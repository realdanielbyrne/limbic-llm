"""M2 + M3a screen: read-back on RecurrentGemma (trained with a 2,048-token window) at about 8K tokens.

Question: does a natively windowed recurrent hybrid lose facts stated beyond its attention window, and does putting
those positions back into its attention layers (``readback.py``) restore them without training?

Items (built with the repo's own task builders, filler from WikiText-103 test, ``--filler``):

  needle    ``evolve.run.ScatterNeedleTask``: ``--pairs`` (4) facts " The code for NAME is DDDD." scattered in filler,
            one queried (multi-key needle).
  update    ``evolve.run.UpdateNeedleTask``: the same, with the queried name stated again later with a new code
            (``--update-form``, default ``now``: " The code for NAME is now DDDD."); the answer is the new code, the
            old one is scored as ``stale``.

  The task's body (facts + filler, up to its query) becomes the document. Chat models (a chat template on the
  tokenizer, ``--chat auto``) get one user turn "<document>\\n\\nWhat is the [current] code for NAME? Answer with the
  4-digit code." and a model turn opened with "The code for NAME is" (``--no-prefill-answer`` leaves it empty);
  base models get "<bos><document>\\n\\n The code for NAME is". Every prompt is exactly ``--seq-len`` tokens (the
  tail filler is trimmed), so a batch needs no padding.

  Cells: ``beyond`` (``--items``): the last fact ends more than W positions before the first question row, i.e. the
  needle is outside every arm's attention band when the question is read. ``inside`` (``--inside-items``, the null
  and the task-difficulty check): the facts sit in the last ``--inside-len`` tokens, inside every band; read-back
  must change nothing there. Scored by exact match of the first number generated (greedy, ``--max-new`` tokens).

Arms (``--arms``; ``selector[:time]``, time default ordinal; see ``readback.py``):

  native            the model as trained (stock sdpa, sliding window W).
  window            W recent columns through the read-back function (implementation control: ≈ native).
  stream            StreamingLLM: ``--sink`` + W − sink recent.
  infllm / random / foreign / oracle[:ordinal|true|none]
                    sink + ``--budget`` read-back columns + W − sink − budget recent. oracle segments: each statement's
                    span centred in budget / n_statements positions (merged where they overlap).

Every arm holds W columns per query in each attention layer (bytes reported). Decoding: the document (all but the
last ``--tail-steps`` tokens) is prefilled in ONE call (HF RecurrentGemma cannot continue a prefill: the recurrent
block's conv state and the sliding mask are only right for a single prefill or one-token steps); the last
``--tail-steps`` tokens (the question, which must fit) and the answer go one token at a time through the KV cache,
so the question's own queries drive InfLLM's selection (``--sel-rows``).

Report (``--out``, one JSON; generated text only in ``--trace-dir`` outside the repo): exact match per task × cell ×
arm with bootstrap intervals, stale rate, hit rate (share of the queried value's digit positions read back, mean over
attention layers, at the last step), paired differences against native and the pre-registered comparisons, and the
gate: oracle:ordinal ≥ 0.5 and native ≤ 0.8 beyond the window (pooled over tasks; per task too).

    python experiments/ca2/readback_screen.py --model google/recurrentgemma-2b-it --device cuda --batch 8 \\
        --out results/evolve/round12/stage1/m2_readback_rg2b_it.json 2>&1 | tee results/.../m2_readback_rg2b_it.log
    CUDA_VISIBLE_DEVICES="" python experiments/ca2/readback_screen.py --tiny --out /tmp/x/rb.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import zlib
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
sys.path.insert(0, str(EXP))
import pretrained_loop as pl  # noqa: E402
from ca2.common import alignment, git_state, paired_boot  # noqa: E402
from ca2.readback import Readback, ReadbackConfig  # noqa: E402
from evolve.run import ScatterNeedleTask, UpdateNeedleTask  # noqa: E402

DEFAULT_ARMS = "native,window,stream,infllm,random,foreign,oracle:ordinal,oracle:true,oracle:none"
QUESTIONS = {"needle": "What is the code for {n}? Answer with the 4-digit code.",
             "update": "What is the current code for {n}? Answer with the 4-digit code."}
ANSWER_PREFIX = "The code for {n} is"
RESULT_KEYS = ("model", "seq_len", "items", "inside_items", "inside_len", "tasks", "pairs", "update_form", "sink",
               "budget", "block", "n_rep", "chunk", "sel_rows", "tail", "tail_steps", "max_new", "chat",
               "prefill_answer", "seed", "tiny", "tiny_window", "filler", "dtype")


# ---------------------------------------------------------------------------------------------------- items
@dataclass
class Item:
    task: str
    cell: str
    uid: str
    ids: list[int]
    answer: str
    stale: str | None
    oracle: list[tuple[int, int]]
    value_pos: list[int]                                          # the queried value's digit positions
    fact_first: int                                               # first token of the earliest statement
    fact_last: int                                                # last token of the latest statement
    seed: int
    meta: dict = field(default_factory=dict)


class Renderer:
    """Prompt pieces around the document: chat (one user turn + answer prefix) or plain continuation."""

    def __init__(self, tok, chat: bool, prefill: bool) -> None:
        self.tok, self.chat, self.prefill = tok, chat, prefill
        if chat:
            hf = tok.tok
            mark = "XQXDOCXQX"
            text = hf.apply_chat_template([{"role": "user", "content": mark}], tokenize=False,
                                          add_generation_prompt=True)
            self.head, self.foot = text.split(mark)
            self.pre = tok.encode(self.head)                     # the template spells <bos> out
        else:
            self.pre = [tok.bos_token_id]

    def post(self, task: str, name: str) -> list[int]:
        if not self.chat:
            return self.tok.encode("\n\n" + " " + ANSWER_PREFIX.format(n=name))
        q = QUESTIONS[task].format(n=name)
        tail = self.foot + (ANSWER_PREFIX.format(n=name) if self.prefill else "")
        return self.tok.encode("\n\n" + q + tail)


def oracle_segments(spans: list[tuple[int, int]], budget: int, sink: int) -> list[tuple[int, int]]:
    """Each statement span (first, last) centred in budget // n positions; overlapping windows are merged."""
    if not spans:
        return []
    per = budget // len(spans)
    iv = []
    for a, b in spans:
        n = b - a + 1
        if n >= per:
            iv.append((a, b))
            continue
        s0 = max(sink, a - (per - n) // 2)
        iv.append((s0, s0 + per - 1))
    iv.sort()
    out = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b - a + 1) for a, b in out]


def _digits(tok, ids: Sequence[int]) -> str:
    return re.sub(r"\D", "", tok.decode(list(ids)))


def build_items(tok, corpus: list[int], rend: Renderer, task: str, cell: str, n: int, args, W: int,
                recent_min: int) -> list[Item]:
    """``n`` prompts of exactly ``args.seq_len`` tokens for one task × cell, deterministic in ``args.seed``."""
    L, Q = args.seq_len, args.tail_steps
    rng = random.Random(f"{args.seed}:{task}:{cell}")
    slack = 16                                                    # the chat question is longer than the task query
    if cell == "beyond":
        S, tail = L + slack, args.tail + slack + Q                # trimming the tail filler keeps it ≥ --tail
    else:
        S, tail = args.inside_len, 32
    kw = dict(pairs=args.pairs, sink=0, corpus=corpus, min_gap=8, tail=tail)
    tk = UpdateNeedleTask(tok, S, form=args.update_form, **kw) if task == "update" else ScatterNeedleTask(tok, S, **kw)
    items = []
    for i in range(n):
        ids, n_ans, _ = tk.sample(rng)
        q_len = tk.last_query_len
        body = ids[:len(ids) - q_len - n_ans]
        qtext = tok.decode(ids[len(ids) - q_len - n_ans:len(ids) - n_ans])
        m = re.match(r"\s*The code for (.+) is$", qtext)
        if not m:
            raise ValueError(f"unexpected query {qtext!r}")
        name = m.group(1)
        post = rend.post(task, name)
        if len(post) > Q:
            raise ValueError(f"the question takes {len(post)} tokens > --tail-steps {Q}")
        pre = list(rend.pre)
        if cell == "beyond":
            excess = len(pre) + len(body) + len(post) - L
            if excess < 0:
                raise ValueError(f"document too short by {-excess} tokens")
            body = body[:len(body) - excess]
            off = len(pre)
        else:
            k = L - len(pre) - len(body) - len(post)
            if k < 0:
                raise ValueError(f"--inside-len {args.inside_len} does not fit --seq-len {L}")
            o = rng.randrange(len(corpus) - k)
            pre = pre + corpus[o:o + k]
            off = len(pre)
        prompt = pre + body + post
        assert len(prompt) == L, (len(prompt), L)
        spans = [(off + a, off + b) for a, b in tk.last_spans]
        if sum(b - a + 1 for a, b in spans) > args.budget:
            raise ValueError(f"the {len(spans)} statements take more than --budget {args.budget} columns")
        r = tk.last_queried
        va = off + tk.last_value_at[r]
        answer = _digits(tok, ids[len(ids) - n_ans:])
        stale = _digits(tok, tk.last_old_answer) if task == "update" else None
        first, last = min(a for a, _ in spans), max(b for _, b in spans)
        q0 = L - Q
        if cell == "beyond" and q0 - last <= W:
            raise ValueError(f"item {i}: last fact at {last} is within W={W} of the question (row {q0}); raise --tail")
        if cell == "inside" and (L - 1 + args.max_new) - first >= recent_min:
            raise ValueError(f"item {i}: facts start at {first}, not inside every arm's {recent_min}-column band")
        items.append(Item(task, cell, f"{task}/{cell}/{i}", prompt, answer, stale,
                          oracle_segments(spans, args.budget, args.sink), list(range(va + 1, va + 1 + len(answer))),
                          first, last, zlib.crc32(f"{args.seed}:{task}:{cell}:{i}".encode()),
                          {"queried_rank": r, "n_statements": len(spans), "gap_to_question": q0 - last}))
    return items


# ---------------------------------------------------------------------------------------------------- model
def load(args):
    if args.tiny:
        from transformers import RecurrentGemmaConfig, RecurrentGemmaForCausalLM
        torch.manual_seed(args.seed)
        tok = pl.ByteTokenizer()
        cfg = RecurrentGemmaConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                                   num_attention_heads=4, num_key_value_heads=1, head_dim=16, lru_width=64,
                                   attention_window_size=args.tiny_window, partial_rotary_factor=0.5,
                                   bos_token_id=256, eos_token_id=256, pad_token_id=256)
        cfg._attn_implementation = "sdpa"
        model = RecurrentGemmaForCausalLM(cfg)
        dev = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        model = model.to(dev, torch.float32)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=pl.DTYPES[args.dtype],
                                                     attn_implementation="sdpa")
        model = model.to(args.device)
        tok = pl.HFTokenizer(AutoTokenizer.from_pretrained(args.model))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def model_facts(model) -> dict:
    c = model.config
    kinds = list(c.layers_block_type)
    attn = [i for i, k in enumerate(kinds) if k == "attention"]
    return {"window": int(c.attention_window_size), "attention_layers": attn, "n_layers": len(kinds),
            "kv_heads": int(c.num_key_value_heads), "head_dim": int(c.head_dim),
            "dtype": str(next(model.parameters()).dtype)}


def memory_bytes(arm: str, facts: dict, L: int, args) -> dict:
    """KV bytes per sequence: attention columns (equal across arms), the off-window archive, InfLLM's keys."""
    sel = arm.split(":")[0]
    elt = 2 if "16" in facts["dtype"] else 4
    per_col = len(facts["attention_layers"]) * facts["kv_heads"] * facts["head_dim"] * 2 * elt
    W = facts["window"]
    out = {"columns_per_query": W, "attention_kv_bytes": W * per_col}
    archive = 0 if sel in ("native", "window", "stream") else max(0, L + args.max_new - W) * per_col
    out["archive_bytes"] = archive
    nb = max(0, (L - args.sink) // args.block)
    reps = nb * args.n_rep * facts["head_dim"] * 4 * len(facts["attention_layers"])
    out["rep_bytes"] = reps if sel == "infllm" else 0
    return out


@torch.no_grad()
def run_batch(model, tok, items: list[Item], arm: str, args) -> list[dict]:
    """Prefill the document in one call, step the question tail and the answer one token at a time; greedy."""
    from transformers import DynamicCache
    B, L, Q = len(items), len(items[0].ids), args.tail_steps
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    ids = torch.tensor([it.ids for it in items], device=dev)
    sel, _, tm = arm.partition(":")
    rb = None
    if sel != "native":
        rb = Readback(model, ReadbackConfig(selector=sel, time=tm or "ordinal", sink=args.sink, budget=args.budget,
                                            block=args.block, n_rep=args.n_rep, chunk=args.chunk,
                                            sel_rows=args.sel_rows))
    model._setup_cache(model.config, B, dev, dtype)
    cache = DynamicCache(config=model.config)
    n0 = L - Q
    stops = {tok.eos_token_id}
    if hasattr(tok, "tok"):
        eot = tok.tok.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot, int) and eot >= 0:
            stops.add(eot)

    def fwd(x, p0):
        pos = torch.arange(p0, p0 + x.shape[1], device=dev)[None].expand(B, -1)
        return model(x, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]

    gen = []
    with (rb.active() if rb else nullcontext()):
        if rb:
            rb.begin(B, oracle=[it.oracle for it in items], rand_seeds=[it.seed for it in items])
        logits = fwd(ids[:, :n0], 0)
        for t in range(n0, L):
            logits = fwd(ids[:, t:t + 1], t)
        for k in range(args.max_new):
            nxt = logits.float().argmax(-1)
            gen.append(nxt)
            if k + 1 < args.max_new:
                logits = fwd(nxt[:, None], L + k)
    gen = torch.stack(gen, 1).tolist()
    recs = []
    for b, it in enumerate(items):
        g = gen[b]
        cut = next((j for j, x in enumerate(g) if x in stops), len(g))
        text = tok.decode(g[:cut])
        m = re.search(r"\d+", text)
        pred = m.group() if m else None
        hit = None
        if rb is not None and rb.last_selected and sel not in ("foreign",):
            hs = []
            for pos, src in rb.last_selected.values():
                got = set(pos[b][pos[b] >= 0].tolist())
                hs.append(sum(p in got for p in it.value_pos) / len(it.value_pos))
            hit = float(np.mean(hs))
        recs.append({"uid": it.uid, "em": int(pred == it.answer), "stale": None if it.stale is None else
                     int(pred == it.stale), "hit": hit, "text": text})
    if rb is not None:
        recs[0]["_max_cols"] = rb.stats["max_cols"]
        recs[0]["_selected_mean"] = rb.selected_mean()
    return recs


# ------------------------------------------------------------------------------------------------- analysis
def summarise(recs: list[dict], seed: int) -> dict:
    em = [r["em"] for r in recs]
    bt = paired_boot(em, seed=seed)
    st = [r["stale"] for r in recs if r["stale"] is not None]
    hits = [r["hit"] for r in recs if r["hit"] is not None]
    return {"n": len(em), "em": bt["diff"], "em_lo": bt["lo"], "em_hi": bt["hi"],
            "stale": float(np.mean(st)) if st else None, "hit": float(np.mean(hits)) if hits else None}


def compare(report: dict, seed: int) -> dict:
    """Paired differences on the beyond cell (pooled over tasks, and per task) and the gate."""
    arms = report["arms"]
    tasks = report["args"]["tasks"].split(",")
    out = {}

    def vec(arm, task=None):
        if arm not in arms or not arms[arm].get("complete"):
            return None
        rs = [r for r in arms[arm]["per_item"] if r["cell"] == "beyond" and (task is None or r["task"] == task)]
        return {r["uid"]: r["em"] for r in rs}

    pairs = [(a, "native") for a in arms if a != "native"]
    pairs += [("oracle:ordinal", "oracle:true"), ("oracle:ordinal", "oracle:none"), ("oracle:ordinal", "infllm"),
              ("infllm", "stream"), ("infllm", "random"), ("oracle:ordinal", "foreign"), ("stream", "window")]
    for scope in [None] + tasks:
        res = {}
        for a, b in pairs:
            va, vb = vec(a, scope), vec(b, scope)
            if not va or not vb:
                continue
            keys = sorted(set(va) & set(vb))
            res[f"{a} - {b}"] = paired_boot([va[k] for k in keys], [vb[k] for k in keys], seed=seed)
        out["pooled" if scope is None else scope] = res
    gate = {}
    for scope in [None] + tasks:
        o, n = vec("oracle:ordinal", scope), vec("native", scope)
        if o and n:
            om, nm = float(np.mean(list(o.values()))), float(np.mean(list(n.values())))
            gate["pooled" if scope is None else scope] = {"oracle_ordinal": om, "native": nm,
                                                          "pass": bool(om >= 0.5 and nm <= 0.8)}
    inside = None
    if "native" in arms and arms["native"].get("complete"):
        ins = [r["em"] for r in arms["native"]["per_item"] if r["cell"] == "inside"]
        inside = float(np.mean(ins)) if ins else None
    gate["inside_native"] = {"em": inside, "requirement_ge_0.9": None if inside is None else bool(inside >= 0.9)}
    return {"paired_beyond": out, "gate": gate}


# ----------------------------------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a = ap.add_argument
    a("--model", default="google/recurrentgemma-2b-it")
    a("--dtype", default="bfloat16", choices=sorted(pl.DTYPES))
    a("--device", default="cuda")
    a("--tiny", action="store_true", help="random 6-layer RecurrentGemma + byte tokenizer (CPU smoke)")
    a("--tiny-window", type=int, default=384)
    a("--seq-len", type=int, default=8192, help="prompt length in tokens (every item exactly)")
    a("--items", type=int, default=50, help="beyond-window items per task")
    a("--inside-items", type=int, default=20, help="inside-window items per task (0 = none)")
    a("--inside-len", type=int, default=1024, help="inside cell: the facts' span of the prompt's end")
    a("--tasks", default="needle,update")
    a("--pairs", type=int, default=4)
    a("--update-form", default="now")
    a("--arms", default=DEFAULT_ARMS)
    a("--sink", type=int, default=4)
    a("--budget", type=int, default=512, help="R: read-back columns per query per attention layer")
    a("--block", type=int, default=128)
    a("--n-rep", type=int, default=4)
    a("--chunk", type=int, default=256, help="prefill query rows per selection step")
    a("--sel-rows", type=int, default=32, help="decode steps: rows whose queries score InfLLM blocks")
    a("--tail", type=int, default=2560, help="beyond cell: minimum filler from the last fact to the question")
    a("--tail-steps", type=int, default=64, help="last prompt tokens fed one at a time (must hold the question)")
    a("--max-new", type=int, default=12)
    a("--batch", type=int, default=8)
    a("--chat", default="auto", choices=["auto", "on", "off"])
    a("--no-prefill-answer", dest="prefill_answer", action="store_false")
    a("--filler", default="~/limbic-data/wikitext103_test.txt")
    a("--seed", type=int, default=0)
    a("--limit-batches", type=int, default=0, help="smoke: only this many batches per task x cell")
    a("--out", default=None)
    a("--trace-dir", default=None)
    a("--force", action="store_true", help="start again when --out holds a run with other settings")
    return ap


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    aj = alignment().judge
    if args.tiny:
        args.dtype = "float32"
        if args.seq_len == 8192:                                  # tiny defaults: W = 384, 1,024 tokens, 2 pairs
            args.seq_len, args.tail, args.inside_len, args.budget, args.block = 1024, 450, 180, 128, 16
            args.sink, args.chunk, args.sel_rows, args.tail_steps, args.pairs = 2, 64, 4, 40, 2
    short = "tiny" if args.tiny else args.model.split("/")[-1]
    name = aj.safe_name(f"m2_readback_{short}")
    args.out = Path(args.out or ROOT / "results" / "evolve" / "round12" / "stage1" / f"{name}.json")
    args.trace_dir = str(aj.assert_private(args.trace_dir or Path("~/limbic-data/ca2_traces") / args.out.stem, ROOT))
    torch.manual_seed(args.seed)
    t_start = time.time()
    model, tok = load(args)
    facts = model_facts(model)
    W = facts["window"]
    chat = args.chat == "on" or (args.chat == "auto" and getattr(getattr(tok, "tok", None), "chat_template", None))
    args.chat = "on" if chat else "off"
    rend = Renderer(tok, bool(chat), args.prefill_answer)
    corpus = tok.encode(Path(args.filler).expanduser().read_text(encoding="utf-8", errors="replace"))
    recent_min = W - args.sink - args.budget
    arms = [s.strip() for s in args.arms.split(",") if s.strip()]
    cells = [("beyond", args.items)] + ([("inside", args.inside_items)] if args.inside_items else [])
    items = []
    for task in args.tasks.split(","):
        for cell, n in cells:
            items += build_items(tok, corpus, rend, task, cell, n, args, W, recent_min)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.out.read_text()) if args.out.exists() else {}
    cur = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    diff = [k for k in RESULT_KEYS if k in report.get("args", {}) and report["args"][k] != cur.get(k)]
    if diff and not args.force:
        raise SystemExit(f"{args.out} holds a run with other settings ({diff}); use another --out or --force")
    if diff:
        report = {}
    report.setdefault("arms", {})
    report.update({"args": cur, "git": git_state(), "model": facts, "torch": torch.__version__,
                   "items": [{"uid": it.uid, "answer_len": len(it.answer), "oracle_cols": sum(n for _, n in it.oracle),
                              "fact_first": it.fact_first, "fact_last": it.fact_last, **it.meta} for it in items],
                   "arm_order": arms})
    trace = Path(args.trace_dir)
    trace.mkdir(parents=True, exist_ok=True)
    print(f"{args.model if not args.tiny else 'tiny'}: W={W}, attention layers {facts['attention_layers']}, "
          f"L={args.seq_len}, {len(items)} items, {len(arms)} arms, chat={args.chat}", flush=True)
    groups: dict[tuple[str, str], list[Item]] = {}
    for it in items:
        groups.setdefault((it.task, it.cell), []).append(it)
    for arm in arms:
        if report["arms"].get(arm, {}).get("complete"):
            print(f"[{arm}] complete in {args.out.name}; skipped", flush=True)
            continue
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        per_item, extra = [], {"max_cols": 0, "selected_mean": []}
        with open(trace / f"{arm.replace(':', '_')}.jsonl", "w") as fh:
            for (task, cell), its in groups.items():
                nb = 0
                for i in range(0, len(its), args.batch):
                    if args.limit_batches and nb >= args.limit_batches:
                        break
                    chunk = its[i:i + args.batch]
                    if arm.startswith("foreign") and len(chunk) < 2:
                        chunk = its[max(0, i - 1):i + 1]          # foreign rolls rows: never a batch of one
                        recs = run_batch(model, tok, chunk, arm, args)[-1:]
                        chunk = chunk[-1:]
                    else:
                        recs = run_batch(model, tok, chunk, arm, args)
                    nb += 1
                    for it, r in zip(chunk, recs):
                        extra["max_cols"] = max(extra["max_cols"], r.pop("_max_cols", 0) or 0)
                        sm = r.pop("_selected_mean", None)
                        if sm is not None:
                            extra["selected_mean"].append(sm)
                        fh.write(json.dumps({"arm": arm, "task": task, "cell": cell, "answer": it.answer,
                                             "stale_answer": it.stale, **r}) + "\n")
                        per_item.append({"uid": r["uid"], "task": task, "cell": cell, "em": r["em"],
                                         "stale": r["stale"], "hit": r["hit"]})
        dt = time.time() - t0
        summ = {}
        for (task, cell) in groups:
            rs = [r for r in per_item if r["task"] == task and r["cell"] == cell]
            if rs:
                summ[f"{task}/{cell}"] = summarise(rs, args.seed)
        rs = [r for r in per_item if r["cell"] == "beyond"]
        if rs:
            summ["pooled/beyond"] = summarise(rs, args.seed)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
        report["arms"][arm] = {"complete": True, "seconds": dt, "peak_gb": peak, "per_item": per_item, "summary": summ,
                               "memory": memory_bytes(arm, facts, args.seq_len, args),
                               "max_cols_seen": extra["max_cols"] or None,
                               "readback_cols_mean": float(np.mean(extra["selected_mean"])) if extra["selected_mean"]
                               else None}
        report["comparisons"] = compare(report, args.seed)
        report["seconds"] = time.time() - t_start
        args.out.write_text(json.dumps(report, indent=1))
        line = "  ".join(f"{k} {v['em']:.2f} [{v['em_lo']:.2f},{v['em_hi']:.2f}]" +
                         (f" stale {v['stale']:.2f}" if v["stale"] is not None else "") +
                         (f" hit {v['hit']:.2f}" if v["hit"] is not None else "") for k, v in summ.items())
        print(f"[{arm}] {dt:.0f}s" + (f" peak {peak:.1f} GB" if peak else "") + f"  {line}", flush=True)
    g = report.get("comparisons", {}).get("gate", {})
    print("gate:", json.dumps(g), flush=True)
    print(f"saved {args.out} (no generated text); traces in {args.trace_dir}", flush=True)
    return report


if __name__ == "__main__":
    main()
