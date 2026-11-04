"""Evaluate a trained candidate by *which* fact was queried, and report what its gate learned.

Mean digit-1 accuracy cannot tell content addressing from positional shortcuts: a memory of only
the first and the last fact scores 2/3 with 3 pairs. This rebuilds a run's candidate from its
result JSON (class, flags, tap, pairs, scattered or fixed layout), loads its ``.pt`` checkpoint,
and on fresh batches reports, by the queried fact's rank (first, middle, last):

  digit-1 accuracy and exact match of the 4-digit code, with the side channels on, and the same for
  the windowed base (side channels off) and full attention as controls (``--controls``);
  for an EventLoop candidate (W1i, W1j): the learned write strength ε (mean over all tokens, and at
  each fact's first digit), the evidence on whether the events became sparse (review w1i-3). ε is
  the gate as run (B17): on the side's input (the sink dropped), and for W1j with its CA1 novelty
  term, recorded in the side's own forward;
  for a slot candidate (W1s), from ``side.loop.last_diag`` recorded in its own forward: fact
  coverage (the queried fact has a retained slot at the query: an onset inside the fact), slot hit
  (the most-read slot is that one; null for the superposed store, which reads no slot), alignment
  hit (the aligned read lands on the queried first digit; 0 by construction for reinstatement and
  ``--w1s-align none``, which align nothing), the null slot's weight, and the gate's AUC for first
  digits against filler tokens.

Variants of the task test generalisation (all need a scattered-layout run):
  --jitter         facts take one of three phrasings; the query keeps the plain one
  --update FORM    one name is stated again with a new code; the answer is the new code, and the
                   stale (old) code is scored too: digit 1 and exact, NEW against OLD. Forms: same,
                   now, new, changed ("U's code has changed from OLD to NEW"), nowold ("... is now
                   NEW (it used to be OLD)"). ``--update-k K`` restates it K times (the last wins;
                   the stale code is the one before it)
  --decoys N       N extra code statements for names never queried (ranks count every statement)
  --order KIND     before / next / first / last: the code stated just before or after a named
                   fact, or the first or last code stated (``run.OrderNeedleTask``)
  --heldout-names  names from a list disjoint from ``pl.NAMES`` (never seen in training)
  --w1s-ctrl       W1s controls: ``empty`` forces the store empty (must score 0.00), ``shuffle``
                   permutes the retained slots across the batch's rows (about chance; additive
                   output only, and --batch ≥ 2)
  --ctrl-conds     both W1s controls as extra conditions of the same pass (``side_empty``,
                   ``side_shuffle``, in "controls"); the slot diagnostics stay those of windowed+side

With more statements than names (e.g. 32 codes on the 20 training names) the queried name still
comes from the run's list and the others from the list ∪ ``names.syllable_pool(2000, 0)``; the
output records this as ``names_rule``.

Evaluation-time overrides of a trained candidate (diagnostics, rule E6; each adds a file suffix):
  --w1s-set KEY=VAL      a candidate flag, parsed by the module's ``split_args`` (types coerced,
                         unknown keys refused; booleans take true/false). Merge order: split_args
                         defaults < the run's flags < --w1s-set. Parameters the override adds are
                         freshly initialised under ``torch.manual_seed(--seed)``, and the checkpoint's
                         missing keys must be exactly those (or come from --w1s-load-extra)
  --w1s-param NAME=VAL   set, or with ``+=`` shift, a parameter (state_dict name) after loading,
                         e.g. ``loop.log_beta_align+=0.3``
  --w1s-load-extra PT    a state-dict fragment loaded after the checkpoint (e.g. a fitted W_z)
  --keep-oracle MODE     with ``--w1s-set w1s_keep_by=external``: the kept entries are chosen by a
                         seeded random table (``random``), the facts' space and digits 1–3
                         (``offsets``), or the attention each token receives in the windowed base
                         (``attn``: eager weights, heads and ``--keep-attn-layers`` averaged)

Evaluation can differ from training (rule E6): ``--pairs``, ``--seq-len`` (a multiple of the window),
``--text-file`` (natural filler) and ``--needle-tail`` change the task at evaluation only. Only the
logits of the answer positions are computed, so long sequences fit in memory; above 2048 tokens
``--batch`` must be given.

    python experiments/evolve/eval_by_query.py results/evolve/round4/r4_w1i_t21_pairs3_scatter.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402
from evolve.names import HELDOUT_NAMES  # noqa: E402,F401  (re-exported: it used to live here)
from evolve.run import (TRAIN_NAMES, UPDATE_STEMS, OrderNeedleTask, ScatterNeedleTask, UpdateNeedleTask,  # noqa: E402
                        oracle_gate, oracle_onsets)

CTRL_CONDS = {"side_empty": "empty", "side_shuffle": "shuffle"}   # --ctrl-conds: condition → side.ctrl
LONG = 2048                                                        # above this, --batch must be explicit


def _score(lg: torch.Tensor, ids: torch.Tensor, n_ans: int) -> list[list[bool]]:
    """Per sequence, whether each answer token is predicted. lg: logits of the last n_ans + 1 positions."""
    pred = lg.argmax(-1).cpu()                                     # (B, n_ans + 1)
    T = ids.shape[1]
    return [[pred[i, k].item() == ids[i, T - n_ans + k].item() for k in range(n_ans)] for i in range(len(ids))]


def _summarise(per_seq: list, P: int, edges: list[int]) -> dict:
    """Accuracy by queried rank and by gap bin from (rank, gap, digit-1 ok, exact ok) tuples."""
    by_rank = {"digit1": [], "exact": [], "n": []}
    for k in range(P):
        sel = [r for r in per_seq if r[0] == k]
        by_rank["n"].append(len(sel))
        by_rank["digit1"].append(sum(r[2] for r in sel) / max(1, len(sel)))
        by_rank["exact"].append(sum(r[3] for r in sel) / max(1, len(sel)))
    bins = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in per_seq if lo < r[1] <= hi]
        if sel:
            key = f"{lo + 1 if lo >= 0 else 0}-{hi if hi < 10 ** 9 else 'max'}"
            bins[key] = {"n": len(sel), "digit1": sum(r[2] for r in sel) / len(sel),
                         "exact": sum(r[3] for r in sel) / len(sel),
                         "by_rank": {str(k): {"n": sum(1 for r in sel if r[0] == k),
                                              "exact": sum(r[3] for r in sel if r[0] == k)
                                              / max(1, sum(1 for r in sel if r[0] == k))} for k in range(P)}}
    n = max(1, len(per_seq))
    return {"digit1": sum(r[2] for r in per_seq) / n, "exact": sum(r[3] for r in per_seq) / n,
            "digit1_by_queried_rank": by_rank["digit1"], "exact_by_queried_rank": by_rank["exact"],
            "count_by_queried_rank": by_rank["n"], "by_gap": bins}


def _auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """P(pos > neg), ties counted half: the gate's ranking of first digits above filler."""
    neg = neg.sort().values
    below = torch.searchsorted(neg, pos, right=False) + torch.searchsorted(neg, pos, right=True)
    return (below.double() / 2).mean().item() / max(1, len(neg))


def _fires(diag: dict, b: int, tq: int, sp: list, same: set, theta: float, acc: dict,
           P: int = 0, pool: int = 8) -> None:
    """Supersession fire diagnostics (G0.3) at the query: for every pair of slots held at tq, whether their
    keys match (cos > θ, the point where supw's tag σ(γ(cos − θ)) passes ½). Pairs are counted by kind:
    two different facts (a false fire), a fact and a filler window, and the old and new statements of an
    updated name (a true fire; ``same`` holds those statements' ranks)."""
    held = [int(x) for x in diag["retained"][b, tq] if x >= 0]
    cs, ck = diag["cand_s"][b].cpu(), diag["cand_key"][b].float().cpu()
    idx = {int(s_): i for i, s_ in enumerate(cs.tolist()) if s_ >= 0}
    def owner(s_):                                                 # the fact the key pools from (≥ half of it)
        lo_k, hi_k = s_ - P, s_ - P + pool - 1
        best = max(((min(hi, hi_k) - max(lo, lo_k) + 1, r) for r, (lo, hi) in enumerate(sp)), default=(0, -1))
        return best[1] if 2 * best[0] >= pool else -1
    for x in range(len(held)):
        for y in range(x + 1, len(held)):
            ox, oy = owner(held[x]), owner(held[y])
            fire = float(float(ck[idx[held[x]]] @ ck[idx[held[y]]]) > theta)
            if ox >= 0 and oy >= 0 and ox != oy:
                acc["fire_true" if {ox, oy} <= same and len(same) > 1 else "fire_false"].append(fire)
            elif (ox >= 0) != (oy >= 0):
                acc["fire_filler"].append(fire)


def _slot_diag(diag: dict, js, vals, spans, qlens, T: int, n_ans: int, sink: int, acc: dict,
               same=None, theta: float = 0.9, P: int = 0, pool: int = 8) -> None:
    """Accumulate W1s's diagnostics for one batch. The query is the answer's space (it predicts
    digit 1); positions are side coordinates (the sink dropped), as in ``last_diag``."""
    p, on = diag["p"].float().cpu(), diag["onsets"].cpu()
    ret, rw, al = diag["retained"].cpu(), diag["read_w"].float().cpu(), diag["align"].cpu()
    K, tq = ret.shape[-1], T - n_ans - sink
    for b, j in enumerate(js):
        sp = [(a - sink, e - sink) for a, e in spans[b]]
        if "cand_key" in diag:
            _fires(diag, b, tq, sp, (same[b] if same else set()), theta, acc, P, pool)
        d1 = [v + 1 - sink for v in vals[b]]
        r = ret[b, tq]
        in_q = (r >= sp[j][0]) & (r <= sp[j][1])                   # retained onsets inside the queried fact
        k, a = int(rw[b, tq].argmax()), int(al[b, tq])
        hit = float(k < K and bool(in_q[k])) if rw[b, tq].sum() > 0 else None   # None: no slot read (superposed)
        acc["seq"].append((j, float(in_q.any()), hit, float(a == d1[j])))
        acc["align_within1"].append(float(a >= 0 and abs(a - d1[j]) <= 1))
        acc["null_weight"].append(rw[b, tq, -1].item() if hit is not None else None)
        fill = torch.ones(p.shape[1], dtype=torch.bool)
        for lo, hi in sp:
            fill[lo:hi + 1] = False
        fill[T - sink - qlens[b] - n_ans:] = False                  # the query and its answer
        acc["p_d1"].append(p[b, d1]), acc["p_fill"].append(p[b, fill])
        acc["onset_in_fact"] += [float(on[b, lo:hi + 1].any()) for lo, hi in sp]
        acc["density"].append(on[b].float().mean().item())
        acc.setdefault("ret256", []).append(float(((r >= 0) & (r > tq - 256)).sum()))   # held slots ≤ 256 old


def _slot_summary(acc: dict, R: int) -> dict:
    def mean(v):                                                   # None entries are not measured
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None
    out = {}
    for i, name in ((1, "fact_coverage"), (2, "slot_hit"), (3, "align_hit")):
        out[name] = mean([r[i] for r in acc["seq"]])
        out[f"{name}_by_queried_rank"] = [mean([r[i] for r in acc["seq"] if r[0] == k]) for k in range(R)]
    pos, neg = torch.cat(acc["p_d1"]), torch.cat(acc["p_fill"])
    out.update({"align_within1": mean(acc["align_within1"]), "null_weight_at_query": mean(acc["null_weight"]),
                "onset_density": mean(acc["density"]), "onset_in_fact": mean(acc["onset_in_fact"]),
                "p_mean_digit1": pos.mean().item(), "p_mean_filler": neg.mean().item(),
                "gate_auc_digit1_vs_filler": _auc(pos, neg)})
    if acc.get("ret256"):                                          # the ord form's scale at age 256 (G0.2)
        out["retained_within_256_at_query"] = mean(acc["ret256"])
    for k in ("fire_false", "fire_filler", "fire_true"):           # supw fire rates (G0.3), cos > θ
        if acc.get(k):
            out[f"{k}_rate"], out[f"{k}_pairs"] = mean(acc[k]), len(acc[k])
    return out


def _tag(s) -> str:
    """A file-name-safe piece of a variant suffix."""
    s = str(s)
    return re.sub(r"[^A-Za-z0-9.+_-]", "_", s) if s else "none"


def parse_sets(mod, items: list[str]) -> list[tuple[str, str, object]]:
    """``--w1s-set KEY=VAL`` items → (key, raw value, parsed value), each parsed by the candidate module's
    ``split_args`` so types and choices are checked. KEY is a ``split_args`` dest (``w1s_recency_form``,
    dashes allowed); keys not among its defaults are refused. Booleans take true/false."""
    defaults = vars(mod.split_args([])[0])
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--w1s-set {item!r}: expected KEY=VAL")
        key, val = item.split("=", 1)
        key = key.strip().lstrip("-").replace("-", "_")
        if key not in defaults:
            raise SystemExit(f"--w1s-set {key}: not a flag of {mod.__name__} (known: {', '.join(sorted(defaults))})")
        if key.endswith("_ctrl"):
            raise SystemExit("--w1s-set: use --w1s-ctrl or --ctrl-conds for the controls")
        flag, d = "--" + key.replace("_", "-"), defaults[key]
        if isinstance(d, bool):
            v = val.strip().lower()
            if v not in ("true", "false", "1", "0", "yes", "no"):
                raise SystemExit(f"--w1s-set {key}: a boolean flag takes true/false, got {val!r}")
            argv = [flag] if v in ("true", "1", "yes") else ["--no-" + flag[2:]]
        else:
            argv = [flag] + (val.replace(",", " ").split() if isinstance(d, (list, tuple)) else [val])
        try:
            ns, rest = mod.split_args(argv)
        except SystemExit as e:                                    # argparse already printed why
            raise SystemExit(f"--w1s-set {item!r}: rejected by {mod.__name__}.split_args ({e})")
        if rest == argv and isinstance(d, bool) and d is False:    # a store_true flag: false is its default
            ns, rest = mod.split_args([])
        if rest:
            raise SystemExit(f"--w1s-set {item!r}: not parsed by {mod.__name__}.split_args: {rest}")
        out.append((key, val, getattr(ns, key)))
    return out


PARAM = re.compile(r"^\s*([\w.]+)\s*(\+=|=)\s*(\S+)\s*$")


def apply_params(side, items: list[str]) -> list[dict]:
    """``--w1s-param NAME=VAL`` (set) or ``NAME+=VAL`` (shift), on ``side.state_dict()`` names, in place."""
    sd = side.state_dict(keep_vars=True)
    done = []
    for item in items:
        m = PARAM.match(item)
        if not m:
            raise SystemExit(f"--w1s-param {item!r}: expected NAME=VAL or NAME+=VAL")
        name, op, val = m.group(1), m.group(2), float(m.group(3))
        if name not in sd:
            raise SystemExit(f"--w1s-param {name}: not in the side's state_dict "
                             f"(e.g. {', '.join(k for k in sd if k.startswith('loop.'))[:300]})")
        t = sd[name]
        with torch.no_grad():
            t.add_(val) if op == "+=" else t.fill_(val)
        done.append({"name": name, "op": op, "value": val,
                     "result": t.detach().float().mean().item() if t.numel() else None})
    return done


@contextmanager
def ctrl_condition(side, cond: str):
    """``side.ctrl`` set for a --ctrl-conds condition, and back to none afterwards."""
    mode = CTRL_CONDS.get(cond)
    if mode is None:
        yield
        return
    side.ctrl = mode
    try:
        yield
    finally:
        side.ctrl = "none"


def keep_layers(spec: str, lm) -> list[int]:
    """``--keep-attn-layers LO-HI`` (inclusive) → the attention blocks among them."""
    lo, hi = (int(x) for x in spec.split("-")) if "-" in spec else (int(spec), int(spec))
    n = len(lm.blocks)
    if not 0 <= lo <= hi < n:
        raise SystemExit(f"--keep-attn-layers {spec}: the model has blocks 0–{n - 1}")
    layers = [i for i in range(lo, hi + 1) if lm.kinds[i] == "attention"]
    if not layers:
        raise SystemExit(f"--keep-attn-layers {spec}: no attention layer in that range")
    return layers


@torch.no_grad()
def received_attention(lm, ids: torch.Tensor, window: int, layers: list[int], sink: int) -> torch.Tensor:
    """(B, T', T') float32, side coordinates (the sink dropped): R[b, q, j] = the attention query q pays
    key j in the windowed base (side off), averaged over heads and ``layers``. Needs eager attention,
    whose modules return (out, weights)."""
    acc = []

    def hook(mod, args, out):
        w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        if not torch.is_tensor(w) or w.dim() != 4:
            raise RuntimeError("no attention weights from the attention module: use --attn-impl eager")
        w = w.float().mean(1)[:, sink:, sink:]
        if acc:
            acc[0] += w
        else:
            acc.append(w.clone())
    hs = [pl._child(lm.blocks[i], pl.ATTN_MIXERS).register_forward_hook(hook) for i in layers]
    try:
        with lm.condition(window, False):
            lm(ids, keep=1)
    finally:
        for h in hs:
            h.remove()
    return acc[0] / len(layers)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("result", help="a round result JSON with a .pt checkpoint next to it")
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--batch", type=int, default=None,
                    help=f"sequences per forward (default 16; must be given above {LONG} tokens)")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--pairs", type=int, default=None, help="evaluate with this many stored pairs (default: as trained)")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="evaluation sequence length, a multiple of the window (default: as trained)")
    ap.add_argument("--text-file", default=None, help="natural-text filler at evaluation only (default: as trained)")
    ap.add_argument("--random-filler", action="store_true",
                    help="random-token filler even if the run was trained with --text-file")
    ap.add_argument("--needle-tail", type=int, default=None,
                    help="minimum filler between the last fact and the query (default: as trained, else 256)")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default=None, help="default: as trained")
    ap.add_argument("--controls", action=argparse.BooleanOptionalAction, default=True,
                    help="also score the windowed base (side channels off) and full attention")
    ap.add_argument("--gap-bins", type=int, nargs="+", default=None,
                    help="bin edges for accuracy against the queried fact's gap (default: 4 even bins above 256)")
    ap.add_argument("--jitter", action="store_true", help="facts in one of three phrasings (query plain)")
    ap.add_argument("--update", choices=list(UPDATE_STEMS), default=None,
                    help="a name restated with a new code (phrasing: same template, ' is now', 'The new code', "
                         "'has changed from OLD to', 'is now NEW (it used to be OLD)')")
    ap.add_argument("--update-k", type=int, default=1, help="with --update: restatements of the name (the last wins)")
    ap.add_argument("--decoys", type=int, default=0, help="extra code statements for names never queried")
    ap.add_argument("--order", choices=["before", "next", "first", "last"], default=None,
                    help="order queries: the code stated just before/after a name's, or the first/last code")
    ap.add_argument("--heldout-names", action="store_true", help="names never used in training (not in pl.NAMES)")
    ap.add_argument("--w1s-ctrl", choices=["none", "empty", "shuffle"], default="none",
                    help="W1s control: an empty store, or retained slots shuffled across rows")
    ap.add_argument("--ctrl-conds", action="store_true",
                    help="add side_empty and side_shuffle conditions (W1s controls) to this pass")
    ap.add_argument("--w1s-set", action="append", default=[], metavar="KEY=VAL",
                    help="override a candidate flag at evaluation (repeatable)")
    ap.add_argument("--w1s-param", action="append", default=[], metavar="NAME=VAL|NAME+=VAL",
                    help="set or shift a side parameter after loading (repeatable)")
    ap.add_argument("--allow-random-code", action="store_true",
                    help="with --w1s-set w1s_code=learned and no --w1s-load-extra: a seeded random W_z (a control)")
    ap.add_argument("--w1s-load-extra", default=None, metavar="PT",
                    help="a state-dict fragment loaded after the checkpoint")
    ap.add_argument("--keep-oracle", choices=["random", "offsets", "attn"], default=None,
                    help="external keep scores (needs --w1s-set w1s_keep_by=external and a --w1s-keep M)")
    ap.add_argument("--keep-attn-layers", default="4-20", metavar="LO-HI",
                    help="--keep-oracle attn: blocks whose attention is averaged (inclusive)")
    ap.add_argument("--out", default=None,
                    help="default: <result>.by_query.json (a variant adds its name, e.g. .by_query.jitter.json)")
    args = ap.parse_args(argv)
    rec = json.loads(Path(args.result).read_text())
    ra, cand = rec["args"], rec.get("candidate")
    if args.ctrl_conds and args.w1s_ctrl != "none":
        raise SystemExit("--ctrl-conds runs both W1s controls itself; drop --w1s-ctrl")
    if (args.w1s_set or args.w1s_param or args.w1s_load_extra or args.keep_oracle) and not cand:
        raise SystemExit("--w1s-set, --w1s-param, --w1s-load-extra and --keep-oracle need a candidate run")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    attn_impl = args.attn_impl or ra.get("attn_impl", "sdpa")
    if args.keep_oracle == "attn" and attn_impl != "eager":
        raise SystemExit("--keep-oracle attn reads attention weights: needs --attn-impl eager")
    if ra.get("tiny"):                                             # random small model, rebuilt from its seed
        pa = pl.parse_args(["--tiny", "--tiny-arch", ra["tiny_arch"], "--seed", str(ra["seed"]),
                            "--dtype", ra["dtype"], "--device", dev, "--attn-impl", attn_impl])
    else:
        pa = pl.parse_args(["--model", ra["model"], "--dtype", "bfloat16", "--device", dev,
                            "--attn-impl", attn_impl])
    model, tok = pl.load_model(pa)
    d = pl.text_config(model).hidden_size
    flags, sets = {}, []
    torch.manual_seed(args.seed)                                   # parameters an override adds: reproducible
    if cand:
        mod_name, cls = cand["class"].split(":")
        mod = importlib.import_module(mod_name)
        Cand = getattr(mod, cls)
        flags = {**vars(mod.split_args([])[0]), **cand["args"]} if hasattr(mod, "split_args") else cand["args"]
        run_flags = dict(flags)
        if args.w1s_set:
            if not hasattr(mod, "split_args"):
                raise SystemExit(f"--w1s-set needs a candidate module with split_args ({mod_name})")
            sets = parse_sets(mod, args.w1s_set)
            keys = {k for k, _, _ in sets}
            if "w1s_time_feat" in keys and "w1s_recency_form" not in keys:   # training resolves one from the other
                raise SystemExit("--w1s-set w1s_time_feat needs an explicit --w1s-set w1s_recency_form=... "
                                 "(training defaults it to none with a time feature; this run recorded "
                                 f"{run_flags.get('w1s_recency_form')!r})")
            flags.update({k: v for k, _, v in sets})
            if (flags.get("w1s_code") == "learned" and run_flags.get("w1s_code", "lex") != "learned"
                    and not args.w1s_load_extra and not args.allow_random_code):
                raise SystemExit("--w1s-set w1s_code=learned on a run trained without it needs --w1s-load-extra "
                                 "with a fitted loop.wz.weight (fit_code.py); --allow-random-code for a random control")
        side = Cand(d, argparse.Namespace(**ra), argparse.Namespace(**flags))   # flags added later: defaults
    else:
        side = pl.SideChannels(d, memory=ra["memory"], time=ra["time"], **pl.loop_kwargs(argparse.Namespace(**ra)))
    state = torch.load(Path(args.result).with_suffix(".pt"), map_location="cpu")
    ref_keys = None
    if sets:                                                       # what the overrides change at construction
        fresh = {k: v.clone() for k, v in side.state_dict().items()}
        torch.manual_seed(args.seed)
        ref_sd = Cand(d, argparse.Namespace(**ra), argparse.Namespace(**run_flags)).state_dict()
        ref_keys = set(ref_sd)
        clobbered = sorted(k for k in fresh if k in ref_sd and k in state and fresh[k].shape == ref_sd[k].shape
                           and not torch.equal(fresh[k], ref_sd[k]))
        if clobbered:                                              # init-only flags the checkpoint overwrites (F4)
            raise SystemExit(f"--w1s-set: {clobbered} are only initial values and the checkpoint overwrites "
                             "them; use --w1s-param NAME=VAL")
        del ref_sd
    missing, unexpected = side.load_state_dict(state, strict=False)
    extra_keys = []
    if args.w1s_load_extra:
        extra = torch.load(args.w1s_load_extra, map_location="cpu")
        foreign = sorted(set(extra) - set(side.state_dict()))
        if foreign:
            raise SystemExit(f"--w1s-load-extra: keys not in the side: {foreign}")
        side.load_state_dict(extra, strict=False)
        extra_keys = sorted(extra)
    if sets or extra_keys:                                         # only what the override adds may be missing
        expected = set()
        if sets:
            expected = set(side.state_dict()) - ref_keys
        got, expected = set(missing) - set(extra_keys), expected - set(extra_keys)   # --w1s-load-extra fills some
        if got != expected:
            raise SystemExit(f"missing keys {sorted(got)} differ from those the overrides add {sorted(expected)}")
    params = apply_params(side, args.w1s_param)
    side = side.to(dev).eval()
    oracle = oracle_gate(argparse.Namespace(**flags))
    external = flags.get("w1s_keep_by") == "external"
    if args.keep_oracle and not external:
        raise SystemExit("--keep-oracle needs --w1s-set w1s_keep_by=external (and --w1s-set w1s_keep=M)")
    if external and not args.keep_oracle:
        raise SystemExit("--w1s-keep-by external needs --keep-oracle (it reads the caller's keep scores)")
    if args.w1s_ctrl != "none":
        if not hasattr(side, "ctrl"):
            raise SystemExit("--w1s-ctrl needs a candidate with a ctrl attribute (W1s)")
        if args.w1s_ctrl == "shuffle" and (args.batch or 16) < 2:
            raise SystemExit("--w1s-ctrl shuffle needs --batch ≥ 2 (a row of its own reads its own store)")
        try:
            side.ctrl = args.w1s_ctrl
        except ValueError as e:                                    # shuffle under reinstatement
            raise SystemExit(str(e))
    W = ra["window"]
    LM = pl.HookedLM
    if hasattr(side, "taps"):                                      # W1s reads earlier blocks too
        from evolve.taps import TapLM
        LM = TapLM
    lm = LM(model, side, ra["layer"], W, ra["window_mode"], ra["window_layers"], 1, pad_id=tok.eos_token_id)
    P = args.pairs or ra["pairs"]
    L = args.seq_len or ra["seq_len"]
    if ra["window_mode"] in ("block", "segment") and L % W:
        raise SystemExit(f"--seq-len {L} must be a multiple of the window {W}")
    if args.batch is None and L > LONG:
        raise SystemExit(f"--seq-len {L} > {LONG}: give --batch explicitly (the default 16 may not fit)")
    batch = args.batch or 16
    corpus = None
    text_file = None if args.random_filler else (args.text_file or ra.get("text_file"))
    if text_file:
        corpus = tok.encode(Path(text_file).read_text(encoding="utf-8", errors="replace"))
    scatter = rec.get("needle_scatter", 0)
    variants = (args.jitter or args.update or args.heldout_names or oracle or args.order or args.decoys
                or args.keep_oracle == "offsets" or args.needle_tail is not None)
    if variants and not scatter:
        raise SystemExit("--jitter, --update, --order, --decoys, --needle-tail, --heldout-names, --keep-oracle "
                         "offsets and an oracle gate need a scattered-layout run")
    if args.jitter and args.update:
        raise SystemExit("--jitter and --update are separate tests")
    if args.order and (args.update or args.jitter or args.decoys):
        raise SystemExit("--order is a separate test (no --update, --jitter or --decoys)")
    if args.update_k != 1 and not args.update:
        raise SystemExit("--update-k needs --update")
    tail = args.needle_tail or rec.get("needle_tail", 256)
    pool = (HELDOUT_NAMES if args.heldout_names
            else TRAIN_NAMES if rec.get("needle_names") in ("train", "pool") else None)   # pool: training only
    kw = dict(min_gap=scatter, tail=tail, block=W, names=pool, decoys=args.decoys)
    try:
        if args.update:
            task = UpdateNeedleTask(tok, L, P, 1, corpus, form=args.update, k=args.update_k, **kw)
        elif args.order:
            task = OrderNeedleTask(tok, L, P, 1, corpus, kind=args.order, **kw)
        elif scatter:
            task = ScatterNeedleTask(tok, L, P, 1, corpus, jitter=args.jitter, **kw)
        else:
            task = pl.NeedleTask(tok, L, P, 1, corpus)
    except ValueError as e:
        raise SystemExit(str(e))
    R = P + args.decoys + (args.update_k if args.update else 0)    # statements, so queried ranks
    edges = [-1] + (args.gap_bins or [256 + i * (L - 256) // 5 for i in range(1, 5)]) + [10 ** 9]
    conds = {"windowed+side": (W, True)}
    if args.controls:
        conds.update({"windowed": (W, False), "full": (0, False)})
    ctrl_run = []
    if args.ctrl_conds:
        if not hasattr(side, "ctrl"):
            raise SystemExit("--ctrl-conds needs a candidate with a ctrl attribute (W1s)")
        if batch < 2:
            raise SystemExit("--ctrl-conds: side_shuffle needs --batch ≥ 2 (a row of its own reads its own store)")
        for c, mode in CTRL_CONDS.items():
            try:
                with ctrl_condition(side, c):
                    pass
            except ValueError:                                     # shuffle under reinstatement: no such control
                continue
            conds[c] = (W, True)
            ctrl_run.append(c)
    layers = keep_layers(args.keep_attn_layers, lm) if args.keep_oracle == "attn" else None
    gen = torch.Generator().manual_seed(args.seed) if args.keep_oracle == "random" else None

    def keep_for(x: torch.Tensor, vals: list) -> tuple[str, torch.Tensor] | None:
        """The external keep input for one batch: (side attribute, tensor) in side coordinates."""
        if args.keep_oracle == "random":
            return "keep_scores", torch.rand(x.shape[0], x.shape[1] - 1, generator=gen).to(dev)
        if args.keep_oracle == "offsets":                          # the space and digits 1–3: successors d1–d4
            tbl = torch.zeros(x.shape[0], x.shape[1] - 1)
            for b, vat in enumerate(vals):
                for v in vat:
                    tbl[b, v - 1:v + 3] = 1.0
            return "keep_scores", tbl.to(dev)
        if args.keep_oracle == "attn":
            return "keep_pair", received_attention(lm, x, W, layers, 1)
        return None

    def set_keep(k) -> None:
        if args.keep_oracle:
            side.keep_scores = side.keep_pair = None
            setattr(side, k[0], k[1])

    rng = random.Random(args.seed)
    per_seq = {c: [] for c in conds}                               # (rank, gap, digit-1 ok, exact ok)
    upd = {c: [] for c in conds}                                   # (new digit 1, old digit 1, new exact, old exact)
    eps_all, eps_d1 = [], []
    slots = {"seq": [], "align_within1": [], "null_weight": [], "p_d1": [], "p_fill": [], "onset_in_fact": [],
             "density": [], "fire_false": [], "fire_filler": [], "fire_true": []}
    theta = float(side.loop.sup_theta) if hasattr(getattr(side, "loop", None), "sup_theta") else 0.9   # DEC9 θ
    events = hasattr(side.loop, "event_strength") and not hasattr(side, "taps")   # never W1s (B17)
    for _ in range(args.n // batch):
        rows, js, vals, gaps, spans, qlens, olds, same = [], [], [], [], [], [], [], []
        for _ in range(batch):
            ids, n_ans, gap = task.sample(rng)
            gaps.append(gap)
            if scatter:
                j, vat = task.last_queried, task.last_value_at
                spans.append(task.last_spans), qlens.append(task.last_query_len)
                same.append(set(getattr(task, "last_update_ranks", None) or [])
                            | ({task.last_old_rank, j} if args.update else set()))   # the updated name's statements
                if args.update:                                    # the same sequence answered with the old code
                    if len(task.last_old_answer) != n_ans:
                        raise SystemExit("old and new codes tokenise to different lengths")
                    olds.append(ids[:-n_ans] + task.last_old_answer)
            else:                                                  # fixed layout: recover j from the text
                text = tok.tok.decode(ids)
                names = [n for n in pl.NAMES if f"for {n} is" in text]
                q = text.rsplit("The code for ", 1)[1].split(" is")[0]
                order = sorted(names, key=lambda n: text.index(f"for {n} is"))
                j, vat = order.index(q), None
            rows.append(ids), js.append(j), vals.append(vat)
        ids = torch.tensor(rows)
        old = torch.tensor(olds) if args.update else None
        if oracle:                                                 # diagnostic gate: the true first digits
            side.oracle_onsets = oracle_onsets(vals, ids.shape[1], 1, dev)
        keep = keep_for(ids, vals) if args.keep_oracle else None
        keep_old = None
        if args.keep_oracle and args.update:                      # same layout; attention is recomputed
            keep_old = keep_for(old, vals) if args.keep_oracle == "attn" else keep
        diag = nov = None
        for c, (w, s_on) in conds.items():
            with ctrl_condition(side, c):
                set_keep(keep)
                with torch.no_grad(), lm.condition(w, s_on):
                    ok = _score(lm(ids, keep=n_ans + 1), ids, n_ans)  # only the answer positions' logits
                if c == "windowed+side":                           # recorded in the side's own forward
                    diag = getattr(side.loop, "last_diag", None)
                    nov = getattr(side.loop, "last_novelty", None)
                for i, j in enumerate(js):                         # ok[i][0] is the space; [1] is digit 1
                    per_seq[c].append((j, gaps[i], float(ok[i][1]), float(all(ok[i]))))
                if args.update:                                    # digit 1 is predicted before either code
                    set_keep(keep_old)
                    with torch.no_grad(), lm.condition(w, s_on):
                        ok_old = _score(lm(old, keep=n_ans + 1), old, n_ans)
                    upd[c] += [(float(ok[i][1]), float(ok_old[i][1]), float(all(ok[i])), float(all(ok_old[i])))
                               for i in range(len(js))]
        if isinstance(diag, dict) and scatter:
            _slot_diag(diag, js, vals, spans, qlens, ids.shape[1], n_ans, 1, slots, same, theta,
                       side.loop.P, side.loop.pool)
        if events:
            with torch.no_grad():
                h_tap = {}
                hook = lm.blocks[lm.layer].register_forward_hook(
                    lambda m, i, o: h_tap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).float()))
                with lm.condition(W, False):
                    lm(ids, keep=1)
                hook.remove()
                h = h_tap["h"][:, 1:]                              # the sink dropped, as the side sees it (B17)
                x = side.loop.norm(side.time(h)[0] if side.time is not None else h)
                extra = None                                       # W1j: the CA1 novelty in the logit (B17)
                if nov is not None and hasattr(side.loop, "kappa"):
                    extra = side.loop.kappa[None, :, None] * nov
                eps = side.loop.event_strength(x, extra).cpu()    # (B, H, T), side coordinates
            eps_all.append(eps.mean().item())
            if scatter:
                d1 = [list(vat) for vat in vals]                  # value_at is the space (with the sink): − 1 + 1
                eps_d1.append(torch.stack([eps[i, :, d1[i]].mean() for i in range(len(d1))]).mean().item())
    if args.keep_oracle:
        side.keep_scores = side.keep_pair = None
    main_ = _summarise(per_seq["windowed+side"], R, edges)
    out = {"result": args.result, "pairs": P, "seq_len": L, "text_file": text_file, "scatter": scatter,
           "n": len(per_seq["windowed+side"]),
           # the side-channel model (kept at the top level, as in earlier rounds)
           "digit1_acc_by_queried_rank": main_["digit1_by_queried_rank"],
           "exact_by_queried_rank": main_["exact_by_queried_rank"],
           "count_by_queried_rank": main_["count_by_queried_rank"],
           "digit1": main_["digit1"], "exact": main_["exact"], "by_gap": main_["by_gap"],
           "controls": {c: _summarise(per_seq[c], R, edges) for c in conds if c != "windowed+side"},
           "per_sequence": per_seq["windowed+side"],
           "missing_keys": list(missing), "unexpected_keys": list(unexpected)}
    nondefault = args.update_k != 1 or args.decoys or args.order
    if args.jitter or args.update or pool or nondefault:
        out["task"] = {"jitter": args.jitter, "update": args.update,
                       "names": "heldout" if args.heldout_names else "train" if pool else "all"}
        if rec.get("needle_names") == "pool":                      # trained on the syllable pool
            out["task"]["trained_names"] = "pool"
        if args.update_k != 1:
            out["task"]["update_k"] = args.update_k
        if args.decoys:
            out["task"]["decoys"] = args.decoys
        if args.order:
            out["task"]["order"] = args.order
    if tail != 256:
        out["needle_tail"] = tail
    rule = getattr(task, "names_rule", None)
    if rule:
        out["names_rule"] = rule
    if args.update:                                                # new = the current code; old = stale
        out["update"] = {c: dict(zip(("new_digit1", "old_digit1", "new_exact", "old_exact"),
                                     (sum(r[i] for r in v) / max(1, len(v)) for i in range(4))))
                         for c, v in upd.items()}
    if args.w1s_ctrl != "none":
        out["w1s_ctrl"] = args.w1s_ctrl
    if ctrl_run:
        out["ctrl_conds"] = ctrl_run
    if oracle:
        out["oracle_gate"] = True
    if sets:
        out["w1s_set"] = {k: v for k, _, v in sets}
    if params:
        out["w1s_param"] = params
    if extra_keys:
        out["w1s_load_extra"] = {"path": args.w1s_load_extra, "keys": extra_keys}
    if args.attn_impl and args.attn_impl != ra.get("attn_impl", "sdpa"):
        out["attn_impl"] = args.attn_impl
    if args.keep_oracle:
        out["keep_oracle"] = args.keep_oracle if layers is None else {"mode": "attn", "layers": layers}
    if hasattr(side, "state_report"):                              # streaming state size (T-independent)
        out["w1s_state"] = side.state_report()
    if slots["seq"]:
        out["w1s"] = _slot_summary(slots, R)
    if events:
        out["eps_mean_all_tokens"] = sum(eps_all) / len(eps_all)
        if eps_d1:
            out["eps_mean_at_fact_digit1"] = sum(eps_d1) / len(eps_d1)
    upd_name = f"update-{args.update}" + (f"-k{args.update_k}" if args.update_k != 1 else "")
    variant = "".join(f".{name}" for name, on in (("jitter", args.jitter), (upd_name, args.update),
                                                    ("heldout", args.heldout_names),
                                                    (f"ctrl-{args.w1s_ctrl}", args.w1s_ctrl != "none"),
                                                    (f"order-{args.order}", args.order),
                                                    (f"decoys{args.decoys}", args.decoys),
                                                    (f"tail{tail}", tail != rec.get("needle_tail", 256)),
                                                    (f"attn-{attn_impl}", "attn_impl" in out),
                                                    (f"keep-{args.keep_oracle}", args.keep_oracle),
                                                    (f"seed{args.seed}", args.seed != 2024)) if on)
    variant += "".join(f".set-{_tag(k)}-{_tag(v)}" for k, v, _ in sets)
    variant += "".join(f".param-{_tag(p['name'])}-{'add' if p['op'] == '+=' else 'set'}{_tag(p['value'])}"
                       for p in params)
    if extra_keys:
        variant += f".extra-{_tag(Path(args.w1s_load_extra).stem)}"
    out["eval"] = {"seed": args.seed, "n_requested": args.n, "batch": batch,   # the settings (review F6)
                   "keep_attn_layers": args.keep_attn_layers if args.keep_oracle == "attn" else None,
                   "allow_random_code": args.allow_random_code}
    path = Path(args.out or Path(args.result).with_suffix(f".by_query{variant}.json"))   # variants kept apart
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    summary = {k: out[k] for k in ("pairs", "seq_len", "exact", "digit1", "digit1_acc_by_queried_rank")}
    summary["controls_exact"] = {c: round(v["exact"], 3) for c, v in out["controls"].items()}
    if "update" in out:
        summary["update"] = {k: round(v, 3) for k, v in out["update"]["windowed+side"].items()}
    if "w1s" in out:
        summary["w1s"] = {k: v if v is None else round(v, 3) for k, v in out["w1s"].items() if not isinstance(v, list)}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
