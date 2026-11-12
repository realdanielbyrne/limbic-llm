"""Thinking-control runner (round 12, C1 decoding grid now, C3-core controllers later), built on reason.py.

An arm is a decoding setting, a thinking mode, an optional real truncation and an optional stop controller:
    <label>@k=v;k=v;...        keys (defaults in brackets):
        T [0.6]  top_p [0.95]  top_k [20]  pp [0] presence penalty  fp [0] frequency penalty
        rp [1.0] transformers' repetition penalty (CTRL; prompt included)  think [on] | off
        trunc=<B>               real truncation: once B reasoning tokens are generated, the canonical close
                                (``decode.CANONICAL_CLOSE``) is forced and the answer is sampled under the same
                                decoding, inside the same ``generate`` call and under the same overall cap
                                (``--max-new-tokens``) as the untruncated arm, so the answer is never cut by a
                                separate answer budget and rows the controller never touched are identical to the
                                untruncated arm's (tokens before B are those of the untruncated arm, same seed)
        ctrl=<det>[+<det>]:<act> detectors: ngram-W-n-rho (``decode.NgramLoopRule``), budget-B, consist-k and
                                conf-tau (``decode.ProbeRule``, fed by forked probes every --probe-every tokens);
                                actions: stop (force the canonical close), bias<beta> (``ThinkCloseBias``), none
                                (record only). trunc and ctrl combine as one controller (any detector fires);
                                with trunc the action must be stop.
        suppress=wait           NoWait: "wait" / "hmm" / "alternatively" tokens set to −inf inside the span
Presets (``--preset``): c1 = (T, pp) ∈ {(0.6, 0), (1.0, 0), (1.0, 1.5)} with thinking on, thinking off at
(0.6, 0) (the setting of reason.py's running ``nothink`` arm), and the real truncation at 1,024 under (1.0, 1.5);
reference arm t06 (the running baselines' decoding). Data default: the first 200 of reason.py's
``gsm8k:train:400`` subset (data seed 0).

Generation. Prompts are reason.py's ``render`` (thinking switch explicit; the prompt ends in ``<think>\\n``, so for
Qwen3 too the span opens in the prompt, as in the running baselines); for models whose thinking markers are
not ``<think>`` / ``</think>`` a hook (``MODEL_HOOKS``) gives the markers and renders with the template alone, the
model then opening the span itself (``SpanTracker`` handles both). Left-padded batches of ``--batch`` rows,
sampled with ``generate``; seeds per batch as reason.py (seed·1,000,003 + sample·7,919 + batch), so arms see the
same problems, batches and seeds and differences are paired. Processors: ``decode.build_processors`` (fixed
order; the presence penalty acts on the whole generated response, OpenAI / vLLM semantics). Stop ids, pad and
the scoring are reason.py's.

Forced-answer probe (``--probe-every`` E = 128 span tokens, as reason.py: cue ``</think>`` + "\\n\\nThe final
answer is \\boxed{", ``--probe-tokens`` greedy tokens). ``--probe-mode fork`` (default) runs it online on a fork of
the generation cache after every E generated tokens (``decode.generate_segmented``; it leaves the generated
tokens unchanged) and re-prefills only the checkpoints left (the final one at n_reason); ``prefill`` re-prefills
every checkpoint after generation, exactly as reason.py. Checkpoints are ``reason.checkpoints(n_reason, E)``.
The probe also returns the answer's confidence (geometric-mean p of its tokens; DEER) and first-token entropy.

Per item (results JSON as per-item arrays, no text; text and per-token arrays only in ``--trace-dir``):
everything reason.py records for unsteered arms (correct, answer in the trace, n_reason, n_gen, truncated,
closed, cks / probe answers / ok, ttc and censoring, acc@b, flips, backtracking, repetition onset, distinct-4,
self-loop rate, agreement, wall time), plus: forced (a controller forced the close), natural_close (the model
closed the span itself; None with thinking off), closed_after_act (a close sampled after a bias controller
acted), n_forced,
probe_conf, tokens_total = n_gen + (probes a probe-driven controller consulted) × ``--probe-cost`` token
equivalents, and the controller's record (detector, action, detected and acting step, span tokens, why).
Per token (``common.TraceWriter``; reason.py's layout so ``stoprules.py`` reads both): ``bk/*`` from
``reason.bookkeeping`` and, from the sampling pass itself (raw logits, T = 1), ``out/ent`` and ``out/logp_next``
at the position predicting the next token (as reason.py's teacher-forced arrays) and ``out/p_close``.
Per arm: ``reason.aggregate`` plus non-closure, forced and token totals (and, over forced rows, the share whose
answer the cap cut, their accuracy and their raw accuracy); against the reference arm (``--ref``,
default the first): ``reason.compare`` (paired bootstrap, clusters = problems) plus Δ natural closure, Δ n_gen
and Δ tokens_total. The JSON is rewritten after every arm; complete arms are skipped on a rerun, and a rerun
with different settings (``RESUME_KEYS``) is refused unless ``--force``.

Usage
  CUDA_VISIBLE_DEVICES="" python experiments/ca2/think.py --tiny --preset c1 --out /tmp/x/t.json \\
      --trace-dir /tmp/x/traces
  CUDA_VISIBLE_DEVICES=1 python experiments/ca2/think.py --model Qwen/Qwen3.5-2B --preset c1 --batch 32 \\
      --probe-batch 16 --name c1_q35 2>&1 | tee results/evolve/round12/stage1/c1_q35.log
  CUDA_VISIBLE_DEVICES=0 python experiments/ca2/think.py --model Qwen/Qwen3-1.7B --preset c1 --batch 16 \\
      --probe-batch 8 --name c1_q3 2>&1 | tee results/evolve/round12/stage1/c1_q3.log
(batches as the running baselines, reason_base_q35 / reason_base_q3, so that the t06 arm draws the same seeds per
batch as their ``none`` arm; the forked probe adds 25 tokens of cache per row, the re-prefill leftovers are one
final checkpoint per row.)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
sys.path.insert(0, str(EXP))
from ca2 import decode as D  # noqa: E402
from ca2 import reason as R  # noqa: E402
from ca2.common import TraceWriter, alignment, git_state, paired_boot  # noqa: E402

PRESETS = {
    "c1": ["t06@T=0.6;pp=0", "t10@T=1.0;pp=0", "t10pp15@T=1.0;pp=1.5", "nothink@think=off;T=0.6;pp=0",
           "t10pp15-trunc1024@T=1.0;pp=1.5;trunc=1024"],
}
DEFAULT_PROBE_COST = 25        # Qwen: </think> + 8 cue tokens prefilled on the fork, + 16 greedy probe tokens

# Models whose thinking markers are not <think>/</think>. Gemma 4 (google/gemma-4-E2B-it), read on the Hub without
# downloading (chat_template.jinja of 2026-07-09, tokenizer_config.json, generation_config.json): enable_thinking
# puts "<|think|>" at the top of a system turn; the prompt ends in "<|turn>model\n" and the model opens the span
# itself with "<|channel>" + "thought\n" and closes it with "<channel|>" (tokenizer soc / eoc tokens); the turn
# ends with "<turn|>" (eot; generation_config eos ids 1, 106, 50); card sampling T 1.0, top-p 0.95, top-k 64.
# NOT verified until the model is downloaded: that the markers are single tokens (checked at load, else the run
# stops), that AutoModelForCausalLM and pretrained_loop.find_blocks / layer_kinds load and read this multimodal
# checkpoint, one rendered prompt and one generated trace by hand, and that "thought\n" after the open marker is
# counted as reasoning (2 tokens; it is).
MODEL_HOOKS = {
    "gemma-4": {"open": "<|channel>", "close": "<channel|>", "stops": ("<turn|>",),
                "template_kwargs": {"enable_thinking": True}, "verified": "markers from the Hub template only"},
}


# ------------------------------------------------------------------------------------------------- arms
@dataclass
class ThinkArm:
    spec: str
    label: str
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence: float = 0.0
    frequency: float = 0.0
    repetition: float = 1.0
    thinking: bool = True
    trunc: int | None = None
    ctrl: str | None = None
    suppress: str | None = None
    key: str = ""

    def __post_init__(self) -> None:
        self.key = R.safe_key(self.label)

    def decoding(self) -> dict:
        return {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k, "presence": self.presence,
                "frequency": self.frequency, "repetition": self.repetition}


def parse_arm(spec: str) -> ThinkArm:
    """One arm from ``<label>@k=v;...`` (module docstring)."""
    label, _, body = spec.strip().partition("@")
    if not label or not re.fullmatch(r"[A-Za-z0-9._-]+", label):
        raise ValueError(f"{spec!r}: the label must be [A-Za-z0-9._-]+")
    arm = ThinkArm(spec.strip(), label)
    conv = {"T": ("temperature", float), "top_p": ("top_p", float), "top_k": ("top_k", int),
            "pp": ("presence", float), "fp": ("frequency", float), "rp": ("repetition", float),
            "trunc": ("trunc", int), "ctrl": ("ctrl", str), "suppress": ("suppress", str)}
    for kv in [x for x in body.split(";") if x.strip()]:
        k, eq, v = kv.partition("=")
        k, v = k.strip(), v.strip()
        if not eq:
            raise ValueError(f"{spec!r}: {kv!r} is not k=v")
        if k == "think":
            if v not in ("on", "off"):
                raise ValueError(f"{spec!r}: think=on|off")
            arm.thinking = v == "on"
        elif k in conv:
            setattr(arm, conv[k][0], conv[k][1](v))
        else:
            raise ValueError(f"{spec!r}: unknown key {k!r}; keys: think, {', '.join(conv)}")
    if arm.ctrl is not None:
        parse_ctrl(arm.ctrl)
    if arm.suppress not in (None, "wait"):
        raise ValueError(f"{spec!r}: suppress=wait")
    if not arm.thinking and (arm.trunc or arm.ctrl or arm.suppress):
        raise ValueError(f"{spec!r}: trunc, ctrl and suppress act on the reasoning span (think=on)")
    if arm.trunc is not None and arm.ctrl is not None and parse_ctrl(arm.ctrl)[1] != "stop":
        raise ValueError(f"{spec!r}: with trunc the controller's action must be stop")
    return arm


def parse_ctrl(s: str) -> tuple[list[tuple[str, tuple]], str, float]:
    """``det[+det]:action`` -> ([(kind, params)], action, beta)."""
    dets, _, act = s.partition(":")
    out = []
    for d in dets.split("+"):
        p = d.split("-")
        if p[0] == "ngram" and len(p) == 4:
            out.append(("ngram", (int(p[1]), int(p[2]), float(p[3]))))
        elif p[0] == "budget" and len(p) == 2:
            out.append(("budget", (int(p[1]),)))
        elif p[0] == "consist" and len(p) == 2:
            out.append(("consist", (int(p[1]),)))
        elif p[0] == "conf" and len(p) == 2:
            out.append(("conf", (float(p[1]),)))
        else:
            raise ValueError(f"controller detector {d!r}: ngram-W-n-rho | budget-B | consist-k | conf-tau")
    m = re.fullmatch(r"stop|none|bias(-?[\d.]+)", act)
    if not m:
        raise ValueError(f"controller action {act!r}: stop | none | bias<beta>")
    return out, ("bias" if act.startswith("bias") else act), float(m.group(1)) if m.group(1) else 0.0


def build_detector(arm: ThinkArm, B: int, P: int) -> tuple[D.Detector | None, str, float, list[D.ProbeRule]]:
    """The arm's detector (AnyOf when several), action, beta and its probe-fed parts."""
    parts: list[D.Detector] = []
    action, beta = "stop", 0.0
    if arm.trunc is not None:
        parts.append(D.BudgetRule(B, arm.trunc))
    if arm.ctrl is not None:
        dets, action, beta = parse_ctrl(arm.ctrl)
        for kind, prm in dets:
            if kind == "ngram":
                parts.append(D.NgramLoopRule(B, P, *prm))
            elif kind == "budget":
                parts.append(D.BudgetRule(B, *prm))
            elif kind == "consist":
                parts.append(D.ProbeRule(B, "consist", k=prm[0]))
            else:
                parts.append(D.ProbeRule(B, "conf", tau=prm[0]))
    if not parts:
        return None, action, beta, []
    det = parts[0] if len(parts) == 1 else D.AnyOf(B, parts)
    return det, action, beta, [p for p in parts if isinstance(p, D.ProbeRule)]


# ------------------------------------------------------------------------------------------- environment
@dataclass
class Model:
    env: R.Env
    open_id: int | None                  # span-open marker (None: tiny, state given by the render)
    hook: dict | None
    facts: dict = field(default_factory=dict)


def env_args(args) -> argparse.Namespace:
    """The namespace ``reason.load_env`` reads (no taps: nothing is steered here)."""
    return argparse.Namespace(tiny=args.tiny, tiny_arch=args.tiny_arch, seed=args.seed, dtype=args.dtype,
                              device=args.device, model=args.model, attn_impl=args.attn_impl, tap=None,
                              inject_before=None, no_attn_tap=True, attn_layer=None, render="auto",
                              thinking="auto", lens_layers=None, release_words="")


def model_hook(name: str | None) -> dict | None:
    n = (name or "").lower()
    for k, h in MODEL_HOOKS.items():
        if k in n:
            return h
    return None


def load_model(args) -> Model:
    env = R.load_env(env_args(args))
    hook = None if args.tiny else model_hook(args.model)
    open_id = None if args.tiny else R._special_id(env.tok, "<think>")
    if hook is not None or args.span_open or args.span_close:
        h = dict(hook or {})
        o, c = args.span_open or h.get("open"), args.span_close or h.get("close")
        oid, cid = R._special_id(env.tok, o) if o else None, R._special_id(env.tok, c) if c else None
        if oid is None or cid is None:
            raise SystemExit(f"{args.model}: thinking markers {o!r} / {c!r} are not single tokens of this tokenizer; "
                             "check the model's chat template and pass --span-open / --span-close")
        env.span_end, env.thinking = cid, True
        env.has_switch = "enable_thinking" in str(getattr(R._hf(env.tok), "chat_template", ""))
        extra = [R._special_id(env.tok, s) for s in h.get("stops", ())]
        env.stops = list(dict.fromkeys(env.stops + [i for i in extra if i is not None]))
        open_id, hook = oid, {**h, "open": o, "close": c, "open_id": oid, "close_id": cid}
    if env.span_end is None:
        raise SystemExit(f"{args.model}: no end-of-thinking token; think.py needs a thinking model")
    facts = {**env.facts, "think_end_id": env.span_end, "think_open_id": open_id, "stop_ids": env.stops,
             "thinking_switch": env.has_switch, "hook": hook}
    return Model(env, open_id, hook, facts)


def render(mdl: Model, question: str, thinking: bool, instruction: str) -> list[int]:
    """reason.py's ``render``; with a hook, the template alone (the model opens the span)."""
    env = mdl.env
    if mdl.hook is None:
        return R.render(env, question, thinking, instruction)
    text = f"{question}\n{instruction}" if instruction else question
    kw = dict(mdl.hook.get("template_kwargs", {}))
    kw["enable_thinking"] = thinking
    s = R._hf(env.tok).apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                           add_generation_prompt=True, **kw)
    return env.tok.encode(s)


# ------------------------------------------------------------------------------------------- reading
def span_index(resp: Sequence[int], start_open: bool, open_id: int | None, close_id: int | None,
               skip: tuple[int, int] | None = None) -> tuple[list[int], int | None]:
    """(pos, close_at): pos[c] = response index after the c-th reasoning token (pos[0] = index of the first),
    close_at = index of the first close marker after the span (None if none). Forced tokens in ``skip``
    = [a, b) are not reasoning tokens."""
    pos, state, close_at = [], start_open, None
    for j, t in enumerate(resp):
        if close_id is not None and t == close_id:
            if state and close_at is None:
                close_at = j
            state = False
            continue
        if open_id is not None and t == open_id:
            state = True
            continue
        if state and not (skip and skip[0] <= j < skip[1]):
            pos.append(j)
    return pos, close_at


@torch.no_grad()
def run_batch(mdl: Model, arm: ThinkArm, rows: list[tuple[R.Item, int]], bi: int, args, writer: TraceWriter
              ) -> list[dict]:
    env, tok = mdl.env, mdl.env.tok
    B = len(rows)
    prompts = [render(mdl, it.question, arm.thinking, args.instruction) for it, _ in rows]
    ids, mask = R.pad_left(prompts, env.pad)
    P = ids.shape[1]
    close_id = env.span_end if arm.thinking else None
    span = D.SpanTracker(close_id, P, open_id=mdl.open_id,
                         prompt_open=None if (mdl.open_id is not None and close_id is not None) else arm.thinking,
                         stops=env.stops)                    # finished rows: span closed, no controller acts
    span.state(ids)                                          # fixes the prompt's span state per row
    det, action, beta, prules = build_detector(arm, B, P)
    close_ids = D.close_sequence(tok, env.span_end) if arm.thinking else []
    online = arm.thinking and args.probe_every > 0 and (args.probe_mode == "fork" or bool(prules))
    procs, parts = D.build_processors(
        prompt_len=P, batch=B, span=span, presence=arm.presence, frequency=arm.frequency, repetition=arm.repetition,
        suppress=D.wait_token_ids(tok) if arm.suppress == "wait" else (),
        controller=(det, action, beta) if det is not None else None, close_id=close_id, close_ids=close_ids,
        finished=(env.stops, env.pad) if online else None)
    gk = ({"do_sample": True, "temperature": arm.temperature, "top_p": arm.top_p, "top_k": arm.top_k}
          if arm.temperature > 0 else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None})
    n_new = args.max_new_tokens                              # one cap for every arm (a forced close included)
    seed = args.seed * 1_000_003 + rows[0][1] * 7919 + bi
    cue = D.cue_sequence(tok, close_id)
    online_res: list[dict[int, D.ProbeResult]] = [dict() for _ in range(B)]
    ctrl = parts.get("controller")

    def on_probe(rec: dict) -> None:
        live = [b for b in range(B) if rec["open"][b] and (ctrl is None or ctrl.act_step[b] < 0)]
        for b in live:
            online_res[b][int(rec["n_span"][b])] = rec["results"][b]
        for pr in prules:
            pr.observe(live, [rec["results"][b].answer for b in live], [rec["results"][b].conf for b in live],
                       rec["t"], [rec["n_span"][b] for b in live])

    torch.manual_seed(seed)
    R._sync(env.dev)
    t0 = time.time()
    if online:
        so = D.generate_segmented(env.model, ids.to(env.dev), mask.to(env.dev), n_new, args.probe_every, gk, procs,
                                  env.stops, env.pad, span=span, on_probe=on_probe, sync=lambda: R._sync(env.dev),
                                  probe=lambda c, s, m: D.forked_probe(env.model, tok, c, s, m, cue,
                                                                       args.probe_tokens, env.stops, env.pad))
        new, t_probe_online = so.new.cpu(), so.seconds_probe
        n_online = [sum(1 for p in so.probes if p["open"][b]) for b in range(B)]
    else:
        out = env.model.generate(input_ids=ids.to(env.dev), attention_mask=mask.to(env.dev), max_new_tokens=n_new,
                                 logits_processor=procs, eos_token_id=env.stops, pad_token_id=env.pad, **gk)
        new, t_probe_online, n_online = out[:, P:].cpu(), 0.0, [0] * B
    R._sync(env.dev)
    t_gen = time.time() - t0 - t_probe_online
    stats = parts["stats"]
    if new.shape[1]:
        stats.finish(new[:, -1].to(env.dev))
    st_arr = stats.arrays()
    # per-row reading
    recs, jobs, job_of = [], [], []
    for b, (it, k) in enumerate(rows):
        r, stopped = R.cut_stop(new[b].tolist(), env.stops)
        acted = ctrl is not None and ctrl.act_step[b] >= 0
        forced = bool(acted and action == "stop")
        skip = (int(ctrl.act_step[b]), int(ctrl.act_step[b]) + len(close_ids)) if forced else None
        start_open = bool(span.prompt_open[b])
        pos, close_at = span_index(r, start_open, mdl.open_id, close_id, skip) if arm.thinking else \
            (list(range(len(r) - (1 if stopped else 0))), None)
        n_reason = len(pos)
        body = [t for t in (r[close_at + 1:] if close_at is not None else r) if t not in env.stops]   # as reason.py
        ans = R.extract_answer(tok.decode(body))
        truncated = not stopped
        ok = R.is_correct(ans, it.gold, it.kind) and not truncated
        probed = bool(arm.thinking and args.probe_every > 0 and k < args.probe_samples)
        cks = R.checkpoints(n_reason, args.probe_every) if probed else []
        mine = []
        for c in cks:
            if c in online_res[b]:
                mine.append(("online", online_res[b][c]))
            else:
                cut = pos[c - 1] + 1 if c >= 1 else 0
                mine.append(("job", len(jobs)))
                jobs.append(prompts[b] + r[:cut])
        job_of.append(mine)
        span_text = tok.decode([r[j] for j in pos])
        bt = R.backtrack_counts(span_text)
        resp_ids = [t for t in r if t not in env.stops]
        rec = dict(uid=it.uid, sample=k, gold=it.gold, answer=ans, correct=bool(ok),
                   raw_correct=bool(R.is_correct(ans, it.gold, it.kind)), truncated=truncated,
                   closed=close_at is not None,
                   natural_close=bool(close_at is not None and not forced) if arm.thinking else None,
                   closed_after_act=bool(not forced and close_at is not None
                                         and close_at >= int(ctrl.act_step[b]))
                   if acted else None,
                   forced=forced, n_forced=len(close_ids) if forced else 0, n_reason=int(n_reason), n_gen=len(r),
                   n_prompt=len(prompts[b]), forced_from=skip[0] if forced else None, cks=cks, probed=probed,
                   backtrack=bt, backtrack_rate=1000.0 * sum(bt[x] for x in R.BACKTRACK_CORE) / max(1, n_reason),
                   rep_onset=alignment().judge.repetition_onset(resp_ids), distinct4=R.distinct_n(resp_ids, 4),
                   self_loop=R.self_loop_rate(span_text), text=tok.decode(resp_ids), seed=seed, batch=bi, row=b,
                   ctrl=ctrl.record(b) if ctrl is not None else None)
        n_ctrl_probes = 0
        if prules and online:
            last = int(ctrl.act_step[b]) if acted else None
            n_ctrl_probes = sum(1 for p in so.probes if p["open"][b] and (last is None or p["t"] <= last))
        rec["n_ctrl_probes"] = n_ctrl_probes
        rec["n_online_probes"] = n_online[b]
        rec["tokens_total"] = len(r) + n_ctrl_probes * args.probe_cost
        recs.append(rec)
    t2 = time.time()
    done = D.probe_prefixes(env.model, tok, jobs, cue, args.probe_tokens, env.stops, env.pad, args.probe_batch,
                            env.dev) if jobs else []
    t_probe = time.time() - t2 + t_probe_online
    text_of = R.TokText(tok)
    tot = max(1, sum(len(R.cut_stop(new[b].tolist(), env.stops)[0]) for b in range(B)))
    for b, (rec, mine) in enumerate(zip(recs, job_of)):
        res = [x if src == "online" else done[x] for src, x in mine]
        rec["probe_answers"] = [p.answer for p in res]
        rec["probe_ok"] = [bool(R.is_correct(p.answer, rec["gold"], rows[b][0].kind)) for p in res]
        rec["probe_conf"] = [p.conf for p in res]
        rec["probe_ent"] = [p.ent_first for p in res]
        rec["probe_online"] = [src == "online" for src, _ in mine]
        if res:
            rec["ttc"], rec["censored"] = R.ttc_from(rec["cks"], rec["probe_ok"])
            rec["flips"] = R.count_flips(rec["probe_answers"])
        else:
            rec["ttc"], rec["censored"], rec["flips"] = None, None, None
        rec["acc_budget"] = (R.budget_acc(args.budget_list, rec["cks"], rec["probe_ok"], rec["n_reason"],
                                          rec["closed"] and not rec["truncated"], rec["correct"])
                             if rec["probed"] else {bb: None for bb in args.budget_list})
        share = rec["n_gen"] / tot
        rec["wall_batch_s"] = t_gen
        rec["wall_s"] = rec["wall_net_s"] = t_gen * share
        rec["probe_s"] = t_probe * share
        # per-token arrays in reason.py's layout
        r = R.cut_stop(new[b].tolist(), env.stops)[0]
        seq = prompts[b] + r
        Np = len(prompts[b])
        bk = R.bookkeeping(seq, Np, rec["n_reason"], rec["n_reason"] if rec["closed"] else None, rec["forced_from"],
                           rec["cks"], rec["probe_ok"], text_of)
        n = len(seq)
        toks = dict(bk)
        for key, src in (("out/ent", "ent"), ("out/logp_next", "logp"), ("out/p_close", "p_close")):
            a = np.full(n, np.nan, np.float32)
            m = min(len(r), st_arr[src].shape[1] if st_arr[src].size else 0)
            if m:
                a[Np - 1: Np - 1 + m] = st_arr[src][b, :m]
            toks[key] = a
        writer.add("tokens", rec, toks)
    return recs


# ------------------------------------------------------------------------------------------- arm loop
EXTRA_ITEM = ("forced", "natural_close", "closed_after_act", "n_forced", "tokens_total", "n_ctrl_probes",
              "n_online_probes")


def per_item(recs: list[dict], budgets: Sequence[int]) -> dict:
    pi = R.per_item_arrays(recs, budgets)
    for k in EXTRA_ITEM:
        pi[k] = [r.get(k) for r in recs]
    pi["probe_conf"] = [r.get("probe_conf") for r in recs]
    pi["probe_ok"] = [r.get("probe_ok") for r in recs]
    pi["cks"] = [r.get("cks") for r in recs]
    for k in ("detector", "action", "detected_step", "detected_span", "act_step", "act_span", "why"):
        pi[f"ctrl_{k}"] = [(r["ctrl"] or {}).get(k) for r in recs]
    return pi


def aggregate(pi: dict, budgets: Sequence[int], tmax: float) -> dict:
    agg = R.aggregate(pi, budgets, tmax)
    nat = [float(x) for x in pi["natural_close"] if x is not None]          # None: thinking off (no span)
    agg["natural_close_rate"] = R._mean(nat)
    agg["closed_after_act_rate"] = R._mean([float(x) for x in pi["closed_after_act"] if x is not None])
    agg["nonclose_rate"] = None if agg["natural_close_rate"] is None else 1.0 - agg["natural_close_rate"]
    agg["forced_rate"] = R._mean([float(x) for x in pi["forced"]])
    fz = [i for i, x in enumerate(pi["forced"]) if x]                       # rows with a forced close
    agg["forced_truncated_rate"] = R._mean([float(pi["truncated"][i]) for i in fz])    # answer cut by the cap
    agg["forced_acc"] = R._mean([float(pi["correct"][i]) for i in fz])
    agg["forced_raw_acc"] = R._mean([float(pi["raw_correct"][i]) for i in fz])         # answer right, cap ignored
    agg["tokens_total_mean"] = R._mean(pi["tokens_total"])
    agg["n_gen_median"] = float(np.median(pi["n_gen"])) if pi["n_gen"] else None
    acted = [x for x in pi["ctrl_act_step"] if x is not None and x >= 0]
    agg["ctrl_act_rate"] = (len(acted) / len(pi["uid"])) if pi["uid"] and any(
        x is not None for x in pi["ctrl_act_step"]) else None
    agg["ctrl_act_step_mean"] = R._mean(acted)
    return agg


def compare(pi: dict, ref: dict, budgets: Sequence[int], seed: int) -> dict:
    out = R.compare(pi, ref, budgets, seed)
    key = lambda p, i: (p["uid"][i], p["sample"][i])
    at = {key(ref, i): i for i in range(len(ref["uid"]))}
    pairs = [(i, at[key(pi, i)]) for i in range(len(pi["uid"])) if key(pi, i) in at]
    if not pairs:
        return out
    cl = [pi["uid"][i] for i, _ in pairs]
    for name in ("natural_close", "n_gen", "tokens_total"):
        a = [float("nan") if pi[name][i] is None else float(pi[name][i]) for i, _ in pairs]
        b = [float("nan") if ref[name][j] is None else float(ref[name][j]) for _, j in pairs]
        out[f"d_{name}"] = paired_boot(a, b, clusters=cl, seed=seed)
    return out


def run_arm(mdl: Model, arm: ThinkArm, items: list[R.Item], args) -> dict:
    trace_dir = Path(args.trace_dir).expanduser()
    adir = trace_dir / arm.key
    if adir.exists():
        shutil.rmtree(adir)                                  # an incomplete or forced arm starts clean
    writer = TraceWriter(arm.key, root=trace_dir, meta={"arm": arm.spec, "run": args.name})
    batches = []
    for k in range(args.samples):
        rk = [(it, k) for it in items]
        batches += [rk[i:i + args.batch] for i in range(0, len(rk), args.batch)]
    t0 = time.time()
    recs = []
    for bi, br in enumerate(batches):
        got = run_batch(mdl, arm, br, bi, args, writer)
        recs += got
        print(f"  [{arm.key}] batch {bi + 1}/{len(batches)}: acc {np.mean([r['correct'] for r in got]):.2f} "
              f"closed {np.mean([bool(r['natural_close']) for r in got]):.2f} reason "
              f"{np.mean([r['n_reason'] for r in got]):.0f} tok  ({time.time() - t0:.0f}s)", flush=True)
    pointer = writer.close()
    pi = per_item(recs, args.budget_list)
    return {"spec": arm.spec, "key": arm.key, "complete": True,
            "setup": {"decoding": arm.decoding(), "thinking": arm.thinking, "trunc": arm.trunc, "ctrl": arm.ctrl,
                      "suppress": arm.suppress, "close_text": D.CANONICAL_CLOSE if arm.thinking else None,
                      "close_ids": D.close_sequence(mdl.env.tok, mdl.env.span_end) if arm.thinking else None},
            "agg": aggregate(pi, args.budget_list, float(args.max_new_tokens)), "per_item": pi, "trace": pointer,
            "seconds": time.time() - t0}


# ------------------------------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = ap.add_argument
    a("--model", default="Qwen/Qwen3.5-2B")
    a("--tiny", action="store_true", help="random tiny model + byte tokenizer, offline (CPU smoke test)")
    a("--tiny-arch", default="qwen3", choices=["qwen3", "qwen35"])
    a("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    a("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    a("--attn-impl", default="sdpa")
    a("--span-open", default=None, help="span-open marker token (models outside the Qwen <think> layout)")
    a("--span-close", default=None, help="span-close marker token")
    a("--data", default="gsm8k:train:400", help="reason.py data spec")
    a("--data-seed", type=int, default=0)
    a("--limit", type=int, default=200, help="first N problems of the spec (0: all)")
    a("--instruction", default=R.INSTRUCTION)
    a("--samples", type=int, default=1)
    a("--seed", type=int, default=0)
    a("--batch", type=int, default=32)
    a("--max-new-tokens", type=int, default=4096)
    a("--arms", default=None, help="';'-separated settings inside an arm, ',' between arms (module docstring)")
    a("--preset", choices=sorted(PRESETS), default="c1")
    a("--ref", default=None, help="label of the reference arm (default: the first arm)")
    a("--probe-every", type=int, default=128, help="forced-answer probe interval in span tokens (0: off)")
    a("--probe-tokens", type=int, default=16)
    a("--probe-samples", type=int, default=1)
    a("--probe-batch", type=int, default=16, help="rows per re-prefill probe batch")
    a("--probe-mode", choices=["fork", "prefill"], default="fork")
    a("--probe-cost", type=int, default=DEFAULT_PROBE_COST, help="token equivalents charged per controller probe")
    a("--budgets", default="256,512,1024,2048")
    a("--name", default=None)
    a("--stage", default="stage1")
    a("--out", default=None, help="results JSON (default results/evolve/round12/<stage>/<name>.json)")
    a("--trace-dir", default=None, help="per-token traces and text (default ~/limbic-data/ca2_traces/<name>)")
    a("--force", action="store_true")
    return ap


RESUME_KEYS = ("model", "tiny", "tiny_arch", "dtype", "attn_impl", "span_open", "span_close", "data", "data_seed",
               "limit", "instruction", "samples", "seed", "batch", "max_new_tokens", "probe_every", "probe_tokens",
               "probe_samples", "probe_mode", "probe_cost", "budgets")


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    aj = alignment().judge
    if args.tiny:
        args.dtype = "float32"
    short = "tiny-" + args.tiny_arch if args.tiny else args.model.split("/")[-1]
    args.name = aj.safe_name(args.name or R.safe_key(f"think_{short}_{args.preset}"))
    args.out = Path(args.out or ROOT / "results" / "evolve" / "round12" / args.stage / f"{args.name}.json")
    args.trace_dir = str(aj.assert_private(args.trace_dir or Path("~/limbic-data/ca2_traces") / args.name, ROOT))
    args.budget_list = [int(b) for b in args.budgets.split(",") if b]
    if args.probe_every > 0 and any(b % args.probe_every for b in args.budget_list):
        raise SystemExit("--budgets must be multiples of --probe-every")
    specs = [s for s in args.arms.split(",") if s.strip()] if args.arms else PRESETS[args.preset]
    arms = [parse_arm(s) for s in specs]
    if len({a_.key for a_ in arms}) != len(arms):
        raise SystemExit("arm labels must be distinct")
    if args.probe_every <= 0 and any(a_.ctrl and re.search(r"consist|conf", a_.ctrl) for a_ in arms):
        raise SystemExit("consist / conf controllers are fed by probes: --probe-every must be > 0")
    ref = args.ref or arms[0].label
    torch.manual_seed(args.seed)
    t0 = time.time()
    mdl = load_model(args)
    if any(not a_.thinking for a_ in arms) and not mdl.env.has_switch:
        raise SystemExit("think=off: this model has no thinking switch")
    spec = "synthetic:6" if args.tiny and args.data == "gsm8k:train:400" else args.data
    items = R.load_items(spec, args.data_seed, args.limit or None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.out.read_text()) if args.out.exists() else {}
    report.setdefault("arms", {})
    prev = report.get("args", {})
    diff = [k for k in RESUME_KEYS if k in prev and prev[k] != getattr(args, k)]
    old = report.get("data", {}).get("uids")
    stale = bool(report["arms"]) and bool(diff or (old is not None and old != [it.uid for it in items]))
    for spec_, res in report["arms"].items():                 # an arm label re-used with other settings
        same = [a_ for a_ in arms if a_.key == res.get("key")]
        if same and same[0].spec != spec_:
            stale = True
            diff.append(f"arm {res.get('key')}: {spec_!r} -> {same[0].spec!r}")
    if stale and not args.force:
        raise SystemExit(f"{args.out} holds a run with different settings ({diff or 'problems'}); "
                         "use another --out, or --force to start it again")
    if stale:
        report["arms"] = {}
    report.update({"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                   "git": git_state(), "versions": R.versions(), "model": mdl.facts,
                   "data": {"spec": spec, "n_items": len(items), "uids": [it.uid for it in items],
                            "scorer": "math_verify" if R._math_verify() else "numeric+normalised-latex"},
                   "close_text": D.CANONICAL_CLOSE, "arm_order": [a_.spec for a_ in arms], "ref": ref,
                   "probe_cost_rule": "per controller probe: cue (close + 8 tokens on Qwen) + --probe-tokens"})
    print(f"{mdl.facts['name']}: thinking end {mdl.env.span_end}, open {mdl.open_id}; {len(items)} problems x "
          f"{args.samples} samples; {len(arms)} arms; ref {ref}", flush=True)
    for a_ in arms:
        if report["arms"].get(a_.spec, {}).get("complete") and not args.force:
            print(f"[{a_.key}] complete in {args.out.name}; skipped (--force reruns)", flush=True)
            continue
        print(f"[{a_.key}] {a_.spec}", flush=True)
        report["arms"][a_.spec] = run_arm(mdl, a_, items, args)
        by_label = {res["key"]: res for res in report["arms"].values()}
        r0 = by_label.get(R.safe_key(ref))
        for res in report["arms"].values():
            if r0 is not None and r0.get("complete") and res is not r0 and res.get("complete"):
                res["vs_ref"] = {"ref": ref, **compare(res["per_item"], r0["per_item"], args.budget_list, args.seed)}
        report["seconds"] = time.time() - t0
        R.write_json(args.out, report)
        ag = report["arms"][a_.spec]["agg"]
        nc = "  -  " if ag["nonclose_rate"] is None else f"{ag['nonclose_rate']:.2f}"
        print(f"[{a_.key}] acc {ag['acc']:.3f}  non-close {nc}  reason {ag['n_reason_mean']:.0f}"
              f"  n_gen {ag['n_gen_mean']:.0f}  TTC median {ag['ttc_km']['median']}", flush=True)
    report["seconds"] = time.time() - t0
    R.write_json(args.out, report)
    print(f"saved {args.out} (no generated text); traces in {args.trace_dir}", flush=True)
    return report


if __name__ == "__main__":
    main()
