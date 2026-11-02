"""Experiment 4: hippocampal feedback loops and a sense of time.

Task (WhatWhenTask): an event token appears, then 20–140 filler tokens, then a
query. The model must say *what* happened (8 event types, chance 0.125) and *when*
(4 log-spaced elapsed-time bins; the bins cover 15, 25, 40 and 40 of the
120 possible gaps, so always guessing the largest bin scores 0.333). Attention only sees 16 tokens and
there is no absolute position embedding, so both answers must come through the
side channels: content from the hippocampal loop, elapsed time from the clock.

Suites (pick with --suite, or build one model from the flags):

  what_when   window | memory | time | memory_time        does each channel do its job?
  stability   closed-loop memory with kernel flat / exp / osc, and osc + CA1 control
              trained on gaps ≤ 140, then run on 520-token sequences to see whether
              feedback stays bounded and new input keeps dominating
  The long rollout of single-event models also runs a zero-shot recency probe: an
  older, different event precedes the target, and it reports how often the model
  names the newer and the older event.

  recency     the stability models on a recency task: an older, different event
              comes 20–60 tokens before the target, and the answer is the most recent.
              Undamped feedback returns both events equally; tests whether new input
              dominates. Reports how often the model names the older event.
  all         what_when + stability
  feedback    closed clamped-sinusoid loops with no feedback, divisive feedback and
              subtractive (inverted-copy) feedback, hard clamp removed, vs the full design
  ablations   the full design without the contrast, and with dentate-gyrus write keys
              read directly (EC → CA3) or through the dentate gyrus

Usage
  python experiments/temporal_loop.py --suite what_when
  python experiments/temporal_loop.py --suite stability --steps 1500
  python experiments/temporal_loop.py --name custom --memory --time --kernel osc \
      --rho 0.6 --loop closed --no-ca1-control --steps 800
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from limbic_llm.temporal_loop import TemporalLoopLM  # noqa: E402
from tasks import WhatWhenTask  # noqa: E402

LOOP_DEFAULTS = dict(kernel="osc", rho=0.7, amp=1.0, loop="open", ca1_control=True, contrast=True, dg_keys=False,
                     feedback=None, clamp=None, project=None, saturate=None, forward_gain=1.0, fb_beta=1.0,
                     read="direct")
_RAW = dict(clamp=False, project=False, saturate=False)               # CA1 parts off: feedback alone

PRESETS = {
    # ---- what_when: which channel carries which answer
    "window":      dict(memory=False, time=False),
    "memory":      dict(memory=True, time=False),
    "time":        dict(memory=False, time=True),
    "memory_time": dict(memory=True, time=True),
    # ---- stability: closed loops, feedback written back into memory
    "closed_flat":     dict(memory=True, time=False, loop="closed", kernel="flat", ca1_control=False),
    "closed_exp":      dict(memory=True, time=False, loop="closed", kernel="exp", ca1_control=False),
    "closed_osc":      dict(memory=True, time=False, loop="closed", kernel="osc", ca1_control=False),
    "closed_osc_ca1":  dict(memory=True, time=False, loop="closed", kernel="osc", ca1_control=True),
    # ---- feedback: what does the CA1 → CA3 back-projection itself do? (no hard clamp)
    "closed_osc_div":  dict(memory=True, time=False, loop="closed", kernel="osc", feedback="divisive", **_RAW),
    "closed_osc_sub":  dict(memory=True, time=False, loop="closed", kernel="osc", feedback="subtractive", **_RAW),
    # ---- ablations of the full design
    "closed_osc_ca1_nocontrast": dict(memory=True, time=False, loop="closed", kernel="osc", contrast=False),
    "closed_osc_ca1_dg":         dict(memory=True, time=False, loop="closed", kernel="osc", dg_keys=True, read="direct"),
    "closed_osc_ca1_dgread":     dict(memory=True, time=False, loop="closed", kernel="osc", dg_keys=True, read="dg"),
}
SUITES = {
    "what_when": ["window", "memory", "time", "memory_time"],
    "stability": ["closed_flat", "closed_exp", "closed_osc", "closed_osc_ca1"],
}
SUITES["all"] = SUITES["what_when"] + SUITES["stability"]
SUITES["recency"] = SUITES["stability"]
SUITES["feedback"] = ["closed_osc", "closed_osc_div", "closed_osc_sub", "closed_osc_ca1"]
SUITES["ablations"] = ["closed_osc_ca1_nocontrast", "closed_osc_ca1_dg", "closed_osc_ca1_dgread"]
CUE_NOISE = (0.0, 0.5, 1.0, 2.0)


def build(cfg: dict, vocab: int, args) -> TemporalLoopLM:
    loop_kw = {k: cfg.get(k, LOOP_DEFAULTS[k]) for k in LOOP_DEFAULTS}
    loop_kw.update(tau=args.tau, d_key=args.d_key)
    return TemporalLoopLM(vocab, args.d, 4, window=args.window, memory=cfg["memory"], time=cfg["time"],
                          **loop_kw)


def answer_acc(logits, tgt, mask):
    pred = logits.argmax(-1)
    return ((pred == tgt) & mask).sum().item() / max(1, mask.sum().item())


@torch.no_grad()
def evaluate(model, task, n=4, B=128):
    """Accuracy on what and when; with a distractor, also how often the older event is named."""
    model.eval()
    w = c = o = 0.0
    for _ in range(n):
        inp, tgt, what, when, _ = task.batch(B)
        lg = model(inp).logits
        w += answer_acc(lg, tgt, what)
        c += answer_acc(lg, tgt, when)
        if task.distractor:
            o += (lg.argmax(-1)[what] == task.last_old).float().mean().item()
    model.train()
    return w / n, c / n, (o / n if task.distractor else None)


@torch.no_grad()
def long_rollout(model, args, seed):
    """~520-token sequences, gaps up to ~500: does recall survive and does feedback stay bounded?"""
    if args.task == "recent":
        task = WhatWhenTask(gaps=(20, 440), length=520, seed=seed, distractor=True)
    else:
        task = WhatWhenTask(gaps=(20, 497), length=520, seed=seed)
    model.eval()
    inp, tgt, what, _, gap = task.batch(64)
    out = model(inp)
    pred = out.logits.argmax(-1)
    q = what.nonzero()
    correct = (pred[what] == tgt[what]).float()
    res = {}
    for lo, hi in [(20, 140), (140, 300), (300, 500)]:
        sel = (gap[q[:, 0]] >= lo) & (gap[q[:, 0]] < hi)
        res[f"what_gap_{lo}_{hi}"] = correct[sel].mean().item() if sel.any() else None
    if task.distractor:
        res["older_named"] = (pred[what] == task.last_old).float().mean().item()
    if args.task == "single":
        # zero-shot recency probe: an older, different event precedes the target.
        # Which one does a model trained on single events name?
        probe = WhatWhenTask(seed=seed + 1, distractor=True)
        p_inp, p_tgt, p_what, _, _ = probe.batch(256)
        p_pred = model(p_inp).logits.argmax(-1)[p_what]
        res["probe_newer_named"] = (p_pred == p_tgt[p_what]).float().mean().item()
        res["probe_older_named"] = (p_pred == probe.last_old).float().mean().item()
    if model.loop is not None and args.task == "single":
        # pattern completion from degraded cues: relative noise on every read query
        c_inp, c_tgt, c_what, _, _ = WhatWhenTask(seed=seed + 2).batch(256)
        for sigma in CUE_NOISE:
            model.loop.cue_noise = sigma
            torch.manual_seed(0)
            res[f"cue_noise_{sigma}"] = answer_acc(model(c_inp).logits, c_tgt, c_what)
        model.loop.cue_noise = 0.0
    if out.loop is not None:
        r = out.loop.inj_ratio
        for lo, hi in [(args.tau, 128), (128, 320), (320, 519)]:
            res[f"inj_ratio_{lo}_{hi}"] = r[:, lo:hi].mean().item()
        res["inj_ratio_max"] = r.max().item()
        res["gain"] = torch.sigmoid(model.loop.gain_logit).item()
    model.train()
    return res


def train(name: str, cfg: dict, args) -> dict:
    torch.manual_seed(args.seed)
    recent = args.task == "recent"
    task = WhatWhenTask(seed=args.seed, distractor=recent)
    val = WhatWhenTask(seed=args.seed + 999, distractor=recent)
    model = build(cfg, task.vocab, args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
    t0, curve, diverged = time.time(), [], None
    for step in range(1, args.steps + 1):
        inp, tgt, what, when, _ = task.batch(args.batch)
        out = model(inp)
        ce = F.cross_entropy(out.logits.flatten(0, 1), tgt.flatten(), reduction="none").view_as(tgt)
        loss = (ce * what).sum() / what.sum() + (ce * when).sum() / when.sum()
        if not math.isfinite(loss.item()):
            diverged = step
            print(f"[{name:15s}] diverged at step {step}", flush=True)
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            w, c, o = evaluate(model, val)
            curve.append((step, w, c) if o is None else (step, w, c, o))
            older = "" if o is None else f" older {o:.3f}"
            print(f"[{name:15s}] step {step:5d} loss {loss.item():.3f} what {w:.3f} when {c:.3f}{older} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    res = {"config": cfg, "params": sum(p.numel() for p in model.parameters()), "curve": curve,
           "diverged_at": diverged, "seconds": time.time() - t0}
    if curve:
        res["what"], res["when"] = curve[-1][1], curve[-1][2]
        if len(curve[-1]) > 3:
            res["older_named"] = curve[-1][3]
    if diverged is None:
        res["long"] = long_rollout(model, args, args.seed + 7)
        print(f"[{name:15s}] long rollout {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res['long'].items()})}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=list(SUITES), help="run a preset suite")
    ap.add_argument("--models", help="comma-separated preset names (overrides --suite)")
    ap.add_argument("--name", default="custom", help="name for a model built from the flags below")
    ap.add_argument("--task", choices=["single", "recent"], default=None,
                    help="single: one event; recent: an older distractor event precedes it "
                         "(default: recent for --suite recency, else single)")
    g = ap.add_argument_group("loop and clock (used when neither --suite nor --models is given)")
    g.add_argument("--memory", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--time", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--kernel", choices=["osc", "exp", "flat"], default="osc",
                   help="osc: clamped damped sinusoid; exp: damped; flat: undamped")
    g.add_argument("--rho", type=float, default=0.7, help="attenuation per circuit")
    g.add_argument("--amp", type=float, default=1.0, help="feedback amplitude, clamped to ≤ 1")
    g.add_argument("--loop", choices=["open", "closed"], default="open",
                   help="open: store pre-injection stream; closed: store post-injection stream")
    g.add_argument("--ca1-control", action=argparse.BooleanOptionalAction, default=True,
                   help="negative-feedback gain control and error-only injection")
    g.add_argument("--contrast", action=argparse.BooleanOptionalAction, default=True,
                   help="inject episode-minus-prior contrast (dynamic activation addition)")
    g.add_argument("--dg-keys", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--read", choices=["direct", "dg"], default="direct",
                   help="read query: dense, directly from the stream (EC → CA3), or through the DG code")
    g.add_argument("--feedback", choices=["none", "divisive", "subtractive"], default=None,
                   help="CA1 → CA3 back-projection: energy-divisive, or an inverted copy of earlier "
                        "injections (default: divisive with --ca1-control, else none)")
    g.add_argument("--clamp", action=argparse.BooleanOptionalAction, default=None,
                   help="hard bound ‖inj‖ ≤ G‖h‖ (default: on with --ca1-control)")
    g.add_argument("--project", action=argparse.BooleanOptionalAction, default=None,
                   help="error-only injection (default: on with --ca1-control)")
    g.add_argument("--saturate", action=argparse.BooleanOptionalAction, default=None,
                   help="tanh amplitude saturation (default: on with --ca1-control)")
    g.add_argument("--forward-gain", type=float, default=1.0, help="open-loop gain A")
    g.add_argument("--fb-beta", type=float, default=1.0, help="subtractive feedback fraction β")
    t = ap.add_argument_group("training")
    t.add_argument("--tau", type=int, default=16, help="loop period in tokens (one circuit)")
    t.add_argument("--window", type=int, default=16)
    t.add_argument("--d", type=int, default=64)
    t.add_argument("--d-key", type=int, default=32)
    t.add_argument("--steps", type=int, default=1500)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.task is None:
        args.task = "recent" if args.suite == "recency" else "single"

    if args.models:
        runs = {n: PRESETS[n] for n in args.models.split(",")}
    elif args.suite:
        runs = {n: PRESETS[n] for n in SUITES[args.suite]}
    else:
        runs = {args.name: dict(memory=args.memory, time=args.time, kernel=args.kernel, rho=args.rho, amp=args.amp,
                                loop=args.loop, ca1_control=args.ca1_control, contrast=args.contrast,
                                dg_keys=args.dg_keys, read=args.read, feedback=args.feedback, clamp=args.clamp,
                                project=args.project, saturate=args.saturate, forward_gain=args.forward_gain,
                                fb_beta=args.fb_beta)}
    stem = args.suite or "custom"
    if args.task == "recent" and args.suite != "recency":
        stem += "_recent"
    out = Path(args.out or f"results/temporal_loop_{stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text())["results"] if out.exists() else {}
    for name, cfg in runs.items():
        if name in results:
            print(f"[{name:15s}] already done, skipping", flush=True)
            continue
        results[name] = train(name, cfg, args)
        out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print("\nbaselines: what 0.125 (chance), when 0.333 (majority bin)")
    for n, r in results.items():
        if r.get("diverged_at"):
            print(f"{n:15s} diverged at step {r['diverged_at']}")
        else:
            lg = r.get("long", {})
            extra = f"  inj late {lg['inj_ratio_320_519']:.3f}" if "inj_ratio_320_519" in lg else ""
            if "older_named" in r:
                extra += f"  older named {r['older_named']:.3f}"
            print(f"{n:15s} what {r['what']:.3f}  when {r['when']:.3f}{extra}")


if __name__ == "__main__":
    main()
