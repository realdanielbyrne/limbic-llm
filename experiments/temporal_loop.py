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
  all         both

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

LOOP_DEFAULTS = dict(kernel="osc", rho=0.7, amp=1.0, loop="open", ca1_control=True, contrast=True, dg_keys=False)

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
}
SUITES = {
    "what_when": ["window", "memory", "time", "memory_time"],
    "stability": ["closed_flat", "closed_exp", "closed_osc", "closed_osc_ca1"],
}
SUITES["all"] = SUITES["what_when"] + SUITES["stability"]


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
    model.eval()
    w = c = 0.0
    for _ in range(n):
        inp, tgt, what, when, _ = task.batch(B)
        lg = model(inp).logits
        w += answer_acc(lg, tgt, what)
        c += answer_acc(lg, tgt, when)
    model.train()
    return w / n, c / n


@torch.no_grad()
def long_rollout(model, args, seed):
    """520-token sequences, gaps 20–500: does recall survive and does feedback stay bounded?"""
    task = WhatWhenTask(gaps=(20, 500), length=520, seed=seed)
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
    task = WhatWhenTask(seed=args.seed)
    val = WhatWhenTask(seed=args.seed + 999)
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
            w, c = evaluate(model, val)
            curve.append((step, w, c))
            print(f"[{name:15s}] step {step:5d} loss {loss.item():.3f} what {w:.3f} when {c:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    res = {"config": cfg, "params": sum(p.numel() for p in model.parameters()), "curve": curve,
           "diverged_at": diverged, "seconds": time.time() - t0}
    if curve:
        res["what"], res["when"] = curve[-1][1], curve[-1][2]
    if diverged is None:
        res["long"] = long_rollout(model, args, args.seed + 7)
        print(f"[{name:15s}] long rollout {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res['long'].items()})}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=list(SUITES), help="run a preset suite")
    ap.add_argument("--models", help="comma-separated preset names (overrides --suite)")
    ap.add_argument("--name", default="custom", help="name for a model built from the flags below")
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

    if args.models:
        runs = {n: PRESETS[n] for n in args.models.split(",")}
    elif args.suite:
        runs = {n: PRESETS[n] for n in SUITES[args.suite]}
    else:
        runs = {args.name: dict(memory=args.memory, time=args.time, kernel=args.kernel, rho=args.rho, amp=args.amp,
                                loop=args.loop, ca1_control=args.ca1_control, contrast=args.contrast,
                                dg_keys=args.dg_keys)}
    out = Path(args.out or f"results/temporal_loop_{args.suite or 'custom'}.json")
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
            print(f"{n:15s} what {r['what']:.3f}  when {r['when']:.3f}{extra}")


if __name__ == "__main__":
    main()
