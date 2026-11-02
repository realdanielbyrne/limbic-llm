"""W1c: run the CPU stability and recency suites on the round candidates (W1, W1i).

The evolution search's must-pass criteria (research/NOTEBOOK.md, "Fitness") are measured on the
small what-when models of ``experiments/temporal_loop.py``: bounded echo over a 520-token rollout,
and the zero-shot recency probe (an older, different event precedes the target; the full design
names the older one 18% of the time, T11–T13). This trains the same ``TemporalLoopLM`` with its
``HippocampalLoop`` replaced by a candidate's loop at the small model's width, using
temporal_loop.py's own training, evaluation and long rollout unchanged.

The candidates are open loops: they store the stream before injection, so injected content never
re-enters memory and cannot run away by self-rehearsal; the rollout reports the injection ratio
anyway. The cue-noise sweep in the rollout acts only on ``HippocampalLoop`` and is meaningless here.

    python experiments/evolve/cpu_suite.py --models w1i,w1i_time,w1,w1_time --steps 3000 \
        --out results/evolve/round5/cpu_suite.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import temporal_loop as tl  # noqa: E402
from evolve.w1_gdn import GDNLoop  # noqa: E402
from evolve.w1i_events import EventLoop  # noqa: E402

CANDIDATES = {
    "w1i":      dict(loop="w1i", time=False),
    "w1i_time": dict(loop="w1i", time=True),
    "w1":       dict(loop="w1", time=False),
    "w1_time":  dict(loop="w1", time=True),
}


def build(cfg: dict, vocab: int, args):
    model = tl.TemporalLoopLM(vocab, args.d, 4, window=args.window, memory=True, time=cfg["time"])
    heads, dh = 4, args.d // 4
    if cfg["loop"] == "w1i":
        model.loop = EventLoop(args.d, heads=heads, d_head=dh, chunk=32)
    else:
        model.loop = GDNLoop(args.d, heads=heads, d_head=dh, chunk=32)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="w1i,w1i_time,w1,w1_time")
    ap.add_argument("--task", choices=["single", "recent"], default="single")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--tau", type=int, default=16, help="only used to slice the injection ratio")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/evolve/round5/cpu_suite.json")
    args = ap.parse_args(argv)
    tl.build = build                                               # temporal_loop.train calls build()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text())["results"] if out.exists() else {}
    for name in args.models.split(","):
        if name in results:
            print(f"[{name}] already done, skipping", flush=True)
            continue
        torch.manual_seed(args.seed)
        results[name] = tl.train(name, CANDIDATES[name], args)
        out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
