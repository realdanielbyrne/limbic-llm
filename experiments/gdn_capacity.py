"""Experiment 3: does dentate-gyrus addressing raise a linear memory's capacity?

A Gated DeltaNet state is a linear associative matrix, so it holds roughly
d_k clean bindings per head, fewer when keys overlap. The hippocampal claim
is that sparse, nearly orthogonal DG codes cut interference, much as
one-hot addresses make Qwenputer's DeltaNet state exact RAM.

Pure Gated DeltaNet LMs (no attention) are trained on associative recall
with an increasing number of key→value pairs:

  gdn      standard Gated DeltaNet, d_k = d_head          (state 16×16 / head)
  gdn_x8   standard Gated DeltaNet, d_k = 8·d_head        (state 16×128 / head)
  tied_x8  tied dense address,      d_k = 8·d_head        (same state)
  dg_x8    tied DG k-WTA address,   d_k = 8·d_head, 5 %   (same state)

``tied_x8`` vs ``dg_x8`` isolates sparsity at an identical state size.

Usage:  python experiments/gdn_capacity.py --pairs 8,16,32 --steps 1500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from limbic_llm import GatedDeltaLM, lm_loss  # noqa: E402
from tasks import AssociativeRecall  # noqa: E402

MODES = {
    "gdn": dict(addressing="gdn", key_expansion=1),
    "gdn_x8": dict(addressing="gdn", key_expansion=8),
    "tied_x8": dict(addressing="tied", key_expansion=8),
    "dg_x8": dict(addressing="dg", key_expansion=8, dg_sparsity=0.05),
}


def evaluate(model, task, n_batches=4, B=128):
    model.eval()
    c = n = 0
    with torch.no_grad():
        for _ in range(n_batches):
            inp, tgt, mask = task.batch(B)
            pred = model(inp).logits.argmax(-1)
            c += ((pred == tgt).float() * mask).sum().item()
            n += mask.sum().item()
    model.train()
    return c / n


def run(mode: str, pairs: int, args) -> dict:
    torch.manual_seed(args.seed)
    mk = lambda s: AssociativeRecall(n_keys=args.n_keys, n_pairs=pairs, gap=args.gap, seed=s)
    task, val = mk(args.seed), mk(args.seed + 999)
    model = GatedDeltaLM(task.vocab, args.d, args.heads, n_layers=args.layers, **MODES[mode])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
    t0, curve = time.time(), []
    for step in range(1, args.steps + 1):
        inp, tgt, mask = task.batch(args.batch)
        tl = lm_loss(model(inp).logits, tgt, reduction="none")
        loss = (tl * mask).sum() / mask.sum() + 0.05 * tl.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            acc = evaluate(model, val)
            curve.append((step, acc))
            print(f"[{mode:8s} P={pairs:2d}] step {step:5d} loss {loss.item():.3f} recall {acc:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return {"params": sum(p.numel() for p in model.parameters()), "final": curve[-1][1],
            "best": max(a for _, a in curve), "curve": curve, "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--pairs", default="8,16,32")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--n-keys", type=int, default=64)
    ap.add_argument("--gap", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gdn_capacity.json")
    args = ap.parse_args()
    modes, loads = args.modes.split(","), [int(p) for p in args.pairs.split(",")]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    # resume: keep finished runs from an earlier (possibly interrupted) sweep
    results = json.loads(out.read_text())["results"] if out.exists() else {}
    for m in modes:
        for p in loads:
            if str(p) in results.get(m, {}):
                print(f"[{m:8s} P={p:2d}] already done, skipping", flush=True)
                continue
            results.setdefault(m, {})[str(p)] = run(m, p, args)
            out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    results = {m: {int(p): r for p, r in results[m].items()} for m in modes}
    print("\nrecall (chance = 1/16 = 0.062)")
    print("mode      " + "".join(f"  P={p:<4d}" for p in loads))
    for m in modes:
        print(f"{m:9s} " + "".join(f"  {results[m][p]['final']:.3f} " for p in loads))


if __name__ == "__main__":
    main()
