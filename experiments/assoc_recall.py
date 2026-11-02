"""Experiment 1 — one-shot associative recall across a gap longer than attention.

Compares, at matched width and similar depth:
  full_attn      transformer with full causal attention (upper reference)
  window_attn    transformer with a sliding window shorter than the gap
  hippocampal    Strategy I  (EC → DG → CA3 store → CA2 → CA1 → Sub → cortex)
  titans_mac     plain Titans MAC (no DG / CA1 / novelty-lr)
  titans_seq     plain Titans MAC + sequential (heteroassociative) writes only
  titans_seq_dg  ablation: + DG sparse writes
  titans_seq_ca1 ablation: + CA1 comparator + novelty-scaled lr
  titans_gdn     Titans MAC with a linear Gated DeltaNet memory (sequential writes)
  titans_gdn_dg  + DG-sparse addresses
  titans_hippo   Strategy II (Titans MAC + DG writes + CA1 comparator + novelty-lr)

Usage:  python experiments/assoc_recall.py --steps 1500 --models all
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

from limbic_llm import BaselineLM, HippocampalLM, TitansHippocampalLM, lm_loss, novelty_alignment_loss  # noqa: E402
from tasks import AssociativeRecall  # noqa: E402


def build(name: str, vocab: int, d: int, local: int):
    if name == "full_attn":
        return BaselineLM(vocab, d, 4, n_layers=3)
    if name == "window_attn":
        return BaselineLM(vocab, d, 4, n_layers=3, window=local)
    if name == "hippocampal":
        return HippocampalLM(vocab, d, 4, n_ec=1, n_hpc=1, n_neocortex=1, segment=local,
                             ca3_capacity=64, ca3_write=12, dg_expansion=8, dg_sparsity=0.05)
    common = dict(variant="mac", chunk=local, window=local, n_persistent=4, mem_depth=2)
    if name == "titans_mac":
        return TitansHippocampalLM(vocab, d, 4, n_layers=2, use_dg=False, use_ca1=False,
                                   novelty_lr=False, sequential=False, **common)
    if name == "titans_seq":   # ablation: plain Titans + sequential writes only
        return TitansHippocampalLM(vocab, d, 4, n_layers=2, use_dg=False, use_ca1=False,
                                   novelty_lr=False, sequential=True, **common)
    if name == "titans_seq_dg":   # ablation: + DG sparse write path
        return TitansHippocampalLM(vocab, d, 4, n_layers=2, use_dg=True, use_ca1=False, novelty_lr=False,
                                   sequential=True, dg_expansion=4, dg_sparsity=0.05, **common)
    if name == "titans_seq_ca1":  # ablation: + CA1 comparator and novelty-scaled lr
        return TitansHippocampalLM(vocab, d, 4, n_layers=2, use_dg=False, use_ca1=True, novelty_lr=True,
                                   sequential=True, **common)
    if name == "titans_gdn":      # Titans MAC with a linear Gated DeltaNet memory
        return TitansHippocampalLM(vocab, d, 4, n_layers=2, use_dg=False, use_ca1=False, novelty_lr=False,
                                   sequential=True, memory_type="gated_delta", mem_heads=4, **common)
    if name == "titans_gdn_dg":   # + DG-sparse addresses
        return TitansHippocampalLM(vocab, d, 4, n_layers=2, use_dg=True, use_ca1=False, novelty_lr=False,
                                   sequential=True, memory_type="gated_delta", mem_heads=4,
                                   dg_expansion=8, dg_sparsity=0.05, **common)
    if name == "titans_hippo":
        return TitansHippocampalLM(vocab, d, 4, n_layers=2, use_dg=True, use_ca1=True,
                                   novelty_lr=True, sequential=True, dg_expansion=4, dg_sparsity=0.05,
                                   **common)
    raise ValueError(name)


def evaluate(model, task, n_batches=8, B=64):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            inp, tgt, mask = task.batch(B)
            pred = model(inp).logits.argmax(-1)
            correct += ((pred == tgt).float() * mask).sum().item()
            total += mask.sum().item()
    model.train()
    return correct / total


def train(name, args):
    torch.manual_seed(args.seed)
    task = AssociativeRecall(gap=args.gap, n_pairs=args.pairs, seed=args.seed)
    val = AssociativeRecall(gap=args.gap, n_pairs=args.pairs, seed=args.seed + 999)
    model = build(name, task.vocab, args.d, args.local)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)
    curve, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        inp, tgt, mask = task.batch(args.batch)
        out = model(inp)
        tl = lm_loss(out.logits, tgt, reduction="none")
        # recall positions dominate the objective; other tokens are noise
        loss = (tl * mask).sum() / mask.sum() + 0.05 * tl.mean()
        if out.novelty is not None:
            loss = loss + 0.1 * novelty_alignment_loss(out.novelty, tl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        for dg in getattr(model, "dentate_gyri", []):
            dg.tick()
        if step % args.eval_every == 0 or step == args.steps:
            acc = evaluate(model, val)
            curve.append((step, acc, loss.item()))
            print(f"[{name:12s}] step {step:5d}  loss {loss.item():.3f}  recall {acc:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return {"params": n_params, "final_recall": curve[-1][1], "best_recall": max(c[1] for c in curve),
            "curve": curve, "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--gap", type=int, default=40)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--local", type=int, default=16, help="attention window / segment / chunk")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/synthetic/assoc_recall.json")
    args = ap.parse_args()
    names = ["full_attn", "window_attn", "hippocampal", "titans_mac", "titans_seq", "titans_hippo"] \
        if args.models == "all" else args.models.split(",")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    results = {n: train(n, args) for n in names}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print("\nchance = %.3f" % (1 / 16))
    for n, r in results.items():
        print(f"{n:12s} params {r['params']:>8,d}  final {r['final_recall']:.3f}  best {r['best_recall']:.3f}")


if __name__ == "__main__":
    main()
