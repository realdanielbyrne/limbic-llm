"""Experiment 2 — continual learning with hippocampal replay consolidation.

Three fact tasks are learned in sequence. Conditions:
  naive    — plain sequential fine-tuning
  replay   — novelty-prioritised replay buffer, interleaved awake replay with
             logit distillation, plus an offline ``sleep`` phase on the slow
             (neocortical) parameters at the end of each task

Metrics: accuracy on every task after each stage, average final accuracy,
and backward transfer BWT = (1/(N−1)) Σ_{i<N} (a_{N,i} − a_{i,i}).

Usage:  python experiments/continual.py --steps 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hippoformer import (Consolidator, HippocampalLM, PrioritizedReplayBuffer, lm_loss,  # noqa: E402
                         novelty_alignment_loss, sequence_priority)
from tasks import FactTask  # noqa: E402


def accuracy(model, task, B=128):
    model.eval()
    with torch.no_grad():
        _, inp, tgt, mask = task.batch(B)
        pred = model(inp).logits.argmax(-1)
        acc = ((pred == tgt).float() * mask).sum() / mask.sum()
    model.train()
    return acc.item()


def run(condition: str, args) -> dict:
    torch.manual_seed(args.seed)
    tasks = [FactTask(t, n_tasks=args.tasks, seed=args.seed) for t in range(args.tasks)]
    model = HippocampalLM(tasks[0].vocab, args.d, 4, n_ec=1, n_hpc=1, n_neocortex=1, segment=12,
                          ca3_capacity=32, ca3_write=6)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    buffer = PrioritizedReplayBuffer(capacity=args.buffer, seed=args.seed)
    cons = Consolidator(model, buffer, list(model.neocortex.parameters()) + list(model.head.parameters()),
                        lr=args.sleep_lr, distill_weight=1.0,
                        loss_mask_fn=lambda tgt: (tgt >= len(tasks) * len(tasks[0].keys)).float())
    matrix = []
    for t, task in enumerate(tasks):
        for step in range(args.steps):
            seq, inp, tgt, mask = task.batch(args.batch)
            out = model(inp)
            tl = lm_loss(out.logits, tgt, reduction="none")
            loss = (tl * mask).sum() / mask.sum() + 0.1 * novelty_alignment_loss(out.novelty, tl)
            if condition == "replay" and len(buffer) and t > 0:
                loss = loss + args.replay_weight * cons.replay_loss(args.batch // 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if condition == "replay" and step >= args.steps // 2 and step % 5 == 0:
                # encode late (well-learned) traces, prioritised by CA1 novelty
                buffer.add(seq[:8], sequence_priority(out.novelty[:8], tl[:8]), out.logits[:8])
        if condition == "replay":
            cons.sleep(steps=args.sleep_steps, batch_size=args.batch)
        row = [accuracy(model, tasks[i]) for i in range(args.tasks)]
        matrix.append(row)
        print(f"[{condition:6s}] after task {t}: " + "  ".join(f"T{i}={a:.2f}" for i, a in enumerate(row)),
              flush=True)
    N = args.tasks
    final = matrix[-1]
    bwt = sum(matrix[-1][i] - matrix[i][i] for i in range(N - 1)) / (N - 1)
    return {"matrix": matrix, "avg_final": sum(final) / N, "bwt": bwt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--buffer", type=int, default=256)
    ap.add_argument("--replay-weight", type=float, default=1.0)
    ap.add_argument("--sleep-steps", type=int, default=100)
    ap.add_argument("--sleep-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/continual.json")
    args = ap.parse_args()
    results = {c: run(c, args) for c in ["naive", "replay"]}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print()
    for c, r in results.items():
        print(f"{c:6s} avg final acc {r['avg_final']:.3f}   BWT {r['bwt']:+.3f}")


if __name__ == "__main__":
    main()
