"""Binding-free ceiling for experiments/tasks.py::AssociativeRecall.

The task scores teacher-forced, shuffled queries; values are drawn with replacement
from n_vals. A model that never binds key to value can still remember the multiset of
studied values and strike each one as it is revealed. Its expected accuracy at each
query is max(count of remaining values) / (number of remaining values).

Usage:  python binding_free_baseline.py --pairs 6 8 16 32 --n 100000
Result (seed 0): 6 -> 0.476, 8 -> 0.423, 16 -> 0.319, 32 -> 0.243.
Compare results/synthetic/assoc_recall.json (full attention 0.485, Titans 0.483-0.490,
Strategy I 0.414) and results/synthetic/gdn_capacity.json (gdn_x8 0.428 / 0.320 / 0.222).
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np


def ceiling(pairs: int, n: int, n_vals: int = 16, seed: int = 0) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    hit = np.zeros(pairs)
    for _ in range(n):
        vals = rng.integers(0, n_vals, pairs)
        rem = Counter(vals.tolist())
        for j, i in enumerate(rng.permutation(pairs)):
            hit[j] += max(rem.values()) / sum(rem.values())
            v = int(vals[i]); rem[v] -= 1
            if rem[v] == 0:
                del rem[v]
    per_pos = (hit / n).tolist()
    return float(np.mean(per_pos)), per_pos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", default=[6, 8, 16, 32])
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--n-vals", type=int, default=16)
    a = ap.parse_args()
    for p in a.pairs:
        m, pp = ceiling(p, a.n if p <= 8 else a.n // 5, a.n_vals)
        print(f"pairs={p:3d}  binding-free recall={m:.3f}  first query={pp[0]:.3f}")
