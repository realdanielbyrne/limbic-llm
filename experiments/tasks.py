"""Synthetic tasks that isolate the two things a hippocampus is for.

1. ``AssociativeRecall`` — one-shot episodic memory across a long gap.
   [k1 v1 … kn vn | filler × G | k_π1 v_π1 … ]  — loss only on recalled values.
   Bindings are new in every sequence, so they cannot be stored in weights;
   with G larger than the attention window, only a memory system can answer.

2. ``FactTask`` — semantic knowledge for continual learning.
   Each task owns a disjoint key set with a fixed key→value mapping that must
   be consolidated into weights. Only the first occurrence of each key in a
   sequence is scored, so in-context copying cannot help.
"""

from __future__ import annotations

import torch


class AssociativeRecall:
    def __init__(self, n_keys=16, n_vals=16, n_filler=16, n_pairs=6, gap=40, seed=0):
        self.n_keys, self.n_vals, self.n_filler = n_keys, n_vals, n_filler
        self.n_pairs, self.gap = n_pairs, gap
        self.vocab = n_keys + n_vals + n_filler
        self.length = 4 * n_pairs + gap
        self.g = torch.Generator().manual_seed(seed)

    def batch(self, B: int):
        P = self.n_pairs
        keys = torch.stack([torch.randperm(self.n_keys, generator=self.g)[:P] for _ in range(B)])
        vals = torch.randint(0, self.n_vals, (B, P), generator=self.g) + self.n_keys
        filler = torch.randint(0, self.n_filler, (B, self.gap), generator=self.g) + self.n_keys + self.n_vals
        order = torch.stack([torch.randperm(P, generator=self.g) for _ in range(B)])
        qk, qv = keys.gather(1, order), vals.gather(1, order)
        study = torch.stack([keys, vals], -1).flatten(1)
        test = torch.stack([qk, qv], -1).flatten(1)
        seq = torch.cat([study, filler, test], 1)
        inp, tgt = seq[:, :-1], seq[:, 1:]
        mask = torch.zeros_like(tgt, dtype=torch.float)
        start = 2 * P + self.gap                       # first query key position in seq
        mask[:, start::2] = 1.0                        # targets that are recalled values
        return inp, tgt, mask


class WhatWhenTask:
    """What happened, and how long ago? Content and elapsed time beyond the attention window.

    [filler … E_k … filler (gap Δ) … Q] → answer E_k (what), then answer bin(Δ) (when).
    Δ is drawn from ``gaps``; ``bins`` are elapsed-time intervals (log-spaced by default).
    Only the two answers are scored.
    """

    def __init__(self, n_filler=16, n_events=8, gaps=(20, 140), bins=(20, 35, 60, 100, 140),
                 pre=(4, 20), length=None, seed=0):
        self.n_filler, self.n_events = n_filler, n_events
        self.gaps, self.bins, self.pre = gaps, bins, pre
        self.E0 = n_filler
        self.Q = n_filler + n_events
        self.B0 = self.Q + 1
        self.vocab = self.B0 + len(bins) - 1
        self.length = length or (pre[1] + gaps[1] + 3)
        self.g = torch.Generator().manual_seed(seed)

    def bin_of(self, gap: torch.Tensor) -> torch.Tensor:
        edges = torch.tensor(self.bins[1:-1])
        return torch.bucketize(gap, edges, right=True).clamp(max=len(self.bins) - 2)

    def batch(self, B: int):
        L = self.length
        seq = torch.randint(0, self.n_filler, (B, L), generator=self.g)
        p = torch.randint(self.pre[0], self.pre[1], (B,), generator=self.g)
        gap = torch.randint(self.gaps[0], self.gaps[1], (B,), generator=self.g)
        ev = torch.randint(0, self.n_events, (B,), generator=self.g)
        b = torch.arange(B)
        q = p + gap
        seq[b, p] = self.E0 + ev
        seq[b, q] = self.Q
        seq[b, q + 1] = self.E0 + ev
        seq[b, q + 2] = self.B0 + self.bin_of(gap)
        inp, tgt = seq[:, :-1], seq[:, 1:]
        what = torch.zeros_like(tgt, dtype=torch.bool)
        when = torch.zeros_like(tgt, dtype=torch.bool)
        what[b, q] = True                # at Q, predict the event type
        when[b, q + 1] = True            # after the type, predict the elapsed-time bin
        return inp, tgt, what, when, gap


class FactTask:
    def __init__(self, task_id: int, n_keys=8, n_vals=16, n_tasks=3, length=24, seed=0):
        g = torch.Generator().manual_seed(1000 + 10 * seed + task_id)
        self.keys = torch.arange(task_id * n_keys, (task_id + 1) * n_keys)
        self.value_of = torch.randint(0, n_vals, (n_keys,), generator=g) + n_tasks * n_keys
        self.vocab = n_tasks * n_keys + n_vals
        self.length = length
        self.g = torch.Generator().manual_seed(seed + task_id)

    def batch(self, B: int):
        n = self.length // 2
        ki = torch.randint(0, len(self.keys), (B, n), generator=self.g)
        seq = torch.stack([self.keys[ki], self.value_of[ki]], -1).flatten(1)
        inp, tgt = seq[:, :-1], seq[:, 1:]
        mask = torch.zeros_like(tgt, dtype=torch.float)
        for b in range(B):                              # score first occurrence only
            seen = set()
            for j in range(n):
                k = int(ki[b, j])
                if k not in seen:
                    mask[b, 2 * j] = 1.0               # tgt[2j] is the value after key j
                    seen.add(k)
        return seq, inp, tgt, mask
