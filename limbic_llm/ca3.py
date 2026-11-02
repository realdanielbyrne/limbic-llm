"""CA3: autoassociative memory via recurrent collaterals (modern Hopfield).

Biology
-------
Each CA3 pyramidal cell contacts ~4 % of all other CA3 cells. Hebbian LTP on
these recurrent collaterals stores co-active patterns in a single trial, and
the recurrent dynamics complete a partial cue to the nearest stored attractor.

Computation
-----------
Two pattern sets are visible to a query ξ_t:

* the current segment's patterns h_{≤t} (working, causal), and
* an episodic store {p_j} written in one shot by earlier segments.

Retrieval is the modern-Hopfield / attention update (Ramsauer et al., 2020)

    ξ ← V^⊤ softmax( β · K ξ ),      K = W_K [p ; h],  V = W_V [p ; h]

iterated ``n_iter`` times with tied weights (recurrent settling).

Writing is novelty-gated and capacity-limited: the ``n_write`` most novel
tokens of a segment are copied into the slots with the lowest salience, every
existing slot's salience decays by γ (heterosynaptic LTD), and slots that are
retrieved gain salience (reconsolidation).

``sequential=True`` (default) makes the store heteroassociative: each slot
holds a cue h_t and its successor h_{t+1}, and retrieval returns successors.
This is the temporally asymmetric STDP account of CA3 sequence memory
(Levy, 1996; Jensen & Lisman, 1996), and it is what lets a cue recall *what
came next* (key → value) rather than only completing itself. With
``sequential=False`` the store is purely autoassociative.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EpisodicStore:
    patterns: torch.Tensor   # (B, S, d)  cue patterns (what a query is matched against)
    salience: torch.Tensor   # (B, S); 0 marks an empty slot
    successors: torch.Tensor | None = None  # (B, S, d) patterns that followed each cue

    @classmethod
    def empty(cls, batch: int, capacity: int, d: int, device=None, dtype=None,
              sequential: bool = False) -> "EpisodicStore":
        z = lambda: torch.zeros(batch, capacity, d, device=device, dtype=dtype)
        return cls(z(), torch.zeros(batch, capacity, device=device, dtype=dtype),
                   z() if sequential else None)

    @property
    def valid(self) -> torch.Tensor:
        return self.salience > 0

    @property
    def values(self) -> torch.Tensor:
        return self.patterns if self.successors is None else self.successors

    def detach(self) -> "EpisodicStore":
        return EpisodicStore(self.patterns.detach(), self.salience.detach(),
                             None if self.successors is None else self.successors.detach())


class CA3Autoassociative(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_heads: int = 4,
        capacity: int = 256,
        n_write: int = 16,
        n_iter: int = 1,
        salience_decay: float = 0.98,
        reconsolidation: float = 0.1,
        sequential: bool = True,
        init_beta: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.capacity = capacity
        self.n_write = n_write
        self.n_iter = n_iter
        self.salience_decay = salience_decay
        self.reconsolidation = reconsolidation
        self.sequential = sequential

        # mossy-fibre terminal field: sparse DG code -> CA3 pattern space
        self.mossy = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model)
        # Hopfield inverse temperature β per head (log-parameterised). A sharp
        # initial β matters: with ~capacity patterns competing, a flat softmax
        # returns their mean and the key projection receives almost no gradient.
        self.log_beta = nn.Parameter(torch.full((n_heads,), float(torch.log(torch.tensor(init_beta)))))
        # cue/query projections operate on normalised vectors (cosine Hopfield)
        nn.init.orthogonal_(self.wq.weight)
        nn.init.orthogonal_(self.wk.weight)
        self.drop = nn.Dropout(dropout)

    # ----------------------------------------------------------------- helpers
    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.n_heads, self.d_head).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = x.shape
        return x.transpose(1, 2).reshape(b, n, self.d_model)

    # ----------------------------------------------------------------- forward
    def forward(
        self, dg_code: torch.Tensor, store: EpisodicStore | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Returns (completed pattern, CA3 pattern h, attention mass on store)."""
        h = self.norm(self.mossy(dg_code))  # (B, T, d)
        B, T, _ = h.shape

        norm = lambda t: F.normalize(t, dim=-1)
        k_cur, v_cur = norm(self._split(self.wk(h))), self._split(self.wv(h))
        causal = torch.ones(T, T, dtype=torch.bool, device=h.device).tril()
        if store is not None:
            S = store.patterns.shape[1]
            k_mem = norm(self._split(self.wk(store.patterns.to(h.dtype))))
            v_mem = self._split(self.wv(store.values.to(h.dtype)))
            k = torch.cat([k_mem, k_cur], dim=2)
            v = torch.cat([v_mem, v_cur], dim=2)
            mem_mask = store.valid[:, None, None, :].expand(B, 1, T, S)
            mask = torch.cat([mem_mask, causal.expand(B, 1, T, T)], dim=-1)
        else:
            S = 0
            k, v = k_cur, v_cur
            mask = causal.expand(B, 1, T, T)

        beta = self.log_beta.exp().view(1, -1, 1, 1)
        xi = h
        for _ in range(self.n_iter):
            q = norm(self._split(self.wq(xi)))
            scores = (q @ k.transpose(-1, -2)) * beta
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = self.drop(scores.softmax(-1))
            retrieved = self.wo(self._merge(attn @ v))
            xi = h + retrieved  # recurrent settling around the cue

        store_mass = attn[..., :S].sum(dim=(1, 2)) / (self.n_heads * T) if S else None
        return xi, h, store_mass

    # ------------------------------------------------------------------- write
    def write(
        self,
        store: EpisodicStore | None,
        h: torch.Tensor,
        novelty: torch.Tensor,
        store_mass: torch.Tensor | None = None,
    ) -> EpisodicStore:
        """One-shot, novelty-gated storage of the segment's most novel patterns.

        h: (B, T, d) CA3 patterns, novelty: (B, T) in [0, 1].
        Slot selection is non-differentiable; the stored patterns keep their
        autograd history so recall losses in later segments can train the
        encoding path (truncate by calling ``store.detach()``).
        """
        B, T, d = h.shape
        if store is None:
            store = EpisodicStore.empty(B, self.capacity, d, h.device, h.dtype, self.sequential)
        n_cues = T - 1 if self.sequential else T   # last token has no successor yet
        with torch.no_grad():
            salience = store.salience.detach() * self.salience_decay
            if store_mass is not None:  # retrieved memories are reconsolidated
                salience = torch.where(
                    salience > 0, salience + self.reconsolidation * store_mass.detach(), salience
                )
            m = min(self.n_write, n_cues, self.capacity)
            if m <= 0:
                return EpisodicStore(store.patterns, salience, store.successors)
            # a transition is as novel as the event it leads to
            nov = (novelty[:, 1:] if self.sequential else novelty).detach()
            idx = nov.topk(m, dim=1).indices                        # most novel events
            new_s = nov.gather(1, idx).clamp_min(1e-4)
            slots = salience.topk(m, dim=1, largest=False).indices  # weakest slots
            salience = salience.scatter(1, slots, new_s)
        gather = lambda src, i: src.gather(1, i.unsqueeze(-1).expand(B, m, d))
        put = lambda dst, src: dst.scatter(1, slots.unsqueeze(-1).expand(B, m, d), src)
        patterns = put(store.patterns, gather(h, idx))
        successors = None
        if self.sequential:
            successors = put(store.successors, gather(h, idx + 1))
        return EpisodicStore(patterns, salience, successors)

    @staticmethod
    def pattern_completion_score(completed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(completed, target, dim=-1).mean()
