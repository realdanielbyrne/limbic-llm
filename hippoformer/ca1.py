"""CA1: comparator of retrieved memory (CA3) against current input (EC).

Biology
-------
CA1 pyramidal cells receive two segregated inputs:

* Schaffer collaterals from CA3 on proximal apical dendrites
  (stratum radiatum) — what memory predicts, and
* the temporoammonic path from EC layer III on distal tufts
  (stratum lacunosum-moleculare) — what the senses report.

Coincidence of the two drives plateau potentials; mismatch signals novelty
(Lisman & Grace, 2005; Hasselmo's encoding/retrieval switch).

Computation (``mode="pointwise"``, one comparison per token / neuron)
    a = W_d · x_EC          (distal compartment)
    r = W_p · ξ_CA3         (proximal compartment)
    φ = [a ; r ; a ⊙ r ; |a − r|]
    y = W_o φ               fused representation
    n = σ( MLP(φ) )         novelty ∈ (0,1),   n → 0 match, n → 1 mismatch

``mode="cross_attention"`` additionally lets each EC query attend causally over
the CA3 sequence before the comparison (the RAG-style variant).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CA1Comparator(nn.Module):
    def __init__(self, d_model: int, d_memory: int | None = None, n_heads: int = 4,
                 mode: str = "pointwise") -> None:
        super().__init__()
        if mode not in {"pointwise", "cross_attention"}:
            raise ValueError(mode)
        d_memory = d_memory or d_model
        self.mode = mode
        self.distal = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.proximal = nn.Sequential(nn.LayerNorm(d_memory), nn.Linear(d_memory, d_model))
        if mode == "cross_attention":
            self.xattn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fuse = nn.Linear(4 * d_model, d_model)
        self.novelty_head = nn.Sequential(nn.Linear(4 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, retrieved: torch.Tensor, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """retrieved: (B,T,d_memory) from CA3/LMM; current: (B,T,d) EC skip path.

        Returns fused (B,T,d) and novelty (B,T).
        """
        a = self.distal(current)
        r = self.proximal(retrieved)
        if self.mode == "cross_attention":
            T = a.shape[1]
            mask = torch.ones(T, T, dtype=torch.bool, device=a.device).triu(1)
            r, _ = self.xattn(a, r, r, attn_mask=mask, need_weights=False)
        phi = torch.cat([a, r, a * r, (a - r).abs()], dim=-1)
        fused = self.fuse(phi)
        novelty = torch.sigmoid(self.novelty_head(phi)).squeeze(-1)
        return fused, novelty

    @staticmethod
    def match_score(retrieved: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """Parameter-free cosine match, useful as a diagnostic."""
        return F.cosine_similarity(retrieved, current, dim=-1)
