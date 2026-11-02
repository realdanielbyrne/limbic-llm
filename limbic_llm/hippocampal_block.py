"""Strategy I: a direct trisynaptic-circuit block.

    x (EC) ──► DG ──► CA3 ──► CA2(FiLM) ──► CA1 ◄── x (temporoammonic skip)
                       ▲                     │
                       └── episodic store ◄──┘  (novelty-gated one-shot write)
                                             ▼
                                         Subiculum ──► x + Δ

The block keeps the residual stream width, so it drops into any
pre-norm transformer stack (context-augmentation mode) or can be stacked on
its own (standalone mode). State (the CA3 episodic store) is carried across
segments by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .ca1 import CA1Comparator
from .ca2 import CA2Modulator
from .ca3 import CA3Autoassociative, EpisodicStore
from .dentate_gyrus import DentateGyrus
from .subiculum import Subiculum


@dataclass
class HippocampalOutput:
    hidden: torch.Tensor          # (B,T,d) residual stream after the block
    novelty: torch.Tensor         # (B,T)   CA1 mismatch signal
    store: EpisodicStore          # updated CA3 episodic store
    dg_code: torch.Tensor         # (B,T,d·r) sparse granule-cell code


class HippocampalBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dg_expansion: int = 8,
        dg_sparsity: float = 0.04,
        dg_reserve: float = 0.0,
        ca3_capacity: int = 256,
        ca3_write: int = 16,
        ca3_iter: int = 1,
        ca1_mode: str = "pointwise",
        use_ca2: bool = True,
        d_context: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dg = DentateGyrus(d_model, dg_expansion, dg_sparsity, dg_reserve)
        self.ca3 = CA3Autoassociative(
            d_model * dg_expansion, d_model, n_heads, ca3_capacity, ca3_write, ca3_iter, dropout=dropout
        )
        self.ca2 = CA2Modulator(d_model, d_context) if use_ca2 else None
        self.ca1 = CA1Comparator(d_model, d_model, n_heads, ca1_mode)
        self.subiculum = Subiculum(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        store: EpisodicStore | None = None,
        context: torch.Tensor | None = None,
        write: bool = True,
    ) -> HippocampalOutput:
        code = self.dg(x)                                   # pattern separation
        completed, h, mass = self.ca3(code, store)          # pattern completion
        if self.ca2 is not None:
            completed = self.ca2(completed, context, source=x)
        fused, novelty = self.ca1(completed, x)             # EC→CA1 skip comparison
        delta = self.subiculum(fused, novelty)
        out = x + self.drop(delta)
        if write:
            store = self.ca3.write(store, h, novelty.detach(), mass)
        elif store is None:
            store = EpisodicStore.empty(x.shape[0], self.ca3.capacity, self.ca3.d_model, x.device, x.dtype,
                                        self.ca3.sequential)
        return HippocampalOutput(out, novelty, store, code)
