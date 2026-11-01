"""Strategy II: Titans layers enhanced with hippocampal components.

Three variants, matching the paper, each optionally upgraded with

* a DG sparse write path inside the LMM          (``use_dg``)
* a CA1 comparator between memory and input       (``use_ca1``)
* novelty-scaled memory learning rate             (``novelty_lr``)
* heteroassociative cue→successor writes          (``sequential``)

With all four switched off these layers reduce to plain Titans
MAC / MAG / MAL, which makes ablations a one-flag change.

Causality: memory is read with the state *before* the current chunk and
written *after* it, and memory tokens prepended in MAC are masked causally,
so no token ever sees information from its own future.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .attention import FeedForward, MultiHeadAttention, causal_mask, with_prefix
from .ca1 import CA1Comparator
from .dentate_gyrus import DentateGyrus
from .neural_memory import MemoryState, NeuralMemory


@dataclass
class TitansOutput:
    hidden: torch.Tensor
    state: MemoryState
    surprise: torch.Tensor          # (B,T) associative-memory loss at write time
    novelty: torch.Tensor | None    # (B,T) CA1 mismatch, if enabled


class PersistentMemory(nn.Module):
    """Data-independent learnable tokens holding task knowledge (slow weights)."""

    def __init__(self, n_tokens: int, d_model: int) -> None:
        super().__init__()
        self.n = n_tokens
        self.tokens = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)

    def forward(self, batch: int) -> torch.Tensor:
        return self.tokens.unsqueeze(0).expand(batch, -1, -1)


class TitansHippocampalLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        variant: str = "mac",
        chunk: int = 32,
        window: int = 32,
        n_persistent: int = 8,
        mem_depth: int = 2,
        use_dg: bool = True,
        dg_expansion: int = 4,
        dg_sparsity: float = 0.05,
        use_ca1: bool = True,
        novelty_lr: bool = True,
        sequential: bool = True,
        meta_grad: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if variant not in {"mac", "mag", "mal"}:
            raise ValueError(variant)
        self.variant = variant
        self.chunk = chunk
        self.window = window
        self.novelty_lr = novelty_lr and use_ca1

        dg = DentateGyrus(d_model, dg_expansion, dg_sparsity) if use_dg else None
        self.memory = NeuralMemory(d_model, d_model, mem_depth, dg=dg, meta_grad=meta_grad,
                                   sequential=sequential)
        self.persistent = PersistentMemory(n_persistent, d_model)
        self.ca1 = CA1Comparator(d_model, d_model, n_heads) if use_ca1 else None

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        if variant == "mag":
            self.gate = nn.Linear(2 * d_model + 1, d_model)

    # ----------------------------------------------------------------- helpers
    def _compare(self, retrieved: torch.Tensor, x: torch.Tensor):
        if self.ca1 is None:
            return retrieved, None
        return self.ca1(retrieved, x)

    def _scan_memory(self, x: torch.Tensor, state: MemoryState, write_src: torch.Tensor | None = None):
        """Chunked read-then-write over the whole sequence (MAG / MAL)."""
        write_src = x if write_src is None else write_src
        reads, fused_all, novs, surps = [], [], [], []
        for s in range(0, x.shape[1], self.chunk):
            xc = x[:, s:s + self.chunk]
            r = self.memory.read(xc, state)
            fused, nov = self._compare(r, xc)
            state, sur = self.memory.write(
                write_src[:, s:s + self.chunk], state,
                nov.detach() if (self.novelty_lr and nov is not None) else None,
            )
            fused_all.append(fused)
            surps.append(sur)
            if nov is not None:
                novs.append(nov)
        novelty = torch.cat(novs, 1) if novs else None
        return torch.cat(fused_all, 1), state, torch.cat(surps, 1), novelty

    def _swa(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        P = self.persistent(B)
        mask = with_prefix(causal_mask(T, x.device, self.window), self.persistent.n)
        kv = self.norm_kv(torch.cat([P, x], 1))
        return self.attn(self.norm_q(x), kv, mask)

    # ----------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, state: MemoryState | None = None) -> TitansOutput:
        B = x.shape[0]
        state = state or self.memory.init_state(B)
        return getattr(self, f"_{self.variant}")(x, state)

    def _mac(self, x: torch.Tensor, state: MemoryState) -> TitansOutput:
        outs, novs, surps = [], [], []
        P = self.persistent(x.shape[0])
        for s in range(0, x.shape[1], self.chunk):
            xc = x[:, s:s + self.chunk]
            c = xc.shape[1]
            r = self.memory.read(xc, state)            # long-term retrieval h_t
            fused, nov = self._compare(r, xc)          # CA1 comparator
            ctx = torch.cat([P, fused, xc], 1)         # [persistent ; memory ; chunk]
            cm = causal_mask(c, x.device)
            mask = with_prefix(torch.cat([cm, cm], -1), self.persistent.n)
            y = xc + self.attn(self.norm_q(xc), self.norm_kv(ctx), mask)
            y = y + self.ff(y)
            state, sur = self.memory.write(
                y, state, nov.detach() if (self.novelty_lr and nov is not None) else None
            )
            outs.append(y)
            surps.append(sur)
            if nov is not None:
                novs.append(nov)
        return TitansOutput(torch.cat(outs, 1), state, torch.cat(surps, 1),
                            torch.cat(novs, 1) if novs else None)

    def _mag(self, x: torch.Tensor, state: MemoryState) -> TitansOutput:
        y_att = self._swa(x)
        y_mem, state, sur, nov = self._scan_memory(x, state)
        n = nov if nov is not None else torch.zeros(x.shape[:2], device=x.device, dtype=x.dtype)
        g = torch.sigmoid(self.gate(torch.cat([y_att, y_mem, n.unsqueeze(-1)], -1)))
        h = x + g * y_att + (1 - g) * y_mem                 # CA2-like multiplicative gating
        return TitansOutput(h + self.ff(h), state, sur, nov)

    def _mal(self, x: torch.Tensor, state: MemoryState) -> TitansOutput:
        y_mem, state, sur, nov = self._scan_memory(x, state)
        h = x + y_mem                                        # memory layer first (DG→CA3→CA1)
        h = h + self._swa(h)                                 # then local attention
        return TitansOutput(h + self.ff(h), state, sur, nov)
