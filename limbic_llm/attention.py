"""Shared attention utilities and a plain pre-norm transformer layer.

The plain layer is the "neocortex" / EC encoder and the baseline model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T: int, device=None, window: int | None = None) -> torch.Tensor:
    """Bool mask (T,T); True = may attend. Optional sliding window."""
    i = torch.arange(T, device=device)
    m = i[None, :] <= i[:, None]
    if window is not None:
        m &= (i[:, None] - i[None, :]) < window
    return m


def with_prefix(mask: torch.Tensor, n_prefix: int) -> torch.Tensor:
    """Prepend ``n_prefix`` always-visible key columns to a (T,S) mask."""
    if n_prefix == 0:
        return mask
    pre = torch.ones(mask.shape[0], n_prefix, dtype=torch.bool, device=mask.device)
    return torch.cat([pre, mask], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h = n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        kv = x if kv is None else kv
        B, T, D = x.shape
        S = kv.shape[1]
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(kv).view(B, S, 2, self.h, self.dh).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        # Explicit softmax attention (not F.scaled_dot_product_attention): Titans
        # meta-gradients differentiate through this twice, and the fused CPU
        # kernels have no double-backward.
        scores = (q @ k.transpose(-1, -2)) / self.dh**0.5
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.dropout(scores.softmax(-1), self.dropout, self.training)
        y = attn @ v
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, mult * d_model), nn.GELU(),
            nn.Linear(mult * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerLayer(nn.Module):
    """Pre-norm causal transformer layer with optional sliding window."""

    def __init__(self, d_model: int, n_heads: int, window: int | None = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.window = window
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = causal_mask(x.shape[1], x.device, self.window)
        x = x + self.attn(self.norm(x), mask=mask)
        return x + self.ff(x)
