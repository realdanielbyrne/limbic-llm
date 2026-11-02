"""CA2: conditional modulation side channel (FiLM).

Biology
-------
CA2 lacks mossy-fibre input but receives vasopressinergic and supramammillary
projections that no other CA field gets; it is required for social memory.
It acts as a side channel that biases CA3→CA1 traffic by context.

Computation
-----------
    γ, β = f(c)                       c = external context or temporal context
    y    = x ⊙ (1 + γ) + β            (initialised to the identity: γ = β = 0)

When no external context is supplied, a slowly drifting *temporal context*
c_t = (1-ρ) Σ_{s≤t} ρ^{t-s} x_s is used (Howard & Kahana's temporal context
model), computed as a causal depthwise convolution with a learnable ρ.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalContext(nn.Module):
    def __init__(self, d_model: int, window: int = 64, init_rho: float = 0.9) -> None:
        super().__init__()
        self.window = window
        self.logit_rho = nn.Parameter(torch.full((d_model,), float(torch.logit(torch.tensor(init_rho)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        L = min(self.window, T)
        rho = torch.sigmoid(self.logit_rho)  # (D,)
        lags = torch.arange(L - 1, -1, -1, device=x.device, dtype=x.dtype)  # oldest first
        kernel = (1 - rho)[:, None] * rho[:, None] ** lags[None, :]  # (D, L)
        xt = F.pad(x.transpose(1, 2), (L - 1, 0))
        return F.conv1d(xt, kernel.unsqueeze(1), groups=D).transpose(1, 2)


class CA2Modulator(nn.Module):
    def __init__(self, d_model: int, d_context: int | None = None, window: int = 64) -> None:
        super().__init__()
        self.temporal = TemporalContext(d_model, window) if d_context is None else None
        d_c = d_model if d_context is None else d_context
        self.film = nn.Sequential(nn.LayerNorm(d_c), nn.Linear(d_c, 2 * d_model))
        nn.init.zeros_(self.film[1].weight)
        nn.init.zeros_(self.film[1].bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None,
                source: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B,T,d) stream to modulate. context: (B,d_c) or (B,T,d_c).

        ``source`` is the signal the temporal context integrates when no
        context is given (defaults to ``x``).
        """
        if context is None:
            if self.temporal is None:
                raise ValueError("CA2 built with d_context requires a context tensor")
            context = self.temporal(x if source is None else source)
        if context.dim() == 2:
            context = context.unsqueeze(1)
        gamma, beta = self.film(context).chunk(2, dim=-1)
        return x * (1 + gamma) + beta
