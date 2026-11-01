"""Subiculum: burst-mode output amplification.

Subicular pyramidal cells fire intrinsic bursts that amplify hippocampal
output to distant targets. Here a SwiGLU projection is scaled by a gain that
depends on the CA1 novelty signal:

    y = g(n) · W_2 ( SiLU(W_g x) ⊙ W_1 x ),     g(n) = softplus(a·n + b)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Subiculum(nn.Module):
    def __init__(self, d_model: int, mult: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, mult * d_model, bias=False)
        self.wg = nn.Linear(d_model, mult * d_model, bias=False)
        self.w2 = nn.Linear(mult * d_model, d_model, bias=False)
        self.gain_a = nn.Parameter(torch.tensor(0.0))
        self.gain_b = nn.Parameter(torch.tensor(0.5413))  # softplus(0.5413) ≈ 1

    def forward(self, x: torch.Tensor, novelty: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm(x)
        y = self.w2(F.silu(self.wg(h)) * self.w1(h))
        if novelty is not None:
            y = y * F.softplus(self.gain_a * novelty + self.gain_b).unsqueeze(-1)
        return y
