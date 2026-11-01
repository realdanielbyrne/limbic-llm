"""Dentate gyrus: pattern separation by expansion + k-winners-take-all.

Biology
-------
EC layer II (~2e5 cells) projects to ~1e6 granule cells, of which only 2-4 %
fire for any input. Strong feedback inhibition (basket / mossy cells) enforces
this sparsity. The result is that overlapping EC patterns are mapped to nearly
orthogonal granule-cell codes before they reach CA3.

Computation
-----------
    h      = W_exp · LN(x) + b                      h ∈ ℝ^{d·r}
    m_i    = 1[ h_i ∈ top-k(h) ],  k = ⌈s · d·r⌉     (lateral inhibition)
    y      = g ⊙ m ⊙ ReLU(h)                         (mossy-fibre gain g)

Adult neurogenesis is modelled by a pool of *reserve* units that are switched on
by :meth:`grow`. Newly born units receive an amplified gradient
(plasticity ∝ 1 + β·exp(-age/τ)), mirroring the 4–6 week hyper-plastic window
of immature granule cells.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DentateGyrus(nn.Module):
    def __init__(
        self,
        d_model: int,
        expansion: int = 8,
        sparsity: float = 0.04,
        reserve_fraction: float = 0.0,
        young_boost: float = 4.0,
        young_tau: float = 2000.0,
    ) -> None:
        super().__init__()
        if not 0.0 < sparsity <= 1.0:
            raise ValueError("sparsity must be in (0, 1]")
        self.d_model = d_model
        self.d_total = d_model * expansion
        self.sparsity = sparsity
        self.young_boost = young_boost
        self.young_tau = young_tau

        self.norm = nn.LayerNorm(d_model)
        self.expand = nn.Linear(d_model, self.d_total)
        # log mossy-fibre gain per granule cell (initialised to gain 1)
        self.log_gain = nn.Parameter(torch.zeros(self.d_total))

        n_reserve = int(round(reserve_fraction * self.d_total))
        alive = torch.ones(self.d_total, dtype=torch.bool)
        if n_reserve:
            alive[-n_reserve:] = False
        self.register_buffer("alive", alive)
        # age in optimisation steps; large value = mature
        self.register_buffer("age", torch.full((self.d_total,), 1e9))

        self.expand.weight.register_hook(self._plasticity_hook)
        self.expand.bias.register_hook(self._plasticity_hook_bias)

    # ------------------------------------------------------------------ utils
    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())

    @property
    def k(self) -> int:
        return max(1, math.ceil(self.sparsity * self.n_alive))

    def _plasticity(self) -> torch.Tensor:
        p = 1.0 + self.young_boost * torch.exp(-self.age / self.young_tau)
        return p * self.alive

    def _plasticity_hook(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self._plasticity().unsqueeze(1).to(grad.dtype)

    def _plasticity_hook_bias(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self._plasticity().to(grad.dtype)

    # ------------------------------------------------------------ neurogenesis
    @torch.no_grad()
    def grow(self, n: int) -> int:
        """Activate up to ``n`` reserve granule cells. Returns number born."""
        dead = (~self.alive).nonzero(as_tuple=True)[0][:n]
        if dead.numel() == 0:
            return 0
        self.alive[dead] = True
        self.age[dead] = 0.0
        nn.init.kaiming_uniform_(self.expand.weight[dead], a=math.sqrt(5))
        self.expand.bias[dead] = 0.0
        self.log_gain[dead] = 0.0
        return int(dead.numel())

    @torch.no_grad()
    def tick(self, steps: int = 1) -> None:
        """Advance the age of every granule cell (call once per optimiser step)."""
        self.age.add_(steps)

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand(self.norm(x))
        h = h.masked_fill(~self.alive, float("-inf"))
        top = h.topk(self.k, dim=-1)
        mask = torch.zeros_like(h, dtype=torch.bool).scatter_(-1, top.indices, True)
        h = torch.where(mask, h, torch.zeros_like(h))
        return F.relu(h) * self.log_gain.exp()

    # ------------------------------------------------------------- objectives
    @staticmethod
    def decorrelation_loss(code: torch.Tensor) -> torch.Tensor:
        """ICA-flavoured penalty: off-diagonal correlation of granule activity.

        Pushes active units toward statistical independence, the property ICA
        optimises explicitly, which minimises interference in CA3.
        ``code``: (..., D). Returns a scalar.
        """
        z = code.reshape(-1, code.shape[-1])
        z = z - z.mean(0, keepdim=True)
        z = z / (z.std(0, keepdim=True) + 1e-5)
        c = (z.T @ z) / max(1, z.shape[0] - 1)
        off = c - torch.diag(torch.diagonal(c))
        return off.pow(2).sum() / (c.shape[0] * (c.shape[0] - 1))

    @staticmethod
    def pattern_separation(x_a: torch.Tensor, x_b: torch.Tensor, code_a: torch.Tensor,
                           code_b: torch.Tensor) -> tuple[float, float]:
        """Cosine overlap of inputs vs. codes (lower output overlap = separation)."""
        cin = F.cosine_similarity(x_a.flatten(), x_b.flatten(), dim=0).item()
        cout = F.cosine_similarity(code_a.flatten(), code_b.flatten(), dim=0).item()
        return cin, cout
