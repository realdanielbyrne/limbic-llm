"""Titans long-term memory module (LMM) with optional dentate-gyrus write path.

Implements Behrouz, Zhong & Mirrokni (2024), arXiv:2501.00663, §3:

    k_t = W_K x_t,  v_t = W_V x_t,  q_t = W_Q x_t           (k, q ℓ2-normalised)
    ℓ(M; x_t) = ‖ M(k_t) − v_t ‖²                            associative loss
    S_t = η_t S_{t−1} − θ_t ∇ℓ(M_{t−1}; x_t)                  past + momentary surprise
    M_t = (1 − α_t) M_{t−1} + S_t                             forgetting = weight decay
    y_t = M*_{t}(q_t)                                         read: forward pass, no update

Parallelisation follows the paper's mini-batch scheme: within a chunk of C
tokens every gradient is taken w.r.t. the memory at the chunk start. As a
simplification, θ_t is kept per token (it weights each token's loss term)
while η and α are averaged per chunk, so one fused update is applied per
chunk.

Hippocampal additions (Strategy II)
-----------------------------------
* ``dg``: keys/values are projected from a sparse DG code, i.e. writes are
  pattern-separated before reaching the associative store.
* ``lr_scale``: an external novelty signal (CA1 mismatch) multiplies θ_t,
  so mismatching events are encoded more strongly (Hasselmo's ACh-gated
  encoding mode).
* ``sequential``: heteroassociative writes k_t = W_K x_t ↦ v_t = W_V x_{t+1}
  (CA3 sequence memory via asymmetric STDP), so a cue retrieves what followed
  it. Transitions are formed within a chunk; the one that spans a chunk
  boundary is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dentate_gyrus import DentateGyrus


@dataclass
class MemoryState:
    weights: list[torch.Tensor]   # per-sample memory parameters, each (B, d_in, d_out)
    momentum: list[torch.Tensor]  # surprise momentum S, same shapes

    def detach(self) -> "MemoryState":
        return MemoryState([w.detach() for w in self.weights], [m.detach() for m in self.momentum])


class NeuralMemory(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_memory: int | None = None,
        depth: int = 2,
        hidden_mult: int = 2,
        dg: DentateGyrus | None = None,
        meta_grad: bool = True,
        max_lr: float = 0.4,
        novelty_gain: float = 1.0,
        sequential: bool = False,
        max_update_norm: float | None = 4.0,
    ) -> None:
        super().__init__()
        d_memory = d_memory or d_model
        self.d_memory = d_memory
        self.sequential = sequential
        self.max_update_norm = max_update_norm
        self.depth = depth
        self.meta_grad = meta_grad
        self.max_lr = max_lr
        self.novelty_gain = novelty_gain
        self.dg = dg

        d_feat = dg.d_total if dg is not None else d_model
        self.norm = nn.LayerNorm(d_model)
        self.wk = nn.Linear(d_feat, d_memory, bias=False)
        self.wv = nn.Linear(d_feat, d_memory, bias=False)
        self.wq = nn.Linear(d_feat, d_memory, bias=False)
        self.out = nn.Linear(d_memory, d_model)

        dims = [d_memory] + [hidden_mult * d_memory] * (depth - 1) + [d_memory]
        self.init_weights = nn.ParameterList(
            nn.Parameter(torch.randn(a, b) * (1.0 / a) ** 0.5) for a, b in zip(dims[:-1], dims[1:])
        )
        # data-dependent θ (lr), η (momentum), α (forgetting)
        self.to_theta = nn.Linear(d_model, 1)
        self.to_eta = nn.Linear(d_model, 1)
        self.to_alpha = nn.Linear(d_model, 1)
        nn.init.constant_(self.to_eta.bias, 0.0)     # η ≈ 0.5 (effective lr θ/(1−η) stays stable)
        nn.init.constant_(self.to_alpha.bias, -4.0)  # α ≈ 0.018

    # ------------------------------------------------------------------ state
    def init_state(self, batch: int) -> MemoryState:
        w = [p.unsqueeze(0).expand(batch, *p.shape) for p in self.init_weights]
        m = [torch.zeros_like(x) for x in w]
        return MemoryState(w, m)

    # ----------------------------------------------------------------- memory
    def _apply(self, weights: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, w in enumerate(weights):
            h = torch.bmm(h, w)
            if i < len(weights) - 1:
                h = F.silu(h)
        return h + x if len(weights) > 1 else h  # residual for deep memories

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return self.dg(h) if self.dg is not None else h

    # ------------------------------------------------------------------- read
    def read(self, x: torch.Tensor, state: MemoryState) -> torch.Tensor:
        q = F.normalize(self.wq(self._features(x)), dim=-1)
        return self.out(self._apply(state.weights, q))

    # ------------------------------------------------------------------ write
    def write(
        self, x: torch.Tensor, state: MemoryState, lr_scale: torch.Tensor | None = None
    ) -> tuple[MemoryState, torch.Tensor]:
        """Memorise a chunk x:(B,C,d). Returns new state and per-token surprise ℓ_t."""
        C = x.shape[1]
        if self.sequential and C < 2:
            return state, torch.zeros(x.shape[:2], device=x.device, dtype=x.dtype)
        feat = self._features(x)
        k = F.normalize(self.wk(feat), dim=-1)
        v = self.wv(feat)
        h = self.norm(x)
        theta = self.max_lr * torch.sigmoid(self.to_theta(h)).squeeze(-1)  # (B,C)
        if lr_scale is not None:
            theta = theta * (1.0 + self.novelty_gain * (lr_scale - 0.5))
        if self.sequential:          # bind each cue to its successor
            k, v, theta = k[:, :-1], v[:, 1:], theta[:, 1:]
        eta = torch.sigmoid(self.to_eta(h)).mean(dim=(1, 2))               # (B,)
        alpha = torch.sigmoid(self.to_alpha(h)).mean(dim=(1, 2))           # (B,)

        create_graph = self.meta_grad and torch.is_grad_enabled() and self.training
        with torch.enable_grad():
            # Differentiate w.r.t. a fresh alias of M so ∇ℓ is the *partial*
            # derivative through M(k_t) only. Using M itself would also pick up
            # paths where x_t was computed from an earlier read of M (MAC/MAL),
            # making training apply a different update rule than inference.
            W = [(w + 0) if (create_graph and w.requires_grad) else w.detach().requires_grad_(True)
                 for w in state.weights]
            pred = self._apply(W, k if create_graph else k.detach())
            target = v if create_graph else v.detach()
            # ‖M(k) − v‖² summed over features, as in the paper; a mean would
            # shrink every write by 1/d and the memory would barely move.
            surprise = (pred - target).pow(2).sum(-1)                       # (B,C)
            th = theta if create_graph else theta.detach()
            loss = (th * surprise).sum()
            grads = torch.autograd.grad(loss, W, create_graph=create_graph)

        # Per-sample trust region: rescale the fused chunk update so its norm is
        # at most max_update_norm. Correlated keys summed over a chunk can
        # otherwise overshoot and diverge under momentum.
        if self.max_update_norm is not None:
            gn = torch.sqrt(sum(g.pow(2).sum(dim=(1, 2)) for g in grads))         # (B,)
            scale = (self.max_update_norm / (gn + 1e-6)).clamp(max=1.0).view(-1, 1, 1)
            grads = [g * scale for g in grads]

        e, a = eta.view(-1, 1, 1), alpha.view(-1, 1, 1)
        new_m, new_w = [], []
        for w, s, g in zip(state.weights, state.momentum, grads):
            s = e * s - g
            new_m.append(s)
            new_w.append((1 - a) * w + s)
        # First-order mode: g is a constant, but η, α and the meta-learned
        # initial weights still receive gradient through this update.
        surprise = surprise.detach()
        if self.sequential:          # align: surprise of transition t→t+1 sits at t+1
            surprise = F.pad(surprise, (1, 0))
        return MemoryState(new_w, new_m), surprise
