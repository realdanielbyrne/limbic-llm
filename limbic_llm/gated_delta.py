"""Gated DeltaNet: a linear associative memory as CA3 / Titans-LMM special case.

Rule (Yang, Kautz & Hatamizadeh, 2024, arXiv:2412.06464), keys ℓ2-normalised:

    S_t = α_t S_{t−1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ,        o_t = S_t q_t

Equivalently, decay then take one SGD step of size β_t on the Titans
associative loss ½‖M k_t − v_t‖² with a *linear* memory M = S and no momentum:

    S_t = α_t S_{t−1} − β_t (α_t S_{t−1} k_t − v_t) k_tᵀ

so α_t plays the role of Titans' forgetting / weight decay (synaptic decay,
LTD) and β_t the role of θ_t (encoding strength); the bracket is the
prediction error that the CA1 comparator computes. With one-hot keys and
α = β = 1 the state is exact RAM: each write replaces one slot (Qwenputer).

The recurrence is evaluated exactly: token by token in ``gated_delta_scan``,
or with the equivalent chunkwise-parallel solve in ``gated_delta_chunked``
(the default, about 12× faster on CPU). (Titans' chunked
update in ``neural_memory.py`` takes every gradient in a chunk at the chunk
start, so two writes to the same key inside a chunk add instead of the second
overwriting the first.)

Addressing modes
----------------
``"gdn"``   Standard Gated DeltaNet: separate q/k/v projections, each passed
            through a short causal depthwise conv + SiLU (the conv supplies the
            token shift that binds a key to the value that follows it).
``"tied"``  RAM-style addressing: a single dense address a_t = norm(W_a x_t)
            serves as the query at t and as the key of the successor write
            (a_{t−1} ↦ v_t). Heteroassociative CA3 sequence memory.
``"dg"``    As ``"tied"`` but the address is a dentate-gyrus code: expansion
            followed by per-head k-winners-take-all, so addresses are sparse
            and nearly orthogonal (soft one-hot). ``tied`` vs ``dg`` at the same
            key dimension isolates the effect of pattern separation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neural_memory import MemoryState


# ----------------------------------------------------------------------- kernel
def gated_delta_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    S0: torch.Tensor | None = None,
    read_before_write: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact gated delta rule.

    q, k: (B,H,T,Dk)   v: (B,H,T,Dv)   alpha, beta: (B,H,T)   S0: (B,H,Dv,Dk)
    Returns outputs (B,H,T,Dv), final state (B,H,Dv,Dk) and the per-token
    prediction error ‖α S k − v‖² (the delta rule's "surprise"), (B,H,T).
    """
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    S = S0 if S0 is not None else k.new_zeros(B, H, Dv, Dk)
    outs, errs = [], []
    for t in range(T):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        if read_before_write:
            outs.append(torch.einsum("bhvk,bhk->bhv", S, qt))
        S = S * alpha[:, :, t, None, None]                      # decay (forget)
        err = vt - torch.einsum("bhvk,bhk->bhv", S, kt)          # prediction error
        S = S + beta[:, :, t, None, None] * torch.einsum("bhv,bhk->bhvk", err, kt)
        errs.append(err.pow(2).sum(-1))
        if not read_before_write:
            outs.append(torch.einsum("bhvk,bhk->bhv", S, qt))
    return torch.stack(outs, 2), S, torch.stack(errs, 2)


def gated_delta_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    S0: torch.Tensor | None = None,
    chunk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunkwise-parallel form of :func:`gated_delta_scan` (same result, exact).

    With u_t = β_t (v_t − α_t S_{t−1} k_t) and in-chunk cumulative decay
    Γ_t = Π_{s≤t} α_s, the state unrolls to S_t = Γ_t S_0 + Σ_{i≤t} (Γ_t/Γ_i) u_i k_iᵀ,
    so the u's of a chunk solve one unit-lower-triangular system

        (I + L) U = diag(β) (V − diag(Γ) K S_0ᵀ),   L_ti = β_t (Γ_t/Γ_i) k_tᵀ k_i  (i < t)

    and outputs / next state follow by matrix products. Chunks run
    sequentially; tokens inside a chunk are processed in parallel.
    """
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    S = S0 if S0 is not None else k.new_zeros(B, H, Dv, Dk)
    pad = (-T) % chunk
    if pad:   # padded steps are no-ops: β = 0, α = 1
        z = lambda t: F.pad(t, (0, 0, 0, pad))
        q, k, v = z(q), z(k), z(v)
        alpha = F.pad(alpha, (0, pad), value=1.0)
        beta = F.pad(beta, (0, pad), value=0.0)
    N, C = (T + pad) // chunk, chunk
    rs = lambda t: t.reshape(B, H, N, C, *t.shape[3:])
    q, k, v, beta = rs(q), rs(k), rs(v), rs(beta)
    g = rs(torch.log(alpha.clamp_min(1e-30))).cumsum(-1)                # log Γ_t, (B,H,N,C)
    diff = g[..., :, None] - g[..., None, :]                             # log Γ_t/Γ_i
    strict = torch.ones(C, C, dtype=torch.bool, device=k.device).tril(-1)
    incl = torch.ones(C, C, dtype=torch.bool, device=k.device).tril()
    decay_strict = torch.exp(diff.masked_fill(~strict, float("-inf")))
    decay_incl = torch.exp(diff.masked_fill(~incl, float("-inf")))
    KK = k @ k.transpose(-1, -2)                                         # (B,H,N,C,C)
    A = torch.eye(C, device=k.device, dtype=k.dtype) + beta[..., None] * decay_strict * KK
    QK = decay_incl * (q @ k.transpose(-1, -2))
    gam = g.exp()                                                        # Γ_t
    gam_last = gam[..., -1:]                                             # Γ_C
    tail = torch.exp(g[..., -1:] - g)                                    # Γ_C/Γ_i without 0/0
    outs, errs = [], []
    for n in range(N):
        kn, vn, qn, bn = k[:, :, n], v[:, :, n], q[:, :, n], beta[:, :, n]
        S0k = gam[:, :, n, :, None] * (kn @ S.transpose(-1, -2))        # Γ_t S_0 k_t, (B,H,C,Dv)
        rhs = bn[..., None] * (vn - S0k)
        U = torch.linalg.solve_triangular(A[:, :, n], rhs, upper=False, unitriangular=True)
        pred = S0k + (decay_strict[:, :, n] * KK[:, :, n]) @ U          # α_t S_{t−1} k_t
        errs.append((vn - pred).pow(2).sum(-1))
        outs.append(gam[:, :, n, :, None] * (qn @ S.transpose(-1, -2)) + QK[:, :, n] @ U)
        w = tail[:, :, n, :, None] * kn                                 # (Γ_C/Γ_i) k_i
        S = gam_last[:, :, n, :, None] * S + U.transpose(-1, -2) @ w
    o = torch.cat(outs, 2)[:, :, :T]
    err = torch.cat(errs, 2)[:, :, :T]
    return o, S, err


def _run(q, k, v, alpha, beta, S0, chunk_size):
    if chunk_size is None or k.shape[2] <= 1:
        return gated_delta_scan(q, k, v, alpha, beta, S0)
    return gated_delta_chunked(q, k, v, alpha, beta, S0, chunk_size)


def kwta(h: torch.Tensor, frac: float) -> torch.Tensor:
    """k-winners-take-all over the last dim, ReLU on winners (DG inhibition)."""
    k = max(1, math.ceil(frac * h.shape[-1]))
    top = h.topk(k, dim=-1)
    out = torch.zeros_like(h).scatter(-1, top.indices, F.relu(top.values))
    return out


class CausalConv1d(nn.Module):
    """Depthwise causal conv with a cache so streaming calls stay exact."""

    def __init__(self, channels: int, kernel: int = 4) -> None:
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, bias=False)
        with torch.no_grad():                       # start near identity (current token)
            self.conv.weight.zero_()
            self.conv.weight[:, 0, -1] = 1.0

    def forward(self, x: torch.Tensor, cache: torch.Tensor | None = None):
        # x: (B,T,C); cache: (B,K−1,C) previous inputs
        B, T, C = x.shape
        if cache is None:
            cache = x.new_zeros(B, self.kernel - 1, C)
        xx = torch.cat([cache, x], 1)
        y = self.conv(xx.transpose(1, 2)).transpose(1, 2)
        return y, xx[:, -(self.kernel - 1):]


class _HeadRMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _gates(proj_a: nn.Linear, proj_b: nn.Linear, x: torch.Tensor):
    """α = exp(−softplus(·)) ∈ (0,1) (decay), β = σ(·) ∈ (0,1) (write strength)."""
    alpha = torch.exp(-F.softplus(proj_a(x)))
    beta = torch.sigmoid(proj_b(x))
    return alpha.transpose(1, 2), beta.transpose(1, 2)        # (B,H,T)


# ------------------------------------------------------------------ mixer layer
@dataclass
class GDNState:
    S: torch.Tensor                       # (B,H,Dv,Dk)
    conv: tuple | None = None             # conv caches for q,k,v ("gdn")
    last_addr: torch.Tensor | None = None # (B,H,Dk) address of the previous token

    def detach(self) -> "GDNState":
        d = lambda t: None if t is None else t.detach()
        return GDNState(self.S.detach(), None if self.conv is None else tuple(d(c) for c in self.conv),
                        d(self.last_addr))


class GatedDeltaNet(nn.Module):
    """Token mixer: drop-in replacement for an attention sublayer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_head: int | None = None,
        addressing: str = "gdn",
        key_expansion: int = 1,
        dg_sparsity: float = 0.05,
        conv_size: int = 4,
        init_decay_bias: float = -5.0,
        output_gate: bool = True,
        chunk_size: int | None = 32,
    ) -> None:
        super().__init__()
        if addressing not in {"gdn", "tied", "dg"}:
            raise ValueError(addressing)
        self.addressing = addressing
        self.chunk_size = chunk_size
        self.h = n_heads
        self.dv = d_head or d_model // n_heads
        self.dk = self.dv * key_expansion
        self.sparsity = dg_sparsity

        if addressing == "gdn":
            self.wq = nn.Linear(d_model, self.h * self.dk, bias=False)
            self.wk = nn.Linear(d_model, self.h * self.dk, bias=False)
            self.conv_q = CausalConv1d(self.h * self.dk, conv_size)
            self.conv_k = CausalConv1d(self.h * self.dk, conv_size)
            self.conv_v = CausalConv1d(self.h * self.dv, conv_size)
        else:
            self.wa = nn.Linear(d_model, self.h * self.dk, bias=addressing == "dg")
        self.wv = nn.Linear(d_model, self.h * self.dv, bias=False)
        self.w_alpha = nn.Linear(d_model, self.h)
        self.w_beta = nn.Linear(d_model, self.h)
        nn.init.constant_(self.w_alpha.bias, init_decay_bias)   # α ≈ 0.993 at init
        self.norm = _HeadRMSNorm(self.dv)
        self.wg = nn.Linear(d_model, self.h * self.dv, bias=False) if output_gate else None
        self.wo = nn.Linear(self.h * self.dv, d_model, bias=False)

    # ------------------------------------------------------------- helpers
    def _heads(self, x: torch.Tensor, d: int) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.h, d).transpose(1, 2)            # (B,H,T,d)

    def address(self, x: torch.Tensor) -> torch.Tensor:
        """RAM address for tied/dg modes, (B,H,T,Dk), unit norm (or zero)."""
        a = self._heads(self.wa(x), self.dk)
        if self.addressing == "dg":
            a = kwta(a, self.sparsity)
        return F.normalize(a, dim=-1)

    def init_state(self, B: int, device=None, dtype=None) -> GDNState:
        return GDNState(torch.zeros(B, self.h, self.dv, self.dk, device=device, dtype=dtype))

    # ------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, state: GDNState | None = None):
        B, T, _ = x.shape
        state = state or self.init_state(B, x.device, x.dtype)
        v_raw = self.wv(x)
        if self.addressing == "gdn":
            cq, ck, cv = state.conv or (None, None, None)
            q, cq = self.conv_q(self.wq(x), cq)
            k, ck = self.conv_k(self.wk(x), ck)
            v, cv = self.conv_v(v_raw, cv)
            q = F.normalize(self._heads(F.silu(q), self.dk), dim=-1)
            k = F.normalize(self._heads(F.silu(k), self.dk), dim=-1)
            v = self._heads(F.silu(v), self.dv)
            conv, last = (cq, ck, cv), None
        else:
            a = self.address(x)
            prev = state.last_addr if state.last_addr is not None else a.new_zeros(B, self.h, self.dk)
            k = torch.cat([prev.unsqueeze(2), a[:, :, :-1]], 2)  # key of write t = address of t−1
            q, v = a, self._heads(v_raw, self.dv)
            conv, last = None, a[:, :, -1]
        alpha, beta = _gates(self.w_alpha, self.w_beta, x)
        o, S, err = _run(q, k, v, alpha, beta, state.S, self.chunk_size)
        o = self.norm(o).transpose(1, 2).reshape(B, T, self.h * self.dv)
        if self.wg is not None:
            o = o * F.silu(self.wg(x))
        return self.wo(o), GDNState(S, conv, last), err.mean(1)


# --------------------------------------------- Titans-compatible memory module
class GatedDeltaMemory(nn.Module):
    """Linear delta-rule memory with the NeuralMemory read/write interface.

    Plugs into ``TitansHippocampalLayer(memory_type="gated_delta")``. Writes run
    the exact per-token recurrence over the chunk; reads use the state before
    the chunk (as in Titans). ``lr_scale`` (CA1 novelty) scales β, and
    ``sequential`` binds a_{t−1} ↦ v_t, with the last address carried across
    chunks so no transition is dropped.
    """

    def __init__(self, d_model: int, d_memory: int | None = None, n_heads: int = 1,
                 use_dg: bool = False, key_expansion: int = 1, dg_sparsity: float = 0.05,
                 sequential: bool = False, novelty_gain: float = 1.0,
                 init_decay_bias: float = -5.0, chunk_size: int | None = 32) -> None:
        super().__init__()
        d_memory = d_memory or d_model
        self.chunk_size = chunk_size
        self.h = n_heads
        self.dv = d_memory // n_heads
        self.dk = self.dv * key_expansion
        self.use_dg = use_dg
        self.sparsity = dg_sparsity
        self.sequential = sequential
        self.novelty_gain = novelty_gain
        self.dg = None                                    # for API parity with NeuralMemory
        self.norm = nn.LayerNorm(d_model)
        self.wk = nn.Linear(d_model, self.h * self.dk, bias=use_dg)
        self.wq = self.wk if (use_dg or sequential) else nn.Linear(d_model, self.h * self.dk, bias=False)
        self.wv = nn.Linear(d_model, self.h * self.dv, bias=False)
        self.out = nn.Linear(self.h * self.dv, d_model)
        self.w_alpha = nn.Linear(d_model, self.h)
        self.w_beta = nn.Linear(d_model, self.h)
        nn.init.constant_(self.w_alpha.bias, init_decay_bias)

    def _heads(self, x, d):
        B, T, _ = x.shape
        return x.view(B, T, self.h, d).transpose(1, 2)

    def _addr(self, proj, h):
        a = self._heads(proj(h), self.dk)
        if self.use_dg:
            a = kwta(a, self.sparsity)
        return F.normalize(a, dim=-1)

    def init_state(self, batch: int) -> MemoryState:
        p = self.wv.weight
        S = p.new_zeros(batch, self.h, self.dv, self.dk)
        return MemoryState([S], [p.new_zeros(batch, self.h, self.dk)])  # momentum slot holds last address

    def read(self, x: torch.Tensor, state: MemoryState) -> torch.Tensor:
        B, T, _ = x.shape
        q = self._addr(self.wq, self.norm(x))
        o = torch.einsum("bhvk,bhtk->bhtv", state.weights[0], q)
        return self.out(o.transpose(1, 2).reshape(B, T, self.h * self.dv))

    def write(self, x: torch.Tensor, state: MemoryState, lr_scale: torch.Tensor | None = None):
        h = self.norm(x)
        k = self._addr(self.wk, h)
        v = self._heads(self.wv(h), self.dv)
        last = k[:, :, -1]
        if self.sequential:
            k = torch.cat([state.momentum[0].unsqueeze(2), k[:, :, :-1]], 2)
        alpha, beta = _gates(self.w_alpha, self.w_beta, h)
        if lr_scale is not None:
            beta = (beta * (1.0 + self.novelty_gain * (lr_scale.unsqueeze(1) - 0.5))).clamp(0, 1)
        _, S, err = _run(k, k, v, alpha, beta, state.weights[0], self.chunk_size)
        return MemoryState([S], [last]), err.mean(1).detach()
