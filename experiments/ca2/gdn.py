"""GDN taps: per-token signals from the frozen Gated DeltaNet layers of a hybrid (Qwen3.5), GDN relevance
sources for the CA2 gain, and a branch-gain law on the GDN output. Nothing is trained.

The layer (``Qwen3_5GatedDeltaNet.forward``), per value head h (16 heads × 128 dims on Qwen3.5-2B):

    x = input_layernorm(h_block)            (zeroed at padding)
    [q; k; v] = SiLU(conv1d_4(in_proj_qkv(x)))        depthwise causal kernel-4 conv (no bias)
    β_t = σ(in_proj_b(x))                   write gate
    g_t = −exp(A_log)·softplus(in_proj_a(x) + dt_bias),   ρ_t = exp(g_t) ∈ (0, 1) retention
    k̂, q̂ = l2norm(k), l2norm(q);  q̃ = q̂/√d_k
    S_t = ρ_t S_{t−1} + k̂_t (β_t e_t)ᵀ,    e_t = v_t − ρ_t S_{t−1}ᵀ k̂_t      ([K, V] layout)
    o_t = S_tᵀ q̃_t                          (read after the write)
    u_t = out_proj(RMSNorm(o_t)·w ⊙ SiLU(z_t))   the branch output added to the residual

Hooks (``GDNTaps``, all read-only except the branch gain): ``linear_attn`` forward pre-hook (with kwargs:
``cache_params`` and the padding mask; the conv tail and S_{−1} are *cloned* there, before the layer updates
the cache in place), ``in_proj_qkv`` / ``in_proj_b`` / ``in_proj_a`` forward hooks (pre-conv mix, b, a),
the ``norm`` forward pre-hook (the model's own o_t and z; the per-token loop runs here) and the
``out_proj`` forward hook (branch output; the branch-gain law u_t ← (1 + g_t) u_t is applied here, after
the state write, so a gain at layer l never changes layer l's state; it changes every layer above).

The post-conv q, k, v are locals of the layer (the conv module itself is never called, so a hook on it
never fires). They are recomputed with the model module's own ``causal_conv1d_fn`` /
``causal_conv1d_update`` on the hooked pre-conv tensor, with the same branch the layer takes: no cache →
zero tail; a cache with history → the cloned ``conv_states`` (prefill continuation or one-token decode).
β and g are recomputed with the layer's exact expressions. The recomputed inputs therefore equal the
kernel's inputs bitwise (tested).

State. With no cache the model starts every call from S = 0 and a zero conv tail (HookedLM folding gives
every W-token block its own zero state), so the signals do too; there is no shadow state. With a cache,
S_{−1} and the tail come from ``cache_params`` when the layer has history.

Per-token loop (``gdn_scan``): an op-for-op copy of ``torch_recurrent_gated_delta_rule`` (its outputs and
final state are bitwise equal to the reference, tested) that also returns, per token and head,

    prediction  p̂_t = ρ_t S_{t−1}ᵀ k̂_t (``kv_mem``), mismatch e_t = v_t − p̂_t (computed directly, never
                as v_new/β), pre-write read r_t = ρ_t S_{t−1}ᵀ q̂_t (unit q̂, the 1/√d scale dropped),
    lagged prediction  p̂^L_t = exp(Σ_{u=t−L+1..t} g_u) S_{t−L}ᵀ k̂_t with L = the conv kernel (4): the
                state from before any token inside the current token's conv footprint, so a match cannot
                come from the overlapping conv windows of the last 3 tokens,
    ‖S_t‖_F, and q̂_t·k̂_t (the self term of the post-write read).

Signals per token, layer and head (fp32; NaN at padding and where undefined):
    beta β, ret ρ, logret g, vnorm ‖v‖, pred ‖p̂‖, err ‖e‖, cos_vp cos(v, p̂), ratio ‖p̂‖/‖v‖,
    fam = 1 − ‖e‖/(‖v‖ + ‖p̂‖) ∈ [0, 1] (0: empty state or opposite; 1: perfect prediction),
    read ‖r‖, str = ‖r‖/(‖r‖ + ‖v‖) (recall strength), qk q̂·k̂, snorm ‖S_t‖_F, out ‖o_t‖ (this loop),
    o_model ‖o_t‖ (the model's kernel, from the norm pre-hook), o_dev ‖o − o_model‖/‖o_model‖ (a runtime
    check of the recompute; ~1e−2 against Triton in bf16), zgate ‖SiLU(z)‖/√d_v (output gate),
    pred_lag, err_lag, cos_vp_lag, ratio_lag, fam_lag, lag_ok (1 where S_{t−L} is known),
    branch ‖u_t‖ (before the gain), bgain g_t of the branch law (0 where none).
``signals()`` returns the head mean of each, (B, T, n_layers) (key ``<name>``), and with ``per_head`` the
per-head values (B, T, n_layers, H) (key ``<name>_head``); the controller traces them as ``tap/gdn/<key>``.

Sources (registered in ``sources.SOURCES`` on import, so ``make_source("gdn_write")`` works):
    gdn_write          β (head mean, layer mean over the GDN layers below the injection site)
    gdn_decay          ρ (retention); ``invert=True`` gives 1 − ρ (forgetting, boundary-like)
    gdn_mismatch       1 − fam (novelty: normalised mismatch)       gdn_mismatch_lag: 1 − fam_lag
    gdn_familiar       fam                                          gdn_recall: str
``GDNSource(signal, transform="zscore")`` maps any signal through σ((x − μ_{t−1})/σ_{t−1}) with prefix
statistics over real tokens (carried state; same arithmetic in prefill and decode).

Branch-gain law (``BranchLaw``): u_t ← (1 + g_t) u_t at chosen GDN layers, g ≥ −1 (−1 zeroes the branch).
Modes: ``fixed`` (a constant or per-row gain, the D4 sweep and the matched-mean control), ``ctrl`` (the
controller's own g_t; layers at or above the injection site only), ``self`` (a signed ``GainPipeline`` on
the layer's own signal, which that layer's gain cannot change), ``replay`` (a donor gain trajectory by
steered step: the yoked control). Gains are zero off the controller's steered positions. With g = 0
throughout a call every hook returns None, so the model output is byte-identical to the base model.
A GDN tap or source on a model without linear-attention layers raises.

Cost: the loop is ~10 small kernels per token per layer (B·H·K·V flops each), i.e. ~37k sequential steps
per 2048-token sequence over 18 layers; ``mismatch=False`` skips it (gates, o_model, zgate, branch only).
"""

from __future__ import annotations

import math
import sys
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402
from ca2.controller import GainPipeline, Tap  # noqa: E402
from ca2.regulators import leaky_scan  # noqa: E402
from ca2.sources import SOURCES, Source, StepContext  # noqa: E402

HEAD_KEYS = ("beta", "ret", "logret", "vnorm", "pred", "err", "cos_vp", "ratio", "fam", "read", "str", "qk",
             "snorm", "out", "o_model", "o_dev", "zgate", "pred_lag", "err_lag", "cos_vp_lag", "ratio_lag",
             "fam_lag", "lag_ok")
CALL_KEYS = ("branch", "bgain")                 # per layer only (no head axis)
GATE_KEYS = ("beta", "ret", "logret", "o_model", "zgate")     # available with mismatch=False
UNIT_SIGNALS = ("beta", "ret", "fam", "fam_lag", "str", "cos_vp", "cos_vp_lag", "qk")   # usable raw as s
LAW_MODES = ("fixed", "ctrl", "self", "replay")


# --------------------------------------------------------------------------------------------- helpers
def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``modeling_qwen3_5.l2norm`` (rsqrt form, FLA-aligned)."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _module_of(lin: nn.Module):
    return sys.modules[type(lin).__module__]


def linear_layers(model: nn.Module, blocks: nn.ModuleList | None = None) -> list[int]:
    """Indices of the blocks whose token mixer is linear attention (GDN)."""
    blocks = pl.find_blocks(model) if blocks is None else blocks
    return [i for i, k in enumerate(pl.layer_kinds(model, blocks)) if k == "linear"]


def is_hybrid(model: nn.Module) -> bool:
    return bool(linear_layers(model))


def gate_params(model: nn.Module) -> dict[int, dict[str, torch.Tensor]]:
    """Per GDN block: A_log, dt_bias (H,) and the token-independent decay rate A = exp(A_log) (fp32, CPU)."""
    blocks = pl.find_blocks(model)
    out = {}
    for i in linear_layers(model, blocks):
        lin = _lin(blocks[i], i)
        out[i] = {"A_log": lin.A_log.detach().float().cpu(), "dt_bias": lin.dt_bias.detach().float().cpu(),
                  "A": lin.A_log.detach().float().exp().cpu()}
    return out


def _lin(block: nn.Module, i: int) -> nn.Module:
    lin = pl._child(block, pl.LINEAR_MIXERS)
    need = ("in_proj_qkv", "in_proj_b", "in_proj_a", "norm", "out_proj", "conv1d", "A_log", "dt_bias")
    if lin is None or not all(hasattr(lin, n) for n in need):
        raise ValueError(f"block {i} has no Gated DeltaNet mixer (need {need})")
    return lin


def recompute_qkv(lin: nn.Module, mixed: torch.Tensor, conv_state: torch.Tensor | None = None,
                  recording_past: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Post-conv q, k, v (B, T, H, d) as the layer computes them (heads repeated to the value heads).

    ``mixed`` is the ``in_proj_qkv`` output (B, T, C); ``conv_state`` the layer's cached pre-conv columns
    (B, C, K) cloned before the call, or None (no cache or no history: zero tail). Same branch and the
    same module-level conv function as ``Qwen3_5GatedDeltaNet.forward``, so the result is bitwise equal.
    """
    M = _module_of(lin)
    B, T, _ = mixed.shape
    x = mixed.transpose(1, 2)
    w = lin.conv1d.weight.squeeze(1)
    if conv_state is not None and T == 1 and not recording_past:
        y = M.causal_conv1d_update(x, conv_state.clone(), w, lin.conv1d.bias, lin.activation)
    else:
        if conv_state is not None:
            x = torch.cat([conv_state.to(x.dtype), x], dim=-1)
        y = M.causal_conv1d_fn(x, w, lin.conv1d.bias, activation=lin.activation)
        if conv_state is not None:
            y = y[:, :, -T:]
    y = y.transpose(1, 2)
    q, k, v = torch.split(y, [lin.key_dim, lin.key_dim, lin.value_dim], dim=-1)
    q = q.reshape(B, T, -1, lin.head_k_dim)
    k = k.reshape(B, T, -1, lin.head_k_dim)
    v = v.reshape(B, T, -1, lin.head_v_dim)
    rep = lin.num_v_heads // lin.num_k_heads
    if rep > 1:
        q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
    return q, k, v


def gates(lin: nn.Module, b: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(β, g) from the ``in_proj_b`` / ``in_proj_a`` outputs, with the layer's exact expressions."""
    beta = b.sigmoid()
    g = -lin.A_log.float().exp() * F.softplus(a.float() + lin.dt_bias)
    return beta, g


# ------------------------------------------------------------------------------------------ the scan
@dataclass
class GDNScan:
    """Output of ``gdn_scan``. Layout (B, T, H, ·) like the kernel's; everything fp32 except ``out``."""

    out: torch.Tensor                  # (B, T, H, V) the reference rule's output, in the input dtype
    final_state: torch.Tensor          # (B, H, K, V) fp32
    o: torch.Tensor                    # (B, T, H, V) fp32 output before the cast
    v: torch.Tensor                    # (B, T, H, V) fp32 values
    pred: torch.Tensor                 # (B, T, H, V) ρ_t S_{t−1}ᵀ k̂_t
    read: torch.Tensor                 # (B, T, H, V) ρ_t S_{t−1}ᵀ q̂_t (unit q̂)
    pred_lag: torch.Tensor | None      # (B, T, H, V) exp(Σ_{t−L+1..t} g) S_{t−L}ᵀ k̂_t
    lag_ok: torch.Tensor               # (T,) bool: S_{t−L} known
    snorm: torch.Tensor                # (B, T, H) ‖S_t‖_F
    qk: torch.Tensor                   # (B, T, H) q̂_t·k̂_t
    history: list                      # last L states S_{T−L}, …, S_{T−1} (each (B, H, K, V))
    g_history: list                    # last L − 1 log-decays g_{T−L+1}, …, g_{T−1} (each (B, H))


def gdn_scan(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
             initial_state: torch.Tensor | None = None, lag: int = 4, history: Sequence | None = None,
             g_history: Sequence | None = None) -> GDNScan:
    """Per-token gated delta rule, an exact copy of ``torch_recurrent_gated_delta_rule`` with
    ``use_qk_l2norm_in_kernel=True`` (inputs (B, T, H, d), g and β (B, T, H)) that also returns the
    prediction, the pre-write read and the lagged prediction.

    ``history``: the L states S_{−L}, …, S_{−1} before this call (S_{−1} = ``initial_state``) and
    ``g_history`` the L − 1 log-decays g_{−L+1}, …, g_{−1}. When both are None the earlier states are zero
    if ``initial_state`` is None (a fresh sequence: exact), and unknown otherwise (``lag_ok`` False where
    t − L < −1). ``lag = 0`` skips the lagged prediction.
    """
    initial_dtype = query.dtype
    B, T, _, Dk = key.shape
    H, Dv = value.shape[-2:]
    decay = g
    query, key, value, beta, decay = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                                      for x in (query, key, value, beta, decay)]
    query = _l2norm(query)
    key = _l2norm(key)
    qhat = query
    query = query / (query.shape[-1] ** 0.5)
    if initial_state is None:
        state = torch.zeros((B, H, Dk, Dv), dtype=value.dtype, device=value.device)
    else:
        state = initial_state.to(value)
    core = torch.zeros_like(value)
    pred = torch.zeros_like(value)
    read = torch.zeros_like(value)
    snorm = value.new_zeros(B, H, T)
    L = int(lag)
    pred_lag = torch.zeros_like(value) if L > 0 else None
    lag_ok = torch.ones(T, dtype=torch.bool)
    if L > 0:
        if history is not None:
            if len(history) != L or g_history is None or len(g_history) != L - 1:
                raise ValueError(f"history needs {L} states and g_history {L - 1} decays")
            hist = deque((s.to(value) for s in history), maxlen=L)
            ghist = deque((x.to(value) for x in g_history), maxlen=max(L - 1, 1))
        else:
            zero = torch.zeros_like(state)
            hist = deque([zero] * (L - 1) + [state], maxlen=L)
            ghist = deque([value.new_zeros(B, H)] * (L - 1), maxlen=max(L - 1, 1))
            if initial_state is not None:
                lag_ok[: max(L - 1, 0)] = False                 # S_{t−L} for t − L < −1 is unknown
        if L == 1:
            ghist.clear()
    for i in range(T):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        if L > 0:
            s_lag = hist[0]
            gsum = decay[:, :, i]
            for x in ghist:
                gsum = gsum + x
            pred_lag[:, :, i] = gsum.exp()[..., None] * (s_lag * k_t.unsqueeze(-1)).sum(dim=-2)
        # --- torch_recurrent_gated_delta_rule, op for op
        decay_t = decay[:, :, i].exp()[..., None, None]
        state = state * decay_t
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        # --- extras read from the same decayed pre-write state
        pred[:, :, i] = kv_mem
        read[:, :, i] = (state * qhat[:, :, i].unsqueeze(-1)).sum(dim=-2)
        # --- reference rule continued
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        snorm[:, :, i] = state.square().sum(dim=(-2, -1)).sqrt()
        if L > 0:
            hist.append(state)
            if L > 1:
                ghist.append(decay[:, :, i])
    qk = (qhat * key).sum(-1)
    t = lambda x: x.transpose(1, 2)                              # noqa: E731
    return GDNScan(out=t(core).contiguous().to(initial_dtype), final_state=state, o=t(core), v=t(value),
                   pred=t(pred), read=t(read), pred_lag=None if pred_lag is None else t(pred_lag), lag_ok=lag_ok,
                   snorm=t(snorm), qk=t(qk), history=list(hist) if L > 0 else [],
                   g_history=list(ghist) if L > 1 else [])


def _nan_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(b > 0, a / b.clamp_min(1e-30), torch.full_like(a, float("nan")))


def _match_stats(v: torch.Tensor, p: torch.Tensor, suffix: str = "") -> dict[str, torch.Tensor]:
    vn, pn = v.norm(dim=-1), p.norm(dim=-1)
    en = (v - p).norm(dim=-1)
    return {f"pred{suffix}": pn, f"err{suffix}": en, f"cos_vp{suffix}": _nan_div((v * p).sum(-1), vn * pn),
            f"ratio{suffix}": _nan_div(pn, vn), f"fam{suffix}": 1.0 - _nan_div(en, vn + pn)}


def scan_stats(scan: GDNScan, beta: torch.Tensor, g: torch.Tensor, o_model: torch.Tensor | None = None,
               z: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Per-token, per-head signals (B, T, H) from a ``GDNScan`` (module docstring for the definitions)."""
    v = scan.v
    vn = v.norm(dim=-1)
    rn = scan.read.norm(dim=-1)
    out = {"beta": beta.float(), "ret": g.float().exp(), "logret": g.float(), "vnorm": vn, "read": rn,
           "str": _nan_div(rn, rn + vn), "qk": scan.qk, "snorm": scan.snorm, "out": scan.o.norm(dim=-1)}
    out.update(_match_stats(v, scan.pred))
    if scan.pred_lag is not None:
        ok = scan.lag_ok.to(v.device)[None, :, None]
        lag = _match_stats(v, scan.pred_lag, "_lag")
        nan = torch.full_like(vn, float("nan"))
        out.update({k: torch.where(ok, x, nan) for k, x in lag.items()})
        out["lag_ok"] = ok.float().expand_as(vn)
    if o_model is not None:
        om = o_model.float()
        out["o_model"] = om.norm(dim=-1)
        out["o_dev"] = _nan_div((scan.o - om).norm(dim=-1), out["o_model"])
    if z is not None:
        out["zgate"] = F.silu(z.float()).norm(dim=-1) / math.sqrt(z.shape[-1])
    return out


# ----------------------------------------------------------------------------------------- branch law
class BranchLaw:
    """u_t ← (1 + g_t) u_t at the GDN ``layers`` (applied on ``out_proj``'s output; module docstring)."""

    def __init__(self, layers: Sequence[int], mode: str = "fixed", gain: "float | torch.Tensor" = 0.0,
                 signal: str = "fam", invert: bool = False, heads: str = "mean",
                 donor: torch.Tensor | None = None, scale: float = 1.0, g_min: float = -0.25, g_max: float = 0.5,
                 g_abs: float = 2.0, smooth: str = "none", leak: float = 0.9, kappa: float = 0.5,
                 polarity: str = "reinforce", restore_kappa: float | None = None,
                 restore_target: float = 0.5) -> None:
        if mode not in LAW_MODES:
            raise ValueError(f"mode must be one of {LAW_MODES}")
        if heads not in ("mean", "max"):
            raise ValueError("heads must be 'mean' or 'max'")
        self.layers = tuple(int(l) for l in layers)
        if not self.layers:
            raise ValueError("a branch law needs at least one layer")
        self.mode, self.signal, self.invert, self.heads, self.scale = mode, signal, bool(invert), heads, float(scale)
        self.g_abs = float(g_abs)
        self.gain = gain.detach().float() if isinstance(gain, torch.Tensor) else float(gain)
        if mode == "fixed":
            gt = torch.as_tensor(self.gain)
            if (gt < -1).any() or (gt > g_abs).any():
                raise ValueError("a fixed branch gain must lie in [-1, g_abs]")
        if mode == "self" and (g_min < -1.0):
            raise ValueError("g_min must be >= -1 (the branch cannot be inverted)")
        if mode == "self" and signal not in UNIT_SIGNALS:
            raise ValueError(f"a self-sourced law needs a signal in [0, 1]: {UNIT_SIGNALS}")
        if mode == "replay":
            if donor is None or donor.dim() != 2 or donor.shape[1] == 0:
                raise ValueError("replay needs a donor gain trajectory (B, N)")
            if (donor < -1).any() or (donor > g_abs).any():
                raise ValueError("donor gains must lie in [-1, g_abs]")
        self.donor = None if donor is None else donor.detach().float()
        self._pipe_kw = dict(g_min=g_min, g_max=g_max, g_abs=g_abs, smooth=smooth, leak=leak, kappa=kappa,
                             polarity=polarity, restore_kappa=restore_kappa, restore_target=restore_target,
                             signed=True)
        self.pipes: dict[int, GainPipeline] = {}
        if mode == "self":
            self.pipes = {l: GainPipeline(**self._pipe_kw) for l in self.layers}

    def reset(self) -> None:
        for p in self.pipes.values():
            p.reset()

    def state_numel(self) -> int:
        return sum(p.state_numel() for p in self.pipes.values())


# ----------------------------------------------------------------------------------------------- taps
class GDNTaps(Tap):
    """Hooks on every (or the chosen) GDN layer; per-token signals and the optional branch-gain law.

    layers     GDN block indices (default: every linear-attention block)
    mismatch   run the per-token loop (prediction, mismatch, read, lagged variant, state norm)
    lag        lagged-state distance L (default: the layer's conv kernel, 4)
    per_head   also return (B, T, n_layers, H) per-head signals (True, or a list of keys)
    law        a ``BranchLaw`` (or None)
    """

    name = "gdn"

    def __init__(self, layers: Sequence[int] | None = None, mismatch: bool = True, lag: int | None = None,
                 per_head: "bool | Sequence[str]" = False, law: BranchLaw | None = None, name: str = "gdn") -> None:
        self.req_layers = None if layers is None else tuple(int(l) for l in layers)
        self.mismatch, self.lag_req, self.law, self.name = bool(mismatch), lag, law, name
        self.per_head = (set(HEAD_KEYS) if per_head is True else set(per_head or ()))
        self.layers: tuple[int, ...] = ()
        self.lins: dict[int, nn.Module] = {}
        self.handles: list = []
        self.att = None
        self.ctrl = None
        self.reset()

    # ------------------------------------------------------------------ attach
    def attach(self, att) -> None:
        kinds = list(att.kinds)
        lin_idx = [i for i, k in enumerate(kinds) if k == "linear"]
        types = getattr(pl.text_config(att.model), "layer_types", None) or []
        if not lin_idx or (types and "linear_attention" not in types):
            raise ValueError("GDN taps need a model with linear-attention (Gated DeltaNet) layers; this one has none")
        layers = lin_idx if self.req_layers is None else [l % len(kinds) for l in self.req_layers]
        bad = [l for l in layers if kinds[l] != "linear"]
        if bad:
            raise ValueError(f"blocks {bad} are not GDN layers (GDN: {lin_idx})")
        self.layers = tuple(sorted(layers))
        self.att, self.ctrl = att, getattr(att, "ctrl", None)
        M = self.inject_before()
        if self.law is not None:
            miss = [l for l in self.law.layers if l not in self.layers]
            if miss:
                raise ValueError(f"branch-law layers {miss} are not tapped")
            if self.law.mode == "ctrl":
                low = [l for l in self.law.layers if l < M]
                if low:
                    raise ValueError(f"mode 'ctrl' reads the controller's gain, set at block {M}'s input; layers {low} "
                                     "run before it (use mode 'self' or 'fixed' there)")
                if self.ctrl is None:
                    raise ValueError("mode 'ctrl' needs a controller")
            if self.law.mode == "self" and not self.mismatch and self.law.signal not in GATE_KEYS:
                raise ValueError(f"signal {self.law.signal!r} needs mismatch=True")
        for l in self.layers:
            lin = _lin(att.blocks[l], l)
            self.lins[l] = lin
            self.handles += [lin.register_forward_pre_hook(self._pre(l), with_kwargs=True),
                             lin.in_proj_qkv.register_forward_hook(self._keep(l, "mixed")),
                             lin.in_proj_b.register_forward_hook(self._keep(l, "b")),
                             lin.in_proj_a.register_forward_hook(self._keep(l, "a")),
                             lin.norm.register_forward_pre_hook(self._norm(l)),
                             lin.out_proj.register_forward_hook(self._out(l))]

    def attach_model(self, model: nn.Module) -> "GDNTaps":
        """Standalone use (no controller): taps on ``model``; gains apply at every real position."""
        blocks = pl.find_blocks(model)
        from types import SimpleNamespace
        self.attach(SimpleNamespace(model=model, blocks=blocks, kinds=pl.layer_kinds(model, blocks), ctrl=None,
                                    inject_before=len(blocks), sink=0))
        return self

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def inject_before(self) -> int:
        att = self.att
        if att is None:
            return 10 ** 9
        return int(getattr(att, "inject_before", len(att.blocks)))

    # ------------------------------------------------------------------- state
    def reset(self) -> None:
        self.cur: dict[int, dict] = {}
        self.stats: dict[int, dict[str, torch.Tensor]] = {}
        self.ring: dict[int, tuple[list, list]] = {}       # carried lag history per layer (cache mode only)
        if self.law is not None:
            self.law.reset()

    def begin_call(self, ctrl) -> None:
        self.ctrl = ctrl if ctrl is not None else self.ctrl
        self.cur, self.stats = {}, {}

    def state_numel(self) -> int:
        n = sum(sum(t.numel() for t in s) + sum(t.numel() for t in gg) for s, gg in self.ring.values())
        return n + (self.law.state_numel() if self.law is not None else 0)

    # ------------------------------------------------------------------- hooks
    def _pre(self, l: int):
        def fn(mod, args, kwargs):
            cache = kwargs.get("cache_params")
            mask = kwargs.get("attention_mask")
            S0 = conv = None
            rec_past = False
            if cache is not None:
                S = cache.get_recurrent_state(l, state_idx=0)
                C = cache.get_conv_state(l, state_idx=0)
                S0 = None if S is None else S.detach().clone()
                conv = None if C is None else C.detach().clone()
                rec_past = bool(cache.is_recording_past(l)) if hasattr(cache, "is_recording_past") else False
            self.cur[l] = {"cache": cache is not None, "mask": None if mask is None else mask.detach(), "S0": S0,
                           "conv": conv, "rec_past": rec_past}
            return None
        return fn

    def _keep(self, l: int, key: str):
        def fn(mod, args, out):
            self.cur.setdefault(l, {})[key] = out.detach()
            return None
        return fn

    def _fold(self, l: int):
        """(W, sink, (B, T)) when HookedLM folded this layer's input in this call, else None."""
        shape = getattr(self.lins[l], "_limbic_fold", None)
        if shape is None:
            return None
        W = self.cur[l]["mixed"].shape[1]
        return W, int(getattr(self.att, "sink", 0)), shape

    def _unfold(self, l: int, x: torch.Tensor) -> torch.Tensor:
        f = self._fold(l)
        return x if f is None else pl.unfold(x, f[0], f[1], f[2])

    def _norm(self, l: int):
        def fn(mod, args):
            c = self.cur[l]
            lin = self.lins[l]
            mixed = c["mixed"]
            B, T, _ = mixed.shape
            H, Dv = lin.num_v_heads, lin.head_v_dim
            o_model = args[0].detach().reshape(B, T, H, Dv)
            z = args[1].detach().reshape(B, T, H, Dv)
            beta, g = gates(lin, c["b"], c["a"])
            if self.mismatch:
                q, k, v = recompute_qkv(lin, mixed, c.get("conv"), c.get("rec_past", False))
                L = int(self.lag_req if self.lag_req is not None else lin.conv_kernel_size)
                S0 = c.get("S0")
                hist = gh = None
                if S0 is not None and l in self.ring and self.ring[l][0] and self.ring[l][0][0].shape[0] == B:
                    hist, gh = self.ring[l]
                scan = gdn_scan(q, k, v, g, beta, S0, L, hist, gh)
                if c.get("cache"):
                    self.ring[l] = ([s.detach() for s in scan.history], [x.detach() for x in scan.g_history])
                else:
                    self.ring.pop(l, None)
                st = scan_stats(scan, beta, g, o_model, z)
            else:
                st = {"beta": beta.float(), "ret": g.float().exp(), "logret": g.float(),
                      "o_model": o_model.float().norm(dim=-1),
                      "zgate": F.silu(z.float()).norm(dim=-1) / math.sqrt(Dv)}
            mask = c.get("mask") if self._fold(l) is None else None
            out = {}
            for key, x in st.items():
                x = self._unfold(l, x)
                if mask is not None and key != "lag_ok":
                    x = torch.where(mask.to(x.device)[..., None].bool(), x, torch.full_like(x, float("nan")))
                out[key] = x
            self.stats[l] = out
            return None
        return fn

    def _out(self, l: int):
        def fn(mod, args, out):
            u = out
            Bf, Tf = u.shape[:2]
            g = self._law_gain(l, Bf, Tf, u.device)
            st = self.stats.setdefault(l, {})
            st["branch"] = self._unfold(l, u.detach().float().norm(dim=-1))
            st["bgain"] = self._unfold(l, g if g is not None else torch.zeros(Bf, Tf, device=u.device))
            if g is None or not bool((g != 0).any()):
                return None
            return (u.float() * (1.0 + g).unsqueeze(-1)).to(u.dtype)
        return fn

    # -------------------------------------------------------------- the law
    def _positions(self, B: int, T: int, dev) -> dict[str, torch.Tensor]:
        """Steered positions and steered-step index in model coordinates (B, T) (unfolded)."""
        ctrl = self.ctrl
        if ctrl is None:
            m = torch.ones(B, T, device=dev)
            any_mask = next((c.get("mask") for c in self.cur.values() if c.get("mask") is not None), None)
            if any_mask is not None:
                m = any_mask.to(dev).float()
            return {"steer": m, "step": (m.cumsum(1) - m).long()}
        off = int(getattr(ctrl, "col_offset", 0))
        pos = ctrl._positions(B, T - off, dev)
        pad = lambda x: torch.cat([x.new_zeros(B, off), x], 1) if off else x      # noqa: E731
        return {"steer": pad(pos["steer"].float()), "step": pad(pos["step"])}

    def _law_gain(self, l: int, Bf: int, Tf: int, dev) -> torch.Tensor | None:
        law = self.law
        if law is None or l not in law.layers:
            return None
        f = self._fold(l)
        B, T = (Bf, Tf) if f is None else f[2]
        pos = self._positions(B, T, dev)
        steer = pos["steer"]
        if law.mode == "fixed":
            gain = torch.as_tensor(law.gain, device=dev, dtype=torch.float32)
            g = (gain[:, None] if gain.dim() == 1 else gain) * steer
        elif law.mode == "ctrl":
            cur = self.ctrl.current.get("g")
            if cur is None:
                raise RuntimeError("the controller has not stepped in this call")
            off = int(getattr(self.ctrl, "col_offset", 0))
            cur = cur.float()
            g = law.scale * (torch.cat([cur.new_zeros(B, off), cur], 1) if off else cur) * steer
        elif law.mode == "replay":
            d = law.donor.to(dev)
            if d.shape[0] != B:
                raise ValueError(f"donor has {d.shape[0]} rows, the batch {B}")
            g = d.gather(1, pos["step"].clamp_min(0) % d.shape[1]) * steer
        else:
            x = self.stats[l].get(law.signal)
            if x is None:
                raise KeyError(f"signal {law.signal!r} not computed at layer {l}")
            s = _pool(x, law.heads)
            s = torch.nan_to_num(s, nan=0.0).clamp(0.0, 1.0)
            s = 1.0 - s if law.invert else s
            g = law.pipes[l](s * steer, steer)["g"]
        g = g.float()
        if f is not None:
            g = pl.fold(g, f[0], f[1])[0]
        return g

    # ----------------------------------------------------------------- output
    def head_stat(self, l: int, key: str) -> torch.Tensor:
        """This call's per-head signal ``key`` at GDN block ``l``: (B, T_model, H) (or (B, T_model))."""
        if l not in self.stats or key not in self.stats[l]:
            raise KeyError(f"GDN signal {key!r} at block {l} is not available in this call "
                           f"(the layer has not run yet, or mismatch=False)")
        return self.stats[l][key]

    def signals(self) -> dict[str, torch.Tensor]:
        if not self.layers or any(l not in self.stats for l in self.layers):
            return {}
        out = {}
        keys = [k for k in HEAD_KEYS if all(k in self.stats[l] for l in self.layers)]
        for k in keys:
            per = [self.stats[l][k] for l in self.layers]
            out[k] = torch.stack([x.nanmean(-1) for x in per], -1)
            if k in self.per_head:
                out[f"{k}_head"] = torch.stack(per, 2)
        for k in CALL_KEYS:
            if all(k in self.stats[l] for l in self.layers):
                out[k] = torch.stack([self.stats[l][k] for l in self.layers], -1)
        return out

    def meta(self) -> dict:
        """Static facts for the run record: tapped layers, head counts, lag, law."""
        lin = next(iter(self.lins.values()), None)
        return {"layers": list(self.layers), "n_heads": None if lin is None else lin.num_v_heads,
                "lag": self.lag_req if self.lag_req is not None else (None if lin is None else lin.conv_kernel_size),
                "mismatch": self.mismatch, "law": None if self.law is None else
                {"mode": self.law.mode, "layers": list(self.law.layers), "signal": self.law.signal,
                 "invert": self.law.invert, "gain": (self.law.gain.tolist() if isinstance(self.law.gain, torch.Tensor)
                                                     else self.law.gain), **self.law._pipe_kw}}


def _pool(x: torch.Tensor, heads: str) -> torch.Tensor:
    """(B, T, H) → (B, T): NaN-aware head mean or max."""
    if heads == "max":
        m = torch.nan_to_num(x, nan=float("-inf")).amax(-1)
        return torch.where(torch.isfinite(m), m, torch.full_like(m, float("nan")))
    return x.nanmean(-1)


# --------------------------------------------------------------------------------------------- sources
class GDNSource(Source):
    """r_t from a GDN signal: head-pooled, averaged over the GDN layers below the injection site.

    signal     a ``GDNTaps`` per-head key; raw use needs one in [0, 1] (``UNIT_SIGNALS``; cosines are
               clipped at 0), any key with ``transform="zscore"``
    layers     GDN blocks to pool (default: every tapped block below ``inject_before``); each must be below it
    heads      'mean' or 'max'
    transform  'raw', or 'zscore': σ((x_t − μ_{t−1})/σ_{t−1}) with prefix statistics over real tokens
               (0.5 until two tokens are counted)
    invert     s ← 1 − s
    Extra traced keys: ``x`` (the pooled signal before the map, NaN where undefined), ``defined``.
    """

    def __init__(self, signal: str = "fam", layers: Sequence[int] | None = None, heads: str = "mean",
                 transform: str = "raw", invert: bool = False, tap: str = "gdn", name: str | None = None) -> None:
        if signal not in HEAD_KEYS:
            raise ValueError(f"unknown GDN signal {signal!r}; known: {HEAD_KEYS}")
        if transform not in ("raw", "zscore"):
            raise ValueError("transform must be 'raw' or 'zscore'")
        if transform == "raw" and signal not in UNIT_SIGNALS:
            raise ValueError(f"signal {signal!r} is not in [0, 1]; use transform='zscore'")
        if heads not in ("mean", "max"):
            raise ValueError("heads must be 'mean' or 'max'")
        self.signal, self.layers, self.heads, self.transform, self.invert = signal, layers, heads, transform, invert
        self.tap = tap
        self.needs = (tap,)
        self.name = name or f"gdn_{signal}"
        self.reset()

    def reset(self) -> None:
        self.stat: torch.Tensor | None = None             # (B, 3): count, sum, sum of squares (zscore)

    def state_numel(self) -> int:
        return 0 if self.stat is None else self.stat.numel()

    def _layers(self, tp: GDNTaps) -> list[int]:
        if not tp.layers:
            raise ValueError("the GDN tap has no layers (a model without linear-attention layers?)")
        M = tp.inject_before()
        if self.layers is None:
            ls = [l for l in tp.layers if l < M]
        else:
            ls = [int(l) for l in self.layers]
            bad = [l for l in ls if l not in tp.layers or l >= M]
            if bad:
                raise ValueError(f"source layers {bad} are not tapped GDN layers below the injection site {M}")
        if not ls:
            raise ValueError(f"no tapped GDN layer below the injection site {M}")
        return ls

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        tp = ctx.taps[self.tap]
        off = ctx.col_offset
        xs = [_pool(tp.head_stat(l, self.signal)[:, off:].float(), self.heads) for l in self._layers(tp)]
        x = torch.stack(xs, -1).nanmean(-1)
        if tuple(x.shape) != tuple(ctx.mask.shape):
            raise ValueError(f"GDN signal shape {tuple(x.shape)} does not match the call {tuple(ctx.mask.shape)}")
        ok = torch.isfinite(x) & (ctx.mask > 0)
        x0 = torch.where(ok, x, torch.zeros_like(x))
        if self.transform == "raw":
            s = x0.clamp(0.0, 1.0)
        else:
            m = ok.float()
            inc = torch.stack([m, x0 * m, x0 * x0 * m], -1)
            run = leaky_scan(torch.ones_like(inc), inc, self.stat)
            self.stat = run[:, -1].detach()
            prev = run - inc                                        # statistics over tokens before t
            n, s1, s2 = prev[..., 0], prev[..., 1], prev[..., 2]
            mu = s1 / n.clamp_min(1.0)
            sd = (s2 / n.clamp_min(1.0) - mu * mu).clamp_min(0.0).add(1e-6).sqrt()
            s = torch.where(n >= 2, torch.sigmoid((x0 - mu) / sd), torch.full_like(x0, 0.5))
        if self.invert:
            s = 1.0 - s
        s = s * ok.float()
        return {"s": s, "x": x, "defined": ok.float()}


GDN_SOURCES = {
    "gdn_write": partial(GDNSource, "beta", name="gdn_write"),
    "gdn_decay": partial(GDNSource, "ret", name="gdn_decay"),
    "gdn_mismatch": partial(GDNSource, "fam", invert=True, name="gdn_mismatch"),
    "gdn_mismatch_lag": partial(GDNSource, "fam_lag", invert=True, name="gdn_mismatch_lag"),
    "gdn_familiar": partial(GDNSource, "fam", name="gdn_familiar"),
    "gdn_recall": partial(GDNSource, "str", name="gdn_recall"),
}
SOURCES.update(GDN_SOURCES)


def hybrid_taps(model: nn.Module, **kw) -> list[Tap]:
    """[GDNTaps(**kw)] on a hybrid model, [] otherwise: pass to ``CA2Attachment(taps=...)`` in every run, so
    every GDN signal is traced whenever the model has GDN layers."""
    return [GDNTaps(**kw)] if is_hybrid(model) else []


def add_gdn_sources(ctrl, tap: str = "gdn", names: Sequence[str] = tuple(GDN_SOURCES)) -> list[Source]:
    """Register the GDN sources on ``ctrl`` as non-driving (traced) sources; names already present are kept."""
    added = []
    for n in names:
        if n not in ctrl.sources:
            added.append(ctrl.add_source(GDN_SOURCES[n](tap=tap)))
    return added
