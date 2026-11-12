"""AttnTap: attention statistics at one full-attention block, recomputed from hooked q and k (sdpa stays on).

Why recompute. The decoder block discards the attention weights, sdpa never returns them, and switching the
model to eager would change that layer's numerics (gain 0 would no longer be byte-identical to the sdpa
baseline). So the tap reads the attention module's own inputs to its kernel and evaluates the probabilities a
second time, in float32, off the model's path:

  q, k     forward hooks on ``self_attn.q_norm`` / ``self_attn.k_norm``: (B, T, H, d) and (B, T, H_kv, d), after
           the per-head RMSNorm, before RoPE. On Qwen3.5 ``q_norm`` sees the query half of the doubled
           ``q_proj`` (the output gate is split off before it), so the head layout is the same as Qwen3.
  RoPE     a forward hook on ``self_attn`` (``with_kwargs``) reads ``position_embeddings`` (cos, sin) and
           applies the attention module's *own* ``apply_rotary_pos_emb`` (looked up in its modeling file) in
           the model dtype, exactly as the model does: Qwen3.5's partial rotary (the first ``cos.shape[-1]``
           dims) and its mrope recomposition are therefore whatever the model computed. Only then is the
           result cast to float32.
  keys     with a KV cache (``past_key_values`` in the kwargs) the keys are the cache's post-RoPE keys of this
           layer, already updated for this call (``cache.layers[i].keys``): all T_k columns; without a cache,
           RoPE(k) of this call.
  mask     visibility V (B, T_q, T_k) from the ``attention_mask`` kwarg as the kernel received it (after any
           pre-hook, so ``pretrained_loop.HookedLM``'s window mask is seen): None → causal with offset
           T_k − T_q; bool → as given (True = attend); float (additive, eager or HookedLM) → mask > min/2.
           V is ANDed with causality and with key reality.
  reality  key_real (B, T_k) from the 2D ``attention_mask`` given to the model (a forward pre-hook on the
           model; ``run.generate`` passes the full-length mask at every step); all ones when the model gets
           none (no padding: HookedLM, unpadded batches). A query is real iff its own column is real; every
           statistic is 0 (masses) or NaN (ratios) at non-real queries.

Per query t (key column j, lag ℓ = t − j; columns in cache coordinates, query t at T_k − T_q + i):

  S_{h,t,j} = scaling · q_{h,t}·k_{g(h),j}  (GQA: query head h reads key head ⌊h / (H / H_kv)⌋, as repeat_kv)
  P_{h,t,·} = softmax_j over V of S        p̄_{t,j} = (1/H) Σ_h P_{h,t,j}  (head-pooled, "active mass")
  eligible  E_{t,j} = V ∧ real_j ∧ ¬sink_j ∧ j ≠ t     (sink: the row's first ``sink`` real tokens)
  elig      Σ_j p̄ E          sink   Σ_{sink} p̄          self   p̄_{t,t} (t not a sink)   n_elig Σ E,  n_vis Σ V
            (elig + sink + self = 1 at every real query: the three partition the visible keys)
  band      (B, T, W_r): p̄_{t,t−ℓ} E_{t,t−ℓ} for ℓ = 1..W_r (``band_ok`` = E at that lag)
  ent       (1/H) Σ_h H(P_h) (per-head entropy, nats);  ent_pool H(p̄);  ent_elig H(p̄E / elig)
  lagbin    (B, T, n_bins): eligible mass by log₂ distance, bin b = lags [2^b, 2^{b+1} − 1] (last bin open):
            the time-relevance profile of the query (how far back it reads)
  mean_lag  Σ p̄E ℓ / elig;  mean_loglag Σ p̄E log ℓ / elig
  per head  head_elig, head_sink, head_self, head_band (Σ over the band), head_ent: (B, T, H)
  scores    the pre-softmax variant, from S over eligible keys, per head then averaged over heads:
            score_mean μ_h, score_sd σ_h, score_band_z (B, T, W_r) z = (S_{t,t−ℓ} − μ_h)/σ_h,
            score_sink_z (z of the sink key's score), score_top_z (max eligible z). z is shift invariant,
            so it does not depend on how much mass the sink takes.
Kept on the device for the sources but not traced (too large): ``qband`` (B, T, H, W_r), the per-head
softmax restricted to the eligible keys, q_{h,t,ℓ} = exp(S_{h,t,t−ℓ} − logsumexp_{j∈E} S_{h,t,j}) — the
head_pool="mean" alternative (eligible keys excluded *before* the softmax, so dormant heads that sit on the
sink count as much as active ones).

Massive keys. The tap cannot know this call's massive flags (the bank computes them at the tap layer L, after
this attention layer has run), so the eligible set above does not exclude them; the sources do, at step time,
from ``ctx.bank.massive_prev`` ‖ ``ctx.bank.massive`` (kept over the relevance window only, the bank's fixed
state), by subtracting the flagged band keys from the band and from ``elig`` / ``n_elig`` (exact for keys
within W_r; flagged keys further back are not excluded).

Sources (registered in ``sources.SOURCES`` on import, so ``make_source("attn_agree")`` works; each reads the
tap through ``ctx.taps[tap]``, the tap must be attached at a full-attention layer < M):
  ``attn_window``  (locality) with p̃_ℓ = band_ℓ k_ℓ / elig', k_ℓ = 1 for non-excluded band keys, elig' the
                   eligible mass after exclusions, frac = Σ_ℓ p̃_ℓ, u = n_band / n_elig' (the share a uniform
                   distribution would give the band), s = clip((frac − u)/(1 − u), 0, 1) (the lift removes
                   the position confound at t < W_r). ``exclude_prev`` drops lag 1 (previous-token heads).
  ``attn_agree``   (attention-weighted gist agreement) a_{t,j} = cos(h_j − b_{t−1}, m_{t−1} − b_{t−1}) with h_j
                   the tap stream (block L output) at key j and m, b the fast / slow band averages carried to
                   t − 1; s = Σ_ℓ p̃_ℓ [a_{t,t−ℓ}]_+ / Σ_ℓ p̃_ℓ. The (T, W_r, D) tensor is never built: one lag at
                   a time, O(B·T·D) each. State: a ring of the last W_r tap vectors (B, W_r, D).
  ``pool``         "active" (default) weights by p̄ (head-pooled before renormalising); "mean" by the mean of
                   the per-head restricted softmaxes (``qband`` renormalised after the exclusions). Both are
                   traced whatever drives s.
  guard            where the eligible mass elig' < ε_a (default 0.02) or the score is otherwise undefined, s_t
                   = s_{t−1} (held; 0 before the first defined step). The hold is a prefix quantity.

bf16. q and k are the model's bf16 values (RoPE applied in bf16 as the model does) upcast to float32; P is the
model's logits evaluated in float32, which is close to but not bitwise what the sdpa kernel computes inside.
That is a readout, not the model's path: gain 0 with the tap attached is byte-identical (every hook returns
None). Cost: row chunks sized so that one (B, H, chunk, T_k) float32 tensor has at most ``chunk_elems``
elements; the full (B, H, T_q, T_k) tensor is never held.

Not supported: ``HookedLM`` segment mode (it folds the batch), static or sliding-window KV caches (key
columns would not be absolute), custom 4D masks passed to the model (key reality would be unknown).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402
from ca2.controller import Tap  # noqa: E402
from ca2.sources import SOURCES, Source, StepContext  # noqa: E402

TAP_KEYS = ("elig", "sink", "self", "n_elig", "n_vis", "qreal", "ent", "ent_pool", "ent_elig", "mean_lag",
            "mean_loglag", "band", "band_ok", "band_mass", "far", "lagbin", "head_elig", "head_sink", "head_self",
            "head_band", "head_ent", "score_mean", "score_sd", "score_band_z", "score_sink_z", "score_top_z")


# ------------------------------------------------------------------------------------------- helpers
def visibility(mask: torch.Tensor | None, B: int, Tq: int, Tk: int, device) -> torch.Tensor:
    """Bool (B, Tq, Tk): which keys each query may attend, from the mask the attention kernel received.

    None → causal with offset Tk − Tq; bool → as given; float (additive) → mask > finfo.min / 2. Always ANDed
    with causality. A (b, h, Tq, Tk) mask with several heads uses head 0 (HF masks are head-independent).
    """
    q = torch.arange(Tq, device=device)[:, None] + (Tk - Tq)
    causal = torch.arange(Tk, device=device)[None, :] <= q
    if mask is None:
        return causal[None].expand(B, Tq, Tk)
    m = mask
    if m.dim() == 2:
        m = m[None, None]
    elif m.dim() == 3:
        m = m[:, None]
    if m.shape[-1] != Tk or m.shape[-2] < Tq:
        raise ValueError(f"attention mask {tuple(mask.shape)} does not match T_q={Tq}, T_k={Tk}")
    m = m[:, 0, -Tq:, :]
    if m.dtype != torch.bool:
        m = m > torch.finfo(m.dtype).min / 2
    return (m.to(device) & causal[None]).expand(B, Tq, Tk)


def _cache_keys(cache, idx: int) -> torch.Tensor | None:
    """Post-RoPE keys (B, H_kv, T_k, d) of layer ``idx`` in a transformers cache (5.x ``layers``, 4.x lists)."""
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return getattr(layers[idx], "keys", None)
    kc = getattr(cache, "key_cache", None)
    return None if kc is None else kc[idx]


def _rope_fn(attn: torch.nn.Module):
    fn = getattr(sys.modules.get(type(attn).__module__), "apply_rotary_pos_emb", None)
    if fn is None:
        raise ValueError(f"no apply_rotary_pos_emb next to {type(attn).__name__}")
    return fn


def hold(s: torch.Tensor, defined: torch.Tensor, last: torch.Tensor | None) -> torch.Tensor:
    """s where defined, else the last defined value along dim 1 (``last`` (B,) carried in; 0 when None)."""
    B, T = s.shape
    i = torch.arange(T, device=s.device).expand(B, T)
    j = torch.where(defined.bool(), i, torch.full_like(i, -1)).cummax(1).values
    prev = torch.zeros(B, device=s.device) if last is None else last.to(s.device)
    return torch.where(j >= 0, s.gather(1, j.clamp_min(0)), prev[:, None].expand(B, T))


# ---------------------------------------------------------------------------------------- statistics
def attn_stats(q: torch.Tensor, k: torch.Tensor, vis: torch.Tensor, key_real: torch.Tensor,
               sink_keys: torch.Tensor, scaling: float, window: int = 32, lag_bins: int = 16,
               chunk_elems: int = 1 << 24) -> dict[str, torch.Tensor]:
    """The statistics of the module docstring for one call.

    q (B, H, T_q, d) and k (B, H_kv, T_k, d) post-RoPE (any float dtype; computed in float32); vis (B, T_q, T_k)
    bool; key_real, sink_keys (B, T_k) bool. Returns (B, T_q, ...) float32 tensors (``band_ok`` bool) plus
    ``qband`` (B, T_q, H, W_r).
    """
    q, k = q.float(), k.float()
    B, H, Tq, d = q.shape
    Hk, Tk = k.shape[1], k.shape[2]
    if H % Hk:
        raise ValueError(f"{H} query heads are not a multiple of {Hk} key heads")
    G, W, dev = H // Hk, int(window), q.device
    col = torch.arange(Tk, device=dev)
    qcol_all = col[Tk - Tq:]
    real_q = key_real[:, Tk - Tq:]
    base = key_real & ~sink_keys                                   # (B, Tk)
    lags = torch.arange(1, W + 1, device=dev)
    edges = [2 ** (b + 1) for b in range(lag_bins - 1)]
    c = max(1, min(Tq, chunk_elems // max(1, B * H * Tk)))
    nan = float("nan")
    parts: dict[str, list] = {}

    for r0 in range(0, Tq, c):
        r1 = min(Tq, r0 + c)
        n = r1 - r0
        qcol = qcol_all[r0:r1]
        S = torch.matmul(q[:, :, r0:r1].reshape(B, Hk, G, n, d), k[:, :, None].transpose(-1, -2)) * scaling
        S = S.reshape(B, H, n, Tk)
        v = vis[:, r0:r1] & key_real[:, None, :]                   # (B, n, Tk)
        rq = real_q[:, r0:r1] & v.any(-1)                          # (B, n) real queries
        P = torch.softmax(S.masked_fill(~v[:, None], float("-inf")), -1)
        P = torch.where(rq[:, None, :, None], P, torch.zeros((), device=dev))
        selfk = col[None, :] == qcol[:, None]                      # (n, Tk)
        E = v & base[:, None, :] & ~selfk[None]                    # (B, n, Tk) eligible keys
        sk = v & sink_keys[:, None, :]
        sv = v & selfk[None] & ~sink_keys[:, None, :]              # the sink's own query counts as sink
        pbar = P.mean(1)
        zero = torch.zeros((), device=dev)
        out: dict[str, torch.Tensor] = {}
        elig = torch.where(E, pbar, zero).sum(-1)
        out["elig"] = elig
        out["sink"] = torch.where(sk, pbar, zero).sum(-1)
        out["self"] = torch.where(sv, pbar, zero).sum(-1)
        out["n_elig"] = E.sum(-1).float()
        out["n_vis"] = torch.where(rq, v.sum(-1).float(), zero)
        out["qreal"] = rq.float()
        out["ent_pool"] = -torch.xlogy(pbar, pbar).sum(-1)
        hent = -torch.xlogy(P, P).sum(-1)                          # (B, H, n)
        out["ent"] = hent.mean(1)
        ok_e = rq & (elig > 0)
        pt = torch.where(E, pbar, zero) / elig.clamp_min(1e-30)[..., None]
        out["ent_elig"] = torch.where(ok_e, -torch.xlogy(pt, pt).sum(-1), torch.full_like(elig, nan))
        lag = (qcol[:, None] - col[None, :])                       # (n, Tk)
        lagf = lag.clamp_min(1).float()
        pe = torch.where(E, pbar, zero)
        out["mean_lag"] = torch.where(ok_e, (pe * lagf).sum(-1) / elig.clamp_min(1e-30), torch.full_like(elig, nan))
        out["mean_loglag"] = torch.where(ok_e, (pe * lagf.log()).sum(-1) / elig.clamp_min(1e-30),
                                         torch.full_like(elig, nan))
        cum = [torch.where(lag[None] < e, pe, zero).sum(-1) for e in edges] + [elig]
        cum = torch.stack(cum, -1)
        out["lagbin"] = torch.cat([cum[..., :1], cum[..., 1:] - cum[..., :-1]], -1)
        # band (lags 1..W)
        kidx = qcol[:, None] - lags[None, :]                        # (n, W)
        inr = kidx >= 0
        kc = kidx.clamp_min(0)
        Eg = E.gather(-1, kc[None].expand(B, n, W)) & inr[None]    # (B, n, W)
        band = torch.where(Eg, pbar.gather(-1, kc[None].expand(B, n, W)), zero)
        out["band"], out["band_ok"] = band, Eg
        out["band_mass"] = band.sum(-1)
        out["far"] = elig - out["band_mass"]
        PE = torch.where(E[:, None], P, zero)
        out["head_elig"] = PE.sum(-1).transpose(1, 2)
        out["head_sink"] = torch.where(sk[:, None], P, zero).sum(-1).transpose(1, 2)
        out["head_self"] = torch.where(sv[:, None], P, zero).sum(-1).transpose(1, 2)
        kcH = kc[None, None].expand(B, H, n, W)
        Pg = torch.where(Eg[:, None], P.gather(-1, kcH), zero)      # (B, H, n, W)
        out["head_band"] = Pg.sum(-1).transpose(1, 2)
        out["head_ent"] = hent.transpose(1, 2)
        # per-head softmax restricted to the eligible keys, at the band lags
        SE = S.masked_fill(~E[:, None], float("-inf"))
        lse = torch.logsumexp(SE, -1)                               # (B, H, n); -inf where no eligible key
        Sg = S.gather(-1, kcH)
        lfin = torch.isfinite(lse)
        qb = torch.where(Eg[:, None] & lfin[..., None], torch.exp(Sg - lse.masked_fill(~lfin, 0.0)[..., None]), zero)
        out["qband"] = qb.permute(0, 2, 1, 3)                       # (B, n, H, W)
        # pre-softmax scores over eligible keys
        ne = E.sum(-1).float()[:, None]                             # (B, 1, n)
        mu = torch.where(E[:, None], S, zero).sum(-1) / ne.clamp_min(1)
        var = torch.where(E[:, None], (S - mu[..., None]) ** 2, zero).sum(-1) / ne.clamp_min(1)
        sd = var.sqrt()
        zok = (ne >= 2) & (sd > 0) & rq[:, None]                    # (B, H, n)
        sdc = sd.clamp_min(1e-30)
        nanH = torch.full_like(mu, nan)
        out["score_mean"] = torch.where(zok, mu, nanH).mean(1)
        out["score_sd"] = torch.where(zok, sd, nanH).mean(1)
        zb = torch.where(zok[..., None] & Eg[:, None], (Sg - mu[..., None]) / sdc[..., None], torch.full_like(Sg, nan))
        out["score_band_z"] = zb.mean(1)
        nsk = sk.sum(-1).float()[:, None]
        ssk = torch.where(sk[:, None], S, zero).sum(-1) / nsk.clamp_min(1)
        out["score_sink_z"] = torch.where(zok & (nsk > 0), (ssk - mu) / sdc, nanH).mean(1)
        top = SE.amax(-1)
        out["score_top_z"] = torch.where(zok, (top - mu) / sdc, nanH).mean(1)
        for key, val in out.items():
            parts.setdefault(key, []).append(val)
    return {key: torch.cat(v, 1) for key, v in parts.items()}


# ---------------------------------------------------------------------------------------------- the tap
class AttnTap(Tap):
    """Attention statistics at one full-attention block (module docstring); signals (B, T) / (B, T, K).

    ``layer`` defaults to the attachment's ``relevance_layer``; ``window`` W_r is the band width; ``sink`` the
    number of leading real tokens treated as the sink (default: the attachment's ``sink`` if it has one, as
    ``SteerSide``, else the controller's ``sink_skip``); ``lag_bins`` the log₂ distance bins; ``chunk_elems``
    bounds one (B, H, rows, T_k) float32 tensor. State: ``sink_cols`` (B, sink), the sink's cache columns,
    carried across KV-cached calls.
    """

    def __init__(self, layer: int | None = None, window: int = 32, sink: int | None = None, lag_bins: int = 16,
                 chunk_elems: int = 1 << 24, name: str = "attn") -> None:
        if window < 1 or lag_bins < 1:
            raise ValueError("window and lag_bins must be >= 1")
        self.layer, self.window, self.sink, self.lag_bins = layer, int(window), sink, int(lag_bins)
        self.chunk_elems, self.name = int(chunk_elems), name
        self.handles: list = []
        self.attn = None
        self._mask2d: torch.Tensor | None = None
        self.reset()
        self.begin_call(None)

    # ------------------------------------------------------------------ Tap interface
    def attach(self, att) -> None:
        L = att.relevance_layer if self.layer is None else int(self.layer) % len(att.blocks)
        if att.kinds[L] != "attention":
            raise ValueError(f"block {L} is a {att.kinds[L]} block; AttnTap needs a full-attention block")
        attn = pl._child(att.blocks[L], pl.ATTN_MIXERS)
        if attn is None or not hasattr(attn, "q_norm") or not hasattr(attn, "k_norm"):
            raise ValueError(f"block {L}: no attention module with q_norm / k_norm (Qwen3 / Qwen3.5 layout)")
        if self.sink is None:
            self.sink = int(getattr(att, "sink", getattr(att.ctrl, "sink_skip", 1)))
        self.layer, self.attn = L, attn
        self.layer_idx = int(getattr(attn, "layer_idx", L))
        self.scaling = float(getattr(attn, "scaling", attn.head_dim ** -0.5))
        self.rope = _rope_fn(attn)
        self.handles += [attn.q_norm.register_forward_hook(self._q_hook),
                         attn.k_norm.register_forward_hook(self._k_hook),
                         attn.register_forward_hook(self._attn_hook, with_kwargs=True),
                         att.model.register_forward_pre_hook(self._model_pre, with_kwargs=True)]

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self) -> None:
        self.sink_cols: torch.Tensor | None = None       # (B, sink) cache columns of the sink, −1 if none

    def state_numel(self) -> int:
        return 0 if self.sink_cols is None else self.sink_cols.numel()

    def begin_call(self, ctrl) -> None:
        self._q = self._k = None
        self.cur: dict[str, torch.Tensor] = {}

    def signals(self) -> dict[str, torch.Tensor]:
        return {k: (v.float() if k == "band_ok" else v) for k, v in self.cur.items() if k in TAP_KEYS}

    # ------------------------------------------------------------------------- hooks
    def _model_pre(self, mod, args, kwargs):
        m = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        self._mask2d = m.detach() if isinstance(m, torch.Tensor) and m.dim() == 2 else None

    def _q_hook(self, mod, args, out):
        self._q = out.detach()

    def _k_hook(self, mod, args, out):
        self._k = out.detach()

    @torch.no_grad()
    def _attn_hook(self, mod, args, kwargs, out):
        if self._q is None or self._k is None:
            return None
        cos, sin = kwargs["position_embeddings"]
        q = self._q.transpose(1, 2)                                  # (B, H, Tq, d), model dtype
        k = self._k.transpose(1, 2)
        qr, kr = self.rope(q, k, cos, sin)
        B, H, Tq, _ = qr.shape
        cache = kwargs.get("past_key_values")
        K = _cache_keys(cache, self.layer_idx) if cache is not None else None
        if K is None:
            K = kr
        Tk = K.shape[2]
        dev = qr.device
        m2 = self._mask2d
        if m2 is not None and m2.shape[1] >= Tk:
            key_real = m2[:, -Tk:].to(dev).bool()
        elif m2 is not None and m2.shape[1] == Tq:
            key_real = torch.cat([torch.ones(B, Tk - Tq, dtype=torch.bool, device=dev), m2.to(dev).bool()], 1)
        else:
            key_real = torch.ones(B, Tk, dtype=torch.bool, device=dev)
        am = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        vis = visibility(am if isinstance(am, torch.Tensor) else None, B, Tq, Tk, dev)
        sink_keys = self._sink_keys(key_real, cache is not None and Tk > Tq)
        self.cur = attn_stats(qr, K, vis, key_real, sink_keys, self.scaling, self.window, self.lag_bins,
                              self.chunk_elems)
        self._q = self._k = None
        return None

    def _sink_keys(self, key_real: torch.Tensor, cached: bool) -> torch.Tensor:
        B, Tk = key_real.shape
        n = int(self.sink)
        col = torch.arange(Tk, device=key_real.device)
        if not cached or self.sink_cols is None:
            ridx = key_real.long().cumsum(1) - 1
            first = key_real & (ridx < n)
            cols = torch.full((B, max(n, 0)), -1, dtype=torch.long, device=key_real.device)
            for i in range(n):
                hit = first & (ridx == i)
                cols[:, i] = torch.where(hit.any(1), hit.float().argmax(1), torch.full((B,), -1, device=hit.device))
            self.sink_cols = cols
        if n == 0:
            return torch.zeros(B, Tk, dtype=torch.bool, device=key_real.device)
        return (col[None, :, None] == self.sink_cols.to(key_real.device)[:, None, :]).any(-1)


# -------------------------------------------------------------------------------------------- sources
class _AttnSource(Source):
    """Shared band bookkeeping of the attention sources (exclusions, both pools, the hold)."""

    def __init__(self, tap: str, window: int | None, exclude_prev: bool, exclude_massive: bool, pool: str,
                 eps: float, name: str) -> None:
        if pool not in ("active", "mean"):
            raise ValueError("pool must be 'active' or 'mean'")
        self.tap, self.window, self.exclude_prev, self.exclude_massive = tap, window, bool(exclude_prev), \
            bool(exclude_massive)
        self.pool, self.eps, self.name, self.needs = pool, float(eps), name, (tap,)
        self.reset()

    def reset(self) -> None:
        self.last: torch.Tensor | None = None            # (B,) held score

    def state_numel(self) -> int:
        return 0 if self.last is None else self.last.numel()

    def _band(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        tap = ctx.taps[self.tap]
        cur = tap.cur
        if not cur:
            raise RuntimeError(f"tap {self.tap!r} did not run in this call (is its layer below the injection site?)")
        o = ctx.col_offset
        W = tap.window if self.window is None else int(self.window)
        if W > tap.window:
            raise ValueError(f"source window {W} exceeds the tap's {tap.window}")
        B, T = ctx.mask.shape
        band, ok = cur["band"][:, o:, :W], cur["band_ok"][:, o:, :W]
        qband = cur["qband"][:, o:, :, :W]
        if band.shape[:2] != (B, T):
            raise ValueError(f"tap columns {tuple(band.shape[:2])} do not match the step's {(B, T)}")
        excl = torch.zeros_like(ok)
        n_mass = torch.zeros(B, T, device=band.device)
        if self.exclude_massive:
            Wb = ctx.bank.massive_prev.shape[1]
            if W > Wb:
                raise ValueError(f"window {W} exceeds the bank's massive window {Wb}")
            flags = torch.cat([ctx.bank.massive_prev, ctx.bank.massive], 1).to(band.device)
            idx = Wb + torch.arange(T, device=band.device)[:, None] - torch.arange(1, W + 1, device=band.device)
            mlag = flags[:, idx]                                     # (B, T, W)
            excl = excl | mlag
            n_mass = (mlag & ok).float().sum(-1)
        if self.exclude_prev:
            excl[..., 0] = True
        excl = excl & ok
        keep = ok & ~excl
        zero = torch.zeros((), device=band.device)
        elig = cur["elig"][:, o:] - torch.where(excl, band, zero).sum(-1)
        n_elig = cur["n_elig"][:, o:] - excl.float().sum(-1)
        n_band = keep.float().sum(-1)
        w_act = torch.where(keep, band, zero) / elig.clamp_min(1e-30)[..., None]
        den_h = 1.0 - torch.where(excl[:, :, None], qband, zero).sum(-1)              # (B, T, H)
        hok = den_h > 1e-6
        w_h = torch.where(keep[:, :, None] & hok[..., None], qband / den_h.clamp_min(1e-6)[..., None], zero)
        nh = hok.float().sum(-1)
        w_mean = w_h.sum(2) / nh.clamp_min(1)[..., None]
        real = cur["qreal"][:, o:] > 0
        base_ok = real & (elig >= self.eps)
        return {"keep": keep, "w_active": w_act, "w_mean": w_mean, "elig": elig, "n_elig": n_elig,
                "n_band": n_band, "n_massive": n_mass, "base_ok": base_ok, "mean_ok": base_ok & (nh > 0), "W": W}

    def _held(self, s: torch.Tensor, defined: torch.Tensor) -> torch.Tensor:
        out = hold(s, defined, self.last)
        self.last = out[:, -1].detach()
        return out


class AttnWindowSource(_AttnSource):
    """s = clip((frac − u)/(1 − u), 0, 1): attention locality over the last W_r tokens, uniform-lifted."""

    def __init__(self, tap: str = "attn", window: int | None = None, exclude_prev: bool = False,
                 exclude_massive: bool = True, pool: str = "active", eps: float = 0.02,
                 name: str = "attn_window") -> None:
        super().__init__(tap, window, exclude_prev, exclude_massive, pool, eps, name)

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        b = self._band(ctx)
        u = b["n_band"] / b["n_elig"].clamp_min(1)
        wok = (b["n_elig"] > b["n_band"]) & (b["n_band"] > 0)
        out = {}
        for pool in ("active", "mean"):
            frac = b[f"w_{pool}"].sum(-1)
            lift = ((frac - u) / (1.0 - u).clamp_min(1e-30)).clamp(0.0, 1.0)
            ok = b["base_ok"] & wok & (b["mean_ok"] if pool == "mean" else True)
            out[f"frac_{pool}"] = torch.where(ok, frac, torch.zeros_like(frac))
            out[f"lift_{pool}"] = torch.where(ok, lift, torch.zeros_like(lift))
            out[f"defined_{pool}"] = ok.float()
        sel = self.pool
        out["s"] = self._held(out[f"lift_{sel}"], out[f"defined_{sel}"] > 0)
        out.update(u=u, elig=b["elig"], n_elig=b["n_elig"], n_band=b["n_band"], n_massive=b["n_massive"])
        return out


class AttnAgreeSource(_AttnSource):
    """s = Σ_ℓ p̃_ℓ [cos(h_{t−ℓ} − b_{t−1}, m_{t−1} − b_{t−1})]_+ / Σ_ℓ p̃_ℓ over the band (attention-weighted)."""

    def __init__(self, tap: str = "attn", window: int | None = None, exclude_prev: bool = False,
                 exclude_massive: bool = True, pool: str = "active", eps: float = 0.02,
                 fast: int = 0, slow: int = -1, name: str = "attn_agree") -> None:
        self.fast, self.slow = fast, slow
        super().__init__(tap, window, exclude_prev, exclude_massive, pool, eps, name)

    def reset(self) -> None:
        super().reset()
        self.ring: torch.Tensor | None = None            # (B, W, D) last tap vectors
        self.ring_ok: torch.Tensor | None = None         # (B, W) real columns

    def state_numel(self) -> int:
        return super().state_numel() + sum(0 if t is None else t.numel() for t in (self.ring, self.ring_ok))

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        b = self._band(ctx)
        W = b["W"]
        h = ctx.h
        B, T, D = h.shape
        dev = h.device
        ring = self.ring if self.ring is not None and self.ring.shape[1] == W else h.new_zeros(B, W, D)
        rok = (self.ring_ok if self.ring_ok is not None and self.ring_ok.shape[1] == W
               else torch.zeros(B, W, dtype=torch.bool, device=dev))
        seq = torch.cat([ring, h], 1)
        sok = torch.cat([rok, ctx.mask > 0], 1)
        bk = ctx.bank
        pm, pb = bk.prev(self.fast), bk.prev(self.slow)
        gd = pm - pb
        gok = (bk.prev_den(self.fast) > 0) & (bk.prev_den(self.slow) > 0) & (gd.norm(dim=-1) > 0)
        a = torch.zeros(B, T, W, device=dev)
        kok = torch.zeros(B, T, W, dtype=torch.bool, device=dev)
        for l in range(1, W + 1):
            hj = seq[:, W - l: W - l + T]
            a[..., l - 1] = F.cosine_similarity(hj - pb, gd, dim=-1, eps=1e-8)
            kok[..., l - 1] = sok[:, W - l: W - l + T]
        keep = b["keep"] & kok
        ap = a.clamp_min(0.0)
        zero = torch.zeros((), device=dev)
        out = {}
        for pool in ("active", "mean"):
            w = torch.where(keep, b[f"w_{pool}"], zero)
            ws = w.sum(-1)
            ok = b["base_ok"] & gok & (ws > 0) & (b["mean_ok"] if pool == "mean" else True)
            agr = (w * ap).sum(-1) / ws.clamp_min(1e-30)
            out[f"agree_{pool}"] = torch.where(ok, agr.clamp(0.0, 1.0), torch.zeros_like(agr))
            out[f"a_signed_{pool}"] = torch.where(ok, (w * torch.where(keep, a, zero)).sum(-1) / ws.clamp_min(1e-30),
                                                  torch.zeros_like(agr))
            out[f"band_w_{pool}"] = torch.where(ok, ws, torch.zeros_like(ws))
            out[f"defined_{pool}"] = ok.float()
        nk = keep.float().sum(-1)
        uok = gok & (nk > 0) & (ctx.mask > 0)
        out["a_uniform"] = torch.where(uok, torch.where(keep, ap, zero).sum(-1) / nk.clamp_min(1), torch.zeros_like(nk))
        out["s"] = self._held(out[f"agree_{self.pool}"], out[f"defined_{self.pool}"] > 0)
        out.update(elig=b["elig"], n_band=b["n_band"], n_massive=b["n_massive"])
        self.ring, self.ring_ok = seq[:, -W:].detach(), sok[:, -W:].detach()
        return out


SOURCES.setdefault("attn_window", AttnWindowSource)
SOURCES.setdefault("attn_agree", AttnAgreeSource)


def add_attention(ctrl, layer: int | None = None, window: int = 32, name: str = "attn",
                  sources: Sequence[str] = ("attn_window", "attn_agree"), **src_kw) -> AttnTap:
    """Build an ``AttnTap`` and register the attention sources on ``ctrl`` (not driving; a source already
    registered under the same name, e.g. the driving one, is kept). Pass the tap to ``CA2Attachment(taps=[tap])``
    or ``SteerSide(taps=[tap])``; every statistic and every source signal is then traced."""
    tap = AttnTap(layer=layer, window=window, name=name)
    for s in sources:
        if s not in ctrl.sources:
            ctrl.add_source(SOURCES[s](tap=name, **src_kw))
    return tap
