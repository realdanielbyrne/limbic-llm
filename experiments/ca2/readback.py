"""Attention read-back on a model trained with a sliding window (round 12, M2 + M3a): ``Readback``.

RecurrentGemma (``google/recurrentgemma-2b(-it)``) interleaves two RG-LRU recurrent blocks with one local-attention
block (layers 2, 5, ..., 23; multi-query: 10 query heads, 1 KV head, head_dim 256, rotary on the first half of each
head). Each attention layer sees the last W = 2,048 positions, its own included (HF mask: kv > q − W). Everything
older reaches the model only through the recurrent state. Read-back offers chosen out-of-window positions back to
every attention layer as extra key/value columns, at a fixed per-query budget of columns, so every arm holds the
same KV memory in attention:

    columns(t) = sink (s) + read-back (≤ R) + recent (D = W − s − R),   recent = positions (t − D, t].

Mechanism (no weights change; the model's own q/k/v projections, rotary and cache run as usual):

  archive   A registered attention function (``readback``; ``Readback.active()`` switches the model's
            ``config._attn_implementation`` to it) appends each call's new rotated keys and values (the last T columns
            the cache hands over) to a per-layer archive, so evicted positions stay available. A forward pre-hook on
            each attention module records the call's ``position_ids`` (rows must share positions: no padding).
  attention For query rows t of a chunk (``chunk`` rows in a multi-token call, one row per decode step), one float32
            softmax over [sink | read-back | recent] columns, then the values; cast back to the query dtype.
            Recent columns keep their true rotary positions (the native computation). A sink column is shown only
            when it has left the recent band (p ≤ t − D).
  time code Read-back entries are re-positioned with the RoPE identity R(a)x · R(b)y = x · R(b − a)y, so only the
            distance δ = t − p' matters:
              true      p' = p, the original position (δ = t − p > D, outside the trained range when > W);
              ordinal   the n entries in original order, contiguous, just before the oldest recent column:
                        δ_i = D + (n − 1 − i), i = 0 the oldest; δ ∈ [D, D + n) ⊂ [0, W) (the P66 winner);
              none      every segment starts at the same slot: δ_j = D + L_max − 1 − j for offset j in its
                        segment; within-segment order kept, between-segment order erased.
            Relative codes are exact per query row: q̃_t (un-rotated, R(−t) q_t) is scored against R(−δ) k̃ (the key
            rotated from p to −δ), which equals R(t) q̃ · R(t − δ) k̃ for every t at once. The sink follows
            StreamingLLM's cache positions under EVERY time code: just beyond the read-back slots,
            δ = D + n_slots + (s − 1 − i) (n_slots = n for ordinal and true, L_max for none, 0 with nothing read
            back), so its distance stays inside [0, W) and oracle:true differs from oracle:ordinal only in the
            read-back entries' code. Only ``all`` (the exactness check) keeps the sink at its true positions.
  causality Selection for the prefill chunk with rows [t0, t1) uses only queries of the previous chunk
            (rows [t0 − chunk, t0)); in a decode step (one row) it uses the last ``sel_rows`` rows, the current one
            included. Only positions p with s ≤ p ≤ t_ref − D are eligible (t_ref = t0, or t in a step), checked
            on every call, so a read-back column is never one the recent band already holds; every row also
            masks read-back entries with p > t − D (only ``all`` needs that mask).
  budget    Every row sees at most s + R + D = W columns (asserted for every selector except ``all``); the counts
            are recorded in ``stats``.

Selectors (``ReadbackConfig.selector``):

  window    s = 0, R = 0, D = W: the native columns through this function (an implementation control).
  stream    StreamingLLM: s sink columns + recent W − s, nothing read back.
  oracle    the runner's segments (``begin(oracle=...)``, the needle facts with context), in position order while
            they fit in R; the same for every layer.
  infllm    InfLLM block retrieval, per layer. Blocks of ``block`` positions from s. When a block is complete its
            ``n_rep`` representative keys are the keys that received the most attention from the block's own
            queries (causal softmax restricted to the block, summed over its queries and heads; InfLLM ranks by
            the raw dot product, which this replaces as the design asks). Block score =
            mean_h Σ_{q ∈ Q} max_j q̃_h · r̃_j / √d over the selection queries Q, un-rotated q̃ and r̃
            (position-free); the top ⌊R / block⌋ eligible blocks are read back.
  random    ⌊R / block⌋ eligible blocks in a fixed random order per row (``begin(rand_seeds=...)``).
  foreign   the oracle segments of the next row of the batch ((b + 1) mod B), read from THAT row's archive:
            another document's columns, same count. Needs B ≥ 2.
  all       every eligible position, true time, per-row eligibility: equals full causal attention (the exactness
            check). No budget.

Public API
  ReadbackConfig(selector, time, sink, budget, block, n_rep, chunk, sel_rows)
  Readback(model, cfg)          attention modules found by their ``rotary_emb`` + ``layer_idx``; W from the config.
      .active()                 context manager: read-back on (config switch + hooks), restored on exit.
      .begin(B, oracle=None, rand_seeds=None)   new sequences: archives cleared, per-row oracle segments
                                ([(start, length), ...] per row) and random seeds.
      .recent / .sink / .budget columns; .columns() = per-query KV columns per layer.
      .last_selected            {layer: (positions (B, M) long, −1 padded; source row (B,))} from the last call.
      .stats                    {"max_cols", "calls", "selected_mean"}.
  readback_attention(...)       the registered attention function (dispatches to the module's controller).
  rope_shift(x, shift, inv_freq) rotate the rotary part of x by position offsets shift (float32).
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

IMPL = "readback"
SELECTORS = ("window", "stream", "oracle", "infllm", "random", "foreign", "all")
TIMES = ("true", "ordinal", "none")
NEG_INF = float("-inf")


# ------------------------------------------------------------------------------------------------ rotary helpers
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


def rope_shift(x: torch.Tensor, shift: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """x (..., M, d) with its rotary part (the first 2·len(inv_freq) dims) rotated by ``shift`` positions (float32).

    ``shift`` broadcasts against x[..., 0]. Rotations compose, so shifting a key rotated at p by a − p puts it at a,
    and shifting by −p un-rotates it. Same formula as the HF rotary (float32 angles), without the cast to x's dtype.
    """
    rd = 2 * inv_freq.numel()
    f = shift.to(torch.float32)[..., None] * inv_freq.to(x.device, torch.float32)
    emb = torch.cat((f, f), -1)
    xr, xp = x[..., :rd].float(), x[..., rd:].float()
    return torch.cat((xr * emb.cos() + rotate_half(xr) * emb.sin(), xp), -1)


def _rep(x: torch.Tensor, H: int) -> torch.Tensor:
    """(B, Hkv, N, d) → broadcastable against H query heads (HF ``repeat_kv``: query head h reads KV head h // g)."""
    g = x.shape[1]
    return x if g in (1, H) else x.repeat_interleave(H // g, 1)


# --------------------------------------------------------------------------------------------------- config
@dataclass
class ReadbackConfig:
    selector: str = "oracle"
    time: str = "ordinal"
    sink: int = 4
    budget: int = 512                                             # R: read-back columns per query
    block: int = 128
    n_rep: int = 4
    chunk: int = 256                                              # query rows per selection step in a prefill call
    sel_rows: int = 32                                            # decode steps: queries scoring the blocks

    def __post_init__(self) -> None:
        if self.selector not in SELECTORS:
            raise ValueError(f"selector {self.selector!r} not in {SELECTORS}")
        if self.time not in TIMES:
            raise ValueError(f"time {self.time!r} not in {TIMES}")
        if self.selector == "window":
            self.sink, self.budget = 0, 0
        if self.selector == "stream":
            self.budget = 0
        if self.selector == "all" and self.time != "true":
            raise ValueError("'all' is the exactness check: true time only")
        if min(self.sink, self.budget, self.chunk - 1, self.sel_rows - 1, self.block - 1, self.n_rep - 1) < 0:
            raise ValueError(f"bad read-back sizes: {self}")
        if self.selector in ("infllm", "random") and self.budget < self.block:
            raise ValueError(f"budget {self.budget} holds no {self.block}-position block")


# ---------------------------------------------------------------------------------------------- layer store
@dataclass
class _Store:
    """One attention layer's archive (rotated keys and values as the module made them) and selection state."""
    k: torch.Tensor | None = None                                 # (B, Hkv, cap, d) base dtype
    v: torch.Tensor | None = None
    n: int = 0                                                    # positions archived (0 … n − 1)
    hist: torch.Tensor | None = None                              # (B, H, ≤ rows, d) float32 un-rotated queries
    pend: torch.Tensor | None = None                              # (B, H, m, d) queries of the incomplete block
    pend_start: int = 0
    reps: list = field(default_factory=list)                      # per complete block: (B, n_rep, d) float32

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        T = k.shape[2]
        if self.k is None or self.n + T > self.k.shape[2]:
            cap = max(2 * (self.n + T), 1024)
            nk = k.new_zeros(*k.shape[:2], cap, k.shape[3])
            nv = v.new_zeros(*v.shape[:2], cap, v.shape[3])
            if self.k is not None:
                nk[:, :, :self.n], nv[:, :, :self.n] = self.k[:, :, :self.n], self.v[:, :, :self.n]
            self.k, self.v = nk, nv
        self.k[:, :, self.n:self.n + T] = k
        self.v[:, :, self.n:self.n + T] = v
        self.n += T


# ------------------------------------------------------------------------------------------------ controller
class Readback:
    """Read-back of out-of-window positions into a sliding-window model's attention layers (module docstring)."""

    def __init__(self, model: nn.Module, cfg: ReadbackConfig) -> None:
        self.model, self.cfg = model, cfg
        self.attn = [m for m in model.modules() if hasattr(m, "rotary_emb") and hasattr(m, "layer_idx")
                     and hasattr(m, "k_proj")]
        if not self.attn:
            raise ValueError("no attention modules with rotary_emb / layer_idx / k_proj found")
        W = getattr(model.config, "sliding_window", None) or getattr(model.config, "attention_window_size", None)
        if not W:
            raise ValueError("the model has no sliding window")
        self.W = int(W)
        self.sink = cfg.sink
        self.budget = cfg.budget if cfg.selector != "all" else 0
        self.recent = self.W - self.sink - self.budget
        if self.recent < 1:
            raise ValueError(f"sink {cfg.sink} + budget {cfg.budget} leave no recent columns in W = {self.W}")
        self.stores: dict[int, _Store] = {}
        self.oracle: list[list[tuple[int, int]]] | None = None
        self.perms: list[list[int]] | None = None
        self.rand_seeds: list[int] | None = None
        self.last_selected: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.stats = {"max_cols": 0, "calls": 0, "selected_sum": 0.0, "selected_n": 0}
        self._handles: list = []

    # ------------------------------------------------------------------------------------------ lifecycle
    def columns(self) -> int:
        return self.W if self.cfg.selector in ("window",) else self.sink + self.budget + self.recent

    def begin(self, B: int, oracle: list[list[tuple[int, int]]] | None = None,
              rand_seeds: list[int] | None = None) -> None:
        """New sequences: clear the archives; per-row oracle segments and random-order seeds."""
        if oracle is not None and len(oracle) != B:
            raise ValueError(f"{len(oracle)} oracle rows for batch {B}")
        if self.cfg.selector == "foreign" and B < 2:
            raise ValueError("foreign read-back needs batch ≥ 2 (rows are rolled by one)")
        if self.cfg.selector in ("oracle", "foreign") and oracle is None:
            raise ValueError(f"{self.cfg.selector} needs oracle segments")
        self.stores = {}
        self.oracle = [sorted((int(a), int(n)) for a, n in row) for row in oracle] if oracle is not None else None
        self.rand_seeds = list(rand_seeds) if rand_seeds is not None else list(range(B))
        self.perms = None
        self.last_selected = {}

    @contextmanager
    def active(self):
        cfg = self.model.config
        old = cfg._attn_implementation
        for m in self.attn:
            m._readback = self
            self._handles.append(m.register_forward_pre_hook(self._pos_hook, with_kwargs=True))
        cfg._attn_implementation = IMPL
        try:
            yield self
        finally:
            cfg._attn_implementation = old
            for h in self._handles:
                h.remove()
            self._handles = []
            for m in self.attn:
                m._readback = None

    @staticmethod
    def _pos_hook(mod, args, kwargs):
        pos = kwargs.get("position_ids")
        if pos is None and len(args) > 1:
            pos = args[1]
        mod._rb_pos = pos
        return None

    # ------------------------------------------------------------------------------------------- selection
    def _blocks_eligible(self, t_ref: int) -> int:
        """Number of whole blocks k (positions s + k·b … s + (k+1)·b − 1) that end at or before t_ref − D."""
        lim = t_ref - self.recent - self.sink + 1
        return max(0, lim // self.cfg.block)

    def _row_segments(self, li: int, st: _Store, t_ref: int, qsel: torch.Tensor | None,
                      inv_freq: torch.Tensor) -> tuple[list[list[tuple[int, int]]], list[int]]:
        """Per row: segments [(start, length)] in position order and the source row of their archive."""
        cfg, s, D, R = self.cfg, self.sink, self.recent, self.budget
        B = st.k.shape[0]
        last = t_ref - D                                          # the latest eligible position
        src = list(range(B))
        if cfg.selector in ("window", "stream"):
            return [[] for _ in range(B)], src
        if cfg.selector == "all":
            n = last - s + 1
            return [[(s, n)] if n > 0 else [] for _ in range(B)], src
        if cfg.selector in ("oracle", "foreign"):
            if cfg.selector == "foreign":
                src = [(b + 1) % B for b in range(B)]
            out = []
            for b in range(B):
                segs, used = [], 0
                for a, n in self.oracle[src[b]]:
                    a2 = max(a, s)
                    n2 = n - (a2 - a)
                    if n2 <= 0 or a2 + n2 - 1 > last or used + n2 > R:
                        continue
                    segs.append((a2, n2))
                    used += n2
                out.append(segs)
            return out, src
        nb = self._blocks_eligible(t_ref)
        take = R // cfg.block
        bl = cfg.block
        if cfg.selector == "random":
            if self.perms is None or len(self.perms[0]) < nb:
                size = max(nb, 4096)
                self.perms = []
                for sd in self.rand_seeds:
                    p = list(range(size))
                    random.Random(sd).shuffle(p)
                    self.perms.append(p)
            out = []
            for b in range(B):
                ks = [k for k in self.perms[b] if k < nb][:take]
                out.append([(s + k * bl, bl) for k in sorted(ks)])
            return out, src
        # infllm
        nb = min(nb, len(st.reps))
        if nb == 0 or qsel is None or qsel.shape[2] == 0:
            return [[] for _ in range(B)], src
        reps = torch.stack(st.reps[:nb], 1)                      # (B, nb, n_rep, d)
        d = reps.shape[-1]
        sc = torch.einsum("bhqd,bkjd->bhqkj", qsel, reps) / d ** 0.5
        score = sc.amax(-1).sum(2).mean(1)                        # (B, nb)
        top = score.topk(min(take, nb), -1).indices
        return [[(s + int(k) * bl, bl) for k in sorted(top[b].tolist())] for b in range(B)], src

    def _update_reps(self, st: _Store, qu: torch.Tensor, p0: int, inv_freq: torch.Tensor) -> None:
        """Keep the queries of the incomplete block; for each block completed, its n_rep representative keys."""
        s, bl = self.sink, self.cfg.block
        if st.pend is None:
            st.pend, st.pend_start = qu, p0
        else:
            st.pend = torch.cat([st.pend, qu], 2)
        while True:
            a = s + len(st.reps) * bl                              # next block's first position
            if st.pend_start > a:                                  # rows missing (cannot happen in order)
                raise RuntimeError(f"block at {a} lost its queries (pending from {st.pend_start})")
            end = st.pend_start + st.pend.shape[2]
            if end < a + bl:
                break
            qb = st.pend[:, :, a - st.pend_start:a - st.pend_start + bl]          # (B, H, b, d)
            pos = torch.arange(a, a + bl, device=qb.device)
            kb = rope_shift(st.k[:, 0, a:a + bl], -pos, inv_freq)                 # (B, b, d) un-rotated
            sc = torch.einsum("bhqd,bkd->bhqk", qb, kb) / qb.shape[-1] ** 0.5
            causal = torch.ones(bl, bl, dtype=torch.bool, device=qb.device).tril()
            p = torch.softmax(sc.masked_fill(~causal, NEG_INF), -1)
            mass = p.sum((1, 2))                                   # (B, b): attention received
            idx = mass.topk(self.cfg.n_rep, -1).indices
            st.reps.append(kb.gather(1, idx[..., None].expand(-1, -1, kb.shape[-1])))
        keep = max(0, (s + len(st.reps) * bl) - st.pend_start)
        st.pend = st.pend[:, :, keep:]
        st.pend_start += keep

    # ------------------------------------------------------------------------------------------- attention
    def attend(self, mod: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               scaling: float) -> torch.Tensor:
        cfg = self.cfg
        pos = mod._rb_pos
        if pos is None:
            raise RuntimeError("no position_ids recorded for this attention call")
        pos = pos.to(query.device)
        if pos.dim() == 2:
            if pos.shape[0] > 1 and not bool((pos == pos[:1]).all()):
                raise ValueError("read-back needs every row at the same positions (no padding)")
            pos = pos[0]
        B, H, T, d = query.shape
        p0 = int(pos[0])
        if T > 1 and not bool((pos == torch.arange(p0, p0 + T, device=pos.device)).all()):
            raise ValueError("positions within a call must be contiguous")
        if cfg.selector == "infllm" and key.shape[1] != 1:
            raise ValueError("infllm representatives assume one KV head (RecurrentGemma); got " + str(key.shape[1]))
        li = mod.layer_idx
        st = self.stores.get(li)
        if st is None or p0 == 0:
            st = self.stores[li] = _Store()
        if p0 != st.n:
            raise ValueError(f"layer {li}: call starts at {p0} but {st.n} positions are archived")
        st.append(key[:, :, -T:], value[:, :, -T:])
        inv = mod.rotary_emb.inv_freq
        rel = cfg.time != "true"
        C = cfg.chunk if T > 1 else 1
        out = torch.empty(B, H, T, d, dtype=torch.float32, device=query.device)
        for c0 in range(0, T, C):
            c1 = min(T, c0 + C)
            rows = pos[c0:c1]
            t0, t1 = p0 + c0, p0 + c1
            qc = query[:, :, c0:c1].float()                        # rotated at the true positions
            need_qu = rel or cfg.selector == "infllm" or (self.sink > 0 and cfg.selector != "all")
            qu = rope_shift(qc, -rows, inv) if need_qu else None
            if cfg.selector == "infllm":
                if T == 1:
                    st.hist = qu if st.hist is None else torch.cat([st.hist, qu], 2)[:, :, -cfg.sel_rows:]
                qsel = st.hist
            else:
                qsel = None
            t_ref = t1 - 1 if (cfg.selector == "all" or T == 1) else t0
            segs, src = self._row_segments(li, st, t_ref, qsel, inv)
            out[:, :, c0:c1] = self._chunk(li, st, qc, qu, rows, segs, src, inv, scaling, t_ref)
            if cfg.selector == "infllm":
                if T > 1:
                    st.hist = qu
                self._update_reps(st, qu, t0, inv)
        self.stats["calls"] += 1
        return out.to(query.dtype).transpose(1, 2).contiguous()

    def _chunk(self, li: int, st: _Store, qc: torch.Tensor, qu: torch.Tensor | None, rows: torch.Tensor,
               segs: list[list[tuple[int, int]]], src: list[int], inv: torch.Tensor, scaling: float,
               t_ref: int) -> torch.Tensor:
        cfg, s, D, W = self.cfg, self.sink, self.recent, self.W
        dev = qc.device
        B = qc.shape[0]
        t0, t1 = int(rows[0]), int(rows[-1]) + 1
        rel = cfg.time != "true"
        # ---- recent band (true positions) and sink
        lo = max(0, t0 - D + 1)
        kr = st.k[:, :, lo:t1].float()
        vr = st.v[:, :, lo:t1].float()
        pr = torch.arange(lo, t1, device=dev)
        rec_vis = (pr[None] > rows[:, None] - D) & (pr[None] <= rows[:, None])          # (C, N)
        ns = min(s, t1)                                           # sink columns, shown once they leave the band
        sk_vis = torch.arange(ns, device=dev)[None] <= rows[:, None] - D                 # (C, ns)
        # ---- read-back entries
        M = max([sum(n for _, n in r) for r in segs] + [0])
        if M:
            pos = torch.full((B, M), -1, dtype=torch.long)
            off = torch.zeros(B, M, dtype=torch.long)
            lmax = torch.zeros(B, 1, dtype=torch.long)
            for b, r in enumerate(segs):
                if r:
                    pp = torch.cat([torch.arange(a, a + n) for a, n in r])
                    oo = torch.cat([torch.arange(n) for _, n in r])
                    pos[b, :len(pp)], off[b, :len(oo)] = pp, oo
                    lmax[b] = max(n for _, n in r)
            pos, off, lmax = pos.to(dev), off.to(dev), lmax.to(dev)
            valid = pos >= 0
            if bool((valid & ((pos < s) | (pos > t_ref - D))).any()):
                raise AssertionError(f"ineligible read-back position (s={s}, t_ref={t_ref}, D={D})")
            sb = torch.tensor(src, device=dev)[:, None]
            pc = pos.clamp_min(0)
            ks = st.k[sb, :, pc].permute(0, 2, 1, 3).float()      # (B, Hkv, M, d) rotated at p
            vs = st.v[sb, :, pc].permute(0, 2, 1, 3).float()
            nval = valid.sum(1, keepdim=True)
            if cfg.time == "ordinal":
                rank = valid.long().cumsum(1) - 1
                delta = D + (nval - 1 - rank)
            elif cfg.time == "none":
                delta = D + lmax - 1 - off
            if rel:
                ks = rope_shift(ks, (-delta - pc)[:, None], inv)
            rb_vis = valid[:, None, :] & (pos[:, None, :] <= rows[None, :, None] - D)    # (B, C, M)
            n_slots = lmax if cfg.time == "none" else nval              # true: the sink sits as under ordinal
            self.last_selected[li] = (torch.where(valid, pos, -1), sb[:, 0])
            self.stats["selected_sum"] += float(nval.float().mean())
            self.stats["selected_n"] += 1
        else:
            ks = vs = rb_vis = None
            n_slots = torch.zeros(B, 1, dtype=torch.long, device=dev)
            self.last_selected[li] = (torch.full((B, 0), -1, dtype=torch.long, device=dev),
                                      torch.tensor(src, device=dev))
        # ---- scores
        H = qc.shape[1]
        kr = _rep(kr, H)
        if ks is not None:
            ks = _rep(ks, H)
        sc_rec = (qc @ kr.transpose(-1, -2)) * scaling
        sc_rec = sc_rec.masked_fill(~rec_vis[None, None], NEG_INF)
        parts_s, parts_v = [], []
        if ns:
            ksk = st.k[:, :, :ns].float()
            if cfg.selector != "all":                             # StreamingLLM cache positions, any time code
                ps = torch.arange(ns, device=dev)
                dsk = D + n_slots + (ns - 1 - ps)[None]                                  # (B, ns)
                ksk = rope_shift(ksk, (-dsk - ps[None])[:, None], inv)
                parts_s.append((qu @ _rep(ksk, H).transpose(-1, -2)) * scaling)
            else:
                parts_s.append((qc @ _rep(ksk, H).transpose(-1, -2)) * scaling)
            parts_s[-1] = parts_s[-1].masked_fill(~sk_vis[None, None], NEG_INF)
            parts_v.append(st.v[:, :, :ns].float())
        if ks is not None:
            q_rb = qu if rel else qc
            sc_rb = (q_rb @ ks.transpose(-1, -2)) * scaling
            parts_s.append(sc_rb.masked_fill(~rb_vis[:, None], NEG_INF))
            parts_v.append(vs)
        parts_s.append(sc_rec)
        parts_v.append(vr)
        # ---- budget
        cols = rec_vis.sum(1)[None] + sk_vis.sum(1)[None]
        if rb_vis is not None:
            cols = cols + rb_vis.sum(2)
        mc = int(cols.max())
        self.stats["max_cols"] = max(self.stats["max_cols"], mc)
        if cfg.selector != "all" and mc > W:
            raise AssertionError(f"{mc} columns exceed the window budget {W}")
        p = torch.softmax(torch.cat(parts_s, -1), -1)
        o, j = 0.0, 0
        for v in parts_v:
            v = _rep(v, qc.shape[1])
            n = v.shape[2]
            o = o + p[..., j:j + n] @ v
            j += n
        return o

    def selected_mean(self) -> float | None:
        n = self.stats["selected_n"]
        return self.stats["selected_sum"] / n if n else None


# --------------------------------------------------------------------------------------- registered function
def readback_attention(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       attention_mask: torch.Tensor | None, scaling: float | None = None, dropout: float = 0.0,
                       **kwargs) -> tuple[torch.Tensor, None]:
    rb = getattr(module, "_readback", None)
    if rb is None:
        raise RuntimeError("readback attention called on a module without an active Readback")
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    return rb.attend(module, query, key, value, scaling), None


AttentionInterface.register(IMPL, readback_attention)
AttentionMaskInterface.register(IMPL, sdpa_mask)                 # the mask is built but not used
