"""Round 13 gated memory circuit (plan ``research/plans/round13_gated_memory_plan.md`` §2): fixed, nothing trained.

Every computation runs in float32 whatever the base model's dtype. The circuit reads two taps of a frozen model
(early: word-level state e_t; deep: contextual state h_t) and returns content only through read-back; nothing here
writes to the residual stream.

Parts
  rms_normalise      the buffer's normalisation (states are RMS-normalised before they enter the filters).
  RingBuffer         the last N (4,096) RMS-normalised states of one tap; fixed size.
  massive_mask       sink positions, and massive activations (raw norm above ``factor`` × the median raw norm of the
                     previous ``ring`` non-sink positions; causal), are excluded from the filters.
  ema_scan           y[t] = Σ_{s ≤ t} a^{t−s} u[s], exact chunked form (the vectorised filter).
  FilterBank         plan §2.2. With x the RMS-normalised deep state, x̃ = x − m (m: running document mean,
                     τ = 4,096, causal, over non-excluded positions; ``centre=False`` uses x):
                         num_k[t] = a_k num_k[t−1] + g[t−1] x̃[t−1]       a_k = exp(−1/τ_k)
                         den_k[t] = a_k den_k[t−1] + g[t−1]
                         band_k[t] = num_k[t] / den_k[t]   (0 while den_k = 0),  strength_k[t] = den_k[t]
                     lag 0 is excluded (band at t uses inputs < t; ``lag0=True`` includes x̃[t]). Fast bank
                     (τ = 4, 16, 64): g = 1 on non-excluded positions. Slow bank (τ = 256, 1024): g = 1 only where
                     the dentate salience exceeds ``gate_threshold`` (``slow_gated=False``: like the fast bank).
                     Outputs: band_k − band_{k+1} for neighbouring bands, plus the slowest band (``band_diff=False``:
                     the raw bands). Strength is exposed for the slow bank only (the fast bank's is ≈ τ).
                     ``forward`` is the whole-sequence form; ``stream()`` returns a per-token form with a fixed-size
                     state (``FilterBankStream``); the two agree (tests).
  DentateGate        plan §2.3: code = top-k(ReLU(R · LN(e_t))²), R a fixed seeded Gaussian expansion (``expand`` ×
                     width, rows scaled 1/√width), k = ``k_frac`` of the units. Key: the sparse code (indices,
                     values). Salience: the code's total strength (sum of its values). Per position, stateless.
  EpisodicStore      plan §2.4: an entry per span holds the pooled sparse key (mean of the span's codes), the context
                     tag (bank outputs at the span's last token; each output unit-normalised, concatenated,
                     unit-normalised), the deep state pooled over the span, an ordinal index and a pointer
                     (start, length) into the K/V archive. Recall: overlap(key_t, key_j) + λ · cos(ctx_t, tag_j),
                     overlap = cosine of the non-negative codes. ``capacity`` bounds the entries (oldest evicted).
  comparator         plan §2.5: pathways d (current deep state), r (deep state of an entry), c (slowest band
                     difference), each centred (minus the document mean m) and unit-normalised;
                     a_t = cos(d, r_top) − mean_{j ≠ top} cos(d, r_j); out = normalise(ReLU(d + r_top + c)²);
                     mismatch = d − r_top.

Ablation switches (step 2d) are ``CircuitConfig`` fields: ``centre``, ``lag0``, ``slow_gated``, ``band_diff``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

TAUS = (4, 16, 64, 256, 1024)
N_FAST = 3                                                         # τ = 4, 16, 64; the rest form the slow bank


@dataclass
class CircuitConfig:
    taus: tuple = TAUS
    n_fast: int = N_FAST
    mean_tau: float = 4096.0
    ring: int = 4096
    sink: int = 4
    massive: float = 10.0
    expand: int = 4
    k_frac: float = 0.05
    seed: int = 0
    gate_threshold: float = math.inf                               # dentate salience; set from dev items
    centre: bool = True
    lag0: bool = False
    slow_gated: bool = True
    band_diff: bool = True

    def __post_init__(self) -> None:
        self.taus = tuple(float(t) for t in self.taus)
        if not 0 <= self.n_fast <= len(self.taus):
            raise ValueError(f"n_fast {self.n_fast} for {len(self.taus)} bands")
        if list(self.taus) != sorted(self.taus):
            raise ValueError("taus must be increasing")

    @property
    def n_slow(self) -> int:
        return len(self.taus) - self.n_fast


# ------------------------------------------------------------------------------------------------ helpers
def rms_normalise(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def unit(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class RingBuffer:
    """The last ``n`` RMS-normalised states (B, n, D) of one tap; ``count`` positions pushed so far."""

    def __init__(self, n: int, B: int, D: int, device="cpu") -> None:
        self.n = int(n)
        self.buf = torch.zeros(B, self.n, D, dtype=torch.float32, device=device)
        self.pos = torch.full((self.n,), -1, dtype=torch.long)
        self.count = 0

    def push(self, h: torch.Tensor) -> None:
        i = self.count % self.n
        self.buf[:, i] = rms_normalise(h)
        self.pos[i] = self.count
        self.count += 1

    def get(self, p: int) -> torch.Tensor:
        """State of absolute position ``p`` (must still be held)."""
        i = p % self.n
        if int(self.pos[i]) != p:
            raise KeyError(f"position {p} has left the ring (holds {self.count - self.n}…{self.count - 1})")
        return self.buf[:, i]

    def pool(self, a: int, b: int) -> torch.Tensor:
        """Mean state over positions a … b − 1."""
        return torch.stack([self.get(p) for p in range(a, b)], 1).mean(1)

    def state_numel(self) -> int:
        return self.buf.numel() + self.pos.numel()


def running_median(norms: torch.Tensor, sink: int, ring: int) -> torch.Tensor:
    """(B, T): median of norms over the previous ``ring`` non-sink positions (max(sink, t − ring) … t − 1); NaN when
    there are none. Causal by construction."""
    B, T = norms.shape
    n = norms.float().clone()
    n[:, :min(sink, T)] = float("nan")
    seq = torch.cat([torch.full((B, ring), float("nan"), device=n.device), n], 1)
    med = torch.empty(B, T, device=n.device)
    step = max(1, (1 << 24) // max(ring, 1))                       # bound the sort's memory
    for b in range(B):
        win = seq[b].unfold(0, ring, 1)[:T]                        # win[t] = norms[t − ring … t − 1]
        for c0 in range(0, T, step):
            med[b, c0:c0 + step] = torch.nanmedian(win[c0:c0 + step], dim=-1).values
    return med


def massive_mask(norms: torch.Tensor, sink: int, ring: int, factor: float) -> torch.Tensor:
    """(B, T) bool: positions excluded from the filters (sink, or raw norm > factor × running median).

    The median at t is over the raw norms of the previous ``ring`` positions that are not sink positions
    (positions max(sink, t − ring) … t − 1); with none, nothing is massive. Causal by construction.
    """
    B, T = norms.shape
    pos = torch.arange(T, device=norms.device)
    excl = (pos < sink)[None].expand(B, T).clone()
    big = norms.float() > factor * running_median(norms, sink, ring)          # NaN median → False
    return excl | big


def _decay_matrix(a: float, C: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.arange(C, device=device, dtype=torch.float64)
    d = i[:, None] - i[None, :]
    L = torch.where(d >= 0, torch.pow(torch.tensor(a, dtype=torch.float64, device=device), d.clamp_min(0)),
                    torch.zeros((), dtype=torch.float64, device=device))
    pw = torch.pow(torch.tensor(a, dtype=torch.float64, device=device), i + 1)
    return L.float(), pw.float()


def ema_scan(u: torch.Tensor, a: float, chunk: int = 128) -> torch.Tensor:
    """y[t] = Σ_{s ≤ t} a^{t−s} u[s] along dim 1 of u (B, T, ...), float32, exact up to rounding.

    Within a chunk a lower-triangular decay matrix (zeros above the diagonal, so later inputs never reach earlier
    rows), across chunks a carried state. Bitwise causal for finite inputs.
    """
    u = u.float()
    B, T = u.shape[:2]
    uf = u.reshape(B, T, -1)
    C = max(1, min(chunk, T))
    L, pw = _decay_matrix(a, C, u.device)
    y = torch.empty_like(uf)
    carry = torch.zeros(B, uf.shape[2], device=u.device)
    for c0 in range(0, T, C):
        c1 = min(T, c0 + C)
        n = c1 - c0
        yc = torch.einsum("ij,bjf->bif", L[:n, :n], uf[:, c0:c1]) + pw[:n, None] * carry[:, None]
        y[:, c0:c1] = yc
        carry = yc[:, -1]
    return y.reshape(u.shape)


def _shift(y: torch.Tensor) -> torch.Tensor:
    """y[t − 1] at t (0 at t = 0): the lag-0-excluded value."""
    out = torch.zeros_like(y)
    out[:, 1:] = y[:, :-1]
    return out


# ---------------------------------------------------------------------------------------------- filter bank
@dataclass
class BankOut:
    bands: torch.Tensor                                            # (B, T, K, D) band contents
    outputs: torch.Tensor                                          # (B, T, K, D) differences + slowest (or raw)
    strength: torch.Tensor                                         # (B, T, n_slow) slow-bank den
    mean: torch.Tensor                                             # (B, T, D) running document mean
    excluded: torch.Tensor                                         # (B, T) bool
    gate: torch.Tensor                                             # (B, T) slow-bank gate g
    x: torch.Tensor | None = None                                  # (B, T, D) RMS-normalised input


class FilterBank:
    """Fixed exponential filter bank over the deep tap (module docstring)."""

    def __init__(self, cfg: CircuitConfig) -> None:
        self.cfg = cfg
        self.a = [math.exp(-1.0 / t) for t in cfg.taus]
        self.a_m = math.exp(-1.0 / cfg.mean_tau)

    def gates(self, excluded: torch.Tensor, salience: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        w = (~excluded).float()
        if self.cfg.slow_gated:
            if salience is None:
                raise ValueError("the gated slow bank needs the dentate salience")
            gs = w * (salience.float() > self.cfg.gate_threshold).float()
        else:
            gs = w
        return w, gs

    def outputs_from(self, bands: torch.Tensor) -> torch.Tensor:
        if not self.cfg.band_diff:
            return bands
        diffs = bands[:, :, :-1] - bands[:, :, 1:]
        return torch.cat([diffs, bands[:, :, -1:]], 2)

    def forward(self, h: torch.Tensor, salience: torch.Tensor | None = None, only_fast: bool = False) -> BankOut:
        """Whole-sequence form. h (B, T, D) raw deep states (any dtype); salience (B, T) the gate signal (dentate
        salience in the plan; ``rank13 --gate-signal`` may choose another). ``only_fast``: the fast bands only (they
        do not depend on the gate), e.g. to compute novelty before the gate signal is known."""
        cfg = self.cfg
        h = h.float()
        norms = h.norm(dim=-1)
        excl = massive_mask(norms, cfg.sink, cfg.ring, cfg.massive)
        x = rms_normalise(h)
        if only_fast:
            w, gs = (~excl).float(), torch.zeros_like(excl, dtype=torch.float32)
        else:
            w, gs = self.gates(excl, salience)
        mnum = ema_scan(w[..., None] * x, self.a_m)
        mden = ema_scan(w, self.a_m)
        mean = torch.where(mden[..., None] > 0, mnum / mden.clamp_min(1e-30)[..., None], torch.zeros_like(mnum))
        xt = x - mean if cfg.centre else x
        bands, strength = [], []
        for k, a in enumerate(self.a):
            if only_fast and k >= cfg.n_fast:
                break
            g = w if k < cfg.n_fast else gs
            num = ema_scan(g[..., None] * xt, a)
            den = ema_scan(g, a)
            if not cfg.lag0:
                num, den = _shift(num), _shift(den)
            bands.append(torch.where(den[..., None] > 0, num / den.clamp_min(1e-30)[..., None], torch.zeros_like(num)))
            if k >= cfg.n_fast:
                strength.append(den)
        bands = torch.stack(bands, 2)
        st = torch.stack(strength, 2) if strength else torch.zeros(*h.shape[:2], 0, device=h.device)
        return BankOut(bands, bands if only_fast else self.outputs_from(bands), st, mean, excl, gs, x)

    __call__ = forward

    def stream(self, B: int, D: int, device="cpu") -> "FilterBankStream":
        return FilterBankStream(self, B, D, device)


class FilterBankStream:
    """Per-token form of ``FilterBank`` with a fixed-size state (num, den per band, the mean, a ring of norms)."""

    def __init__(self, bank: FilterBank, B: int, D: int, device="cpu") -> None:
        self.bank, self.cfg = bank, bank.cfg
        K = len(self.cfg.taus)
        self.num = torch.zeros(B, K, D, device=device)
        self.den = torch.zeros(B, K, device=device)
        self.mnum = torch.zeros(B, D, device=device)
        self.mden = torch.zeros(B, device=device)
        self.ring = torch.full((B, self.cfg.ring), float("nan"), device=device)
        self.ptr = 0
        self.t = 0

    def state_numel(self) -> int:
        return sum(x.numel() for x in (self.num, self.den, self.mnum, self.mden, self.ring)) + 2

    def step(self, h: torch.Tensor, salience: torch.Tensor | None = None) -> dict:
        """One position: h (B, D) raw state, salience (B,). Returns the outputs at this position."""
        cfg, bank = self.cfg, self.bank
        h = h.float()
        nrm = h.norm(dim=-1)
        med = torch.nanmedian(self.ring, dim=-1).values
        excl = (nrm > cfg.massive * med) | torch.tensor(self.t < cfg.sink, device=h.device)
        x = rms_normalise(h)
        w, gs = bank.gates(excl[:, None], None if salience is None else salience[:, None])
        w, gs = w[:, 0], gs[:, 0]
        self.mnum = bank.a_m * self.mnum + w[:, None] * x
        self.mden = bank.a_m * self.mden + w
        mean = torch.where(self.mden[:, None] > 0, self.mnum / self.mden.clamp_min(1e-30)[:, None],
                           torch.zeros_like(self.mnum))
        xt = x - mean if cfg.centre else x
        a = torch.tensor(bank.a, device=h.device)
        g = torch.stack([w if k < cfg.n_fast else gs for k in range(len(bank.a))], 1)     # (B, K)

        def content():
            return torch.where(self.den[..., None] > 0, self.num / self.den.clamp_min(1e-30)[..., None],
                               torch.zeros_like(self.num))
        if cfg.lag0:
            self.num = a[None, :, None] * self.num + g[..., None] * xt[:, None]
            self.den = a[None] * self.den + g
            bands, den = content(), self.den.clone()
        else:
            bands, den = content(), self.den.clone()
            self.num = a[None, :, None] * self.num + g[..., None] * xt[:, None]
            self.den = a[None] * self.den + g
        if self.t >= cfg.sink:
            self.ring[:, self.ptr] = nrm
            self.ptr = (self.ptr + 1) % cfg.ring
        self.t += 1
        out = bank.outputs_from(bands[:, None])[:, 0]
        return {"bands": bands, "outputs": out, "strength": den[:, cfg.n_fast:], "mean": mean, "excluded": excl,
                "gate": gs}


def context_tag(outputs: torch.Tensor) -> torch.Tensor:
    """(..., K, D) bank outputs → (..., K·D): each output unit-normalised, concatenated, unit-normalised."""
    return unit(unit(outputs).flatten(-2))


# --------------------------------------------------------------------------------------------- dentate gate
class DentateGate:
    """Fixed random expansion + squared ReLU + top-k (module docstring). Stateless per position."""

    def __init__(self, d_model: int, expand: int = 4, k_frac: float = 0.05, seed: int = 0, device="cpu") -> None:
        self.D, self.U = int(d_model), int(expand * d_model)
        self.k = max(1, int(round(k_frac * self.U)))
        g = torch.Generator().manual_seed(int(seed))
        self.R = (torch.randn(self.U, self.D, generator=g) / math.sqrt(self.D)).to(device)

    def to(self, device) -> "DentateGate":
        self.R = self.R.to(device)
        return self

    def forward(self, e: torch.Tensor, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """e (..., D) → (idx (..., k) long, val (..., k) float32, salience (...))."""
        shp = e.shape[:-1]
        z = F.layer_norm(e.float().reshape(-1, self.D), (self.D,))
        idx = torch.empty(z.shape[0], self.k, dtype=torch.long, device=z.device)
        val = torch.empty(z.shape[0], self.k, device=z.device)
        for c0 in range(0, z.shape[0], chunk):
            p = F.relu(z[c0:c0 + chunk] @ self.R.T).pow(2)
            v, i = p.topk(self.k, -1)
            idx[c0:c0 + chunk], val[c0:c0 + chunk] = i, v
        return idx.reshape(*shp, self.k), val.reshape(*shp, self.k), val.sum(-1).reshape(shp)

    __call__ = forward

    def dense(self, idx: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(*idx.shape[:-1], self.U, device=idx.device)
        return out.scatter_(-1, idx, val)


def code_overlap(q: torch.Tensor, idx: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    """Cosine between a dense key q (U,) and sparse codes (..., k): the dentate overlap per position."""
    qn = unit(q.float())
    dot = (qn[idx] * val).sum(-1)
    return dot / val.norm(dim=-1).clamp_min(1e-12)


# -------------------------------------------------------------------------------------------- episodic store
@dataclass
class EpisodicStore:
    """CA3 entries (one per span). Tensors on one device; ``capacity`` keeps the newest entries."""
    keys: torch.Tensor                                             # (n, U) unit-normalised pooled codes
    tags: torch.Tensor                                             # (n, K·D) context tags
    states: torch.Tensor                                           # (n, D) pooled deep states (RMS-normalised)
    ordinal: torch.Tensor                                          # (n,) long
    pointer: list = field(default_factory=list)                    # [(start, length)] into the K/V archive

    @classmethod
    def from_spans(cls, spans: list[tuple[int, int]], gate: DentateGate, idx: torch.Tensor, val: torch.Tensor,
                   outputs: torch.Tensor, deep: torch.Tensor, capacity: int | None = None) -> "EpisodicStore":
        """spans [(a, b)] (b exclusive) in position order; idx/val (T, k) codes; outputs (T, K, D); deep (T, D)."""
        if capacity is not None and len(spans) > capacity:
            spans = spans[-capacity:]
        dev = idx.device
        n = len(spans)
        T = idx.shape[0]
        owner = torch.full((T,), -1, dtype=torch.long, device=dev)
        for j, (a, b) in enumerate(spans):
            owner[a:b] = j
        pos = (owner >= 0).nonzero().squeeze(1)
        cnt = torch.bincount(owner[pos], minlength=n).float().clamp_min(1)
        keys = torch.zeros(n, gate.U, device=dev)
        rows = owner[pos][:, None].expand(-1, idx.shape[1])
        keys.index_put_((rows.reshape(-1), idx[pos].reshape(-1)), val[pos].reshape(-1), accumulate=True)
        keys = unit(keys / cnt[:, None])
        x = rms_normalise(deep)
        states = torch.zeros(n, deep.shape[-1], device=dev).index_add_(0, owner[pos], x[pos]) / cnt[:, None]
        last = torch.tensor([b - 1 for _, b in spans], device=dev, dtype=torch.long)
        tags = context_tag(outputs[last])
        return cls(keys, tags, states, torch.arange(n, device=dev), [(a, b - a) for a, b in spans])

    def __len__(self) -> int:
        return self.keys.shape[0]

    def terms(self, key: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The two recall terms per entry: overlap(key, key_j) and cos(ctx, tag_j)."""
        return self.keys @ unit(key.float()), self.tags @ unit(ctx.float())

    def recall(self, key: torch.Tensor, ctx: torch.Tensor | None = None, lam: float = 0.0) -> torch.Tensor:
        """score_j = overlap(key, key_j) + λ · cos(ctx, tag_j)."""
        s = self.keys @ unit(key.float())
        if lam and ctx is not None:
            s = s + lam * (self.tags @ unit(ctx.float()))
        return s


# ------------------------------------------------------------------------------------------------ comparator
def comparator(d: torch.Tensor, states: torch.Tensor, top: int, c: torch.Tensor,
               mean: torch.Tensor | None = None) -> dict:
    """CA1 comparator (module docstring). d (D,) the current deep state, RMS-normalised here (so a raw state gives
    the same result as a normalised one: the states and the mean live on the RMS-normalised scale), states (n, D)
    the entries' pooled RMS-normalised states, c (D,), mean (D,) the document mean of the RMS-normalised states."""
    m = torch.zeros_like(d, dtype=torch.float32) if mean is None else mean.float()
    du = unit(rms_normalise(d) - m)
    ru = unit(states.float() - m)
    cu = unit(c.float())
    cos = ru @ du
    others = torch.cat([cos[:top], cos[top + 1:]])
    a = cos[top] - (others.mean() if others.numel() else torch.zeros((), device=cos.device))
    out = unit(F.relu(du + ru[top] + cu).pow(2))
    return {"a": float(a), "cos_top": float(cos[top]), "cos_others": float(others.mean()) if others.numel() else 0.0,
            "out": out, "mismatch": du - ru[top]}


# ------------------------------------------------------------------------------------- training-free saliences
def novelty(x: torch.Tensor, mean: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """1 − cos(x̃_t, band_t): the centred (RMS-normalised) state at t against a fast band's content at t (inputs < t
    with lag 0 excluded). (..., T, D) → (..., T); 1 where the band is still empty. Causal (inputs ≤ t)."""
    xt = x.float() - mean.float()
    c = (unit(xt) * unit(band.float())).sum(-1)
    return 1.0 - c


def key_novelty(gate: "DentateGate", idx: torch.Tensor, val: torch.Tensor, tau: float = 64.0,
                chunk: int = 1024) -> torch.Tensor:
    """1 − overlap(code_t, exponential average of the codes at positions < t, time constant τ). idx/val (T, k) →
    (T,); 1 at t = 0. Strictly causal: the average excludes t."""
    T = idx.shape[0]
    a = math.exp(-1.0 / tau)
    out = torch.empty(T, device=idx.device)
    hist = torch.zeros(gate.U, device=idx.device)                  # Σ_{s < t} a^{t−1−s} code_s, carried
    L, pw = _decay_matrix(a, chunk, idx.device)
    for c0 in range(0, T, chunk):
        c1 = min(T, c0 + chunk)
        n = c1 - c0
        dense = gate.dense(idx[c0:c1], val[c0:c1])                 # (n, U)
        incl = L[:n, :n] @ dense + pw[:n, None] * hist[None]       # inclusive EMA at each row
        prev = torch.cat([hist[None], incl[:-1]], 0)               # exclusive: the row before
        num = (dense * prev).sum(-1)
        den = dense.norm(dim=-1) * prev.norm(dim=-1)
        out[c0:c1] = 1.0 - torch.where(den > 0, num / den.clamp_min(1e-30), torch.zeros_like(num))
        hist = incl[-1]
    return out


def norm_ratio(e: torch.Tensor, sink: int, ring: int) -> torch.Tensor:
    """‖e_t‖ / running median of the previous ``ring`` non-sink norms (1 where there is none). (B, T, D) → (B, T)."""
    n = e.float().norm(dim=-1)
    med = running_median(n, sink, ring)
    return torch.where(torch.isfinite(med) & (med > 0), n / med, torch.ones_like(n))
