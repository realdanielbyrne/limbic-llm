"""GistBank: bias-corrected running averages of the stream (the EC timescale bank behind the CA2 gist).

For each band i with rate β_i, over the *counted* tokens u (real, not the sink, optionally not massive),

    A_i(t) = Σ_{u≤t} β_i^{t−u} use_u x_u / Σ_{u≤t} β_i^{t−u} use_u        (0 before any counted token)

with x = h (or h ⊙ keep when ``dim_keep`` zeroes massive dimensions). The default two bands
(β = 0.9, 0.995) are exactly ``SteeringController``'s fast average m_t and slow average b_t
(``experiments/alignment/steering.py`` ``_ema``): the same (T, T) decay kernel K[t, u] = β^{t−u}, the
same einsum and the same carried numerator/denominator across calls, so the default gist

    d̂_t = (A_fast(t) − A_slow(t)) / ‖A_fast(t) − A_slow(t)‖

reproduces it. Because both averages are normalised means of the same tokens, a constant offset cancels.
The first ``sink_skip`` real tokens of each row (the attention sink, ~80× the median norm on Qwen3) are
never counted. State: per band a numerator (B, D) and a denominator (B,), plus the real-token count —
fixed size.

Exposed per call (``BankStep``):
  ``avg(i)``   A_i(t), (B, T, D);
  ``prev(i)``  A_i(t−1), the average carried to the step before t, (B, T, D), with ``prev_den(i)`` > 0
               where it is defined (the gist agreement source reads these, so a_t never contains h_t);
  ``gist(i, j)`` normalise(A_i − A_j), default (fast, slowest);
  ``massive``  (B, T) flags, and ``massive_prev`` (B, W_r) the flags of the W_r columns before this call.

Massive tokens. Column t is flagged when log‖h_t‖ − mean_{u<t, counted} log‖h_u‖ > log ρ_m (default
ρ_m = 10), the prefix mean coming from a carried (sum, count). The stream is the tap's pre-injection
output (block L), not block L's input as an early design note had it: the tap is where the bank reads,
and massive activations persist in the residual stream across blocks. Flags are kept only for the last
W_r = ``massive_window`` columns (column-aligned, padding included, so an attention tap can index keys
by column): fixed state. ``exclude_massive`` drops flagged tokens from the averages (off by default, so
the default bank matches ``steering.py``).

``massive_dims(h, ratio)`` lists the dimensions whose largest |activation| exceeds ``ratio`` × the
median |activation| (from one forward pass over neutral text); ``dim_keep_mask`` turns that list into
the keep vector the bank and the controller apply to the averages and to d̂.

Window reset (the ``window_gist`` control). With ``window=W`` the averages restart at every W-token
block of real tokens, aligned like ``pretrained_loop.window_mask`` block mode: block(t) =
⌊(idx_t − offset)/W⌋, idx_t the real-token index of the row and offset = ``window_offset`` (the sink).
The kernel then becomes (B, T, T), K[b, t, u] = β^{t−u}·1[block_b(t) = block_b(u)], and the carry
applies only while the block has not changed, so no information crosses a window boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def _kernel(rate: float, i: torch.Tensor) -> torch.Tensor:
    """Causal decay kernel K[t, u] = rate^{t−u} for u ≤ t, else 0, (T, T) (as ``SteeringController._kernel``)."""
    diff = i[:, None] - i[None, :]
    return torch.where(diff >= 0, rate ** diff.clamp_min(0), torch.zeros_like(diff))


def massive_dims(h: torch.Tensor, ratio: float = 50.0, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Dimensions d with max_t |h_{t,d}| > ratio · median_{t,d} |h_{t,d}| over real tokens, as a LongTensor.

    h (B, T, D) or (N, D); ``mask`` (B, T) selects real tokens (exclude the sink yourself if wanted).
    """
    x = h.float().abs()
    x = x[mask.bool()] if mask is not None else x.reshape(-1, x.shape[-1])
    peak = x.max(0).values
    return torch.nonzero(peak > ratio * x.median()).flatten()


def dim_keep_mask(d_model: int, dims: torch.Tensor | list[int]) -> torch.Tensor:
    keep = torch.ones(d_model)
    keep[torch.as_tensor(dims, dtype=torch.long)] = 0.0
    return keep


@dataclass
class BankStep:
    """Everything the bank computed for one call; bands are indexed like ``GistBank.betas``."""

    num: list[torch.Tensor]                  # (B, T, D) per band
    den: list[torch.Tensor]                  # (B, T) per band
    carry: list[tuple[torch.Tensor, torch.Tensor] | None]   # (B, D), (B,) before the call per band (None: none)
    carry_ok: torch.Tensor | None            # (B, T) bool: carried state still valid at t (window reset)
    mask: torch.Tensor                       # (B, T) real tokens
    use: torch.Tensor                        # (B, T) counted tokens
    idx: torch.Tensor                        # (B, T) real-token index in the row (−1 at padding)
    massive: torch.Tensor                    # (B, T) bool
    massive_prev: torch.Tensor               # (B, W_r) bool, columns before this call
    logn: torch.Tensor                       # (B, T) log ‖h‖
    _avg: dict = field(default_factory=dict)

    def avg(self, i: int) -> torch.Tensor:
        i = i % len(self.num)
        if i not in self._avg:
            num, den = self.num[i], self.den[i]
            self._avg[i] = torch.where(den.unsqueeze(-1) > 0, num / den.clamp_min(1e-30).unsqueeze(-1),
                                       torch.zeros_like(num))
        return self._avg[i]

    def _shift(self, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        first = torch.zeros_like(x[:, :1]) if c is None else c[:, None].to(x)
        out = torch.cat([first, x[:, :-1]], 1)
        if self.carry_ok is not None:                    # a window boundary at t: nothing carried into it
            ok = torch.cat([self.carry_ok[:, :1], self._same_block[:, 1:]], 1)
            out = out * ok.view(ok.shape + (1,) * (out.dim() - 2)).to(out)
        return out

    def prev_den(self, i: int) -> torch.Tensor:
        """Denominator of A_i(t−1), (B, T); > 0 where the previous-step average is defined."""
        i = i % len(self.num)
        return self._shift(self.den[i], None if self.carry[i] is None else self.carry[i][1])

    def prev(self, i: int) -> torch.Tensor:
        """A_i(t−1): the band-i average before h_t entered it, (B, T, D)."""
        i = i % len(self.num)
        num = self._shift(self.num[i], None if self.carry[i] is None else self.carry[i][0])
        den = self.prev_den(i)
        return torch.where(den.unsqueeze(-1) > 0, num / den.clamp_min(1e-30).unsqueeze(-1), torch.zeros_like(num))

    def gist(self, i: int = 0, j: int = -1) -> torch.Tensor:
        """normalise(A_i − A_j), (B, T, D); zero where undefined."""
        return F.normalize(self.avg(i) - self.avg(j), dim=-1)

    _same_block: torch.Tensor | None = None  # (B, T) bool: block(t) == block(t − 1) (window reset only)


class GistBank:
    """Bias-corrected running averages at several rates, sink skip, massive-token flags. Fixed-size state."""

    def __init__(self, betas: tuple[float, ...] = (0.9, 0.995), sink_skip: int = 1, exclude_massive: bool = False,
                 massive_ratio: float = 10.0, massive_window: int = 64, dim_keep: torch.Tensor | None = None,
                 window: int | None = None, window_offset: int | None = None) -> None:
        if not betas or not all(0.0 < b < 1.0 for b in betas):
            raise ValueError("betas must be in (0, 1)")
        if massive_ratio <= 1.0 or massive_window < 0:
            raise ValueError("massive_ratio must exceed 1 and massive_window be >= 0")
        if window is not None and window < 1:
            raise ValueError("window must be >= 1")
        self.betas = tuple(float(b) for b in betas)
        self.sink_skip, self.exclude_massive = int(sink_skip), bool(exclude_massive)
        self.log_ratio, self.massive_window = math.log(massive_ratio), int(massive_window)
        self.dim_keep = None if dim_keep is None else dim_keep.detach().float().flatten()
        self.window = window
        self.window_offset = self.sink_skip if window_offset is None else int(window_offset)
        self.reset()

    def reset(self) -> None:
        self.ema: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.betas)
        self.n_seen: torch.Tensor | None = None          # (B,) real tokens seen
        self.log_sum: torch.Tensor | None = None         # (B,) Σ log‖h‖ over counted tokens (massive prefix)
        self.log_cnt: torch.Tensor | None = None         # (B,)
        self.flags: torch.Tensor | None = None           # (B, W_r) massive flags of the last W_r columns
        self.last_block: torch.Tensor | None = None      # (B,) window block of the last real token (window reset)

    def state_numel(self) -> int:
        n = sum(a.numel() + b.numel() for e in self.ema if e is not None for a, b in [e])
        for t in (self.n_seen, self.log_sum, self.log_cnt, self.flags, self.last_block):
            n += 0 if t is None else t.numel()
        return n

    def update(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> BankStep:
        """Advance the bank over one call's stream h (B, T, D) with real-token ``mask`` (B, T)."""
        h = h.float()
        B, T, D = h.shape
        dev = h.device
        mask = torch.ones(B, T, device=dev) if mask is None else mask.to(dev).float()
        n_prev = self.n_seen if self.n_seen is not None else torch.zeros(B, device=dev)
        idx = n_prev[:, None] + mask.cumsum(1) - 1               # index of each real token within its sequence
        use = mask * (idx >= self.sink_skip).float()             # skip the attention sink (RV1.C1)
        self.n_seen = (n_prev + mask.sum(1)).detach()
        # massive-token flags from a carried prefix mean of log-norms over counted tokens
        logn = torch.log(h.norm(dim=-1).clamp_min(1e-30))
        s0 = self.log_sum if self.log_sum is not None else torch.zeros(B, device=dev)
        c0 = self.log_cnt if self.log_cnt is not None else torch.zeros(B, device=dev)
        cs = s0[:, None] + (use * logn).cumsum(1) - use * logn  # exclusive prefix sums
        cc = c0[:, None] + use.cumsum(1) - use
        massive = (use > 0) & (cc > 0) & (logn - cs / cc.clamp_min(1) > self.log_ratio)
        self.log_sum, self.log_cnt = (cs[:, -1] + use[:, -1] * logn[:, -1]).detach(), (cc[:, -1] + use[:, -1]).detach()
        W_r = self.massive_window
        fprev = self.flags if self.flags is not None else torch.zeros(B, W_r, dtype=torch.bool, device=dev)
        self.flags = torch.cat([fprev, massive], 1)[:, -W_r:].detach() if W_r else fprev
        if self.exclude_massive:
            use = use * (~massive).float()
        x = h if self.dim_keep is None else h * self.dim_keep.to(dev)
        i = torch.arange(T, device=dev, dtype=torch.float32)
        carry = list(self.ema)
        carry_ok = same = None
        if self.window is None:
            nums, dens = [], []
            for k, beta in enumerate(self.betas):
                num, den = self._ema(x, use, beta, k, i)
                nums.append(num)
                dens.append(den)
        else:
            blk = (idx - self.window_offset).div(self.window, rounding_mode="floor")
            blk = torch.where(idx >= self.window_offset, blk, torch.full_like(idx, -1.0))
            blk = torch.where(mask > 0, blk, torch.full_like(blk, -2.0))   # padding: its own block
            last = self.last_block if self.last_block is not None else torch.full((B,), -3.0, device=dev)
            carry_ok = blk == last[:, None]
            same = torch.cat([carry_ok[:, :1], blk[:, 1:] == blk[:, :-1]], 1)
            self.last_block = blk[:, -1].detach()
            nums, dens = [], []
            for k, beta in enumerate(self.betas):
                K = _kernel(beta, i)[None] * (blk[:, :, None] == blk[:, None, :]).float()
                num = torch.einsum("btu,bud->btd", K, x * use.unsqueeze(-1))
                den = torch.einsum("btu,bu->bt", K, use)
                if self.ema[k] is not None:
                    decay = (beta ** (i + 1))[None, :] * carry_ok.float()
                    num = num + self.ema[k][0][:, None, :] * decay[:, :, None]
                    den = den + self.ema[k][1][:, None] * decay
                self.ema[k] = (num[:, -1].detach(), den[:, -1].detach())
                nums.append(num)
                dens.append(den)
        step = BankStep(nums, dens, carry, carry_ok, mask, use, torch.where(mask > 0, idx, torch.full_like(idx, -1.0)),
                        massive, fprev, logn)
        step._same_block = same
        return step

    def _ema(self, x: torch.Tensor, use: torch.Tensor, beta: float, k: int, i: torch.Tensor):
        """``SteeringController._ema`` verbatim, returning numerator and denominator."""
        K = _kernel(beta, i)
        num = torch.einsum("tu,bud->btd", K, x * use.unsqueeze(-1))
        den = use @ K.T
        if self.ema[k] is not None:
            decay = beta ** (i + 1)
            num = num + self.ema[k][0][:, None, :] * decay[None, :, None]
            den = den + self.ema[k][1][:, None] * decay[None, :]
        self.ema[k] = (num[:, -1].detach(), den[:, -1].detach())
        return num, den
