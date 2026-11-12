"""Relevance sources for the CA2 gain: a protocol, the gist-agreement source and the gain controls.

A source maps one forward call's ``StepContext`` to per-token signals in a dict of (B, T) float32
tensors. The key ``"s"`` is the relevance score in [0, 1] and is required; ``"release"`` (bool-valued),
if present, is a release event (R3); every other key is an extra signal. The controller calls *every
registered source on every step*, whether or not it drives the gain, and stores each signal in the
per-token trace as ``src/<source name>/<key>`` (measurement policy: every signal in every arm).

Protocol (duck-typed; subclass ``Source`` for the defaults):

    name: str          unique key in the controller and the trace
    final: bool        True for gain controls: "s" is then r_t itself and skips polarity, smoothing,
                       release and regulators (so a replayed gain is replayed exactly)
    needs: tuple[str]  names of taps the source reads (``ctx.taps[name]``); checked at the first step
    reset()            start a new sequence (called by ``CA2Controller.reset``)
    __call__(ctx) -> dict[str, Tensor (B, T)]

Causality: a source may use only ``ctx`` (this call, positions ≤ t through each column) and state it
carried from earlier calls. ``ctx.bank.prev(i)`` gives the band averages *before* h_t, so a score at t
never contains the token's own state unless the source chooses so.

Reading a tap for the current call. Taps (``controller.Tap``) fill their own per-call buffers from
their hooks during the forward pass, before the controller's step runs at the injection site; a
source reads them through ``ctx.taps[name]`` (the tap object; its buffer layout is the tap's own
documented contract). Tap tensors are in *model* coordinates; when a wrapper hides leading columns
from the controller (``SteerSide`` strips the sink), slice them with ``[:, ctx.col_offset:]``.

Sources here:
  ``agree``           a_t = cos(h_t − b_{t−1}, m_{t−1} − b_{t−1}), m and b the fast and slow band averages
                      carried to t−1 (R3/RV2). s_t = [a_t]_+, or the clip map (a_t − a_lo)/(a_hi − a_lo)
                      clamped to [0, 1]; release when a_t < −θ_r. a = 0 where the previous averages are
                      undefined (before two counted tokens) and at uncounted positions.
  ``constant``        s_t = c (with g_min = g_max, or c = 1, a fixed gain through the same pipeline).
  ``yoked``           (final) r_t replayed from a donor trajectory recorded on another prompt: row b reads
                      donor[b, k mod len_b], k the steered-step index (or the real-token index). Arrange
                      the donor rows as a derangement with ``derange`` (shuffling a row's own trajectory
                      would read its future, so it is not offered). Independent of this row's content.
  ``position_mean``   (final) r_t = table[min(k, N − 1)], the mean r at each position across documents
                      (measured on another run): the position-matched control. A (B, N) table gives each row
                      its own (e.g. leave-one-out) trajectory: r_{b,t} = table[b, min(k, N − 1)].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class StepContext:
    """One forward call as the gain step sees it. Shapes: B rows, T columns of this call, D width."""

    h: torch.Tensor                    # (B, T, D) float32, the tap's pre-injection stream
    x: torch.Tensor                    # (B, T, D) float32, the injection-site stream (is h when M = L + 1)
    mask: torch.Tensor                 # (B, T) float, real tokens
    use: torch.Tensor                  # (B, T) float, tokens the bank counted (real, not sink, maybe not massive)
    steer: torch.Tensor                # (B, T) float, steered positions (gain may be non-zero only here)
    idx: torch.Tensor                  # (B, T) float, real-token index in the row (−1 at padding)
    step: torch.Tensor                 # (B, T) long, steered-step index k (steered positions before t)
    col: torch.Tensor                  # (T,) long, absolute column of each position since reset
    first: bool                        # first call since reset (the prefill under KV-cached decoding)
    call: int                          # calls since reset
    ids: torch.Tensor | None           # (B, T) long input ids of this call (None without the ids hook)
    bank: Any                          # bank.BankStep of this call
    taps: dict = field(default_factory=dict)      # name -> Tap
    col_offset: int = 0                # leading model columns the controller does not see (SteerSide sink)
    ctrl: Any = None                   # the controller (read-only use)


class Source:
    name = "source"
    final = False
    needs: tuple[str, ...] = ()

    def reset(self) -> None:
        pass

    def state_numel(self) -> int:
        return 0

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class AgreeSource(Source):
    """Gist agreement from the averages carried to t − 1; reinforce polarity gives s = [a]_+."""

    def __init__(self, fast: int = 0, slow: int = -1, a_lo: float | None = None, a_hi: float | None = None,
                 theta_release: float | None = None, name: str = "agree") -> None:
        if (a_lo is None) != (a_hi is None) or (a_lo is not None and not a_lo < a_hi):
            raise ValueError("give both a_lo < a_hi, or neither")
        self.fast, self.slow, self.a_lo, self.a_hi = fast, slow, a_lo, a_hi
        self.theta_release, self.name = theta_release, name

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        b = ctx.bank
        pm, pb = b.prev(self.fast), b.prev(self.slow)
        ok = (b.prev_den(self.fast) > 0) & (b.prev_den(self.slow) > 0) & (ctx.use > 0)
        a = F.cosine_similarity(ctx.h - pb, pm - pb, dim=-1, eps=1e-8)
        a = torch.where(ok, a, torch.zeros_like(a))
        if self.a_lo is None:
            s = a.clamp_min(0.0)
        else:
            s = ((a - self.a_lo) / (self.a_hi - self.a_lo)).clamp(0.0, 1.0)
        out = {"s": s, "a": a, "defined": ok.float()}
        if self.theta_release is not None:
            out["release"] = (ok & (a < -self.theta_release)).float()
        return out


class ConstantSource(Source):
    def __init__(self, c: float = 1.0, name: str = "constant") -> None:
        if not 0.0 <= c <= 1.0:
            raise ValueError("c must be in [0, 1]")
        self.c, self.name = float(c), name

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        return {"s": torch.full(ctx.mask.shape, self.c, device=ctx.mask.device)}


def _index(ctx: StepContext, index: str) -> torch.Tensor:
    if index == "step":
        return ctx.step.long()
    if index == "position":
        return ctx.idx.long().clamp_min(0)
    raise ValueError("index must be 'step' or 'position'")


class YokedSource(Source):
    """Replay a donor r trajectory (B, N); ``lengths`` (B,) valid entries per row (cycled), default N."""

    final = True

    def __init__(self, donor: torch.Tensor, lengths: torch.Tensor | None = None, index: str = "step",
                 name: str = "yoked") -> None:
        donor = donor.detach().float()
        if donor.dim() != 2 or donor.shape[1] == 0:
            raise ValueError("donor must be (B, N) with N >= 1")
        if lengths is None:
            lengths = torch.full((donor.shape[0],), donor.shape[1], dtype=torch.long)
        lengths = lengths.long()
        if (lengths < 1).any() or (lengths > donor.shape[1]).any():
            raise ValueError("lengths must be in [1, N]")
        if not ((donor[torch.arange(donor.shape[1])[None] < lengths[:, None]] >= 0).all()
                and (donor[torch.arange(donor.shape[1])[None] < lengths[:, None]] <= 1).all()):
            raise ValueError("donor values must lie in [0, 1]")
        self.donor, self.lengths, self.index, self.name = donor, lengths, index, name

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        if ctx.mask.shape[0] != self.donor.shape[0]:
            raise ValueError(f"donor has {self.donor.shape[0]} rows, the batch {ctx.mask.shape[0]}")
        donor, n = self.donor.to(ctx.mask.device), self.lengths.to(ctx.mask.device)
        k = _index(ctx, self.index) % n[:, None]
        return {"s": donor.gather(1, k)}


class PositionMeanSource(Source):
    """r_t = table[min(k, N − 1)], a per-position mean gain trajectory measured on other documents; a (B, N)
    table holds one trajectory per row (row b reads table[b, min(k, N − 1)])."""

    final = True

    def __init__(self, table: torch.Tensor, index: str = "position", name: str = "position_mean") -> None:
        table = table.detach().float()
        if table.dim() != 2:
            table = table.flatten()
        if table.numel() == 0 or table.shape[-1] == 0 or (table < 0).any() or (table > 1).any():
            raise ValueError("table must be non-empty, (N,) or (B, N), with values in [0, 1]")
        self.table, self.index, self.name = table, index, name

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        t = self.table.to(ctx.mask.device)
        k = _index(ctx, self.index).clamp(0, t.shape[-1] - 1)
        if t.dim() == 1:
            return {"s": t[k]}
        if t.shape[0] != k.shape[0]:
            raise ValueError(f"table has {t.shape[0]} rows, the batch {k.shape[0]}")
        return {"s": t.gather(1, k)}


def derange(r: torch.Tensor, lengths: torch.Tensor | None = None, shift: int = 1):
    """Donor rows for a yoked arm: row b gets the trajectory of row (b + shift) mod B (B ≥ 2)."""
    B = r.shape[0]
    if B < 2 or shift % B == 0:
        raise ValueError("a derangement needs B >= 2 and shift not a multiple of B")
    perm = (torch.arange(B) + shift) % B
    return r[perm], (None if lengths is None else lengths[perm])


SOURCES = {"agree": AgreeSource, "constant": ConstantSource}


def make_source(spec: "str | Source", **kw) -> Source:
    """A source from a name in ``SOURCES`` (with keyword arguments) or an instance (returned as is)."""
    if isinstance(spec, str):
        if spec not in SOURCES:
            raise ValueError(f"unknown source {spec!r}; build controls (yoked, position_mean) or tap sources "
                             f"as instances. Known names: {sorted(SOURCES)}")
        return SOURCES[spec](**kw)
    return spec
