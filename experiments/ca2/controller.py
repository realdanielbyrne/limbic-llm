"""CA2Controller: a training-free, dynamic-gain side channel that injects along the stream's own gist.

At the injection site (the input of block M; by default M = L + 1, i.e. block L's output)

    h_t ← h_t + g_t ‖h_t‖ d̂_t,          g_t = g_min + (g_max − g_min) · r_t,   r_t ∈ [0, 1],

with nothing trained: zero parameters, buffers and plain state only, float32 whatever the model dtype.

Gain pipeline (``GainPipeline``), per step, every recurrence a ``regulators.leaky_scan`` (same arithmetic
in prefill and decode):
  1. raw score s_t ∈ [0, 1] from the driving source (``sources.py``);
  2. polarity: ``reinforce`` p = s (Daniel's "momentum": more gain with more agreement); ``restore``
     p = 1 − s, or σ(κ_p(c* − s)) with ``restore_kappa`` (inject more the further the state is from target);
  3. smoothing over steered positions: ``none`` r̃ = p; ``ema`` r̃_t = Σ λ^{t−u} p_u / Σ λ^{t−u} (bias
     corrected, stays in [0, 1]); ``integral`` E_t = λE_{t−1} + p_t, r̃ = tanh(κE_t) (steering's graded law);
  4. release (R3): on a release event the carried state is multiplied by ρ_r before p_t is added;
  5. regulators: r_t = r̃_t · ρ_t, ρ_t = exp(−γR_t) the repetition regulator (R6, ``ngram`` or ``hidden``);
  6. g_t = g_min + (g_max − g_min) r_t at steered positions, 0 at padding, at the sink and before the
     steer point. 0 ≤ g_min ≤ g_max ≤ G_abs is asserted. ``gain_override`` (B,) replaces g by a fixed
     per-row gain (the fixed-at-matched-mean control; the law still runs and is traced as ``g_law``);
  7. accumulated-dose cap (optional): g'_t = min(g_t, C − λ_d D_{t−1}), D_t = λ_d D_{t−1} + g'_t; asserted
     feasible (g_min ≤ (1 − λ_d)C), so g stays in [g_min, g_max].
A ``final`` source (the yoked and position-mean controls) supplies r_t directly and skips 2–5.

Steer point (``steer_from``): ``last_prompt`` (default; the last column of the first call, then every
later call: run.generate's prefill + decode; N1: steering the prompt fails on chat models), ``generated``
(decode calls only), ``all`` (every real non-sink token), an int k or a (B,) tensor of per-row k (real-token
index ≥ k, index 0 being the row's first real token, i.e. the sink when present; use this for single-pass
teacher-forced runs, where ``last_prompt`` would steer one token).

Directions (all unit, matched norm; ``dim_keep`` zeroes massive dimensions; ``error_only`` then removes
the component along h, d ← normalise(d − (d·ĥ)ĥ)):
  ``gist``         normalise(A_fast − A_slow), the bank's default bands (= ``SteeringController`` gist);
  ``anti_gist``    −gist (sign control; same gain trajectory under teacher forcing);
  ``mean``         normalise(A_slow);
  ``random``       a fixed seeded unit vector (as ``SteeringController`` random with the same seed);
  ``self``         h_t/‖h_t‖, a pure norm rescale;
  ``foreign``      another row's gist at the same column (row b reads row (b + 1) mod B; B ≥ 2);
  ``window_gist``  the gist of a second bank that restarts at every ``window``-token block (no
                   cross-window content; aligned like ``pretrained_loop`` block mode).

Controller contract of ``experiments/alignment/run.py`` (``generate`` and ``perplexity`` drive it
unchanged): ``reset()``, ``record``, ``pending_mask`` (B, T) for the next prefill, ``pending_decision``
(accepted, unused), ``trace[i]`` a dict per forward call with ``"g"`` (effective gain g·‖d̂‖), ``"s"``
(driving score) and ``"E"`` (smoother state) of shape (B, T_call), and ``summary(gen_ids, eos_ids)``
returning gain_mean, gain_max, gain_prompt_mean, capped_frac, E_final, s_prompt, s_gen_mean, s_gen,
gen_mask (plus gain_sd, frac_at_gmin, frac_at_gmax, n_resp). ``forward(h)`` applies the whole step to
one tensor, so ``steering.Attachment`` or ``common.SteerSide`` can host it too.

Per-token trace (``record = True``; CPU tensors, one dict per forward call, model-column aligned).
Every signal is recorded in every arm, whether or not it drives the gain:
  gain       g (effective), g_raw (before ‖d̂‖), g_law (the law's gain before override and dose cap),
             s, p, E, r_smooth, r, rho (regulator factor), release, dose;
  injection  inj_ratio ‖Δh‖/‖h‖, dnorm ‖d̂‖, cos_dh cos(d̂, h) before the error projection, cos_dh_out
             after it, cos_d_gist cos(d̂, gist), dcos_lag{k} cos(d̂_t, d̂_{t−k}) for k in ``lags``
             (NaN where undefined), hnorm ‖h‖ at the tap, xnorm ‖h‖ at the injection site;
  sources    src/<name>/<key> for every registered source (the ``agree`` source is always registered);
  regulators rep_ngram, rep_hidden (both always computed), rep_R;
  bookkeeping mask, use, steer, idx (real-token index), step (steered-step index), col (column), ids,
             massive (flag);
  taps       tap/<name>/<key> for every registered tap (see below).
``trace_tensors()`` concatenates the calls into (B, T_total, ...) tensors.

Extension points (taps). A tap is any object with the ``Tap`` interface; the attention tap
(``experiments/ca2/attention.py``) and the GDN tap (``experiments/ca2/gdn.py``) plug in without editing
this file:
  registration  ``CA2Attachment(model, ctrl, ..., taps=[tap])`` (or ``SteerSide(..., taps=[tap])``) calls
                ``ctrl.add_tap(tap)`` (keyed by ``tap.name``, which must be unique) and then
                ``tap.attach(att)``, where the tap registers its own torch hooks on ``att.model`` /
                ``att.blocks`` (``att.kinds`` gives 'attention' / 'linear' per block; ``att.tap``,
                ``att.inject_before`` and ``att.relevance_layer`` are the layer indices) and keeps the
                handles; ``att.remove()`` calls ``tap.remove()``.
  call cycle    a forward hook on the input embeddings opens each model call: ``ctrl.begin_call(ids)``,
                which calls ``tap.begin_call(ctrl)`` on every tap (clear per-call buffers there); a forward
                hook on the model closes it: ``ctrl.end_call()``. ``ctrl.reset()`` calls ``tap.reset()``
                (new sequence: drop carried state, e.g. a cache read or running statistics).
  sources       a source that reads a tap names it in ``needs`` and reads ``ctx.taps[name]`` during the
                step. The step runs at the injection site (block M's input), so taps at any layer < M
                have run for this call; ``relevance_layer`` must be < M (asserted).
  gain consumers a hook that runs after the step in the same call (any layer ≥ M, e.g. a GDN branch-gain
                law on ``out_proj``) reads ``ctrl.current``: this call's g, g_eff, r, s, steer, mask, d
                (device tensors, cleared at the next ``begin_call``). A law that needs its own gain below M
                instantiates its own ``GainPipeline`` (same arithmetic, its own state).
  trace         per-token signals enter the trace in two ways: ``tap.signals()`` returns a dict of
                (B, T) or (B, T, K) tensors in model coordinates, collected at ``end_call`` as
                ``tap/<name>/<key>``; or ``ctrl.log(key, tensor)`` at any time during the call (use a
                ``tap/<name>/`` prefix). Both slice off ``ctrl.col_offset`` leading columns (the sink under
                ``SteerSide``) and move to CPU. Nothing is recorded unless ``ctrl.record``.
  fixed state   ``tap.state_numel()`` counts carried state (the fixed-size check sums it).
``NormTap`` below is a minimal working tap (residual norm at every block output, the downstream-norm
measure of the plan) and doubles as the reference implementation.

Placement (``CA2Attachment(tap=L, inject_before=M, relevance_layer=R)``): the gist is read at block L's
output (pre-injection); the injection is a forward pre-hook on block M (M > L + 1), or, for M = L + 1,
it rewrites block L's output in the same hook (identical to ``steering.Attachment``). A forward hook on
the input embeddings sees ``input_ids`` in prefill and in decode (ids for the n-gram regulator, release
ids and the trace). With g = 0 at every position of a call the hooks return None, so the model output
is byte-identical to the base model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402
from ca2.bank import BankStep, GistBank  # noqa: E402
from ca2.regulators import (HiddenRepeat, NgramRepeat, RepetitionRegulator, dose_cap_scan, leaky_scan,  # noqa: E402
                            project_error, release_events)
from ca2.sources import Source, StepContext, make_source  # noqa: E402

DIRECTIONS = ("gist", "anti_gist", "mean", "random", "self", "foreign", "window_gist")
SMOOTH = ("none", "ema", "integral")
POLARITY = ("reinforce", "restore")
STEER_FROM = ("last_prompt", "generated", "all")
REGULATORS = ("none", "ngram", "hidden")


# ------------------------------------------------------------------------------------------- taps
class Tap:
    """Base class and interface of a tap (see the module docstring, "Extension points")."""

    name = "tap"

    def attach(self, att: "CA2Attachment") -> None:
        pass

    def remove(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def begin_call(self, ctrl: "CA2Controller") -> None:
        pass

    def signals(self) -> dict[str, torch.Tensor]:
        return {}

    def state_numel(self) -> int:
        return 0


class NormTap(Tap):
    """‖h‖ at every block output (after any injection there), per token: signal ``norm`` (B, T, n_blocks)."""

    name = "norms"

    def __init__(self) -> None:
        self.handles: list = []
        self.buf: dict[int, torch.Tensor] = {}
        self.n = 0

    def attach(self, att: "CA2Attachment") -> None:
        self.n = len(att.blocks)
        for i, b in enumerate(att.blocks):
            self.handles.append(b.register_forward_hook(self._hook(i)))

    def _hook(self, i: int):
        def fn(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            self.buf[i] = h.detach().float().norm(dim=-1)
        return fn

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def begin_call(self, ctrl: "CA2Controller") -> None:
        self.buf = {}

    def signals(self) -> dict[str, torch.Tensor]:
        if len(self.buf) != self.n or not self.n:
            return {}
        return {"norm": torch.stack([self.buf[i] for i in range(self.n)], -1)}


# ---------------------------------------------------------------------------------- gain pipeline
class GainPipeline:
    """Score → polarity → smoothing (with release) → regulators → g = g_min + (g_max − g_min) r. Stateful."""

    def __init__(self, g_min: float = 0.0, g_max: float = 0.2, g_abs: float = 0.3, smooth: str = "ema",
                 leak: float = 0.9, kappa: float = 0.5, polarity: str = "reinforce",
                 restore_kappa: float | None = None, restore_target: float = 0.5, release_rho: float = 1.0,
                 signed: bool = False) -> None:
        lo = -g_abs if signed else 0.0
        if not lo <= g_min <= g_max <= g_abs:
            raise ValueError(f"need {lo} <= g_min <= g_max <= g_abs, got {g_min}, {g_max}, {g_abs}")
        if smooth not in SMOOTH or polarity not in POLARITY:
            raise ValueError(f"smooth in {SMOOTH}, polarity in {POLARITY}")
        if not 0.0 <= leak < 1.0 or not 0.0 <= release_rho <= 1.0 or kappa <= 0:
            raise ValueError("need leak in [0, 1), release_rho in [0, 1], kappa > 0")
        self.g_min, self.g_max, self.g_abs = float(g_min), float(g_max), float(g_abs)
        self.smooth, self.leak, self.kappa, self.polarity = smooth, float(leak), float(kappa), polarity
        self.restore_kappa, self.restore_target = restore_kappa, float(restore_target)
        self.release_rho = float(release_rho)
        self.reset()

    def reset(self) -> None:
        self.num: torch.Tensor | None = None             # (B,) ema numerator or integral
        self.den: torch.Tensor | None = None             # (B,) ema denominator

    def state_numel(self) -> int:
        return sum(0 if t is None else t.numel() for t in (self.num, self.den))

    def __call__(self, s: torch.Tensor, steer: torch.Tensor, release: torch.Tensor | None = None,
                 rho: torch.Tensor | None = None, final: bool = False) -> dict[str, torch.Tensor]:
        s, steer = s.float(), steer.float()
        if final:
            p = E = rs = s
            r = s.clamp(0.0, 1.0)
        else:
            if self.polarity == "reinforce":
                p = s
            elif self.restore_kappa is None:
                p = 1.0 - s
            else:
                p = torch.sigmoid(self.restore_kappa * (self.restore_target - s))
            coef = torch.where(steer > 0, torch.full_like(s, self.leak), torch.ones_like(s))
            if release is not None:
                coef = coef * torch.where(release.bool() & (steer > 0), torch.full_like(s, self.release_rho),
                                          torch.ones_like(s))
            x = p * steer
            if self.smooth == "none":
                E = rs = p
            elif self.smooth == "ema":
                s0 = None if self.num is None else torch.stack([self.num, self.den], -1)
                nd = leaky_scan(coef.unsqueeze(-1), torch.stack([x, steer], -1), s0)   # one loop for both
                num, den = nd[..., 0], nd[..., 1]
                self.num, self.den = num[:, -1].detach(), den[:, -1].detach()
                E = rs = torch.where(den > 0, num / den.clamp_min(1e-30), torch.zeros_like(num))
            else:
                E = leaky_scan(coef, x, self.num)
                self.num = E[:, -1].detach()
                rs = torch.tanh(self.kappa * E)
            r = rs if rho is None else rs * rho
            r = r.clamp(0.0, 1.0)
        g = (self.g_min + (self.g_max - self.g_min) * r) * steer
        return {"p": p, "E": E, "r_smooth": rs, "r": r, "g": g}


# ---------------------------------------------------------------------------------------- controller
class CA2Controller(nn.Module):
    """Dynamic-gain CA2 injection (module docstring). Call ``reset()`` between sequences."""

    def __init__(self, d_model: int, source: "str | Source" = "constant", g_min: float = 0.0, g_max: float = 0.2,
                 g_abs: float = 0.3, smooth: str = "ema", leak: float = 0.9, kappa: float = 0.5,
                 polarity: str = "reinforce", restore_kappa: float | None = None, restore_target: float = 0.5,
                 release_rho: float = 1.0, release_ids: Sequence[int] = (), direction: str = "gist",
                 betas: tuple[float, ...] = (0.9, 0.995), sink_skip: int = 1, exclude_massive: bool = False,
                 massive_ratio: float = 10.0, massive_window: int = 64, dim_keep: torch.Tensor | None = None,
                 error_only: bool = False, steer_from: "str | int | torch.Tensor" = "last_prompt",
                 gain_override: torch.Tensor | None = None, regulator: str = "none", rep_gamma: float = 1.0,
                 rep_leak: float = 0.9, rep_n: int = 4, rep_window: int = 32, rep_lags: int = 8,
                 dose_cap: float | None = None, dose_leak: float = 0.99, window: int | None = None,
                 lags: tuple[int, ...] = (1, 4, 16, 64), seed: int = 0, foreign_shift: int = 1,
                 sources: Iterable["str | Source"] = ("agree",)) -> None:
        super().__init__()
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        if regulator not in REGULATORS:
            raise ValueError(f"regulator must be one of {REGULATORS}")
        if isinstance(steer_from, str) and steer_from not in STEER_FROM:
            raise ValueError(f"steer_from must be one of {STEER_FROM}, an int or a (B,) tensor")
        if direction == "window_gist" and window is None:
            raise ValueError("window_gist needs window=W")
        if dose_cap is not None and not (0.0 <= dose_leak < 1.0 and g_min <= (1.0 - dose_leak) * dose_cap):
            raise ValueError("the dose cap is infeasible: need dose_leak in [0, 1) and g_min <= (1 - dose_leak) * cap")
        if gain_override is not None and ((gain_override < 0).any() or (gain_override > g_abs).any()):
            raise ValueError("gain_override must lie in [0, g_abs]")
        self.d_model = int(d_model)
        self.pipeline = GainPipeline(g_min, g_max, g_abs, smooth, leak, kappa, polarity, restore_kappa,
                                     restore_target, release_rho)
        self.g_min, self.g_max, self.g_abs = self.pipeline.g_min, self.pipeline.g_max, self.pipeline.g_abs
        self.direction, self.error_only, self.steer_from = direction, bool(error_only), steer_from
        self.release_ids = tuple(int(i) for i in release_ids)
        self.regulator, self.lags, self.foreign_shift = regulator, tuple(int(k) for k in lags), int(foreign_shift)
        self.dose_cap, self.dose_leak = dose_cap, float(dose_leak)
        self.bank = GistBank(betas, sink_skip, exclude_massive, massive_ratio, massive_window, dim_keep)
        self.wbank = (GistBank(betas, sink_skip, exclude_massive, massive_ratio, massive_window, dim_keep,
                               window=window) if direction == "window_gist" else None)
        self.sink_skip = int(sink_skip)
        self.ngram, self.hrep = NgramRepeat(rep_n, rep_window), HiddenRepeat(rep_lags)
        self.rep_reg = RepetitionRegulator(rep_gamma, rep_leak)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("d_rand", F.normalize(torch.randn(self.d_model, generator=g), dim=0))
        self.register_buffer("dim_keep", None if dim_keep is None else dim_keep.detach().float().flatten())
        self.gain_override = None if gain_override is None else gain_override.detach().float()
        self.sources: dict[str, Source] = {}
        self.taps: dict[str, Tap] = {}
        drv = make_source(source)
        for s in sources:
            s = make_source(s)
            if s.name != drv.name:
                self.add_source(s)
        self.add_source(drv, drive=True)
        self.col_offset = 0
        self.record = False
        self.reset()

    # ------------------------------------------------------------------ registration
    def add_source(self, source: "str | Source", drive: bool = False) -> Source:
        source = make_source(source)
        if source.name in self.sources:
            raise ValueError(f"a source named {source.name!r} is already registered")
        self.sources[source.name] = source
        if drive:
            self.drive = source.name
        return source

    def add_tap(self, tap: Tap) -> Tap:
        if tap.name in self.taps and self.taps[tap.name] is not tap:
            raise ValueError(f"a tap named {tap.name!r} is already registered")
        self.taps[tap.name] = tap
        return tap

    # ------------------------------------------------------------------------- state
    def reset(self) -> None:
        self.bank.reset()
        if self.wbank is not None:
            self.wbank.reset()
        self.pipeline.reset()
        self.ngram.reset()
        self.hrep.reset()
        self.rep_reg.reset()
        for s in self.sources.values():
            s.reset()
        for t in self.taps.values():
            t.reset()
        self.n_real: torch.Tensor | None = None          # (B,) real tokens seen
        self.n_steer: torch.Tensor | None = None         # (B,) steered tokens seen
        self.n_col = 0                                   # columns seen
        self.dose: torch.Tensor | None = None            # (B,) accumulated dose
        self.d_ring: torch.Tensor | None = None          # (B, max lag, D) earlier directions
        self.d_ring_ok: torch.Tensor | None = None       # (B, max lag)
        self.pending_mask: torch.Tensor | None = None    # (B, T) padding mask of the next (prefill) call
        self.pending_decision: torch.Tensor | None = None  # accepted for run.generate; unused
        self.prefill_mask: torch.Tensor | None = None
        self.calls = 0
        self.trace: list[dict] = []
        self.current: dict[str, torch.Tensor] = {}
        self._open = False
        self._stepped = False
        self._ids: torch.Tensor | None = None
        self._pos: dict | None = None
        self._obs: tuple | None = None
        self._rec: dict | None = None
        self._checked = False

    def state_numel(self) -> int:
        """Carried state size (excluding the trace); constant once the batch is fixed."""
        n = self.bank.state_numel() + (self.wbank.state_numel() if self.wbank is not None else 0)
        n += self.pipeline.state_numel() + self.ngram.state_numel() + self.hrep.state_numel()
        n += self.rep_reg.state_numel()
        n += sum(s.state_numel() for s in self.sources.values()) + sum(t.state_numel() for t in self.taps.values())
        for t in (self.n_real, self.n_steer, self.dose, self.d_ring, self.d_ring_ok):
            n += 0 if t is None else t.numel()
        return n

    # ---------------------------------------------------------------------- call cycle
    def begin_call(self, ids: torch.Tensor | None = None) -> None:
        """Open a forward call (the ids hook calls this); closes the previous one."""
        if self._open:
            self.end_call()
        self._open, self._stepped = True, False
        self._ids = None if ids is None else ids.detach().long()
        self._pos, self._obs, self.current = None, None, {}
        self._rec = {}
        if self.record:
            self.trace.append(self._rec)
        for t in self.taps.values():
            t.begin_call(self)

    def end_call(self) -> None:
        """Close the call: collect tap signals into the trace."""
        if not self._open:
            return
        if self.record and self._rec is not None:
            for t in self.taps.values():
                for k, v in t.signals().items():
                    self._rec[f"tap/{t.name}/{k}"] = self._cpu(v)
        self._open = False

    def log(self, key: str, value: torch.Tensor) -> None:
        """Record a per-token signal (B, T_model, ...) of the current call in the trace under ``key``."""
        if self.record and self._rec is not None:
            self._rec[key] = self._cpu(value)

    def _cpu(self, v: torch.Tensor) -> torch.Tensor:
        v = v.detach()
        if self.col_offset and v.dim() >= 2:
            v = v[:, self.col_offset:]
        return v.cpu()

    def _positions(self, B: int, T: int, dev) -> dict:
        if self._pos is not None:
            return self._pos
        first = self.calls == 0
        mask = self.pending_mask
        self.pending_mask = None
        mask = torch.ones(B, T, device=dev) if mask is None else mask.to(dev).float()[:, -T:]
        n_prev = self.n_real if self.n_real is not None else torch.zeros(B, device=dev)
        idx = n_prev[:, None] + mask.cumsum(1) - 1
        real = mask * (idx >= self.sink_skip).float()
        sf = self.steer_from
        if isinstance(sf, str):
            if sf == "all" or (sf == "last_prompt" and not (first and T > 1)) or (sf == "generated" and not first):
                rule = torch.ones(B, T, device=dev)
            elif sf == "last_prompt":
                rule = torch.zeros(B, T, device=dev)
                rule[:, -1] = 1.0
            else:
                rule = torch.zeros(B, T, device=dev)
        else:
            k = torch.as_tensor(sf, device=dev, dtype=torch.float32)
            rule = (idx >= (k[:, None] if k.dim() == 1 else k)).float()
        steer = real * rule
        ns = self.n_steer if self.n_steer is not None else torch.zeros(B, device=dev)
        step = (ns[:, None] + steer.cumsum(1) - steer).long()
        col = torch.arange(self.n_col, self.n_col + T, device=dev)
        self._pos = {"first": first, "mask": mask, "idx": torch.where(mask > 0, idx, torch.full_like(idx, -1.0)),
                     "real": real, "steer": steer, "step": step, "col": col}
        return self._pos

    # --------------------------------------------------------------------------- step
    def observe(self, h: torch.Tensor) -> None:
        """Read the tap stream (block L output, pre-injection) for this call."""
        if not self._open or self._stepped:
            self.begin_call(None)
        if self._obs is not None:
            raise RuntimeError("observe() twice in one call")
        h32 = h.detach().float()
        B, T, _ = h32.shape
        if self.d_rand.device != h32.device:
            self.to(h32.device)
        pos = self._positions(B, T, h32.device)
        if pos["first"]:
            self.prefill_mask = pos["mask"].detach().cpu()
        bstep = self.bank.update(h32, pos["mask"])
        wstep = self.wbank.update(h32, pos["mask"]) if self.wbank is not None else None
        self._obs = (h, h32, bstep, wstep)

    def inject(self, x: torch.Tensor) -> torch.Tensor | None:
        """Apply the step at the injection site; returns the new stream, or None when g = 0 throughout."""
        if self._obs is None:
            raise RuntimeError("inject() before observe() in this call")
        h_src, h32, bstep, wstep = self._obs
        self._obs = None
        x32 = h32 if x is h_src else x.detach().float()
        if x32.shape != h32.shape:
            raise ValueError(f"injection-site shape {tuple(x32.shape)} differs from the tap's {tuple(h32.shape)}")
        B, T, D = x32.shape
        dev = x32.device
        pos = self._positions(B, T, dev)
        mask, steer, real = pos["mask"], pos["steer"], pos["real"]
        ids = self._ids if self._ids is not None and tuple(self._ids.shape) == (B, T) else None
        ctx = StepContext(h=h32, x=x32, mask=mask, use=bstep.use, steer=steer, idx=pos["idx"], step=pos["step"],
                          col=pos["col"], first=pos["first"], call=self.calls,
                          ids=None if ids is None else ids.to(dev), bank=bstep, taps=self.taps,
                          col_offset=self.col_offset, ctrl=self)
        if not self._checked:
            for s in self.sources.values():
                missing = [n for n in getattr(s, "needs", ()) if n not in self.taps]
                if missing:
                    raise KeyError(f"source {s.name!r} needs taps {missing}; attach them first")
            self._checked = True
        outs = {}
        for name, src in self.sources.items():
            o = src(ctx)
            if "s" not in o or tuple(o["s"].shape) != (B, T):
                raise ValueError(f"source {name!r} must return 's' of shape {(B, T)}")
            outs[name] = o
        drv = outs[self.drive]
        s = drv["s"].float()
        release = release_events(ctx.ids, self.release_ids, drv.get("release"), (B, T), dev)
        rep_ngram = self.ngram(ctx.ids, mask) if ctx.ids is not None else torch.zeros(B, T, device=dev)
        rep_hidden = self.hrep(h32, bstep.use)
        rho = R = None
        if self.regulator != "none":
            R, rho = self.rep_reg(rep_ngram if self.regulator == "ngram" else rep_hidden, real)
        pipe = self.pipeline(s, steer, release, rho, final=self.sources[self.drive].final)
        g_law = pipe["g"]
        g = g_law if self.gain_override is None else self.gain_override.to(dev)[:, None] * steer
        dose = None
        if self.dose_cap is not None:
            g, dose = dose_cap_scan(g, real, self.dose_leak, self.dose_cap, self.dose)
            self.dose = dose[:, -1].detach()
        # direction
        gist = bstep.gist(0, -1)
        d = self._direction(x32, bstep, wstep, gist)
        if self.dim_keep is not None:
            d = F.normalize(d * self.dim_keep, dim=-1)
        dfull = d.expand(B, T, D) if d.dim() == 1 else d
        cos_dh = F.cosine_similarity(dfull, x32, dim=-1, eps=1e-12)
        if self.error_only:
            d, _ = project_error(d, x32)
            dfull = d
        dnorm = dfull.norm(dim=-1)
        out = x32 + g.unsqueeze(-1) * x32.norm(dim=-1, keepdim=True) * d
        g_eff = g * dnorm
        self.current = {"g": g, "g_eff": g_eff, "r": pipe["r"], "s": s, "steer": steer, "mask": mask, "d": dfull}
        if self.record:
            self._trace_step(pos, outs, pipe, g, g_law, g_eff, s, release, rho, R, rep_ngram, rep_hidden, dose,
                             dfull, dnorm, cos_dh, gist, h32, x32, bstep, ids)
        self._advance(pos, dfull, dnorm)
        self.calls += 1
        self._stepped = True
        if not bool((g != 0).any()):
            return None
        return out.to(x.dtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """The whole step on one tensor (tap = injection site); returns h unchanged when g = 0."""
        self.observe(h)
        out = self.inject(h)
        return h if out is None else out

    def _direction(self, x32, bstep: BankStep, wstep: BankStep | None, gist: torch.Tensor) -> torch.Tensor:
        m = self.direction
        if m == "random":
            return self.d_rand
        if m == "self":
            return F.normalize(x32, dim=-1)
        if m == "gist":
            return gist
        if m == "anti_gist":
            return -gist
        if m == "mean":
            return F.normalize(bstep.avg(-1), dim=-1)
        if m == "foreign":
            if gist.shape[0] < 2:
                raise ValueError("the foreign direction needs at least two rows")
            return gist.roll(-self.foreign_shift, 0)
        return wstep.gist(0, -1)

    def _advance(self, pos: dict, dfull: torch.Tensor, dnorm: torch.Tensor) -> None:
        mask = pos["mask"]
        zero = torch.zeros_like(mask[:, 0])
        self.n_real = ((self.n_real if self.n_real is not None else zero) + mask.sum(1)).detach()
        self.n_steer = ((self.n_steer if self.n_steer is not None else zero) + pos["steer"].sum(1)).detach()
        self.n_col += mask.shape[1]
        if self.lags:
            K = max(self.lags)
            B, T, D = dfull.shape
            ring = self.d_ring if self.d_ring is not None else dfull.new_zeros(B, K, D)
            rok = (self.d_ring_ok if self.d_ring_ok is not None
                   else torch.zeros(B, K, dtype=torch.bool, device=dfull.device))
            self.d_ring = torch.cat([ring, dfull.detach()], 1)[:, -K:]
            self.d_ring_ok = torch.cat([rok, (dnorm > 0) & (mask > 0)], 1)[:, -K:]

    def _trace_step(self, pos, outs, pipe, g, g_law, g_eff, s, release, rho, R, rep_ngram, rep_hidden, dose,
                    dfull, dnorm, cos_dh, gist, h32, x32, bstep, ids) -> None:
        B, T, D = dfull.shape
        nan = torch.full((B, T), float("nan"), device=g.device)
        rec = {"g": g_eff, "g_raw": g, "g_law": g_law, "s": s, "p": pipe["p"], "E": pipe["E"],
               "r_smooth": pipe["r_smooth"], "r": pipe["r"], "rho": torch.ones_like(s) if rho is None else rho,
               "rep_R": torch.zeros_like(s) if R is None else R, "release": release.float(),
               "dose": nan if dose is None else dose, "rep_ngram": rep_ngram, "rep_hidden": rep_hidden,
               "inj_ratio": g.abs() * dnorm, "dnorm": dnorm, "cos_dh": cos_dh,
               "cos_dh_out": F.cosine_similarity(dfull, x32, dim=-1, eps=1e-12),
               "hnorm": h32.norm(dim=-1), "xnorm": x32.norm(dim=-1), "massive": bstep.massive.float(),
               "mask": pos["mask"], "use": bstep.use, "steer": pos["steer"], "idx": pos["idx"],
               "step": pos["step"], "col": pos["col"][None].expand(B, T),
               "ids": ids if ids is not None else torch.full((B, T), -1, dtype=torch.long, device=g.device)}
        gok = gist.norm(dim=-1) > 0
        cg = F.cosine_similarity(dfull, gist, dim=-1, eps=1e-12)
        rec["cos_d_gist"] = torch.where(gok & (dnorm > 0), cg, nan)
        if self.lags:
            K = max(self.lags)
            ring = self.d_ring if self.d_ring is not None else dfull.new_zeros(B, K, D)
            rok = self.d_ring_ok if self.d_ring_ok is not None else torch.zeros(B, K, dtype=torch.bool, device=g.device)
            seq = torch.cat([ring, dfull], 1)
            ok = torch.cat([rok, (dnorm > 0) & (pos["mask"] > 0)], 1)
            for k in self.lags:
                c = F.cosine_similarity(dfull, seq[:, K - k: K - k + T], dim=-1, eps=1e-12)
                rec[f"dcos_lag{k}"] = torch.where(ok[:, K - k: K - k + T] & ok[:, K:], c, nan)
        for name, o in outs.items():
            for k, v in o.items():
                rec[f"src/{name}/{k}"] = v.float()
        flat = [k for k, v in rec.items() if v.is_floating_point() and tuple(v.shape) == (B, T)]
        if flat:                                          # one device-to-host copy for all (B, T) float signals
            host = torch.stack([rec[k].detach().float() for k in flat]).cpu()
            for i, k in enumerate(flat):
                self._rec[k] = host[i]
        for k, v in rec.items():
            if k not in self._rec:
                self._rec[k] = v.detach().cpu()

    # ------------------------------------------------------------------------ reports
    def trace_tensors(self) -> dict[str, torch.Tensor]:
        """Concatenate the per-call trace along columns: key -> (B, T_total, ...); missing calls filled with
        NaN (float) or −1 (integer)."""
        if not self.trace:
            return {}
        keys: dict[str, torch.Tensor] = {}
        for rec in self.trace:
            for k, v in rec.items():
                keys.setdefault(k, v)
        widths = []
        for rec in self.trace:
            ref = rec.get("g", next(iter(rec.values()), None))
            widths.append(None if ref is None else ref.shape[1])
        out = {}
        for k, proto in keys.items():
            parts = []
            for rec, w in zip(self.trace, widths):
                if k in rec:
                    parts.append(rec[k])
                elif w is not None:
                    fill = -1 if not proto.is_floating_point() else float("nan")
                    parts.append(torch.full((proto.shape[0], w) + tuple(proto.shape[2:]), fill, dtype=proto.dtype))
            try:
                out[k] = torch.cat(parts, 1)
            except RuntimeError:
                continue                                  # inconsistent shapes across calls: left out
        return out

    def summary(self, gen_ids: torch.Tensor | None = None, eos_ids=None) -> dict:
        """Per-sequence statistics of the last generation over real response positions (as ``SteeringController``):
        the last prompt token plus each generated token fed back before the first EOS."""
        recs = [r for r in self.trace if "g" in r]
        if not recs:
            return {}
        pre, dec = recs[0], recs[1:]
        B = pre["g"].shape[0]
        pm = self.prefill_mask if self.prefill_mask is not None else torch.ones_like(pre["g"])
        if dec:
            s_dec, g_dec, E_dec, gr_dec = (torch.cat([t[k] for t in dec], 1) for k in ("s", "g", "E", "g_raw"))
        else:
            s_dec = g_dec = E_dec = gr_dec = torch.zeros(B, 0)
        gen_mask = torch.ones_like(s_dec)
        if gen_ids is not None and eos_ids is not None and s_dec.shape[1]:
            eos = torch.tensor(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
            hit = torch.isin(gen_ids[:, : s_dec.shape[1]].cpu(), eos)
            gen_mask = (hit.long().cumsum(1) == 0).float()
        rmask = torch.cat([torch.ones(B, 1), gen_mask], 1)
        g = torch.cat([pre["g"][:, -1:], g_dec], 1) * rmask
        graw = torch.cat([pre["g_raw"][:, -1:], gr_dec], 1)
        E = torch.cat([pre["E"][:, -1:], E_dec], 1)
        n = rmask.sum(1)
        last = (n - 1).long()
        n_gen = gen_mask.sum(1)
        mean = g.sum(1) / n
        sd = (((g - mean[:, None]) ** 2 * rmask).sum(1) / n).sqrt()
        tol = 1e-6 * max(1.0, self.g_max)
        varies = self.g_max > self.g_min
        at_max = ((graw >= self.g_max - tol).float() * rmask).sum(1) / n
        at_min = ((graw <= self.g_min + tol).float() * rmask).sum(1) / n
        nanv = torch.full((B,), float("nan"))
        s_gen_mean = torch.where(n_gen > 0, (s_dec * gen_mask).sum(1) / n_gen.clamp_min(1), nanv)
        return {"gain_mean": mean, "gain_max": g.max(1).values, "gain_sd": sd,
                "capped_frac": at_max if varies else nanv, "frac_at_gmax": at_max if varies else nanv,
                "frac_at_gmin": at_min if varies else nanv,
                "gain_prompt_mean": (pre["g"] * pm).sum(1) / pm.sum(1).clamp_min(1),
                "E_final": E.gather(1, last[:, None])[:, 0], "s_prompt": pre["s"][:, -1], "s_gen_mean": s_gen_mean,
                "s_gen": s_dec, "gen_mask": gen_mask, "n_resp": n}


# ---------------------------------------------------------------------------------------- attachment
def _ids_from(args, kwargs):
    x = args[0] if args else kwargs.get("input", kwargs.get("input_ids"))
    return x if isinstance(x, torch.Tensor) and not x.is_floating_point() and x.dim() == 2 else None


class CA2Attachment:
    """Hooks a ``CA2Controller`` (and its taps) onto a Hugging Face causal LM.

    tap L (default n_blocks // 2): the gist is read at block L's output; inject_before M (default L + 1):
    the injection is applied at block M's input; relevance_layer R (default L): the layer whose tap feeds
    the relevance source, R < M. Registers, in order: an ids hook on the input embeddings (opens each
    call), the block hooks, a model forward hook (closes each call), then each tap's ``attach``.
    """

    def __init__(self, model: nn.Module, ctrl: CA2Controller, tap: int | None = None,
                 inject_before: int | None = None, relevance_layer: int | None = None, taps: Sequence[Tap] = (),
                 blocks: nn.ModuleList | None = None) -> None:
        self.model, self.ctrl = model, ctrl
        self.blocks = pl.find_blocks(model) if blocks is None else blocks
        self.kinds = pl.layer_kinds(model, self.blocks)
        n = len(self.blocks)
        L = n // 2 if tap is None else tap % n
        M = L + 1 if inject_before is None else int(inject_before)
        R = L if relevance_layer is None else int(relevance_layer)
        if not L < M <= n or (M == n and M != L + 1):
            raise ValueError(f"need tap < inject_before <= n_blocks ({n}); got {L}, {M}")
        if not 0 <= R < M:
            raise ValueError(f"relevance_layer {R} must be below inject_before {M}")
        self.tap, self.inject_before, self.relevance_layer = L, M, R
        self.handles = [model.get_input_embeddings().register_forward_hook(self._ids_hook, with_kwargs=True),
                        self.blocks[L].register_forward_hook(self._out_hook)]
        if M != L + 1:
            self.handles.append(self.blocks[M].register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        self.handles.append(model.register_forward_hook(self._end_hook))
        self.taps = list(taps)
        for t in self.taps:
            ctrl.add_tap(t)
            t.attach(self)

    def _ids_hook(self, mod, args, kwargs, out):
        self.ctrl.begin_call(_ids_from(args, kwargs))

    def _out_hook(self, mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        self.ctrl.observe(h)
        if self.inject_before != self.tap + 1:
            return None
        y = self.ctrl.inject(h)
        if y is None:
            return None
        return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y

    def _pre_hook(self, mod, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        y = self.ctrl.inject(x)
        if y is None:
            return None
        if args:
            return (y,) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = y
        return args, kwargs

    def _end_hook(self, mod, args, out):
        self.ctrl.end_call()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        for t in self.taps:
            t.remove()
            if self.ctrl.taps.get(t.name) is t:
                del self.ctrl.taps[t.name]
