"""CA2 norm-steering controller: a graded, regulated steering side channel.

Design: research/NOTEBOOK.md, "CA2 as a norm-steering controller" and
"re-aligning an unaligned model". A score s_t ∈ [0, 1] (a linear harm probe on the
residual stream, or any ``score_fn``) acts as a CA1-style decision. A leaky
integral E_t = λ E_{t−1} + s_t lets sustained pressure escalate and decay away. The
injection pushes the stream along a steering direction d̂ (for alignment, the
refusal direction), bounded by a cap:

    none         g_t = 0
    static       g_t = c                         activation addition / CAA
    conditional  g_t = c · 1[s_prompt ≥ θ]       CAST-style switch, decided on the prompt
    graded       g_t = G · tanh(κ E_t)           the escalating controller, g_t ≤ G

    h_t ← h_t + g_t ‖h_t‖ d̂

Gains are fractions of ‖h_t‖, so G is directly comparable with the saturation
knee of stability experiment S1, and every direction mode below injects the same
norm at the same g (a matched-norm comparison; the trace records the effective
gain g·‖d̂‖, which is 0 where a gist direction is still undefined).

Score source (RV1.C3, RV1.D4). ``conditional`` and ``graded`` need a score: a probe
dict {w, b, mu, sd}, or ``score_fn(h) -> (B, T)`` probabilities (for example a
temperature-scaled head). A model-agnostic text classifier or a guard model gives
one score per prompt, not per token: pass ``external_decision=True`` and set
``pending_decision`` (B,) before each generation; that source can drive only the
``conditional`` mode. Constructing a mode without the score it needs raises.

Direction modes:

``fixed``      d̂ given (the refusal direction for alignment).
``gist``       the current line of thought: two bias-corrected exponential averages of
               the stream before injection, a fast one and a slow one,
                   m_t = Σ_u β^{t−u} h_u / Σ_u β^{t−u},   b_t likewise with β_s,
                   d̂_t = (m_t − b_t)/‖m_t − b_t‖,
               over real tokens only, skipping the first ``gist_skip`` real tokens of
               each sequence. The first token is the attention sink, whose norm is
               ~80× the median on Qwen3 (RV1.C1); with it included, or with
               zero-initialised averages, d̂ pointed along the sink and the stream mean
               rather than the current thought. Because both averages are normalised
               means of the same tokens, a constant offset cancels exactly, so no
               corpus mean has to be subtracted.
``anti_gist``  −d̂_gist (sign control).
``mean``       b_t/‖b_t‖, the slow average alone (does adding the stream mean do it?).
``random``     a fixed random unit direction, seeded (does any perturbation do it?).

The last three are the matched-norm controls of RV1.C2. The averages read the
stream before injection, so *within one forward pass* the injection does not feed
back into them. Across decoding steps it does (RV1.C6): the injection changes the
sampled token, which sets every later h_t, so a high gist gain can lock generation
into a repetition attractor. Measure it with ``judge.repetition_onset``.

``score_positions="last"`` (default) counts the score at the last prompt token
during prefill, then at each generated token. The probe of ``probe.py`` is fit
only at the last prompt token, so its calibration on generated tokens is an
extrapolation; ``run.py`` measures it (RV1.A2) rather than assuming it. State (E,
the averages, the conditional decision) carries across forward calls, so prefill
followed by token-by-token decoding with a KV cache matches a single pass. Call
``reset()`` between sequences. The controller runs in float32 whatever the
model's dtype (bf16 index arithmetic is unsafe; see register B4).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("none", "static", "conditional", "graded")
DIRECTIONS = ("fixed", "gist", "anti_gist", "mean", "random")


class SteeringController(nn.Module):
    def __init__(self, direction: torch.Tensor | None, probe: dict | None = None, mode: str = "graded",
                 gain: float = 0.15, cap: float = 0.3, kappa: float = 0.5, leak: float = 0.9,
                 threshold: float = 0.5, score_positions: str = "last", direction_mode: str = "fixed",
                 beta: float = 0.9, beta_slow: float = 0.995, gist_skip: int = 1, d_model: int | None = None,
                 score_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
                 external_decision: bool = False, seed: int = 0) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if direction_mode not in DIRECTIONS:
            raise ValueError(f"direction_mode must be one of {DIRECTIONS}")
        if direction_mode == "fixed" and direction is None:
            raise ValueError("a fixed direction is required unless direction_mode is a gist or control mode")
        if score_positions not in ("last", "all"):
            raise ValueError(score_positions)
        if not 0.0 <= leak < 1.0:
            raise ValueError("leak must be in [0, 1)")
        if not (0.0 < beta < 1.0 and 0.0 < beta_slow < 1.0):
            raise ValueError("beta and beta_slow must be in (0, 1)")
        probe = probe or None
        if probe is not None and score_fn is not None:
            raise ValueError("give a probe or a score_fn, not both")
        per_token = probe is not None or score_fn is not None
        if mode == "graded" and not per_token:              # RV1.C3: was a silent no-op (s ≡ 0)
            raise ValueError("graded mode needs a per-token score: pass a probe or a score_fn")
        if mode == "conditional" and not (per_token or external_decision):
            raise ValueError("conditional mode needs a probe, a score_fn or external_decision=True")
        if external_decision and mode != "conditional":
            raise ValueError("an external (per-prompt) decision can drive only the conditional mode")
        if direction is None and d_model is None:
            raise ValueError("d_model is required when no direction is given")
        D = int(direction.numel()) if direction is not None else int(d_model)
        if direction_mode == "random":
            g = torch.Generator().manual_seed(seed)
            direction = torch.randn(D, generator=g)
        elif direction is None:
            direction = torch.zeros(D)
        self.mode, self.gain, self.cap, self.kappa = mode, float(gain), float(cap), float(kappa)
        self.leak, self.threshold, self.score_positions = float(leak), float(threshold), score_positions
        self.direction_mode, self.beta, self.beta_slow = direction_mode, float(beta), float(beta_slow)
        self.gist_skip = int(gist_skip)
        self.score_fn, self.external_decision = score_fn, bool(external_decision)
        self.register_buffer("d", F.normalize(direction.detach().float().flatten(), dim=0))
        self.has_probe = probe is not None
        p = probe or {}
        self.register_buffer("w", p.get("w", torch.zeros(D)).detach().float())
        self.register_buffer("mu", p.get("mu", torch.zeros(D)).detach().float())
        self.register_buffer("sd", p.get("sd", torch.ones(D)).detach().float())
        self.b = float(p.get("b", 0.0))
        self.record = False
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.E: torch.Tensor | None = None             # (B,) integral after the last position seen
        self.ema: dict[str, torch.Tensor] = {}         # bias-corrected averages: numerators (B, D), weights (B,)
        self.n_seen: torch.Tensor | None = None        # (B,) real tokens seen so far (for gist_skip)
        self.cond: torch.Tensor | None = None          # (B,) conditional-mode decision
        self.pending_mask: torch.Tensor | None = None  # (B, T) padding mask for the next (prefill) call
        self.pending_decision: torch.Tensor | None = None  # (B,) external per-prompt score in [0, 1]
        self.prefill_mask: torch.Tensor | None = None  # (B, T) the mask the prefill call used
        self.calls = 0
        self.trace: list[dict] = []

    def score(self, h: torch.Tensor) -> torch.Tensor:
        """Score s_t ∈ [0, 1] per position, (B, T); zero when the controller has no score source."""
        if self.score_fn is not None:
            return self.score_fn(h).float()
        if not self.has_probe:
            return torch.zeros(h.shape[:2], device=h.device)
        x = (h - self.mu) / self.sd
        return torch.sigmoid(x @ self.w + self.b)

    # ---------------------------------------------------------------- forward
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        dt = h.dtype
        h = h.float()
        B, T, _ = h.shape
        first = self.calls == 0
        mask = self.pending_mask
        self.pending_mask = None
        mask = torch.ones(B, T, device=h.device) if mask is None else mask.to(h.device).float()[:, -T:]
        if first:
            self.prefill_mask = mask.detach().cpu()
        s_all = self.score(h) * mask
        s = s_all
        last_only = self.score_positions == "last" and first and T > 1
        if last_only:
            s = torch.zeros_like(s_all)
            s[:, -1] = s_all[:, -1]                    # prefill: only the last prompt token
        # leaky integral E_t = λ^{t+1} E_prev + Σ_{u≤t} λ^{t−u} s_u, indices in float32
        i = torch.arange(T, device=h.device, dtype=torch.float32)
        E = s @ self._kernel(self.leak, i).T
        if self.E is not None:
            E = E + self.E[:, None] * self.leak ** (i + 1)[None, :]
        if self.mode == "none":
            g = torch.zeros_like(E)
        elif self.mode == "static":
            g = torch.full_like(E, self.gain)
        elif self.mode == "conditional":
            if self.cond is None:                      # decided once, on the prompt's last token
                if self.external_decision:
                    if self.pending_decision is None:
                        raise RuntimeError("external_decision: set pending_decision before generating")
                    self.cond = self.pending_decision.to(h.device).float() >= self.threshold
                else:
                    self.cond = s_all[:, -1] >= self.threshold
            g = self.gain * self.cond.float()[:, None].expand(B, T)
            if last_only:
                g = torch.cat([torch.zeros(B, T - 1, device=h.device), g[:, -1:]], 1)
        else:
            g = self.cap * torch.tanh(self.kappa * E)
        g = g * mask
        d = self._direction(h, mask, i)                # (D,) or (B, T, D), from the pre-injection stream
        out = h + g.unsqueeze(-1) * h.norm(dim=-1, keepdim=True) * d
        self.E = E[:, -1].detach()
        self.calls += 1
        if self.record:
            g_eff = g if d.dim() == 1 else g * d.norm(dim=-1)
            self.trace.append({"s": s_all.detach().cpu(), "E": E.detach().cpu(), "g": g_eff.detach().cpu()})
        return out.to(dt)

    @staticmethod
    def _kernel(rate: float, i: torch.Tensor) -> torch.Tensor:
        """Causal decay kernel K[t, u] = rate^{t−u} for u ≤ t, else 0, (T, T)."""
        diff = i[:, None] - i[None, :]
        return torch.where(diff >= 0, rate ** diff.clamp_min(0), torch.zeros_like(diff))

    def _ema(self, x: torch.Tensor, use: torch.Tensor, beta: float, key: str, i: torch.Tensor) -> torch.Tensor:
        """Bias-corrected average Σ β^{t−u} use_u x_u / Σ β^{t−u} use_u, (B, T, D); 0 before any used token."""
        K = self._kernel(beta, i)
        num = torch.einsum("tu,bud->btd", K, x * use.unsqueeze(-1))
        den = use @ K.T
        if key in self.ema:
            decay = beta ** (i + 1)
            num = num + self.ema[key][0][:, None, :] * decay[None, :, None]
            den = den + self.ema[key][1][:, None] * decay[None, :]
        self.ema[key] = (num[:, -1].detach(), den[:, -1].detach())
        return torch.where(den.unsqueeze(-1) > 0, num / den.clamp_min(1e-30).unsqueeze(-1), torch.zeros_like(num))

    def _direction(self, h: torch.Tensor, mask: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        if self.direction_mode in ("fixed", "random"):
            return self.d
        n_prev = self.n_seen if self.n_seen is not None else torch.zeros(h.shape[0], device=h.device)
        idx = n_prev[:, None] + mask.cumsum(1) - 1      # index of each real token within its sequence
        use = mask * (idx >= self.gist_skip).float()    # skip the attention sink (RV1.C1)
        self.n_seen = (n_prev + mask.sum(1)).detach()
        bg = self._ema(h, use, self.beta_slow, "slow", i)
        if self.direction_mode == "mean":
            return F.normalize(bg, dim=-1)
        m = self._ema(h, use, self.beta, "fast", i)
        d = F.normalize(m - bg, dim=-1)
        return -d if self.direction_mode == "anti_gist" else d

    # ---------------------------------------------------------------- reports
    def summary(self, gen_ids: torch.Tensor | None = None, eos_ids=None) -> dict:
        """Per-sequence statistics of the last generation, over real positions only (RV1.A5).

        Response positions are the last prompt token plus each generated token fed back
        before the first EOS (``gen_ids`` (B, n), the generated ids; ``eos_ids`` an id or a
        list of ids). Prompt statistics use the prefill's padding mask. Gains are the
        effective gains g·‖d̂‖.
        """
        if not self.trace:
            return {}
        pre, dec = self.trace[0], self.trace[1:]
        B = pre["g"].shape[0]
        pm = self.prefill_mask if self.prefill_mask is not None else torch.ones_like(pre["g"])
        if dec:
            s_dec, g_dec, E_dec = (torch.cat([t[k] for t in dec], 1) for k in ("s", "g", "E"))
        else:
            s_dec = g_dec = E_dec = torch.zeros(B, 0)
        gen_mask = torch.ones_like(s_dec)
        if gen_ids is not None and eos_ids is not None and s_dec.shape[1]:
            eos = torch.tensor(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
            # decode call k processes generated token k − 1; it is real iff no EOS came at or before it
            hit = torch.isin(gen_ids[:, : s_dec.shape[1]].cpu(), eos)
            gen_mask = (hit.long().cumsum(1) == 0).float()
        rmask = torch.cat([torch.ones(B, 1), gen_mask], 1)
        g = torch.cat([pre["g"][:, -1:], g_dec], 1) * rmask
        E = torch.cat([pre["E"][:, -1:], E_dec], 1)
        n = rmask.sum(1)
        last = (n - 1).long()
        n_gen = gen_mask.sum(1)
        s_gen_mean = torch.where(n_gen > 0, (s_dec * gen_mask).sum(1) / n_gen.clamp_min(1),
                                 torch.full((B,), float("nan")))
        return {"gain_mean": g.sum(1) / n, "gain_max": g.max(1).values,
                "capped_frac": ((g >= 0.95 * max(self.cap, 1e-9)).float() * rmask).sum(1) / n,
                "gain_prompt_mean": (pre["g"] * pm).sum(1) / pm.sum(1).clamp_min(1),
                "E_final": E.gather(1, last[:, None])[:, 0],
                "s_prompt": pre["s"][:, -1], "s_gen_mean": s_gen_mean,
                "s_gen": s_dec, "gen_mask": gen_mask}


class Attachment:
    """Forward hook that applies a controller to one decoder block's output."""

    def __init__(self, block: nn.Module, controller: SteeringController) -> None:
        self.controller = controller
        self.handle = block.register_forward_hook(self._hook)

    def _hook(self, mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.controller.d.device != h.device:             # multi-GPU placement
            self.controller.to(h.device)
        h = self.controller(h)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def remove(self) -> None:
        self.handle.remove()
