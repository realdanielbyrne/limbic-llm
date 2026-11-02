"""CA2 norm-steering controller: a graded, regulated steering side channel.

Design: research/NOTEBOOK.md, "CA2 as a norm-steering controller" and
"re-aligning an unaligned model". A linear probe on the residual stream gives a
harm probability s_t, a calibrated CA1-style decision. A leaky integral
E_t = λ E_{t−1} + s_t lets sustained pressure escalate and decay away. The
injection pushes the stream along a steering direction d̂ (for alignment, the
refusal direction), bounded by a cap:

    none         g_t = 0
    static       g_t = c                         activation addition / CAA
    conditional  g_t = c · 1[s_prompt ≥ θ]       CAST-style switch, decided on the prompt
    graded       g_t = G · tanh(κ E_t)           the escalating controller, g_t ≤ G

    h_t ← h_t + g_t ‖h_t‖ d̂

Gains are fractions of ‖h_t‖, so G is directly comparable with the saturation
knee of stability experiment S1.

Direction. ``direction="fixed"`` uses d̂ (the refusal direction for alignment).
``direction="gist"`` boosts the current line of thought instead: two exponential
averages of the stream before injection, a fast one for the current thought and a
slow one for the background,

    m_t = β m_{t−1} + (1 − β) h_t,    b_t = β_s b_{t−1} + (1 − β_s) h_t,    d̂_t = (m_t − b_t)/‖m_t − b_t‖

which is momentum on the residual stream, keeping only what distinguishes the
current thought. Both averages read the stream before injection (open loop), so
the injection never feeds back into them; with the cap this keeps the positive
feedback bounded (stability programme S1–S8).

``score_positions="last"`` (default) counts the probe only where it was trained:
the last prompt token during prefill, then each generated token. State (E, the
conditional decision) carries across forward calls, so prefill followed by
token-by-token decoding with a KV cache matches a single pass. Call ``reset()``
between sequences. The controller runs in float32 whatever the model's dtype
(bf16 index arithmetic is unsafe; see register C4).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("none", "static", "conditional", "graded")


class SteeringController(nn.Module):
    def __init__(self, direction: torch.Tensor | None, probe: dict | None = None, mode: str = "graded",
                 gain: float = 0.15, cap: float = 0.3, kappa: float = 0.5, leak: float = 0.9,
                 threshold: float = 0.5, score_positions: str = "last", direction_mode: str = "fixed",
                 beta: float = 0.9, beta_slow: float = 0.995, d_model: int | None = None) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if direction_mode not in ("fixed", "gist"):
            raise ValueError(direction_mode)
        if direction_mode == "fixed" and direction is None:
            raise ValueError("a fixed direction is required unless direction_mode='gist'")
        self.direction_mode, self.beta, self.beta_slow = direction_mode, float(beta), float(beta_slow)
        if direction is None:
            direction = torch.zeros(int(d_model))
        if score_positions not in ("last", "all"):
            raise ValueError(score_positions)
        if not 0.0 <= leak < 1.0:
            raise ValueError("leak must be in [0, 1)")
        self.mode, self.gain, self.cap, self.kappa = mode, float(gain), float(cap), float(kappa)
        self.leak, self.threshold, self.score_positions = float(leak), float(threshold), score_positions
        self.register_buffer("d", F.normalize(direction.detach().float().flatten(), dim=0))
        D = self.d.numel()
        probe = probe or {}
        self.register_buffer("w", probe.get("w", torch.zeros(D)).detach().float())
        self.register_buffer("mu", probe.get("mu", torch.zeros(D)).detach().float())
        self.register_buffer("sd", probe.get("sd", torch.ones(D)).detach().float())
        self.b = float(probe.get("b", -30.0 if not probe else 0.0))     # no probe → s ≈ 0
        self.record = False
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.E: torch.Tensor | None = None            # (B,) integral after the last position seen
        self.m: torch.Tensor | None = None            # (B, D) fast average of the stream (gist mode)
        self.bg: torch.Tensor | None = None           # (B, D) slow average (background)
        self.cond: torch.Tensor | None = None         # (B,) conditional-mode decision
        self.pending_mask: torch.Tensor | None = None  # (B, T) padding mask for the next (prefill) call
        self.calls = 0
        self.trace: list[dict] = []

    def score(self, h: torch.Tensor) -> torch.Tensor:
        """Harm probability s_t from the linear probe, (B, T)."""
        x = (h - self.mu) / self.sd
        return torch.sigmoid(x @ self.w + self.b)

    # ---------------------------------------------------------------- forward
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        dt = h.dtype
        h = h.float()
        B, T, _ = h.shape
        mask = self.pending_mask
        self.pending_mask = None
        mask = torch.ones(B, T, device=h.device) if mask is None else mask.to(h.device).float()[:, -T:]
        s_all = self.score(h) * mask
        s = s_all
        if self.score_positions == "last" and self.calls == 0 and T > 1:
            s = torch.zeros_like(s_all)
            s[:, -1] = s_all[:, -1]                    # prefill: only the last prompt token
        # leaky integral E_t = λ^{t+1} E_prev + Σ_{u≤t} λ^{t−u} s_u, indices in float32
        i = torch.arange(T, device=h.device, dtype=torch.float32)
        diff = i[:, None] - i[None, :]
        K = torch.where(diff >= 0, self.leak ** diff.clamp_min(0), torch.zeros_like(diff))
        E = s @ K.T
        if self.E is not None:
            E = E + self.E[:, None] * self.leak ** (i + 1)[None, :]
        if self.mode == "none":
            g = torch.zeros_like(E)
        elif self.mode == "static":
            g = torch.full_like(E, self.gain)
        elif self.mode == "conditional":
            if self.cond is None:                      # decided once, on the prompt's last token
                self.cond = s_all[:, -1] >= self.threshold
            g = self.gain * self.cond.float()[:, None].expand(B, T)
            if self.calls == 0 and self.score_positions == "last" and T > 1:
                g = torch.cat([torch.zeros(B, T - 1, device=h.device), g[:, -1:]], 1)
        else:
            g = self.cap * torch.tanh(self.kappa * E)
        g = g * mask
        if self.direction_mode == "gist":
            d = self._gist(h, mask, i)                 # (B, T, D), from the pre-injection stream
        else:
            d = self.d
        out = h + g.unsqueeze(-1) * h.norm(dim=-1, keepdim=True) * d
        self.E = E[:, -1].detach()
        self.calls += 1
        if self.record:
            self.trace.append({"s": s_all.detach().cpu(), "E": E.detach().cpu(), "g": g.detach().cpu()})
        return out.to(dt)

    def _ema(self, x: torch.Tensor, mask: torch.Tensor, beta: float, prev: torch.Tensor | None,
             i: torch.Tensor) -> torch.Tensor:
        """m_t = β m_{t−1} + (1 − β) x_t over real positions (pads skipped), (B, T, D)."""
        diff = i[:, None] - i[None, :]
        K = torch.where(diff >= 0, beta ** diff.clamp_min(0), torch.zeros_like(diff))   # (T, T)
        m = (1 - beta) * torch.einsum("tu,bud->btd", K, x * mask.unsqueeze(-1))
        if prev is not None:
            m = m + prev[:, None, :] * (beta ** (i + 1))[None, :, None]
        return m

    def _gist(self, h: torch.Tensor, mask: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        m = self._ema(h, mask, self.beta, self.m, i)
        bg = self._ema(h, mask, self.beta_slow, self.bg, i)
        self.m, self.bg = m[:, -1].detach(), bg[:, -1].detach()
        return F.normalize(m - bg, dim=-1)

    def summary(self) -> dict:
        """Per-sequence statistics of the last generation, from the recorded trace."""
        if not self.trace:
            return {}
        g = torch.cat([t["g"] for t in self.trace], 1)
        E = torch.cat([t["E"] for t in self.trace], 1)
        s = torch.cat([t["s"] for t in self.trace], 1)
        return {"gain_mean": g.mean(1), "gain_max": g.max(1).values, "E_final": E[:, -1],
                "capped_frac": (g >= 0.95 * max(self.cap, 1e-9)).float().mean(1), "s_mean": s.mean(1)}


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
