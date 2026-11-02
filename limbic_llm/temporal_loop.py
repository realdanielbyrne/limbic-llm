"""Temporal loop: hippocampal feedback into the token stream, and a sense of time.

Two side channels for a transformer's residual stream (see docs/design/temporal-loop.md).

HippocampalLoop (CA3 store → CA2 contrast → CA1 control → re-injection)
-----------------------------------------------------------------------
The stream h_t at a tap layer is written to an associative memory and read back
into the stream. Content written at time s re-enters at time t with weight

    w(Δ) = A ρ^{Δ/τ − 1} [cos(2πΔ/τ)]₊,     Δ = t − s ≥ τ,   A ≤ 1,  ρ < 1     ("osc")

so feedback returns in phase windows once per circuit of period τ, is attenuated
by ρ on every circuit, and is clamped to be excitatory and bounded. New input
always enters at weight 1, so it dominates. "exp" drops the oscillation and
"flat" (ρ = 1) is the undamped loop, kept as the unstable control.

Retrieval is content-addressed: relevance a(t, s) = exp(b (q̂_t · k̂_s − 1)) uses a
dense query read directly from the stream (the entorhinal → CA3 perforant path)
and keys that may be sparse dentate-gyrus codes. Feedback is

    F_t = Σ_s w(t − s) a(t, s) v_s,     Δ_t = F_t − m_t μ_t    (contrast against the prior)

where μ_t is the mean of earlier values and m_t the total weight, so Δ_t is what
this episode adds beyond the model's generic expectation (CA2, dynamic activation
addition). With CA1 control the injection is negative-feedback regulated:

    e_t = Δ_t − proj_{h_t}(Δ_t)                      inject only what h_t lacks
    g_t = G tanh(‖Δ_t‖/λ) / (1 + κ E_t)              E_t: recent injection energy
    h_t ← h_t + g_t rms(h_t) ê_t                     ‖injection‖ ≤ G ‖h_t‖, G < 1

so the closed-loop gain is set by the feedback path, as in a negative-feedback
amplifier, and a strong contrast is amplified only up to the clamp.

The parts of that control can be switched separately (``ca1_control`` sets all of them):

    feedback   "divisive"     g ← g / (1 + κE_t)            magnitude-only inhibition
               "subtractive"  Δ_t ← Δ_t − β ȳ_t,  ȳ_t = Σ_{s≤t−τ} k(t−s) inj_s
                                                 an inverted copy of earlier injections
                                                 (content-specific, like a self-negative
                                                 prompt in activation addition)
               "none"
    clamp      rescale the injection to g‖h‖, g ≤ G < 1 (a hard bound, by construction)
    project    error-only injection
    saturate   tanh(‖Δ‖/λ) amplitude

With forward gain A, subtractive feedback gives the op-amp law y = AΔ/(1 + Aβ) → Δ/β
in steady state, while divisive feedback on energy only compresses, y ∝ A^{1/3}.
Because the inverted copy arrives a circuit late, high loop gain can ring.

Retrieval reads with a dense query taken directly from the stream (``read="direct"``,
the entorhinal layer II → CA3 path) or through the same dentate-gyrus code as the keys
(``read="dg"``). ``cue_noise`` adds relative Gaussian noise to the read cue, for tests
of pattern completion from degraded cues.

``loop="open"`` writes the pre-injection stream (acetylcholine-style: encoding is
not contaminated by recall). ``loop="closed"`` writes the post-injection stream,
so injected content is itself stored and re-enters on later circuits; it is
computed exactly in chunks of τ, because nothing within τ steps can feed back.

TimeChannel (an internal clock)
-------------------------------
A learned salience u_t ∈ (0, 1) drives a bank of leaky integrators and damped
oscillators at log-spaced timescales τ_j. This is the Laplace stage of Shankar &
Howard (2012), closer to LEC temporal-context cells (Tsao et al. 2018); their
scale-invariant time cells need an approximate inverse transform, not implemented:

    L_j(t) = (1 − ρ_j) Σ_{s≤t} ρ_j^{t−s} u_s,
    C_j(t) + i S_j(t) = (1 − ρ_j) Σ_{s≤t} ρ_j^{t−s} e^{iω_j (t−s)} u_s,   ρ_j = e^{−1/τ_j}

Each feature is bounded in [−1, 1], and together they encode how long ago salient
events happened, which positional encodings do not provide once an event has left
the attention window. A zero-initialised projection adds them to the stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import TransformerLayer
from .gated_delta import kwta


# ----------------------------------------------------------------------- kernels
def loop_kernel(T: int, tau: int, kind: str = "osc", rho: float = 0.7, amp: float = 1.0,
                device=None, dtype=torch.float32) -> torch.Tensor:
    """Re-entry weights W[t, s] = w(t − s); zero for t − s < τ. Shape (T, T)."""
    if kind not in {"osc", "exp", "flat"}:
        raise ValueError(kind)
    amp = min(float(amp), 1.0)                                   # clamp: feedback never exceeds new input
    # index arithmetic in float32: in bf16, arange(512) has only 385 distinct values, which
    # merges positions and lets the causal mask admit future tokens
    i = torch.arange(T, device=device, dtype=torch.float32)
    d = i[:, None] - i[None, :]
    circuits = d / tau
    if kind == "flat":
        w = torch.full_like(d, amp)
    else:
        w = amp * rho ** (circuits - 1).clamp_min(0)
        if kind == "osc":
            w = w * torch.cos(2 * math.pi * circuits).clamp_min(0)
    return torch.where(d >= tau, w, torch.zeros_like(w)).to(dtype)


def decay_kernel(T: int, rho: float, omega: float = 0.0, device=None, dtype=torch.float32,
                 strict: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) parts of (1 − ρ) ρ^{t−s} e^{iω(t−s)} for s ≤ t (s < t if strict)."""
    i = torch.arange(T, device=device, dtype=torch.float32)        # float32 indices (see loop_kernel)
    d = i[:, None] - i[None, :]
    mask = d > 0 if strict else d >= 0
    base = torch.where(mask, (1 - rho) * rho ** d.clamp_min(0), torch.zeros_like(d))
    return (base * torch.cos(omega * d)).to(dtype), (base * torch.sin(omega * d)).to(dtype)


# ------------------------------------------------------------------- time channel
class TimeChannel(nn.Module):
    def __init__(self, d_model: int, timescales=(4, 8, 16, 32, 64, 128, 256), oscillate: bool = True) -> None:
        super().__init__()
        self.timescales = tuple(float(t) for t in timescales)
        self.oscillate = oscillate
        n = len(self.timescales) * (3 if oscillate else 1)
        self.norm = nn.LayerNorm(d_model)
        self.salience = nn.Linear(d_model, 1)
        nn.init.constant_(self.salience.bias, -2.0)
        self.out = nn.Linear(n, d_model)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def features(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = torch.sigmoid(self.salience(self.norm(h))).squeeze(-1)          # (B, T)
        T = h.shape[1]
        feats = []
        for tau in self.timescales:
            rho = math.exp(-1.0 / tau)
            kl, _ = decay_kernel(T, rho, 0.0, h.device, h.dtype)          # leaky integrator
            feats.append(u @ kl.T)
            if self.oscillate:                                           # damped oscillator, period 4τ
                kc, ks = decay_kernel(T, rho, 2 * math.pi / (4 * tau), h.device, h.dtype)
                feats += [u @ kc.T, u @ ks.T]
        return torch.stack(feats, -1), u                                     # (B, T, n), (B, T)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict]:
        z, u = self.features(h)
        return h + self.out(z), {"salience": u, "time_features": z}


# -------------------------------------------------------------- hippocampal loop
@dataclass
class LoopStats:
    inj_ratio: torch.Tensor          # (B, T) ‖injection‖ / ‖h‖
    gain: torch.Tensor               # (B, T)
    extras: dict = field(default_factory=dict)


class HippocampalLoop(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_key: int = 32,
        tau: int = 16,
        kernel: str = "osc",
        rho: float = 0.7,
        amp: float = 1.0,
        loop: str = "open",
        contrast: bool = True,
        ca1_control: bool = True,
        dg_keys: bool = False,
        dg_expansion: int = 4,
        dg_sparsity: float = 0.1,
        sharpness: float = 6.0,
        max_gain: float = 0.9,
        energy_kappa: float = 4.0,
        energy_rho: float = 0.9,
        feedback: str | None = None,
        clamp: bool | None = None,
        project: bool | None = None,
        saturate: bool | None = None,
        forward_gain: float = 1.0,
        fb_beta: float = 1.0,
        read: str = "direct",
    ) -> None:
        super().__init__()
        if loop not in {"open", "closed"}:
            raise ValueError(loop)
        feedback = feedback if feedback is not None else ("divisive" if ca1_control else "none")
        if feedback not in {"none", "divisive", "subtractive"}:
            raise ValueError(feedback)
        if read not in {"direct", "dg"}:
            raise ValueError(read)
        self.tau, self.kernel, self.rho, self.amp, self.loop = tau, kernel, rho, amp, loop
        self.contrast, self.ca1_control, self.dg_keys = contrast, ca1_control, dg_keys
        self.feedback = feedback
        self.clamp = ca1_control if clamp is None else clamp
        self.project = ca1_control if project is None else project
        self.saturate = ca1_control if saturate is None else saturate
        self.forward_gain, self.fb_beta, self.read = forward_gain, fb_beta, read
        self.cue_noise = 0.0                                             # set at evaluation time
        self.sharpness, self.max_gain = sharpness, max_gain
        self.energy_kappa, self.energy_rho = energy_kappa, energy_rho
        self.dg_sparsity = dg_sparsity
        dk = d_key * (dg_expansion if dg_keys else 1)
        self.norm = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, dk, bias=False)                     # EC → CA3 direct (dense read)
        self.wk = nn.Linear(d_model, dk, bias=dg_keys)                   # DG → CA3 (write address)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.gain_logit = nn.Parameter(torch.tensor(-1.0))              # learned loop gain
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    # ---------------------------------------------------------------- pieces
    def _keys(self, x: torch.Tensor) -> torch.Tensor:
        k = self.wk(x)
        if self.dg_keys:
            k = kwta(k, self.dg_sparsity)
        return F.normalize(k, dim=-1)

    def _inject(self, h: torch.Tensor, delta: torch.Tensor, energy: torch.Tensor,
                ybar: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.sigmoid(self.gain_logit) * self.forward_gain          # forward gain A·g
        x = delta
        if self.feedback == "subtractive" and ybar is not None:
            x = x - self.fb_beta * ybar                                  # inverted copy of past injections
        div = 1 + self.energy_kappa * energy if self.feedback == "divisive" else torch.ones_like(energy)
        lam = self.log_lambda.exp()
        xnorm = x.norm(dim=-1)
        e = x
        if self.project:                                                 # error-only injection
            hn = F.normalize(h, dim=-1)
            e = x - (x * hn).sum(-1, keepdim=True) * hn
        if self.clamp:                                                   # ‖inj‖ = gain‖h‖, gain ≤ G
            s = torch.tanh(xnorm / lam) if self.saturate else torch.ones_like(xnorm)
            gain = self.max_gain * g.clamp(max=1.0) * s / div
            rms = h.pow(2).mean(-1, keepdim=True).sqrt()
            return gain.unsqueeze(-1) * rms * F.normalize(e, dim=-1) * math.sqrt(h.shape[-1]), gain
        # unclamped: amplitude follows the signal, optionally saturating at λ
        s = lam * torch.tanh(xnorm / lam) / xnorm.clamp_min(1e-6) if self.saturate else torch.ones_like(xnorm)
        gain = g * s / div
        return gain.unsqueeze(-1) * e, gain

    # --------------------------------------------------------------- forward
    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, LoopStats]:
        B, T, D = h.shape
        tau = self.tau
        W = loop_kernel(T, tau, self.kernel, self.rho, self.amp, h.device, h.dtype)
        x_pre = self.norm(h)
        cue = x_pre
        if self.cue_noise > 0:                                           # degraded retrieval cue
            rms = x_pre.pow(2).mean(-1, keepdim=True).sqrt()
            cue = x_pre + self.cue_noise * rms * torch.randn_like(x_pre)
        q = self._keys(cue) if self.read == "dg" else F.normalize(self.wq(cue), dim=-1)
        pieces = [h[:, :tau]]                                            # first circuit: nothing to return
        ratios = [h.new_zeros(B, min(tau, T))]
        gain_list = [h.new_zeros(B, min(tau, T))]
        energies = [h.new_zeros(B, min(tau, T))]                         # per-position injection energy
        injs = [h.new_zeros(B, min(tau, T), D)]                          # past injections (subtractive)
        use_hist = self.feedback != "none"
        for start in range(tau, T, tau):                                 # nothing can return before τ
            stop = min(start + tau, T)
            if self.loop == "closed":
                mem = self.norm(torch.cat(pieces, 1))                    # post-injection stream is stored
            else:
                mem = x_pre[:, :start]                                   # pre-injection stream is stored
            k, v = self._keys(mem), self.wv(mem)                         # (B, S, dk), (B, S, D)
            qc = q[:, start:stop]
            a = torch.exp(self.sharpness * (qc @ k.transpose(1, 2) - 1))  # relevance in (0, 1]
            Wc = W[start:stop, :start]
            M = Wc.unsqueeze(0) * a                                       # (B, C, S)
            Fb = M @ v
            if self.contrast:
                avail = (Wc > 0).to(h.dtype)                              # items old enough to return
                cnt = avail.sum(-1).clamp_min(1)
                prior = (avail.unsqueeze(0) @ v) / cnt.view(1, -1, 1)    # mean of returnable values
                delta = Fb - M.sum(-1, keepdim=True) * prior
            else:
                delta = Fb
            # CA1 → CA3 inhibitory back-projection: arrives at least one circuit later,
            # decayed by energy_rho per step. Divisive mode uses its energy E_t, subtractive
            # mode an inverted copy ȳ_t of the earlier injections themselves.
            ybar = None
            if use_hist and start > tau:
                d = (torch.arange(start, stop, device=h.device)[:, None]
                     - torch.arange(start, device=h.device)[None, :]).float()
                kE = torch.where(d >= tau, (1 - self.energy_rho) * self.energy_rho ** (d - tau),
                                 torch.zeros_like(d)).to(h.dtype)
                E = torch.cat(energies, 1) @ kE.T
                if self.feedback == "subtractive":
                    ybar = kE.unsqueeze(0) @ torch.cat(injs, 1)
            else:
                E = h.new_zeros(B, stop - start)
            inj, gain = self._inject(h[:, start:stop], delta, E, ybar)
            injs.append(inj)
            pieces.append(h[:, start:stop] + inj)
            r = inj.norm(dim=-1) / h[:, start:stop].norm(dim=-1).clamp_min(1e-6)
            ratios.append(r)
            gain_list.append(gain)
            energies.append(r.detach() ** 2)
        return torch.cat(pieces, 1), LoopStats(torch.cat(ratios, 1), torch.cat(gain_list, 1))


# ---------------------------------------------------------------------- model
@dataclass
class TemporalLoopOutput:
    logits: torch.Tensor
    loop: LoopStats | None
    time: dict | None


class TemporalLoopLM(nn.Module):
    """Windowed transformer (cortex) with hippocampal loop and time side channels.

    tokens → embed → TransformerLayer(window) → [TimeChannel] → [HippocampalLoop]
           → TransformerLayer(window) → head
    No absolute position embedding by default, so elapsed time can only come
    from the side channels, not from subtracting stored positions.
    """

    def __init__(self, vocab: int, d_model: int = 64, n_heads: int = 4, window: int = 16,
                 memory: bool = True, time: bool = True, pos: bool = False, max_len: int = 2048,
                 time_kwargs: dict | None = None, **loop_kwargs) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.pos = nn.Embedding(max_len, d_model) if pos else None
        self.lower = TransformerLayer(d_model, n_heads, window)
        self.time = TimeChannel(d_model, **(time_kwargs or {})) if time else None
        self.loop = HippocampalLoop(d_model, **loop_kwargs) if memory else None
        self.upper = TransformerLayer(d_model, n_heads, window)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> TemporalLoopOutput:
        h = self.tok(tokens)
        if self.pos is not None:
            h = h + self.pos(torch.arange(tokens.shape[1], device=tokens.device))
        h = self.lower(h)
        tstats = lstats = None
        if self.time is not None:
            h, tstats = self.time(h)
        if self.loop is not None:
            h, lstats = self.loop(h)
        h = self.upper(h)
        return TemporalLoopOutput(self.head(self.norm(h)), lstats, tstats)
