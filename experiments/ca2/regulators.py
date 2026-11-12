"""Recurrences and regulators of the CA2 gain pipeline: leaky scans, release, repetition (R6), projection, dose cap.

Every recurrence of the gain pipeline is one ``leaky_scan``,

    s_t = c_t · s_{t−1} + x_t,            s_{−1} = state0 (zeros by default),

an explicit loop over T in float32, vectorised over the batch (and any trailing dims). A prefill of T
tokens and T single-token decode calls therefore run the *same* arithmetic per position, so a scan is
bitwise equal across the two paths (unlike a matrix-kernel formulation, whose reduction order depends
on T). Positions that do not count (padding, positions before the steer point) use c_t = 1, x_t = 0:
the state is frozen there, so "time" is counted in counted tokens.

Release on backtracking (R3). When a release event fires at t (the driving source reports one, e.g.
the gist agreement a_t < −θ_r, or the current token id is in ``release_ids``), the carried state is
multiplied by ρ_r before x_t is added: c_t ← ρ_r · c_t. ``release_events`` builds the (B, T) event mask.

Repetition regulator (R6). ρ_t = exp(−γ R_t), R_t = λ_r R_{t−1} + rep_t, with rep_t one of
  ``ngram``   1 if the n-gram (n = 4) ending at t already occurred inside the previous ``window`` (32)
              ids, as ``alignment/judge.py`` degeneration counts n-grams; a ring of the last ``window``
              ids (−1 = padding / not seen) carries across calls;
  ``hidden``  max over lags k = 1..K of [cos(h_t, h_{t−k})]_+, from a ring of the last K normalised
              pre-injection states.
Both signals are computed on every step for the trace (``NgramRepeat``, ``HiddenRepeat``); the
regulator (``RepetitionRegulator``) multiplies r_t by ρ_t only when selected. ρ ∈ (0, 1], so a
regulated gain stays inside [g_min, g_max]. While the text repeats R grows and ρ decays toward
exp(−γ/(1 − λ_r)); once it stops R decays geometrically and ρ recovers to 1.

Error-only projection (``temporal_loop.py`` law): d ← normalise(d − (d·ĥ)ĥ), ĥ = h/‖h‖, so the
injection carries no component along the current state; unit norm is kept (matched norm), and the
cosine removed is returned for the trace.

Accumulated-dose cap. On a recurrent (GDN) model the per-token gain is not the dose: the state
integrates it. With D_t = λ_d D_{t−1} + g'_t over real positions, the applied gain is
    g'_t = min(g_t, C − λ_d D_{t−1}),
so D_t ≤ C at every t. The cap is always feasible with g'_t ≥ g_min when g_min ≤ (1 − λ_d)·C (asserted
by the controller), so the dose cap never pushes a steered gain below g_min.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def leaky_scan(coef: torch.Tensor, x: torch.Tensor, state0: torch.Tensor | None = None) -> torch.Tensor:
    """s_t = coef_t · s_{t−1} + x_t along dim 1, (B, T, ...) float32; ``state0`` (B, ...) or None (zeros)."""
    coef, x = torch.broadcast_tensors(coef.float(), x.float())
    out = torch.empty_like(x)
    s = torch.zeros_like(x[:, 0]) if state0 is None else state0.float().expand_as(x[:, 0]).clone()
    for t in range(x.shape[1]):
        s = coef[:, t] * s + x[:, t]
        out[:, t] = s
    return out


def release_events(ids: torch.Tensor | None, release_ids, src_release: torch.Tensor | None,
                   shape: tuple[int, int], device) -> torch.Tensor:
    """(B, T) bool: the driving source's release flag, or the current token id in ``release_ids``."""
    ev = torch.zeros(shape, dtype=torch.bool, device=device)
    if src_release is not None:
        ev = ev | src_release.to(device).bool()
    if ids is not None and len(release_ids):
        ev = ev | torch.isin(ids.to(device), torch.as_tensor(list(release_ids), device=device))
    return ev


def project_error(d: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove the component of d along h and renormalise; returns (d', cos(d, h)). Zero d stays zero.

    d (D,) or (B, T, D); h (B, T, D). The output is (B, T, D).
    """
    d = d.expand_as(h) if d.dim() == 1 else d
    hn = F.normalize(h, dim=-1)
    c = (d * hn).sum(-1, keepdim=True)
    dn = d.norm(dim=-1, keepdim=True)
    resid = d - c * hn
    # d (anti)parallel to h (e.g. the mean direction at the first counted token): nothing is left once the
    # component along h is removed, and normalising the rounding residue would inject a noise direction
    live = resid.norm(dim=-1, keepdim=True) > 1e-4 * dn
    dp = torch.where(live & (dn > 0), F.normalize(resid, dim=-1), torch.zeros_like(d))
    cos = torch.where(dn[..., 0] > 0, c[..., 0] / dn[..., 0].clamp_min(1e-30), torch.zeros_like(c[..., 0]))
    return dp, cos


def dose_cap_scan(g: torch.Tensor, real: torch.Tensor, lam: float, cap: float,
                  state0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the accumulated-dose cap: g'_t = min(g_t, C − λ D_{t−1}), D_t = λ D_{t−1} + g'_t at real positions
    (frozen elsewhere). g, real (B, T); returns (g', D) float32."""
    g = g.float()
    real = real.float()
    D = torch.zeros_like(g[:, 0]) if state0 is None else state0.float().clone()
    gs, Ds = torch.empty_like(g), torch.empty_like(g)
    for t in range(g.shape[1]):
        gt = torch.minimum(g[:, t], cap - lam * D) * real[:, t]
        D = torch.where(real[:, t] > 0, lam * D + gt, D)
        gs[:, t], Ds[:, t] = gt, D
    return gs, Ds


class NgramRepeat:
    """rep_t = 1 if the n-gram ending at t occurred inside the previous ``window`` ids (all ids real)."""

    def __init__(self, n: int = 4, window: int = 32) -> None:
        if not 1 <= n <= window:
            raise ValueError("need 1 <= n <= window")
        self.n, self.window = int(n), int(window)
        self.reset()

    def reset(self) -> None:
        self.ring: torch.Tensor | None = None                 # (B, window) last ids, −1 = none / padding

    def __call__(self, ids: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        W, n = self.window, self.n
        cur = torch.where(real.bool(), ids.long(), torch.full_like(ids.long(), -1))
        ring = self.ring if self.ring is not None else torch.full((B, W), -1, dtype=torch.long, device=ids.device)
        seq = torch.cat([ring, cur], 1)                       # (B, W + T); current t sits at W + t
        A = [seq[:, W - k: W - k + T] for k in range(n)]      # ids t − k
        ok = torch.ones(B, T, dtype=torch.bool, device=ids.device)
        for a in A:
            ok &= a >= 0
        rep = torch.zeros(B, T, dtype=torch.bool, device=ids.device)
        for lag in range(1, W - n + 2):                       # earlier gram fully inside ids[t − W .. t − 1]
            m = ok.clone()
            for k in range(n):
                m &= A[k] == seq[:, W - lag - k: W - lag - k + T]
            rep |= m
        self.ring = seq[:, -W:].detach()
        return rep.float()

    def state_numel(self) -> int:
        return 0 if self.ring is None else self.ring.numel()


class HiddenRepeat:
    """rep_t = max over lags 1..K of [cos(h_t, h_{t−k})]_+ over counted earlier positions (0 if none)."""

    def __init__(self, lags: int = 8) -> None:
        self.lags = int(lags)
        self.reset()

    def reset(self) -> None:
        self.ring: torch.Tensor | None = None                 # (B, K, D) normalised earlier states
        self.ring_ok: torch.Tensor | None = None              # (B, K) bool

    def __call__(self, h: torch.Tensor, use: torch.Tensor) -> torch.Tensor:
        B, T, D = h.shape
        K = self.lags
        hn = F.normalize(h.float(), dim=-1)
        ring = self.ring if self.ring is not None else hn.new_zeros(B, K, D)
        rok = self.ring_ok if self.ring_ok is not None else torch.zeros(B, K, dtype=torch.bool, device=h.device)
        seq, ok = torch.cat([ring, hn], 1), torch.cat([rok, use.bool()], 1)
        rep = hn.new_zeros(B, T)
        for lag in range(1, K + 1):
            c = (hn * seq[:, K - lag: K - lag + T]).sum(-1).clamp_min(0.0)
            c = torch.where(ok[:, K - lag: K - lag + T] & use.bool(), c, torch.zeros_like(c))
            rep = torch.maximum(rep, c)
        self.ring, self.ring_ok = seq[:, -K:].detach(), ok[:, -K:].detach()
        return rep

    def state_numel(self) -> int:
        return 0 if self.ring is None else self.ring.numel() + self.ring_ok.numel()


class RepetitionRegulator:
    """ρ_t = exp(−γ R_t), R_t = λ_r R_{t−1} + rep_t over real positions (frozen elsewhere)."""

    def __init__(self, gamma: float = 1.0, leak: float = 0.9) -> None:
        if gamma < 0 or not 0.0 <= leak < 1.0:
            raise ValueError("need gamma >= 0 and leak in [0, 1)")
        self.gamma, self.leak = float(gamma), float(leak)
        self.reset()

    def reset(self) -> None:
        self.R: torch.Tensor | None = None                    # (B,)

    def __call__(self, rep: torch.Tensor, real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real = real.float()
        coef = torch.where(real > 0, torch.full_like(real, self.leak), torch.ones_like(real))
        R = leaky_scan(coef, rep.float() * real, self.R)
        self.R = R[:, -1].detach()
        return R, torch.exp(-self.gamma * R)

    def state_numel(self) -> int:
        return 0 if self.R is None else self.R.numel()
