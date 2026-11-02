"""The op-amp test: how does CA1 → CA3 feedback shape the closed-loop gain?

An untrained closed HippocampalLoop is driven by a random token stream, with the
hard clamp, error-only projection and saturation all removed, so the only thing
regulating the injection is the feedback path. The open-loop (forward) gain A is
swept for three kinds of feedback:

  none          y = A g Δ
  divisive      y = A g Δ / (1 + κE),  E ∝ recent ‖y‖²      → compresses, y ∝ A^{1/3}
  subtractive   y = A g (Δ − β ȳ),     ȳ = lagged copy of y  → op-amp, y → Δ/β

and β is swept for subtractive feedback at high A. A lagged, smoothed copy can only
cancel what persists from circuit to circuit, so subtractive feedback acts as a
high-pass filter on the injection: it is tested both on a varied stream (24 tokens)
and on a persistent one (a single repeated token, damped kernel, no contrast, since the
contrast would cancel a constant signal), where y_sub / y_none → 1/(1 + K). Because the inverted copy arrives
at least one circuit (τ) late, through a leaky integrator with rate ρ_E, the error
obeys e_t = −K Σ_{d≥τ} (1 − ρ_E) ρ_E^{d−τ} e_{t−d}, K = A g β, whose characteristic
polynomial is z^τ − ρ_E z^{τ−1} + K (1 − ρ_E). Above the critical K, where a root leaves
the unit circle, the loop rings with growing amplitude, as an uncompensated op-amp does.

Usage
  python experiments/loop_systems.py                 # writes results/loop_systems.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from limbic_llm.temporal_loop import HippocampalLoop  # noqa: E402


def _roots(tau: int, rho: float, K: float) -> torch.Tensor:
    """Roots of z^τ − ρ z^{τ−1} + K(1 − ρ), as eigenvalues of its companion matrix."""
    C = torch.zeros(tau, tau, dtype=torch.float64)
    C[1:, :-1] = torch.eye(tau - 1, dtype=torch.float64)
    C[0, 0], C[0, -1] = rho, -K * (1 - rho)                        # first row: −(coefficients)
    return torch.linalg.eigvals(C)


def critical_gain(tau: int, rho: float) -> tuple[float, float]:
    """Smallest K at which z^τ − ρ z^{τ−1} + K(1 − ρ) has a root on the unit circle,
    and the ringing period (tokens) at that point."""
    lo, hi = 0.0, 1.0
    def unstable(K):
        return _roots(tau, rho, K).abs().max().item() >= 1.0
    while not unstable(hi):
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if unstable(mid) else (mid, hi)
    r = _roots(tau, rho, hi)
    z = r[r.abs().argmax()]
    ang = abs(torch.angle(z).item())
    period = 2 * math.pi / ang if ang > 1e-9 else float("inf")
    return hi, period


@torch.no_grad()
def run(feedback: str, A: float, beta: float, kernel: str, seeds: int, T: int, D: int, tau: int,
        vocab: int = 24, contrast: bool = True) -> dict:
    late, mid, peak = [], [], []
    for seed in range(seeds):
        torch.manual_seed(seed)
        loop = HippocampalLoop(D, tau=tau, kernel=kernel, loop="closed", feedback=feedback, contrast=contrast,
                               clamp=False, project=False, saturate=False, forward_gain=A, fb_beta=beta)
        emb = torch.randn(vocab, D)
        h = emb[torch.randint(0, vocab, (8, T))]
        _, st = loop(h)
        r = st.inj_ratio
        mid.append(r[:, T // 2: 3 * T // 4].mean().item())
        late.append(r[:, 3 * T // 4:].mean().item())
        peak.append(r.max().item())
    return {"feedback": feedback, "A": A, "beta": beta, "kernel": kernel, "vocab": vocab,
            "K": A * torch.sigmoid(torch.tensor(-1.0)).item() * beta if feedback == "subtractive" else None,
            "inj_mid": sum(mid) / len(mid), "inj_late": sum(late) / len(late), "inj_max": max(peak)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gains", default="0.5,1,2,4,8,16,32")
    ap.add_argument("--betas", default="0.25,0.5,1,2,4")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--tau", type=int, default=16)
    ap.add_argument("--out", default="results/loop_systems.json")
    args = ap.parse_args()
    gains = [float(x) for x in args.gains.split(",")]
    betas = [float(x) for x in args.betas.split(",")]
    g0 = torch.sigmoid(torch.tensor(-1.0)).item()
    Kc, period = critical_gain(args.tau, 0.9)
    print(f"initial g = {g0:.3f}; subtractive loop rings for K = A·g·β > {Kc:.3f} "
          f"(A·β > {Kc / g0:.2f}), period ≈ {period:.1f} tokens")

    rows = []
    for kernel in ("osc", "flat"):
        for fb in ("none", "divisive", "subtractive"):
            for A in gains:
                rows.append(run(fb, A, 1.0, kernel, args.seeds, args.T, args.d, args.tau))
                r = rows[-1]
                print(f"{kernel:4s} {fb:11s} A={A:5.1f}  inj mid {r['inj_mid']:9.3f}  late {r['inj_late']:9.3f}  "
                      f"max {r['inj_max']:10.3f}", flush=True)
    for beta in betas:
        rows.append(run("subtractive", 4.0, beta, "osc", args.seeds, args.T, args.d, args.tau))
        r = rows[-1]
        print(f"osc  subtractive A=  4.0 β={beta:4.2f} K={r['K']:.2f}  inj late {r['inj_late']:9.3f}  "
              f"max {r['inj_max']:10.3f}", flush=True)
    print("persistent input (one repeated token): subtractive / none should approach 1/(1 + K)")
    for A in (1.0, 2.0, 4.0, 6.0):
        none = run("none", A, 1.0, "exp", args.seeds, args.T, args.d, args.tau, vocab=1, contrast=False)
        sub = run("subtractive", A, 1.0, "exp", args.seeds, args.T, args.d, args.tau, vocab=1, contrast=False)
        rows += [none, sub]
        K = sub["K"]
        print(f"exp  persistent A={A:4.1f} K={K:.2f}  none {none['inj_late']:.4f}  sub {sub['inj_late']:.4f}  "
              f"ratio {sub['inj_late'] / none['inj_late']:.3f}  (1/(1+K) = {1 / (1 + K):.3f})", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "g0": g0, "K_crit": Kc, "ring_period": period,
                               "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
