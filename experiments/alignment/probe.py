"""Activations, the harm probe (A1), and steering directions.

* ``activations``: the residual stream after one decoder block at the last prompt
  token. Prompts are left-padded, so the last position is always real.
* ``fit_probe``: logistic regression on standardised activations. This is the
  CA1-style calibrated decision s_t = σ(w·(h − μ)/σ + b). Reports held-out AUROC and
  expected calibration error.
* ``diff_of_means``: the refusal direction of Arditi et al. (2024), the mean
  activation on harmful prompts minus the mean on harmless ones. Taken from an
  aligned model, it is the direction the hook injects into an unaligned one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def activations(model, block, batches) -> torch.Tensor:
    """Block-output activation at the last position of each prompt, (N, D) float32 on CPU.

    ``batches`` yields (ids, mask) pairs of left-padded prompts.
    """
    grabbed = []

    def hook(mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        grabbed.append(h[:, -1].float().cpu())

    handle = block.register_forward_hook(hook)
    dev = model.get_input_embeddings().weight.device
    try:
        for ids, mask in batches:
            model(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False)
    finally:
        handle.remove()
    return torch.cat(grabbed)


def diff_of_means(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """mean(X | y = 1) − mean(X | y = 0)."""
    y = y.bool()
    return X[y].mean(0) - X[~y].mean(0)


def fit_probe(X: torch.Tensor, y: torch.Tensor, l2: float = 1e-2, steps: int = 400, lr: float = 0.05,
              seed: int = 0) -> dict:
    """Logistic regression on standardised features; returns {w, b, mu, sd} in input space terms."""
    torch.manual_seed(seed)
    X, y = X.float(), y.float()
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-4)
    Z = (X - mu) / sd
    w = torch.zeros(Z.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Z @ w + b, y) + l2 * w.pow(2).sum() / len(y)
        loss.backward()
        opt.step()
    return {"w": w.detach(), "b": float(b.detach()), "mu": mu, "sd": sd}


def predict(probe: dict, X: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(((X.float() - probe["mu"]) / probe["sd"]) @ probe["w"] + probe["b"])


def auroc(scores: torch.Tensor, y: torch.Tensor) -> float:
    """Area under the ROC curve by the rank-sum (Mann–Whitney) statistic, ties averaged."""
    scores, y = scores.double(), y.bool()
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    for v in scores.unique():                         # average ranks over ties
        tie = scores == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(probs: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    probs, y = probs.double(), y.double()
    edges = torch.linspace(0, 1, bins + 1, dtype=torch.double)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (probs >= lo) & ((probs < hi) if hi < 1 else (probs <= hi))
        if sel.any():
            err += sel.double().mean().item() * abs(probs[sel].mean().item() - y[sel].mean().item())
    return err


def split(n: int, frac: float = 0.8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    k = int(round(frac * n))
    return perm[:k], perm[k:]


def probe_report(X: torch.Tensor, y: torch.Tensor, seed: int = 0) -> tuple[dict, dict]:
    """Fit on 80 %, report on the held-out 20 %, then refit on everything for use."""
    tr, te = split(len(y), seed=seed)
    p = fit_probe(X[tr], y[tr], seed=seed)
    pt = predict(p, X[te])
    report = {"auroc": auroc(pt, y[te]), "ece": ece(pt, y[te]),
              "acc": float(((pt >= 0.5).float() == y[te].float()).float().mean()), "n_test": int(len(te))}
    return fit_probe(X, y, seed=seed), report


def residual_direction(probe: dict) -> torch.Tensor:
    """The probe's weight mapped back to residual-stream coordinates (w / σ)."""
    return probe["w"] / probe["sd"]
