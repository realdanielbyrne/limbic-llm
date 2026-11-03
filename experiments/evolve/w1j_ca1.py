"""Round-8 candidate W1j: W1i with an explicit CA1 comparator driving its event gate.

In the hippocampus the decision to encode is not learned from scratch. CA1 compares the direct
cortical input (EC III) with CA3's prediction (Schaffer collaterals), and the mismatch drives
encoding through neuromodulation: the hippocampal–VTA dopamine loop (Lisman & Grace 2005) and the
acetylcholine encode/retrieve switch (Hasselmo 2006). W1i had to discover its event detector
from a weak gradient, and most seeds never did (P36). The failing runs sit on the "everything but
which code" plateau, because three parts must co-adapt (NOTEBOOK, round 7 analysis). W1j builds
the comparison in, so the event part arrives with structure:

    err_t   = ‖v_t − α_t S_{t−1} a_{t−1}‖²            the chain store's own prediction error: what
                                                      arrived (EC) against what it predicted (CA3)
    rel_t   = err_t / (‖v_t‖² + ε)                    scale-free: 1 = predicted nothing, 0 = exactly
    n_t     = rel_t − m_{t−1},  m_t = λ m_{t−1} + (1 − λ) rel_t    novelty against the recent baseline
    e_t     = σ(w_e · x_t + b_e + κ_h n_t)            CA1 novelty added to W1i's event logit (κ_h learned)

The comparison has no parameters and is meaningful from the first step. It is detached, so the
memory cannot learn to predict worse in order to trigger writes; κ_h (per head, starting at 4)
learns how much novelty should gate encoding. Everything else is W1i: the chain store S, the
event-gated goal field G with refractory trace and optional write budget, and the CA1 injection.
Novelty alone cannot pick out the facts in random-token filler, where every token is unexpected;
it can in natural text, where the digits of a code stand out. The learned term w_e · x_t stays.

Run with ``experiments/evolve/run.py --candidate evolve.w1j_ca1:W1jSide``; W1i's --w1i-* flags
apply, plus --w1j-*.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.w1i_events import EventLoop, causal_decay
from evolve.w1i_events import split_args as w1i_split_args
from limbic_llm.gated_delta import _gates, _run
from limbic_llm.temporal_loop import LoopStats, TimeChannel


def split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Take the --w1i-* and --w1j-* flags; the rest go to ``pretrained_loop``."""
    w, rest = w1i_split_args(argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--w1j-kappa", type=float, default=4.0, help="initial novelty gain κ on the event logit")
    ap.add_argument("--w1j-baseline", type=float, default=0.9, help="λ of the novelty baseline (running mean)")
    j, rest = ap.parse_known_args(rest)
    return argparse.Namespace(**vars(w), **vars(j)), rest


class CA1Loop(EventLoop):
    """W1i (chain store S + event-gated goal field G) with CA1 novelty in the event gate."""

    def __init__(self, d_model: int, kappa: float = 4.0, baseline: float = 0.9, **kw) -> None:
        super().__init__(d_model, **kw)
        self.kappa = nn.Parameter(torch.full((self.mem.h,), float(kappa)))
        self.baseline = baseline

    def novelty(self, err: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """n_t = rel_t − m_{t−1}: relative prediction error against its recent causal mean. (B, H, T)."""
        rel = (err / (v.pow(2).sum(-1) + 1e-6)).float()                    # (B, H, T)
        lam = torch.full((rel.shape[1],), self.baseline, device=rel.device)
        K = causal_decay(rel.shape[2], lam, rel.device) * (1 - self.baseline)
        m = torch.einsum("hts,bhs->bht", K, rel)                           # m_t (includes t)
        m_prev = torch.cat([torch.zeros_like(m[..., :1]), m[..., :-1]], -1)
        return rel - m_prev

    def _read(self, x: torch.Tensor) -> torch.Tensor:
        m, (B, T, _) = self.mem, x.shape
        a = m.address(x)
        v = m._heads(m.wv(x), m.dv)
        alpha, beta = _gates(m.w_alpha, m.w_beta, x)
        o, _, err = _run(a, self._shift(a), v, alpha, beta, None, m.chunk_size)    # chain store S (CA3)
        n = self.novelty(err.detach(), v.detach())                                # CA1 mismatch
        self.last_novelty = n
        eps = self.event_strength(x, self.kappa[None, :, None] * n)
        c = self.goal_context(x)
        ga = torch.exp(-F.softplus(self.g_alpha(x))).transpose(1, 2)
        og, _, _ = _run(c, self._shift(c), v, ga, eps, None, m.chunk_size)        # goal field G
        o = o + torch.sigmoid(self.goal_logit) * og
        o = m.norm(o).transpose(1, 2).reshape(B, T, m.h * m.dv)
        if m.wg is not None:
            o = o * F.silu(m.wg(x))
        return m.wo(o)


class W1jSide(nn.Module):
    """TimeChannel then CA1Loop; the ``SideChannels`` interface used by ``pretrained_loop``."""

    def __init__(self, d_model: int, args: argparse.Namespace, w: argparse.Namespace | None = None) -> None:
        super().__init__()
        w = w or split_args([])[0]
        self.time = TimeChannel(d_model) if getattr(args, "time", True) else None
        self.loop = CA1Loop(d_model, kappa=w.w1j_kappa, baseline=w.w1j_baseline, heads=w.w1i_heads,
                            d_head=w.w1i_dhead, taus=tuple(w.w1i_taus), event_bias=w.w1i_event_bias,
                            refractory=w.w1i_refractory, goal_gain=w.w1i_goal_gain,
                            chain_addressing=w.w1i_chain_addressing, chain_expansion=w.w1i_chain_expansion,
                            budget=w.w1i_budget, budget_window=w.w1i_budget_window)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, LoopStats]:
        if self.time is not None:
            h, _ = self.time(h)
        return self.loop(h)
