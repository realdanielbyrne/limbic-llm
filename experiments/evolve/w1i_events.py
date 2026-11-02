"""Round-4 candidate W1i: an event-gated goal field (closest to Samsonovich & Ascoli 2005).

S&A store traversed transitions in CA3→CA3 synapses (J) and select where to go with a goal field
in CA3→CA1 synapses (W). The goal field is written *rarely*, at sharp-wave events, in hindsight:
the context that led to a goal is bound to it. Rounds 1–4 showed why "rarely" matters here:
- a goal store keyed by drifting temporal context and written at every token fails, because
  consecutive contexts are nearly identical and the delta rule overwrites;
- W1g's start store learned a positional shortcut (the first fact), not content (P18).

A toy test (3 facts, the middle one queried) found that context keys work when (i) only the first
value of each fact is written, and (ii) the name carries more weight in the context than the
tokens the facts share. Both are learned here:

    b_t = W_c x_t                                   key features; their magnitude is salience
    c_t = norm( Σ_{s≤t} ρ_h^{t−s} b_s )             drifting context (TCM), ρ_h learned
    e_t = σ(w_e · x_t + b_e)                        event detector (starts sparse, b_e = −2)
    r_t = Σ_{s≤t} λ^{t−s}(1−λ) e_s                  recent-event trace
    ε_t = e_t (1 − r_{t−1})                          refractory: no re-write right after an event
    G:  v_t filed under c_{t−1} with strength ε_t, no decay (α ≈ 1, learned)
    o_t = S_t a_t + σ(g) · G_t c_t                  S is W1's chain store, unchanged

G is read with the current context and written with the previous one (hindsight: the context that
led to the event). Fixed state (G, the context and the trace are fixed-size; rule E2), causal
(E1). The kernels are evaluated in parallel in float32. Same CA1 injection as W1.

Run with ``experiments/evolve/run.py --candidate evolve.w1i_events:W1iSide``; flags below.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.w1g_goal import StartLoop
from limbic_llm.gated_delta import _gates, _run
from limbic_llm.temporal_loop import LoopStats, TimeChannel


def split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Take the --w1i-* flags; the rest go to ``pretrained_loop``."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--w1i-heads", type=int, default=5)
    ap.add_argument("--w1i-dhead", type=int, default=64)
    ap.add_argument("--w1i-taus", type=float, nargs="+", default=[2.0, 4.0, 8.0],
                    help="initial context time constants, cycled over heads")
    ap.add_argument("--w1i-event-bias", type=float, default=-2.0, help="initial event-detector bias")
    ap.add_argument("--w1i-refractory", type=float, default=0.7, help="λ of the recent-event trace (0 = off)")
    ap.add_argument("--w1i-goal-gain", type=float, default=0.0, help="initial logit of the G read weight")
    return ap.parse_known_args(argv)


def causal_decay(T: int, rho: torch.Tensor, device) -> torch.Tensor:
    """(H, T, T) kernel ρ_h^{t−s} for s ≤ t, else 0. Index arithmetic in float32."""
    i = torch.arange(T, device=device, dtype=torch.float32)
    d = i[:, None] - i[None, :]
    return torch.where(d >= 0, rho.float()[:, None, None] ** d.clamp_min(0), torch.zeros((), device=device))


class EventLoop(StartLoop):
    """W1's chain store S plus an event-gated, context-keyed goal field G."""

    def __init__(self, d_model: int, heads: int = 5, d_head: int = 64, taus=(2.0, 4.0, 8.0),
                 event_bias: float = -2.0, refractory: float = 0.7, goal_gain: float = 0.0,
                 chunk: int = 64) -> None:
        super().__init__(d_model, "tied", heads, d_head, goal_expansion=1, goal_gain=goal_gain, chunk=chunk)
        H = heads
        rho = torch.tensor([math.exp(-1.0 / taus[i % len(taus)]) for i in range(H)])
        self.rho_logit = nn.Parameter(torch.log(rho / (1 - rho)))
        self.event = nn.Linear(d_model, H)
        nn.init.constant_(self.event.bias, event_bias)
        nn.init.constant_(self.g_alpha.bias, -9.0)                         # no decay to start (α ≈ 1)
        self.refractory = refractory

    def goal_context(self, x: torch.Tensor) -> torch.Tensor:
        """c_t = norm(Σ_{s≤t} ρ^{t−s} b_s), b = W_c x unnormalised (magnitude = salience). (B,H,T,dk)."""
        b = self.mem._heads(self.wc(x), self.gk).float()
        K = causal_decay(x.shape[1], torch.sigmoid(self.rho_logit), x.device)
        return F.normalize(torch.einsum("hts,bhsk->bhtk", K, b), dim=-1)

    def event_strength(self, x: torch.Tensor) -> torch.Tensor:
        """ε_t = e_t (1 − r_{t−1}), (B, H, T)."""
        e = torch.sigmoid(self.event(x)).transpose(1, 2).float()         # (B, H, T)
        if not self.refractory:
            return e
        lam = torch.full((e.shape[1],), self.refractory, device=x.device)
        K = causal_decay(e.shape[2], lam, x.device) * (1 - self.refractory)
        r = torch.einsum("hts,bhs->bht", K, e)
        r_prev = torch.cat([torch.zeros_like(r[..., :1]), r[..., :-1]], -1)
        return e * (1 - r_prev).clamp(0, 1)

    def _read(self, x: torch.Tensor) -> torch.Tensor:
        m, (B, T, _) = self.mem, x.shape
        a = m.address(x)
        v = m._heads(m.wv(x), m.dv)
        alpha, beta = _gates(m.w_alpha, m.w_beta, x)
        o, _, _ = _run(a, self._shift(a), v, alpha, beta, None, m.chunk_size)        # chain store S
        c = self.goal_context(x)
        ga, _ = _gates(self.g_alpha, self.g_beta, x)
        og, _, _ = _run(c, self._shift(c), v, ga, self.event_strength(x), None, m.chunk_size)   # goal field G
        o = o + torch.sigmoid(self.goal_logit) * og
        o = m.norm(o).transpose(1, 2).reshape(B, T, m.h * m.dv)
        if m.wg is not None:
            o = o * F.silu(m.wg(x))
        return m.wo(o)


class W1iSide(nn.Module):
    """TimeChannel then EventLoop; the ``SideChannels`` interface used by ``pretrained_loop``."""

    def __init__(self, d_model: int, args: argparse.Namespace, w: argparse.Namespace | None = None) -> None:
        super().__init__()
        w = w or split_args([])[0]
        self.time = TimeChannel(d_model) if getattr(args, "time", True) else None
        self.loop = EventLoop(d_model, w.w1i_heads, w.w1i_dhead, tuple(w.w1i_taus), w.w1i_event_bias,
                              w.w1i_refractory, w.w1i_goal_gain)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, LoopStats]:
        if self.time is not None:
            h, _ = self.time(h)
        return self.loop(h)
