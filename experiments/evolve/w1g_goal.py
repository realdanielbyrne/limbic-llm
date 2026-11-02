"""Round-3 candidates W1g (a separate start store) and W1h (CA3 settling of its cue).

W1 retrieves a stored code once its first digit is given (the successor chain, Samsonovich &
Ascoli's J), but with 2–3 stored codes it picks the first digit at chance (register P11/P13).
The round-3 probe (``probe_cue.py``) found why: the queried name is linearly decodable at the
space token before the answer (0.95–1.00), but W1's trained addresses there collapse across facts
(cosine 0.79–0.88), because successor binding pushes them toward token identity. Under a delta
rule the shared key is then overwritten, and the store returns the most recent fact.

**W1g (start store).** A second successor store G with its own key projection, so its addresses
are not shaped by the chain. Its keys are dentate-gyrus codes (expansion, then k-WTA):

    a_t = norm(W_a x_t)              S: v_t filed under a_{t−1}     the chain (W1, unchanged)
    b_t = norm(kWTA(W_c x_t))        G: v_t filed under b_{t−1}     the start: keyed on name AND token
    o_t = S_t a_t + σ(g) · G_t b̃_t

The keys must be conjunctive. A linear key adds name and token features, so "space + Name1" and
"space + Name2" still share the space component (cosine 0.5), and under the delta rule the later
fact overwrites half of the earlier. In a unit test with one-hot tokens a linear G key picked the
wrong fact (margin −0.19), while DG codes (2× expansion, 5% active) picked the right one (mean
+0.39 over 8 random projections). That is pattern separation of conjunctions, the DG's job.

Values, head norm, output gate and projection are shared with S. G has its own decay and write
gates. Both stores are fixed-size (rule E2) and causal (E1).

**W1h (CA3 settling, ``--w1g-settle K``).** The cue b_t is settled in an autoassociative store A
(CA3 recurrent collaterals, written b_{t−1} → b_{t−1}, so it holds positions before t only)
before G is read:

    x⁰ = b_t,   x^{k+1} = norm( kWTA( g_rec · A_{t−1} x^k + γ · b_t ) ),   b̃_t = x^K

- **Recurrence:** K reads of A per position with updated queries.
- **Sparse activity gating:** k-WTA on the settling state at every step (feedback inhibition).
- **High gain:** g_rec is learned and starts at 4. γ, the clamped cue, starts at 1.
- **Stability:** divisive normalisation at every step.

A cue slightly biased toward one stored pattern settles onto that pattern (pattern completion).
K = 0 is W1g.

Run with ``experiments/evolve/run.py --candidate evolve.w1g_goal:W1gSide``; flags below.
(An earlier draft keyed G by a drifting temporal context. A unit test showed that consecutive
writes under nearly identical context keys overwrite each other under the delta rule, so it was
replaced before any GPU run.)
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.w1_gdn import GDNLoop
from limbic_llm.gated_delta import _gates, _run, kwta
from limbic_llm.temporal_loop import LoopStats, TimeChannel


def split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Take the --w1g-* flags; the rest go to ``pretrained_loop``."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--w1g-addressing", choices=["tied", "dg"], default="tied", help="the chain store S")
    ap.add_argument("--w1g-heads", type=int, default=5)
    ap.add_argument("--w1g-dhead", type=int, default=64)
    ap.add_argument("--w1g-key-expansion", type=int, default=1)
    ap.add_argument("--w1g-goal-expansion", type=int, default=2, help="DG expansion of G's keys")
    ap.add_argument("--w1g-goal-sparsity", type=float, default=0.05, help="DG k-WTA fraction of G's keys")
    ap.add_argument("--w1g-goal-gain", type=float, default=0.0, help="initial logit of the G read weight")
    ap.add_argument("--w1g-no-goal", action="store_true", help="ablation: S only (W1 with these heads)")
    ap.add_argument("--w1g-settle", type=int, default=0, help="W1h: CA3 settling steps for G's cue (0 = W1g)")
    ap.add_argument("--w1g-settle-gain", type=float, default=4.0, help="W1h: initial recurrent gain")
    ap.add_argument("--w1g-settle-sparsity", type=float, default=0.05, help="W1h: k-WTA fraction of the cue")
    return ap.parse_known_args(argv)


class StartLoop(GDNLoop):
    """W1's chain store S plus a start store G (own keys), optionally with CA3 settling of G's cue."""

    def __init__(self, d_model: int, addressing: str = "tied", heads: int = 5, d_head: int = 64,
                 key_expansion: int = 1, goal_expansion: int = 2, goal_sparsity: float = 0.05,
                 goal_gain: float = 0.0, use_goal: bool = True, settle: int = 0, settle_gain: float = 4.0,
                 settle_sparsity: float = 0.05, chunk: int = 64) -> None:
        super().__init__(d_model, addressing, heads, d_head, key_expansion, chunk=chunk)
        H = heads
        self.gk, self.goal_sparsity = d_head * goal_expansion, goal_sparsity
        self.wc = nn.Linear(d_model, H * self.gk, bias=False)             # G's own key projection (DG)
        self.g_alpha = nn.Linear(d_model, H)                               # G's decay and write gates
        self.g_beta = nn.Linear(d_model, H)
        nn.init.constant_(self.g_alpha.bias, -7.0)
        self.goal_logit = nn.Parameter(torch.tensor(goal_gain))
        self.use_goal = use_goal
        self.settle, self.settle_sparsity = settle, settle_sparsity
        if settle:
            self.rec_gain = nn.Parameter(torch.tensor(settle_gain))       # high recurrent gain
            self.cue_gain = nn.Parameter(torch.tensor(1.0))               # clamped cue (EC → CA3)

    @staticmethod
    def _shift(z: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.zeros_like(z[:, :, :1]), z[:, :, :-1]], 2)

    def settle_cue(self, b: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """CA3 settling of the cue b (B, H, T, Dk) in an autoassociative store of positions < t."""
        prev = self._shift(b)                                              # A writes b_{t−1} → b_{t−1}
        x = b
        for _ in range(self.settle):
            r, _, _ = _run(x, prev, prev, alpha, beta, None, self.mem.chunk_size)   # A_{t−1} x
            x = F.normalize(kwta(self.rec_gain * r + self.cue_gain * b, self.settle_sparsity), dim=-1)
        return x

    def goal_keys(self, x: torch.Tensor) -> torch.Tensor:
        """DG codes for G: expansion, k-WTA per head, unit norm. (B, H, T, gk)."""
        return F.normalize(kwta(self.mem._heads(self.wc(x), self.gk), self.goal_sparsity), dim=-1)

    def _read(self, x: torch.Tensor) -> torch.Tensor:
        m, (B, T, _) = self.mem, x.shape
        a = m.address(x)                                                   # (B, H, T, Dk)
        v = m._heads(m.wv(x), m.dv)
        alpha, beta = _gates(m.w_alpha, m.w_beta, x)
        o, _, _ = _run(a, self._shift(a), v, alpha, beta, None, m.chunk_size)       # chain store S
        if self.use_goal:
            b = self.goal_keys(x)
            ga, gb = _gates(self.g_alpha, self.g_beta, x)
            q = self.settle_cue(b, ga, gb) if self.settle else b
            og, _, _ = _run(q, self._shift(b), v, ga, gb, None, m.chunk_size)       # start store G
            o = o + torch.sigmoid(self.goal_logit) * og
        o = m.norm(o).transpose(1, 2).reshape(B, T, m.h * m.dv)
        if m.wg is not None:
            o = o * F.silu(m.wg(x))
        return m.wo(o)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, LoopStats]:
        o = self._read(self.norm(h))
        hn = F.normalize(h, dim=-1)
        e = o - (o * hn).sum(-1, keepdim=True) * hn                        # error-only injection (CA1)
        s = torch.tanh(o.norm(dim=-1) / self.log_lambda.exp())
        gain = self.max_gain * torch.sigmoid(self.gain_logit) * s
        rms = h.pow(2).mean(-1, keepdim=True).sqrt()
        inj = gain.unsqueeze(-1) * rms * F.normalize(e, dim=-1) * math.sqrt(h.shape[-1])
        inj = torch.cat([torch.zeros_like(inj[:, :1]), inj[:, 1:]], 1)
        ratio = inj.norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-6)
        return h + inj, LoopStats(ratio, gain)


class W1gSide(nn.Module):
    """TimeChannel then StartLoop; the ``SideChannels`` interface used by ``pretrained_loop``."""

    def __init__(self, d_model: int, args: argparse.Namespace, w: argparse.Namespace | None = None) -> None:
        super().__init__()
        w = w or split_args([])[0]
        self.time = TimeChannel(d_model) if getattr(args, "time", True) else None
        self.loop = StartLoop(d_model, w.w1g_addressing, w.w1g_heads, w.w1g_dhead, w.w1g_key_expansion,
                              w.w1g_goal_expansion, w.w1g_goal_sparsity, w.w1g_goal_gain, not w.w1g_no_goal,
                              w.w1g_settle, w.w1g_settle_gain, w.w1g_settle_sparsity)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, LoopStats]:
        if self.time is not None:
            h, _ = self.time(h)
        return self.loop(h)
