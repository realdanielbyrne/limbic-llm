"""Round-1 candidate W1: a Gated DeltaNet side memory with successor-bound keys, plus the clock.

Replaces ``HippocampalLoop`` (an O(T) kernel-weighted read over every stored position) with a
fixed-size associative state (``limbic_llm.gated_delta.GatedDeltaNet``):

    a_t = norm(W_a x_t)                       address of position t (dense, or DG k-WTA codes)
    S_t = α_t S_{t−1}(I − β_t a_{t−1}a_{t−1}ᵀ) + β_t v_t a_{t−1}ᵀ     write: v_t filed under a_{t−1}
    o_t = S_t a_t                             read: "what followed the last context like mine"

Filing each value under the *previous* address (successor binding, RESULTS §3, register M5) is
the shift the horizon runs lacked: there the loop recovered exactly one of four digits (P4a),
because key and value came from the same position. The read at t is injected at t, where the
model predicts token t + 1, through the same CA1 control as the loop: error-only (component
along h removed), norm ``g·‖h‖`` with g ≤ G·σ(gain_logit), saturating. The memory stores the
stream before injection (open loop), so it obeys rule E2 (state independent of T) and E1 (causal).

Run through ``experiments/evolve/run.py --candidate evolve.w1_gdn:W1Side``; extra flags below.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from limbic_llm.gated_delta import GatedDeltaNet
from limbic_llm.temporal_loop import LoopStats, TimeChannel


def split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Take the --w1-* flags; the rest go to ``pretrained_loop``."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--w1-addressing", choices=["tied", "dg"], default="tied")
    ap.add_argument("--w1-heads", type=int, default=8)
    ap.add_argument("--w1-dhead", type=int, default=64)
    ap.add_argument("--w1-key-expansion", type=int, default=1)
    ap.add_argument("--w1-sparsity", type=float, default=0.05)
    ap.add_argument("--w1-decay-bias", type=float, default=-7.0, help="α = exp(−softplus(b)); −7 → α^500 ≈ 0.63")
    ap.add_argument("--w1-beta-bias", type=float, default=0.0, help="β = σ(b) at init (write strength)")
    ap.add_argument("--w1-chunk", type=int, default=64)
    return ap.parse_known_args(argv)


class GDNLoop(nn.Module):
    """Fixed-state CA3 memory at the tap, injected through CA1 control (clamp, projection, saturation)."""

    tau = 1                                                       # reads are usable from the second token

    def __init__(self, d_model: int, addressing: str = "tied", heads: int = 8, d_head: int = 64,
                 key_expansion: int = 1, sparsity: float = 0.05, decay_bias: float = -7.0,
                 beta_bias: float = 0.0, chunk: int = 64, max_gain: float = 0.9) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mem = GatedDeltaNet(d_model, n_heads=heads, d_head=d_head, addressing=addressing,
                                 key_expansion=key_expansion, dg_sparsity=sparsity,
                                 init_decay_bias=decay_bias, chunk_size=chunk)
        nn.init.constant_(self.mem.w_beta.bias, beta_bias)
        self.max_gain = max_gain
        self.gain_logit = nn.Parameter(torch.tensor(-1.0))
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, LoopStats]:
        o, _, _ = self.mem(self.norm(h))                          # (B, T, D), causal
        hn = F.normalize(h, dim=-1)
        e = o - (o * hn).sum(-1, keepdim=True) * hn               # error-only injection
        s = torch.tanh(o.norm(dim=-1) / self.log_lambda.exp())    # saturation
        gain = self.max_gain * torch.sigmoid(self.gain_logit) * s
        rms = h.pow(2).mean(-1, keepdim=True).sqrt()
        inj = gain.unsqueeze(-1) * rms * F.normalize(e, dim=-1) * math.sqrt(h.shape[-1])
        inj = torch.cat([torch.zeros_like(inj[:, :1]), inj[:, 1:]], 1)   # nothing stored before t = 0
        ratio = inj.norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-6)
        return h + inj, LoopStats(ratio, gain)


class W1Side(nn.Module):
    """TimeChannel then GDNLoop; the ``SideChannels`` interface used by ``pretrained_loop``."""

    def __init__(self, d_model: int, args: argparse.Namespace, w: argparse.Namespace | None = None) -> None:
        super().__init__()
        w = w or split_args([])[0]
        self.time = TimeChannel(d_model) if getattr(args, "time", True) else None
        self.loop = GDNLoop(d_model, w.w1_addressing, w.w1_heads, w.w1_dhead, w.w1_key_expansion,
                            w.w1_sparsity, w.w1_decay_bias, w.w1_beta_bias, w.w1_chunk)
        self.w1 = vars(w)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, LoopStats]:
        if self.time is not None:
            h, _ = self.time(h)
        return self.loop(h)
