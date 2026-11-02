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

Round 2 (W1b) gives the address context, so the queried name can select the start of a chain.
With two stored codes W1 recalled digits 2–4 but flipped a coin on digit 1 (P11): its address
at the space after "is" is the same in both facts. Two options:

  --w1-ctx K          addresses from a learned causal depthwise conv over the last K positions,
                      a_t = norm(W_a conv_K(x)_{≤t}); values and gates still come from x_t
                      (initialised to the current position, so K = 1 is W1)
  --w1-addressing gdn GatedDeltaNet's own mode: separate q/k/v with short causal convs and no
                      explicit successor binding (the conv has to learn the shift)

Run through ``experiments/evolve/run.py --candidate evolve.w1_gdn:W1Side``; extra flags below.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from limbic_llm.gated_delta import CausalConv1d, GatedDeltaNet, _gates, _run
from limbic_llm.temporal_loop import LoopStats, TimeChannel


def split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Take the --w1-* flags; the rest go to ``pretrained_loop``."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--w1-addressing", choices=["tied", "dg", "gdn"], default="tied")
    ap.add_argument("--w1-heads", type=int, default=8)
    ap.add_argument("--w1-dhead", type=int, default=64)
    ap.add_argument("--w1-key-expansion", type=int, default=1)
    ap.add_argument("--w1-sparsity", type=float, default=0.05)
    ap.add_argument("--w1-decay-bias", type=float, default=-7.0, help="α = exp(−softplus(b)); −7 → α^500 ≈ 0.63")
    ap.add_argument("--w1-beta-bias", type=float, default=0.0, help="β = σ(b) at init (write strength)")
    ap.add_argument("--w1-chunk", type=int, default=64)
    ap.add_argument("--w1-ctx", type=int, default=1, help="context positions in the address (tied/dg); 1 = W1")
    return ap.parse_known_args(argv)


class GDNLoop(nn.Module):
    """Fixed-state CA3 memory at the tap, injected through CA1 control (clamp, projection, saturation)."""

    tau = 1                                                       # reads are usable from the second token

    def __init__(self, d_model: int, addressing: str = "tied", heads: int = 8, d_head: int = 64,
                 key_expansion: int = 1, sparsity: float = 0.05, decay_bias: float = -7.0,
                 beta_bias: float = 0.0, chunk: int = 64, max_gain: float = 0.9, ctx: int = 1) -> None:
        super().__init__()
        if ctx > 1 and addressing == "gdn":
            raise ValueError("--w1-ctx applies to tied/dg addressing; gdn mode has its own convs")
        self.norm = nn.LayerNorm(d_model)
        self.ctx = CausalConv1d(d_model, ctx) if ctx > 1 else None   # starts as the current position
        self.mem = GatedDeltaNet(d_model, n_heads=heads, d_head=d_head, addressing=addressing,
                                 key_expansion=key_expansion, dg_sparsity=sparsity,
                                 init_decay_bias=decay_bias, chunk_size=chunk)
        nn.init.constant_(self.mem.w_beta.bias, beta_bias)
        self.max_gain = max_gain
        self.gain_logit = nn.Parameter(torch.tensor(-1.0))
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    def _context_read(self, x: torch.Tensor) -> torch.Tensor:
        """GatedDeltaNet's tied/dg path with addresses from the context conv (values from x_t)."""
        m, (B, T, _) = self.mem, x.shape
        a = m.address(self.ctx(x)[0])                             # (B, H, T, Dk), context address
        k = torch.cat([a.new_zeros(B, m.h, 1, m.dk), a[:, :, :-1]], 2)   # key of write t = address of t − 1
        v = m._heads(m.wv(x), m.dv)
        alpha, beta = _gates(m.w_alpha, m.w_beta, x)
        o, _, _ = _run(a, k, v, alpha, beta, None, m.chunk_size)
        o = m.norm(o).transpose(1, 2).reshape(B, T, m.h * m.dv)
        if m.wg is not None:
            o = o * F.silu(m.wg(x))
        return m.wo(o)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, LoopStats]:
        x = self.norm(h)
        o = self._context_read(x) if self.ctx is not None else self.mem(x)[0]   # (B, T, D), causal
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
                            w.w1_sparsity, w.w1_decay_bias, w.w1_beta_bias, w.w1_chunk, ctx=w.w1_ctx)
        self.w1 = vars(w)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, LoopStats]:
        if self.time is not None:
            h, _ = self.time(h)
        return self.loop(h)
