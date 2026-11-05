"""Round-10 candidate W1s: episodic slots that keep whole windows, read by lexical and learned keys.

W1i and W1j wrote each fact into a superposed delta-rule store and learned content selection in
only 1–3 of 5 seeds (P36, P40, P46–P48). The round-10 merit tests (NOTEBOOK, 3 Nov 2026) put the
barrier in superposition, twice over: within a fact, consecutive digits overwrite each other; between
facts, keys taken at the space after "is", where the name has faded, collide. W1s keeps each event's
whole window in a slot of its own, in order, and reads it by offset:

    gate     d_t = w_g·(x_t − m_t)                x = [LN(h³); LN(h²¹)], m_t its causal running mean
             point    p_t = σ(a·d_t / sd_t + b)        sd_t: causal running deviation of w_g·x
             hyst     u_t = Σ_{s≤t} λ^{t−s}(1−λ) d_s,  z_t = u_t / sd_t(u),
                      g_t = σ(a·z_t + b + c(g_{t−1} − ½)),  p_t = g_t(1 − g_{t−1})
             schmitt  the hard form of hyst: on at z_t ≥ θ_on, off at z_t < θ_off; p_t = σ(4(z_t − θ_on))
                      where it switches on (no gradient through the state, only through p)
             oracle   p = 1 at each fact's first digit (``side.oracle_onsets``; diagnosis only, breaks E4)
             dense    p = σ(b) at every position (ablation: no event selection)
    windows  onsets are the strict local maxima of p over ±r; onset s keeps [s−P, s−P+n−1] and closes
             (becomes readable) at c_s = s + max(r, n−P−1), once both its suppression and its window
             are known; with the hysteresis gate, entries from s on are weighted by the on-state g
    store    slots: onset s has the fixed priority π_s = log p_s + δ·s, and the store at t holds the top
             K by π among the windows closed by t. That is exactly a streaming store of K slots that
             evicts the lowest priority (``stream``): δ = 0 keeps the strongest events, large δ is FIFO.
             superposed (ablation): the same keys and values written into one delta-rule state
             (``limbic_llm.gated_delta._run``), each window when it closes
    keys     lex      mean of LN(h³) over the window's first 8 entries; the query is the mean over the
                      last 8 tokens; centred cosine, centred on the retained keys (no fitted parameters)
             learned  W_k [LN(h³); LN(h²¹)] → 128, one map for keys and queries
    read     a_t = softmax over the retained slots + null of β_lex·ĉos_lex + β_L·cos_L + log p_k − ω·age_k
             (β learned, from 30 and 10). Inside slot k, α = softmax_j β_a cos(z_t, z_j) aligns the
             current token with a stored one, and the value read is the one stored at j + 1
    output   additive   v = A·LN(h²¹) ∈ ℝ³² is stored, o = B·r goes through W1's CA1 injection
             reinstate  attention at layers ≥ L sees the retained windows' columns with the bias
                        log Σ_k a_{t,k} 1[c ∈ window_k] + b (``reinstate_bias``, applied by
                        ``taps.TapLM``); the gate, keys and read run on the taps, and the tap-21
                        output is h unchanged

The drive is divided by its causal running deviation (the plan has w_g·(x_t − m_t) + b_g): a latch
or a hard threshold means something only against a known noise scale, and with it the initial
density is 0.12 ± 0.05 at any input scale, with no fitting (``gate_bias``). ``--no-w1s-centre``
drops the mean subtraction only.

The gate gets its gradient through log p_k in the read softmax (and through the window weights of
the hysteresis gate); retention and suppression are hard, which is what keeps the parallel form
equal to the streaming store. At initialisation the gate fires at about 0.12 per token (P35), so
there are about 30 candidate windows per 512 tokens against K = 16 and the facts' windows are
retained from the first step. The parallel form uses (B, T, T) weight matrices and matmuls, never a
(B, T, K, D) tensor.

Rules. E1: every quantity at t uses positions ≤ t (a window is read only once closed). E2: the
streaming state is K slots of n entries, a ring buffer of the positions not yet closed (c_s − s +
max(P, r) + 1) and the gate's scalars, whatever T (``stream`` returns its size: 0.89M floats at width
2048 with K = 16). E3: 0.74M parameters at width 2048 (the combination C).

Taps: ``W1sSide.taps`` (block indices, default 3) are read by ``taps.TapLM``, which sets
``side.extra = {i: h_i}`` before each call. Without ``extra`` (unit tests, the CPU suite) every tap
falls back to h. ``--w1s-key-taps`` lists the streams of the gate and the learned key: tap indices,
plus any other index (21) for the main tap h; the lexical key uses the lowest.

Run with ``experiments/evolve/run.py --candidate evolve.w1s_slots:W1sSide``; flags below.
"""

from __future__ import annotations

import argparse
import functools
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.w1_gdn import GDNLoop
from evolve.w1i_events import causal_decay
from limbic_llm.gated_delta import _run
from limbic_llm.temporal_loop import LoopStats, TimeChannel

TAPS_REQUIRED = True                                              # run.py builds taps.TapLM for this module
GATES = ("point", "hyst", "schmitt", "oracle", "dense")
NEG = -1e4                                                        # reinstatement bias: nothing reinstated
EPS = 1e-8                                                        # floor of squared norms in the cosines
SCHMITT_GAIN = 4.0                                                # p = σ(4(z − θ_on)) at a Schmitt onset
MAX_BETA = 40.0                                                   # exp(β(cos − 1)) stays above float32's floor
ALIGN_FLOOR = 1e-12                                               # a slot's alignment normaliser (E shifted to ≤ 1)
RECENCY_FORMS = ("linear", "none", "log", "ord", "supr", "supw")


def _ints(s: str) -> tuple[int, ...]:
    s = s.strip().lower()
    return () if s in ("", "none") else tuple(int(x) for x in s.split(","))


def _opt(w: argparse.Namespace, name: str, default):
    """A round-11 flag, or its default for namespaces built before it existed."""
    v = getattr(w, name, default)
    return default if v is None and default is not None else v


def rope_freqs(dims: int, periods: str) -> torch.Tensor:
    """Angular frequencies (float64) of the rotated pairs: periods log-spaced from the shortest to the longest."""
    lo, hi = (float(x) for x in periods.split(","))
    if dims % 2 or dims < 2:
        raise ValueError("--w1s-rope-dims must be even and ≥ 2")
    per = torch.logspace(math.log10(lo), math.log10(hi), dims // 2, dtype=torch.float64)
    return 2 * math.pi / per


def rotate(x: torch.Tensor, pos: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Partial rotary encoding: pairs (2f, 2f+1) of the first 2·len(freqs) dims of x (..., d) turn by
    pos·ω_f; the other dims are unchanged. pos (...) holds integer positions; angles are float64 (E5)."""
    R = 2 * len(freqs)
    ang = pos.to(torch.float64)[..., None] * freqs.to(pos.device)
    c, s_ = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0:R:2], x[..., 1:R:2]
    rot = torch.stack([x1 * c - x2 * s_, x1 * s_ + x2 * c], -1).flatten(-2)
    return torch.cat([rot, x[..., R:]], -1)


def newer_sum(x: torch.Tensor) -> torch.Tensor:
    """Σ_{k > i} x[..., k] along the last dim (candidates in time order: the mass of newer ones)."""
    return F.pad(x.flip(-1).cumsum(-1).flip(-1)[..., 1:], (0, 1))


def split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Take the --w1s-* flags; the rest go to ``pretrained_loop``."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--w1s-taps", default="3", help="early blocks read through taps.TapLM (comma list, '' = none)")
    ap.add_argument("--w1s-gate", choices=GATES, default="hyst")
    ap.add_argument("--w1s-centre", action=argparse.BooleanOptionalAction, default=True,
                    help="subtract the causal running mean of the gate input")
    ap.add_argument("--w1s-density", type=float, default=0.12, help="initial firing density of the gate")
    ap.add_argument("--w1s-lam", type=float, default=0.8, help="initial λ of the hyst/schmitt integrator")
    ap.add_argument("--w1s-latch", type=float, default=6.0, help="initial latch strength c of the hyst gate")
    ap.add_argument("--w1s-schmitt", type=float, nargs=2, default=[1.2, 0.8], metavar=("ON", "OFF"),
                    help="Schmitt thresholds in running standard deviations of u")
    ap.add_argument("--w1s-pre", type=int, default=6, help="P: entries recorded before the onset")
    ap.add_argument("--w1s-win", type=int, default=24, help="n: entries per window")
    ap.add_argument("--w1s-nms", type=int, default=8, help="r: non-maximum suppression radius")
    ap.add_argument("--w1s-store", choices=["slots", "superposed"], default="slots")
    ap.add_argument("--w1s-K", type=int, default=16, help="slots")
    ap.add_argument("--w1s-delta", type=float, default=0.0, help="δ in π_s = log p_s + δ·s (large = FIFO)")
    ap.add_argument("--w1s-key", choices=["lex", "learned", "both"], default="both")
    ap.add_argument("--w1s-key-taps", default=None,
                    help="streams of the gate and learned key (default: the taps, plus 21 = h when additive)")
    ap.add_argument("--w1s-pool", type=int, default=8, help="entries pooled into a key, and tokens into a query")
    ap.add_argument("--w1s-dk", type=int, default=128, help="learned key width")
    ap.add_argument("--w1s-dv", type=int, default=32, help="stored value width (additive output)")
    ap.add_argument("--w1s-align", choices=["lex", "learned", "none"], default="lex",
                    help="offset alignment inside a slot (none: the window's values averaged)")
    ap.add_argument("--w1s-recency", type=float, default=0.002, help="initial ω per token of age (0 = off)")
    ap.add_argument("--w1s-out", choices=["additive", "reinstate"], default="additive")
    ap.add_argument("--w1s-L", type=int, default=4, help="reinstate: first attention layer that sees the slots")
    ap.add_argument("--w1s-ctrl", choices=["none", "empty", "shuffle"], default="none",
                    help="evaluation controls: store forced empty, or slots taken from another batch row")
    # round 11 (research/plans/round11_runbook.md §3); every default reproduces round 10
    ap.add_argument("--w1s-recency-form", choices=RECENCY_FORMS, default=None,
                    help="age term of the read: linear ω·age (round 10), none, log ω·log(1+age), ord ω·#newer "
                         "slots, supr (read-time supersession), supw (write-time supersession tags). Default: "
                         "linear, or none with a --w1s-time-feat")
    ap.add_argument("--w1s-time-feat", choices=["none", "rope", "ordinal"], default="none",
                    help="time as data in the learned-key score: rotary slot time, or a query-chosen ordinal rank")
    ap.add_argument("--w1s-rope-dims", type=int, default=32, help="rope: rotated dims of the learned key and query")
    ap.add_argument("--w1s-rope-periods", default="64,65536", help="rope: shortest,longest period (log-spaced)")
    ap.add_argument("--w1s-ord-dim", type=int, default=16, help="ordinal: width of the rank embedding")
    ap.add_argument("--w1s-sup-w", type=float, default=3.0, help="supr: initial w_s (logits per unit of newer mass)")
    ap.add_argument("--w1s-sup-gamma", type=float, default=40.0, help="supw: initial γ (sharpness of the match)")
    ap.add_argument("--w1s-sup-theta", type=float, default=0.9, help="supw: initial θ (key cosine of a match)")
    ap.add_argument("--w1s-sup-omega", type=float, default=3.0, help="supw: initial ω_s (logits per unit of tag)")
    ap.add_argument("--w1s-lex-init", type=float, default=None, help="initial log β_lex (default log 30)")
    ap.add_argument("--w1s-align-init", type=float, default=None,
                    help="initial log β_align (default log 30 for lex alignment, log 10 for learned)")
    ap.add_argument("--w1s-code", choices=["lex", "learned"], default="lex",
                    help="alignment code: the lexical state (centred cosine) or z = W_z·x (plain cosine)")
    ap.add_argument("--w1s-dc", type=int, default=64, help="learned alignment code width")
    ap.add_argument("--w1s-keep", type=int, default=0, help="keep M transitions per window (0 = all)")
    ap.add_argument("--w1s-keep-by", choices=["first", "value", "learned", "external"], default="first",
                    help="which M: the first, top value norm ‖v_{j+1}‖, a learned score w·x_{j+1} + b of the "
                         "successor (like the value norm, it scores the token whose value is stored), or scores the "
                         "caller sets (side.keep_scores / side.keep_pair; evaluation diagnostics only)")
    ap.add_argument("--w1s-keep-init", choices=["zero", "value"], default="zero",
                    help="learned keep: start from 0 (w·x + b only) or from the value norm (‖v_{j+1}‖ + w·x + b; G0.4)")
    ns, rest = ap.parse_known_args(argv)
    if ns.w1s_recency_form is None:                               # resolved here so the JSON records it
        ns.w1s_recency_form = "linear" if ns.w1s_time_feat == "none" else "none"
    return ns, rest


# ----------------------------------------------------------------------- helpers
def window_mean(x: torch.Tensor, lo: int, n: int) -> torch.Tensor:
    """Mean of x over positions [t+lo, t+lo+n−1] ∩ [0, T) for every t. (B, T, D); n exact terms each."""
    B, T, D = x.shape
    pl_, pr = max(0, -lo), max(0, lo + n - 1)
    w = F.pad(x, (0, 0, pl_, pr)).unfold(1, n, 1)[:, lo + pl_:lo + pl_ + T]   # (B, T, D, n) view
    t = torch.arange(T, device=x.device)
    cnt = ((t + lo + n).clamp(max=T) - (t + lo).clamp(min=0)).clamp(min=1)
    return w.sum(-1) / cnt.to(x.dtype)[None, :, None]


def cummean(x: torch.Tensor) -> torch.Tensor:
    """Causal running mean along dim 1 (the current position included)."""
    n = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)
    return x.cumsum(1) / n.view(1, -1, *[1] * (x.dim() - 2))


def strict_peaks(p: torch.Tensor, r: int) -> torch.Tensor:
    """True where p_t exceeds every other p within ±r (ties suppress both). (B, T) bool."""
    if r == 0:
        return torch.ones_like(p, dtype=torch.bool)
    T = p.shape[1]
    w = F.pad(p, (r, r), value=float("-inf")).unfold(1, r, 1)
    return (p > w[:, :T].amax(-1)) & (p > w[:, r + 1:r + 1 + T].amax(-1))


def running_sd(u: torch.Tensor) -> torch.Tensor:
    """Causal running standard deviation of u (B, T), about its running mean."""
    m, m2 = cummean(u), cummean(u * u)
    return ((m2 - m * m).clamp_min(0) + 1e-6).sqrt()


def schmitt(z: torch.Tensor, on: float, off: float) -> torch.Tensor:
    """Hard hysteresis in parallel: on from the last z ≥ on until the next z < off (on ≥ off). (B, T) bool.

    The state is the kind of the most recent threshold crossing: a cummax over crossing indices."""
    idx = torch.arange(z.shape[-1], device=z.device).expand_as(z)
    up, dn = z >= on, z < off
    last = torch.where(up | dn, idx, torch.full_like(idx, -1)).cummax(-1).values
    return up.gather(-1, last.clamp_min(0)) & (last >= 0)


def schmitt_loop(z: torch.Tensor, on: float, off: float) -> torch.Tensor:
    """The sequential definition of ``schmitt`` (reference for the tests and ``stream``)."""
    s, out = torch.zeros_like(z[..., 0], dtype=torch.bool), []
    for t in range(z.shape[-1]):
        s = (z[..., t] >= on) | (s & ~(z[..., t] < off))
        out.append(s)
    return torch.stack(out, -1)


class _Latch(torch.autograd.Function):
    """g_t = σ(u_t + c(g_{t−1} − ½)), g_{−1} = 0, for u (B, T): the only sequential part of W1s.

    Both loops run on the CPU, T steps over B values each; the backward pass is the reverse loop
    δ_t = g_t(1 − g_t)(∂L/∂g_t + c δ_{t+1}), ∂L/∂u_t = δ_t, ∂L/∂c = Σ_t δ_t (g_{t−1} − ½)."""

    @staticmethod
    def forward(ctx, u, c):
        uc, cc = u.detach().cpu(), float(c)
        g, prev = torch.empty_like(uc), uc.new_zeros(uc.shape[0])
        for t in range(uc.shape[1]):
            prev = torch.sigmoid(uc[:, t] + cc * (prev - 0.5))
            g[:, t] = prev
        ctx.g, ctx.cc, ctx.dev, ctx.cdev = g, cc, u.device, c.device
        return g.to(u.device)

    @staticmethod
    def backward(ctx, dg):
        g, cc, dg = ctx.g, ctx.cc, dg.detach().cpu()
        dz, nxt = torch.empty_like(g), g.new_zeros(g.shape[0])
        for t in range(g.shape[1] - 1, -1, -1):
            nxt = g[:, t] * (1 - g[:, t]) * (dg[:, t] + cc * nxt)
            dz[:, t] = nxt
        dc = (dz * (F.pad(g[:, :-1], (1, 0)) - 0.5)).sum()
        return dz.to(ctx.dev), dc.to(ctx.cdev)


def latch(u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return _Latch.apply(u, c)


@functools.lru_cache(maxsize=None)
def gate_bias(gate: str, rho: float, c: float = 0.0, lam: float = 0.8) -> float:
    """Initial gate bias b_g for firing density ρ. The drive is standardised (unit running deviation),
    so this needs no data: point, E σ(z + b) ≈ σ(b / √(1 + π/8)) for z ~ N(0, 1); hyst, a bisection
    on the latch driven by a unit-variance AR(λ) process (the leaky integral of a white drive),
    deterministic (fixed seed). Mean on-state ρ: −0.8 for ρ = 0.12, c = 6, λ = 0.8."""
    logit = math.log(rho / (1 - rho))
    if gate == "point":
        return logit * math.sqrt(1 + math.pi / 8)
    if gate != "hyst":
        return logit
    e = torch.randn(128, 512, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    z = e.clone()
    for t in range(1, z.shape[1]):
        z[:, t] = lam * z[:, t - 1] + math.sqrt(1 - lam * lam) * e[:, t]
    lo, hi = -10.0, 10.0
    for _ in range(25):
        b = (lo + hi) / 2
        prev, tot = torch.zeros(z.shape[0], dtype=z.dtype), 0.0
        for t in range(z.shape[1]):
            prev = torch.sigmoid(z[:, t] + b + c * (prev - 0.5))
            tot += prev.mean().item()
        lo, hi = (lo, b) if tot / z.shape[1] > rho else (b, hi)
    return (lo + hi) / 2


def check_ctrl(ctrl: str, out: str) -> None:
    """Reject a control that cannot mean what it says. Reinstatement reopens columns of the row's own
    sequence, so another row's windows would still reinstate this row's tokens (often its own facts:
    41% of queried first digits on the scattered layout); ``empty`` is its content control."""
    if ctrl not in ("none", "empty", "shuffle"):
        raise ValueError(ctrl)
    if ctrl == "shuffle" and out == "reinstate":
        raise ValueError("--w1s-ctrl shuffle is not a control for --w1s-out reinstate (use empty)")


def _rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x[b, idx[b, ...]] for x (B, T, ...) and integer idx (B, ...)."""
    b = torch.arange(x.shape[0], device=x.device).view(-1, *[1] * (idx.dim() - 1))
    return x[b, idx]


# ----------------------------------------------------------------------- memory
class SlotLoop(GDNLoop):
    """Gate → windows → K slots (or a superposed state) → keyed read with offset alignment.

    A ``GDNLoop`` for its CA1 injection (``forward``, ``gain_logit``, ``tau``): its read is replaced
    by ``_context_read``, and its GatedDeltaNet is dropped. ``streams`` names the gate's and the
    learned key's inputs (tap indices, ``"h"`` for the main tap) and ``lex`` the lexical stream.
    After every call ``last_diag`` holds detached tensors: ``p`` (B, T) onset strength, ``g`` the
    on-state, ``onsets`` (B, T) candidates, ``retained`` (B, T, K) onset positions of the slots held
    at t (most recent first, −1 = empty), ``read_w`` (B, T, K + 1) their read weights with the null
    last, and ``align`` (B, T) the stored position whose value was read at t, in the slot with the
    largest weight (−1 when that is the null, or without alignment). With the superposed store,
    ``retained`` lists the K most recent closed windows and ``read_w`` is zero.
    """

    def __init__(self, d_model: int, streams: list, lex, w: argparse.Namespace) -> None:
        super().__init__(d_model, heads=1, d_head=1)
        del self.mem
        self.ctx = nn.Identity()                                  # GDNLoop.forward reads through _context_read
        if w.w1s_win < 2 or not 0 <= w.w1s_pre < w.w1s_win:
            raise ValueError("--w1s-win must be ≥ 2 and 0 ≤ --w1s-pre < --w1s-win")
        if w.w1s_store == "superposed" and (w.w1s_out == "reinstate" or w.w1s_gate == "dense"):
            raise ValueError("the superposed store has no windows to reinstate, and needs a sparse gate")
        check_ctrl(w.w1s_ctrl, w.w1s_out)
        self.streams, self.lex = list(streams), lex
        self.gate, self.centre, self.store = w.w1s_gate, w.w1s_centre, w.w1s_store
        self.P, self.n, self.r = w.w1s_pre, w.w1s_win, 0 if w.w1s_gate == "dense" else w.w1s_nms
        self.delay = max(self.r, self.n - self.P - 1)              # c_s − s
        self.K, self.delta, self.pool = w.w1s_K, w.w1s_delta, min(w.w1s_pool, w.w1s_win)
        self.key, self.align, self.out, self.ctrl = w.w1s_key, w.w1s_align, w.w1s_out, w.w1s_ctrl
        self.use_lex, self.use_learned = self.key in ("lex", "both"), self.key in ("learned", "both")
        self.thresholds = tuple(w.w1s_schmitt)
        M, rho = len(streams), w.w1s_density
        c = w.w1s_latch if self.gate == "hyst" else 0.0

        def split_linear(d_out):                                  # one Linear on the concatenated streams
            lins = nn.ModuleList(nn.Linear(d_model, d_out, bias=False) for _ in streams)
            with torch.no_grad():
                for lin in lins:
                    lin.weight.mul_(1 / math.sqrt(M))
            return lins

        self.gate_w = split_linear(1)
        self.gate_a = nn.Parameter(torch.tensor(0.0))             # log gain on the standardised drive
        self.gate_b = nn.Parameter(torch.tensor(gate_bias(self.gate, rho, c, w.w1s_lam)))   # density ρ at init
        if self.gate in ("hyst", "schmitt"):
            self.lam_logit = nn.Parameter(torch.tensor(math.log(w.w1s_lam / (1 - w.w1s_lam))))
        if self.gate == "hyst":
            self.latch_c = nn.Parameter(torch.tensor(float(c)))
        self.wk = split_linear(w.w1s_dk) if self.use_learned or self.align == "learned" else None
        lex_init, align_init = _opt(w, "w1s_lex_init", None), _opt(w, "w1s_align_init", None)
        lex0 = 30.0 if lex_init is None else math.exp(lex_init)
        self.log_beta_lex = (nn.Parameter(torch.tensor(math.log(30.0) if lex_init is None else float(lex_init)))
                             if self.use_lex else None)
        self.log_beta_learned = nn.Parameter(torch.tensor(math.log(10.0))) if self.use_learned else None
        self.log_beta_align = (nn.Parameter(torch.tensor(math.log(30.0 if self.align == "lex" else 10.0)
                                                         if align_init is None else float(align_init)))
                               if self.align != "none" else None)
        # null threshold: a centred lexical cosine of 0.3. On frozen-Qwen layer-3 states (merit captures,
        # oracle onsets, 3 facts + 12 false windows) the queried fact's window scores 0.53 on average
        # (10th percentile 0.43) and the best other window 0.45: a higher null would win at initialisation.
        # It scales with the initial β_lex, so a lexical key started near off (--w1s-lex-init) is not outvoted.
        self.null_logit = nn.Parameter(torch.tensor(0.3 * (lex0 * self.use_lex + 10.0 * self.use_learned)))
        self.log_omega = nn.Parameter(torch.tensor(math.log(w.w1s_recency))) if w.w1s_recency > 0 else None
        if self.out == "additive":
            self.wa = nn.Linear(d_model, w.w1s_dv, bias=False)    # v = A LN(h²¹)
            self.wb = nn.Linear(w.w1s_dv, d_model, bias=False)    # o = B r
        else:
            self.rbias = nn.Parameter(torch.tensor(0.0))
        self._round11(d_model, w)                                 # after every round-10 module (same init)
        self._in = None
        self.last_diag: dict = {}

    def _round11(self, d_model: int, w: argparse.Namespace) -> None:
        """Round-11 options (time as data, supersession, compression). Each creates its parameters only
        when it is on, so round-10 checkpoints load with no missing keys under the defaults."""
        tf = _opt(w, "w1s_time_feat", "none")
        rf = _opt(w, "w1s_recency_form", None) or ("linear" if tf == "none" else "none")
        self.rform, self.time_feat = rf, tf
        self.code, self.keep, self.keep_by = _opt(w, "w1s_code", "lex"), _opt(w, "w1s_keep", 0), _opt(w, "w1s_keep_by", "first")
        self.keep_init = _opt(w, "w1s_keep_init", "zero")
        if self.keep_init != "zero" and self.keep_by != "learned":
            raise ValueError("--w1s-keep-init applies to --w1s-keep-by learned")
        if rf not in RECENCY_FORMS or tf not in ("none", "rope", "ordinal"):
            raise ValueError(f"--w1s-recency-form {rf} / --w1s-time-feat {tf}")
        slots_add = self.store == "slots" and self.out == "additive"
        if self.store == "superposed" and (tf != "none" or rf not in ("linear", "none")):
            raise ValueError("--w1s-time-feat and --w1s-recency-form act on the slot read; the superposed store "
                             "reads no slots")
        if self.keep_by != "first" and not self.keep:
            raise ValueError(f"--w1s-keep-by {self.keep_by} needs --w1s-keep M > 0")
        if tf != "none" and not self.use_learned:
            raise ValueError("--w1s-time-feat acts on the learned key: needs --w1s-key learned or both")
        if rf == "supw" and not (self.use_learned or self.use_lex):
            raise ValueError("supw needs keys")
        if self.code == "learned" and not (slots_add and self.align == "lex"):
            raise ValueError("--w1s-code learned replaces the lexical alignment code of additive slots "
                             "(--w1s-store slots --w1s-out additive --w1s-align lex)")
        if self.keep and not slots_add:
            raise ValueError("--w1s-keep needs additive slots")
        if self.keep < 0 or self.keep >= self.n:
            raise ValueError("--w1s-keep must be in [0, --w1s-win)")
        self.keep_scores = self.keep_pair = None                  # external keep scores, set by the caller
        if rf == "supr":
            self.log_sup_w = nn.Parameter(torch.tensor(math.log(_opt(w, "w1s_sup_w", 3.0))))
        if rf == "supw":
            self.log_sup_gamma = nn.Parameter(torch.tensor(math.log(_opt(w, "w1s_sup_gamma", 40.0))))
            self.sup_theta = nn.Parameter(torch.tensor(float(_opt(w, "w1s_sup_theta", 0.9))))
            self.log_sup_omega = nn.Parameter(torch.tensor(math.log(_opt(w, "w1s_sup_omega", 3.0))))
        if tf == "rope":
            dk = self.wk[0].out_features
            self.rope_dims = min(_opt(w, "w1s_rope_dims", 32), dk - dk % 2)
            self.rope_periods = _opt(w, "w1s_rope_periods", "64,65536")
            rope_freqs(self.rope_dims, self.rope_periods)          # validate now
        if tf == "ordinal":
            de = _opt(w, "w1s_ord_dim", 16)
            self.ord_E = nn.Parameter(torch.zeros(self.K, de))    # rank 0 = newest … K − 1
            self.ord_u = nn.Linear(self.wk[0].out_features, de, bias=False)
            with torch.no_grad():
                self.ord_u.weight.normal_(0, 0.1)
        if self.code == "learned":
            self.wz = nn.Linear(d_model, _opt(w, "w1s_dc", 64), bias=False)
        if self.keep and self.keep_by == "learned":
            self.keep_w = nn.Linear(d_model, 1)
            with torch.no_grad():
                self.keep_w.weight.normal_(0, 0.02)
                self.keep_w.bias.fill_(0.0)
        # the lexical stream is kept (ring, slot keys, centring) only where something lexical reads it
        self.need_x = self.use_lex or (self.align == "lex" and self.code == "lex") or self.store == "superposed"

    # ------------------------------------------------------------ features
    def _streams(self, h, extra) -> dict:
        """LayerNorm (no affine) of each stream read: tap i from ``extra`` (else h), "h" = the main tap."""
        xs = {}
        for m in dict.fromkeys(self.streams + [self.lex]):
            src = h if m == "h" else (extra or {}).get(m, h)
            if src is None:
                raise ValueError(f"tap {m} is missing from extra")
            xs[m] = F.layer_norm(src, src.shape[-1:])
        return xs

    def _gate(self, xs: dict, oracle) -> tuple[torch.Tensor, torch.Tensor]:
        """Onset strength p and on-state g, (B, T) each."""
        x0 = xs[self.lex]
        (B, T), dt = x0.shape[:2], x0.dtype
        if self.gate == "oracle":
            if oracle is None:
                raise ValueError("--w1s-gate oracle needs side.oracle_onsets (B, T)")
            p = oracle.to(x0.device, dt)
            return p, p
        if self.gate == "dense":
            p = torch.sigmoid(self.gate_b).to(dt).expand(B, T)
            return p, p
        y = sum(lin(xs[m]).squeeze(-1) for lin, m in zip(self.gate_w, self.streams))
        d = y - cummean(y) if self.centre else y                  # w·(x_t − m_t) = w·x_t − mean_{s≤t} w·x_s
        if self.gate == "point":
            p = torch.sigmoid(self.gate_a.exp() * d / running_sd(y) + self.gate_b)
            return p, p
        lam = torch.sigmoid(self.lam_logit)
        u = d @ (causal_decay(T, lam[None], d.device)[0].to(dt) * (1 - lam)).T   # leaky integrator
        z = u / running_sd(u)
        if self.gate == "hyst":
            g = latch(self.gate_a.exp() * z + self.gate_b, self.latch_c)
            return g * (1 - F.pad(g[:, :-1], (1, 0))), g
        on = schmitt(z, *self.thresholds)
        rise = on & ~F.pad(on[:, :-1], (1, 0))
        return rise.to(dt) * torch.sigmoid(SCHMITT_GAIN * (z - self.thresholds[0])), on.to(dt)

    def _features(self, x, h, extra, oracle) -> dict:
        """Per-position features: the lexical stream (detached: no fitted parameters), learned codes y,
        values v (additive), and the gate."""
        xs = self._streams(h, extra)
        f = {"x": xs[self.lex].detach()}
        f["y"] = sum(lin(xs[m]) for lin, m in zip(self.wk, self.streams)) if self.wk is not None else None
        f["v"] = self.wa(x) if x is not None and self.out == "additive" else None
        f["p"], f["g"] = self._gate(xs, oracle)
        if self.code == "learned":                                # z = W_z x, the alignment code
            f["z"] = self.wz(f["x"])
        if self.keep:                                             # each transition j → j+1's keep score
            if self.keep_by == "external":
                if self.keep_scores is None and self.keep_pair is None:
                    raise ValueError("--w1s-keep-by external needs side.keep_scores or side.keep_pair")
                f["ks"] = self.keep_scores.to(f["x"]) if self.keep_scores is not None else None
                f["kp"] = self.keep_pair.to(f["x"]) if self.keep_pair is not None else None
            elif self.keep_by in ("value", "learned"):            # scored at the successor j + 1
                sc = f["v"].norm(dim=-1) if self.keep_by == "value" else self.keep_w(f["x"]).squeeze(-1)
                if self.keep_by == "learned" and self.keep_init == "value":    # starts as value-norm keep
                    sc = sc + f["v"].norm(dim=-1)
                f["ks"] = F.pad(sc[:, 1:], (0, 1))
        return f

    # --------------------------------------------------------------- store
    def _store(self, f: dict) -> dict:
        """Candidate onsets, their windows, retention and keys, from store-side features ``f``."""
        p, x = f["p"], f["x"]
        (B, T), dt, dev = p.shape, p.dtype, p.device
        onsets = (torch.ones_like(p, dtype=torch.bool) if self.gate == "dense"
                  else strict_peaks(p, self.r) & (p > 0))
        N = T if self.gate == "dense" else -(-T // (self.r + 1))  # strict maxima are > r apart
        pos = torch.arange(T, device=dev)
        srt = torch.where(onsets, pos, pos + T).sort(1).values[:, :N]   # onset positions, in time order
        valid = srt < T
        s = torch.where(valid, srt, torch.zeros_like(srt))
        ps = p.gather(1, s)
        lp = torch.log(ps.clamp_min(1e-30))
        # retention: candidate i is held at t if closed by t and fewer than K closed ones beat it
        closed = valid[:, None, :] & ((s + self.delay)[:, None, :] <= pos[None, :, None])     # (B, T, N)
        pi = torch.where(valid, lp.detach() + self.delta * s.to(dt), torch.full_like(lp, float("-inf")))
        beats = valid[:, :, None] & ((pi[:, :, None] > pi[:, None, :])
                                     | ((pi[:, :, None] == pi[:, None, :]) & (s[:, :, None] > s[:, None, :])))
        M = beats.cumsum(1)                                       # M[b, m, i] = #{i' ≤ m : i' beats i}
        last = (closed.sum(-1) - 1).clamp_min(0)                  # closed ones are a prefix (time order)
        rank = M.gather(1, last[..., None].expand(B, T, N))
        ret = closed & (rank < self.K)
        if self.ctrl == "empty":
            ret = torch.zeros_like(ret)
        j = pos.view(1, 1, T)
        w0 = (s - self.P).clamp_min(0)[..., None]
        w1 = (s - self.P + self.n - 1).clamp_max(T - 1)[..., None]
        win = valid[..., None] & (j >= w0) & (j <= w1)            # (B, N, T) window positions
        mem = win & (j < w1)                                      # positions whose successor is stored
        if self.keep:                                             # keep M transitions per window
            mem = mem & self._keep_mask(f, s, w0, w1, mem)
        st = dict(s=s, valid=valid, ps=ps, lp=lp, onsets=onsets, closed=closed, ret=ret, win=win, mem=mem)
        if self.keep and self.keep_by == "learned":               # the scorer's gradient: σ(score) on kept values
            st["kscale"] = torch.sigmoid(f["ks"])
        if self.gate == "hyst":                                   # entries from the onset on weigh g
            gn = F.pad(f["g"][:, 1:], (0, 1))                     # g at the successor position j + 1
            st["wgt"] = torch.where(j + 1 < s[..., None], torch.ones_like(gn[:, None]), gn[:, None])
        if self.use_lex or (self.align == "lex" and self.code == "lex"):
            st["K"] = _rows(window_mean(x, -self.P, self.pool), s)        # (B, N, D) lexical keys (0 if clipped away)
        if self.wk is not None:
            st["KL"] = _rows(window_mean(f["y"], -self.P, self.pool), s)  # (B, N, dk) learned keys
        if self.rform == "supw":                                  # S[i, j]: newer candidate j matches i
            Kc = F.normalize(st["KL"] if self.use_learned else st["K"], dim=-1)
            later = valid[:, None, :] & valid[:, :, None] & (s[:, None, :] > s[:, :, None])
            S = torch.sigmoid(self.log_sup_gamma.exp() * (Kc @ Kc.transpose(1, 2) - self.sup_theta)) * later
            st["sup"] = closed.to(S.dtype) @ S.transpose(1, 2)   # (B, T, N): tags from candidates closed by t
        return st

    def _keep_mask(self, f: dict, s, w0, w1, mem) -> torch.Tensor:
        """(B, N, T) bool: the M transitions kept in each window. ``first``: the first M; otherwise the top M
        by score, ties to the earlier position (a stable sort over time-ordered positions). Scores of a
        window are known when it closes (successors ≤ w1 ≤ c_s); ``keep_pair`` sums the attention a
        successor received from queries up to the window's close c_s only."""
        B, N, T = mem.shape
        M, j = self.keep, torch.arange(T, device=mem.device).view(1, 1, T)
        if self.keep_by == "first":
            return j < w0 + M
        if f.get("kp") is not None:                               # received attention of j + 1 from (j + 1, c_s]
            C = f["kp"].cumsum(1)                                 # C[b, q, k] = Σ_{q' ≤ q} R[b, q', k]
            c = (s + self.delay).clamp_max(T - 1)
            rows = _rows(C, c)                                    # (B, N, T): Σ_{q ≤ c_s} R[q, k]
            diag = C.diagonal(dim1=1, dim2=2)                     # Σ_{q ≤ k} R[q, k]
            got = rows - diag[:, None, :]
            sc = F.pad(got[..., 1:], (0, 1))                      # transition j scored by its successor j + 1
        else:
            sc = f["ks"][:, None, :].expand(B, N, T)
        key = torch.where(mem, sc.detach(), torch.full_like(sc, float("-inf")))
        idx = key.sort(dim=-1, descending=True, stable=True).indices[..., :M]
        keep = torch.zeros_like(mem).scatter(-1, idx, True)
        return keep

    def _select(self, f: dict, st: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Read weights over the candidates (B, T, N), zero unless retained, and the null weight (B, T)."""
        ret, s = st["ret"], st["s"]
        B, T, N = ret.shape
        logit = st["lp"][:, None, :].expand(B, T, N)
        if "K" in st:                                             # centred on the retained keys (≥ 2 of them)
            K = st["K"]
            with torch.no_grad():
                m = ret.to(K.dtype)
                cnt = m.sum(-1, keepdim=True)
                mu = (cnt >= 2).to(K.dtype) * (m @ K) / cnt.clamp_min(1)        # (B, T, D)
                st["mu"], st["mu2"] = mu, mu.pow(2).sum(-1, keepdim=True)
                if self.use_lex:
                    qc = window_mean(f["x"], 1 - self.pool, self.pool) - mu
                    kq = qc @ K.transpose(1, 2) - (qc * mu).sum(-1, keepdim=True)
                    kn = K.pow(2).sum(-1)[:, None] - 2 * (mu @ K.transpose(1, 2)) + st["mu2"]
                    cos = kq / (qc.pow(2).sum(-1, keepdim=True).clamp_min(EPS) * kn.clamp_min(EPS)).sqrt()
            if self.use_lex:
                logit = logit + self.log_beta_lex.exp() * cos
        if self.use_learned:
            q = F.normalize(window_mean(f["y"], 1 - self.pool, self.pool), dim=-1)
            if self.time_feat == "rope":                          # the score depends on content and t − s jointly
                fr = rope_freqs(self.rope_dims, self.rope_periods)
                qr = rotate(q, torch.arange(T, device=s.device).view(1, T), fr)
                kr = rotate(F.normalize(st["KL"], dim=-1), s, fr)
                logit = logit + self.log_beta_learned.exp() * (qr @ kr.transpose(1, 2))
            else:
                logit = logit + self.log_beta_learned.exp() * (q @ F.normalize(st["KL"], dim=-1).transpose(1, 2))
            if self.time_feat == "ordinal":                       # u_t·E[r_k], r_k = rank among retained (0 = newest)
                rank = newer_sum(ret.to(torch.long)).clamp_max(self.K - 1)
                U = self.ord_u(q) @ self.ord_E.T                  # (B, T, K)
                logit = logit + U.gather(-1, rank)
        if self.log_omega is not None and self.rform == "linear":
            age = torch.arange(T, device=s.device).view(1, T, 1) - s[:, None, :]
            logit = logit - self.log_omega.exp() * age.to(logit.dtype)
        elif self.log_omega is not None and self.rform == "log":
            age = torch.arange(T, device=s.device).view(1, T, 1) - s[:, None, :]
            logit = logit - self.log_omega.exp() * torch.log1p(age.clamp_min(0).to(logit.dtype))
        elif self.log_omega is not None and self.rform == "ord":
            logit = logit - self.log_omega.exp() * newer_sum(ret.to(logit.dtype))
        elif self.rform == "supw":
            logit = logit - self.log_sup_omega.exp() * st["sup"]
        logit = torch.cat([logit.masked_fill(~ret, float("-inf")), self.null_logit.to(logit.dtype).expand(B, T, 1)], -1)
        a = torch.softmax(logit, -1)
        if self.rform == "supr":                                  # less weight where newer matching slots carry mass
            mass = newer_sum(a[..., :N].masked_fill(~ret, 0.0))
            logit = torch.cat([logit[..., :N] - self.log_sup_w.exp() * mass, logit[..., N:]], -1)
            a = torch.softmax(logit, -1)
        return a[..., :N], a[..., N]

    def _alignment(self, f: dict, st: dict) -> torch.Tensor | None:
        """E[t, j] = exp(β_a (cos(z_t, z_j) − m_t)), (B, T, T); None without alignment.

        m_t is the best cosine among the entries held at t (detached). Within a slot the read normalises
        E over the slot's entries, so the shift changes nothing mathematically; it keeps the best-matching
        slot's normaliser ≥ 1, where exp(β(cos − 1)) underflowed and its division's backward overflowed
        to NaN (CPU suite, 4 Nov 2026). ``stream`` applies the same rule."""
        if self.align == "none":
            return None
        beta = self.log_beta_align.exp().clamp(max=MAX_BETA)
        if self.code == "learned":                                # z = W_z x, plain cosine (no centring)
            zq, zs = F.normalize(f["z"], dim=-1), F.normalize(st["z"], dim=-1)
            cos = zq @ zs.transpose(1, 2)
        elif self.align == "learned":
            zq, zs = F.normalize(f["y"], dim=-1), F.normalize(st["y"], dim=-1)
            cos = zq @ zs.transpose(1, 2)
        else:
            with torch.no_grad():                                 # lexical: centred on the retained keys
                Z, mu, mu2 = st["x"], st["mu"], st["mu2"]
                zc = f["x"] - mu
                num = zc @ Z.transpose(1, 2) - (zc * mu).sum(-1, keepdim=True)
                zn = Z.pow(2).sum(-1)[:, None] - 2 * (mu @ Z.transpose(1, 2)) + mu2
                cos = num / (zc.pow(2).sum(-1, keepdim=True).clamp_min(EPS) * zn.clamp_min(EPS)).sqrt()
        with torch.no_grad():                                     # the best match among the entries held at t
            held = (st["ret"].to(cos.dtype) @ st["mem"].to(cos.dtype)) > 0
            m = cos.detach().masked_fill(~held, float("-inf")).amax(-1, keepdim=True)
            m = torch.where(torch.isfinite(m), m, torch.ones_like(m))
        return torch.exp(beta * (cos - m))

    def _read(self, f: dict, st: dict) -> torch.Tensor:
        """The recalled value r (B, T, dv): slots with offset alignment, or the superposed state."""
        if self.store == "superposed":
            return self._read_superposed(f, st)
        a, a_null = self._select(f, st)
        st["a"], st["a_null"] = a, a_null
        E = st["E"] = self._alignment(f, st)
        mem = st["mem"].to(a.dtype)
        mw = mem * st["wgt"] if "wgt" in st else mem
        if "kscale" in st:                                        # σ(score) of the kept transitions (the scorer's gradient)
            mw = mw * st["kscale"][:, None, :]
        if E is None:                                             # the window's values, averaged
            A = (a / mem.sum(-1).clamp_min(1)[:, None, :]) @ mw
        else:                                                     # α_{t,k,j} = E[t, j] / Σ_{j ∈ k} E[t, j]
            A = E * ((a / (E @ mem.transpose(1, 2)).clamp_min(ALIGN_FLOOR)) @ mw)
        v = st["v"]
        return A @ F.pad(v[:, 1:], (0, 0, 0, 1))                  # the value stored at j + 1

    def _read_superposed(self, f: dict, st: dict) -> torch.Tensor:
        """Ablation: the slots' keys and values in one delta-rule state, read r_t = S_t q_t.

        Window k is written when it closes, entry by entry: key [κ_k; ζ_j] (window key and token
        code, both unit, over √2), value v_{j+1}, strength p_k (× the on-state for hyst). κ is the
        lexical key centred by the running mean of the lexical stream and/or the learned key; ζ is
        the alignment code. The writes and the per-position reads are merged into one time-ordered
        sequence for ``_run``."""
        s, valid = st["s"], st["valid"]
        (B, N), T, n1 = s.shape, f["x"].shape[1], self.n - 1
        kap, kq = [], []
        if self.use_lex or self.align == "lex":
            ms, mq = cummean(st["x"]), cummean(f["x"])
        if self.use_lex:
            kap.append(F.normalize(st["K"] - _rows(ms, (s + self.delay).clamp_max(T - 1)), dim=-1))
            kq.append(F.normalize(window_mean(f["x"], 1 - self.pool, self.pool) - mq, dim=-1))
        if self.use_learned:
            kap.append(F.normalize(st["KL"], dim=-1))
            kq.append(F.normalize(window_mean(f["y"], 1 - self.pool, self.pool), dim=-1))
        kap, kq = torch.cat(kap, -1) / math.sqrt(len(kap)), torch.cat(kq, -1) / math.sqrt(len(kq))
        e = torch.arange(n1, device=s.device)
        j = s[..., None] - self.P + e                             # (B, N, n − 1) entry positions
        ok = valid[..., None] & (j >= 0) & (j + 1 < T)
        jf = j.clamp(0, T - 2).reshape(B, N * n1)
        keys = kap[:, :, None].expand(B, N, n1, kap.shape[-1]).reshape(B, N * n1, -1)
        if self.align != "none":
            zs, zq = ((F.normalize(st["x"] - ms, dim=-1), F.normalize(f["x"] - mq, dim=-1)) if self.align == "lex"
                      else (F.normalize(st["y"], dim=-1), F.normalize(f["y"], dim=-1)))
            keys = torch.cat([keys, _rows(zs, jf)], -1) / math.sqrt(2)
            kq = torch.cat([kq, zq], -1) / math.sqrt(2)
        vals = _rows(st["v"], jf + 1)
        beta = st["ps"][..., None].expand(B, N, n1).reshape(B, N * n1)
        if "wgt" in st:
            beta = beta * st["wgt"].gather(2, jf.view(B, N, n1)).reshape(B, N * n1)   # weight of v_{j+1}
        beta = beta.clamp(0, 1) * ok.reshape(B, N * n1).to(beta.dtype) * (self.ctrl != "empty")
        # one sequence: window k's writes at its closing time c_k, then the read at each t
        c = torch.where(valid & (s + self.delay < T), s + self.delay, torch.full_like(s, T))
        order_key = torch.cat([(c[..., None] * self.n + e).reshape(B, N * n1),
                               (torch.arange(T, device=s.device) * self.n + self.n - 1).expand(B, T)], 1)
        order = order_key.argsort(dim=1, stable=True)

        def seq(writes, reads):
            return _rows(torch.cat([writes, reads], 1), order)[:, None]
        o, _, _ = _run(seq(kq.new_zeros(B, N * n1, kq.shape[-1]), kq), seq(keys, keys.new_zeros(B, T, keys.shape[-1])),
                       seq(vals, vals.new_zeros(B, T, vals.shape[-1])), beta.new_ones(B, 1, order.shape[1]),
                       seq(beta, beta.new_zeros(B, T)), None, 64)
        return _rows(o[:, 0], order.argsort(1)[:, N * n1:])          # the reads, back in time order

    # ------------------------------------------------------------- outputs
    def _run_store(self, x, h, extra, oracle) -> tuple[dict, dict]:
        f = self._features(x, h, extra, oracle)
        if self.ctrl == "shuffle" and f["p"].shape[0] < 2:
            raise ValueError("--w1s-ctrl shuffle needs a batch of ≥ 2 rows (one row would read its own store)")
        sf = {k: (v.roll(1, 0) if v is not None else None) for k, v in f.items()} if self.ctrl == "shuffle" else f
        st = self._store(sf)
        st.update(x=sf["x"], y=sf["y"], v=sf["v"], p=sf["p"], g=sf["g"])
        if "z" in sf:
            st["z"] = sf["z"]
        return f, st

    def _context_read(self, x: torch.Tensor) -> torch.Tensor:
        h, extra, oracle = self._in
        f, st = self._run_store(x, h, extra, oracle)
        r = self._read(f, st)
        self.last_diag = self._diag(st)
        return self.wb(r)

    def forward(self, h: torch.Tensor, extra: dict | None = None, oracle: torch.Tensor | None = None):
        """``GDNLoop.forward`` (CA1 injection) with the slot read; returns (h + injection, LoopStats)."""
        self._in = (h, extra, oracle)
        try:
            return super().forward(h)
        finally:
            self._in = None

    def read(self, h: torch.Tensor, extra: dict | None = None, oracle: torch.Tensor | None = None) -> torch.Tensor:
        """The read o = B r before the CA1 injection (B, T, D); compared with ``stream`` in the tests."""
        self._in = (h, extra, oracle)
        try:
            return self._context_read(self.norm(h))
        finally:
            self._in = None

    def reinstate_bias(self, extra: dict, oracle: torch.Tensor | None = None) -> torch.Tensor:
        """(B, T, T) float32: log Σ_k a_{t,k} 1[c ∈ window_k] + b, or NEG where nothing is reinstated.

        Rows are query positions t, columns positions c (side coordinates). Uses the taps only."""
        f, st = self._run_store(None, None, extra, oracle)
        a, a_null = self._select(f, st)
        st.update(a=a, a_null=a_null, E=None)
        S = a @ st["win"].to(a.dtype)
        self.last_diag = self._diag(st)
        return torch.where(S > 0, torch.log(S.clamp_min(1e-30)) + self.rbias, torch.full_like(S, NEG)).float()

    @torch.no_grad()
    def _diag(self, st: dict) -> dict:
        s, ret = st["s"], st["ret"] if self.store == "slots" or self.ctrl == "empty" else st["closed"]
        B, T, N = ret.shape
        held = torch.where(ret, s[:, None, :], torch.full_like(ret, -1, dtype=s.dtype))
        if N < self.K:
            held = F.pad(held, (0, self.K - N), value=-1)
        top = held.topk(self.K, -1)
        d = dict(p=st["p"].detach(), g=st["g"].detach(), onsets=st["onsets"], retained=top.values)
        if "KL" in st or "K" in st:                               # candidates' keys (supw fire diagnostics, G0.3)
            d["cand_s"] = s.where(st["valid"], torch.full_like(s, -1))
            d["cand_key"] = F.normalize((st["KL"] if self.use_learned else st["K"]).detach(), dim=-1)
        if "a" not in st:                                         # superposed: no slot read
            d["read_w"] = torch.zeros(B, T, self.K + 1, device=s.device)
            d["align"] = torch.full((B, T), -1, device=s.device, dtype=torch.long)
            return d
        a = F.pad(st["a"].detach(), (0, max(0, self.K - N)))
        w = a.gather(-1, top.indices) * (top.values >= 0)
        d["read_w"] = torch.cat([w, st["a_null"].detach()[..., None]], -1)
        k = torch.cat([st["a"].detach(), st["a_null"].detach()[..., None]], -1).argmax(-1)    # (B, T)
        if st["E"] is None:                                       # no alignment (or reinstatement)
            d["align"] = torch.full_like(k, -1)
            return d
        rows = _rows(st["mem"], k.clamp_max(N - 1))               # (B, T, T) positions of the chosen slot
        j = (st["E"].detach() * rows).argmax(-1) + 1
        d["align"] = torch.where(k < N, j, torch.full_like(j, -1))
        return d

    # ----------------------------------------------------------- reference
    @torch.no_grad()
    def stream(self, h: torch.Tensor | None = None, extra: dict | None = None,
               oracle: torch.Tensor | None = None) -> dict:
        """Pure-Python streaming reference (rule E2): one position at a time, a ring buffer of recent
        entries, K slots that evict the lowest priority (or one delta-rule state when superposed).
        It never calls the parallel code (``_store``, ``_select``, ``_read``), only the shared maps.

        A slot holds its key(s) and the transitions it keeps, each (offset j, code_j, w·v_{j+1}): the
        alignment code of entry j (the lexical state, z = W_z x, or the learned code) and the weighted
        successor value. With ``--w1s-keep M`` only M transitions are kept, chosen when the window
        closes; without it, every transition of the window.

        Returns ``o`` (B, T, D) (additive) or ``rbias`` (B, T, T) (reinstate), ``p``, ``onsets``,
        ``retained`` (per row and t, the set of held onsets), ``read_w`` (per row and t, onset → weight,
        with the null under None), ``state_floats`` (the state's size at the end, the largest row),
        ``state_components`` (its parts) and ``evictions``.
        """
        if self.ctrl == "shuffle":
            raise ValueError("stream() has no shuffle control")
        xs = self._streams(h, extra)
        x0 = xs[self.lex]
        (B, T, D), dt = x0.shape, x0.dtype
        xn = self.norm(h) if self.out == "additive" else None
        P, n, r, delay, pool = self.P, self.n, self.r, self.delay, self.pool
        ring_len = max(delay + P + 1, delay + r + 1, pool)
        lin = lambda mods, t, b: sum(m_(xs[k][b, t]) for m_, k in zip(mods, self.streams))
        res = dict(p=torch.zeros(B, T, dtype=dt), onsets=torch.zeros(B, T, dtype=torch.bool),
                   retained=[[None] * T for _ in range(B)], read_w=[[None] * T for _ in range(B)], evictions=0)
        if self.out == "additive":
            res["o"] = torch.zeros(B, T, xn.shape[-1], dtype=dt)
        else:
            res["rbias"] = torch.full((B, T, T), NEG, dtype=torch.float32)
        cosine = lambda a, b: F.normalize(a, dim=-1) @ F.normalize(b, dim=-1)
        keyed_x = self.use_lex or (self.align == "lex" and self.code == "lex")   # slots keep a lexical key
        fr = rope_freqs(self.rope_dims, self.rope_periods) if self.time_feat == "rope" else None
        kscores = self.keep_scores.to(dt) if self.keep and self.keep_by == "external" and self.keep_scores is not None else None
        kpair = self.keep_pair.to(dt) if self.keep and self.keep_by == "external" and self.keep_pair is not None else None
        if self.keep and self.keep_by == "external" and kscores is None and kpair is None:
            raise ValueError("--w1s-keep-by external needs side.keep_scores or side.keep_pair")
        for b in range(B):
            ring, slots, S = deque(maxlen=ring_len), [], None
            ysum = y2sum = u = gprev = usum = u2sum = 0.0
            sprev, xsum = False, torch.zeros(D, dtype=dt)
            for t in range(T):
                # ---- ingest position t: features and the gate
                x = xs[self.lex][b, t]
                e = dict(t=t)
                if self.need_x:
                    e["x"] = x
                if self.store == "superposed":                    # centred codes, running mean at t
                    xsum = xsum + x
                    e["zc"], e["mq"] = F.normalize(x - xsum / (t + 1), dim=-1), xsum / (t + 1)
                e["y"] = lin(self.wk, t, b) if self.wk is not None else None
                e["v"] = self.wa(xn[b, t]) if xn is not None else None
                if self.code == "learned":
                    e["z"] = self.wz(x)
                if self.keep and self.keep_by == "value":         # this position's score as a successor
                    e["ks"] = float(e["v"].norm())
                elif self.keep and self.keep_by == "learned":
                    e["ks"] = float(self.keep_w(x).squeeze(-1)) + (float(e["v"].norm()) if self.keep_init == "value" else 0.0)
                if self.gate == "oracle":
                    p = g = float(oracle[b, t])
                elif self.gate == "dense":
                    p = g = torch.sigmoid(self.gate_b).item()
                else:
                    yg = lin(self.gate_w, t, b).squeeze(-1)
                    ysum, y2sum = ysum + yg, y2sum + yg * yg
                    d = yg - ysum / (t + 1) if self.centre else yg
                    sd = lambda s1, s2: ((s2 / (t + 1) - (s1 / (t + 1)) ** 2).clamp_min(0) + 1e-6).sqrt()
                    if self.gate == "point":
                        p = g = torch.sigmoid(self.gate_a.exp() * d / sd(ysum, y2sum) + self.gate_b)
                    else:
                        lam = torch.sigmoid(self.lam_logit)
                        u = lam * u + (1 - lam) * d
                        usum, u2sum = usum + u, u2sum + u * u
                        z = u / sd(usum, u2sum)
                        if self.gate == "hyst":
                            g = torch.sigmoid(self.gate_a.exp() * z + self.gate_b + self.latch_c * (gprev - 0.5))
                            p, gprev = g * (1 - gprev), g
                        else:
                            on = bool(z >= self.thresholds[0]) or (sprev and not bool(z < self.thresholds[1]))
                            p = float(on and not sprev) * torch.sigmoid(SCHMITT_GAIN * (z - self.thresholds[0]))
                            g, sprev = float(on), on
                e["p"], e["g"] = float(p), float(g)
                res["p"][b, t] = e["p"]
                ring.append(e)
                at = {q["t"]: q for q in ring}
                # ---- the candidate whose suppression and window are now known
                s = t - delay
                if s >= 0:
                    ps = at[s]["p"]
                    nb = [at[j]["p"] for j in range(max(0, s - r), s + r + 1) if j != s]
                    if self.gate == "dense" or (ps > 0 and all(ps > q for q in nb)):
                        res["onsets"][b, s] = True
                        w0, w1 = max(0, s - P), s - P + n - 1
                        head = [at[j] for j in range(w0, s - P + pool)]
                        ent = [at[j] for j in range(w0, w1 + 1)]
                        wgt = [1.0 if (self.gate != "hyst" or q["t"] < s) else q["g"] for q in ent]
                        lp = math.log(max(ps, 1e-30))
                        # a head clipped away at position 0 gives a zero key, as window_mean does
                        hmean = lambda k: (torch.stack([q[k] for q in head]).mean(0) if head
                                           else torch.zeros_like(e[k]))
                        slot = dict(s=s, w0=w0, w1=w1, pi=lp + self.delta * s, lp=lp, ps=ps, ent=ent, wgt=wgt,
                                    K=hmean("x") if keyed_x or self.store == "superposed" else None,
                                    KL=hmean("y") if self.wk is not None else None, sup=0.0)
                        slot["trans"] = self._stream_transitions(slot, t, b, kscores, kpair)
                        if self.store == "superposed":
                            if self.ctrl != "empty":
                                S = self._stream_write(S, slot, e)
                        elif self.ctrl != "empty":
                            if self.rform == "supw":              # tag every held (older) slot this one matches
                                kn = slot["KL"] if self.use_learned else slot["K"]
                                for q in slots:
                                    ko = q["KL"] if self.use_learned else q["K"]
                                    q["sup"] = q["sup"] + torch.sigmoid(
                                        self.log_sup_gamma.exp() * (cosine(ko, kn) - self.sup_theta))
                            slots.append(slot)
                            if len(slots) > self.K:               # evict the lowest priority (older on ties)
                                slots.remove(min(slots, key=lambda q: (q["pi"], q["s"])))
                                res["evictions"] += 1
                # ---- read at t
                win = list(ring)[-pool:]
                q_lex = torch.stack([q["x"] for q in win]).mean(0) if self.need_x else None
                q_l = torch.stack([q["y"] for q in win]).mean(0) if self.wk is not None else None
                if self.store == "superposed":
                    rvec = self._stream_read(S, e, q_lex, q_l)
                    res["retained"][b][t], res["read_w"][b][t] = set(), {None: 0.0}
                else:
                    mu = None
                    if keyed_x:
                        mu = (torch.stack([q["K"] for q in slots]).mean(0) if len(slots) >= 2
                              else torch.zeros(D, dtype=dt))
                    logits = []
                    for i, q in enumerate(slots):
                        lg = q["lp"]
                        if self.use_lex:
                            qc = q_lex - mu
                            kq = qc @ q["K"] - qc @ mu
                            kn = q["K"] @ q["K"] - 2 * (mu @ q["K"]) + mu @ mu
                            lg = lg + self.log_beta_lex.exp() * kq / (max(qc @ qc, EPS) * max(kn, EPS)) ** 0.5
                        if self.use_learned:
                            if self.time_feat == "rope":
                                qr = rotate(F.normalize(q_l, dim=-1), torch.tensor(t), fr)
                                kr = rotate(F.normalize(q["KL"], dim=-1), torch.tensor(q["s"]), fr)
                                lg = lg + self.log_beta_learned.exp() * (qr @ kr)
                            else:
                                lg = lg + self.log_beta_learned.exp() * cosine(q_l, q["KL"])
                            if self.time_feat == "ordinal":
                                rank = min(len(slots) - 1 - i, self.K - 1)       # slots are in time order
                                lg = lg + self.ord_u(F.normalize(q_l, dim=-1)) @ self.ord_E[rank]
                        if self.log_omega is not None and self.rform == "linear":
                            lg = lg - self.log_omega.exp() * (t - q["s"])
                        elif self.log_omega is not None and self.rform == "log":
                            lg = lg - self.log_omega.exp() * math.log1p(t - q["s"])
                        elif self.log_omega is not None and self.rform == "ord":
                            lg = lg - self.log_omega.exp() * (len(slots) - 1 - i)
                        elif self.rform == "supw":
                            lg = lg - self.log_sup_omega.exp() * q["sup"]
                        logits.append(torch.as_tensor(lg, dtype=dt))
                    a = torch.softmax(torch.stack(logits + [self.null_logit.to(dt)]), 0)
                    if self.rform == "supr":                      # second pass: newer slots' first-pass mass
                        lg2 = [logits[i] - self.log_sup_w.exp() * sum(a[k] for k in range(i + 1, len(slots)))
                               for i in range(len(slots))]
                        a = torch.softmax(torch.stack([torch.as_tensor(v, dtype=dt) for v in lg2]
                                                      + [self.null_logit.to(dt)]), 0)
                    res["retained"][b][t] = {q["s"] for q in slots}
                    res["read_w"][b][t] = {**{q["s"]: float(a[i]) for i, q in enumerate(slots)}, None: float(a[-1])}
                    rvec, cover = 0.0, torch.zeros(T, dtype=dt)
                    cs_by = {}                                    # each slot's cosines to the current token
                    if self.out == "additive" and self.align != "none":
                        for i, q in enumerate(slots):
                            tr = q["trans"]
                            if not tr:
                                continue
                            if self.code == "learned":
                                cs_by[i] = torch.stack([cosine(e["z"], c_) for _, c_, _ in tr])
                            elif self.align == "lex":
                                zc = x - mu
                                cs_by[i] = torch.stack([(zc @ c_ - zc @ mu) / (max(zc @ zc, EPS) * max(
                                    c_ @ c_ - 2 * (mu @ c_) + mu @ mu, EPS)) ** 0.5 for _, c_, _ in tr])
                            else:
                                cs_by[i] = torch.stack([cosine(e["y"], c_) for _, c_, _ in tr])
                    m_t = max((float(v.max()) for v in cs_by.values()), default=1.0)   # the best held match
                    for i, q in enumerate(slots):
                        if self.out == "reinstate":
                            cover[q["w0"]:q["w1"] + 1] += a[i]
                            continue
                        tr = q["trans"]
                        if not tr:                                # clipped at 0: no entry has a stored successor
                            continue
                        if self.align == "none":
                            al = torch.full((len(tr),), 1.0 / len(tr), dtype=dt)
                        else:
                            E = torch.exp(self.log_beta_align.exp().clamp(max=MAX_BETA) * (cs_by[i] - m_t))
                            al = E / E.sum().clamp_min(ALIGN_FLOOR)
                        vals = torch.stack([v_ for _, _, v_ in tr])
                        rvec = rvec + a[i] * (al @ vals)
                    if self.out == "reinstate":
                        res["rbias"][b, t] = torch.where(cover > 0, torch.log(cover.clamp_min(1e-30)) + self.rbias, NEG)
                if self.out == "additive":
                    res["o"][b, t] = self.wb(rvec) if torch.is_tensor(rvec) else 0.0
            comp = self._stream_components(ring, slots, S)
            if sum(comp.values()) >= res.get("state_floats", -1):
                res["state_floats"], res["state_components"] = sum(comp.values()), comp
        return res

    def _stream_transitions(self, slot: dict, t: int, b: int, kscores, kpair) -> list:
        """The transitions a closing window keeps: (offset j, code_j, w_{j+1}·v_{j+1}) in time order. With
        --w1s-keep M, the first M or the top M by score (ties to the earlier offset); a learned score
        also scales the kept value by σ(score). Values are None for reinstatement (nothing is replayed)."""
        ent, wgt = slot["ent"], slot["wgt"]
        idx = list(range(len(ent) - 1))                           # transition i: ent[i] → ent[i + 1]
        if self.keep:
            if self.keep_by == "first":
                kept = [i for i in idx if ent[i]["t"] < slot["w0"] + self.keep]
            else:
                if self.keep_by in ("value", "learned"):
                    sc = [ent[i + 1]["ks"] for i in idx]
                elif kpair is not None:                           # attention received by j + 1 from (j + 1, t]
                    sc = [float(sum(kpair[b, q_, ent[i + 1]["t"]] for q_ in range(ent[i + 1]["t"] + 1, t + 1)))
                          for i in idx]
                else:
                    sc = [float(kscores[b, ent[i]["t"]]) for i in idx]
                kept = sorted(sorted(idx, key=lambda i: (-sc[i], i))[:self.keep])
        else:
            kept = idx
        out = []
        for i in kept:
            c_, nx = ent[i], ent[i + 1]
            code = c_["z"] if self.code == "learned" else c_.get("x") if self.align == "lex" else c_["y"]
            if self.align == "none":
                code = None
            val = None
            if nx["v"] is not None:
                val = wgt[i + 1] * nx["v"]
                if self.keep and self.keep_by == "learned":
                    val = torch.sigmoid(torch.tensor(nx["ks"], dtype=val.dtype)) * val
            out.append((c_["t"], code, val))
        return out

    def _stream_write(self, S, slot, e):
        """Superposed store: write slot's entries (key [κ; ζ_j], value v_{j+1}) by the delta rule."""
        parts = []
        if self.use_lex:
            parts.append(F.normalize(slot["K"] - e["mq"], dim=-1))
        if self.use_learned:
            parts.append(F.normalize(slot["KL"], dim=-1))
        kap = torch.cat(parts) / math.sqrt(len(parts))
        for c_, nxt, w_ in zip(slot["ent"][:-1], slot["ent"][1:], slot["wgt"][1:]):
            k = kap
            if self.align != "none":
                z = c_["zc"] if self.align == "lex" else F.normalize(c_["y"], dim=-1)
                k = torch.cat([kap, z]) / math.sqrt(2)
            if S is None:
                S = k.new_zeros(nxt["v"].shape[0], k.shape[0])
            beta = min(max(slot["ps"] * w_, 0.0), 1.0)
            S = S + beta * torch.outer(nxt["v"] - S @ k, k)
        return S

    def _stream_read(self, S, e, q_lex, q_l):
        parts = []
        if self.use_lex:
            parts.append(F.normalize(q_lex - e["mq"], dim=-1))
        if self.use_learned:
            parts.append(F.normalize(q_l, dim=-1))
        q = torch.cat(parts) / math.sqrt(len(parts))
        if self.align != "none":
            z = e["zc"] if self.align == "lex" else F.normalize(e["y"], dim=-1)
            q = torch.cat([q, z]) / math.sqrt(2)
        return S @ q if S is not None else q.new_zeros(self.wa.out_features)

    def _stream_components(self, ring, slots, S) -> dict:
        """Floats held by the streaming state, by part: the ring buffer, the slots' keys, their stored
        codes and values (per entry, or per kept transition with its offset), supersession tags, the
        superposed state, and the gate's scalars with the lexical running sum."""
        def numel(v):
            return v.numel() if torch.is_tensor(v) else (1 if isinstance(v, float) else 0)
        c = dict(ring=sum(numel(v) for q in ring for k, v in q.items() if k != "t"),
                 slot_keys=0, slot_codes=0, slot_values=0, slot_weights=0, sup_tags=0, superposed=0, gate=0)
        for q in slots:                                           # keys, then per entry: code, value, weight
            c["slot_keys"] += numel(q["K"]) + numel(q["KL"]) + 2
            c["sup_tags"] += 1 if self.rform == "supw" else 0
            if self.keep:                                         # per kept transition: code, value, offset
                for _, code, val in q["trans"]:
                    c["slot_codes"] += numel(code)
                    c["slot_values"] += numel(val)
                    c["slot_weights"] += 1
            else:
                for c_ in q["ent"]:
                    c["slot_codes"] += (numel(c_.get("x")) * (self.align == "lex" and self.code == "lex")
                                        + numel(c_.get("z")) * (self.code == "learned")
                                        + numel(c_["y"]) * (self.align == "learned"))
                    c["slot_values"] += numel(c_["v"])
                    c["slot_weights"] += 1
        c["superposed"] = S.numel() if S is not None else 0
        c["gate"] = 7 + (ring[0]["x"].numel() if self.need_x else 0)   # gate state and the lexical running sum
        return c

    def _stream_size(self, ring, slots, S) -> int:
        """Floats held by the streaming state: ring buffer, slots (or S), gate scalars and running sums."""
        return sum(self._stream_components(ring, slots, S).values())

    def analytic_state(self, d_model: int) -> dict:
        """The streaming state at full occupancy (a full ring, K full slots of unclipped windows), by part,
        without running ``stream``: what ``_stream_components`` reports once the store is full."""
        D = d_model
        dk = self.wk[0].out_features if self.wk is not None else 0
        dv = self.wa.out_features if self.out == "additive" else 0
        dc = self.wz.out_features if self.code == "learned" else 0
        ring_len = max(self.delay + self.P + 1, self.delay + self.r + 1, self.pool)
        per = (D * self.need_x + dk + dv + dc + 2 + (self.keep > 0 and self.keep_by in ("value", "learned"))
               + 2 * D * (self.store == "superposed"))
        keyed_x = self.use_lex or (self.align == "lex" and self.code == "lex")
        code = dc if self.code == "learned" else D if self.align == "lex" else dk if self.align == "learned" else 0
        c = dict(ring=ring_len * per, slot_keys=0, slot_codes=0, slot_values=0, slot_weights=0, sup_tags=0,
                 superposed=0, gate=7 + D * self.need_x)
        if self.store == "superposed":
            kdim = (D if self.use_lex else 0) + (dk if self.use_learned else 0) + (code if self.align != "none" else 0)
            c["superposed"] = dv * kdim
            return c
        m = self.keep if self.keep else self.n                    # entries (or kept transitions) per slot
        c["slot_keys"] = self.K * (D * keyed_x + dk + 2)
        c["slot_codes"] = self.K * m * code
        c["slot_values"] = self.K * m * dv
        c["slot_weights"] = self.K * m
        c["sup_tags"] = self.K * (self.rform == "supw")
        return c


# ----------------------------------------------------------------------- side
class W1sSide(nn.Module):
    """TimeChannel then SlotLoop; the ``SideChannels`` interface used by ``pretrained_loop``, with taps.

    ``extra`` ({block: (B, T, D) float32, sink stripped}) and ``oracle_onsets`` ((B, T) bool, side
    coordinates) are set by the caller before each forward. In reinstate mode the forward returns h
    unchanged and the read happens in ``reinstate_bias``, called by ``taps.TapLM`` once the taps ran.
    """

    def __init__(self, d_model: int, args: argparse.Namespace, w: argparse.Namespace | None = None) -> None:
        super().__init__()
        w = w or split_args([])[0]
        self.taps = _ints(w.w1s_taps)
        self.out_mode, self.reinstate_L = w.w1s_out, w.w1s_L
        self.extra: dict | None = None
        self.oracle_onsets: torch.Tensor | None = None
        reinstate = self.out_mode == "reinstate"
        kt = _ints(w.w1s_key_taps) if w.w1s_key_taps is not None else self.taps + (() if reinstate else (-1,))
        main = [i for i in kt if i not in self.taps]
        if len(main) > 1:
            raise ValueError(f"--w1s-key-taps {kt}: indices not in --w1s-taps {self.taps} mean the main tap; "
                             "give at most one")
        streams = [i if i in self.taps else "h" for i in kt]
        if not streams:
            raise ValueError("W1s needs at least one stream (--w1s-taps / --w1s-key-taps)")
        if reinstate:
            if "h" in streams or not self.taps:
                raise ValueError("reinstatement reads the taps only: --w1s-key-taps must be taps (< --w1s-L)")
            if max(self.taps) >= self.reinstate_L:
                raise ValueError(f"taps {self.taps} must lie below --w1s-L {self.reinstate_L}")
        lex = min(streams, key=lambda m: math.inf if m == "h" else m)
        self.time = TimeChannel(d_model) if getattr(args, "time", True) else None
        self.loop = SlotLoop(d_model, streams, lex, w)
        self.w1s = vars(w)

    @property
    def gate(self) -> str:
        return self.loop.gate

    @property
    def ctrl(self) -> str:
        """The evaluation control (``--w1s-ctrl``); ``eval_by_query.py`` sets it on the side."""
        return self.loop.ctrl

    @ctrl.setter
    def ctrl(self, value: str) -> None:
        check_ctrl(value, self.out_mode)
        self.loop.ctrl = value

    @property
    def keep_scores(self) -> torch.Tensor | None:
        """(B, T) score of each transition j → j+1 for ``--w1s-keep-by external`` (evaluation diagnostics)."""
        return self.loop.keep_scores

    @keep_scores.setter
    def keep_scores(self, value: torch.Tensor | None) -> None:
        self.loop.keep_scores = value

    @property
    def keep_pair(self) -> torch.Tensor | None:
        """(B, T, T) R[b, q, k], attention key k received from query q (side coordinates), for
        ``--w1s-keep-by external``: a transition is scored by what its successor received up to the close."""
        return self.loop.keep_pair

    @keep_pair.setter
    def keep_pair(self, value: torch.Tensor | None) -> None:
        self.loop.keep_pair = value

    def state_report(self) -> dict:
        """The streaming state at full occupancy (T-independent, rule E2) by part, and the trainable parameters."""
        comp = self.loop.analytic_state(self.loop.wb.out_features if self.out_mode == "additive"
                                        else self.loop.gate_w[0].in_features)
        return dict(state_floats=sum(comp.values()), state_components=comp,
                    params=sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, LoopStats]:
        if self.out_mode == "reinstate":                          # the read went into attention
            z = h.new_zeros(h.shape[:2])
            return h, LoopStats(z, z)
        if self.time is not None:
            h, _ = self.time(h)
        return self.loop(h, self.extra, self.oracle_onsets)

    def reinstate_bias(self, extra: dict) -> torch.Tensor:
        """(B, T, T) float32 attention bias for layers ≥ ``reinstate_L`` (see ``SlotLoop.reinstate_bias``)."""
        return self.loop.reinstate_bias(extra, self.oracle_onsets)
