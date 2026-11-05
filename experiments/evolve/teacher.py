"""Training-time teachers from the frozen model's own full-attention pass: round-11 arms A14 and A15.

Both arms use the frozen base model under the ``full`` condition (``lm.condition(0, False)``: no window,
side off) as a teacher while training, and nothing of it at evaluation, so rule E4 holds at test time
(RV4.R1 and R2.1, research/reviews/2026-11-04_side_memory/README.md §5; DEC12). Both default to off,
and with both off ``run.train`` calls ``pretrained_loop.train`` exactly as before (no proxy, no patch).

A14  self-distillation (``--distill-frac F`` > 0)
     A fraction F of the training steps (drawn per step from their own RNG, seeded from --seed, which
     also draws the text offsets; the needle batches' RNG and a --needle-curriculum's stage counter
     advance on needle steps only) train on a
     general-text batch instead of a needle batch: B contiguous windows of --seq-len tokens from
     --distill-text, after the sink, as ``pl.TextTask`` builds them. The loss on those steps is

         T² · KL( softmax(z_full / T) ‖ softmax(z_side / T) ),   T = --distill-temp

     averaged over the batch and the last N = --distill-positions positions (N ≤ seq-len − window is
     enforced, so every position is beyond the window, where full and windowed differ). z_full is the
     frozen model under the full condition (the teacher, in no_grad), z_side the same model under
     ``condition(window, True)`` (the student). The T² keeps the gradient's scale across temperatures
     (Hinton et al. 2015); at the default T = 1 it is 1. Memory (the vocabulary is 151k): neither
     (B, N, V) logits tensor is ever materialised. Each pass returns the backbone's final hidden states
     (a forward hook on ``model.base_model``), with ``lm(ids, keep=1)`` computing the logits of the last
     position only (which checks, once, that the LM head is a plain linear map of those states). The
     KL is then computed in chunks of --distill-chunk positions under activation checkpointing, so the
     peak is one chunk's float32 logits of each model, and backward recomputes them chunk by chunk.
A15  teacher attention-mass admission target (``--admit-target teacher``)
     On needle training batches, a second no_grad pass of the frozen model under the full condition
     records, at the attention layers in --admit-layers LO-HI (default all), the attention from the
     query rows (the query span " The code for NAME is" and the answer's space, which predicts digit
     1) onto every position, averaged over heads, layers and those rows. Those layers run eager
     attention for this pass only: their modules see a config proxy whose ``_attn_implementation`` is
     "eager", with an explicit causal mask (sdpa's model-level mask is None, and eager attention
     without one would see the future). The student pass keeps the loaded implementation (sdpa). For
     each candidate onset s of the gate (``SlotLoop._store``: the strict local maxima of p, side
     coordinates) whose window [s − P, s − P + n − 1] closed before the query (s + delay < q0, so the
     read could use it at every query row), the target is the attention mass inside that window,
     normalised over those candidates. The student is the gate's p over the same candidates,
     normalised (log p from the store, with its gradient). The loss λ · KL(target ‖ student),
     λ = --admit-weight, averaged over the rows that have a target, is added to the needle loss. Its
     gradient reaches only the gate (through log p_s). The teacher pass runs before the student pass,
     so ``lm.last_stats`` (logged by ``pl.train``) is the student's.

How it reaches ``pl.train`` without editing it: ``training`` (a context manager, entered by ``run.train``)
hands ``pl.train`` a proxy of the task whose ``batch`` returns a text batch tagged ``ids.distill`` on a
distillation step and, under A15, records each needle row's query length (``ids.query_len``, from the
task's own ``_record``), and replaces ``pl.scores`` for the duration of training: a tagged batch returns
the per-position KL as ``ce`` with an all-true mask (so ``pl.train``'s loss is its mean, and its logged
``token_acc`` is the teacher–student argmax agreement); under A15 a needle batch returns ``ce + λ·KL``,
so ``pl.train``'s mean over the answer tokens is the needle loss plus the admission loss. Both are
restored on exit. ``summary`` gives the settings and mean losses that ``run.py`` writes to the result
JSON (``distill``, ``admit``).

Eager attention costs B·H·T² per captured layer (transient; only the query rows are kept), which is
small at T = 513 but grows quadratically for Stage-C lengths.
"""

from __future__ import annotations

import argparse
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402

DEFAULT_TEXT = "/home/realdanielbyrne/limbic-data/wikitext103_train.txt"
DEFAULTS = dict(distill_frac=0.0, distill_text=DEFAULT_TEXT, distill_positions=256, distill_temp=1.0,
                distill_chunk=32, admit_target="none", admit_weight=1.0, admit_layers="")
NO_TARGET_GATES = ("oracle", "dense")                             # p fixed or flat: nothing to learn


def split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Take the --distill-* and --admit-* flags; the rest go on to the candidate and ``pretrained_loop``."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--distill-frac", type=float, default=DEFAULTS["distill_frac"],
                    help="A14: share of training steps on general text with the full-attention KL (0 = off)")
    ap.add_argument("--distill-text", default=DEFAULTS["distill_text"], help="A14: the general-text corpus")
    ap.add_argument("--distill-positions", type=int, default=DEFAULTS["distill_positions"], metavar="N",
                    help="A14: the last N positions carry the KL (N ≤ seq-len − window)")
    ap.add_argument("--distill-temp", type=float, default=DEFAULTS["distill_temp"], metavar="T",
                    help="A14: temperature of both distributions (the KL is scaled by T²)")
    ap.add_argument("--distill-chunk", type=int, default=DEFAULTS["distill_chunk"],
                    help="A14: positions per checkpointed KL chunk (memory only; the loss is the same)")
    ap.add_argument("--admit-target", choices=["none", "teacher"], default=DEFAULTS["admit_target"],
                    help="A15: teacher = the full model's attention mass per candidate window as a target for p")
    ap.add_argument("--admit-weight", type=float, default=DEFAULTS["admit_weight"], metavar="LAMBDA",
                    help="A15: weight of KL(target ‖ gate)")
    ap.add_argument("--admit-layers", default=DEFAULTS["admit_layers"], metavar="LO-HI",
                    help="A15: blocks whose attention is averaged (inclusive; attention layers only; '' = all)")
    return ap.parse_known_args(argv)


def distill_on(ns) -> bool:
    return ns is not None and ns.distill_frac > 0


def admit_on(ns) -> bool:
    return ns is not None and ns.admit_target != "none"


def active(ns) -> bool:
    return distill_on(ns) or admit_on(ns)


def check(ns: argparse.Namespace, args: argparse.Namespace, scatter: int, cand_ns) -> None:
    """Refuse settings that cannot mean what they say, before the model is loaded (SystemExit)."""
    changed = {k for k, v in DEFAULTS.items() if getattr(ns, k) != v}
    if not distill_on(ns) and changed & {k for k in DEFAULTS if k.startswith("distill_") and k != "distill_frac"}:
        raise SystemExit("--distill-text/-positions/-temp/-chunk need --distill-frac > 0")
    if not admit_on(ns) and changed & {"admit_weight", "admit_layers"}:
        raise SystemExit("--admit-weight/--admit-layers need --admit-target teacher")
    if not 0 <= ns.distill_frac <= 1:
        raise SystemExit(f"--distill-frac {ns.distill_frac} must be in [0, 1]")
    if distill_on(ns):
        if args.task != "needle":
            raise SystemExit("--distill-frac replaces a share of needle batches: needs --task needle")
        if not args.window:
            raise SystemExit("--distill-frac needs a window (the student is the windowed model)")
        if args.window_mode == "segment":
            raise SystemExit("--distill-frac does not support --window-mode segment (the hidden states are folded)")
        if not 1 <= ns.distill_positions <= args.seq_len - args.window:
            raise SystemExit(f"--distill-positions {ns.distill_positions} must be in [1, seq-len − window = "
                             f"{args.seq_len - args.window}] (positions beyond the window)")
        if ns.distill_temp <= 0 or ns.distill_chunk < 1:
            raise SystemExit("--distill-temp must be > 0 and --distill-chunk ≥ 1")
        if not Path(ns.distill_text).is_file():
            raise SystemExit(f"--distill-text {ns.distill_text}: no such file")
        if cand_ns is not None and any(k.endswith("_gate") and v == "oracle" for k, v in vars(cand_ns).items()):
            raise SystemExit("--distill-frac: the oracle gate has no fact onsets on text batches (review R4)")
    if admit_on(ns):
        if args.task != "needle" or not scatter:
            raise SystemExit("--admit-target needs the needle task with --needle-scatter (query spans from its layout)")
        if cand_ns is None or not hasattr(cand_ns, "w1s_gate"):
            raise SystemExit("--admit-target needs a W1s candidate (its gate's candidate windows)")
        if cand_ns.w1s_gate in NO_TARGET_GATES:
            raise SystemExit(f"--admit-target: the {cand_ns.w1s_gate} gate has no learned p to train")
        if ns.admit_weight <= 0:
            raise SystemExit("--admit-weight must be > 0")


def admit_layers(spec: str, lm) -> list[int]:
    """``LO-HI`` (inclusive) or ``I`` → the attention blocks among them; '' = every attention block."""
    n = len(lm.blocks)
    if spec.strip():
        lo, hi = (int(x) for x in spec.split("-")) if "-" in spec else (int(spec), int(spec))
    else:
        lo, hi = 0, n - 1
    if not 0 <= lo <= hi < n:
        raise SystemExit(f"--admit-layers {spec}: the model has blocks 0–{n - 1}")
    layers = [i for i in range(lo, hi + 1) if lm.kinds[i] == "attention"]
    if not layers:
        raise SystemExit(f"--admit-layers {spec}: no attention layer in that range")
    return layers


# ------------------------------------------------------------------ A14: text
class TextWindows:
    """Contiguous --seq-len windows of a tokenised corpus, with the sink prepended (as ``pl.TextTask``)."""

    def __init__(self, tok, path: str, seq_len: int, sink: int) -> None:
        self.ids = torch.tensor(tok.encode(Path(path).read_text(encoding="utf-8", errors="replace")))
        self.seq_len, self.sink, self.bos = seq_len, sink, tok.bos_token_id
        if len(self.ids) <= seq_len:
            raise ValueError(f"{path} has {len(self.ids)} tokens; need more than --seq-len {seq_len}")

    def batch(self, B: int, rng: random.Random) -> torch.Tensor:
        o = [rng.randrange(len(self.ids) - self.seq_len) for _ in range(B)]
        rows = torch.stack([self.ids[i:i + self.seq_len] for i in o])
        if self.sink:
            rows = torch.cat([torch.full((B, self.sink), self.bos, dtype=rows.dtype), rows], 1)
        return rows


# ------------------------------------------------------------------ A14: KL
def hidden_states(lm, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One ``lm`` call under the current condition: the backbone's final hidden states (B, T, D), with the
    graph if grad is on, and the last position's logits (B, 1, V). Going through ``lm`` keeps every
    hook's bookkeeping (``HookedLM`` and ``TapLM`` reset their state per call)."""
    box = {}
    hd = lm.model.base_model.register_forward_hook(lambda mod, a, out: box.__setitem__("h", out[0]))
    try:
        lg = lm(ids, keep=1)
    finally:
        hd.remove()
    return box["h"], lg


def lm_head(lm) -> nn.Linear:
    head = lm.model.get_output_embeddings()
    if not isinstance(head, nn.Linear):
        raise SystemExit(f"--distill-frac needs a linear LM head, got {type(head).__name__}")
    return head


def check_head(head: nn.Linear, h: torch.Tensor, lg: torch.Tensor) -> None:
    """The head applied to the captured states must give the model's own logits (no soft-capping etc.)."""
    mine = head(h[:, -1:].to(head.weight.device)).float()
    ref = lg[:, -1:].float().to(mine.device)
    err = (mine - ref).abs().max().item()
    if not err <= 1e-2 * (1 + ref.abs().max().item()):
        raise SystemExit(f"the LM head of the captured hidden states differs from the model's logits by {err:.3g}: "
                         "this model post-processes its logits; --distill-frac does not support it")


def _kl_chunk(s_h, t_h, weight, bias, temp: float):
    zs = F.linear(s_h, weight, bias).float() / temp
    zt = F.linear(t_h, weight, bias).float() / temp
    lt, ls = zt.log_softmax(-1), zs.log_softmax(-1)
    kl = (lt.exp() * (lt - ls)).sum(-1) * (temp * temp)
    return kl, (zt.argmax(-1) == zs.argmax(-1)).float()


def chunked_kl(s_h: torch.Tensor, t_h: torch.Tensor, head: nn.Linear, temp: float = 1.0,
               chunk: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position T²·KL(teacher ‖ student) (B, N) float32 and argmax agreement (B, N) bool, from the
    final hidden states of the student ``s_h`` and the teacher ``t_h`` (B, N, D) through ``head``. Chunks
    of ``chunk`` positions; with grad, each chunk is checkpointed (backward recomputes its logits)."""
    dev = head.weight.device
    s_h, t_h = s_h.to(dev), t_h.to(dev).detach()
    kls, ags = [], []
    for i in range(0, s_h.shape[1], chunk):
        a, b = s_h[:, i:i + chunk], t_h[:, i:i + chunk]
        if torch.is_grad_enabled() and a.requires_grad:
            kl, ag = checkpoint(_kl_chunk, a, b, head.weight, head.bias, temp, use_reentrant=False)
        else:
            kl, ag = _kl_chunk(a, b, head.weight, head.bias, temp)
        kls.append(kl)
        ags.append(ag)
    return torch.cat(kls, 1), torch.cat(ags, 1).bool()


def kl_reference(t_logits: torch.Tensor, s_logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """The direct T²·KL(teacher ‖ student) per position from full logits (tests, small vocabularies)."""
    lt, ls = (t_logits.float() / temp).log_softmax(-1), (s_logits.float() / temp).log_softmax(-1)
    return F.kl_div(ls, lt, log_target=True, reduction="none").sum(-1) * (temp * temp)


class Distiller:
    """The A14 loss on one tagged text batch: (ce = per-position KL, ok = agreement, m = all true)."""

    def __init__(self, lm, ns: argparse.Namespace) -> None:
        self.head, self.N, self.temp, self.chunk = lm_head(lm), ns.distill_positions, ns.distill_temp, ns.distill_chunk
        self.checked, self.losses, self.agree = False, [], []

    def __call__(self, lm, ids: torch.Tensor):
        with torch.no_grad(), lm.condition(0, False):             # the teacher: full attention, side off
            t_h, t_lg = hidden_states(lm, ids)
        if not self.checked:
            check_head(self.head, t_h, t_lg)
            self.checked = True
        s_h, _ = hidden_states(lm, ids)                           # the student: pl.train's condition(window, True)
        kl, agree = chunked_kl(s_h[:, -self.N:], t_h[:, -self.N:], self.head, self.temp, self.chunk)
        self.losses.append(kl.detach().mean().item())
        self.agree.append(agree.float().mean().item())
        return kl, agree, torch.ones_like(agree)


# ------------------------------------------------------------- A15: attention
class _EagerConfig:
    """A read-through proxy of a module's config that reports eager attention (this module only)."""

    _attn_implementation = "eager"

    def __init__(self, cfg) -> None:
        object.__setattr__(self, "_cfg", cfg)

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_cfg"), k)


def capture_attention(lm, ids: torch.Tensor, layers: list[int], rows: int, keep: int = 1
                      ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """The frozen model under the full condition, in no_grad, with eager attention at ``layers`` only:
    ({layer: (B, rows, T) float32 attention of the last ``rows`` query positions onto every position,
    averaged over heads}, the last ``keep`` positions' logits). Full coordinates (the sink included).

    The config proxy, the causal mask and the hooks are removed on exit, whatever happens."""
    T = ids.shape[1]
    got: dict[int, torch.Tensor] = {}

    def pre(mod, args, kwargs):                                   # explicit causal mask (sdpa passes None)
        h = args[0] if args else kwargs["hidden_states"]
        kwargs["attention_mask"] = pl.window_mask(h.shape[1], h.shape[1], "sliding", 0, h.device, h.dtype)
        return args, kwargs

    def post_for(i):
        def post(mod, args, out):
            w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            if not torch.is_tensor(w) or w.dim() != 4 or w.shape[-1] != T:
                raise RuntimeError(f"block {i}: no (B, H, T, T) attention weights under the eager proxy")
            got[i] = w[:, :, -rows:].float().mean(1)
        return post
    mods = [pl._child(lm.blocks[i], pl.ATTN_MIXERS) for i in layers]
    if any(m is None or not hasattr(m, "config") for m in mods):
        raise SystemExit("--admit-target: an attention module without a config (cannot switch it to eager)")
    saved, handles = [m.config for m in mods], []
    try:
        for i, m in zip(layers, mods):
            m.config = _EagerConfig(m.config)
            handles.append(m.register_forward_pre_hook(pre, with_kwargs=True))   # after the window hooks
            handles.append(m.register_forward_hook(post_for(i)))
        with torch.no_grad(), lm.condition(0, False):
            lg = lm(ids, keep=keep)
    finally:
        for hd in handles:
            hd.remove()
        for m, c in zip(mods, saved):
            m.config = c
    return got, lg


def query_rows(T: int, n_ans: torch.Tensor, qlen: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Each row's query rows [q0, a0] in full coordinates: a0 = T − n_ans is the answer's space, which
    predicts digit 1, and q0 = a0 − qlen the query's first token. Returns (q0 (B,), a0 (B,), R = the
    number of last rows that cover every row's span)."""
    a0 = T - n_ans
    q0 = a0 - qlen
    return q0, a0, int(T - q0.min())


def row_mean(att: torch.Tensor, q0: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
    """(B, R, T) attention of the last R rows → (B, T): the mean over each row's own query rows [q0, a0]."""
    B, R, T = att.shape
    r = torch.arange(T - R, T, device=att.device)
    sel = ((r[None] >= q0.to(att.device)[:, None]) & (r[None] <= a0.to(att.device)[:, None])).to(att.dtype)
    return (sel[..., None] * att).sum(1) / sel.sum(1, keepdim=True).clamp_min(1)


def window_targets(att: torch.Tensor, s: torch.Tensor, valid: torch.Tensor, P: int, n: int, delay: int,
                   q0: torch.Tensor, sink: int) -> dict:
    """The A15 target over the gate's candidates.

    ``att`` (B, T_full): the query rows' mean attention (full coordinates, sink first); ``s``, ``valid``
    (B, N): candidate onsets in side coordinates (``SlotLoop._store``); ``q0`` (B,): each row's first
    query position (full coordinates). A candidate is eligible if valid and closed before the query,
    s + delay < q0 − sink; its mass is the attention inside [s − P, s − P + n − 1] ∩ [0, T). Overlapping
    windows each count the shared positions. Returns ``target`` (B, N), normalised over the eligible
    candidates (zero elsewhere), ``elig`` (B, N), ``has`` (B,) rows with eligible mass > 0, and
    ``covered`` (B,): the share of the pre-query attention (sink excluded) inside the eligible windows'
    union, a diagnostic of how much of what full attention reads the candidates can hold."""
    a = att[:, sink:].float().to(s.device)
    B, T = a.shape
    qs = (q0.to(s.device) - sink)[:, None]                        # first query position, side coordinates
    C = F.pad(a.cumsum(1), (1, 0))                                # C[:, k] = Σ_{c < k} a[:, c]
    lo = (s - P).clamp(0, T)
    hi = (s - P + n).clamp(0, T)                                  # exclusive end
    elig = valid & (s + delay < qs)
    mass = (C.gather(1, hi) - C.gather(1, lo)).clamp_min(0) * elig
    tot = mass.sum(1, keepdim=True)
    has = (tot[:, 0] > 0) & elig.any(1)
    target = torch.where(has[:, None], mass / tot.clamp_min(1e-30), torch.zeros_like(mass))
    diff = torch.zeros(B, T + 1, device=s.device)                 # union of the eligible windows
    one = elig.to(diff.dtype)
    diff.scatter_add_(1, lo, one)
    diff.scatter_add_(1, hi, -one)
    union = diff[:, :T].cumsum(1) > 0.5
    pre = torch.arange(T, device=s.device)[None] < qs
    covered = (a * (union & pre)).sum(1) / (a * pre).sum(1).clamp_min(1e-30)
    return dict(target=target, elig=elig, has=has, covered=covered)


def admit_kl(lp: torch.Tensor, target: torch.Tensor, elig: torch.Tensor, has: torch.Tensor) -> torch.Tensor:
    """Per-row KL(target ‖ q) (B,), q = softmax of log p over the eligible candidates; 0 where ``has`` is
    false. ``lp`` (B, N) carries the gate's gradient; ``target`` is constant."""
    ok = elig & has[:, None]
    logit = torch.where(ok, lp, torch.full_like(lp, float("-inf")))
    logit = torch.where(has[:, None], logit, torch.zeros_like(lp))   # rows without a target: finite, unused
    lq = logit.log_softmax(-1)
    pos = ok & (target > 0)
    t = target.to(lq.dtype)
    kl = torch.where(pos, t * (t.clamp_min(1e-30).log() - lq), torch.zeros_like(lq)).sum(-1)
    return torch.where(has, kl, torch.zeros_like(kl))


@contextmanager
def store_tap(loop):
    """Keep the store dict of the next ``loop._store`` call (live tensors, with grad) in the yielded box,
    by an instance attribute over the class method; removed on exit."""
    box: dict = {}
    orig = loop._store
    had = "_store" in vars(loop)

    def tapped(f):
        st = orig(f)
        box["st"] = st
        return st
    loop._store = tapped
    try:
        yield box
    finally:
        if had:
            loop._store = orig
        else:
            del loop._store


class Admitter:
    """The A15 term on one needle batch: the teacher pass, then the student's (``prev``) with its store."""

    def __init__(self, lm, ns: argparse.Namespace) -> None:
        loop = getattr(lm.side, "loop", None)
        if loop is None or not hasattr(loop, "_store"):
            raise SystemExit("--admit-target needs a W1s side (side.loop._store)")
        self.loop, self.weight, self.layers = loop, ns.admit_weight, admit_layers(ns.admit_layers, lm)
        self.losses, self.covered, self.n_elig = [], [], []

    def __call__(self, lm, ids: torch.Tensor, mask: torch.Tensor, prev):
        qlen = getattr(ids, "query_len", None)
        if qlen is None or len(qlen) != len(ids):
            raise RuntimeError("A15 needs each row's query length (ids.query_len from the task proxy)")
        T = ids.shape[1]
        q0, a0, R = query_rows(T, mask.sum(1).long().cpu(), torch.tensor(qlen))
        got, _ = capture_attention(lm, ids, self.layers, R)        # before the student: last_stats stays its
        att = row_mean(torch.stack([got[i] for i in self.layers]).mean(0), q0, a0)
        with store_tap(self.loop) as box:
            ce, ok, m = prev(lm, ids, mask)
        st = box.get("st")
        if st is None:
            raise RuntimeError("A15: the student pass did not run the side's store")
        tg = window_targets(att, st["s"], st["valid"], self.loop.P, self.loop.n, self.loop.delay, q0, lm.sink)
        kl = admit_kl(st["lp"], tg["target"], tg["elig"], tg["has"])
        has = tg["has"]
        loss = self.weight * (kl * has).sum() / has.sum().clamp_min(1)
        self.losses.append(loss.detach().item() / self.weight)
        self.covered.append(tg["covered"].mean().item())
        self.n_elig.append(tg["elig"].sum(1).float().mean().item())
        return ce + loss.to(ce.device), ok, m


# --------------------------------------------------------------- training
class _Batches:
    """The task as ``pl.train`` sees it under A14/A15 (``batch``; every other attribute passes through)."""

    def __init__(self, task, text: TextWindows | None, frac: float, seed: int, track_query: bool) -> None:
        self.task, self.text, self.frac, self.track = task, text, frac, track_query
        self.drng = random.Random(seed * 1_000_003 + 14)          # A14's own draws (steps, then offsets)
        self.kinds: list[str] = []                                # per training step: "d" text, "n" needle

    def __getattr__(self, k):
        return getattr(self.task, k)

    def batch(self, B: int, rng: random.Random):
        if self.frac > 0 and (self.frac >= 1 or self.drng.random() < self.frac):
            ids = self.text.batch(B, self.drng)
            ids.distill = True
            self.kinds.append("d")
            return ids, torch.ones_like(ids, dtype=torch.bool), None
        self.kinds.append("n")
        if not self.track:
            return self.task.batch(B, rng)
        qlens, orig = [], self.task._record

        def rec(*a, **kw):                                        # (j, value_at, natural, heads, query_len, facts)
            qlens.append(kw["query_len"] if "query_len" in kw else a[4])
            return orig(*a, **kw)
        self.task._record = rec
        try:
            ids, mask, gaps = self.task.batch(B, rng)
        finally:
            del self.task._record
        ids.query_len = qlens
        return ids, mask, gaps


@contextmanager
def training(lm, task, args: argparse.Namespace, ns: argparse.Namespace):
    """Install A14/A15 for one ``pl.train`` call: yields (the task proxy, the stats dict). ``pl.scores``
    is replaced for the duration and restored on exit. The text is tokenised with the task's tokenizer."""
    distiller = Distiller(lm, ns) if distill_on(ns) else None
    admitter = Admitter(lm, ns) if admit_on(ns) else None
    text = TextWindows(task.tok, ns.distill_text, args.seq_len, lm.sink) if distiller is not None else None
    proxy = _Batches(task, text, ns.distill_frac, args.seed, admitter is not None)
    prev = pl.scores

    def scores(lm_, ids, mask):
        if getattr(ids, "distill", False):
            return distiller(lm_, ids)
        if admitter is not None:
            return admitter(lm_, ids, mask, prev)
        return prev(lm_, ids, mask)
    stats = {"distiller": distiller, "admitter": admitter, "text_tokens": len(text.ids) if text is not None else None,
             "kinds": proxy.kinds}
    pl.scores = scores
    try:
        yield proxy, stats
    finally:
        pl.scores = prev


def _tenths(v: list[float]) -> list[float]:
    """Means of ten consecutive, near-equal parts of v (fewer when v is short)."""
    k = min(10, len(v))
    cuts = [round(i * len(v) / k) for i in range(k + 1)]
    return [sum(v[a:b]) / (b - a) for a, b in zip(cuts[:-1], cuts[1:])] if k else []


def summary(ns: argparse.Namespace, stats: dict | None) -> dict:
    """The JSON records: ``distill`` and/or ``admit`` (settings, steps, mean loss, its trend by tenths)."""
    out = {}
    stats = stats or {}
    d, a = stats.get("distiller"), stats.get("admitter")
    if distill_on(ns):
        v = d.losses if d is not None else []
        out["distill"] = {"frac": ns.distill_frac, "text": ns.distill_text, "text_tokens": stats.get("text_tokens"),
                          "positions": ns.distill_positions, "temp": ns.distill_temp, "chunk": ns.distill_chunk,
                          "steps": len(v), "loss_mean": sum(v) / len(v) if v else None, "loss_by_tenth": _tenths(v),
                          "agree_mean": sum(d.agree) / len(d.agree) if v else None,
                          "loss": "T^2 KL(full || windowed+side), mean over the last N positions",
                          # pl.train's curve entries at "d" steps are this KL (token_acc = argmax agreement)
                          "step_kinds": "".join(stats.get("kinds") or [])}
    if admit_on(ns):
        v = a.losses if a is not None else []
        out["admit"] = {"target": ns.admit_target, "weight": ns.admit_weight, "layers_spec": ns.admit_layers,
                        "layers": a.layers if a is not None else None, "steps": len(v),
                        "kl_mean": sum(v) / len(v) if v else None, "kl_by_tenth": _tenths(v),
                        "covered_mean": sum(a.covered) / len(v) if v else None,
                        "eligible_mean": sum(a.n_elig) / len(v) if v else None,
                        "loss": "weight * KL(teacher window mass || gate p), over candidates closed before the query"}
    return out

