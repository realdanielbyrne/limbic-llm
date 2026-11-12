"""Time-lagged cross-layer injection on a frozen LM: the state of an EARLIER token read at layer L, added at
layer L' at the current token, for skip-ahead (L' > L), same-layer (L' = L) and feedback (L' < L) cells.

Two teacher-forced passes per batch, nothing trained (zero parameters; hooks and plain tensors only).

Pass 1 (unsteered, under the arm's attention condition): the block outputs H^L_p (float32, sink column
excluded, text positions p = 0 .. T−1) at every source layer L, the logits z^base, ‖h_l‖ at every block, and in
a windowed condition the full-attention logits z^full.

Pass 2 (steered): at the output of block L', at every text position t ≥ ``--steer-from`` where u_t is defined,

    h^{L'}_t ← h^{L'}_t + g ‖h^{L'}_t‖ u_t,        u_t a unit vector built from pass-1 states H^L_p, p < t only.

Exactness. The source states come from the unsteered pass. A skip-ahead cell (L' > L) is exact: with a single
injection at L', the states at L ≤ L' − 1 of pass 2 equal those of pass 1. The same-layer cell (L' = L) is exact
as defined, with the source taken BEFORE the injection: block L's pre-injection output does not depend on an
injection at its own output, so pass-2 pre-injection states at L equal pass 1. Both are checked bitwise per arm
(``checks.source_unchanged``; ``fb_relchange`` = 0). A feedback cell (L' < L) is the first-order (open-loop)
approximation of a closed loop: in pass 2 the injection at L' moves the states at L that a closed loop would
read. For feedback cells the runner records two first-order DIAGNOSTICS of that approximation (not bounds):
    fb_relchange_t = ‖H2^L_t − H^L_t‖ / ‖H^L_t‖  (H2: pass-2 states at L; measured at token t, while the source
                     reads earlier tokens), and
    fb_ucos_t      = cos(u_t, u'_t), u' the same construction from H2: the agreement of ONE closed-loop iterate
                     with the open-loop direction. It does not bound the distance to a closed-loop fixed point or
                     the error in loss. JSON: ``cell_kind`` ("skip", "same", "feedback"), ``exact`` and
                     ``approximation`` per arm.

Sources (``--sources``; every one strictly causal: u_t uses states of tokens < t only; m_t = (1/t) Σ_{p<t} H_p the
causal running mean of the source layer; A^τ_j = Σ_{p≤j} λ^{j−p} H_p / Σ_{p≤j} λ^{j−p}, λ = e^{−1/τ}, a
bias-corrected causal exponential average; N(v) = v/‖v‖, undefined (no injection) where ‖v‖ ≤ 1e-4 ‖m_t‖):
    lag:k               u_t = N(H_{t−k} − m_t)                     the single state of token t − k, centred
    ema:τ[:skip=m]      u_t = N(A^τ_{t−1−m} − m_t)                 recent average (optionally skipping the
                                                                   last m tokens), centred
    slowfast:τs:τf      u_t = N(A^{τs}_{t−1} − A^{τf}_{t−1})       older context emphasised over recent
    beyond:W            u_t = N((1/b_t) Σ_{p<b_t} H_p),  b_t = ⌊t/W⌋W   the mean of the tokens OUTSIDE t's
                                                                   attention block (block mode aligned to the
                                                                   first text token, as ``pl.window_mask``):
                                                                   content the windowed model cannot attend to.
                                                                   W must be a multiple of ``--window`` (checked;
                                                                   else b_t would fall inside the window).
                                                                   Not centred: any causal mean used for
                                                                   centring would include in-window tokens.
A source is first defined at t_src (lag:k → max(k, 2), ema:τ:skip=m → max(1 + m, 2), slowfast → 2, beyond:W → W;
at t = 1 the lag:1, ema:τ and slowfast vectors are exactly 0); the true arm steers from
t_0 = max(``--steer-from``, t_src).

Variants (every cell, same norm, paired rows, the same steered positions as the true arm wherever defined):
    true      as above;
    foreign   the same construction from another document's pass-1 states at the same positions (row b reads
              row (b + 1) mod B of the batch; batches are interleaved, so that row is a distant part of the
              file): same norm and statistics, wrong content. true − foreign is the pre-stated comparison;
    shuffle   the true construction taken at a far-away random earlier position: u_t ← u_{s_t}, s_t uniform over
              the positions s ≤ t − G where the true source is defined (``--shuffle-gap`` G; seeded per row), and
              injected only where the true arm injects. u_{s_t} uses tokens < s_t < t, so it stays causal; it keeps
              the content type but breaks the lag. It is undefined for t < G + t_src, so it steers from
              t_late = max(``--steer-from``, G + t_src): on t ≥ t_late its steered set equals the true arm's;
    random    a seeded unit vector per row, constant over positions, steered from the same t_0 as the true arm it
              is paired with (keyed by L', gain and t_0: ``random:<seed>:to<L'>@g`` when t_0 = ``--steer-from``,
              ``random:<seed>:from<t_0>:to<L'>@g`` otherwise);
    none      g = 0 with the injection arithmetic running (must reproduce the unsteered logits bitwise).

Conditions (``text.CONDITIONS``, one ``pl.HookedLM`` per condition, block window with a sink, its side off):
``full``; ``attn`` attention windowed to W, GDN state intact; ``all`` every token mixer windowed (hybrids only).
Scored targets are those predicted from text positions ≥ W in every condition (``TextTask.first``), so arms
and conditions are paired on identical tokens.

Measures per arm (results JSON; per item = per row, paired across arms): loss and Δloss against the
unsteered run (= ``none``, checked bitwise) with a paired bootstrap interval; top-1 accuracy, entropy and its
change against the unsteered run (signed and absolute); KL to the unsteered run (and loss / KL to the
full-context run and gap closure in windowed conditions); Δloss by offset within the window (buckets 0–W/8,
W/8–W/2, W/2–W, and each of the W offsets) and by position (log₂ buckets); the same offset buckets restricted to
t ≥ t_late (``late_off_*``, for the matched shuffle contrast); the steered share; the injection ratio
‖Δh‖/‖h‖ (= g where steered); cos(u_t, h^{L'}_t); the residual-norm ratio ‖h_l(arm)‖/‖h_l(unsteered)‖ at every
block; the logit-lens rank of the target at ``--lens-depths`` (default 4 depths, as ``text.py``); a logit-lens
readout of the injection (the final norm and unembedding applied to u_t ‖h^{L'}_t‖): ``lens_recent`` = the share
of its top-``--lens-k`` tokens that are among the last ``--lens-recent`` (32) input tokens x_{t−31..t}, a
repetition-risk measure (compare with the random and foreign arms); the feedback diagnostics above; the git
commit and torch / transformers / fla versions.

``contrasts`` holds, for every true arm, paired bootstraps over rows of (keys ``*_early``: the same over offsets
0 .. W/8 − 1 only, where the Stage-0 effect sits):
    vs_none                 true − unsteered;
    minus_foreign           true − foreign;
    minus_random            true − random (same L', gain and t_0);
    minus_shuffle_matched   true − shuffle over t ≥ t_late, where both arms steer the same tokens (residual
                            mismatch: the true arm's injections at t_0 .. t_late − 1 still reach later tokens
                            through attention and the GDN state);
    minus_shuffle           true − shuffle over all scored tokens (dose-mismatched: ``dose_shuffle`` < 1);
with the steered share of each arm (``steered_frac_<control>``, ``dose_<control>`` = its ratio to the true arm).
Per token (``common.TraceWriter`` through ``measure.ArmTraces``, outside the repo): out/loss, out/ent, out/rank,
out/top1, out/p_tgt, out/p_max, out/loss_base, out/ent_base, out/kl_base (+ out/loss_full, out/kl_full), inj/on,
inj/ratio, inj/cos_uh, inj/hnorm, lens/recent, lens/rank_d<d>, out/norm_ratio (blocks L' .. n−1), fb/relchange
(cells with a source layer), fb/ucos (feedback cells), pos/token, pos/text, pos/offset, pos/block, pos/scored.

Statistics. Every interval is a 95% paired bootstrap, uncorrected. ``--analyse`` additionally gives
Benjamini–Hochberg q values over every interval of the report, the expected number of uncorrected false
positives, and states how the displayed gain was chosen (post hoc on the same rows unless ``--at-gain``).
``--cells-from`` selection (``select_cells``): cells with any true − foreign interval (whole rows or early
offsets) or helpful (negative) Δloss vs none surviving BH at ``--select-q`` over the screen's tests, plus the
``--cells-top`` cells with the most negative true − foreign t statistic. The follow-up run must use rows
disjoint from the screen's (``--row-offset``; refused otherwise), and stores ``selection_provenance``.

Cost. Pass 1 (and the full pass) are shared by a group of ``--group`` arms per batch; each arm costs one
forward pass plus logit-lens unembeddings (its injected vectors and the target ranks at the lens depths).
``--count`` prints the arm counts without loading weights. ``--screen`` runs one gain (0.05) over the whole
grid; ``--cells-from <screen JSON>`` then restricts a full gain sweep to the selected cells, on other rows.

    python experiments/ca2/laggrid.py --model Qwen/Qwen3.5-2B-Base --screen 2>&1 \\
        | tee results/evolve/round12/stage0/laggrid_screen_Qwen3.5-2B-Base.log
    python experiments/ca2/laggrid.py --model Qwen/Qwen3.5-2B-Base --cells-from <screen.json> --cells-top 6 \\
        --row-offset 128
    python experiments/ca2/laggrid.py --analyse results/evolve/round12/stage0/laggrid_Qwen3.5-2B-Base.json
    python experiments/ca2/laggrid.py --tiny --tiny-arch qwen35          # CPU smoke
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402
from ca2 import common  # noqa: E402
from ca2 import measure as ms  # noqa: E402
from ca2 import text as tx  # noqa: E402

SOURCE_KINDS = ("lag", "ema", "slowfast", "beyond")
VARIANTS = ("true", "foreign", "shuffle", "random")
GRID = {"hybrid": ("7,11,12,19", "3,7,11,12,15,19"), "attention": ("7,14,21", "3,7,14,21,25"),
        "tiny": ("1,2", "0,1,2")}
SOURCES = "lag:1,lag:8,lag:64,ema:16,ema:256,slowfast:256:8,beyond:64"
TINY_SOURCES = "lag:1,lag:8,ema:16,ema:16:skip=4,slowfast:32:4,beyond:16"
APPROX = {"skip": "exact: the source layer lies below the injection, pass-2 source states equal pass 1",
          "same": "exact as defined: the source is the pre-injection state at L (equal in both passes)",
          "feedback": "first-order (open-loop) approximation of a closed loop: the source states come from the "
                      "unsteered pass; fb_relchange and fb_ucos are first-order diagnostics of the error, "
                      "not bounds"}


# ------------------------------------------------------------------------------------------------ CLI
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a = p.add_argument
    a("--model", default="Qwen/Qwen3.5-2B-Base")
    a("--dtype", default="bfloat16", choices=list(pl.DTYPES))
    a("--device", default="cuda")
    a("--attn-impl", default="sdpa")
    a("--tiny", action="store_true", help="tiny random model + byte tokenizer, CPU, small defaults")
    a("--tiny-arch", default="qwen35", choices=["qwen35", "qwen3"])
    a("--seed", type=int, default=0)
    a("--text-file", default="~/limbic-data/wikitext103_test.txt")
    a("--max-chars", type=int, default=None, help="read only the first N characters (default: all; tiny: 200k)")
    a("--seq-len", type=int, default=None, help="text tokens per row (default 1024; tiny 96)")
    a("--n-seqs", type=int, default=None, help="rows (default 128; tiny 4)")
    a("--row-offset", type=int, default=0, help="skip the first N rows of the file (disjoint rows for a "
                                                  "--cells-from follow-up)")
    a("--batch", type=int, default=None, help="rows per forward (default 2; >= 2 for the foreign variant)")
    a("--sink", type=int, default=1)
    a("--window", type=int, default=None, help="attention window W (default 64; tiny 16)")
    a("--conditions", default="full,attn,all", help="'all' is dropped on a model without linear layers")
    a("--layers", default=None, help="source layers L (default: hybrid 7,11,12,19; attention 7,14,21; tiny 1,2)")
    a("--targets", default=None, help="target layers L' (default: hybrid 3,7,11,12,15,19; attention "
                                      "3,7,14,21,25; tiny 0,1,2)")
    a("--cells", default=None, help="restrict to these L-L' pairs, e.g. 7-11,11-7")
    a("--cells-from", default=None, help="screen JSON: restrict to the cells that matter there")
    a("--cells-top", type=int, default=4, help="with --cells-from: also the top K cells by true - foreign")
    a("--select-q", type=float, default=0.10, help="with --cells-from: BH level of the selection")
    a("--sources", default=None, help=f"default {SOURCES} (tiny: {TINY_SOURCES})")
    a("--variants", default="true,foreign,shuffle,random")
    a("--random-seeds", default="0")
    a("--gains", default=None, help="default 0.02,0.05,0.1,0.2 (tiny 0.3); --screen: 0.05")
    a("--screen", action="store_true", help="one gain (0.05) over the whole grid")
    a("--steer-from", type=int, default=None, help="first steered text position (default 32; tiny 8)")
    a("--shuffle-gap", type=int, default=None, help="shuffle draws s_t <= t - G (default 128; tiny 16)")
    a("--lens-k", type=int, default=10)
    a("--lens-recent", type=int, default=32)
    a("--lens-depths", default=None, help="blocks for the target's logit-lens rank (default 4 depths as text.py; "
                                          "'none' disables)")
    a("--group", type=int, default=32, help="arms sharing one unsteered pass per batch")
    a("--trace-tokens", type=int, default=-1, help="rows per arm with per-token arrays (-1: all)")
    a("--arms", default=None, help="comma list of arm names or fnmatch patterns (e.g. 'none,true:lag:8:*')")
    a("--boot", type=int, default=2000)
    a("--name", default=None)
    a("--stage", default="stage0")
    a("--out", default=None)
    a("--trace-dir", default=None)
    a("--force", action="store_true")
    a("--count", action="store_true", help="print the arm counts per condition and exit (no weights loaded)")
    a("--analyse", default=None, metavar="JSON", help="print the true - control matrices of a results JSON")
    a("--control", default="foreign", choices=["foreign", "shuffle", "random", "none"], help="--analyse")
    a("--at-gain", type=float, default=None, help="--analyse: this gain instead of the best one")
    a("--bucket", default="all", choices=["all", "early"], help="--analyse: all scored tokens, or offsets < W/8")
    args = p.parse_args(argv)
    tiny = dict(seq_len=96, n_seqs=4, batch=2, window=16, steer_from=8, shuffle_gap=16, max_chars=200_000,
                gains="0.3", sources=TINY_SOURCES)
    real = dict(seq_len=1024, n_seqs=128, batch=2, window=64, steer_from=32, shuffle_gap=128, max_chars=None,
                gains="0.02,0.05,0.1,0.2", sources=SOURCES)
    if args.screen and args.gains is None:
        args.gains = "0.05"
    for k, v in (tiny if args.tiny else real).items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.tiny and args.dtype == "bfloat16" and args.device != "cuda":
        args.dtype = "float32"
    for v in args.variants.split(","):
        if v and v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; known {VARIANTS}")
    if args.row_offset < 0:
        raise SystemExit("--row-offset must be >= 0")
    return args


def check_sources(args) -> list[dict]:
    """Parsed ``--sources``; SystemExit on a bad spec or a ``beyond:W`` whose W is not a multiple of the window
    (then b_t = ⌊t/W⌋W would lie inside t's attention block and the source would read attendable tokens)."""
    out = []
    for s in args.sources.split(","):
        if not s.strip():
            continue
        try:
            src = parse_source(s)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        if src["kind"] == "beyond" and (args.window < 1 or src["W"] % args.window):
            raise SystemExit(f"{src['name']}: W must be a multiple of --window {args.window} so that every token it "
                             "averages lies outside the attention window")
        out.append(src)
    return out


def ints(s: str | None) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()] if s else []


# --------------------------------------------------------------------------------------------- sources
def parse_source(spec: str) -> dict:
    """'lag:8', 'ema:256', 'ema:16:skip=4' (or skip4), 'slowfast:256:8', 'beyond:64' -> dict with a canonical
    ``name`` (characters usable in arm names)."""
    parts = spec.strip().split(":")
    kind = parts[0]
    try:
        if kind == "lag" and len(parts) == 2 and int(parts[1]) >= 1:
            return {"kind": "lag", "k": int(parts[1]), "name": f"lag:{int(parts[1])}"}
        if kind == "ema" and len(parts) in (2, 3):
            tau, skip = float(parts[1]), 0
            if len(parts) == 3:
                if not parts[2].startswith("skip"):
                    raise ValueError
                skip = int(parts[2][4:].lstrip("="))
            if tau <= 0 or skip < 0:
                raise ValueError
            return {"kind": "ema", "tau": tau, "skip": skip,
                    "name": f"ema:{tau:g}" + (f":skip{skip}" if skip else "")}
        if kind == "slowfast" and len(parts) == 3:
            ts, tf = float(parts[1]), float(parts[2])
            if ts <= tf or tf <= 0:
                raise ValueError
            return {"kind": "slowfast", "tau_s": ts, "tau_f": tf, "name": f"slowfast:{ts:g}:{tf:g}"}
        if kind == "beyond" and len(parts) == 2 and int(parts[1]) >= 1:
            return {"kind": "beyond", "W": int(parts[1]), "name": f"beyond:{int(parts[1])}"}
    except ValueError:
        pass
    raise ValueError(f"bad source {spec!r}; use lag:k, ema:tau[:skip=m], slowfast:tau_s:tau_f (tau_s > tau_f), "
                     "beyond:W")


def source_start(src: dict) -> int:
    """t_src: the first text position where the source can be defined (module docstring)."""
    return {"lag": lambda: max(src["k"], 2), "ema": lambda: max(1 + src["skip"], 2), "slowfast": lambda: 2,
            "beyond": lambda: src["W"]}[src["kind"]]()


def steer_start(src: dict, steer_from: int) -> int:
    """t_0 = max(steer_from, t_src): the first position the true arm (and its random control) steers."""
    return max(int(steer_from), source_start(src))


def late_start(src: dict, steer_from: int, gap: int) -> int:
    """t_late = max(steer_from, G + t_src): from here the shuffle arm steers the same tokens as the true arm."""
    return max(int(steer_from), int(gap) + source_start(src))


def mean_before(H: torch.Tensor) -> torch.Tensor:
    """m_t = (1/t) Σ_{p<t} H_p, (B, T, D); m_0 = 0."""
    B, T, D = H.shape
    cs = torch.cat([H.new_zeros(B, 1, D), H.cumsum(1)[:, :-1]], 1)
    n = torch.arange(T, device=H.device, dtype=H.dtype).clamp_min(1)
    return cs / n[None, :, None]


def ema_inclusive(H: torch.Tensor, tau: float) -> torch.Tensor:
    """A^τ_j = Σ_{p≤j} λ^{j−p} H_p / Σ_{p≤j} λ^{j−p} for every j (a sequential scan: strictly causal), (B, T, D)."""
    lam = math.exp(-1.0 / tau)
    num = H.new_zeros(H.shape[0], H.shape[2])
    den = 0.0
    out = torch.empty_like(H)
    for j in range(H.shape[1]):
        num = lam * num + H[:, j]
        den = lam * den + 1.0
        out[:, j] = num / den
    return out


def shift_right(X: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Y_t = X_{t−k} (zero and undefined for t < k): (Y, ok (T,))."""
    T = X.shape[1]
    Y = torch.zeros_like(X)
    ok = torch.zeros(T, dtype=torch.bool, device=X.device)
    if k < T:
        Y[:, k:] = X[:, : T - k]
        ok[k:] = True
    return Y, ok


def source_vectors(H: torch.Tensor, src: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The true construction (module docstring): u (B, T, D) unit or zero, ok (B, T) bool; u_t uses H_{<t}."""
    B, T, D = H.shape
    m = mean_before(H)
    t = torch.arange(T, device=H.device)
    if src["kind"] == "lag":
        x, ok = shift_right(H, src["k"])
        v, ok = x - m, ok & (t >= 1)
    elif src["kind"] == "ema":
        a, ok = shift_right(ema_inclusive(H, src["tau"]), 1 + src["skip"])
        v = a - m
    elif src["kind"] == "slowfast":
        a, ok = shift_right(ema_inclusive(H, src["tau_s"]) - ema_inclusive(H, src["tau_f"]), 1)
        v = a
    elif src["kind"] == "beyond":
        W = src["W"]
        b = (t // W) * W                                            # first position of t's block
        cs = torch.cat([H.new_zeros(B, 1, D), H.cumsum(1)], 1)       # cs[:, j] = Σ_{p<j} H_p
        v = cs[:, b] / b.clamp_min(1).to(H.dtype)[None, :, None]
        ok = b >= 1
    else:
        raise ValueError(src)
    vn = v.norm(dim=-1)
    ok = ok[None, :].expand(B, T) & (vn > 1e-4 * m.norm(dim=-1)) & (vn > 0)
    u = torch.where(ok[..., None], v / vn.clamp_min(1e-30)[..., None], torch.zeros_like(v))
    return u, ok


def shuffle_index(rows, T: int, gap: int, seed: int, ok: torch.Tensor | None = None) -> torch.Tensor:
    """s_t uniform over the positions s ≤ t − gap with ok[b, s] (all positions when ``ok`` is None), seeded by row;
    −1 where there is none: (B, T) long. s_t depends on ok[:, ≤ t − gap] only (causal)."""
    okn = (np.ones((len(rows), T), dtype=bool) if ok is None
           else ok.detach().cpu().numpy().astype(bool).reshape(len(rows), T))
    out = np.full((len(rows), T), -1, dtype=np.int64)
    hi = np.arange(T) - gap
    for b, r in enumerate(rows):
        rng = np.random.default_rng([int(seed), int(r), 7])
        x = rng.random(T)
        pos = np.flatnonzero(okn[b])                                   # defined positions, ascending
        if pos.size == 0:
            continue
        cnt = np.cumsum(okn[b])                                        # cnt[j] = #defined in [0, j]
        n_t = np.where(hi >= 0, cnt[np.maximum(hi, 0)], 0)
        k = np.minimum(np.floor(x * n_t).astype(np.int64), np.maximum(n_t - 1, 0))
        out[b] = np.where(n_t > 0, pos[k], -1)
    return torch.from_numpy(out)


def random_vectors(rows, D: int, seed: int, rs: int, device) -> torch.Tensor:
    """One seeded unit vector per row: (B, 1, D) float32."""
    out = []
    for r in rows:
        g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(rs) * 7_919 + int(r) + 1)
        out.append(F.normalize(torch.randn(D, generator=g), dim=0))
    return torch.stack(out)[:, None].to(device)


def variant_vectors(arm: dict, H: torch.Tensor, rows, args, true_uv=None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """u (B, T, D), ok (B, T) of an arm from the source states H (B, T, D) (unused by random / none)."""
    v = arm["variant"]
    if v in ("random", "none"):
        B, T, D = len(rows), H.shape[1], arm["d_model"]
        dev = H.device
        u = random_vectors(rows, D, args.seed, arm.get("seed", 0), dev).expand(B, T, D)
        ok = (torch.arange(T, device=dev) >= int(arm.get("start") or 0))[None].expand(B, T)
        return u, ok.contiguous()
    u, ok = true_uv if true_uv is not None else source_vectors(H, parse_source(arm["source"]))
    if v == "true":
        return u, ok
    if v == "foreign":
        if u.shape[0] < 2:
            raise ValueError("the foreign variant needs batches of >= 2 rows")
        return u.roll(-1, 0), ok.roll(-1, 0)
    if v == "shuffle":
        idx = shuffle_index(rows, u.shape[1], args.shuffle_gap, args.seed, ok).to(u.device)
        ic = idx.clamp_min(0)
        us = u.gather(1, ic[..., None].expand(-1, -1, u.shape[2]))
        oks = (idx >= 0) & ok.gather(1, ic) & ok                    # a defined source, where the true arm steers
        return torch.where(oks[..., None], us, torch.zeros_like(us)), oks
    raise ValueError(v)


# ----------------------------------------------------------------------------------------------- hooks
class StateCapture:
    """Forward hooks keeping block outputs (float32, sink columns dropped) at the given layers. Registered before
    an ``Injector`` on the same block (as ``run_group`` does), it sees the pre-injection output."""

    def __init__(self, blocks, layers, sink: int) -> None:
        self.h: dict[int, torch.Tensor] = {}
        self.sink = int(sink)
        self.handles = [blocks[l].register_forward_hook(self._hook(l)) for l in sorted(set(layers))]

    def _hook(self, l: int):
        def fn(mod, args, out):
            self.h[l] = (out[0] if isinstance(out, tuple) else out)[:, self.sink:].detach().float()
        return fn

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


class Injector:
    """h_t ← h_t + g‖h_t‖u_t at block ``layer``'s output on text positions t ≥ steer_from where ``ok``; float32
    arithmetic, cast back to the model dtype (with g = 0 the output equals the input bitwise). Records the
    pre-injection norm, cos(u, h) and the injection ratio, (B, T) each, text coordinates."""

    def __init__(self, block, u: torch.Tensor, ok: torch.Tensor, g: float, steer_from: int, sink: int) -> None:
        self.u, self.g, self.sink = u, float(g), int(sink)
        T = u.shape[1]
        on = ok.clone()
        on[:, : min(steer_from, T)] = False
        self.on = on
        self.rec: dict[str, torch.Tensor] = {}
        self.handle = block.register_forward_hook(self._hook)

    def _hook(self, mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        s = self.sink
        hb = h[:, s:].float()
        hn = hb.norm(dim=-1)
        on = self.on.to(h.device)
        u = self.u.to(h.device)
        delta = (self.g * hn * on.float())[..., None] * u
        nan = torch.full_like(hn, float("nan"))
        self.rec = {"hnorm": hn.detach(), "cos_uh": torch.where(on, F.cosine_similarity(u, hb, dim=-1, eps=1e-12),
                                                                nan).detach(),
                    "ratio": (delta.norm(dim=-1) / hn.clamp_min(1e-12)).detach(), "on": on.float()}
        y = torch.cat([h[:, :s], (hb + delta).to(h.dtype)], 1)
        return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y

    def remove(self) -> None:
        self.handle.remove()


@torch.no_grad()
def lens_recent(model, u: torch.Tensor, hnorm: torch.Tensor, on: torch.Tensor, ids_text: torch.Tensor, k: int,
                n_recent: int, chunk_elems: int = 1 << 26) -> torch.Tensor:
    """Share of the top-k logit-lens tokens of x_t = u_t‖h_t‖ (final norm + unembedding) that are among the input
    tokens x_{t−n_recent+1 .. t}; NaN where not ``on``. (B, T)."""
    norm, head = ms.final_norm(model), model.get_output_embeddings()
    dt, dev = head.weight.dtype, head.weight.device
    B, T, D = u.shape
    V = head.weight.shape[0]
    ids = ids_text.to(dev)
    pad = torch.full((B, n_recent - 1), -1, dtype=ids.dtype, device=dev)
    recent = torch.cat([pad, ids], 1).unfold(1, n_recent, 1)          # (B, T, n_recent): x_{t−n+1 .. t}
    out = torch.full((B, T), float("nan"), device=dev)
    step = max(1, chunk_elems // max(1, B * V))
    for a in range(0, T, step):
        sl = slice(a, a + step)
        x = (u[:, sl].to(dev) * hnorm[:, sl, None].to(dev)).to(dt)
        top = head(norm(x)).float().topk(k, dim=-1).indices              # (B, t, k)
        hit = (top[..., :, None] == recent[:, sl, None, :]).any(-1).float().mean(-1)
        out[:, sl] = hit
    return torch.where(on.to(dev), out, torch.full_like(out, float("nan")))


# ----------------------------------------------------------------------------------------------- arms
def cell_kind(L: int, Lp: int) -> str:
    return "skip" if Lp > L else ("same" if Lp == L else "feedback")


def arm_name(variant: str, source: str | None = None, L: int | None = None, Lp: int | None = None,
             g: float = 0.0, seed: int = 0, start: int | None = None) -> str:
    """``start``: the random arm's t_0 when it differs from ``--steer-from`` (None otherwise)."""
    if variant == "none":
        return "none"
    if variant == "random":
        return f"random:{seed}:" + (f"from{start}:" if start is not None else "") + f"to{Lp}@{g:g}"
    return f"{variant}:{source}:L{L}to{Lp}@{g:g}"


def random_start(src: dict, steer_from: int) -> int | None:
    """The ``start`` key of the random arm paired with a true arm of source ``src``."""
    t0 = steer_start(src, steer_from)
    return None if t0 == int(steer_from) else t0


def grid_cells(args, n_blocks: int, family: str) -> list[tuple[int, int]]:
    Ld, Td = GRID[family]
    Ls, Ts = ints(args.layers or Ld), ints(args.targets or Td)
    for l in Ls + Ts:
        if not 0 <= l < n_blocks:
            raise SystemExit(f"layer {l} outside 0..{n_blocks - 1}")
    cells = [(L, Lp) for L in Ls for Lp in Ts]
    if args.cells:
        want = {tuple(int(x) for x in c.split("-")) for c in args.cells.split(",") if c}
        bad = [c for c in want if not all(0 <= x < n_blocks for x in c)]
        if bad:
            raise SystemExit(f"cells {bad} outside 0..{n_blocks - 1}")
        cells = sorted(want) if not (args.layers or args.targets) else [c for c in cells if c in want]
    if args.cells_from:
        keep = set(select_cells(json.loads(Path(args.cells_from).expanduser().read_text()), args.cells_top,
                                args.select_q))
        cells = [c for c in cells if c in keep]
    return cells


def make_arms(args, cells, d_model: int) -> list[dict]:
    """Every arm of one condition: none, then per (L, L', source, gain) the true / foreign / shuffle variants,
    and per (L', gain, seed, t_0 of the sources) the random one; sorted so that arms sharing a source layer are
    adjacent. Sourced arms carry ``late_from`` = t_late (the matched shuffle region)."""
    variants = [v for v in args.variants.split(",") if v]
    gains = [float(x) for x in args.gains.split(",") if x.strip()]
    srcs = [parse_source(s) for s in args.sources.split(",") if s.strip()]
    targets = sorted({c[1] for c in cells})
    arms = [{"name": "none", "variant": "none", "L": None, "Lp": targets[0] if targets else 0, "source": None,
             "gain": 0.0, "seed": 0, "d_model": d_model, "start": None, "late_from": None}]
    for L, Lp in cells:
        for src in srcs:
            for g in gains:
                for v in ("true", "foreign", "shuffle"):
                    if v in variants:
                        arms.append({"name": arm_name(v, src["name"], L, Lp, g), "variant": v, "L": L, "Lp": Lp,
                                     "source": src["name"], "gain": g, "seed": 0, "d_model": d_model,
                                     "start": steer_start(src, args.steer_from),
                                     "late_from": late_start(src, args.steer_from, args.shuffle_gap)})
    if "random" in variants:
        starts = sorted({random_start(s, args.steer_from) for s in srcs} or {None}, key=lambda x: x or -1)
        for Lp in targets:
            for g in gains:
                for rs in ints(args.random_seeds):
                    for st in starts:
                        arms.append({"name": arm_name("random", Lp=Lp, g=g, seed=rs, start=st), "variant": "random",
                                     "L": None, "Lp": Lp, "source": None, "gain": g, "seed": rs, "d_model": d_model,
                                     "start": st, "late_from": None})
    if args.arms:
        pats = [p for p in args.arms.split(",") if p]
        arms = [a for a in arms if any(fnmatch.fnmatchcase(a["name"], p) for p in pats)]
    order = {"none": 0, "true": 1, "foreign": 1, "shuffle": 1, "random": 2}
    arms.sort(key=lambda a: (order[a["variant"]], -1 if a["L"] is None else a["L"], a["source"] or "", a["Lp"],
                             a["gain"], a["variant"]))
    return arms


def conditions(args, hybrid: bool) -> tuple[list[str], list[str]]:
    conds = [c for c in args.conditions.split(",") if c]
    bad = [c for c in conds if c not in tx.CONDITIONS]
    if bad:
        raise SystemExit(f"unknown conditions {bad}; known {list(tx.CONDITIONS)}")
    skipped = []
    if not hybrid and "all" in conds:
        conds.remove("all")
        skipped.append("all: identical to attn on a model without linear-attention layers")
    return conds, skipped


def family_of(args, kinds) -> str:
    return "tiny" if args.tiny else ("hybrid" if ms.is_hybrid(kinds) else "attention")


def count_arms(args) -> dict:
    """Arm counts per condition from the config only (no weights)."""
    if args.tiny:
        kinds = ["linear", "attention"] * 2 if args.tiny_arch == "qwen35" else ["attention"] * 4
    else:
        import transformers
        cfg = transformers.AutoConfig.from_pretrained(args.model)
        cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        types = getattr(cfg, "layer_types", None) or ["full_attention"] * cfg.num_hidden_layers
        kinds = ["linear" if "linear" in t else "attention" for t in types]
    check_sources(args)
    cells = grid_cells(args, len(kinds), family_of(args, kinds))
    conds, _ = conditions(args, ms.is_hybrid(kinds))
    n = len(make_arms(args, cells, 1))
    return {"family": family_of(args, kinds), "n_blocks": len(kinds), "cells": len(cells), "conditions": conds,
            "arms_per_condition": n, "arms_total": n * len(conds)}


# ------------------------------------------------------------------------------------------- one group
class Acc:
    """Per-item sums of one arm (results JSON ``per_item``)."""

    def __init__(self) -> None:
        self.items: dict[str, list] = {}
        self.checks: dict = {"gain0_equal": None, "source_unchanged": None}
        self.time = {"arm_s": 0.0, "tokens": 0}

    def put(self, key: str, rows) -> None:
        self.items.setdefault(key, []).extend(np.asarray(rows).tolist())


def lens_depths(args, n_blocks: int) -> list[int]:
    if args.lens_depths and args.lens_depths.strip().lower() == "none":
        return []
    return tx.lens_depths(args, n_blocks)


def run_group(group: list[dict], cond: str, lm: pl.HookedLM, model, blocks, batches, args, writers: dict,
              accs: dict, base_time: dict) -> None:
    s, win = args.sink, (args.window if cond != "full" else 0)
    depths = lens_depths(args, len(blocks))
    n_blocks = len(blocks)
    W = args.window
    off_e, pos_e = ms.offset_edges(W), ms.log2_edges(args.seq_len)
    Ls = sorted({a["L"] for a in group if a["L"] is not None})
    for bi, (ids, scored, rows) in enumerate(batches):
        B, T = ids.shape
        Tt = T - s
        dev = lm.input_device
        sc_text = torch.cat([scored[:, 1:], torch.zeros_like(scored[:, :1])], 1)[:, s:]   # target → predicting pos
        tg, valid = ms.shifted_targets(ids)
        valid[:, :s] = False
        t0 = ms.timer()
        cap, ncap = StateCapture(blocks, Ls, s), ms.NormCapture(blocks)
        try:
            with lm.condition(win, False):
                refs = {"base": lm(ids)}
        finally:
            cap.remove()
            ncap.remove()
        H1, base_norm = cap.h, ncap.norms()[:, s:].cpu()
        if win:
            with lm.condition(0, False):
                refs["full"] = lm(ids)
        rm = {r: ms.token_metrics(z, tg, valid) for r, z in refs.items()}
        base_time["base_s"] += ms.timer() - t0
        cut = lambda x: x[:, s:].detach().float().cpu()                     # noqa: E731
        ref_tok = {f"out/loss_{r}": cut(m["loss"]) for r, m in rm.items()}
        ref_tok["out/ent_base"] = cut(rm["base"]["ent"])
        ids_text = ids[:, s:]
        tp = torch.arange(Tt, dtype=torch.float32).expand(B, -1)
        offset = tp % W
        cache: dict = {}
        for arm in group:
            acc, t0 = accs[arm["name"]], ms.timer()
            L, Lp, g = arm["L"], arm["Lp"], float(arm["gain"])
            if arm["variant"] in ("random", "none"):
                u, ok = variant_vectors(arm, torch.zeros(B, Tt, 1, device=dev), rows, args)
            else:
                key = (L, arm["source"])
                if key not in cache:
                    cache.clear()                                   # arms are sorted: one source at a time
                    cache[key] = source_vectors(H1[L], parse_source(arm["source"]))
                u, ok = variant_vectors(arm, H1[L], rows, args, cache[key])
            # hook order: the source capture BEFORE the injector (same cells read the pre-injection state at L),
            # the norm and lens captures after it (they see the injected output at L')
            fcap = StateCapture(blocks, [L], s) if L is not None else None
            inj = Injector(blocks[Lp], u, ok, g, args.steer_from, s)
            ncap = ms.NormCapture(blocks)
            lens = ms.LensCapture(blocks, depths)
            try:
                with lm.condition(win, False):
                    lg = lm(ids)
            finally:
                inj.remove()
                ncap.remove()
                lens.remove()
                if fcap is not None:
                    fcap.remove()
            m = ms.token_metrics(lg, tg, valid, refs)
            if arm["variant"] == "none":
                eq = bool(torch.equal(lg, refs["base"]))
                c = acc.checks["gain0_equal"]
                acc.checks["gain0_equal"] = eq if c is None else c and eq
            del lg
            ranks = lens.ranks(model, tg.to(dev), valid)
            lens.h.clear()
            rec = {k: v.cpu() for k, v in inj.rec.items()}
            tok = {f"out/{k}": cut(m[k]) for k in ("loss", "ent", "rank", "top1", "p_tgt", "p_max", "kl_base")}
            tok.update(ref_tok)
            tok.update({"inj/on": rec["on"], "inj/ratio": rec["ratio"], "inj/cos_uh": rec["cos_uh"],
                        "inj/hnorm": rec["hnorm"]})
            if win:
                tok["out/kl_full"] = cut(m["kl_full"])
            for d, r in ranks.items():
                tok[f"lens/rank_d{d}"] = cut(r)
            tok["lens/recent"] = lens_recent(model, u, inj.rec["hnorm"], inj.on, ids_text, args.lens_k,
                                             args.lens_recent).cpu()
            nr = ncap.norms()[:, s:].cpu() / base_norm.clamp_min(1e-12)
            tok["out/norm_ratio"] = nr[..., Lp:]
            if fcap is not None:
                H2 = fcap.h[L]
                rel = (H2 - H1[L]).norm(dim=-1) / H1[L].norm(dim=-1).clamp_min(1e-12)
                tok["fb/relchange"] = rel.cpu()
                if Lp >= L:                                     # skip and same cells: exact, checked bitwise
                    z = bool((H2 == H1[L]).all())
                    c = acc.checks["source_unchanged"]
                    acc.checks["source_unchanged"] = z if c is None else c and z
                else:
                    u2, ok2 = variant_vectors(arm, H2, rows, args)
                    both = (ok & ok2 & inj.on.to(ok.device))
                    tok["fb/ucos"] = torch.where(both, (u2 * u).sum(-1), torch.full_like(rel, float("nan"))).cpu()
            tok["pos/token"] = ids_text.detach().cpu().long()
            tok["pos/text"], tok["pos/offset"], tok["pos/scored"] = tp, offset, sc_text.float()
            tok["pos/block"] = torch.div(tp, W, rounding_mode="floor")
            acc.time["arm_s"] += ms.timer() - t0
            acc.time["tokens"] += int(B * Tt)
            accumulate(acc, tok, sc_text.bool(), rows, win, (off_e, pos_e), W, n_blocks, nr, args, depths,
                       arm.get("late_from"))
            for b in range(B):
                item = {"row": rows[b], "start": rows[b] * args.seq_len, "cond": cond, "arm": arm["name"],
                        "batch": bi}
                writers[arm["name"]].add(item, ms.row_dict(tok, b), False)
        del refs, H1


def accumulate(acc: Acc, tok: dict, scored: torch.Tensor, rows, win: int, edges, W: int, n_blocks: int,
               nr: torch.Tensor, args, depths: list[int], late_from: int | None) -> None:
    sc = scored & torch.isfinite(tok["out/loss"])
    nll, n = ms.masked_sum(tok["out/loss"], sc)
    acc.put("row", rows)
    acc.put("start", [r * args.seq_len for r in rows])
    acc.put("nll_sum", nll)
    acc.put("n_tok", n)
    refs = ("base", "full") if win else ("base",)
    for r in refs:
        acc.put(f"loss_{r}_sum", ms.masked_sum(tok[f"out/loss_{r}"], sc)[0])
        acc.put(f"kl_{r}_sum", ms.masked_sum(tok[f"out/kl_{r}"], sc)[0])
    for k in ("top1", "ent", "ent_base"):
        acc.put(f"{k}_sum", ms.masked_sum(tok[f"out/{k}"], sc)[0])
    dent = tok["out/ent"] - tok["out/ent_base"]
    acc.put("dent_sum", ms.masked_sum(dent, sc)[0])
    acc.put("absdent_sum", ms.masked_sum(dent.abs(), sc)[0])
    for d in depths:
        rk = tok[f"lens/rank_d{d}"]
        acc.put(f"lens_logrank_d{d}_sum", ms.masked_sum(torch.log1p(rk), sc)[0])
        acc.put(f"lens_top1_d{d}_sum", ms.masked_sum((rk == 0).float(), sc)[0])
    on = sc & tok["inj/on"].bool()
    acc.put("on_n", ms.masked_sum(tok["inj/on"], sc)[0])
    for k, key in (("ratio", "inj/ratio"), ("cos_uh", "inj/cos_uh"), ("lens_recent", "lens/recent")):
        sm, ct = ms.masked_sum(tok[key], on)
        acc.put(f"{k}_sum", sm)
        acc.put(f"{k}_n", ct)
    for k in ("fb/relchange", "fb/ucos"):
        if k in tok:
            sm, ct = ms.masked_sum(tok[k], on)
            acc.put(f"{k.split('/')[1]}_sum", sm)
            acc.put(f"{k.split('/')[1]}_n", ct)
    w = sc.float()[..., None] * torch.isfinite(nr).float()
    acc.put("normratio_sum", (torch.nan_to_num(nr) * w).sum(1).double().numpy())
    acc.put("normratio_n", w.sum(1).double().numpy())
    off_e, pos_e = edges
    tp = tok["pos/text"]
    for bname, bidx, nb in (("off", ms.bucketize(tok["pos/offset"], off_e), len(off_e) - 1),
                            ("offw", tok["pos/offset"].long(), W),
                            ("pos", ms.bucketize(tp, pos_e), len(pos_e) - 1)):
        bidx = torch.where(sc, bidx, torch.full_like(bidx, -1))
        for k in ("loss", "loss_base", "kl_base") + (("loss_full",) if win else ()):
            sm, ct = ms.bucket_sums(tok[f"out/{k}"], bidx, nb)
            acc.put(f"{bname}_{k}_sum", sm)
            if k == "loss":
                acc.put(f"{bname}_n", ct)
    if late_from is not None:                           # t ≥ t_late: the matched region of the shuffle contrast
        late = sc & (tp >= late_from)
        bidx = torch.where(late, ms.bucketize(tok["pos/offset"], off_e), torch.full_like(tp, -1, dtype=torch.long))
        for k in ("loss", "loss_base"):
            sm, ct = ms.bucket_sums(tok[f"out/{k}"], bidx, len(off_e) - 1)
            acc.put(f"late_off_{k}_sum", sm)
            if k == "loss":
                acc.put("late_off_n", ct)
        acc.put("late_on_n", ms.masked_sum(tok["inj/on"], late)[0])


def finish_arm(arm: dict, cond: str, acc: Acc, args, win: int, prov: dict | None = None) -> dict:
    it = {k: np.asarray(v, dtype=np.float64) for k, v in acc.items.items()}
    n = it["n_tok"]
    tot = lambda k: float(it[k].sum() / max(n.sum(), 1))                    # noqa: E731
    rat = lambda k: (float(it[f"{k}_sum"].sum() / it[f"{k}_n"].sum())        # noqa: E731
                     if f"{k}_n" in it and it[f"{k}_n"].sum() > 0 else None)
    L = it["nll_sum"] / np.maximum(n, 1)
    agg = {"loss": tot("nll_sum"), "n_tok": int(n.sum()), "n_seq": int(len(n)), "loss_base": tot("loss_base_sum"),
           "kl_base": tot("kl_base_sum"), "steered_frac": tot("on_n"), "inj_ratio": rat("ratio"),
           "cos_uh": rat("cos_uh"), "lens_recent": rat("lens_recent"), "fb_relchange": rat("relchange"),
           "fb_ucos": rat("ucos")}
    for k in ("top1", "ent", "ent_base", "dent", "absdent"):
        agg[k] = tot(f"{k}_sum")
    for k in [k for k in it if k.startswith("lens_") and k.endswith("_sum") and "_d" in k]:
        agg[k[:-4]] = tot(k)
    if "late_off_n" in it:
        agg["late_from"], agg["late_steered_frac"] = arm.get("late_from"), float(
            it["late_on_n"].sum() / max(it["late_off_n"].sum(), 1))
    agg["d_loss_vs_none"] = ms.boot(L, it["loss_base_sum"] / np.maximum(n, 1), n=args.boot, seed=args.seed)
    if win:
        agg["loss_full"], agg["kl_full"] = tot("loss_full_sum"), tot("kl_full_sum")
        agg["d_loss_vs_full"] = ms.boot(L, it["loss_full_sum"] / np.maximum(n, 1), n=args.boot, seed=args.seed)
        agg["closure"] = tx.closure_boot(it["nll_sum"], it["loss_base_sum"], it["loss_full_sum"], n, args.boot,
                                         args.seed)
    for b in ("off", "offw", "pos"):
        nb = it[f"{b}_n"]
        agg[f"loss_by_{b}"] = (it[f"{b}_loss_sum"].sum(0) / np.maximum(nb.sum(0), 1)).tolist()
        agg[f"dloss_by_{b}"] = ((it[f"{b}_loss_sum"] - it[f"{b}_loss_base_sum"]).sum(0)
                                / np.maximum(nb.sum(0), 1)).tolist()
        agg[f"kl_by_{b}"] = (it[f"{b}_kl_base_sum"].sum(0) / np.maximum(nb.sum(0), 1)).tolist()
        agg[f"n_by_{b}"] = nb.sum(0).tolist()
    nr = it["normratio_sum"].sum(0) / np.maximum(it["normratio_n"].sum(0), 1)
    agg["norm_ratio_by_block"], agg["norm_ratio_final"] = nr.tolist(), float(nr[-1])
    tm = dict(acc.time)
    tm["tok_per_s_arm"] = tm["tokens"] / max(tm["arm_s"], 1e-9)
    ck = cell_kind(arm["L"], arm["Lp"]) if arm["L"] is not None else None
    return {"condition": cond, "window": win, "arm": arm["name"],
            **{k: arm.get(k) for k in ("variant", "L", "Lp", "source", "gain", "seed", "start", "late_from")},
            "cell_kind": ck, "exact": ck != "feedback", "approximation": APPROX.get(ck),
            "complete": True, "agg": agg, "checks": acc.checks, "time": tm, "per_item": it,
            "edges": {"offset": ms.offset_edges(args.window), "position": ms.log2_edges(args.seq_len)},
            **(prov or {})}


# ------------------------------------------------------------------------------------------- contrasts
def row_means(a: dict, key: str = "loss", region: str = "all", bucket: str = "all") -> np.ndarray:
    """Per-row mean of ``key`` ('loss' or 'loss_base') over the scored tokens of ``region`` ('all', or 'late':
    t ≥ t_late) and ``bucket`` ('all', or 'early': offsets 0 .. W/8 − 1); NaN for a row without such tokens."""
    pi = a["per_item"]
    arr = lambda k: np.asarray(pi[k], dtype=np.float64)                     # noqa: E731
    if region == "all" and bucket == "all":
        num, den = arr("nll_sum" if key == "loss" else f"{key}_sum"), arr("n_tok")
    else:
        pre = "off" if region == "all" else "late_off"
        if f"{pre}_n" not in pi:
            return np.full(len(pi["row"]), np.nan)
        num, den = arr(f"{pre}_{key}_sum"), arr(f"{pre}_n")
        num, den = (num[:, 0], den[:, 0]) if bucket == "early" else (num.sum(1), den.sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def per_row_loss(a: dict) -> np.ndarray:
    return row_means(a)


def p_two(b: dict | None) -> float | None:
    """Two-sided bootstrap p value of a ``paired_boot`` record (None without one)."""
    if not b or b.get("p_le_0") is None:
        return None
    return float(min(1.0, 2.0 * min(b["p_le_0"], b["p_ge_0"])))


def bh(p: list[float]) -> list[float]:
    """Benjamini–Hochberg q values (same order as ``p``)."""
    m = len(p)
    if not m:
        return []
    order = np.argsort(p)
    q = np.empty(m)
    run = 1.0
    for i in range(m - 1, -1, -1):
        run = min(run, p[order[i]] * m / (i + 1))
        q[order[i]] = run
    return q.tolist()


def contrasts(data: dict, cond: str, n_boot: int, seed: int) -> dict:
    """For every complete true arm of ``cond``: Δloss vs none and the paired true − control differences over rows
    (module docstring), whole rows and early offsets, with the steered share of each arm."""
    arms = {a["arm"]: a for a in data["arms"].values() if a.get("condition") == cond and a.get("complete")}
    hdr = data.get("args", {})
    steer_from, gap = int(hdr.get("steer_from", 0)), int(hdr.get("shuffle_gap", 0))
    out = {}
    bt = lambda x, y: ms.clean(ms.boot(x, y, n=n_boot, seed=seed))         # noqa: E731
    for name, a in arms.items():
        if a.get("variant") != "true":
            continue
        src = parse_source(a["source"])
        sf = a["agg"]["steered_frac"]
        rec = {"L": a["L"], "Lp": a["Lp"], "source": a["source"], "gain": a["gain"], "cell_kind": a["cell_kind"],
               "kl_base": a["agg"]["kl_base"], "lens_recent": a["agg"]["lens_recent"], "steered_frac": sf,
               "t0": steer_start(src, steer_from), "late_from": late_start(src, steer_from, gap)}
        for bk, sfx in (("all", ""), ("early", "_early")):
            rec[f"vs_none{sfx}"] = bt(row_means(a, "loss", "all", bk), row_means(a, "loss_base", "all", bk))
        ctrl = {"foreign": arm_name("foreign", a["source"], a["L"], a["Lp"], a["gain"]),
                "shuffle": arm_name("shuffle", a["source"], a["L"], a["Lp"], a["gain"]),
                "random": arm_name("random", Lp=a["Lp"], g=a["gain"], seed=0,
                                   start=random_start(src, steer_from))}
        for c, cn in ctrl.items():
            b = arms.get(cn)
            if b is None or b["per_item"]["row"] != a["per_item"]["row"]:
                continue
            for bk, sfx in (("all", ""), ("early", "_early")):
                rec[f"minus_{c}{sfx}"] = bt(row_means(a, "loss", "all", bk), row_means(b, "loss", "all", bk))
                if c == "shuffle":
                    rec[f"minus_shuffle_matched{sfx}"] = bt(row_means(a, "loss", "late", bk),
                                                            row_means(b, "loss", "late", bk))
            rec[f"lens_recent_{c}"] = b["agg"]["lens_recent"]
            rec[f"steered_frac_{c}"] = b["agg"]["steered_frac"]
            rec[f"dose_{c}"] = (b["agg"]["steered_frac"] / sf) if sf else None
            if c == "shuffle":
                rec["dose_shuffle_matched"] = ((b["agg"].get("late_steered_frac") or 0.0)
                                               / a["agg"]["late_steered_frac"]
                                               if a["agg"].get("late_steered_frac") else None)
        out[name] = rec
    return out


def excludes0(b: dict | None) -> bool:
    return bool(b and b.get("lo") is not None and (b["lo"] > 0 or b["hi"] < 0))


SELECT_KEYS = ("minus_foreign", "minus_foreign_early", "vs_none", "vs_none_early")


def select_cells(data: dict, top: int, q: float = 0.10) -> list[tuple[int, int]]:
    """Cells of a screen JSON that matter: any true − foreign interval (two-sided; whole rows or early offsets) or
    helpful Δloss vs none (one-sided, p = share of bootstrap means ≥ 0; whole rows or early offsets) surviving
    Benjamini–Hochberg at ``q`` over every such test of the screen, plus the ``top`` cells with the most negative
    true − foreign t statistic (diff / se, whole rows or early offsets; over conditions and sources)."""
    return selection(data, top, q)["cells"]


def selection(data: dict, top: int, q: float = 0.10) -> dict:
    """``select_cells`` with its provenance: the criterion, the number of tests, the BH-selected and top cells."""
    tests, score = [], {}
    for cond, recs in data.get("contrasts", {}).items():
        for r in recs.values():
            c = (int(r["L"]), int(r["Lp"]))
            for k in SELECT_KEYS:
                b = r.get(k)
                if not b or b.get("diff") is None:
                    continue
                p = p_two(b) if k.startswith("minus") else b.get("p_ge_0")
                if p is not None:
                    tests.append((c, p))
                if k.startswith("minus") and b.get("se"):
                    score[c] = min(score.get(c, math.inf), b["diff"] / b["se"])
    qs = bh([p for _, p in tests])
    sig = sorted({c for (c, _), qv in zip(tests, qs) if qv <= q})
    best = [c for c, _ in sorted(score.items(), key=lambda kv: kv[1])[:max(0, top)]]
    return {"cells": sorted(set(sig) | set(best)), "bh_cells": sig, "top_cells": sorted(best), "n_tests": len(tests),
            "criterion": f"BH q <= {q:g} over {len(tests)} tests ({', '.join(SELECT_KEYS)}; minus_*: two-sided, "
                         f"vs_none*: one-sided helpful), plus the {top} cells with the most negative true - foreign "
                         "t statistic"}


# -------------------------------------------------------------------------------------------- analyse
def analyse(path: str, control: str = "foreign", at_gain: float | None = None, n_boot: int = 2000,
            seed: int = 0, show: bool = True, bucket: str = "all") -> dict:
    """Per condition and source, the (L, L') matrix of true − control per-row loss differences (``control``
    'shuffle' uses the matched region t ≥ t_late; ``bucket`` 'early' the offsets < W/8), with 95% intervals.
    Matrices are stored for every gain; the printed one is at ``at_gain``, or else at the gain whose true arms have
    the lowest mean Δloss vs none over cells — a choice made post hoc on the same rows, stated in the output.
    '*' marks an interval excluding 0 (uncorrected); '**' also a Benjamini–Hochberg q < 0.05 over every interval
    of the report (all conditions, sources, gains). Contrasts are recomputed from the per-item sums."""
    d = json.loads(Path(path).expanduser().read_text())
    key = ("vs_none" if control == "none" else "minus_shuffle_matched" if control == "shuffle"
           else f"minus_{control}") + ("_early" if bucket == "early" else "")
    rule = (f"gain fixed by --at-gain {at_gain:g}" if at_gain is not None else
            "gain chosen POST HOC on these rows (lowest mean dloss vs none over cells); intervals not adjusted "
            "for that choice")
    out: dict = {"file": str(path), "control": control, "bucket": bucket, "key": key, "gain_rule": rule,
                 "conditions": {}}
    conds = sorted({a["condition"] for a in d["arms"].values()}, key=lambda c: list(tx.CONDITIONS).index(c))
    by: dict = {}                                       # (cond, src) -> records
    for cond in conds:
        for r in contrasts(d, cond, n_boot, seed).values():
            by.setdefault((cond, r["source"]), []).append(r)
    flat = [(ck, r) for ck, recs in by.items() for r in recs if r.get(key) and r[key].get("diff") is not None]
    qs = bh([p_two(r[key]) for _, r in flat])
    qmap = {(ck, r["L"], r["Lp"], r["gain"]): qv for (ck, r), qv in zip(flat, qs)}
    n_sig = sum(excludes0(r[key]) for _, r in flat)
    n_bh = sum(qv < 0.05 for qv in qs)
    lines = [f"[laggrid] {path}  true - {control} ({key}), nats per scored token; 95% paired bootstrap",
             f"  * interval excludes 0 (uncorrected): {n_sig}/{len(flat)} over the report; expected by chance at 5%: "
             f"{0.05 * len(flat):.1f}.  ** BH q < 0.05 over those {len(flat)}: {n_bh}",
             f"  {rule}"]
    out.update(n_tests=len(flat), n_uncorrected=n_sig, expected_false_positives=0.05 * len(flat), n_bh=n_bh)
    for (cond, src), recs in sorted(by.items(), key=lambda kv: (conds.index(kv[0][0]), kv[0][1])):
        oc = out["conditions"].setdefault(cond, {})
        gains = sorted({r["gain"] for r in recs})
        if at_gain is not None:
            g = min(gains, key=lambda x: abs(x - at_gain))
        else:
            g = min(gains, key=lambda x: np.mean([r["vs_none"]["diff"] for r in recs if r["gain"] == x
                                                  and r["vs_none"]["diff"] is not None] or [math.inf]))
        by_gain = {}
        for gg in gains:
            cells = {(r["L"], r["Lp"]): r.get(key) for r in recs if r["gain"] == gg}
            Ls, Lps = sorted({c[0] for c in cells}), sorted({c[1] for c in cells})
            mat = {f"{L}-{Lp}": cells.get((L, Lp)) for L in Ls for Lp in Lps}
            qm = {f"{L}-{Lp}": qmap.get(((cond, src), L, Lp, gg)) for L in Ls for Lp in Lps}
            by_gain[f"{gg:g}"] = {"matrix": mat, "q": qm, "rows_L": Ls, "cols_Lp": Lps,
                                  "significant": [k for k, v in mat.items() if excludes0(v)],
                                  "significant_bh": [k for k, v in qm.items() if v is not None and v < 0.05]}
        sel = by_gain[f"{g:g}"]
        oc[src] = {"gain": g, "gain_rule": rule, **sel, "by_gain": by_gain}
        mat, Ls, Lps = sel["matrix"], sel["rows_L"], sel["cols_Lp"]
        lines.append(f"\n{cond}  {src}  gain {g:g}  (rows L, columns L'; significant: {len(sel['significant'])}/"
                     f"{len(mat)} uncorrected, {len(sel['significant_bh'])} BH)")
        lines.append("  L\\L' " + "".join(f"{Lp:>27d}" for Lp in Lps))
        for L in Ls:
            row = []
            for Lp in Lps:
                v = mat.get(f"{L}-{Lp}")
                if not v or v.get("diff") is None:
                    row.append(f"{'-':>27s}")
                    continue
                qv = sel["q"].get(f"{L}-{Lp}")
                star = ("**" if qv is not None and qv < 0.05 else "* ") if excludes0(v) else "  "
                row.append(f"{v['diff']:+.4f}{star}[{v['lo']:+.4f},{v['hi']:+.4f}]".rjust(27))
            lines.append(f"  {L:>4d} " + "".join(row))
    if show:
        print("\n".join(lines), flush=True)
    out["text"] = "\n".join(lines)
    return out


# ---------------------------------------------------------------------------------------------- main
def make_rows(args, tok, need_foreign: bool) -> tuple[list, dict]:
    """``text.make_batches`` restricted to rows [``--row-offset``, ``--row-offset`` + ``--n-seqs``) of the file, in
    the same interleaved layout (batch k = rows k, k + n_b, ... of the selection). Row ids stay absolute, so the
    random and shuffle seeds and the trace items keep their meaning across offsets."""
    one, info = tx.make_batches(SimpleNamespace(
        text_file=args.text_file, max_chars=args.max_chars, seq_len=args.seq_len, sink=args.sink, window=args.window,
        n_seqs=args.row_offset + args.n_seqs, batch=1, directions="", impulse_directions=""), tok)
    rows = [b for b in one if b[2][0] >= args.row_offset]
    n = len(rows)
    if need_foreign:
        n -= n % args.batch
    if n < 1 or (need_foreign and args.batch < 2):
        raise SystemExit(f"rows from {args.row_offset}: {len(rows)} available; need at least one full batch, and "
                         "--batch >= 2 for the foreign variant")
    rows = rows[:n]
    nb = -(-n // args.batch)
    batches = []
    for k in range(nb):
        idx = list(range(k, n, nb))
        batches.append((torch.cat([rows[i][0] for i in idx]), torch.cat([rows[i][1] for i in idx]),
                        [rows[i][2][0] for i in idx]))
    info.update(n_rows=n, n_rows_requested=args.n_seqs, row_offset=args.row_offset,
                row_starts=[r[2][0] * args.seq_len for r in rows], batches=[b[2] for b in batches],
                layout="interleaved: batch k = rows k, k + n_b, ... of the selection (absolute row ids)")
    return batches, info


def selection_provenance(args, task: dict) -> dict | None:
    """With ``--cells-from``: the screen file, its rows, the selection and its criterion; SystemExit when this run's
    rows overlap the screen's on the same text (the follow-up must not re-measure the rows it was selected on)."""
    if not args.cells_from:
        return None
    path = Path(args.cells_from).expanduser()
    scr = json.loads(path.read_text())
    st = scr.get("task", {})
    same_text = Path(str(st.get("file", ""))).expanduser().resolve() == Path(task["file"]).expanduser().resolve()
    overlap = 0
    if same_text:
        a_len, b_len = int(st.get("seq_len", args.seq_len)), int(args.seq_len)
        ivs = sorted((int(x), int(x) + a_len) for x in st.get("row_starts", []))
        for x in task["row_starts"]:
            overlap += any(lo < x + b_len and x < hi for lo, hi in ivs)
    if overlap:
        last = max(int(x) for x in st["row_starts"]) + int(st.get("seq_len", args.seq_len))
        raise SystemExit(f"--cells-from {path}: {overlap} of this run's rows overlap the rows the cells were selected "
                         f"on; pass --row-offset {-(-last // args.seq_len)} (or another --text-file)")
    sel = selection(scr, args.cells_top, args.select_q)
    return {"source": str(path), "source_git": scr.get("git"), "source_rows": {
        "row_offset": st.get("row_offset", 0), "n_rows": st.get("n_rows"), "file": st.get("file")},
        "this_rows": {"row_offset": args.row_offset, "n_rows": task["n_rows"]}, "rows_disjoint": True,
        "same_text": same_text, **sel}


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.analyse:
        return analyse(args.analyse, args.control, args.at_gain, args.boot, args.seed, bucket=args.bucket)
    if args.count:
        c = count_arms(args)
        print(f"[laggrid] {json.dumps(c)}", flush=True)
        return c
    check_sources(args)
    torch.manual_seed(args.seed)
    if args.name is None and args.screen:
        args.name = ms.default_name(args, "laggrid_screen")
    out, trace_root = ms.resolve_paths(args, "laggrid", args.stage)
    model, tok, blocks, kinds = ms.load_model(args)
    hybrid = ms.is_hybrid(kinds)
    family = family_of(args, kinds)
    cells = grid_cells(args, len(blocks), family)
    if not cells:
        raise SystemExit("no (L, L') cells selected")
    conds, skipped = conditions(args, hybrid)
    need_foreign = "foreign" in args.variants
    batches, task = make_rows(args, tok, need_foreign)
    prov_sel = selection_provenance(args, task)
    d_model = int(pl.text_config(model).hidden_size)
    facts = ms.model_facts(model, tok, blocks, kinds, args.model if not args.tiny else f"tiny-{args.tiny_arch}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vers, git = ms.versions(), common.git_state()
    kernel = {k: vers.get(k) for k in ("torch", "transformers", "fla")}
    header = {"runner": "laggrid", "stage": args.stage, "name": args.name, "args": vars(args),
              "git": git, "versions": vers, "model": facts, "task": task, "family": family,
              "cells": [list(c) for c in cells], "cell_kinds": {f"{L}-{Lp}": cell_kind(L, Lp) for L, Lp in cells},
              "approximation": APPROX, "trainable_parameters": int(n_params), "conditions_skipped": skipped,
              "trace_root": str(trace_root), "lens_depths": lens_depths(args, len(blocks)),
              "selection_provenance": prov_sel}
    compat = {k: getattr(args, k) for k in ("model", "tiny", "tiny_arch", "dtype", "attn_impl", "seq_len", "sink",
                                            "window", "steer_from", "seed", "text_file", "max_chars", "batch",
                                            "shuffle_gap", "lens_k", "lens_recent", "lens_depths", "row_offset")}
    compat.update(n_rows=task["n_rows"], **{f"version_{k}": v for k, v in kernel.items()})
    res = ms.ResultFile(out, header, compat, force=args.force)
    arm_prov = {"git": git.get("commit"), "git_dirty": git.get("dirty"), "versions": kernel,
                "attn_impl": args.attn_impl}
    print(f"[laggrid] {facts['name']} ({family}) cells={len(cells)} conds={conds} rows={task['n_rows']} out={out}",
          flush=True)
    for cond in conds:
        lm = pl.HookedLM(model, side=None, layer=0, window=args.window, mode="block", window_layers=tx.CONDITIONS[cond],
                         sink=args.sink, pad_id=tok.bos_token_id)
        lm.side_on = False
        try:
            res.data.setdefault("conditions", {})[cond] = {
                "window": 0 if cond == "full" else args.window, "window_layers": tx.CONDITIONS[cond],
                "check_window": pl.check_window(lm) if cond != "full" else "full attention",
                "windowed_blocks": lm.windowed}
            res.save()
            todo = []
            for arm in make_arms(args, cells, d_model):
                if res.done(f"{cond}/{arm['name']}") and not args.force:
                    continue
                todo.append(arm)
            print(f"[laggrid] {cond}: {len(todo)} arms to run", flush=True)
            win = args.window if cond != "full" else 0
            for gi in range(0, len(todo), max(1, args.group)):
                group = todo[gi: gi + max(1, args.group)]
                writers, accs = {}, {}
                for arm in group:
                    meta = {"runner": "laggrid", "cond": cond, "arm": arm, "model": facts["name"], "args": vars(args)}
                    writers[arm["name"]] = ms.ArmTraces(trace_root, cond, arm["name"], meta, args.trace_tokens)
                    accs[arm["name"]] = Acc()
                bt = {"base_s": 0.0}
                run_group(group, cond, lm, model, blocks, batches, args, writers, accs, bt)
                for arm in group:
                    rec = finish_arm(arm, cond, accs[arm["name"]], args, win, arm_prov)
                    rec["time"]["group_base_s"] = bt["base_s"]
                    rec["traces"] = writers[arm["name"]].close()
                    res.put(f"{cond}/{arm['name']}", rec)
                    a = rec["agg"]
                    dl = a["d_loss_vs_none"]
                    print(f"[laggrid] {cond}/{arm['name']}: loss {a['loss']:.4f} d {dl['diff']:+.4f} "
                          f"[{dl['lo']:+.4f},{dl['hi']:+.4f}] kl {a['kl_base']:.4f}", flush=True)
            res.data.setdefault("contrasts", {})[cond] = contrasts(res.data, cond, args.boot, args.seed)
            res.save()
        finally:
            for h in lm.handles:
                h.remove()
    return res.data


if __name__ == "__main__":
    main()
