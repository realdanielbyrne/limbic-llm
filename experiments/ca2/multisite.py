"""Multi-site CA2 injection: the same additive activation in smaller doses across several tap points.

Hypothesis (Daniel, 10 Nov 2026). The repetition a fixed-gain gist causes in free generation (Stage 0: collapse
from g ≈ 0.3–0.5 at every single site tried, more than a random direction of equal norm) comes from
over-stimulation at one point; the same additive activation split over several sites should give the effect
with less collapse. Nothing is trained: zero parameters, every controller is a ``CA2Controller`` at a fixed gain.

Mechanism. K sites (block outputs) L_1 < ... < L_K. Site i has its OWN controller, reading its own block's
pre-injection output h⁽ⁱ⁾ and injecting at that block's output (inject_before = L_i + 1):

    h⁽ⁱ⁾_t ← h⁽ⁱ⁾_t + g_i ‖h⁽ⁱ⁾_t‖ d̂⁽ⁱ⁾_t,

d̂⁽ⁱ⁾ being site i's own direction (``CA2Controller`` directions on site i's own ``GistBank``):
    gist           normalise(A⁽ⁱ⁾_fast − A⁽ⁱ⁾_slow);   anti_gist −gist;   mean normalise(A⁽ⁱ⁾_slow);
    random         an independent seeded unit vector per site, seed r + 1000·(i − 1) (site 1 uses seed r, the
                   single-site vector; r = ``--random-seed``);
    random_shared  one vector (seed r) at every site (for K = 1 identical to ``random``; not run there).
Dose rules for a total gain G:
    lin   g_i = G / K       equal summed dose, Σ g_i = G
    sqrt  g_i = G / √K      equal summed norm, √(Σ g_i²) = G, if the injections were orthogonal
    full  g_i = G           no splitting (the upper reference)
For K = 1 the rules coincide: dose label ``one``, run once.

Feed-forward dependence. Within one pass the sites run in depth order: site i's injection is part of the stream
every higher site j > i reads (its bank, its ‖h⁽ʲ⁾‖ and therefore its injection size), but nothing flows down
within the pass and there is no loop within a pass; across tokens everything couples through attention and the
GDN state. With several sites the gist of an upper site therefore partly contains the lower injections (a lower
site pushes the stream along its gist, the upper running averages absorb that push). Nominal G is not the dose,
so every arm records the realised effect:
    inj_ratio_by_block        ‖Δh_l‖/‖h_l‖ = g_l‖d̂_l‖ at each block l (0 off the sites; h_l the stream there,
                              lower injections included), mean over steered scored tokens;
    inj_ratio_base_by_block   ‖Δh_l‖/‖h_l^base‖, the same injection relative to the unsteered stream;
    norm_ratio_by_block       ‖h_l‖/‖h_l^base‖ at every block output (after any injection there);
    cos_consec                cos(d̂⁽ⁱ⁾_t, d̂⁽ⁱ⁺¹⁾_t) between consecutive sites' injected directions, compared as
                              raw vectors of the residual space (no map between layers);
    kl_base                   KL(arm ‖ unsteered) per token,
so configurations are compared at MATCHED EFFECT (KL), not only at matched nominal gain.

Signals (plan, "Measurement policy": every registered signal at every site). Each controller carries the
``agree`` and ``constant`` sources and ``measure.GistDriftSource``; with ``--signals all`` also the attention
sources (attn_window, attn_agree) on the deepest full-attention block ≤ its site and, on a hybrid, the six GDN
sources pooled over the GDN layers ≤ its site (as ``measure.register_signals`` for one site). The taps are
shared: one ``gdn.GDNTaps`` and one ``attention.AttnTap`` per distinct attention block, attached once (not K
times: K copies of the GDN hooks would hold K × 18 (B, T, 6144) tensors) and owned by the lowest site's
controller, which resets them, opens their calls and traces them once (``tap/gdn/*``, ``tap/attn<l>/*``); the
other controllers read them through a read-only view. The GDN per-token mismatch loop runs on the first
``--gdn-mismatch-n`` rows of every arm, once per pass. ``--signals lean`` keeps only agree, constant and gist
drift (faster generation with many sites). Per-token keys: ``site<L>/<controller key>`` per site,
``bk/<key>`` (ids, col, mask, idx, step, steer; identical at every site), ``tap/*``, ``out/*``, ``multi/*``.

Modes.
  text  teacher-forced, the conditions of ``text.py`` (full attention; attention windowed to W with GDN intact;
        every token mixer windowed, GDN reset), the rows of ``text.make_batches`` (same rows in every arm).
        Each site is hosted as ``text.py`` hosts its one site (``common.SteerSide``: sink stripped, controller
        reset at every call, steering from text position ``--steer-from``), the block-output hook replicating
        ``HookedLM._tap``; ``pl.HookedLM`` supplies only the window. Per arm: loss, Δloss against the unsteered
        pass with a paired bootstrap over rows, KL to it (and to full context in windowed conditions, with the
        gap closure of ``text.closure_boot``), loss / Δloss / KL by offset within the window and by position,
        entropy, the realised-effect measures above, per-site gain distribution and pooled signal means.
  gen   free generation with ``ceiling.py``'s prompt construction (``ceiling.gen_prompts``), sampling (T =
        ``--gen-temp``, the same seeded noise per batch in every arm: generator seed s·1000003 + batch) and
        metrics (collapse = ``judge.repetition_onset`` is not None, onset, distinct-4 = 1 − ``judge.degeneration``,
        entropy, NLL of the own sample) plus the self-loop rate (``reason.self_loop_rate`` on the decoded
        continuation; the text itself never leaves the process). All arms run in one process with one
        ``--gen-batch``; every arm, the unsteered one included, runs on each of ``--gen-seeds`` sampling seeds
        (args.seed, args.seed + 1, ...), so the baseline's sampling noise is measured and steered arms are paired
        with it on (prompt, seed). Sites are hosted by ``CA2Attachment`` (KV cache, steer_from = last prompt
        token). With ``--gen-tf`` (default) each steered continuation is teacher-forced twice, steered (fresh
        controllers, steer_from = the last prompt column, so the same schedule) and unsteered, giving per-token
        KL(steered ‖ unsteered) along it (T = 1 distributions, as ``reason.py``), the paired norm ratio by block,
        and ``tf_lp_dev``, the largest |log p| difference between the generation and its teacher-forced
        replay on the sampled tokens (a check that the replay is the generation).

Arms: the cross product site set × dose rule × direction × total gain, plus ``none`` (every site of every set
at g = 0 with every signal attached; byte-identical to the base, checked in text mode). Names:
``L<set>:<dose>:<direction>@<G>``, e.g. ``L12:one:gist@0.3``, ``L7,12,19:sqrt:random_shared@0.5``,
``L2-21:lin:gist@0.75`` (the set label is the ``--site-sets`` spec). ``--arms`` takes ';'-separated fnmatch
patterns (names contain commas).

Results JSON (``measure.ResultFile``: atomic rewrite, resume per arm, refuses to resume under other settings):
header (runner, args, git, versions, model facts, task, gen prompt facts, site sets, compat) and ``arms``:
    text/<cond>/<arm>   mode, condition, window, arm, meta (set, sites, K, dose, direction, G, g_site, seeds),
                        agg, per_site, checks, time, per_item (per row), edges, traces (pointer and hash)
    gen/seed<s>/<arm>   mode, seed, arm, meta, agg, per_site, checks, time, per_item (per prompt), traces
and ``gen_summary`` (per arm across seeds: collapse rate per seed, mean and SD, paired Δ against none with
prompts as bootstrap clusters). No generated text is stored in the results JSON; per-token arrays (sampled ids
included) go only to the trace directory outside the repository.

Analysis (``--analyse <json>``; reads the results JSON only). For each site set and dose rule (and direction),
the collapse rate and the text Δloss as functions of KL, the single-site curve as the reference; curves are
interpolated linearly in KL between neighbouring gains (G = 0 is the none arm, KL = 0), per item (the same
weights for every item, so paired items stay paired), at common KL values (``--kl-grid``, default 5 geometric
points up to the smaller of the two curves' largest KL; both curves pass through the none point, so the range
starts at 0, not at either curve's first gain). The KL axis (``--kl-from``): ``text`` = the arm's mean KL in
text condition ``--kl-cond`` (natural text, same tokens in every arm: the preferred axis, so run text mode, at
least ``--conditions full``, next to gen), ``gen`` = its mean KL along its own continuations, ``gen64`` = the
same over its first 64 sampled tokens, ``auto`` = text when every arm involved has it, else gen64. Gen KL is
measured on each arm's own tokens: inside a repetition loop steered and unsteered models agree on the repeated
token, so the KL of an arm that collapses is pulled down and the curve need not rise with G (gen64 mostly sees
tokens before the onset). Matching needs a one-to-one G ↔ KL map, so a curve whose KL is not non-decreasing in G
gets no matched comparison. Per group, ``coverage`` records the KL range of both curves, the grid, the single-
site collapse knee (the lowest-G reference arm whose collapse exceeds the unsteered one, paired bootstrap) and
any issue (non-monotone curve, empty grid, distributed range stopping below the knee, so every matched point
lies where neither arm collapses); ``--analyse`` prints them. ``--dist-gains`` gives the K > 1 arms their own
total gains so their curves can reach the knee. Also the unsteered collapse rate with its across-seed spread.
PRE-STATED comparison: at matched KL, collapse(distributed) − collapse(single site), paired over (prompt,
seed), bootstrap with prompts as clusters; "lower" when the interval lies below 0.

Resume guard: every setting that changes the numbers or what is recorded (model, dtype, attention kernel,
lengths, window, sampling, batch sizes, ``--gen-tf``, ...) must match the stored ones; site sets, directions,
doses and gains may be added on a resume (the header keeps the union of the site sets).

Reused, not rewritten: ``text`` (CONDITIONS, make_batches, build_ctrl, closure_boot, floats, lens-free parts of
the arm loop's arithmetic), ``ceiling.gen_prompts`` and its sampling loop's settings, ``measure`` (loading,
paths, token_metrics, NormCapture, buckets, sums, GistDriftSource, OptionalSource, ResultFile, ArmTraces,
attention_below), ``common`` (SteerSide, paired_boot, gain_stats, git_state), ``controller`` (CA2Controller,
CA2Attachment), ``gdn.GDNTaps`` and its sources, ``attention.AttnTap`` and its sources, ``reason.self_loop_rate``.
Copied (single-site code that cannot host K sites): the per-row accumulation of ``text.run_arm`` (without
contiguity, logit lens, impulse and the attn_up / attn_last taps), the block hook of ``HookedLM._tap``, the
sampling loop of ``ceiling.run_gen``, the source registration of ``measure.register_signals`` (with shared taps).

    python experiments/ca2/multisite.py --model Qwen/Qwen3.5-2B-Base --modes text,gen --conditions full \\
        2>&1 | tee results/evolve/round12/stage0/multisite_q35.log
    python experiments/ca2/multisite.py --analyse results/evolve/round12/stage0/multisite_Qwen3.5-2B-Base.json
    python experiments/ca2/multisite.py --tiny --tiny-arch qwen35          # CPU smoke
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402
from ca2 import ceiling as ce  # noqa: E402
from ca2 import common  # noqa: E402
from ca2 import measure as ms  # noqa: E402
from ca2 import text as tx  # noqa: E402
from ca2.controller import CA2Attachment, CA2Controller, Tap  # noqa: E402
from ca2.sources import SOURCES, ConstantSource  # noqa: E402

MODES = ("text", "gen")
DOSES = ("lin", "sqrt", "full")
DIRECTIONS = ("gist", "anti_gist", "mean", "random", "random_shared")
SEED_STRIDE = 1000
BOOK = ("ids", "col", "mask", "idx", "step", "steer")          # identical at every site: traced once as bk/<key>
OUT_KEYS = ("loss", "ent", "rank", "top1", "p_tgt", "p_max")


# ------------------------------------------------------------------------------------------------ CLI
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a = p.add_argument
    a("--analyse", default=None, metavar="JSON", help="analyse a results JSON of this runner and exit")
    a("--analyse-out", default=None, help="write the analysis JSON here (default: print only)")
    a("--kl-from", default="auto", choices=["auto", "text", "gen", "gen64"],
      help="effect axis: text KL (same tokens in every arm), gen = KL along the arm's own continuation, gen64 = "
           "the same over its first 64 sampled tokens; auto = text when every arm has it, else gen64")
    a("--kl-cond", default="full", help="text condition whose KL is the effect axis (--kl-from text)")
    a("--kl-grid", default=None, help="common KL values (default: 5 geometric points over the overlap)")
    a("--ref-set", default=None, help="single-site reference set label (default: the first set with one site)")
    a("--model", default="Qwen/Qwen3.5-2B-Base")
    a("--dtype", default="bfloat16", choices=list(pl.DTYPES))
    a("--device", default="cuda")
    a("--attn-impl", default="sdpa")
    a("--tiny", action="store_true", help="tiny random model + byte tokenizer, CPU, small defaults")
    a("--tiny-arch", default="qwen35", choices=["qwen35", "qwen3", "gpt2"])
    a("--seed", type=int, default=0)
    a("--text-file", default="~/limbic-data/wikitext103_test.txt")
    a("--max-chars", type=int, default=None, help="read only the first N characters (default: all; tiny: 200k)")
    a("--modes", default="text,gen")
    a("--site-sets", default=None, help="';'-separated site sets, each a comma list with a-b ranges (default: "
                                        "hybrid 12;7,12,19;2,6,10,14,18,22;2-21, Qwen3 14;7,14,21;6,10,14,18,22;4-24)")
    a("--doses", default="lin,sqrt", help=f"dose rules among {DOSES} (K = 1 runs once as 'one')")
    a("--directions", default="gist,random,random_shared,mean,anti_gist", help=f"text-mode directions {DIRECTIONS}")
    a("--gen-directions", default="gist,random,random_shared", help="gen-mode directions")
    a("--gains", default=None, help="total gains G (default 0.1,0.2,0.3,0.5,0.75; tiny 0.2,0.5)")
    a("--dist-gains", default=None, help="total gains G of the multi-site (K > 1) arms (default: --gains); higher "
                                         "values let the distributed KL curves reach the single-site collapse knee")
    a("--random-seed", type=int, default=0, help="seed r of the random directions (site i: r + 1000·(i − 1))")
    a("--g-abs", type=float, default=2.0)
    a("--betas", default="0.9,0.995")
    a("--signals", default="all", choices=["all", "lean"], help="taps and tap sources at every site, or not")
    a("--conditions", default="full,attn,all")
    a("--window", type=int, default=None, help="attention window W (default 64; tiny 16)")
    a("--n-seqs", type=int, default=None, help="text rows (default 64; tiny 4)")
    a("--seq-len", type=int, default=None, help="text tokens per row (default 1024; tiny 96)")
    a("--batch", type=int, default=None, help="text rows per forward (default 2)")
    a("--sink", type=int, default=1)
    a("--steer-from", type=int, default=None, help="first steered text position (default 32; tiny 8)")
    a("--gen-n", type=int, default=None, help="generation prompts (default 96; tiny 4)")
    a("--gen-prompt-len", type=int, default=None, help="prompt tokens after the sink (default 32; tiny 8)")
    a("--gen-new", type=int, default=None, help="new tokens (default 512; tiny 16)")
    a("--gen-temp", type=float, default=0.7)
    a("--gen-batch", type=int, default=None, help="prompts per generation batch, one for all arms (default 16)")
    a("--gen-seeds", type=int, default=None, help="sampling seeds per arm (default 3; tiny 2)")
    a("--gen-tf", action=argparse.BooleanOptionalAction, default=True,
      help="teacher-force each steered continuation (steered and unsteered) for per-token KL")
    a("--tf-batch", type=int, default=None, help="rows per teacher-forced pass (default 4; tiny 2)")
    a("--attn-window", type=int, default=32, help="W_r of the attention taps (capped at W in tiny runs)")
    a("--gdn-mismatch-n", type=int, default=8, help="rows (text) / prompts (gen) per arm running the GDN loop")
    a("--gdn-per-head", default="beta,ret,fam,fam_lag,err,str,cos_vp", help="per-head GDN keys on that subsample")
    a("--trace-tokens", type=int, default=None, help="items per arm with per-token arrays (-1: all; default 16)")
    a("--arms", default=None, help="';'-separated fnmatch patterns of arm names to run (e.g. 'none;L12:*')")
    a("--boot", type=int, default=2000)
    a("--name", default=None)
    a("--stage", default="stage0")
    a("--out", default=None)
    a("--trace-dir", default=None)
    a("--force", action="store_true")
    args = p.parse_args(argv)
    if args.analyse:
        return args
    tiny = dict(seq_len=96, n_seqs=4, batch=2, window=16, steer_from=8, gen_n=4, gen_prompt_len=8, gen_new=16,
                gen_batch=2, gen_seeds=2, tf_batch=2, max_chars=200_000, gains="0.2,0.5", trace_tokens=-1)
    real = dict(seq_len=1024, n_seqs=64, batch=2, window=64, steer_from=32, gen_n=96, gen_prompt_len=32,
                gen_new=512, gen_batch=16, gen_seeds=3, tf_batch=4, max_chars=None, gains="0.1,0.2,0.3,0.5,0.75",
                trace_tokens=16)
    for k, v in (tiny if args.tiny else real).items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.tiny:
        args.attn_window = min(args.attn_window, args.window, args.steer_from)
        if args.dtype == "bfloat16" and args.device != "cuda":
            args.dtype = "float32"
    for flag, val, known in (("--doses", args.doses, DOSES), ("--directions", args.directions, DIRECTIONS),
                             ("--gen-directions", args.gen_directions, DIRECTIONS), ("--modes", args.modes, MODES),
                             ("--conditions", args.conditions, tuple(tx.CONDITIONS))):
        bad = [x for x in val.split(",") if x and x not in known]
        if bad:
            p.error(f"{flag}: unknown {bad}; known {list(known)}")
    if args.gen_seeds < 1 or args.gen_batch < 1 or args.tf_batch < 1:
        p.error("--gen-seeds, --gen-batch and --tf-batch must be >= 1")
    # fields text.make_batches / text.build_ctrl / ceiling.gen_prompts read
    args.impulse_start, args.impulse_len, args.impulse_directions = 0, 1, ""
    return args


# ----------------------------------------------------------------------------------------------- arms
def parse_sites(spec: str) -> list[int]:
    """'7,12,19' -> [7, 12, 19]; '2-5' -> [2, 3, 4, 5]; mixed lists allowed; sorted and deduplicated."""
    out: list[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError(f"empty site set {spec!r}")
    return sorted(set(out))


def default_site_sets(kinds: Sequence[str]) -> str:
    n = len(kinds)
    if ms.is_hybrid(kinds) and n == 24:
        return "12;7,12,19;2,6,10,14,18,22;2-21"
    if not ms.is_hybrid(kinds) and n == 28:
        return "14;7,14,21;6,10,14,18,22;4-24"
    mid = min(n // 2, n - 2)
    three = sorted({max(0, n // 4), mid, min(n - 2, (3 * n) // 4)})
    return f"{mid};{','.join(map(str, three))};0-{n - 2}"


def site_sets_of(args, kinds) -> dict[str, list[int]]:
    n = len(kinds)
    out: dict[str, list[int]] = {}
    for spec in (args.site_sets or default_site_sets(kinds)).split(";"):
        spec = spec.replace(" ", "")
        if not spec:
            continue
        sites = parse_sites(spec)
        if sites[0] < 0 or sites[-1] >= n - 1:
            raise SystemExit(f"site set {spec}: sites must lie in [0, {n - 2}] (below the last block)")
        out[f"L{spec}"] = sites
    if not out:
        raise SystemExit("no site sets")
    return out


def site_gains(G: float, K: int, dose: str) -> list[float]:
    """Per-site gains g_i for total gain G over K sites (module docstring)."""
    if K == 1 or dose in ("one", "full", "none"):
        return [float(G)] * K
    if dose == "lin":
        return [float(G) / K] * K
    if dose == "sqrt":
        return [float(G) / math.sqrt(K)] * K
    raise ValueError(f"unknown dose {dose!r}")


def site_seeds(direction: str, K: int, r: int) -> list[int]:
    if direction == "random":
        return [r + SEED_STRIDE * i for i in range(K)]
    return [r] * K


def make_arm(label: str, sites: list[int], dose: str, direction: str, G: float, r: int) -> dict:
    K = len(sites)
    name = "none" if G == 0 and label == "union" else f"{label}:{dose}:{direction}@{G:g}"
    return {"name": name, "set": label, "sites": list(sites), "K": K, "dose": dose, "direction": direction,
            "ctrl_direction": "random" if direction.startswith("random") else direction, "G": float(G),
            "g_site": site_gains(G, K, dose), "seeds": site_seeds(direction, K, r)}


def make_arms(site_sets: dict[str, list[int]], doses: Sequence[str], dirs: Sequence[str], gains: Sequence[float],
              r: int, patterns: str | None = None, dist_gains: Sequence[float] | None = None) -> list[dict]:
    """none, then per set × dose × direction × G; ``dist_gains`` (default ``gains``) are the G of the K > 1 sets."""
    union = sorted({s for v in site_sets.values() for s in v})
    arms = [make_arm("union", union, "none", "gist", 0.0, r)]
    for label, sites in site_sets.items():
        K = len(sites)
        for dose in (["one"] if K == 1 else list(doses)):
            for d in dirs:
                if K == 1 and d == "random_shared":
                    continue
                for G in (gains if K == 1 or not dist_gains else dist_gains):
                    if G > 0:
                        arms.append(make_arm(label, sites, dose, d, G, r))
    if patterns:
        pats = [x for x in patterns.split(";") if x]
        arms = [a for a in arms if any(fnmatch.fnmatchcase(a["name"], q) for q in pats)]
    return arms


def arm_meta(arm: dict) -> dict:
    return {k: arm[k] for k in ("set", "sites", "K", "dose", "direction", "G", "g_site", "seeds")}


def trace_name(name: str) -> str:
    return name.replace(",", ".")


# ------------------------------------------------------------------------------------- controllers, taps
def site_arm(arm: dict, i: int) -> dict:
    """The single-site arm dict ``text.build_ctrl`` takes, for site i."""
    return {"kind": "fixed", "direction": arm["ctrl_direction"], "seed": int(arm["seeds"][i]),
            "gain": float(arm["g_site"][i])}


def gen_ctrl(arm: dict, i: int, d_model: int, args, steer_from) -> CA2Controller:
    """Site i's controller for generation, as ``ceiling.run_gen`` builds its one controller."""
    g = float(arm["g_site"][i])
    return CA2Controller(d_model, source=ConstantSource(1.0), g_min=0.0, g_max=g, g_abs=max(args.g_abs, g),
                         smooth="none", direction=arm["ctrl_direction"], betas=tuple(tx.floats(args.betas)),
                         sink_skip=args.sink, steer_from=steer_from, seed=int(arm["seeds"][i]),
                         massive_window=max(64, args.attn_window), sources=("agree",))


class _View(Tap):
    """Read-only view of a tap owned by another controller: sources read it (attribute access is delegated), but
    reset, begin_call and signals are the base class's no-ops, so only the owner manages and traces it."""

    def __init__(self, inner: Tap) -> None:
        self.__dict__["_inner"] = inner
        self.__dict__["name"] = inner.name

    def __getattr__(self, key: str):
        return getattr(self.__dict__["_inner"], key)


class SiteSignals:
    """Every relevance signal at every site with shared taps (module docstring, "Signals")."""

    def __init__(self, model, blocks, kinds, sites: Sequence[int], sink: int, attn_window: int, gdn_mismatch: bool,
                 per_head: Sequence[str] = (), signals: str = "all") -> None:
        self.kinds, self.sites, self.signals = list(kinds), list(sites), signals
        self.gdn = None
        self.attn: dict[int, Tap] = {}
        if signals != "all":
            return
        host = SimpleNamespace(model=model, blocks=blocks, kinds=self.kinds, ctrl=None, inject_before=len(blocks),
                               relevance_layer=self.sites[-1], tap=self.sites[-1], sink=int(sink))
        if ms.is_hybrid(self.kinds):
            self.gdn = ms.gdn_mod().GDNTaps(mismatch=gdn_mismatch, per_head=list(per_head))
            self.gdn.attach(host)
        A = ms.attention_mod()
        for site in self.sites:
            a = ms.attention_below(self.kinds, site)
            if a is not None and a not in self.attn and ms.has_qk_norm(blocks, a):
                t = A.AttnTap(layer=a, window=attn_window, name=f"attn{a}")
                t.attach(host)
                self.attn[a] = t

    def register(self, ctrls: Sequence[CA2Controller]) -> None:
        """Sources on every controller; the shared taps on the first (owner) and views on the others."""
        G = ms.gdn_mod() if self.gdn is not None else None
        taps = ([self.gdn] if self.gdn is not None else []) + list(self.attn.values())
        for i, (c, site) in enumerate(zip(ctrls, self.sites)):
            if "gist_drift" not in c.sources:
                c.add_source(ms.GistDriftSource())
            for t in taps:
                c.add_tap(t if i == 0 else _View(t))
            if self.signals != "all":
                continue
            a = ms.attention_below(self.kinds, site)
            if a in self.attn:
                for s in ("attn_window", "attn_agree"):
                    if s not in c.sources:
                        c.add_source(SOURCES[s](tap=f"attn{a}"))
            if self.gdn is not None:
                layers = [l for l in self.gdn.layers if l <= site]
                if not layers:
                    continue
                for n, make in G.GDN_SOURCES.items():
                    if n in c.sources:
                        continue
                    src = make(tap="gdn", layers=layers)
                    c.add_source(src if src.signal in G.GATE_KEYS else ms.OptionalSource(src))

    def remove(self) -> None:
        for t in ([self.gdn] if self.gdn is not None else []) + list(self.attn.values()):
            t.remove()


class TextHost:
    """K controllers hosted as ``text.py`` hosts one: a ``common.SteerSide`` per site (sink stripped, reset at every
    call) and a block-output hook replicating ``HookedLM._tap`` (returns None when the gain is 0 throughout)."""

    def __init__(self, model, blocks, ctrls: Sequence[CA2Controller], sites: Sequence[int], sink: int) -> None:
        self.sink = int(sink)
        self.sides = [common.SteerSide(c, model, sink=self.sink, taps=(), tap_layer=L) for c, L in zip(ctrls, sites)]
        self.handles = [blocks[L].register_forward_hook(self._hook(sd)) for sd, L in zip(self.sides, sites)]

    def _hook(self, side: common.SteerSide):
        s = self.sink

        def fn(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            x = h[:, s:].float()
            body, _ = side(x)
            if body is x:
                return None
            y = torch.cat([h[:, :s], body.to(h.dtype)], 1)
            return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y
        return fn

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        for sd in self.sides:
            sd.remove()


class GenHost:
    """K ``CA2Attachment`` instances (one controller each, inject at the site's own output; KV cache works)."""

    def __init__(self, model, ctrls: Sequence[CA2Controller], sites: Sequence[int]) -> None:
        self.atts = [CA2Attachment(model, c, tap=L) for c, L in zip(ctrls, sites)]

    def remove(self) -> None:
        for a in self.atts:
            a.remove()
        self.atts = []


def consec_cos(ctrls: Sequence[CA2Controller]) -> torch.Tensor | None:
    """cos(d̂⁽ⁱ⁾, d̂⁽ⁱ⁺¹⁾) of this call, (B, T, K − 1); NaN where either direction is zero (undefined gist)."""
    ds = [c.current.get("d") for c in ctrls]
    if len(ds) < 2 or any(d is None for d in ds):
        return None
    out = []
    for a, b in zip(ds[:-1], ds[1:]):
        a, b = a.float(), b.float()
        c = F.cosine_similarity(a, b, dim=-1, eps=1e-12)
        ok = (a.norm(dim=-1) > 0) & (b.norm(dim=-1) > 0)
        out.append(torch.where(ok, c, torch.full_like(c, float("nan"))))
    return torch.stack(out, -1)


def assemble(traces: Sequence[dict], sites: Sequence[int]) -> dict[str, torch.Tensor]:
    """One per-token dict from K controller traces: site<L>/<key>, bk/<key> (site 1), tap/* (the owner's)."""
    out: dict[str, torch.Tensor] = {}
    for i, (tt, L) in enumerate(zip(traces, sites)):
        for k, v in tt.items():
            if k.startswith("tap/"):
                if i == 0:
                    out[k] = v
            elif k in BOOK:
                if i == 0:
                    out[f"bk/{k}"] = v
            else:
                out[f"site{L}/{k}"] = v
    return out


def _put(items: dict, key: str, rows) -> None:
    items.setdefault(key, []).extend(np.asarray(rows).tolist())


def _signal_sums(acc: dict, tt: dict, sel: torch.Tensor) -> None:
    for k, v in tt.items():
        if v.dim() == 2 and v.is_floating_point() and tuple(v.shape) == tuple(sel.shape):
            sm, ct = ms.masked_sum(v, sel)
            cur = acc.setdefault(k, [0.0, 0.0])
            cur[0] += float(sm.sum())
            cur[1] += float(ct.sum())


def _mean(sums: dict) -> dict:
    return {k: (v[0] / v[1] if v[1] else None) for k, v in sorted(sums.items())}


# ------------------------------------------------------------------------------------------- text mode
def run_text_arm(arm: dict, cond: str, lm: pl.HookedLM, model, blocks, kinds, batches, args, d_model: int,
                 writer: ms.ArmTraces) -> dict:
    s, win = args.sink, (args.window if cond != "full" else 0)
    n_blk, T_text = len(blocks), args.seq_len
    sites, K = arm["sites"], arm["K"]
    off_e, pos_e = ms.offset_edges(args.window), ms.log2_edges(T_text)
    per_head = [k for k in args.gdn_per_head.split(",") if k]
    n_sub_b = len(batches) if args.gdn_mismatch_n < 0 else -(-args.gdn_mismatch_n // args.batch)
    items: dict[str, list] = {}
    sig: list[dict] = [{} for _ in sites]
    gs: list[list] = [[] for _ in sites]
    checks = {"gain0_equal": None, "kl_base_max_none": None, "site_gain_dev": 0.0}
    tm = {"base_s": 0.0, "arm_s": 0.0, "metrics_s": 0.0, "tokens": 0}
    text_pos = torch.arange(T_text).float()
    for bi, (ids, scored, rows) in enumerate(batches):
        B, T = ids.shape
        scored = torch.cat([scored[:, 1:], torch.zeros_like(scored[:, :1])], 1)   # target mask → predicting position
        tg, valid = ms.shifted_targets(ids)
        valid[:, :s] = False
        t0 = ms.timer()
        refs = {}
        ncap = ms.NormCapture(blocks)
        try:
            with lm.condition(win, False):
                refs["base"] = lm(ids)
        finally:
            ncap.remove()
        base_norm = ncap.norms()[:, s:].float().cpu()
        if win:
            with lm.condition(0, False):
                refs["full"] = lm(ids)
        tm["base_s"] += ms.timer() - t0
        sub = bi < n_sub_b
        ctrls = [tx.build_ctrl(site_arm(arm, i), d_model, args, T_text) for i in range(K)]
        sset = SiteSignals(model, blocks, kinds, sites, s, args.attn_window, sub, per_head, args.signals)
        sset.register(ctrls)
        for c in ctrls:
            c.record = True
        host = TextHost(model, blocks, ctrls, sites, s)
        ncap = ms.NormCapture(blocks)                     # registered after the host: sees the injected outputs
        t0 = ms.timer()
        try:
            with lm.condition(win, False):                # HookedLM supplies the window only; the host injects
                lg = lm(ids)
        finally:
            ncap.remove()
            host.remove()
            sset.remove()
        tm["arm_s"] += ms.timer() - t0
        t0 = ms.timer()
        arm_norm = ncap.norms()[:, s:].float().cpu()
        if arm["G"] == 0:
            eq = bool(torch.equal(lg, refs["base"]))
            checks["gain0_equal"] = eq if checks["gain0_equal"] is None else (checks["gain0_equal"] and eq)
        m = ms.token_metrics(lg, tg, valid, refs)
        mb = ms.token_metrics(refs["base"], tg, valid)
        m["loss_base"], m["ent_base"] = mb["loss"], mb["ent"]
        if win:
            m["loss_full"] = ms.token_metrics(refs["full"], tg, valid)["loss"]
        cc = consec_cos(ctrls)
        cc = None if cc is None else cc.float().cpu()
        traces = [c.trace_tensors() for c in ctrls]
        del lg, refs, ctrls
        cut = lambda x: x[:, s:].detach().float().cpu()                         # noqa: E731
        tok = assemble(traces, sites)
        for k in OUT_KEYS + ("kl_base", "loss_base", "ent_base") + (("kl_full", "loss_full") if win else ()):
            tok[f"out/{k}"] = cut(m[k])
        tok["out/norm_ratio"] = arm_norm / base_norm.clamp_min(1e-12)
        if cc is not None:
            tok["multi/cos_consec"] = cc
        tp = text_pos.expand(B, -1)
        tok["pos/text"] = tp
        tok["pos/offset"] = tp % args.window
        tok["pos/block"] = tp // args.window
        tok["pos/scored"] = scored[:, s:].float()
        # ---- per-row sums over the scored tokens
        sc = scored[:, s:].bool() & torch.isfinite(tok["out/loss"])
        steer = traces[0]["steer"] > 0
        stsc = steer & sc
        nll, n = ms.masked_sum(tok["out/loss"], sc)
        _put(items, "row", rows)
        _put(items, "start", [r * args.seq_len for r in rows])
        _put(items, "nll_sum", nll)
        _put(items, "n_tok", n)
        for k in ("top1", "ent", "ent_base", "kl_base", "loss_base") + (("kl_full", "loss_full") if win else ()):
            _put(items, f"{k}_sum", ms.masked_sum(tok[f"out/{k}"], sc)[0])
        dent = tok["out/ent"] - tok["out/ent_base"]
        _put(items, "absdent_sum", ms.masked_sum(dent.abs(), sc)[0])
        nr = tok["out/norm_ratio"]
        w = sc.float()[..., None] * torch.isfinite(nr).float()
        _put(items, "normratio_sum", (torch.nan_to_num(nr) * w).sum(1).double().numpy())
        _put(items, "normratio_n", w.sum(1).double().numpy())
        _put(items, "steer_n", ms.masked_sum(torch.ones_like(tok["out/loss"]), stsc)[1])
        inj, injb = [], []
        for i, (tr, L) in enumerate(zip(traces, sites)):
            inj.append(ms.masked_sum(tr["inj_ratio"], stsc)[0])
            rel = tr["inj_ratio"] * tr["xnorm"] / base_norm[..., L].clamp_min(1e-12)
            injb.append(ms.masked_sum(rel, stsc)[0])
            g = tr["g_raw"][steer]
            gs[i].append(g)
            if g.numel():
                checks["site_gain_dev"] = max(checks["site_gain_dev"], float((g - arm["g_site"][i]).abs().max()))
            _signal_sums(sig[i], tr, sc)
        _put(items, "inj_sum", np.stack(inj, 1))
        _put(items, "injbase_sum", np.stack(injb, 1))
        if cc is not None:
            ok = torch.isfinite(cc) & stsc[..., None]
            _put(items, "cos_sum", torch.where(ok, cc, torch.zeros_like(cc)).double().sum(1).numpy())
            _put(items, "cos_n", ok.double().sum(1).numpy())
        allv = torch.isfinite(tok["out/loss"])
        bo, bp = ms.bucketize(tok["pos/offset"], off_e), ms.bucketize(tp, pos_e)
        for bname, bidx, keep in (("off", bo, sc), ("pos", bp, sc), ("offall", bo, allv), ("posall", bp, allv)):
            bidx = torch.where(keep, bidx, torch.full_like(bidx, -1))
            nb_ = len(off_e) - 1 if bname.startswith("off") else len(pos_e) - 1
            for k in ("loss", "loss_base", "kl_base") + (("loss_full", "kl_full") if win else ()):
                sm, ct = ms.bucket_sums(tok[f"out/{k}"], bidx, nb_)
                _put(items, f"{bname}_{k}_sum", sm)
                if k == "loss":
                    _put(items, f"{bname}_n", ct)
        if arm["G"] == 0:
            kmax = float(torch.nan_to_num(tok["out/kl_base"], nan=0.0).max())
            checks["kl_base_max_none"] = max(checks["kl_base_max_none"] or 0.0, kmax)
        tm["metrics_s"] += ms.timer() - t0
        tm["tokens"] += int(B * (T - s))
        for b in range(B):
            item = {"row": rows[b], "start": rows[b] * args.seq_len, "cond": cond, "arm": arm["name"],
                    "nll_sum": float(nll[b]), "n_tok": float(n[b]), "batch": bi, "gdn_loop": sub}
            writer.add(item, ms.row_dict(tok, b), sub)
    return finish_text(arm, cond, win, items, sig, gs, checks, tm, args, n_blk, (off_e, pos_e))


def finish_text(arm, cond, win, items, sig, gs, checks, tm, args, n_blk: int, edges) -> dict:
    it = {k: np.asarray(v, dtype=np.float64) for k, v in items.items()}
    n = it["n_tok"]
    tot = lambda k: float(it[f"{k}_sum"].sum() / max(n.sum(), 1))             # noqa: E731
    agg = {"loss": float(it["nll_sum"].sum() / max(n.sum(), 1)), "n_tok": int(n.sum()), "n_seq": int(len(n))}
    agg["ppl"] = math.exp(agg["loss"])
    for k in ("top1", "ent", "ent_base", "kl_base", "loss_base", "absdent") + (("kl_full", "loss_full") if win else ()):
        agg[k] = tot(k)
    agg["dent"] = agg["ent"] - agg["ent_base"]
    L = it["nll_sum"] / np.maximum(n, 1)
    Lb = it["loss_base_sum"] / np.maximum(n, 1)
    agg["d_loss"] = agg["loss"] - agg["loss_base"]
    agg["d_loss_vs_base"] = ms.boot(L, Lb, n=args.boot, seed=args.seed)   # base = the unsteered pass = none
    agg["kl_base_boot"] = ms.boot(it["kl_base_sum"] / np.maximum(n, 1), n=args.boot, seed=args.seed)
    if win:
        agg["d_loss_vs_full"] = ms.boot(L, it["loss_full_sum"] / np.maximum(n, 1), n=args.boot, seed=args.seed)
        agg["closure"] = tx.closure_boot(it["nll_sum"], it["loss_base_sum"], it["loss_full_sum"], n, args.boot,
                                         args.seed)
    for b in ("off", "pos", "offall", "posall"):
        nb = it[f"{b}_n"]
        den = np.maximum(nb.sum(0), 1)
        agg[f"loss_by_{b}"] = (it[f"{b}_loss_sum"].sum(0) / den).tolist()
        agg[f"dloss_by_{b}"] = ((it[f"{b}_loss_sum"] - it[f"{b}_loss_base_sum"]).sum(0) / den).tolist()
        agg[f"kl_by_{b}"] = (it[f"{b}_kl_base_sum"].sum(0) / den).tolist()
        agg[f"n_by_{b}"] = nb.sum(0).tolist()
    agg["norm_ratio_by_block"] = (it["normratio_sum"].sum(0) / np.maximum(it["normratio_n"].sum(0), 1)).tolist()
    agg["norm_ratio_final"] = agg["norm_ratio_by_block"][-1]
    stn = max(float(it["steer_n"].sum()), 1.0)
    inj = it["inj_sum"].reshape(len(n), -1).sum(0) / stn
    injb = it["injbase_sum"].reshape(len(n), -1).sum(0) / stn
    by, byb = [0.0] * n_blk, [0.0] * n_blk
    for i, L_ in enumerate(arm["sites"]):
        by[L_], byb[L_] = float(inj[i]), float(injb[i])
    agg["inj_ratio_by_block"], agg["inj_ratio_base_by_block"] = by, byb
    agg["inj_ratio_total"], agg["inj_ratio_base_total"] = float(inj.sum()), float(injb.sum())
    if "cos_sum" in it:
        c = it["cos_sum"].sum(0) / np.maximum(it["cos_n"].sum(0), 1)
        agg["cos_consec"] = c.tolist()
        agg["cos_consec_mean"] = float(it["cos_sum"].sum() / max(it["cos_n"].sum(), 1))
    per_site = {}
    for i, L_ in enumerate(arm["sites"]):
        g = torch.cat(gs[i]) if gs[i] else torch.zeros(0)
        gst = common.gain_stats(g, torch.ones_like(g), 0.0, float(arm["g_site"][i])) if g.numel() else {"n": 0}
        per_site[str(L_)] = {"g": arm["g_site"][i], "gain_dist": gst, "inj_ratio": float(inj[i]),
                             "inj_ratio_base": float(injb[i]), "signal_means": _mean(sig[i])}
    tm = dict(tm)
    tm["overhead_ratio"] = tm["arm_s"] / max(tm["base_s"] / (2 if win else 1), 1e-9)
    tm["tok_per_s_arm"] = tm["tokens"] / max(tm["arm_s"], 1e-9)
    return {"mode": "text", "condition": cond, "window": win, "arm": arm["name"], "meta": arm_meta(arm),
            "complete": True, "agg": agg, "per_site": per_site, "checks": checks, "time": tm, "per_item": it,
            "edges": {"offset": edges[0], "position": edges[1]}}


# -------------------------------------------------------------------------------------------- gen mode
def _self_loop(tok, ids: Sequence[int]) -> float | None:
    from ca2.reason import self_loop_rate
    dec = getattr(tok, "decode", None)
    if dec is None:
        return None
    r = self_loop_rate(dec(list(ids)))
    return float("nan") if r is None else float(r)


@torch.no_grad()
def teacher_forced(arm: dict, model, blocks, full: torch.Tensor, P: int, args, d_model: int) -> dict:
    """Steered (fresh controllers, steer_from = column P − 1) and unsteered passes over prompt + continuation, in
    chunks of ``--tf-batch`` rows: KL(steered ‖ unsteered), log p of each next token, norm ratio by block."""
    dev = model.get_input_embeddings().weight.device
    B, T = full.shape
    out = {"kl": torch.full((B, T), float("nan")), "lp": torch.full((B, T), float("nan")),
           "norm_ratio": torch.full((B, T, len(blocks)), float("nan")), "base_norm": torch.zeros(B, T, len(blocks))}
    for a in range(0, B, args.tf_batch):
        x = full[a:a + args.tf_batch].to(dev)
        ctrls = [gen_ctrl(arm, i, d_model, args, P - 1) for i in range(arm["K"])]
        host = GenHost(model, ctrls, arm["sites"])
        nc = ms.NormCapture(blocks)
        try:
            lg_s = model(input_ids=x, use_cache=False).logits
        finally:
            nc.remove()
            host.remove()
        ns = nc.norms()
        nc = ms.NormCapture(blocks)
        try:
            lg_b = model(input_ids=x, use_cache=False).logits
        finally:
            nc.remove()
        nb = nc.norms()
        tg, valid = ms.shifted_targets(x)
        mm = ms.token_metrics(lg_s, tg, valid, {"base": lg_b})
        del lg_s, lg_b
        sl = slice(a, a + x.shape[0])
        out["kl"][sl] = mm["kl_base"].float().cpu()
        out["lp"][sl] = (-mm["loss"]).float().cpu()
        out["norm_ratio"][sl] = (ns / nb.clamp_min(1e-12)).float().cpu()
        out["base_norm"][sl] = nb.float().cpu()
    return out


@torch.no_grad()
def run_gen_arm(arm: dict, seed: int, model, blocks, kinds, prompts: torch.Tensor, args, d_model: int, tok,
                writer: ms.ArmTraces) -> dict:
    """Sampled continuations with a KV cache, K sites; the same seeded noise per batch in every arm (paired)."""
    aj = common.alignment().judge
    dev = model.get_input_embeddings().weight.device
    per_head = [k for k in args.gdn_per_head.split(",") if k]
    n_sub_b = 10 ** 9 if args.gdn_mismatch_n < 0 else -(-args.gdn_mismatch_n // args.gen_batch)
    sites, K, steered = arm["sites"], arm["K"], arm["G"] > 0
    n_blk, N = len(blocks), args.gen_new
    items: dict[str, list] = {}
    sig: list[dict] = [{} for _ in sites]
    gs: list[list] = [[] for _ in sites]
    acc = {"nr": np.zeros(n_blk), "nr_n": np.zeros(n_blk), "inj": np.zeros(K), "injb": np.zeros(K), "st_n": 0.0,
           "injb_n": 0.0, "cos": np.zeros(max(K - 1, 0)), "cos_n": np.zeros(max(K - 1, 0))}
    checks = {"site_gain_dev": 0.0, "tf_lp_dev_max": None}
    tm = {"gen_s": 0.0, "tf_s": 0.0, "tokens": 0}
    for bi, a in enumerate(range(0, len(prompts), args.gen_batch)):
        ids = prompts[a:a + args.gen_batch].to(dev)
        B, P = ids.shape
        sub = bi < n_sub_b
        ctrls = [gen_ctrl(arm, i, d_model, args, "last_prompt") for i in range(K)]
        sset = SiteSignals(model, blocks, kinds, sites, args.sink, args.attn_window, sub, per_head, args.signals)
        sset.register(ctrls)
        host = GenHost(model, ctrls, sites)
        gen = torch.Generator(device=dev).manual_seed(seed * 1_000_003 + bi)
        for c in ctrls:
            c.reset()
            c.record = True
        ent, lp, pmax, new, coss = [], [], [], [], []
        t0 = ms.timer()
        try:
            out = model(input_ids=ids, use_cache=True)
            cc = consec_cos(ctrls)
            if cc is not None:
                coss.append(cc.float().cpu())
            cache = out.past_key_values
            for step in range(N):
                z = out.logits[:, -1].float()
                lps = torch.log_softmax(z, -1)
                nxt = torch.multinomial(torch.softmax(z / args.gen_temp, -1), 1, generator=gen)
                ent.append(-(lps.exp() * lps).sum(-1))
                lp.append(lps.gather(-1, nxt)[:, 0])
                pmax.append(lps.amax(-1).exp())
                new.append(nxt)
                out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                cc = consec_cos(ctrls)
                if cc is not None:
                    coss.append(cc.float().cpu())
        finally:
            host.remove()
            sset.remove()
        tm["gen_s"] += ms.timer() - t0
        tm["tokens"] += B * N
        del out, cache
        gen_ids = torch.cat(new, 1).cpu()
        ent, lp, pmax = (torch.stack(x, 1).float().cpu() for x in (ent, lp, pmax))
        traces = [c.trace_tensors() for c in ctrls]
        Ttot = P + N
        full = torch.cat([ids.cpu(), gen_ids], 1)
        gm = torch.zeros(B, Ttot, dtype=torch.bool)
        gm[:, P - 1:P + N - 1] = True                       # columns whose next-token distribution was sampled
        tf = None
        if steered and args.gen_tf:
            t0 = ms.timer()
            tf = teacher_forced(arm, model, blocks, full, P, args, d_model)
            tm["tf_s"] += ms.timer() - t0

        def pad(x):
            return torch.cat([torch.full((B, P - 1), float("nan")), x, torch.full((B, 1), float("nan"))], 1)

        tok_ = assemble(traces, sites)
        tok_["gen/ids"] = full
        tok_["gen/is_gen"] = torch.cat([torch.zeros(B, P), torch.ones(B, N)], 1)
        tok_["gen/ent"], tok_["gen/lp_next"], tok_["gen/pmax"] = pad(ent), pad(lp), pad(pmax)
        if coss:
            tok_["multi/cos_consec"] = torch.cat(coss, 1)
        if tf is not None:
            tok_["tf/kl"], tok_["tf/lp_next"], tok_["tf/norm_ratio"] = tf["kl"], tf["lp"], tf["norm_ratio"]
        tok_ = {k: v for k, v in tok_.items() if v.shape[1] == Ttot}
        steer = traces[0]["steer"] > 0
        for i, (tr, L) in enumerate(zip(traces, sites)):
            g = tr["g_raw"][steer]
            gs[i].append(g)
            if g.numel():
                checks["site_gain_dev"] = max(checks["site_gain_dev"], float((g - arm["g_site"][i]).abs().max()))
            acc["inj"][i] += float(ms.masked_sum(tr["inj_ratio"], steer)[0].sum())
            if tf is not None:
                rel = tr["inj_ratio"] * tr["xnorm"] / tf["base_norm"][..., L].clamp_min(1e-12)
                acc["injb"][i] += float(ms.masked_sum(rel, steer)[0].sum())
            _signal_sums(sig[i], tr, steer)
        acc["st_n"] += float(steer.sum())
        if tf is not None:
            acc["injb_n"] += float(steer.sum())
            nr = tf["norm_ratio"]
            w = gm[..., None].float() * torch.isfinite(nr).float()
            acc["nr"] += (torch.nan_to_num(nr) * w).sum((0, 1)).double().numpy()
            acc["nr_n"] += w.sum((0, 1)).double().numpy()
        if coss and K > 1:
            cc = tok_["multi/cos_consec"]
            ok = torch.isfinite(cc) & steer[..., None]
            acc["cos"] += torch.where(ok, cc, torch.zeros_like(cc)).double().sum((0, 1)).numpy()
            acc["cos_n"] += ok.double().sum((0, 1)).numpy()
        for b in range(B):
            g_ids = gen_ids[b].tolist()
            onset = aj.repetition_onset(g_ids)
            rec = {"prompt": a + b, "seed": seed, "collapse": float(onset is not None),
                   "onset": float("nan") if onset is None else float(onset),
                   "distinct4": 1.0 - aj.degeneration(g_ids, 4), "self_loop": _self_loop(tok, g_ids),
                   "ent": float(ent[b].mean()), "nll": float(-lp[b].mean()), "n_gen": len(g_ids),
                   "gain_mean": float(traces[0]["g_raw"][b, P - 1:].mean())}
            if tf is not None:
                kb = tf["kl"][b][gm[b]]
                rec["kl_mean"] = float(torch.nanmean(kb))
                rec["kl_first64"] = float(torch.nanmean(kb[:64]))
                dev_lp = (tf["lp"][b][gm[b]] - lp[b]).abs()
                rec["tf_lp_dev"] = float(dev_lp.max())
                checks["tf_lp_dev_max"] = max(checks["tf_lp_dev_max"] or 0.0, rec["tf_lp_dev"])
            elif not steered:
                rec["kl_mean"] = rec["kl_first64"] = 0.0
            for k, v in rec.items():
                items.setdefault(k, []).append(v)
            writer.add({**rec, "arm": arm["name"]}, ms.row_dict(tok_, b), sub)
    it = {k: np.asarray([np.nan if v is None else v for v in vs], dtype=np.float64) for k, vs in items.items()}
    agg = {"collapse_rate": float(it["collapse"].mean()), "distinct4": float(it["distinct4"].mean()),
           "onset_median": float(np.nanmedian(it["onset"])) if np.isfinite(it["onset"]).any() else None,
           "self_loop": float(np.nanmean(it["self_loop"])) if np.isfinite(it["self_loop"]).any() else None,
           "ent": float(it["ent"].mean()), "nll": float(it["nll"].mean()), "n": int(len(it["collapse"]))}
    for k in ("kl_mean", "kl_first64"):
        if k in it:
            agg[k] = float(np.nanmean(it[k]))
    if acc["nr_n"].sum():
        agg["norm_ratio_by_block"] = (acc["nr"] / np.maximum(acc["nr_n"], 1)).tolist()
        agg["norm_ratio_final"] = agg["norm_ratio_by_block"][-1]
    inj = acc["inj"] / max(acc["st_n"], 1.0)
    by = [0.0] * n_blk
    for i, L in enumerate(sites):
        by[L] = float(inj[i])
    agg["inj_ratio_by_block"], agg["inj_ratio_total"] = by, float(inj.sum())
    if acc["injb_n"]:
        injb = acc["injb"] / acc["injb_n"]
        byb = [0.0] * n_blk
        for i, L in enumerate(sites):
            byb[L] = float(injb[i])
        agg["inj_ratio_base_by_block"], agg["inj_ratio_base_total"] = byb, float(injb.sum())
    if K > 1 and acc["cos_n"].sum():
        agg["cos_consec"] = (acc["cos"] / np.maximum(acc["cos_n"], 1)).tolist()
        agg["cos_consec_mean"] = float(acc["cos"].sum() / acc["cos_n"].sum())
    per_site = {}
    for i, L in enumerate(sites):
        g = torch.cat(gs[i]) if gs[i] else torch.zeros(0)
        gst = common.gain_stats(g, torch.ones_like(g), 0.0, float(arm["g_site"][i])) if g.numel() else {"n": 0}
        per_site[str(L)] = {"g": arm["g_site"][i], "gain_dist": gst, "inj_ratio": float(inj[i]),
                            "signal_means": _mean(sig[i])}
    tm["tok_per_s"] = tm["tokens"] / max(tm["gen_s"], 1e-9)
    return {"mode": "gen", "seed": seed, "arm": arm["name"], "meta": arm_meta(arm), "complete": True, "agg": agg,
            "per_site": per_site, "checks": checks, "time": tm, "per_item": it}


def gen_summary(res: ms.ResultFile, args) -> dict:
    """Per arm across seeds: collapse rate per seed, mean, SD; paired Δ against none (prompts as clusters)."""
    by: dict[str, dict[int, dict]] = {}
    for key, a in res.data["arms"].items():
        if key.startswith("gen/") and a.get("complete"):
            by.setdefault(a["arm"], {})[int(a["seed"])] = a
    none = by.get("none", {})
    out = {}
    for name, recs in by.items():
        seeds = sorted(recs)
        rates = [recs[s]["agg"]["collapse_rate"] for s in seeds]
        row = {"seeds": seeds, "collapse_rate_by_seed": rates, "collapse_rate": float(np.mean(rates)),
               "collapse_rate_sd": float(np.std(rates, ddof=1)) if len(rates) > 1 else None,
               "distinct4": float(np.mean([recs[s]["agg"]["distinct4"] for s in seeds])),
               "kl_mean": (float(np.mean([recs[s]["agg"]["kl_mean"] for s in seeds]))
                           if all(recs[s]["agg"].get("kl_mean") is not None for s in seeds) else None)}
        if name != "none":
            ca, cn, da, dn, cl = [], [], [], [], []
            for s in seeds:
                if s not in none:
                    continue
                pa, pn = recs[s]["per_item"], none[s]["per_item"]
                if list(pa["prompt"]) != list(pn["prompt"]):
                    continue
                ca += list(pa["collapse"])
                cn += list(pn["collapse"])
                da += list(pa["distinct4"])
                dn += list(pn["distinct4"])
                cl += list(pa["prompt"])
            if cl:
                row["d_collapse_vs_none"] = common.paired_boot(ca, cn, clusters=cl, n=args.boot, seed=args.seed)
                row["d_distinct4_vs_none"] = common.paired_boot(da, dn, clusters=cl, n=args.boot, seed=args.seed)
        out[name] = row
    res.data["gen_summary"] = ms.clean(out)
    res.save()
    return out


# ----------------------------------------------------------------------------------------------- analysis
def _locate(points: list[tuple[float, float]], kappa: float) -> tuple[int, float] | None:
    """First segment (i, i + 1) of the curve [(G, KL)] sorted by G with KL_i ≤ κ ≤ KL_{i+1}: (i, weight)."""
    for i in range(len(points) - 1):
        k0, k1 = points[i][1], points[i + 1][1]
        if k0 is None or k1 is None:
            continue
        lo, hi = min(k0, k1), max(k0, k1)
        if lo <= kappa <= hi:
            return i, (0.0 if k1 == k0 else (kappa - k0) / (k1 - k0))
    return None


def _interp(vecs: list[np.ndarray], loc: tuple[int, float] | None) -> np.ndarray | None:
    if loc is None:
        return None
    i, w = loc
    return (1.0 - w) * vecs[i] + w * vecs[i + 1]


def _monotone(points: list[tuple[float, float]]) -> bool:
    """KL non-decreasing in G along the curve (sorted by G); matching by KL needs a one-to-one G ↔ KL map."""
    ks = [k for _, k in points]
    return all(k is not None for k in ks) and all(b >= a for a, b in zip(ks, ks[1:]))


def _grid(curves: list[list[tuple[float, float]]], given: list[float] | None) -> list[float]:
    """Common KL values over the shared range (0, min over curves of max KL]. Every curve starts at the shared
    none point (G = 0, KL = 0), so each one is defined, by interpolation from that point, on all of (0, its max];
    the default is 5 geometric points from the smallest first positive KL of any curve (capped at the top) to it."""
    his = [max(k for _, k in c) for c in curves]
    hi = min(his) if his else 0.0
    if not hi or hi <= 0:
        return []
    if given:
        return [k for k in given if 0.0 < k <= hi]
    los = [min([k for _, k in c if k > 0], default=None) for c in curves]
    los = [x for x in los if x is not None]
    lo = min(min(los), hi) if los else hi
    if lo == hi:
        return [float(hi)]
    return [float(x) for x in np.geomspace(lo, hi, 5)]


def _ref_knee(rcurve: list[tuple[float, float]], rvecs: list[np.ndarray], clus: np.ndarray, boot: int,
              seed: int) -> dict | None:
    """The single-site collapse knee: the lowest-G reference arm whose collapse exceeds the unsteered arm's (paired
    bootstrap over (prompt, seed), prompts as clusters, interval above 0): its G and KL. None if no arm does."""
    for (G, k), v in zip(rcurve[1:], rvecs[1:]):
        bt = common.paired_boot(v, rvecs[0], clusters=clus, n=boot, seed=seed)
        if bt["lo"] is not None and bt["lo"] > 0:
            return {"G": G, "kl": k, "collapse": float(v.mean()), "d_collapse_vs_none": bt}
    return None


def analyse(path: str | Path, kl_from: str = "auto", kl_cond: str = "full", kl_grid: list[float] | None = None,
            ref_set: str | None = None, boot: int = 2000, seed: int = 0) -> dict:
    """Effect-matched curves and the pre-stated comparison (module docstring, "Analysis")."""
    d = json.loads(Path(path).expanduser().read_text())
    text: dict[str, dict[str, dict]] = {}
    gen: dict[str, dict[int, dict]] = {}
    meta: dict[str, dict] = {}
    for key, a in d.get("arms", {}).items():
        if not a.get("complete"):
            continue
        meta[a["arm"]] = a["meta"]
        if a["mode"] == "text":
            text.setdefault(a["condition"], {})[a["arm"]] = a
        else:
            gen.setdefault(a["arm"], {})[int(a["seed"])] = a
    sets = {m["set"]: list(m["sites"]) for m in meta.values() if m["set"] != "union"}
    sets.update(d.get("site_sets", {}))                 # the header's sets, plus any set an arm on file was run with
    if ref_set is None:
        ref_set = next((k for k, v in sets.items() if len(v) == 1), None)
    groups: dict[tuple[str, str, str], list[tuple[float, str]]] = {}
    for name, m in meta.items():
        if name != "none":
            groups.setdefault((m["set"], m["dose"], m["direction"]), []).append((m["G"], name))
    for v in groups.values():
        v.sort()

    def kl_text(name: str, cond: str) -> float | None:
        if name == "none":
            return 0.0
        r = text.get(cond, {}).get(name)
        return None if r is None else r["agg"]["kl_base"]

    def kl_gen(name: str, key: str = "kl_mean") -> float | None:
        if name == "none":
            return 0.0
        v = [r["agg"].get(key) for r in gen.get(name, {}).values()]
        return float(np.mean(v)) if v and all(x is not None for x in v) else None

    def gen_items(name: str, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
        recs = gen[name]
        c = np.concatenate([np.asarray(recs[s]["per_item"]["collapse"], dtype=float) for s in seeds])
        p = np.concatenate([np.asarray(recs[s]["per_item"]["prompt"], dtype=float) for s in seeds])
        return c, p

    def text_items(name: str, cond: str) -> np.ndarray:
        pi = text[cond][name]["per_item"]
        n = np.maximum(np.asarray(pi["n_tok"], dtype=float), 1)
        return (np.asarray(pi["nll_sum"]) - np.asarray(pi["loss_base_sum"])) / n

    out: dict = {"source": str(path), "ref_set": ref_set, "kl_cond": kl_cond, "groups": [], "comparison": []}
    # unsteered collapse and its across-seed spread
    if "none" in gen:
        seeds = sorted(gen["none"])
        rates = [gen["none"][s]["agg"]["collapse_rate"] for s in seeds]
        c, p = gen_items("none", seeds)
        out["unsteered"] = {"seeds": seeds, "collapse_rate_by_seed": rates, "collapse_rate": float(np.mean(rates)),
                            "sd": float(np.std(rates, ddof=1)) if len(rates) > 1 else None,
                            "min": float(min(rates)), "max": float(max(rates)),
                            "pooled": common.paired_boot(c, clusters=p, n=boot, seed=seed)}
    for (gset, dose, direction), pts in sorted(groups.items()):
        rdir = "random" if direction == "random_shared" else direction
        ref = groups.get((ref_set, "one", rdir), []) if ref_set else []
        names = [n for _, n in pts]
        rnames = [n for _, n in ref]
        g: dict = {"set": gset, "sites": sets.get(gset), "dose": dose, "direction": direction,
                   "K": meta[names[0]]["K"], "ref": f"{ref_set}:one:{rdir}" if ref else None}
        # --- collapse against KL
        src = kl_from
        if src == "auto":
            src = "text" if all(kl_text(n, kl_cond) is not None for n in names + rnames) else "gen64"
        klf = ((lambda n: kl_text(n, kl_cond)) if src == "text" else
               (lambda n: kl_gen(n, "kl_first64")) if src == "gen64" else kl_gen)
        g["kl_from"] = src
        have_gen = "none" in gen and all(n in gen for n in names)
        if have_gen:
            seeds = sorted(set(gen["none"]).intersection(*[set(gen[n]) for n in names + [r for r in rnames
                                                                                          if r in gen]]))
            g["seeds"] = seeds
            curve = [(0.0, 0.0)] + [(G, klf(n)) for G, n in pts]
            vecs = [gen_items("none", seeds)[0]] + [gen_items(n, seeds)[0] for n in names]
            clus = gen_items("none", seeds)[1]
            g["curve"] = []
            for i, ((G, k), v) in enumerate(zip(curve, vecs)):
                nm = "none" if i == 0 else names[i - 1]
                g["curve"].append({"G": G, "kl": k, "collapse_rate": float(v.mean()),
                                   "distinct4": float(np.mean([gen[nm][s]["agg"]["distinct4"] for s in seeds]))})
            cov: dict = {"kl_max": max([k for _, k in curve if k is not None], default=None),
                         "monotone": _monotone(curve), "grid": [], "issues": []}
            g["coverage"] = cov
            if not cov["monotone"]:
                cov["issues"].append(f"KL ({src}) not non-decreasing in G: no matched comparison")
            if ref and gset != ref_set and all(r in gen for r in rnames) and None not in [k for _, k in curve]:
                rcurve = [(0.0, 0.0)] + [(G, klf(n)) for G, n in ref]
                if None in [k for _, k in rcurve]:
                    cov["issues"].append(f"reference has no {src} KL on some arm: no matched comparison")
                else:
                    rvecs = [vecs[0]] + [gen_items(n, seeds)[0] for n in rnames]
                    cov["kl_max_ref"] = max(k for _, k in rcurve)
                    cov["ref_monotone"] = _monotone(rcurve)
                    knee = _ref_knee(rcurve, rvecs, clus, boot, seed)
                    cov["ref_knee"] = knee
                    if knee is None:
                        cov["issues"].append("no reference arm collapses above the unsteered rate")
                    elif cov["kl_max"] < knee["kl"]:
                        cov["issues"].append(f"KL range [0, {cov['kl_max']:.4g}] stops below the single-site "
                                             f"collapse knee (G {knee['G']:g}, KL {knee['kl']:.4g}): every matched "
                                             f"point lies where the single site does not collapse; raise the gains")
                    if not cov["ref_monotone"]:
                        cov["issues"].append(f"reference KL ({src}) not non-decreasing in G: no matched comparison")
                    grid = _grid([curve, rcurve], kl_grid) if cov["monotone"] and cov["ref_monotone"] else []
                    cov["grid"] = grid
                    if not grid and cov["monotone"] and cov["ref_monotone"]:
                        cov["issues"].append("empty KL grid (no shared KL range, or no --kl-grid value inside it)")
                    for kappa in grid:
                        a_, b_ = _interp(vecs, _locate(curve, kappa)), _interp(rvecs, _locate(rcurve, kappa))
                        if a_ is None or b_ is None:
                            continue
                        bt = common.paired_boot(a_, b_, clusters=clus, n=boot, seed=seed)
                        verdict = ("lower" if bt["hi"] is not None and bt["hi"] < 0 else
                                   "higher" if bt["lo"] is not None and bt["lo"] > 0 else "unclear")
                        out["comparison"].append({"set": gset, "dose": dose, "direction": direction, "kl": kappa,
                                                  "kl_from": src, "collapse": float(a_.mean()),
                                                  "collapse_ref": float(b_.mean()), "d_collapse": bt,
                                                  "verdict": verdict})
        # --- text Δloss against KL, per condition (KL of that condition)
        g["dloss"] = {}
        for cond, recs in text.items():
            if "none" not in recs or not all(n in recs for n in names):
                continue
            curve = [(0.0, 0.0)] + [(G, recs[n]["agg"]["kl_base"]) for G, n in pts]
            vecs = [text_items("none", cond)] + [text_items(n, cond) for n in names]
            ent = {"curve": [{"G": G, "kl": k, "d_loss": float(np.mean(v))} for (G, k), v in zip(curve, vecs)],
                   "matched": [], "monotone": _monotone(curve)}
            if ref and gset != ref_set and all(r in recs for r in rnames):
                rcurve = [(0.0, 0.0)] + [(G, recs[n]["agg"]["kl_base"]) for G, n in ref]
                rvecs = [vecs[0]] + [text_items(n, cond) for n in rnames]
                ent["ref_monotone"] = _monotone(rcurve)
                ent["grid"] = _grid([curve, rcurve], kl_grid) if ent["monotone"] and ent["ref_monotone"] else []
                for kappa in ent["grid"]:
                    a_, b_ = _interp(vecs, _locate(curve, kappa)), _interp(rvecs, _locate(rcurve, kappa))
                    if a_ is None or b_ is None:
                        continue
                    ent["matched"].append({"kl": kappa, "d_loss": float(a_.mean()), "d_loss_ref": float(b_.mean()),
                                           "diff": common.paired_boot(a_, b_, n=boot, seed=seed)})
            g["dloss"][cond] = ent
        out["groups"].append(g)
    return ms.clean(out)


def analyse_main(args) -> dict:
    grid = [float(x) for x in args.kl_grid.split(",") if x.strip()] if args.kl_grid else None
    out = analyse(args.analyse, args.kl_from, args.kl_cond, grid, args.ref_set, args.boot, args.seed)
    if "unsteered" in out:
        u = out["unsteered"]
        print(f"[multisite] unsteered collapse {u['collapse_rate']:.3f} by seed {u['collapse_rate_by_seed']} "
              f"sd {u['sd']}", flush=True)
    for g in out["groups"]:
        for msg in g.get("coverage", {}).get("issues", []):
            print(f"[multisite] WARNING {g['set']}:{g['dose']}:{g['direction']} ({g['kl_from']}): {msg}", flush=True)
    for c in out["comparison"]:
        b = c["d_collapse"]
        print(f"[multisite] {c['set']}:{c['dose']}:{c['direction']} KL {c['kl']:.4g} ({c['kl_from']}): collapse "
              f"{c['collapse']:.3f} vs single {c['collapse_ref']:.3f}, d {b['diff']:+.3f} "
              f"[{b['lo']:+.3f}, {b['hi']:+.3f}] {c['verdict']}", flush=True)
    if args.analyse_out:
        p = Path(args.analyse_out).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------------------------------- main
def release_hooked(lm: pl.HookedLM) -> None:
    """Remove a HookedLM's hooks and clear the ``_limbic_fold`` marker its fold pre-hook leaves on each GDN mixer:
    ``gdn.GDNTaps`` reads that marker, so after an ``all``-condition run a later GDN tap (generation) would
    otherwise unfold its signals with the stale window shape."""
    for h in lm.handles:
        h.remove()
    lm.handles = []
    for b in lm.blocks:
        m = pl._child(b, pl.LINEAR_MIXERS)
        if m is not None and getattr(m, "_limbic_fold", None) is not None:
            m._limbic_fold = None


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.analyse:
        return analyse_main(args)
    torch.manual_seed(args.seed)
    out, trace_root = ms.resolve_paths(args, "multisite", args.stage)
    model, tok, blocks, kinds = ms.load_model(args)
    hybrid = ms.is_hybrid(kinds)
    sets = site_sets_of(args, kinds)
    modes = [m for m in args.modes.split(",") if m]
    gains = tx.floats(args.gains)
    dist_gains = tx.floats(args.dist_gains) if args.dist_gains else gains
    doses = [x for x in args.doses.split(",") if x]
    d_model = int(pl.text_config(model).hidden_size)
    facts = ms.model_facts(model, tok, blocks, kinds, args.model if not args.tiny else f"tiny-{args.tiny_arch}")
    batches, task = tx.make_batches(args, tok)
    union = sorted({s for v in sets.values() for s in v})
    # a resume may add site sets: the header keeps the union of the stored and the new ones (analyse reads them)
    old_sets: dict = {}
    if Path(out).exists() and not args.force:
        try:
            old_sets = json.loads(Path(out).read_text()).get("site_sets", {}) or {}
        except (OSError, ValueError):
            old_sets = {}
    all_sets = {**old_sets, **sets}
    all_union = sorted({int(s) for v in all_sets.values() for s in v})
    site_facts = {str(L): {"kind": kinds[L], "attention_below": ms.attention_below(kinds, L),
                           "gdn_below": [l for l, k in enumerate(kinds) if k == "linear" and l <= L]}
                  for L in all_union}
    header = {"runner": "multisite", "stage": args.stage, "name": args.name, "args": vars(args),
              "git": common.git_state(), "versions": ms.versions(), "model": facts, "task": task,
              "site_sets": all_sets, "sites": site_facts, "doses": doses, "gains": gains, "dist_gains": dist_gains,
              "modes": modes, "trace_root": str(trace_root)}
    compat = {k: getattr(args, k) for k in ("model", "tiny", "tiny_arch", "dtype", "attn_impl", "seq_len", "sink",
                                            "window", "steer_from", "betas", "g_abs", "seed", "text_file",
                                            "max_chars", "batch", "attn_window", "random_seed", "signals", "gen_n",
                                            "gen_prompt_len", "gen_new", "gen_temp", "gen_batch", "gen_tf",
                                            "tf_batch")}
    compat["n_rows"] = task["n_rows"]
    res = ms.ResultFile(out, header, compat, force=args.force)
    print(f"[multisite] {facts['name']} sets={sets} modes={modes} out={out}", flush=True)
    if "text" in modes:
        conds = [c for c in args.conditions.split(",") if c]
        skipped = []
        if not hybrid and "all" in conds and "attn" in conds:
            conds.remove("all")
            skipped.append("all: identical to attn on a model without linear-attention layers")
        res.data["conditions_skipped"] = skipped
        arms = make_arms(sets, doses, [x for x in args.directions.split(",") if x], gains, args.random_seed,
                         args.arms, dist_gains)
        for cond in conds:
            lm = pl.HookedLM(model, side=None, layer=union[0], window=args.window, mode="block",
                             window_layers=tx.CONDITIONS[cond], sink=args.sink, pad_id=tok.bos_token_id)
            try:
                res.data.setdefault("conditions", {})[cond] = {
                    "window": 0 if cond == "full" else args.window, "window_layers": tx.CONDITIONS[cond],
                    "check_window": pl.check_window(lm) if cond != "full" else "full attention",
                    "windowed_blocks": lm.windowed}
                res.save()
                for arm in arms:
                    key = f"text/{cond}/{arm['name']}"
                    if res.done(key) and not args.force:
                        print(f"[multisite] skip {key} (complete)", flush=True)
                        continue
                    meta = {"runner": "multisite", "key": key, "arm": arm, "model": facts["name"]}
                    w = ms.ArmTraces(trace_root, f"text-{cond}", trace_name(arm["name"]), meta, args.trace_tokens)
                    rec = run_text_arm(arm, cond, lm, model, blocks, kinds, batches, args, d_model, w)
                    rec["traces"] = w.close()
                    res.put(key, rec)
                    ag = rec["agg"]
                    print(f"[multisite] {key}: dloss {ag['d_loss']:+.4f} kl {ag['kl_base']:.4f} "
                          f"inj {ag['inj_ratio_total']:.3f} norm {ag['norm_ratio_final']:.3f} "
                          f"t {rec['time']['arm_s']:.1f}s", flush=True)
            finally:
                release_hooked(lm)
    if "gen" in modes:
        prompts = ce.gen_prompts(args, tok)
        res.data["gen_prompts"] = {"n": int(len(prompts)), "len": int(prompts.shape[1]),
                                   "rows": "first rows of text.make_batches (ceiling.gen_prompts)",
                                   "seeds": [args.seed + k for k in range(args.gen_seeds)]}
        res.save()
        arms = make_arms(sets, doses, [x for x in args.gen_directions.split(",") if x], gains, args.random_seed,
                         args.arms, dist_gains)
        for seed in [args.seed + k for k in range(args.gen_seeds)]:
            for arm in arms:
                key = f"gen/seed{seed}/{arm['name']}"
                if res.done(key) and not args.force:
                    print(f"[multisite] skip {key} (complete)", flush=True)
                    continue
                meta = {"runner": "multisite", "key": key, "arm": arm, "model": facts["name"], "seed": seed}
                w = ms.ArmTraces(trace_root, f"gen-seed{seed}", trace_name(arm["name"]), meta, args.trace_tokens)
                rec = run_gen_arm(arm, seed, model, blocks, kinds, prompts, args, d_model, tok, w)
                rec["traces"] = w.close()
                res.put(key, rec)
                ag = rec["agg"]
                print(f"[multisite] {key}: collapse {ag['collapse_rate']:.3f} distinct4 {ag['distinct4']:.3f} "
                      f"kl {ag.get('kl_mean')} tok/s {rec['time']['tok_per_s']:.0f}", flush=True)
        gen_summary(res, args)
    return res.data


if __name__ == "__main__":
    main()
