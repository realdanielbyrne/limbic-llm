"""C0 (round 12): offline study of training-free stop rules for thinking, on the traces of an unsteered arm.

Input. One arm directory written by ``reason.py`` (the ``none`` arm of ``reason_base_q35`` / ``reason_base_q3``) or
by ``think.py`` (same layout): ``<arm>/tokens/items.jsonl`` (per item: uid, sample, cks, probe_answers, probe_ok,
n_reason, n_gen, closed, truncated, correct, acc_budget, tok_shard / tok_offset / n_tok; think.py adds probe_conf)
and ``<arm>/tokens/tokens_<shard>.npz`` (per token: ``bk__ids``, ``bk__reason`` (1 on reasoning tokens),
``out__ent`` (entropy of the next-token distribution at that position, T = 1)). Only probed rows are used. No text
is read and nothing generated reaches the output.

Policies. A policy decides at the probe checkpoints c_0 < c_1 < … (every E = 128 reasoning tokens and the
span's last token, ``reason.checkpoints``) from what is known at c_i only, and stops at the first i where it
fires. A stop at c_i yields the forced answer at c_i (the probe: ``</think>`` + "The final answer is \\boxed{",
greedy), which is what a hard stop there gives up to the answer's sampling; it is charged what a real stop
generates (the canonical close and the answer, below). The checkpoint at n_reason of a
span that closed by itself is not a stop: there the natural answer stands. Every policy runs under the same hard
cap as the traces (``--max-new-tokens``, 4,096): a span still open at the cap is stopped there.

Cost, in generated-token equivalents (the critic's rule: probes are not free), the same quantity as think.py's
``tokens_total`` (generated tokens, a forced close and its answer included, + κ per controller probe):
    tokens = n_gen + κ·m                  after a natural close (n_gen = reasoning + close marker + answer + stop)
    tokens = c_stop + κ·m + (L − 1) + A   after a stop at c_stop, m probes run
with κ = ``--probe-cost`` (default 25: the cue, ``</think>`` + 8 tokens prefilled on a fork of the cache, plus 16
greedy tokens), L = ``--close-tokens`` the canonical close's length in tokens (``decode.close_sequence``,
marker included: 25 on Qwen3.5, 24 on Qwen3) and A the row's post-span length: n_gen − n_reason (marker + answer
+ stop) when its span closed by itself, else the median of that over the input's closed rows (``set_tails``).
The answer after a stop is charged as long as the natural one: a stop saves reasoning tokens, not answer tokens.
Rules that decide without probes run none (m = 0; the forced answer stands in for the answer the stop
generates); probe-driven rules run one at every checkpoint up to the stop (m = i + 1), or at every checkpoint
before the close.

Rules (the decision at c_i, ``fire_i``):
  ngram-W-n-ρ     the n-gram loop rule of ``judge.repetition_onset`` (and ``decode.NgramLoopRule``) on the
                  reasoning tokens: onset t* = the first t ≥ W at which the W tokens before t have a repeated
                  n-gram share 1 − distinct/(W − n + 1) > ρ; fire_i = [t* ≤ c_i]
  seen-W-n-ρ      share of the last W reasoning positions whose n-gram already occurred earlier in the span > ρ
                  (the design's "8-gram already seen" loop rule)
  rep-n-k         some n-gram has occurred ≥ k times in the span so far (Pipis et al. 2512.12895: 30-gram × 20)
  ent-W-τ         mean predictive entropy over the last W reasoning positions ≤ τ (a free per-token signal; the
                  reason.py traces have no probe confidence)
  consist-k[-m]   the last k probe answers parse and agree (numbers as numbers, ``decode.answer_key``) and c_i ≥ m
                  (answer consistency; Liu and Wang 2506.02536, Dynasor)
  conf-τ          the probe answer's confidence q ≥ τ (DEER 2504.15895: geometric-mean probability of the answer
                  tokens); only on traces that record ``probe_conf`` (think.py), skipped otherwise
  or(a,b), and(a,b)   combinations of the best settings of two families (chosen on the selection half)
  oracle          the first checkpoint from which every forced answer is correct (``reason.ttc_from``); the first
                  checkpoint when there is none. A ceiling, not a rule
  none            no rule (the trace as generated, capped)
Frontier. Real budget truncation at b ∈ {E, 2E, …, cap}: accuracy ``reason.budget_acc`` (natural answer if the
span closed within b, else the forced answer at b) and tokens n_gen if closed within b, else b + (L − 1) + A. The
frontier's accuracy at a policy's mean tokens T̄ is the linear interpolation between the budgets whose mean
tokens bracket T̄ (``frontier_at``); outside the frontier's range it is undefined (reported as None).

Protocol. Problems are split by a seeded hash (SHA-256 of "<split seed>:<uid>", even = selection, odd =
held-out). On the selection half every setting is scored by Δ = acc − frontier(T̄); the best setting of each
family and the best overall are chosen there and reported on the held-out half with a paired bootstrap over
problems (``--boot`` resamples; each resample recomputes the policy's accuracy and mean tokens, the frontier
curve and the interpolation): Δacc at matched tokens (vertical distance, the pre-registered C0 comparison) and
tokens saved at matched accuracy (horizontal distance, against the frontier's running-maximum envelope). The
number of settings scanned on the selection half is reported.

Usage
  python experiments/ca2/stoprules.py --tiny --out /tmp/x/c0_tiny.json                    # synthetic traces
  python experiments/ca2/stoprules.py --trace ~/limbic-data/ca2_traces/reason_base_q35/none --name c0_q35
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
sys.path.insert(0, str(EXP))
from ca2 import reason as R  # noqa: E402
from ca2.common import _sha256, git_state  # noqa: E402

PROBE_COST = 25
NGRAM_GRID = [(W, n, rho) for W in (32, 64, 128, 256) for n in (4, 8, 16) for rho in (0.3, 0.5, 0.7) if W >= 2 * n]
SEEN_GRID = [(W, n, rho) for W in (128, 256) for n in (8, 16) for rho in (0.3, 0.5, 0.7)]
REP_GRID = [(30, 20), (30, 10), (30, 5), (16, 5)]
ENT_GRID = [(W, tau) for W in (64, 128) for tau in (0.05, 0.1, 0.2, 0.3, 0.5)]
CONSIST_GRID = [(k, m) for k in (2, 3, 4, 5) for m in (0, 512)]
CONF_GRID = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
FAMILIES = ("ngram", "seen", "rep", "ent", "consist", "conf")
PROBE_FAMILIES = ("consist", "conf")


# ------------------------------------------------------------------------------------------------- items
@dataclass
class Trace:
    """One probed (problem, sample) row."""
    uid: str
    sample: int
    cks: list[int]
    answers: list[str | None]
    ok: list[bool]
    n_reason: int
    n_gen: int
    closed: bool                         # the span closed by itself (not truncated, not forced)
    correct: bool                        # natural answer correct (False when truncated)
    span_ids: np.ndarray                 # reasoning token ids, in order
    span_ent: np.ndarray | None          # predictive entropy at those positions
    conf: list[float | None] | None = None
    acc_budget: dict | None = None       # as stored by the runner (for the frontier check)
    keys: list = field(default_factory=list)
    tail: float | None = None            # tokens a stop adds after c_stop: (L − 1) + A (``set_tails``)

    def __post_init__(self) -> None:
        from ca2.decode import answer_key
        self.keys = [answer_key(a) for a in self.answers]


def load_traces(arm_dir: str | Path) -> tuple[list[Trace], dict]:
    """Probed rows of one arm directory (reason.py / think.py layout); returns (traces, provenance)."""
    d = Path(arm_dir).expanduser()
    tdir = d / "tokens" if (d / "tokens" / "items.jsonl").exists() else d
    items = [json.loads(line) for line in (tdir / "items.jsonl").read_text().splitlines() if line.strip()]
    shards: dict[int, dict[str, np.ndarray]] = {}

    def shard(k: int) -> dict[str, np.ndarray]:
        if k not in shards:
            z = np.load(tdir / f"tokens_{k:04d}.npz")
            shards[k] = {"ids": z["bk__ids"], "reason": z["bk__reason"],
                         "ent": z["out__ent"] if "out__ent" in z.files else None}
        return shards[k]

    out = []
    for it in items:
        if not it.get("probed") or not it.get("cks"):
            continue
        s = shard(int(it.get("tok_shard", 0)))
        a, n = int(it["tok_offset"]), int(it["n_tok"])
        rs = s["reason"][a:a + n] > 0
        ids = s["ids"][a:a + n][rs].astype(np.int64)
        ent = s["ent"][a:a + n][rs].astype(np.float32) if s["ent"] is not None else None
        closed = bool(it["closed"]) and not bool(it["truncated"]) and not it.get("forced")
        out.append(Trace(it["uid"], int(it["sample"]), list(it["cks"]), list(it["probe_answers"]),
                         [bool(x) for x in it["probe_ok"]], int(it["n_reason"]), int(it["n_gen"]), closed,
                         bool(it["correct"]), ids, ent, it.get("probe_conf"),
                         {int(k): v for k, v in (it.get("acc_budget") or {}).items()}))
    man = d / "manifest.json"
    prov = {"dir": str(d), "n_items_file": len(items), "n_probed": len(out),
            "manifest_sha256": _sha256(man) if man.exists() else None}
    return out, prov


def synthetic(n: int = 240, seed: int = 0, every: int = 128, cap: int = 4096) -> list[Trace]:
    """Synthetic traces with a planted structure: the answer forms at a random checkpoint and stays correct
    (probes before it give scattered wrong answers); the reasoning goes on overthinking, and in half the rows
    falls into a token loop some time after the answer formed and never closes. Answer consistency (k = 2) is
    then near-optimal; the frontier needs long budgets for the late formers."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        form = rng.choice([1, 1, 2, 3, 4, 6, 9, 14])                   # checkpoint index where the answer forms
        loops = i % 2 == 0
        if loops:
            n_reason, closed = cap, False
            onset = min(cap - 64, (form + 1 + rng.randrange(0, 3)) * every + rng.randrange(0, every))
        else:
            n_reason, closed = min(cap - 1, (form + 4 + rng.randrange(0, 8)) * every + rng.randrange(1, every)), True
            onset = None
        cks = R.checkpoints(n_reason, every)
        answers = [("42" if j >= form else str(100 + rng.randrange(0, 10 ** 6))) for j in range(len(cks))]
        ok = [a == "42" for a in answers]
        ids = [rng.randrange(0, 5000) for _ in range(n_reason)]
        if onset is not None:
            cyc = [rng.randrange(0, 5000) for _ in range(rng.randrange(3, 12))]
            ids[onset:] = (cyc * (n_reason // len(cyc) + 1))[: n_reason - onset]
        ent = np.full(n_reason, 1.5, np.float32)
        n_gen = n_reason + (40 if closed else 0)
        out.append(Trace(f"synthetic/{i}", 0, cks, answers, ok, n_reason, n_gen, closed, closed and ok[-1],
                         np.asarray(ids, np.int64), ent, None, None))
    return out


def set_tails(trs: Sequence[Trace], close_tokens: int) -> float:
    """Set each trace's stop tail (L − 1) + A (module docstring, "Cost"); returns the median A used for rows
    whose span did not close by itself (0 when no row closed)."""
    post = [tr.n_gen - tr.n_reason for tr in trs if tr.closed]
    med = float(np.median(post)) if post else 0.0
    for tr in trs:
        tr.tail = float(close_tokens - 1) + (float(tr.n_gen - tr.n_reason) if tr.closed else med)
    return med


def _tail(tr: Trace) -> float:
    if tr.tail is None:
        raise ValueError(f"{tr.uid}: stop tail not set (stoprules.set_tails)")
    return tr.tail


def split_half(uid: str, seed: int) -> int:
    """0 = selection, 1 = held-out (seeded hash of the problem id)."""
    return int(hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest(), 16) % 2


# ------------------------------------------------------------------------------------------- signals
def gram_ids(ids: np.ndarray, n: int) -> np.ndarray:
    """Integer id per n-gram start position (equal ids ⇔ equal n-grams); length max(0, T − n + 1)."""
    T = len(ids)
    if T < n:
        return np.zeros(0, np.int64)
    table: dict[tuple, int] = {}
    seq = ids.tolist()
    return np.asarray([table.setdefault(tuple(seq[s:s + n]), len(table)) for s in range(T - n + 1)], np.int64)


def window_onset(g: np.ndarray, T: int, W: int, n: int, rho: float) -> int | None:
    """``judge.repetition_onset(ids, n, W, rho)`` in O(T): the first t ≥ W with
    1 − distinct(grams starting in [t − W, t − n]) / (W − n + 1) > rho; None if never (or W < 2n)."""
    if W < 2 * n or T < W:
        return None
    G = W - n + 1
    cnt: dict[int, int] = {}
    for s in range(0, G):
        cnt[int(g[s])] = cnt.get(int(g[s]), 0) + 1
    t = W
    while True:
        if 1.0 - len(cnt) / G > rho:
            return t
        if t >= T:
            return None
        out_g, in_g = int(g[t - W]), int(g[t - n + 1])            # window [t − W + 1, t + 1)
        c = cnt[out_g] - 1
        if c:
            cnt[out_g] = c
        else:
            del cnt[out_g]
        cnt[in_g] = cnt.get(in_g, 0) + 1
        t += 1


def seen_share(g: np.ndarray, T: int, W: int, n: int) -> np.ndarray:
    """share[t] for t = 0..T: of the gram starts in [t − W, t − n] (grams inside the last W tokens), the share
    whose n-gram occurred at an earlier start; NaN while t < W."""
    out = np.full(T + 1, np.nan)
    if len(g) == 0:
        return out
    first: dict[int, int] = {}
    seen = np.zeros(len(g), np.float64)
    for s, x in enumerate(g.tolist()):
        if x in first:
            seen[s] = 1.0
        else:
            first[x] = s
    cs = np.concatenate([[0.0], np.cumsum(seen)])
    G = W - n + 1
    for t in range(W, T + 1):
        out[t] = (cs[t - n + 1] - cs[t - W]) / G
    return out


def rep_onset(g: np.ndarray, n: int, k: int) -> int | None:
    """First t (tokens) at which some n-gram has occurred k times among grams ending by t; None if never."""
    cnt: dict[int, int] = {}
    for s, x in enumerate(g.tolist()):
        c = cnt.get(x, 0) + 1
        cnt[x] = c
        if c >= k:
            return s + n
    return None


# ------------------------------------------------------------------------------------------- rules
@dataclass
class Rule:
    name: str
    family: str
    probes: bool                                            # decides from probes (charged at every checkpoint)
    fire: Callable[[Trace], np.ndarray]                     # (len(cks),) bool, causal


def _at_onset(onset: int | None, cks: Sequence[int]) -> np.ndarray:
    c = np.asarray(cks)
    return np.zeros(len(c), bool) if onset is None else c >= onset


class SignalCache:
    """Per-trace n-gram ids, memoised across settings."""

    def __init__(self) -> None:
        self.g: dict[tuple[int, int], np.ndarray] = {}

    def grams(self, tr: Trace, i: int, n: int) -> np.ndarray:
        k = (i, n)
        if k not in self.g:
            self.g[k] = gram_ids(tr.span_ids, n)
        return self.g[k]


def build_rules(has_conf: bool, has_ent: bool, cache: SignalCache, index: dict[int, int]) -> list[Rule]:
    """Every scanned setting (``index`` maps id(trace) → its position, for the n-gram cache)."""
    rules: list[Rule] = []
    for W, n, rho in NGRAM_GRID:
        rules.append(Rule(f"ngram-{W}-{n}-{rho:g}", "ngram", False, lambda tr, W=W, n=n, rho=rho: _at_onset(
            window_onset(cache.grams(tr, index[id(tr)], n), len(tr.span_ids), W, n, rho), tr.cks)))
    for W, n, rho in SEEN_GRID:
        def f(tr, W=W, n=n, rho=rho):
            sh = seen_share(cache.grams(tr, index[id(tr)], n), len(tr.span_ids), W, n)
            on = np.nonzero(np.nan_to_num(sh, nan=0.0) > rho)[0]
            return _at_onset(int(on[0]) if on.size else None, tr.cks)
        rules.append(Rule(f"seen-{W}-{n}-{rho:g}", "seen", False, f))
    for n, k in REP_GRID:
        rules.append(Rule(f"rep-{n}-{k}", "rep", False, lambda tr, n=n, k=k: _at_onset(
            rep_onset(cache.grams(tr, index[id(tr)], n), n, k), tr.cks)))
    if has_ent:
        for W, tau in ENT_GRID:
            def fe(tr, W=W, tau=tau):
                e = tr.span_ent
                cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(e.astype(np.float64), nan=0.0))])
                out = np.zeros(len(tr.cks), bool)
                for i, c in enumerate(tr.cks):
                    c = min(c, len(e))
                    if c >= W:
                        out[i] = (cs[c] - cs[c - W]) / W <= tau
                return out
            rules.append(Rule(f"ent-{W}-{tau:g}", "ent", False, fe))
    for k, m in CONSIST_GRID:
        def fc(tr, k=k, m=m):
            out = np.zeros(len(tr.cks), bool)
            for i in range(k - 1, len(tr.cks)):
                h = tr.keys[i - k + 1:i + 1]
                out[i] = tr.cks[i] >= m and h[0] is not None and all(x == h[0] for x in h)
            return out
        rules.append(Rule(f"consist-{k}" + (f"-{m}" if m else ""), "consist", True, fc))
    if has_conf:
        for tau in CONF_GRID:
            rules.append(Rule(f"conf-{tau:g}", "conf", True, lambda tr, tau=tau: np.asarray(
                [q is not None and q >= tau for q in tr.conf], bool)))
    return rules


def combine(a: Rule, b: Rule, op: str) -> Rule:
    f = (lambda tr: a.fire(tr) | b.fire(tr)) if op == "or" else (lambda tr: a.fire(tr) & b.fire(tr))
    return Rule(f"{op}({a.name},{b.name})", op, a.probes or b.probes, f)


# ------------------------------------------------------------------------------------------- outcomes
def outcome(tr: Trace, fire: np.ndarray | None, probes: bool, kappa: float) -> tuple[float, float, bool]:
    """(correct, tokens, stopped early) of a policy on one trace (module docstring, "Cost")."""
    n = len(tr.cks)
    last_real = n - 1 if not tr.closed else n - 2                  # the closed span's last checkpoint is no stop
    i = None
    if fire is not None:
        hit = np.nonzero(fire[: last_real + 1])[0]
        i = int(hit[0]) if hit.size else None
    if i is None and not tr.closed:
        i = n - 1                                                  # the cap
    if i is None:                                                  # natural close
        m = (n - 1) if probes else 0
        return float(tr.correct), tr.n_gen + kappa * m, False
    m = (i + 1) if probes else 0
    return float(tr.ok[i]), tr.cks[i] + kappa * m + _tail(tr), True


def oracle_outcome(tr: Trace, kappa: float) -> tuple[float, float, bool]:
    ttc, censored = R.ttc_from(tr.cks, tr.ok)
    i = tr.cks.index(ttc) if not censored else 0
    if tr.closed and i == len(tr.cks) - 1:
        return float(tr.correct), float(tr.n_gen), False
    return float(tr.ok[i]), tr.cks[i] + _tail(tr), True


def frontier_matrix(trs: Sequence[Trace], budgets: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """(acc, tokens), each (N, n_budgets): real truncation at each budget (``reason.budget_acc``); a truncated
    row is charged its close and answer like a stop (no probe: the truncation decides from the count alone)."""
    acc = np.full((len(trs), len(budgets)), np.nan)
    tok = np.full((len(trs), len(budgets)), np.nan)
    for r, tr in enumerate(trs):
        ba = R.budget_acc(budgets, tr.cks, tr.ok, tr.n_reason, tr.closed, tr.correct)
        for j, b in enumerate(budgets):
            acc[r, j] = np.nan if ba[b] is None else ba[b]
            tok[r, j] = tr.n_gen if (tr.closed and tr.n_reason <= b) else min(b, tr.n_reason) + _tail(tr)
    return acc, tok


def frontier_at(f_acc: np.ndarray, f_tok: np.ndarray, t: float) -> float | None:
    """Frontier accuracy at mean tokens t: linear between the budgets whose mean tokens bracket t (sorted by
    tokens); None outside the frontier's range."""
    o = np.argsort(f_tok, kind="stable")
    x, y = f_tok[o], f_acc[o]
    if not (x[0] <= t <= x[-1]):
        return None
    return float(np.interp(t, x, y))


def frontier_tokens_at(f_acc: np.ndarray, f_tok: np.ndarray, a: float) -> float | None:
    """Frontier tokens needed for accuracy a, on its running-maximum envelope; None if never reached."""
    o = np.argsort(f_tok, kind="stable")
    x, y = f_tok[o], np.maximum.accumulate(f_acc[o])
    if a > y[-1]:
        return None
    if a <= y[0]:
        return float(x[0])
    j = int(np.nonzero(y >= a)[0][0])
    if y[j] == y[j - 1]:
        return float(x[j])
    return float(x[j - 1] + (a - y[j - 1]) * (x[j] - x[j - 1]) / (y[j] - y[j - 1]))


def score(cor: np.ndarray, tok: np.ndarray, f_acc: np.ndarray, f_tok: np.ndarray) -> dict:
    """Point estimates of a policy against the frontier on the same rows (budget means over rows with a value)."""
    fa, ft = np.nanmean(f_acc, 0), np.nanmean(f_tok, 0)
    acc, t = float(cor.mean()), float(tok.mean())
    fat = frontier_at(fa, ft, t)
    ftk = frontier_tokens_at(fa, ft, acc)
    return {"acc": acc, "tokens": t, "frontier_acc": fat, "d_acc": None if fat is None else acc - fat,
            "frontier_tokens": ftk, "tokens_saved": None if ftk is None else ftk - t,
            "tokens_saved_frac": None if ftk is None or ftk <= 0 else (ftk - t) / ftk}


def boot(cor: np.ndarray, tok: np.ndarray, f_acc: np.ndarray, f_tok: np.ndarray, clusters: Sequence[str],
         n: int, seed: int) -> dict:
    """Paired cluster bootstrap of Δacc at matched tokens and tokens saved at matched accuracy (each resample
    recomputes the policy, the frontier curve and the interpolation)."""
    keys, inv = np.unique(np.asarray(clusters), return_inverse=True)
    C = len(keys)
    # per-cluster sums, so a resample of clusters is a weighted mean
    S = lambda v: np.bincount(inv, weights=v, minlength=C)
    cnt = np.bincount(inv, minlength=C).astype(np.float64)
    sc, st = S(cor), S(tok)
    fa = np.stack([S(np.nan_to_num(f_acc[:, j])) for j in range(f_acc.shape[1])], 1)
    fn = np.stack([S(np.isfinite(f_acc[:, j]).astype(np.float64)) for j in range(f_acc.shape[1])], 1)
    ft = np.stack([S(f_tok[:, j]) for j in range(f_tok.shape[1])], 1)
    rng = np.random.default_rng(seed)
    d_acc, saved = [], []
    for _ in range(n):
        w = np.bincount(rng.integers(0, C, C), minlength=C).astype(np.float64)
        N = (w * cnt).sum()
        a, t = (w * sc).sum() / N, (w * st).sum() / N
        fam = (w[:, None] * fa).sum(0) / np.maximum((w[:, None] * fn).sum(0), 1e-12)
        ftm = (w[:, None] * ft).sum(0) / N
        x = frontier_at(fam, ftm, t)
        d_acc.append(np.nan if x is None else a - x)
        y = frontier_tokens_at(fam, ftm, a)
        saved.append(np.nan if y is None else y - t)

    def ci(v):
        v = np.asarray(v, np.float64)
        ok = v[np.isfinite(v)]
        if ok.size < max(10, n // 10):
            return {"lo": None, "hi": None, "n_valid": int(ok.size), "p_le_0": None}
        lo, hi = np.quantile(ok, [0.025, 0.975])
        return {"lo": float(lo), "hi": float(hi), "n_valid": int(ok.size), "p_le_0": float((ok <= 0).mean())}
    return {"d_acc": ci(d_acc), "tokens_saved": ci(saved), "n_clusters": C, "n_boot": n}


# ------------------------------------------------------------------------------------------- study
def evaluate(trs: Sequence[Trace], rule: Rule | None, kappa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    res = [outcome(tr, rule.fire(tr) if rule else None, bool(rule and rule.probes), kappa) for tr in trs]
    a = np.asarray(res, np.float64)
    return a[:, 0], a[:, 1], a[:, 2]


def study(trs: list[Trace], args) -> dict:
    """The whole C0 analysis on a list of traces (module docstring)."""
    t0 = time.time()
    kappa = float(args.probe_cost)
    E = int(args.probe_every)
    cap = int(args.max_new_tokens)
    budgets = list(range(E, cap + 1, E))
    half = np.asarray([split_half(tr.uid, args.split_seed) for tr in trs])
    sel = [tr for tr, h in zip(trs, half) if h == 0]
    held = [tr for tr, h in zip(trs, half) if h == 1]
    if not sel or not held:
        raise SystemExit("both halves need rows")
    has_conf = all(tr.conf is not None and len(tr.conf) == len(tr.cks) for tr in trs)
    has_ent = all(tr.span_ent is not None and len(tr.span_ent) == len(tr.span_ids) for tr in trs)
    cache = SignalCache()
    index = {id(tr): i for i, tr in enumerate(trs)}
    med_post = set_tails(trs, int(args.close_tokens))
    rules = build_rules(has_conf, has_ent, cache, index)
    # frontier check: the simulator reproduces the runner's stored acc@b
    mism = 0
    for tr in trs:
        if tr.acc_budget:
            ba = R.budget_acc(sorted(tr.acc_budget), tr.cks, tr.ok, tr.n_reason, tr.closed, tr.correct)
            mism += sum(1 for b, v in tr.acc_budget.items() if v != ba[b])
    S_acc, S_tok = frontier_matrix(sel, budgets)
    H_acc, H_tok = frontier_matrix(held, budgets)

    def frontier_rows(fa, ft):
        return [{"budget": b, "acc": float(np.nanmean(fa[:, j])), "tokens": float(np.nanmean(ft[:, j]))}
                for j, b in enumerate(budgets)]
    # selection half: every setting
    scanned = {}
    for r in rules:
        c, t, s = evaluate(sel, r, kappa)
        scanned[r.name] = {"family": r.family, "probes": r.probes, "fire_rate": float(s.mean()),
                           **score(c, t, S_acc, S_tok)}
    key = lambda name: (-math.inf if scanned[name]["d_acc"] is None else scanned[name]["d_acc"],
                        -scanned[name]["tokens"])
    best_of = {}
    for fam in FAMILIES:
        names = [n for n in scanned if scanned[n]["family"] == fam]
        if names:
            best_of[fam] = max(names, key=key)
    by_name = {r.name: r for r in rules}
    # combinations of the top-3 settings of two families, chosen on the selection half
    top = {fam: sorted([n for n in scanned if scanned[n]["family"] == fam], key=key, reverse=True)[:3]
           for fam in FAMILIES}
    pairs = [("or", "ngram", "consist"), ("or", "seen", "consist"), ("and", "consist", "ent"),
             ("and", "consist", "conf"), ("or", "ngram", "conf")]
    combos = []
    for op, fa_, fb_ in pairs:
        for a_ in top.get(fa_, []):
            for b_ in top.get(fb_, []):
                combos.append(combine(by_name[a_], by_name[b_], op))
    for r in combos:
        c, t, s = evaluate(sel, r, kappa)
        scanned[r.name] = {"family": r.family, "probes": r.probes, "fire_rate": float(s.mean()),
                           **score(c, t, S_acc, S_tok)}
        by_name[r.name] = r
    for fam in ("or", "and"):
        names = [n for n in scanned if scanned[n]["family"] == fam]
        if names:
            best_of[fam] = max(names, key=key)
    best = max([n for n in scanned], key=key)
    # held-out half: the chosen settings, the oracle and no rule
    cl = [tr.uid for tr in held]
    report_rows = {}

    def held_row(label, c, t, s, extra=None):
        row = {"acc": float(c.mean()), "tokens": float(t.mean()), "stop_rate": float(s.mean()),
               **score(c, t, H_acc, H_tok), "boot": boot(c, t, H_acc, H_tok, cl, args.boot, args.seed)}
        row.update(extra or {})
        report_rows[label] = row
    for fam, name in sorted(best_of.items()):
        c, t, s = evaluate(held, by_name[name], kappa)
        held_row(f"best_{fam}", c, t, s, {"setting": name, "probes": by_name[name].probes})
    c, t, s = evaluate(held, by_name[best], kappa)
    held_row("best_overall", c, t, s, {"setting": best, "probes": by_name[best].probes})
    c, t, s = evaluate(held, None, kappa)
    held_row("none", c, t, s)
    o = np.asarray([oracle_outcome(tr, kappa) for tr in held], np.float64)
    held_row("oracle", o[:, 0], o[:, 1], o[:, 2])
    return {"n_rows": len(trs), "n_problems": len({tr.uid for tr in trs}), "n_selection": len(sel),
            "n_heldout": len(held), "probe_cost": kappa, "probe_every": E, "cap": cap, "budgets": budgets,
            "close_tokens": int(args.close_tokens), "median_post_close": med_post,
            "cost_rule": "natural close n_gen + κ·m; stop or truncation c + κ·m + (L − 1) + A (A: the row's "
                         "n_gen − n_reason if it closed, else the median over closed rows); m = 0 without probes",
            "has_probe_conf": has_conf, "has_entropy": has_ent,
            "frontier_check_mismatches": mism,
            "n_settings_scanned": len(scanned), "n_settings_by_family": {
                f: sum(1 for v in scanned.values() if v["family"] == f) for f in FAMILIES + ("or", "and")},
            "selection_objective": "d_acc = acc - frontier accuracy at the setting's mean tokens (ties: fewer tokens)",
            "frontier": {"selection": frontier_rows(S_acc, S_tok), "heldout": frontier_rows(H_acc, H_tok)},
            "chosen": {**best_of, "overall": best}, "selection": scanned, "heldout": report_rows,
            "seconds": time.time() - t0}


def table(res: dict) -> str:
    f = lambda x, p=3: "  -  " if x is None else f"{x:.{p}f}"
    lines = [f"held-out ({res['n_heldout']} rows; selection {res['n_selection']}; {res['n_settings_scanned']} "
             f"settings scanned; probe cost {res['probe_cost']:g} tok/probe)",
             f"{'policy':<14}{'setting':<44}{'acc':>7}{'tokens':>9}{'front@T':>9}{'Δacc':>8}{'95% CI':>18}"
             f"{'saved':>8}"]
    for k, r in res["heldout"].items():
        b = r["boot"]["d_acc"]
        ci = "" if b["lo"] is None else f"[{b['lo']:+.3f},{b['hi']:+.3f}]"
        d = "  -  " if r["d_acc"] is None else "%+.3f" % r["d_acc"]
        lines.append(f"{k:<14}{str(r.get('setting', '')):<44}{r['acc']:>7.3f}{r['tokens']:>9.0f}"
                     f"{f(r['frontier_acc']):>9}{d:>8}{ci:>18}{f(r['tokens_saved'], 0):>8}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = ap.add_argument
    a("--trace", default=None, help="arm directory of a reason.py / think.py run (e.g. .../reason_base_q35/none)")
    a("--tiny", action="store_true", help="synthetic traces with a planted structure (tests)")
    a("--tiny-n", type=int, default=240)
    a("--probe-cost", type=float, default=PROBE_COST, help="token equivalents per probe")
    a("--probe-every", type=int, default=128)
    a("--close-tokens", type=int, default=25,
      help="canonical close length in tokens, marker included (decode.close_sequence: 25 Qwen3.5, 24 Qwen3)")
    a("--max-new-tokens", type=int, default=4096, help="the traces' cap (frontier budgets up to it)")
    a("--split-seed", type=int, default=0)
    a("--boot", type=int, default=2000)
    a("--seed", type=int, default=0)
    a("--name", default=None)
    a("--out", default=None, help="results JSON (default results/evolve/round12/stage1/<name>.json)")
    return ap


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if not args.tiny and not args.trace:
        raise SystemExit("--trace <arm dir> or --tiny")
    name = args.name or ("c0_tiny" if args.tiny else "c0_" + Path(args.trace).expanduser().parent.name)
    out = Path(args.out or ROOT / "results" / "evolve" / "round12" / "stage1" / f"{name}.json")
    if args.tiny:
        trs, prov = synthetic(args.tiny_n, args.seed, args.probe_every, args.max_new_tokens), {"synthetic": True}
    else:
        trs, prov = load_traces(args.trace)
    res = study(trs, args)
    report = {"args": vars(args), "git": git_state(), "input": prov, **res}
    out.parent.mkdir(parents=True, exist_ok=True)
    R.write_json(out, report)
    print(table(res), flush=True)
    print(f"saved {out}", flush=True)
    return report


if __name__ == "__main__":
    main()
