"""Read a round's evaluations by the pre-registered rules (round-11 plan, "How results are read"; DEC11).

Inputs are ``eval_by_query.py`` JSONs, one per (run, cell). ``--dir D`` scans D for ``<run>_<cell>.json``, the
cell being the last ``_`` piece (``e11_r11_A11_s1_nat512p3.json`` → run ``e11_r11_A11_s1``, cell ``nat512p3``);
files without ``per_sequence`` (training JSONs) and names with a dot are skipped. ``--run NAME=TEMPLATE``
names a run by a path template with ``{cell}`` (repeatable; several templates for one name are tried in order,
cell by cell, and replace a discovered run of that name). A run's seed is the last ``_s<digits>`` in its name.
Round-10 Stage-B's ``update`` cell (update new, 512 tokens, k 1) is read as ``updnew512``.

Every evaluation whose ``windowed`` or ``side_empty`` exact is not 0 is void: listed, and left out of every
statistic. A missing control is not clean: without a ``windowed`` control (``--no-controls``) the cell is
unverified, listed and left out like a void one; without a ``side_empty`` one (every evaluation made before
``--ctrl-conds``, the round-10 references among them) it is listed and kept.

Evaluation-time modifications (``--w1s-param``, ``--w1s-set``, ``--w1s-ctrl``, ``--w1s-load-extra``,
``--keep-oracle``, the oracle gate, ``--attn-impl``) are recorded per cell and reported per run; two cells whose
modifications differ are not paired unless ``--allow-modified`` (Phase 0's E-lex / E-ord pair a modified
evaluation with the plain checkpoint on purpose).

Pairing (strong check). Two cells are paired only when their ``per_sequence`` rank and gap agree row for row
(and n, the eval seed, the task, the length and pairs, the filler file's name and, where both have it,
``per_sequence_repeat``); otherwise the pair is refused, with the reason, and the cell is left out. A
reference made of several runs (the mechanism column) is their row-by-row mean, each run checked so.

Groups. The statistic is the mean of the per-cell paired differences. Cells whose sequences are the same rows
are averaged row by row first, so their correlation enters the interval; the weights keep the statistic the
mean over cells. Same rows: identical (rank, gap) fingerprints (held-out, nat512p3 and para1–4); (rank, gap)
equal on ≥ 95% of rows (the overlap is reported); or update cells of one draw (updsame/updnow/updnew at one
length: the same names, codes and order, gaps shifted by the stem's length). A group of one 0/1 cell (names)
decides near the edge on Newcombe's interval, as the cell does.
  grid           the 18 DEC2 cells (nat/rand × 512/1024/2048 × 1/3/5 codes)
  names          held-out names
  query          para1, 2 and 4 (para3 is out: full attention 0.44 < 0.8); jitter is reported beside it with
                 its full-attention ceiling
  updates_new    new-code exact on update same/now/new at 512 and 2048, update-k 2 and 3
  updates_stale  the stale (old-code) exact on the same cells, lower is better (``per_sequence_stale``; an
                 evaluation made before that existed gives only the unpaired cell-level difference)
  robustness     rand4096p8/16/24, rand8192p32, decoys20 (n = 96): reported only
  pick           (recipe level and factorial only) the mean of the grid, names and query statistics: the
                 Stage-B ranking
  gate           (reported only) per run, from the ``w1s`` block: fact coverage, onset-in-fact, gate AUC and
                 onset density, on natural against random grid cells, jitter, the paraphrases and held-out
                 names, over the cells both sides of a comparison have; and the late-drop step, from the run's
                 training JSON (the evaluation's ``result``): the first logged step from which a trailing median
                 of the logged loss stays ≤ ``--drop-loss`` (``late_drop``; ``never``, or ``too few points``
                 below ``--drop-min-points`` = 30 logged points: runs logged every steps/10 read no step)

Intervals.
  checkpoint level   the per-sequence paired interval, mean ± 1.96·sd/√n (two trained checkpoints). Near
                     the ceiling or floor (either side's level ≥ ``--near`` or ≤ 1 − ``--near``) ``--interval
                     auto`` decides on the alternative instead: Newcombe's paired score interval (method 10,
                     Wilson limits) for a single 0/1 cell, a bootstrap over rows (within each shared-row
                     cluster) for groups and averaged references. Every interval is reported.
  recipe level       d̄ ± half: d̄ the difference of the recipes' mean group levels, from repeated runs of fixed
                     recipes (``--noise NAME=RUN,RUN,...``, repeatable). The pre-registered half-width is
                     t·s_seed·√(1/k_A + 1/k_B), s_seed the pooled between-run SD (always reported, ``between``).
                     On shared seeds the seed effect cancels in d̄ but not in s_seed, so with ``--s-seed auto``
                     (default) and same-seed replicates in the noise sets (df ≥ ``--min-within-df`` = 3; e.g. a
                     run and its ``run.py --init-seed`` twin) the verdict uses σ_run from within-seed residuals
                     (equal seed compositions) or Var(d̄) = c·σ_seed² + (1/k_A + 1/k_B)·σ_run²
                     (``recipe_interval``).
                     A recipe is better or worse only where the interval clears 0 and the sign holds on every
                     shared seed. Stage-B pick: an Updates verdict that is missing (void or absent cells, no
                     --noise) is ``unknown``, never "not worse".

Reports (any combination):
  --arms REGEX ... --ref TEMPLATE ...   every matching run against the same-seed reference (``{seed}`` in the
                                        template: ``e11ref_r10_I1_s{seed}``); one column per template
  seed 4 (``--rescue-seed``)            two more columns for those arms: rescue, against ``--rescue-ref`` (I1
                                        seed 4), and mechanism, against the row mean of ``--mechanism-refs`` (I1's
                                        learned seeds 0, 2 and 3, round-10 Stage B). Only the second is evidence
                                        of mechanism
  --compare A B                         any two runs (arm against arm), every cell they share
  --recipe NAME=RUN,... --recipe-ref R  recipe-level comparisons of every recipe against R, ranked on pick, with
                                        the Stage-B default (the top recipe; if it is worse on updates, also the
                                        best one that is not)
  --design CSV                          factorial main effects: a CSV with a ``run`` column, an optional ``seed``
                                        column and one column per factor (0/1, -/+, off/on). The design is checked
                                        first (balance, orthogonality, resolution and aliases; below III refused,
                                        below IV warned). Each effect is the least-squares main effect with seed
                                        blocks (the mean over seeds of mean on − mean off when nothing is missing);
                                        intervals from the model's residual, and from the noise when ``--noise`` is
                                        given; missing runs are refit, and effects they leave unestimable named

    python experiments/evolve/analyse_round.py --dir results/evolve/round11/eval --dir results/evolve/round10/stageB \\
        --arms 'e11_r11_.*' --ref 'e11ref_r10_I1_s{seed}' --out summary.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

GRID = [f"{f}{L}p{p}" for f in ("nat", "rand") for L in (512, 1024, 2048) for p in (1, 3, 5)]
UPDATES = ["updsame512", "updnow512", "updnew512", "updsame2048", "updnow2048", "updnew2048", "updk2", "updk3"]
ROBUSTNESS = ["rand4096p8", "rand4096p16", "rand4096p24", "rand8192p32", "decoys20"]
GROUPS = {
    "grid": {"cells": GRID, "metric": "exact"},
    "names": {"cells": ["heldout"], "metric": "exact"},
    "query": {"cells": ["para1", "para2", "para4"], "metric": "exact"},
    "updates_new": {"cells": UPDATES, "metric": "exact"},
    "updates_stale": {"cells": UPDATES, "metric": "stale", "lower_better": True},
    "robustness": {"cells": ROBUSTNESS, "metric": "exact", "reported_only": True},
}
PICK = ("grid", "names", "query")                                   # the Stage-B ranking statistic: their mean
BESIDE = {"query": ["jitter"]}                                      # reported alongside a group, not in it
GATE_KEYS = ("fact_coverage", "onset_in_fact", "gate_auc_digit1_vs_filler", "onset_density")
GATE_SETS = {"natural": [c for c in GRID if c.startswith("nat")], "random": [c for c in GRID if c.startswith("rand")],
             "jitter": ["jitter"], "query": GROUPS["query"]["cells"], "names": ["heldout"]}
Z = 1.959963984540054
STEM = re.compile(r"^(?P<run>[^.]+)_(?P<cell>[A-Za-z0-9]+)$")


# ----------------------------------------------------------------------------------------------- statistics
def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def var(v) -> float:
    """Sample variance (n − 1); 0 for fewer than two values."""
    n = len(v)
    if n < 2:
        return 0.0
    m = sum(v) / n
    return sum((x - m) ** 2 for x in v) / (n - 1)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularised incomplete beta (modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    p = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - p if t > 0 else p


def t_quantile(q: float, df: float) -> float:
    """Student-t quantile by bisection on the CDF (no scipy in the repo's environment)."""
    if df <= 0:
        return float("inf")
    if q == 0.5:
        return 0.0
    if q < 0.5:
        return -t_quantile(1.0 - q, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < q:
        hi *= 2.0
        if hi > 1e8:
            return float("inf")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval of a proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_paired(a: list[float], b: list[float], z: float = Z) -> tuple[float, float]:
    """Newcombe (1998) method 10: score interval for the difference of two paired proportions mean(a) − mean(b),
    from Wilson limits of the two marginals and the continuity-corrected φ of the 2×2 table."""
    n = len(a)
    A = sum(1 for x, y in zip(a, b) if x and y)
    B = sum(1 for x, y in zip(a, b) if x and not y)
    C = sum(1 for x, y in zip(a, b) if y and not x)
    D = n - A - B - C
    p1, p2 = (A + B) / n, (A + C) / n
    l1, u1 = wilson(A + B, n, z)
    l2, u2 = wilson(A + C, n, z)
    den = (A + B) * (C + D) * (A + C) * (B + D)
    if den == 0:
        phi = 0.0
    elif A * D > B * C:
        phi = max(A * D - B * C - n / 2, 0.0) / math.sqrt(den)
    else:
        phi = (A * D - B * C) / math.sqrt(den)
    theta = p1 - p2
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return max(-1.0, theta - dl), min(1.0, theta + du)


def percentile(sorted_v: list[float], q: float) -> float:
    """Linear-interpolated percentile of an ascending list."""
    if not sorted_v:
        return float("nan")
    pos = q * (len(sorted_v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (pos - lo)


def verdict(lo: float, hi: float, lower_better: bool = False) -> str:
    if lo > 0:
        return "worse" if lower_better else "better"
    if hi < 0:
        return "better" if lower_better else "worse"
    return "tie"


def solve(M: list[list[float]], ys: list[list[float]]) -> list[list[float]] | None:
    """Gauss–Jordan: M⁻¹·[ys...] for a small dense system; None if singular."""
    n = len(M)
    A = [row[:] + [y[i] for y in ys] for i, row in enumerate(M)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-10:
            return None
        A[c], A[p] = A[p], A[c]
        piv = A[c][c]
        A[c] = [x / piv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0.0:
                f = A[r][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return [[A[i][n + j] for i in range(n)] for j in range(len(ys))]


# ------------------------------------------------------------------------------------------------ loading
def fingerprint(rows: list[tuple[int, int]]) -> str:
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()[:12]


def cell_alias(cell: str, c: dict) -> str:
    """Round-10 Stage-B's ``update`` cell is round 11's ``updnew512`` when its settings say so."""
    t = c["task"]
    if cell == "update" and t.get("update") == "new" and c["seq_len"] == 512 and "update_k" not in t:
        return "updnew512"
    return cell


def load_cell(path: str) -> dict | None:
    """An eval_by_query JSON → the fields the analysis reads; None if it has no per_sequence."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    ps = d.get("per_sequence") if isinstance(d, dict) else None
    if not ps:
        return None
    ctrl = d.get("controls") or {}
    upd = (d.get("update") or {}).get("windowed+side")
    task = {k: v for k, v in (d.get("task") or {}).items() if k != "trained_names"}   # a training fact, not the cell
    rows = [(int(r[0]), int(r[1])) for r in ps]
    stale = d.get("per_sequence_stale")
    return {"path": str(path), "n": len(ps), "rows": rows, "fp": fingerprint(rows),
            "exact": [float(r[3]) for r in ps], "stale": [float(x) for x in stale] if stale is not None else None,
            "repeat": d.get("per_sequence_repeat"), "seed": (d.get("eval") or {}).get("seed"),
            "task": task, "seq_len": d.get("seq_len"), "pairs": d.get("pairs"),
            "filler": Path(d["text_file"]).name if d.get("text_file") else None,
            "level_exact": float(d["exact"]), "level_stale": float(upd["old_exact"]) if upd else None,
            "full": ctrl.get("full", {}).get("exact"), "windowed": ctrl.get("windowed", {}).get("exact"),
            "side_empty": ctrl.get("side_empty", {}).get("exact"),
            "full_stale": ((d.get("update") or {}).get("full") or {}).get("old_exact"),
            "w1s": d.get("w1s"), "result": d.get("result"), "raw_cell": None,
            "mods": {k: d[k] for k in MODS if k in d}}


MODS = ("w1s_param", "w1s_set", "w1s_ctrl", "w1s_load_extra", "keep_oracle", "oracle_gate", "attn_impl")


def draw_key(c: dict) -> str | None:
    """Update cells (``task.update`` set) that differ only in the update's phrasing draw the same names, codes and
    order from the same eval seed (run.py: the forms take the same random numbers), so their rows are the same
    sequences up to the stem's length: the key they share. None for other cells."""
    t = c["task"]
    if not t.get("update"):
        return None
    rest = {k: v for k, v in t.items() if k != "update"}
    return fingerprint([c["n"], c["seed"], c["seq_len"], c["pairs"], c["filler"], sorted(rest.items())])


def is_void(c: dict) -> bool:
    """A control that is present and not 0. A missing control is not clean: see ``unverified``."""
    return bool(c["windowed"]) or bool(c["side_empty"])


def unverified(c: dict) -> bool:
    """No windowed control (``--no-controls``): the E-control rule cannot be checked, so the cell is left out like
    a void one (listed apart). A missing side_empty alone is listed (``controls_unknown``) but kept: every
    evaluation made before ``--ctrl-conds`` lacks it (round-10 Stage B, the references)."""
    return c["windowed"] is None


def excluded(c: dict) -> bool:
    return c["void"] or c.get("unverified", False)


def discover(dirs: list[str]) -> dict[str, dict[str, str]]:
    """{run: {cell: path}} from every ``<run>_<cell>.json`` in ``dirs`` (paths only; read later)."""
    runs: dict[str, dict[str, str]] = {}
    for d in dirs:
        for p in sorted(glob.glob(str(Path(d) / "*.json"))):
            m = STEM.match(Path(p).stem)
            if not m:
                continue
            prev = runs.setdefault(m["run"], {}).get(m["cell"])
            if prev is not None and Path(prev).resolve() != Path(p).resolve():
                raise SystemExit(f"run {m['run']} cell {m['cell']} found twice: {prev} and {p}")
            runs[m["run"]][m["cell"]] = p
    return runs


def from_templates(templates: list[str]) -> dict[str, str]:
    """{cell: path} from ``{cell}`` path templates, the first template winning per cell."""
    out: dict[str, str] = {}
    for t in templates:
        if "{cell}" not in t:
            raise SystemExit(f"--run template {t!r} needs {{cell}}")
        pre, post = t.split("{cell}", 1)
        rx = re.compile("^" + re.escape(pre) + r"(?P<cell>[A-Za-z0-9]+)" + re.escape(post) + "$")
        for p in sorted(glob.glob(pre + "*" + post)):
            m = rx.match(p)
            if m and m["cell"] not in out:
                out[m["cell"]] = p
    return out


def load_runs(paths: dict[str, dict[str, str]],
              report: dict | None = None) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """Read every cell; aliases applied; returns ({run: {cell: data}}, void list). ``report`` (if given) gets the
    cells without a windowed control (``unverified``, left out), without a side_empty one (``controls_unknown``,
    kept) and each run's evaluation-time modifications (``modified``: {run: {cell: mods}})."""
    report = {} if report is None else report
    for k in ("unverified", "controls_unknown"):
        report.setdefault(k, [])
    report.setdefault("modified", {})
    runs, voids = {}, []
    for run, cells in paths.items():
        got = {}
        for cell, p in cells.items():
            c = load_cell(p)
            if c is None:
                continue
            name = cell_alias(cell, c)
            if name in got and name != cell:                     # an explicit updnew512 wins over the alias
                continue
            c["raw_cell"] = cell
            c["void"] = is_void(c)
            c["unverified"] = unverified(c)
            c["draw"] = draw_key(c)
            if c["void"]:
                voids.append(f"{run}/{name}")
            if c["unverified"]:
                report["unverified"].append(f"{run}/{name}")
            if c["side_empty"] is None:
                report["controls_unknown"].append(f"{run}/{name}")
            if c["mods"]:
                report["modified"].setdefault(run, {})[name] = c["mods"]
            got[name] = c
        if got:
            runs[run] = got
    return runs, voids


def run_seed(name: str) -> int | None:
    m = re.findall(r"_s(\d+)", name)
    return int(m[-1]) if m else None


# ------------------------------------------------------------------------------------------------ pairing
def pair_check(a: dict, b: dict, allow_modified: bool = False) -> str | None:
    """Why two cells cannot be paired row by row, or None (they draw the same sequences). Cells whose
    evaluation-time modifications (``MODS``: --w1s-param, --w1s-set, --w1s-ctrl, ...) differ are refused unless
    ``allow_modified`` (``--allow-modified``; Phase 0's E-lex / E-ord pair the modified checkpoint with the plain
    one on purpose)."""
    if a["n"] != b["n"]:
        return f"n {a['n']} vs {b['n']}"
    for k in ("seed", "seq_len", "pairs", "filler"):
        if a[k] is not None and b[k] is not None and a[k] != b[k]:
            return f"{k} {a[k]} vs {b[k]}"
    if a["task"] != b["task"]:
        return f"task {a['task']} vs {b['task']}"
    for i, (x, y) in enumerate(zip(a["rows"], b["rows"])):
        if x != y:
            return f"row {i}: (rank, gap) {list(x)} vs {list(y)}"
    if a["repeat"] is not None and b["repeat"] is not None and a["repeat"] != b["repeat"]:
        return "per_sequence_repeat differs"
    ma, mb = a.get("mods") or {}, b.get("mods") or {}
    if ma != mb and not allow_modified:
        return f"eval-time modifications differ: {json.dumps(ma)} vs {json.dumps(mb)} (--allow-modified pairs them)"
    return None


def values(c: dict, metric: str) -> list[float] | None:
    return c["exact"] if metric == "exact" else c["stale"]


def paired_cell(a: dict, refs: list[dict], metric: str, allow_modified: bool = False) -> dict:
    """Per-row differences of ``a`` against the row mean of ``refs``, or a refusal with its reason."""
    for r in refs:
        why = pair_check(a, r, allow_modified)
        if why:
            return {"refused": why, "ref": r["path"]}
    va, vr = values(a, metric), [values(r, metric) for r in refs]
    if va is None or any(v is None for v in vr):
        out = {"refused": f"no per-sequence {metric} (evaluated before per_sequence_stale)"}
        la, lr = a[f"level_{metric}"], [r[f"level_{metric}"] for r in refs]
        if la is not None and None not in lr:
            out["unpaired_diff"] = la - sum(lr) / len(lr)
        return out
    ref = [sum(col) / len(col) for col in zip(*vr)]
    binary = len(refs) == 1 and all(x in (0.0, 1.0) for x in va + ref)
    return {"d": [x - y for x, y in zip(va, ref)], "a": va, "b": ref, "fp": a["fp"], "binary": binary,
            "rows": a["rows"], "draw": a.get("draw")}


def near(level_a: float, level_b: float, edge: float) -> bool:
    return any(x >= edge or x <= 1.0 - edge for x in (level_a, level_b))


def cell_stats(p: dict, opts, lower_better: bool = False) -> dict:
    """The checkpoint-level interval of one paired cell (normal, plus Newcombe for 0/1 cells or a bootstrap)."""
    d, n = p["d"], len(p["d"])
    m = sum(d) / n
    half = Z * math.sqrt(var(d) / n) if n > 1 else float("inf")
    la, lb = sum(p["a"]) / n, sum(p["b"]) / n
    out = {"diff": m, "a": la, "b": lb, "n": n, "normal": [m - half, m + half]}
    if p["binary"]:
        out["newcombe"] = list(newcombe_paired(p["a"], p["b"]))
    elif opts.bootstrap:
        out["bootstrap"] = bootstrap([(1.0, [d])], opts)
    out.update(decide(out, near(la, lb, opts.near), opts, lower_better, alt="newcombe" if p["binary"] else "bootstrap"))
    return out


def decide(st: dict, is_near: bool, opts, lower_better: bool, alt: str) -> dict:
    method = "normal"
    if opts.interval == "alt" or (opts.interval == "auto" and is_near):
        method = alt if alt in st else "normal"
    lo, hi = st[method]
    return {"near_edge": is_near, "method": method, "lo": lo, "hi": hi, "verdict": verdict(lo, hi, lower_better)}


def bootstrap(clusters: list[tuple[float, list[list[float]]]], opts) -> list[float]:
    """Percentile interval of Σ w·mean(row means) with rows resampled within each cluster (seeded).
    ``clusters``: (weight, [per-cell row vectors of equal length])."""
    rng = random.Random(opts.boot_seed)
    rows = [(w, [sum(col) / len(col) for col in zip(*vecs)]) for w, vecs in clusters]
    stats = []
    for _ in range(opts.bootstrap):
        stats.append(sum(w * sum(rng.choices(r, k=len(r))) / len(r) for w, r in rows))
    stats.sort()
    return [percentile(stats, 0.025), percentile(stats, 0.975)]


ROW_OVERLAP = 0.95                                                  # share of equal (rank, gap) rows: same rows


def row_clusters(cells: dict[str, dict]) -> tuple[dict[str, list[str]], dict]:
    """Cells whose rows are (nearly) the same sequences, joined (transitively) into one cluster, which the group
    statistic averages row by row: the same (rank, gap) fingerprint; or equal n and (rank, gap) equal on at least
    ``ROW_OVERLAP`` of the rows; or update cells of one draw (``draw_key``: the same names, codes and order, only
    the update's phrasing differs; updsame/updnow/updnew at one length, whose gaps differ by the stem's length,
    so neither test above sees them; their per-row differences correlate, +0.44 at 2048 in round 11).
    Row-index clustering is unbiased whether or not the joined cells are correlated, so joining cells that are
    in fact independent costs no bias. Returns ({cluster key: cells}, {"a~b": overlap} for joins that are not
    fingerprint matches)."""
    names = list(cells)
    parent = {c: c for c in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    overlap = {}
    for i, c in enumerate(names):
        for e in names[i + 1:]:
            p, q = cells[c], cells[e]
            if len(p["d"]) != len(q["d"]):
                continue
            if p["fp"] == q["fp"]:
                parent[find(e)] = find(c)
                continue
            rp, rq = p.get("rows"), q.get("rows")
            frac = sum(x == y for x, y in zip(rp, rq)) / len(rp) if rp and rq and len(rp) == len(rq) else 0.0
            draw = p.get("draw") is not None and p.get("draw") == q.get("draw")
            if frac >= ROW_OVERLAP or draw:
                parent[find(e)] = find(c)
                rank = sum(x[0] == y[0] for x, y in zip(rp, rq)) / len(rp) if rp and rq and len(rp) == len(rq) else None
                overlap[f"{c}~{e}"] = {"rows_equal": frac, "rank_equal": rank, "same_draw": draw}
    out: dict[str, list[str]] = {}
    for c in names:                                                # clusters in the cells' order
        out.setdefault(find(c), []).append(c)
    return out, overlap


def group_stats(cells: dict[str, dict], opts, lower_better: bool = False) -> dict | None:
    """The group statistic from paired cells {cell: paired_cell}: the mean over cells, rows shared by cells of
    one ``row_clusters`` cluster averaged row by row first (so the normal interval carries their correlation).
    A group of one 0/1 cell decides near the edge on Newcombe's interval, as that cell's own row does."""
    if not cells:
        return None
    K = len(cells)
    by_fp, overlap = row_clusters(cells)
    stat = la = lb = v = 0.0
    clusters = []
    for fp, names in by_fp.items():
        w = len(names) / K
        rows = [sum(col) / len(col) for col in zip(*[cells[c]["d"] for c in names])]
        n = len(rows)
        stat += w * sum(rows) / n
        la += w * mean([mean(cells[c]["a"]) for c in names])
        lb += w * mean([mean(cells[c]["b"]) for c in names])
        v += w * w * var(rows) / n
        clusters.append((w, [cells[c]["d"] for c in names]))
    half = Z * math.sqrt(v)
    out = {"diff": stat, "a": la, "b": lb, "cells": sorted(cells), "n_cells": K,
           "shared_rows": [sorted(n) for n in by_fp.values() if len(n) > 1], "normal": [stat - half, stat + half]}
    if overlap:
        out["row_overlap"] = overlap
    alt = "bootstrap"
    if K == 1 and next(iter(cells.values()))["binary"]:            # one 0/1 cell: as cell_stats decides it
        (p,) = cells.values()
        out["newcombe"] = list(newcombe_paired(p["a"], p["b"]))
        alt = "newcombe"
    if opts.bootstrap:
        out["bootstrap"] = bootstrap(clusters, opts)
    out.update(decide(out, near(la, lb, opts.near), opts, lower_better, alt=alt))
    return out


def compare(runs: dict, a: str, refs: list[str], opts) -> dict:
    """Checkpoint-level comparison of run ``a`` against ``refs`` (one run, or several averaged row by row):
    every cell they share, the groups, the refused pairs and the gate block's differences."""
    A = runs[a]
    R = [runs[r] for r in refs]
    allow = getattr(opts, "allow_modified", False)
    shared = sorted(set(A).intersection(*[set(r) for r in R]))
    out = {"a": a, "ref": refs, "cells": {}, "refused": {}, "void": [], "unverified": [], "groups": {}, "beside": {}}
    paired = {"exact": {}, "stale": {}}
    for c in shared:
        if A[c]["void"] or any(r[c]["void"] for r in R):
            out["void"].append(c)
            continue
        if excluded(A[c]) or any(excluded(r[c]) for r in R):       # no windowed control: not checked, left out
            out["unverified"].append(c)
            continue
        row = {"full": A[c]["full"], "ref_full": mean([r[c]["full"] for r in R])}
        for metric in ("exact", "stale"):
            if metric == "stale" and A[c]["level_stale"] is None:
                continue
            p = paired_cell(A[c], [r[c] for r in R], metric, allow)
            if "refused" in p:
                out["refused"][f"{c}:{metric}"] = p["refused"]
                if "unpaired_diff" in p:
                    row[f"{metric}_unpaired_diff"] = p["unpaired_diff"]
                continue
            paired[metric][c] = p
            row[metric] = cell_stats(p, opts, lower_better=(metric == "stale"))
        out["cells"][c] = row
    for g, spec in GROUPS.items():
        got = {c: paired[spec["metric"]][c] for c in spec["cells"] if c in paired[spec["metric"]]}
        st = group_stats(got, opts, spec.get("lower_better", False))
        if st is not None:
            st["missing"] = [c for c in spec["cells"] if c not in got]
            if spec.get("reported_only"):
                st["reported_only"] = True
            out["groups"][g] = st
        else:                                                      # nothing paired: say what there is instead
            unp = [out["cells"][c][f"{spec['metric']}_unpaired_diff"] for c in spec["cells"]
                   if f"{spec['metric']}_unpaired_diff" in out["cells"].get(c, {})]
            if unp:
                out["groups"][g] = {"unpaired_only": True, "unpaired_diff_mean": mean(unp), "n_cells": len(unp),
                                    "reported_only": spec.get("reported_only", False)}
        for c in BESIDE.get(g, []):
            if c in out["cells"]:
                out["beside"][c] = out["cells"][c]
    keep = [c for c in shared if c not in out["void"] and c not in out["unverified"]]
    ga, gr = gate_block(A, opts, keep), [gate_block(r, opts, keep) for r in R]   # the same cells on both sides
    out["gate"] = {"a": ga, "ref": gr[0] if len(gr) == 1 else None,
                   "diff": gate_diff(ga, gr[0]) if len(gr) == 1 else None}
    return out


# --------------------------------------------------------------------------------------------- gate block
def late_drop(cells: dict, opts) -> dict | None:
    """The late-drop step of the run behind these evaluations (their ``result``), read from its logged training
    loss, robustly: ``pl.train`` logs one single batch's loss per point, and single batches spike (0.98 among
    0.001s in round 11), so a raw rule ("every later point ≤ the threshold") lets one batch decide. Here:

      smoothed   the trailing median (``--drop-smooth median``, default; or ``mean``) of the last k logged
                 losses, k = ``--drop-window`` (0: max(5, points // 20)); a median ignores spikes in fewer than
                 half the window
      step       the first logged step from which the smoothed loss stays ≤ ``--drop-loss`` to the end
      status     ``learned`` (step set); ``never`` (the smoothed loss is above the threshold at the end: no step);
                 ``too few points`` (fewer than ``--drop-min-points`` logged, default 30: the step is not read;
                 runs logged every steps/10 have 10 and need ``--log-every``); ``not logged`` (no curve);
                 ``not found`` (no training JSON)

    ``step`` is None unless the status is ``learned``. The curriculum's end (``needle_curriculum``, P:STEPS
    stages) is reported: a step before it was read while the task still had fewer pairs."""
    res = next((c["result"] for c in cells.values() if c.get("result")), None)
    if not res:
        return None
    cands = [Path(res), Path(opts.root) / res] + [p / Path(res).name for p in {Path(c["path"]).parent.parent
                                                                               for c in cells.values()}]
    path = next((p for p in cands if p.is_file() and p.suffix == ".json"), None)
    if path is None:
        return {"result": res, "found": False, "status": "not found", "step": None}
    try:
        rec = json.loads(path.read_text())
    except ValueError:
        return {"result": res, "found": False, "status": "not found", "step": None}
    curve = [c for c in rec.get("curve") or [] if isinstance(c, dict) and "step" in c and "loss" in c]
    thr = opts.drop_loss
    out = {"result": str(path), "found": True, "step": None, "threshold": thr, "points": len(curve)}
    if not curve:
        return {**out, "status": "not logged"}
    out["log_every"] = curve[1]["step"] - curve[0]["step"] if len(curve) > 1 else None
    out["final_loss"] = curve[-1]["loss"]
    cur = str(rec.get("needle_curriculum") or "")
    stages = [st.split(":") for st in cur.split(",") if ":" in st]
    if stages:
        try:
            out["curriculum_end"] = sum(int(n) for _, n in stages)
        except ValueError:
            pass
    min_pts = getattr(opts, "drop_min_points", 30)
    if len(curve) < min_pts:
        return {**out, "status": "too few points", "min_points": min_pts}
    k = getattr(opts, "drop_window", 0) or max(5, len(curve) // 20)
    how = getattr(opts, "drop_smooth", "median")
    loss = [c["loss"] for c in curve]
    smooth = {}
    for i in range(k - 1, len(loss)):
        w = sorted(loss[i - k + 1:i + 1])
        smooth[i] = (w[k // 2] if k % 2 else (w[k // 2 - 1] + w[k // 2]) / 2) if how == "median" else sum(w) / k
    out.update({"window": k, "smooth": how, "final_smoothed": smooth.get(len(loss) - 1)})
    step = None
    for i in range(len(loss) - 1, k - 2, -1):                     # walk back while the smoothed loss stays low
        if smooth[i] > thr:
            break
        step = curve[i]["step"]
    out["step"] = step
    out["status"] = "learned" if step is not None else "never"
    return out


def gate_block(run: dict, opts, only: list[str] | None = None) -> dict:
    """The gate-robustness numbers of one run (reported only): the w1s block's means per cell set, over the cells
    in ``only`` if given (a comparison passes the cells both sides have, so a difference is not a difference of
    cell sets)."""
    out = {}
    for name, cells in GATE_SETS.items():
        use = [c for c in cells if c in run and (only is None or c in only) and not excluded(run[c])
               and run[c].get("w1s")]
        got = [run[c]["w1s"] for c in use]
        if got:
            out[name] = {k: mean([w.get(k) for w in got]) for k in GATE_KEYS}
            out[name]["cells"] = len(got)
            out[name]["cell_names"] = use
    if "natural" in out and "random" in out:
        out["natural_minus_random"] = {k: (out["natural"][k] - out["random"][k])
                                       if out["natural"][k] is not None and out["random"][k] is not None else None
                                       for k in GATE_KEYS}
    out["late_drop"] = late_drop(run, opts)
    return out


def gate_diff(a: dict, b: dict) -> dict:
    """a − b per gate set, only where both sides averaged the same cells (else listed in ``cells_differ``); the
    late-drop step difference only when both runs read a step (``late_drop_status`` says why not)."""
    out, differ = {}, {}
    for name in GATE_SETS:
        if name in a and name in b:
            if a[name].get("cell_names") != b[name].get("cell_names"):
                differ[name] = [a[name].get("cell_names"), b[name].get("cell_names")]
                continue
            out[name] = {k: a[name][k] - b[name][k] for k in GATE_KEYS
                         if a[name].get(k) is not None and b[name].get(k) is not None}
    if "natural_minus_random" in a and "natural_minus_random" in b and not ({"natural", "random"} & set(differ)):
        out["natural_minus_random"] = {k: a["natural_minus_random"][k] - b["natural_minus_random"][k]
                                       for k in GATE_KEYS if a["natural_minus_random"].get(k) is not None
                                       and b["natural_minus_random"].get(k) is not None}
    if differ:
        out["cells_differ"] = differ
    la, lb = a.get("late_drop") or {}, b.get("late_drop") or {}
    sa, sb = la.get("step"), lb.get("step")
    out["late_drop_step"] = sa - sb if sa is not None and sb is not None else None
    out["late_drop_status"] = [la.get("status"), lb.get("status")]
    return out


# ------------------------------------------------------------------------------------------- recipe level
def common_cells(runs: dict, names: list[str], cells: list[str], metric: str,
                 allow_modified: bool = False) -> tuple[list[str], dict]:
    """The cells of ``cells`` that every run in ``names`` has, not void (nor unverified), with a level for
    ``metric``, and whose sequences agree across the runs (strong check against the first run); the others with
    why."""
    keep, why = [], {}
    for c in cells:
        have = [runs[r].get(c) for r in names]
        if any(h is None for h in have):
            why[c] = "missing in " + ", ".join(r for r, h in zip(names, have) if h is None)
        elif any(h["void"] for h in have):
            why[c] = "void"
        elif any(excluded(h) for h in have):
            why[c] = "unverified (no windowed control)"
        elif any(h[f"level_{metric}"] is None for h in have):
            why[c] = f"no {metric} level"
        else:
            bad = [(r, pair_check(have[0], h, allow_modified)) for r, h in zip(names[1:], have[1:])]
            bad = [(r, w) for r, w in bad if w]
            if bad:
                why[c] = f"unpaired with {names[0]}: {bad[0][0]} ({bad[0][1]})"
            else:
                keep.append(c)
    return keep, why


def level(run: dict, cells: list[str], metric: str) -> float:
    return sum(run[c][f"level_{metric}"] for c in cells) / len(cells)


def levels(runs: dict, names: list[str], group: str,
           allow_modified: bool = False) -> tuple[dict[str, float] | None, dict]:
    """Each run's group level on the cells all of them share ({run: level}, info). ``pick``: the mean of the
    grid, names and query levels."""
    if group == "pick":
        parts, info = {}, {}
        for g in PICK:
            lv, inf = levels(runs, names, g, allow_modified)
            if lv is None:
                return None, {"missing_group": g, **inf}
            parts[g], info[g] = lv, inf
        return {r: sum(parts[g][r] for g in PICK) / len(PICK) for r in names}, info
    spec = GROUPS[group]
    cells, why = common_cells(runs, names, spec["cells"], spec["metric"], allow_modified)
    info = {"cells": cells, "left_out": why}
    if not cells:
        return None, info
    return {r: level(runs[r], cells, spec["metric"]) for r in names}, info


def seed_key(run: str):
    """The data seed of a run (``run_seed``); a run without one is its own seed."""
    s = run_seed(run)
    return s if s is not None else ("run", run)


def seed_noise(sets: list[list[tuple[object, float]]]) -> dict:
    """Noise estimates from repeated runs of fixed recipes, each set a list of (data seed, group level).

      s_seed, df             the pooled between-run SD (each set's sample variance, pooled): σ_seed² + σ_run²
                             when a set's runs are on different data seeds (the pre-registered estimator)
      s_within, df_within    σ_run: pooled within-seed residuals, Σ (x − mean of its seed in its set)², on
                             n − (seeds) df per set; only same-seed replicates (``run.py --init-seed``, or a
                             rerun) contribute, so the seed main effect is removed before pooling
      seed_var, df_seed      σ_seed² by the one-way random-effects method of moments: (SS between seeds −
                             df_seed·σ̂_run²) / Σ (n − Σ n_s²/n), floored at 0; None without both a same-seed
                             replicate and two seeds in one set"""
    ss = df = ssw = dfw = ssb = dfb = coef = 0.0
    for members in sets:
        x = [v for _, v in members]
        n = len(x)
        if n < 2:
            continue
        ss += var(x) * (n - 1)
        df += n - 1
        by: dict = {}
        for sd, v in members:
            by.setdefault(sd, []).append(v)
        m = sum(x) / n
        for v in by.values():
            mv = sum(v) / len(v)
            ssw += sum((y - mv) ** 2 for y in v)
            ssb += len(v) * (mv - m) ** 2
        dfw += n - len(by)
        dfb += len(by) - 1
        coef += n - sum(len(v) ** 2 for v in by.values()) / n
    out = {"s_seed": math.sqrt(ss / df) if df else None, "df": int(df),
           "s_within": math.sqrt(ssw / dfw) if dfw else None, "df_within": int(dfw), "seed_var": None,
           "df_seed": int(dfb)}
    if dfw and coef > 0:
        out["seed_var"] = max(0.0, (ssb - dfb * ssw / dfw) / coef)
    return out


def pooled_sd(runs: dict, noise: dict[str, list[str]], group: str, like: list[str],
              allow_modified: bool = False) -> dict:
    """s_seed and σ_run (``seed_noise``) of the group level over the ``noise`` sets (each a fixed recipe's
    repeated runs), on the cells ``like``'s runs share (a set lacking one is left out)."""
    want, want_info = levels(runs, like, group, allow_modified)
    used, skipped, sets = [], {}, []
    for name, members in noise.items():
        members = [m for m in members if m in runs]
        if len(members) < 2:
            skipped[name] = "fewer than 2 runs found"
            continue
        lv, info = levels(runs, list(like) + members, group, allow_modified)
        if lv is None or want is None or not same_cells(want_info, info):
            skipped[name] = "does not have every cell of the comparison"
            continue
        x = [lv[m] for m in members]
        sets.append([(seed_key(m), lv[m]) for m in members])
        used.append({"set": name, "runs": members, "levels": x, "sd": math.sqrt(var(x)),
                     "seeds": [run_seed(m) for m in members]})
    return {**seed_noise(sets), "sets": used, "skipped": skipped}


def same_cells(i1: dict, i2: dict) -> bool:
    """True if two ``levels`` infos used the same cells (for pick: in every part)."""
    if "cells" in i1 or "cells" in i2:
        return i1.get("cells") == i2.get("cells")
    return all(same_cells(i1[g], i2.get(g, {})) for g in PICK if g in i1)


def seed_coef(A: list[str], B: list[str]) -> float:
    """c = Σ_s (n_A,s/k_A − n_B,s/k_B)²: the factor of σ_seed² in Var(d̄) under level = recipe + seed + noise.
    0 when the two recipes have the same seed composition (the seed effect cancels in d̄); 1/k_A + 1/k_B when
    every run is on a seed of its own (the pre-registered formula's case)."""
    na, nb = {}, {}
    for r in A:
        na[seed_key(r)] = na.get(seed_key(r), 0) + 1
    for r in B:
        nb[seed_key(r)] = nb.get(seed_key(r), 0) + 1
    return sum((na.get(s, 0) / len(A) - nb.get(s, 0) / len(B)) ** 2 for s in set(na) | set(nb))


def recipe_interval(noise: dict, k_a: int, k_b: int, c_seed: float, how: str = "auto",
                    min_df: int = 3) -> tuple[dict | None, dict | None]:
    """(the interval the verdict uses, the between-run one) as {"estimator", "sd", "df", "half"}.

      between       t_df · s_seed · √(1/k_A + 1/k_B): the pre-registered estimator. When the recipes share seeds
                    the seed effect cancels in d̄ while s_seed includes it, so the interval is wide (≈ 99% where
                    95% is meant, in round-11 simulations); exact when every run is on a seed of its own
      within-seed   (the same seed composition, c = 0) t_dfw · σ̂_run · √(1/k_A + 1/k_B)
      seed-aware    (otherwise) Var(d̄) = c·σ̂_seed² + (1/k_A + 1/k_B)·σ̂_run², t on Satterthwaite's df

    ``how``: ``auto`` uses within-seed / seed-aware when the noise sets' same-seed replicates give df_within ≥
    ``min_df`` (``--min-within-df``, default 3), and between otherwise; ``between`` always uses between. The
    choice depends on the design (the df) only, never on the observed spreads, so it does not bias coverage;
    the floor is there because on 1 or 2 df the t quantile (12.7, 4.3) makes the calibrated interval wider in
    expectation than the conservative one unless σ_seed is large against σ_run."""
    kk = 1 / k_a + 1 / k_b
    between = None
    if noise.get("s_seed") is not None and noise.get("df"):
        sd = noise["s_seed"] * math.sqrt(kk)
        between = {"estimator": "between", "sd": sd, "df": noise["df"], "half": t_quantile(0.975, noise["df"]) * sd}
    if how == "between" or (noise.get("df_within") or 0) < max(1, min_df):
        return between, between
    sw2, dfw = noise["s_within"] ** 2, noise["df_within"]
    if c_seed < 1e-12:
        sd = math.sqrt(kk * sw2)
        return {"estimator": "within-seed", "sd": sd, "df": dfw, "half": t_quantile(0.975, dfw) * sd,
                "c_seed": 0.0}, between
    if noise.get("seed_var") is None or not noise.get("df_seed"):
        return between, between                                    # no estimate of σ_seed alone
    u1, u2 = c_seed * noise["seed_var"], kk * sw2
    v = u1 + u2
    den = (u1 * u1 / noise["df_seed"] if u1 > 0 else 0.0) + u2 * u2 / dfw
    df = v * v / den if den > 0 else dfw
    sd = math.sqrt(v)
    return {"estimator": "seed-aware", "sd": sd, "df": df, "half": t_quantile(0.975, df) * sd,
            "c_seed": c_seed}, between


def recipe_compare(runs: dict, A: list[str], B: list[str], noise: dict, opts) -> dict:
    """Recipe level: d̄ ± half per group (and pick), ``recipe_interval`` choosing the estimator (``interval``
    names it; the between-run interval is always reported as ``between``), with the per-seed differences and
    whether the sign holds on every seed both recipes have."""
    out = {}
    allow = getattr(opts, "allow_modified", False)
    how = getattr(opts, "s_seed", "auto")
    for g in list(GROUPS) + ["pick"]:
        lower = GROUPS.get(g, {}).get("lower_better", False)
        lv, info = levels(runs, A + B, g, allow)
        if lv is None:
            out[g] = {"missing": info}
            continue
        la, lb = [lv[r] for r in A], [lv[r] for r in B]
        d = sum(la) / len(la) - sum(lb) / len(lb)
        seeds = sorted({run_seed(r) for r in A} & {run_seed(r) for r in B} - {None})
        per_seed = {}
        for s in seeds:
            xa = [lv[r] for r in A if run_seed(r) == s]
            xb = [lv[r] for r in B if run_seed(r) == s]
            per_seed[str(s)] = sum(xa) / len(xa) - sum(xb) / len(xb)
        sign = (all(v > 0 for v in per_seed.values()) and d > 0) or (all(v < 0 for v in per_seed.values()) and d < 0)
        row = {"diff": d, "a_levels": dict(zip(A, la)), "b_levels": dict(zip(B, lb)), "per_seed": per_seed,
               "sign_every_seed": bool(per_seed) and sign, "k_a": len(A), "k_b": len(B),
               "cells": info.get("cells") if "cells" in info else {p: info[p]["cells"] for p in PICK},
               "reported_only": GROUPS.get(g, {}).get("reported_only", False)}
        noise_sd = pooled_sd(runs, noise, g, A + B, allow) if noise else None
        row["noise"] = noise_sd
        use = between = None
        if noise_sd:
            use, between = recipe_interval(noise_sd, len(A), len(B), seed_coef(A, B), how,
                                           getattr(opts, "min_within_df", 3))
        if between is not None:
            row["between"] = {"lo": d - between["half"], "hi": d + between["half"], "df": between["df"],
                              "verdict": verdict(d - between["half"], d + between["half"], lower)}
        if use is not None:
            row["interval"] = use
            row["lo"], row["hi"] = d - use["half"], d + use["half"]
            v = verdict(row["lo"], row["hi"], lower)
            row["verdict"] = v if v == "tie" or row["sign_every_seed"] else "tie (sign not on every seed)"
        else:
            row["verdict"] = "no s_seed (give --noise)"
        out[g] = row
    return out


UPDATE_GROUPS = ("updates_new", "updates_stale")


def updates_state(r: dict) -> str:
    """A ranking row's Updates verdict: ``worse`` if either group is worse; ``not worse`` only if both have a
    verdict (better / tie); otherwise ``unknown`` (a missing group, void or absent cells, or no --noise)."""
    vs = [str(r.get(g) or "") for g in UPDATE_GROUPS]
    if any(v.startswith("worse") for v in vs):
        return "worse"
    if all(v.startswith(("better", "tie")) for v in vs):
        return "not worse"
    return "unknown"


def stage_b_pick(ranking: list[dict]) -> dict:
    """The default for Daniel: the top recipe on pick; if it is worse on updates, also the best one that is not.
    If its Updates verdict is unknown, that is said (``top_worse_on_updates`` None, a warning) and the best
    recipe known not to be worse is named beside it; a recipe with an unknown verdict is never the ``also``."""
    if not ranking:
        return {}
    top = ranking[0]
    state = updates_state(top)
    out = {"top": top["recipe"], "top_updates": state,
           "top_worse_on_updates": {"worse": True, "not worse": False}.get(state)}
    if state != "not worse":
        nxt = next((r for r in ranking[1:] if updates_state(r) == "not worse"), None)
        out["also"] = nxt["recipe"] if nxt else None
        if state == "unknown":
            why = "; ".join(f"{g}: {top.get(g) or 'missing'}" for g in UPDATE_GROUPS)
            out["warning"] = (f"the Updates verdict of {top['recipe']} is unknown ({why}): it is picked on pick "
                              "alone; " + (f"{nxt['recipe']} is the best recipe known not to be worse on Updates"
                                           if nxt else "no recipe has a known Updates verdict that is not worse"))
    unknown = [r["recipe"] for r in ranking if updates_state(r) == "unknown"]
    if unknown:
        out["unknown_updates"] = unknown
    out["note"] = "ties go to the simpler recipe (not judged here); a recipe without the linear age penalty needs a DEC"
    return out


# ---------------------------------------------------------------------------------------------- factorial
LEVELS = {"1": 1, "+": 1, "+1": 1, "on": 1, "yes": 1, "true": 1, "0": -1, "-": -1, "-1": -1, "off": -1, "no": -1,
          "false": -1}


def read_design(path: str) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "run" not in rows[0]:
        raise SystemExit(f"--design {path}: needs a header with a 'run' column")
    factors = [k for k in rows[0] if k not in ("run", "seed")]
    out = []
    for r in rows:
        lv = {}
        for k in factors:
            v = str(r[k]).strip().lower()
            if v not in LEVELS:
                raise SystemExit(f"--design: run {r['run']} factor {k}: level {r[k]!r} (use 0/1, -/+, off/on)")
            lv[k] = LEVELS[v]
        seed = int(r["seed"]) if r.get("seed") not in (None, "") else run_seed(r["run"])
        out.append({"run": r["run"].strip(), "seed": seed, "levels": lv})
    return factors, out


ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}


def _block_centre(col: list[float], blocks: list) -> list[float]:
    """``col`` minus its mean within each block (the seed blocks of the model)."""
    sums: dict = {}
    for x, b in zip(col, blocks):
        t = sums.setdefault(b, [0.0, 0])
        t[0] += x
        t[1] += 1
    return [x - sums[b][0] / sums[b][1] for x, b in zip(col, blocks)]


def _corr(u: list[float], v: list[float]) -> float | None:
    nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(x * x for x in v))
    if nu < 1e-12 or nv < 1e-12:
        return None
    return sum(x * y for x, y in zip(u, v)) / (nu * nv)


def design_check(factors: list[str], design: list[dict]) -> dict:
    """The design's aliasing, from its ±1 columns with the seed blocks taken out (each column centred within its
    seed). Main effects must be balanced in every seed block and mutually orthogonal (else ``problems``: the
    factorial refuses). The defining words are the factor products constant within every block (all subsets for
    ≤ 10 factors, else up to 4); the resolution is the shortest word's length (``full`` if none). Main effects
    against two-factor interactions (resolution III) and two-factor interactions against each other (IV) are
    listed with their correlation: 1 is a full alias (a regular fraction), between 0 and 1 a partial one (a
    non-regular design; the resolution then reads at most III or IV from these lists)."""
    from itertools import combinations
    blocks = [r["seed"] for r in design]
    X = {f: [float(r["levels"][f]) for r in design] for f in factors}
    C = {f: _block_centre(X[f], blocks) for f in factors}
    out = {"problems": [], "warnings": [], "words": [], "main_vs_2fi": {}, "2fi_vs_2fi": {}}
    for f in factors:
        for b in sorted(set(blocks), key=lambda x: (x is None, str(x))):
            tot = sum(x for x, bb in zip(X[f], blocks) if bb == b)
            if abs(tot) > 1e-9:
                out["problems"].append(f"{f} unbalanced in seed {b} (Σx = {tot:+.0f})")
        if all(abs(x) < 1e-12 for x in C[f]):
            out["problems"].append(f"{f} constant within every seed (confounded with the seed blocks)")
    for f, g in combinations(factors, 2):
        r = _corr(C[f], C[g])
        if r is not None and abs(r) > 1e-9:
            out["problems"].append(f"{f} and {g} not orthogonal (r = {r:+.3f})" if abs(r) < 1 - 1e-9 else
                                   f"{f} aliased with {g}")
    kmax = len(factors) if len(factors) <= 10 else 4
    for k in range(1, kmax + 1):
        for sub in combinations(factors, k):
            col = [math.prod(X[f][i] for f in sub) for i in range(len(design))]
            if all(abs(x) < 1e-12 for x in _block_centre(col, blocks)):
                out["words"].append("·".join(sub))
    lens = [w.count("·") + 1 for w in out["words"]]
    res = min(lens) if lens else None
    pairs = list(combinations(factors, 2))
    P = {gh: _block_centre([X[gh[0]][i] * X[gh[1]][i] for i in range(len(design))], blocks) for gh in pairs}
    for f in factors:
        al = {}
        for gh in pairs:
            if f in gh:
                continue
            r = _corr(C[f], P[gh])
            if r is not None and abs(r) > 1e-9:
                al["×".join(gh)] = round(r, 4)
        if al:
            out["main_vs_2fi"][f] = al
    for a, b in combinations(pairs, 2):
        if set(a) & set(b):
            continue
        r = _corr(P[a], P[b])
        if r is not None and abs(r) > 1e-9:
            out["2fi_vs_2fi"]["×".join(a) + " ~ " + "×".join(b)] = round(r, 4)
    if res is None:                                                # no word: a full factorial or non-regular
        res = 3 if out["main_vs_2fi"] else 4 if out["2fi_vs_2fi"] else None
        out["resolution"] = ROMAN.get(res, "full") if res else "full (≥ V)"
    else:
        out["resolution"] = ROMAN.get(res, str(res))
    out["resolution_n"] = res
    if out["main_vs_2fi"]:
        out["warnings"].append("resolution below IV: main effects aliased with two-factor interactions: " +
                               "; ".join(f"{f} ~ {', '.join(k)}" for f, k in out["main_vs_2fi"].items()))
    return out


def _rank(M: list[list[float]], tol: float = 1e-9) -> int:
    A = [row[:] for row in M]
    rank, cols = 0, len(A[0]) if A else 0
    for c in range(cols):
        p = max(range(rank, len(A)), key=lambda r: abs(A[r][c]), default=None)
        if p is None or abs(A[p][c]) < tol:
            continue
        A[rank], A[p] = A[p], A[rank]
        for r in range(len(A)):
            if r != rank and A[r][c] != 0.0:
                f = A[r][c] / A[rank][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[rank])]
        rank += 1
        if rank == len(A):
            break
    return rank


def factorial(runs: dict, factors: list[str], design: list[dict], noise: dict, opts) -> dict:
    """Main effects per group: per seed, mean(on) − mean(off); averaged over seeds (``raw_effect``). The
    reported ``effect`` is the least-squares one, 2β from y = seed block + Σ β_F x_F (x = ±1), on the runs
    present: equal to the raw effect for a complete balanced design, and still adjusted for the other factors
    when runs are missing. Intervals are centred on it: from the residual of that model, and from the noise
    (σ_run from same-seed replicates when the noise sets have one, else s_seed: the effects are within-seed
    contrasts) times 2√(X'X)⁻¹_jj when --noise is given.

    The intended design (missing runs included) is checked first (``design_check``): unbalanced or
    non-orthogonal main effects, or a resolution below III, are refused (SystemExit); below IV is warned and
    each effect lists its two-factor aliases. With runs missing, a factor whose column is no longer independent
    of the others is ``not estimable`` (dropped from the fit), and the correlations the gaps introduce are
    listed. A singular fit gives no verdicts."""
    chk = design_check(factors, design)
    if chk["problems"] or (chk["resolution_n"] is not None and chk["resolution_n"] < 3):
        raise SystemExit("--design: not a usable main-effects design: " + "; ".join(
            chk["problems"] or [f"resolution {chk['resolution']} (words {', '.join(chk['words'])})"]))
    allow = getattr(opts, "allow_modified", False)
    how = getattr(opts, "s_seed", "auto")
    missing = [r["run"] for r in design if r["run"] not in runs]
    design = [r for r in design if r["run"] in runs]
    names = [r["run"] for r in design]
    out = {"factors": factors, "runs": names, "missing_runs": missing, "groups": {}, "design": chk}
    if not design:
        return out
    seeds = sorted({r["seed"] for r in design}, key=lambda s: (s is None, s))
    blocks = [r["seed"] for r in design]
    Xb = [[1.0] + [1.0 if r["seed"] == s else 0.0 for s in seeds[1:]] for r in design]
    est = []
    for f in factors:                                              # estimable: its column is not in the others' span
        cols = [[float(r["levels"][g]) for r in design] for g in factors]
        full = [xb + [c[i] for c in cols] for i, xb in enumerate(Xb)]
        drop = [xb + [c[i] for g, c in zip(factors, cols) if g != f] for i, xb in enumerate(Xb)]
        if _rank(full) > _rank(drop):
            est.append(f)
    out["not_estimable"] = [f for f in factors if f not in est]
    if missing:
        C = {f: _block_centre([float(r["levels"][f]) for r in design], blocks) for f in est}
        out["induced_correlation"] = {f"{f}~{g}": round(r, 4) for i, f in enumerate(est) for g in est[i + 1:]
                                      if (r := _corr(C[f], C[g])) is not None and abs(r) > 1e-9}
    for g in list(GROUPS) + ["pick"]:
        lv, info = levels(runs, names, g, allow)
        if lv is None:
            out["groups"][g] = {"missing": info}
            continue
        y = [lv[r["run"]] for r in design]
        X = [xb + [float(r["levels"][f]) for f in est] for xb, r in zip(Xb, design)]
        p, N = len(X[0]), len(X)
        XtX = [[sum(X[i][a] * X[i][b] for i in range(N)) for b in range(p)] for a in range(p)]
        Xty = [sum(X[i][a] * y[i] for i in range(N)) for a in range(p)]
        inv = solve(XtX, [[1.0 if i == j else 0.0 for i in range(p)] for j in range(p)])
        beta = res_sd = None
        if inv is not None:
            beta = [sum(inv[j][i] * Xty[i] for i in range(p)) for j in range(p)]
            rss = sum((y[i] - sum(X[i][j] * beta[j] for j in range(p))) ** 2 for i in range(N))
            res_df = N - p
            res_sd = math.sqrt(rss / res_df) if res_df > 0 else None
        noise_sd = pooled_sd(runs, noise, g, names, allow) if noise else None
        sig = None
        if noise_sd:
            if how != "between" and (noise_sd.get("df_within") or 0) >= max(1, getattr(opts, "min_within_df", 3)):
                sig = {"estimator": "within-seed", "s": noise_sd["s_within"], "df": noise_sd["df_within"]}
            elif noise_sd.get("s_seed") is not None and noise_sd.get("df"):
                sig = {"estimator": "between", "s": noise_sd["s_seed"], "df": noise_sd["df"]}
        lower = GROUPS.get(g, {}).get("lower_better", False)
        effs = {}
        for f in factors:
            per_seed = {}
            for s in seeds:
                on = [yy for yy, r in zip(y, design) if r["seed"] == s and r["levels"][f] > 0]
                off = [yy for yy, r in zip(y, design) if r["seed"] == s and r["levels"][f] < 0]
                if on and off:
                    per_seed[str(s)] = sum(on) / len(on) - sum(off) / len(off)
            raw = mean(list(per_seed.values()))
            n_on = sum(1 for r in design if r["levels"][f] > 0)
            n_off = len(design) - n_on
            row = {"effect": raw, "raw_effect": raw, "per_seed": per_seed, "n_on": n_on, "n_off": n_off}
            if f not in est:
                row["not_estimable"] = True
                effs[f] = row
                continue
            j = len(Xb[0]) + est.index(f)
            scale = None
            if beta is not None:
                row["ols_effect"] = row["effect"] = 2 * beta[j]
                scale = 2 * math.sqrt(inv[j][j])
                if res_sd is not None:
                    half = t_quantile(0.975, N - p) * res_sd * scale
                    e = row["effect"]
                    row["residual"] = {"lo": e - half, "hi": e + half, "df": N - p,
                                       "verdict": verdict(e - half, e + half, lower)}
            if sig is not None and row["effect"] is not None and inv is not None:
                half = t_quantile(0.975, sig["df"]) * sig["s"] * scale
                e = row["effect"]
                row["seed_noise"] = {"lo": e - half, "hi": e + half, "df": sig["df"], "estimator": sig["estimator"],
                                     "verdict": verdict(e - half, e + half, lower)}
            if f in chk["main_vs_2fi"]:
                row["aliases"] = chk["main_vs_2fi"][f]
            effs[f] = row
        out["groups"][g] = {"effects": effs, "residual_sd": res_sd, "singular": inv is None,
                            "cells": info.get("cells") if "cells" in info else {q: info[q]["cells"] for q in PICK}}
    return out


# --------------------------------------------------------------------------------------------------- text
def fmt(st: dict | None, key: str = "diff") -> str:
    if not st:
        return "-"
    if "lo" not in st:
        return f"{st[key]:+.3f}" if st.get(key) is not None else "-"
    tag = {"normal": "", "newcombe": "N", "bootstrap": "B"}.get(st.get("method"), "")
    return f"{st[key]:+.3f} [{st['lo']:+.3f},{st['hi']:+.3f}]{tag} {st['verdict']}"


def group_lines(groups: dict) -> list[str]:
    lines = []
    for g in GROUPS:
        st = groups.get(g)
        if not st:
            continue
        extra = " (reported only)" if st.get("reported_only") else ""
        if st.get("unpaired_only"):
            lines.append(f"    {g:14s} unpaired only: mean cell-level Δ {st['unpaired_diff_mean']:+.3f} over "
                         f"{st['n_cells']} cells{extra}")
        else:
            lines.append(f"    {g:14s} {fmt(st)}  cells {st['n_cells']}{extra}")
    return lines


def table(res: dict) -> str:
    lines = []
    if res["void"]:
        lines.append("VOID (windowed or side_empty exact ≠ 0): " + ", ".join(res["void"]))
    chk = res.get("checks") or {}
    if chk.get("unverified"):
        lines.append("UNVERIFIED (no windowed control; left out): " + ", ".join(chk["unverified"]))
    if chk.get("controls_unknown"):
        lines.append(f"side_empty control absent (kept; made before --ctrl-conds): "
                     f"{len(chk['controls_unknown'])} cells")
    for run, cells in (chk.get("modified") or {}).items():
        kinds = sorted({json.dumps(m, sort_keys=True) for m in cells.values()})
        lines.append(f"MODIFIED evaluation {run} ({len(cells)} cells): " + " | ".join(kinds)
                     + ("" if res["settings"].get("allow_modified") else "  (paired only with the same modifications)"))
    for arm, cols in res.get("arms", {}).items():
        lines.append(f"\n{arm}")
        for label, cmp_ in cols.items():
            if "error" in cmp_:
                lines.append(f"  {label:12s} {cmp_['error']}")
                continue
            lines.append(f"  vs {label} ({', '.join(cmp_['ref'])})")
            lines += group_lines(cmp_["groups"])
            for c, row in cmp_["beside"].items():
                full = " / ".join("-" if x is None else f"{x:.2f}" for x in (row["full"], row["ref_full"]))
                lines.append(f"    {c:14s} {fmt(row.get('exact'))}  full {full}")
            if cmp_["refused"]:
                lines.append(f"    refused pairs: {len(cmp_['refused'])} ({next(iter(cmp_['refused'].items()))})")
            gd = (cmp_.get("gate") or {}).get("diff") or {}
            if gd.get("natural_minus_random") or gd.get("late_drop_step") is not None or gd.get("late_drop_status"):
                nm = gd.get("natural_minus_random", {})
                cov, onf = nm.get("fact_coverage", float("nan")), nm.get("onset_in_fact", float("nan"))
                st = gd.get("late_drop_status") or [None, None]
                ld = (gd["late_drop_step"] if gd.get("late_drop_step") is not None
                      else f"- ({st[0]} / {st[1]})")
                lines.append(f"    gate Δ (nat−rand): coverage {cov:+.3f}, onset-in-fact {onf:+.3f}; "
                             f"late-drop step Δ {ld}")
            if gd.get("cells_differ"):
                lines.append(f"    gate sets on different cells (no Δ): {', '.join(gd['cells_differ'])}")
    for c in res.get("compare", []):
        lines.append(f"\n{c['a']} vs {' + '.join(c['ref'])}")
        lines += group_lines(c["groups"])
        if c["refused"]:
            lines.append(f"    refused pairs: {len(c['refused'])} ({next(iter(c['refused'].items()))})")
    for name, rc in res.get("recipe_compare", {}).items():
        lines.append(f"\nrecipe {name}")
        for g, row in rc.items():
            if "diff" in row:
                sd = (row.get("noise") or {}).get("s_seed")
                sw = (row.get("noise") or {}).get("s_within")
                seeds = json.dumps({k: round(v, 3) for k, v in row["per_seed"].items()})
                tail = f"  s_seed {sd:.3f}" if sd is not None else ""
                if row.get("interval") and row["interval"]["estimator"] != "between":
                    bt = row.get("between")
                    tail += (f"  σ_run {sw:.3f} ({row['interval']['estimator']}, df {row['interval']['df']:.1f})"
                             + (f"; between [{bt['lo']:+.3f},{bt['hi']:+.3f}] {bt['verdict']}" if bt else ""))
                tail += "  (reported only)" if row["reported_only"] else ""
                lines.append(f"    {g:14s} {fmt(row)}  per seed {seeds}{tail}")
            else:
                lines.append(f"    {g:14s} not computed: {json.dumps(row.get('missing'))[:160]}")
    if res.get("ranking"):
        lines.append("\nranking on pick: " + ", ".join(f"{r['recipe']} {r['pick']:+.3f}" for r in res["ranking"]))
        sb = res["stage_b_pick"]
        if sb.get("warning"):
            lines.append(f"WARNING: {sb['warning']}")
        lines.append(f"Stage-B default: {json.dumps(sb)}")
        unk = [r["recipe"] for r in res["ranking"] if updates_state(r) == "unknown"]
        if unk:
            lines.append("Updates verdict unknown (void/missing cells or no --noise): " + ", ".join(unk))
    fac = res.get("factorial")
    if fac:
        dc = fac.get("design") or {}
        lines.append(f"\nfactorial ({len(fac['runs'])} runs; missing {fac['missing_runs'] or 'none'}; resolution "
                     f"{dc.get('resolution')}; words {', '.join(dc.get('words') or []) or 'none'})")
        for w in dc.get("warnings") or []:
            lines.append(f"  WARNING: {w}")
        if fac.get("not_estimable"):
            lines.append(f"  NOT ESTIMABLE with the missing runs: {', '.join(fac['not_estimable'])}")
        if fac.get("induced_correlation"):
            lines.append(f"  missing runs correlate: {json.dumps(fac['induced_correlation'])} (effects adjusted by "
                         "least squares)")
        for g, row in fac["groups"].items():
            if "effects" not in row:
                continue
            if row.get("singular"):
                lines.append(f"    {g:14s} SINGULAR (aliased columns): no effects or verdicts")
                continue
            parts = []
            for f, e in row["effects"].items():
                iv, tag = (e["seed_noise"], "s") if "seed_noise" in e else (e.get("residual"), "r")
                if e.get("not_estimable"):
                    parts.append(f"{f} not estimable")
                    continue
                if e["effect"] is None:
                    parts.append(f"{f} -")
                    continue
                al = f" (aliased: {', '.join(e['aliases'])})" if e.get("aliases") else ""
                parts.append(f"{f} {e['effect']:+.3f}" + (f" [{iv['lo']:+.3f},{iv['hi']:+.3f}]{tag} {iv['verdict']}"
                                                          if iv else "") + al)
            lines.append(f"    {g:14s} " + "; ".join(parts))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------- main
def parse_kv(items: list[str], flag: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"{flag} {it!r}: expected NAME=...")
        k, v = it.split("=", 1)
        out.setdefault(k.strip(), []).extend(x.strip() for x in v.split(",") if x.strip())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", default=[], help="scan for <run>_<cell>.json (repeatable)")
    ap.add_argument("--run", action="append", default=[], metavar="NAME=TEMPLATE",
                    help="a run from a path template with {cell} (repeatable; replaces a discovered run)")
    ap.add_argument("--arms", nargs="*", default=None, metavar="REGEX", help="runs for the arm table (full match)")
    ap.add_argument("--ref", action="append", default=[], metavar="TEMPLATE",
                    help="same-seed reference run name, {seed} filled from the arm (one column each)")
    ap.add_argument("--rescue-seed", type=int, default=4, help="arms on this seed get the rescue and mechanism columns")
    ap.add_argument("--rescue-ref", default="e10b_r10_I1_s4", help="rescue column reference (I1 on that seed)")
    ap.add_argument("--mechanism-refs", default="e10b_r10_I1_s0,e10b_r10_I1_s2,e10b_r10_I1_s3",
                    help="mechanism column: runs averaged row by row (I1's learned seeds, round-10 Stage B)")
    ap.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"), help="run A against run B")
    ap.add_argument("--recipe", action="append", default=[], metavar="NAME=RUN,RUN",
                    help="a recipe's runs (repeatable)")
    ap.add_argument("--recipe-ref", default=None, help="compare every recipe against this one, ranked on pick")
    ap.add_argument("--noise", action="append", default=[], metavar="NAME=RUN,RUN",
                    help="repeated runs of one fixed recipe, for s_seed (repeatable)")
    ap.add_argument("--design", default=None, help="factorial design CSV (run, [seed,] factors)")
    ap.add_argument("--interval", choices=["auto", "normal", "alt"], default="auto",
                    help="auto: normal, or Newcombe / bootstrap near the ceiling; alt: always the alternative")
    ap.add_argument("--near", type=float, default=0.95, help="a level ≥ this (or ≤ 1 − this) is near the edge")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples (0: none)")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--drop-loss", type=float, default=0.1, help="late-drop threshold on the logged training loss")
    ap.add_argument("--drop-min-points", type=int, default=30,
                    help="late drop: fewer logged points than this read 'too few points' (no step)")
    ap.add_argument("--drop-window", type=int, default=0, help="late drop: smoothing window in logged points "
                                                               "(0: max(5, points // 20))")
    ap.add_argument("--drop-smooth", choices=["median", "mean"], default="median", help="late drop: the smoother")
    ap.add_argument("--s-seed", choices=["auto", "between"], default="auto",
                    help="recipe-level noise: auto = σ_run from same-seed replicates when the noise sets have one "
                         "(seed effect removed), else the between-run s_seed; between = always s_seed")
    ap.add_argument("--min-within-df", type=int, default=3,
                    help="--s-seed auto: the same-seed replicates' df needed to use σ_run (else s_seed)")
    ap.add_argument("--allow-modified", action="store_true",
                    help="pair evaluations whose eval-time modifications (--w1s-param, --w1s-set, ...) differ")
    ap.add_argument("--root", default=".", help="the repository root, for the evaluations' training JSON paths")
    ap.add_argument("--list", action="store_true", help="print the runs and their cells, and stop")
    ap.add_argument("--out", default=None, help="the summary JSON")
    ap.add_argument("--table", default=None, help="also write the readable table here")
    opts = ap.parse_args(argv)

    paths = discover(opts.dir)
    for name, templates in parse_kv(opts.run, "--run").items():
        paths[name] = from_templates(templates)
    checks: dict = {}
    runs, voids = load_runs(paths, checks)
    if opts.list:
        for r in sorted(runs):
            print(f"{r} (seed {run_seed(r)}): {' '.join(sorted(runs[r]))}")
        return {"runs": sorted(runs)}
    res = {"settings": {k: v for k, v in vars(opts).items() if k not in ("list",)}, "void": voids, "checks": checks,
           "runs": {r: {"seed": run_seed(r), "cells": {c: {"path": d["path"], "n": d["n"], "fp": d["fp"],
                                                            "exact": d["level_exact"], "stale": d["level_stale"],
                                                            "full": d["full"], "void": d["void"],
                                                            "unverified": d["unverified"], "mods": d["mods"]}
                                                        for c, d in sorted(cs.items())}}
                    for r, cs in sorted(runs.items())}}
    noise = {k: v for k, v in parse_kv(opts.noise, "--noise").items()}

    if opts.arms is not None:
        pats = [re.compile(p) for p in (opts.arms or [".*"])]
        mech = [m for m in opts.mechanism_refs.split(",") if m]
        refs_all = {opts.rescue_ref, *mech}
        arms = [r for r in sorted(runs) if any(p.fullmatch(r) for p in pats) and r not in refs_all]
        res["arms"] = {}
        for a in arms:
            s, cols = run_seed(a), {}
            for t in opts.ref:
                ref = t.replace("{seed}", str(s))
                cols[t] = compare(runs, a, [ref], opts) if ref in runs and ref != a else {"error": f"no run {ref}"}
            if s == opts.rescue_seed:
                cols["rescue"] = (compare(runs, a, [opts.rescue_ref], opts) if opts.rescue_ref in runs
                                  else {"error": f"no run {opts.rescue_ref}"})
                have = [m for m in mech if m in runs]
                cols["mechanism"] = (compare(runs, a, have, opts) if have == mech and have
                                     else {"error": f"missing {sorted(set(mech) - set(have))}"})
                if "error" not in cols["mechanism"]:            # each learned seed alone (means only)
                    quick = argparse.Namespace(**{**vars(opts), "bootstrap": 0})
                    one = {m: compare(runs, a, [m], quick)["groups"] for m in mech}
                    cols["mechanism"]["per_ref"] = {m: {g: st.get("diff") for g, st in gs.items()}
                                                    for m, gs in one.items()}
            res["arms"][a] = cols

    res["compare"] = []
    for a, b in opts.compare:
        for r in (a, b):
            if r not in runs:
                raise SystemExit(f"--compare: no run {r} (see --list)")
        res["compare"].append(compare(runs, a, [b], opts))

    recipes = parse_kv(opts.recipe, "--recipe")
    for name, members in list(recipes.items()) + list(noise.items()):
        gone = [m for m in members if m not in runs]
        if gone:
            raise SystemExit(f"recipe/noise {name}: no runs {gone} (see --list)")
    if opts.recipe_ref:
        if opts.recipe_ref not in recipes:
            raise SystemExit(f"--recipe-ref {opts.recipe_ref}: not a --recipe")
        res["recipe_compare"], ranking = {}, []
        for name, members in recipes.items():
            if name == opts.recipe_ref:
                continue
            rc = recipe_compare(runs, members, recipes[opts.recipe_ref], noise, opts)
            res["recipe_compare"][f"{name} vs {opts.recipe_ref}"] = rc
            if "diff" in rc.get("pick", {}):
                ranking.append({"recipe": name, "pick": rc["pick"]["diff"], "pick_verdict": rc["pick"]["verdict"],
                                **{g: rc[g].get("verdict") or "missing (no cells in common)" for g in UPDATE_GROUPS
                                   if g in rc}})
        ranking.sort(key=lambda r: -r["pick"])
        res["ranking"], res["stage_b_pick"] = ranking, stage_b_pick(ranking)
    if noise:
        res["noise"] = {name: {"runs": m, "seeds": [run_seed(x) for x in m]} for name, m in noise.items()}
    if opts.design:
        factors, design = read_design(opts.design)
        res["factorial"] = factorial(runs, factors, design, noise, opts)

    text = table(res)
    print(text)
    if opts.table:
        Path(opts.table).write_text(text + "\n")
    if opts.out:
        Path(opts.out).parent.mkdir(parents=True, exist_ok=True)
        Path(opts.out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
