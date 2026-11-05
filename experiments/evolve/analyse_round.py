"""Read a round's evaluations by the pre-registered rules (round-11 plan, "How results are read"; DEC11).

Inputs are ``eval_by_query.py`` JSONs, one per (run, cell). ``--dir D`` scans D for ``<run>_<cell>.json``, the
cell being the last ``_`` piece (``e11_r11_A11_s1_nat512p3.json`` → run ``e11_r11_A11_s1``, cell ``nat512p3``);
files without ``per_sequence`` (training JSONs) and names with a dot are skipped. ``--run NAME=TEMPLATE``
names a run by a path template with ``{cell}`` (repeatable; several templates for one name are tried in order,
cell by cell, and replace a discovered run of that name). A run's seed is the last ``_s<digits>`` in its name.
Round-10 Stage-B's ``update`` cell (update new, 512 tokens, k 1) is read as ``updnew512``.

Every evaluation whose ``windowed`` or ``side_empty`` exact is not 0 is void: listed, and left out of every
statistic.

Pairing (strong check). Two cells are paired only when their ``per_sequence`` rank and gap agree row for row
(and n, the eval seed, the task, the length and pairs, the filler file's name and, where both have it,
``per_sequence_repeat``); otherwise the pair is refused, with the reason, and the cell is left out. A
reference made of several runs (the mechanism column) is their row-by-row mean, each run checked so.

Groups. The statistic is the mean of the per-cell paired differences. Cells whose sequences are the same rows
(identical (rank, gap) fingerprints: held-out, nat512p3 and para1–4; updnow and updnew) are averaged row by
row first, so their correlation enters the interval; the weights keep the statistic the mean over cells.
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
                 names; and the late-drop step (the first logged step from which the training loss stays
                 ≤ ``--drop-loss``), read from the run's training JSON (the evaluation's ``result``)

Intervals.
  checkpoint level   the per-sequence paired interval, mean ± 1.96·sd/√n (two trained checkpoints). Near
                     the ceiling or floor (either side's level ≥ ``--near`` or ≤ 1 − ``--near``) ``--interval
                     auto`` decides on the alternative instead: Newcombe's paired score interval (method 10,
                     Wilson limits) for a single 0/1 cell, a bootstrap over rows (within each shared-row
                     cluster) for groups and averaged references. Every interval is reported.
  recipe level       d̄ ± t·s_seed·√(1/k_A + 1/k_B): d̄ the difference of the recipes' mean group levels, s_seed
                     the pooled between-run SD of the group level over repeated runs of a fixed recipe
                     (``--noise NAME=RUN,RUN,...``, repeatable), t on its pooled degrees of freedom. A recipe is
                     better or worse only where the interval clears 0 and the sign holds on every shared seed.

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
                                        column and one column per factor (0/1, -/+, off/on). Each effect is the
                                        mean over seeds of (mean on − mean off); intervals from the residual of a
                                        main-effects model with seed blocks, and from s_seed when ``--noise`` is given

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
            "w1s": d.get("w1s"), "result": d.get("result"), "raw_cell": None}


def is_void(c: dict) -> bool:
    return bool(c["windowed"]) or bool(c["side_empty"])


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


def load_runs(paths: dict[str, dict[str, str]]) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """Read every cell; aliases applied; returns ({run: {cell: data}}, void list)."""
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
            if c["void"]:
                voids.append(f"{run}/{name}")
            got[name] = c
        if got:
            runs[run] = got
    return runs, voids


def run_seed(name: str) -> int | None:
    m = re.findall(r"_s(\d+)", name)
    return int(m[-1]) if m else None


# ------------------------------------------------------------------------------------------------ pairing
def pair_check(a: dict, b: dict) -> str | None:
    """Why two cells cannot be paired row by row, or None (they draw the same sequences)."""
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
    return None


def values(c: dict, metric: str) -> list[float] | None:
    return c["exact"] if metric == "exact" else c["stale"]


def paired_cell(a: dict, refs: list[dict], metric: str) -> dict:
    """Per-row differences of ``a`` against the row mean of ``refs``, or a refusal with its reason."""
    for r in refs:
        why = pair_check(a, r)
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
    return {"d": [x - y for x, y in zip(va, ref)], "a": va, "b": ref, "fp": a["fp"], "binary": binary}


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


def group_stats(cells: dict[str, dict], opts, lower_better: bool = False) -> dict | None:
    """The group statistic from paired cells {cell: paired_cell}: the mean over cells, rows shared by cells with
    the same fingerprint averaged row by row first (so the normal interval carries their correlation)."""
    if not cells:
        return None
    K = len(cells)
    by_fp: dict[str, list[str]] = {}
    for c, p in cells.items():
        by_fp.setdefault(p["fp"], []).append(c)
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
    if opts.bootstrap:
        out["bootstrap"] = bootstrap(clusters, opts)
    out.update(decide(out, near(la, lb, opts.near), opts, lower_better, alt="bootstrap"))
    return out


def compare(runs: dict, a: str, refs: list[str], opts) -> dict:
    """Checkpoint-level comparison of run ``a`` against ``refs`` (one run, or several averaged row by row):
    every cell they share, the groups, the refused pairs and the gate block's differences."""
    A = runs[a]
    R = [runs[r] for r in refs]
    shared = sorted(set(A).intersection(*[set(r) for r in R]))
    out = {"a": a, "ref": refs, "cells": {}, "refused": {}, "void": [], "groups": {}, "beside": {}}
    paired = {"exact": {}, "stale": {}}
    for c in shared:
        if A[c]["void"] or any(r[c]["void"] for r in R):
            out["void"].append(c)
            continue
        row = {"full": A[c]["full"], "ref_full": mean([r[c]["full"] for r in R])}
        for metric in ("exact", "stale"):
            if metric == "stale" and A[c]["level_stale"] is None:
                continue
            p = paired_cell(A[c], [r[c] for r in R], metric)
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
    ga, gr = gate_block(A, opts), [gate_block(r, opts) for r in R]
    out["gate"] = {"a": ga, "ref": gr[0] if len(gr) == 1 else None,
                   "diff": gate_diff(ga, gr[0]) if len(gr) == 1 else None}
    return out


# --------------------------------------------------------------------------------------------- gate block
def late_drop(cells: dict, opts) -> dict | None:
    """The late-drop step of the run behind these evaluations (their ``result``): the first logged step from
    which every logged training loss is ≤ ``--drop-loss`` (None if the last one is above it)."""
    res = next((c["result"] for c in cells.values() if c.get("result")), None)
    if not res:
        return None
    cands = [Path(res), Path(opts.root) / res] + [p / Path(res).name for p in {Path(c["path"]).parent.parent
                                                                               for c in cells.values()}]
    path = next((p for p in cands if p.is_file() and p.suffix == ".json"), None)
    if path is None:
        return {"result": res, "found": False}
    try:
        curve = json.loads(path.read_text()).get("curve") or []
    except ValueError:
        return {"result": res, "found": False}
    curve = [c for c in curve if isinstance(c, dict) and "step" in c and "loss" in c]
    if not curve:
        return {"result": res, "found": True, "step": None}
    step = None
    for i in range(len(curve)):
        if all(c["loss"] <= opts.drop_loss for c in curve[i:]):
            step = curve[i]["step"]
            break
    every = curve[1]["step"] - curve[0]["step"] if len(curve) > 1 else None
    return {"result": str(path), "found": True, "step": step, "log_every": every, "final_loss": curve[-1]["loss"],
            "threshold": opts.drop_loss}


def gate_block(run: dict, opts) -> dict:
    """The gate-robustness numbers of one run (reported only): the w1s block's means per cell set."""
    out = {}
    for name, cells in GATE_SETS.items():
        got = [run[c]["w1s"] for c in cells if c in run and not run[c]["void"] and run[c].get("w1s")]
        if got:
            out[name] = {k: mean([w.get(k) for w in got]) for k in GATE_KEYS}
            out[name]["cells"] = len(got)
    if "natural" in out and "random" in out:
        out["natural_minus_random"] = {k: (out["natural"][k] - out["random"][k])
                                       if out["natural"][k] is not None and out["random"][k] is not None else None
                                       for k in GATE_KEYS}
    out["late_drop"] = late_drop(run, opts)
    return out


def gate_diff(a: dict, b: dict) -> dict:
    out = {}
    for name in list(GATE_SETS) + ["natural_minus_random"]:
        if name in a and name in b:
            out[name] = {k: a[name][k] - b[name][k] for k in GATE_KEYS
                         if a[name].get(k) is not None and b[name].get(k) is not None}
    sa, sb = (a.get("late_drop") or {}).get("step"), (b.get("late_drop") or {}).get("step")
    out["late_drop_step"] = sa - sb if sa is not None and sb is not None else None
    return out


# ------------------------------------------------------------------------------------------- recipe level
def common_cells(runs: dict, names: list[str], cells: list[str], metric: str) -> tuple[list[str], dict]:
    """The cells of ``cells`` that every run in ``names`` has, not void, with a level for ``metric``, and whose
    sequences agree across the runs (strong check against the first run); the others with why."""
    keep, why = [], {}
    for c in cells:
        have = [runs[r].get(c) for r in names]
        if any(h is None for h in have):
            why[c] = "missing in " + ", ".join(r for r, h in zip(names, have) if h is None)
        elif any(h["void"] for h in have):
            why[c] = "void"
        elif any(h[f"level_{metric}"] is None for h in have):
            why[c] = f"no {metric} level"
        else:
            bad = [(r, pair_check(have[0], h)) for r, h in zip(names[1:], have[1:])]
            bad = [(r, w) for r, w in bad if w]
            if bad:
                why[c] = f"unpaired with {names[0]}: {bad[0][0]} ({bad[0][1]})"
            else:
                keep.append(c)
    return keep, why


def level(run: dict, cells: list[str], metric: str) -> float:
    return sum(run[c][f"level_{metric}"] for c in cells) / len(cells)


def levels(runs: dict, names: list[str], group: str) -> tuple[dict[str, float] | None, dict]:
    """Each run's group level on the cells all of them share ({run: level}, info). ``pick``: the mean of the
    grid, names and query levels."""
    if group == "pick":
        parts, info = {}, {}
        for g in PICK:
            lv, inf = levels(runs, names, g)
            if lv is None:
                return None, {"missing_group": g, **inf}
            parts[g], info[g] = lv, inf
        return {r: sum(parts[g][r] for g in PICK) / len(PICK) for r in names}, info
    spec = GROUPS[group]
    cells, why = common_cells(runs, names, spec["cells"], spec["metric"])
    info = {"cells": cells, "left_out": why}
    if not cells:
        return None, info
    return {r: level(runs[r], cells, spec["metric"]) for r in names}, info


def pooled_sd(runs: dict, noise: dict[str, list[str]], group: str, like: list[str]) -> dict:
    """s_seed: the pooled between-run SD of the group level over the ``noise`` sets (each a fixed recipe's
    repeated runs), on the cells ``like``'s runs share (a set lacking one is left out)."""
    want, want_info = levels(runs, like, group)
    ss, df, used, skipped = 0.0, 0, [], {}
    for name, members in noise.items():
        members = [m for m in members if m in runs]
        if len(members) < 2:
            skipped[name] = "fewer than 2 runs found"
            continue
        lv, info = levels(runs, list(like) + members, group)
        if lv is None or want is None or not same_cells(want_info, info):
            skipped[name] = "does not have every cell of the comparison"
            continue
        x = [lv[m] for m in members]
        ss += var(x) * (len(x) - 1)
        df += len(x) - 1
        used.append({"set": name, "runs": members, "levels": x, "sd": math.sqrt(var(x))})
    return {"s_seed": math.sqrt(ss / df) if df else None, "df": df, "sets": used, "skipped": skipped}


def same_cells(i1: dict, i2: dict) -> bool:
    """True if two ``levels`` infos used the same cells (for pick: in every part)."""
    if "cells" in i1 or "cells" in i2:
        return i1.get("cells") == i2.get("cells")
    return all(same_cells(i1[g], i2.get(g, {})) for g in PICK if g in i1)


def recipe_compare(runs: dict, A: list[str], B: list[str], noise: dict, opts) -> dict:
    """Recipe level: d̄ ± t·s_seed·√(1/k_A + 1/k_B) per group (and pick), with the per-seed differences and
    whether the sign holds on every seed both recipes have."""
    out = {}
    for g in list(GROUPS) + ["pick"]:
        lower = GROUPS.get(g, {}).get("lower_better", False)
        lv, info = levels(runs, A + B, g)
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
        noise_sd = pooled_sd(runs, noise, g, A + B) if noise else None
        row["noise"] = noise_sd
        if noise_sd and noise_sd["s_seed"] is not None:
            half = t_quantile(0.975, noise_sd["df"]) * noise_sd["s_seed"] * math.sqrt(1 / len(A) + 1 / len(B))
            row["lo"], row["hi"] = d - half, d + half
            v = verdict(row["lo"], row["hi"], lower)
            row["verdict"] = v if v == "tie" or row["sign_every_seed"] else "tie (sign not on every seed)"
        else:
            row["verdict"] = "no s_seed (give --noise)"
        out[g] = row
    return out


def stage_b_pick(ranking: list[dict]) -> dict:
    """The default for Daniel: the top recipe on pick; if it is worse on updates, also the best one that is not."""
    if not ranking:
        return {}
    top = ranking[0]
    worse = lambda r: any(str(r.get(g, "")).startswith("worse") for g in ("updates_new", "updates_stale"))
    out = {"top": top["recipe"], "top_worse_on_updates": worse(top)}
    if worse(top):
        nxt = next((r for r in ranking[1:] if not worse(r)), None)
        out["also"] = nxt["recipe"] if nxt else None
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


def factorial(runs: dict, factors: list[str], design: list[dict], noise: dict, opts) -> dict:
    """Main effects per group: per seed, mean(on) − mean(off); averaged over seeds. Intervals: the residual of
    y = seed block + Σ β_F x_F (x = ±1, effect = 2β), and s_seed·√(1/n_on + 1/n_off) when --noise is given."""
    missing = [r["run"] for r in design if r["run"] not in runs]
    design = [r for r in design if r["run"] in runs]
    names = [r["run"] for r in design]
    out = {"factors": factors, "runs": names, "missing_runs": missing, "groups": {}}
    if not design:
        return out
    seeds = sorted({r["seed"] for r in design}, key=lambda s: (s is None, s))
    for g in list(GROUPS) + ["pick"]:
        lv, info = levels(runs, names, g)
        if lv is None:
            out["groups"][g] = {"missing": info}
            continue
        y = [lv[r["run"]] for r in design]
        X = [[1.0] + [1.0 if r["seed"] == s else 0.0 for s in seeds[1:]] + [float(r["levels"][f]) for f in factors]
             for r in design]
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
        noise_sd = pooled_sd(runs, noise, g, names) if noise else None
        lower = GROUPS.get(g, {}).get("lower_better", False)
        effs = {}
        for k, f in enumerate(factors):
            per_seed = {}
            for s in seeds:
                on = [yy for yy, r in zip(y, design) if r["seed"] == s and r["levels"][f] > 0]
                off = [yy for yy, r in zip(y, design) if r["seed"] == s and r["levels"][f] < 0]
                if on and off:
                    per_seed[str(s)] = sum(on) / len(on) - sum(off) / len(off)
            e = mean(list(per_seed.values()))
            n_on = sum(1 for r in design if r["levels"][f] > 0)
            n_off = len(design) - n_on
            row = {"effect": e, "per_seed": per_seed, "n_on": n_on, "n_off": n_off}
            if beta is not None:
                j = 1 + (len(seeds) - 1) + k
                row["ols_effect"] = 2 * beta[j]
                if res_sd is not None and e is not None:
                    half = t_quantile(0.975, N - p) * 2 * res_sd * math.sqrt(inv[j][j])
                    row["residual"] = {"lo": e - half, "hi": e + half, "df": N - p,
                                       "verdict": verdict(e - half, e + half, lower)}
            if noise_sd and noise_sd["s_seed"] is not None and n_on and n_off and e is not None:
                half = t_quantile(0.975, noise_sd["df"]) * noise_sd["s_seed"] * math.sqrt(1 / n_on + 1 / n_off)
                row["seed_noise"] = {"lo": e - half, "hi": e + half, "df": noise_sd["df"],
                                     "verdict": verdict(e - half, e + half, lower)}
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
            if gd.get("natural_minus_random") or gd.get("late_drop_step") is not None:
                nm = gd.get("natural_minus_random", {})
                cov, onf = nm.get("fact_coverage", float("nan")), nm.get("onset_in_fact", float("nan"))
                lines.append(f"    gate Δ (nat−rand): coverage {cov:+.3f}, onset-in-fact {onf:+.3f}; "
                             f"late-drop step Δ {gd.get('late_drop_step')}")
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
                seeds = json.dumps({k: round(v, 3) for k, v in row["per_seed"].items()})
                tail = f"  s_seed {sd:.3f}" if sd is not None else ""
                tail += "  (reported only)" if row["reported_only"] else ""
                lines.append(f"    {g:14s} {fmt(row)}  per seed {seeds}{tail}")
            else:
                lines.append(f"    {g:14s} not computed: {json.dumps(row.get('missing'))[:160]}")
    if res.get("ranking"):
        lines.append("\nranking on pick: " + ", ".join(f"{r['recipe']} {r['pick']:+.3f}" for r in res["ranking"]))
        lines.append(f"Stage-B default: {json.dumps(res['stage_b_pick'])}")
    fac = res.get("factorial")
    if fac:
        lines.append(f"\nfactorial ({len(fac['runs'])} runs; missing {fac['missing_runs'] or 'none'})")
        for g, row in fac["groups"].items():
            if "effects" not in row:
                continue
            parts = []
            for f, e in row["effects"].items():
                iv, tag = (e["seed_noise"], "s") if "seed_noise" in e else (e.get("residual"), "r")
                if e["effect"] is None:
                    parts.append(f"{f} -")
                    continue
                parts.append(f"{f} {e['effect']:+.3f}" + (f" [{iv['lo']:+.3f},{iv['hi']:+.3f}]{tag}" if iv else ""))
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
    ap.add_argument("--root", default=".", help="the repository root, for the evaluations' training JSON paths")
    ap.add_argument("--list", action="store_true", help="print the runs and their cells, and stop")
    ap.add_argument("--out", default=None, help="the summary JSON")
    ap.add_argument("--table", default=None, help="also write the readable table here")
    opts = ap.parse_args(argv)

    paths = discover(opts.dir)
    for name, templates in parse_kv(opts.run, "--run").items():
        paths[name] = from_templates(templates)
    runs, voids = load_runs(paths)
    if opts.list:
        for r in sorted(runs):
            print(f"{r} (seed {run_seed(r)}): {' '.join(sorted(runs[r]))}")
        return {"runs": sorted(runs)}
    res = {"settings": {k: v for k, v in vars(opts).items() if k not in ("list",)}, "void": voids,
           "runs": {r: {"seed": run_seed(r), "cells": {c: {"path": d["path"], "n": d["n"], "fp": d["fp"],
                                                            "exact": d["level_exact"], "stale": d["level_stale"],
                                                            "full": d["full"], "void": d["void"]}
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
                                **{g: rc[g].get("verdict") for g in ("updates_new", "updates_stale") if g in rc}})
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
