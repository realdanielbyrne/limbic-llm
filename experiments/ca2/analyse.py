"""Pre-stated comparisons from the round-12 result JSONs (``text.py``, ``ceiling.py``) and their traces.

Every interval is a percentile bootstrap over sequences (the unit of a training-free run: "seeds" are
document samples), paired wherever two arms ran on the same rows; ratios are ratios of sums, with numerator
and denominator resampled jointly. Two-sided p = 2·min(P*(Δ ≤ 0), P*(Δ ≥ 0)); Holm's step-down correction
runs over the primary family only. Everything else is exploratory and labelled so.

Text (per condition):
  vs_none        every arm − none: Δloss (nats/token), also by offset bucket and position bucket.
  matched        for a candidate direction (gist, mean) at gain g: candidate − each control at the same g
                 (anti_gist, mean/gist, self, random = the mean of random:k per sequence, foreign, window_gist),
                 and candidate − best control, the minimum over controls taken inside each bootstrap replicate.
                 PRIMARY: gist − best control, per condition and gain (Holm over that family).
  closure        C = (L_win − L_arm)/(L_win − L_full) (windowed conditions), overall and by offset / position
                 bucket; where the gap's interval includes 0 only the numerator is read.
  gdn_state      within the hybrid: the arm's effect with GDN intact (attn) − with GDN reset (all), per sequence;
                 and the base value of the retained state, L_none(all) − L_none(attn).
  contiguity     Δ boost of c_{+1} against the control (arm − none), overall, by lag bin, in / out of block.
  impulse        KL(arm ‖ none) and Δloss by lag after the burst, with intervals; the half-life bin; the mass
                 beyond the window.
  gain           the gain distribution of each arm, with the "did not vary" flag (SD < 0.1·(g_max − g_min)).
  cross_model    with two text JSONs of different models: difference of each arm's Δloss-vs-none (unpaired).
Ceiling: the knee per site and direction (``ceiling.knees``), the dose-response table, and for the branch
sweep each (condition, layer group, target, g) against none, GDN branch − attention control and GDN branch −
MLP control at the same g, and the single-layer ranking.
With ``--traces``: which per-token signal (in the none arm, same tokens) predicts where an arm helped or hurt:
Spearman ρ between Δloss_t = loss_arm − loss_none and every scalar trace key (and each column of keys with
≤ 32 columns), on steered, scored tokens.

    python experiments/ca2/analyse.py results/evolve/round12/stage0/text_Qwen3.5-2B-Base.json [--traces]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ca2 import measure as ms  # noqa: E402

CONTROLS = ("anti_gist", "mean", "gist", "self", "random", "foreign", "window_gist")


# ------------------------------------------------------------------------------------------ statistics
def p_two(b: dict) -> float | None:
    if b.get("p_le_0") is None:
        return None
    return float(min(1.0, 2 * min(b["p_le_0"], b["p_ge_0"])))


def holm(ps: dict[str, float | None]) -> dict[str, float | None]:
    """Holm step-down adjusted p-values (None stays None)."""
    keys = sorted([k for k, p in ps.items() if p is not None], key=lambda k: ps[k])
    m, out, run = len(keys), {k: None for k in ps}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def boot_stat(cols: dict[str, np.ndarray], fn, n: int, seed: int) -> dict:
    """Bootstrap of fn(resampled columns) over units (rows of the equal-length arrays in ``cols``)."""
    m = len(next(iter(cols.values())))
    if m == 0:
        return {"diff": None, "n_units": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n, m))
    with np.errstate(divide="ignore", invalid="ignore"):
        bs = np.array([fn({k: v[i] for k, v in cols.items()}) for i in idx])
        point = float(fn(cols))
    fin = bs[np.isfinite(bs)]
    if fin.size == 0:
        return {"diff": point, "lo": None, "hi": None, "n_units": m}
    lo, hi = np.quantile(fin, [0.025, 0.975])
    return {"diff": point, "lo": float(lo), "hi": float(hi), "se": float(fin.std(ddof=1)) if fin.size > 1 else 0.0,
            "n_units": m, "p_le_0": float((fin <= 0).mean()), "p_ge_0": float((fin >= 0).mean())}


def seq_loss(a: dict) -> np.ndarray:
    p = a["per_item"]
    return np.asarray(p["nll_sum"], dtype=float) / np.maximum(np.asarray(p["n_tok"], dtype=float), 1)


def arr(a: dict, k: str) -> np.ndarray:
    return np.asarray(a["per_item"][k], dtype=float)


# ------------------------------------------------------------------------------------------------ text
def by_cond(d: dict) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for key, a in d["arms"].items():
        if a.get("complete") and "per_item" in a:
            out.setdefault(a["condition"], {})[a["arm"]] = a
    return out


def ratio_diff(a: dict, b: dict, num: str, den: str, n: int, seed: int) -> list[dict]:
    """Per bucket (columns of per-item (n_seq, n_b) arrays): Σnum_a/Σden_a − Σnum_b/Σden_b, jointly resampled."""
    A, Ad, B_, Bd = arr(a, num), arr(a, den), arr(b, num), arr(b, den)
    out = []
    for j in range(A.shape[1]):
        cols = {"a": A[:, j], "ad": Ad[:, j], "b": B_[:, j], "bd": Bd[:, j]}
        out.append(boot_stat(cols, lambda c: c["a"].sum() / c["ad"].sum() - c["b"].sum() / c["bd"].sum(), n, seed))
    return out


def text_condition(cond: str, arms: dict[str, dict], n: int, seed: int) -> dict:
    out: dict = {"vs_none": {}, "matched": {}, "closure": {}, "contiguity": {}, "impulse": {}, "gain": {}}
    none = arms.get("none")
    rows = None if none is None else none["per_item"]["row"]
    same = {k: a for k, a in arms.items() if rows is None or a["per_item"]["row"] == rows}
    L = {k: seq_loss(a) for k, a in same.items()}
    for k, a in same.items():
        out["gain"][k] = a.get("gain_dist")
        if none is None or k == "none":
            continue
        r = {"d_loss": ms.boot(L[k], L["none"], n=n, seed=seed)}
        r["d_loss"]["p"] = p_two(r["d_loss"])
        if "off_loss_sum" in a["per_item"]:
            r["d_loss_by_offset"] = ratio_diff(a, none, "off_loss_sum", "off_n", n, seed)
            r["d_loss_by_position"] = ratio_diff(a, none, "pos_loss_sum", "pos_n", n, seed)
        out["vs_none"][k] = r
        if a.get("window") and "loss_full_sum" in a["per_item"]:
            cols = {"a": arr(a, "nll_sum"), "w": arr(a, "loss_base_sum"), "f": arr(a, "loss_full_sum"),
                    "n": arr(a, "n_tok")}
            c = {"C": boot_stat(cols, lambda x: (x["w"].sum() - x["a"].sum()) / (x["w"].sum() - x["f"].sum()), n, seed),
                 "gap": boot_stat(cols, lambda x: (x["w"].sum() - x["f"].sum()) / x["n"].sum(), n, seed),
                 "num": boot_stat(cols, lambda x: (x["w"].sum() - x["a"].sum()) / x["n"].sum(), n, seed)}
            for b in ("off", "pos"):
                if f"{b}_loss_full_sum" not in a["per_item"]:
                    continue
                A_, W_, F_, N_ = (arr(a, f"{b}_{k}") for k in ("loss_sum", "loss_base_sum", "loss_full_sum", "n"))
                c[f"by_{b}"] = []
                for j in range(N_.shape[1]):
                    cj = {"a": A_[:, j], "w": W_[:, j], "f": F_[:, j], "n": N_[:, j]}
                    if cj["n"].sum() == 0:
                        c[f"by_{b}"].append(None)
                        continue
                    gp = boot_stat(cj, lambda x: (x["w"].sum() - x["f"].sum()) / x["n"].sum(), n, seed)
                    ex = bool(gp.get("lo") is not None and (gp["lo"] > 0 or gp["hi"] < 0))
                    c[f"by_{b}"].append({
                        "gap": gp, "gap_excludes_0": ex,
                        "num": boot_stat(cj, lambda x: (x["w"].sum() - x["a"].sum()) / x["n"].sum(), n, seed),
                        "C": boot_stat(cj, lambda x: (x["w"].sum() - x["a"].sum()) / (x["w"].sum() - x["f"].sum()),
                                       n, seed) if ex else None})
            g = c["gap"]
            c["gap_excludes_0"] = bool(g.get("lo") is not None and (g["lo"] > 0 or g["hi"] < 0))
            c["read"] = "C" if c["gap_excludes_0"] else "numerator only (the gap's interval includes 0)"
            out["closure"][k] = c
        # contiguity: Δ boost (c_{+1} vs control) against none
        pc = {}
        for dname in ("p1", "p2", "m1", "m2"):
            cols = {"s": arr(a, f"cont_{dname}_all_sum"), "n": arr(a, f"cont_{dname}_all_n"),
                    "s0": arr(none, f"cont_{dname}_all_sum"), "n0": arr(none, f"cont_{dname}_all_n")}
            pc[f"{dname}_all"] = boot_stat(cols, lambda x: x["s"].sum() / x["n"].sum() - x["s0"].sum() / x["n0"].sum(),
                                           n, seed)
        for tag in ("in", "out"):
            pc[f"p1_{tag}_by_lag"] = ratio_diff(a, none, f"cont_p1_{tag}_sum", f"cont_p1_{tag}_n", n, seed)
        out["contiguity"][k] = pc
        if "imp_n" in a["per_item"]:
            kl = arr(a, "imp_kl_sum")
            cnt = arr(a, "imp_n")
            dl = arr(a, "imp_dloss_sum")
            bins = []
            for j in range(kl.shape[1]):
                cols = {"k": kl[:, j], "d": dl[:, j], "n": cnt[:, j]}
                if cols["n"].sum() == 0:
                    bins.append(None)
                    continue
                bins.append({"kl": boot_stat(cols, lambda x: x["k"].sum() / x["n"].sum(), n, seed),
                             "dloss": boot_stat(cols, lambda x: x["d"].sum() / x["n"].sum(), n, seed)})
            first = next((b["kl"]["diff"] for b in bins if b), None)
            half = next((j for j, b in enumerate(bins) if b and first and b["kl"]["diff"] < 0.5 * first), None)
            edges = a.get("edges", {}).get("lag")
            out["impulse"][k] = {"by_lag": bins, "lag_edges": edges, "half_life_bin": half,
                                 "half_life_lag": None if half is None or not edges else edges[half],
                                 "pre_burst_kl_max": a.get("checks", {}).get("imp_pre_kl_max")}
    # matched controls at the same gain
    gains = sorted({k.split("@")[1] for k in same if "@" in k and not k.startswith("impulse")})
    fam = {}
    for g in gains:
        for cand in ("gist", "mean"):
            ck = f"{cand}@{g}"
            if ck not in L:
                continue
            ctrl = {}
            rnd = [L[k] for k in L if k.startswith("random:") and k.endswith(f"@{g}")]
            if rnd:
                ctrl["random"] = np.mean(rnd, 0)
            for c in CONTROLS:
                if c != cand and c != "random" and f"{c}@{g}" in L:
                    ctrl[c] = L[f"{c}@{g}"]
            res = {c: ms.boot(L[ck], v, n=n, seed=seed) for c, v in ctrl.items()}
            for v in res.values():
                v["p"] = p_two(v)
            if ctrl:
                cols = {"x": L[ck], **{f"c_{c}": v for c, v in ctrl.items()}}
                best = boot_stat(cols, lambda x: x["x"].mean() - min(x[f"c_{c}"].mean() for c in ctrl), n, seed)
                best["p"] = p_two(best)
                best["controls"] = sorted(ctrl)
                best["best_point"] = min(ctrl, key=lambda c: float(ctrl[c].mean()))
                res["best_control"] = best
                if cand == "gist":
                    fam[f"{cond}/gist@{g}"] = best["p"]
            out["matched"][ck] = res
    out["primary_family"] = fam
    return out


def gdn_state(conds: dict, n: int, seed: int) -> dict:
    if "attn" not in conds or "all" not in conds:
        return {}
    A, R = conds["attn"], conds["all"]
    out = {}
    if "none" in A and "none" in R and A["none"]["per_item"]["row"] == R["none"]["per_item"]["row"]:
        out["retained_state_value"] = ms.boot(seq_loss(R["none"]), seq_loss(A["none"]), n=n, seed=seed)
    for k in A:
        if k == "none" or k not in R or "none" not in A or "none" not in R:
            continue
        ea = seq_loss(A[k]) - seq_loss(A["none"])
        er = seq_loss(R[k]) - seq_loss(R["none"])
        out[k] = ms.boot(ea, er, n=n, seed=seed)
    return out


def analyse_text(d: dict, n: int, seed: int) -> dict:
    conds = by_cond(d)
    out = {"kind": "text", "model": d["model"]["name"], "site": d.get("site"), "conditions": {}}
    fam = {}
    for c, arms in conds.items():
        out["conditions"][c] = text_condition(c, arms, n, seed)
        fam.update(out["conditions"][c].pop("primary_family"))
    out["primary"] = {"family": "gist − best control per (condition, gain)", "p": fam, "p_holm": holm(fam)}
    out["gdn_state"] = gdn_state(conds, n, seed)
    return out


def cross_model(texts: list[dict], n: int, seed: int) -> dict:
    """Unpaired difference of Δloss-vs-none between two models' text runs (independent resampling)."""
    if len(texts) < 2:
        return {}
    a, b = texts[0], texts[1]
    ca, cb = by_cond(a), by_cond(b)
    out = {"models": [a["model"]["name"], b["model"]["name"]]}
    rng = np.random.default_rng(seed)
    for c in set(ca) & set(cb):
        for k in set(ca[c]) & set(cb[c]):
            if k == "none" or "none" not in ca[c] or "none" not in cb[c]:
                continue
            ea = seq_loss(ca[c][k]) - seq_loss(ca[c]["none"])
            eb = seq_loss(cb[c][k]) - seq_loss(cb[c]["none"])
            ia, ib = rng.integers(0, len(ea), (n, len(ea))), rng.integers(0, len(eb), (n, len(eb)))
            bs = ea[ia].mean(1) - eb[ib].mean(1)
            lo, hi = np.quantile(bs, [0.025, 0.975])
            out[f"{c}/{k}"] = {"diff": float(ea.mean() - eb.mean()), "lo": float(lo), "hi": float(hi),
                               "effect_a": float(ea.mean()), "effect_b": float(eb.mean())}
    return out


# --------------------------------------------------------------------------------------------- ceiling
def analyse_ceiling(d: dict, n: int, seed: int) -> dict:
    import argparse as _ap
    from ca2 import ceiling as ce
    args = _ap.Namespace(**d["args"])
    out = {"kind": "ceiling", "model": d["model"]["name"], **ce.knees(d, args), "curves": {}, "branch": {}}
    arms = d["arms"]
    for key, a in arms.items():
        if not a.get("complete"):
            continue
        parts = key.split("/")
        if parts[0].startswith("L") and len(parts) == 3:
            site, name = parts[0], parts[2]
            spec, _, g = name.partition("@")
            ag = a["agg"]
            out["curves"].setdefault(site, {}).setdefault(spec, []).append(
                {"g": float(g or 0), "dloss": ag["loss"] - ag["loss_base"], "kl": ag["kl_base"],
                 "absdent": ag.get("absdent"), "norm_final": ag.get("norm_ratio_final"),
                 "d_attn_up_ent": (ag.get("d_attn_up_ent") or {}).get("diff"),
                 "d_attn_last_ent": (ag.get("d_attn_last_ent") or {}).get("diff"),
                 "gen": (arms.get(f"gen/{site}/{name}") or {}).get("agg")})
    for site in out["curves"].values():
        for v in site.values():
            v.sort(key=lambda r: r["g"])
    # branch sweep
    for cond in {k.split("/")[1] for k in arms if k.startswith("branch/")}:
        pre = f"branch/{cond}/"
        none = arms.get(pre + "none")
        if not none or not none.get("complete"):
            continue
        Ln = seq_loss(none)
        tab, ctrl = {}, {}
        for key, a in arms.items():
            if not key.startswith(pre) or key == pre + "none" or not a.get("complete"):
                continue
            br = a.get("branch") or {}
            r = ms.boot(seq_loss(a), Ln, n=n, seed=seed)
            r["p"] = p_two(r)
            tab[key[len(pre):]] = {**br, "d_loss": r, "kl": a["agg"]["kl_base"]}
        for name, r in tab.items():
            if r.get("target") != "gdn":
                continue
            for c in ("attn", "mlp"):
                other = name.replace(":gdn@", f":{c}@")
                if other in tab:
                    b = ms.boot(seq_loss(arms[pre + name]), seq_loss(arms[pre + other]), n=n, seed=seed)
                    b["p"] = p_two(b)
                    ctrl[f"{name} - {c}"] = b
        singles = {}
        for name, r in tab.items():
            if r.get("target") == "gdn" and str(r.get("set", "")).startswith("L"):
                singles.setdefault(f"{r['g']:g}", []).append((r["set"], r["d_loss"]["diff"]))
        rank = {g: [s for s, _ in sorted(v, key=lambda x: x[1])] for g, v in singles.items()}
        out["branch"][cond] = {"vs_none": tab, "gdn_vs_controls": ctrl, "single_layer_rank_by_dloss": rank}
    return out


# ---------------------------------------------------------------------------------------------- traces
def _npz(pointer: dict, stream: str = "tokens") -> dict[str, np.ndarray] | None:
    man = json.loads(Path(pointer["manifest"]).read_text())
    files = sorted(f["path"] for f in man["files"] if f.get("arm") == stream and f["path"].endswith(".npz"))
    if not files:
        return None
    parts = [np.load(Path(pointer["dir"]) / f) for f in files]
    return {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0].files}


def _rank(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20 or np.std(x[ok]) == 0:
        return None
    return float(np.corrcoef(_rank(x[ok]), _rank(y[ok]))[0, 1])


def signal_predictors(d: dict, arm_filter: str, max_tok: int, seed: int, top: int = 30) -> dict:
    """Spearman ρ of Δloss_t (arm − none, same tokens) with each signal of the none arm (module docstring)."""
    out = {}
    conds = by_cond(d)
    rng = np.random.default_rng(seed)
    for c, arms in conds.items():
        none = arms.get("none")
        if not none or not none.get("traces"):
            continue
        sig = _npz(none["traces"])
        if sig is None:
            continue
        for k, a in arms.items():
            if k == "none" or not a.get("traces") or (arm_filter and not k.startswith(arm_filter)):
                continue
            tk = _npz(a["traces"])
            if tk is None or len(tk["out__loss"]) != len(sig["out__loss"]):
                continue
            dl = tk["out__loss"].astype(np.float64) - sig["out__loss"].astype(np.float64)
            use = np.isfinite(dl) & (tk["steer"] > 0) & (tk.get("pos__scored", np.ones_like(dl)) > 0)
            idx = np.flatnonzero(use)
            if idx.size > max_tok:
                idx = np.sort(rng.choice(idx, max_tok, replace=False))
            rho = {}
            for key, v in sig.items():
                if key.startswith("out__") or v.dtype.kind not in "fiu" or key in ("col", "ids", "idx", "step"):
                    continue
                if v.ndim == 1:
                    rho[key] = spearman(v[idx].astype(np.float64), dl[idx])
                elif v.ndim == 2 and v.shape[1] <= 32:
                    for j in range(v.shape[1]):
                        rho[f"{key}[{j}]"] = spearman(v[idx, j].astype(np.float64), dl[idx])
            rho["out__loss_none"] = spearman(sig["out__loss"][idx].astype(np.float64), dl[idx])
            rho["out__ent_none"] = spearman(sig["out__ent"][idx].astype(np.float64), dl[idx])
            ranked = sorted([(kk, r) for kk, r in rho.items() if r is not None], key=lambda x: -abs(x[1]))
            out[f"{c}/{k}"] = {"n_tok": int(idx.size), "mean_dloss": float(np.nanmean(dl[idx])) if idx.size else None,
                               "top": ranked[:top], "all": rho}
    return out


# ------------------------------------------------------------------------------------------------- main
def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("results", nargs="+", help="result JSON(s) of text.py / ceiling.py")
    p.add_argument("--out", default=None, help="analysis JSON (default: <first result>.analysis.json)")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--traces", action="store_true", help="also read the trace files (signal predictors)")
    p.add_argument("--pred-arms", default="", help="prefix filter of arms for the signal predictors (default all)")
    p.add_argument("--pred-max-tok", type=int, default=200_000)
    args = p.parse_args(argv)
    docs = [json.loads(Path(r).expanduser().read_text()) for r in args.results]
    out = {"inputs": [str(Path(r).expanduser()) for r in args.results], "boot": args.boot, "results": []}
    texts = []
    for d in docs:
        if d.get("runner") == "text":
            a = analyse_text(d, args.boot, args.seed)
            texts.append(d)
            if args.traces:
                a["signal_predictors"] = signal_predictors(d, args.pred_arms, args.pred_max_tok, args.seed)
        elif d.get("runner") == "ceiling":
            a = analyse_ceiling(d, args.boot, args.seed)
        else:
            raise SystemExit(f"unknown runner {d.get('runner')!r}")
        out["results"].append(a)
    out["cross_model"] = cross_model(texts, args.boot, args.seed)
    path = Path(args.out).expanduser() if args.out else Path(args.results[0]).expanduser().with_suffix(".analysis.json")
    path.write_text(json.dumps(ms.clean(out), indent=1))
    print(f"[analyse] wrote {path}")
    return out


if __name__ == "__main__":
    main()
