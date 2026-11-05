"""The round reader (experiments/evolve/analyse_round.py) on synthetic eval_by_query JSONs: discovery, the
strong pairing check, the group statistic with shared rows, the checkpoint- and recipe-level intervals, the
rescue and mechanism columns, factorial main effects and the gate block. Pure Python: no model, no torch."""

import csv
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from evolve import analyse_round as ar  # noqa: E402

N = 40
SHARED = {"heldout": "s512", "nat512p3": "s512", "para1": "s512", "para2": "s512", "para3": "s512", "para4": "s512",
          "updnow512": "u512", "updnew512": "u512", "update": "u512", "updnow2048": "u2048", "updnew2048": "u2048"}


def rows_for(cell: str, n: int = N) -> list[list[int]]:
    """Deterministic (rank, gap) rows; cells with the same SHARED key draw the same rows."""
    key = SHARED.get(cell, cell)
    off = sum(map(ord, key)) % 97
    return [[i % 3, 280 + off + 7 * i] for i in range(n)]


def task_for(cell: str) -> dict:
    t = {"jitter": cell == "jitter", "update": None, "names": "heldout" if cell == "heldout" else "train"}
    for form in ("same", "now", "new"):
        if cell.startswith(f"upd{form}"):
            t["update"] = form
    if cell == "update":
        t["update"] = "new"
    if cell in ("updk2", "updk3"):
        t["update"], t["update_k"] = "same", int(cell[-1])
    if cell.startswith("para"):
        t["paraphrase"] = int(cell[-1])
    return t


def ones(level: float, n: int = N, start: int = 0) -> list[float]:
    """n 0/1 outcomes with round(level·n) ones, starting at row ``start`` (cyclically)."""
    k = round(level * n)
    return [1.0 if (i - start) % n < k else 0.0 for i in range(n)]


def write_cell(d: Path, run: str, cell: str, exact, *, stale=None, windowed=0.0, side_empty=0.0, full=0.99,
               w1s=None, result=None, rows=None, seq_len=None, extra=None):
    exact = ones(exact) if isinstance(exact, float) else exact
    rows = rows or rows_for(cell, len(exact))
    sl = seq_len or (2048 if "2048" in cell else 1024 if "1024" in cell else 512)
    rec = {"result": result, "pairs": 3, "seq_len": sl, "text_file": None, "scatter": 8, "n": len(exact),
           "exact": sum(exact) / len(exact), "per_sequence": [r + [x, x] for r, x in zip(rows, exact)],
           "per_sequence_repeat": [bool(i % 2) for i in range(len(exact))], "task": task_for(cell),
           "controls": {"windowed": {"exact": windowed}, "full": {"exact": full}, "side_empty": {"exact": side_empty}},
           "eval": {"seed": 2024}}
    if task_for(cell)["update"]:
        st = stale if stale is not None else [0.0] * len(exact)
        st = ones(st) if isinstance(st, float) else st
        rec["update"] = {"windowed+side": {"new_exact": rec["exact"], "old_exact": sum(st) / len(st)},
                         "full": {"new_exact": 0.4, "old_exact": 0.3}}
        if stale is not None:
            rec["per_sequence_stale"] = st
    if w1s:
        rec["w1s"] = w1s
    rec.update(extra or {})
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{run}_{cell}.json").write_text(json.dumps(rec))


def opts(**kw):
    o = dict(interval="auto", near=0.95, bootstrap=200, boot_seed=0, drop_loss=0.1, root=".")
    o.update(kw)
    return type("O", (), o)()


def load(d: Path) -> dict:
    runs, voids = ar.load_runs(ar.discover([str(d)]))
    return runs


# ------------------------------------------------------------------------------------------------ helpers
def test_t_quantile_wilson_and_newcombe_against_known_values():
    assert [round(ar.t_quantile(0.975, df), 3) for df in (1, 2, 5, 10, 30)] == [12.706, 4.303, 2.571, 2.228, 2.042]
    assert ar.t_quantile(0.975, 10 ** 6) == pytest.approx(ar.Z, abs=1e-4)
    assert ar.t_quantile(0.025, 4) == pytest.approx(-ar.t_quantile(0.975, 4))
    lo, hi = ar.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    # Newcombe (1998), paired example (12, 9, 2, 21): method 10 gives 0.0112 to 0.2954
    a = [1] * 12 + [1] * 9 + [0] * 2 + [0] * 21
    b = [1] * 12 + [0] * 9 + [1] * 2 + [0] * 21
    lo, hi = ar.newcombe_paired(a, b)
    assert (round(lo, 4), round(hi, 4)) == (0.0112, 0.2954)
    lo, hi = ar.newcombe_paired([1] * 100, [1] * 100)            # at the ceiling: a non-degenerate interval
    assert lo < 0 < hi and hi - lo > 0.05


def test_discovery_skips_training_and_dotted_files_and_aliases_update(tmp_path):
    write_cell(tmp_path, "e11_r11_A11_s1", "nat512p3", 0.9)
    write_cell(tmp_path, "e10b_r10_I1_s0", "update", 0.8)
    (tmp_path / "r11_A11_s1.json").write_text(json.dumps({"args": {}, "curve": []}))    # training JSON
    (tmp_path / "r11_A11_s1.by_query.jitter.json").write_text("{}")
    runs = load(tmp_path)
    assert set(runs) == {"e11_r11_A11_s1", "e10b_r10_I1_s0"}
    assert set(runs["e10b_r10_I1_s0"]) == {"updnew512"} and runs["e10b_r10_I1_s0"]["updnew512"]["raw_cell"] == "update"
    assert ar.run_seed("e11_r11_A11_s4") == 4 and ar.run_seed("r11_I1_s4b") == 4 and ar.run_seed("x") is None
    t = tmp_path / "other"
    write_cell(t, "zz", "rand512p1", 0.5)
    got = ar.from_templates([str(t / "zz_{cell}.json"), str(tmp_path / "e11_r11_A11_s1_{cell}.json")])
    assert set(got) == {"rand512p1", "nat512p3"}


def test_strong_pairing_refuses_and_reports(tmp_path):
    for c in ("rand512p1", "rand512p3", "nat512p1"):
        write_cell(tmp_path, "A_s1", c, 0.8)
    write_cell(tmp_path, "B_s1", "rand512p1", 0.5)
    bad = rows_for("rand512p3")
    bad[5] = [bad[5][0], bad[5][1] + 1]                            # one gap differs
    write_cell(tmp_path, "B_s1", "rand512p3", 0.5, rows=bad)
    write_cell(tmp_path, "B_s1", "nat512p1", ones(0.5, N - 4))     # n differs
    runs = load(tmp_path)
    res = ar.compare(runs, "A_s1", ["B_s1"], opts())
    assert {c for c, row in res["cells"].items() if "exact" in row} == {"rand512p1"}
    g5 = rows_for("rand512p3")[5][1]
    assert res["refused"]["rand512p3:exact"] == f"row 5: (rank, gap) [2, {g5}] vs [2, {g5 + 1}]"
    assert res["refused"]["nat512p1:exact"].startswith("n 40 vs 36")
    assert res["groups"]["grid"]["cells"] == ["rand512p1"] and "rand512p3" in res["groups"]["grid"]["missing"]
    a, b = runs["A_s1"]["rand512p1"], dict(runs["B_s1"]["rand512p1"])
    b["task"] = {**b["task"], "names": "heldout"}
    assert ar.pair_check(a, b).startswith("task")
    b = dict(runs["B_s1"]["rand512p1"], repeat=[True] * N)
    assert ar.pair_check(a, b) == "per_sequence_repeat differs"
    b = dict(runs["B_s1"]["rand512p1"], seed=7)
    assert ar.pair_check(a, b) == "seed 2024 vs 7"


def test_void_evaluations_are_listed_and_left_out(tmp_path):
    write_cell(tmp_path, "A_s1", "rand512p1", 0.8, windowed=0.01)
    write_cell(tmp_path, "A_s1", "rand512p3", 0.8, side_empty=0.02)
    write_cell(tmp_path, "A_s1", "rand512p5", 0.8)
    for c in ("rand512p1", "rand512p3", "rand512p5"):
        write_cell(tmp_path, "B_s1", c, 0.5)
    runs, voids = ar.load_runs(ar.discover([str(tmp_path)]))
    assert sorted(voids) == ["A_s1/rand512p1", "A_s1/rand512p3"]
    res = ar.compare(runs, "A_s1", ["B_s1"], opts())
    assert sorted(res["void"]) == ["rand512p1", "rand512p3"] and res["groups"]["grid"]["cells"] == ["rand512p5"]


def test_cells_sharing_rows_are_averaged_row_by_row(tmp_path):
    """Two cells on the same rows with the same differences carry one cell's information: the group interval
    is that cell's, not √2 narrower. On different rows (independent) it is √2 narrower. The statistic is
    the mean of the per-cell differences either way."""
    a1, a2 = ones(0.7), ones(0.6, start=3)
    for c in ("para1", "para2"):
        write_cell(tmp_path, "A_s1", c, a1 if c == "para1" else a2)
        write_cell(tmp_path, "B_s1", c, ones(0.4, start=11))
    runs = load(tmp_path)
    o = opts(bootstrap=0)
    res = ar.compare(runs, "A_s1", ["B_s1"], o)
    g = res["groups"]["query"]
    d1 = [x - y for x, y in zip(a1, ones(0.4, start=11))]
    d2 = [x - y for x, y in zip(a2, ones(0.4, start=11))]
    assert g["diff"] == pytest.approx((sum(d1) + sum(d2)) / (2 * N)) and g["shared_rows"] == [["para1", "para2"]]
    rows = [(x + y) / 2 for x, y in zip(d1, d2)]
    half = ar.Z * math.sqrt(ar.var(rows) / N)
    assert g["normal"] == pytest.approx([g["diff"] - half, g["diff"] + half])
    # the same differences on independent rows: each cell's variance enters with weight 1/4
    t2 = tmp_path / "indep"
    for c, a in (("rand512p1", a1), ("rand512p3", a2)):
        write_cell(t2, "A_s1", c, a)
        write_cell(t2, "B_s1", c, ones(0.4, start=11))
    gi = ar.compare(load(t2), "A_s1", ["B_s1"], o)["groups"]["grid"]
    half_i = ar.Z * math.sqrt((ar.var(d1) + ar.var(d2)) / 4 / N)
    assert gi["diff"] == pytest.approx(g["diff"]) and gi["normal"][1] - gi["diff"] == pytest.approx(half_i)
    assert gi["shared_rows"] == []


def test_interval_method_switches_near_the_ceiling(tmp_path):
    write_cell(tmp_path, "A_s1", "rand512p1", ones(1.0))
    write_cell(tmp_path, "B_s1", "rand512p1", ones(0.975))          # 39 of 40: near the ceiling
    write_cell(tmp_path, "A_s1", "rand512p3", ones(0.7))
    write_cell(tmp_path, "B_s1", "rand512p3", ones(0.5))
    runs = load(tmp_path)
    res = ar.compare(runs, "A_s1", ["B_s1"], opts())
    c1, c3 = res["cells"]["rand512p1"]["exact"], res["cells"]["rand512p3"]["exact"]
    assert c1["near_edge"] and c1["method"] == "newcombe" and c1["lo"] < 0       # 1 of 40 is no evidence
    assert not c3["near_edge"] and c3["method"] == "normal" and "newcombe" in c3
    g = res["groups"]["grid"]
    assert g["method"] == "normal" and "bootstrap" in g                         # the group's level is 0.85/0.74
    res = ar.compare(runs, "A_s1", ["B_s1"], opts(interval="normal"))
    assert res["cells"]["rand512p1"]["exact"]["method"] == "normal"
    res = ar.compare(runs, "A_s1", ["B_s1"], opts(interval="alt"))
    assert res["cells"]["rand512p3"]["exact"]["method"] == "newcombe" and res["groups"]["grid"]["method"] == "bootstrap"


def test_stale_is_paired_when_recorded_and_lower_is_better(tmp_path):
    for c in ("updsame512", "updnow512", "updnew512"):
        write_cell(tmp_path, "A_s1", c, 0.8, stale=ones(0.1))
        write_cell(tmp_path, "B_s1", c, 0.6, stale=ones(0.5))
    write_cell(tmp_path, "C_s1", "updsame512", 0.6)                 # older evaluations: no per_sequence_stale
    write_cell(tmp_path, "C_s1", "updnow512", 0.6)
    runs = load(tmp_path)
    res = ar.compare(runs, "A_s1", ["B_s1"], opts())
    st = res["groups"]["updates_stale"]
    assert st["diff"] == pytest.approx(-0.4) and st["verdict"] == "better"
    # updsame512 joins updnow/updnew512: one update draw (row_clusters), though its fingerprint differs here
    assert st["shared_rows"] == [["updnew512", "updnow512", "updsame512"]]
    assert res["groups"]["updates_new"]["verdict"] == "better"
    res = ar.compare(runs, "A_s1", ["C_s1"], opts())
    assert "updsame512:stale" in res["refused"] and "per_sequence_stale" in res["refused"]["updsame512:stale"]
    g = res["groups"]["updates_stale"]
    assert g["unpaired_only"] and g["unpaired_diff_mean"] == pytest.approx(0.1)
    assert res["groups"]["updates_new"]["n_cells"] == 2


def test_arm_table_rescue_and_mechanism_columns(tmp_path):
    """--arms with a same-seed --ref; on seed 4 the rescue (I1 s4) and mechanism (row mean of I1 s0, s2, s3)
    columns. The references are not arms."""
    for c in ("rand512p1", "heldout"):
        write_cell(tmp_path, "e11_r11_X_s1", c, 0.9)
        write_cell(tmp_path, "e11_r11_X_s4", c, 0.9)
        write_cell(tmp_path, "ref_I1_s1", c, 0.8)
        write_cell(tmp_path, "ref_I1_s4", c, 0.5)
        for s, lv in ((0, 0.7), (2, 0.8), (3, 0.9)):
            write_cell(tmp_path, f"e10b_r10_I1_s{s}", c, lv)
        write_cell(tmp_path, "e10b_r10_I1_s4", c, 0.55)
    out = tmp_path / "sum.json"
    res = ar.main(["--dir", str(tmp_path), "--arms", "e11_.*", "--ref", "ref_I1_s{seed}", "--bootstrap", "100",
                   "--out", str(out)])
    assert set(res["arms"]) == {"e11_r11_X_s1", "e11_r11_X_s4"}
    assert set(res["arms"]["e11_r11_X_s1"]) == {"ref_I1_s{seed}"}
    s4 = res["arms"]["e11_r11_X_s4"]
    assert s4["ref_I1_s{seed}"]["groups"]["grid"]["diff"] == pytest.approx(0.4)
    assert s4["rescue"]["ref"] == ["e10b_r10_I1_s4"] and s4["rescue"]["groups"]["grid"]["diff"] == pytest.approx(0.35)
    m = s4["mechanism"]
    assert m["ref"] == ["e10b_r10_I1_s0", "e10b_r10_I1_s2", "e10b_r10_I1_s3"]
    assert m["groups"]["grid"]["diff"] == pytest.approx(0.1) and m["groups"]["names"]["diff"] == pytest.approx(0.1)
    assert m["cells"]["rand512p1"]["exact"]["method"] in ("normal", "bootstrap")   # averaged: not 0/1, no Newcombe
    assert "newcombe" not in m["cells"]["rand512p1"]["exact"]
    assert m["per_ref"]["e10b_r10_I1_s0"]["grid"] == pytest.approx(0.2)
    saved = json.loads(out.read_text())["arms"]["e11_r11_X_s4"]["mechanism"]
    assert saved["groups"]["grid"]["diff"] == pytest.approx(0.1)
    res = ar.main(["--dir", str(tmp_path), "--arms", "e11_.*", "--mechanism-refs", "e10b_r10_I1_s0,nope",
                   "--bootstrap", "0"])
    assert "error" in res["arms"]["e11_r11_X_s4"]["mechanism"]
    # a reference evaluated on other sequences is refused, not paired
    write_cell(tmp_path, "e10b_r10_I1_s2", "rand512p1", 0.8, rows=[[0, 999 + i] for i in range(N)])
    res = ar.main(["--dir", str(tmp_path), "--arms", "e11_r11_X_s4", "--bootstrap", "0"])
    mech = res["arms"]["e11_r11_X_s4"]["mechanism"]
    assert mech["refused"]["rand512p1:exact"].startswith("row 0: (rank, gap)")
    assert "grid" not in mech["groups"] and mech["groups"]["names"]["diff"] == pytest.approx(0.1)


def _recipe_dir(tmp_path, levels: dict[str, float]):
    for run, lv in levels.items():
        for c in ar.GRID:
            write_cell(tmp_path, run, c, lv)
    return tmp_path


def test_recipe_level_interval_and_sign_rule(tmp_path):
    lv = {"A_s1": 0.90, "A_s2": 0.85, "A_s4": 0.95, "B_s1": 0.80, "B_s2": 0.80, "B_s4": 0.75,
          "N_s0": 0.70, "N_s2": 0.75, "N_s3": 0.80, "M_s4": 0.60, "M_s4b": 0.65}
    runs = load(_recipe_dir(tmp_path, lv))
    A, B = ["A_s1", "A_s2", "A_s4"], ["B_s1", "B_s2", "B_s4"]
    noise = {"N": ["N_s0", "N_s2", "N_s3"], "M": ["M_s4", "M_s4b"]}
    rc = ar.recipe_compare(runs, A, B, noise, opts())
    g = rc["grid"]
    assert g["diff"] == pytest.approx(0.9 - 0.783333333, abs=1e-6)
    ss = ar.var([0.70, 0.75, 0.80]) * 2 + ar.var([0.60, 0.65]) * 1
    s = math.sqrt(ss / 3)
    assert g["noise"]["s_seed"] == pytest.approx(s) and g["noise"]["df"] == 3
    half = ar.t_quantile(0.975, 3) * s * math.sqrt(1 / 3 + 1 / 3)
    assert [g["lo"], g["hi"]] == pytest.approx([g["diff"] - half, g["diff"] + half])
    assert g["per_seed"] == pytest.approx({"1": 0.1, "2": 0.05, "4": 0.2}) and g["sign_every_seed"]
    assert g["verdict"] == ("better" if g["lo"] > 0 else "tie")
    assert "missing" in rc["names"] and "missing" in rc["pick"]      # no held-out cells in these runs
    # one seed with the opposite sign: never "better", whatever the interval
    runs2 = load(_recipe_dir(tmp_path / "flip", {**lv, "A_s2": 0.70}))
    g2 = ar.recipe_compare(runs2, A, B, {"N": noise["N"]}, opts())["grid"]
    assert not g2["sign_every_seed"] and not g2["verdict"].startswith("better")
    g3 = ar.recipe_compare(runs, A, B, {}, opts())["grid"]
    assert "lo" not in g3 and g3["verdict"].startswith("no s_seed")


def test_recipe_ranking_and_stage_b_default(tmp_path):
    cells = ar.GRID + ["heldout", "para1", "para2", "para4", "updsame512"]
    levels = {"I1_s1": (0.800, 0.500), "I1_s2": (0.825, 0.525), "P_s1": (0.950, 0.200), "P_s2": (0.925, 0.225),
              "Q_s1": (0.900, 0.525), "Q_s2": (0.875, 0.500), "N_s1": (0.800, 0.500), "N_s1b": (0.825, 0.500)}
    for run, (lv, upd) in levels.items():
        for c in cells:
            write_cell(tmp_path, run, c, upd if c.startswith("upd") else lv, stale=0.1 if c.startswith("upd") else None)
    res = ar.main(["--dir", str(tmp_path), "--recipe", "I1=I1_s1,I1_s2", "--recipe", "P=P_s1,P_s2",
                   "--recipe", "Q=Q_s1,Q_s2", "--recipe-ref", "I1", "--noise", "N=N_s1,N_s1b", "--bootstrap", "0"])
    assert [r["recipe"] for r in res["ranking"]] == ["P", "Q"]
    assert res["ranking"][0]["updates_new"] == "worse"
    assert res["stage_b_pick"]["top"] == "P" and res["stage_b_pick"]["top_worse_on_updates"]
    assert res["stage_b_pick"]["also"] == "Q"
    pk = res["recipe_compare"]["P vs I1"]["pick"]
    assert pk["diff"] == pytest.approx(0.125) and pk["cells"]["names"] == ["heldout"]
    with pytest.raises(SystemExit, match="not a --recipe"):
        ar.main(["--dir", str(tmp_path), "--recipe", "I1=I1_s1", "--recipe-ref", "Z"])
    with pytest.raises(SystemExit, match="no runs"):
        ar.main(["--dir", str(tmp_path), "--recipe", "I1=I1_s1,nope", "--recipe-ref", "I1"])


def test_factorial_main_effects(tmp_path):
    """A 2^3 design on two seeds with y = base + seed + 0.05·x_A − 0.025·x_B (+0 for C): effects (on − off)
    +0.10, −0.05 and 0, the residual interval of zero width; with noise sets, the s_seed interval."""
    design = tmp_path / "design.csv"
    lv, rows = {}, []
    for s, base in ((1, 0.70), (4, 0.60)):
        for i in range(8):
            xa, xb, xc = (1 if i & 1 else -1), (1 if i & 2 else -1), (1 if i & 4 else -1)
            run = f"F{i}_s{s}"
            lv[run] = base + 0.05 * xa - 0.025 * xb
            rows.append({"run": run, "A": "+" if xa > 0 else "-", "B": "on" if xb > 0 else "off", "C": int(xc > 0)})
    lv.update({"N_s0": 0.70, "N_s2": 0.72})
    with open(design, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "A", "B", "C"])
        w.writeheader()
        w.writerows(rows)
    # levels on a 40-row cell are multiples of 1/40: put each run's level exactly on one grid cell
    for run, y in lv.items():
        write_cell(tmp_path, run, "rand512p1", ones(round(y * 40) / 40))
    runs = load(tmp_path)
    factors, des = ar.read_design(str(design))
    fac = ar.factorial(runs, factors, des, {"N": ["N_s0", "N_s2"]}, opts())
    eff = fac["groups"]["grid"]["effects"]
    def y(f, s, sign):
        return ar.mean([runs[r["run"]]["rand512p1"]["level_exact"] for r in des
                        if r["seed"] == s and r["levels"][f] * sign > 0])
    want = {f: sum(y(f, s, 1) - y(f, s, -1) for s in (1, 4)) / 2 for f in "ABC"}
    for f in "ABC":
        assert eff[f]["effect"] == pytest.approx(want[f]) and eff[f]["ols_effect"] == pytest.approx(want[f])
        assert eff[f]["n_on"] == 8 and eff[f]["n_off"] == 8 and set(eff[f]["per_seed"]) == {"1", "4"}
    assert eff["A"]["effect"] == pytest.approx(0.1) and eff["B"]["effect"] == pytest.approx(-0.05)
    assert eff["C"]["effect"] == pytest.approx(0.0, abs=1e-12)
    assert eff["A"]["residual"]["verdict"] == "better" and eff["B"]["residual"]["verdict"] == "worse"
    assert eff["A"]["residual"]["df"] == 16 - 5 and fac["groups"]["grid"]["residual_sd"] == pytest.approx(0.0, abs=1e-9)
    sn = eff["A"]["seed_noise"]
    s = math.sqrt(ar.var([runs["N_s0"]["rand512p1"]["level_exact"], runs["N_s2"]["rand512p1"]["level_exact"]]))
    half = ar.t_quantile(0.975, 1) * s * math.sqrt(1 / 8 + 1 / 8)
    assert sn["df"] == 1 and sn["hi"] - sn["lo"] == pytest.approx(2 * half)
    bad = tmp_path / "bad.csv"
    bad.write_text("run,A\nF0_s1,maybe\n")
    with pytest.raises(SystemExit, match="level"):
        ar.read_design(str(bad))


def test_gate_block_and_late_drop(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    curve = [{"step": s, "loss": l} for s, l in ((600, 0.5), (1200, 0.3), (1800, 0.05), (2400, 0.12), (3000, 0.04),
                                                  (3600, 0.02))]
    (train / "r_A_s1.json").write_text(json.dumps({"curve": curve}))
    early = [{"step": 600, "loss": 0.02}, {"step": 1200, "loss": 0.01}]
    (train / "r_B_s1.json").write_text(json.dumps({"curve": early}))
    ev = tmp_path / "ev"
    for run, cov in (("A_s1", 0.9), ("B_s1", 0.8)):
        res_path = str(train / f"r_{run.split('_')[0]}_s1.json")
        for c in ("nat512p1", "nat512p3", "rand512p1", "jitter"):
            w = {"fact_coverage": cov if c.startswith("nat") else cov - 0.1, "onset_in_fact": 0.95,
                 "gate_auc_digit1_vs_filler": 0.5, "onset_density": 0.06}
            write_cell(ev, run, c, 0.8, w1s=w, result=res_path)
    runs = load(ev)
    ga = ar.gate_block(runs["A_s1"], opts())
    assert ga["natural"]["fact_coverage"] == pytest.approx(0.9) and ga["natural"]["cells"] == 2
    assert ga["natural_minus_random"]["fact_coverage"] == pytest.approx(0.1) and ga["jitter"]["cells"] == 1
    ld = ga["late_drop"]                                           # 6 logged points: too few to read a step
    assert ld["step"] is None and ld["status"] == "too few points" and ld["log_every"] == 600 and ld["points"] == 6
    res = ar.compare(runs, "A_s1", ["B_s1"], opts(bootstrap=0))
    assert res["gate"]["diff"]["late_drop_step"] is None
    assert res["gate"]["diff"]["late_drop_status"] == ["too few points", "too few points"]
    assert res["gate"]["diff"]["natural"]["fact_coverage"] == pytest.approx(0.1)
    ld = ar.gate_block(runs["A_s1"], opts(drop_min_points=0, drop_window=3))["late_drop"]
    assert ld["status"] == "learned" and ld["step"] == 3000           # median of 3 from 3000 on: ≤ 0.1
    (train / "r_A_s1.json").write_text(json.dumps({"curve": curve[:-2] + [{"step": 3000, "loss": 0.3}] * 2}))
    ld = ar.gate_block(runs["A_s1"], opts(drop_min_points=0, drop_window=3))["late_drop"]
    assert ld["step"] is None and ld["status"] == "never"


def test_cli_writes_json_and_table(tmp_path):
    for c in ("rand512p1", "para1", "para2", "jitter"):
        write_cell(tmp_path, "A_s1", c, 0.9, full=0.57 if c == "jitter" else 0.99)
        write_cell(tmp_path, "B_s1", c, 0.7, full=0.57 if c == "jitter" else 0.99)
    out, tab = tmp_path / "o" / "s.json", tmp_path / "t.txt"
    ar.main(["--dir", str(tmp_path), "--compare", "A_s1", "B_s1", "--out", str(out), "--table", str(tab),
             "--bootstrap", "50"])
    res = json.loads(out.read_text())
    c = res["compare"][0]
    assert c["groups"]["query"]["shared_rows"] == [["para1", "para2"]] and c["beside"]["jitter"]["full"] == 0.57
    text = tab.read_text()
    assert "A_s1 vs B_s1" in text and "query" in text and "grid" in text
    assert res["runs"]["A_s1"]["cells"]["para1"]["fp"] == res["runs"]["B_s1"]["cells"]["para1"]["fp"]
    with pytest.raises(SystemExit, match="no run"):
        ar.main(["--dir", str(tmp_path), "--compare", "A_s1", "Z_s1"])
    assert ar.main(["--dir", str(tmp_path), "--list"]) == {"runs": ["A_s1", "B_s1"]}


# ------------------------------------------------------------------------- review 3 (Q8–Q15) regressions
def _updates_runs(tmp_path, levels, void=None, cells=None):
    """Runs with grid/names/query cells at ``lv`` and update cells at ``upd`` (exact) with stale 0.1."""
    cells = cells or ar.GRID[:3] + ["heldout", "para1", "para2", "para4", "updsame512", "updnow512"]
    for run, (lv, upd) in levels.items():
        for c in cells:
            w = 0.025 if void and (run, c) in void else 0.0
            write_cell(tmp_path, run, c, upd if c.startswith("upd") else lv, windowed=w,
                       stale=0.1 if c.startswith("upd") else None)


def test_stage_b_pick_never_reads_an_unknown_updates_verdict_as_not_worse(tmp_path):
    """Q8: without --noise, or when void/absent cells leave no Updates verdict, the top recipe's Updates state is
    'unknown' (top_worse_on_updates None, a warning), the best recipe known not to be worse is named beside it,
    and the ranking row says the verdict is missing instead of carrying None."""
    levels = {"I1_s1": (0.80, 0.70), "I1_s2": (0.82, 0.70), "X_s1": (0.95, 0.20), "X_s2": (0.93, 0.20),
              "Y_s1": (0.88, 0.70), "Y_s2": (0.86, 0.70), "N_s1": (0.80, 0.70), "N_s1b": (0.83, 0.68)}
    _updates_runs(tmp_path, levels)
    base = ["--dir", str(tmp_path), "--recipe", "I1=I1_s1,I1_s2", "--recipe", "X=X_s1,X_s2", "--recipe",
            "Y=Y_s1,Y_s2", "--recipe-ref", "I1", "--bootstrap", "0"]
    res = ar.main(base)                                            # no --noise: nothing is known about Updates
    sb = res["stage_b_pick"]
    assert sb["top"] == "X" and sb["top_updates"] == "unknown" and sb["top_worse_on_updates"] is None
    assert sb["also"] is None and "unknown" in sb["warning"] and sb["unknown_updates"] == ["X", "Y"]
    # void update cells in X: the Updates groups have no cells; the ranking says so
    _updates_runs(tmp_path / "v", levels, void={("X_s1", "updsame512"), ("X_s1", "updnow512")})
    res = ar.main([a if a != str(tmp_path) else str(tmp_path / "v") for a in base] + ["--noise", "N=N_s1,N_s1b"])
    row = next(r for r in res["ranking"] if r["recipe"] == "X")
    assert row["updates_new"].startswith("missing") and ar.updates_state(row) == "unknown"
    sb = res["stage_b_pick"]
    assert sb["top"] == "X" and sb["top_worse_on_updates"] is None and sb["also"] == "Y"   # Y: known, a tie
    assert "WARNING" in ar.table(res)
    # the three states, directly
    rk = [{"recipe": "A", "updates_new": "worse", "updates_stale": "tie"},
          {"recipe": "B", "updates_new": "no s_seed (give --noise)", "updates_stale": "tie"},
          {"recipe": "C", "updates_new": "tie (sign not on every seed)", "updates_stale": "better"}]
    sb = ar.stage_b_pick(rk)
    assert sb["top_worse_on_updates"] is True and sb["also"] == "C" and sb["unknown_updates"] == ["B"]


def _curve(losses, every=50):
    return [{"step": every * (i + 1), "loss": x} for i, x in enumerate(losses)]


def _late_drop(tmp_path, curve, **kw):
    tr = tmp_path / "train.json"
    tr.write_text(json.dumps({"curve": curve, "needle_curriculum": "1:1500,2:1500"}))
    cells = {"x": {"result": str(tr), "path": str(tmp_path / "ev" / "x.json")}}
    return ar.late_drop(cells, opts(**kw))


def test_late_drop_reads_a_smoothed_curve_and_tells_never_from_not_logged(tmp_path):
    """Q9: single-batch spikes neither move the step (a 0.98 spike among 0.001s, even as the last point) nor make
    a failed run read as learned; 'never', 'too few points' and 'not logged' are distinct statuses."""
    rng = __import__("random").Random(0)
    learned = [0.8 - 0.006 * i for i in range(60)] + [0.002 + 0.002 * rng.random() for _ in range(60)]
    for i in (70, 85, 100, 119):                                   # isolated spikes, one of them last
        learned[i] = 0.98
    ld = _late_drop(tmp_path, _curve(learned))
    assert ld["status"] == "learned" and ld["points"] == 120 and ld["window"] == 6 and ld["curriculum_end"] == 3000
    assert 3000 <= ld["step"] <= 3400                              # the drop at point 60 (step 3000), not 6000
    failed = [0.3 + 0.1 * rng.random() for _ in range(120)]
    for i in range(0, 120, 9):
        failed[i] = 0.001                                          # isolated lucky batches
    failed[-1] = 0.001
    ld = _late_drop(tmp_path, _curve(failed))
    assert ld["status"] == "never" and ld["step"] is None
    ld = _late_drop(tmp_path, _curve(learned[:10], every=600))
    assert ld["status"] == "too few points" and ld["step"] is None and ld["points"] == 10
    assert _late_drop(tmp_path, [])["status"] == "not logged"
    mean_ld = _late_drop(tmp_path, _curve(learned), drop_smooth="mean")   # a mean lets the last spike decide
    assert mean_ld["status"] == "never"


def test_gate_block_differences_use_the_cells_both_sides_have(tmp_path):
    """Q11: the reference has 2 of the 4 natural cells; the arm's gate means over its 4 would mix the cell sets.
    The Δ is taken over the 2 shared cells, on both sides."""
    cov = {"nat512p1": 0.9, "nat512p3": 0.9, "nat2048p1": 0.5, "nat2048p3": 0.5}
    for c, v in cov.items():
        w = {"fact_coverage": v, "onset_in_fact": 0.9, "gate_auc_digit1_vs_filler": 0.5, "onset_density": 0.05}
        write_cell(tmp_path, "A_s1", c, 0.8, w1s=w)
        if c.startswith("nat512"):
            write_cell(tmp_path, "R_s1", c, 0.8, w1s={**w, "fact_coverage": v - 0.1})
    runs = load(tmp_path)
    assert ar.gate_block(runs["A_s1"], opts())["natural"]["fact_coverage"] == pytest.approx(0.7)   # alone: 4 cells
    g = ar.compare(runs, "A_s1", ["R_s1"], opts(bootstrap=0))["gate"]
    assert g["a"]["natural"]["cells"] == g["ref"]["natural"]["cells"] == 2
    assert g["diff"]["natural"]["fact_coverage"] == pytest.approx(0.1)   # not 0.7 − 0.8
    a, b = dict(g["a"]), dict(g["ref"])
    b["natural"] = {**b["natural"], "cell_names": ["nat512p1"]}
    assert "natural" not in ar.gate_diff(a, b) and ar.gate_diff(a, b)["cells_differ"]["natural"]


def test_update_cells_of_one_draw_share_rows_even_with_shifted_gaps(tmp_path):
    """Q12: updsame2048 and updnow2048 draw the same sequences; the update stem's length shifts the gaps, so their
    fingerprints differ. They are clustered row by row (the variance carries their correlation); cells with
    ≥ 95% of rows equal are too; independent cells are not."""
    base = rows_for("updsame2048")
    shifted = [[r, g + 2] for r, g in base]
    a = ones(0.7, start=0)
    b = ones(0.4, start=5)
    write_cell(tmp_path, "A_s1", "updsame2048", a, rows=base)
    write_cell(tmp_path, "A_s1", "updnow2048", a, rows=shifted)
    write_cell(tmp_path, "B_s1", "updsame2048", b, rows=base)
    write_cell(tmp_path, "B_s1", "updnow2048", b, rows=shifted)
    runs = load(tmp_path)
    assert runs["A_s1"]["updsame2048"]["fp"] != runs["A_s1"]["updnow2048"]["fp"]
    g = ar.compare(runs, "A_s1", ["B_s1"], opts(bootstrap=0))["groups"]["updates_new"]
    assert g["shared_rows"] == [["updnow2048", "updsame2048"]]
    ov = g["row_overlap"]["updsame2048~updnow2048"]
    assert ov["same_draw"] and ov["rank_equal"] == 1.0 and ov["rows_equal"] == 0.0
    d = [x - y for x, y in zip(a, b)]                              # identical d on both: one cell's information
    half = ar.Z * math.sqrt(ar.var(d) / N)
    assert g["normal"][1] - g["diff"] == pytest.approx(half)
    # ≥ 95% of rows equal (39 of 40) on non-update cells: joined, with the overlap reported
    r2 = rows_for("rand512p1")
    r3 = [list(x) for x in r2]
    r3[0][1] += 1
    t = tmp_path / "near"
    for run, v in (("A_s1", a), ("B_s1", b)):
        write_cell(t, run, "rand512p1", v, rows=r2)
        write_cell(t, run, "rand512p3", v, rows=r3)
        write_cell(t, run, "rand512p5", v)                         # other rows: independent
    g = ar.compare(load(t), "A_s1", ["B_s1"], opts(bootstrap=0))["groups"]["grid"]
    assert g["shared_rows"] == [["rand512p1", "rand512p3"]]
    assert g["row_overlap"]["rand512p1~rand512p3"]["rows_equal"] == pytest.approx(39 / 40)


def test_names_group_of_one_binary_cell_decides_as_the_cell_does(tmp_path):
    """Q13: 4 discordant rows of 384, all favouring A, near the ceiling: Newcombe says tie for the held-out cell;
    the Names group (that one cell) must say the same, not the bootstrap's 'better'."""
    n = 384
    b = [1.0] * n
    for i in range(10):
        b[i] = 0.0
    a = list(b)
    for i in range(4):
        a[i] = 1.0
    write_cell(tmp_path, "A_s1", "heldout", a)
    write_cell(tmp_path, "B_s1", "heldout", b)
    res = ar.compare(load(tmp_path), "A_s1", ["B_s1"], opts(bootstrap=2000))
    cell, grp = res["cells"]["heldout"]["exact"], res["groups"]["names"]
    assert cell["method"] == "newcombe" and cell["verdict"] == "tie"
    assert grp["method"] == "newcombe" and grp["verdict"] == "tie" and grp["lo"] == pytest.approx(cell["lo"])
    assert grp["bootstrap"][0] > 0                                 # the bootstrap alone would have said better


def test_missing_controls_are_not_clean_and_modified_evaluations_do_not_pair(tmp_path):
    """Q14: no windowed control → unverified, left out of every statistic (listed); no side_empty control →
    kept but listed; an evaluation with eval-time modifications is refused against a plain one unless
    --allow-modified, and every run's modifications are reported."""
    for c in ("rand512p1", "rand512p3", "rand512p5"):
        write_cell(tmp_path, "B_s1", c, 0.5)
    write_cell(tmp_path, "A_s1", "rand512p1", 0.8, extra={"controls": {"full": {"exact": 0.99}}})   # --no-controls
    write_cell(tmp_path, "A_s1", "rand512p3", 0.8, extra={"controls": {"windowed": {"exact": 0.0},
                                                                         "full": {"exact": 0.99}}})
    write_cell(tmp_path, "A_s1", "rand512p5", 0.8)
    rep = {}
    runs, voids = ar.load_runs(ar.discover([str(tmp_path)]), rep)
    assert voids == [] and rep["unverified"] == ["A_s1/rand512p1"]
    assert sorted(rep["controls_unknown"]) == ["A_s1/rand512p1", "A_s1/rand512p3"]
    res = ar.compare(runs, "A_s1", ["B_s1"], opts(bootstrap=0))
    assert res["unverified"] == ["rand512p1"] and res["groups"]["grid"]["cells"] == ["rand512p3", "rand512p5"]
    # E-lex style: the checkpoint evaluated with --w1s-param against the plain one
    t = tmp_path / "mod"
    for c in ("rand512p1", "rand512p3"):
        write_cell(t, "P_s1", c, 0.5)
        write_cell(t, "M_s1", c, 0.8, extra={"w1s_param": ["loop.log_beta_lex=-30"], "w1s_ctrl": "shuffle"})
    rep = {}
    runs, _ = ar.load_runs(ar.discover([str(t)]), rep)
    assert rep["modified"]["M_s1"]["rand512p1"] == {"w1s_param": ["loop.log_beta_lex=-30"], "w1s_ctrl": "shuffle"}
    res = ar.compare(runs, "M_s1", ["P_s1"], opts(bootstrap=0))
    assert "grid" not in res["groups"] and res["refused"]["rand512p1:exact"].startswith("eval-time modifications")
    res = ar.compare(runs, "M_s1", ["P_s1"], opts(bootstrap=0, allow_modified=True))
    assert res["groups"]["grid"]["diff"] == pytest.approx(0.3)
    out = ar.main(["--dir", str(t), "--compare", "M_s1", "P_s1", "--bootstrap", "0"])
    assert out["checks"]["modified"]["M_s1"] and "MODIFIED evaluation M_s1" in ar.table(out)
    lv, info = ar.levels(runs, ["M_s1", "P_s1"], "grid")
    assert lv is None and "eval-time modifications" in info["left_out"]["rand512p1"]


def _fac_design(tmp_path, rows, factors, lv, name="design.csv"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run"] + factors)
        w.writeheader()
        w.writerows(rows)
    for run, y in lv.items():
        write_cell(tmp_path, run, "rand512p1", ones(round(y * 40) / 40))
    return path


def _fractional(gen):
    """A 2^(4−1) design in A, B, C (full) and D = gen(A, B, C), on seeds 1 and 4; y = base + 0.1·A + 0.1·A·B."""
    rows, lv = [], {}
    for s, base in ((1, 0.5), (4, 0.45)):
        for i in range(8):
            a, b, c = (1 if i & 1 else -1), (1 if i & 2 else -1), (1 if i & 4 else -1)
            d = gen(a, b, c)
            run = f"F{i}_s{s}"
            lv[run] = base + 0.1 * a + 0.1 * a * b
            rows.append({"run": run, "A": int(a > 0), "B": int(b > 0), "C": int(c > 0), "D": int(d > 0)})
    return rows, lv


def test_factorial_checks_resolution_aliases_and_missing_runs(tmp_path):
    """Q10: a resolution-IV design (D = ABC) puts the A×B interaction in no main effect; a resolution-III one
    (D = AB) is warned, and D carries the alias A×B; a resolution-II design (D = A) or unbalanced columns are
    refused. With a run missing, the effect is the least-squares one (centred intervals), and a factor that the
    gaps leave without contrast is reported not estimable."""
    rows, lv = _fractional(lambda a, b, c: a * b * c)
    path = _fac_design(tmp_path, rows, list("ABCD"), lv)
    runs = load(tmp_path)
    fac = ar.factorial(runs, *ar.read_design(str(path)), {}, opts())
    assert fac["design"]["resolution"] == "IV" and fac["design"]["words"] == ["A·B·C·D"]
    assert not fac["design"]["warnings"] and fac["design"]["2fi_vs_2fi"]["A×B ~ C×D"] == 1.0
    eff = fac["groups"]["grid"]["effects"]
    assert eff["A"]["effect"] == pytest.approx(0.2) and eff["D"]["effect"] == pytest.approx(0.0, abs=1e-9)
    t3 = tmp_path / "r3"
    rows, lv = _fractional(lambda a, b, c: a * b)
    path = _fac_design(t3, rows, list("ABCD"), lv)
    fac = ar.factorial(load(t3), *ar.read_design(str(path)), {}, opts())
    assert fac["design"]["resolution"] == "III" and "A·B·D" in fac["design"]["words"]
    assert fac["design"]["warnings"] and fac["groups"]["grid"]["effects"]["D"]["aliases"] == {"A×B": 1.0}
    text = ar.table({"void": [], "settings": {}, "factorial": fac})
    assert "resolution III" in text and "WARNING" in text and "aliased: A×B" in text
    for gen, msg in ((lambda a, b, c: a, "aliased with"), (lambda a, b, c: 1, "unbalanced")):
        t2 = tmp_path / f"bad{msg[:3]}"
        rows, lv = _fractional(gen)
        path = _fac_design(t2, rows, list("ABCD"), lv)
        with pytest.raises(SystemExit, match=msg):
            ar.factorial(load(t2), *ar.read_design(str(path)), {}, opts())
    # a missing run: y depends on A and B only; the least-squares effects stay exact, the raw ones do not
    rows, lv = _fractional(lambda a, b, c: a * b * c)
    lv = {r: y + 0.05 * (1 if rows[i]["B"] else -1) for i, (r, y) in enumerate(lv.items())}
    tm = tmp_path / "miss"
    path = _fac_design(tm, rows, list("ABCD"), lv)
    (tm / "F3_s4_rand512p1.json").unlink()
    fac = ar.factorial(load(tm), *ar.read_design(str(path)), {}, opts())
    assert fac["missing_runs"] == ["F3_s4"] and fac["not_estimable"] == [] and fac["induced_correlation"]
    c = fac["groups"]["grid"]["effects"]["C"]
    assert c["effect"] == c["ols_effect"] and c["raw_effect"] != pytest.approx(c["effect"], abs=1e-6)
    r = c["residual"]
    assert (r["lo"] + r["hi"]) / 2 == pytest.approx(c["ols_effect"])
    # every "D on" run missing: D is constant in what is left, so not estimable
    tn = tmp_path / "nest"
    path = _fac_design(tn, rows, list("ABCD"), lv)
    for i, row in enumerate(rows):
        if row["D"]:
            (tn / f"{row['run']}_rand512p1.json").unlink()
    fac = ar.factorial(load(tn), *ar.read_design(str(path)), {}, opts())
    assert fac["not_estimable"] == ["D"] and fac["groups"]["grid"]["effects"]["D"]["not_estimable"]
    assert "NOT ESTIMABLE" in ar.table({"void": [], "settings": {}, "factorial": fac})


def _simulate_coverage(a_seeds, b_seeds, noise_seeds, reps=4000, s_seed=0.05, s_run=0.03, seed=0):
    """Null simulations of level = seed effect + run noise: the share of intervals covering 0, for the
    estimator recipe_interval chooses (auto) and for the between-run one."""
    rng = __import__("random").Random(seed)
    hit = {"auto": 0, "between": 0}
    est = None
    for _ in range(reps):
        beta = {}

        def x(s):
            return beta.setdefault(s, rng.gauss(0, s_seed)) + rng.gauss(0, s_run)
        A = [x(s) for s in a_seeds]
        B = [x(s) for s in b_seeds]
        sets = [[(s, x(s)) for s in ss] for ss in noise_seeds]
        d = sum(A) / len(A) - sum(B) / len(B)
        noise = ar.seed_noise(sets)
        ra = [f"A_s{s}" for s in a_seeds]
        rb = [f"B_s{s}" for s in b_seeds]
        use, between = ar.recipe_interval(noise, len(A), len(B), ar.seed_coef(ra, rb))
        est = use["estimator"]
        hit["auto"] += abs(d) <= use["half"]
        hit["between"] += abs(d) <= between["half"]
    return {k: v / reps for k, v in hit.items()}, est


def test_recipe_interval_removes_the_seed_effect_and_covers_at_its_level():
    """Q15: on the same seeds the seed effect cancels in d̄, but the between-run s_seed includes it: that interval
    covers ~99.8%. With same-seed replicates in the noise sets (df ≥ 3) the within-seed estimator covers 95%,
    and the seed-aware one (seed compositions that differ) about 95%; with no replicates the between-run
    interval is kept."""
    noise = [[1, 2, 4, 4], [0, 0, 3, 3], [5, 5, 6]]                # df_within 1 + 2 + 1 = 4
    cov, est = _simulate_coverage([1, 2, 4], [1, 2, 4], noise)
    assert est == "within-seed" and 0.935 <= cov["auto"] <= 0.965 and cov["between"] > 0.99
    cov, est = _simulate_coverage([1, 2, 4], [1, 2, 4, 4], noise)
    assert est == "seed-aware" and 0.93 <= cov["auto"] <= 0.975 and cov["between"] > 0.98
    cov, est = _simulate_coverage([4], [0, 1, 2, 3], noise)          # disjoint seeds: σ_seed does not cancel
    assert est == "seed-aware" and 0.93 <= cov["auto"] <= 0.975
    cov, est = _simulate_coverage([1, 2, 4], [1, 2, 4], [[0, 1, 2, 3]], reps=500)
    assert est == "between"                                         # no same-seed replicate: as pre-registered
    # the estimator's arithmetic on known values
    n = ar.seed_noise([[(1, 0.5), (1, 0.7), (2, 0.9)]])
    assert n["df_within"] == 1 and n["s_within"] == pytest.approx(math.sqrt(0.02))
    assert n["df"] == 2 and n["s_seed"] == pytest.approx(math.sqrt(ar.var([0.5, 0.7, 0.9])))
    assert ar.seed_coef(["A_s1", "A_s2"], ["B_s1", "B_s2"]) == 0.0
    assert ar.seed_coef(["A_s4"], ["B_s0", "B_s1"]) == pytest.approx(1 / 1 + 1 / 2)


def test_recipe_compare_reports_the_estimator_and_the_between_interval(tmp_path):
    lv = {"A_s1": 0.90, "A_s2": 0.85, "B_s1": 0.80, "B_s2": 0.76,
          "N_s1": 0.70, "N_s1b": 0.72, "N_s2": 0.80, "N_s2b": 0.79, "N_s3": 0.75, "N_s3b": 0.78}
    runs = load(_recipe_dir(tmp_path, lv))
    noise = {"N": ["N_s1", "N_s1b", "N_s2", "N_s2b", "N_s3", "N_s3b"]}
    g = ar.recipe_compare(runs, ["A_s1", "A_s2"], ["B_s1", "B_s2"], noise, opts())["grid"]
    x = {r: runs[r]["rand512p1"]["level_exact"] for r in noise["N"]}             # levels are k/40
    sw = math.sqrt(sum((x[f"N_s{s}"] - x[f"N_s{s}b"]) ** 2 / 2 for s in (1, 2, 3)) / 3)
    assert g["interval"]["estimator"] == "within-seed" and g["noise"]["s_within"] == pytest.approx(sw, abs=1e-9)
    assert g["hi"] - g["diff"] == pytest.approx(ar.t_quantile(0.975, 3) * sw * 1.0)
    assert "between" in g and g["between"]["hi"] - g["between"]["lo"] > g["hi"] - g["lo"]
    g2 = ar.recipe_compare(runs, ["A_s1", "A_s2"], ["B_s1", "B_s2"], noise, opts(s_seed="between"))["grid"]
    assert g2["interval"]["estimator"] == "between" and g2["lo"] == pytest.approx(g2["between"]["lo"])
