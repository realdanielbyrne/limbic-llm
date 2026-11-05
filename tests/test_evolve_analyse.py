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
    assert st["shared_rows"] == [["updnew512", "updnow512"]] and res["groups"]["updates_new"]["verdict"] == "better"
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
    assert ga["late_drop"]["step"] == 3000 and ga["late_drop"]["log_every"] == 600     # 2400 is above 0.1
    res = ar.compare(runs, "A_s1", ["B_s1"], opts(bootstrap=0))
    assert res["gate"]["diff"]["late_drop_step"] == 3000 - 600
    assert res["gate"]["diff"]["natural"]["fact_coverage"] == pytest.approx(0.1)
    (train / "r_A_s1.json").write_text(json.dumps({"curve": curve[:-2] + [{"step": 3000, "loss": 0.3}]}))
    assert ar.gate_block(runs["A_s1"], opts())["late_drop"]["step"] is None


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
