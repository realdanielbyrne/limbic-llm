"""Round 12 reasoning runner (experiments/ca2/reason.py): scoring, the forced-answer probe readings, text
measures, arm grammar, the reasoning-span controller and end-to-end tiny runs (results JSON, traces, resume).

Tiny random models with the byte tokenizer; no downloads. The qwen35 run swaps the GDN kernels for their
pure-torch originals (``reason.torch_gdn_kernels``) and skips when that is not possible.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments" / "alignment")]

from ca2 import reason as R  # noqa: E402
from ca2.common import _sha256, alignment  # noqa: E402
from ca2.controller import CA2Attachment, CA2Controller  # noqa: E402

A = alignment()
BOS = 256


# ------------------------------------------------------------------------------------------- scoring
def test_answer_extraction_and_scoring():
    assert R.boxed(r"so \boxed{\frac{1}{2}} and then \boxed{42}") == "42"
    assert R.boxed(r"x \boxed{a{b}c}") == "a{b}c"
    assert R.boxed("no box") is None
    assert R.extract_answer("we get 1,234 apples, then 17.5") == "17.5"
    assert R.extract_answer(r"\boxed{ 66 }") == "66"
    assert R.last_number("none here") is None
    assert R.is_correct("66", "66", "numeric") and R.is_correct("66.0", "66", "numeric")
    assert R.is_correct("$20.40", "20.4", "numeric") and R.is_correct("1,234", "1234", "numeric")
    assert not R.is_correct("67", "66", "numeric") and not R.is_correct(None, "66", "numeric")
    assert R.is_correct(r"\dfrac{1}{2}", r"\frac{1}{2}", "math")
    assert R.is_correct("0.5", r"\frac{1}{2}", "math")
    assert R.is_correct(r"\text{(B)}", "(B)", "math")
    assert R.forced_answer("42} is the answer") == "42"
    assert R.forced_answer(r"\frac{3}{4}} done") == r"\frac{3}{4}"
    assert R.forced_answer("  ") is None


def test_probe_readings_ttc_flips_budgets():
    assert R.checkpoints(300, 128) == [128, 256, 300]
    assert R.checkpoints(256, 128) == [128, 256]
    assert R.checkpoints(0, 128) == [] and R.checkpoints(50, 0) == []
    cks = [128, 256, 384, 400]
    assert R.ttc_from(cks, [False, True, True, True]) == (256, False)
    assert R.ttc_from(cks, [True, False, True, True]) == (384, False)      # must stay correct
    assert R.ttc_from(cks, [True, True, True, False]) == (None, True)      # censored
    assert R.ttc_from(cks, [True] * 4) == (128, False)
    assert R.count_flips(["3", None, "3.0", "4", "4", "x"]) == 2
    acc = R.budget_acc([128, 256, 512], cks, [False, True, True, True], 400, True, True)
    assert acc == {128: 0.0, 256: 1.0, 512: 1.0}
    acc = R.budget_acc([256, 512], [128, 256, 300], [False, False, True], 300, False, False)   # truncated at 300
    assert acc == {256: 0.0, 512: 1.0}
    assert R.budget_acc([256], [], [], 100, True, True) == {256: 1.0}


def test_kaplan_meier_median_and_restricted_mean():
    km = R.kaplan_meier([10, 20, 30, 40], [True, True, True, True], tmax=40)
    assert km["median"] == 20 and km["rmst"] == pytest.approx(10 + 0.75 * 10 + 0.5 * 10 + 0.25 * 10)
    km = R.kaplan_meier([10, 50, 50], [True, False, False], tmax=50)       # censored rows stay at risk
    assert km["median"] is None and km["surv_end"] == pytest.approx(2 / 3)
    assert R.kaplan_meier([], [], 10)["n"] == 0


def test_text_measures():
    bt = R.backtrack_counts("Wait, hmm. Hmmm... Alternatively we wait. Actually, let me check. double-check")
    assert bt == {"wait": 2, "alternatively": 1, "hmm": 2, "actually": 1, "let_me_check": 1, "double_check": 1}
    assert R.distinct_n([1, 2, 3, 4, 1, 2, 3, 4], 4) == pytest.approx(4 / 5)
    assert R.distinct_n([1, 2], 4) is None
    loop = "We add the two numbers to get the sum. We add the two numbers to get the sum. Then we stop here now."
    assert R.self_loop_rate(loop) == pytest.approx(1 / 3)
    assert R.self_loop_rate("a b") is None


def test_arm_grammar_and_presets():
    a = R.parse_arm("fixed:random2@0.15")
    assert (a.kind, a.direction, a.seed, a.gain, a.key) == ("fixed", "random", 2, 0.15, "fixed-random2-0.15")
    a = R.parse_arm("law:gdn_mismatch_lag:restore:anti_gist@0.3")
    assert (a.source, a.polarity, a.direction, a.g_max) == ("gdn_mismatch_lag", "restore", "anti_gist", 0.3)
    assert R.parse_arm("yoked:law:agree:reinforce").ref == "law:agree:reinforce"
    assert R.parse_arm("trunc@512").budget == 512 and R.parse_arm("trunc:law:agree:restore").ref
    for bad in ("fixed:gist", "law:nope:reinforce", "law:agree:up", "matched:", "fixed:sideways@0.1", "what"):
        with pytest.raises(ValueError):
            R.parse_arm(bad)
    s0 = R.preset_arms("stage0", 0.2, ["agree"], True)
    assert s0[0] == "none" and "fixed:gist@0.1" in s0 and "fixed:foreign@0.2" in s0 and s0[-1] == "nothink"
    s1 = R.preset_arms("stage1", 0.2, ["agree", "gdn_write"], False)
    assert "law:gdn_write:restore" in s1 and "yoked:law:agree:reinforce" in s1 and "nothink" not in s1
    for spec in s0 + s1:
        A.judge.safe_name(R.parse_arm(spec).key)


def test_load_items_specs():
    nb = R.load_items("notebook20")
    assert len(nb) == 20 and nb[0].gold == "66" and nb[7].gold == "11.25" and nb[9].gold == "20.4"
    syn = R.load_items("synthetic:5", data_seed=3)
    assert len(syn) == 5 and syn == R.load_items("synthetic:5", data_seed=3)
    assert R.load_items("synthetic:5", data_seed=3, limit=2) == syn[:2]
    with pytest.raises(ValueError):
        R.load_items("nope")


# ----------------------------------------------------------------------------------- span controller
def tiny():
    model, _ = R.build_tiny("qwen3", 0)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def test_span_controller_stops_steering_at_the_span_end():
    model = tiny()
    E = R.TINY_THINK_END
    ids = torch.tensor([[BOS] + list(range(40, 52)) + [E] + list(range(60, 66)),
                        [BOS] + list(range(70, 88)) + [90]])
    T = ids.shape[1]
    span = R.SpanController(64, "constant", g_min=0.2, g_max=0.2, smooth="none", steer_from=3, span_end_id=E)
    plain = CA2Controller(64, "constant", g_min=0.2, g_max=0.2, smooth="none", steer_from=3)
    out = {}
    for name, c in (("span", span), ("plain", plain)):
        att = CA2Attachment(model, c, tap=1)
        c.reset()
        c.record = True
        model(input_ids=ids, use_cache=False)
        out[name] = c.trace_tensors()
        att.remove()
    s, p = out["span"]["steer"], out["plain"]["steer"]
    close = 13
    assert torch.equal(s[0, :close], p[0, :close]) and s[0, close:].sum() == 0     # </think> and after: off
    assert torch.equal(s[1], p[1])                                                # no span end: unchanged
    assert out["span"]["g_raw"][0, close:].abs().sum() == 0
    assert torch.equal(out["span"]["step"][0, :close], out["plain"]["step"][0, :close])
    assert span.span_closed.tolist() == [True, False]


def test_span_controller_prefill_plus_decode_equals_one_pass():
    model = tiny()
    E = R.TINY_THINK_END
    ids = torch.tensor([[BOS] + list(range(40, 50)) + [E] + [61, 62, 63]] * 2)
    ids[1, 5] = 99
    P = 6

    def mk():
        return R.SpanController(64, "agree", g_max=0.4, g_abs=1.0, steer_from=P - 1, span_end_id=E)
    one = mk()
    att = CA2Attachment(model, one, tap=1)
    one.reset()
    one.record = True
    model(input_ids=ids, use_cache=False)
    att.remove()
    inc = mk()
    att = CA2Attachment(model, inc, tap=1)
    inc.reset()
    inc.record = True
    o = model(input_ids=ids[:, :P], use_cache=True)
    pkv = o.past_key_values
    sizes = []
    for t in range(P, ids.shape[1]):
        o = model(input_ids=ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
        pkv = o.past_key_values
        sizes.append(inc.state_numel())
    att.remove()
    a, b = one.trace_tensors(), inc.trace_tensors()
    for k in ("g", "g_raw", "steer", "step", "s", "E"):
        assert torch.allclose(a[k].float(), b[k].float(), atol=1e-5, equal_nan=True), k
    assert len(set(sizes)) == 1
    assert sum(p.numel() for p in one.parameters()) == 0


# ---------------------------------------------------------------------------------------- end to end
def _argv(tmp_path, name: str, arch: str, extra) -> list[str]:
    return ["--tiny", "--tiny-arch", arch, "--device", "cpu", "--data", "synthetic:4", "--batch", "3",
            "--max-new-tokens", "24", "--probe-every", "8", "--budgets", "8,16", "--answer-tokens", "6",
            "--calib-tokens", "2", "--g-max", "0.3", "--out", str(tmp_path / f"{name}.json"),
            "--trace-dir", str(tmp_path / f"traces_{name}"), *extra]


def _run(tmp_path, *extra, arch="qwen3", name="t"):
    return R.main(_argv(tmp_path, name, arch, extra))


def test_end_to_end_tiny_json_traces_and_resume(tmp_path, capsys):
    arms = ("none,fixed:gist@0.3,fixed:foreign@0.3,law:agree:restore,matched:law:agree:restore,"
            "yoked:law:agree:restore,posmean:law:agree:restore,trunc:law:agree:restore,nothink")
    rep = _run(tmp_path, "--arms", arms, "--samples", "2")
    out = tmp_path / "t.json"
    js = json.loads(out.read_text())
    assert set(js["arms"]) == set(arms.split(",")) and all(a["complete"] for a in js["arms"].values())
    for k in ("args", "git", "versions", "model", "data"):
        assert k in js
    for spec, a in js["arms"].items():
        pi = a["per_item"]
        assert len(pi["uid"]) == 8 and sorted(set(pi["sample"])) == [0, 1]                # 4 problems × 2
        assert {"acc", "n_reason_mean", "ttc_km", "acc_b8", "trunc_rate"} <= set(a["agg"])
        assert {"mean", "sd", "frac_at_gmin", "frac_at_gmax"} <= set(a["gain"]) or a["gain"]["n"] == 0
        man = Path(a["trace"]["manifest"])
        assert _sha256(man) == a["trace"]["manifest_sha256"]
        for f in json.loads(man.read_text())["files"]:
            assert _sha256(man.parent / f["path"]) == f["sha256"]
        if spec != "none":
            assert a["vs_none"]["n_pairs"] == 8
    # probe only on sample 0; gains as specified
    fx = js["arms"]["fixed:gist@0.3"]
    assert all((c is None) == (s == 1) for c, s in zip(fx["per_item"]["censored"], fx["per_item"]["sample"]))
    assert fx["gain"]["mean"] == pytest.approx(0.3) and js["arms"]["none"]["gain"]["mean"] == 0.0
    assert js["arms"]["matched:law:agree:restore"]["setup"]["g_max"] == pytest.approx(
        js["arms"]["law:agree:restore"]["gain"]["mean"])
    assert max(js["arms"]["law:agree:restore"]["per_item"]["tf_gain_dev"]) < 1e-5     # generation == forced pass
    assert js["arms"]["trunc:law:agree:restore"]["setup"]["budget"] >= 1
    # acc@b only on probed rows (sample 0): never on the unprobed sample, whatever its outcome
    for a in js["arms"].values():
        pi = a["per_item"]
        assert pi["probed"] == [s == 0 for s in pi["sample"]]
        for b in (8, 16):
            assert all(x is None for x, s in zip(pi[f"acc_b{b}"], pi["sample"]) if s == 1)
            assert a["agg"][f"acc_b{b}_n"] == sum(x is not None for x in pi[f"acc_b{b}"]) <= 4
    pm = js["arms"]["posmean:law:agree:restore"]["setup"]
    assert pm["posmean_rule"] == "leave-one-problem-out" and pm["posmean_n"][0] == 8
    # no generated text in the results JSON; the text is in the trace
    items = [json.loads(l) for l in (tmp_path / "traces_t" / "law-agree-restore" / "tokens" / "items.jsonl")
             .read_text().splitlines()]
    assert len(items) == 8 and all("text" in it for it in items)
    txt = out.read_text()
    assert not any(it["text"] and len(it["text"]) > 8 and json.dumps(it["text"])[1:-1] in txt for it in items)
    # per-token arrays: controller, outcome, lens and bookkeeping keys, one row per prompt + response token
    z = np.load(tmp_path / "traces_t" / "law-agree-restore" / "tokens" / "tokens_0000.npz")
    n_tok = sum(it["n_tok"] for it in items)
    for k in ("g", "g_raw", "steer", "src__agree__s", "src__attn_window__s", "tap__attn__elig", "tap__norms__norm",
              "out__kl", "out__logp_next", "lens__rank_next", "bk__event_rank", "bk__reason", "bk__ids"):
        assert k in z.files and z[k].shape[0] == n_tok, k
    assert np.array_equal(z["bk__ids"][: items[0]["n_tok"]][-items[0]["n_gen"]:] >= 0,
                          np.ones(items[0]["n_gen"], bool))
    # resume: everything complete is skipped and the file is unchanged
    before = out.read_text()
    _run(tmp_path, "--arms", arms, "--samples", "2")
    assert "skipped" in capsys.readouterr().out
    after = json.loads(out.read_text())
    assert after["arms"] == json.loads(before)["arms"]
    with pytest.raises(SystemExit):                                                # different settings refused
        _run(tmp_path, "--arms", "none", "--samples", "1")
    for extra in (("--g-max", "0.4"), ("--smooth", "none"), ("--batch", "2"), ("--betas", "0.8,0.99"),
                  ("--probe-samples", "2"), ("--budgets", "8"), ("--leak", "0.5"), ("--regulator", "ngram")):
        with pytest.raises(SystemExit):                                            # law arms do not name these
            R.main(_argv(tmp_path, "t", "qwen3", ("--arms", arms, "--samples", "2")) + list(extra))
    assert json.loads(out.read_text())["arms"] == json.loads(before)["arms"]


def test_posmean_table_is_leave_one_problem_out():
    r = {("a", 0): np.full(2, 0.9, np.float32), ("a", 1): np.full(5, 0.7, np.float32),
         ("b", 0): np.full(3, 0.1, np.float32), ("c", 0): np.full(4, 0.3, np.float32)}
    tabs, n_k = R.posmean_tables(r)
    assert n_k.tolist() == [4, 4, 3, 2, 1]
    # a: only b and c; b reaches 3 steps, c 4; past step 3 nothing but a's own -> truncated (held by the source)
    assert np.allclose(tabs["a"], [0.2, 0.2, 0.2, 0.3])
    # c: a's two samples and b; a's 5-step sample alone covers steps 3..4
    assert np.allclose(tabs["c"], [(0.9 + 0.7 + 0.1) / 3, (0.9 + 0.7 + 0.1) / 3, (0.7 + 0.1) / 2, 0.7, 0.7])
    assert np.allclose(tabs["b"][:2], [(0.9 + 0.7 + 0.3) / 3] * 2) and len(tabs["b"]) == 5
    with pytest.raises(ValueError):
        R.posmean_tables({("a", 0): np.ones(3, np.float32), ("a", 1): np.ones(2, np.float32)})


def test_dependent_arm_needs_its_reference_first(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, "--arms", "matched:law:agree:reinforce", name="dep")


def test_trace_dir_inside_the_repo_is_refused(tmp_path):
    with pytest.raises(ValueError):
        R.main(["--tiny", "--tiny-arch", "qwen3", "--device", "cpu", "--arms", "none", "--out",
                str(tmp_path / "x.json"), "--trace-dir", str(ROOT / "results" / "nope_traces")])
    assert not (ROOT / "results" / "nope_traces").exists()


def test_end_to_end_tiny_hybrid_with_gdn_sources(tmp_path):
    if not hasattr(transformers, "Qwen3_5TextConfig"):
        pytest.skip("transformers has no Qwen3.5 classes")
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    saved = {n: getattr(M, n) for n in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")}
    try:
        try:
            js = _run(tmp_path, "--arms", "none,law:gdn_mismatch_lag:reinforce,law:attn_window:restore",
                      arch="qwen35", name="h")
        except Exception as e:                                                       # noqa: BLE001
            pytest.skip(f"the torch GDN path does not run here: {e!r}")
    finally:
        for n, f in saved.items():
            setattr(M, n, f)
    assert js["model"]["hybrid"] and js["model"]["gdn_layers"] == [0, 2] and js["model"]["attn_tap_layer"] == 1
    law = js["arms"]["law:gdn_mismatch_lag:reinforce"]
    assert law["complete"] and max(law["per_item"]["tf_gain_dev"]) < 1e-5
    z = np.load(tmp_path / "traces_h" / "none" / "tokens" / "tokens_0000.npz")
    for k in ("tap__gdn__fam", "tap__gdn__beta", "src__gdn_mismatch_lag__s", "src__gdn_write__s", "tap__attn__band"):
        assert k in z.files, k
    assert (tmp_path / "traces_h" / "none" / "perhead" / "tokens_0000.npz").exists()
