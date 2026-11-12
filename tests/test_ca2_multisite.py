"""Multi-site CA2 runner (``experiments/ca2/multisite.py``): gain 0 at every site is byte-identical to the base,
K = 1 reproduces the single-site hosting of ``text.py`` and ``ceiling.py`` exactly, the dose rules, strict
causality, prefill + cached decode against one pass, zero trainable parameters, the tiny end-to-end runs of both
modes and ``--analyse``, and the resume guard. CPU only: tiny random models (Qwen3 4 attention blocks; Qwen3.5
GDN, attention, GDN, attention) with the byte tokenizer."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import pretrained_loop as pl  # noqa: E402
from ca2 import common  # noqa: E402
from ca2 import measure as ms  # noqa: E402
from ca2 import multisite as M  # noqa: E402
from ca2 import text as tx  # noqa: E402
from ca2.controller import CA2Attachment  # noqa: E402

D, BOS = 64, 256


@pytest.fixture
def text_file(tmp_path):
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "river", "stone", "light", "north", "seven", "table"]
    p = tmp_path / "text.txt"
    p.write_text(" ".join(words[i] for i in rng.integers(0, len(words), 30_000)))
    return p


@pytest.fixture
def torch_kernels(monkeypatch):
    """The Qwen3.5 GDN kernels swapped for their torch originals for the test, restored afterwards."""
    if not hasattr(transformers, "Qwen3_5TextConfig"):
        pytest.skip("transformers has no Qwen3.5 classes")
    import transformers.models.qwen3_5.modeling_qwen3_5 as Q
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        if not hasattr(Q, name):
            pytest.skip(f"{name} not found")
        monkeypatch.setattr(Q, name, getattr(Q, name))
    ms.swap_torch_gdn()
    return Q


def tiny(arch: str, dtype=torch.float32):
    if arch == "qwen3":
        m, _ = ms.build_tiny_qwen3(0)
    else:
        m, _ = pl.build_tiny(arch, 0)
    m = m.to(dtype).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    blocks = pl.find_blocks(m)
    return m, blocks, pl.layer_kinds(m, blocks)


def ids_(B=2, T=40, seed=0):
    x = torch.randint(0, 256, (B, T), generator=torch.Generator().manual_seed(seed))
    x[:, 0] = BOS
    return x


def margs(**kw):
    a = M.parse_args(["--tiny", "--text-file", "unused"])
    a.steer_from, a.window, a.attn_window = 4, 16, 8
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def text_pass(model, blocks, kinds, arm, ids, args, window=0, layers="attention", signals="all", record=True):
    """One teacher-forced multi-site pass as ``run_text_arm`` makes it; returns (logits, controllers)."""
    lm = pl.HookedLM(model, side=None, layer=arm["sites"][0], window=args.window, mode="block", window_layers=layers,
                     sink=args.sink, pad_id=BOS)
    ctrls = [tx.build_ctrl(M.site_arm(arm, i), D, args, ids.shape[1] - args.sink) for i in range(arm["K"])]
    sset = M.SiteSignals(model, blocks, kinds, arm["sites"], args.sink, args.attn_window, True, ["beta"], signals)
    sset.register(ctrls)
    for c in ctrls:
        c.record = record
    host = M.TextHost(model, blocks, ctrls, arm["sites"], args.sink)
    try:
        with lm.condition(window, False):
            lg = lm(ids)
    finally:
        host.remove()
        sset.remove()
        M.release_hooked(lm)
    return lg, ctrls


def base_pass(model, ids, args, window=0, layers="attention"):
    lm = pl.HookedLM(model, side=None, window=args.window, mode="block", window_layers=layers, sink=args.sink,
                     pad_id=BOS)
    try:
        with lm.condition(window, False):
            return lm(ids)
    finally:
        M.release_hooked(lm)


# ------------------------------------------------------------------------------------------- arms, doses
def test_site_parsing_dose_rules_and_arm_grid():
    assert M.parse_sites("2-5") == [2, 3, 4, 5] and M.parse_sites("7,12,19") == [7, 12, 19]
    assert M.parse_sites("4,1-2,4") == [1, 2, 4]
    G = 0.6
    assert M.site_gains(G, 3, "lin") == pytest.approx([0.2] * 3)
    assert M.site_gains(G, 4, "sqrt") == pytest.approx([0.3] * 4)
    assert M.site_gains(G, 3, "full") == pytest.approx([0.6] * 3)
    assert M.site_gains(G, 1, "lin") == M.site_gains(G, 1, "sqrt") == [0.6]
    hyb = ["linear", "linear", "linear", "attention"] * 6
    assert M.default_site_sets(hyb) == "12;7,12,19;2,6,10,14,18,22;2-21"
    assert M.default_site_sets(["attention"] * 28) == "14;7,14,21;6,10,14,18,22;4-24"
    sets = {"L12": [12], "L7,12,19": [7, 12, 19]}
    arms = M.make_arms(sets, ["lin", "sqrt"], ["gist", "random", "random_shared"], [0.1, 0.3], r=5)
    names = [a["name"] for a in arms]
    assert names[0] == "none" and arms[0]["sites"] == [7, 12, 19] and arms[0]["g_site"] == [0.0] * 3
    assert len(arms) == 1 + 2 * 2 + 2 * 3 * 2                       # single site: one dose, no random_shared
    assert "L12:one:gist@0.3" in names and "L12:one:random_shared@0.3" not in names
    a = next(x for x in arms if x["name"] == "L7,12,19:sqrt:random@0.3")
    assert a["seeds"] == [5, 1005, 2005] and a["g_site"] == pytest.approx([0.3 / math.sqrt(3)] * 3)
    b = next(x for x in arms if x["name"] == "L7,12,19:lin:random_shared@0.1")
    assert b["seeds"] == [5, 5, 5] and b["ctrl_direction"] == "random" and b["g_site"] == pytest.approx([0.1 / 3] * 3)
    single = next(x for x in arms if x["name"] == "L12:one:random@0.1")
    assert single["seeds"] == [5]                                   # site 1 of 'random' is the single-site vector
    kept = M.make_arms(sets, ["lin"], ["gist"], [0.1], 0, "none;L7,12,19:*")
    assert [x["name"] for x in kept] == ["none", "L7,12,19:lin:gist@0.1"]


@pytest.mark.parametrize("dose", ["lin", "sqrt", "full"])
def test_dose_rules_give_the_stated_per_site_gains(dose):
    model, blocks, kinds = tiny("qwen3")
    args = margs()
    arm = M.make_arm("L0-2", [0, 1, 2], dose, "gist", 0.6, 0)
    _, ctrls = text_pass(model, blocks, kinds, arm, ids_(), args)
    want = M.site_gains(0.6, 3, dose)
    for c, g in zip(ctrls, want):
        tr = c.trace_tensors()
        st = tr["steer"] > 0
        assert st.any() and torch.allclose(tr["g_raw"][st], torch.full_like(tr["g_raw"][st], g))
        assert (tr["g_raw"][~st] == 0).all()
        live = st & (tr["dnorm"] > 0)
        assert torch.allclose(tr["inj_ratio"][live], torch.full_like(tr["inj_ratio"][live], g), atol=1e-6)


# --------------------------------------------------------------------------------------- byte identity
@pytest.mark.parametrize("arch", ["qwen3", "qwen35"])
def test_gain_zero_at_every_site_is_byte_identical(arch, request):
    if arch == "qwen35":
        request.getfixturevalue("torch_kernels")
    model, blocks, kinds = tiny(arch)
    args = margs()
    ids = ids_()
    arm = M.make_arm("union", [0, 1, 2], "none", "gist", 0.0, 0)
    for window, layers in ((0, "attention"), (16, "attention"), (16, "all")):
        base = base_pass(model, ids, args, window, layers)
        lg, ctrls = text_pass(model, blocks, kinds, arm, ids, args, window, layers)
        assert torch.equal(lg, base), (window, layers)
        assert all(c.trace_tensors()["steer"].sum() > 0 for c in ctrls)          # the schedule was live
    # generation host: every site attached at g = 0 with the shared taps
    plain = model(input_ids=ids, use_cache=False).logits
    ctrls = [M.gen_ctrl(arm, i, D, args, 3) for i in range(3)]
    sset = M.SiteSignals(model, blocks, kinds, [0, 1, 2], args.sink, args.attn_window, True, ["beta"])
    sset.register(ctrls)
    host = M.GenHost(model, ctrls, [0, 1, 2])
    for c in ctrls:
        c.reset()
        c.record = True
    try:
        lg = model(input_ids=ids, use_cache=False).logits
    finally:
        host.remove()
        sset.remove()
    assert torch.equal(lg, plain)
    tt = M.assemble([c.trace_tensors() for c in ctrls], [0, 1, 2])
    assert "site2/src/agree/s" in tt and "site0/src/gist_drift/lag1" in tt and "bk/steer" in tt
    if arch == "qwen35":
        assert "tap/gdn/beta" in tt and "site2/src/gdn_write/s" in tt and "site0/src/gdn_write/s" in tt
        assert "tap/attn1/elig" in tt and "site1/src/attn_window/s" in tt and "site0/src/attn_window/s" not in tt


# ---------------------------------------------------------------------------------- K = 1 equivalence
@pytest.mark.parametrize("direction", ["gist", "mean", "random"])
def test_one_site_reproduces_text_single_site_hosting(direction):
    """K = 1 through TextHost equals ``text.py``'s SteerSide inside HookedLM, bitwise, logits and trace."""
    model, blocks, kinds = tiny("qwen3")
    args = margs(random_seed=3)
    ids = ids_()
    arm = M.make_arm("L1", [1], "one", direction, 0.4, 3)
    for window in (0, 16):
        lg, ctrls = text_pass(model, blocks, kinds, arm, ids, args, window, signals="lean")
        ref_ctrl = tx.build_ctrl(M.site_arm(arm, 0), D, args, ids.shape[1] - 1)
        ref_ctrl.record = True
        lm = pl.HookedLM(model, side=None, layer=1, window=args.window, mode="block", window_layers="attention",
                         sink=1, pad_id=BOS)
        side = common.SteerSide(ref_ctrl, model, sink=1, tap_layer=1)
        lm.side = side
        try:
            with lm.condition(window, True):
                ref = lm(ids)
        finally:
            side.remove()
            M.release_hooked(lm)
        assert torch.equal(lg, ref)
        a, b = ctrls[0].trace_tensors(), ref_ctrl.trace_tensors()
        for k in ("g", "g_raw", "inj_ratio", "dnorm", "cos_dh", "hnorm"):
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0, equal_nan=True)
        assert not torch.equal(lg, base_pass(model, ids, args, window))


def test_one_site_reproduces_ceiling_single_site_generation():
    """K = 1 through GenHost equals one ``CA2Attachment`` (ceiling.run_gen's hosting): same sampled tokens."""
    model, blocks, kinds = tiny("qwen3")
    args = margs()
    ids = ids_(2, 9)
    arm = M.make_arm("L2", [2], "one", "gist", 0.5, 0)

    def sample(attach):
        ctrls = [M.gen_ctrl(arm, 0, D, args, "last_prompt")]
        h = attach(ctrls)
        ctrls[0].reset()
        g = torch.Generator().manual_seed(0)
        out, new, lg = model(input_ids=ids, use_cache=True), [], []
        for _ in range(12):
            z = out.logits[:, -1].float()
            lg.append(z)
            nxt = torch.multinomial(torch.softmax(z / 0.7, -1), 1, generator=g)
            new.append(nxt)
            out = model(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True)
        h.remove()
        return torch.cat(new, 1), torch.stack(lg, 1)

    a_ids, a_lg = sample(lambda c: M.GenHost(model, c, [2]))
    b_ids, b_lg = sample(lambda c: CA2Attachment(model, c[0], tap=2))
    assert torch.equal(a_ids, b_ids) and torch.equal(a_lg, b_lg)


# -------------------------------------------------------------------------------------------- causality
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("arch,direction", [("qwen3", "gist"), ("qwen3", "random"), ("qwen35", "gist")])
def test_strict_causality_bitwise(arch, direction, dtype, request):
    if arch == "qwen35":
        request.getfixturevalue("torch_kernels")
    model, blocks, kinds = tiny(arch, dtype)
    args = margs()
    arm = M.make_arm("L0-2", [0, 1, 2], "sqrt", direction, 0.6, 0)
    ids, t0 = ids_(), 23
    ids2 = ids.clone()
    ids2[:, t0:] = (ids2[:, t0:] + 7) % 256
    for window, layers in ((0, "attention"), (16, "attention")):
        lg1, c1 = text_pass(model, blocks, kinds, arm, ids, args, window, layers)
        lg2, c2 = text_pass(model, blocks, kinds, arm, ids2, args, window, layers)
        assert not torch.equal(lg1[:, t0:], lg2[:, t0:])
        torch.testing.assert_close(lg1[:, :t0], lg2[:, :t0], rtol=0, atol=0)
        u = t0 - args.sink                                          # trace columns are text positions
        for a, b in zip(c1, c2):
            ta, tb = a.trace_tensors(), b.trace_tensors()
            for k in ("g", "inj_ratio", "hnorm", "src/agree/a", "src/gist_drift/lag1"):
                torch.testing.assert_close(ta[k][:, :u], tb[k][:, :u], rtol=0, atol=0, equal_nan=True)
            assert ta["g"][:, :u].abs().sum() > 0
        cc1, cc2 = M.consec_cos(c1), M.consec_cos(c2)
        torch.testing.assert_close(cc1[:, :u], cc2[:, :u], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("arch", ["qwen3", "qwen35"])
def test_prefill_then_cached_decode_equals_one_pass(arch, request):
    """Multi-site arm through GenHost: prefill P columns, then one token per call, against one pass. Not bitwise
    (prefill and decode use different matmul shapes; the GDN layers switch from the chunked to the recurrent rule),
    so float32 noise tolerance, as tests/test_ca2.py."""
    if arch == "qwen35":
        request.getfixturevalue("torch_kernels")
    model, blocks, kinds = tiny(arch)
    args = margs()
    ids, P = ids_(2, 30), 11
    arm = M.make_arm("L0-2", [0, 1, 2], "lin", "gist", 0.9, 0)

    def run(chunks):
        ctrls = [M.gen_ctrl(arm, i, D, args, 4) for i in range(3)]
        sset = M.SiteSignals(model, blocks, kinds, [0, 1, 2], args.sink, args.attn_window, False, [])
        sset.register(ctrls)
        host = M.GenHost(model, ctrls, [0, 1, 2])
        for c in ctrls:
            c.reset()
            c.record = True
        lg, cache = [], None
        try:
            for a, b in chunks:
                out = model(input_ids=ids[:, a:b], past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                lg.append(out.logits.float())
        finally:
            host.remove()
            sset.remove()
        return torch.cat(lg, 1), [c.trace_tensors() for c in ctrls]

    lg1, tr1 = run([(0, 30)])
    lg2, tr2 = run([(0, P)] + [(t, t + 1) for t in range(P, 30)])
    assert torch.allclose(lg1, lg2, atol=1e-4)
    for a, b in zip(tr1, tr2):
        for k in ("g", "g_raw", "inj_ratio", "src/agree/a", "hnorm"):
            assert torch.allclose(a[k], b[k], atol=1e-4, equal_nan=True), k
        assert a["g"].abs().sum() > 0
    base = model(input_ids=ids, use_cache=False).logits.float()
    assert not torch.allclose(lg1, base, atol=1e-3)


# ---------------------------------------------------------------------------------------- no parameters
def test_zero_trainable_parameters_and_frozen_model_untouched():
    model, blocks, kinds = tiny("qwen3")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    args = margs()
    arm = M.make_arm("L0-2", [0, 1, 2], "full", "gist", 0.5, 0)
    _, ctrls = text_pass(model, blocks, kinds, arm, ids_(), args)
    assert sum(p.numel() for c in ctrls for p in c.parameters()) == 0
    assert not any(p.requires_grad for p in model.parameters())
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(len(b._forward_hooks) == 0 and len(b._forward_pre_hooks) == 0 for b in blocks)


# ------------------------------------------------------------------------------------------- end to end
def _argv(tmp, text_file, arch, extra=()):
    return ["--tiny", "--tiny-arch", arch, "--dtype", "float32", "--device", "cpu", "--text-file", str(text_file),
            "--out", str(tmp / "ms.json"), "--trace-dir", str(tmp / "traces"), "--site-sets", "2;0-2",
            "--doses", "lin,sqrt", "--directions", "gist,random_shared", "--gen-directions", "gist,random_shared",
            "--gains", "0.3,0.9", "--boot", "100", *extra]


@pytest.mark.parametrize("arch", ["qwen35", "qwen3"])
def test_end_to_end_both_modes_and_analyse(tmp_path, text_file, torch_kernels, arch):
    d = M.main(_argv(tmp_path, text_file, arch))
    arms = d["arms"]
    conds = ["full", "attn", "all"] if arch == "qwen35" else ["full", "attn"]
    assert set(d["conditions"]) == set(conds)
    names = {"none", "L2:one:gist@0.3", "L2:one:gist@0.9", "L0-2:lin:gist@0.3", "L0-2:sqrt:random_shared@0.9"}
    for c in conds:
        assert names <= {k.split("/", 2)[2] for k in arms if k.startswith(f"text/{c}/")}
        none = arms[f"text/{c}/none"]
        assert none["checks"]["gain0_equal"] is True and none["agg"]["kl_base"] == 0.0
        a = arms[f"text/{c}/L0-2:sqrt:gist@0.9"]
        ag = a["agg"]
        assert ag["kl_base"] > 0 and a["checks"]["site_gain_dev"] < 1e-6
        g = 0.9 / math.sqrt(3)
        assert ag["inj_ratio_by_block"][:3] == pytest.approx([g] * 3, rel=1e-4) and ag["inj_ratio_by_block"][3] == 0
        assert len(ag["cos_consec"]) == 2 and len(ag["norm_ratio_by_block"]) == 4
        assert set(a["per_site"]) == {"0", "1", "2"}
        assert a["per_site"]["1"]["gain_dist"]["mean"] == pytest.approx(g, rel=1e-5)
        s1 = arms[f"text/{c}/L2:one:gist@0.9"]["agg"]
        assert s1["norm_ratio_by_block"][0] == pytest.approx(1.0) and s1["norm_ratio_by_block"][1] == pytest.approx(1.0)
        assert "cos_consec" not in s1 and "d_loss_vs_base" in s1
        if c != "full":
            assert "closure" in ag and "kl_full" in ag
        pi = a["per_item"]
        assert np.allclose(np.sum(pi["off_n"], 1), pi["n_tok"])
    # traces: per-site keys, shared taps once, outcome keys
    man = json.loads(Path(arms["text/attn/L0-2:lin:gist@0.3"]["traces"]["manifest"]).read_text())
    keys = {k for f in man["files"] for k in f.get("keys", {})}
    for k in ("site0__g_raw", "site2__inj_ratio", "site1__src__gist_drift__lag1", "bk__steer", "out__kl_base",
              "out__norm_ratio", "multi__cos_consec", "pos__offset"):
        assert k in keys, k
    if arch == "qwen35":
        assert {"tap__gdn__beta", "site2__src__gdn_mismatch__x", "tap__attn1__elig", "tap__gdn__fam_head"} <= keys
    assert not any(k.startswith("site1__tap") for k in keys)
    # generation: every arm on every seed, paired summary, teacher-forced KL
    seeds = d["gen_prompts"]["seeds"]
    assert seeds == [0, 1]
    for s in seeds:
        n = arms[f"gen/seed{s}/none"]
        assert n["agg"]["n"] == 4 and n["agg"]["kl_mean"] == 0.0
        a = arms[f"gen/seed{s}/L0-2:lin:gist@0.9"]
        assert a["agg"]["kl_mean"] > 0 and a["checks"]["tf_lp_dev_max"] < 1e-3
        assert len(a["agg"]["norm_ratio_by_block"]) == 4 and len(a["agg"]["cos_consec"]) == 2
        assert a["per_item"]["prompt"] == n["per_item"]["prompt"]
    gs = d["gen_summary"]
    assert gs["none"]["seeds"] == [0, 1] and len(gs["none"]["collapse_rate_by_seed"]) == 2
    assert "d_collapse_vs_none" in gs["L0-2:sqrt:gist@0.3"]
    txt = json.dumps({k: v for k, v in d.items() if k != "args"})
    assert "gen/ids" not in txt and "alpha" not in txt              # no generated text in the results JSON
    gman = json.loads(Path(arms["gen/seed0/L0-2:lin:gist@0.9"]["traces"]["manifest"]).read_text())
    gkeys = {k for f in gman["files"] for k in f.get("keys", {})}
    assert {"tf__kl", "gen__ids", "site0__g", "multi__cos_consec"} <= gkeys
    # analysis
    out = M.main(["--analyse", str(tmp_path / "ms.json"), "--boot", "100", "--analyse-out", str(tmp_path / "a.json")])
    assert out["ref_set"] == "L2" and out["unsteered"]["seeds"] == [0, 1]
    grp = {(g["set"], g["dose"], g["direction"]): g for g in out["groups"]}
    g = grp[("L0-2", "lin", "gist")]
    assert g["ref"] == "L2:one:gist" and g["kl_from"] == "text" and len(g["curve"]) == 3
    assert g["curve"][0]["kl"] == 0.0 and g["curve"][2]["kl"] > g["curve"][1]["kl"]
    assert grp[("L0-2", "sqrt", "random_shared")]["ref"] is None     # its reference L2:one:random was not run
    assert set(g["dloss"]) == set(conds) and g["dloss"]["full"]["curve"][0]["d_loss"] == 0.0
    cmp_ = [c for c in out["comparison"] if c["set"] == "L0-2"]
    assert cmp_ and all(c["verdict"] in ("lower", "higher", "unclear") for c in cmp_)
    assert all("d_collapse" in c and c["d_collapse"]["n_units"] == 4 for c in cmp_)
    assert json.loads((tmp_path / "a.json").read_text())["groups"]
    gen_only = M.analyse(tmp_path / "ms.json", kl_from="gen", boot=50)
    assert {g["kl_from"] for g in gen_only["groups"]} == {"gen"}


def test_resume_skips_complete_arms_and_refuses_other_settings(tmp_path, text_file, torch_kernels, monkeypatch):
    argv = _argv(tmp_path, text_file, "qwen3", ["--site-sets", "1;0-1", "--directions", "gist",
                                                "--gen-directions", "gist", "--gains", "0.5", "--doses", "lin",
                                                "--conditions", "full"])
    d = M.main(argv)
    n_arms = len(d["arms"])
    assert n_arms == 3 + 3 * 2                                       # none + 2 arms, text (one condition) + 2 seeds
    called = []
    monkeypatch.setattr(M, "run_text_arm", lambda *a, **k: called.append("text"))
    monkeypatch.setattr(M, "run_gen_arm", lambda *a, **k: called.append("gen"))
    M.main(argv)
    assert not called
    with pytest.raises(SystemExit):
        M.main(argv + ["--seq-len", "80"])
    with pytest.raises(SystemExit):
        M.main(argv + ["--gen-batch", "4"])                          # one generation batch size per results file
    with pytest.raises(SystemExit):
        M.main(argv + ["--random-seed", "7"])
    with pytest.raises(SystemExit):
        M.main(argv + ["--attn-impl", "eager"])                      # other kernels, other bf16 numerics
    with pytest.raises(SystemExit):
        M.main(argv + ["--no-gen-tf"])                               # later arms would carry no kl_mean
    d2 = json.loads((tmp_path / "ms.json").read_text())
    assert d2["compat"]["seq_len"] == 96 and len(d2["arms"]) == n_arms
    M.main(argv + ["--site-sets", "0-1"])                            # a resume naming only some sets
    assert not called
    d3 = json.loads((tmp_path / "ms.json").read_text())
    assert d3["site_sets"] == {"L1": [1], "L0-1": [0, 1]}            # the union is kept: analyse still finds L1
    assert M.analyse(tmp_path / "ms.json", boot=50)["ref_set"] == "L1"


def test_trace_dir_inside_repo_is_refused(tmp_path, text_file, torch_kernels):
    with pytest.raises(SystemExit):
        M.main(["--tiny", "--tiny-arch", "qwen3", "--text-file", str(text_file), "--trace-dir", str(ROOT / "tmp_ms"),
                "--out", str(tmp_path / "x.json")])
    assert not (ROOT / "tmp_ms").exists()


def test_analysis_interpolation_is_paired_and_linear_in_kl():
    curve = [(0.0, 0.0), (0.1, 0.2), (0.3, 0.6)]
    vecs = [np.zeros(4), np.array([0.0, 1.0, 0.0, 1.0]), np.ones(4)]
    i, w = M._locate(curve, 0.4)
    assert i == 1 and w == pytest.approx(0.5)
    assert M._interp(vecs, (i, w)) == pytest.approx([0.5, 1.0, 0.5, 1.0])
    assert M._locate(curve, 0.7) is None
    g = M._grid([curve, [(0.0, 0.0), (0.2, 0.3), (0.5, 0.9)]], None)
    assert g[0] == pytest.approx(0.2) and g[-1] == pytest.approx(0.6) and len(g) == 5
    assert M._grid([curve], [0.1, 0.7]) == [0.1]
    assert M._monotone(curve) and not M._monotone([(0.0, 0.0), (0.1, 0.3), (0.2, 0.2)])


def test_grid_starts_at_the_shared_none_point_not_at_the_first_gain():
    """Review finding: one curve's whole KL range lies below the other's first positive KL. Both curves pass
    through (G 0, KL 0), so the grid is (0, min max KL], not empty, and every grid point is located on both."""
    dist = [(0.0, 0.0), (0.375, 0.01), (0.75, 0.015)]                 # e.g. L2-21:lin, per-site g ≤ 0.0375
    ref = [(0.0, 0.0), (0.1, 0.02), (0.3, 0.1)]
    g = M._grid([dist, ref], None)
    assert len(g) == 5 and g[-1] == pytest.approx(0.015) and g[0] == pytest.approx(0.01)
    for k in g:
        assert M._locate(dist, k) is not None and M._locate(ref, k) is not None
    assert M._locate(ref, g[0])[0] == 0                                # between none and the first gain
    assert M._grid([dist, [(0.0, 0.0), (0.1, 0.0)]], None) == []        # a curve with no positive KL: nothing


def _synthetic(tmp_path, collapse: dict[str, list[float]], kl: dict[str, tuple[float, float]],
               sets_in_header: dict | None = None) -> Path:
    """A results JSON with gen arms only: per arm the per-prompt collapse (same on every seed) and
    (kl_mean, kl_first64). Arm names 'L<set>:<dose>:gist@G'."""
    arms = {}
    for name, c in collapse.items():
        if name == "none":
            m = {"set": "union", "sites": [1, 2, 3], "K": 3, "dose": "none", "direction": "gist", "G": 0.0}
        else:
            lab, dose, rest = name.split(":")
            sites = M.parse_sites(lab[1:])
            m = {"set": lab, "sites": sites, "K": len(sites), "dose": dose, "direction": "gist",
                 "G": float(rest.split("@")[1])}
        km, k64 = kl.get(name, (0.0, 0.0))
        for s in (0, 1):
            arms[f"gen/seed{s}/{name}"] = {
                "mode": "gen", "seed": s, "arm": name, "meta": m, "complete": True,
                "agg": {"collapse_rate": float(np.mean(c)), "distinct4": 0.5, "kl_mean": km, "kl_first64": k64},
                "per_item": {"prompt": list(range(len(c))), "collapse": list(c)}}
    p = tmp_path / "syn.json"
    p.write_text(json.dumps({"site_sets": sets_in_header if sets_in_header is not None else {}, "arms": arms}))
    return p


def test_analysis_flags_a_distributed_curve_below_the_single_site_knee(tmp_path):
    n = 40
    z, h, f = [0.0] * n, [1.0] * (n // 2) + [0.0] * (n // 2), [1.0] * n
    p = _synthetic(tmp_path, {"none": z, "L2:one:gist@0.1": z, "L2:one:gist@0.3": h, "L2:one:gist@0.5": f,
                              "L1-3:lin:gist@0.3": z, "L1-3:lin:gist@0.75": z},
                   {"L2:one:gist@0.1": (0.02, 0.02), "L2:one:gist@0.3": (0.1, 0.1), "L2:one:gist@0.5": (0.3, 0.3),
                    "L1-3:lin:gist@0.3": (0.01, 0.01), "L1-3:lin:gist@0.75": (0.015, 0.015)},
                   sets_in_header={"L1-3": [1, 2, 3]})                 # L2 missing from the header (old resume)
    out = M.analyse(p, kl_from="gen", boot=200)
    assert out["ref_set"] == "L2"                                     # recovered from the arms' meta
    g = next(x for x in out["groups"] if x["set"] == "L1-3")
    cov = g["coverage"]
    assert cov["monotone"] and cov["ref_monotone"] and len(cov["grid"]) == 5
    assert cov["ref_knee"]["G"] == 0.3 and cov["ref_knee"]["kl"] == pytest.approx(0.1)
    assert any("knee" in s for s in cov["issues"])
    rows = [c for c in out["comparison"] if c["set"] == "L1-3"]
    assert len(rows) == 5 and all(c["verdict"] == "unclear" for c in rows)


def test_analysis_refuses_matching_on_a_non_monotone_gen_kl(tmp_path):
    """Review finding: gen KL is read along the arm's own continuation; a collapsed arm reads a LOWER KL. Such a
    curve gets no matched comparison; auto without text arms uses kl_first64 (gen64), here monotone."""
    n = 20
    z, f = [0.0] * n, [1.0] * n
    p = _synthetic(tmp_path, {"none": z, "L2:one:gist@0.3": z, "L2:one:gist@0.75": f, "L1-3:lin:gist@0.3": z,
                              "L1-3:lin:gist@0.75": z},
                   {"L2:one:gist@0.3": (0.2, 0.2), "L2:one:gist@0.75": (0.05, 0.4),
                    "L1-3:lin:gist@0.3": (0.1, 0.1), "L1-3:lin:gist@0.75": (0.3, 0.3)})
    gen = M.analyse(p, kl_from="gen", boot=100)
    g = next(x for x in gen["groups"] if x["set"] == "L1-3")
    assert g["coverage"]["ref_monotone"] is False and g["coverage"]["grid"] == []
    assert any("not non-decreasing" in s for s in g["coverage"]["issues"])
    assert not [c for c in gen["comparison"] if c["set"] == "L1-3"]
    auto = M.analyse(p, boot=100)
    g = next(x for x in auto["groups"] if x["set"] == "L1-3")
    assert g["kl_from"] == "gen64" and g["coverage"]["ref_monotone"] is True
    assert [c for c in auto["comparison"] if c["set"] == "L1-3"]
