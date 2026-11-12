"""Round-12 runners: the measures in ``experiments/ca2/measure.py`` and tiny end-to-end runs of ``text.py``,
``ceiling.py`` and ``analyse.py`` on the CPU (random tiny models, byte tokenizer, the WikiText test file or a
generated stand-in)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import pretrained_loop as pl  # noqa: E402
from ca2 import measure as ms  # noqa: E402
from ca2.controller import CA2Attachment, CA2Controller  # noqa: E402

D = 64


@pytest.fixture
def text_file(tmp_path):
    """A small text with repeats (contiguity has something to find), outside the repository."""
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "river", "stone", "light", "north", "seven", "table"]
    body = " ".join(words[i] for i in rng.integers(0, len(words), 30_000))
    p = tmp_path / "text.txt"
    p.write_text(body)
    return p


@pytest.fixture
def torch_kernels(monkeypatch):
    """Let the runner swap the Qwen3.5 GDN kernels for their torch originals, and undo it afterwards (every test
    that calls a runner's ``main`` uses it, so the swap never leaks into other test files)."""
    if not hasattr(transformers, "Qwen3_5TextConfig"):
        pytest.skip("transformers has no Qwen3.5 classes")
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        if not hasattr(M, name):
            pytest.skip(f"{name} not found")
        monkeypatch.setattr(M, name, getattr(M, name))
    return M


def tiny_qwen3():
    m, _ = ms.build_tiny_qwen3(0)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ---------------------------------------------------------------------------------------------- metrics
def test_token_metrics_match_direct_computation():
    g = torch.Generator().manual_seed(0)
    B, T, V = 2, 7, 11
    z, zr = torch.randn(B, T, V, generator=g), torch.randn(B, T, V, generator=g)
    tg = torch.randint(0, V, (B, T), generator=g)
    valid = torch.ones(B, T, dtype=torch.bool)
    valid[:, -1] = False
    cand = torch.randint(-1, V, (B, T, 3), generator=g)
    m = ms.token_metrics(z, tg, valid, {"base": zr}, cand=cand, chunk_elems=B * V * 2)   # chunks of 2 columns
    lp, lq = F.log_softmax(z, -1), F.log_softmax(zr, -1)
    loss = -lp.gather(-1, tg[..., None])[..., 0]
    ent = -(lp.exp() * lp).sum(-1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    rank = (z > z.gather(-1, tg[..., None])).sum(-1).float()
    for k, ref in (("loss", loss), ("ent", ent), ("kl_base", kl), ("rank", rank), ("top1", (rank == 0).float())):
        assert torch.allclose(m[k][valid], ref[valid], atol=1e-5), k
        assert torch.isnan(m[k][~valid]).all()
    want = torch.where(cand >= 0, lp.gather(-1, cand.clamp_min(0)), torch.full(cand.shape, float("nan")))
    got = m["lp"]
    assert torch.allclose(torch.nan_to_num(got[valid], nan=7.0), torch.nan_to_num(want[valid], nan=7.0), atol=1e-5)
    m0 = ms.token_metrics(z, tg, valid, {"self": z})
    assert m0["kl_self"][valid].abs().max() < 1e-6


def test_contiguity_against_brute_force():
    ids = torch.tensor([[9, 1, 2, 3, 1, 4, 2, 5, 1, 6, 3, 7]])
    s, W = 1, 4
    c = ms.contiguity(ids, s, W, s, seed=0, row_ids=[0])
    x = ids[0].tolist()
    for t in range(len(x)):
        prev = [j for j in range(s, t) if x[j] == x[t]]
        if t < s or not prev:
            assert math.isnan(float(c["lag"][0, t]))
            assert (c["cand"][0, t] == -1).all()
            continue
        j = prev[-1]
        assert c["lag"][0, t] == t - j
        for k, off in enumerate(ms.CONT_OFFSETS):
            q = j + off
            assert int(c["cand"][0, t, k]) == (x[q] if s <= q <= t else -1)
        assert c["inblock"][0, t] == float((j - s) // W == (t - s) // W)
        if t + 1 < len(x):
            assert c["hit"][0, t] == float(x[j + 1] == x[t + 1])
        ctrl = int(c["cand"][0, t, 4])
        assert ctrl == -1 or ctrl in x[s + 1: t + 1]
    again = ms.contiguity(ids, s, W, s, seed=0, row_ids=[0])
    assert torch.equal(again["cand"], c["cand"])


def test_buckets_and_sums():
    e = ms.offset_edges(64)
    assert e == [0, 8, 32, 64]
    assert ms.offset_edges(512) == [0, 64, 256, 512]
    x = torch.tensor([[0.0, 7.0, 8.0, 31.0, 32.0, 63.0, 64.0, float("nan")]])
    assert ms.bucketize(x, e).tolist() == [[0, 0, 1, 1, 2, 2, -1, -1]]
    assert ms.lag_edges(10) == [1, 2, 4, 8, 16]
    s, c = ms.bucket_sums(torch.tensor([[1.0, 2.0, 3.0, float("nan")]]), torch.tensor([[0, 1, 1, 1]]), 2)
    assert s.tolist() == [[1.0, 5.0]] and c.tolist() == [[1.0, 2.0]]


def test_closure_bootstrap_and_holm():
    from ca2.text import closure_boot
    from ca2.analyse import holm
    n = np.full(50, 100.0)
    full = np.full(50, 200.0)
    win = full + 50.0 + np.random.default_rng(0).normal(0, 1, 50)
    arm = win - 0.4 * (win - full)
    r = closure_boot(arm, win, full, n, 500, 0)
    assert abs(r["C"] - 0.4) < 1e-9 and r["gap_excludes_0"] and r["C_ci"][0] <= 0.4 <= r["C_ci"][1]
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj == {"a": 0.03, "b": 0.06, "c": 0.06, "d": None}


# ------------------------------------------------------------------------------------------- sources
def test_gist_drift_matches_bank_and_is_split_invariant():
    g = torch.Generator().manual_seed(1)
    h = torch.randn(2, 40, D, generator=g)

    def run(chunks):
        ctrl = CA2Controller(D, sink_skip=1, steer_from="all", g_max=0.0)
        ctrl.add_source(ms.GistDriftSource(lags=(1, 4)))
        ctrl.record = True
        for a, b in chunks:
            ctrl.begin_call(None)
            ctrl.forward(h[:, a:b])
        return ctrl.trace_tensors()

    one, two = run([(0, 40)]), run([(0, 17), (17, 40)])
    for k in ("gnorm", "cos_h", "lag1", "lag4", "rel"):
        a, b = one[f"src/gist_drift/{k}"], two[f"src/gist_drift/{k}"]
        assert torch.allclose(torch.nan_to_num(a, nan=9.0), torch.nan_to_num(b, nan=9.0), atol=1e-5), k
    # lag-1 cosine from the bank's gist directly
    from ca2.bank import GistBank
    st = GistBank((0.9, 0.995), 1).update(h, None)
    gist = st.gist(0, -1)
    c = (gist[:, 1:] * gist[:, :-1]).sum(-1)
    got = one["src/gist_drift/lag1"][:, 1:]
    ok = torch.isfinite(got)
    assert ok.any() and torch.allclose(got[ok], c[ok], atol=1e-5)


def test_optional_source_fills_missing_signal():
    class Missing(ms.Source):
        name, needs = "m", ()

        def __call__(self, ctx):
            raise KeyError("absent")

    ctrl = CA2Controller(D, g_max=0.0, steer_from="all")
    ctrl.add_source(ms.OptionalSource(Missing()))
    ctrl.record = True
    ctrl.begin_call(None)
    ctrl.forward(torch.randn(1, 5, D))
    tt = ctrl.trace_tensors()
    assert torch.isnan(tt["src/m/x"]).all() and (tt["src/m/s"] == 0).all() and (tt["src/m/defined"] == 0).all()


# ------------------------------------------------------------------------------------------- model parts
def test_lens_at_last_block_reproduces_the_model_rank():
    model = tiny_qwen3()
    blocks = pl.find_blocks(model)
    ids = torch.randint(0, 256, (2, 12), generator=torch.Generator().manual_seed(0))
    tg, valid = ms.shifted_targets(ids)
    lens = ms.LensCapture(blocks, [len(blocks) - 1])
    lg = model(input_ids=ids, use_cache=False).logits
    lens.remove()
    r = lens.ranks(model, tg, valid)[len(blocks) - 1]
    m = ms.token_metrics(lg, tg, valid)
    assert torch.equal(torch.nan_to_num(r, nan=-1), torch.nan_to_num(m["rank"], nan=-1))


def test_branch_scale_identity_and_ablation():
    model = tiny_qwen3()
    blocks = pl.find_blocks(model)
    ids = torch.randint(0, 256, (1, 10), generator=torch.Generator().manual_seed(0))
    base = model(input_ids=ids, use_cache=False).logits
    bs = ms.BranchScale(blocks, [1], "mlp", 0.0, 3)
    assert torch.equal(model(input_ids=ids, use_cache=False).logits, base)
    bs.remove()
    bs = ms.BranchScale(blocks, [1], "mlp", -1.0, 3)
    a = model(input_ids=ids, use_cache=False).logits
    bs.remove()
    h = blocks[1].mlp.register_forward_hook(
        lambda m, i, o: torch.cat([o[:, :3], torch.zeros_like(o[:, 3:])], 1))
    b = model(input_ids=ids, use_cache=False).logits
    h.remove()
    assert torch.equal(a, b) and torch.equal(a[:, :3], base[:, :3]) and not torch.equal(a, base)
    bs = ms.BranchScale(blocks, [1, 3], "attn", 0.5, 0)
    c = model(input_ids=ids, use_cache=False).logits
    bs.remove()
    assert not torch.equal(c, base)
    assert ms.neighbour_attention(["linear", "attention", "linear", "attention"], [0, 2]) == [1, 3]
    sets = ms.gdn_layer_sets(["linear", "linear", "linear", "attention"] * 6)
    assert sets["pre"] == [2, 6, 10, 14, 18, 22] and sets["band"] == [8, 9, 10, 12, 13, 14] and len(sets["all"]) == 18


def test_register_signals_without_gdn_layers():
    model = tiny_qwen3()
    blocks = pl.find_blocks(model)
    kinds = pl.layer_kinds(model, blocks)
    ctrl = CA2Controller(D, g_max=0.0)
    taps = ms.register_signals(ctrl, blocks, kinds, 1, 4, True)
    assert {t.name for t in taps} == {"attn", "norms"}
    assert {"agree", "constant", "gist_drift", "attn_window", "attn_agree"} <= set(ctrl.sources)
    att = CA2Attachment(model, ctrl, tap=1, taps=taps)
    ctrl.reset()
    ctrl.record = True
    model(input_ids=torch.randint(0, 256, (1, 9)), use_cache=False)
    att.remove()
    tt = ctrl.trace_tensors()
    assert "tap/attn/elig" in tt and "src/gist_drift/lag1" in tt and "tap/norms/norm" in tt


def test_trace_dir_inside_repo_is_refused(text_file, torch_kernels):
    from ca2 import text as tx
    with pytest.raises(SystemExit):
        tx.main(["--tiny", "--tiny-arch", "qwen3", "--text-file", str(text_file), "--trace-dir", str(ROOT / "tmp_tr"),
                 "--out", str(text_file.parent / "x.json")])
    assert not (ROOT / "tmp_tr").exists()


# --------------------------------------------------------------------------------------------- end to end
def _text_argv(tmp, text_file, arch, extra=()):
    return ["--tiny", "--tiny-arch", arch, "--text-file", str(text_file), "--out", str(tmp / "text.json"),
            "--trace-dir", str(tmp / "traces"), "--directions", "gist,mean,random:0,foreign,window_gist",
            "--gains", "0.3", "--impulse-directions", "gist", "--n-seqs", "4", "--boot", "200", *extra]


def test_text_end_to_end_hybrid(tmp_path, text_file, torch_kernels, monkeypatch):
    from ca2 import analyse, text as tx
    d = tx.main(_text_argv(tmp_path, text_file, "qwen35"))
    arms = d["arms"]
    assert set(d["conditions"]) == {"full", "attn", "all"}
    for c in ("full", "attn", "all"):
        assert arms[f"{c}/none"]["checks"]["gain0_equal"] is True
        assert arms[f"{c}/none"]["agg"]["kl_base"] == 0.0
        assert arms[f"{c}/impulse:gist@0.3"]["checks"]["imp_pre_kl_max"] == 0.0     # nothing before the burst
        assert arms[f"{c}/gist@0.3"]["agg"]["kl_base"] > 0
        assert arms[f"{c}/gist@0.3"]["gain_dist"]["mean"] == pytest.approx(0.3)
    assert "full/window_gist@0.3" not in arms and "attn/window_gist@0.3" in arms
    w = arms["attn/gist@0.3"]
    assert abs(w["agg"]["loss_base"] - arms["attn/none"]["agg"]["loss"]) < 1e-9
    assert abs(w["agg"]["loss_full"] - arms["full/none"]["agg"]["loss"]) < 1e-9
    assert "d_loss_vs_none" in w["agg"] and w["agg"]["closure"]["n_units"] == 4
    # offset / position buckets (closure, loss and Δloss by bucket) cover exactly the scored tokens; the first
    # block (steered, no window gap) is only in the descriptive all-token buckets
    for c in ("full", "attn", "all"):
        pi = arms[f"{c}/gist@0.3"]["per_item"]
        for b in ("off", "pos"):
            assert np.allclose(np.sum(pi[f"{b}_n"], 1), pi["n_tok"]), (c, b)
            assert np.all(np.sum(pi[f"{b}all_n"], 1) > np.asarray(pi["n_tok"])), (c, b)
            sc = np.sum(pi[f"{b}_loss_sum"], 1)
            assert np.allclose(sc, pi["nll_sum"], rtol=1e-5), (c, b)
    assert len(w["agg"]["closure_by_off"]) == len(w["agg"]["loss_by_offall"])
    # traces: every registered signal, main and subsample streams, hashed
    man = json.loads(Path(w["traces"]["manifest"]).read_text())
    keys = {k for f in man["files"] for k in f.get("keys", {})}
    for k in ("out__loss", "out__kl_full", "out__norm_ratio", "src__gdn_write__s", "src__gdn_mismatch__x",
              "tap__gdn__beta", "tap__gdn__fam", "tap__gdn__fam_head", "tap__attn__elig", "src__attn_agree__s",
              "src__gist_drift__lag16", "cont__lp_p1", "lens__rank_d1", "tap__norms__norm", "pos__offset"):
        assert k in keys, k
    # resume: a second run skips every complete arm
    called = []
    monkeypatch.setattr(tx, "run_arm", lambda *a, **k: called.append(1))
    tx.main(_text_argv(tmp_path, text_file, "qwen35"))
    assert not called
    out = analyse.main([str(tmp_path / "text.json"), "--boot", "100", "--traces", "--pred-arms", "gist"])
    r = out["results"][0]
    assert "attn/gist@0.3" in r["primary"]["p_holm"] and "gist@0.3" in r["gdn_state"]
    assert r["signal_predictors"]["attn/gist@0.3"]["top"]


def test_text_deterministic_and_comparator(tmp_path, text_file, torch_kernels):
    from ca2 import text as tx
    a = tx.main(_text_argv(tmp_path / "a", text_file, "qwen3", ["--conditions", "full,attn,all"]))
    b = tx.main(_text_argv(tmp_path / "b", text_file, "qwen3", ["--conditions", "full,attn,all"]))
    assert set(a["conditions"]) == {"full", "attn"} and a["conditions_skipped"]
    for k, v in a["arms"].items():
        assert v["agg"]["loss"] == b["arms"][k]["agg"]["loss"], k
        assert v["per_item"]["nll_sum"] == b["arms"][k]["per_item"]["nll_sum"], k
    g = tx.main(["--tiny", "--tiny-arch", "gpt2", "--text-file", str(text_file), "--out", str(tmp_path / "g.json"),
                 "--trace-dir", str(tmp_path / "gtr"), "--directions", "gist", "--gains", "0.3",
                 "--impulse-directions", "", "--conditions", "full,attn"])
    assert g["arms"]["attn/none"]["checks"]["gain0_equal"] is True


def test_ceiling_end_to_end(tmp_path, text_file, torch_kernels):
    from ca2 import analyse, ceiling as ce
    d = ce.main(["--tiny", "--tiny-arch", "qwen35", "--text-file", str(text_file), "--out", str(tmp_path / "c.json"),
                 "--trace-dir", str(tmp_path / "ctr"), "--sites", "1", "--directions", "gist,random:0",
                 "--gains", "0,0.2,2", "--gen-directions", "gist", "--branch-gains=-1,1", "--branch-sets", "pre",
                 "--branch-singles", "", "--boot", "100"])
    arms = d["arms"]
    assert arms["L1/full/none"]["checks"]["gain0_equal"] is True
    assert arms["branch/full/none"]["checks"]["gain0_equal"] is True
    want = {"branch/full/branch:pre:gdn@-1", "branch/full/branch:pre:attn@-1", "branch/attn/branch:pre:mlp@1"}
    assert want <= set(arms)
    assert arms["gen/L1/gist@2"]["agg"]["n"] == 4 and "collapse_rate" in arms["gen/L1/none"]["agg"]
    nr = arms["L1/full/gist@2"]["agg"]["norm_ratio_by_block"]
    assert nr[0] == pytest.approx(1.0) and nr[1] != pytest.approx(1.0)     # blocks below the injection unchanged
    assert d["branch_conditions"]["attn"]                         # windowed attention, GDN intact
    k = d["per_site"]["1"]["gist"]
    assert set(k) >= {"g_sat", "g_col", "g_knee", "g_on"}
    assert d["recommend"]["site"] == 1
    out = analyse.main([str(tmp_path / "c.json"), "--boot", "100"])
    r = out["results"][0]
    assert r["branch"]["full"]["gdn_vs_controls"] and r["curves"]["L1"]["gist"]


def test_ceiling_branch_conditions_are_validated_and_all_resets_gdn():
    from ca2 import ceiling as ce, text as tx
    with pytest.raises(SystemExit):
        ce.parse_args(["--tiny", "--branch-conditions", "full,windowed"])
    a = ce.parse_args(["--tiny", "--branch-conditions", "full,attn,all"])
    assert [tx.CONDITIONS[c] for c in a.branch_conditions.split(",")] == ["attention", "attention", "all"]


def test_ceiling_recommend_ranks_a_site_without_a_knee_above_one_with_a_knee():
    from ca2 import ceiling as ce
    args = ce.parse_args(["--tiny", "--gains", "0,0.05,0.5,2", "--directions", "gist"])

    def arm(kl, absdent, norm):
        return {"complete": True, "agg": {"kl_base": kl, "absdent": absdent, "norm_ratio_final": norm}}
    arms = {}
    for site, knee in ((1, 0.5), (2, None)):
        for g in (0.05, 0.5, 2.0):
            sat = knee is not None and g >= knee
            arms[f"L{site}/full/gist@{g:g}"] = arm(1.0, 10.0 if sat else 0.0, 1.0)
    res = {"arms": arms, "sites": [1, 2], "model": {"kinds": ["linear_attention", "full_attention"] * 2}}
    out = ce.knees(res, args)
    assert out["per_site"]["1"]["gist"]["g_knee"] == 0.5 and out["per_site"]["2"]["gist"]["g_knee"] is None
    assert out["recommend"]["site"] == 2 and out["recommend"]["g_max"] == 2.0
