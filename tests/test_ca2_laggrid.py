"""Time-lagged cross-layer injection (``experiments/ca2/laggrid.py``) on the CPU: the source constructions
against brute force, strict causality of every source and variant (vectors and logits), ``beyond:W`` never
reading the current window, the foreign and shuffle controls, gain 0 reproducing the unsteered logits bitwise,
the feedback approximation flagged with a finite relative change, zero trainable parameters, tiny end-to-end
runs on both tiny architectures with ``--analyse``, and the resume guard."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import pretrained_loop as pl  # noqa: E402
from ca2 import laggrid as lg  # noqa: E402
from ca2 import measure as ms  # noqa: E402

ALL_SOURCES = ["lag:1", "lag:5", "ema:4", "ema:8:skip=3", "slowfast:16:2", "beyond:6"]
ARGS = SimpleNamespace(seed=0, shuffle_gap=5, seq_len=40)


@pytest.fixture
def text_file(tmp_path):
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "river", "stone", "light", "north", "seven", "table"]
    p = tmp_path / "text.txt"
    p.write_text(" ".join(words[i] for i in rng.integers(0, len(words), 30_000)))
    return p


@pytest.fixture
def torch_kernels(monkeypatch):
    """Pure-torch Qwen3.5 GDN kernels for the CPU, restored afterwards (pattern of test_ca2_runners.py)."""
    if not hasattr(transformers, "Qwen3_5TextConfig"):
        pytest.skip("transformers has no Qwen3.5 classes")
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        if not hasattr(M, name):
            pytest.skip(f"{name} not found")
        monkeypatch.setattr(M, name, getattr(M, name))
    ms.swap_torch_gdn()
    return M


def tiny(arch: str):
    a = SimpleNamespace(tiny=True, tiny_arch=arch, device="cpu", dtype="float32", seed=0, attn_impl="sdpa")
    return ms.load_model(a)


def states(B=3, T=40, D=8, seed=0):
    return torch.randn(B, T, D, generator=torch.Generator().manual_seed(seed))


def arm(variant, source="lag:1", L=1, Lp=2, g=0.3, d=8):
    return {"variant": variant, "source": source, "L": L, "Lp": Lp, "gain": g, "seed": 0, "d_model": d}


# --------------------------------------------------------------------------------------------- sources
def test_sources_match_brute_force():
    H = states().double()
    B, T, D = H.shape

    def m(t):
        return H[:, :t].mean(1) if t else torch.zeros(B, D, dtype=H.dtype)

    def ema(j, tau):
        lam = math.exp(-1 / tau)
        w = torch.tensor([lam ** (j - p) for p in range(j + 1)], dtype=H.dtype)
        return (w[None, :, None] * H[:, : j + 1]).sum(1) / w.sum()

    for spec in ALL_SOURCES:
        src = lg.parse_source(spec)
        u, ok = lg.source_vectors(H, src)
        for t in range(T):
            if src["kind"] == "lag":
                v = H[:, t - src["k"]] - m(t) if t >= src["k"] else None
            elif src["kind"] == "ema":
                j = t - 1 - src["skip"]
                v = ema(j, src["tau"]) - m(t) if j >= 0 else None
            elif src["kind"] == "slowfast":
                v = ema(t - 1, src["tau_s"]) - ema(t - 1, src["tau_f"]) if t >= 1 else None
            else:
                b = (t // src["W"]) * src["W"]
                v = H[:, :b].mean(1) if b else None
            if v is None:
                assert not ok[:, t].any(), (spec, t)
                continue
            good = v.norm(dim=-1) > 1e-4 * m(t).norm(dim=-1)
            assert torch.equal(ok[:, t], good), (spec, t)
            want = v / v.norm(dim=-1, keepdim=True)
            assert torch.allclose(u[:, t][good], want[good], atol=1e-9), (spec, t)
    assert lg.parse_source("ema:16:skip4")["name"] == lg.parse_source("ema:16:skip=4")["name"] == "ema:16:skip4"
    for bad in ("lag:0", "slowfast:2:8", "beyond", "gist", "ema:4:x3"):
        with pytest.raises(ValueError):
            lg.parse_source(bad)


@pytest.mark.parametrize("spec", ALL_SOURCES)
@pytest.mark.parametrize("variant", ["true", "foreign", "shuffle"])
def test_source_vectors_are_strictly_causal(spec, variant):
    """Changing the states at positions >= t0 leaves u_t for t <= t0 bitwise unchanged (u_t reads H_{<t})."""
    H = states()
    rows = [0, 5, 9]
    a = arm(variant, spec)
    u, ok = lg.variant_vectors(a, H, rows, ARGS)
    for t0 in (3, 11, 25):
        H2 = H.clone()
        H2[:, t0:] = torch.randn(H2[:, t0:].shape, generator=torch.Generator().manual_seed(t0)) * 5
        u2, ok2 = lg.variant_vectors(a, H2, rows, ARGS)
        assert torch.equal(u[:, : t0 + 1], u2[:, : t0 + 1]) and torch.equal(ok[:, : t0 + 1], ok2[:, : t0 + 1])
    assert not torch.equal(u, lg.variant_vectors(a, H * 2 + 1, rows, ARGS)[0])     # the vectors do read H


def test_beyond_never_reads_the_current_window():
    H = states(T=48)
    W = 8
    u, ok = lg.source_vectors(H, lg.parse_source(f"beyond:{W}"))
    assert not ok[:, :W].any() and ok[:, W:].all()
    for t in range(W, 48):
        b = (t // W) * W
        H2 = H.clone()
        H2[:, b:] += 100.0                                         # every token of t's block (and later)
        u2, ok2 = lg.source_vectors(H2, lg.parse_source(f"beyond:{W}"))
        assert torch.equal(u[:, t], u2[:, t]), t
        if b >= W and b - 1 >= 0:                                   # the last token before the block does count
            H3 = H.clone()
            H3[:, b - 1] += 100.0
            assert not torch.equal(lg.source_vectors(H3, lg.parse_source(f"beyond:{W}"))[0][:, t], u[:, t])


def test_foreign_shuffle_and_random_controls():
    H = states()
    rows = [0, 5, 9]
    ut, okt = lg.variant_vectors(arm("true", "ema:4"), H, rows, ARGS)
    uf, okf = lg.variant_vectors(arm("foreign", "ema:4"), H, rows, ARGS)
    B = H.shape[0]
    for b in range(B):
        assert torch.equal(uf[b], ut[(b + 1) % B]) and torch.equal(okf[b], okt[(b + 1) % B])
        assert not torch.equal(uf[b], ut[b])
    nf, nt = uf.norm(dim=-1), ut.norm(dim=-1)
    assert torch.allclose(nf[okf], torch.ones_like(nf[okf]), atol=1e-6)
    assert torch.allclose(nt[okt], torch.ones_like(nt[okt]), atol=1e-6) and okf.sum() == okt.sum()
    with pytest.raises(ValueError):
        lg.variant_vectors(arm("foreign", "ema:4"), H[:1], rows[:1], ARGS)
    idx0 = lg.shuffle_index(rows, 40, ARGS.shuffle_gap, 0)
    t = torch.arange(40)
    assert ((idx0 == -1) == (t < ARGS.shuffle_gap)[None]).all() and (idx0 <= (t - ARGS.shuffle_gap).clamp_min(-1)).all()
    idx = lg.shuffle_index(rows, 40, ARGS.shuffle_gap, 0, okt)
    us, oks = lg.variant_vectors(arm("shuffle", "ema:4"), H, rows, ARGS)
    for b in range(B):
        for tt in range(40):
            if idx[b, tt] >= 0:
                assert idx[b, tt] <= tt - ARGS.shuffle_gap and okt[b, idx[b, tt]]       # only defined sources
            if idx[b, tt] >= 0 and okt[b, tt]:
                assert oks[b, tt] and torch.equal(us[b, tt], ut[b, idx[b, tt]])
            else:
                assert not oks[b, tt] and not us[b, tt].any()
    ur, okr = lg.variant_vectors(arm("random", None, None, 2, d=8), H, rows, ARGS)
    assert okr.all() and torch.equal(ur[:, 0], ur[:, -1]) and not torch.equal(ur[0], ur[1])
    assert torch.allclose(ur.norm(dim=-1), torch.ones(B, 40))
    ur7, okr7 = lg.variant_vectors({**arm("random", None, None, 2, d=8), "start": 7}, H, rows, ARGS)
    assert torch.equal(ur7, ur) and not okr7[:, :7].any() and okr7[:, 7:].all()


@pytest.mark.parametrize("spec", ["lag:1", "lag:64", "ema:256", "slowfast:256:8", "beyond:64"])
def test_controls_steer_the_true_arms_tokens(spec):
    """Review finding (dose): at the real defaults (T = 1024, G = 128, steer from 32) the shuffle arm steers exactly
    the true arm's tokens from t_late = max(steer_from, G + t_src) on, and the paired random arm exactly the true
    arm's tokens; ``late_from`` / ``start`` of ``make_arms`` agree with the masks."""
    args = SimpleNamespace(seed=0, shuffle_gap=128, steer_from=32, sources=spec, gains="0.05",
                           variants="true,foreign,shuffle,random", random_seeds="0", arms=None)
    H = states(B=2, T=1024, D=8)
    rows = [3, 11]
    src = lg.parse_source(spec)
    a = {name: x for x in lg.make_arms(args, [(1, 2)], 8) for name in [x["variant"]]}
    first = lambda ok: int(ok.float().argmax())                                       # noqa: E731
    on = {}
    for v in ("true", "foreign", "shuffle", "random"):
        u, ok = lg.variant_vectors(a[v], H, rows, args)
        o = ok.clone()
        o[:, : args.steer_from] = False                                     # as the Injector does
        on[v] = o
    t_late = lg.late_start(src, 32, 128)
    assert a["true"]["late_from"] == a["shuffle"]["late_from"] == t_late
    assert torch.equal(on["random"], on["true"]) and torch.equal(on["foreign"], on["true"])
    assert first(on["true"][0]) == lg.steer_start(src, 32)
    assert torch.equal(on["shuffle"][:, t_late:], on["true"][:, t_late:]) and not on["shuffle"][:, :t_late].any()
    scored = slice(64, 1024)
    frac = {v: float(on[v][:, scored].float().mean()) for v in on}
    assert frac["true"] == 1.0 and frac["random"] == 1.0
    assert frac["shuffle"] == pytest.approx((1024 - max(64, t_late)) / 960)
    if lg.random_start(src, 32) is None:
        assert a["random"]["name"] == "random:0:to2@0.05"
    else:
        assert a["random"]["name"] == f"random:0:from{lg.steer_start(src, 32)}:to2@0.05"


def test_beyond_must_be_a_multiple_of_the_window():
    """Review finding: beyond:W with W not a multiple of --window would average tokens inside the window."""
    base = SimpleNamespace(window=64, sources="lag:1,beyond:64")
    assert [s["name"] for s in lg.check_sources(base)] == ["lag:1", "beyond:64"]
    assert lg.check_sources(SimpleNamespace(window=64, sources="beyond:128"))
    for w, s in ((128, "beyond:64"), (64, "beyond:96"), (16, "beyond:8")):
        with pytest.raises(SystemExit):
            lg.check_sources(SimpleNamespace(window=w, sources=s))
    with pytest.raises(SystemExit):
        lg.main(["--tiny", "--window", "128", "--sources", "beyond:64", "--count"])
    with pytest.raises(SystemExit):
        lg.main(["--tiny", "--device", "cpu", "--window", "16", "--sources", "beyond:24"])


def test_bh_and_selection_need_corrected_evidence():
    """Review finding (forking paths): one strong cell survives BH; 40 marginal intervals (p = 0.04 each, one per
    cell would pass the old uncorrected rule) do not; the top-K cells are added by t statistic."""
    assert lg.bh([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])

    def b(pp, d):
        return {"diff": d, "lo": d - 1, "hi": d + 1, "se": 1.0, "p_le_0": pp / 2, "p_ge_0": 1 - pp / 2}

    def rec(L, Lp, p, diff, early_p=1.0):
        return {"L": L, "Lp": Lp, "minus_foreign": b(p, diff), "minus_foreign_early": b(early_p, 0.0),
                "vs_none": {"diff": 0.1, "lo": 0.05, "hi": 0.15, "se": 0.01, "p_le_0": 0.0, "p_ge_0": 1.0},
                "vs_none_early": {"diff": 0.1, "lo": 0.05, "hi": 0.15, "se": 0.01, "p_le_0": 0.0, "p_ge_0": 1.0}}
    recs = {f"c{L}-{Lp}": rec(L, Lp, 0.04, 0.5) for L in range(8) for Lp in range(5)}
    recs["strong"] = rec(9, 9, 1e-5, -3.0)
    data = {"contrasts": {"attn": recs}}
    sel = lg.selection(data, 0)
    assert sel["cells"] == [(9, 9)] and sel["n_tests"] == 4 * 41
    recs["early"] = rec(9, 8, 1.0, 0.0, early_p=1e-6)                       # an early-offset-only effect counts
    assert (9, 8) in lg.select_cells(data, 0)
    assert set(lg.select_cells(data, 1)) == {(9, 9), (9, 8)}


# ------------------------------------------------------------------------------------------- injection
def _lm(model, window_layers="attention"):
    lm = pl.HookedLM(model, side=None, layer=0, window=8, mode="block", window_layers=window_layers, sink=1,
                     pad_id=256)
    lm.side_on = False
    return lm


@pytest.mark.parametrize("arch", ["qwen35", "qwen3"])
def test_gain_zero_reproduces_the_unsteered_logits(arch, torch_kernels):
    model, _, blocks, kinds = tiny(arch)
    g = torch.Generator().manual_seed(1)
    ids = torch.cat([torch.full((2, 1), 256), torch.randint(0, 256, (2, 30), generator=g)], 1)
    lm = _lm(model, "all" if arch == "qwen35" else "attention")
    for win in (0, 8):
        with lm.condition(win, False):
            base = lm(ids)
        for Lp in range(len(blocks)):
            u, ok = lg.variant_vectors(arm("random", None, None, Lp, d=64), torch.zeros(2, 30, 1), [0, 1], ARGS)
            inj = lg.Injector(blocks[Lp], u, ok, 0.0, 2, 1)
            with lm.condition(win, False):
                out = lm(ids)
            inj.remove()
            assert torch.equal(out, base), (win, Lp)
            assert inj.rec["ratio"].abs().max() == 0 and inj.on[:, :2].sum() == 0 and inj.on[:, 2:].all()
        inj = lg.Injector(blocks[1], u, ok, 0.3, 2, 1)
        with lm.condition(win, False):
            out = lm(ids)
        inj.remove()
        assert torch.equal(out[:, :3], base[:, :3]) and not torch.equal(out, base)    # nothing before steer_from
        assert torch.allclose(inj.rec["ratio"][:, 2:], torch.full((2, 28), 0.3), atol=1e-6)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0
    for h in lm.handles:
        h.remove()


@pytest.mark.parametrize("arch", ["qwen35", "qwen3"])
def test_injection_is_strictly_causal_end_to_end(arch, torch_kernels):
    """Pass 1 + pass 2 for every source type: changing tokens after t0 leaves the injected vectors up to t0 + 1
    and the logits up to t0 bitwise unchanged, in full and windowed conditions."""
    model, _, blocks, kinds = tiny(arch)
    g = torch.Generator().manual_seed(2)
    T, s, t0 = 30, 1, 17
    ids = torch.cat([torch.full((2, 1), 256), torch.randint(0, 256, (2, T), generator=g)], 1)
    ids2 = ids.clone()
    ids2[:, s + t0 + 1:] = torch.randint(0, 256, (2, T - t0 - 1), generator=g)
    lm = _lm(model, "all" if arch == "qwen35" else "attention")
    L, Lp = 2, 1                                                    # a feedback cell: the hardest case
    for win in (0, 8):
        for spec in ALL_SOURCES:
            for variant in ("true", "foreign", "shuffle"):
                outs = []
                for x in (ids, ids2):
                    cap = lg.StateCapture(blocks, [L], s)
                    with lm.condition(win, False):
                        lm(x)
                    cap.remove()
                    u, ok = lg.variant_vectors(arm(variant, spec, L, Lp, d=64), cap.h[L], [0, 1], ARGS)
                    inj = lg.Injector(blocks[Lp], u, ok, 0.5, 2, s)
                    with lm.condition(win, False):
                        z = lm(x)
                    inj.remove()
                    outs.append((u, ok, z))
                (u1, k1, z1), (u2, k2, z2) = outs
                assert torch.equal(u1[:, : t0 + 2], u2[:, : t0 + 2]) and torch.equal(k1[:, : t0 + 2], k2[:, : t0 + 2])
                assert torch.equal(z1[:, : s + t0 + 1], z2[:, : s + t0 + 1]), (win, spec, variant)
                assert not torch.equal(z1, z2)
    for h in lm.handles:
        h.remove()


def test_lens_recent_counts_recent_tokens():
    model, _, blocks, kinds = tiny("qwen3")
    head = model.get_output_embeddings().weight
    ids = torch.randint(0, 256, (1, 40), generator=torch.Generator().manual_seed(3))
    # u_t along the unembedding row of x_t itself: its top-1 lens token is (almost surely) a recent token
    u = torch.nn.functional.normalize(head[ids[0]].float(), dim=-1)[None]
    on = torch.ones(1, 40, dtype=torch.bool)
    on[:, :3] = False
    r = lg.lens_recent(model, u, torch.full((1, 40), 30.0), on, ids, k=1, n_recent=4)
    assert torch.isnan(r[:, :3]).all() and r[:, 3:].mean() > 0.5
    r0 = lg.lens_recent(model, u, torch.full((1, 40), 30.0), on, torch.full_like(ids, 300), k=1, n_recent=4)
    assert (r0[:, 3:] == 0).all()


# --------------------------------------------------------------------------------------------- end to end
def _argv(tmp, text_file, arch, extra=()):
    return ["--tiny", "--tiny-arch", arch, "--device", "cpu", "--text-file", str(text_file), "--out",
            str(tmp / "lag.json"), "--trace-dir", str(tmp / "traces"), "--boot", "100", *extra]


def test_end_to_end_hybrid_feedback_flag_analyse_and_resume(tmp_path, text_file, torch_kernels, monkeypatch, capsys):
    argv = _argv(tmp_path, text_file, "qwen35", ["--sources", "lag:1,ema:4:skip=2,beyond:16", "--group", "7"])
    d = lg.main(argv)
    arms = d["arms"]
    assert set(d["conditions"]) == {"full", "attn", "all"} and d["trainable_parameters"] == 0
    assert d["cell_kinds"] == {"1-0": "feedback", "1-1": "same", "1-2": "skip", "2-0": "feedback", "2-1": "feedback",
                               "2-2": "same"}
    n_cond = 6 * 3 * 3 + 3 * 2 + 1                  # random arms: t_0 = 8 (lag:1, ema) and t_0 = 16 (beyond:16)
    assert len(arms) == 3 * n_cond == 3 * lg.count_arms(lg.parse_args(argv))["arms_per_condition"]
    assert d["lens_depths"] == [0, 1, 2] and d["compat"]["attn_impl"] == "sdpa" and "version_torch" in d["compat"]
    for c in ("full", "attn", "all"):
        none = arms[f"{c}/none"]
        assert none["checks"]["gain0_equal"] is True and none["agg"]["kl_base"] == 0.0
        assert none["agg"]["d_loss_vs_none"]["diff"] == 0.0 and none["agg"]["dent"] == 0.0
        for k, a in arms.items():
            if not k.startswith(c + "/"):
                continue
            ag = a["agg"]
            assert a["git"] == d["git"]["commit"] and a["versions"]["torch"] == torch.__version__
            assert 0 <= ag["top1"] <= 1 and ag["ent"] > 0 and ag["absdent"] >= abs(ag["dent"])
            assert {"lens_logrank_d0", "lens_top1_d2"} <= set(ag)
            if a["variant"] in ("none", "random"):
                continue
            assert ag["kl_base"] > 0 and abs(ag["inj_ratio"] - 0.3) < 1e-5 and math.isfinite(ag["fb_relchange"])
            assert ag["loss_base"] == pytest.approx(none["agg"]["loss"], abs=1e-9)
            assert ag["ent_base"] == pytest.approx(none["agg"]["ent"], abs=1e-9)
            assert np.allclose(np.sum(a["per_item"]["off_n"], 1), a["per_item"]["n_tok"])
            assert np.allclose(np.sum(a["per_item"]["offw_n"], 1), a["per_item"]["n_tok"])
            assert len(ag["dloss_by_offw"]) == 16 and ag["norm_ratio_by_block"][: a["Lp"]] == [1.0] * a["Lp"]
            if a["cell_kind"] == "feedback":
                assert a["exact"] is False and "open-loop" in a["approximation"] and "not bounds" in a["approximation"]
                assert ag["fb_relchange"] > 0 and -1 <= ag["fb_ucos"] <= 1
            else:                                       # skip and same: the source is unchanged, bitwise
                assert a["exact"] is True and a["checks"]["source_unchanged"] is True
                assert ag["fb_relchange"] == 0.0 and ag["fb_ucos"] is None
        if c != "full":
            assert "closure" in arms[f"{c}/true:lag:1:L1to2@0.3"]["agg"]
    # the controls steer the true arm's tokens: random fully, shuffle on t >= t_late
    for c in ("full", "attn", "all"):
        for name, cr in d["contrasts"][c].items():
            assert cr["dose_random"] == pytest.approx(1.0) and cr["dose_foreign"] == pytest.approx(1.0), name
            assert cr["dose_shuffle_matched"] == pytest.approx(1.0) and cr["dose_shuffle"] < 1.0, name
    cr = d["contrasts"]["attn"]["true:beyond:16:L2to0@0.3"]
    assert {"minus_foreign", "minus_shuffle", "minus_shuffle_matched", "minus_random", "vs_none"} <= set(cr)
    assert {"minus_foreign_early", "minus_shuffle_matched_early", "minus_random_early", "vs_none_early"} <= set(cr)
    assert cr["t0"] == 16 and cr["late_from"] == 32
    assert arms["attn/random:0:from16:to0@0.3"]["agg"]["steered_frac"] == pytest.approx(
        arms["attn/true:beyond:16:L2to0@0.3"]["agg"]["steered_frac"])
    man = json.loads(Path(arms["attn/true:lag:1:L2to0@0.3"]["traces"]["manifest"]).read_text())
    keys = {k: v for f in man["files"] for k, v in f.get("keys", {}).items()}
    for k in ("out__loss", "out__ent", "out__rank", "out__top1", "out__p_tgt", "out__p_max", "out__ent_base",
              "out__kl_base", "out__kl_full", "out__loss_full", "inj__ratio", "inj__cos_uh", "lens__recent",
              "lens__rank_d0", "lens__rank_d2", "out__norm_ratio", "fb__relchange", "fb__ucos", "pos__token",
              "pos__offset", "pos__block"):
        assert k in keys, k
    assert keys["pos__token"]["dtype"] == "int32"
    for k in ("top1_sum", "ent_sum", "dent_sum", "absdent_sum", "late_off_loss_sum", "late_off_n"):
        assert k in arms["attn/true:lag:1:L2to0@0.3"]["per_item"], k
    # --analyse: matrices per condition and source, intervals, significance marks, the gain rule, BH
    out = lg.main(["--analyse", str(tmp_path / "lag.json"), "--boot", "100"])
    m = out["conditions"]["attn"]["lag:1"]
    assert m["rows_L"] == [1, 2] and m["cols_Lp"] == [0, 1, 2] and m["gain"] == 0.3 and "0.3" in m["by_gain"]
    assert m["matrix"]["2-0"]["lo"] <= m["matrix"]["2-0"]["diff"] <= m["matrix"]["2-0"]["hi"]
    assert set(m["q"]) == set(m["matrix"]) and out["n_tests"] == 3 * 3 * 6
    txt = capsys.readouterr().out
    assert "true - foreign" in txt and "POST HOC" in txt and "expected by chance" in txt and "BH" in txt
    out = lg.main(["--analyse", str(tmp_path / "lag.json"), "--boot", "100", "--control", "shuffle", "--bucket",
                   "early", "--at-gain", "0.3"])
    assert out["key"] == "minus_shuffle_matched_early" and "--at-gain" in out["gain_rule"]
    assert "POST HOC" not in capsys.readouterr().out
    assert lg.select_cells(d, 2) and set(lg.select_cells(d, 100)) == {(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)}
    # resume: a second run skips every complete arm; other settings (also the attention kernel) are refused
    called = []
    monkeypatch.setattr(lg, "run_group", lambda *a, **k: called.append(1))
    lg.main(argv)
    assert not called
    for extra in (["--steer-from", "4"], ["--attn-impl", "eager"]):
        with pytest.raises(SystemExit):
            lg.main(argv + extra)


def test_end_to_end_attention_model_and_screen(tmp_path, text_file):
    argv = _argv(tmp_path, text_file, "qwen3", ["--sources", "lag:8,slowfast:32:4", "--cells", "1-2,2-1",
                                                "--variants", "true,foreign"])
    d = lg.main(argv)
    assert set(d["conditions"]) == {"full", "attn"} and d["conditions_skipped"]
    assert set(d["arms"]) == {f"{c}/{n}" for c in ("full", "attn") for n in
                              ["none"] + [f"{v}:{s}:L{L}to{Lp}@0.3" for v in ("true", "foreign")
                                          for s in ("lag:8", "slowfast:32:4") for L, Lp in ((1, 2), (2, 1))]}
    assert d["arms"]["attn/foreign:lag:8:L2to1@0.3"]["cell_kind"] == "feedback"
    assert d["task"]["row_offset"] == 0 and d["task"]["row_starts"] == [0, 96, 192, 288]
    s = lg.parse_args(["--screen", "--model", "x"])
    assert s.gains == "0.05" and s.n_seqs == 128 and s.batch == 2 and s.seq_len == 1024
    c = lg.count_arms(lg.parse_args(["--tiny", "--tiny-arch", "qwen3", "--screen"]))
    assert c["conditions"] == ["full", "attn"] and c["arms_per_condition"] == 6 * 6 * 3 + 3 * 2 + 1


def test_cells_from_needs_disjoint_rows(tmp_path, text_file):
    """Review finding (forking paths): a --cells-from follow-up on the screen's rows is refused; with
    --row-offset it runs on other rows, keeps the row layout, and records the selection's provenance."""
    scr = _argv(tmp_path, text_file, "qwen3", ["--sources", "lag:1", "--cells", "1-2,2-1", "--variants",
                                               "true,foreign", "--conditions", "attn", "--screen"])
    d0 = lg.main(scr)
    (tmp_path / "f").mkdir()
    with pytest.raises(SystemExit, match="--row-offset 4"):
        lg.main(_argv(tmp_path / "f", text_file, "qwen3", ["--sources", "lag:1", "--variants", "true,foreign",
                                                           "--conditions", "attn", "--cells-from",
                                                           str(tmp_path / "lag.json"), "--cells-top", "1"]))
    (tmp_path / "g").mkdir()
    d = lg.main(_argv(tmp_path / "g", text_file, "qwen3", ["--sources", "lag:1", "--variants", "true,foreign",
                                                           "--conditions", "attn", "--cells-from",
                                                           str(tmp_path / "lag.json"), "--cells-top", "1",
                                                           "--row-offset", "4"]))
    p = d["selection_provenance"]
    assert p["rows_disjoint"] and p["source_rows"]["row_offset"] == 0 and p["this_rows"]["row_offset"] == 4
    assert "BH" in p["criterion"] and len(p["top_cells"]) == 1 and set(map(tuple, d["cells"])) == set(
        map(tuple, p["cells"]))
    assert d["task"]["row_starts"] == [384, 480, 576, 672] and d["task"]["batches"] == [[4, 6], [5, 7]]
    assert d0["task"]["batches"] == [[0, 2], [1, 3]]
    a0, a1 = d0["arms"]["attn/none"]["per_item"], d["arms"]["attn/none"]["per_item"]
    assert a0["row"] == [0, 2, 1, 3] and a1["row"] == [4, 6, 5, 7] and a1["start"][0] == 384
