"""Round 12 Stage 0a runner (experiments/ca2/hybrid_diag.py) on tiny random models, CPU only.

The whole runner end to end in --tiny mode (every diagnostic, results JSON, hashed trace manifests, resume and
--force), the trace directory refusal inside the repository, and the pieces the diagnostics rest on: cache
surgery (a cache rebuilt from its parts continues like one pass; an empty GDN state equals no GDN cache;
dropping the attention cache changes the logits), the gain-0 rig being byte-identical to the base model, a
pulse leaving everything before it and every GDN layer at or below the injection block unchanged, chunked
outcomes against the model's own logits, non-overlapping text slots, and the correlation helpers.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("transformers has no Qwen3.5 classes", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments" / "alignment")]

from ca2 import hybrid_diag as hd  # noqa: E402
from ca2.common import _sha256  # noqa: E402

KERNELS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")


@pytest.fixture
def torch_gdn(monkeypatch):
    """The runner's kernel swap, undone after the test (other tests may rely on the default dispatch)."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    for name in KERNELS:
        fn = getattr(M, name, None)
        if fn is None:
            pytest.skip(f"{name} not found")
        monkeypatch.setattr(M, name, fn)                  # registers the restore
    done = hd.use_torch_gdn_kernels()
    if not done.startswith("torch"):
        pytest.skip(f"no pure-torch GDN kernels: {done}")
    return done


@pytest.fixture
def env(torch_gdn, tmp_path):
    a = hd.parse_args(["--tiny", "--device", "cpu", "--out", str(tmp_path / "r.json"),
                       "--trace-dir", str(tmp_path / "tr")])
    model, tok, blocks, kinds, facts = hd.load_any(a, "main")
    facts["which"] = "main"
    text = hd._fallback_text(tok)
    return hd.Env(a, model, tok, blocks, kinds, facts, text)


def test_tiny_end_to_end_resume_and_force(torch_gdn, tmp_path):
    out, tr = tmp_path / "r.json", tmp_path / "traces"
    argv = ["--tiny", "--device", "cpu", "--out", str(out), "--trace-dir", str(tr)]
    res = hd.main(argv)
    assert set(res["diags"]) == set(hd.DIAGS)
    disk = json.loads(out.read_text())
    for d in hd.DIAGS:
        blk = disk["diags"][d]
        assert blk["complete"] and blk["trace"]["n_files"] > 0
        man = Path(blk["trace"]["manifest"])
        assert man.is_file() and man.parent == tr / d
        for f in json.loads(man.read_text())["files"]:
            assert _sha256(man.parent / f["path"]) == f["sha256"]
    assert disk["model"]["n_trainable"] == 0 and disk["comparator"]["linear_layers"] == []
    gs = disk["diags"]["gate_shift"]["arms"]
    assert all(v["max_abs_delta_at_or_below_L"] == 0 for v in gs.values())
    for k, v in disk["diags"]["impulse"]["arms"].items():
        assert v["pre_pulse_kl_max"] == 0, k
    assert disk["diags"]["impulse_cmp"]["settings"]["gdn_signals"] == []
    stamp = {d: disk["diags"][d]["finished"] for d in hd.DIAGS}
    mt = (tr / "census" / "manifest.json").stat().st_mtime_ns
    hd.main(argv)                                                  # everything complete: skipped
    again = json.loads(out.read_text())
    assert {d: again["diags"][d]["finished"] for d in hd.DIAGS} == stamp
    assert (tr / "census" / "manifest.json").stat().st_mtime_ns == mt
    hd.main(argv + ["--diag", "census", "--force"])
    forced = json.loads(out.read_text())
    assert forced["diags"]["census"]["finished"] >= stamp["census"]
    assert forced["diags"]["census"]["summary"]["mean_loss"] == disk["diags"]["census"]["summary"]["mean_loss"]
    assert forced["diags"]["attenuation"] == disk["diags"]["attenuation"]
    assert forced["diags"]["census"]["run_settings"]["layer"] == forced["diags"]["attenuation"]["run_settings"]["layer"]
    for extra in (["--layer", "2"], ["--no-mismatch"], ["--attn-window", "8"], ["--seed", "1"]):
        with pytest.raises(SystemExit):                             # complete blocks made with other settings
            hd.main(argv + ["--diag", "gate_shift"] + extra)
    assert json.loads(out.read_text()) == forced                   # refused before anything is written


def test_trace_dir_inside_the_repository_is_refused(tmp_path):
    with pytest.raises(ValueError):
        hd.main(["--tiny", "--device", "cpu", "--out", str(tmp_path / "r.json"),
                 "--trace-dir", str(ROOT / "results" / "should_not_exist"), "--diag", "census"])
    assert not (ROOT / "results" / "should_not_exist").exists()


def test_cache_surgery(env):
    E = env
    ids = E.rows(E.pick(48, 3, hd.random.Random(0)), 47)
    P = 30
    one = hd.forward(E.model, ids)
    ctx, cont = ids[:, :P], ids[:, P:]
    parts = hd.cache_parts(hd.forward(E.model, ctx, use_cache=True).cache)
    assert set(parts["rec"]) == set(E.lin) and set(parts["kv"]) == {i for i, k in enumerate(E.kinds) if k == "attention"}
    full = hd.continue_call(E, cont, hd.make_cache(E.model, **parts), P)
    torch.testing.assert_close(full.hf, one.hf[:, P:], atol=1e-5, rtol=0)
    none = hd.continue_call(E, cont, hd.make_cache(E.model), P)
    zc = {i: torch.zeros_like(t) for i, t in parts["conv"].items()}
    zr = {i: torch.zeros_like(t) for i, t in parts["rec"].items()}
    zero = hd.continue_call(E, cont, hd.make_cache(E.model, conv=zc, rec=zr), P)
    assert torch.equal(zero.hf, none.hf)                           # an empty GDN state is no GDN cache
    rec = hd.continue_call(E, cont, hd.make_cache(E.model, conv=parts["conv"], rec=parts["rec"]), P)
    assert (rec.hf - none.hf).abs().max() > 1e-3                    # the restored state changes the output
    win = hd.kv_window(parts["kv"], P - 1, 1)                       # sink + every other column: the whole cache
    same = hd.continue_call(E, cont, hd.make_cache(E.model, kv=win, conv=parts["conv"], rec=parts["rec"]), P)
    assert torch.equal(same.hf, full.hf)
    short = hd.kv_window(parts["kv"], 4, 1)
    assert all(k.shape[-2] == 5 for k, _ in short.values())


def test_gain_zero_rig_is_byte_identical_and_a_pulse_is_causal(env):
    E = env
    ids = E.rows(E.pick(40, 2, hd.random.Random(1)), 39)
    base = hd.forward(E.model, ids)
    r0 = hd.run_arm(E, ids, gain=0.0)
    assert torch.equal(r0.hf, base.hf)
    assert "tap/gdn/fam" in r0.trace and "src/agree/s" in r0.trace and "tap/norms/norm" in r0.trace
    t0 = 12
    keep = [E.L, len(E.blocks) - 1]
    b = hd.run_arm(E, ids, keep=keep, gain=0.0, steer_from=t0)
    p = hd.run_arm(E, ids, keep=keep, direction="random", pulse=(t0, 3), gain=0.5)
    assert torch.equal(p.hf[:, :t0], b.hf[:, :t0])
    assert (p.hf[:, t0] - b.hf[:, t0]).abs().max() > 0
    g = p.trace["g_raw"]
    assert torch.all(g[:, t0:t0 + 3] == 0.5) and torch.all(g[:, :t0] == 0) and torch.all(g[:, t0 + 3:] == 0)
    hL = (p.blocks[E.L] - b.blocks[E.L]).abs().amax(-1)
    assert torch.all(hL[:, t0 + 3:] == 0)                          # block L itself is changed only where injected
    for l in E.lin:
        if l <= E.L:
            assert torch.equal(torch.nan_to_num(p.gdn[l]["beta"]), torch.nan_to_num(b.gdn[l]["beta"]))
    o = hd.outcomes(E.model, p.hf, hd.next_targets(ids), ref=b.hf)
    assert torch.all(o["kl"][:, :t0] == 0) and o["kl"][:, t0].min() > 0


def test_outcomes_match_the_model_logits(env):
    E = env
    ids = E.rows(E.pick(33, 2, hd.random.Random(2)), 32)
    logits = E.model(input_ids=ids).logits.float()
    r = hd.forward(E.model, ids)
    o = hd.outcomes(E.model, r.hf, hd.next_targets(ids), budget=7 * 512)   # several chunks
    lp = logits.log_softmax(-1)
    ref = -lp[:, :-1].gather(-1, ids[:, 1:, None])[..., 0]
    torch.testing.assert_close(o["loss"][:, :-1], ref, atol=1e-5, rtol=0)
    assert torch.isnan(o["loss"][:, -1]).all()
    torch.testing.assert_close(o["ent"], -(lp.exp() * lp).sum(-1), atol=1e-5, rtol=0)
    assert torch.equal(o["top1"][:, :-1], (logits[:, :-1].argmax(-1) == ids[:, 1:]).float())
    o2 = hd.outcomes(E.model, r.hf, hd.next_targets(ids), ref=r.hf)
    assert torch.all(o2["kl"] == 0) and torch.all(o2["agree"] == 1)


def test_pick_slots_do_not_overlap(env):
    E = env
    used = E.used()
    rng = hd.random.Random(3)
    picks = [(s, 50) for s in E.pick(50, 20, rng, used)] + [(s, 7) for s in E.pick(7, 40, rng, used)]
    picks += [(s, 333) for s in E.pick(333, 5, rng, used)]
    iv = sorted(picks)
    for (s1, l1), (s2, _) in zip(iv, iv[1:]):
        assert s1 + l1 <= s2
    assert E.pick(50, 4, hd.random.Random(9)) == E.pick(50, 4, hd.random.Random(9))


def test_helpers():
    assert hd.lag_bins(10) == [(0, 1), (1, 2), (2, 4), (4, 8), (8, 11)]
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(200, 3)), rng.normal(size=200)
    x[:, 1] += 2 * y
    x[5, 2] = np.nan
    r = hd._pearson(x, y)
    for j in range(3):
        ok = np.isfinite(x[:, j])
        assert abs(r[j] - np.corrcoef(x[ok, j], y[ok])[0, 1]) < 1e-12
    mono = np.stack([y, np.exp(y), -y ** 3], 1)
    np.testing.assert_allclose(hd._spearman(mono, y), [1.0, 1.0, -1.0], atol=1e-12)
    assert hd._clean({"a": float("nan"), "b": np.float32(1.23456789), "c": torch.tensor([1.0, float("inf")])}) == \
        {"a": None, "b": 1.23457, "c": [1.0, None]}
