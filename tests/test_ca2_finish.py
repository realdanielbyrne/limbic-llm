"""C2 / C3-inject (experiments/ca2/finish.py): direction construction (deterministic, stated positions only), the
span- and stop-gated CA2 injection, gain laws and their range, gain 0 reproducing the unsteered arm token for token,
the yoked and matched-mean controls, the teacher-forced screen's labels and selectivity on synthetic traces, the
truncation frontier, and tiny end-to-end runs of the three subcommands.

Tiny random models with the byte tokenizer; no downloads.
"""

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments" / "alignment")]

from ca2 import decode as D  # noqa: E402
from ca2 import finish as Fn  # noqa: E402
from ca2 import reason as R  # noqa: E402
from ca2.common import TraceWriter  # noqa: E402
from ca2.controller import CA2Attachment, CA2Controller  # noqa: E402
from ca2.sources import ConstantSource, StepContext  # noqa: E402

BOS = PAD = STOP = 256
CLOSE = R.TINY_THINK_END


def tiny(arch="qwen3"):
    model, _ = R.build_tiny(arch, 0)
    return model.float().eval()


# ------------------------------------------------------------------------------------------- directions
def test_random_direction_is_the_controllers_own():
    for s in (0, 3):
        c = CA2Controller(64, direction="random", seed=s)
        f = Fn.FinishController(64, None, seed=s)
        assert torch.equal(Fn.random_direction(64, s), c.d_rand) and torch.equal(f.d_fixed, c.d_rand)
        assert torch.equal(Fn.FinishController(64, None, sign=-1.0, seed=s).d_fixed, -c.d_rand)


def test_conclude_positions():
    assert Fn.conclude_positions(10, 100, 110, 4) == ([106, 107, 108, 109], [58, 59, 60, 61])
    assert Fn.conclude_positions(10, 100, None, 4) is None
    assert Fn.conclude_positions(10, 12, 22, 4) is None                     # windows closer than k
    assert Fn.conclude_positions(10, 100, 120, 4) is None                   # close after the span


def _row(ids, P, n, close, cks=(), oks=(), answers=(), uid="u", natural=True):
    ids = np.asarray(ids, np.int64)
    return Fn.TraceRow(uid, 0, ids, P, n, close, natural, True, list(cks), list(oks), list(answers),
                       np.zeros(len(ids), np.float32))


def _seq(rng, P, n):
    return [BOS] + [rng.randrange(32, 127) for _ in range(P - 1)] + [rng.randrange(32, 127) for _ in range(n)] + \
        [CLOSE, 10, 10, 52, STOP]


def test_directions_are_deterministic_and_read_only_the_stated_positions():
    model = tiny()
    blocks = R.pl.find_blocks(model)
    rng = random.Random(0)
    on = [[BOS] + [rng.randrange(32, 127) for _ in range(rng.randrange(6, 12))] for _ in range(5)]
    off = [p + [CLOSE, 10, 10] for p in on]
    a = Fn.think_direction(model, blocks, on, off, [1, 3], PAD, batch=2)
    b = Fn.think_direction(model, blocks, on, off, [1, 3], PAD, batch=2)
    for L in (1, 3):
        assert torch.equal(a[L]["d"], b[L]["d"])
        cap = {}
        h = blocks[L].register_forward_hook(lambda m, x, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        man = []
        for p, q in zip(on, off):                             # unpadded single passes
            model(input_ids=torch.tensor([q]))
            hq = cap["h"][0, -1]
            model(input_ids=torch.tensor([p]))
            man.append(hq - cap["h"][0, -1])
        h.remove()
        assert torch.allclose(a[L]["d"], torch.stack(man).mean(0), atol=1e-5)
    # conclude: states only at C and M; tokens after the close never matter, a token inside C does
    rows = [_row(_seq(rng, 8, 40), 8, 40, 48, uid=f"r{i}") for i in range(3)]
    c0 = Fn.conclude_direction(model, blocks, rows, [2], PAD, k=4)
    alt = [_row(np.concatenate([r.ids[:49], (r.ids[49:] + 1) % 128]), 8, 40, 48, uid=r.uid) for r in rows]
    assert torch.equal(c0[2]["d"], Fn.conclude_direction(model, blocks, alt, [2], PAD, k=4)[2]["d"])
    ch = [_row(r.ids.copy(), 8, 40, 48, uid=r.uid) for r in rows]
    ch[0].ids[46] = (ch[0].ids[46] + 1) % 128                 # inside C (44..47)
    assert not torch.equal(c0[2]["d"], Fn.conclude_direction(model, blocks, ch, [2], PAD, k=4)[2]["d"])
    st = Fn.states_at(model, blocks, [r.ids.tolist() for r in rows], [[44, 45, 46, 47, 26, 27, 28, 29]] * 3, [2], PAD)
    man = torch.stack([s[:4].mean(0) - s[4:].mean(0) for s in st[2]]).mean(0)
    assert torch.allclose(c0[2]["d"], man, atol=1e-6) and c0[2]["stats"]["n_rows"] == 3
    unnat = [_row(r.ids, 8, 40, 48, natural=False) for r in rows]
    assert Fn.conclude_direction(model, blocks, unnat, [2], PAD, k=4)[2]["d"] is None


# ------------------------------------------------------------------------------------------- injection gate
def _manual(model, ctrl, prompts, steps):
    ids, mask = R.pad_left(prompts, PAD)
    B = ids.shape[0]
    ctrl.reset()
    ctrl.pending_mask = mask
    pos = (mask.cumsum(1) - 1).clamp_min(0)
    o = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    cache, logits = o.past_key_values, [o.logits]
    for t in steps:
        mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype)], 1)
        o = model(input_ids=torch.tensor(t)[:, None], attention_mask=mask, position_ids=mask.sum(1, keepdim=True) - 1,
                  past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        logits.append(o.logits)
    return ctrl.gain_log(), logits


@pytest.mark.parametrize("arch", ["qwen3", "qwen35"])
def test_injection_acts_only_inside_the_span_and_never_on_finished_rows(arch, request):
    if arch == "qwen35":
        request.getfixturevalue("torch_gdn")
    model = tiny(arch)
    d = torch.randn(64)
    prompts = [[BOS, 70, 71, 72, 10], [BOS, 70, 10], [BOS, 73, 74, 10]]       # BOS is a stop id: prompt exempt
    steps = [[65, 65, 65], [66, 66, STOP], [CLOSE, 67, PAD], [67, 68, PAD], [STOP, 69, PAD], [PAD, 70, PAD]]
    ctrl = Fn.FinishController(64, d, stops=[STOP], source=ConstantSource(1.0), g_min=0.0, g_max=0.1,
                               smooth="none", span_end_id=CLOSE, log_gain=True)
    att = CA2Attachment(model, ctrl, tap=1, inject_before=2)
    G, steered = _manual(model, ctrl, prompts, steps)       # steps[t][b]: the token fed to row b at step t
    att.remove()
    want = torch.tensor([[1, 1, 1, 0, 0, 0, 0],               # closes at step 3 (the close itself is not steered)
                         [1, 1, 1, 1, 1, 1, 1],               # open throughout
                         [1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)   # stop inside the span: finished from it
    assert torch.equal(G[2], want)
    assert torch.allclose(G[0], 0.1 * want) and ((G[0] == 0) | (G[0] == G[0].max())).all()
    plain = Fn.FinishController(64, d, stops=[STOP], source=ConstantSource(1.0), g_min=0.0, g_max=0.0,
                                smooth="none", span_end_id=CLOSE, log_gain=True)
    att = CA2Attachment(model, plain, tap=1, inject_before=2)
    G0, base = _manual(model, plain, prompts, steps)
    att.remove()
    assert (G0[0] == 0).all()
    _, mask = R.pad_left(prompts, PAD)
    P = mask.shape[1]
    assert torch.equal(steered[0][:, : P - 1], base[0][:, : P - 1])           # prompt columns before the last
    assert not torch.equal(steered[0][:, -1], base[0][:, -1])
    # and with the model unattached, g = 0 equals the bare model bitwise
    _, bare = _manual(model, Fn.FinishController(64, d, g_max=0.0, log_gain=True), prompts, steps)
    assert all(torch.equal(x, y) for x, y in zip(bare, base))


@pytest.fixture
def torch_gdn():
    if not hasattr(transformers, "Qwen3_5TextConfig"):
        pytest.skip("transformers has no Qwen3.5 classes")
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    saved = {n: getattr(M, n) for n in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")}
    R.torch_gdn_kernels()
    yield
    for n, f in saved.items():
        setattr(M, n, f)


def test_paused_controller_is_inert():
    model = tiny()
    ctrl = Fn.FinishController(64, torch.randn(64), source=ConstantSource(1.0), g_max=0.2, smooth="none",
                               log_gain=True)
    att = CA2Attachment(model, ctrl, tap=1, inject_before=2)
    ids = torch.tensor([[BOS, 70, 71, 72]])
    ctrl.reset()
    ctrl.paused = True
    a = model(input_ids=ids).logits
    assert ctrl.calls == 0 and not ctrl.glog
    ctrl.paused = False
    b = model(input_ids=ids).logits
    att.remove()
    c = model(input_ids=ids).logits
    assert torch.equal(a, c) and not torch.equal(b[:, -1], c[:, -1])


# ------------------------------------------------------------------------------------------- gain laws
def _ctx(step, ids=None, steer=None):
    B, T = step.shape
    m = torch.ones(B, T)
    return StepContext(h=torch.zeros(B, T, 4), x=torch.zeros(B, T, 4), mask=m, use=m,
                       steer=m if steer is None else steer, idx=m, step=step, col=torch.arange(T), first=False,
                       call=1, ids=ids, bank=None)


def test_gain_laws():
    e = Fn.ElapsedSource(10, 2)
    s = e(_ctx(torch.tensor([[0, 2, 7, 12, 30]])))["s"]
    assert torch.allclose(s, torch.tensor([[0.0, 0.0, 0.5, 1.0, 1.0]]))
    c = Fn.ConsistSource(3)
    c.set_rows(2)
    seq = [("12", "5"), ("12.0", "6"), ("12", None), ("13", "6")]
    lv = []
    for a0, a1 in seq:
        c.observe([0, 1], [a0, a1])
        lv.append(c.level.tolist())
    assert lv == [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 0.0]]
    assert c(_ctx(torch.zeros(2, 3, dtype=torch.long)))["s"].shape == (2, 3)
    r = Fn.RepeatSource(2, 8, 0.5)
    ids = torch.tensor([[1, 2, 1, 2, 1, 2]])
    out = r(_ctx(torch.zeros(1, 6, dtype=torch.long), ids=ids))
    assert out["rep"].tolist() == [[0, 0, 0, 1, 1, 1]] and 0 < out["s"][0, -1] < 1
    law = Fn.make_law("elapsed-10+consist-3")
    assert [type(p).__name__ for p in law.parts] == ["ElapsedSource", "ConsistSource"]
    assert Fn.consist_parts(law) == [law.parts[1]]
    for bad in ("elapsed", "consist-1", "rep-2", "foo-3", "elapsed-0"):
        with pytest.raises(ValueError):
            Fn.make_law(bad)


def test_arm_grammar_and_order():
    a = Fn.parse_finish_arm("d@T=0.6;pp=0;inj=-think;block=14;gmin=0;gmax=0.2;law=consist-3")
    assert (a.temperature, a.presence, a.inject.kind, a.inject.sign, a.inject.block, a.inject.mode) == \
        (0.6, 0.0, "think", -1.0, 14, "law")
    assert a.spec.endswith("law=consist-3") and Fn.parse_finish_arm("x@T=1.0").inject is None
    assert Fn.parse_finish_arm("m@inj=random;g=match:d").inject.match == "d"
    for bad in ("x@inj=think", "x@g=0.1", "x@inj=up;g=0.1", "x@inj=think;g=0.1;law=consist-3;gmin=0;gmax=1",
                "x@inj=think;law=consist-3", "x@inj=think;gmin=0.3;gmax=0.2;law=consist-3",
                "x@inj=think;yoke=d;gmax=0.2", "x@think=off;inj=think;g=0.1", "x@inj=think;g=2"):
        with pytest.raises(ValueError):
            Fn.parse_finish_arm(bad)
    arms = [Fn.parse_finish_arm(s) for s in Fn.preset_arms("T=0.6;pp=0", "think", 14, 0.2, "consist-3", 0.1)]
    assert [x.label for x in arms] == ["dyn", "fixm", "yoke", "rand", "neg", "el", "fix"]
    assert arms[4].inject.sign == -1.0 and arms[3].inject.kind == "random"
    Fn.check_arm_order(arms)
    with pytest.raises(ValueError):
        Fn.check_arm_order(arms[1:])
    with pytest.raises(ValueError):
        Fn.check_arm_order([Fn.parse_finish_arm("f@inj=think;g=0.1"), Fn.parse_finish_arm("y@inj=think;yoke=f")])


# ------------------------------------------------------------------------------------------- screen measures
def test_labels_and_selectivity_on_synthetic_traces():
    P, n = 5, 40
    ids = np.zeros(P + n + 1, np.int64)
    r = _row(ids, P, n, P + n, cks=[8, 16, 24, 32, 40], oks=[False, False, True, True, True],
             answers=["3", "4", "7", "7.0", "7"])
    f, u = Fn.formation_labels(r, "correct")
    m = np.arange(len(ids)) - P + 1
    assert (f == ((m >= 24) & (m <= n))).all() and (u == ((m >= 1) & (m <= 16))).all()
    f2, u2 = Fn.formation_labels(r, "stable")
    assert (f2 == ((m >= 24) & (m <= n))).all() and (u2 == ((m >= 1) & (m <= 16))).all()
    r2 = _row(ids, P, n, P + n, cks=[8, 16, 24, 32, 40], oks=[False, True, False, True, True],
              answers=["3", "7", "9", "7", "7"])
    f3, u3 = Fn.formation_labels(r2, "correct")
    assert (f3 == ((m >= 32) & (m <= n))).all() and (u3 == ((m >= 1) & (m <= 24))).all()
    cens = _row(ids, P, n, P + n, cks=[8, 16], oks=[True, False], answers=["7", "3"])
    assert not Fn.formation_labels(cens, "correct")[0].any() and not Fn.formation_labels(cens, "correct")[1].any()
    # Δ = 2 where formed, 0.5 where not, 9 in the unlabelled gap; boundaries tagged
    rows = []
    for k, (a, b) in enumerate([(2.0, 0.5), (3.0, 1.0), (1.0, 0.0)]):
        delta = np.where(f, a, np.where(u, b, 9.0))[P:P + n]
        bnd = np.zeros(n)
        bnd[[3, 30]] = 1
        e = Fn.row_effects(delta, np.full(n, 0.1), np.zeros(n), f[P:P + n], u[P:P + n], bnd, np.ones(n, bool))
        assert (e["F"], e["U"], e["Fb"], e["Ub"], e["nF"], e["nU"]) == (a, b, a, b, 17, 16)
        rows.append(e)
    s = Fn.selectivity(rows)
    assert s["n_rows"] == 3 and math.isclose(s["E_F"], 2.0) and math.isclose(s["E_U"], 0.5)
    assert math.isclose(s["ratio"], 4.0) and math.isclose(s["diff"]["diff"], 1.5) and s["diff"]["lo"] > 0
    rand = Fn.selectivity([dict(e, F=1.0, U=1.0) for e in rows])
    assert Fn.screen_gate(s, [rand])["pass"]
    assert not Fn.screen_gate(s, [Fn.selectivity([dict(e, F=3.0, U=1.2) for e in rows])])["pass"]   # 4 < 2 × 2.5
    neg = Fn.selectivity([dict(e, U=-0.2) for e in rows])
    assert neg["ratio"] is None and neg["sign_selective"] and Fn.screen_gate(neg, [rand])["pass"]
    assert not Fn.screen_gate(Fn.selectivity([dict(e, F=-1.0) for e in rows]), [rand])["pass"]


def test_position_matched_contrast_removes_a_pure_position_effect():
    """Δ rising with the span offset alone (no formation effect) gives E_F − E_U > 0 but a matched contrast ≈ 0; a
    label effect at fixed offset survives the matching. Rows form at different offsets, so bins hold both labels."""
    rng = np.random.default_rng(0)
    rows_pos, rows_lab = [], []
    n, width, nb = 96, 16, 6                                     # probes every 16 span tokens, bins of 16
    m = np.arange(1, n + 1)
    for k in range(24):
        ttc = 32 + 16 * (k % 4)                                  # formed from ttc on, not formed up to ttc − 16
        f, u = m >= ttc, m <= ttc - 16
        noise = rng.normal(0, 0.01, n)
        for rows, delta in ((rows_pos, 0.02 * m + noise), (rows_lab, 0.5 * f + noise)):
            e = Fn.row_effects(delta, np.zeros(n), np.zeros(n), f, u, np.zeros(n), np.ones(n, bool))
            e["Fbin"], e["Ubin"] = Fn.binned_effects(delta, f, u, m, width, nb)
            rows.append(e)
    pos, lab = Fn.selectivity(rows_pos), Fn.selectivity(rows_lab)
    assert pos["diff"]["lo"] > 0.5                               # the confound: later positions, larger Δ
    mp, ml = Fn.matched_contrast(rows_pos, 500), Fn.matched_contrast(rows_lab, 500)
    assert mp["n_bins"] == 2 and abs(mp["diff"]) < 0.01 and mp["lo"] < 0 < mp["hi"]   # same offsets: nothing left
    assert ml["lo"] > 0.4 and math.isclose(ml["diff"], 0.5, abs_tol=0.02)
    rnd = dict(Fn.selectivity([dict(e, F=1.0, U=1.0) for e in rows_pos]), matched={"diff": 0.0})
    gp = Fn.screen_gate({**pos, "matched": mp}, [rnd])
    assert gp["pass"] and not gp["pass_matched"]
    gl = Fn.screen_gate({**lab, "matched": ml}, [rnd])
    assert gl["pass"] and gl["pass_matched"]


def test_frontier_envelope_and_distances():
    env = Fn.envelope([(900, 0.83), (1200, 0.80), (600, 0.76), (2400, 0.82), (3500, 0.21)])
    assert env == [(600, 0.76), (900, 0.83), (1200, 0.83), (2400, 0.83), (3500, 0.83)]
    assert Fn.env_acc(env, 500) is None and math.isclose(Fn.env_acc(env, 750), 0.795)
    assert math.isclose(Fn.env_tokens(env, 0.795), 750) and Fn.env_tokens(env, 0.9) is None
    assert Fn.env_tokens(env, 0.70) is None and Fn.env_tokens(env, 0.76) == 600
    N = 50
    mk = lambda acc, tok: {"uid": [f"u{i}" for i in range(N)], "sample": [0] * N,
                           "correct": [i < acc * N for i in range(N)], "tokens_total": [tok] * N}
    front = {"a": mk(0.6, 500), "b": mk(0.8, 1000)}
    d = Fn.frontier_distance(mk(0.7, 600), front, n_boot=200, other_pi=mk(0.7, 700))
    assert math.isclose(d["horizontal"]["value"], 150) and math.isclose(d["vertical"]["value"], 0.7 - 0.64)
    assert math.isclose(d["horizontal_minus_other"]["value"], 100)


def test_import_refuses_other_settings(tmp_path):
    src = {"args": {"seed": 1, "batch": 32}, "data": {"uids": ["a", "b"]},
           "arms": {"x@T=1": {"key": "x", "complete": True, "agg": {}, "per_item": {}, "spec": "x@T=1"}}}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(src))
    assert set(Fn.import_arms({}, [f"{p}:x"], {"seed": 1, "batch": 32}, ["a", "b"])) == {"x"}
    for args, uids in (({"seed": 0, "batch": 32}, ["a", "b"]), ({"seed": 1, "batch": 32}, ["a"])):
        with pytest.raises(SystemExit):
            Fn.import_arms({}, [f"{p}:x"], args, uids)
    with pytest.raises(SystemExit):
        Fn.import_arms({}, [f"{p}:y"], {"seed": 1, "batch": 32}, ["a", "b"])


# ------------------------------------------------------------------------------------------- end to end
def _fake_traces(root: Path, n_rows: int = 6) -> Path:
    """A think.py-layout trace of tiny rows that close the span themselves, with probe records."""
    rng = random.Random(1)
    w = TraceWriter("fake", root=root)
    for i in range(n_rows):
        P, n = 12, 48
        seq = _seq(rng, P, n)
        cks = R.checkpoints(n, 8)
        oks = [c >= 24 for c in cks]
        rec = dict(uid=f"synthetic/{i}", sample=0, n_prompt=P, n_reason=n, closed=True, forced_from=None,
                   truncated=False, correct=True, cks=cks, probe_ok=oks,
                   probe_answers=["1" if not o else "2" for o in oks])
        bk = R.bookkeeping(seq, P, n, n, None, cks, oks, lambda t: chr(t) if t < 128 else "")
        w.add("tokens", rec, {k: v for k, v in bk.items()})
    w.close()
    return root / "fake"


def _run_argv(tmp, name, arms, *extra):
    return ["run", "--tiny", "--device", "cpu", "--data", "synthetic:4", "--batch", "3", "--max-new-tokens", "160",
            "--probe-every", "8", "--budgets", "8,16", "--arms", arms, "--out", str(tmp / f"{name}.json"),
            "--trace-dir", str(tmp / f"tr_{name}"), "--name", name, *extra]


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("finish")
    tr = _fake_traces(tmp / "traces")
    dj = Fn.main(["directions", "--tiny", "--device", "cpu", "--name", "dirs", "--out", str(tmp / "dirs.json"),
                  "--dir-out", str(tmp / "dirs.pt"), "--conclude-traces", str(tr), "--window", "4",
                  "--min-conclude", "2"])
    arms = ("a@T=1.0,z@T=1.0;inj=think;block=2;g=0,z2@T=1.0;inj=conclude;block=2;gmin=0;gmax=0;law=elapsed-8,"
            "dyn@T=1.0;inj=think;block=2;gmin=0.05;gmax=0.25;law=elapsed-24+consist-2,"
            "yk@T=1.0;inj=think;block=2;yoke=dyn,fm@T=1.0;inj=think;block=2;g=match:dyn,"
            "rd@T=1.0;inj=random;block=2;gmin=0;gmax=0.25;law=rep-2-16,"
            "ti@T=1.0;trunc=16;inj=think;block=2;g=0.1,f16@T=1.0;trunc=16")
    js = Fn.main(_run_argv(tmp, "r", arms, "--dirs", str(tmp / "dirs.pt"), "--frontier", "f16,a", "--bias", "fm",
                           "--trunc-ref", "f16", "--random-arm", "rd"))
    return tmp, tr, dj, js


def _tok(tmp, name, key):
    d = tmp / f"tr_{name}" / key / "tokens"
    items = [json.loads(line) for line in (d / "items.jsonl").read_text().splitlines()]
    z = np.load(d / "tokens_0000.npz")
    return items, [z["bk__ids"][it["tok_offset"]:it["tok_offset"] + it["n_tok"]] for it in items]


def _arm(js, label):
    return next(v for v in js["arms"].values() if v["key"] == label)


def test_directions_file_and_report(e2e):
    tmp, tr, dj, _ = e2e
    blob = torch.load(tmp / "dirs.pt", weights_only=False)
    assert blob["blocks"] == [2, 3] and set(blob["dirs"]) == {"think", "conclude"}
    assert dj["conclude"]["rows"]["fit_windows"] == 6 and dj["conclude"]["rows"]["enough"]
    assert all(blob["dirs"][k][L] is not None for k in ("think", "conclude") for L in (2, 3))
    assert dj["think"]["stats"]["2"]["n_rows"] == 16 and "think|conclude" in dj["cosines"]["2"]


def test_gain_zero_reproduces_the_unsteered_arm_token_for_token(e2e):
    tmp, _, _, js = e2e
    _, base = _tok(tmp, "r", "a")
    for k in ("z", "z2"):
        _, ids = _tok(tmp, "r", k)
        assert all(np.array_equal(x, y) for x, y in zip(ids, base)), k
        assert _arm(js, k)["per_item"]["n_gen"] == _arm(js, "a")["per_item"]["n_gen"]
    _, dyn = _tok(tmp, "r", "dyn")
    assert not all(np.array_equal(x, y) for x, y in zip(dyn, base))      # a non-zero gain does change the text


def test_gain_range_span_gate_and_records(e2e):
    tmp, _, _, js = e2e
    for lab, lo, hi in (("dyn", 0.05, 0.25), ("rd", 0.0, 0.25), ("ti", 0.1, 0.1), ("yk", 0.05, 0.25)):
        res = _arm(js, lab)
        gain = Fn.load_gain(Path(res["setup"]["inject"]["gain_file"]["path"]))
        assert len(gain) == 4
        for (uid, k), (g, r) in gain.items():
            assert g.size and (g >= lo - 1e-6).all() and (g <= hi + 1e-6).all(), lab
            assert np.allclose(g, lo + (hi - lo) * r, atol=1e-6)
        pi = res["per_item"]
        for i in range(4):
            assert pi["inj_n_steered"][i] == len(gain[(pi["uid"][i], 0)][0])
    ti, pre = _arm(js, "ti")["per_item"], D.close_sequence(R.pl.ByteTokenizer(), CLOSE).index(CLOSE)
    for i in range(4):                                       # last prompt column + 16 span tokens + forced text
        if ti["forced"][i] and ti["n_gen"][i] > 16 + pre + 1:
            assert ti["inj_n_steered"][i] == 1 + 16 + pre
    dyn = _arm(js, "dyn")
    g = Fn.load_gain(Path(dyn["setup"]["inject"]["gain_file"]["path"]))
    for (uid, k), (gg, r) in g.items():                      # elapsed-24 dominates early: r_k >= k / 24
        kk = np.arange(len(r))
        assert (r >= np.clip(kk / 24, 0, 1) - 1e-6).all()
    pi = dyn["per_item"]                                     # the consistency law is charged its probes
    assert all(t == n + p * 25 for t, n, p in zip(pi["tokens_total"], pi["n_gen"], pi["n_ctrl_probes"]))
    assert all(p == q for p, q in zip(pi["n_ctrl_probes"], pi["n_online_probes"])) and any(pi["n_ctrl_probes"])


def test_yoked_and_matched_controls(e2e):
    tmp, _, _, js = e2e
    dyn, yk, fm = _arm(js, "dyn"), _arm(js, "yk"), _arm(js, "fm")
    gd = Fn.load_gain(Path(dyn["setup"]["inject"]["gain_file"]["path"]))
    gy = Fn.load_gain(Path(yk["setup"]["inject"]["gain_file"]["path"]))
    uids = dyn["per_item"]["uid"]
    for i, u in enumerate(uids):
        donor = uids[(i + 1) % len(uids)]
        assert yk["per_item"]["inj_donor"][i] == donor != u
        rd, ry = gd[(donor, 0)][1], gy[(u, 0)][1]
        want = rd[np.minimum(np.arange(len(ry)), len(rd) - 1)]
        assert np.allclose(ry, want, atol=1e-6)
    assert yk["setup"]["inject"]["g_min_used"] == 0.05 and yk["setup"]["inject"]["g_max_used"] == 0.25
    m = dyn["agg"]["inj_gain_mean_pooled"]
    assert math.isclose(fm["setup"]["inject"]["g_max_used"], m)
    gf = Fn.load_gain(Path(fm["setup"]["inject"]["gain_file"]["path"]))
    assert all(np.allclose(g, m, atol=1e-6) for g, _ in gf.values())
    pool = sum(float(g.sum()) for g, _ in gd.values()) / sum(len(g) for g, _ in gd.values())
    assert math.isclose(m, pool, rel_tol=1e-6)


def test_run_report_analysis_and_no_text(e2e):
    tmp, _, _, js = e2e
    an = js["analysis"]
    assert an["frontier_arms"] == ["f16", "a"] and set(an["distances"]) >= {"dyn", "rd", "fm", "f16"}
    pre = an["preregistered"]
    assert {"vs_bias", "vs_trunc", "vs_random"} <= set(pre) and pre["vs_trunc"]["paired"]["n_pairs"] == 4
    assert js["finish"]["dirs"]["sha256"] == Fn._sha256(tmp / "dirs.pt")
    raw = (tmp / "r.json").read_text()
    assert '"text"' not in raw
    # rerun: complete arms are skipped; a different direction file is refused
    js2 = Fn.main(_run_argv(tmp, "r", "a@T=1.0,dyn@T=1.0;inj=think;block=2;gmin=0.05;gmax=0.25;"
                                      "law=elapsed-24+consist-2", "--dirs", str(tmp / "dirs.pt"), "--frontier", "a"))
    for k in ("n_gen", "n_reason", "inj_gain_mean", "tokens_total"):
        assert _arm(js2, "dyn")["per_item"][k] == _arm(js, "dyn")["per_item"][k]
    other = tmp / "dirs2.pt"
    blob = torch.load(tmp / "dirs.pt", weights_only=False)
    blob["meta"]["x"] = 1
    torch.save(blob, other)
    with pytest.raises(SystemExit):
        Fn.main(_run_argv(tmp, "r", "a@T=1.0", "--dirs", str(other)))


def test_screen_end_to_end_tiny(e2e):
    tmp, tr, _, _ = e2e
    js = Fn.main(["screen", "--tiny", "--device", "cpu", "--name", "scr", "--out", str(tmp / "scr.json"),
                  "--dirs", str(tmp / "dirs.pt"), "--traces", str(tr), "--gains", "0,0.2", "--rows", "4",
                  "--n-boot", "200"])
    assert js["n_rows"] == 4 and js["directions"] == ["think", "-think", "conclude", "-conclude", "random0", "random1"]
    for L in (2, 3):
        for d in js["directions"]:
            z = js["results"][f"{L}|{d}|0|correct"]
            assert z["E_F"] == 0.0 and z["E_U"] == 0.0 and z["kl_mean"] == 0.0     # g = 0: the base model exactly
            s = js["results"][f"{L}|{d}|0.2|correct"]
            assert s["n_rows"] == 4 and s["E_F"] != 0.0 and s["kl_mean"] > 0
            assert all(r["nF"] == 25 and r["nU"] == 16 for r in s["rows"])    # m >= 24 of 48; m <= 16
    assert set(js["gates"]) == {f"{L}|{d}|{g}|{lk}" for L in (2, 3) for d in ("think", "-think", "conclude",
                                "-conclude") for g in ("0", "0.2") for lk in ("correct", "stable")}


def test_end_to_end_tiny_hybrid_gain_zero_and_law(tmp_path, torch_gdn):
    arms = ("a@T=1.0,z@T=1.0;inj=random;block=1;g=0,"
            "dyn@T=1.0;inj=random;block=1;gmin=0;gmax=0.3;law=elapsed-8+consist-2")
    js = Fn.main(_run_argv(tmp_path, "h", arms, "--tiny-arch", "qwen35", "--frontier", "a"))
    assert js["model"]["hybrid"]
    _, base = _tok(tmp_path, "h", "a")
    _, z = _tok(tmp_path, "h", "z")
    _, dyn = _tok(tmp_path, "h", "dyn")
    assert all(np.array_equal(x, y) for x, y in zip(z, base))
    assert not all(np.array_equal(x, y) for x, y in zip(dyn, base))
    g = Fn.load_gain(Path(_arm(js, "dyn")["setup"]["inject"]["gain_file"]["path"]))
    assert all(((gg >= 0) & (gg <= 0.3 + 1e-6)).all() for gg, _ in g.values())
    assert all((r >= np.clip(np.arange(len(r)) / 8, 0, 1) - 1e-6).all() and np.allclose(gg, 0.3 * r, atol=1e-6)
               for gg, r in g.values())                    # r = max(elapsed ramp over 8 steps, consistency)
    assert all(gg[-1] > 0.29 for gg, _ in g.values() if len(gg) > 8)


def test_screen_refuses_the_evaluation_problems(e2e):
    """Rows of the direction file's evaluation set (meta heldout_from) are never screened unless asked (dev)."""
    tmp, tr, _, _ = e2e
    ev = tmp / "eval.json"
    ev.write_text(json.dumps({"data": {"uids": [f"synthetic/{i}" for i in range(6)]}}))
    blob = torch.load(tmp / "dirs.pt", weights_only=False)
    blob["meta"]["heldout_from"] = str(ev)
    torch.save(blob, tmp / "dirs_eval.pt")
    argv = ["screen", "--tiny", "--device", "cpu", "--name", "scr_ev", "--out", str(tmp / "scr_ev.json"),
            "--dirs", str(tmp / "dirs_eval.pt"), "--traces", str(tr), "--gains", "0.2", "--rows", "2",
            "--blocks", "2", "--n-boot", "50", "--kinds", "conclude"]
    with pytest.raises(SystemExit):
        Fn.main(argv)
    ev.write_text(json.dumps({"data": {"uids": ["synthetic/0", "synthetic/1"]}}))
    js = Fn.main(argv)
    assert js["uids"] == ["synthetic/2", "synthetic/3"] and not js["rows_policy"]["dev_screen"]
    assert js["rows_policy"]["conclude_built_from_screen_rows"] == 2
    s = js["results"]["2|conclude|0.2|correct"]
    assert s["fresh"]["n_rows"] == 0 and "matched" in s and "pass_matched" in js["gates"]["2|conclude|0.2|correct"]
    js = Fn.main(argv + ["--allow-eval-rows"])
    assert js["uids"] == ["synthetic/0", "synthetic/1"] and js["rows_policy"]["dev_screen"]
