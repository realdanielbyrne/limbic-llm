"""C1 / C3 thinking-control runner (experiments/ca2/think.py): arm grammar and the c1 preset, then tiny end-to-end
runs: every arm kind, pairing across arms (real truncation keeps the untruncated arm's tokens up to the budget),
probe-cost accounting of controllers, forked against re-prefill probes, the traces stoprules.py reads, no text
in the results JSON, resume and the settings guard.

Tiny random models with the byte tokenizer; no downloads.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments" / "alignment")]

from ca2 import decode as D  # noqa: E402
from ca2 import stoprules as S  # noqa: E402
from ca2 import think as T  # noqa: E402
from ca2.common import _sha256  # noqa: E402

ARMS = ("a@T=0.6;pp=0,b@T=1.0;pp=1.5,nt@think=off,tr@T=1.0;pp=1.5;trunc=16,"
        "ng@T=1.0;ctrl=ngram-16-2-0.3:stop,cs@T=1.0;ctrl=consist-2:stop,cb@T=1.0;ctrl=consist-2:bias3,"
        "ws@T=1.0;suppress=wait")


# ------------------------------------------------------------------------------------------- grammar
def test_arm_grammar_and_c1_preset():
    a = T.parse_arm("x@T=1.0;pp=1.5;top_k=20;rp=1.1;trunc=1024")
    assert (a.temperature, a.presence, a.top_k, a.repetition, a.trunc, a.thinking) == (1.0, 1.5, 20, 1.1, 1024, True)
    assert T.parse_arm("y@think=off").thinking is False
    assert T.parse_ctrl("ngram-256-8-0.5+consist-3:bias4") == ([("ngram", (256, 8, 0.5)), ("consist", (3,))],
                                                                "bias", 4.0)
    for bad in ("@T=1", "x y@T=1", "x@Q=1", "x@think=maybe", "x@think=off;trunc=5", "x@ctrl=ngram-1:stop",
                "x@ctrl=consist-2:jump", "x@trunc=5;ctrl=consist-2:bias2", "x@suppress=hmm"):
        with pytest.raises(ValueError):
            T.parse_arm(bad)
    c1 = [T.parse_arm(s) for s in T.PRESETS["c1"]]
    grid = {(x.temperature, x.presence) for x in c1 if x.thinking and x.trunc is None}
    assert grid == {(0.6, 0.0), (1.0, 0.0), (1.0, 1.5)}
    assert [x for x in c1 if not x.thinking] and [x.trunc for x in c1 if x.trunc] == [1024]
    assert c1[0].label == "t06" and all(x.top_p == 0.95 and x.top_k == 20 for x in c1)
    assert D.CANONICAL_CLOSE == ("\n\nConsidering the limited time by the user, I have to give the solution based "
                                 "on the thinking directly now.\n</think>\n\n")


# ------------------------------------------------------------------------------------------- end to end
def _argv(tmp_path, name, arms, *extra, arch="qwen3"):
    return ["--tiny", "--tiny-arch", arch, "--device", "cpu", "--data", "synthetic:4", "--batch", "3",
            "--max-new-tokens", "160", "--probe-every", "8", "--budgets", "8,16",
            "--arms", arms, "--out", str(tmp_path / f"{name}.json"), "--trace-dir", str(tmp_path / f"tr_{name}"),
            *extra]


def _items(tmp_path, name, key):
    d = tmp_path / f"tr_{name}" / key / "tokens"
    items = [json.loads(l) for l in (d / "items.jsonl").read_text().splitlines()]
    z = np.load(d / "tokens_0000.npz")
    return items, {k: z[k] for k in z.files}


def test_end_to_end_tiny_every_arm_kind_resume_and_guard(tmp_path, capsys):
    js = T.main(_argv(tmp_path, "t", ARMS))
    out = tmp_path / "t.json"
    arms = ARMS.split(",")
    assert list(js["arms"]) == arms and all(a["complete"] for a in js["arms"].values())
    for k in ("args", "git", "versions", "model", "data", "close_text"):
        assert k in js
    for spec, a in js["arms"].items():
        pi = a["per_item"]
        assert len(pi["uid"]) == 4 and pi["uid"] == [f"synthetic/{i}" for i in range(4)]
        assert {"acc", "n_reason_mean", "ttc_km", "acc_b8", "acc_b16", "nonclose_rate", "tokens_total_mean",
                "rep_rate", "distinct4_mean", "self_loop_mean", "wall_s_sum"} <= set(a["agg"])
        man = Path(a["trace"]["manifest"])
        assert _sha256(man) == a["trace"]["manifest_sha256"]
        if spec != arms[0]:
            assert a["vs_ref"]["ref"] == "a" and a["vs_ref"]["n_pairs"] == 4 and "d_acc" in a["vs_ref"]
    nt = js["arms"]["nt@think=off"]
    assert nt["agg"]["nonclose_rate"] is None and all(x is None for x in nt["per_item"]["natural_close"])
    # real truncation: the controller acts at exactly 16 reasoning tokens with the canonical close; the tokens
    # before it are the untruncated arm's (same decoding, seed and batch)
    tr = js["arms"]["tr@T=1.0;pp=1.5;trunc=16"]["per_item"]
    assert tr["ctrl_act_span"] == [16] * 4 and tr["n_reason"] == [16] * 4 and all(tr["forced"])
    ib, zb = _items(tmp_path, "t", "b")
    it, zt = _items(tmp_path, "t", "tr")
    close = D.close_sequence(T.R.build_tiny("qwen3", 0)[1], T.R.TINY_THINK_END)
    for x, y in zip(ib, it):
        P = x["n_prompt"]
        a_ = zb["bk__ids"][x["tok_offset"]:x["tok_offset"] + x["n_tok"]]
        b_ = zt["bk__ids"][y["tok_offset"]:y["tok_offset"] + y["n_tok"]]
        assert np.array_equal(a_[:P + 16], b_[:P + 16])
        assert b_[P + 16:P + 16 + len(close)].tolist() == close
        assert y["forced_from"] == 16
    # the forced answer runs under the arm's one cap (--max-new-tokens 160), not a separate answer budget: a
    # forced row the cap cut has generated all 160 tokens, so a long answer after the close is not scored as cut
    for ng, tc in zip(tr["n_gen"], tr["truncated"]):
        assert ng == 160 if tc else ng < 160
    assert any(n > 16 + len(close) + 16 for n in tr["n_gen"])        # answers longer than any small answer budget
    ag = js["arms"]["tr@T=1.0;pp=1.5;trunc=16"]["agg"]
    assert ag["forced_truncated_rate"] == np.mean(tr["truncated"]) and ag["forced_raw_acc"] is not None
    assert "forced_truncated_rate" in js["arms"]["b@T=1.0;pp=1.5"]["agg"]
    # probe-driven controller: charged per probe run up to (and including) the one that triggered it
    cs = js["arms"]["cs@T=1.0;ctrl=consist-2:stop"]["per_item"]
    for ng, n_p, tot, act in zip(cs["n_gen"], cs["n_ctrl_probes"], cs["tokens_total"], cs["ctrl_act_step"]):
        assert tot == ng + n_p * T.DEFAULT_PROBE_COST
        if act is not None and act >= 0:
            assert n_p == act // 8 and n_p >= 2
    assert all(x == 0 for x in js["arms"]["b@T=1.0;pp=1.5"]["per_item"]["n_ctrl_probes"])
    cb = js["arms"]["cb@T=1.0;ctrl=consist-2:bias3"]["per_item"]
    assert not any(cb["forced"])                                      # the bias action never forces tokens
    # the trace holds text and the per-token keys stoprules.py reads; the results JSON holds no text
    txt = out.read_text()
    assert all("text" in x for x in ib)
    assert not any(x["text"] and len(x["text"]) > 8 and json.dumps(x["text"])[1:-1] in txt for x in ib)
    for k in ("bk__ids", "bk__reason", "out__ent", "out__logp_next", "out__p_close"):
        assert k in zb and zb[k].shape[0] == sum(x["n_tok"] for x in ib)
    trs, _ = S.load_traces(tmp_path / "tr_t" / "b")
    assert len(trs) == 4 and all(t.conf is not None and len(t.conf) == len(t.cks) for t in trs)
    # resume: complete arms are skipped, the file is unchanged; other settings are refused
    before = json.loads(out.read_text())["arms"]
    T.main(_argv(tmp_path, "t", ARMS))
    assert "skipped" in capsys.readouterr().out
    assert json.loads(out.read_text())["arms"] == before
    for extra in (("--batch", "2"), ("--probe-every", "16"), ("--seed", "1"), ("--max-new-tokens", "100"),
                  ("--probe-cost", "10")):
        with pytest.raises(SystemExit):
            T.main(_argv(tmp_path, "t", ARMS, *extra))
    with pytest.raises(SystemExit):                                   # a label re-used for another setting
        T.main(_argv(tmp_path, "t", "a@T=0.7;pp=0"))
    assert json.loads(out.read_text())["arms"] == before


def test_forked_probes_equal_reprefill_probes(tmp_path):
    arms = "a@T=1.0;pp=1.5"
    fork = T.main(_argv(tmp_path, "f", arms, "--probe-mode", "fork"))["arms"][arms]["per_item"]
    pre = T.main(_argv(tmp_path, "p", arms, "--probe-mode", "prefill"))["arms"][arms]["per_item"]
    for k in ("n_gen", "n_reason", "cks", "probe_ok", "acc_b8", "acc_b16", "ttc"):
        assert fork[k] == pre[k], k
    assert all(fork["n_online_probes"]) and not any(pre["n_online_probes"])
    for a, b in zip(fork["probe_conf"], pre["probe_conf"]):
        assert np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-4, atol=1e-6)


def test_end_to_end_tiny_hybrid(tmp_path):
    if not hasattr(transformers, "Qwen3_5TextConfig"):
        pytest.skip("transformers has no Qwen3.5 classes")
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    saved = {n: getattr(M, n) for n in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")}
    try:
        arms = "a@T=1.0;pp=1.5,tr@T=1.0;pp=1.5;trunc=16"
        js = T.main(_argv(tmp_path, "h", arms, arch="qwen35"))
    finally:
        for n, f in saved.items():
            setattr(M, n, f)
    assert js["model"]["hybrid"]
    tr = js["arms"]["tr@T=1.0;pp=1.5;trunc=16"]["per_item"]
    un = js["arms"]["a@T=1.0;pp=1.5"]["per_item"]
    for k in range(4):                                     # forced at 16, or closed by itself before (as untruncated)
        if tr["forced"][k]:
            assert tr["n_reason"][k] == 16 and tr["ctrl_act_span"][k] == 16
        else:                                              # untouched rows: the untruncated arm's row, whole
            assert tr["natural_close"][k] and tr["n_reason"][k] == un["n_reason"][k] < 16
            assert (tr["n_gen"][k], tr["correct"][k], tr["truncated"][k]) == \
                (un["n_gen"][k], un["correct"][k], un["truncated"][k])
    assert any(tr["forced"])


def test_trace_dir_inside_the_repo_is_refused(tmp_path):
    with pytest.raises(ValueError):
        T.main(["--tiny", "--device", "cpu", "--arms", "a@T=0.6", "--out", str(tmp_path / "x.json"),
                "--trace-dir", str(ROOT / "results" / "nope_think_traces")])
    assert not (ROOT / "results" / "nope_think_traces").exists()
