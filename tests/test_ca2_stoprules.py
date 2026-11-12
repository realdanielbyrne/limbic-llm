"""C0 offline stop-rule study (experiments/ca2/stoprules.py): the loop signals against ``judge`` and brute force,
the cost accounting (probes charged in token equivalents), the frontier against ``reason.budget_acc``, the
interpolation, the problem-level split, recovery of a planted rule on synthetic traces, and reading the traces
of a real (tiny) reason.py run.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments" / "alignment")]

from ca2 import reason as R  # noqa: E402
from ca2 import stoprules as S  # noqa: E402
from ca2.common import alignment  # noqa: E402

A = alignment()


def seqs(seed: int, k: int = 24) -> list[list[int]]:
    rng = random.Random(seed)
    out = []
    for j in range(k):
        L = rng.randrange(10, 400)
        if j % 3 == 0:
            s = [rng.randrange(0, 300) for _ in range(L)]
        elif j % 3 == 1:
            cyc = [rng.randrange(0, 300) for _ in range(rng.randrange(1, 10))]
            pre = [rng.randrange(0, 300) for _ in range(rng.randrange(0, 150))]
            s = pre + (cyc * (L // len(cyc) + 1))[:L]
        else:
            s = [rng.randrange(0, 4) for _ in range(L)]
        out.append(s)
    return out


# ------------------------------------------------------------------------------------------- signals
@pytest.mark.parametrize("W,n,rho", [(32, 4, 0.5), (64, 8, 0.3), (16, 2, 0.7), (128, 16, 0.5)])
def test_window_onset_equals_judge_repetition_onset(W, n, rho):
    for s in seqs(W + n):
        ids = np.asarray(s, np.int64)
        got = S.window_onset(S.gram_ids(ids, n), len(ids), W, n, rho)
        assert got == A.judge.repetition_onset(s, n=n, window=W, threshold=rho)


def test_seen_share_and_rep_onset_against_brute_force():
    for s in seqs(1, 12):
        ids = np.asarray(s, np.int64)
        for W, n in ((16, 2), (64, 8)):
            g = S.gram_ids(ids, n)
            sh = S.seen_share(g, len(ids), W, n)
            grams = [tuple(s[i:i + n]) for i in range(len(s) - n + 1)]
            for t in range(W, len(s) + 1):
                starts = range(t - W, t - n + 1)
                want = sum(1 for p in starts if grams[p] in grams[:p]) / (W - n + 1)
                assert sh[t] == pytest.approx(want)
        for n, k in ((3, 4), (8, 2)):
            g = S.gram_ids(ids, n)
            want = None
            for t in range(n, len(s) + 1):
                grams = [tuple(s[i:i + n]) for i in range(t - n + 1)]
                if grams and max(grams.count(x) for x in set(grams)) >= k:
                    want = t
                    break
            assert S.rep_onset(g, n, k) == want


# ------------------------------------------------------------------------------------------- costs
def trace(closed=True, ok=(False, True, True, True), answers=("1", "2", "2", "2"), cks=(128, 256, 384, 450),
          n_gen=520, correct=True, conf=None, tail=None) -> S.Trace:
    n_reason = cks[-1]
    tr = S.Trace("u/0", 0, list(cks), list(answers), list(ok), n_reason, n_gen, closed, correct and closed,
                 np.arange(n_reason), np.ones(n_reason, np.float32), conf)
    tr.tail = tail
    return tr


def test_outcome_charges_probes_in_token_equivalents():
    k, L = 25.0, 40.0                                                                 # L: the stop's close + answer
    tr = trace(tail=L)
    fire = np.array([False, True, True, True])
    assert S.outcome(tr, fire, probes=False, kappa=k) == (1.0, 256 + L, True)          # no probe; close + answer
    assert S.outcome(tr, fire, probes=True, kappa=k) == (1.0, 256 + 2 * k + L, True)   # probes at 128 and 256
    # firing only at the closed span's last checkpoint is no stop: the natural answer and n_gen stand
    last = np.array([False, False, False, True])
    assert S.outcome(tr, last, probes=False, kappa=k) == (1.0, 520.0, False)
    assert S.outcome(tr, last, probes=True, kappa=k) == (1.0, 520 + 3 * k, False)      # 3 probes before the close
    assert S.outcome(tr, None, probes=False, kappa=k) == (1.0, 520.0, False)
    # an open span (truncated at the cap) is stopped at the cap; nothing fires
    op = trace(closed=False, ok=(False, True, True, False), cks=(128, 256, 384, 512), n_gen=512, correct=False,
               tail=L)
    assert S.outcome(op, np.zeros(4, bool), probes=False, kappa=k) == (0.0, 512 + L, True)
    assert S.outcome(op, np.zeros(4, bool), probes=True, kappa=k) == (0.0, 512 + 4 * k + L, True)
    # oracle: first checkpoint from which the answer stays correct
    assert S.oracle_outcome(tr, k) == (1.0, 256 + L, True)
    with pytest.raises(ValueError):                                                   # a stop needs its tail
        S.outcome(trace(), fire, probes=False, kappa=k)


def test_stop_is_charged_its_close_and_answer():
    """A stop pays the canonical close and an answer as long as the row's natural one (n_gen − n_reason), or the
    median of the closed rows when the span never closed; the frontier's truncated rows pay the same. So a stop's
    charge equals the n_gen think.py records for a forced close: c + L_close + answer + stop."""
    k, L = 25.0, 24
    # the review's case: closes at 600 with n_gen 900 (marker + 298 answer tokens + stop); probes agree from 128
    a = trace(cks=(128, 256, 384, 512, 600), ok=(True,) * 5, answers=("7",) * 5, n_gen=900)
    b = trace(closed=False, cks=(128, 256), ok=(False, False), answers=("1", "2"), n_gen=256, correct=False)
    c = trace(cks=(128, 200), ok=(True, True), answers=("7", "7"), n_gen=300)
    med = S.set_tails([a, b, c], L)
    assert med == np.median([300, 100]) == 200.0
    assert a.tail == (L - 1) + 300 and c.tail == (L - 1) + 100 and b.tail == (L - 1) + 200
    fire = np.array([False, True, True, True, True])                                 # consist-2 fires at 256
    cost = S.outcome(a, fire, probes=True, kappa=k)[1]
    answer = 900 - 600 - 2                                                           # natural answer, no marker/stop
    assert cost == 256 + 2 * k + (L + answer + 1)        # think.py's forced row: n_gen = c + L + answer + stop
    assert 900 - cost < 900 - (256 + 2 * k)                                          # saves less than c + κ·m implied
    acc, tok = S.frontier_matrix([a, b], [128, 256, 512, 640])
    assert tok[0].tolist() == [128 + a.tail, 256 + a.tail, 512 + a.tail, 900.0]
    assert tok[1].tolist() == [128 + b.tail, 256 + b.tail, 256 + b.tail, 256 + b.tail]


def test_frontier_reproduces_budget_acc_and_interpolates():
    k = 10.0
    trs = [trace(tail=k), trace(closed=False, ok=(False, False, True, True), cks=(128, 256, 384, 512), n_gen=512,
                                correct=False, tail=k)]
    budgets = [128, 256, 384, 512]
    acc, tok = S.frontier_matrix(trs, budgets)
    for r, tr in enumerate(trs):
        ba = R.budget_acc(budgets, tr.cks, tr.ok, tr.n_reason, tr.closed, tr.correct)
        assert acc[r].tolist() == [ba[b] for b in budgets]
    assert tok[0].tolist() == [128 + k, 256 + k, 384 + k, 520.0]                      # closed by 512: n_gen
    assert tok[1].tolist() == [128 + k, 256 + k, 384 + k, 512 + k]
    fa, ft = np.array([0.2, 0.6, 0.8]), np.array([100.0, 200.0, 400.0])
    assert S.frontier_at(fa, ft, 150.0) == pytest.approx(0.4)
    assert S.frontier_at(fa, ft, 300.0) == pytest.approx(0.7)
    assert S.frontier_at(fa, ft, 50.0) is None and S.frontier_at(fa, ft, 500.0) is None
    assert S.frontier_tokens_at(fa, ft, 0.7) == pytest.approx(300.0)
    assert S.frontier_tokens_at(fa, ft, 0.9) is None
    # the inverse uses the running-maximum envelope (a dip does not count)
    assert S.frontier_tokens_at(np.array([0.2, 0.6, 0.5, 0.8]), np.array([1.0, 2.0, 3.0, 4.0]), 0.7) == \
        pytest.approx(3.5)


def test_split_is_by_problem_and_seeded():
    uids = [f"gsm8k/train/{i}" for i in range(400)]
    h = [S.split_half(u, 0) for u in uids]
    assert 150 < sum(h) < 250 and h == [S.split_half(u, 0) for u in uids]
    assert h != [S.split_half(u, 1) for u in uids]


# ------------------------------------------------------------------------------------------- study
def _args(**kw):
    a = S.build_parser().parse_args(["--tiny", "--boot", "300"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_planted_rule_is_recovered_and_beats_the_frontier():
    trs = S.synthetic(240, seed=0)
    res = S.study(trs, _args())
    best = res["chosen"]["overall"]
    assert "consist" in best
    h = res["heldout"]["best_overall"]
    assert h["d_acc"] > 0.05 and h["boot"]["d_acc"]["lo"] > 0
    assert res["heldout"]["oracle"]["tokens"] <= h["tokens"]
    assert res["n_settings_scanned"] == len(res["selection"]) and res["n_settings_scanned"] > 60
    assert res["n_selection"] + res["n_heldout"] == 240
    assert res["frontier_check_mismatches"] == 0
    # the loop rules fire on the looping half only
    sel = res["selection"]
    assert 0.3 < sel[res["chosen"]["ngram"]]["fire_rate"] <= 0.6


def test_probe_cost_is_charged_and_can_sink_a_probe_rule():
    trs = S.synthetic(240, seed=0)
    S.set_tails(trs, 25)
    rule = next(r for r in S.build_rules(False, True, S.SignalCache(), {id(t): i for i, t in enumerate(trs)})
                if r.name == "consist-2")
    c0, t0, _ = S.evaluate(trs, rule, 0.0)
    c1, t1, _ = S.evaluate(trs, rule, 100.0)
    assert np.array_equal(c0, c1)
    m = np.asarray([S.outcome(tr, rule.fire(tr), True, 1.0)[1] - S.outcome(tr, rule.fire(tr), True, 0.0)[1]
                    for tr in trs])
    assert np.allclose(t1 - t0, 100.0 * m) and (m >= 2).all()                          # ≥ 2 probes for k = 2
    expensive = S.study(trs, _args(probe_cost=4000.0))                            # probes dearer than tokens
    sel = expensive["selection"]["consist-2"]
    assert sel["d_acc"] is None or sel["d_acc"] < 0                                    # no longer above the frontier
    assert "consist" not in expensive["chosen"]["overall"]


def test_main_tiny_writes_json(tmp_path):
    rep = S.main(["--tiny", "--tiny-n", "80", "--boot", "200", "--out", str(tmp_path / "c0.json")])
    js = json.loads((tmp_path / "c0.json").read_text())
    assert js["n_settings_scanned"] == rep["n_settings_scanned"] and "heldout" in js and "selection" in js
    assert {"budget", "acc", "tokens"} <= set(js["frontier"]["heldout"][0])


def test_reads_reason_py_traces(tmp_path):
    """A tiny reason.py run's none arm: the loader takes its items and per-token arrays, and the simulated
    frontier reproduces the acc@b reason.py stored."""
    pytest.importorskip("transformers")
    R.main(["--tiny", "--tiny-arch", "qwen3", "--device", "cpu", "--data", "synthetic:6", "--batch", "3",
            "--max-new-tokens", "40", "--probe-every", "8", "--budgets", "8,16,24", "--calib-tokens", "0",
            "--arms", "none", "--out", str(tmp_path / "r.json"), "--trace-dir", str(tmp_path / "traces")])
    trs, prov = S.load_traces(tmp_path / "traces" / "none")
    assert len(trs) == 6 and prov["n_probed"] == 6 and prov["manifest_sha256"]
    js = json.loads((tmp_path / "r.json").read_text())["arms"]["none"]["per_item"]
    for tr, nr in zip(trs, js["n_reason"]):
        assert len(tr.span_ids) == tr.n_reason == nr and len(tr.span_ent) == tr.n_reason
        ba = R.budget_acc([8, 16, 24], tr.cks, tr.ok, tr.n_reason, tr.closed, tr.correct)
        assert all(ba[b] == tr.acc_budget[b] for b in (8, 16, 24))
    res = S.study(trs, S.build_parser().parse_args(["--tiny", "--boot", "50", "--probe-every", "8",
                                                    "--max-new-tokens", "40"]))
    assert res["frontier_check_mismatches"] == 0
