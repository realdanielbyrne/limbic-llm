"""The evaluation tools of the evolution search (experiments/evolve/run.py, eval_by_query.py): the
scattered needle task's variants (jitter, update, held-out names), the oracle-onset plumbing and
the slot diagnostics. Uses the tiny byte-tokenizer model only (CPU, no downloads)."""

import json
import random
import re
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from evolve import eval_by_query  # noqa: E402
from evolve import run as evolve_run  # noqa: E402

pl = evolve_run.pl


def keep_pl(monkeypatch):
    """run.main leaves pretrained_loop's task, side, train and evaluate patched; undo after the test."""
    for name in ("NeedleTask", "SideChannels", "HookedLM", "train", "evaluate", "scores"):
        monkeypatch.setattr(pl, name, getattr(pl, name))


def tok():
    return pl.build_tiny("gpt2", seed=0)[1]


def text(ids) -> str:
    return bytes(i for i in ids if i < 256).decode("latin-1")


class _RefScatter(pl.NeedleTask):
    """``ScatterNeedleTask`` as it was before the names / jitter / update options (commit 296322a)."""

    def __init__(self, *a, min_gap: int = 8, tail: int = 256, block: int = 0, mix: float = 1.0,
                 curriculum: list[tuple[int, int]] | None = None, **kw) -> None:
        super().__init__(*a, **kw)
        self.min_gap, self.tail, self.block, self.mix = min_gap, tail, block, mix
        self.curriculum, self._active, self._calls = curriculum or [], False, 0
        if self.corpus and mix < 1.0 and self.pool is None:              # random filler for the mixed case
            self.pool = self.tok.filler_pool()

    def sample(self, rng):
        enc = self.tok.encode
        names = rng.sample(pl.NAMES, self.pairs)
        values = ["".join(rng.choice("0123456789") for _ in range(4)) for _ in names]
        facts = [enc(f" The code for {n} is") + enc(f" {v}") + enc(".") for n, v in zip(names, values)]
        heads = [len(enc(f" The code for {n} is")) for n in names]
        j = rng.randrange(self.pairs)
        query, answer = enc(f" The code for {names[j]} is"), enc(f" {values[j]}")
        n_fill = self.seq_len - sum(map(len, facts)) - len(query) - len(answer)
        spare = n_fill - self.tail - self.min_gap * (self.pairs - 1)
        if spare < 0:
            raise ValueError(f"--seq-len {self.seq_len} too short for {self.pairs} scattered pairs")
        for _ in range(1000):                                              # filler before each fact
            cuts = sorted(rng.randint(0, spare) for _ in range(self.pairs))
            pre = [cuts[0]] + [cuts[i] - cuts[i - 1] + self.min_gap for i in range(1, self.pairs)]
            if not self.block or not self._straddles(pre, facts):
                break
        natural = bool(self.corpus) and (self.mix >= 1.0 or rng.random() < self.mix)   # per sequence

        def fill(k):
            if natural:
                o = rng.randrange(len(self.corpus) - k)
                return self.corpus[o:o + k]
            return [self.pool[rng.randrange(len(self.pool))] for _ in range(k)]
        ids, value_at = [self.bos] * self.sink, []
        for f, h, k in zip(facts, heads, pre):
            ids += fill(k)
            value_at.append(len(ids) + h)
            ids += f
        ids += fill(self.seq_len + self.sink - len(ids) - len(query) - len(answer)) + query + answer
        gap = (len(ids) - len(answer)) - value_at[j]
        self.last_queried, self.last_value_at, self.last_natural = j, value_at, natural   # for eval_by_query.py
        self.last_heads, self.last_query_len = heads, len(query)          # for diag_novelty.py
        return ids, len(answer), gap

    def batch(self, B: int, rng):
        """With a curriculum, training batches use fewer pairs early ((pairs, steps) stages, then
        the full count). Only active inside training (see ``train`` below), so the evaluation
        batches, built before training, always use the full number of pairs."""
        if not (self.curriculum and self._active):
            return super().batch(B, rng)
        step, self._calls = self._calls, self._calls + 1
        pairs = self.pairs
        for p, n in self.curriculum:
            if step < n:
                pairs = p
                break
            step -= n
        full, self.pairs = self.pairs, pairs
        try:
            return super().batch(B, rng)
        finally:
            self.pairs = full

    def _straddles(self, pre: list[int], facts: list[list[int]]) -> bool:
        """True if any fact would span two window blocks (positions counted after the sink)."""
        p = 0
        for k, f in zip(pre, facts):
            p += k
            if p // self.block != (p + len(f) - 1) // self.block:
                return True
            p += len(f)
        return False


CORPUS = "the cat sat on the mat and looked at the dog, which was asleep by the door. " * 60


@pytest.mark.parametrize("pairs,corpus,mix,block,curriculum", [
    (3, False, 1.0, 64, None), (5, True, 1.0, 64, None), (3, True, 0.5, 64, None), (1, True, 1.0, 0, None),
    (3, False, 1.0, 64, [(1, 3), (2, 3)])])
def test_default_scatter_task_is_unchanged(pairs, corpus, mix, block, curriculum):
    """With default options the task draws the same random numbers as before: the same sequences,
    layout records and batches for 100 seeds (training data of earlier rounds stays reproducible)."""
    t = tok()
    c = t.encode(CORPUS) if corpus else None
    kw = dict(min_gap=8, block=block, mix=mix, curriculum=curriculum)
    new, ref = evolve_run.ScatterNeedleTask(t, 512, pairs, 1, c, **kw), _RefScatter(t, 512, pairs, 1, c, **kw)
    for seed in range(100):
        a, b = random.Random(seed), random.Random(seed)
        assert new.sample(a) == ref.sample(b), seed
        for k in ("last_queried", "last_value_at", "last_natural", "last_heads", "last_query_len"):
            assert getattr(new, k) == getattr(ref, k), (seed, k)
        new._active = ref._active = bool(curriculum)
        bn, br = new.batch(2, a), ref.batch(2, b)
        new._active = ref._active = False
        assert torch.equal(bn[0], br[0]) and torch.equal(bn[1], br[1]) and bn[2] == br[2], seed
        assert a.random() == b.random()                             # the generators stay in step


def _check_facts(ids, task, n_stmts, stems):
    """Each statement: value_at is the space before 4 digits and a period; spans cover the statement."""
    assert len(ids) == 513 and len(task.last_value_at) == n_stmts == len(task.last_spans)
    for v, (lo, hi), h in zip(task.last_value_at, task.last_spans, task.last_heads):
        assert ids[v] == ord(" ") and text(ids[v + 1:v + 5]).isdigit() and ids[v + 5] == ord(".") == ids[hi]
        assert hi == v + 5 and lo == v - h
        head = text(ids[lo:v])
        assert any(head == s.format(n=n) for s in stems for n in task.names), head
    q = text(ids[-(task.last_query_len + 5):-5])
    assert q == evolve_run.PLAIN.format(n=q.split()[-2])            # the query keeps the plain phrasing


def test_jitter_task_varies_the_phrasing_and_keeps_the_layout():
    t = tok()
    task = evolve_run.ScatterNeedleTask(t, 512, 3, 1, min_gap=8, block=64, jitter=True)
    rng, seen = random.Random(0), set()
    for _ in range(60):
        ids, n_ans, gap = task.sample(rng)
        _check_facts(ids, task, 3, evolve_run.JITTER_STEMS)
        v = task.last_value_at[task.last_queried]
        assert ids[v:v + 5] == ids[-n_ans:] and gap == len(ids) - n_ans - v   # the answer is the queried code
        for (lo, hi), v in zip(task.last_spans, task.last_value_at):
            assert (lo - 1) // 64 == (hi - 1) // 64                 # inside one window block (after the sink)
            seen |= {i for i, st in enumerate(evolve_run.JITTER_STEMS)
                     if any(text(ids[lo:v]) == st.format(n=n) for n in pl.NAMES)}
    assert seen == {0, 1, 2}                                        # every phrasing occurs


@pytest.mark.parametrize("form", ["same", "now", "new"])
def test_update_task_answers_the_new_code(form):
    """--update: a name is stated twice; the answer is the later (new) code, whose first digit
    differs from the old one, and the old code is kept for scoring the stale answer."""
    t = tok()
    task = evolve_run.UpdateNeedleTask(t, 512, 3, 1, min_gap=8, block=64, form=form)
    rng = random.Random(1)
    stems = [evolve_run.PLAIN, evolve_run.UPDATE_STEMS[form]]
    for _ in range(60):
        ids, n_ans, gap = task.sample(rng)
        _check_facts(ids, task, 4, stems)
        j, o = task.last_queried, task.last_old_rank
        assert o < j                                                # the update comes after the old statement
        vn, vo = task.last_value_at[j], task.last_value_at[o]
        name = text(ids[-(task.last_query_len + 5):-5]).split()[-2]
        assert name in text(ids[task.last_spans[j][0]:vn]) and name in text(ids[task.last_spans[o][0]:vo])
        assert text(ids[task.last_spans[j][0]:vn]) == evolve_run.UPDATE_STEMS[form].format(n=name)
        assert ids[-n_ans:] == ids[vn:vn + 5]                       # the answer is the NEW code
        assert task.last_old_answer == ids[vo:vo + 5] and len(task.last_old_answer) == n_ans
        assert ids[vn + 1] != ids[vo + 1]                           # digit 1 tells them apart
        assert gap == len(ids) - n_ans - vn
        others = [text(ids[lo:v]) for i, ((lo, _), v) in enumerate(zip(task.last_spans, task.last_value_at))
                  if i not in (j, o)]
        assert all(name not in s for s in others)


def test_heldout_names_are_disjoint_and_used():
    assert not set(eval_by_query.HELDOUT_NAMES) & set(pl.NAMES)
    assert len(eval_by_query.HELDOUT_NAMES) >= 5
    task = evolve_run.ScatterNeedleTask(tok(), 512, 5, 1, min_gap=8, block=64, names=eval_by_query.HELDOUT_NAMES)
    rng = random.Random(2)
    for _ in range(30):
        ids, _, _ = task.sample(rng)
        body = text(ids)
        assert not any(f" {n} " in body or f" {n}," in body for n in pl.NAMES)
        assert sum(f"for {n} is" in body for n in eval_by_query.HELDOUT_NAMES) == 5


def test_train_names_restrict_the_sampled_names():
    assert evolve_run.TRAIN_NAMES == pl.NAMES[:20] and len(set(pl.NAMES) - set(evolve_run.TRAIN_NAMES)) == 5
    task = evolve_run.ScatterNeedleTask(tok(), 512, 5, 1, min_gap=8, block=64, names=evolve_run.TRAIN_NAMES)
    rng, used = random.Random(3), set()
    for _ in range(200):
        body = text(task.sample(rng)[0])
        used |= {n for n in pl.NAMES if f"for {n} is" in body}
    assert used == set(evolve_run.TRAIN_NAMES)


def test_batch_records_value_at_and_oracle_onsets():
    """batch() keeps each row's value_at (also on the ids), and the oracle onsets mark first digits."""
    task = evolve_run.ScatterNeedleTask(tok(), 512, 3, 1, min_gap=8, block=64)
    ids, _, _ = task.batch(4, random.Random(4))
    assert ids.value_at is task.last_batch_value_at and len(ids.value_at) == 4
    on = evolve_run.oracle_onsets(ids.value_at, ids.shape[1], 1)
    assert on.shape == (4, 512) and on.sum() == 12
    for b, vat in enumerate(ids.value_at):
        for v in vat:
            assert on[b, v] and chr(int(ids[b, v + 1])).isdigit() and ids[b, v] == ord(" ")   # side t = v + 1 − 1
    assert evolve_run.oracle_gate(types.SimpleNamespace(w1s_gate="oracle"))
    assert not evolve_run.oracle_gate(types.SimpleNamespace(w1s_gate="hyst")) and not evolve_run.oracle_gate(None)
    seen = {}
    lm = types.SimpleNamespace(side=nn.Linear(1, 1), sink=1)
    old, evolve_run._scores = evolve_run._scores, lambda lm, i, m: seen.setdefault("on", lm.side.oracle_onsets)
    try:
        evolve_run.scores(lm, ids, None)
    finally:
        evolve_run._scores = old
    assert torch.equal(seen["on"], on)


FAKE = '''
"""A stand-in candidate with an oracle gate: records the onsets it was handed at each forward."""
import argparse
import torch
import torch.nn as nn

SEEN = []


def split_args(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--fake-gate", default="oracle")
    return ap.parse_known_args(argv)


class Loop(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain_logit = nn.Parameter(torch.zeros(()))
        self.tau = 1


TAPS_REQUIRED = True


class FakeSide(nn.Module):
    def __init__(self, d_model, args, w=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.loop, self.oracle_onsets, self.taps = Loop(), None, (1,)

    def forward(self, h):
        assert self.oracle_onsets is not None and self.oracle_onsets.shape == h.shape[:2]
        SEEN.append(self.oracle_onsets.clone())
        return h * self.scale, None
'''


def test_runner_and_eval_end_to_end(tmp_path, monkeypatch):
    """run.py trains a candidate with an oracle gate on train names and records them; eval_by_query.py
    rebuilds it on CPU and writes each task variant to its own file. A candidate with
    ``TAPS_REQUIRED`` (or ``.taps``) runs on ``taps.TapLM``, here a stand-in that only counts."""
    keep_pl(monkeypatch)
    (tmp_path / "fake_oracle.py").write_text(FAKE)
    monkeypatch.syspath_prepend(str(tmp_path))
    import fake_oracle
    built = []

    class TapLM(pl.HookedLM):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            built.append(self)
    monkeypatch.setitem(sys.modules, "evolve.taps", types.SimpleNamespace(TapLM=TapLM))
    base = pl.HookedLM
    out = tmp_path / "r.json"
    evolve_run.main(["--tiny", "--task", "needle", "--window", "64", "--seq-len", "512", "--pairs", "3",
                     "--steps", "2", "--batch", "2", "--eval-n", "4", "--device", "cpu", "--needle-scatter", "8",
                     "--needle-names", "train", "--candidate", "fake_oracle:FakeSide", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert rec["needle_names"] == "train" and rec["candidate"]["args"]["fake_gate"] == "oracle"
    assert pl.HookedLM is base and pl.scores is evolve_run._scores and len(built) == 1   # patches undone
    assert fake_oracle.SEEN and all(int(s.sum()) == 3 * len(s) for s in fake_oracle.SEEN)
    for variant, flags in (("", []), (".jitter", ["--jitter"]), (".update-now", ["--update", "now"]),
                           (".heldout", ["--heldout-names"])):
        fake_oracle.SEEN.clear()
        eval_by_query.main([str(out), "--n", "4", "--batch", "2"] + flags)
        res = json.loads(out.with_suffix(f".by_query{variant}.json").read_text())
        assert res["n"] == 4 and len(res["digit1_acc_by_queried_rank"]) == (4 if variant == ".update-now" else 3)
        assert fake_oracle.SEEN and all(int(s.sum()) == (4 if variant == ".update-now" else 3) * len(s)
                                        for s in fake_oracle.SEEN)
        if variant == ".update-now":
            assert set(res["update"]["windowed+side"]) == {"new_digit1", "old_digit1", "new_exact", "old_exact"}
        if variant:
            assert res["task"]["names"] == ("heldout" if variant == ".heldout" else "train")
    assert len(built) == 5


def test_eval_scores_w1j_events_with_novelty(tmp_path, monkeypatch):
    """B17: the ε that eval_by_query.py reports for W1j includes the CA1 novelty term, as in W1j's
    own forward pass (it used to call event_strength without it)."""
    from evolve.w1j_ca1 import CA1Loop
    keep_pl(monkeypatch)
    out = tmp_path / "j.json"
    evolve_run.main(["--tiny", "--task", "needle", "--window", "64", "--seq-len", "512", "--pairs", "3",
                     "--steps", "1", "--batch", "2", "--eval-n", "2", "--device", "cpu", "--needle-scatter", "8",
                     "--candidate", "evolve.w1j_ca1:W1jSide", "--w1i-heads", "2", "--w1i-dhead", "16",
                     "--out", str(out)])
    calls, orig = [], CA1Loop.event_strength

    def spy(self, x, extra_logit=None):
        calls.append((x, extra_logit))
        return orig(self, x, extra_logit)
    monkeypatch.setattr(CA1Loop, "event_strength", spy)
    eval_by_query.main([str(out), "--n", "2", "--batch", "2", "--no-controls"])
    res = json.loads(out.with_suffix(".by_query.json").read_text())
    (x_run, e_run), (x_rep, e_rep) = calls                          # the side's forward, then the report
    assert e_run is not None and torch.equal(e_run, e_rep)
    assert torch.allclose(x_run, x_rep, atol=1e-5)                 # the same input: the sink dropped
    assert "eps_mean_at_fact_digit1" in res and "w1s" not in res


def test_slot_diagnostics_on_a_perfect_store():
    """_slot_diag on hand-made last_diag tensors: a store that retains and reads the queried fact,
    aligned on its first digit, with a gate high only at first digits, scores 1 everywhere."""
    sink, T, n_ans, K = 1, 61, 5, 4
    vals = [[11, 31], [21, 41]]                                     # spaces before the digits (with the sink)
    spans = [[(v - 6, v + 5) for v in r] for r in vals]
    js, qlens = [1, 0], [6, 6]
    B, Ts = 2, T - sink
    p = torch.zeros(B, Ts)
    onsets = torch.zeros(B, Ts, dtype=torch.bool)
    retained = torch.full((B, Ts, K), -1, dtype=torch.long)
    read_w = torch.zeros(B, Ts, K + 1)
    align = torch.full((B, Ts), -1, dtype=torch.long)
    tq = T - n_ans - sink
    for b in range(B):
        d1 = [v + 1 - sink for v in vals[b]]
        p[b, d1] = 1.0
        onsets[b, d1] = True
        retained[b, tq, :2] = torch.tensor(d1)
        read_w[b, tq, js[b]] = 1.0
        align[b, tq] = d1[js[b]]
    acc = {"seq": [], "align_within1": [], "null_weight": [], "p_d1": [], "p_fill": [], "onset_in_fact": [],
           "density": []}
    diag = {"p": p, "onsets": onsets, "retained": retained, "read_w": read_w, "align": align}
    eval_by_query._slot_diag(diag, js, vals, spans, qlens, T, n_ans, sink, acc)
    s = eval_by_query._slot_summary(acc, 2)
    assert s["fact_coverage"] == s["slot_hit"] == s["align_hit"] == s["gate_auc_digit1_vs_filler"] == 1.0
    assert s["onset_in_fact"] == 1.0 and s["null_weight_at_query"] == 0.0
    read_w[:, tq] = 0
    read_w[:, tq, -1] = 1.0                                         # everything to the null slot: no hit
    acc = {k: [] for k in acc}
    eval_by_query._slot_diag(diag, js, vals, spans, qlens, T, n_ans, sink, acc)
    s = eval_by_query._slot_summary(acc, 2)
    assert s["slot_hit"] == 0.0 and s["fact_coverage"] == 1.0 and s["null_weight_at_query"] == 1.0
    read_w.zero_()                                                  # superposed store: no slot read at all
    acc = {k: [] for k in acc}
    eval_by_query._slot_diag(diag, js, vals, spans, qlens, T, n_ans, sink, acc)
    s = eval_by_query._slot_summary(acc, 2)
    assert s["slot_hit"] is None and s["null_weight_at_query"] is None and s["slot_hit_by_queried_rank"] == [None] * 2
    assert eval_by_query._auc(torch.tensor([0.5]), torch.tensor([0.5, 0.5])) == 0.5


@pytest.mark.parametrize("flags", [["--w1s-taps", "1"],
                                   ["--w1s-taps", "1", "--w1s-gate", "oracle"],
                                   ["--w1s-taps", "0", "--w1s-out", "reinstate", "--w1s-L", "1",
                                    "--attn-impl", "eager"]])
def test_w1s_runs_end_to_end_with_taps(tmp_path, monkeypatch, flags):
    """The real W1s on the real ``taps.TapLM``: run.py trains it on the tiny model (taps below the main
    tap 3), and eval_by_query.py rebuilds it, reports the slot diagnostics, and runs the controls:
    with the store forced empty the read is the null everywhere and no fact is covered."""
    keep_pl(monkeypatch)
    out = tmp_path / "w1s.json"
    evolve_run.main(["--tiny", "--task", "needle", "--window", "64", "--seq-len", "512", "--pairs", "3",
                     "--steps", "2", "--batch", "2", "--eval-n", "4", "--device", "cpu", "--needle-scatter", "8",
                     "--layer", "3", "--candidate", "evolve.w1s_slots:W1sSide", "--out", str(out)] + flags)
    rec = json.loads(out.read_text())
    assert rec["results"]["windowed"]["exact"] == 0.0
    reinstate = "reinstate" in flags
    with pytest.raises(SystemExit, match="batch"):                 # one row: the shuffle is the identity
        eval_by_query.main([str(out), "--n", "4", "--batch", "1", "--w1s-ctrl", "shuffle"])
    if reinstate:                                                   # reinstatement has no shuffle control
        with pytest.raises(SystemExit, match="reinstate"):
            eval_by_query.main([str(out), "--n", "4", "--batch", "2", "--w1s-ctrl", "shuffle"])
    for variant, extra in (("", []), (".ctrl-empty", ["--w1s-ctrl", "empty"]),
                           (".ctrl-shuffle", ["--w1s-ctrl", "shuffle"])):
        if reinstate and variant == ".ctrl-shuffle":
            continue
        eval_by_query.main([str(out), "--n", "4", "--batch", "2"] + extra)
        res = json.loads(out.with_suffix(f".by_query{variant}.json").read_text())
        assert not res["missing_keys"] and not res["unexpected_keys"]
        w = res["w1s"]
        assert 0 <= w["fact_coverage"] <= 1 and 0 <= w["null_weight_at_query"] <= 1
        if variant == ".ctrl-empty":
            assert res["exact"] == 0.0 and w["fact_coverage"] == 0.0 and w["null_weight_at_query"] == 1.0
        if "oracle" in flags and not variant:
            assert w["fact_coverage"] == 1.0 and w["gate_auc_digit1_vs_filler"] == 1.0


# ------------------------------------------------------------------ round 11: tasks, names, overrides
def _digest(task, n=40, seed=0) -> str:
    """A hash of n samples (sequences and layout records), one batch and the generator's next draw."""
    import hashlib
    h, rng = hashlib.sha256(), random.Random(seed)
    for _ in range(n):
        ids, n_ans, gap = task.sample(rng)
        h.update(json.dumps([ids, n_ans, gap, task.last_queried, task.last_value_at, task.last_spans, task.last_heads,
                             task.last_query_len, getattr(task, "last_old_rank", None),
                             getattr(task, "last_old_answer", None)]).encode())
    ids, mask, gaps = task.batch(3, rng)
    h.update(json.dumps([ids.tolist(), mask.tolist(), gaps, ids.value_at]).encode())
    h.update(repr(rng.random()).encode())
    return h.hexdigest()[:16]


# computed from the code before round 11 (commit 4d1c271): the defaults must keep these sequences
GOLDEN = {"scatter_p3": "9ee88c373812a0e0", "scatter_p5_corpus": "95c1210e1af4b1e5",
          "scatter_p3_train": "bcb13329b4d27b5c", "scatter_p3_jitter": "e4ac050d30b70e97",
          "update_same": "9f49743450bd910a", "update_same_p5_corpus": "6bef2f0751116820",
          "update_now": "78210e376f7f4801", "update_now_p5_corpus": "913f0a8417e52240",
          "update_new": "979a685bfa0f4ab5", "update_new_p5_corpus": "c70a1136f10f8f49"}


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_default_sequences_match_the_pre_round11_golden(case):
    t = tok()
    c = t.encode(CORPUS)
    S, U = evolve_run.ScatterNeedleTask, evolve_run.UpdateNeedleTask
    build = {"scatter_p3": lambda: S(t, 512, 3, 1, min_gap=8, block=64),
             "scatter_p5_corpus": lambda: S(t, 512, 5, 1, c, min_gap=8, block=64),
             "scatter_p3_train": lambda: S(t, 1024, 3, 1, min_gap=8, block=64, names=evolve_run.TRAIN_NAMES),
             "scatter_p3_jitter": lambda: S(t, 512, 3, 1, c, min_gap=8, block=64, jitter=True)}
    for f in ("same", "now", "new"):
        build[f"update_{f}"] = lambda f=f: U(t, 512, 3, 1, min_gap=8, block=64, form=f)
        build[f"update_{f}_p5_corpus"] = lambda f=f: U(t, 1024, 5, 1, c, min_gap=8, block=64, form=f)
    assert _digest(build[case]()) == GOLDEN[case]


def _stmt(ids, task, r) -> str:
    lo, hi = task.last_spans[r]
    return text(ids[lo:hi + 1])


@pytest.mark.parametrize("form", ["changed", "nowold"])
def test_update_forms_naming_the_old_code_answer_the_new_one(form):
    """changed / nowold state OLD and NEW in one statement; value_at points at NEW's space, the answer
    is NEW, the old statement comes earlier and its code is the stale answer."""
    task = evolve_run.UpdateNeedleTask(tok(), 512, 3, 1, min_gap=8, block=64, form=form)
    rng = random.Random(5)
    for _ in range(40):
        ids, n_ans, gap = task.sample(rng)
        j, o = task.last_queried, task.last_old_rank
        vn, vo = task.last_value_at[j], task.last_value_at[o]
        new, old = text(ids[vn + 1:vn + 5]), text(ids[vo + 1:vo + 5])
        name = text(ids[-(task.last_query_len + 5):-5]).split()[-2]
        assert o < j and ids[-n_ans:] == ids[vn:vn + 5] and new.isdigit() and new[0] != old[0]
        assert task.last_old_answer == ids[vo:vo + 5] and gap == len(ids) - n_ans - vn
        want = (f" {name}'s code has changed from {old} to {new}." if form == "changed"
                else f" The code for {name} is now {new} (it used to be {old}).")
        assert _stmt(ids, task, j) == want
        assert _stmt(ids, task, o) == f" The code for {name} is {old}."
        assert task.last_spans[o][1] < task.last_spans[j][0]        # NEW's statement after OLD's


def test_update_k_answers_the_last_restatement():
    task = evolve_run.UpdateNeedleTask(tok(), 960, 3, 1, min_gap=8, block=64, form="now", k=3)
    rng = random.Random(6)
    for _ in range(40):
        ids, n_ans, _ = task.sample(rng)
        ranks = task.last_update_ranks
        assert len(ranks) == 4 and ranks == sorted(ranks) and len(task.last_value_at) == 3 + 3
        codes = [text(ids[task.last_value_at[r] + 1:task.last_value_at[r] + 5]) for r in ranks]
        assert all(a[0] != b[0] for a, b in zip(codes, codes[1:]))  # digit 1 differs from the previous code
        assert task.last_queried == ranks[-1] and text(ids[-4:]) == codes[-1]
        assert task.last_old_rank == ranks[-2] and text(task.last_old_answer[1:]) == codes[-2]
        name = text(ids[-(task.last_query_len + 5):-5]).split()[-2]
        assert all(name in _stmt(ids, task, r) for r in ranks)
        assert sum(name in _stmt(ids, task, r) for r in range(6)) == 4


@pytest.mark.parametrize("update", [None, "same"])
def test_decoys_are_never_queried(update):
    """decoys: extra statements for other names, among the facts; the query is one of the pairs and
    the queried rank covers the positions of all statements."""
    t = tok()
    kw = dict(min_gap=8, block=64, decoys=4)
    task = (evolve_run.UpdateNeedleTask(t, 960, 3, 1, form=update, **kw) if update
            else evolve_run.ScatterNeedleTask(t, 960, 3, 1, **kw))
    rng, ranks = random.Random(7), set()
    for _ in range(80):
        ids, _, _ = task.sample(rng)
        n_st = 3 + 4 + (1 if update else 0)
        assert len(task.last_value_at) == n_st and len(task.last_decoy_ranks) == 4
        assert task.last_queried not in task.last_decoy_ranks
        name = text(ids[-(task.last_query_len + 5):-5]).split()[-2]
        assert not any(name in _stmt(ids, task, r) for r in task.last_decoy_ranks)
        stmts = [_stmt(ids, task, r).split()[3] for r in range(n_st)]   # " The code for NAME is ..."
        assert len(set(stmts)) == 3 + 4                             # distinct names (the update repeats one)
        ranks.add(task.last_queried)
    assert len(ranks) >= n_st - 1                                   # the queried rank is not tied to a position


@pytest.mark.parametrize("kind", ["before", "next", "first", "last"])
def test_order_tasks_answer_the_right_fact(kind):
    task = evolve_run.OrderNeedleTask(tok(), 512, 4, 1, min_gap=8, block=64, kind=kind)
    rng = random.Random(8)
    for _ in range(40):
        ids, n_ans, gap = task.sample(rng)
        r, a = task.last_queried, task.last_anchor_rank
        codes = [text(ids[v + 1:v + 5]) for v in task.last_value_at]
        names = [_stmt(ids, task, i).split()[3] for i in range(4)]
        assert text(ids[-4:]) == codes[r] and gap == len(ids) - n_ans - task.last_value_at[r]
        assert r == {"before": a - 1, "next": a + 1, "first": 0, "last": 3}[kind]
        q = text(ids[-(task.last_query_len + 5):-5])
        assert q == evolve_run.ORDER_QUERIES[kind].format(n=names[a])
    with pytest.raises(ValueError):
        evolve_run.OrderNeedleTask(tok(), 512, 1, 1, kind="before")


def test_tail_must_exceed_the_window(tmp_path, monkeypatch):
    t = tok()
    with pytest.raises(ValueError, match="exceed"):
        evolve_run.ScatterNeedleTask(t, 512, 3, 1, min_gap=8, tail=64, block=64)
    task = evolve_run.ScatterNeedleTask(t, 512, 3, 1, min_gap=8, tail=200, block=64)
    rng = random.Random(9)
    for _ in range(30):
        ids, n_ans, _ = task.sample(rng)
        assert len(ids) - n_ans - task.last_query_len - (task.last_spans[-1][1] + 1) >= 200
    keep_pl(monkeypatch)
    with pytest.raises(SystemExit, match="exceed"):
        evolve_run.main(["--tiny", "--window", "64", "--needle-scatter", "8", "--needle-tail", "64",
                         "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit, match="needle-scatter"):
        evolve_run.main(["--tiny", "--needle-tail", "300", "--out", str(tmp_path / "x.json")])


def test_syllable_pool_is_disjoint_deterministic_and_prefix_stable():
    from evolve.names import HELDOUT_NAMES, syllable_pool
    assert eval_by_query.HELDOUT_NAMES is HELDOUT_NAMES             # re-exported
    a, b = syllable_pool(2000, 0), syllable_pool(2000, 0)
    assert a == b and len(set(a)) == 2000 and syllable_pool(50, 0) == a[:50]
    assert syllable_pool(50, 1) != a[:50]
    assert not set(a) & set(HELDOUT_NAMES) and not set(a) & set(pl.NAMES)
    assert all(n[0].isupper() and n[1:].islower() and len(n) in (4, 6) for n in a)   # 2–3 CV syllables


def test_name_rule_beyond_the_list():
    """More statements than names: the queried name from the configured list, the rest distinct and
    drawn from list ∪ syllable pool; within the list the rule is off."""
    from evolve.names import syllable_pool
    t, pool = tok(), set(evolve_run.TRAIN_NAMES) | set(syllable_pool(2000, 0))
    small = evolve_run.ScatterNeedleTask(t, 512, 3, 1, min_gap=8, block=64, names=evolve_run.TRAIN_NAMES)
    assert small.names_rule is None
    for cls, kw in ((evolve_run.ScatterNeedleTask, {}), (evolve_run.UpdateNeedleTask, {"form": "now"}),
                    (evolve_run.OrderNeedleTask, {"kind": "before"})):
        task = cls(t, 4096, 24, 1, min_gap=8, block=64, names=evolve_run.TRAIN_NAMES, **kw)
        assert "syllable_pool(2000, seed 0)" in task.names_rule
        rng, seen = random.Random(10), set()
        for _ in range(20):
            ids, _, _ = task.sample(rng)
            q = text(ids[-(task.last_query_len + 5):-5])
            qn = q.split()[3] if cls is not evolve_run.OrderNeedleTask else q.split()[-2][:-2]
            names = [_stmt(ids, task, r).split()[3] for r in range(len(task.last_value_at))]
            assert qn in evolve_run.TRAIN_NAMES and set(names) <= pool
            assert len(set(names)) == 24 and qn in names
            seen |= set(names) - set(evolve_run.TRAIN_NAMES)
        assert seen                                                 # pool names do occur


FAKE_SET = '''
"""A stand-in candidate for the override plumbing: flag types, a flag that adds a parameter, a state report."""
import argparse
import torch
import torch.nn as nn

BUILT = []


def split_args(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--fk-int", type=int, default=3)
    ap.add_argument("--fk-float", type=float, default=0.5)
    ap.add_argument("--fk-mode", choices=["a", "b"], default="a")
    ap.add_argument("--fk-flag", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--fk-switch", action="store_true")
    ap.add_argument("--fk-pair", type=float, nargs=2, default=[1.0, 2.0])
    ap.add_argument("--fk-opt", type=float, default=None)
    ap.add_argument("--fk-keep-by", choices=["first", "external"], default="first")
    return ap.parse_known_args(argv)


class Loop(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain_logit = nn.Parameter(torch.zeros(()))
        self.log_beta_align = nn.Parameter(torch.tensor(1.0))
        self.tau = 1


class FakeSide(nn.Module):
    def __init__(self, d_model, args, w):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.loop = Loop()
        if w.fk_flag:
            self.extra_p = nn.Parameter(torch.randn(()))
        BUILT.append(dict(vars(w)))

    def forward(self, h):
        return h * self.scale, None

    def state_report(self):
        return {"state_floats": 7, "state_components": {"x": 7}, "params": sum(p.numel() for p in self.parameters())}
'''


@pytest.fixture
def fake_set(tmp_path, monkeypatch):
    """A trained stand-in run (fake_set:FakeSide) and its module."""
    keep_pl(monkeypatch)
    (tmp_path / "fake_set.py").write_text(FAKE_SET)
    monkeypatch.syspath_prepend(str(tmp_path))
    import fake_set as mod
    out = tmp_path / "f.json"
    evolve_run.main(["--tiny", "--task", "needle", "--window", "64", "--seq-len", "512", "--pairs", "3",
                     "--steps", "2", "--batch", "2", "--eval-n", "2", "--device", "cpu", "--needle-scatter", "8",
                     "--candidate", "fake_set:FakeSide", "--fk-int", "5", "--out", str(out)])
    mod.BUILT.clear()
    return out, mod


def test_w1s_set_coerces_types_and_rejects_unknown_keys(fake_set):
    _, mod = fake_set
    got = {k: v for k, _, v in eval_by_query.parse_sets(
        mod, ["fk_int=7", "fk-float=0.25", "fk_mode=b", "fk_flag=true", "fk_switch=true", "fk_pair=3,4", "fk_opt=1.5"])}
    assert got == {"fk_int": 7, "fk_float": 0.25, "fk_mode": "b", "fk_flag": True, "fk_switch": True,
                   "fk_pair": [3.0, 4.0], "fk_opt": 1.5}
    assert type(got["fk_int"]) is int and type(got["fk_float"]) is float
    got = {k: v for k, _, v in eval_by_query.parse_sets(mod, ["fk_flag=false", "fk_switch=false"])}
    assert got == {"fk_flag": False, "fk_switch": False}
    for bad in ("fk_nope=1", "fk_mode=c", "fk_int=x", "fk_flag=maybe", "fk_int"):
        with pytest.raises(SystemExit):
            eval_by_query.parse_sets(mod, [bad])


def test_run_records_tail_state_and_refuses_external_keep(fake_set, tmp_path):
    out, _ = fake_set
    rec = json.loads(out.read_text())
    assert rec["needle_tail"] == 256 and "needle_name_pool" not in rec and "needle_names" not in rec
    assert rec["w1s_state"] == {"state_floats": 7, "state_components": {"x": 7}, "params": 3}
    with pytest.raises(SystemExit, match="keep-by external"):
        evolve_run.main(["--tiny", "--needle-scatter", "8", "--candidate", "fake_set:FakeSide",
                         "--fk-keep-by", "external", "--out", str(tmp_path / "x.json")])


def test_run_trains_on_the_name_pool(fake_set, tmp_path):
    from evolve.names import syllable_pool
    out = tmp_path / "p.json"
    evolve_run.main(["--tiny", "--task", "needle", "--window", "64", "--seq-len", "512", "--pairs", "3",
                     "--steps", "1", "--batch", "2", "--eval-n", "2", "--device", "cpu", "--needle-scatter", "8",
                     "--needle-names", "pool", "--needle-name-pool", "50", "--needle-tail", "300",
                     "--candidate", "fake_set:FakeSide", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert rec["needle_names"] == "pool" and rec["needle_name_pool"] == 50 and rec["needle_tail"] == 300
    task = pl.NeedleTask(tok(), 512, 3, 1)                          # the task run.py installed
    assert task.names == syllable_pool(50, 0) and task.tail == 300
    res_path = out.with_suffix(".by_query.json")
    eval_by_query.main([str(out), "--n", "2", "--batch", "2", "--no-controls"])
    res = json.loads(res_path.read_text())
    assert res["task"] == {"jitter": False, "update": None, "names": "train", "trained_names": "pool"}
    assert res["needle_tail"] == 300


def test_w1s_set_merge_order_missing_keys_and_load_extra(fake_set, tmp_path):
    out, mod = fake_set
    eval_by_query.main([str(out), "--n", "2", "--batch", "2", "--no-controls", "--w1s-set", "fk_float=0.25",
                        "--w1s-set", "fk_flag=true"])
    w = mod.BUILT[0]                                                # defaults < run flags < --w1s-set
    assert (w["fk_int"], w["fk_float"], w["fk_flag"], w["fk_mode"]) == (5, 0.25, True, "a")
    res = json.loads(out.with_suffix(".by_query.set-fk_float-0.25.set-fk_flag-true.json").read_text())
    assert res["w1s_set"] == {"fk_float": 0.25, "fk_flag": True} and res["missing_keys"] == ["extra_p"]
    assert res["w1s_state"]["state_floats"] == 7
    a = json.loads(out.with_suffix(".by_query.set-fk_float-0.25.set-fk_flag-true.json").read_text())
    eval_by_query.main([str(out), "--n", "2", "--batch", "2", "--no-controls", "--w1s-set", "fk_float=0.25",
                        "--w1s-set", "fk_flag=true"])                    # the added parameter is seeded
    assert json.loads(out.with_suffix(".by_query.set-fk_float-0.25.set-fk_flag-true.json").read_text()) == a
    extra = tmp_path / "wz.pt"
    torch.save({"extra_p": torch.tensor(2.0)}, extra)
    eval_by_query.main([str(out), "--n", "2", "--batch", "2", "--no-controls", "--w1s-set", "fk_flag=true",
                        "--w1s-load-extra", str(extra)])
    res = json.loads(out.with_suffix(".by_query.set-fk_flag-true.extra-wz.json").read_text())
    assert res["w1s_load_extra"]["keys"] == ["extra_p"]
    torch.save({"nope": torch.tensor(2.0)}, extra)
    with pytest.raises(SystemExit, match="not in the side"):
        eval_by_query.main([str(out), "--n", "2", "--batch", "2", "--w1s-set", "fk_flag=true",
                            "--w1s-load-extra", str(extra)])
    st = torch.load(out.with_suffix(".pt"))                         # a checkpoint missing a trained key
    del st["scale"]
    bad = tmp_path / "bad.json"
    bad.write_text(out.read_text())
    torch.save(st, bad.with_suffix(".pt"))
    with pytest.raises(SystemExit, match="missing keys"):
        eval_by_query.main([str(bad), "--n", "2", "--batch", "2", "--w1s-set", "fk_flag=true"])
    with pytest.raises(SystemExit, match="not a flag"):
        eval_by_query.main([str(out), "--n", "2", "--batch", "2", "--w1s-set", "fk_nope=1"])


def test_w1s_param_shifts_and_sets_after_loading(fake_set):
    out, _ = fake_set
    ck = torch.load(out.with_suffix(".pt"))["loop.log_beta_align"].item()
    eval_by_query.main([str(out), "--n", "2", "--batch", "2", "--no-controls",
                        "--w1s-param", "loop.log_beta_align+=0.3", "--w1s-param", "scale=2"])
    res = json.loads(out.with_suffix(".by_query.param-loop.log_beta_align-add0.3.param-scale-set2.0.json").read_text())
    p = {d["name"]: d for d in res["w1s_param"]}
    assert p["loop.log_beta_align"]["result"] == pytest.approx(ck + 0.3) and p["scale"]["result"] == 2.0
    with pytest.raises(SystemExit, match="state_dict"):
        eval_by_query.main([str(out), "--n", "2", "--batch", "2", "--w1s-param", "loop.nope+=1"])


def test_eval_task_variants_and_refusals(fake_set):
    out, _ = fake_set
    for flags, suffix, R in ((["--update", "changed"], ".update-changed", 4),
                             (["--update", "same", "--update-k", "2"], ".update-same-k2", 5),
                             (["--decoys", "2"], ".decoys2", 5), (["--order", "before"], ".order-before", 3),
                             (["--order", "first"], ".order-first", 3)):
        eval_by_query.main([str(out), "--n", "4", "--batch", "2"] + flags)
        res = json.loads(out.with_suffix(f".by_query{suffix}.json").read_text())
        assert res["n"] == 4 and len(res["count_by_queried_rank"]) == R
        assert res["controls"]["windowed"]["exact"] == 0.0
        if "--update" in flags:
            assert set(res["update"]) == {"windowed+side", "windowed", "full"}
    with pytest.raises(SystemExit, match="--batch"):                # long cells need an explicit batch
        eval_by_query.main([str(out), "--n", "2", "--seq-len", "4096"])
    for flags, msg in ((["--ctrl-conds"], "ctrl attribute"),
                       (["--ctrl-conds", "--w1s-ctrl", "empty"], "drop --w1s-ctrl"), (["--update-k", "2"], "--update"),
                       (["--order", "next", "--update", "now"], "separate"), (["--needle-tail", "64"], "exceed"),
                       (["--keep-oracle", "random"], "external")):
        with pytest.raises(SystemExit, match=msg):
            eval_by_query.main([str(out), "--n", "2", "--batch", "2"] + flags)


def test_received_attention_is_causal_windowed_and_needs_eager():
    model, _ = pl.build_tiny("gpt2", seed=0, attn_impl="eager")
    lm = pl.HookedLM(model.eval(), None, 3, 64, "block", "all", 1, pad_id=256)   # eval: no attention dropout
    ids = torch.randint(0, 256, (2, 1 + 192), generator=torch.Generator().manual_seed(0))
    R = eval_by_query.received_attention(lm, ids, 64, [0, 1, 2, 3], 1)
    T = 192
    assert R.shape == (2, T, T) and (R >= 0).all() and (R.sum(-1) <= 1 + 1e-5).all()
    q, j = torch.meshgrid(torch.arange(T), torch.arange(T), indexing="ij")
    assert (R[:, j > q] == 0).all() and (R[:, q // 64 != j // 64] == 0).all()   # causal, inside its block
    assert (R.sum(-1) > 0.0).all()
    model, _ = pl.build_tiny("gpt2", seed=0, attn_impl="sdpa")
    lm = pl.HookedLM(model, None, 3, 64, "block", "all", 1, pad_id=256)
    with pytest.raises(RuntimeError, match="eager"):
        eval_by_query.received_attention(lm, ids, 64, [0], 1)


@pytest.fixture
def w1s_run(tmp_path, monkeypatch):
    keep_pl(monkeypatch)
    out = tmp_path / "w1s.json"
    evolve_run.main(["--tiny", "--task", "needle", "--window", "64", "--seq-len", "512", "--pairs", "3",
                     "--steps", "2", "--batch", "2", "--eval-n", "4", "--device", "cpu", "--needle-scatter", "8",
                     "--layer", "3", "--candidate", "evolve.w1s_slots:W1sSide", "--w1s-taps", "1", "--out", str(out)])
    return out


def test_ctrl_conds_keep_the_windowed_side_diagnostics(w1s_run):
    """--ctrl-conds scores side_empty and side_shuffle in the same pass: each equals its --w1s-ctrl run,
    side_empty scores 0, and the slot diagnostics and main scores are those of the run without them
    (they used to be overwritten by the last side-on condition)."""
    out = w1s_run
    base = ["--n", "8", "--batch", "4"]
    eval_by_query.main([str(out)] + base)
    eval_by_query.main([str(out)] + base + ["--ctrl-conds", "--out", str(out.with_suffix(".cc.json"))])
    plain = json.loads(out.with_suffix(".by_query.json").read_text())
    cc = json.loads(out.with_suffix(".cc.json").read_text())
    assert cc["ctrl_conds"] == ["side_empty", "side_shuffle"]
    assert cc["w1s"] == plain["w1s"] and cc["per_sequence"] == plain["per_sequence"]
    assert {k: v for k, v in cc["controls"].items() if k in plain["controls"]} == plain["controls"]
    assert cc["controls"]["side_empty"]["exact"] == 0.0 and cc["controls"]["side_empty"]["digit1"] == 0.0
    for ctrl in ("empty", "shuffle"):
        eval_by_query.main([str(out)] + base + ["--w1s-ctrl", ctrl])
        one = json.loads(out.with_suffix(f".by_query.ctrl-{ctrl}.json").read_text())
        mine = cc["controls"][f"side_{ctrl}"]
        assert mine["digit1_by_queried_rank"] == one["digit1_acc_by_queried_rank"]
        assert mine["exact"] == one["exact"]
    eval_by_query.main([str(out)] + base + ["--update", "now", "--ctrl-conds"])
    res = json.loads(out.with_suffix(".by_query.update-now.json").read_text())
    assert set(res["update"]) == {"windowed+side", "windowed", "full", "side_empty", "side_shuffle"}
    assert res["update"]["side_empty"]["old_exact"] == 0.0 and res["update"]["side_empty"]["new_exact"] == 0.0
    with pytest.raises(SystemExit, match="batch"):
        eval_by_query.main([str(out), "--n", "2", "--batch", "1", "--ctrl-conds"])


@pytest.mark.parametrize("mode", ["random", "offsets", "attn"])
def test_keep_oracles_feed_the_side(w1s_run, mode):
    out = w1s_run
    seen = []
    from evolve import w1s_slots
    orig = w1s_slots.W1sSide.forward

    def spy(self, h):
        seen.append((self.keep_scores, self.keep_pair, h.shape))
        return orig(self, h)
    w1s_slots.W1sSide.forward = spy
    try:
        extra = ["--attn-impl", "eager", "--keep-attn-layers", "1-2"] if mode == "attn" else []
        eval_by_query.main([str(out), "--n", "4", "--batch", "2", "--no-controls", "--w1s-set", "w1s_keep=4",
                            "--w1s-set", "w1s_keep_by=external", "--keep-oracle", mode] + extra)
    finally:
        w1s_slots.W1sSide.forward = orig
    sfx = ".attn-eager" if mode == "attn" else ""
    res = json.loads(out.with_suffix(f".by_query{sfx}.keep-{mode}.set-w1s_keep-4.set-w1s_keep_by-external.json")
                     .read_text())
    assert res["w1s_set"] == {"w1s_keep": 4, "w1s_keep_by": "external"}
    assert seen and all(s[2][1] == 512 for s in seen)
    for ks, kp, shape in seen:
        if mode == "attn":
            assert ks is None and kp.shape == (2, 512, 512)
        else:
            assert kp is None and ks.shape == (2, 512)
            if mode == "offsets":
                assert (ks.sum(1) == 3 * 4).all()                   # 3 facts × (space, digits 1–3)
    if mode == "offsets":                                           # the space before each fact's digits
        task = evolve_run.ScatterNeedleTask(tok(), 512, 3, 1, min_gap=8, block=64)
        rng = random.Random(2024)
        ids, _, _ = task.sample(rng)
        ks = seen[0][0][0]
        for v in task.last_value_at:
            assert ks[v - 1:v + 3].tolist() == [1.0] * 4 and ids[v] == ord(" ")


def test_supw_fire_rates_are_reported(w1s_run):
    """G0.3: the W1s summary reports the rate of key matches (cos > θ) between slots held at the query, by
    kind: different facts (false), fact and filler, and the old and new statements of an updated name (true)."""
    out = w1s_run
    eval_by_query.main([str(out), "--n", "8", "--batch", "4", "--update", "same", "--pairs", "3",
                        "--w1s-set", "w1s_recency_form=supw", "--out", str(out.with_suffix(".fire.json"))])
    w = json.loads(out.with_suffix(".fire.json").read_text())["w1s"]
    assert "fire_false_rate" in w and 0.0 <= w["fire_false_rate"] <= 1.0 and w["fire_false_pairs"] > 0


# --------------------------------------------------- fixes from the round-11 code review (4 Nov 2026)
def test_name_rule_uses_every_list_name_so_familiarity_is_no_cue():
    """F8: beyond the list, every list name appears in every sequence, so a familiar name does not mark
    the queried fact (it used to be the only list name in ~80% of sequences)."""
    t = tok()
    for cls, kw in ((evolve_run.ScatterNeedleTask, {}), (evolve_run.UpdateNeedleTask, {"form": "now"})):
        task = cls(t, 4096, 24, 1, min_gap=8, block=64, names=evolve_run.TRAIN_NAMES, **kw)
        rng = random.Random(3)
        for _ in range(10):
            ids, _, _ = task.sample(rng)
            names = [_stmt(ids, task, r).split()[3] for r in range(len(task.last_value_at))]
            assert set(evolve_run.TRAIN_NAMES) <= set(names), cls


def test_update_k_digit1_differs_from_every_earlier_code():
    """F15: with k restatements, the final code's digit 1 differs from all earlier codes of the name."""
    t = tok()
    task = evolve_run.UpdateNeedleTask(t, 1024, 3, 1, min_gap=8, block=64, form="same", k=3)
    rng = random.Random(0)
    for _ in range(50):
        ids, n_ans, _ = task.sample(rng)
        codes = [_stmt(ids, task, r).split()[-1].rstrip(".") for r in task.last_update_ranks]
        assert len({c[0] for c in codes}) == len(codes), codes
    with pytest.raises(ValueError):
        evolve_run.UpdateNeedleTask(t, 1024, 3, 1, min_gap=8, block=64, form="same", k=10)


def test_syllable_pool_has_no_prefix_collisions():
    """F11: no pool name starts with a training or held-out name ('Tomi' would share the token ' Tom')."""
    from evolve.names import HELDOUT_NAMES, syllable_pool
    taken = set(HELDOUT_NAMES) | set(pl.NAMES)
    assert not any(p.startswith(n) for p in syllable_pool(2000, 0) for n in taken)


def test_fire_owner_is_the_fact_the_key_pools_from():
    """F7: a slot whose onset sits one token before a fact, but whose key pools mostly from it, belongs
    to that fact (it used to count as filler)."""
    diag = {"retained": torch.tensor([[[99, 300]]]), "cand_s": torch.tensor([[99, 300]]),
            "cand_key": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])}
    acc = {"fire_false": [], "fire_filler": [], "fire_true": []}
    eval_by_query._fires(diag, 0, 0, [(100, 110), (300, 310)], set(), 0.9, acc, P=0, pool=8)
    assert acc["fire_false"] == [1.0] and not acc["fire_filler"]


def test_overrides_that_would_mislead_are_refused(w1s_run):
    """F4: an init-only --w1s-set the checkpoint overwrites; F5: a time feature without an explicit recency
    form; F14: a learned code with no fitted W_z (unless --allow-random-code). F6: the seed is recorded."""
    out, base = w1s_run, ["--n", "4", "--batch", "2"]
    with pytest.raises(SystemExit, match="initial values"):
        eval_by_query.main([str(out)] + base + ["--w1s-set", "w1s_recency=0.01"])
    with pytest.raises(SystemExit, match="recency_form"):
        eval_by_query.main([str(out)] + base + ["--w1s-set", "w1s_time_feat=rope"])
    eval_by_query.main([str(out)] + base + ["--w1s-set", "w1s_time_feat=rope", "--w1s-set", "w1s_recency_form=none",
                                            "--out", str(out.with_suffix(".rope.json"))])
    with pytest.raises(SystemExit, match="fitted"):
        eval_by_query.main([str(out)] + base + ["--w1s-set", "w1s_code=learned"])
    eval_by_query.main([str(out)] + base + ["--w1s-set", "w1s_code=learned", "--allow-random-code", "--seed", "7",
                                            "--out", str(out.with_suffix(".rnd.json"))])
    rec = json.loads(out.with_suffix(".rnd.json").read_text())
    assert rec["eval"]["seed"] == 7 and rec["eval"]["allow_random_code"]


# ------------------------------------- round 11 eval cells: paraphrased queries (RV4.E3), repeated digits (E12)
class _QwenLikeTok:
    """A word-level stand-in for Qwen's tokenizer (no downloads): its pre-tokenizer split, approximately (a word
    takes its leading space, contractions such as "'s" and runs of punctuation are tokens of their own, and every
    digit is one token), with ids handed out on first sight. 0 is BOS/EOS."""

    SPLIT = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
    WORDS = " the a of and to in was by on at it his her they then over under after night river stone"

    def __init__(self) -> None:
        self.bos_token_id = self.eos_token_id = 0
        self.ids, self.pieces = {}, ["<|endoftext|>"]
        self._pool = self.encode(self.WORDS)

    def encode(self, text: str) -> list[int]:
        out = []
        for p in self.SPLIT.findall(text):
            if p not in self.ids:
                self.ids[p] = len(self.pieces)
                self.pieces.append(p)
            out.append(self.ids[p])
        return out

    def decode(self, ids) -> str:
        return "".join(self.pieces[int(i)] for i in ids if int(i))

    def filler_pool(self) -> list[int]:
        return list(self._pool)


def test_qwen_like_tokenizer_splits_like_qwen():
    q = _QwenLikeTok()
    pieces = lambda s: [q.pieces[i] for i in q.encode(s)]          # noqa: E731
    assert pieces(" The code for Alice is") == [" The", " code", " for", " Alice", " is"]
    assert pieces(" 4071") == [" ", "4", "0", "7", "1"]              # an answer: the space, then one token per digit
    assert pieces(" Q: What is the code for Alice? A:") == [" Q", ":", " What", " is", " the", " code", " for",
                                                              " Alice", "?", " A", ":"]
    assert pieces(" Alice's code is") == [" Alice", "'s", " code", " is"]


def _queried(tok_, ids, task):
    """The queried statement's name and the value tokens at its value_at."""
    r = task.last_queried
    (lo, _), v = task.last_spans[r], task.last_value_at[r]
    return tok_.decode(ids[lo:v]).split()[3], ids[v:v + 5]          # " The code for NAME is [now]"


@pytest.mark.parametrize("k", sorted(evolve_run.PARAPHRASE_QUERIES))
@pytest.mark.parametrize("cls,kw", [(evolve_run.ScatterNeedleTask, {}), (evolve_run.UpdateNeedleTask, {"form": "now"}),
                                    (evolve_run.ScatterNeedleTask, {"names": "heldout"})])
def test_paraphrased_query_renders_answers_and_pairs_with_the_plain_cell(k, cls, kw):
    """--paraphrase K on a Qwen-like tokenizer: the query is the template with the queried name, the answer is that
    fact's " DDDD" (5 tokens), and the sequence is the plain cell's, draw for draw, up to the end of the last filler:
    same facts, value_at, spans, queried rank and gap, the same distance from the last fact to the answer, and the
    generator in step (so the next sample pairs too)."""
    q = _QwenLikeTok()
    kw = dict(kw)
    if kw.get("names") == "heldout":
        kw["names"] = eval_by_query.HELDOUT_NAMES
    tpl = evolve_run.PARAPHRASE_QUERIES[k]
    para = cls(q, 512, 3, 1, min_gap=8, block=64, query=tpl, **kw)
    plain = cls(q, 512, 3, 1, min_gap=8, block=64, **kw)
    a, b = random.Random(k), random.Random(k)
    for _ in range(30):
        ids, n_ans, gap = para.sample(a)
        ids0, n_ans0, gap0 = plain.sample(b)
        assert len(ids) == len(ids0) == 513 and n_ans == n_ans0 == 5 and gap == gap0
        for key in ("last_queried", "last_value_at", "last_spans", "last_heads", "last_old_rank", "last_old_answer"):
            assert getattr(para, key, None) == getattr(plain, key, None), key
        name, value = _queried(q, ids, para)
        ql, ql0 = para.last_query_len, plain.last_query_len
        assert q.decode(ids[-(ql + n_ans):-n_ans]) == tpl.format(n=name)
        assert q.decode(ids0[-(ql0 + n_ans):-n_ans]) == evolve_run.PLAIN.format(n=name)
        assert ids[-n_ans:] == value == ids0[-n_ans:] and re.fullmatch(r" \d{4}", q.decode(value))
        cut = len(ids) - n_ans - max(ql, ql0)
        assert ids[:cut] == ids0[:cut]                              # everything before the longer query
        end = max(hi for _, hi in para.last_spans)                  # the last fact's last token
        assert len(ids) - ql - n_ans - (end + 1) >= para.tail - max(0, ql - ql0)
        if kw.get("names"):
            assert name in eval_by_query.HELDOUT_NAMES
        assert a.getstate() == b.getstate()


def test_paraphrase_shorter_query_continues_natural_filler():
    """Template 2 is shorter than the plain query (bytes): the filler gains the difference, continuing the corpus
    where the last filler stopped (no seam in natural text), and the generator stays in step with the plain cell."""
    t = tok()
    c = t.encode(CORPUS)
    para = evolve_run.ScatterNeedleTask(t, 512, 3, 1, c, min_gap=8, block=64, query=evolve_run.PARAPHRASE_QUERIES[2])
    plain = evolve_run.ScatterNeedleTask(t, 512, 3, 1, c, min_gap=8, block=64)
    a, b = random.Random(11), random.Random(11)
    for _ in range(20):
        ids, n_ans, _ = para.sample(a)
        ids0, _, _ = plain.sample(b)
        ql, ql0 = para.last_query_len, plain.last_query_len
        assert ql < ql0
        end = para.last_spans[-1][1]
        tail = text(ids[end + 1:len(ids) - n_ans - ql])
        assert tail.startswith(text(ids0[end + 1:len(ids0) - n_ans - ql0])) and tail in CORPUS * 3
    assert a.getstate() == b.getstate()


def test_paraphrase_refusals():
    t = tok()
    with pytest.raises(ValueError, match="own queries"):
        evolve_run.OrderNeedleTask(t, 512, 3, 1, min_gap=8, block=64, kind="before",
                                   query=evolve_run.PARAPHRASE_QUERIES[1])
    with pytest.raises(ValueError, match="{n}"):
        evolve_run.ScatterNeedleTask(t, 512, 3, 1, min_gap=8, block=64, query=" What is the code?")
    long_q = evolve_run.ScatterNeedleTask(t, 512, 3, 1, min_gap=8, tail=10, query=" Please tell me the code for {n} now")
    with pytest.raises(ValueError, match="needle-tail"):           # 17 bytes longer than the plain query, tail 10
        long_q.sample(random.Random(0))


def test_plain_query_template_keeps_the_golden_sequences():
    """query=PLAIN given explicitly is the default: the pre-round-11 golden digests still hold."""
    t = tok()
    assert _digest(evolve_run.ScatterNeedleTask(t, 512, 3, 1, min_gap=8, block=64, query=evolve_run.PLAIN)) \
        == GOLDEN["scatter_p3"]
    assert _digest(evolve_run.UpdateNeedleTask(t, 512, 3, 1, min_gap=8, block=64, form="now", query=evolve_run.PLAIN)) \
        == GOLDEN["update_now"]


@pytest.mark.parametrize("make", [tok, _QwenLikeTok])
def test_answer_code_and_repeat_flag_on_constructed_sequences(make):
    t = make()
    for code, want in (("1123", True), ("1234", False), ("9090", True), ("0000", True), ("5072", False)):
        ids = [t.bos_token_id] + t.encode(" the cat sat The code for Bob is") + t.encode(f" {code}")
        got = eval_by_query.answer_code(t, ids[-5:])
        assert got == code and eval_by_query.has_repeat(got) is want
    with pytest.raises(ValueError, match="digits"):
        eval_by_query.answer_code(t, t.encode(" is"))


def test_repeat_split_counts_and_accuracies():
    """_summarise with repeat flags: digit 1 and exact split by "the code repeats a digit", with the counts; an
    empty group is None; without flags the summary is as before (no new keys)."""
    per_seq = [(0, 300, 1.0, 1.0), (1, 310, 0.0, 0.0), (2, 320, 1.0, 0.0), (0, 330, 1.0, 1.0), (1, 340, 0.0, 0.0)]
    rep = [True, True, False, False, False]
    s = eval_by_query._summarise(per_seq, 3, [-1, 10 ** 9], rep)
    assert s["exact_by_repeat"] == {"repeat": 0.5, "no_repeat": pytest.approx(1 / 3), "n_repeat": 2, "n_no_repeat": 3}
    assert s["digit1_by_repeat"] == {"repeat": 0.5, "no_repeat": pytest.approx(2 / 3), "n_repeat": 2, "n_no_repeat": 3}
    s = eval_by_query._summarise(per_seq, 3, [-1, 10 ** 9], [True] * 5)
    assert s["exact_by_repeat"]["no_repeat"] is None and s["exact_by_repeat"]["n_no_repeat"] == 0
    plain = eval_by_query._summarise(per_seq, 3, [-1, 10 ** 9])
    assert set(plain) == {"digit1", "exact", "digit1_by_queried_rank", "exact_by_queried_rank",
                          "count_by_queried_rank", "by_gap"}
    with pytest.raises(ValueError, match="repeat flags"):
        eval_by_query._summarise(per_seq, 3, [-1, 10 ** 9], rep[:4])


def test_repeat_share_of_sampled_codes_is_about_half():
    """1 − 10·9·8·7 / 10⁴ = 0.496 of 4-digit codes repeat a digit; read from sampled sequences' answer tokens."""
    t = tok()
    task = evolve_run.ScatterNeedleTask(t, 512, 3, 1, min_gap=8, block=64)
    rng = random.Random(12)
    flags = []
    for _ in range(3000):
        ids, n_ans, _ = task.sample(rng)
        flags.append(eval_by_query.has_repeat(eval_by_query.answer_code(t, ids[-n_ans:])))
    assert abs(sum(flags) / len(flags) - 0.496) < 0.03


PRE_E12_KEYS = {"result", "pairs", "seq_len", "text_file", "scatter", "n", "digit1_acc_by_queried_rank",
                "exact_by_queried_rank", "count_by_queried_rank", "digit1", "exact", "by_gap", "controls",
                "per_sequence", "missing_keys", "unexpected_keys", "w1s_state", "w1s", "eval"}
PRE_E12_CONTROL_KEYS = {"digit1", "exact", "digit1_by_queried_rank", "exact_by_queried_rank", "count_by_queried_rank",
                        "by_gap"}
E12_KEYS = {"exact_by_repeat", "digit1_by_repeat"}


def _check_repeat_split(res):
    """The repeat split in the main output and every control agrees with per_sequence + per_sequence_repeat."""
    rep = res["per_sequence_repeat"]
    assert len(rep) == len(res["per_sequence"]) == res["n"] and all(len(r) == 4 for r in res["per_sequence"])
    for name in ("digit1", "exact"):
        want = eval_by_query._by_repeat(res["per_sequence"], rep)[f"{name}_by_repeat"]
        assert res[f"{name}_by_repeat"] == want
        assert want["n_repeat"] + want["n_no_repeat"] == res["n"]
    for c in res["controls"].values():
        assert E12_KEYS <= set(c) and c["exact_by_repeat"]["n_repeat"] == sum(rep)


def test_default_eval_output_only_adds_the_repeat_split(w1s_run):
    """With default flags the output keeps every key it had (no "task"), and adds only the E12 split: the top-level
    exact/digit1_by_repeat and per_sequence_repeat, and the two splits in each control."""
    out = w1s_run
    eval_by_query.main([str(out), "--n", "8", "--batch", "4"])
    res = json.loads(out.with_suffix(".by_query.json").read_text())
    assert set(res) == PRE_E12_KEYS | E12_KEYS | {"per_sequence_repeat"}
    assert set(res["controls"]) == {"windowed", "full"}
    assert all(set(c) == PRE_E12_CONTROL_KEYS | E12_KEYS for c in res["controls"].values())
    _check_repeat_split(res)


@pytest.mark.parametrize("k,extra,suffix", [(1, ["--heldout-names", "--update", "now"], ".update-now.heldout.para1"),
                                            (2, ["--ctrl-conds"], ".para2")])
def test_paraphrase_end_to_end_on_w1s(w1s_run, k, extra, suffix):
    """eval_by_query.py --paraphrase on the tiny W1s run: its own file, the template recorded, the repeat split in
    every condition (side controls too), windowed at 0, and per_sequence paired row by row with the plain cell."""
    out = w1s_run
    base = [str(out), "--n", "8", "--batch", "4"] + extra
    eval_by_query.main(base + ["--paraphrase", str(k)])
    res = json.loads(out.with_suffix(f".by_query{suffix}.json").read_text())
    assert res["task"]["paraphrase"] == k and res["task"]["query_template"] == evolve_run.PARAPHRASE_QUERIES[k]
    assert res["controls"]["windowed"]["exact"] == 0.0
    _check_repeat_split(res)
    if "--update" in extra:
        assert res["task"]["names"] == "heldout" and res["task"]["update"] == "now"
        assert set(res["update"]) == {"windowed+side", "windowed", "full"}
    else:
        assert {"side_empty", "side_shuffle"} <= set(res["controls"])
    eval_by_query.main(base + ["--out", str(out.with_suffix(".plain.json"))])
    plain = json.loads(out.with_suffix(".plain.json").read_text())
    assert [r[:2] for r in plain["per_sequence"]] == [r[:2] for r in res["per_sequence"]]   # rank and gap
    assert plain["per_sequence_repeat"] == res["per_sequence_repeat"]
    with pytest.raises(SystemExit, match="paraphrase"):
        eval_by_query.main([str(out), "--n", "4", "--batch", "2", "--order", "before", "--paraphrase", "1"])


def test_paraphrase_keeps_the_last_fact_beyond_the_window():
    """Review R7: a paraphrased query that eats the tail filler down to the window or less is refused."""
    t = tok()
    q = " Remember, as stated much earlier in this long text, the code for {n}. It is"
    task = evolve_run.ScatterNeedleTask(t, 512, 3, 1, min_gap=8, block=64, tail=70, query=q)
    with pytest.raises(ValueError, match="window"):
        task.sample(random.Random(0))


# ------------------------------------------------------------------ round-11 Stage A2: stale per sequence, --init-from
@pytest.mark.parametrize("extra", [[], ["--update-k", "2", "--ctrl-conds"]])
def test_per_sequence_stale_is_parallel_and_adds_only_that_key(w1s_run, extra):
    """--update: per_sequence_stale (top level and in each control) is the stale exact per sequence, parallel to
    per_sequence: its mean is update[cond]["old_exact"]. Nothing else changes; without --update it is absent."""
    out = w1s_run
    base = [str(out), "--n", "8", "--batch", "4", "--update", "now"] + extra
    eval_by_query.main(base + ["--out", str(out.with_suffix(".stale.json"))])
    res = json.loads(out.with_suffix(".stale.json").read_text())
    ps = res["per_sequence_stale"]
    assert len(ps) == len(res["per_sequence"]) == res["n"] and set(ps) <= {0.0, 1.0}
    assert sum(ps) / len(ps) == pytest.approx(res["update"]["windowed+side"]["old_exact"])
    for c, v in res["controls"].items():
        assert len(v["per_sequence_stale"]) == res["n"]
        assert sum(v["per_sequence_stale"]) / res["n"] == pytest.approx(res["update"][c]["old_exact"])
    assert set(res["controls"]["windowed"]["per_sequence_stale"]) == {0.0}
    if "--ctrl-conds" in extra:
        assert set(res["controls"]["side_empty"]["per_sequence_stale"]) == {0.0}
    want = PRE_E12_KEYS | E12_KEYS | {"per_sequence_repeat", "task", "update", "per_sequence_stale"}
    assert set(res) == want | ({"ctrl_conds"} if "--ctrl-conds" in extra else set())
    assert all(set(c) == PRE_E12_CONTROL_KEYS | E12_KEYS | {"per_sequence_stale"} for c in res["controls"].values())
    eval_by_query.main([str(out), "--n", "8", "--batch", "4", "--out", str(out.with_suffix(".plain.json"))])
    assert "per_sequence_stale" not in json.loads(out.with_suffix(".plain.json").read_text())


INIT = ["--tiny", "--task", "needle", "--window", "64", "--seq-len", "512", "--pairs", "3", "--batch", "2",
        "--eval-n", "2", "--device", "cpu", "--needle-scatter", "8"]


def test_init_from_loads_fragments_before_training_and_records_them(fake_set, tmp_path):
    """--init-from: the fragments' values are in the side when training starts (an untrained parameter keeps
    them; a trained one starts there), they beat --init-gain, and the JSON lists each file's keys."""
    out0, _ = fake_set
    assert "init_from" not in json.loads(out0.read_text())        # default: nothing loaded, nothing recorded
    f1, f2 = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save({"loop.log_beta_align": torch.tensor(3.0), "loop.gain_logit": torch.tensor(1.25)}, f1)
    torch.save({"scale": torch.tensor(2.0, dtype=torch.float64)}, f2)   # cast to the side's dtype on load
    out = tmp_path / "i.json"
    evolve_run.main(INIT + ["--steps", "2", "--init-gain", "0.5", "--candidate", "fake_set:FakeSide",
                            "--init-from", str(f1), "--init-from", str(f2), "--out", str(out)])
    st = torch.load(out.with_suffix(".pt"))
    assert st["loop.log_beta_align"].item() == 3.0 and st["loop.gain_logit"].item() == 1.25   # no gradient: kept
    assert st["scale"].dtype == torch.float32 and abs(st["scale"].item() - 2.0) < 0.01         # trained from 2.0
    assert abs(torch.load(out0.with_suffix(".pt"))["scale"].item() - 1.0) < 0.01                 # (vs from 1.0)
    rec = json.loads(out.read_text())
    assert rec["init_from"] == [{"path": str(f1), "keys": ["loop.gain_logit", "loop.log_beta_align"]},
                                {"path": str(f2), "keys": ["scale"]}]


@pytest.mark.parametrize("frag,flags,msg", [
    ({"nope": torch.tensor(1.0)}, [], "keys not in the side"),
    ({"scale": torch.ones(3)}, [], "shapes differ"),
    ({"scale": torch.tensor(1.0)}, ["--steps", "0"], "--steps > 0"),
    ({}, [], "non-empty dict"),
    ({"scale": 1.0}, [], "not tensors"),
])
def test_init_from_refusals(fake_set, tmp_path, frag, flags, msg):
    f = tmp_path / "f.pt"
    torch.save(frag, f)
    argv = INIT + ["--steps", "1"] + flags + ["--candidate", "fake_set:FakeSide", "--init-from", str(f),
                                              "--out", str(tmp_path / "x.json")]
    with pytest.raises(SystemExit, match=msg):
        evolve_run.main(argv)


def test_init_from_refuses_a_key_in_two_fragments_and_a_missing_file(fake_set, tmp_path):
    f1, f2 = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save({"scale": torch.tensor(2.0)}, f1)
    torch.save({"scale": torch.tensor(3.0)}, f2)
    base = INIT + ["--steps", "1", "--candidate", "fake_set:FakeSide", "--out", str(tmp_path / "x.json")]
    with pytest.raises(SystemExit, match="one source per key"):
        evolve_run.main(base + ["--init-from", str(f1), "--init-from", str(f2)])
    with pytest.raises(SystemExit, match="no such file"):
        evolve_run.main(base + ["--init-from", str(tmp_path / "none.pt")])


def test_init_from_w1s_fitted_code_reaches_training(tmp_path, monkeypatch):
    """C5r's use: a fitted {loop.wz.weight, loop.log_beta_align} fragment into a W1s run with the learned code.
    The side holds the fragment's values when pl.train starts, the other parameters are the constructor's, and
    the checkpoint loads into a fresh side with no missing keys."""
    keep_pl(monkeypatch)
    from evolve.w1s_slots import W1sSide, split_args
    flags = ["--w1s-taps", "1", "--w1s-code", "learned", "--w1s-dc", "8"]
    g = torch.Generator().manual_seed(5)
    frag = {"loop.wz.weight": torch.randn(8, 64, generator=g), "loop.log_beta_align": torch.tensor(2.5)}
    f = tmp_path / "wz.pt"
    torch.save(frag, f)
    seen = {}
    orig = evolve_run._train

    def spy(lm, task, args):
        seen.update({k: v.detach().clone() for k, v in lm.side.state_dict().items()})
        return orig(lm, task, args)
    monkeypatch.setattr(evolve_run, "_train", spy)

    def run(out, extra):
        evolve_run.main(INIT + ["--steps", "1", "--layer", "3", "--candidate", "evolve.w1s_slots:W1sSide"]
                        + flags + extra + ["--out", str(out)])
    run(tmp_path / "base.json", [])
    base = dict(seen)
    run(tmp_path / "init.json", ["--init-from", str(f)])
    assert torch.equal(seen["loop.wz.weight"], frag["loop.wz.weight"])
    assert seen["loop.log_beta_align"].item() == 2.5
    assert all(torch.equal(seen[k], base[k]) for k in base if k not in frag)   # the rest as constructed
    assert not torch.equal(base["loop.wz.weight"], frag["loop.wz.weight"])
    rec = json.loads((tmp_path / "init.json").read_text())
    assert rec["init_from"] == [{"path": str(f), "keys": ["loop.log_beta_align", "loop.wz.weight"]}]
    side = W1sSide(64, type("A", (), {"time": True})(), split_args(flags)[0])
    missing, unexpected = side.load_state_dict(torch.load(tmp_path / "init.pt"), strict=False)
    assert not missing and not unexpected
