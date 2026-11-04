"""The evaluation tools of the evolution search (experiments/evolve/run.py, eval_by_query.py): the
scattered needle task's variants (jitter, update, held-out names), the oracle-onset plumbing and
the slot diagnostics. Uses the tiny byte-tokenizer model only (CPU, no downloads)."""

import json
import random
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
