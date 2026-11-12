"""Round 12 decoding control (experiments/ca2/decode.py): logits processors against hand computations and
reference implementations, the reasoning-span gate in both layouts (Qwen3: the model opens the span; Qwen3.5: the
prompt opens it), forced tokens, the n-gram loop rule against ``judge.repetition_onset``, the stop controller, and
segmented generation with a forked probe against unsegmented ``generate``.

Tiny random models with the byte tokenizer (reason.build_tiny / pretrained_loop.build_tiny); no downloads.
"""

import random
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments" / "alignment")]

from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor  # noqa: E402

from ca2 import decode as D  # noqa: E402
from ca2 import reason as R  # noqa: E402
from ca2.common import alignment  # noqa: E402
import pretrained_loop as pl  # noqa: E402  (on sys.path through ca2)

A = alignment()
_BT = pl.ByteTokenizer()
BOS = 256
CLOSE = R.TINY_THINK_END              # 257 stands for </think>
OPEN = 258                            # a tiny stand-in for <think> (the Qwen3 layout, where the model opens the span)


# ------------------------------------------------------------------------------------------- helpers
def tiny(arch: str = "qwen3"):
    model = R.build_tiny("qwen3", 0)[0] if arch == "qwen3" else pl.build_tiny("qwen35", 0)[0]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture
def torch_gdn():
    """Qwen3.5's GDN kernels swapped for their pure-torch originals (CPU), restored afterwards."""
    if not hasattr(transformers, "Qwen3_5TextConfig"):
        pytest.skip("transformers has no Qwen3.5 classes")
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    saved = {n: getattr(M, n) for n in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")}
    R.torch_gdn_kernels()
    yield
    for n, f in saved.items():
        setattr(M, n, f)


def prompts(n: int = 3, seed: int = 0, open_span: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    seqs = [[BOS] + [rng.randrange(32, 127) for _ in range(rng.randrange(5, 12))] + ([OPEN] if open_span else [])
            for _ in range(n)]
    return R.pad_left(seqs, BOS)


def ref_presence(input_ids: torch.Tensor, scores: torch.Tensor, P, presence: float, frequency: float = 0.0):
    """OpenAI / vLLM semantics written out per row and token: logit[v] −= presence·[c_v > 0] + frequency·c_v,
    with c_v the count of v in the generated part only."""
    out = scores.clone()
    for b in range(input_ids.shape[0]):
        p = int(P[b]) if isinstance(P, (list, torch.Tensor)) else int(P)
        counts: dict[int, int] = {}
        for v in input_ids[b, p:].tolist():
            counts[v] = counts.get(v, 0) + 1
        for v, c in counts.items():
            out[b, v] -= presence * (c > 0) + frequency * c
    return out


# ------------------------------------------------------------------------------------------- penalties
def test_presence_penalty_matches_the_openai_reference():
    g = torch.Generator().manual_seed(0)
    B, P, T, V = 4, 6, 15, 40
    ids = torch.randint(0, 12, (B, T), generator=g)                  # small alphabet: repeats are frequent
    scores = torch.randn(B, V, generator=g)
    for pp, fp in ((1.5, 0.0), (0.7, 0.3), (0.0, 0.0)):
        got = D.PresencePenalty(pp, P, frequency=fp)(ids, scores)
        assert torch.allclose(got, ref_presence(ids, scores, P, pp, fp), atol=0, rtol=0)
    # per-row prompt widths (prompt tokens are never penalised)
    Pr = torch.tensor([3, 6, 9, 15])
    got = D.PresencePenalty(1.5, Pr)(ids, scores)
    assert torch.equal(got, ref_presence(ids, scores, Pr, 1.5))
    assert torch.equal(got[3], scores[3])                             # nothing generated yet in row 3
    # hand computation: one row, generated part [5, 5, 7]: only tokens 5 and 7 drop, by exactly 1.5, once each
    ids1 = torch.tensor([[1, 2, 5, 5, 7]])
    s1 = torch.zeros(1, 10)
    out = D.PresencePenalty(1.5, 2)(ids1, s1)
    want = torch.zeros(1, 10)
    want[0, 5] = want[0, 7] = -1.5
    assert torch.equal(out, want)
    # not transformers' repetition penalty (multiplicative, prompt included)
    s2 = torch.tensor([[2.0, -2.0, 1.0, 0.0, 0.0, 3.0, 0.0, -1.0, 0.0, 0.0]])
    rp = RepetitionPenaltyLogitsProcessor(1.5)(ids1, s2.clone())
    pp = D.PresencePenalty(1.5, 2)(ids1, s2.clone())
    assert rp[0, 1] != s2[0, 1] and pp[0, 1] == s2[0, 1]              # prompt token 2 (id 1): rp only
    assert rp[0, 5] == pytest.approx(3.0 / 1.5) and pp[0, 5] == pytest.approx(3.0 - 1.5)
    assert rp[0, 7] == pytest.approx(-1.5) and pp[0, 7] == pytest.approx(-2.5)


def test_presence_penalty_inside_generate_equals_a_manual_loop():
    """Greedy generation with the processor equals a hand-written decode loop applying the reference penalty."""
    model = tiny()
    ids, mask = prompts(2, seed=3)
    P = ids.shape[1]
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=12, do_sample=False,
                         logits_processor=LogitsProcessorList([D.PresencePenalty(3.0, P)]), pad_token_id=BOS,
                         eos_token_id=None)
    seq = ids.clone()
    m = mask.clone()
    for _ in range(12):
        pos = (m.cumsum(1) - 1).clamp_min(0)
        logits = model(input_ids=seq, attention_mask=m, position_ids=pos).logits[:, -1].float()
        nxt = ref_presence(seq, logits, P, 3.0).argmax(-1)
        seq = torch.cat([seq, nxt[:, None]], 1)
        m = torch.cat([m, torch.ones_like(m[:, :1])], 1)
    assert torch.equal(out, seq)


# ------------------------------------------------------------------------------------------- span gate
def test_span_tracker_both_layouts():
    P = 4
    # Qwen3.5 layout: the prompt ends inside the span; Qwen3 layout: the model emits the open marker itself
    q35 = torch.tensor([[BOS, 9, 9, OPEN, 10, 11, CLOSE, 12], [BOS, BOS, 9, OPEN, 10, 11, 12, 13]])
    q3 = torch.tensor([[BOS, 9, 9, 9, OPEN, 10, CLOSE, 12], [BOS, 9, 9, 9, 10, OPEN, 11, 12]])
    sp = D.SpanTracker(CLOSE, P, open_id=OPEN)
    st = [sp.state(q35[:, :t]) for t in range(P, q35.shape[1] + 1)]
    assert sp.prompt_open.tolist() == [True, True]
    assert [s[0].tolist() for s in st] == [[True, True], [True, True], [True, True], [False, True], [False, True]]
    assert st[-1][1].tolist() == [2, 4]                                # reasoning tokens, markers excluded
    sp3 = D.SpanTracker(CLOSE, P, open_id=OPEN)
    st3 = [sp3.state(q3[:, :t]) for t in range(P, q3.shape[1] + 1)]
    assert sp3.prompt_open.tolist() == [False, False]
    assert [s[0].tolist() for s in st3] == [[False, False], [True, False], [True, True], [False, True],
                                            [False, True]]
    assert st3[-1][1].tolist() == [1, 2]
    # an explicit prompt state (tiny byte tokenizer: no open marker)
    sp0 = D.SpanTracker(CLOSE, 2, prompt_open=True)
    assert sp0.state(torch.tensor([[1, 2, 3, CLOSE, 4]]))[0].tolist() == [False]


def test_processors_act_only_inside_the_span():
    P = 4
    ids = torch.tensor([[BOS, 9, 9, 9, OPEN, 10, 11],          # Qwen3 layout, span open (opened by the model)
                        [BOS, 9, 9, 9, 10, 11, 12],            # Qwen3 layout, span never opened
                        [BOS, 9, 9, OPEN, 10, CLOSE, 11]])     # Qwen3.5 layout, span closed
    V = 300
    s = torch.zeros(3, V)
    sp = D.SpanTracker(CLOSE, P, open_id=OPEN)
    bias = D.ThinkCloseBias(CLOSE, 3, sp)
    bias.set([0, 1, 2], 4.0)
    out = bias(ids, s)
    assert out[:, CLOSE].tolist() == [4.0, 0.0, 0.0] and out.abs().sum() == 4.0
    sup = D.SuppressTokens([10, 20], sp)(ids, s)
    assert torch.isinf(sup[0, [10, 20]]).all() and torch.isfinite(sup[1:]).all()
    pres = D.PresencePenalty(1.0, P, span=sp)(ids, s)                 # span-restricted variant
    assert pres[0, 10] == -1.0 and pres[0, 11] == -1.0 and pres[1].abs().sum() == 0 and pres[2].abs().sum() == 0
    pres_all = D.PresencePenalty(1.0, P)(ids, s)                      # vendor default: the whole response
    assert pres_all[1, 12] == -1.0 and pres_all[2, CLOSE] == -1.0
    # loop rule: a repetitive window outside the span never fires
    loop = torch.tensor([[BOS] * P + [1, 2] * 8 + [OPEN] + [1, 2] * 8])
    det = D.NgramLoopRule(1, P, window=8, n=2, threshold=0.5)
    det.span = D.SpanTracker(CLOSE, P, open_id=OPEN)
    for t in range(P + 1, loop.shape[1] + 1):
        det(loop[:, :t], torch.zeros(1, V))
    assert int(det.fired_at[0]) == 17                                 # not before the model opened the span (t=17)
    assert int(det.fired_span[0]) >= 0


def test_think_close_bias_is_settable_between_steps():
    sp = D.SpanTracker(CLOSE, 1, prompt_open=True)
    b = D.ThinkCloseBias(CLOSE, 2, sp)
    ids = torch.tensor([[BOS, 5], [BOS, 6]])
    s = torch.randn(2, 300)
    assert torch.equal(b(ids, s), s)                                  # zero bias: untouched
    b.set([1], 2.5)
    out = b(ids, s)
    assert out[1, CLOSE] == pytest.approx(float(s[1, CLOSE]) + 2.5) and torch.equal(out[0], s[0])
    b.set([1], 0.0)
    assert torch.equal(b(ids, s), s)


# ------------------------------------------------------------------------------------------- forced tokens
def test_force_tokens_emit_exactly_the_queue_then_release():
    f = D.ForceTokens(2, prompt_len=1)
    f.push(0, [7, 8, 9])
    s = torch.randn(2, 20)
    ids = torch.tensor([[BOS], [BOS]])
    o1 = f(ids, s)
    assert o1[0].argmax() == 7 and torch.isinf(o1[0]).sum() == 19 and torch.equal(o1[1], s[1])
    f(ids, s), f(ids, s)
    assert f.pending(0) == 0 and torch.equal(f(ids, s), s)            # released
    assert [v for _, v in f.emitted[0]] == [7, 8, 9]
    # inside generate, under sampling at a high temperature with top-k and top-p
    model = tiny()
    ids, mask = prompts(3, seed=1)
    q = [65, 66, 67, 68, 69]
    for seed in (0, 1):
        f = D.ForceTokens(3, ids.shape[1])
        f.push(1, q)
        torch.manual_seed(seed)
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=9, do_sample=True, temperature=1.7,
                             top_k=50, top_p=0.9, logits_processor=LogitsProcessorList([f]), pad_token_id=BOS,
                             eos_token_id=None)
        new = out[:, ids.shape[1]:]
        assert new[1, :5].tolist() == q and f.pending(1) == 0
    # rows without a queue are untouched: the same seed without the processor gives the same rows 0 and 2
    torch.manual_seed(1)
    plain = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=9, do_sample=True, temperature=1.7,
                           top_k=50, top_p=0.9, pad_token_id=BOS, eos_token_id=None)[:, ids.shape[1]:]
    assert torch.equal(plain[0, :5], new[0, :5]) and torch.equal(plain[2, :5], new[2, :5])


def test_close_sequence_and_cue():
    bt = _BT
    ids = D.close_sequence(bt, CLOSE)
    assert ids.count(CLOSE) == 1 and bt.decode(ids) == D.CANONICAL_CLOSE.replace("</think>", "")
    assert D.CLOSE_SENTENCE == ("Considering the limited time by the user, I have to give the solution based on the "
                                "thinking directly now.")
    assert D.CANONICAL_CLOSE.endswith(".\n</think>\n\n")
    assert D.cue_sequence(bt, CLOSE)[0] == CLOSE and bt.decode(D.cue_sequence(bt, None)) == R.CUE_ANSWER
    with pytest.raises(ValueError):
        D.close_sequence(bt, CLOSE, text="no marker")


# ------------------------------------------------------------------------------------------- loop rule
@pytest.mark.parametrize("W,n,thr", [(32, 4, 0.5), (16, 2, 0.3), (64, 8, 0.7)])
def test_ngram_loop_rule_agrees_with_repetition_onset(W, n, thr):
    rng = random.Random(W * 7 + n)
    seqs = []
    for k in range(12):
        L = rng.randrange(W - 5, 3 * W)
        if k % 3 == 0:
            s = [rng.randrange(0, 200) for _ in range(L)]                                      # random
        elif k % 3 == 1:
            cyc = [rng.randrange(0, 200) for _ in range(rng.randrange(2, 9))]
            pre = [rng.randrange(0, 200) for _ in range(rng.randrange(0, 2 * W))]
            s = pre + (cyc * (L // len(cyc) + 1))[:L]                                           # random, then a loop
        else:
            s = [rng.randrange(0, 6) for _ in range(L)]                                        # small alphabet
        seqs.append(s)
    P = 3
    for s in seqs:
        ids = torch.tensor([[BOS] * P + s])
        det = D.NgramLoopRule(1, P, window=W, n=n, threshold=thr)
        det.span = D.SpanTracker(CLOSE, P, prompt_open=True)
        for t in range(P, ids.shape[1] + 1):                         # every processor call: t − P generated tokens
            det(ids[:, :t], torch.zeros(1, 4))
        want = A.judge.repetition_onset(s, n=n, window=W, threshold=thr)
        got = int(det.fired_at[0])
        assert (got if got >= 0 else None) == want


def test_ngram_repeat_share_equals_judge_degeneration():
    rng = random.Random(0)
    for _ in range(30):
        W, n = rng.randrange(4, 40), rng.randrange(1, 6)
        win = [rng.randrange(0, 5) for _ in range(W)]
        got = float(D.ngram_repeat_share(torch.tensor([win]), n)[0])
        assert got == pytest.approx(A.judge.degeneration(win, n), abs=1e-6)


# ------------------------------------------------------------------------------------------- stop controller
def test_stop_controller_forces_the_close_at_the_first_detection():
    model = tiny()
    ids, mask = prompts(2, seed=2, open_span=False)
    P = ids.shape[1]
    sp = D.SpanTracker(CLOSE, P, prompt_open=True)
    close_ids = [10, 10, CLOSE, 10]
    det = D.BudgetRule(2, 5)
    procs, parts = D.build_processors(prompt_len=P, batch=2, span=sp, controller=(det, "stop", 0.0),
                                      close_id=CLOSE, close_ids=close_ids, stats=True)
    torch.manual_seed(0)
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=14, do_sample=True, temperature=1.0,
                         top_k=0, top_p=1.0, logits_processor=procs, pad_token_id=BOS, eos_token_id=None)
    new = out[:, P:]
    ctrl = parts["controller"]
    for b in range(2):
        a = int(ctrl.act_step[b])
        assert a == 5 and new[b, a:a + 4].tolist() == close_ids                # exactly at the budget
        assert ctrl.record(b)["why"] == "budget-5" and ctrl.record(b)["act_span"] == 5
    # bias action: the bias is set on the acting row only, and only inside the span
    sp2 = D.SpanTracker(CLOSE, P, prompt_open=True)
    det2 = D.BudgetRule(2, 3)
    procs2, parts2 = D.build_processors(prompt_len=P, batch=2, span=sp2, controller=(det2, "bias", 50.0),
                                        close_id=CLOSE, close_ids=close_ids, stats=False)
    out2 = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6, do_sample=False,
                          logits_processor=procs2, pad_token_id=BOS, eos_token_id=None)[:, P:]
    assert out2[:, 3].tolist() == [CLOSE, CLOSE] and parts2["bias"].bias.tolist() == [50.0, 50.0]
    assert (out2[:, 4:] != CLOSE).all()                                        # span closed: the bias is off


def test_finished_rows_are_never_acted_on():
    """A row that emitted a stop id is finished: its span is closed from there and the pads it is fed are not
    reasoning, so neither the budget nor the n-gram rule fires on it and FinishedRows' pad is not overridden by
    the forced close (Qwen's pad id is itself a stop; each segmented ``generate`` call resets HF's own mask)."""
    STOP = PAD = 9
    P, V = 2, 300
    gen0 = [7, STOP] + [PAD] * 8                                     # stopped at step 1, inside the span
    gen1 = [1, 2, 3, 4, 6, 7, 8, 10, 11, 1]                          # still reasoning
    ids = torch.tensor([[0, 0] + gen0, [0, 0] + gen1])
    for given in (True, False):                                      # stops on the tracker, or from ``finished``
        for det in (D.BudgetRule(2, 4), D.NgramLoopRule(2, P, 8, 2, 0.5)):
            sp = D.SpanTracker(CLOSE, P, prompt_open=True, stops=[STOP] if given else ())
            procs, parts = D.build_processors(prompt_len=P, batch=2, span=sp, controller=(det, "stop", 0.0),
                                              close_id=CLOSE, close_ids=[3, CLOSE, 4], finished=([STOP], PAD))
            open_, n_span = sp.state(ids)
            assert open_.tolist() == [False, True] and n_span.tolist() == [1, 10]
            out = procs(ids, torch.zeros(2, V))
            ctrl = parts["controller"]
            assert int(ctrl.act_step[0]) == -1 and int(det.fired_at[0]) == -1 and not parts["force"].pending(0)
            assert int(out[0].argmax()) == PAD and out[0].isfinite().sum() == 1
            if isinstance(det, D.BudgetRule):                         # the live row is still controlled
                assert int(ctrl.act_step[1]) == 10 and int(out[1].argmax()) == 3
    # in a segmented generation: rows that stop before the budget are never forced, and stay padded
    model = tiny()
    stops = [BOS, 70]
    hit = 0
    for seed in range(6):
        ids, mask = prompts(4, seed=seed, open_span=False)
        P = ids.shape[1]
        sp = D.SpanTracker(CLOSE, P, prompt_open=True, stops=stops)
        procs, parts = D.build_processors(prompt_len=P, batch=4, span=sp, controller=(D.BudgetRule(4, 30), "stop",
                                                                                     0.0),
                                          close_id=CLOSE, close_ids=[10, CLOSE, 10], finished=(stops, BOS))
        torch.manual_seed(seed)
        so = D.generate_segmented(model, ids, mask, 48, 8, dict(do_sample=True, temperature=1.3, top_k=0,
                                                               top_p=1.0), procs, stops, BOS)
        ctrl = parts["controller"]
        for b in range(4):
            g = so.new[b].tolist()
            first = next((j for j, x in enumerate(g) if x in stops), None)
            closed = CLOSE in g[:30]                                 # the model closed its span itself first
            if first is not None and first < 30:
                hit += 1
                assert int(ctrl.act_step[b]) == -1 and all(x == BOS for x in g[first + 1:])
            else:
                assert int(ctrl.act_step[b]) == (-1 if closed else 30)
    assert hit >= 1


def test_probe_rules_consistency_and_confidence():
    pr = D.ProbeRule(2, "consist", k=3)
    pr.observe([0, 1], ["5", "5"], [None, None], 10, [10, 10])
    pr.observe([0, 1], ["5.0", "6"], [None, None], 20, [20, 20])
    pr.observe([0, 1], ["5", "6"], [None, None], 30, [30, 30])
    fired = pr.update(None, torch.tensor([True, True]), torch.tensor([30, 30]), 31)
    assert fired.tolist() == [True, False] and int(pr.fired_span[0]) == 30
    pc = D.ProbeRule(2, "conf", tau=0.9)
    pc.observe([0, 1], ["1", "2"], [0.95, 0.5], 7, [7, 7])
    assert pc.update(None, torch.tensor([True, True]), torch.tensor([7, 7]), 8).tolist() == [True, False]


# ------------------------------------------------------------------------------------------- segmented
def _segmented_equals_generate(model, sample: bool):
    ids, mask = prompts(3, seed=5, open_span=False)
    P = ids.shape[1]
    stops = [BOS, 70]                                                    # 70 ('F') ends some rows early
    gk = dict(do_sample=True, temperature=1.3, top_k=40, top_p=0.95) if sample else dict(do_sample=False)
    N = 41
    torch.manual_seed(11)
    ref = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=N, eos_token_id=stops,
                         pad_token_id=BOS, **gk)[:, P:]
    sp = D.SpanTracker(CLOSE, P, prompt_open=True)
    procs, _ = D.build_processors(prompt_len=P, batch=3, span=sp, finished=(stops, BOS), stats=True)
    cue = [CLOSE] + list(b"\n\nThe final answer is \\boxed{")
    seen = []
    torch.manual_seed(11)
    so = D.generate_segmented(model, ids, mask, N, 8, gk, procs, stops, BOS, span=sp,
                              probe=lambda c, s, m: D.forked_probe(model, _BT, c, s, m, cue, 6, stops, BOS),
                              on_probe=seen.append)
    return ref, so, seen, (ids, mask, cue, stops)



@pytest.mark.parametrize("sample", [True, False])
def test_segmented_generation_with_forked_probe_reproduces_generate_qwen3(sample):
    model = tiny("qwen3")
    ref, so, seen, (ids, mask, cue, stops) = _segmented_equals_generate(model, sample)
    n = min(ref.shape[1], so.new.shape[1])
    assert ref.shape[1] == so.new.shape[1] and torch.equal(ref[:, :n], so.new[:, :n])
    assert len(seen) >= 3 and so.calls >= 4
    # the forked probe equals a re-prefill probe of the same prefix (cache reuse changes nothing)
    rec = seen[1]
    t = rec["t"]
    pref = [ids[b][mask[b] > 0].tolist() + so.new[b, :t].tolist() for b in range(3)]
    re = D.probe_prefixes(model, _BT, pref, cue, 6, stops, BOS, 8, torch.device("cpu"))
    for b in range(3):
        if rec["open"][b]:
            assert re[b].answer == rec["results"][b].answer
            assert re[b].conf == pytest.approx(rec["results"][b].conf, rel=1e-4, abs=1e-6)


def test_segmented_generation_with_forked_probe_reproduces_generate_qwen35(torch_gdn):
    model = tiny("qwen35")
    for sample in (True, False):
        ref, so, seen, _ = _segmented_equals_generate(model, sample)
        assert ref.shape == so.new.shape and torch.equal(ref, so.new)
        assert len(seen) >= 3


def test_cache_fork_restores_every_layer(torch_gdn):
    for arch in ("qwen3", "qwen35"):
        model = tiny(arch)
        ids, mask = prompts(2, seed=4)
        pos = (mask.cumsum(1) - 1).clamp_min(0)
        o = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        cache = o.past_key_values
        snap = {i: {k: v.clone() for k, v in vars(L).items() if isinstance(v, torch.Tensor)}
                for i, L in enumerate(cache.layers)}
        fork = D.CacheFork(cache)
        ext = torch.cat([ids, torch.full((2, 3), 65)], 1)
        m2 = torch.cat([mask, torch.ones(2, 3, dtype=mask.dtype)], 1)
        model.generate(input_ids=ext, attention_mask=m2, past_key_values=cache, max_new_tokens=4, do_sample=False,
                       pad_token_id=BOS, eos_token_id=None)
        cache = fork.restore(cache)
        for i, L in enumerate(cache.layers):
            now = {k: v for k, v in vars(L).items() if isinstance(v, torch.Tensor)}
            assert set(now) == set(snap[i]), (arch, i)
            for k, v in snap[i].items():
                assert torch.equal(now[k], v), (arch, i, k)


def test_token_stats_entropy_and_logp():
    model = tiny()
    ids, mask = prompts(2, seed=6)
    P = ids.shape[1]
    st = D.TokenStats(close_id=CLOSE)
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=5, do_sample=False,
                         logits_processor=LogitsProcessorList([st]), pad_token_id=BOS, eos_token_id=None)
    st.finish(out[:, -1])
    arr = st.arrays()
    m = torch.cat([mask, torch.ones(2, 5, dtype=mask.dtype)], 1)
    full = model(input_ids=out, attention_mask=m, position_ids=(m.cumsum(1) - 1).clamp_min(0)).logits.float()
    lp = torch.log_softmax(full[:, P - 1:-1], -1)                                # predicting generated tokens
    ent = -(lp.exp() * lp).sum(-1)
    tok_lp = lp.gather(2, out[:, P:, None])[..., 0]
    assert torch.allclose(torch.from_numpy(arr["ent"]), ent, atol=1e-4)
    assert torch.allclose(torch.from_numpy(arr["logp"]), tok_lp, atol=1e-4)
    assert torch.allclose(torch.from_numpy(arr["p_close"]), lp[..., CLOSE].exp(), atol=1e-5)
