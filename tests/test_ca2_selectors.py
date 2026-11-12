"""CPU tests for experiments/ca2/selectors13.py: the shared span builder, the step-0 scorers on planted targets, the
repaired InfLLM's norm invariance, attention received against the model's own attention, and the retriever."""

import math
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import pretrained_loop as pl  # noqa: E402
from ca2 import selectors13 as S  # noqa: E402
from ca2.circuit import DentateGate, EpisodicStore, context_tag  # noqa: E402

TOK = pl.ByteTokenizer()
TB = S.TokenTables(TOK)


# ------------------------------------------------------------------------------------------------ span builder
def test_cut_spans_sentences_cap_and_decimals():
    text = " The mill stood 3 @.@ 5 km away . He said \"no .\" The U.S. army came ." + " word" * 30 + " ."
    ids = TOK.encode(text)
    sp = S.cut_spans(ids, TB, 0, len(ids), cap=48)
    pieces = [TOK.decode(ids[a:b]) for a, b in sp]
    assert pieces[0] == " The mill stood 3 @.@ 5 km away ."           # "@.@" does not cut
    assert pieces[1] == " He said \"no .\""
    assert all(b - a <= 48 for a, b in sp)
    assert sp[0][0] == 0 and sp[-1][1] == len(ids)
    assert all(sp[i][1] == sp[i + 1][0] for i in range(len(sp) - 1))  # contiguous cover
    nl = TOK.encode("one two\nthree four")
    assert [TOK.decode(nl[a:b]) for a, b in S.cut_spans(nl, TB, 0, len(nl))] == ["one two\n", "three four"]


def test_span_scores_top3_and_budget_prefix():
    spans = [(0, 5), (5, 7), (7, 20)]
    v = torch.tensor([1., 9, 2, 3, 0, 4, 5, 0, 0, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 0])
    s = S.span_scores(v, spans, cap=48)
    assert torch.allclose(s, torch.tensor([(9 + 3 + 2) / 3, 4.5, 7.0]))
    v2 = v.clone()
    v2[5:7] = float("-inf")
    assert S.span_scores(v2, spans, cap=48)[1] == float("-inf")
    od = S.order(s, torch.arange(3))
    assert od.tolist() == [2, 0, 1]
    assert S.take(od, [5, 2, 13], budget=17) == [2]                   # the strict prefix stops at the first misfit
    assert S.take(od, [5, 2, 13], budget=18) == [2, 0]
    r = S.evaluate(s, spans, [0, 1], budget=18, tiebreak=torch.arange(3))
    assert (r[0].rank, r[0].in_budget, r[1].rank, r[1].in_budget) == (2, True, 3, False)
    tie = S.order(torch.zeros(3), torch.tensor([2, 0, 1]))            # ties: the fixed random order
    assert tie.tolist() == [1, 2, 0]


def test_budget_holds_at_most_the_budget():
    torch.manual_seed(0)
    spans, a = [], 0
    while a < 3000:
        n = int(torch.randint(5, 49, ()))
        spans.append((a, a + n))
        a += n
    s = torch.rand(len(spans))
    sel = S.take(S.order(s, torch.randperm(len(spans))), [b - a for a, b in spans], 512)
    assert sum(spans[j][1] - spans[j][0] for j in sel) <= 512 and len(sel) >= 512 // 48


def test_percentile_and_salience_product():
    m = torch.tensor([True, True, True, False])
    p = S.percentile(torch.tensor([3., 1, 2, 10]), m)
    assert torch.allclose(p[:3], torch.tensor([1., 1 / 3, 2 / 3])) and p[3] == float("-inf")
    w = S.with_salience(torch.tensor([3., 1, 2, 0]), torch.tensor([1., 3, 2, 0]), m)
    assert torch.allclose(w[:3], torch.tensor([1 / 3, 1 / 3, 4 / 9]))


# ------------------------------------------------------------------------------------------------- scorers
def planted_qk(T=80, H=3, R=5, d=16, target=33, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = [torch.randn(T, d, generator=g) * 0.3 for _ in range(2)]
    q = [torch.randn(H, R, d, generator=g) * 0.3 for _ in range(2)]
    for li in range(2):
        k[li][target] = q[li].mean((0, 1)) * 12
    return q, k


@pytest.mark.parametrize("name", list(S.QK_VARIANTS))
def test_qk_family_ranks_a_planted_key_first(name):
    q, k = planted_qk()
    inv = 1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 8))
    cand = torch.ones(80, dtype=torch.bool)
    cand[:2] = False
    s = S.qk_scores(q, k, torch.arange(75, 80), cand, inv, S.QK_VARIANTS[name], ord_delta=40)
    assert int(s.argmax()) == 33 and s[0] == float("-inf")


def test_rope_matches_readback_rope_shift():
    from ca2.readback import rope_shift
    inv = 1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 8))
    x = torch.randn(2, 7, 16)
    pos = torch.arange(100, 107)
    assert torch.allclose(S.rope(x, pos, inv), rope_shift(x, pos, inv), atol=1e-6)


def test_infllm_invariant_to_key_norms_and_finds_target():
    q, k = planted_qk()
    recv = torch.rand(80)
    spans = [(a, min(80, a + 10)) for a in range(0, 80, 10)]
    recv[33] = 5.0                                                 # the planted key is a representative
    s = S.infllm_scores(q, k, recv, spans, n_rep=3)
    assert int(s.argmax()) == 33
    assert int(torch.isfinite(s).sum()) == 3 * len(spans)
    scale = torch.rand(80) * 10 + 0.1
    s2 = S.infllm_scores(q, [kk * scale[:, None] for kk in k], recv, spans, n_rep=3)
    fin = torch.isfinite(s)
    assert torch.equal(fin, torch.isfinite(s2)) and torch.allclose(s[fin], s2[fin], atol=1e-5)
    ss = S.span_scores(s, spans)
    assert int(ss.argmax()) == 3


def test_attention_received_matches_brute_force():
    torch.manual_seed(0)
    H, T, d, W = 2, 30, 8, 7
    q, k = torch.randn(H, T, d), torch.randn(T, d)
    inv = 1.0 / (10000 ** (torch.arange(0, 4, 2).float() / 4))
    got = S.attention_received(q, k, W, inv, chunk=4)
    qr, kr = S.rope(q, torch.arange(T), inv), S.rope(k, torch.arange(T), inv)
    sc = torch.einsum("htd,pd->htp", qr, kr) / math.sqrt(d)
    t, p = torch.arange(T)[:, None], torch.arange(T)[None]
    vis = (p <= t) & (p > t - W)
    pr = torch.softmax(sc.masked_fill(~vis, float("-inf")), -1).sum((0, 1))
    seen = vis.sum(0).float()
    assert torch.allclose(got, pr / seen, atol=1e-5)


def test_attention_received_matches_the_model_attention():
    """Through the tiny RecurrentGemma's own q/k projections: the eager attention weights' column sums."""
    from transformers import RecurrentGemmaConfig, RecurrentGemmaForCausalLM
    torch.manual_seed(0)
    cfg = RecurrentGemmaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                               num_attention_heads=4, num_key_value_heads=1, head_dim=16, lru_width=64,
                               attention_window_size=8, partial_rotary_factor=0.5)
    cfg._attn_implementation = "eager"
    m = RecurrentGemmaForCausalLM(cfg).eval()
    attn = m.model.layers[2].temporal_block
    got = {}
    hs = [attn.q_proj.register_forward_hook(lambda mod, a, o: got.__setitem__("q", o)),
          attn.k_proj.register_forward_hook(lambda mod, a, o: got.__setitem__("k", o)),
          attn.register_forward_hook(lambda mod, a, o: got.__setitem__("w", o[1]))]
    ids = torch.randint(3, 97, (1, 25))
    with torch.no_grad():
        m(ids, use_cache=False)
    for h in hs:
        h.remove()
    q = got["q"][0].view(25, 4, 16).permute(1, 0, 2)
    k = got["k"][0].view(25, 16)
    recv = S.attention_received(q, k, 8, attn.rotary_emb.inv_freq, attn.scaling)
    w = got["w"][0]                                                # (H, T, T)
    seen = (torch.clamp(torch.arange(25) + 8, max=25) - torch.arange(25)).float()
    assert torch.allclose(recv, w.sum((0, 1)) / seen, atol=1e-5)


def test_cos_and_dentate_rank_the_planted_word():
    torch.manual_seed(0)
    T, D = 60, 32
    early = torch.randn(T, D)
    early[60 - 3] = early[20] + 0.05 * torch.randn(D)               # a question row repeats position 20's word
    cand = torch.zeros(T, dtype=torch.bool)
    cand[:50] = True
    qr = torch.arange(55, 60)
    for agg in ("max",):
        s = S.cos_scores(early, qr, cand, early[:50].mean(0), agg)
        assert int(s.argmax()) == 20
    gate = DentateGate(D, seed=0)
    idx, val, _ = gate(early)
    s, key = S.dentate_scores(idx, val, qr[2:3], cand, gate.U)
    assert int(s.argmax()) == 20 and key.shape == (gate.U,)


def test_store_scores_and_references():
    torch.manual_seed(0)
    gate = DentateGate(16, seed=0)
    e = torch.randn(40, 16)
    idx, val, _ = gate(e)
    spans = [(a, a + 5) for a in range(0, 40, 5)]
    st = EpisodicStore.from_spans(spans, gate, idx, val, torch.randn(40, 5, 16), torch.randn(40, 16))
    key = gate.dense(idx[10:15], val[10:15]).mean(0)
    pos, sp = S.store_scores(st, key, context_tag(torch.randn(5, 16)), 0.0, spans, 40)
    assert int(sp.argmax()) == 2 and torch.equal(pos[10:15], sp[2].expand(5))
    o = S.oracle_scores(spans, [5], 40)
    assert int(S.span_scores(o, spans).argmax()) == 5
    r1, r2 = S.random_scores(spans, 40, 7), S.random_scores(spans, 40, 7)
    assert torch.equal(r1, r2) and not torch.equal(r1, S.random_scores(spans, 40, 8))


def test_retriever_with_a_stand_in():
    def emb(texts, query):
        voc = ["mill", "code", "alice", "river", "bob"]
        return torch.tensor([[float(w in t.lower()) for w in voc] for t in texts]) + 1e-3
    r = S.Retriever("stand-in", embed_fn=emb)
    s = r.score("What is the code for Alice?", ["The river ran.", "The code for Alice is 1234.", "Bob's mill."])
    assert int(s.argmax()) == 1


def _cached(name):
    root = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    return (root / ("models--" + name.replace("/", "--"))).exists()


@pytest.mark.skipif(not _cached(S.BGE), reason="bge-small not in the HF cache")
def test_bge_retriever_offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    r = S.Retriever(S.BGE)
    spans = [" The river ran past the old mill .", " Ruiz, the pharmacist, lives in Lisbon.",
             " Okafor, the pilot, lives in Osaka.", " The railway station opened in 1890 ."]
    s = r.score("In which city does the person who dispenses medicine live?", spans)
    assert int(s.argmax()) == 1


# ----------------------------------------------------------------------------------------- step 0b
def test_with_salience_keeps_masked_positions_at_minus_inf():
    m = torch.tensor([True, True, False, False])
    w = S.with_salience(torch.tensor([1., 2, 3, 4]), torch.tensor([4., 3, 2, 1]), m)
    assert torch.isfinite(w[:2]).all() and (w[2:] == float("-inf")).all()   # step 0 gave +inf here
    w2 = S.with_salience(torch.tensor([1., float("-inf"), 3, 4]), torch.tensor([4., 3, 2, 1]), m)
    assert w2[1] == float("-inf")


def test_rank_many_equals_evaluate():
    torch.manual_seed(0)
    spans, a = [], 0
    while len(spans) < 60:
        n = int(torch.randint(3, 49, ()))
        spans.append((a, a + n))
        a += n
    lengths = torch.tensor([b - a for a, b in spans])
    tb = torch.randperm(60)
    sc = torch.randn(5, 60)
    sc[1] = 0.0                                                     # all ties
    sc[2, :10] = float("-inf")
    sc[3, 5] = float("nan")
    rk, cb = S.rank_many(sc, lengths, [7, 30], tb)
    for i in range(5):
        r = S.evaluate(sc[i], spans, [7, 30], 512, tb)
        od = S.order(sc[i], tb)
        for j, t in enumerate([7, 30]):
            assert int(rk[i, j]) == r[j].rank
            pl = od.tolist().index(t)
            assert int(cb[i, j]) == int(lengths[od[:pl]].sum())
            assert (int(cb[i, j]) + int(lengths[t]) <= 512) == r[j].in_budget


def test_span_scores_batched_equals_single():
    spans = [(0, 5), (5, 7), (7, 20)]
    P = torch.randn(4, 20)
    P[1, 5:7] = float("-inf")
    SS = S.span_scores(P, spans)
    for i in range(4):
        assert torch.equal(SS[i], S.span_scores(P[i], spans))


def test_position_reference():
    cand = torch.tensor([False, True, True, False, True])
    p = S.position_scores(cand)
    assert p.tolist() == [float("-inf"), 1.0, 2.0, float("-inf"), 4.0]
