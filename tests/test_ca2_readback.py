"""CPU tests for experiments/ca2/readback.py (attention read-back) and readback_screen.py (the M2 + M3a runner).

Tiny random RecurrentGemma (6 layers, attention at 2 and 5, window 8, MQA, partial rotary), float32.
"""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from transformers import DynamicCache, RecurrentGemmaConfig, RecurrentGemmaForCausalLM  # noqa: E402
from transformers.models.recurrent_gemma.modeling_recurrent_gemma import apply_rotary_pos_emb  # noqa: E402

from ca2 import readback as rbm  # noqa: E402
from ca2.readback import Readback, ReadbackConfig, rope_shift  # noqa: E402

W = 8
B, T = 2, 40
ORACLE = [[(5, 2)], [(9, 2)]]


def tiny(window=W, seed=0):
    torch.manual_seed(seed)
    cfg = RecurrentGemmaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                               num_attention_heads=4, num_key_value_heads=1, head_dim=16, lru_width=64,
                               attention_window_size=window, partial_rotary_factor=0.5)
    cfg._attn_implementation = "sdpa"
    return RecurrentGemmaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def models():
    m = tiny()
    cf = copy.deepcopy(m.config)
    cf.attention_window_size = 10 ** 6
    mf = RecurrentGemmaForCausalLM(cf).eval()
    mf.load_state_dict(m.state_dict())
    torch.manual_seed(1)
    ids = torch.randint(3, 97, (B, T))
    return m, mf, ids


def attn_modules(model):
    return [m for m in model.modules() if hasattr(m, "rotary_emb") and hasattr(m, "layer_idx")]


def stepped(model, ids, n0):
    """Prefill ids[:, :n0] in one call, then one token per call through the KV cache; all logits."""
    model._setup_cache(model.config, ids.shape[0], "cpu", torch.float32)
    c = DynamicCache(config=model.config)
    pos = lambda a, b: torch.arange(a, b)[None].expand(ids.shape[0], -1)  # noqa: E731
    out = [model(ids[:, :n0], position_ids=pos(0, n0), past_key_values=c, use_cache=True).logits]
    for t in range(n0, ids.shape[1]):
        out.append(model(ids[:, t:t + 1], position_ids=pos(t, t + 1), past_key_values=c, use_cache=True).logits)
    return torch.cat(out, 1)


def run(m, ids, cfg, oracle=ORACLE, n0=None):
    rb = Readback(m, cfg)
    with torch.no_grad(), rb.active():
        rb.begin(ids.shape[0], oracle=oracle, rand_seeds=[3, 4])
        out = m(ids, use_cache=False).logits if n0 is None else stepped(m, ids, n0)
    return out, rb


# ------------------------------------------------------------------------------------------------- exactness
@pytest.mark.parametrize("sink", [0, 3])
@pytest.mark.parametrize("n0", [None, 13])
def test_all_true_equals_full_attention(models, sink, n0):
    """Every out-of-window position read back at its true position = full causal attention (TF and cached)."""
    m, mf, ids = models
    with torch.no_grad():
        full = mf(ids, use_cache=False).logits
        nat = m(ids, use_cache=False).logits
    assert (nat - full).abs().max() > 1e-2                        # the window matters on this sequence
    out, rb = run(m, ids, ReadbackConfig(selector="all", time="true", sink=sink, chunk=5), n0=n0)
    assert torch.allclose(out, full, atol=1e-5), (out - full).abs().max()
    assert rb.stats["max_cols"] == T


def test_all_layer_output_equals_full_attention_layer(models):
    """The attention layer's own output (after o_proj) equals full attention at every attention layer."""
    m, mf, ids = models
    outs = {}

    def grab(tag):
        def hook(mod, args, out):
            outs[(tag, mod.layer_idx)] = out[0].detach().clone()
        return hook
    hs = [a.register_forward_hook(grab("full")) for a in attn_modules(mf)]
    with torch.no_grad():
        mf(ids, use_cache=False)
    for h in hs:
        h.remove()
    rb = Readback(m, ReadbackConfig(selector="all", time="true", sink=2, chunk=7))
    hs = [a.register_forward_hook(grab("rb")) for a in rb.attn]
    with torch.no_grad(), rb.active():
        rb.begin(B)
        m(ids, use_cache=False)
    for h in hs:
        h.remove()
    layers = sorted(li for tag, li in outs if tag == "full")
    assert layers == [2, 5]
    for li in layers:
        assert torch.allclose(outs[("rb", li)], outs[("full", li)], atol=1e-5)


@pytest.mark.parametrize("n0", [None, 13])
def test_window_equals_native(models, n0):
    m, _, ids = models
    with torch.no_grad():
        nat = m(ids, use_cache=False).logits
    out, rb = run(m, ids, ReadbackConfig(selector="window", chunk=5), n0=n0)
    assert torch.allclose(out, nat, atol=1e-5)
    assert rb.stats["max_cols"] == W
    assert m.config._attn_implementation == "sdpa"               # restored on exit
    with torch.no_grad():
        assert torch.equal(m(ids, use_cache=False).logits, nat)


def test_short_sequence_readback_changes_nothing(models):
    """Inside the window (T ≤ recent band) every budgeted arm equals the native model."""
    m = tiny(window=16)
    ids = models[2][:, :10]
    with torch.no_grad():
        nat = m(ids, use_cache=False).logits
    for sel, tm in [("stream", "ordinal"), ("oracle", "ordinal"), ("oracle", "true"), ("oracle", "none"),
                    ("infllm", "ordinal"), ("random", "ordinal"), ("foreign", "ordinal")]:
        out, _ = run(m, ids, ReadbackConfig(selector=sel, time=tm, sink=1, budget=4, block=2, n_rep=1, chunk=3))
        assert torch.allclose(out, nat, atol=1e-5), (sel, tm)


# --------------------------------------------------------------------------------------------------- rotary
def test_rope_shift_matches_module_and_inverts():
    m = tiny()
    attn = attn_modules(m)[0]
    torch.manual_seed(2)
    k = torch.randn(1, 1, 12, 16)
    pos = torch.arange(100, 112)[None]
    cos, sin = attn.rotary_emb(k, pos)
    _, k_mod = apply_rotary_pos_emb(k, k, cos, sin)
    k_ours = rope_shift(k, pos[0], attn.rotary_emb.inv_freq)
    assert torch.allclose(k_ours, k_mod, atol=1e-5)
    assert torch.allclose(rope_shift(k_ours, -pos[0], attn.rotary_emb.inv_freq), k, atol=1e-5)
    assert torch.equal(k_ours[..., 8:], k[..., 8:])               # partial rotary: the second half is untouched


def test_relative_score_identity():
    """q̃ · R(−δ)k̃ = R(t)q̃ · R(t − δ)k̃ for every t (the relative time codes are exact per row)."""
    m = tiny()
    inv = attn_modules(m)[0].rotary_emb.inv_freq
    torch.manual_seed(3)
    q, k = torch.randn(16), torch.randn(16)
    for t, delta in [(50, 7), (3000, 2040), (9, 0)]:
        a = (q * rope_shift(k[None], torch.tensor([-delta]), inv)[0]).sum()
        qt = rope_shift(q[None], torch.tensor([t]), inv)[0]
        b = (qt * rope_shift(k[None], torch.tensor([t - delta]), inv)[0]).sum()
        assert torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("time_code", ["ordinal", "none", "true"])
def test_time_codes_equal_explicit_positions(models, time_code):
    """At a decode step, the layer output equals attention recomputed from pre-RoPE q/k placed at the positions the
    time code assigns (the module's own rotary), sink + read-back + recent."""
    m, _, ids = models
    attn = attn_modules(m)[0]                  # layer 2: its inputs do not depend on the arm
    caught = {"q": [], "k": [], "v": [], "o": []}
    hs = [attn.q_proj.register_forward_hook(lambda mod, a, o: caught["q"].append(o)),
          attn.k_proj.register_forward_hook(lambda mod, a, o: caught["k"].append(o)),
          attn.v_proj.register_forward_hook(lambda mod, a, o: caught["v"].append(o)),
          attn.register_forward_hook(lambda mod, a, o: caught["o"].append(o[0]))]
    s, R = 2, 4
    oracle = [[(4, 2), (12, 2)], [(6, 3)]]
    cfg = ReadbackConfig(selector="oracle", time=time_code, sink=s, budget=R, chunk=5)
    _, rb = run(m, ids, cfg, oracle=oracle, n0=30)
    for h in hs:
        h.remove()
    D = W - s - R
    t = T - 1
    d = 16
    q = torch.cat(caught["q"], 1).view(B, T, 4, d)[:, t]          # (B, H, d) pre-RoPE
    k = torch.cat(caught["k"], 1).view(B, T, d)
    v = torch.cat(caught["v"], 1).view(B, T, d)
    o_rb = caught["o"][-1][:, 0]                                  # (B, hidden)
    for b in range(B):
        sel = [p for a, n in oracle[b] for p in range(a, a + n)]
        n = len(sel)
        if time_code == "ordinal":
            vpos = [t - (D + n - 1 - i) for i in range(n)]
            n_slots = n
        elif time_code == "none":
            lmax = max(nn for _, nn in oracle[b])
            vpos = [t - (D + lmax - 1 - j) for a, nn in oracle[b] for j in range(nn)]
            n_slots = lmax
        else:
            vpos = sel
            n_slots = n                                           # the sink sits as under ordinal
        sink_pos = [t - (D + n_slots + s - 1 - i) for i in range(s)]                 # StreamingLLM, every code
        rec = list(range(t - D + 1, t + 1))
        cols = list(range(s)) + sel + rec
        kpos = sink_pos + vpos + rec
        kk = k[b, cols][None, None]
        cos, sin = attn.rotary_emb(kk, torch.tensor([kpos]))
        _, kr = apply_rotary_pos_emb(kk, kk, cos, sin)
        qq = q[b][None, :, None]                                  # (1, H, 1, d)
        cq, sq = attn.rotary_emb(qq, torch.tensor([[t]]))
        qr, _ = apply_rotary_pos_emb(qq, qq, cq, sq)
        p = torch.softmax((qr @ kr.transpose(-1, -2)) * d ** -0.5, -1)
        o = (p @ v[b, cols][None, None]).transpose(1, 2).reshape(1, -1)
        assert torch.allclose(attn.o_proj(o)[0], o_rb[b], atol=1e-5), (time_code, b)
    assert rb.last_selected[2][0][0].tolist() == [4, 5, 12, 13]


@pytest.mark.parametrize("n0", [None, 13])
def test_sink_placement_does_not_depend_on_the_time_code(n0):
    """With nothing read back (empty oracle) oracle:true, oracle:ordinal and stream are the same computation: the
    sink sits at StreamingLLM cache positions under every time code, never at an out-of-range true distance."""
    Wl = 16
    m = tiny(window=Wl)
    torch.manual_seed(5)
    ids = torch.randint(3, 97, (B, 80))
    kw = dict(sink=2, budget=4, block=4, chunk=8, sel_rows=4)
    out = {tm: run(m, ids, ReadbackConfig(selector="oracle", time=tm, **kw), oracle=[[], []], n0=n0)[0]
           for tm in ("ordinal", "true", "none")}
    st, _ = run(m, ids, ReadbackConfig(selector="stream", time="true", sink=2, chunk=8), n0=n0)
    st_o, _ = run(m, ids, ReadbackConfig(selector="stream", time="ordinal", sink=2, chunk=8), n0=n0)
    assert torch.allclose(out["true"], out["ordinal"], atol=1e-5), (out["true"] - out["ordinal"]).abs().max()
    assert torch.allclose(out["none"], out["ordinal"], atol=1e-5)
    assert torch.allclose(st, st_o, atol=1e-5)
    with torch.no_grad():
        nat = m(ids, use_cache=False).logits
    assert (out["true"][:, 2 * Wl:] - nat[:, 2 * Wl:]).abs().max() > 1e-3   # the sink does act past the window


# ------------------------------------------------------------------------------------------ budget, selection
SEL = [("stream", "ordinal"), ("oracle", "ordinal"), ("oracle", "true"), ("oracle", "none"), ("infllm", "ordinal"),
       ("infllm", "true"), ("random", "ordinal"), ("foreign", "ordinal")]


@pytest.mark.parametrize("sel,tm", SEL)
@pytest.mark.parametrize("n0", [None, 13])
def test_budget_and_eligibility(models, sel, tm, n0):
    m, _, ids = models
    cfg = ReadbackConfig(selector=sel, time=tm, sink=1, budget=4, block=2, n_rep=1, chunk=5, sel_rows=3)
    _, rb = run(m, ids, cfg, n0=n0)
    assert rb.stats["max_cols"] <= W
    D = rb.recent
    assert D == W - 1 - (0 if sel == "stream" else 4)
    for li, (pos, src) in rb.last_selected.items():
        assert int((pos >= 0).sum(1).max()) <= rb.budget       # per row
        p = pos[pos >= 0]
        if p.numel():
            assert int(p.min()) >= rb.sink and int(p.max()) <= T - 1 - D
    if sel != "stream":
        assert rb.selected_mean() and rb.selected_mean() > 0


def test_ineligible_position_is_refused(models):
    m, _, ids = models
    with pytest.raises(AssertionError, match="ineligible"):
        rb = Readback(m, ReadbackConfig(selector="oracle", sink=1, budget=4, chunk=5))
        rb.recent = 1                                              # a smaller band: the oracle is now ineligible
        rb._row_segments = lambda *a, **k: ([[(T - 2, 1)]] * B, [0, 1])
        with torch.no_grad(), rb.active():
            rb.begin(B, oracle=ORACLE)
            m(ids, use_cache=False)


def test_budget_assertion_fires(models):
    m, _, ids = models
    rb = Readback(m, ReadbackConfig(selector="oracle", sink=1, budget=2, chunk=5))
    rb._row_segments = lambda li, st, t_ref, *a: ([[(1, 4)] if t_ref - rb.recent >= 4 else []] * B, [0, 1])
    with pytest.raises(AssertionError, match="exceed the window budget"):
        with torch.no_grad(), rb.active():
            rb.begin(B, oracle=ORACLE)
            m(ids, use_cache=False)


@pytest.mark.parametrize("sel", ["stream", "oracle", "random", "foreign"])
def test_teacher_forced_equals_cached_for_fixed_selection(models, sel):
    """Selection that does not read queries: a one-row-chunk teacher-forced pass equals KV-cached decoding."""
    m, _, ids = models
    cfg = ReadbackConfig(selector=sel, sink=1, budget=4, block=2, chunk=1)
    a, _ = run(m, ids, cfg)
    b, _ = run(m, ids, cfg, n0=11)
    assert torch.allclose(a, b, atol=1e-5)


def test_foreign_reads_the_next_row(models):
    m, _, ids = models
    _, rb = run(m, ids, ReadbackConfig(selector="foreign", sink=1, budget=4, chunk=5))
    pos, src = rb.last_selected[5]
    assert src.tolist() == [1, 0]
    assert pos[0][pos[0] >= 0].tolist() == [9, 10] and pos[1][pos[1] >= 0].tolist() == [5, 6]
    with pytest.raises(ValueError, match="batch"):
        Readback(m, ReadbackConfig(selector="foreign", sink=1, budget=4)).begin(1, oracle=[[(5, 2)]])


def test_infllm_ranking_by_hand():
    """Block score = mean_h Σ_q max_j q·r_kj / √d; the top ⌊R/b⌋ eligible blocks, in position order."""
    m = tiny(window=8)
    rb = Readback(m, ReadbackConfig(selector="infllm", sink=1, budget=4, block=2, n_rep=2))
    d = 16
    st = rbm._Store(k=torch.zeros(1, 1, 64, d), n=64)
    e = torch.eye(d)
    # block k's representatives: block 0 → e0, e1; block 1 → e2, 3·e3; block 2 → e4, e5; block 3 → 2·e0, e6
    st.reps = [torch.stack([e[0], e[1]])[None], torch.stack([e[2], 3 * e[3]])[None],
               torch.stack([e[4], e[5]])[None], torch.stack([2 * e[0], e[6]])[None]]
    q = torch.zeros(1, 2, 2, d)                                   # (B, H, rows, d)
    q[0, 0, 0] = e[0] + e[3]                                      # head 0: block 3 (2) + block 1 (3) ...
    q[0, 1, 1] = e[4]
    # hand: block0: h0 max(1,0)=1, row1 0 → 1; h1 row1: 0 → mean 0.5; block1: h0 3 → 1.5; block2: h1 1 → 0.5;
    # block3: h0 2 → 1.0. Top 2 → blocks 1 and 3.
    t_ref = rb.recent + rb.sink + 4 * 2 - 1                       # all 4 blocks eligible
    segs, src = rb._row_segments(2, st, t_ref, q, None)
    assert segs == [[(1 + 1 * 2, 2), (1 + 3 * 2, 2)]]
    segs, _ = rb._row_segments(2, st, t_ref - 2, q, None)         # block 3 not yet eligible → blocks 1 and 0 (tie 0/2)
    assert (1 + 1 * 2, 2) in segs[0] and (1 + 3 * 2, 2) not in segs[0]


def test_representatives_are_the_most_attended_keys():
    m = tiny(window=16)
    rb = Readback(m, ReadbackConfig(selector="infllm", sink=1, budget=8, block=4, n_rep=2))
    inv = rb.attn[0].rotary_emb.inv_freq
    torch.manual_seed(5)
    d, H, n = 16, 4, 13
    k_un = torch.randn(1, n, d)
    pos = torch.arange(n)
    st = rbm._Store()
    st.append(rope_shift(k_un, pos, inv)[:, None], torch.zeros(1, 1, n, d))
    qu = torch.randn(1, H, n, d)
    rb._update_reps(st, qu[:, :, :6], 0, inv)
    rb._update_reps(st, qu[:, :, 6:], 6, inv)
    assert len(st.reps) == 3                                      # blocks at 1–4, 5–8, 9–12
    for kb, rep in enumerate(st.reps):
        a = 1 + 4 * kb
        sc = torch.einsum("hqd,kd->hqk", qu[0, :, a:a + 4], k_un[0, a:a + 4]) / d ** 0.5
        sc = sc.masked_fill(~torch.ones(4, 4, dtype=torch.bool).tril(), float("-inf"))
        mass = torch.softmax(sc, -1).sum((0, 1))
        want = k_un[0, a + mass.topk(2).indices]
        assert torch.allclose(rep[0], want, atol=1e-5)


def test_config_validation():
    with pytest.raises(ValueError):
        ReadbackConfig(selector="all", time="ordinal")
    with pytest.raises(ValueError):
        ReadbackConfig(selector="nope")
    with pytest.raises(ValueError):
        ReadbackConfig(selector="infllm", budget=64, block=128)
    assert ReadbackConfig(selector="window").sink == 0 and ReadbackConfig(selector="stream").budget == 0
    m = tiny(window=8)
    with pytest.raises(ValueError, match="no recent"):
        Readback(m, ReadbackConfig(selector="oracle", sink=4, budget=4))


# -------------------------------------------------------------------------------------------------- runner
@pytest.fixture()
def filler(tmp_path):
    words = ("the river ran past an old mill where nothing much happened for several long years and then "
             "a small town grew around it with markets churches and a railway station ").split()
    p = tmp_path / "filler.txt"
    p.write_text(" ".join(words * 400))
    return p


def test_items_layout(filler):
    from ca2 import readback_screen as scr
    args = scr.build_parser().parse_args(["--tiny", "--filler", str(filler)])
    args.seq_len, args.tail, args.inside_len, args.budget, args.sink, args.tail_steps, args.pairs = \
        1024, 450, 180, 128, 2, 40, 2
    import pretrained_loop as pl
    tok = pl.ByteTokenizer()
    rend = scr.Renderer(tok, chat=False, prefill=True)
    corpus = tok.encode(filler.read_text())
    Wt = 384
    for task in ("needle", "update"):
        for cell in ("beyond", "inside"):
            items = scr.build_items(tok, corpus, rend, task, cell, 5, args, Wt, Wt - args.sink - args.budget)
            again = scr.build_items(tok, corpus, rend, task, cell, 5, args, Wt, Wt - args.sink - args.budget)
            assert [it.ids for it in items] == [it.ids for it in again]               # deterministic
            for it in items:
                assert len(it.ids) == args.seq_len
                assert tok.decode([it.ids[p] for p in it.value_pos]) == it.answer
                assert tok.decode(it.ids[-args.tail_steps:]).rstrip().endswith(" is")      # the question is stepped
                assert sum(n for _, n in it.oracle) <= args.budget
                covered = {p for a, n in it.oracle for p in range(a, a + n)}
                assert set(it.value_pos) <= covered
                if cell == "beyond":
                    assert (args.seq_len - args.tail_steps) - it.fact_last > Wt
                else:
                    assert (args.seq_len - 1 + args.max_new) - it.fact_first < Wt - args.sink - args.budget
                if task == "update":
                    assert it.stale and it.stale != it.answer
    # 32 columns per statement, centred: (90, 121) and (107, 138) overlap and merge
    assert scr.oracle_segments([(100, 110), (120, 125)], 64, 2) == [(90, 49)]
    assert scr.oracle_segments([(100, 110), (300, 305)], 64, 2) == [(90, 32), (287, 32)]


def test_runner_tiny_end_to_end(filler, tmp_path):
    from ca2 import readback_screen as scr
    out = tmp_path / "rb.json"
    rep = scr.main(["--tiny", "--device", "cpu", "--items", "2", "--inside-items", "2", "--batch", "2",
                    "--arms", "native,window,stream,oracle:ordinal,foreign,infllm", "--filler", str(filler),
                    "--out", str(out), "--trace-dir", str(tmp_path / "traces")])
    saved = json.loads(out.read_text())
    assert all("text" not in r for res in saved["arms"].values() for r in res["per_item"])   # text: traces only
    assert set(saved["arms"]) == {"native", "window", "stream", "oracle:ordinal", "foreign", "infllm"}
    for arm, res in saved["arms"].items():
        assert res["complete"] and res["memory"]["columns_per_query"] == 384
        assert {"needle/beyond", "update/beyond", "needle/inside", "pooled/beyond"} <= set(res["summary"])
    nat = {r["uid"]: r["em"] for r in saved["arms"]["native"]["per_item"]}
    win = {r["uid"]: r["em"] for r in saved["arms"]["window"]["per_item"]}
    assert nat == win                                             # float32: the implementation control agrees
    assert saved["arms"]["oracle:ordinal"]["summary"]["needle/beyond"]["hit"] == 1.0
    assert "gate" in saved["comparisons"] and "pooled" in saved["comparisons"]["paired_beyond"]
    assert (tmp_path / "traces" / "native.jsonl").exists()
    assert rep["arms"]["native"]["complete"]
    with pytest.raises(SystemExit):                              # other settings into the same file: refused
        scr.main(["--tiny", "--device", "cpu", "--items", "3", "--inside-items", "0", "--arms", "native",
                  "--filler", str(filler), "--out", str(out), "--trace-dir", str(tmp_path / "traces")])
