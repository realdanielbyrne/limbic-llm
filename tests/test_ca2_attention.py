"""Round 12 CA2 attention tap (experiments/ca2/attention.py): recomputed attention statistics against an eager
twin (prefill with left padding, KV-cached decode), Qwen3.5 full-attention blocks (partial rotary, gated q_proj),
HookedLM's window mask, strict causality, prefill + cached decode equal to one pass for the attention sources,
gain-0 byte identity with the tap active, and the sources' arithmetic.

Tiny random models with the byte tokenizer; no downloads. Plan test numbers are given in the section headers.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments" / "alignment")]

import pretrained_loop as pl  # noqa: E402
from ca2.attention import (TAP_KEYS, AttnAgreeSource, AttnTap, AttnWindowSource, add_attention,  # noqa: E402
                           attn_stats, hold, visibility)
from ca2.common import SteerSide, alignment  # noqa: E402
from ca2.controller import CA2Attachment, CA2Controller, NormTap  # noqa: E402
from ca2.sources import SOURCES, make_source  # noqa: E402

D, T, BOS, W = 64, 33, 256, 8
A = alignment()


def tiny_qwen3(dtype=torch.float32, attn="sdpa", seed=0):
    torch.manual_seed(seed)
    cfg = transformers.Qwen3Config(
        vocab_size=512, hidden_size=D, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, max_position_embeddings=1024,
        bos_token_id=BOS, eos_token_id=BOS, pad_token_id=BOS)
    model = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation=attn).to(dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def tokens(B=2, n=T, seed=0):
    ids = torch.randint(0, 256, (B, n), generator=torch.Generator().manual_seed(seed))
    ids[:, 0] = BOS
    return ids


def padded(B=2, n=T, pad=3, seed=0):
    ids = tokens(B, n, seed)
    mask = torch.ones(B, n, dtype=torch.long)
    ids[0, :pad], mask[0, :pad] = BOS, 0
    ids[0, pad] = BOS
    return ids, mask


def run_once(model, ctrl, ids, mask=None, **kw):
    ctrl.reset()
    ctrl.record = True
    ctrl.pending_mask = mask
    pos = A.probe.position_ids(mask) if mask is not None else None
    return model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, **kw)


def incremental(model, ctrl, ids, mask, P, **kw):
    if ctrl is not None:
        ctrl.reset()
        ctrl.record = True
        ctrl.pending_mask = mask[:, :P]
    pos = A.probe.position_ids(mask)
    out = model(input_ids=ids[:, :P], attention_mask=mask[:, :P], position_ids=pos[:, :P], use_cache=True, **kw)
    outs, pkv = [out], out.past_key_values
    for t in range(P, ids.shape[1]):
        o = model(input_ids=ids[:, t:t + 1], attention_mask=mask[:, :t + 1], position_ids=pos[:, t:t + 1],
                  past_key_values=pkv, use_cache=True, **kw)
        outs.append(o)
        pkv = o.past_key_values
    return outs


def same(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


# ------------------------------------------------------------------------------ reference from probabilities
def reference(P, key_real, Wr=W, nbins=16, sink=1, vis=None):
    """Statistics from the eager probabilities P (B, H, Tq, Tk), written as plain loops (the definitions)."""
    P = P.double()
    B, H, Tq, Tk = P.shape
    vis_m = None if vis is None else vis.expand(B, Tq, Tk)
    off = Tk - Tq
    ridx = key_real.long().cumsum(1) - 1
    nan = float("nan")
    out = {k: torch.zeros(B, Tq) for k in ("elig", "sink", "self", "n_elig", "qreal", "ent_pool", "ent", "band_mass")}
    for k in ("ent_elig", "mean_lag", "score_sd", "score_sink_z", "score_top_z"):
        out[k] = torch.full((B, Tq), nan)
    out["band"] = torch.zeros(B, Tq, Wr)
    out["band_ok"] = torch.zeros(B, Tq, Wr)
    out["score_band_z"] = torch.full((B, Tq, Wr), nan)
    out["lagbin"] = torch.zeros(B, Tq, nbins)
    out["head_elig"] = torch.zeros(B, Tq, H)
    out["head_sink"] = torch.zeros(B, Tq, H)
    out["head_band"] = torch.zeros(B, Tq, H)
    out["qband"] = torch.zeros(B, Tq, H, Wr)
    for b in range(B):
        for i in range(Tq):
            t = off + i
            if not key_real[b, t]:
                continue
            vis = [j for j in range(t + 1) if key_real[b, j] and (vis_m is None or vis_m[b, i, j])]
            el = [j for j in vis if j != t and ridx[b, j] >= sink]
            sk = [j for j in vis if ridx[b, j] < sink]
            pb = P[b, :, i].mean(0)
            out["qreal"][b, i] = 1
            e = float(pb[el].sum())
            out["elig"][b, i] = e
            out["sink"][b, i] = float(pb[sk].sum())
            out["self"][b, i] = float(pb[t]) if ridx[b, t] >= sink else 0.0
            out["n_elig"][b, i] = len(el)
            out["ent_pool"][b, i] = float(-(pb[vis] * pb[vis].log()).sum())
            out["ent"][b, i] = float(-(P[b, :, i, vis] * P[b, :, i, vis].log()).sum(-1).mean())
            for h in range(H):
                out["head_elig"][b, i, h] = float(P[b, h, i, el].sum())
                out["head_sink"][b, i, h] = float(P[b, h, i, sk].sum())
            if el:
                pt = pb[el] / e
                out["ent_elig"][b, i] = float(-(pt * pt.log()).sum())
                lags = torch.tensor([t - j for j in el], dtype=torch.float64)
                out["mean_lag"][b, i] = float((pb[el] * lags).sum() / e)
                for j in el:
                    lag = t - j
                    out["lagbin"][b, i, min(int(math.floor(math.log2(lag))), nbins - 1)] += float(pb[j])
                logp = P[b, :, i, el].log()                         # scores up to a per-row constant
                mu, sd = logp.mean(-1), logp.std(-1, unbiased=False)
                if len(el) >= 2:
                    out["score_sd"][b, i] = float(sd.mean())
                    out["score_top_z"][b, i] = float(((logp.max(-1).values - mu) / sd).mean())
                    if sk:
                        out["score_sink_z"][b, i] = float(((P[b, :, i, sk].log().mean(-1) - mu) / sd).mean())
                for l in range(1, Wr + 1):
                    j = t - l
                    if j in el:
                        out["band"][b, i, l - 1] = float(pb[j])
                        out["band_ok"][b, i, l - 1] = 1
                        out["head_band"][b, i] += P[b, :, i, j].float()
                        out["qband"][b, i, :, l - 1] = (P[b, :, i, j] / P[b, :, i, el].sum(-1)).float()
                        if len(el) >= 2:
                            out["score_band_z"][b, i, l - 1] = float(((P[b, :, i, j].log() - mu) / sd).mean())
            out["band_mass"][b, i] = out["band"][b, i].sum()
    return out


CHECK = ("elig", "sink", "self", "n_elig", "qreal", "ent_pool", "ent", "ent_elig", "mean_lag", "band", "band_ok",
         "band_mass", "lagbin", "head_elig", "head_sink", "head_band", "score_sd", "score_band_z", "score_sink_z",
         "score_top_z")


def compare(got, ref, keys=CHECK, atol=1e-5):
    for k in keys:
        torch.testing.assert_close(got[k].float().cpu(), ref[k].float(), atol=atol, rtol=1e-4, equal_nan=True,
                                   msg=lambda m, k=k: f"{k}: {m}")


def tapped(model, layer=1, sink_skip=1, **kw):
    ctrl = CA2Controller(D, "constant", g_min=0.0, g_max=0.0, sink_skip=sink_skip, **kw)
    tap = AttnTap(window=W)
    att = CA2Attachment(model, ctrl, tap=layer, taps=[tap])
    return ctrl, tap, att


# --------------------------------------------------------------- 7. statistics against the eager twin
def test_prefill_statistics_match_eager_twin_with_left_padding():
    model, eager = tiny_qwen3(), tiny_qwen3(attn="eager")
    ids, mask = padded()
    ctrl, tap, att = tapped(model)
    run_once(model, ctrl, ids, mask)
    got = dict(tap.cur)
    tr = ctrl.trace[0]
    att.remove()
    ref_out = eager(input_ids=ids, attention_mask=mask, position_ids=A.probe.position_ids(mask), use_cache=False,
                    output_attentions=True)
    ref = reference(ref_out.attentions[1], mask.bool())
    compare(got, ref)
    torch.testing.assert_close(got["qband"].cpu(), ref["qband"], atol=1e-5, rtol=1e-4)
    for k in TAP_KEYS:                                              # every statistic is in the trace
        assert f"tap/attn/{k}" in tr, k
    assert tr["tap/attn/qreal"][0, :3].sum() == 0 and tr["tap/attn/sink"][0, 3:].gt(0).all()
    # no mass on future keys: masses at each query sum to one over visible keys
    tot = got["elig"] + got["sink"] + got["self"]
    real = got["qreal"] > 0
    assert torch.allclose(tot[real], torch.ones(()), atol=1e-5)
    # the time-relevance profile partitions the eligible mass
    assert torch.allclose(got["lagbin"].sum(-1), got["elig"], atol=1e-6)


def test_decode_statistics_match_eager_twin_with_cached_keys():
    model, eager = tiny_qwen3(), tiny_qwen3(attn="eager")
    ids, mask = padded()
    P = 12
    ctrl, tap, att = tapped(model)
    incremental(model, ctrl, ids, mask, P)
    tt = ctrl.trace_tensors()
    run_once(model, ctrl, ids, mask)
    one = ctrl.trace_tensors()
    att.remove()
    outs = incremental(eager, None, ids, mask, P, output_attentions=True)
    for step, o in enumerate(outs):
        a = o.attentions[1]                                          # (B, H, Tq, Tk)
        Tk = a.shape[-1]
        ref = reference(a, mask[:, :Tk].bool())
        c0 = 0 if step == 0 else P + step - 1
        got = {k: tt[f"tap/attn/{k}"][:, c0:c0 + a.shape[2]] for k in CHECK}
        compare(got, ref)
    for k in CHECK:                                                  # prefill + decode equals one pass
        torch.testing.assert_close(tt[f"tap/attn/{k}"], one[f"tap/attn/{k}"], atol=1e-5, rtol=1e-4, equal_nan=True)


def test_visibility_forms():
    B, Tq, Tk = 2, 3, 5
    causal = visibility(None, B, Tq, Tk, "cpu")
    assert causal.shape == (B, Tq, Tk) and causal[0, 0].tolist() == [True, True, True, False, False]
    bm = causal[:, None].clone()
    bm[0, 0, :, 0] = False
    assert torch.equal(visibility(bm, B, Tq, Tk, "cpu"), bm[:, 0])
    add = torch.zeros(B, 1, Tq, Tk).masked_fill(~bm, torch.finfo(torch.float32).min)
    assert torch.equal(visibility(add, B, Tq, Tk, "cpu"), bm[:, 0])
    inf = torch.zeros(1, 1, Tq, Tk).masked_fill(~causal[:1, None], float("-inf"))
    assert torch.equal(visibility(inf, B, Tq, Tk, "cpu"), causal)
    full = torch.zeros(1, 1, Tq, Tk)                                 # an all-zero additive mask is still causal
    assert torch.equal(visibility(full, B, Tq, Tk, "cpu"), causal)
    with pytest.raises(ValueError):
        visibility(torch.zeros(1, 1, Tq, Tk + 1), B, Tq, Tk, "cpu")


# --------------------------------------------------------------------------- Qwen3.5 full-attention blocks
@pytest.fixture
def qwen35_pair(monkeypatch):
    """The tiny Qwen3.5 hybrid (GDN, attention, GDN, attention) in sdpa and eager, GDN kernels on pure torch."""
    if not hasattr(transformers, "Qwen3_5TextConfig"):
        pytest.skip("transformers has no Qwen3.5 classes")
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        fn = getattr(M, name, None)
        if fn is None:
            pytest.skip(f"{name} not found")
        monkeypatch.setattr(M, name, getattr(fn, "__wrapped__", fn))
    out = []
    for impl in ("sdpa", "eager"):
        model, _ = pl.build_tiny("qwen35", seed=0, attn_impl=impl)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        out.append(model)
    return out


def test_qwen35_attention_block_statistics_match_eager(qwen35_pair):
    model, eager = qwen35_pair
    blk = pl.find_blocks(model)[1]
    assert type(blk.self_attn).__name__ == "Qwen3_5Attention"
    rot = model.model.rotary_emb.inv_freq.numel() * 2
    assert rot < blk.self_attn.head_dim                              # partial rotary: the tap must keep the rest
    ids, mask = padded()
    P = 12
    ctrl, tap, att = tapped(model, layer=1)
    run_once(model, ctrl, ids, mask)
    got = dict(tap.cur)
    one = ctrl.trace_tensors()
    incremental(model, ctrl, ids, mask, P)
    inc = ctrl.trace_tensors()
    att.remove()
    ref_out = eager(input_ids=ids, attention_mask=mask, position_ids=A.probe.position_ids(mask), use_cache=False,
                    output_attentions=True)
    attn_layers = [a for a in ref_out.attentions if a is not None]
    compare(got, reference(attn_layers[0], mask.bool()))
    outs = incremental(eager, None, ids, mask, P, output_attentions=True)
    for step, o in enumerate(outs):
        a = [x for x in o.attentions if x is not None][0]
        c0 = 0 if step == 0 else P + step - 1
        got = {k: inc[f"tap/attn/{k}"][:, c0:c0 + a.shape[2]] for k in CHECK}
        compare(got, reference(a, mask[:, :a.shape[-1]].bool()))
    for k in CHECK:
        torch.testing.assert_close(inc[f"tap/attn/{k}"], one[f"tap/attn/{k}"], atol=1e-5, rtol=1e-4, equal_nan=True)
    with pytest.raises(ValueError):                                  # a GDN block has no attention weights
        CA2Attachment(model, CA2Controller(D), tap=2, relevance_layer=0, taps=[AttnTap()])


def test_qwen35_attention_sources_causal_and_zero_gain(qwen35_pair):
    model, _ = qwen35_pair
    ids = tokens()
    base = model(input_ids=ids, use_cache=False).logits
    ctrl = CA2Controller(D, AttnAgreeSource(), g_min=0.0, g_max=0.0, steer_from=2)
    tap = add_attention(ctrl, layer=1, window=W)
    att = CA2Attachment(model, ctrl, tap=1, inject_before=2, taps=[tap, NormTap()])
    assert torch.equal(run_once(model, ctrl, ids).logits, base)
    att.remove()
    ctrl = CA2Controller(D, AttnWindowSource(), g_max=0.5, g_abs=1.0, steer_from=2)
    tap = add_attention(ctrl, layer=1, window=W)
    att = CA2Attachment(model, ctrl, tap=1, inject_before=2, taps=[tap])
    ids2 = ids.clone()
    ids2[:, 19:] = (ids2[:, 19:] + 5) % 256
    lg1 = run_once(model, ctrl, ids).logits
    tr1 = ctrl.trace_tensors()
    lg2 = run_once(model, ctrl, ids2).logits
    tr2 = ctrl.trace_tensors()
    att.remove()
    same(lg1[:, :19], lg2[:, :19])
    for k in tr1:
        if k != "ids":
            same(tr1[k][:, :19], tr2[k][:, :19])
    assert tr1["g"].abs().sum() > 0


# ---------------------------------------------------------------------------------- HookedLM window mask
@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_hookedlm_block_window_mask(attn):
    """Block mode: keys outside the query's W-block (other than the sink) get no mass and are never eligible;
    the statistics equal those from the eager twin's own weights under the same additive window mask."""
    model = tiny_qwen3(attn=attn)
    ids = tokens(n=1 + 4 * W)
    ctrl = CA2Controller(D, sink_skip=0, steer_from=0, g_min=0.0, g_max=0.0)
    ctrl.record = True
    tap = add_attention(ctrl, window=W)
    side = SteerSide(ctrl, model=model, sink=1, taps=[tap], tap_layer=1)
    lm = pl.HookedLM(model, side, layer=1, window=W, mode="block", window_layers="all", sink=1, pad_id=BOS)
    lm(ids)
    got = dict(tap.cur)
    tr = side.last_trace
    for h in lm.handles:
        h.remove()
    side.remove()
    Tm = ids.shape[1]
    i = torch.arange(Tm)
    blk = (i - 1).div(W, rounding_mode="floor")
    for t in range(1, Tm):
        for l in range(1, W + 1):
            j = t - l
            inside = j >= 1 and blk[j] == blk[t]
            assert bool(got["band_ok"][0, t, l - 1]) == inside
            if not inside:
                assert got["band"][:, t, l - 1].eq(0).all()
    assert got["n_elig"][0, 1:].tolist() == [float((i[1:t + 1] != t).logical_and(blk[1:t + 1] == blk[t]).sum())
                                             for t in range(1, Tm)]
    assert tr["tap/attn/elig"].shape == (2, Tm - 1)                  # sink column sliced off in the trace
    assert "src/attn_window/s" in tr and "src/attn_agree/s" in tr
    if attn == "eager":                                              # reference: the module's own weights
        caught = {}
        mod = pl.find_blocks(model)[1].self_attn
        hk = mod.register_forward_hook(lambda m, a, o: caught.__setitem__("w", o[1]))
        lm2 = pl.HookedLM(model, None, layer=1, window=W, mode="block", window_layers="all", sink=1, pad_id=BOS)
        with lm2.condition(W, False):
            lm2(ids)
        hk.remove()
        for h in lm2.handles:
            h.remove()
        wm = pl.window_mask(Tm, W, "block", 1, "cpu", torch.float32)[:, 0] > torch.finfo(torch.float32).min / 2
        compare(got, reference(caught["w"], torch.ones(2, Tm, dtype=torch.bool), vis=wm))


# ------------------------------------------------------------------------------------- 2. strict causality
SRC = {
    "attn_window": lambda: dict(source=AttnWindowSource()),
    "attn_window_mean_prev": lambda: dict(source=AttnWindowSource(pool="mean", exclude_prev=True)),
    "attn_agree": lambda: dict(source=AttnAgreeSource(), smooth="integral", kappa=2.0),
    "attn_agree_mean": lambda: dict(source=AttnAgreeSource(pool="mean", window=4), regulator="ngram"),
}


def build(name, **kw):
    base = dict(g_max=0.5, g_abs=1.0, steer_from=2)
    base.update(SRC[name]())
    base.update(kw)
    ctrl = CA2Controller(D, **base)
    tap = add_attention(ctrl, window=W)
    return ctrl, tap


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name", list(SRC))
def test_strict_causality_bitwise_with_attention_sources(name, dtype):
    model = tiny_qwen3(dtype)
    ctrl, tap = build(name)
    att = CA2Attachment(model, ctrl, tap=1, taps=[tap, NormTap()])
    ids, t0 = tokens(), 19
    ids2 = ids.clone()
    ids2[:, t0:] = (ids2[:, t0:] + 7) % 256
    lg1 = run_once(model, ctrl, ids).logits.float()
    tr1 = ctrl.trace_tensors()
    lg2 = run_once(model, ctrl, ids2).logits.float()
    tr2 = ctrl.trace_tensors()
    att.remove()
    assert not torch.equal(lg1[:, t0:], lg2[:, t0:])
    same(lg1[:, :t0], lg2[:, :t0])
    assert set(tr1) == set(tr2)
    for k in tr1:
        if k != "ids":
            same(tr1[k][:, :t0], tr2[k][:, :t0])
    assert tr1["g"][:, :t0].abs().sum() > 0
    assert any(k.startswith("tap/attn/") for k in tr1) and "src/attn_agree/s" in tr1 and "src/attn_window/s" in tr1


# ------------------------------------------------------------------------------ 3. prefill + decode
@pytest.mark.parametrize("name", list(SRC))
@pytest.mark.parametrize("direction", ["gist", "random"])
def test_prefill_then_cached_decode_equals_one_pass_for_attention_sources(name, direction):
    model = tiny_qwen3()
    ids, mask = padded()
    ctrl, tap = build(name, direction=direction, steer_from="all")
    att = CA2Attachment(model, ctrl, tap=1, taps=[tap])
    lg1 = run_once(model, ctrl, ids, mask).logits
    tr1 = ctrl.trace_tensors()
    lg2 = torch.cat([o.logits for o in incremental(model, ctrl, ids, mask, 12)], 1)
    tr2 = ctrl.trace_tensors()
    att.remove()
    m = mask.bool()
    assert torch.allclose(lg1[m], lg2[m], atol=1e-4)
    keys = [k for k in tr1 if k.startswith(("src/attn", "tap/attn")) or k in ("g", "g_raw", "s", "r", "E")]
    for k in keys:
        assert torch.allclose(tr1[k], tr2[k], atol=1e-5, rtol=1e-4, equal_nan=True), k
    assert tr1["g"].abs().sum() > 0 and tr1["s"].std() > 0


def test_generate_with_attention_source_matches_teacher_forced_pass():
    model = tiny_qwen3()
    prompts = [[BOS] + list(range(70, 85)), [BOS] + list(range(90, 99))]
    ids, mask = A.data.pad_left(prompts, BOS)
    ctrl = CA2Controller(D, AttnAgreeSource(), g_max=0.5, g_abs=1.0)
    tap = add_attention(ctrl, window=W)
    att = CA2Attachment(model, ctrl, tap=1, taps=[tap])
    import argparse
    new, st = A.run.generate(model, pl.ByteTokenizer(), ids, mask, ctrl, argparse.Namespace(max_new_tokens=7))
    g_gen = torch.cat([ctrl.trace[0]["g"][:, -1:]] + [t["g"] for t in ctrl.trace[1:]], 1)
    el_gen = torch.cat([ctrl.trace[0]["tap/attn/elig"][:, -1:]] + [t["tap/attn/elig"] for t in ctrl.trace[1:]], 1)
    full = torch.cat([ids, new], 1)
    fmask = torch.cat([mask, torch.ones_like(new)], 1)
    ctrl.steer_from = mask.sum(1) - 1
    run_once(model, ctrl, full, fmask)
    P = ids.shape[1]
    sl = slice(P - 1, P - 1 + g_gen.shape[1])
    g_tf, el_tf = ctrl.trace[0]["g"][:, sl], ctrl.trace[0]["tap/attn/elig"][:, sl]
    att.remove()
    assert torch.allclose(g_tf, g_gen, atol=1e-5) and torch.allclose(el_tf, el_gen, atol=1e-5)
    assert g_gen.abs().sum() > 0


# ------------------------------------------------------------------------------ 6. gain 0 is byte-identical
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_gain_with_attention_tap_is_byte_identical(dtype):
    model = tiny_qwen3(dtype)
    ids, mask = padded()
    pos = A.probe.position_ids(mask)
    base = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
    base_inc = torch.cat([o.logits for o in incremental(model, None, ids, mask, 12)], 1)
    ctrl, tap = build("attn_agree", g_min=0.0, g_max=0.0)
    att = CA2Attachment(model, ctrl, tap=1, inject_before=3, taps=[tap, NormTap()])
    zero = run_once(model, ctrl, ids, mask).logits
    assert "tap/attn/lagbin" in ctrl.trace[0] and "src/attn_agree/agree_active" in ctrl.trace[0]
    zero_inc = torch.cat([o.logits for o in incremental(model, ctrl, ids, mask, 12)], 1)
    att.remove()
    off = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
    assert torch.equal(zero, base) and torch.equal(zero_inc, base_inc) and torch.equal(off, base)


# ----------------------------------------------------------------------------- source arithmetic, 4, 5
def test_sources_reduce_the_band_as_defined():
    """attn_window's lift and attn_agree's weighted agreement recomputed from the traced tap statistics and
    the bank, for the active pool; the gain stays in range, nothing is a parameter, the model is untouched."""
    from ca2.bank import GistBank
    model = tiny_qwen3()
    ids = tokens()
    flags = [p.requires_grad for p in model.parameters()]
    seen = {}
    cap = pl.find_blocks(model)[1].register_forward_hook(lambda m, a, o: seen.__setitem__(
        "h", (o[0] if isinstance(o, tuple) else o).detach().float()))     # before the attachment: pre-injection
    ctrl = CA2Controller(D, AttnWindowSource(eps=0.0), g_min=0.05, g_max=0.25, g_abs=0.3, steer_from=2)
    tap = add_attention(ctrl, window=W, eps=0.0)
    att = CA2Attachment(model, ctrl, tap=1, taps=[tap])
    run_once(model, ctrl, ids)
    tr = ctrl.trace_tensors()
    att.remove()
    cap.remove()
    st = tr["steer"].bool()
    assert (tr["g_raw"][st] >= 0.05 - 1e-7).all() and (tr["g_raw"][st] <= 0.25 + 1e-7).all()
    assert (tr["g_raw"][~st] == 0).all()
    band, ok, elig, nel = (tr[f"tap/attn/{k}"] for k in ("band", "band_ok", "elig", "n_elig"))
    assert not tr["massive"].any()
    nb = ok.sum(-1)
    frac = band.sum(-1) / elig.clamp_min(1e-30)
    u = nb / nel.clamp_min(1)
    dfn = (elig > 0) & (nel > nb) & (nb > 0) & (tr["tap/attn/qreal"] > 0)
    lift = ((frac - u) / (1 - u)).clamp(0, 1)
    assert dfn.any() and torch.allclose(tr["src/attn_window/lift_active"][dfn], lift[dfn], atol=1e-6)
    assert torch.equal(tr["src/attn_window/defined_active"].bool(), dfn)
    assert torch.allclose(tr["s"], hold(torch.where(dfn, lift, 0 * lift), dfn, None), atol=1e-6)
    hs = seen["h"]
    assert torch.allclose(hs.norm(dim=-1), tr["hnorm"], atol=1e-5)    # the stream the bank read
    bs = GistBank().update(hs)
    pm, pb = bs.prev(0), bs.prev(-1)
    a_ref = torch.zeros_like(band)
    for l in range(1, W + 1):
        hj = torch.cat([torch.zeros(2, l, D), hs[:, :-l]], 1)
        a_ref[..., l - 1] = torch.nn.functional.cosine_similarity(hj - pb, pm - pb, dim=-1, eps=1e-8)
    w = torch.where(ok > 0, band, torch.zeros(()))
    agr = (w * a_ref.clamp_min(0)).sum(-1) / w.sum(-1).clamp_min(1e-30)
    d2 = tr["src/attn_agree/defined_active"].bool()
    assert d2.any() and torch.allclose(tr["src/attn_agree/agree_active"][d2], agr[d2].clamp(0, 1), atol=1e-5)
    assert sum(p.numel() for p in ctrl.parameters()) == 0
    assert [p.requires_grad for p in model.parameters()] == flags


def test_state_is_fixed_size_during_decode():
    model = tiny_qwen3()
    ids, mask = padded(n=40)
    ctrl, tap = build("attn_agree")
    att = CA2Attachment(model, ctrl, tap=1, taps=[tap])
    ctrl.reset()
    ctrl.pending_mask = mask[:, :12]
    pos = A.probe.position_ids(mask)
    o = model(input_ids=ids[:, :12], attention_mask=mask[:, :12], position_ids=pos[:, :12], use_cache=True)
    pkv, sizes = o.past_key_values, []
    for t in range(12, 40):
        o = model(input_ids=ids[:, t:t + 1], attention_mask=mask[:, :t + 1], position_ids=pos[:, t:t + 1],
                  past_key_values=pkv, use_cache=True)
        pkv = o.past_key_values
        sizes.append((ctrl.state_numel(), tap.state_numel()))
    att.remove()
    assert len(set(sizes)) == 1 and sizes[0][1] == 2                  # sink columns (B, 1)


def test_massive_band_keys_are_excluded_by_the_sources():
    """A massive flag on a band key removes it from the band and from the eligible mass (both pools)."""
    torch.manual_seed(3)
    B, Tn, Wr, H = 1, 6, 4, 2

    class FakeTap:
        window = Wr
        cur = {}

    band = torch.rand(B, Tn, Wr) * 0.1
    ok = torch.ones(B, Tn, Wr, dtype=torch.bool)
    qband = torch.rand(B, Tn, H, Wr) * 0.1
    FakeTap.cur = {"band": band, "band_ok": ok, "qband": qband, "elig": torch.full((B, Tn), 0.8),
                   "n_elig": torch.full((B, Tn), 10.0), "qreal": torch.ones(B, Tn)}
    from ca2.bank import GistBank
    bstep = GistBank(massive_window=8).update(torch.randn(B, Tn, D))
    bstep.massive = torch.zeros(B, Tn, dtype=torch.bool)
    bstep.massive[0, 2] = True                                       # column 2 is massive
    from ca2.sources import StepContext
    ctx = StepContext(h=torch.randn(B, Tn, D), x=None, mask=torch.ones(B, Tn), use=torch.ones(B, Tn),
                      steer=torch.ones(B, Tn), idx=torch.arange(Tn).float()[None], step=torch.zeros(B, Tn).long(),
                      col=torch.arange(Tn), first=True, call=0, ids=None, bank=bstep, taps={"attn": FakeTap})
    src = AttnWindowSource(eps=0.0)
    o = src(ctx)
    t = 4                                                            # query 4 sees column 2 at lag 2
    e = 0.8 - band[0, t, 1]
    frac = (band[0, t].sum() - band[0, t, 1]) / e
    u = 3 / 9
    assert torch.allclose(o["frac_active"][0, t], frac) and o["n_massive"][0, t] == 1
    assert torch.allclose(o["lift_active"][0, t], ((frac - u) / (1 - u)).clamp(0, 1))
    qm = qband[0, t].clone()
    den = 1 - qm[:, 1]
    qm[:, 1] = 0
    assert torch.allclose(o["frac_mean"][0, t], (qm.sum(-1) / den).mean())
    assert o["n_massive"][0, 1] == 0 and o["n_massive"][0, 3] == 1    # query 3: column 2 at lag 1


def test_hold_and_registration():
    s = torch.tensor([[0.3, 0.9, 0.5, 0.1]])
    d = torch.tensor([[False, True, False, True]])
    assert torch.equal(hold(s, d, torch.tensor([0.7])), torch.tensor([[0.7, 0.9, 0.9, 0.1]]))
    assert "attn_window" in SOURCES and isinstance(make_source("attn_agree"), AttnAgreeSource)
    with pytest.raises(KeyError):                                    # the source needs its tap
        CA2Controller(D, "attn_window", steer_from="all")(torch.randn(1, 4, D))
    q = torch.randn(1, 4, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    st = attn_stats(q, k, visibility(None, 1, 5, 5, "cpu"), torch.ones(1, 5, dtype=torch.bool),
                    torch.tensor([[True, False, False, False, False]]), 8 ** -0.5, window=3, chunk_elems=1)
    st2 = attn_stats(q, k, visibility(None, 1, 5, 5, "cpu"), torch.ones(1, 5, dtype=torch.bool),
                     torch.tensor([[True, False, False, False, False]]), 8 ** -0.5, window=3)
    for key in st:                                                   # row chunking does not change the result
        torch.testing.assert_close(st[key], st2[key], equal_nan=True)
