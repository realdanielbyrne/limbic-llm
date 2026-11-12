"""Round 12 CA2 core (experiments/ca2): equivalence with steering.py, causality, prefill/decode, gain range,
no parameters, byte-identical side-off, controls, massive tokens, the repetition regulator, the tap extension
point, run.generate / run.perplexity compatibility, HookedLM hosting and trace storage.

A tiny random Qwen3 (4 attention layers, hidden 64, GQA 4:2, head 16) with the byte tokenizer; no downloads.
Plan test numbers are given in the section headers.
"""

import argparse
import math
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments" / "alignment")]

import pretrained_loop as pl  # noqa: E402
from ca2.bank import GistBank, dim_keep_mask, massive_dims  # noqa: E402
from ca2.common import SteerSide, TraceWriter, alignment, gain_stats, paired_boot, row_tokens  # noqa: E402
from ca2.controller import DIRECTIONS, CA2Attachment, CA2Controller, GainPipeline, NormTap, Tap  # noqa: E402
from ca2.regulators import leaky_scan  # noqa: E402
from ca2.sources import (AgreeSource, ConstantSource, PositionMeanSource, Source, YokedSource,  # noqa: E402
                         derange)
from steering import Attachment, SteeringController  # noqa: E402

D, T, BOS = 64, 33, 256
A = alignment()


def tiny_qwen3(dtype=torch.float32, seed=0, attn="sdpa"):
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
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 256, (B, n), generator=g)
    ids[:, 0] = BOS
    return ids


def padded(B=2, n=T, pad=3, seed=0):
    ids = tokens(B, n, seed)
    mask = torch.ones(B, n, dtype=torch.long)
    ids[0, :pad], mask[0, :pad] = BOS, 0
    ids[0, pad] = BOS                                             # row 0's sink after its padding
    return ids, mask


def donor(B, N=48, seed=3):
    return torch.rand(B, N, generator=torch.Generator().manual_seed(seed))


# fresh source configurations (instances carry state, so build per controller)
CONFIGS = {
    "agree_ema": lambda B: dict(source="agree"),
    "agree_integral_release_ngram": lambda B: dict(source=AgreeSource(theta_release=0.0), smooth="integral",
                                                   kappa=2.0, release_rho=0.3, regulator="ngram"),
    "agree_none_hidden_dose": lambda B: dict(source="agree", smooth="none", regulator="hidden", dose_cap=2.0,
                                             dose_leak=0.9),
    "agree_restore_clip": lambda B: dict(source=AgreeSource(a_lo=-0.5, a_hi=0.5), polarity="restore",
                                         restore_kappa=4.0),
    "constant": lambda B: dict(source=ConstantSource(0.6)),
    "yoked": lambda B: dict(source=YokedSource(donor(B))),
    "position_mean": lambda B: dict(source=PositionMeanSource(torch.linspace(0, 1, 40))),
}


def make(B=2, config="agree_ema", **kw):
    base = dict(g_max=0.5, g_abs=1.0, steer_from=2, window=8)
    base.update(CONFIGS[config](B))
    base.update(kw)
    return CA2Controller(D, **base)


def forward(model, ctrl, ids, mask=None):
    ctrl.reset()
    ctrl.record = True
    ctrl.pending_mask = mask
    pos = A.probe.position_ids(mask) if mask is not None else None
    return model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits.float()


def incremental(model, ctrl, ids, mask, P):
    """Prefill P columns through the KV cache, then teacher-force one token per call."""
    ctrl.reset()
    ctrl.record = True
    ctrl.pending_mask = mask[:, :P]
    pos = A.probe.position_ids(mask)
    out = model(input_ids=ids[:, :P], attention_mask=mask[:, :P], position_ids=pos[:, :P], use_cache=True)
    logits, pkv = [out.logits.float()], out.past_key_values
    for t in range(P, ids.shape[1]):
        o = model(input_ids=ids[:, t:t + 1], attention_mask=mask[:, :t + 1], position_ids=pos[:, t:t + 1],
                  past_key_values=pkv, use_cache=True)
        logits.append(o.logits.float())
        pkv = o.past_key_values
    return torch.cat(logits, 1)


def same(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


# --------------------------------------------------------------------------------- 1. steering equivalence
@pytest.mark.parametrize("direction", ["gist", "anti_gist", "mean"])
def test_bank_with_constant_gain_injects_what_steering_controller_injects(direction):
    torch.manual_seed(1)
    h = torch.randn(2, 14, D) * 3 + 1.0
    mask = torch.ones(2, 14)
    mask[0, :3] = 0
    ref = SteeringController(None, mode="static", gain=0.2, direction_mode=direction, d_model=D)
    new = CA2Controller(D, "constant", g_min=0.2, g_max=0.2, direction=direction, steer_from="all")
    for c in (ref, new):                                          # one pass, left padding
        c.reset()
        c.pending_mask = mask
    assert torch.allclose(ref(h), new(h), atol=1e-6)
    outs = []
    for c in (ref, new):                                          # prefill + decode
        c.reset()
        c.pending_mask = mask[:, :8]
        outs.append(torch.cat([c(h[:, :8])] + [c(h[:, t:t + 1]) for t in range(8, 14)], 1))
    assert torch.allclose(outs[0], outs[1], atol=1e-6)


def test_bank_default_bands_match_steering_averages():
    torch.manual_seed(2)
    h = torch.randn(3, 10, D)
    ref = SteeringController(None, mode="static", gain=0.1, direction_mode="gist", d_model=D)
    ref(h)
    st = GistBank().update(h)
    assert torch.allclose(st.num[0][:, -1], ref.ema["fast"][0], atol=1e-6)
    assert torch.allclose(st.num[1][:, -1], ref.ema["slow"][0], atol=1e-6)
    assert torch.allclose(st.den[1][:, -1], ref.ema["slow"][1], atol=1e-6)
    # previous-step averages: prev(i)[t] is avg(i)[t − 1], and the carried value at the first column
    assert torch.allclose(st.prev(0)[:, 1:], st.avg(0)[:, :-1])
    bank = GistBank()
    bank.update(h[:, :6])
    st2 = bank.update(h[:, 6:])
    assert torch.allclose(st2.prev(1)[:, 0], st.avg(1)[:, 5], atol=1e-6)


def test_model_generation_matches_steering_controller_static():
    model = tiny_qwen3()
    prompts = [[BOS] + list(range(40, 52)), [BOS] + list(range(60, 67))]
    ids, mask = A.data.pad_left(prompts, BOS)
    args = argparse.Namespace(max_new_tokens=6)
    ref = SteeringController(None, mode="static", gain=0.3, direction_mode="gist", d_model=D)
    att = Attachment(pl.find_blocks(model)[1], ref)
    new_ref, st_ref = A.run.generate(model, pl.ByteTokenizer(), ids, mask, ref, args)
    g_ref = torch.cat([ref.trace[0]["g"][:, -1:]] + [t["g"] for t in ref.trace[1:]], 1)
    att.remove()
    ctrl = CA2Controller(D, "constant", g_min=0.3, g_max=0.3, g_abs=1.0, direction="gist", steer_from="all")
    att = CA2Attachment(model, ctrl, tap=1)                       # static mode steers the prompt as well
    new, st = A.run.generate(model, pl.ByteTokenizer(), ids, mask, ctrl, args)
    g_new = torch.cat([ctrl.trace[0]["g"][:, -1:]] + [t["g"] for t in ctrl.trace[1:]], 1)
    att.remove()
    assert torch.equal(new, new_ref)
    assert torch.allclose(g_new, g_ref, atol=1e-6)
    assert torch.allclose(st["gain_mean"], st_ref["gain_mean"], atol=1e-6)


# ------------------------------------------------------------------------------------ 2. causality
CAUSAL = [(c, "gist") for c in CONFIGS] + [("agree_ema", d) for d in DIRECTIONS if d != "gist"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("config,direction", CAUSAL)
def test_strict_causality_bitwise(config, direction, dtype):
    model = tiny_qwen3(dtype)
    ctrl = make(2, config, direction=direction)
    att = CA2Attachment(model, ctrl, tap=1, taps=[NormTap()])
    ids, t0 = tokens(), 19
    ids2 = ids.clone()
    ids2[:, t0:] = (ids2[:, t0:] + 7) % 256
    lg1 = forward(model, ctrl, ids)
    tr1 = ctrl.trace_tensors()
    lg2 = forward(model, ctrl, ids2)
    tr2 = ctrl.trace_tensors()
    att.remove()
    assert not torch.equal(lg1[:, t0:], lg2[:, t0:])
    same(lg1[:, :t0], lg2[:, :t0])
    assert set(tr1) == set(tr2)
    for k in tr1:
        if k == "ids":
            continue
        same(tr1[k][:, :t0], tr2[k][:, :t0])
    assert tr1["g"][:, :t0].abs().sum() > 0                       # the gain acted before t0


# ------------------------------------------------------------------------------ 3. prefill + decode
@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("direction", ["gist", "window_gist", "random"])
def test_prefill_then_cached_decode_equals_one_pass(config, direction):
    """Not bitwise: prefill and decode use different matmul shapes in the model; the gain recurrences are
    leaky scans (same arithmetic) and the bank uses the steering kernel, so the tolerance is float32 noise."""
    model = tiny_qwen3()
    ids, mask = padded()
    ctrl = make(2, config, direction=direction, steer_from="all")
    att = CA2Attachment(model, ctrl, tap=1)
    lg1 = forward(model, ctrl, ids, mask)
    tr1 = ctrl.trace_tensors()
    lg2 = incremental(model, ctrl, ids, mask, 12)
    tr2 = ctrl.trace_tensors()
    att.remove()
    m = mask.bool()
    assert torch.allclose(lg1[m], lg2[m], atol=1e-4)
    for k in ("g", "g_raw", "s", "r", "E", "rho", "dose", "src/agree/a", "rep_ngram", "rep_hidden"):
        assert torch.allclose(tr1[k], tr2[k], atol=1e-5, equal_nan=True), k
    assert tr1["g"].abs().sum() > 0


def test_generate_matches_teacher_forced_pass():
    model = tiny_qwen3()
    prompts = [[BOS] + list(range(70, 85)), [BOS] + list(range(90, 99))]
    ids, mask = A.data.pad_left(prompts, BOS)
    ctrl = CA2Controller(D, "agree", g_max=0.5, g_abs=1.0, regulator="ngram")
    att = CA2Attachment(model, ctrl, tap=1)
    n_new = 7
    new, st = A.run.generate(model, pl.ByteTokenizer(), ids, mask, ctrl, argparse.Namespace(max_new_tokens=n_new))
    g_gen = torch.cat([ctrl.trace[0]["g"][:, -1:]] + [t["g"] for t in ctrl.trace[1:]], 1)
    assert ctrl.trace[0]["g"][:, :-1].abs().sum() == 0            # prompt positions are not steered
    full = torch.cat([ids, new], 1)
    fmask = torch.cat([mask, torch.ones_like(new)], 1)
    ctrl.steer_from = mask.sum(1) - 1                             # real-token index of the last prompt token
    forward(model, ctrl, full, fmask)
    P = ids.shape[1]
    g_tf = ctrl.trace[0]["g"][:, P - 1: P - 1 + g_gen.shape[1]]
    att.remove()
    assert torch.allclose(g_tf, g_gen, atol=1e-5)
    assert ctrl.trace[0]["g"][:, : P - 1].abs().sum() == 0
    for k in ("gain_mean", "gain_max", "gain_prompt_mean", "capped_frac", "E_final", "s_prompt", "s_gen_mean",
              "s_gen", "gen_mask", "gain_sd", "frac_at_gmin", "frac_at_gmax", "n_resp"):
        assert k in st


def test_run_perplexity_contract_and_zero_gain_identity():
    model = tiny_qwen3()
    tok = pl.ByteTokenizer()
    texts = ["the quick brown fox jumps over the lazy dog " * 3, "pack my box with five dozen liquor jugs " * 3]
    args = argparse.Namespace(ppl_prefix=8, ppl_len=60, batch=2)
    base = A.run.perplexity(model, tok, texts, None, args)
    zero = CA2Controller(D, "agree", g_min=0.0, g_max=0.0)
    att = CA2Attachment(model, zero, tap=1, taps=[NormTap()])
    pz = A.run.perplexity(model, tok, texts, zero, args)
    att.remove()
    assert pz["ppl"] == base["ppl"] and pz["gain_mean"] == 0.0
    ctrl = CA2Controller(D, "constant", g_min=0.4, g_max=0.4, g_abs=1.0)
    att = CA2Attachment(model, ctrl, tap=1)
    p = A.run.perplexity(model, tok, texts, ctrl, args)
    att.remove()
    assert p["gain_mean"] > 0 and p["ppl"] != base["ppl"]


# ------------------------------------------------------------------------- 4. gain range, positions, state
@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("smooth", ["none", "ema", "integral"])
@pytest.mark.parametrize("polarity", ["reinforce", "restore"])
@pytest.mark.parametrize("regulator", ["none", "ngram", "hidden"])
def test_gain_stays_in_range_and_is_zero_off_the_steered_positions(config, smooth, polarity, regulator):
    torch.manual_seed(4)
    B, P, n = 3, 9, 16
    kw = dict(g_min=0.05, g_max=0.25, g_abs=0.3, steer_from="last_prompt", smooth=smooth, polarity=polarity,
              regulator=regulator)
    ctrl = CA2Controller(D, **{**CONFIGS[config](B), **kw})
    ctrl.record = True
    h = torch.randn(B, P + n, D) * torch.rand(B, P + n, 1) * 4
    ids = torch.randint(0, 6, (B, P + n))                       # small vocabulary: repeats happen
    mask = torch.ones(B, P)
    mask[0, :2] = 0
    ctrl.pending_mask = mask
    ctrl.begin_call(ids[:, :P])
    ctrl(h[:, :P])
    sizes = []
    for t in range(P, P + n):
        ctrl.begin_call(ids[:, t:t + 1])
        ctrl(h[:, t:t + 1])
        sizes.append(ctrl.state_numel())
    tr = ctrl.trace_tensors()
    st = tr["steer"].bool()
    g = tr["g_raw"]
    assert (g[st] >= 0.05 - 1e-7).all() and (g[st] <= 0.25 + 1e-7).all()
    assert (g[~st] == 0).all() and (tr["g"][~st] == 0).all()
    assert (tr["g"] <= 0.25 + 1e-6).all()
    assert not st[:, : P - 1].any() and st[:, P - 1:].all()     # last prompt token on; padding, sink, prompt off
    assert len(set(sizes)) == 1                                  # fixed state


def test_dose_cap_bounds_the_accumulated_gain():
    torch.manual_seed(5)
    ctrl = CA2Controller(D, "constant", g_min=0.02, g_max=0.3, g_abs=0.3, dose_cap=1.0, dose_leak=0.9,
                         steer_from="all")
    ctrl.record = True
    ctrl(torch.randn(2, 40, D))
    tr = ctrl.trace_tensors()
    assert (tr["dose"] <= 1.0 + 1e-6).all() and tr["dose"].max() > 0.99
    st = tr["steer"].bool()
    assert (tr["g_raw"][st] >= 0.02 - 1e-7).all() and (tr["g_raw"][st] < 0.3).any()
    with pytest.raises(ValueError):
        CA2Controller(D, "constant", g_min=0.2, g_max=0.3, dose_cap=1.0, dose_leak=0.9)


def test_leaky_scan_is_bitwise_across_split_calls():
    torch.manual_seed(6)
    c, x = torch.rand(2, 30), torch.randn(2, 30)
    full = leaky_scan(c, x)
    s, parts = None, []
    for t in range(30):
        o = leaky_scan(c[:, t:t + 1], x[:, t:t + 1], s)
        s = o[:, -1]
        parts.append(o)
    assert torch.equal(full, torch.cat(parts, 1))


# ------------------------------------------------------------------------------------ 5. no parameters
def test_zero_trainable_parameters_and_frozen_model_untouched():
    model = tiny_qwen3()
    flags = [p.requires_grad for p in model.parameters()]
    n_model = sum(p.numel() for p in model.parameters())
    ctrl = make(2, "agree_integral_release_ngram", dim_keep=dim_keep_mask(D, [3, 5]))
    att = CA2Attachment(model, ctrl, tap=1, taps=[NormTap()])
    forward(model, ctrl, tokens())
    assert sum(p.numel() for p in ctrl.parameters()) == 0
    assert {n for n, _ in ctrl.named_buffers()} <= {"d_rand", "dim_keep"}
    side = SteerSide(CA2Controller(D, sink_skip=0, steer_from=0), model=model)
    assert sum(p.numel() for p in side.parameters()) == 0
    side.remove()
    att.remove()
    assert [p.requires_grad for p in model.parameters()] == flags
    assert sum(p.numel() for p in model.parameters()) == n_model


# ------------------------------------------------------------------------------- 6. side off is identical
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_side_off_and_zero_gain_are_byte_identical(dtype):
    model = tiny_qwen3(dtype)
    ids, mask = padded()
    pos = A.probe.position_ids(mask)
    base = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
    ctrl = make(2, "agree_integral_release_ngram", g_min=0.0, g_max=0.0)
    att = CA2Attachment(model, ctrl, tap=1, inject_before=3, taps=[NormTap()])
    ctrl.reset()
    ctrl.record = True
    ctrl.pending_mask = mask
    zero = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
    assert "tap/norms/norm" in ctrl.trace[0] and "src/agree/a" in ctrl.trace[0]
    att.remove()
    ctrl2 = make(2, "constant")
    att2 = CA2Attachment(model, ctrl2, tap=1)
    att2.remove()
    off = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
    assert torch.equal(zero, base) and torch.equal(off, base)


def test_injection_site_before_a_later_block():
    """inject_before = L + 2: blocks ≤ L + 1 keep their outputs, blocks ≥ L + 2 change; self = x/‖x‖."""
    model = tiny_qwen3()
    ids = tokens()
    probe = CA2Controller(D, "constant", g_min=0.0, g_max=0.0)
    att = CA2Attachment(model, probe, tap=0, taps=[NormTap()])
    forward(model, probe, ids)
    n0 = probe.trace[0]["tap/norms/norm"]
    att.remove()
    ctrl = CA2Controller(D, "constant", g_min=0.5, g_max=0.5, g_abs=1.0, direction="self", steer_from="all")
    att = CA2Attachment(model, ctrl, tap=0, inject_before=2, taps=[NormTap()])
    forward(model, ctrl, ids)
    n1 = ctrl.trace[0]["tap/norms/norm"]
    tr = ctrl.trace[0]
    att.remove()
    assert torch.equal(n1[..., :2], n0[..., :2]) and not torch.allclose(n1[..., 2:], n0[..., 2:])
    st = tr["steer"].bool()
    assert torch.allclose(tr["xnorm"][st], n0[..., 1][st], rtol=1e-5)   # injected at block 1's output
    with pytest.raises(ValueError):
        CA2Attachment(model, ctrl, tap=1, inject_before=3, relevance_layer=3)


# --------------------------------------------------------------------------- tap extension point
class _CountTap(Tap):
    """Test tap: per-call buffer of ‖x‖ at block 0's input; logs a second signal through ctrl.log."""

    name = "count"

    def __init__(self):
        self.handles, self.buf, self.calls, self.resets = [], None, 0, 0

    def attach(self, att):
        self.ctrl = att.ctrl
        self.handles.append(att.blocks[0].register_forward_pre_hook(self._pre))

    def _pre(self, mod, args):
        self.buf = args[0].detach().float().norm(dim=-1)
        self.ctrl.log("tap/count/logged", self.buf * 2)

    def remove(self):
        for h in self.handles:
            h.remove()

    def reset(self):
        self.resets += 1

    def begin_call(self, ctrl):
        self.calls += 1
        self.buf = None

    def signals(self):
        return {"norm0": self.buf}


class _TapSource(Source):
    name, needs = "from_tap", ("count",)

    def __call__(self, ctx):
        x = ctx.taps["count"].buf[:, ctx.col_offset:]
        return {"s": torch.sigmoid(x - x.mean()), "raw": x}


def test_tap_registration_sources_and_trace():
    model = tiny_qwen3()
    tap = _CountTap()
    ctrl = CA2Controller(D, _TapSource(), g_max=0.3, g_abs=0.3, steer_from="all")
    att = CA2Attachment(model, ctrl, tap=1, taps=[tap])
    ids = tokens()
    forward(model, ctrl, ids)
    tr = ctrl.trace[0]
    assert tap.calls == 1 and tap.resets >= 1
    assert torch.equal(tr["tap/count/norm0"], tr["tap/count/logged"] / 2)
    assert torch.equal(tr["src/from_tap/raw"], tr["tap/count/norm0"])
    assert "src/agree/a" in tr                                   # every registered source is traced
    att.remove()
    assert "count" not in ctrl.taps
    with pytest.raises(KeyError):                               # a source whose tap is missing
        c2 = CA2Controller(D, _TapSource(), steer_from="all")
        c2(torch.randn(1, 4, D))


# ------------------------------------------------------------------------------------------ 9. controls
def _synthetic(ctrl, h, mask=None):
    ctrl.reset()
    ctrl.record = True
    ctrl.pending_mask = mask
    out = ctrl(h)
    return out, ctrl.trace_tensors()


def test_fixed_mean_override_matches_the_dynamic_mean_with_zero_variance():
    torch.manual_seed(7)
    h = torch.randn(3, 30, D)
    dyn = CA2Controller(D, "agree", g_max=0.3, steer_from="all")
    _, tr = _synthetic(dyn, h)
    st = tr["steer"].bool()
    means = torch.stack([tr["g_raw"][b][st[b]].mean() for b in range(3)])
    fix = CA2Controller(D, "agree", g_max=0.3, steer_from="all", gain_override=means)
    _, tf = _synthetic(fix, h)
    for b in range(3):
        v = tf["g_raw"][b][st[b]]
        assert torch.allclose(v.mean(), means[b], atol=1e-6) and (v == v[0]).all()     # constant: variance 0
    assert torch.equal(tf["g_law"], tr["g_raw"])                # the law still runs and is traced


def test_yoked_replays_the_donor_and_ignores_this_rows_content():
    torch.manual_seed(8)
    h = torch.randn(2, 25, D)
    dyn = CA2Controller(D, "agree", g_max=0.3, steer_from="all")
    _, tr = _synthetic(dyn, h)
    st = tr["steer"].bool()
    r = torch.stack([tr["r"][b][st[b]] for b in range(2)])        # the donor's r over its steered steps
    dr, _ = derange(r)
    yk = CA2Controller(D, YokedSource(dr), g_max=0.3, steer_from="all")
    _, ty = _synthetic(yk, h)
    for b in range(2):
        assert torch.equal(ty["g_raw"][b][st[b]], tr["g_raw"][1 - b][st[1 - b]])
    h2 = h.clone()
    h2[0] = torch.randn(25, D) * 5
    _, ty2 = _synthetic(yk, h2)
    assert torch.equal(ty2["g_raw"], ty["g_raw"])
    assert not torch.equal(ty2["src/agree/a"], ty["src/agree/a"])


def test_position_mean_source_reads_the_table_by_position():
    table = torch.linspace(0, 1, 11)
    ctrl = CA2Controller(D, PositionMeanSource(table), g_max=0.2, steer_from="all")
    _, tr = _synthetic(ctrl, torch.randn(2, 16, D))
    idx = tr["idx"].long()
    st = tr["steer"].bool()
    assert torch.allclose(tr["g_raw"][st], 0.2 * table[idx.clamp_max(10)][st])


def test_direction_controls_do_what_they_claim():
    torch.manual_seed(9)
    h = torch.randn(3, 20, D) * 2 + 0.5
    kw = dict(g_min=0.2, g_max=0.2, steer_from="all")
    outs = {}
    for d in ("gist", "anti_gist", "random", "self", "mean", "foreign"):
        c = CA2Controller(D, "agree", direction=d, seed=11, **kw)
        out, tr = _synthetic(c, h)
        outs[d] = (out - h, tr, c)
    g_gist = outs["gist"][1]["g_raw"]
    for d in outs:
        assert torch.equal(outs[d][1]["g_raw"], g_gist)          # the gain never depends on the direction
    assert torch.allclose(outs["anti_gist"][0], -outs["gist"][0], atol=1e-7)
    ref = SteeringController(None, mode="static", gain=0.2, direction_mode="random", d_model=D, seed=11)
    unit = outs["random"][0] / (0.2 * h.norm(dim=-1, keepdim=True))
    st = outs["random"][1]["steer"].bool()
    assert torch.allclose(unit[st], ref.d.expand_as(unit)[st], atol=1e-5)
    assert torch.allclose(outs["self"][0][st], 0.2 * h[st], atol=1e-6)
    bank = GistBank().update(h)
    gist = bank.gist(0, -1)
    assert torch.allclose(outs["foreign"][0], 0.2 * h.norm(dim=-1, keepdim=True) * gist.roll(-1, 0), atol=1e-6)
    mean = torch.nn.functional.normalize(bank.avg(-1), dim=-1)
    assert torch.allclose(outs["mean"][0], 0.2 * h.norm(dim=-1, keepdim=True) * mean, atol=1e-6)
    with pytest.raises(ValueError):
        CA2Controller(D, direction="foreign", steer_from="all")(torch.randn(1, 4, D))


def test_window_gist_restarts_at_every_window_block():
    torch.manual_seed(10)
    W, n = 6, 20
    h = torch.randn(2, n, D)
    c = CA2Controller(D, "constant", g_min=0.2, g_max=0.2, direction="window_gist", window=W, sink_skip=0,
                      steer_from="all")
    out, _ = _synthetic(c, h)
    for start in range(0, n, W):
        fresh = CA2Controller(D, "constant", g_min=0.2, g_max=0.2, direction="gist", sink_skip=0, steer_from="all")
        o, _ = _synthetic(fresh, h[:, start:start + W])
        assert torch.allclose(out[:, start:start + W], o, atol=1e-6)


def test_error_only_projection_removes_the_component_along_h():
    torch.manual_seed(12)
    h = torch.randn(2, 18, D) + 2.0
    c = CA2Controller(D, "constant", g_min=0.2, g_max=0.2, direction="mean", error_only=True, steer_from="all")
    out, tr = _synthetic(c, h)
    delta = out - h
    live = tr["dnorm"] > 0
    assert torch.allclose((delta * h).sum(-1)[live], torch.zeros(()), atol=1e-4)
    assert torch.allclose(tr["dnorm"][live], torch.ones(()), atol=1e-6)
    assert (tr["cos_dh"][live].abs() > 0.1).any() and (tr["cos_dh_out"][live].abs() < 1e-5).all()


def test_massive_tokens_are_flagged_and_excluded_from_the_averages():
    torch.manual_seed(13)
    h = torch.randn(1, 24, D)
    v = torch.nn.functional.normalize(torch.randn(D), dim=0)
    h[0, 10] = 100 * h[0, :10].norm(dim=-1).mean() * v           # a delimiter-like spike
    res = {}
    for ex in (False, True):
        c = CA2Controller(D, "constant", g_min=0.2, g_max=0.2, direction="gist", exclude_massive=ex,
                          steer_from="all")
        _, tr = _synthetic(c, h)
        assert tr["massive"][0].nonzero().flatten().tolist() == [10]
        res[ex] = tr
    d_off = torch.nn.functional.cosine_similarity(GistBank().update(h).gist(0, -1)[0, 11:16], v[None], dim=-1)
    assert (d_off.abs() > 0.9).all()                              # the spike dominates the gist when counted
    bank = GistBank(exclude_massive=True)
    d_on = torch.nn.functional.cosine_similarity(bank.update(h).gist(0, -1)[0, 11:16], v[None], dim=-1)
    assert (d_on.abs() < 0.5).all()
    st = bank.update(torch.randn(1, 3, D))
    assert st.massive_prev.shape == (1, 64) and st.massive_prev[0, -14].item()   # kept over the window only
    dims = massive_dims(torch.cat([torch.randn(1, 50, D), 0 * torch.randn(1, 1, D)], 1).index_put_(
        (torch.tensor([0]), torch.tensor([7]), torch.tensor([5])), torch.tensor(500.0)))
    assert dims.tolist() == [5]


def test_repetition_regulator_decays_during_repetition_and_recovers():
    torch.manual_seed(14)
    gamma, lam = 1.0, 0.8
    ctrl = CA2Controller(D, "constant", g_min=0.0, g_max=0.2, regulator="ngram", rep_gamma=gamma, rep_leak=lam,
                         steer_from="all", smooth="none")
    ctrl.record = True
    fresh = torch.arange(100, 150)
    ids = torch.cat([fresh[:12], torch.tensor([1, 2, 3, 4] * 7), fresh[12:42]])[None]   # repeat at 12..39
    n = ids.shape[1]
    h = torch.randn(1, n, D)
    ctrl.begin_call(ids[:, :10])
    ctrl(h[:, :10])
    for t in range(10, n):                                       # decode path: the id ring carries
        ctrl.begin_call(ids[:, t:t + 1])
        ctrl(h[:, t:t + 1])
    tr = ctrl.trace_tensors()
    rep, rho = tr["rep_ngram"][0], tr["rho"][0]
    assert rep[:19].sum() == 0 and rep[19:40].all() and rep[40:].sum() == 0   # first repeated 4-gram ends at 19
    R, ref = 0.0, []
    for t in range(n):
        R = (lam * R + float(rep[t])) if t >= 1 else 0.0          # the sink is not a real-for-regulation token
        ref.append(math.exp(-gamma * R))
    assert torch.allclose(rho, torch.tensor(ref), atol=1e-5)
    assert rho[39] < 0.01 and (rho[18:40].diff() < 0).all()        # decays while repeating
    assert rho[-1] > 0.95 and (rho[39:].diff() > 0).all()          # and recovers
    assert torch.allclose(tr["g_raw"][0][1:], 0.2 * rho[1:], atol=1e-6)


def test_release_ids_multiply_the_carried_state():
    p = GainPipeline(g_min=0.0, g_max=1.0, g_abs=1.0, smooth="integral", leak=0.5, kappa=1.0, release_rho=0.0)
    s = torch.ones(1, 5)
    steer = torch.ones(1, 5)
    rel = torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.bool)
    E = p(s, steer, rel)["E"][0]
    assert torch.allclose(E, torch.tensor([1.0, 1.5, 1.0, 1.5, 1.75]))
    ctrl = CA2Controller(D, "constant", smooth="integral", leak=0.5, release_rho=0.0, release_ids=[9],
                         steer_from="all", sink_skip=0)
    ctrl.record = True
    ctrl.begin_call(torch.tensor([[5, 5, 9, 5, 5]]))
    ctrl(torch.randn(1, 5, D))
    assert torch.allclose(ctrl.trace[0]["E"][0], torch.tensor([1.0, 1.5, 1.0, 1.5, 1.75]))
    assert ctrl.trace[0]["release"][0].tolist() == [0, 0, 1, 0, 0]


# --------------------------------------------------------------------------- HookedLM host (SteerSide)
def test_steerside_in_hookedlm_zero_gain_identity_and_window_isolation():
    model = tiny_qwen3()
    ids = tokens(n=1 + 32)
    base_lm = pl.HookedLM(model, None, layer=1, window=8, mode="block", window_layers="all", sink=1, pad_id=BOS)
    with base_lm.condition(8, False):
        base = base_lm(ids)
    for h in base_lm.handles:
        h.remove()

    def side_lm(**kw):
        ctrl = CA2Controller(D, sink_skip=0, steer_from=0, g_abs=1.0, **kw)
        ctrl.record = True
        side = SteerSide(ctrl, model=model, sink=1, taps=[NormTap()], tap_layer=1)
        return pl.HookedLM(model, side, layer=1, window=8, mode="block", window_layers="all", sink=1, pad_id=BOS), side

    lm, side = side_lm(source="constant", g_min=0.0, g_max=0.0)
    assert torch.equal(lm(ids), base)
    tr = side.last_trace
    assert tr["g"].shape == (2, 32) and torch.equal(tr["ids"], ids[:, 1:])
    assert tr["tap/norms/norm"].shape == (2, 32, 4)
    for h in lm.handles:
        h.remove()
    side.remove()
    ids2 = ids.clone()
    ids2[:, 1:9] = (ids2[:, 1:9] + 3) % 256                      # change window block 0 only
    for direction, isolated in (("window_gist", True), ("gist", False)):
        lm, side = side_lm(source="constant", g_min=0.6, g_max=0.6, direction=direction, window=8)
        a, b = lm(ids), lm(ids2)
        assert torch.equal(a[:, 9:], b[:, 9:]) == isolated, direction
        for h in lm.handles:
            h.remove()
        side.remove()


# ------------------------------------------------------------------------------- storage and statistics
def test_trace_writer_stores_outside_the_repo_with_hashes(tmp_path):
    with pytest.raises(ValueError):
        TraceWriter("x", root=ROOT / "results")
    ctrl = CA2Controller(D, "agree", steer_from="all")
    _, tr = _synthetic(ctrl, torch.randn(2, 9, D))
    w = TraceWriter("unit", root=tmp_path, meta={"note": "test"}, shard_tokens=10)
    for b in range(2):
        w.add("gist_0.1", {"item": b, "correct": bool(b)}, row_tokens(tr, b))
    ptr = w.close()
    import hashlib
    import json
    man = json.loads(Path(ptr["manifest"]).read_text())
    assert hashlib.sha256(Path(ptr["manifest"]).read_bytes()).hexdigest() == ptr["manifest_sha256"]
    npz = [f for f in man["files"] if f["path"].endswith(".npz")]
    assert npz and all(hashlib.sha256((Path(ptr["dir"]) / f["path"]).read_bytes()).hexdigest() == f["sha256"]
                       for f in man["files"])
    import numpy as np
    arr = np.load(Path(ptr["dir"]) / npz[0]["path"])
    assert arr["g"].dtype == np.float16 and arr["src__agree__a"].shape[0] == 18      # both items in one shard
    assert arr["ids"].dtype == np.int32
    items = [json.loads(x) for x in (Path(ptr["dir"]) / "gist_0.1" / "items.jsonl").read_text().splitlines()]
    assert [(i["tok_shard"], i["tok_offset"], i["n_tok"]) for i in items] == [(0, 0, 9), (0, 9, 9)]
    assert np.array_equal(arr["g"][9:].astype(np.float32), tr["g"][1].numpy().astype(np.float16).astype(np.float32))


def test_paired_boot_and_gain_stats():
    import numpy as np
    a = np.arange(20, dtype=float)
    r = paired_boot(a + 1.0, a, n=2000)
    assert r["diff"] == pytest.approx(1.0) and r["lo"] == pytest.approx(1.0) and r["n_units"] == 20
    rc = paired_boot([1, 3, 2, 2], [0, 0, 0, 0], clusters=[0, 0, 1, 1], n=500)
    assert rc["n_units"] == 2 and rc["diff"] == pytest.approx(2.0)
    gs = gain_stats(torch.tensor([[0.0, 0.1, 0.2, 0.2]]), torch.ones(1, 4), 0.0, 0.2)
    assert gs["frac_at_gmax"] == 0.5 and gs["frac_at_gmin"] == 0.25 and gs["n"] == 4


# ------------------------------------------------------------------- other architectures (GPT-2, Qwen3.5)
def test_attachment_on_tiny_gpt2_through_common_load():
    from ca2.common import load
    model, tok, blocks, kinds = load(None, dtype="float32", device="cpu", tiny=True, tiny_arch="gpt2")
    ids = tokens()
    base = model(input_ids=ids, use_cache=False).logits
    zero = CA2Controller(D, "agree", g_min=0.0, g_max=0.0)
    att = CA2Attachment(model, zero, tap=1, taps=[NormTap()])
    assert torch.equal(forward(model, zero, ids), base.float())
    att.remove()
    ctrl = make(2, "agree_integral_release_ngram")
    att = CA2Attachment(model, ctrl, tap=1)
    lg1 = forward(model, ctrl, ids)
    lg2 = incremental(model, ctrl, ids, torch.ones_like(ids), 10)
    att.remove()
    assert not torch.allclose(lg1, base) and torch.allclose(lg1, lg2, atol=1e-4)


@pytest.fixture
def torch_gdn(monkeypatch):
    """Swap the hub-dispatched GDN kernels (Triton under flash-linear-attention, which fails on CPU) for their
    pure-torch originals; skip when this transformers has no Qwen3.5 or the wrapper exposes no original."""
    if not hasattr(transformers, "Qwen3_5TextConfig"):
        pytest.skip("transformers has no Qwen3.5 classes")
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        fn = getattr(M, name, None)
        if fn is None:
            pytest.skip(f"{name} not found")
        monkeypatch.setattr(M, name, getattr(fn, "__wrapped__", fn))
    model, _ = pl.build_tiny("qwen35", seed=0)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def test_hybrid_qwen35_causality_decode_and_zero_gain(torch_gdn):
    model = torch_gdn
    ids = tokens()
    base = model(input_ids=ids, use_cache=False).logits
    zero = CA2Controller(D, "agree", g_min=0.0, g_max=0.0)
    att = CA2Attachment(model, zero, tap=0, inject_before=2, taps=[NormTap()])
    assert torch.equal(forward(model, zero, ids), base.float())
    att.remove()
    ctrl = make(2, "agree_none_hidden_dose")
    att = CA2Attachment(model, ctrl, tap=0, inject_before=2)      # read after a GDN block, inject into one
    assert att.kinds[0] == att.kinds[2] == "linear"
    lg1 = forward(model, ctrl, ids)
    tr1 = ctrl.trace_tensors()
    ids2 = ids.clone()
    ids2[:, 19:] = (ids2[:, 19:] + 5) % 256
    lg2 = forward(model, ctrl, ids2)
    tr2 = ctrl.trace_tensors()
    lg3 = incremental(model, ctrl, ids, torch.ones_like(ids), 12)
    tr3 = ctrl.trace_tensors()
    att.remove()
    assert not torch.allclose(lg1, base.float())
    same(lg1[:, :19], lg2[:, :19])
    for k in ("g", "s", "r", "src/agree/a"):
        same(tr1[k][:, :19], tr2[k][:, :19])
        assert torch.allclose(tr1[k], tr3[k], atol=1e-5)
    assert torch.allclose(lg1, lg3, atol=1e-4)
