"""Round 12 GDN taps (experiments/ca2/gdn.py): recompute exactness, the mismatch copy of the reference rule,
full pass against prefill + cached decode, strict causality, the branch-gain law (own state unchanged,
layers above changed, −1 = zeroed branch), byte identity at zero gain, GDN sources and their refusal on a
model without GDN layers, HookedLM folding, and (GPU only) Triton against the torch kernels.

Plan tests 8 and 10. Tiny random Qwen3.5 hybrids in float32: ``pl.build_tiny("qwen35")`` (GDN, attention,
GDN, attention; 2 key heads and 4 value heads, so heads are repeated) and an equal-heads variant as in the
real model. With flash-linear-attention installed the hub wrapper dispatches the GDN kernels to Triton,
which fails on CPU; the ``gdn`` fixture swaps the module-level names for their pure-torch originals and
checks that the torch functions are the ones that run.
"""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("transformers has no Qwen3.5 classes", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments" / "alignment")]

import pretrained_loop as pl  # noqa: E402
from ca2.common import SteerSide, alignment  # noqa: E402
from ca2.controller import CA2Attachment, CA2Controller, NormTap  # noqa: E402
from ca2.gdn import (GDN_SOURCES, HEAD_KEYS, BranchLaw, GDNSource, GDNTaps, add_gdn_sources, gate_params,  # noqa: E402
                     gates, gdn_scan, hybrid_taps, linear_layers, recompute_qkv)
from ca2.sources import make_source  # noqa: E402

D, T, BOS = 64, 29, 256
A = alignment()
KERNELS = {"chunk": "torch_chunk_gated_delta_rule", "recurrent": "torch_recurrent_gated_delta_rule"}


# ------------------------------------------------------------------------------------------- fixtures
@pytest.fixture
def gdn(monkeypatch):
    """Force the pure-torch GDN kernels. Each module-level kernel name is replaced by a recording wrapper
    around its ``__wrapped__`` original; a cached prefill + one decode step then proves both torch
    functions are the ones the layer calls. Skips with a reason when that is not possible."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    calls: list = []
    for kind, name in KERNELS.items():
        fn = getattr(M, name, None)
        if fn is None:
            pytest.skip(f"{name} not found in modeling_qwen3_5")
        orig = getattr(fn, "__wrapped__", None)
        if orig is None or getattr(orig, "__module__", "") != M.__name__:
            pytest.skip(f"{name} exposes no pure-torch __wrapped__ original")

        def rec(*args, _orig=orig, _kind=kind, **kw):
            out = _orig(*args, **kw)
            names = ("query", "key", "value")
            got = dict(zip(names, args))
            got.update({k: kw[k] for k in names + ("g", "beta", "initial_state") if k in kw})
            calls.append(SimpleNamespace(kind=_kind, out=out, **{k: (None if v is None else v.detach().clone())
                                                                  for k, v in got.items()}))
            return out
        monkeypatch.setattr(M, name, rec)
    model = tiny()
    ids = tokens(1, 6)
    try:
        o = model(input_ids=ids[:, :5], use_cache=True)
        model(input_ids=ids[:, 5:], past_key_values=o.past_key_values, use_cache=True)
    except Exception as e:                                         # noqa: BLE001
        pytest.skip(f"the torch GDN path does not run here: {e!r}")
    kinds = {c.kind for c in calls}
    if kinds != {"chunk", "recurrent"}:
        pytest.skip(f"the swap did not select the torch path (calls seen: {sorted(kinds)})")
    calls.clear()
    return SimpleNamespace(M=M, calls=calls)


def tiny(equal_heads=False, dtype=torch.float32, seed=0):
    if equal_heads:
        torch.manual_seed(seed)
        cfg = transformers.Qwen3_5TextConfig(
            vocab_size=512, hidden_size=D, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, linear_num_key_heads=4, linear_num_value_heads=4,
            linear_key_head_dim=16, linear_value_head_dim=16, max_position_embeddings=1024,
            layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
            bos_token_id=BOS, eos_token_id=BOS, pad_token_id=BOS)
        model = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa")
    else:
        model, _ = pl.build_tiny("qwen35", seed=seed)
    model = model.to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _spread_gates(model, seed)
    return model


def _spread_gates(model, seed):
    """Random init leaves A_log/dt_bias at their defaults; spread them so heads differ (decays 0.3–0.99)."""
    g = torch.Generator().manual_seed(seed + 7)
    for blk in pl.find_blocks(model):
        lin = getattr(blk, "linear_attn", None)
        if lin is not None:
            H = lin.A_log.numel()
            lin.A_log.copy_(torch.empty(H).uniform_(-4.0, 0.5, generator=g).to(lin.A_log.dtype))
            lin.dt_bias.copy_(torch.empty(H).uniform_(-3.0, 1.0, generator=g).to(lin.dt_bias.dtype))


def tokens(B=2, n=T, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 256, (B, n), generator=g)
    ids[:, 0] = BOS
    return ids


def padded(B=2, n=T, pad=3, seed=0):
    ids = tokens(B, n, seed)
    mask = torch.ones(B, n, dtype=torch.long)
    ids[0, :pad], mask[0, :pad] = BOS, 0
    ids[0, pad] = BOS
    return ids, mask


def run(model, ids, mask=None, ctrl=None, cache=False):
    if ctrl is not None:
        ctrl.reset()
        ctrl.record = True
        ctrl.pending_mask = mask
    pos = A.probe.position_ids(mask) if mask is not None else None
    return model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=cache).logits.float()


def incremental(model, ids, mask, P, ctrl=None, taps=None, keep=None):
    """Prefill P columns with the cache, then one token per call; collects tap signals per call."""
    if ctrl is not None:
        ctrl.reset()
        ctrl.record = True
        ctrl.pending_mask = mask[:, :P]
    pos = A.probe.position_ids(mask)
    sig = []
    out = model(input_ids=ids[:, :P], attention_mask=mask[:, :P], position_ids=pos[:, :P], use_cache=True)
    logits, pkv = [out.logits.float()], out.past_key_values
    if taps is not None:
        sig.append(taps.signals())
    for t in range(P, ids.shape[1]):
        o = model(input_ids=ids[:, t:t + 1], attention_mask=mask[:, :t + 1], position_ids=pos[:, t:t + 1],
                  past_key_values=pkv, use_cache=True)
        logits.append(o.logits.float())
        pkv = o.past_key_values
        if taps is not None:
            sig.append(taps.signals())
    cat = {k: torch.cat([s[k] for s in sig], 1) for k in sig[0]} if sig else {}
    return torch.cat(logits, 1), cat, pkv


def same(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


def close(a, b, atol=1e-5):
    torch.testing.assert_close(a, b, rtol=0, atol=atol, equal_nan=True)


# ------------------------------------------------------------------------------------- 8. the kernels
def test_swap_selects_the_torch_kernels(gdn):
    model = tiny()
    ids = tokens(1, 7)
    o = model(input_ids=ids[:, :6], use_cache=True)
    model(input_ids=ids[:, 6:], past_key_values=o.past_key_values, use_cache=True)
    assert [c.kind for c in gdn.calls] == ["chunk", "chunk", "recurrent", "recurrent"]     # 2 GDN layers
    for name in KERNELS.values():
        assert getattr(gdn.M, name).__name__ == "rec"


def _ref_inputs(B=2, Tn=37, H=3, K=8, V=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, k = torch.randn(B, Tn, H, K, generator=g), torch.randn(B, Tn, H, K, generator=g)
    v = torch.randn(B, Tn, H, V, generator=g)
    a = -torch.rand(B, Tn, H, generator=g) * 1.5
    beta = torch.rand(B, Tn, H, generator=g)
    S0 = torch.randn(B, H, K, V, generator=g) * 0.5
    return q, k, v, a, beta, S0


def test_scan_is_bitwise_the_reference_recurrent_rule(gdn):
    ref, chunk = gdn.M.torch_recurrent_gated_delta_rule, gdn.M.torch_chunk_gated_delta_rule   # torch originals
    q, k, v, g, beta, S0 = _ref_inputs()
    for init in (None, S0):
        out, fin = ref(q, k, v, g=g, beta=beta, initial_state=init, output_final_state=True,
                       use_qk_l2norm_in_kernel=True)
        sc = gdn_scan(q, k, v, g, beta, init)
        assert torch.equal(sc.out, out) and torch.equal(sc.final_state, fin)
        co, cf = chunk(q, k, v, g=g, beta=beta, initial_state=init, output_final_state=True,
                       use_qk_l2norm_in_kernel=True)
        close(sc.out, co, 1e-5)
        close(sc.final_state, cf, 1e-5)


def test_mismatch_prediction_read_and_lag_match_states_from_the_reference(gdn):
    ref = gdn.M.torch_recurrent_gated_delta_rule
    q, k, v, g, beta, S0 = _ref_inputs(Tn=21)
    L = 4
    for init in (None, S0):
        sc = gdn_scan(q, k, v, g, beta, init, lag=L)
        states = [torch.zeros_like(S0) if init is None else init]           # S_{−1}, S_0, …
        for t in range(q.shape[1]):
            states.append(ref(q[:, :t + 1], k[:, :t + 1], v[:, :t + 1], g=g[:, :t + 1], beta=beta[:, :t + 1],
                              initial_state=init, output_final_state=True, use_qk_l2norm_in_kernel=True)[1])
        kh = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        qh = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        for t in range(q.shape[1]):
            rho = g[:, t].exp()[..., None]
            pred = rho * torch.einsum("bhkv,bhk->bhv", states[t], kh[:, t])
            close(sc.pred[:, t], pred, 1e-6)
            close(v[:, t] - sc.pred[:, t], v[:, t] - pred, 1e-6)                  # e_t, computed directly
            close(sc.read[:, t], rho * torch.einsum("bhkv,bhk->bhv", states[t], qh[:, t]), 1e-6)
            j = t - L + 1                                                         # index of S_{t−L} in states
            if j >= 0:
                lag = (g[:, t - L + 1: t + 1].sum(1).exp()[..., None]
                       * torch.einsum("bhkv,bhk->bhv", states[j], kh[:, t]))
                assert bool(sc.lag_ok[t])
                close(sc.pred_lag[:, t], lag, 1e-6)
            elif init is None:                                                    # earlier states are zero
                assert bool(sc.lag_ok[t]) and torch.equal(sc.pred_lag[:, t], torch.zeros_like(sc.pred_lag[:, t]))
            else:
                assert not bool(sc.lag_ok[t])                                     # unknown without history
        # split the sequence: carried history reproduces the one-shot scan bitwise
        a = gdn_scan(q[:, :9], k[:, :9], v[:, :9], g[:, :9], beta[:, :9], init, lag=L)
        b = gdn_scan(q[:, 9:], k[:, 9:], v[:, 9:], g[:, 9:], beta[:, 9:], a.final_state, L, a.history, a.g_history)
        for f in ("o", "pred", "read", "pred_lag", "snorm"):
            same(torch.cat([getattr(a, f), getattr(b, f)], 1), getattr(sc, f))
        same(b.final_state, sc.final_state)
        assert bool(b.lag_ok.all())


@pytest.mark.parametrize("equal_heads", [False, True])
def test_recomputed_qkv_gates_and_read_match_the_kernel_and_the_norm_prehook(gdn, equal_heads):
    model = tiny(equal_heads)
    tap = GDNTaps().attach_model(model)
    blocks = pl.find_blocks(model)
    seen = {}
    hs = [blocks[l].linear_attn.norm.register_forward_pre_hook(
        lambda m, a, l=l: seen.__setitem__(l, a[0].detach().clone())) for l in tap.layers]
    ids, mask = padded(n=T)
    for m in (None, mask):
        gdn.calls.clear()
        run(model, ids, m)
        assert len(gdn.calls) == len(tap.layers)
        for l, call in zip(tap.layers, gdn.calls):
            lin, cur = tap.lins[l], tap.cur[l]
            q, k, v = recompute_qkv(lin, cur["mixed"], cur["conv"])
            beta, g = gates(lin, cur["b"], cur["a"])
            for x, y in ((q, call.query), (k, call.key), (v, call.value), (beta, call.beta), (g, call.g)):
                assert torch.equal(x, y)
            o_model = seen[l].reshape(q.shape[0], q.shape[1], lin.num_v_heads, lin.head_v_dim)
            sc = gdn_scan(q, k, v, g, beta)
            close(sc.out, o_model, 1e-5)
            dev = tap.head_stat(l, "o_dev")
            real = (m if m is not None else torch.ones_like(ids)).bool()
            ok = real[..., None] & (tap.head_stat(l, "o_model") > 0)          # the pad id embeds to 0: o = 0
            assert ok.float().mean() > 0.8 and float(dev[ok].max()) < 1e-4
            assert torch.isnan(tap.head_stat(l, "fam")[~real]).all()
    # cached decode: the conv tail and S_{−1} come from the cache, cloned before the in-place update
    gdn.calls.clear()
    pos = A.probe.position_ids(mask)
    o = model(input_ids=ids[:, :12], attention_mask=mask[:, :12], position_ids=pos[:, :12], use_cache=True)
    pkv = o.past_key_values
    for t in range(12, 15):
        gdn.calls.clear()
        S_before = {l: pkv.layers[l].recurrent_states[0].clone() for l in tap.layers}
        o = model(input_ids=ids[:, t:t + 1], attention_mask=mask[:, :t + 1], position_ids=pos[:, t:t + 1],
                  past_key_values=pkv, use_cache=True)
        pkv = o.past_key_values
        for l, call in zip(tap.layers, gdn.calls):
            lin, cur = tap.lins[l], tap.cur[l]
            assert call.kind == "recurrent" and torch.equal(cur["S0"], S_before[l])
            q, k, v = recompute_qkv(lin, cur["mixed"], cur["conv"])
            for x, y in ((q, call.query), (k, call.key), (v, call.value)):
                assert torch.equal(x, y)
            assert float(tap.head_stat(l, "o_dev").max()) == 0.0                 # same rule, same inputs
    for h in hs:
        h.remove()
    tap.remove()


# --------------------------------------------------------------- full pass vs prefill + cached decode
@pytest.mark.parametrize("equal_heads", [False, True])
def test_signals_agree_between_full_pass_and_prefill_plus_decode(gdn, equal_heads):
    model = tiny(equal_heads)
    tap = GDNTaps(per_head=True).attach_model(model)
    ids, mask = padded(n=T)
    full = run(model, ids, mask)
    sig_full = tap.signals()
    run(model, ids, mask, cache=True)
    sig_cached_one = tap.signals()
    inc, sig_inc, _ = incremental(model, ids, mask, 10, taps=tap)
    tap.remove()
    assert set(sig_full) >= {*(HEAD_KEYS), "branch", "bgain"} and sig_full["fam"].shape == (2, T, len(tap.layers))
    assert sig_full["fam_head"].shape[-1] == tap.lins[tap.layers[0]].num_v_heads
    for k in sig_full:
        close(sig_cached_one[k], sig_full[k], 1e-6)
        close(sig_inc[k], sig_full[k], 1e-5 if k not in ("o_dev",) else 1e-3)
    close(inc, full, 1e-4)
    real = mask.bool() & (ids != BOS)                       # the pad/BOS id embeds to 0 in the tiny model: v = 0
    for k in ("beta", "ret", "fam", "fam_lag", "str", "lag_ok"):
        x = sig_full[k][real]
        assert torch.isfinite(x).all() and (x >= 0).all() and (x <= 1).all(), k
    assert (sig_full["lag_ok"][real] == 1).all()
    # the state fills: the prediction is 0 on the first token (empty state), familiarity positive later
    assert torch.equal(sig_full["pred"][1, 0], torch.zeros_like(sig_full["pred"][1, 0]))
    assert (sig_full["fam"][1, 5:] > 0).all()


# ------------------------------------------------------------------------------- causality (plan 2, 8)
def _ctrl(**kw):
    base = dict(source="gdn_mismatch_lag", g_max=0.5, g_abs=1.0, steer_from=2)
    base.update(kw)
    ctrl = CA2Controller(D, **base)
    add_gdn_sources(ctrl)
    return ctrl


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_strict_causality_bitwise_with_gdn_sources_and_branch_law(gdn, dtype):
    model = tiny(dtype=dtype)
    law = BranchLaw([2], mode="self", signal="fam_lag", g_min=-0.3, g_max=0.6, smooth="ema")
    ctrl = _ctrl(smooth="integral", kappa=2.0)
    tap = GDNTaps(per_head=True, law=law)
    att = CA2Attachment(model, ctrl, tap=1, inject_before=2, taps=[tap, NormTap()])
    ids = tokens()
    t0 = 17
    ids2 = ids.clone()
    ids2[:, t0:] = (ids2[:, t0:] + 11) % 256
    lg1 = run(model, ids, ctrl=ctrl)
    tr1 = ctrl.trace_tensors()
    lg2 = run(model, ids2, ctrl=ctrl)
    tr2 = ctrl.trace_tensors()
    att.remove()
    assert not torch.equal(lg1[:, t0:], lg2[:, t0:])
    same(lg1[:, :t0], lg2[:, :t0])
    gkeys = [k for k in tr1 if k.startswith("tap/gdn/") or k.startswith("src/gdn")]
    assert len(gkeys) > 40
    for k in tr1:
        same(tr1[k][:, :t0], tr2[k][:, :t0])
    assert (tr1["tap/gdn/bgain"][:, 2:, 1].abs() > 0).any()                     # the law acted (layer 2)


def test_controller_with_gdn_sources_prefill_decode_equals_one_pass(gdn):
    model = tiny()
    ids, mask = padded(n=T)

    def build():
        law = BranchLaw([2], mode="self", signal="beta", g_min=-0.2, g_max=0.4, smooth="ema")
        ctrl = _ctrl(source=GDNSource("err_lag", transform="zscore", name="gdn_err_z"))
        tap = GDNTaps(law=law)
        return ctrl, CA2Attachment(model, ctrl, tap=1, inject_before=2, taps=[tap])
    ctrl, att = build()
    lg1 = run(model, ids, mask, ctrl=ctrl)
    tr1 = ctrl.trace_tensors()
    att.remove()
    ctrl, att = build()
    lg2, _, _ = incremental(model, ids, mask, 9, ctrl=ctrl)
    tr2 = ctrl.trace_tensors()
    att.remove()
    close(lg2, lg1, 1e-4)
    keys = [k for k in tr1 if k.startswith(("tap/gdn/", "src/")) or k in ("g", "s", "r")]
    for k in keys:
        close(tr2[k], tr1[k], 2e-4 if k != "tap/gdn/o_dev" else 1e-2)
    for name in GDN_SOURCES:
        s = tr1[f"src/{name}/s"]
        assert (s >= 0).all() and (s <= 1).all() and torch.equal(s[mask == 0], torch.zeros_like(s[mask == 0]))


# ------------------------------------------------------------------------------- the branch-gain law
def test_branch_gain_leaves_its_own_state_and_changes_the_layers_above(gdn):
    model = tiny()
    ids = tokens()
    _, _, base = incremental(model, ids, torch.ones_like(ids), 20)
    tap = GDNTaps(law=BranchLaw([0], gain=0.8)).attach_model(model)
    _, sig, steered = incremental(model, ids, torch.ones_like(ids), 20, taps=tap)
    tap.remove()
    for kind in ("recurrent_states", "conv_states"):
        assert torch.equal(getattr(steered.layers[0], kind)[0], getattr(base.layers[0], kind)[0])
        assert not torch.allclose(getattr(steered.layers[2], kind)[0], getattr(base.layers[2], kind)[0])
    assert torch.allclose(sig["bgain"][..., 0], torch.full_like(sig["bgain"][..., 0], 0.8))
    assert torch.equal(sig["bgain"][..., 1], torch.zeros_like(sig["bgain"][..., 1]))


def test_branch_gain_minus_one_equals_zeroing_the_branch(gdn):
    model = tiny()
    ids, mask = padded()
    zeroed = copy.deepcopy(model)
    pl.find_blocks(zeroed)[2].linear_attn.out_proj.weight.zero_()
    ref = run(zeroed, ids, mask)
    tap = GDNTaps(law=BranchLaw([2], gain=-1.0)).attach_model(model)
    out = run(model, ids, mask)
    tap.remove()
    assert torch.equal(out, ref)
    assert not torch.equal(out, run(model, ids, mask))


def test_law_modes_ctrl_replay_and_self(gdn):
    model = tiny()
    ids = tokens()
    ctrl = CA2Controller(D, source="gdn_write", g_min=0.1, g_max=0.5, g_abs=1.0, steer_from=3)
    tap = GDNTaps(law=BranchLaw([2], mode="ctrl", scale=2.0))
    att = CA2Attachment(model, ctrl, tap=0, inject_before=1, taps=[tap])
    run(model, ids, ctrl=ctrl)
    tr = ctrl.trace_tensors()
    att.remove()
    same(tr["tap/gdn/bgain"][..., 1], 2.0 * tr["g_raw"])
    assert (tr["tap/gdn/bgain"][:, :3] == 0).all() and (tr["tap/gdn/bgain"][:, 3:, 1] > 0).all()
    with pytest.raises(ValueError):                                         # layer 0 runs before the gain
        CA2Attachment(model, CA2Controller(D, g_abs=1.0), tap=1, inject_before=2,
                      taps=[GDNTaps(law=BranchLaw([0], mode="ctrl"))])
    donor = torch.rand(2, 7, generator=torch.Generator().manual_seed(1)) - 0.5
    ctrl = CA2Controller(D, g_abs=1.0, steer_from=3)
    tap = GDNTaps(law=BranchLaw([0], mode="replay", donor=donor))
    att = CA2Attachment(model, ctrl, tap=1, taps=[tap])
    run(model, ids, ctrl=ctrl)
    tr = ctrl.trace_tensors()
    att.remove()
    step = tr["step"]
    want = donor.gather(1, step % 7) * tr["steer"]
    same(tr["tap/gdn/bgain"][..., 0], want)
    ctrl = CA2Controller(D, g_abs=1.0, steer_from=3)
    tap = GDNTaps(law=BranchLaw([0, 2], mode="self", signal="fam", invert=True, g_min=-0.4, g_max=0.3))
    att = CA2Attachment(model, ctrl, tap=1, taps=[tap])
    run(model, ids, ctrl=ctrl)
    tr = ctrl.trace_tensors()
    att.remove()
    bg, st = tr["tap/gdn/bgain"], tr["steer"][..., None].expand(-1, -1, 2)
    assert (bg[st > 0] >= -0.4 - 1e-7).all() and (bg[st > 0] <= 0.3 + 1e-7).all()
    assert (bg[st == 0] == 0).all() and bg[st > 0].std() > 0
    with pytest.raises(ValueError):
        BranchLaw([0], gain=-1.5)


# ------------------------------------------------------------------- zero gain is byte-identical (6)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_gain_with_every_tap_active_is_byte_identical(gdn, dtype):
    model = tiny(dtype=dtype)
    ids, mask = padded()
    base = run(model, ids, mask)
    for law in (BranchLaw([0, 2], gain=0.0), BranchLaw([0], mode="self", g_min=0.0, g_max=0.0)):
        ctrl = _ctrl(g_min=0.0, g_max=0.0)
        att = CA2Attachment(model, ctrl, tap=1, inject_before=2, taps=[GDNTaps(per_head=True, law=law), NormTap()])
        out = run(model, ids, mask, ctrl=ctrl)
        assert torch.equal(out, base)
        assert "tap/gdn/fam" in ctrl.trace[0]
        att.remove()
    assert torch.equal(run(model, ids, mask), base)


# ------------------------------------------------------------------------- refusal on non-hybrids
def test_gdn_taps_and_sources_refuse_a_model_without_gdn_layers():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=512, hidden_size=D, intermediate_size=128, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                   bos_token_id=BOS, eos_token_id=BOS, pad_token_id=BOS)
    model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    assert linear_layers(model) == [] and hybrid_taps(model) == []
    with pytest.raises(ValueError, match="linear-attention"):
        GDNTaps().attach_model(model)
    ctrl = CA2Controller(D, source="gdn_write")
    with pytest.raises(ValueError, match="linear-attention"):
        CA2Attachment(copy.deepcopy(model), ctrl, taps=[GDNTaps()])    # (a failed attach leaves its hooks)
    ctrl = CA2Controller(D, source="gdn_mismatch")
    att = CA2Attachment(model, ctrl)
    with pytest.raises(KeyError):
        run(model, tokens(), ctrl=ctrl)
    att.remove()


def test_sources_registry_shapes_and_parameters(gdn):
    for name in GDN_SOURCES:
        assert make_source(name).name == name
    with pytest.raises(ValueError):
        GDNSource("err")                                                    # not in [0, 1] without zscore
    model = tiny()
    assert len(hybrid_taps(model)) == 1 and set(gate_params(model)) == {0, 2}
    ctrl = _ctrl()
    tap = GDNTaps(per_head=["fam", "beta"])
    att = CA2Attachment(model, ctrl, tap=1, inject_before=2, taps=[tap])
    assert sum(p.numel() for p in ctrl.parameters()) == 0
    run(model, tokens(), ctrl=ctrl)
    tr = ctrl.trace_tensors()
    assert tr["tap/gdn/fam"].shape == (2, T, 2) and tr["tap/gdn/fam_head"].shape == (2, T, 2, 4)
    assert "tap/gdn/err_head" not in tr
    with pytest.raises(ValueError):                                         # layer 2 is not below M = 2
        GDNSource("beta", layers=[2])._layers(tap)
    n1 = ctrl.state_numel()
    run(model, tokens(seed=1), ctrl=ctrl)
    assert ctrl.state_numel() == n1
    o = model(input_ids=tokens()[:, :5], use_cache=True)                     # cache mode: the lag ring
    model(input_ids=tokens()[:, 5:6], past_key_values=o.past_key_values, use_cache=True)
    assert tap.state_numel() > 0
    att.remove()


# ------------------------------------------------------------------------------- HookedLM folding
def test_hookedlm_folded_gdn_layers_reset_every_window(gdn):
    model = tiny()
    W = 8
    ids = tokens(n=1 + 3 * W)
    base_lm = pl.HookedLM(model, None, layer=1, window=W, mode="block", window_layers="all", sink=1, pad_id=BOS)
    with base_lm.condition(W, False):
        base = base_lm(ids)
    for h in base_lm.handles:
        h.remove()
    ctrl = CA2Controller(D, source="gdn_familiar", sink_skip=0, steer_from=0, g_abs=1.0, g_min=0.0, g_max=0.0)
    ctrl.record = True
    tap = GDNTaps(law=BranchLaw([2], gain=0.0))
    side = SteerSide(ctrl, model=model, sink=1, taps=[tap], tap_layer=1)
    lm = pl.HookedLM(model, side, layer=1, window=W, mode="block", window_layers="all", sink=1, pad_id=BOS)
    assert torch.equal(lm(ids), base)
    tr = side.last_trace
    assert tr["tap/gdn/pred"].shape == (2, 3 * W, 2) and tr["src/gdn_familiar/s"].shape == (2, 3 * W)
    starts = torch.arange(0, 3 * W, W)
    assert (tr["tap/gdn/pred"][:, starts] == 0).all()                      # every window starts empty
    others = torch.ones(3 * W, dtype=torch.bool)
    others[starts] = False
    assert (tr["tap/gdn/pred"][:, others] > 0).all()
    for h in lm.handles:
        h.remove()
    side.remove()
    ctrl = CA2Controller(D, source="gdn_familiar", sink_skip=0, steer_from=0, g_abs=1.0, g_max=0.3)
    tap = GDNTaps(law=BranchLaw([2], gain=0.5))
    side = SteerSide(ctrl, model=model, sink=1, taps=[tap], tap_layer=1)
    lm = pl.HookedLM(model, side, layer=1, window=W, mode="block", window_layers="all", sink=1, pad_id=BOS)
    ctrl.record = True
    out = lm(ids)
    assert not torch.equal(out, base)
    bg = side.last_trace["tap/gdn/bgain"]
    assert torch.allclose(bg[..., 1], torch.full_like(bg[..., 1], 0.5)) and (bg[..., 0] == 0).all()
    for h in lm.handles:
        h.remove()
    side.remove()


# ------------------------------------------------------------------------------ 10. Triton vs torch
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (the Triton kernels)")
def test_triton_and_torch_kernels_agree_at_bf16_tolerance(monkeypatch):
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    model = tiny(equal_heads=True).to("cuda", torch.bfloat16)
    ids = tokens(n=70).cuda()
    tap = GDNTaps().attach_model(model)

    def once():
        lg = model(input_ids=ids, use_cache=False).logits.float()
        s = {k: v.clone() for k, v in tap.signals().items()}
        o = model(input_ids=ids[:, :60], use_cache=True)
        pkv = o.past_key_values
        for t in range(60, 70):
            pkv = model(input_ids=ids[:, t:t + 1], past_key_values=pkv, use_cache=True).past_key_values
        return lg, s, pkv
    lg_t, s_t, pkv_t = once()
    for name in KERNELS.values():
        fn = getattr(M, name)
        monkeypatch.setattr(M, name, getattr(fn, "__wrapped__", fn))
    lg_c, s_c, pkv_c = once()
    tap.remove()
    torch.testing.assert_close(lg_t, lg_c, rtol=0.05, atol=0.05)
    for l in linear_layers(model):
        torch.testing.assert_close(pkv_t.layers[l].recurrent_states[0], pkv_c.layers[l].recurrent_states[0],
                                   rtol=0.05, atol=0.05)
    for k in ("beta", "ret", "fam", "fam_lag", "str", "o_model"):
        torch.testing.assert_close(s_t[k], s_c[k], rtol=0.05, atol=0.05, equal_nan=True)
    assert float(s_t["o_dev"].nan_to_num(0).max()) < 0.1                  # recompute vs Triton output
