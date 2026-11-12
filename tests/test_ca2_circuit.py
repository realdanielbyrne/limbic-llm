"""CPU tests for experiments/ca2/circuit.py (round 13 gated memory circuit): causality (bitwise, fp32 and bf16-rounded
input), fixed-size streaming state, streaming = vectorised, and the mechanisms on toys."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from ca2.circuit import (CircuitConfig, DentateGate, EpisodicStore, FilterBank, RingBuffer, code_overlap,  # noqa: E402
                         comparator, context_tag, ema_scan, massive_mask)

B, T, D = 2, 300, 16
ABLATIONS = [dict(), dict(centre=False), dict(lag0=True), dict(slow_gated=False), dict(band_diff=False)]


def data(seed=0, massive_at=(37, 150)):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, T, D, generator=g) + 0.5
    for p in massive_at:
        h[:, p] *= 40.0                                            # massive activations
    sal = torch.rand(B, T, generator=g)
    return h, sal


def cfg(**kw):
    base = dict(sink=3, ring=64, gate_threshold=0.6, mean_tau=100.0)
    base.update(kw)
    return CircuitConfig(**base)


def test_ema_scan_matches_loop():
    torch.manual_seed(0)
    u = torch.randn(2, 77, 3)
    for a in (0.5, 0.9, math.exp(-1 / 1024)):
        y = ema_scan(u, a, chunk=16)
        ref = torch.zeros_like(u)
        acc = torch.zeros(2, 3)
        for t in range(77):
            acc = a * acc + u[:, t]
            ref[:, t] = acc
        assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5)


def test_massive_and_sink_excluded():
    h, _ = data()
    ex = massive_mask(h.norm(dim=-1), sink=3, ring=64, factor=10.0)
    assert ex[:, :3].all()
    assert ex[:, 37].all() and ex[:, 150].all()
    assert int(ex.sum()) == B * 5


@pytest.mark.parametrize("kw", ABLATIONS)
def test_stream_equals_vectorised_and_fixed_state(kw):
    h, sal = data()
    bank = FilterBank(cfg(**kw))
    vec = bank(h, sal)
    st = bank.stream(B, D)
    n0 = st.state_numel()
    for t in range(T):
        o = st.step(h[:, t], sal[:, t])
        assert st.state_numel() == n0                              # fixed-size state
        assert torch.equal(o["excluded"], vec.excluded[:, t])
        assert torch.allclose(o["bands"], vec.bands[:, t], atol=1e-5, rtol=1e-4), (kw, t)
        assert torch.allclose(o["outputs"], vec.outputs[:, t], atol=1e-5, rtol=1e-4)
        assert torch.allclose(o["strength"], vec.strength[:, t], atol=1e-4, rtol=1e-4)
        assert torch.allclose(o["mean"], vec.mean[:, t], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("kw", ABLATIONS)
@pytest.mark.parametrize("bf16", [False, True])
def test_filter_bank_strictly_causal_bitwise(kw, bf16):
    h, sal = data(1)
    if bf16:
        h = h.bfloat16().float()
    bank = FilterBank(cfg(**kw))
    ref = bank(h, sal)
    lag0 = kw.get("lag0", False)
    for t in (5, 64, 151, 299):
        h2, s2 = h.clone(), sal.clone()
        h2[:, t:] = torch.randn_like(h2[:, t:]) * 3
        if bf16:
            h2 = h2.bfloat16().float()
        s2[:, t:] = 1.0 - s2[:, t:]
        out = bank(h2, s2)
        upto = t if lag0 else t + 1                                # lag 0 excluded: the band at t ignores input t
        for name in ("bands", "outputs", "strength"):
            assert torch.equal(getattr(out, name)[:, :upto], getattr(ref, name)[:, :upto]), (name, t, kw)
        assert torch.equal(out.excluded[:, :t], ref.excluded[:, :t])
        assert torch.equal(out.mean[:, :t], ref.mean[:, :t])
        if not lag0 and t > 0:                                    # the band at t does change with input t − 1
            h3 = h.clone()
            h3[:, t - 1] += 1.0
            assert not torch.equal(bank(h3, sal).bands[:, t], ref.bands[:, t]) or bool(ref.excluded[0, t - 1])


def test_excluded_positions_do_not_enter_the_bands():
    h, sal = data()
    bank = FilterBank(cfg())
    ref = bank(h, sal)
    h2 = h.clone()
    g = torch.Generator().manual_seed(9)
    for p in (37, 150):                                           # another direction, the same (massive) norm
        v = torch.randn(B, D, generator=g)
        h2[:, p] = v / v.norm(dim=-1, keepdim=True) * h[:, p].norm(dim=-1, keepdim=True)
    out = bank(h2, sal)
    assert torch.equal(out.excluded, ref.excluded)
    assert torch.allclose(out.bands, ref.bands, atol=1e-6)


def test_slow_strength_counts_gated_events():
    T2 = 120
    h = torch.randn(1, T2, D)
    sal = torch.zeros(1, T2)
    ev = [10, 11, 40, 77, 100]
    sal[0, ev] = 1.0
    c = CircuitConfig(taus=(4, 1e9), n_fast=1, sink=0, ring=32, gate_threshold=0.5, massive=1e9)
    out = FilterBank(c)(h, sal)
    for t in range(T2):
        assert abs(float(out.strength[0, t, 0]) - sum(1 for e in ev if e < t)) < 1e-3
    a = math.exp(-1 / 16)                                          # a finite τ: a discounted count
    c2 = CircuitConfig(taus=(4, 16), n_fast=1, sink=0, ring=32, gate_threshold=0.5, massive=1e9)
    s2 = FilterBank(c2)(h, sal).strength[0, :, 0]
    for t in (12, 50, 119):
        assert abs(float(s2[t]) - sum(a ** (t - 1 - e) for e in ev if e < t)) < 1e-4
    ug = FilterBank(CircuitConfig(taus=(4, 1e9), n_fast=1, sink=0, ring=32, slow_gated=False, massive=1e9))(h)
    assert abs(float(ug.strength[0, 50, 0]) - 50) < 1e-3          # ungated: every position counts


def test_centring_removes_the_document_constant():
    v = torch.randn(D)
    h = v.expand(1, 200, D).clone() + 0.01 * torch.randn(1, 200, D)
    c = cfg(sink=0, mean_tau=50.0, slow_gated=False)
    out = FilterBank(c)(h)
    assert out.bands[0, 100:].norm(dim=-1).max() < 0.2 * math.sqrt(D)
    raw = FilterBank(cfg(sink=0, mean_tau=50.0, slow_gated=False, centre=False))(h)
    assert raw.bands[0, 100:].norm(dim=-1).min() > 0.8 * math.sqrt(D)


def test_band_outputs_are_differences_plus_slowest():
    h, sal = data()
    out = FilterBank(cfg())(h, sal)
    b = out.bands
    assert torch.allclose(out.outputs[:, :, 0], b[:, :, 0] - b[:, :, 1])
    assert torch.allclose(out.outputs[:, :, -1], b[:, :, -1])
    raw = FilterBank(cfg(band_diff=False))(h, sal)
    assert torch.equal(raw.outputs, raw.bands)


# ------------------------------------------------------------------------------------------------- dentate
@pytest.mark.parametrize("bf16", [False, True])
def test_dentate_causal_bitwise_and_sparse(bf16):
    gate = DentateGate(D, expand=4, k_frac=0.05, seed=3)
    e = torch.randn(1, 50, D)
    if bf16:
        e = e.bfloat16()
    idx, val, sal = gate(e)
    assert idx.shape == (1, 50, gate.k) and gate.k == round(0.05 * 4 * D)
    assert (val >= 0).all() and torch.allclose(sal, val.sum(-1))
    for t in (0, 17, 49):
        e2 = e.clone()
        e2[:, t:] = torch.randn_like(e2[:, t:].float()).to(e2.dtype)
        i2, v2, s2 = gate(e2)
        assert torch.equal(i2[:, :t], idx[:, :t]) and torch.equal(v2[:, :t], val[:, :t])
        assert torch.equal(s2[:, :t], sal[:, :t])
    # streaming (one position at a time) = vectorised; a fixed seed gives a fixed expansion
    for t in range(0, 50, 7):
        i1, v1, _ = gate(e[:, t])
        assert torch.equal(i1, idx[:, t]) and torch.allclose(v1, val[:, t])
    assert torch.equal(DentateGate(D, seed=3).R, gate.R)


def test_dentate_code_overlap_finds_the_same_word():
    gate = DentateGate(32, seed=0)
    torch.manual_seed(1)
    words = torch.randn(10, 32)
    e = words + 0.05 * torch.randn(10, 32)
    idx, val, _ = gate(e)
    qi, qv, _ = gate(words[4:5])
    qkey = gate.dense(qi, qv)[0]
    ov = code_overlap(qkey, idx, val)
    assert int(ov.argmax()) == 4


# -------------------------------------------------------------------------------------------- store, comparator
def test_store_recall_and_context():
    torch.manual_seed(0)
    gate = DentateGate(16, seed=0)
    Tn = 60
    early = torch.randn(Tn, 16)
    deep = torch.randn(Tn, 16)
    idx, val, sal = gate(early)
    outs = torch.randn(Tn, 5, 16)
    spans = [(a, a + 6) for a in range(0, 60, 6)]
    st = EpisodicStore.from_spans(spans, gate, idx, val, outs, deep)
    assert len(st) == 10 and st.pointer[3] == (18, 6)
    key = gate.dense(idx[24:30], val[24:30]).mean(0)              # the pooled key of span 4
    assert int(st.recall(key).argmax()) == 4
    ctx = outs[59]                                                # the tag of span 9
    s0 = st.recall(key * 0 + 1e-9, context_tag(ctx), lam=1.0)
    assert int(s0.argmax()) == 9
    cap = EpisodicStore.from_spans(spans, gate, idx, val, outs, deep, capacity=4)
    assert len(cap) == 4 and cap.pointer[0] == (36, 6)


def _states(n, Dd, g):
    """Raw deep states with a large shared component (as in a document), their RMS-normalised form and mean."""
    shared = torch.randn(Dd, generator=g) * 3
    raw = (shared + torch.randn(n, Dd, generator=g)) * 22.0        # raw scale: norm ≈ 1,000 at D = 64·…
    x = raw * torch.rsqrt(raw.pow(2).mean(-1, keepdim=True))
    return raw, x, x.mean(0)


def test_comparator_separates_the_matching_entry():
    g = torch.Generator().manual_seed(0)
    Dd, n = 64, 20
    a_match, a_other = [], []
    for trial in range(50):
        raw, states, mean = _states(n, Dd, g)
        j = trial % n
        d = raw[j] + 0.3 * 22 * torch.randn(Dd, generator=g)       # the current state, raw scale
        c = torch.randn(Dd, generator=g)
        a_match.append(comparator(d, states, j, c, mean)["a"])
        a_other.append(comparator(d, states, (j + 1) % n, c, mean)["a"])
    pairs = sum(x > y for x in a_match for y in a_other) / (len(a_match) * len(a_other))
    assert pairs > 0.95                                           # AUC of a_t for "top is the matching entry"
    assert abs(sum(a_other) / len(a_other)) < 0.2                 # the null is near 0
    cm = comparator(states[0], states, 0, c, mean)
    assert cm["out"].shape == (Dd,) and abs(float(cm["out"].norm()) - 1) < 1e-5
    assert cm["mismatch"].shape == (Dd,)


def test_comparator_raw_scale_equals_normalised():
    """Review bug (step 0): the runner passed the raw deep state (norm ≈ 1,100) against RMS-normalised states and
    mean (norm ≈ 50), so the centring did nothing. The comparator now RMS-normalises d itself."""
    g = torch.Generator().manual_seed(1)
    raw, states, mean = _states(30, 2560, g)
    d_raw = raw[7] + 5 * torch.randn(2560, generator=g)
    assert float(d_raw.norm()) > 500 and abs(float(states[0].norm()) - 2560 ** 0.5) < 1
    c = torch.randn(2560, generator=g)
    x = d_raw * torch.rsqrt(d_raw.pow(2).mean())
    a_raw = comparator(d_raw, states, 7, c, mean)["a"]
    a_norm = comparator(x, states, 7, c, mean)["a"]
    assert abs(a_raw - a_norm) < 1e-5
    # what the old code computed: unit(d_raw − mean) without normalising d; centring is then a no-op and the
    # shared component makes every cosine large, so the contrast shrinks
    from ca2.circuit import unit
    du = unit(d_raw - mean)
    ru = unit(states - mean)
    cos = ru @ du
    old = float(cos[7] - torch.cat([cos[:7], cos[8:]]).mean())
    assert abs(old - a_raw) > 1e-3


def test_ring_buffer_fixed_size():
    rb = RingBuffer(8, 1, 4)
    n0 = rb.state_numel()
    xs = torch.randn(20, 1, 4)
    for x in xs:
        rb.push(x)
    assert rb.state_numel() == n0
    assert torch.allclose(rb.get(19)[0], xs[19, 0] / xs[19, 0].pow(2).mean().sqrt(), atol=1e-5)
    with pytest.raises(KeyError):
        rb.get(5)
    assert rb.pool(14, 20).shape == (1, 4)


# ------------------------------------------------------------------------------- step 0b: new saliences
from ca2.circuit import key_novelty, norm_ratio, novelty, rms_normalise  # noqa: E402


def test_only_fast_bands_equal_the_full_bank():
    h, sal = data()
    full = FilterBank(cfg())(h, sal)
    fast = FilterBank(cfg())(h, None, only_fast=True)
    assert fast.bands.shape[2] == 3
    assert torch.equal(fast.bands, full.bands[:, :, :3]) and torch.equal(fast.mean, full.mean)
    assert torch.allclose(fast.x, rms_normalise(h))


@pytest.mark.parametrize("bf16", [False, True])
def test_novelty_causal_bitwise(bf16):
    h, _ = data(2)
    if bf16:
        h = h.bfloat16().float()
    bank = FilterBank(cfg())

    def nov(x):
        f = bank(x, None, only_fast=True)
        return torch.stack([novelty(f.x, f.mean, f.bands[:, :, k]) for k in range(3)])
    ref = nov(h)
    assert ref.shape == (3, B, T) and torch.isfinite(ref).all()
    for t in (4, 90, 299):
        h2 = h.clone()
        h2[:, t + 1:] = torch.randn_like(h2[:, t + 1:]) * 2
        if bf16:
            h2 = h2.bfloat16().float()
        assert torch.equal(nov(h2)[:, :, :t + 1], ref[:, :, :t + 1])


@pytest.mark.parametrize("bf16", [False, True])
def test_key_novelty_causal_and_exact(bf16):
    gate = DentateGate(D, seed=1)
    e = torch.randn(130, D)
    if bf16:
        e = e.bfloat16().float()
    idx, val, _ = gate(e)
    kn = key_novelty(gate, idx, val, tau=8.0, chunk=32)
    a = math.exp(-1 / 8)
    hist = torch.zeros(gate.U)
    dense = gate.dense(idx, val)
    for t in range(130):
        c = 0.0 if hist.norm() == 0 else float((dense[t] @ hist) / (dense[t].norm() * hist.norm()))
        assert abs(float(kn[t]) - (1 - c)) < 1e-4, t
        hist = a * hist + dense[t]
    assert kn[0] == 1.0
    for t in (5, 64, 100):
        e2 = e.clone()
        e2[t + 1:] = torch.randn_like(e2[t + 1:])
        i2, v2, _ = gate(e2)
        assert torch.equal(key_novelty(gate, i2, v2, tau=8.0, chunk=32)[:t + 1], kn[:t + 1])


def test_key_novelty_marks_a_new_word():
    gate = DentateGate(D, seed=2)
    w = torch.randn(D)
    e = w + 0.05 * torch.randn(100, D)
    e[70] = torch.randn(D)                                            # a different word
    idx, val, _ = gate(e)
    kn = key_novelty(gate, idx, val, tau=16.0)
    assert int(kn[10:].argmax()) + 10 == 70


@pytest.mark.parametrize("bf16", [False, True])
def test_early_norm_causal(bf16):
    e = torch.randn(1, 200, D)
    e[0, 120] *= 30
    if bf16:
        e = e.bfloat16().float()
    r = norm_ratio(e, sink=2, ring=32)
    assert int(r[0, 3:].argmax()) + 3 == 120 and r[0, 0] == 1.0
    for t in (10, 119, 150):
        e2 = e.clone()
        e2[:, t + 1:] *= 5
        assert torch.equal(norm_ratio(e2, sink=2, ring=32)[:, :t + 1], r[:, :t + 1])
