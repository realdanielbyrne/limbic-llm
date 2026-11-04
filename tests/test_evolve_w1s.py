"""Round-10 candidate W1s (experiments/evolve/w1s_slots.py): rules E1-E3, the streaming reference, the
slot mechanism on a one-hot toy, and the gates."""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from evolve.w1s_slots import (NEG, SlotLoop, W1sSide, gate_bias, latch, schmitt, schmitt_loop,  # noqa: E402
                              split_args)

D = 32
ARGS = type("A", (), {"time": True})()


def side(*flags, d=D, time=True):
    torch.manual_seed(0)
    return W1sSide(d, type("A", (), {"time": time})(), split_args(list(flags))[0]).eval()


def every(B, T, k, start):
    """Oracle onsets every k positions from ``start``."""
    o = torch.zeros(B, T, dtype=torch.bool)
    o[:, start::k] = True
    return o


def run(m, h, extra):
    """The side's output: (B, T, D) for additive, the reinstatement bias (B, T, T) otherwise."""
    if m.out_mode == "reinstate":
        return m.reinstate_bias(extra)
    m.extra = extra
    try:
        return m(h)[0]
    finally:
        m.extra = None


# --------------------------------------------------------------------------- E1
ALIGN = {"lex": "lex", "learned": "learned", "both": "lex"}


@pytest.mark.parametrize("gate", ["point", "hyst", "schmitt", "oracle"])
@pytest.mark.parametrize("store,out", [("slots", "additive"), ("slots", "reinstate"), ("superposed", "additive")])
@pytest.mark.parametrize("key", ["lex", "learned", "both"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_strictly_causal(gate, store, out, key, dtype):
    """E1: perturbing h and every tap from t0 on leaves everything before t0 bitwise unchanged (the
    output rows, or the reinstatement bias rows), also for bf16-rounded float32 input."""
    m = side(f"--w1s-gate={gate}", f"--w1s-store={store}", f"--w1s-out={out}", f"--w1s-key={key}",
             f"--w1s-align={ALIGN[key]}", "--w1s-K=4")
    with torch.no_grad():
        m.loop.gain_logit.fill_(2.0)
        m.time.out.weight.normal_(0, 0.1)
    T = 160
    m.oracle_onsets = every(2, T, 13, 5)
    h = torch.randn(2, T, D).to(dtype).float()
    ex = {3: torch.randn(2, T, D).to(dtype).float()}
    with torch.no_grad():
        y = run(m, h, ex)
        if store == "slots":
            assert (m.loop.last_diag["retained"] >= 0).any()      # the store is in use
        for t0 in (1, 7, 30, 69, 150):
            h2, e2 = h.clone(), {3: ex[3].clone()}
            h2[:, t0:] += torch.randn_like(h2[:, t0:])
            e2[3][:, t0:] += torch.randn_like(e2[3][:, t0:])
            y2 = run(m, h2, e2)
            assert torch.equal(y[:, :t0], y2[:, :t0]), t0
            assert not torch.equal(y, y2)


@pytest.mark.parametrize("flags", [["--w1s-align=none"], ["--w1s-key-taps=21"], ["--w1s-gate=dense", "--w1s-K=4"],
                                   ["--w1s-ctrl=shuffle"], ["--no-w1s-centre", "--w1s-delta=0.5"]])
def test_strictly_causal_other_arms(flags):
    """E1 for the remaining options: no alignment, everything at the main tap, the dense gate,
    the shuffle control, no centring with a recency-weighted priority."""
    m = side(*flags)
    with torch.no_grad():
        m.loop.gain_logit.fill_(2.0)
    T = 120
    h, ex = torch.randn(2, T, D), {3: torch.randn(2, T, D)}
    with torch.no_grad():
        y = run(m, h, ex)
        for t0 in (1, 30, 100):
            h2, e2 = h.clone(), {3: ex[3].clone()}
            h2[:, t0:] += torch.randn_like(h2[:, t0:])
            e2[3][:, t0:] += torch.randn_like(e2[3][:, t0:])
            assert torch.equal(y[:, :t0], run(m, h2, e2)[:, :t0]), t0


# --------------------------------------------------------------------------- E2
STREAM_CFGS = {
    "slots": ["--w1s-key=both", "--w1s-align=lex"],
    "learned+delta": ["--w1s-key=learned", "--w1s-align=learned", "--w1s-delta=0.05", "--no-w1s-centre"],
    "noalign": ["--w1s-key=lex", "--w1s-align=none", "--w1s-recency=0"],
    "reinstate": ["--w1s-out=reinstate", "--w1s-key=both"],
    "superposed": ["--w1s-store=superposed", "--w1s-key=both", "--w1s-align=lex"],
}


@pytest.mark.parametrize("gate", ["point", "hyst", "schmitt", "oracle", "dense"])
@pytest.mark.parametrize("cfg", list(STREAM_CFGS))
def test_stream_equals_parallel(gate, cfg):
    """E2: the parallel forward equals the pure-Python streaming store (K = 2, so slots are evicted):
    the read (or the reinstatement bias), the gate, the onsets, the slots held at every t and their
    read weights."""
    if gate == "dense" and cfg == "superposed":
        pytest.skip("the superposed store needs a sparse gate")
    m = side(f"--w1s-gate={gate}", "--w1s-K=2", *STREAM_CFGS[cfg], d=16).double()
    B, T = 2, 150
    torch.manual_seed(1)
    h, ex = torch.randn(B, T, 16, dtype=torch.float64), {3: torch.randn(B, T, 16, dtype=torch.float64)}
    oracle = every(B, T, 11, 3)
    with torch.no_grad():
        par = (m.loop.reinstate_bias(ex, oracle) if m.out_mode == "reinstate" else m.loop.read(h, ex, oracle))
    d = m.loop.last_diag
    ref = m.loop.stream(h, ex, oracle)
    torch.testing.assert_close(par, ref["rbias"] if m.out_mode == "reinstate" else ref["o"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(d["p"], ref["p"], atol=1e-6, rtol=1e-6)
    closes = T - m.loop.delay                                     # later candidates never close
    assert torch.equal(d["onsets"][:, :closes], ref["onsets"][:, :closes])
    assert ref["evictions"] > 0 or cfg == "superposed"
    if cfg == "superposed":
        return
    for b in range(B):
        for t in range(T):
            held = {int(s) for s in d["retained"][b, t] if s >= 0}
            assert held == ref["retained"][b][t], (b, t)
            w = {int(s): float(a) for s, a in zip(d["retained"][b, t], d["read_w"][b, t]) if s >= 0}
            for s, a in ref["read_w"][b][t].items():
                assert abs((w[s] if s is not None else float(d["read_w"][b, t, -1])) - a) < 1e-6, (b, t, s)


@pytest.mark.parametrize("flags", [["--w1s-pool=1"], ["--w1s-pool=1", "--w1s-out=reinstate"],
                                   ["--w1s-pre=5", "--w1s-win=6"], ["--w1s-pre=5", "--w1s-win=6", "--w1s-align=none"],
                                   ["--w1s-pre=5", "--w1s-win=6", "--w1s-store=superposed"]])
def test_stream_equals_parallel_at_windows_clipped_at_zero(flags):
    """E2 at the sequence start: a key head before position 0 (--w1s-pool ≤ --w1s-pre) is the zero key,
    and a window [0, 0] (--w1s-pre = --w1s-win − 1, onset 0) stores no successor, in both forms."""
    m = side("--w1s-gate=oracle", *flags, d=16).double()
    B, T = 2, 60
    torch.manual_seed(1)
    h, ex = torch.randn(B, T, 16, dtype=torch.float64), {3: torch.randn(B, T, 16, dtype=torch.float64)}
    oracle = torch.zeros(B, T, dtype=torch.bool)
    oracle[:, 0] = oracle[:, 30] = True
    with torch.no_grad():
        par = (m.loop.reinstate_bias(ex, oracle) if m.out_mode == "reinstate" else m.loop.read(h, ex, oracle))
    ref = m.loop.stream(h, ex, oracle)
    torch.testing.assert_close(par, ref["rbias"] if m.out_mode == "reinstate" else ref["o"], atol=1e-5, rtol=1e-5)


def test_state_size_is_independent_of_T():
    """E2: the streaming state (ring buffer, K slots, gate scalars) has the same size at T = 100 and 400."""
    m = side("--w1s-gate=point", "--w1s-K=2", d=16)
    sizes = []
    for T in (100, 400):
        torch.manual_seed(2)
        ref = m.loop.stream(torch.randn(1, T, 16), {3: torch.randn(1, T, 16)})
        assert ref["evictions"] > 0
        sizes.append(ref["state_floats"])
    assert sizes[0] == sizes[1]


# --------------------------------------------------------------------------- E3
@pytest.mark.parametrize("flags", [[], ["--w1s-out=reinstate"], ["--w1s-store=superposed"], ["--w1s-key=lex"],
                                   ["--w1s-gate=point", "--w1s-key-taps=21"]])
def test_parameter_budget_at_qwen_width(flags):
    """E3: at Qwen3-1.7B's width every arm stays within 4.4M parameters and 1M buffer floats."""
    m = W1sSide(2048, ARGS, split_args(flags)[0])
    assert sum(p.numel() for p in m.parameters()) <= 4_400_000
    assert sum(b.numel() for b in m.buffers()) <= 1_000_000


def test_streaming_state_at_qwen_width():
    """E3: the streaming state with all K = 16 slots full stays within 1M floats at width 2048."""
    m = side("--w1s-gate=oracle", d=2048)
    T = 200
    torch.manual_seed(0)
    ref = m.loop.stream(torch.randn(1, T, 2048), {3: torch.randn(1, T, 2048)}, every(1, T, 10, 2))
    assert len(ref["retained"][0][-1]) == 16
    assert ref["state_floats"] <= 1_000_000


# --------------------------------------------------------------------- mechanism
V = 64                                                          # one-hot vocabulary
NAMES, IS, SP, DIGITS, FILLER = [0, 1, 2, 3, 4, 5], 6, 7, list(range(8, 18)), list(range(20, 64))


def toy(q, stored=(0, 1, 2), seed=0):
    """Three facts [name, is, sp, d, d, d, d] separated by 60 filler tokens, 12 false windows in the
    filler (3 per gap), then the query [name_q, is, sp, d, d, d, d] (teacher-forced digits). Returns
    one-hot inputs (1, T, V), oracle onsets (1, T) and the positions: per fact, its name and digits."""
    g = torch.Generator().manual_seed(seed)
    fill = lambda k: [FILLER[i] for i in torch.randint(len(FILLER), (k,), generator=g)]
    digs = [[8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 8, 13]]   # distinct within each fact
    toks, onsets, facts = [], [], []
    for i, name in enumerate(stored):
        onsets += [len(toks) + 10, len(toks) + 22, len(toks) + 34]
        toks += fill(60)
        facts.append(dict(name=len(toks), d=[len(toks) + 3 + k for k in range(4)]))
        onsets.append(len(toks) + 3)
        toks += [name, IS, SP] + digs[i]
    onsets += [len(toks) + 10, len(toks) + 22, len(toks) + 34]
    toks += fill(60)
    qpos = len(toks)
    toks += [q, IS, SP] + (digs[stored.index(q)] if q in stored else [8, 9, 10, 11])
    x = torch.eye(V)[toks][None]
    o = torch.zeros(1, len(toks), dtype=torch.bool)
    o[0, onsets] = True
    return x, o, facts, qpos


TOY = ("--w1s-gate=oracle", "--w1s-pre=3", "--w1s-pool=4")   # a short head and pool keep the one-hot name salient


def spy_read(monkeypatch):
    """Capture SlotLoop._read's output r (the recalled value) on every call."""
    out = []
    real = SlotLoop._read

    def wrapped(self, f, st):
        r = real(self, f, st)
        out.append((r.detach(), st["v"].detach()))
        return r
    monkeypatch.setattr(SlotLoop, "_read", wrapped)
    return out


@pytest.mark.parametrize("q", [0, 1, 2])
def test_oracle_slots_return_the_queried_fact_at_every_rank(q, monkeypatch):
    """With the oracle gate, 3 facts and 12 false windows held in 16 slots, the lexical key picks the
    queried fact's slot and the alignment reads its next digit, for the first, middle and last fact:
    at the query's space (→ digit 1) and at each teacher-forced digit (→ the next one)."""
    reads = spy_read(monkeypatch)
    m = side(*TOY, "--w1s-key=lex", "--w1s-align=lex", d=V, time=False)
    x, o, facts, qpos = toy(q)
    m.oracle_onsets, m.extra = o, {3: x}
    with torch.no_grad():
        m(x)
    d = m.loop.last_diag
    (r, v), = reads
    fact = facts[q]
    for k, t in enumerate(range(qpos + 2, qpos + 6)):            # space, d1, d2, d3 predict d1..d4
        assert (d["retained"][0, t] >= 0).sum() == 15
        slot = d["retained"][0, t][d["read_w"][0, t, :-1].argmax()]
        assert slot == fact["d"][0] and d["read_w"][0, t, -1] < 0.05, (t, slot)
        assert d["align"][0, t] == fact["d"][k], (t, d["align"][0, t])
        cos = torch.nn.functional.normalize(v[0], dim=-1) @ torch.nn.functional.normalize(r[0, t], dim=0)
        assert cos[fact["d"][k]] > cos.max() - 1e-6, (t, cos.argmax())   # recalled: the next digit's value


def test_no_matching_slot_reads_the_null():
    """With the (learned) null threshold between the cosine the shared frame gives (0.42 here: "is",
    space) and the one a matching name gives (0.71), a query for a name that was never stored reads
    the null at the answer position, and a stored name reads its slot."""
    m = side(*TOY, "--w1s-key=lex", d=V, time=False)
    with torch.no_grad():
        m.loop.null_logit.fill_(0.55 * 30)
    for q, stored in ((4, (0, 1, 2)), (0, (0, 1, 2))):
        x, o, _, qpos = toy(q, stored)
        m.oracle_onsets, m.extra = o, {3: x}
        with torch.no_grad():
            m(x)
        null = m.loop.last_diag["read_w"][0, qpos + 2, -1]
        assert null > 0.9 if q == 4 else null < 0.05, (q, null)


def test_priority_delta_zero_keeps_the_strongest_and_large_delta_is_fifo():
    """K = 2 and three events of decreasing strength: δ = 0 keeps the two strongest (the first two),
    a large δ keeps the two most recent (FIFO)."""
    T, pos, w = 140, (20, 60, 100), (10.0, 5.0, 2.0)
    x = torch.eye(V)[[FILLER[0]] * T]
    for p_, t in zip(pos, (0, 1, 2)):
        x[p_] = torch.eye(V)[t]
    for delta, keep in ((0.0, {20, 60}), (1.0, {60, 100})):
        m = side("--w1s-gate=point", "--no-w1s-centre", "--w1s-K=2", f"--w1s-delta={delta}",
                 "--w1s-key-taps=3", d=V, time=False)
        with torch.no_grad():
            m.loop.gate_w[0].weight.zero_()
            m.loop.gate_w[0].weight[0, :3] = torch.tensor(w)
            m.extra = {3: x[None]}
            m(x[None])
        d = m.loop.last_diag
        assert d["onsets"][0].nonzero().flatten().tolist() == list(pos)
        assert {int(s) for s in d["retained"][0, -1] if s >= 0} == keep, delta


def test_superposed_store_loses_digits_within_a_fact(monkeypatch):
    """The same keys and values in one delta-rule state lose digits that the slots keep: within a fact,
    entries share the window key, and between facts the token codes interfere."""
    reads = spy_read(monkeypatch)
    hits = {}
    for store in ("slots", "superposed"):
        m = side(*TOY, "--w1s-key=lex", "--w1s-align=lex", f"--w1s-store={store}", d=V, time=False)
        x, o, facts, qpos = toy(1)
        m.oracle_onsets, m.extra = o, {3: x}
        with torch.no_grad():
            m(x)
        r, v = reads[-1]
        targets = torch.stack([v[0, p] for p in facts[1]["d"]])                     # the 4 digits' values
        cands = torch.nn.functional.normalize(v[0], dim=-1)
        rec = torch.nn.functional.normalize(r[0, qpos + 2:qpos + 6], dim=-1) @ cands.T   # vs every position
        hits[store] = sum(int(rec[k].argmax() == facts[1]["d"][k]) for k in range(4))
        assert targets.shape == (4, v.shape[-1])
    assert hits["slots"] == 4 and hits["superposed"] < 4, hits


@pytest.mark.parametrize("store", ["slots", "superposed"])
def test_empty_store_control_injects_nothing(store):
    """--w1s-ctrl empty: no slot is held (nothing is written), the read is the null, and the side adds
    nothing to h."""
    m = side("--w1s-ctrl=empty", f"--w1s-store={store}", time=False)
    with torch.no_grad():
        m.loop.gain_logit.fill_(2.0)
        h = torch.randn(2, 120, D)
        m.extra = {3: torch.randn(2, 120, D)}
        y, stats = m(h)
    assert torch.equal(y, h) and stats.inj_ratio.abs().max() == 0
    assert (m.loop.last_diag["retained"] < 0).all()             # the superposed store lists no closed windows
    if store == "slots":
        assert (m.loop.last_diag["read_w"][..., -1] == 1).all()


def test_shuffle_control_reads_another_rows_slots():
    """--w1s-ctrl shuffle: row b holds the slots computed from row b − 1."""
    m = side("--w1s-gate=oracle")
    h, ex = torch.randn(3, 120, D), {3: torch.randn(3, 120, D)}
    o = torch.zeros(3, 120, dtype=torch.bool)
    for b, k in enumerate((9, 13, 17)):
        o[b, 5::k] = True
    m.oracle_onsets = o
    with torch.no_grad():
        run(m, h, ex)
        base = m.loop.last_diag["retained"]
        m.loop.ctrl = "shuffle"
        run(m, h, ex)
    assert torch.equal(m.loop.last_diag["retained"], base.roll(1, 0))


def test_shuffle_control_needs_two_rows_and_additive_output():
    """--w1s-ctrl shuffle is refused where it would not move the store to another row: a one-row batch
    (the roll is the identity), and reinstatement (the shuffled windows reopen this row's own columns)."""
    m = side("--w1s-gate=oracle")
    m.ctrl = "shuffle"
    m.oracle_onsets = every(1, 120, 11, 3)
    with pytest.raises(ValueError, match="2 rows"):
        run(m, torch.randn(1, 120, D), {3: torch.randn(1, 120, D)})
    r = side("--w1s-out=reinstate")
    with pytest.raises(ValueError, match="reinstate"):
        r.ctrl = "shuffle"
    with pytest.raises(ValueError, match="reinstate"):
        side("--w1s-out=reinstate", "--w1s-ctrl=shuffle")
    r.ctrl = "empty"


def test_reinstate_bias_and_identity_forward():
    """Reinstatement: no slots → every entry NEG; held windows → finite on their columns only, rows
    with read weight; forward(h) returns h unchanged; the gate gets gradient through the bias."""
    m = side("--w1s-out=reinstate", "--w1s-gate=point")
    T = 120
    h, ex = torch.randn(2, T, D), {3: torch.randn(2, T, D)}
    y, stats = m(h)
    assert y is h and stats.inj_ratio.abs().max() == 0
    rb = m.reinstate_bias(ex)
    assert rb.shape == (2, T, T) and rb.dtype == torch.float32
    held = m.loop.last_diag["retained"]
    assert (rb[:, :17] == NEG).all()                             # nothing closed yet
    for b, t in ((0, 60), (1, 119)):
        cols = (rb[b, t] > NEG).nonzero().flatten()
        assert len(cols) > 0 and cols.max() <= t
        starts = held[b, t][held[b, t] >= 0]
        assert all(any(s - 6 <= c <= s + 17 for s in starts) for c in cols.tolist())
    m.loop.ctrl = "empty"
    assert (m.reinstate_bias(ex) == NEG).all()
    m.loop.ctrl = "none"
    rb = m.reinstate_bias(ex)
    rb.masked_fill(rb == NEG, 0).sum().backward()
    assert m.loop.gate_w[0].weight.grad.abs().sum() > 0


# -------------------------------------------------------------------------- gate
@pytest.mark.parametrize("gate", ["point", "hyst", "schmitt"])
@pytest.mark.parametrize("out", ["additive", "reinstate"])
def test_gradient_reaches_the_gate(gate, out):
    """The gate learns through log p_k in the read softmax (and the hysteresis window weights)."""
    m = side(f"--w1s-gate={gate}", f"--w1s-out={out}").train()
    h, ex = torch.randn(2, 200, D), {3: torch.randn(2, 200, D)}
    if out == "additive":
        m.extra = ex
        y = m(h)[0].pow(2).sum()
    else:
        rb = m.reinstate_bias(ex)
        y = rb.masked_fill(rb == NEG, 0).sum()
    y.backward()
    names = {"point": ["gate_a"], "schmitt": ["lam_logit"], "hyst": ["gate_a", "latch_c", "lam_logit"]}[gate]
    params = [m.loop.gate_w[0].weight] + [getattr(m.loop, n) for n in names]
    for p in params:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


@pytest.mark.parametrize("gate", ["point", "hyst", "schmitt", "dense"])
def test_initial_firing_density(gate):
    """At initialisation the gate's on-state averages 0.12 ± 0.05 (pitfall P35), whatever the input
    scale: the drive is standardised by its causal running deviation."""
    m = side(f"--w1s-gate={gate}", "--w1s-store=superposed" if gate != "dense" else "--w1s-key=lex", d=64)
    for scale in (1.0, 30.0):
        torch.manual_seed(3)
        h, ex = scale * torch.randn(4, 1024, 64), {3: scale * torch.randn(4, 1024, 64)}
        with torch.no_grad():
            run(m, h, ex)
        g = m.loop.last_diag["g"][:, 64:].mean().item()
        assert abs(g - 0.12) < 0.05, (scale, g)


def test_hysteresis_latch_holds():
    """Inside the bistable band (|drive| < 0.41 for c = 6) the latch keeps its state: off stays off,
    a positive pulse turns it on and it stays on, a negative pulse turns it off again."""
    v = torch.zeros(1, 60)
    v[0, 20:23], v[0, 40:43] = 3.0, -3.0
    g = latch(v, torch.tensor(6.0))[0]
    assert g[:20].max() < 0.1 and g[23:40].min() > 0.9 and g[43:].max() < 0.1
    pulse_free = torch.sigmoid(v)[0]                             # without the latch: no memory
    assert pulse_free[30] == 0.5


def test_latch_gradient_matches_autograd_loop():
    """The latch's hand-written backward equals autograd through the plain loop."""
    torch.manual_seed(0)
    u = torch.randn(3, 40, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(latch, (u, c))
    g, prev = [], torch.zeros(3, dtype=torch.float64)
    for t in range(40):
        prev = torch.sigmoid(u[:, t] + c * (prev - 0.5))
        g.append(prev)
    assert torch.allclose(torch.stack(g, 1), latch(u, c))


def test_schmitt_parallel_equals_loop():
    torch.manual_seed(0)
    z = torch.randn(4, 500).cumsum(-1) * 0.1
    for on, off in ((1.2, 0.8), (0.5, -0.5), (0.0, 0.0)):
        assert torch.equal(schmitt(z, on, off), schmitt_loop(z, on, off))


def test_gate_bias_calibration():
    """The hyst bias is the bisection result for density 0.12 with c = 6; point uses the probit rule."""
    assert abs(gate_bias("hyst", 0.12, 6.0, 0.8) + 0.84) < 0.05
    assert abs(gate_bias("point", 0.12) + 2.35) < 0.01


def test_missing_taps_fall_back_to_h_and_split_args():
    """Without ``extra`` every tap reads h (unit tests, CPU suite); split_args leaves other flags."""
    w, rest = split_args(["--w1s-K", "8", "--steps", "10", "--w1s-taps", "3,5", "--w1s-key-taps", "3,5,21"])
    assert w.w1s_K == 8 and rest == ["--steps", "10"]
    m = W1sSide(D, ARGS, w)
    assert m.taps == (3, 5) and m.loop.streams == [3, 5, "h"] and m.loop.lex == 3
    y, _ = m(torch.randn(1, 50, D))
    assert y.shape == (1, 50, D)
    with pytest.raises(ValueError):
        W1sSide(D, ARGS, split_args(["--w1s-out=reinstate", "--w1s-key-taps=3,21"])[0])
    with pytest.raises(ValueError):
        W1sSide(D, ARGS, split_args(["--w1s-out=reinstate", "--w1s-taps=5", "--w1s-L=4"])[0])


def test_ctrl_on_the_side_sets_the_loop():
    """``eval_by_query.py --w1s-ctrl`` sets ``side.ctrl``; it is the loop's control."""
    m = side()
    assert m.ctrl == "none"
    m.ctrl = "empty"
    assert m.loop.ctrl == "empty"
    with pytest.raises(ValueError):
        m.ctrl = "off"
