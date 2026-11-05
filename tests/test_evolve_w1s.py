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


# ===================================================================== round 11
# research/plans/round11_runbook.md §3: time as data (rope, ordinal), recency forms (log, ord, supr, supw),
# a learned alignment code, keeping M transitions per window. Each is tested on the round-10 recipe I1.
from evolve.w1s_slots import newer_sum, rope_freqs, rotate  # noqa: E402

I1 = ("--w1s-gate=point", "--no-w1s-centre", "--w1s-pre=0", "--w1s-key=learned", "--w1s-taps=")
R11 = {
    "log": ["--w1s-recency-form=log"],
    "ord": ["--w1s-recency-form=ord"],
    "supr": ["--w1s-recency-form=supr", "--w1s-sup-w=5"],
    "supw": ["--w1s-recency-form=supw", "--w1s-sup-theta=0.2", "--w1s-sup-gamma=8"],   # fires on random keys
    "none": ["--w1s-recency-form=none"],
    "rope": ["--w1s-time-feat=rope", "--w1s-rope-dims=8", "--w1s-rope-periods=4,4096"],
    "ordinal": ["--w1s-time-feat=ordinal"],
    "rope+linear": ["--w1s-time-feat=rope", "--w1s-recency-form=linear", "--w1s-rope-dims=8"],
    "code": ["--w1s-code=learned", "--w1s-dc=8"],
    "keep-first": ["--w1s-keep=5", "--w1s-keep-by=first"],
    "keep-value": ["--w1s-keep=5", "--w1s-keep-by=value"],
    "keep-learned": ["--w1s-keep=5", "--w1s-keep-by=learned"],
    "keep-ext": ["--w1s-keep=5", "--w1s-keep-by=external"],
    "keep-pair": ["--w1s-keep=5", "--w1s-keep-by=external"],
    "lexgate": ["--w1s-key=both", "--w1s-lex-init=-2", "--w1s-align-init=3.5"],
    "comb": ["--w1s-recency-form=supr", "--w1s-code=learned", "--w1s-dc=8", "--w1s-win=12", "--w1s-K=3",
             "--w1s-keep=6", "--w1s-keep-by=learned"],
}


def r11(name, *extra, d=D):
    m = side(*I1, *R11[name], *extra, d=d)
    with torch.no_grad():
        m.loop.gain_logit.fill_(2.0)
        if hasattr(m.loop, "ord_E"):                             # zero at init: give the rank feature a value
            m.loop.ord_E.normal_(0, 1.0)
    return m


def set_ext(m, name, B, T, seed=5):
    """External keep scores (diagnostic modes) for the keep-ext / keep-pair configurations."""
    g = torch.Generator().manual_seed(seed)
    m.keep_scores = m.keep_pair = None
    if name == "keep-ext":
        m.keep_scores = torch.rand(B, T, generator=g, dtype=torch.float64)
    elif name == "keep-pair":
        m.keep_pair = torch.rand(B, T, T, generator=g, dtype=torch.float64).tril()


@pytest.mark.parametrize("name", list(R11))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_round11_strictly_causal(name, dtype):
    """E1 for every round-11 flag on I1: perturbing h from t0 on leaves every output row before t0
    bitwise unchanged, for float32 and bf16-rounded input (K = 2: an eviction burst)."""
    m = r11(name, "--w1s-K=2")
    T = 200
    torch.manual_seed(0)
    h = torch.randn(2, T, D).to(dtype).float()
    set_ext(m, name, 2, T)
    if m.keep_scores is not None:
        m.keep_scores = m.keep_scores.float()
    if m.keep_pair is not None:
        m.keep_pair = m.keep_pair.float()
    with torch.no_grad():
        y = m(h)[0]
        assert (m.loop.last_diag["retained"] >= 0).any()
        for t0 in (1, 23, 60, 131, 199):
            h2 = h.clone()
            h2[:, t0:] += torch.randn_like(h2[:, t0:])
            kp = None
            if m.keep_pair is not None:                          # the future's attention changes too
                kp = m.keep_pair.clone()
                m.keep_pair = kp.clone()
                m.keep_pair[:, t0:] += 1.0
            if m.keep_scores is not None:
                ks = m.keep_scores.clone()
                m.keep_scores[:, t0:] += torch.rand_like(m.keep_scores[:, t0:])
            y2 = m(h2)[0]
            if kp is not None:
                m.keep_pair = kp
            if m.keep_scores is not None:
                m.keep_scores = ks
            assert torch.equal(y[:, :t0], y2[:, :t0]), (name, t0)
            assert not torch.equal(y, y2)


@pytest.mark.parametrize("name", list(R11))
@pytest.mark.parametrize("gate", ["point", "oracle"])
def test_round11_stream_equals_parallel(name, gate):
    """E2 for every round-11 flag on I1 (K = 2, so slots are evicted): the read, the slots held at every t
    and their read weights equal the independent streaming store's."""
    m = r11(name, "--w1s-K=2", f"--w1s-gate={gate}", d=16).double()
    B, T = 2, 170
    torch.manual_seed(1)
    h = torch.randn(B, T, 16, dtype=torch.float64)
    oracle = every(B, T, 11, 3)
    set_ext(m, name, B, T)
    with torch.no_grad():
        par = m.loop.read(h, None, oracle)
    d = m.loop.last_diag
    ref = m.loop.stream(h, None, oracle)
    torch.testing.assert_close(par, ref["o"], atol=1e-6, rtol=1e-6)
    assert ref["evictions"] > 0
    for b in range(B):
        for t in range(T):
            held = {int(s) for s in d["retained"][b, t] if s >= 0}
            assert held == ref["retained"][b][t], (b, t)
            w = {int(s): float(a) for s, a in zip(d["retained"][b, t], d["read_w"][b, t]) if s >= 0}
            for s, a in ref["read_w"][b][t].items():
                assert abs((w[s] if s is not None else float(d["read_w"][b, t, -1])) - a) < 1e-6, (name, b, t, s)


@pytest.mark.parametrize("name", ["keep-value", "keep-ext"])
def test_keep_ties_go_to_the_earlier_transition(name):
    """Top-M keep with tied scores keeps the earliest transitions, in both forms (no arbitrary topk order)."""
    m = r11(name, "--w1s-gate=oracle", d=16).double()
    B, T = 1, 120
    torch.manual_seed(2)
    h = torch.randn(B, T, 16, dtype=torch.float64)
    if name == "keep-value":
        with torch.no_grad():
            m.loop.wa.weight.zero_()                              # every value norm 0: all tied
    else:
        m.keep_scores = torch.ones(B, T, dtype=torch.float64)
    oracle = every(B, T, 30, 10)
    with torch.no_grad():
        par = m.loop.read(h, None, oracle)
    ref = m.loop.stream(h, None, oracle)
    torch.testing.assert_close(par, ref["o"], atol=1e-9, rtol=1e-9)
    ent = m.loop._store(m.loop._features(m.loop.norm(h), h, None, oracle))
    s0 = int(ent["s"][0, 0])
    kept = ent["mem"][0, 0].nonzero().flatten().tolist()
    assert kept == list(range(s0, s0 + 5)), kept


@pytest.mark.parametrize("name", ["linear", "log", "ord", "supw", "rope", "ordinal", "code", "keep-learned", "comb"])
def test_round11_state_is_independent_of_T_and_matches_the_analytic_count(name):
    """E2: the streaming state has the same size at T = 512, 2048 and 4096, and equals ``analytic_state``
    once the ring and the K slots are full."""
    m = side(*I1, *R11.get(name, []), "--w1s-K=2", d=16)
    sizes = []
    for T in (512, 2048, 4096):
        torch.manual_seed(2)
        ref = m.loop.stream(torch.randn(1, T, 16))
        assert ref["evictions"] > 0
        sizes.append(ref["state_floats"])
    assert sizes[0] == sizes[1] == sizes[2], sizes
    assert ref["state_components"] == m.loop.analytic_state(16), (ref["state_components"], m.loop.analytic_state(16))


def test_learned_code_and_keep_shrink_the_state_at_qwen_width():
    """At width 2048, the learned 64-d code drops the 2048-d codes and keys (A5), and n = 12 with K = 64 (A8)
    stays within E2's 1M floats; I1's own state is unchanged from round 10 (0.89M)."""
    i1 = W1sSide(2048, ARGS, split_args(list(I1))[0]).state_report()["state_floats"]
    a5 = W1sSide(2048, ARGS, split_args([*I1, "--w1s-code=learned", "--w1s-dc=64"])[0]).state_report()["state_floats"]
    a8 = W1sSide(2048, ARGS, split_args([*I1, "--w1s-recency-form=supr", "--w1s-code=learned", "--w1s-dc=64",
                                         "--w1s-win=12", "--w1s-K=64"])[0]).state_report()["state_floats"]
    assert 850_000 < i1 < 950_000, i1
    assert a5 * 5 <= i1, (a5, i1)
    assert a8 <= 1_000_000, a8


@pytest.mark.parametrize("name", list(R11))
def test_round11_parameter_budget_at_qwen_width(name):
    """E3: every round-11 arm stays within 4.4M parameters and 1M buffer floats at width 2048."""
    m = W1sSide(2048, ARGS, split_args([*I1, *R11[name]])[0])
    assert sum(p.numel() for p in m.parameters()) <= 4_400_000
    assert sum(b.numel() for b in m.buffers()) <= 1_000_000


@pytest.mark.parametrize("name", list(R11))
def test_round11_empty_store_injects_nothing(name):
    m = r11(name, "--w1s-ctrl=empty")
    T = 150
    set_ext(m, name, 2, T)
    for k in ("keep_scores", "keep_pair"):
        if getattr(m, k) is not None:
            setattr(m, k, getattr(m, k).float())
    h = torch.randn(2, T, D)
    with torch.no_grad():
        y, stats = m(h)
    assert torch.equal(y, h) and stats.inj_ratio.abs().max() == 0
    assert (m.loop.last_diag["read_w"][..., -1] == 1).all()


def _supw_store(alpha: float, T: int = 120):
    """Two oracle onsets with lexical keys u (at 10) and cos·u + sin·u⊥ (at 60): the older slot's tag at the end."""
    m = side("--w1s-gate=oracle", "--w1s-key=lex", "--w1s-pre=0", "--w1s-pool=4", "--w1s-recency-form=supw",
             d=V, time=False)
    x = torch.eye(V)[[FILLER[0]] * T].clone()
    u, w = torch.eye(V)[0], torch.eye(V)[1]
    x[10:14] = u
    x[60:64] = math.cos(alpha) * u + math.sin(alpha) * w
    o = torch.zeros(1, T, dtype=torch.bool)
    o[0, [10, 60]] = True
    with torch.no_grad():
        st = m.loop._store(m.loop._features(m.loop.norm(x[None]), x[None], {3: x[None]}, o))
    i = int((st["s"][0] == 10).nonzero()[0, 0])
    return float(st["sup"][0, -1, i])


import math  # noqa: E402


def test_supw_tag_is_monotone_in_the_match():
    """supw: the older slot's supersession tag rises monotonically as the newer key's cosine to it rises."""
    tags = [_supw_store(a) for a in (1.5, 1.0, 0.6, 0.4, 0.2, 0.0)]
    assert all(b > a for a, b in zip(tags, tags[1:])), tags
    assert tags[0] < 1e-6 and tags[-1] > 0.9, tags


def test_supw_tag_does_not_grow_with_length_on_random_filler():
    """supw with its default γ = 40, θ = 0.9: a slot's tag from ~900 unrelated candidates at T = 8192 stays
    what it is at T = 512 (sub-θ matches do not add up into a length-dependent penalty)."""
    m = side(*I1, "--w1s-recency-form=supw", d=32)
    tags = []
    for T in (512, 8192):
        torch.manual_seed(4)
        h = torch.randn(1, T, 32)
        with torch.no_grad():
            st = m.loop._store(m.loop._features(m.loop.norm(h), h, None, None))
        tags.append(float(st["sup"][0, -1, 0]))
    assert abs(tags[1] - tags[0]) < 0.05, tags


def test_rope_score_depends_on_the_lag_only():
    """Partial rotary time: rotate(q, t)·rotate(k, s) is unchanged by a common shift of t and s, the dims
    beyond rope-dims are untouched, and norms are kept."""
    fr = rope_freqs(8, "4,4096")
    torch.manual_seed(0)
    q, k = torch.randn(16, dtype=torch.float64), torch.randn(16, dtype=torch.float64)
    a = rotate(q, torch.tensor(700), fr) @ rotate(k, torch.tensor(130), fr)
    b = rotate(q, torch.tensor(7700), fr) @ rotate(k, torch.tensor(7130), fr)
    assert abs(a - b) < 1e-9
    r = rotate(q, torch.tensor(91), fr)
    assert torch.equal(r[8:], q[8:]) and abs(r.norm() - q.norm()) < 1e-12


def test_newer_sum_counts_newer_candidates():
    x = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    assert newer_sum(x).tolist() == [[2.0, 2.0, 1.0, 0.0]]


def test_recency_form_resolution_and_defaults():
    """Defaults reproduce round 10 (linear); a time feature defaults to no age penalty; explicit
    combinations are kept; the lexical-key gate scales the null so it does not outvote the keys."""
    assert split_args([])[0].w1s_recency_form == "linear"
    assert split_args(["--w1s-time-feat=rope"])[0].w1s_recency_form == "none"
    assert split_args(["--w1s-time-feat=rope", "--w1s-recency-form=linear"])[0].w1s_recency_form == "linear"
    m = side(*I1, "--w1s-key=both", "--w1s-lex-init=-2")
    assert abs(m.loop.log_beta_lex.item() + 2) < 1e-6
    assert abs(m.loop.null_logit.item() - 0.3 * (math.exp(-2) + 10)) < 1e-5
    with pytest.raises(ValueError):
        side("--w1s-key=lex", "--w1s-time-feat=rope")          # time features act on the learned key
    with pytest.raises(ValueError):
        side(*I1, "--w1s-code=learned", "--w1s-out=reinstate", "--w1s-taps=3", "--w1s-key-taps=3")


def test_round10_checkpoint_loads_under_the_defaults():
    """A round-10 I1 checkpoint loads into today's W1sSide with no missing or unexpected keys."""
    import json
    root = Path(__file__).resolve().parents[1] / "results/evolve/round10"
    js, pt = root / "r10_I1_s1.json", root / "r10_I1_s1.pt"
    if not pt.exists():
        pytest.skip("round-10 checkpoints are not in the repository (geof only)")
    rec = json.loads(js.read_text())
    flags = {**vars(split_args([])[0]), **rec["candidate"]["args"]}
    import argparse
    m = W1sSide(2048, argparse.Namespace(**rec["args"]), argparse.Namespace(**flags))
    missing, unexpected = m.load_state_dict(torch.load(pt, map_location="cpu"), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    assert m.loop.rform == "linear" and m.loop.code == "lex" and m.loop.keep == 0


def test_keep_scorer_toy_learns_through_the_module():
    """The keep-scorer toy (experiments/evolve/toy_keep.py) runs through the module's real path, and a
    few steps already move the held-out tag hit above chance (4/23) with a strong shared digit direction.
    The Stage-0 run (10 seeds, 300 steps) is a recorded check, not this test."""
    from evolve import toy_keep
    r = toy_keep.run(0, steps=60, lr=0.05, share=3.0, batch=2, d=64)
    assert r["curve"][0]["loss"] > 0 and r["heldout_tag_hit"] > 0.3, r


def test_round11_options_that_would_do_nothing_are_refused():
    """Review F0/F3: time features and recency forms on the superposed store (which reads no slots), and a
    keep rule without M, are errors rather than silent no-ops."""
    for flags in (["--w1s-store=superposed", "--w1s-time-feat=rope"], ["--w1s-store=superposed", "--w1s-recency-form=supw"],
                  [*I1, "--w1s-keep-by=value"], [*I1, "--w1s-keep-by=external"]):
        with pytest.raises(ValueError):
            side(*flags)
    side("--w1s-store=superposed")                               # the round-10 default still builds


def test_learned_keep_can_start_from_the_value_norm():
    """G0.4 → A6: --w1s-keep-init value scores w·x_{j+1} + b + ‖v_{j+1}‖, so with w = 0, b = 0 it keeps what
    value-norm keep keeps; parallel equals stream; it is refused for other keep rules."""
    common = (*I1, "--w1s-K=2", "--w1s-gate=oracle", "--w1s-keep=5")
    a = side(*common, "--w1s-keep-by=learned", "--w1s-keep-init=value", d=16).double()
    b = side(*common, "--w1s-keep-by=value", d=16).double()
    with torch.no_grad():
        a.loop.keep_w.weight.zero_()
        a.loop.keep_w.bias.zero_()
    B, T = 2, 150
    torch.manual_seed(3)
    h = torch.randn(B, T, 16, dtype=torch.float64)
    o = every(B, T, 11, 3)
    sa = a.loop._store(a.loop._features(a.loop.norm(h), h, None, o))
    sb = b.loop._store(b.loop._features(b.loop.norm(h), h, None, o))
    assert torch.equal(sa["mem"], sb["mem"])
    with torch.no_grad():
        a.loop.keep_w.weight.normal_(0, 0.3)
        par = a.loop.read(h, None, o)
    torch.testing.assert_close(par, a.loop.stream(h, None, o)["o"], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError):
        side(*I1, "--w1s-keep=5", "--w1s-keep-by=value", "--w1s-keep-init=value")


def test_alignment_gradients_stay_finite_when_matches_underflow():
    """Regression (CPU suite, 4 Nov 2026): with β at its clamp and stored codes far from the query,
    exp(β(cos − 1)) underflowed and the normaliser's backward overflowed to NaN. The read now shifts
    E by each row's best held match; gradients stay finite and parallel still equals stream."""
    m = side(*I1, "--w1s-K=2", "--w1s-gate=oracle", "--w1s-align-init=3.68", d=16).double()
    B, T = 2, 150
    torch.manual_seed(5)
    h = 5 * torch.randn(B, T, 16, dtype=torch.float64)
    o = every(B, T, 11, 3)
    torch.testing.assert_close(m.loop.read(h, None, o).detach(), m.loop.stream(h, None, o)["o"], atol=1e-6, rtol=1e-6)
    m32 = side(*I1, "--w1s-K=4", "--w1s-gate=oracle", "--w1s-align-init=3.68", "--w1s-code=learned", "--w1s-dc=4").train()
    with torch.no_grad():
        m32.loop.gain_logit.fill_(2.0)
    m32.oracle_onsets = every(2, 200, 9, 2)
    y = m32(30 * torch.randn(2, 200, D))[0].pow(2).sum()
    y.backward()
    assert all(torch.isfinite(p.grad).all() for p in m32.parameters() if p.grad is not None)


# ============================================================ A13: native keys (RV4.R3)
# The slot key is the mean of layer L's pre-rotary K over the window head, the query the mean of its Q over the
# last pool tokens, scored β_N cos. Here the captured activations are random tensors of their own width DN
# (≠ D), as ``taps.TapLM`` would put them in extra; the TapLM path is tested at the end on a tiny Qwen3.
NATIVE = ("--w1s-gate=point", "--no-w1s-centre", "--w1s-pre=0", "--w1s-key=native", "--w1s-taps=")   # I1 → A13
NL, DN = 21, 24
NATIVE_CFGS = {
    "lex-align": [],
    "learned-align": ["--w1s-align=learned"],
    "noalign": ["--w1s-align=none"],
    "supw": ["--w1s-recency-form=supw", "--w1s-sup-theta=0.2", "--w1s-sup-gamma=8"],
    "keep": ["--w1s-keep=5", "--w1s-keep-by=value"],
    "code": ["--w1s-code=learned", "--w1s-dc=8"],
}


def native_extra(B, T, dn=DN, seed=7, dtype=torch.float32, L=NL):
    g = torch.Generator().manual_seed(seed)
    return {("k", L): torch.randn(B, T, dn, generator=g, dtype=dtype),
            ("q", L): torch.randn(B, T, dn, generator=g, dtype=dtype)}


def native(*flags, d=D):
    m = side(*NATIVE, *flags, d=d)
    with torch.no_grad():
        m.loop.gain_logit.fill_(2.0)
    return m


def test_native_defaults_change_nothing():
    """Regression: with every default (and I1's flags) nothing of A13 exists: no attention taps, no β_N, the
    null logit and the parameter list as before, so outputs and checkpoints are those of round 10/11."""
    for flags in ([], list(I1), ["--w1s-out=reinstate"], ["--w1s-store=superposed"]):
        m = side(*flags)
        assert m.attn_taps == () and not m.loop.use_native and not hasattr(m.loop, "log_beta_native")
        assert all("native" not in k for k in m.state_dict())
    assert split_args([])[0].w1s_key == "both" and split_args([])[0].w1s_native_layer == 21
    m = side()
    assert m.loop.null_logit.item() == pytest.approx(0.3 * 40.0, abs=0)


@pytest.mark.parametrize("cfg", list(NATIVE_CFGS))
@pytest.mark.parametrize("gate", ["point", "oracle"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_strictly_causal(cfg, gate, dtype):
    """E1: perturbing h and the captured K and Q from t0 on leaves every output row before t0 bitwise
    unchanged, for float32 and bf16-rounded input (K = 2: evictions); the native read is live."""
    m = native(*NATIVE_CFGS[cfg], f"--w1s-gate={gate}", "--w1s-K=2")
    T = 200
    m.oracle_onsets = every(2, T, 13, 5)
    torch.manual_seed(0)
    h = torch.randn(2, T, D).to(dtype).float()
    ex = {k: v.to(dtype).float() for k, v in native_extra(2, T).items()}
    with torch.no_grad():
        y = run(m, h, ex)
        assert (m.loop.last_diag["retained"] >= 0).any()
        live = {k: v.clone() for k, v in ex.items()}
        live[("q", NL)][:, 100:] += 1.0                           # only the native query changes
        assert not torch.equal(run(m, h, live)[:, 100:], y[:, 100:])
        for t0 in (1, 23, 60, 131, 199):
            h2, e2 = h.clone(), {k: v.clone() for k, v in ex.items()}
            h2[:, t0:] += torch.randn_like(h2[:, t0:]).to(dtype).float()
            for v in e2.values():
                v[:, t0:] += torch.randn_like(v[:, t0:]).to(dtype).float()
            y2 = run(m, h2, e2)
            assert torch.equal(y[:, :t0], y2[:, :t0]), (cfg, t0)
            assert not torch.equal(y, y2)


@pytest.mark.parametrize("cfg", [*NATIVE_CFGS, "fallback"])
@pytest.mark.parametrize("gate", ["point", "oracle"])
def test_native_stream_equals_parallel(cfg, gate):
    """E2: the parallel read equals the independent streaming store (K = 2, so slots are evicted): the read,
    the slots held at every t and their read weights; ``fallback`` has no extra (K = Q = h)."""
    m = native(*NATIVE_CFGS.get(cfg, []), "--w1s-K=2", f"--w1s-gate={gate}", d=16).double()
    B, T = 2, 170
    torch.manual_seed(1)
    h = torch.randn(B, T, 16, dtype=torch.float64)
    ex = None if cfg == "fallback" else native_extra(B, T, dtype=torch.float64)
    oracle = every(B, T, 11, 3)
    with torch.no_grad():
        par = m.loop.read(h, ex, oracle)
    d = m.loop.last_diag
    ref = m.loop.stream(h, ex, oracle)
    torch.testing.assert_close(par, ref["o"], atol=1e-6, rtol=1e-6)
    assert ref["evictions"] > 0
    for b in range(B):
        for t in range(T):
            held = {int(s) for s in d["retained"][b, t] if s >= 0}
            assert held == ref["retained"][b][t], (b, t)
            w = {int(s): float(a) for s, a in zip(d["retained"][b, t], d["read_w"][b, t]) if s >= 0}
            for s, a in ref["read_w"][b][t].items():
                assert abs((w[s] if s is not None else float(d["read_w"][b, t, -1])) - a) < 1e-6, (cfg, b, t, s)


@pytest.mark.parametrize("cfg", ["lex-align", "noalign"])
def test_native_state_is_independent_of_T_and_matches_the_analytic_count(cfg):
    """E2: the streaming state (the ring now holds each position's K and Q, every slot its native key) has the
    same size at T = 512 and 2048, and equals ``analytic_state`` at the captured width DN."""
    m = native(*NATIVE_CFGS[cfg], "--w1s-K=2", d=16)
    sizes = []
    for T in (512, 2048):
        torch.manual_seed(2)
        ref = m.loop.stream(torch.randn(1, T, 16), native_extra(1, T))
        assert ref["evictions"] > 0
        sizes.append(ref["state_floats"])
    assert sizes[0] == sizes[1], sizes
    assert m.loop.native_dim == DN
    assert ref["state_components"] == m.loop.analytic_state(16), (ref["state_components"], m.loop.analytic_state(16))
    lex = 16 if cfg == "lex-align" else 0                         # the lexical alignment centres on lexical keys
    assert ref["state_components"]["slot_keys"] == 2 * (lex + DN + 2)   # K = 2 slots; no learned key


def test_native_budget_and_state_at_qwen_width():
    """E3 at width 2048 with Qwen3-1.7B's K/Q width (8 KV heads × 128 = 1024): one parameter more than the
    learned key's β and none of its 2048 × 128 map, so fewer parameters than I1; the full streaming state
    (K = 16 slots of n = 24) stays within 1M floats and equals the analytic count."""
    i1 = W1sSide(2048, ARGS, split_args(list(I1))[0])
    m = side(*NATIVE, "--w1s-gate=oracle", d=2048)
    n_par = sum(p.numel() for p in m.parameters())
    assert n_par <= 4_400_000 and n_par < sum(p.numel() for p in i1.parameters())
    assert sum(b.numel() for b in m.buffers()) <= 1_000_000
    T = 200
    torch.manual_seed(0)
    ref = m.loop.stream(torch.randn(1, T, 2048), native_extra(1, T, dn=1024), every(1, T, 10, 2))
    assert len(ref["retained"][0][-1]) == 16
    rep = m.state_report()
    assert ref["state_floats"] == rep["state_floats"] <= 1_000_000, (ref["state_floats"], rep)


def test_native_key_has_no_learned_map_and_beta_gets_gradient():
    """β_N (from log 10) is the native key's only parameter and is trained; the captured activations are detached;
    W_k exists only for a learned alignment."""
    m = native("--w1s-gate=oracle").train()
    assert m.loop.wk is None and abs(m.loop.log_beta_native.item() - math.log(10.0)) < 1e-6
    assert native("--w1s-align=learned").loop.wk is not None
    m.oracle_onsets = every(2, 150, 9, 2)
    ex = {k: v.requires_grad_() for k, v in native_extra(2, 150).items()}
    m.extra = ex
    m(torch.randn(2, 150, D))[0].pow(2).sum().backward()
    assert m.loop.log_beta_native.grad is not None and m.loop.log_beta_native.grad.abs() > 0
    assert all(v.grad is None for v in ex.values())


def test_native_controls_empty_and_shuffle():
    """--w1s-ctrl empty injects nothing; shuffle keys row b's slots by row b − 1's K (the query stays its own)."""
    m = native("--w1s-ctrl=empty")
    h, ex = torch.randn(2, 150, D), native_extra(2, 150)
    with torch.no_grad():
        y, stats = (setattr(m, "extra", ex), m(h))[1]
    assert torch.equal(y, h) and stats.inj_ratio.abs().max() == 0
    m = native("--w1s-gate=oracle")
    m.oracle_onsets = every(3, 120, 10, 2)
    h, ex = torch.randn(3, 120, D), native_extra(3, 120)
    with torch.no_grad():
        f, st = m.loop._run_store(m.loop.norm(h), h, ex, m.oracle_onsets)
        m.loop.ctrl = "shuffle"
        fs, sts = m.loop._run_store(m.loop.norm(h), h, ex, m.oracle_onsets)
    assert torch.equal(sts["KN"], st["KN"].roll(1, 0)) and torch.equal(fs["qn"], f["qn"])


def test_native_refusals_and_missing_activations():
    """Native keys key the additive slot read only; time features act on the learned key; an extra without
    layer L's K/Q is an error (a GPU run must not silently key on h); the side declares its attention tap."""
    for flags in (["--w1s-store=superposed"], ["--w1s-out=reinstate", "--w1s-taps=3", "--w1s-key-taps=3"],
                  ["--w1s-time-feat=rope"]):
        with pytest.raises(ValueError):
            side(*NATIVE, *flags)
    m = native("--w1s-native-layer=9")
    assert m.attn_taps == (9,) and m.taps == ()
    with pytest.raises(ValueError, match="layer 9"):
        run(m, torch.randn(1, 60, D), native_extra(1, 60))       # captured at 21, not 9
    with pytest.raises(ValueError, match="same shape"):
        run(m, torch.randn(1, 60, D), {("k", 9): torch.randn(1, 60, 8), ("q", 9): torch.randn(1, 60, 16)})


@pytest.mark.parametrize("q", [0, 1, 2])
def test_native_key_returns_the_queried_fact_through_the_module(q, monkeypatch):
    """Mechanism, through the module's real path: with K = Q = a fixed random projection of the token (what a
    frozen layer's shared projection would give), the oracle gate, 3 facts and 12 false windows, the native
    key picks the queried fact's slot at the answer position (no fitted key: β_N, the null and ω at their
    initial values), and the lexical alignment reads its next digit at each teacher-forced position. The
    selection is soft at initialisation (0.35–0.5 here): the cosine is not centred, so the frame every fact
    shares ("is", space) also scores the other facts; β_N is learned and can sharpen it."""
    reads = spy_read(monkeypatch)
    m = side(*TOY, "--w1s-key=native", "--w1s-align=lex", d=V, time=False)
    x, o, facts, qpos = toy(q)
    R = torch.randn(V, 48, generator=torch.Generator().manual_seed(3)) / 48 ** 0.5
    m.oracle_onsets, m.extra = o, {3: x, ("k", NL): x @ R, ("q", NL): x @ R}
    with torch.no_grad():
        m(x)
    d = m.loop.last_diag
    (r, v), = reads
    fact = facts[q]
    t = qpos + 2                                                  # the space after "is": the answer position
    slot = d["retained"][0, t][d["read_w"][0, t, :-1].argmax()]
    w = d["read_w"][0, t, :-1]
    assert slot == fact["d"][0] and w.max() > 0.3 and w.max() > 1.3 * w.topk(2).values[1], (slot, w)
    assert d["align"][0, t] == fact["d"][0]
    for k, t in enumerate(range(qpos + 2, qpos + 6)):            # space, d1, d2, d3 predict d1..d4 (where read)
        i = int(d["read_w"][0, t].argmax())
        if i < m.loop.K and d["retained"][0, t, i] == fact["d"][0]:
            assert d["align"][0, t] == fact["d"][k], (t, d["align"][0, t])


def _tiny_qwen3(dtype=torch.float32):
    import transformers
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, max_position_embeddings=1024, bos_token_id=256, eos_token_id=256,
        pad_token_id=256)
    model = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


TINY = (*NATIVE, "--w1s-native-layer=2", "--w1s-win=8", "--w1s-nms=2", "--w1s-pool=4")   # main tap 2 = L


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_through_taplm_on_a_tiny_qwen3(dtype):
    """End to end on a random 4-layer Qwen3 (GQA 4:2, head 16) with the main tap and L at block 2: the side gets
    K and Q of width 2 × 16 in extra; the windowed control is byte-identical to HookedLM; E1 holds bitwise at
    the LM level (bf16 too: the captured activations are bf16-rounded); and the parallel read on the captured
    activations equals the streaming store."""
    import pretrained_loop as pl
    from evolve.taps import TapLM
    W_, T_ = 8, 49
    s = side(*TINY, d=64)
    with torch.no_grad():
        s.loop.gain_logit.fill_(2.0)
    lm = TapLM(_tiny_qwen3(dtype), s, 2, W_, "block", "all", 1, pad_id=256)
    base = pl.HookedLM(_tiny_qwen3(dtype), s, 2, W_, "block", "all", 1, pad_id=256)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 256, (2, T_), generator=g)
    ids[:, 0] = 256
    seen = []
    real = s.loop.forward
    s.loop.forward = lambda h, extra=None, oracle=None: (seen.append((h.clone(), dict(extra))),
                                                         real(h, extra, oracle))[1]
    with torch.no_grad():
        for window in (W_, 0):
            with base.condition(window, False), lm.condition(window, False):
                assert torch.equal(base(ids), lm(ids))
        assert seen == []
        with lm.condition(W_, True):
            ref = lm(ids)
            h, ex = seen[-1]
            assert ex[("k", 2)].shape == ex[("q", 2)].shape == (2, T_ - 1, 32) and ex[("k", 2)].dtype == torch.float32
            assert s.loop.native_dim == 32 and s.extra is None
            for t0 in (1, 9, 30, 48):
                x = ids.clone()
                x[:, t0:] = torch.randint(0, 256, x[:, t0:].shape, generator=torch.Generator().manual_seed(t0))
                assert torch.equal(lm(x)[:, :t0], ref[:, :t0]), t0
    s.loop.forward = real
    m = s.double()
    hd, exd = h.double(), {k: v.double() for k, v in ex.items() if isinstance(k, tuple)}
    with torch.no_grad():
        par = m.loop.read(hd, exd)
    torch.testing.assert_close(par, m.loop.stream(hd, exd)["o"], atol=1e-6, rtol=1e-6)


def test_native_state_report_is_marked_provisional_before_the_width_is_seen():
    """Review R0: before a forward with captured K/Q, the native key's width is unknown and the state report
    (an overcount at the model width) says so; once the width is seen, the mark goes."""
    m = W1sSide(64, ARGS, split_args([*I1[:-1], "--w1s-key=native", "--w1s-taps="])[0])
    assert "provisional" in m.state_report()
    m.loop.native_dim = 32
    rep = m.state_report()
    assert "provisional" not in rep and rep["state_floats"] > 0
