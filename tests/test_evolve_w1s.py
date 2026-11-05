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


# ============================================================ gate track (round 11, Stage A2)
# research/plans/round11_combination_plan.md, "Gate track": the hysteresis and Schmitt gates on I1 (no centring) with
# every round-11 option, and --w1s-gate-norm init, which calibrates the uncentred drive's initial operating point.
GATE_TRACK = {name: R11[name] for name in ("none", "log", "ord", "supr", "supw", "rope", "ordinal", "keep-value",
                                           "keep-learned", "code")}
GATE_TRACK.update({"linear": [], "align-init": ["--w1s-align-init=3.5"], "native": ["--w1s-key=native"]})
GATE_NORM = {"none": [], "init": ["--w1s-gate-norm=init"], "seq": ["--w1s-gate-norm=seq"]}
SHIFT = {"none": 0.0, "init": 1.5, "seq": 1.5}                    # the norms are tested where they matter: an offset


def ln_like(B, T, d, a=0.0, scale=1.0, seed=0, dtype=torch.float32, c=0.0):
    """States shaped like an LM's before LayerNorm: a direction shared by every token (the same for every seed;
    token cosine about a²/(1 + a²) after LN) plus noise, times ``scale``; rounded through ``dtype``. ``c`` adds a
    per-sequence offset c·v_b (v_b a random direction of norm √d for each sequence, like a document's own mean;
    drawn from a generator of its own, so c = 0 gives the states of before)."""
    u = torch.randn(d, generator=torch.Generator().manual_seed(123), dtype=torch.float64)
    u = u / u.norm() * d ** 0.5
    x = a * u + torch.randn(B, T, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    if c:
        v = torch.randn(B, 1, d, generator=torch.Generator().manual_seed(seed + 7919), dtype=torch.float64)
        x = x + c * v / v.norm(dim=-1, keepdim=True) * d ** 0.5
    return (scale * x).to(dtype).to(torch.float64 if dtype == torch.float64 else torch.float32)


def gt(gate, cfg, norm, d=D):
    """I1 with the gate swapped (no centring), one round-11 option, K = 2 (evictions)."""
    m = side(*I1, *GATE_TRACK[cfg], f"--w1s-gate={gate}", "--w1s-K=2", *GATE_NORM[norm], d=d)
    with torch.no_grad():
        m.loop.gain_logit.fill_(2.0)
        if hasattr(m.loop, "ord_E"):
            m.loop.ord_E.normal_(0, 1.0)
    return m


def gt_extra(cfg, B, T, dtype=torch.float32):
    return ({k: v.to(dtype) for k, v in native_extra(B, T, dtype=torch.float64).items()} if cfg == "native"
            else None)


@pytest.mark.parametrize("norm", list(GATE_NORM))
@pytest.mark.parametrize("cfg", list(GATE_TRACK))
@pytest.mark.parametrize("gate", ["hyst", "schmitt"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gate_track_strictly_causal(gate, cfg, norm, dtype):
    """E1 for the hysteresis and Schmitt gates on I1 with each round-11 option (and the calibrated drive): perturbing
    h (and the native K/Q) from t0 on leaves every output row before t0 bitwise unchanged, for float32 and
    bf16-rounded input. μ̂ is calibrated on a batch of its own first, as the first training batch would."""
    m = gt(gate, cfg, norm)
    T = 200
    if norm == "init":
        with torch.no_grad():
            run(m, ln_like(2, T, D, SHIFT[norm], seed=9, dtype=dtype), gt_extra(cfg, 2, T))
        assert torch.isfinite(m.loop.gate_mu).all()
    h = ln_like(2, T, D, SHIFT[norm], seed=0, dtype=dtype)
    ex = gt_extra(cfg, 2, T)
    if ex is not None:
        ex = {k: v.to(dtype).float() for k, v in ex.items()}
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        y = run(m, h, ex)
        assert (m.loop.last_diag["retained"] >= 0).any()
        for t0 in (1, 23, 60, 131, 199):
            h2 = h.clone()
            h2[:, t0:] = (h2[:, t0:] + torch.randn(h2[:, t0:].shape, generator=g)).to(dtype).float()
            e2 = None
            if ex is not None:
                e2 = {k: v.clone() for k, v in ex.items()}
                for v in e2.values():
                    v[:, t0:] = (v[:, t0:] + torch.randn(v[:, t0:].shape, generator=g)).to(dtype).float()
            y2 = run(m, h2, e2)
            assert torch.equal(y[:, :t0], y2[:, :t0]), (gate, cfg, norm, t0)
            assert not torch.equal(y, y2)


@pytest.mark.parametrize("norm", list(GATE_NORM))
@pytest.mark.parametrize("cfg", list(GATE_TRACK))
@pytest.mark.parametrize("gate", ["hyst", "schmitt"])
def test_gate_track_stream_equals_parallel(gate, cfg, norm):
    """E2 for the hysteresis and Schmitt gates on I1 with each round-11 option (K = 2, so slots are evicted): the
    read, the gate, the onsets, the slots held at every t and their read weights equal the streaming store's."""
    m = gt(gate, cfg, norm, d=16).double()
    B, T = 2, 170
    if norm == "init":
        m.loop.calibrate_gate(m.loop._streams(ln_like(B, T, 16, SHIFT[norm], seed=9, dtype=torch.float64), None))
    h = ln_like(B, T, 16, SHIFT[norm], seed=1, dtype=torch.float64)
    ex = gt_extra(cfg, B, T, torch.float64)
    with torch.no_grad():
        par = m.loop.read(h, ex)
    d = m.loop.last_diag
    ref = m.loop.stream(h, ex)
    torch.testing.assert_close(par, ref["o"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(d["p"], ref["p"], atol=1e-6, rtol=1e-6)
    closes = T - m.loop.delay
    assert torch.equal(d["onsets"][:, :closes], ref["onsets"][:, :closes])
    assert ref["evictions"] > 0
    for b in range(B):
        for t in range(T):
            held = {int(s) for s in d["retained"][b, t] if s >= 0}
            assert held == ref["retained"][b][t], (b, t)
            w = {int(s): float(a) for s, a in zip(d["retained"][b, t], d["read_w"][b, t]) if s >= 0}
            for s, a in ref["read_w"][b][t].items():
                assert abs((w[s] if s is not None else float(d["read_w"][b, t, -1])) - a) < 1e-6, (cfg, b, t, s)


def _operating_point(gate, flags, seed, h, calib=None):
    """(onset density, mean on-state over t ≥ 64) of an untrained I1-like side (module seed ``seed``) on h;
    ``calib``, if given, is the first batch it sees (which sets μ̂ under --w1s-gate-norm init)."""
    torch.manual_seed(seed)
    m = W1sSide(64, ARGS, split_args([*I1, f"--w1s-gate={gate}", *flags])[0]).eval()
    with torch.no_grad():
        if calib is not None:
            run(m, calib, None)
        run(m, h, None)
    d = m.loop.last_diag
    return d["onsets"].float().mean().item(), d["g"][:, 64:].mean().item()


@pytest.mark.parametrize("gate", ["hyst", "schmitt"])
def test_uncentred_drive_starts_the_latch_saturated(gate):
    """The problem the calibration fixes. On LN-like states with a shared offset (token cosine 0.9), the uncentred
    drive w·x_t sits several running deviations from 0, with a sign and size set by the seed (w·u): the
    hysteresis latch and the Schmitt trigger start always on or always off (on-fraction < 0.03 or > 0.97) for
    half the seeds (already at cosine 0.5 for three of six), while the centred gate starts at its calibrated
    operating point (P35's starved gate)."""
    h = ln_like(4, 1024, 64, a=3.0, seed=3)
    on = [_operating_point(gate, [], s, h)[1] for s in range(6)]
    assert sum(g < 0.03 or g > 0.97 for g in on) >= 3 and max(on) - min(on) > 0.9, on
    centred = [_operating_point(gate, ["--w1s-centre"], s, h)[1] for s in range(6)]
    assert all(0.07 < g < 0.22 for g in centred), centred


@pytest.mark.parametrize("gate", ["point", "hyst", "schmitt"])
@pytest.mark.parametrize("a,scale", [(1.0, 1.0), (1.0, 30.0), (3.0, 1.0), (3.0, 30.0)])
def test_gate_norm_init_restores_the_initial_operating_point(gate, a, scale):
    """--w1s-gate-norm init: calibrated on a first batch of the same kind of states (another draw), the uncentred
    gate starts where the centred one does, per module seed: the same onset density (± 0.006) and on-fraction
    (± 0.04; 0.12 ± 0.03 for point, 0.05–0.25 for the latches, whose per-seed level varies with the centring
    too), over T = 1024, at input scales 1 and 30 and token cosines 0.5 and 0.9. Without it the point gate's
    mean p and the latch's on-fraction follow the seed."""
    h, calib = ln_like(4, 1024, 64, a, scale, seed=3), ln_like(4, 512, 64, a, scale, seed=99)
    for s in range(6):
        dens, on = _operating_point(gate, ["--w1s-gate-norm=init"], s, h, calib)
        dens_c, on_c = _operating_point(gate, ["--w1s-centre"], s, h)
        assert abs(dens - dens_c) < 0.006 and abs(on - on_c) < 0.04, (s, dens, dens_c, on, on_c)
        assert (abs(on - 0.12) < 0.03) if gate == "point" else (0.05 < on < 0.25), (s, on)
    raw = [_operating_point(gate, [], s, h)[1] for s in range(6)]
    assert max(raw) - min(raw) > 0.2, raw                         # uncalibrated: the seed decides


def _latch_by_sequence(gate, flags, seed, h, calib=None):
    """Per-sequence mean on-state over t ≥ 64 of an untrained I1-like side (module seed ``seed``), (B,)."""
    torch.manual_seed(seed)
    m = W1sSide(64, ARGS, split_args([*I1, f"--w1s-gate={gate}", *flags])[0]).eval()
    with torch.no_grad():
        if calib is not None:
            run(m, calib, None)
        run(m, h, None)
    return m.loop.last_diag["g"][:, 64:].mean(1)


@pytest.mark.parametrize("gate", ["hyst", "schmitt"])
def test_gate_norm_init_leaves_per_sequence_offsets(gate):
    """The limit of --w1s-gate-norm init (review 3, Q0). μ̂ removes the offset SHARED by sequences only. Give each
    sequence an offset of its own (c = 0.3: per-sequence means at cosine ≈ 0.99 to each other) and the
    hysteresis/Schmitt drive, which the leaky integral and z = u/sd(u) amplify about 3× in units of z, starts
    14-18% of the sequences (16 sequences × 6 module seeds) with the latch always on or always off (on-fraction
    < 0.03 or > 0.97; c = 0.6: about 40%), while the centred drive starts none so. Without the per-sequence
    offset (c = 0, what the restoration test uses) init saturates none either."""
    def saturated(flags, h, calib=None):
        g = torch.cat([_latch_by_sequence(gate, flags, s, h, calib) for s in range(6)])
        return int(((g < 0.03) | (g > 0.97)).sum()), len(g)
    h0, cal0 = ln_like(16, 1024, 64, 3.0, seed=3), ln_like(8, 512, 64, 3.0, seed=99)
    assert saturated(["--w1s-gate-norm=init"], h0, cal0)[0] == 0
    h, cal = ln_like(16, 1024, 64, 3.0, seed=3, c=0.3), ln_like(8, 512, 64, 3.0, seed=99, c=0.3)
    n_init, n = saturated(["--w1s-gate-norm=init"], h, cal)
    assert n_init >= 0.1 * n, (n_init, n)                        # the documented limit (13-17 of 96 here)
    assert saturated(["--w1s-centre"], h)[0] == 0


@pytest.mark.parametrize("gate", ["point", "hyst", "schmitt"])
@pytest.mark.parametrize("c", [0.0, 0.3, 1.0])
def test_gate_norm_seq_removes_per_sequence_offsets(gate, c):
    """--w1s-gate-norm seq removes each sequence's own offset (its mean over the first W = 64 tokens, frozen after):
    with per-sequence offsets up to c = 1 (and the shared one, a = 3) at most 5% of the sequences start below 0.03
    or above 0.97, and none is pinned (< 0.005 or > 0.995), whatever c; init saturates 14-18% at c = 0.3 and up
    to 2/3 at c = 1. The residue is the frozen estimate's own error (≈ σ/√W, amplified like any offset): 4 and 2 of
    96 sequences at on-fraction 0.01-0.025 for hyst and schmitt at W = 64, 1 and 0 at W = 128, none at 256.
    The initial operating point is the centred gate's per module seed (onset density ± 0.006; mean on-state
    0.12 ± 0.03 for point, 0.05–0.25 for the latches)."""
    h = ln_like(16, 1024, 64, 3.0, seed=3, c=c)
    gs = []
    for s in range(6):
        g = _latch_by_sequence(gate, ["--w1s-gate-norm=seq"], s, h)
        assert int(((g < 0.005) | (g > 0.995)).sum()) == 0, (s, g)
        gs.append(g)
        dens, on = _operating_point(gate, ["--w1s-gate-norm=seq"], s, h)
        dens_c, on_c = _operating_point(gate, ["--w1s-centre"], s, h)
        assert abs(dens - dens_c) < 0.006, (s, dens, dens_c)
        assert (abs(on - 0.12) < 0.03) if gate == "point" else (0.05 < on < 0.25), (s, on, on_c)
    g = torch.cat(gs)
    assert int(((g < 0.03) | (g > 0.97)).sum()) <= 0.05 * len(g), g


@pytest.mark.parametrize("gate", ["point", "hyst", "schmitt"])
def test_gate_norm_seq_is_the_first_window_mean_frozen(gate):
    """seq's drive is d_t = w·x_t − mean_{s ≤ min(t, W − 1)} w·x_s: at t = 0 it is 0 (the gate is neutral: point p_0 =
    σ(b)), before W it is the running-mean centring, and from W on the offset is frozen (``first_mean``; W = 1
    freezes at the first token)."""
    from evolve.w1s_slots import first_mean
    y = torch.randn(3, 50, dtype=torch.float64)
    for W in (1, 7, 50, 80):
        fm = first_mean(y, W)
        for t in (0, 3, 6, 7, 20, 49):
            torch.testing.assert_close(fm[:, t], y[:, :min(t + 1, W)].mean(1))
    m = side(*I1, f"--w1s-gate={gate}", "--w1s-gate-norm=seq", "--w1s-gate-norm-window=16")
    assert m.loop.gate_norm_window == 16 and "loop.gate_mu" not in m.state_dict()
    h = ln_like(2, 120, D, 2.0, seed=4)
    with torch.no_grad():
        xs = m.loop._streams(h, None)
        p, _ = m.loop._gate(xs, None)
        if gate == "point":
            torch.testing.assert_close(p[:, 0], torch.sigmoid(m.loop.gate_b).expand(2))
        ys = sum(lin(xs[k]).squeeze(-1) for lin, k in zip(m.loop.gate_w, m.loop.streams))
        assert torch.equal(first_mean(ys, 16)[:, 15:], first_mean(ys, 16)[:, 15:16].expand(2, 105))


def test_gate_norm_seq_state_refusals_and_defaults():
    """seq keeps one more streaming scalar (the first-W sum), has no buffer and no parameter of its own, and its
    flags are refused where they would do nothing: --w1s-gate-norm-window without seq, seq with the centring or
    with the oracle/dense gates. The window's default is 64 and changes nothing else."""
    assert split_args([])[0].w1s_gate_norm_window == 64
    for gate in ("hyst", "point"):
        m = side(*I1, f"--w1s-gate={gate}", "--w1s-gate-norm=seq", "--w1s-K=2", d=16)
        ref = side(*I1, f"--w1s-gate={gate}", "--w1s-K=2", d=16)
        assert list(m.state_dict()) == list(ref.state_dict())
        assert all(torch.equal(p, q) for p, q in zip(m.parameters(), ref.parameters()))
        sizes = []
        for T in (300, 900):
            torch.manual_seed(2)
            r = m.loop.stream(ln_like(1, T, 16, 1.5, seed=T))
            assert r["evictions"] > 0
            sizes.append(r["state_floats"])
        assert sizes[0] == sizes[1] and r["state_components"] == m.loop.analytic_state(16)
        assert r["state_components"]["gate"] == ref.loop.analytic_state(16)["gate"] + 1
        assert m.state_report()["gate_norm"] == {"mode": "seq", "window": 64}
    for flags in (["--w1s-gate-norm-window=32"], [*I1, "--w1s-gate-norm=init", "--w1s-gate-norm-window=32"],
                  [*I1, "--w1s-centre", "--w1s-gate-norm=seq"], [*I1, "--w1s-gate=oracle", "--w1s-gate-norm=seq"],
                  [*I1, "--w1s-gate=dense", "--w1s-gate-norm=seq"],
                  [*I1, "--w1s-gate-norm=seq", "--w1s-gate-norm-window=0"]):
        with pytest.raises(ValueError, match="gate-norm"):
            side(*flags)


@pytest.mark.parametrize("gate", ["point", "hyst", "schmitt"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("W", [1, 16, 64])
def test_gate_norm_seq_strictly_causal_with_every_gate(gate, dtype, W):
    """E1 for --w1s-gate-norm seq with the point, hyst and Schmitt gates (and per-sequence offsets): perturbing h from
    t0 on, before and after the window freezes, leaves every output row before t0 bitwise unchanged."""
    m = side(*I1, f"--w1s-gate={gate}", "--w1s-gate-norm=seq", f"--w1s-gate-norm-window={W}", "--w1s-K=2")
    with torch.no_grad():
        m.loop.gain_logit.fill_(2.0)
    T = 200
    h = ln_like(2, T, D, 1.5, seed=0, dtype=dtype, c=0.5)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        y = run(m, h, None)
        assert (m.loop.last_diag["retained"] >= 0).any()
        for t0 in (1, 10, 16, 23, 64, 65, 131, 199):
            h2 = h.clone()
            h2[:, t0:] = (h2[:, t0:] + torch.randn(h2[:, t0:].shape, generator=g)).to(dtype).float()
            y2 = run(m, h2, None)
            assert torch.equal(y[:, :t0], y2[:, :t0]), (gate, W, t0)
            assert not torch.equal(y, y2)


@pytest.mark.parametrize("gate", ["point", "hyst", "schmitt"])
@pytest.mark.parametrize("W", [1, 16, 64])
def test_gate_norm_seq_stream_equals_parallel_with_every_gate(gate, W):
    """E2 for --w1s-gate-norm seq with every gate: the read, p, the onsets, the slots held at every t and their read
    weights equal the streaming store's (K = 2: evictions)."""
    m = side(*I1, f"--w1s-gate={gate}", "--w1s-gate-norm=seq", f"--w1s-gate-norm-window={W}", "--w1s-K=2",
             d=16).double()
    with torch.no_grad():
        m.loop.gain_logit.fill_(2.0)
    B, T = 2, 170
    h = ln_like(B, T, 16, 1.5, seed=1, dtype=torch.float64, c=0.5)
    with torch.no_grad():
        par = m.loop.read(h, None)
    d = m.loop.last_diag
    ref = m.loop.stream(h, None)
    torch.testing.assert_close(par, ref["o"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(d["p"], ref["p"], atol=1e-6, rtol=1e-6)
    closes = T - m.loop.delay
    assert torch.equal(d["onsets"][:, :closes], ref["onsets"][:, :closes]) and ref["evictions"] > 0
    for b in range(B):
        for t in range(T):
            assert {int(s) for s in d["retained"][b, t] if s >= 0} == ref["retained"][b][t], (b, t)
            w = {int(s): float(a) for s, a in zip(d["retained"][b, t], d["read_w"][b, t]) if s >= 0}
            for s, a in ref["read_w"][b][t].items():
                assert abs((w[s] if s is not None else float(d["read_w"][b, t, -1])) - a) < 1e-6, (b, t, s)


@pytest.mark.parametrize("gate", ["point", "hyst", "schmitt"])
def test_gate_norm_init_has_no_saturated_onset_at_t0(gate):
    """Review 3, Q1: at t = 0 the running variance is 0, so the 1e-3 floor made the uncentred drive's argument
    ≈ 1000·d_0: p_0 was 0 or 1, and when 1 the slot at s = 0 had the top priority and, with δ = 0, was never
    evicted (init: ≈ 57 of 96 sequences). Under --w1s-gate-norm init the running variances are now seeded with the
    calibrated prior σ̂², so position 0 is an ordinary position: no sequence (of 96, 6 module seeds) has an onset at
    t = 0 with p > 0.9 (before: 55-59), p_0 is never saturated (point, hyst: none above 0.99 or below 1e-3), and a
    slot from t = 0 is held at the end in at most 10% of the sequences (7 and 3 of 96 for point and schmitt, about
    the rate of any early position; before: 57-59). seq has d_0 = 0 (neutral) by construction."""
    h, cal = ln_like(16, 512, 64, 3.0, seed=3), ln_like(8, 512, 64, 3.0, seed=99)
    for norm in ("init", "seq"):
        hot = held = 0
        p0 = []
        for s in range(6):
            torch.manual_seed(s)
            m = W1sSide(64, ARGS, split_args([*I1, f"--w1s-gate={gate}", f"--w1s-gate-norm={norm}"])[0]).eval()
            with torch.no_grad():
                if norm == "init":
                    run(m, cal, None)
                run(m, h, None)
            dg = m.loop.last_diag
            hot += int(((dg["p"][:, 0] > 0.9) & dg["onsets"][:, 0]).sum())
            held += int((dg["retained"][:, -1] == 0).any(-1).sum())
            p0.append(dg["p"][:, 0])
        p0 = torch.cat(p0)
        assert hot == 0 and held <= 0.1 * len(p0), (norm, hot, held)
        assert (p0 <= 0.99).all() and (gate == "schmitt" or (p0 >= 1e-3).all()), (norm, p0)


def test_running_sd_prior():
    """The prior seeds the running variance with n0 pseudo-positions: sd_0 = √(n0/(n0 + 1)·prior), and it fades as
    n0/(n0 + t + 1); without it the function is the round-10 one, bitwise."""
    from evolve.w1s_slots import GATE_PRIOR_N, running_sd
    u = torch.randn(2, 400, dtype=torch.float64) * 3 + 1
    plain = running_sd(u)
    m, m2 = u.cumsum(1) / torch.arange(1, 401), (u * u).cumsum(1) / torch.arange(1, 401)
    assert torch.equal(plain, ((m2 - m * m).clamp_min(0) + 1e-6).sqrt())
    sd = running_sd(u, torch.tensor(4.0, dtype=torch.float64))
    torch.testing.assert_close(sd[:, 0], torch.full((2,), (GATE_PRIOR_N / (GATE_PRIOR_N + 1) * 4.0 + 1e-6) ** 0.5,
                                                    dtype=torch.float64))
    assert (sd[:, -1] / plain[:, -1] - 1).abs().max() < 0.03


def test_gate_norm_init_records_where_mu_came_from_and_refuses_text_batches():
    """Review 3, Q4: state_report (run.py's w1s_state) says where μ̂ came from: the first batch seen (its size,
    whether training, how many calls ran before: 0 in training is the first training step), an explicit
    calibration, a checkpoint, or nothing yet. A batch the caller marks side.batch_kind = "distill" (an A14 text
    window) is refused for calibration; once μ̂ is set, such a batch runs and leaves μ̂ alone. Defaults report
    nothing new."""
    assert "gate_norm" not in side(*I1).state_report() and "align_stream" not in side(*I1).state_report()
    m = side(*I1, "--w1s-gate=hyst", "--w1s-gate-norm=init")
    assert m.state_report()["gate_norm"]["mu_source"] == {"source": "not calibrated"}
    h = ln_like(2, 150, D, 1.5, seed=1)
    m.train()
    m.batch_kind = "distill"
    with pytest.raises(RuntimeError, match="distill"):
        run(m, h, None)
    assert torch.isnan(m.loop.gate_mu).all()
    m.batch_kind = None
    run(m, h, None)
    src = m.state_report()["gate_norm"]
    assert src["mu_source"] == {"source": "first batch seen", "batch": 2, "positions": 150, "training": True,
                                "calls_before": 1, "kind": "unknown"}, src
    assert src["prior_var"] > 0 and len(src["mu_norm"]) == 1
    mu = m.loop.gate_mu.clone()
    m.batch_kind = "distill"
    run(m, ln_like(2, 150, D, -2.0, seed=2), None)
    assert torch.equal(m.loop.gate_mu, mu)
    fresh = side(*I1, "--w1s-gate=hyst", "--w1s-gate-norm=init")
    fresh.load_state_dict(m.state_dict())
    assert fresh.state_report()["gate_norm"]["mu_source"]["source"].startswith("loaded")
    fresh.batch_kind = "needle"
    fresh.loop.calibrate_gate(fresh.loop._streams(h, None))
    assert fresh.state_report()["gate_norm"]["mu_source"]["source"] == "explicit"
    assert fresh.state_report()["gate_norm"]["mu_source"]["kind"] == "needle"


def test_gate_norm_defaults_change_nothing_and_refusals():
    """Regression: by default (and with I1's flags) there is no μ̂ and no new key, so outputs and round-10/11
    checkpoints are unchanged; init adds one buffer per gate stream (no parameter, NaN until calibrated) and is
    refused where it would do nothing or double the centring."""
    assert split_args([])[0].w1s_gate_norm == "none"
    for flags in ([], list(I1), [*I1, "--w1s-gate=hyst"], ["--w1s-out=reinstate"], ["--w1s-store=superposed"]):
        m = side(*flags)
        assert m.loop.gate_norm == "none" and "loop.gate_mu" not in m.state_dict()
    m = side(*I1, "--w1s-gate=hyst", "--w1s-gate-norm=init")
    ref = side(*I1, "--w1s-gate=hyst")
    assert set(m.state_dict()) - set(ref.state_dict()) == {"loop.gate_mu"}
    assert [k for k, _ in m.named_parameters()] == [k for k, _ in ref.named_parameters()]
    assert all(torch.equal(p, q) for p, q in zip(m.parameters(), ref.parameters()))   # same initialisation
    assert m.loop.gate_mu.shape == (1, D) and torch.isnan(m.loop.gate_mu).all()
    two = side("--w1s-gate=point", "--no-w1s-centre", "--w1s-gate-norm=init")         # taps 3 and h
    assert two.loop.gate_mu.shape == (2, D)
    for flags in (["--w1s-gate-norm=init"], [*I1, "--w1s-centre", "--w1s-gate-norm=init"],
                  [*I1, "--w1s-gate=oracle", "--w1s-gate-norm=init"], [*I1, "--w1s-gate=dense", "--w1s-gate-norm=init"]):
        with pytest.raises(ValueError, match="gate-norm"):
            side(*flags)
    big = W1sSide(2048, ARGS, split_args([*I1, "--w1s-gate=schmitt", "--w1s-gate-norm=init"])[0])
    assert sum(b.numel() for b in big.buffers()) <= 1_000_000


def test_gate_norm_calibrates_once_and_never_adapts():
    """μ̂ is set by the first batch (the mean of the LayerNormed gate stream over all its positions), then fixed: a
    batch of other states leaves it unchanged, a checkpoint restores it (the loaded side is not recalibrated and
    computes what the original does, bitwise), and calibrate_gate recalibrates on request."""
    m = side(*I1, "--w1s-gate=hyst", "--w1s-gate-norm=init")
    h1, h2 = ln_like(2, 150, D, 1.5, seed=1), ln_like(2, 150, D, -2.0, seed=2)
    with torch.no_grad():
        y1 = run(m, h1, None)
    mu = m.loop.gate_mu.clone()
    torch.testing.assert_close(mu[0], torch.nn.functional.layer_norm(h1, (D,)).mean((0, 1)))
    with torch.no_grad():
        y2 = run(m, h2, None)
    assert torch.equal(m.loop.gate_mu, mu)
    fresh = side(*I1, "--w1s-gate=hyst", "--w1s-gate-norm=init")
    fresh.load_state_dict(m.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(run(fresh, h2, None), y2) and torch.equal(run(fresh, h1, None), y1)
    assert torch.equal(fresh.loop.gate_mu, mu)
    fresh.loop.calibrate_gate(fresh.loop._streams(h2, None))
    assert not torch.equal(fresh.loop.gate_mu, mu)


@pytest.mark.parametrize("first", ["stream", "parallel"])
def test_gate_norm_stream_calibrates_like_the_forward(first):
    """An uncalibrated side calibrates μ̂ from the same LayerNormed streams whichever form runs first, so the
    streaming store equals the parallel read on that batch."""
    m = side(*I1, "--w1s-gate=schmitt", "--w1s-gate-norm=init", "--w1s-K=2", d=16).double()
    h = ln_like(2, 150, 16, 1.5, seed=4, dtype=torch.float64)
    with torch.no_grad():
        if first == "stream":
            ref = m.loop.stream(h)
            par = m.loop.read(h)
        else:
            par = m.loop.read(h)
            ref = m.loop.stream(h)
    torch.testing.assert_close(par, ref["o"], atol=1e-6, rtol=1e-6)
    assert ref["evictions"] > 0


@pytest.mark.parametrize("gate", ["point", "hyst", "schmitt"])
@pytest.mark.parametrize("norm", list(GATE_NORM))
def test_admission_target_trains_the_uncentred_gates(gate, norm):
    """A15's admission loss (``teacher.admit_kl`` on the store's log p over the candidates) reaches the uncentred
    gates' parameters, with and without the calibrated drive: finite and non-zero, as in training."""
    from evolve import teacher
    m = side(*I1, f"--w1s-gate={gate}", *GATE_NORM[norm]).train()
    h = ln_like(2, 300, D, SHIFT[norm], seed=5)
    if norm == "init":
        m.loop.calibrate_gate(m.loop._streams(ln_like(2, 300, D, SHIFT[norm], seed=9), None))
    with teacher.store_tap(m.loop) as box:
        run(m, h, None)
    st = box["st"]
    target = torch.rand(st["lp"].shape, generator=torch.Generator().manual_seed(0)) * st["valid"]
    target = target / target.sum(1, keepdim=True).clamp_min(1e-30)
    has = st["valid"].any(1)
    assert has.all()
    teacher.admit_kl(st["lp"], target, st["valid"], has).sum().backward()
    names = {"point": ["gate_a", "gate_b"], "hyst": ["gate_a", "gate_b", "latch_c", "lam_logit"],
             "schmitt": ["lam_logit"]}[gate]
    for p in [m.loop.gate_w[0].weight] + [getattr(m.loop, n) for n in names]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, (gate, norm)


def test_diag_gate_init_on_a_tiny_model(tmp_path):
    """experiments/evolve/diag_gate_init.py on the random tiny GPT-2: every gate × norm arm is measured on the same
    natural- and random-filler needle sequences, init calibrates μ̂ on a batch of its own (natural, as a training
    run's first batch), and the readings are in range. From a run JSON the recipe and task come from the run,
    and command-line W1s flags override it."""
    import json
    import pretrained_loop as pl
    from evolve import diag_gate_init as dg
    text = tmp_path / "filler.txt"
    text.write_text("The river runs past the old mill and the quiet town wakes slowly under grey skies. " * 300)
    common = ["--tiny", "--device", "cpu", "--window", "64", "--seq-len", "512", "--pairs", "3", "--layer", "2"]
    out = dg.main([*common, "--text-file", str(text), *I1[1:], "--norms", "none,init,seq,centre", "--n", "4",
                   "--batch", "2", "--side-seeds", "2", "--out", str(tmp_path / "diag.json")])
    assert set(out["arms"]) == {f"{g}/{n}" for g in ("point", "hyst", "schmitt")
                                for n in ("none", "init", "seq", "centre")}
    assert json.loads((tmp_path / "diag.json").read_text())["arms"].keys() == out["arms"].keys()
    assert out["recipe"]["w1s_key"] == "learned" and out["recipe"]["w1s_pre"] == 0 and out["layer"] == 2
    assert out["side_seeds"] == [0, 1]
    for arm, rec in out["arms"].items():
        assert rec["flags"]["w1s_gate"] == arm.split("/")[0]
        assert set(rec["fillers"]) == {"natural", "random"} and set(rec["seeds"]) == {"0", "1"}
        for f, r in rec["fillers"].items():                       # pooled over the two seeds
            assert r["n"] == 8 and r["n_per_seed"] == 4 and 0 < r["onset_density"] < 0.2 and 0 <= r["on_fraction"] <= 1
            for k in ("onset_in_fact", "coverage_all_facts", "fact_coverage", "latch_saturated", "p_low_sequences",
                      "p_high_sequences", "onset_in_fact_p05"):
                assert 0 <= r[k] <= 1, (arm, k, r[k])
            assert 0 <= r["p_onset_median"] <= 1
            per = [rec["seeds"][s]["fillers"][f] for s in ("0", "1")]
            assert all(p["n"] == 4 for p in per)
            assert abs(r["latch_saturated"] - sum(p["latch_saturated"] for p in per) / 2) < 1e-9
            acr = r["across_seeds"]
            assert acr["latch_saturated"]["by_seed"] == {0: per[0]["latch_saturated"], 1: per[1]["latch_saturated"]}
            assert acr["on_fraction"]["min"] <= r["on_fraction"] <= acr["on_fraction"]["max"]
            assert acr["seeds_saturated"] in (0.0, 0.5, 1.0)
        if arm.endswith("/init"):
            assert rec["calibration"]["filler"] == "natural" and all(x > 0 for x in rec["calibration"]["gate_mu_norm"])
            assert all("calibration" in rec["seeds"][s] for s in ("0", "1"))
        else:
            assert "calibration" not in rec
    run = {"args": vars(pl.parse_args(common[:1] + common[3:])), "needle_scatter": 8, "needle_tail": 256,
           "candidate": {"class": "evolve.w1s_slots:W1sSide", "args": vars(split_args([*I1, "--w1s-K=4"])[0])}}
    js = tmp_path / "run.json"
    js.write_text(json.dumps(run))
    out2 = dg.main([str(js), "--device", "cpu", "--gates", "hyst", "--norms", "init", "--n", "2", "--batch", "2",
                    "--w1s-K", "6"])
    assert (tmp_path / "run.gate_init.json").exists() and out2["result"] == str(js)
    assert out2["recipe"]["w1s_K"] == 6 and out2["recipe"]["w1s_key"] == "learned"
    assert set(out2["arms"]) == {"hyst/init"} and set(out2["arms"]["hyst/init"]["fillers"]) == {"random"}
    assert out2["arms"]["hyst/init"]["calibration"]["filler"] == "random"


# ============================================================ Stage A2 (round11_combination_plan.md)
# "Code needed" 1-3: rotary and ordinal time together, pointer replay, and the alignment code's own stream. Each is
# tested on I1 with every gate (the hysteresis and Schmitt gates too: the plan's gate track), against rules E1-E3.
A2 = {
    "rope+ordinal": ["--w1s-time-feat=rope,ordinal", "--w1s-rope-dims=8", "--w1s-rope-periods=4,4096"],
    "rope+ordinal+linear": ["--w1s-time-feat=rope,ordinal", "--w1s-rope-dims=8", "--w1s-recency-form=linear"],
    "rope+ordinal+supr": ["--w1s-time-feat=ordinal,rope", "--w1s-rope-dims=8", "--w1s-recency-form=supr",
                          "--w1s-sup-w=5"],
    "pointer": ["--w1s-replay=pointer"],
    "pointer-learned": ["--w1s-replay=pointer", "--w1s-align=learned"],
    "pointer-code": ["--w1s-replay=pointer", "--w1s-code=learned", "--w1s-dc=8"],
    "pointer-keep": ["--w1s-replay=pointer", "--w1s-keep=5", "--w1s-keep-by=value"],
    "pointer+time": ["--w1s-replay=pointer", "--w1s-time-feat=rope,ordinal", "--w1s-rope-dims=8"],
    "A11a": ["--w1s-taps=3", "--w1s-key-taps=21", "--w1s-align-stream=3"],              # code on 3, gate/key on h
    "A11k": ["--w1s-taps=3", "--w1s-key-taps=3,21", "--w1s-key=both", "--w1s-align-stream=h"],   # code on h
    "A11k-code": ["--w1s-taps=3", "--w1s-key-taps=3,21", "--w1s-key=both", "--w1s-align-stream=h",
                  "--w1s-code=learned", "--w1s-dc=8"],
    "A11a-pointer": ["--w1s-taps=3", "--w1s-key-taps=21", "--w1s-align-stream=3", "--w1s-replay=pointer"],
}
GATES4 = ["point", "hyst", "schmitt", "oracle"]


def a2(name, *extra, d=D):
    m = side(*I1, *A2[name], *extra, d=d)
    with torch.no_grad():
        m.loop.gain_logit.fill_(2.0)
        if hasattr(m.loop, "ord_E"):                             # zero at init: give the rank feature a value
            m.loop.ord_E.normal_(0, 1.0)
    return m


@pytest.mark.parametrize("name", list(A2))
@pytest.mark.parametrize("gate", GATES4)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stage_a2_strictly_causal(name, gate, dtype):
    """E1: perturbing h and tap 3 from t0 on leaves every output row before t0 bitwise unchanged, for float32 and
    bf16-rounded input (K = 2: evictions), with every gate."""
    m = a2(name, "--w1s-K=2", f"--w1s-gate={gate}")
    T = 200
    m.oracle_onsets = every(2, T, 13, 5)
    torch.manual_seed(0)
    h = torch.randn(2, T, D).to(dtype).float()
    ex = {3: torch.randn(2, T, D).to(dtype).float()}
    with torch.no_grad():
        y = run(m, h, ex)
        assert (m.loop.last_diag["retained"] >= 0).any()
        for t0 in (1, 23, 60, 131, 199):
            h2, e2 = h.clone(), {3: ex[3].clone()}
            h2[:, t0:] += torch.randn_like(h2[:, t0:]).to(dtype).float()
            e2[3][:, t0:] += torch.randn_like(e2[3][:, t0:]).to(dtype).float()
            y2 = run(m, h2, e2)
            assert torch.equal(y[:, :t0], y2[:, :t0]), (name, t0)
            assert not torch.equal(y, y2)


@pytest.mark.parametrize("name", list(A2))
@pytest.mark.parametrize("gate", GATES4)
def test_stage_a2_stream_equals_parallel(name, gate):
    """E2: the parallel read equals the independent streaming store (K = 2, so slots are evicted): the read, the
    gate, the slots held at every t and their read weights, and (pointer) the entry read at every t."""
    m = a2(name, "--w1s-K=2", f"--w1s-gate={gate}", d=16).double()
    B, T = 2, 170
    torch.manual_seed(1)
    h, ex = torch.randn(B, T, 16, dtype=torch.float64), {3: torch.randn(B, T, 16, dtype=torch.float64)}
    oracle = every(B, T, 11, 3)
    with torch.no_grad():
        if m.loop.replay == "pointer":                           # a low null: a slot is read wherever one is held
            m.loop.null_logit.fill_(-5.0)
        par = m.loop.read(h, ex, oracle)
    d = m.loop.last_diag
    ref = m.loop.stream(h, ex, oracle)
    torch.testing.assert_close(par, ref["o"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(d["p"], ref["p"], atol=1e-6, rtol=1e-6)
    assert ref["evictions"] > 0
    if m.loop.replay == "pointer":
        assert torch.equal(torch.where(ref["ptr"] >= 0, ref["ptr"] + 1, -1), d["align"])
        assert (ref["ptr"] >= 0).float().mean() > 0.5                     # the pointer is in use
    for b in range(B):
        for t in range(T):
            held = {int(s) for s in d["retained"][b, t] if s >= 0}
            assert held == ref["retained"][b][t], (b, t)
            w = {int(s): float(a) for s, a in zip(d["retained"][b, t], d["read_w"][b, t]) if s >= 0}
            for s, a in ref["read_w"][b][t].items():
                assert abs((w[s] if s is not None else float(d["read_w"][b, t, -1])) - a) < 1e-6, (name, b, t, s)


@pytest.mark.parametrize("name", ["rope+ordinal", "pointer", "pointer-code", "A11a", "A11k", "A11k-code"])
def test_stage_a2_state_is_independent_of_T_and_matches_the_analytic_count(name):
    """E2: the streaming state has the same size at T = 512 and 2048 and equals ``analytic_state`` once the ring
    and the K slots are full (pointer: one more scalar; a split alignment stream: its codes and keys)."""
    m = side(*I1, *A2[name], "--w1s-K=2", d=16)
    sizes = []
    for T in (512, 2048):
        torch.manual_seed(2)
        ref = m.loop.stream(torch.randn(1, T, 16), {3: torch.randn(1, T, 16)})
        assert ref["evictions"] > 0
        sizes.append(ref["state_floats"])
    assert sizes[0] == sizes[1], sizes
    assert ref["state_components"] == m.loop.analytic_state(16), (ref["state_components"], m.loop.analytic_state(16))
    base = side(*I1, "--w1s-K=2", d=16).loop.analytic_state(16)
    comp = ref["state_components"]
    if name.startswith("pointer"):
        assert comp["pointer"] == 1
    if name in ("A11a", "A11k"):                                  # A11k: the lexical key's stream and the code's
        assert comp["slot_keys"] == base["slot_keys"] + (2 * 16 if name == "A11k" else 0), (comp, base)
        assert comp["slot_codes"] == base["slot_codes"]
        ring_len = max(m.loop.delay + m.loop.P + 1, m.loop.delay + m.loop.r + 1, m.loop.pool)
        assert comp["ring"] == base["ring"] + ring_len * 16 * (name == "A11k"), (comp, base)


@pytest.mark.parametrize("name", list(A2))
def test_stage_a2_budget_and_state_at_qwen_width(name):
    """E3 and E2 at width 2048: every Stage-A2 arm stays within 4.4M parameters, and its full streaming state within
    1M floats except A11k with the lexical key: it keeps two 2048-d lexical streams (block 3 for the key, h for the
    code), so its ring and slot keys hold one more 2048-d vector each: 1.005M (I1: 0.907M). The pointer adds one
    parameter (κ) and one state scalar."""
    m = W1sSide(2048, ARGS, split_args([*I1, *A2[name]])[0])
    i1 = W1sSide(2048, ARGS, split_args(list(I1))[0])
    n_par = sum(p.numel() for p in m.parameters())
    assert n_par <= 4_400_000 and sum(b.numel() for b in m.buffers()) <= 1_000_000
    state, ref = m.state_report()["state_floats"], i1.state_report()["state_floats"]
    if name.startswith("A11k") and "--w1s-code=learned" not in A2[name]:
        lp = m.loop
        ring_len = max(lp.delay + lp.P + 1, lp.delay + lp.r + 1, lp.pool)
        assert state == ref + (ring_len + lp.K) * 2048, (state, ref)
    else:
        assert state <= 1_000_000, m.state_report()
    if name == "pointer":
        assert n_par == sum(p.numel() for p in i1.parameters()) + 1
        assert m.state_report()["state_floats"] == i1.state_report()["state_floats"] + 1


def test_stage_a2_defaults_change_nothing():
    """Regression: the defaults (and values that name the default stream) build the round-10/11 module: no κ, no
    alignment stream of its own, the same parameters, and bitwise the same output."""
    w = split_args([])[0]
    assert (w.w1s_replay, w.w1s_align_stream, w.w1s_time_feat) == ("match", "lex", "none")
    a11 = ("--w1s-taps=3", "--w1s-key-taps=3,21", "--w1s-key=both")
    for flags, same in (([], ["--w1s-align-stream=3"]), (list(I1), ["--w1s-align-stream=h"]),
                        (list(I1), ["--w1s-align-stream=21"]), ([*I1, *a11], ["--w1s-align-stream=3"]),
                        ([*I1, "--w1s-time-feat=rope"], ["--w1s-replay=match"])):
        torch.manual_seed(0)                                      # 21 names the main tap only with --layer 21
        m0 = W1sSide(D, type("A", (), {"time": True, "layer": 21})(), split_args(flags)[0]).eval()
        torch.manual_seed(0)
        m1 = W1sSide(D, type("A", (), {"time": True, "layer": 21})(), split_args([*flags, *same])[0]).eval()
        assert not m0.loop.split_align and not m1.loop.split_align and m1.loop.replay == "match"
        assert not hasattr(m1.loop, "ptr_kappa") and "pointer" not in m1.loop.analytic_state(D)
        assert list(m0.state_dict()) == list(m1.state_dict())
        torch.manual_seed(4)
        h, ex = torch.randn(2, 150, D), {3: torch.randn(2, 150, D)}
        with torch.no_grad():
            assert torch.equal(run(m0, h, ex), run(m1, h, ex)), flags


def test_time_feat_list_parses_and_resolves():
    """--w1s-time-feat takes a comma list (canonical order rope,ordinal); a time feature still defaults the recency
    form to none, an explicit one is kept; single values are unchanged; unknown names are refused."""
    for s, canon in (("rope,ordinal", "rope,ordinal"), ("ordinal,rope", "rope,ordinal"), ("rope", "rope"),
                     ("ordinal", "ordinal"), ("none", "none"), ("rope, ordinal", "rope,ordinal")):
        w = split_args([f"--w1s-time-feat={s}"])[0]
        assert w.w1s_time_feat == canon
        assert w.w1s_recency_form == ("linear" if canon == "none" else "none")
    assert split_args(["--w1s-time-feat=rope,ordinal", "--w1s-recency-form=linear"])[0].w1s_recency_form == "linear"
    for bad in ("rope,foo", "learned"):
        with pytest.raises(SystemExit):
            split_args([f"--w1s-time-feat={bad}"])
    m = side(*I1, *A2["rope+ordinal"])
    assert m.loop.tfeats == {"rope", "ordinal"} and hasattr(m.loop, "ord_E") and m.loop.rope_dims == 8
    assert side(*I1, "--w1s-time-feat=rope").loop.tfeats == {"rope"}
    with pytest.raises(ValueError):                              # time features act on the learned key
        side("--w1s-key=lex", "--w1s-time-feat=rope,ordinal")


def test_rope_and_ordinal_are_both_active_together():
    """With both time features the score is the rotary one plus u_t·E[r_k]: with E = 0 the read is bitwise the
    rope-only read (the ordinal parameters are created last, so the shared ones are equal), and with E ≠ 0 it
    differs from both single-feature reads."""
    flags = ("--w1s-rope-dims=8", "--w1s-rope-periods=4,4096", "--w1s-gate=oracle", "--w1s-K=3")
    both, rope = side(*I1, "--w1s-time-feat=rope,ordinal", *flags), side(*I1, "--w1s-time-feat=rope", *flags)
    ordi = side(*I1, "--w1s-time-feat=ordinal", *flags)
    torch.manual_seed(0)
    h, o = torch.randn(2, 150, D), every(2, 150, 9, 2)
    with torch.no_grad():
        assert torch.equal(both.loop.read(h, None, o), rope.loop.read(h, None, o))
        both.loop.ord_E.normal_(0, 1.0)
        ordi.loop.ord_E.copy_(both.loop.ord_E)
        ordi.loop.ord_u.weight.copy_(both.loop.ord_u.weight)
        y = both.loop.read(h, None, o)
        assert not torch.equal(y, rope.loop.read(h, None, o)) and not torch.equal(y, ordi.loop.read(h, None, o))


# ------------------------------------------------------------------ pointer replay
def toy_codes(q, codes, seed=0, query=None):
    """``toy`` with given 4-digit codes (repeated digits allowed): three facts [name, is, sp, d, d, d, d] in
    60-token filler gaps with 3 false windows each, then the query for fact q, its digits teacher-forced (or the
    digits ``query``)."""
    g = torch.Generator().manual_seed(seed)
    fill = lambda k: [FILLER[i] for i in torch.randint(len(FILLER), (k,), generator=g)]
    toks, onsets, facts = [], [], []
    for i, code in enumerate(codes):
        onsets += [len(toks) + 10, len(toks) + 22, len(toks) + 34]
        toks += fill(60)
        facts.append(dict(name=len(toks), d=[len(toks) + 3 + k for k in range(4)]))
        onsets.append(len(toks) + 3)
        toks += [NAMES[i], IS, SP] + list(code)
    onsets += [len(toks) + 10, len(toks) + 22, len(toks) + 34]
    toks += fill(60)
    qpos = len(toks)
    toks += [NAMES[q], IS, SP] + list(codes[q] if query is None else query)
    o = torch.zeros(1, len(toks), dtype=torch.bool)
    o[0, onsets] = True
    return torch.eye(V)[toks][None], o, facts, qpos


def spy_store(monkeypatch):
    """Capture SlotLoop._read's alignment weights E, entries, candidates and values, and its output r."""
    out = []
    real = SlotLoop._read

    def wrapped(self, f, st):
        r = real(self, f, st)
        out.append(({k: st[k].detach() for k in ("E", "mem", "s", "v")}, r.detach()))
        return r
    monkeypatch.setattr(SlotLoop, "_read", wrapped)
    return out


@pytest.mark.parametrize("code", [(8, 9, 8, 10), (8, 9, 8, 9), (9, 9, 10, 11)])
def test_pointer_replays_a_code_with_repeated_digits(code, monkeypatch):
    """One-hot toy: the queried fact's code repeats a digit (half of 4-digit codes do), so at a repeated digit the
    plain match ties the two stored copies and reads both successors half and half. The pointer, carried from the
    unambiguous "is" and space, reads the right successor at every teacher-forced digit (alignment weight > 0.9 in
    the fact's slot, the recalled value closest to the next digit's), and the streaming form agrees."""
    reads = spy_store(monkeypatch)
    codes = [(12, 13, 14, 15), code, (16, 17, 12, 13)]
    x, o, facts, qpos = toy_codes(1, codes)
    fact, res = facts[1], {}
    for replay in ("match", "pointer"):
        m = side(*TOY, "--w1s-key=lex", "--w1s-align=lex", f"--w1s-replay={replay}", d=V, time=False)
        m.oracle_onsets, m.extra = o, {3: x}
        with torch.no_grad():
            m(x)
        st, r = reads[-1]
        d = m.loop.last_diag
        i = int((st["s"][0] == fact["d"][0]).nonzero()[0, 0])     # the fact's candidate
        vn = torch.nn.functional.normalize(st["v"][0], dim=-1)
        res[replay] = []
        for k, t in enumerate(range(qpos + 2, qpos + 6)):         # space, d1, d2, d3 predict d1..d4
            assert d["retained"][0, t][d["read_w"][0, t, :-1].argmax()] == fact["d"][0], (replay, t)
            e = st["E"][0, t] * st["mem"][0, i]
            j = fact["d"][k] - 1                                  # the entry whose successor is digit k + 1
            res[replay].append(float(e[j] / e.sum()))
            if replay == "pointer":
                rec = vn @ torch.nn.functional.normalize(r[0, t], dim=0)
                assert rec[j + 1] > rec.max() - 1e-6, (t, rec.argmax())       # recalled: the next digit's value
        if replay == "pointer":
            assert [int(a) for a in d["align"][0, qpos + 2:qpos + 6]] == fact["d"], res
            with torch.no_grad():
                par = m.loop.read(x, {3: x}, o)
            ref = m.loop.stream(x, {3: x}, o)
            torch.testing.assert_close(par, ref["o"], atol=1e-5, rtol=1e-5)
            assert torch.equal(ref["ptr"][0, qpos + 2:qpos + 6] + 1, torch.tensor(fact["d"]))
    ambiguous = [k for k in range(1, 4) if code[k - 1] in code[:k - 1] + code[k:]]   # the current digit repeats
    assert ambiguous, code
    assert all(w > 0.9 for w in res["pointer"]), res
    assert all(abs(res["match"][k] - 0.5) < 0.1 for k in ambiguous), res


def test_pointer_gives_way_to_the_match_when_the_token_is_not_the_successor(monkeypatch):
    """The pointer advances only while the current token matches the entry it lands on: the query skips the stored
    third digit (stored 8 9 8 10, query 8 9 10), so at "10" the pointer's entry (the stored 8) does not match and
    the re-match (the stored 10) carries the alignment; the next pointer continues from there."""
    reads = spy_store(monkeypatch)
    x, o, facts, qpos = toy_codes(1, [(12, 13, 14, 15), (8, 9, 8, 10), (16, 17, 12, 13)], query=(8, 9, 10, 11))
    m = side(*TOY, "--w1s-key=lex", "--w1s-align=lex", "--w1s-replay=pointer", d=V, time=False)
    m.oracle_onsets, m.extra = o, {3: x}
    with torch.no_grad():
        m(x)
    st, _ = reads[-1]
    fact = facts[1]
    i = int((st["s"][0] == fact["d"][0]).nonzero()[0, 0])
    t = qpos + 5                                                  # the query's "10"
    e = st["E"][0, t] * st["mem"][0, i]
    assert m.loop.last_diag["align"][0, t - 1] == fact["d"][2]    # at "9" the replay pointed at the stored 8
    assert float(e[fact["d"][3]] / e.sum()) > 0.99                # the stored 10 wins: the match, not the pointer
    assert m.loop.last_diag["align"][0, t] == fact["d"][3] + 1


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("cfg", [["--w1s-gate=dense"], ["--w1s-gate=dense", "--w1s-align=learned"]])
def test_pointer_stream_equals_parallel_at_exact_ties(cfg, dtype):
    """Review 3, Q6: states from a 6-token vocabulary repeat exactly, so slots (identical keys, no age term) and
    entries (identical codes) tie mathematically, and the two forms round the tied weights differently. An exact
    argmax then chose the older slot in one form and the newer in the other, and the read differed by O(1) (0.46).
    Both forms now take the first entry within a relative PTR_TIE of the maximum: the pointer and the read agree."""
    torch.manual_seed(1)
    emb, toks = torch.randn(6, 32), torch.randint(0, 6, (3, 400))
    m = side(*I1, "--w1s-replay=pointer", "--w1s-K=4", "--w1s-recency-form=none", *cfg).to(dtype)
    with torch.no_grad():
        m.loop.gain_logit.fill_(2.0)
        m.loop.null_logit.fill_(-5.0)
        par = m.loop.read(emb[toks].to(dtype), None, None)
    d = m.loop.last_diag
    ref = m.loop.stream(emb[toks].to(dtype), None, None)
    assert (ref["ptr"] >= 0).float().mean() > 0.5
    assert torch.equal(torch.where(ref["ptr"] >= 0, ref["ptr"] + 1, -1), d["align"])
    torch.testing.assert_close(par, ref["o"], **({} if dtype == torch.float32 else dict(atol=1e-9, rtol=1e-9)))
    from evolve.w1s_slots import first_max
    x = torch.tensor([[0.3, 0.5, 0.5 * (1 + 1e-7), 0.1], [0.2, 0.1, 0.0, 0.2], [float("-inf")] * 4],
                     dtype=torch.float64)
    assert first_max(x).tolist() == [1, 0, 0]


def test_pointer_kappa_and_codes_get_gradients_and_refusals():
    """κ (from --w1s-ptr-init) is the pointer's only parameter and is trained; with a split alignment stream a
    learned code's map gets gradient through tap 3. Options that would do nothing are errors: a pointer without an
    alignment or without additive slots, an alignment stream for a code that is not lexical (or not read), and an
    unknown stream name."""
    m = a2("pointer", "--w1s-gate=oracle", "--w1s-ptr-init=2.5").train()
    assert abs(m.loop.ptr_kappa.item() - 2.5) < 1e-6
    m.oracle_onsets = every(2, 150, 9, 2)
    with torch.no_grad():
        m.loop.null_logit.fill_(-5.0)
    m(torch.randn(2, 150, D))[0].pow(2).sum().backward()
    assert m.loop.ptr_kappa.grad is not None and torch.isfinite(m.loop.ptr_kappa.grad) and m.loop.ptr_kappa.grad != 0
    c = a2("A11a", "--w1s-gate=oracle", "--w1s-code=learned", "--w1s-dc=8").train()
    c.oracle_onsets, c.extra = every(2, 150, 9, 2), {3: torch.randn(2, 150, D)}
    c(torch.randn(2, 150, D))[0].pow(2).sum().backward()
    assert c.loop.wz.weight.grad.abs().sum() > 0 and c.loop.astream == 3
    for flags in (["--w1s-replay=pointer", "--w1s-align=none"], ["--w1s-replay=pointer", "--w1s-store=superposed"],
                  ["--w1s-replay=pointer", "--w1s-out=reinstate", "--w1s-taps=3", "--w1s-key-taps=3"],
                  ["--w1s-taps=3", "--w1s-key-taps=21", "--w1s-align-stream=3", "--w1s-align=learned"],
                  ["--w1s-taps=3", "--w1s-key-taps=21", "--w1s-align-stream=3", "--w1s-align=none"],
                  ["--w1s-taps=3", "--w1s-key-taps=21", "--w1s-align-stream=3", "--w1s-store=superposed"],
                  ["--w1s-taps=1,3", "--w1s-key-taps=1", "--w1s-align-stream=3", "--w1s-out=reinstate"],
                  ["--w1s-align-stream=block3"]):
        with pytest.raises(ValueError, match="--w1s-(replay pointer|align-stream)"):
            side(*I1, *flags)


def test_align_stream_reads_the_chosen_stream():
    """--w1s-align-stream: A11a (gate and key on h, code on tap 3) reads tap 3 although it is not a key tap: the
    alignment code is LN(tap 3), and changing tap 3 alone changes the read but not the gate, the slots held or their
    read weights. A11k (gate and key on [3, h], code on h) takes the code from h, the lexical key still from tap 3;
    an index outside --w1s-taps means the main tap, as in --w1s-key-taps."""
    m = a2("A11a", "--w1s-gate=point")
    assert m.taps == (3,) and m.loop.streams == ["h"] and m.loop.lex == "h" and m.loop.astream == 3
    torch.manual_seed(6)
    h, ex = torch.randn(2, 150, D), {3: torch.randn(2, 150, D)}
    with torch.no_grad():
        f = m.loop._features(m.loop.norm(h), h, ex, None)
        assert torch.equal(f["xa"], torch.nn.functional.layer_norm(ex[3], (D,)))
        assert torch.equal(f["x"], torch.nn.functional.layer_norm(h, (D,)))
        y, d0 = run(m, h, ex), dict(m.loop.last_diag)
        y2 = run(m, h, {3: ex[3] + torch.randn_like(ex[3])})
        d1 = m.loop.last_diag
    assert (d0["retained"] >= 0).any() and not torch.equal(y, y2)
    for k in ("p", "onsets", "retained", "read_w"):
        assert torch.equal(d0[k], d1[k]), k
    k = a2("A11k")
    assert k.loop.streams == [3, "h"] and k.loop.lex == 3 and k.loop.astream == "h" and k.loop.split_align
    with torch.no_grad():
        f = k.loop._features(k.loop.norm(h), h, ex, None)
        st = k.loop._store(f)
    assert torch.equal(f["xa"], torch.nn.functional.layer_norm(h, (D,)))
    assert torch.equal(f["x"], torch.nn.functional.layer_norm(ex[3], (D,))) and "K" in st and "Ka" in st


def test_align_stream_refuses_a_block_that_is_not_tapped():
    """Review 3, Q5: a block index that is neither in --w1s-taps nor the main tap (--layer) is an error, not a silent
    fall-back to the main tap: I1 + --w1s-align-stream 3 (no tap 3) would otherwise be I1 bitwise under an A11a
    label. The main tap's own index is accepted when --layer gives it (it means h). A value that names the lexical
    stream is still allowed, and state_report records what a non-default value resolved to (the A11a slip
    --w1s-taps 3 --w1s-align-stream 3 without --w1s-key-taps 21: stream 3 = lex, no split)."""
    for flags in (["--w1s-align-stream=3"], ["--w1s-taps=3", "--w1s-key-taps=3", "--w1s-align-stream=7"],
                  ["--w1s-align-stream=21"]):
        with pytest.raises(ValueError, match="--w1s-align-stream"):
            side(*I1, *flags)
    L21 = type("A", (), {"time": True, "layer": 21})()
    with pytest.raises(ValueError, match="main tap"):
        W1sSide(D, L21, split_args([*I1, "--w1s-align-stream=7"])[0])
    m = W1sSide(D, L21, split_args([*I1, "--w1s-taps=3", "--w1s-key-taps=3", "--w1s-align-stream=21"])[0])
    assert m.loop.astream == "h" and m.loop.split_align
    assert m.state_report()["align_stream"] == {"spec": "21", "stream": "h", "lex": "3", "split": True}
    slip = side(*I1, "--w1s-taps=3", "--w1s-align-stream=3")
    assert slip.state_report()["align_stream"] == {"spec": "3", "stream": "3", "lex": "3", "split": False}
    assert a2("A11a").state_report()["align_stream"]["split"] is True
