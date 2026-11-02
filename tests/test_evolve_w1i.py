"""Round-4 candidate W1i (experiments/evolve/w1i_events.py): rules E1-E3 and the event-gated goal field."""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from evolve.w1i_events import EventLoop, W1iSide, split_args  # noqa: E402
from limbic_llm.gated_delta import gated_delta_scan  # noqa: E402

D = 64


def side(*flags):
    torch.manual_seed(0)
    return W1iSide(D, type("A", (), {"time": True})(), split_args(list(flags))[0]).eval()


@pytest.mark.parametrize("chain", ["tied", "dg"])
@pytest.mark.parametrize("refractory", ["0.7", "0"])
@pytest.mark.parametrize("budget", ["0", "2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_strictly_causal(chain, refractory, budget, dtype):
    """E1: perturbing h[:, t0:] leaves the output before t0 bitwise unchanged, also for bf16-rounded
    float32 input (the pipeline casts the tap activations to float32 before the side channels)."""
    m = side("--w1i-heads=2", "--w1i-dhead=16", f"--w1i-refractory={refractory}", "--w1i-event-bias=1",
             f"--w1i-chain-addressing={chain}", f"--w1i-budget={budget}", "--w1i-budget-window=16")
    with torch.no_grad():
        m.loop.gain_logit.fill_(2.0)
        m.loop.goal_logit.fill_(2.0)
        m.time.out.weight.normal_(0, 0.1)
    h = torch.randn(2, 70, D).to(dtype).float()
    y, _ = m(h)
    for t0 in (1, 7, 8, 33, 69):
        h2 = h.clone()
        h2[:, t0:] += torch.randn_like(h2[:, t0:])
        assert torch.equal(y[:, :t0], m(h2)[0][:, :t0]), t0


@pytest.mark.parametrize("flags", [[], ["--w1i-chain-addressing=dg"]])
def test_parameter_budget_at_qwen_width(flags):
    """E3: W1i (dense or DG chain keys) at Qwen3-1.7B's width stays within 4.4M trainable parameters."""
    m = W1iSide(2048, type("A", (), {"time": True})(), split_args(flags)[0])
    assert sum(p.numel() for p in m.parameters()) <= 4_400_000


def _three_facts(q: int):
    """Three facts 'name is sp digit' separated by filler, then the query for fact q. One-hot."""
    names, is_, sp, digs = [0, 1, 2], 3, 4, [5, 6, 7]
    rows = []
    for i in range(3):
        rows += [(names[i], "n"), (is_, "is"), (sp, "sp"), (digs[i], "d")] + [(t, "f") for t in [8, 9, 10, 11] * 3]
    rows += [(t, "f") for t in [8, 9, 10, 11] * 4] + [(names[q], "n"), (is_, "is"), (sp, "sp")]
    return rows, digs


@pytest.mark.parametrize("q", [0, 1, 2])
def test_event_gated_context_selects_any_queried_fact(q):
    """Writing only at events (each fact's first digit), with the name salient in the context,
    reads the queried fact's digit for the first, middle and last fact; writing at every token
    does not, because consecutive context keys overwrite each other."""
    loop = EventLoop(D, heads=1, d_head=D, taus=(2.0,))
    with torch.no_grad():
        w = torch.eye(D)
        w[:3, :3] *= 4.0                                            # names salient
        loop.wc.weight.copy_(w)
    rows, digs = _three_facts(q)
    x = torch.eye(D)[[t for t, _ in rows]][None]
    c = loop.goal_context(x)                                        # (1, 1, T, D)
    k = torch.cat([torch.zeros_like(c[:, :, :1]), c[:, :, :-1]], 2)
    v = torch.eye(D)[[20 + t for t, _ in rows]][None, None]
    one = torch.ones(1, 1, len(rows))
    events = torch.tensor([1.0 if lab == "d" else 0.0 for _, lab in rows])[None, None]
    pos = {d: [i for i, (t, _) in enumerate(rows) if t == d][0] for d in digs}
    pick = lambda o: max(range(3), key=lambda n: (o[0, 0, -1] @ v[0, 0, pos[digs[n]]]).item())
    o_event, _, _ = gated_delta_scan(c, k, v, one, events)
    assert pick(o_event) == q
    o_dense, _, _ = gated_delta_scan(c, k, v, one, one)
    assert pick(o_dense) == 2                                       # dense writes: the last fact wins


def test_refractory_suppresses_writes_right_after_an_event():
    """ε_t = e_t (1 − r_{t−1}): a run of detector firings writes strongly once, then weakly."""
    loop = EventLoop(D, heads=1, d_head=16, refractory=0.7)
    with torch.no_grad():
        loop.event.weight.zero_()
        loop.event.bias.fill_(-20.0)
    x = torch.zeros(1, 12, D)
    with torch.no_grad():
        loop.event.weight[0, 0] = 40.0                              # fires where x[..., 0] = 1
    x[0, 3:7, 0] = 1.0                                              # four consecutive firings
    eps = loop.event_strength(x)[0, 0]
    assert eps[3] > 0.99 and eps[4] < 0.75 and eps[3] > eps[4] > 0
    assert eps[:3].max() < 1e-6 and eps[8:].max() < 1e-6


def test_read_wires_the_goal_field_as_described(monkeypatch):
    """The code the GPU runs: G is keyed by the *previous* context, read with the current one, and
    written with the event strength (RV review w1i-1: the mechanism test above rebuilds this path)."""
    import evolve.w1i_events as w1i
    calls = []
    real = w1i._run

    def spy(q, k, v, alpha, beta, S0, chunk):
        calls.append((q, k, beta))
        return real(q, k, v, alpha, beta, S0, chunk)

    monkeypatch.setattr(w1i, "_run", spy)
    torch.manual_seed(0)
    loop = EventLoop(D, heads=2, d_head=16)
    x = torch.randn(2, 30, D)
    with torch.no_grad():
        loop._read(x)
        c = loop.goal_context(x)
        eps = loop.event_strength(x)
    assert len(calls) == 2                                          # S, then G
    q, k, beta = calls[1]
    assert torch.equal(q, c) and torch.equal(k, loop._shift(c)) and torch.equal(beta, eps)
    assert not hasattr(loop, "g_beta")                              # writes come from the events only


def test_write_budget_caps_writes_per_window():
    """The budget caps the writes in any W-token window at B. It counts attempted events, so a
    detector that fires at every token is starved after its first writes: dense firing earns
    nothing, which is the sparsity pressure."""
    loop = EventLoop(D, heads=1, d_head=16, refractory=0.0, budget=2.0, budget_window=16)
    with torch.no_grad():
        loop.event.weight.zero_()
        loop.event.bias.fill_(10.0)                                 # e ≈ 1 at every token
    eps = loop.event_strength(torch.zeros(1, 80, D))[0, 0]
    windows = torch.stack([eps[t:t + 16].sum() for t in range(0, 64)])
    assert windows.max() <= 2.0 and eps[0] > 0.9                   # the first write passes; capped at B
    assert eps[20:].max() < 1e-6                                    # constant firing is starved
    off = EventLoop(D, heads=1, d_head=16, refractory=0.0)
    with torch.no_grad():
        off.event.weight.zero_()
        off.event.bias.fill_(10.0)
    assert off.event_strength(torch.zeros(1, 80, D))[0, 0][:16].sum() > 15   # no budget: every token writes
