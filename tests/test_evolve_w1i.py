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


@pytest.mark.parametrize("refractory", ["0.7", "0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_strictly_causal(refractory, dtype):
    """E1: perturbing h[:, t0:] leaves the output before t0 bitwise unchanged, also for bf16 input."""
    m = side("--w1i-heads=2", "--w1i-dhead=16", f"--w1i-refractory={refractory}", "--w1i-event-bias=1")
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


def test_parameter_budget_at_qwen_width():
    """E3: W1i at Qwen3-1.7B's width stays within 4.4M trainable parameters."""
    m = W1iSide(2048, type("A", (), {"time": True})(), split_args([])[0])
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
