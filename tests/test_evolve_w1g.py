"""Round-3 candidates W1g/W1h (experiments/evolve/w1g_goal.py): rules E1-E3 and both mechanisms."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from evolve.w1g_goal import StartLoop, W1gSide, split_args  # noqa: E402
from limbic_llm.gated_delta import _run  # noqa: E402

D = 64


def side(*flags):
    torch.manual_seed(0)
    return W1gSide(D, type("A", (), {"time": True})(), split_args(list(flags))[0]).eval()


@pytest.mark.parametrize("settle", [0, 2])
@pytest.mark.parametrize("addressing", ["tied", "dg"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_strictly_causal(settle, addressing, dtype):
    """E1: perturbing h[:, t0:] leaves the output before t0 bitwise unchanged, also for bf16 input."""
    m = side(f"--w1g-addressing={addressing}", "--w1g-heads=2", "--w1g-dhead=16", "--w1g-key-expansion=2",
             f"--w1g-settle={settle}")
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


def test_state_size_independent_of_length():
    """E2: the stores are GDN states (H, Dv, Dk) whatever T is; settling adds no carried state."""
    for T in (16, 64, 257):
        a = torch.randn(1, 2, T, 16)
        _, S, _ = _run(a, a, torch.randn(1, 2, T, 16), torch.ones(1, 2, T), torch.ones(1, 2, T), None, 8)
        assert S.shape == (1, 2, 16, 16)


@pytest.mark.parametrize("flags", [[], ["--w1g-settle=3"]])
def test_parameter_budget_at_qwen_width(flags):
    """E3: W1g and W1h at Qwen3-1.7B's width stay within 4.4M trainable parameters."""
    m = W1gSide(2048, type("A", (), {"time": True})(), split_args(flags)[0])
    assert sum(p.numel() for p in m.parameters()) <= 4_400_000


def test_no_goal_is_w1():
    """With the start store off, the read path is W1's (GatedDeltaNet tied mode) exactly."""
    torch.manual_seed(0)
    loop = StartLoop(D, heads=2, d_head=16, chunk=8, use_goal=False)
    x = torch.randn(2, 40, D)
    assert torch.allclose(loop._read(x), loop.mem(x)[0], atol=1e-5)


def _two_fact_sequence():
    """One-hot tokens; positions inside a fact or the query also carry its name (as the round-3
    probe found the name decodable at the space token). Values depend on the token."""
    n1, n2, is_, sp, d1, d2 = 0, 1, 2, 3, 4, 5
    name_dim = {n1: 20, n2: 21}
    facts = [(n1, [n1, is_, sp, d1]), (n2, [n2, is_, sp, d2])]
    rows = [(t, nm) for nm, toks in facts for t in toks]
    rows += [(t, None) for t in [6, 7, 8, 9] * 5] + [(t, n1) for t in [n1, is_, sp]]
    x = torch.zeros(1, len(rows), D)
    for i, (t, nm) in enumerate(rows):
        x[0, i, t] = 1.0
        if nm is not None:
            x[0, i, name_dim[nm]] = 1.0
    v = torch.eye(D)[[30 + t for t, _ in rows]][None, None]
    pick = lambda o: (o[0, 0, -1] @ v[0, 0, 3] - o[0, 0, -1] @ v[0, 0, 7]).item()   # d1 minus d2
    return x, v, torch.ones(1, 1, len(rows)), pick


def test_chain_store_returns_the_most_recent_fact():
    """S keyed by token identity: the shared key is overwritten, so the later fact wins."""
    x, v, one, pick = _two_fact_sequence()
    tok_only = torch.zeros(D, D)
    tok_only[:10, :10] = torch.eye(10)
    a = F.normalize(x @ tok_only.T, dim=-1)[:, None]
    assert pick(_run(a, StartLoop._shift(a), v, one, one, None, 8)[0]) < -0.5


def test_start_store_selects_the_queried_fact():
    """G with DG keys (the module's goal_keys) reads the queried fact's digit; a linear key sees
    the name too but still lets the later fact overwrite the shared token component."""
    x, v, one, pick = _two_fact_sequence()
    shift = StartLoop._shift
    lin = F.normalize(x, dim=-1)[:, None]
    assert pick(_run(lin, shift(lin), v, one, one, None, 8)[0]) < 0
    margins = []
    for seed in range(8):
        torch.manual_seed(seed)
        loop = StartLoop(D, heads=1, d_head=D, goal_expansion=2, goal_sparsity=0.05, chunk=8)
        with torch.no_grad():
            b = loop.goal_keys(x)
            margins.append(pick(_run(b, shift(b), v, one, one, None, 8)[0]))
    assert sum(margins) / len(margins) > 0.25 and sum(m > 0 for m in margins) >= 6


def test_ca3_settling_completes_a_mixed_cue():
    """W1h: a cue mixing two stored sparse patterns 0.55 / 0.45 settles onto the majority pattern."""
    loop = StartLoop(D, heads=1, d_head=D, goal_expansion=1, settle=3, settle_sparsity=6 / D, chunk=8)
    P1, P2 = torch.zeros(D), torch.zeros(D)
    P1[:6], P2[6:12] = 1.0, 1.0
    P1, P2 = F.normalize(P1, dim=0), F.normalize(P2, dim=0)
    cue = F.normalize(0.55 * P1 + 0.45 * P2, dim=0)
    b = torch.stack([P1, P2, cue])[None, None]                     # (1, 1, 3, D): store P1, P2, then cue
    one = torch.ones(1, 1, 3)
    with torch.no_grad():
        settled = loop.settle_cue(b, one, one)[0, 0, -1]
    assert F.cosine_similarity(settled, P1, dim=0) > 0.95
    assert F.cosine_similarity(cue, P1, dim=0) < 0.8               # without settling it stays a mix
