"""Round-8 candidate W1j (experiments/evolve/w1j_ca1.py): rules E1-E3 and the CA1 novelty path."""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from evolve.w1j_ca1 import CA1Loop, W1jSide, split_args  # noqa: E402
from limbic_llm.gated_delta import gated_delta_scan  # noqa: E402

D = 64


def side(*flags):
    torch.manual_seed(0)
    return W1jSide(D, type("A", (), {"time": True})(), split_args(list(flags))[0]).eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_strictly_causal(dtype):
    """E1: perturbing h[:, t0:] leaves the output before t0 bitwise unchanged (bf16-rounded input too)."""
    m = side("--w1i-heads=2", "--w1i-dhead=16", "--w1i-event-bias=1")
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
    """E3: W1j adds only κ (one per head) to W1i."""
    m = W1jSide(2048, type("A", (), {"time": True})(), split_args([])[0])
    assert sum(p.numel() for p in m.parameters()) <= 4_400_000


def test_novelty_feeds_the_event_gate(monkeypatch):
    """The code the GPU runs: the event gate gets κ · novelty from the chain store's own error."""
    torch.manual_seed(0)
    loop = CA1Loop(D, heads=2, d_head=16)
    seen = {}
    real = loop.event_strength

    def spy(x, extra_logit=None):
        seen["extra"] = extra_logit
        return real(x, extra_logit)

    monkeypatch.setattr(loop, "event_strength", spy)
    x = torch.randn(2, 30, D)
    with torch.no_grad():
        loop._read(x)
    assert torch.allclose(seen["extra"], loop.kappa[None, :, None] * loop.last_novelty)
    assert not loop.last_novelty.requires_grad                    # detached: no incentive to predict worse


def test_novelty_is_high_where_the_memory_did_not_predict():
    """A transition seen before is predicted (low error); a new one is not (high novelty)."""
    loop = CA1Loop(D, heads=1, d_head=D, baseline=0.5)
    T = 12
    keys = torch.nn.functional.normalize(torch.eye(D)[[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 5, 6]], dim=-1)[None, None]
    vals = torch.eye(D)[[1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 9, 9]][None, None]
    one = torch.ones(1, 1, T)
    _, _, err = gated_delta_scan(keys, keys, vals, one, one)
    n = loop.novelty(err, vals)[0, 0]
    assert n[9] < 0.0 and n[10] > 0.5                              # repeated vs new transition
