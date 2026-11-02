"""Alignment track (experiments/alignment): controller, probe, judges, data rule, end to end."""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments" / "alignment"), str(ROOT / "experiments")]

from judge import assert_private, degeneration, refused  # noqa: E402
from probe import auroc, diff_of_means, ece, fit_probe, predict  # noqa: E402
from steering import SteeringController  # noqa: E402

D = 16


def make(mode="graded", **kw):
    torch.manual_seed(0)
    probe = {"w": torch.randn(D), "b": 0.0, "mu": torch.zeros(D), "sd": torch.ones(D)}
    return SteeringController(torch.randn(D), probe, mode=mode, **kw)


@pytest.mark.parametrize("mode", ["static", "conditional", "graded"])
@pytest.mark.parametrize("direction_mode", ["fixed", "gist"])
def test_incremental_decoding_matches_one_pass(mode, direction_mode):
    torch.manual_seed(1)
    h = torch.randn(2, 12, D)
    full = make(mode, score_positions="all", direction_mode=direction_mode)
    a = full(h)
    inc = make(mode, score_positions="all", direction_mode=direction_mode)
    parts = [inc(h[:, :7])] + [inc(h[:, t:t + 1]) for t in range(7, 12)]
    assert torch.allclose(a, torch.cat(parts, 1), atol=1e-5)


def test_graded_gain_is_capped_escalates_and_recovers():
    c = make("graded", cap=0.3, kappa=2.0, leak=0.9, score_positions="all")
    c.w.zero_(); c.b = 20.0                                       # constant s = 1: sustained pressure
    c.record = True
    c(torch.randn(1, 40, D))
    g = c.trace[-1]["g"][0]
    assert g.max() <= 0.3 + 1e-6 and g[5] > g[0] and g[-1] > 0.29
    c.b = -20.0                                                   # pressure removed: E decays as 0.9^t
    c(torch.randn(1, 80, D))
    assert c.trace[-1]["g"][0, -1] < 0.01


def test_conditional_decides_on_the_prompt_and_injects_only_when_triggered():
    h = torch.randn(1, 6, D)
    for b, expect in ((20.0, True), (-20.0, False)):
        c = make("conditional", gain=0.2)
        c.w.zero_(); c.b = b
        out = c(h)
        changed = not torch.allclose(out[:, -1], h[:, -1])
        assert changed == expect
        assert torch.allclose(out[:, :-1], h[:, :-1])            # prefill: only from the last prompt token


def test_left_padding_does_not_change_real_positions():
    torch.manual_seed(2)
    h = torch.randn(1, 5, D)
    a = make("graded", score_positions="all")(h)
    c = make("graded", score_positions="all")
    c.pending_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    b = c(torch.cat([torch.randn(1, 3, D), h], 1))
    assert torch.allclose(a, b[:, 3:], atol=1e-5)


def test_none_mode_is_identity_and_bf16_input_returns_bf16():
    h = torch.randn(2, 9, D).to(torch.bfloat16)
    out = make("none")(h)
    assert out.dtype == torch.bfloat16 and torch.equal(out, h)


def test_gist_direction_is_unit_and_moves_along_the_recent_average():
    c = make("static", gain=0.5, direction_mode="gist", beta=0.5, beta_slow=0.99)
    h = torch.zeros(1, 30, D)
    h[:, 20:, 0] = 1.0                                            # a new "idea" along axis 0
    h[:, :, 1] = 1.0                                              # background along axis 1
    out = c(h)
    delta = (out - h)[0, -1]
    assert delta[0] > 0 and abs(float(delta.norm() / h[0, -1].norm()) - 0.5) < 1e-4


def test_probe_auroc_calibration_and_direction():
    torch.manual_seed(3)
    y = torch.tensor([1] * 200 + [0] * 200)
    X = torch.randn(400, D) + 3.0 * y[:, None] * torch.eye(D)[0]
    p = fit_probe(X[::2], y[::2])
    s = predict(p, X[1::2])
    assert auroc(s, y[1::2]) > 0.9 and ece(s, y[1::2]) < 0.15
    assert auroc(torch.ones(10), torch.tensor([1, 0] * 5)) == 0.5
    d = diff_of_means(X, y)
    assert d[0] > 2.5 and d[1:].abs().max() < 0.5


def test_judges_and_the_data_rule(tmp_path):
    assert refused("I'm sorry, but I can't help with that.")
    assert refused("I cannot provide instructions for that.")
    assert not refused("Sure. Here is a summary of the article.")
    assert degeneration([1, 2, 3, 4] * 10) > 0.8 and degeneration(list(range(40))) == 0.0
    with pytest.raises(ValueError):
        assert_private(ROOT / "results" / "private", ROOT)
    assert assert_private(tmp_path / "ok", ROOT).exists()


def test_end_to_end_tiny_run_writes_labels_only(tmp_path):
    pytest.importorskip("transformers")
    import run
    rep = run.main(["--tiny", "--n-train", "24", "--n-eval", "4", "--batch", "4", "--max-new-tokens", "6",
                    "--device", "cpu", "--name", "pytest_tiny", "--private-dir", str(tmp_path / "priv")])
    out = ROOT / "results" / "alignment" / "pytest_tiny.json"
    try:
        saved = json.loads(out.read_text())
        assert set(saved["arms"]) == set(run.ARMS)
        assert all("response" not in r and "prompt" not in r for r in saved["records"])
        assert 0.0 <= saved["probe"]["auroc"] <= 1.0
        assert (tmp_path / "priv" / "pytest_tiny_graded.jsonl").exists()   # text only in the private dir
    finally:
        out.unlink(missing_ok=True)
