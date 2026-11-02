"""Alignment track (experiments/alignment): controller, probe, judges, data rule, end to end."""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments" / "alignment"), str(ROOT / "experiments")]

from judge import (assert_private, counted_refusal, degeneration, private_file, refused,  # noqa: E402
                   repetition_onset, safe_name)
from probe import auroc, diff_of_means, ece, fit_probe, predict  # noqa: E402
from steering import DIRECTIONS, SteeringController  # noqa: E402

D = 16


def make(mode="graded", **kw):
    torch.manual_seed(0)
    probe = {"w": torch.randn(D), "b": 0.0, "mu": torch.zeros(D), "sd": torch.ones(D)}
    return SteeringController(torch.randn(D), probe, mode=mode, **kw)


# Conditional is excluded: with score_positions="all" its decision is taken at the last position of the
# first call, which differs between one pass and prefill + decode by design (RV1.A4). The "last" path,
# which run.py uses, is tested against a reference below.
@pytest.mark.parametrize("mode", ["static", "graded"])
@pytest.mark.parametrize("direction_mode", DIRECTIONS)
def test_incremental_decoding_matches_one_pass(mode, direction_mode):
    torch.manual_seed(1)
    h = torch.randn(2, 12, D)
    full = make(mode, score_positions="all", direction_mode=direction_mode)
    a = full(h)
    inc = make(mode, score_positions="all", direction_mode=direction_mode)
    parts = [inc(h[:, :7])] + [inc(h[:, t:t + 1]) for t in range(7, 12)]
    assert torch.allclose(a, torch.cat(parts, 1), atol=1e-5)


def reference_last(c: SteeringController, h: torch.Tensor, P: int) -> torch.Tensor:
    """Naive per-token "last" path for one unpadded sequence (T, D), fixed direction: the score counts
    at the last prompt token (index P − 1) and at every later token."""
    s = torch.sigmoid(((h - c.mu) / c.sd) @ c.w + c.b)
    cond = bool(s[P - 1] >= c.threshold)
    E, out = 0.0, []
    for t in range(len(h)):
        counted = t >= P - 1
        E = c.leak * E + (float(s[t]) if counted else 0.0)
        if c.mode == "static":
            g = c.gain
        elif c.mode == "conditional":
            g = c.gain * cond if counted else 0.0
        else:
            g = c.cap * torch.tanh(torch.tensor(c.kappa * E)).item()
        out.append(h[t] + g * h[t].norm() * c.d)
    return torch.stack(out)


@pytest.mark.parametrize("mode", ["static", "conditional", "graded"])
@pytest.mark.parametrize("bias", [-1.0, 1.0])                 # conditional off / on for this seed
def test_last_path_padded_prefill_and_decode_matches_reference(mode, bias):
    """RV1.A4: left-padded prefill of two prompts of different lengths, then decode steps, against a
    per-token reference run on each sequence alone."""
    torch.manual_seed(4)
    lens, n_dec = (5, 9), 6
    seqs = [torch.randn(n + n_dec, D) for n in lens]
    c = make(mode, gain=0.2, cap=0.3, kappa=1.0)
    c.b = bias
    T = max(lens)
    pre = torch.zeros(2, T, D)
    mask = torch.zeros(2, T)
    for j, n in enumerate(lens):
        pre[j, T - n:] = seqs[j][:n]
        mask[j, T - n:] = 1
    c.pending_mask = mask
    outs = [c(pre)] + [c(torch.stack([seqs[j][lens[j] + k] for j in range(2)])[:, None]) for k in range(n_dec)]
    for j, n in enumerate(lens):
        got = torch.cat([outs[0][j, T - n:]] + [o[j] for o in outs[1:]])
        assert torch.allclose(got, reference_last(c, seqs[j], n), atol=1e-5), (mode, j)


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
    for dm in DIRECTIONS:
        a = make("graded", score_positions="all", direction_mode=dm)(h)
        c = make("graded", score_positions="all", direction_mode=dm)
        c.pending_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
        b = c(torch.cat([torch.randn(1, 3, D), h], 1))
        assert torch.allclose(a, b[:, 3:], atol=1e-5), dm


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


def test_gist_excludes_the_sink_and_cancels_the_background():
    """RV1.C1: a massive first token (the attention sink) and a constant background must not leak
    into the gist direction."""
    h = torch.zeros(1, 30, D)
    h[:, :, 1] = 1.0                                              # background along axis 1
    h[:, 20:, 0] = 1.0                                            # the current thought along axis 0
    h[:, 0, 2] = 100.0                                            # sink: ~80x the median norm, as on Qwen3
    d = lambda c: ((c(h) - h)[0, -1] / (0.5 * h[0, -1].norm()))   # the injected unit direction
    gist = d(make("static", gain=0.5, direction_mode="gist", beta=0.5, beta_slow=0.99))
    assert gist[0] > 0.99 and abs(gist[1]) < 1e-4 and abs(gist[2]) < 1e-4
    leaky = d(make("static", gain=0.5, direction_mode="gist", beta=0.5, beta_slow=0.99, gist_skip=0))
    assert abs(leaky[2]) > 0.3                                    # with the sink included it dominates
    anti = d(make("static", gain=0.5, direction_mode="anti_gist", beta=0.5, beta_slow=0.99))
    assert torch.allclose(anti, -gist, atol=1e-5)
    mean = d(make("static", gain=0.5, direction_mode="mean", beta_slow=0.99))
    assert mean[1] > 0.8 and mean[0] > 0 and abs(mean[2]) < 1e-4   # the stream mean, sink excluded


def test_random_direction_is_fixed_unit_and_seeded():
    a = make("static", direction_mode="random", seed=3)
    b = make("static", direction_mode="random", seed=3)
    c = make("static", direction_mode="random", seed=4)
    assert torch.equal(a.d, b.d) and not torch.equal(a.d, c.d) and abs(float(a.d.norm()) - 1) < 1e-6
    h = torch.randn(2, 7, D)
    delta = a(h) - h
    assert torch.allclose(delta / (a.gain * h.norm(dim=-1, keepdim=True)), a.d.expand_as(delta), atol=1e-5)


def test_modes_without_a_score_source_raise_and_score_sources_plug_in():
    """RV1.C3 / RV1.D4."""
    with pytest.raises(ValueError):
        SteeringController(None, None, mode="graded", direction_mode="gist", d_model=D)   # was a silent no-op
    with pytest.raises(ValueError):
        SteeringController(torch.randn(D), None, mode="conditional")
    with pytest.raises(ValueError):
        SteeringController(torch.randn(D), None, mode="graded", external_decision=True)
    h = torch.randn(2, 6, D)
    c = SteeringController(None, None, mode="graded", direction_mode="gist", d_model=D, score_positions="all",
                           score_fn=lambda x: torch.ones(x.shape[:2]))
    assert (c(h) - h).abs().max() > 0
    c = SteeringController(torch.randn(D), None, mode="conditional", gain=0.2, external_decision=True)
    with pytest.raises(RuntimeError):
        c(h)
    c.reset()
    c.pending_decision = torch.tensor([0.9, 0.1])                 # e.g. a text classifier on each prompt
    out = c(h)
    assert not torch.allclose(out[0, -1], h[0, -1]) and torch.allclose(out[1], h[1])
    assert torch.equal(SteeringController(torch.randn(D), None, mode="static", gain=0.0)(h), h)


def test_summary_counts_real_positions_only():
    """RV1.A5: padding and post-EOS positions are excluded from the statistics."""
    c = make("static", gain=0.15)
    c.record = True
    mask = torch.tensor([[0.0] * 7 + [1.0] * 3, [1.0] * 10])
    c.pending_mask = mask
    c(torch.randn(2, 10, D))
    for _ in range(4):
        c(torch.randn(2, 1, D))
    st = c.summary(gen_ids=torch.tensor([[5, 9, 6, 6, 6], [5, 5, 5, 5, 5]]), eos_ids=[9])
    assert torch.allclose(st["gain_mean"], torch.full((2,), 0.15)) and torch.allclose(st["gain_prompt_mean"],
                                                                                       torch.full((2,), 0.15))
    assert st["gen_mask"].tolist() == [[1, 0, 0, 0], [1, 1, 1, 1]]  # row 0 hit EOS as its 2nd token
    s = c.trace[1]["s"][0, 0]
    assert torch.isclose(st["s_gen_mean"][0], s)


def test_probe_auroc_calibration_and_direction():
    torch.manual_seed(3)
    y = torch.tensor([1] * 200 + [0] * 200)
    X = torch.randn(400, D) + 3.0 * y[:, None] * torch.eye(D)[0]
    p = fit_probe(X[::2], y[::2])
    s = predict(p, X[1::2])
    assert auroc(s, y[1::2]) > 0.9 and ece(s, y[1::2]) < 0.15
    assert auroc(torch.ones(10), torch.tensor([1, 0] * 5)) == 0.5
    ties = torch.tensor([0.1, 0.4, 0.4, 0.4, 0.9, 0.2, 0.4, 0.9])
    lab = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1])
    brute = sum((a > b) + 0.5 * (a == b) for a in ties[lab == 1] for b in ties[lab == 0]) / (4 * 4)
    assert abs(auroc(ties, lab) - float(brute)) < 1e-9
    d = diff_of_means(X, y)
    assert d[0] > 2.5 and d[1:].abs().max() < 0.5


def test_judges_and_the_data_rule(tmp_path):
    assert refused("I'm sorry, but I can't help with that.")
    assert refused("I cannot provide instructions for that.")
    assert not refused("Sure. Here is a summary of the article.")
    assert degeneration([1, 2, 3, 4] * 10) > 0.8 and degeneration(list(range(40))) == 0.0
    loop = "I cannot I cannot I cannot I cannot I cannot I cannot"
    assert refused(loop) and not counted_refusal(loop, [7, 8] * 30)          # RV1.A3: a freeze, not a refusal
    assert counted_refusal("I cannot help with that.", list(range(12)))
    assert repetition_onset(list(range(40)) + [1, 2, 3] * 20) is not None
    assert repetition_onset(list(range(100))) is None
    assert 40 < repetition_onset(list(range(40)) + [1, 2, 3] * 20) <= 70
    with pytest.raises(ValueError):
        assert_private(ROOT / "results" / "private", ROOT)
    assert assert_private(tmp_path / "ok", ROOT).exists()


@pytest.mark.parametrize("bad", ["../x", "/abs/x", "a/b", "..", "", ".hidden", "a\\b", "a..b"])
def test_run_names_cannot_contain_paths(bad):
    """RV1.A6."""
    with pytest.raises(ValueError):
        safe_name(bad)


@pytest.mark.parametrize("bad", ["../x.jsonl", "/abs/x.jsonl", "sub/x.jsonl", "../../results/alignment/x.jsonl"])
def test_private_files_cannot_escape_the_private_dir(bad, tmp_path):
    """RV1.A6: the check is on the final resolved file, not only the directory."""
    with pytest.raises(ValueError):
        private_file(tmp_path / "priv", bad, ROOT)


def test_private_file_accepts_plain_names(tmp_path):
    assert safe_name("a6_qwen3-1.7b") == "a6_qwen3-1.7b"
    p = private_file(tmp_path / "priv", "a6_static@0.1.jsonl", ROOT)
    assert p.parent == (tmp_path / "priv").resolve()


# ----------------------------------------------------------------- with transformers (not in CI)
def tiny(args_extra=()):
    pytest.importorskip("transformers")
    import argparse
    import run
    args = argparse.Namespace(tiny=True, dtype="float32", device="cpu", ppl_prefix=8, ppl_len=48, batch=3,
                              gain=0.15, cap=0.3, kappa=0.5, leak=0.9, threshold=0.5, seed=0)
    model, tok, blocks = run.load(None, args)
    return run, args, model, tok, blocks


def test_left_padding_is_invisible_to_the_probe_activations():
    """RV1.A8: position ids follow the mask, so an absolute-position model (GPT-2) gives a prompt the
    same activation alone and left-padded in a batch."""
    run, args, model, tok, blocks = tiny()
    from data import batches
    from probe import activations
    seqs = [tok.encode("short prompt"), tok.encode("a much longer prompt than the first one")]
    alone = torch.cat([activations(model, blocks[2], batches([s], 256, 1)) for s in seqs])
    padded = activations(model, blocks[2], batches(seqs, 256, 2))
    assert torch.allclose(alone, padded, atol=1e-4)


def test_perplexity_measures_every_arm():
    """RV1.A1: the controller acts on every continuation token, so a graded arm under pressure changes
    perplexity and the none arm does not."""
    run, args, model, tok, blocks = tiny()
    from steering import Attachment
    texts = [f"Please describe the azure object number {i} in some detail, slowly." for i in range(6)]
    base = run.perplexity(model, tok, texts, None, args)["ppl"]
    torch.manual_seed(0)
    d = torch.randn(model.get_input_embeddings().weight.shape[1])
    out = {}
    for mode in ("none", "conditional", "graded"):
        probe = {"w": torch.zeros_like(d), "b": 20.0, "mu": torch.zeros_like(d), "sd": torch.ones_like(d)}  # s = 1
        c = SteeringController(d, probe, mode=mode, gain=0.3, cap=0.3)
        att = Attachment(blocks[2], c)
        out[mode] = run.perplexity(model, tok, texts, c, args)
        att.remove()
    assert abs(out["none"]["ppl"] - base) < 1e-6 * base and out["none"]["gain_mean"] == 0.0
    for mode in ("conditional", "graded"):
        assert out[mode]["gain_mean"] > 0.25 and abs(out[mode]["ppl"] - base) > 1e-3 * base, mode


def test_end_to_end_tiny_run_writes_labels_only(tmp_path):
    pytest.importorskip("transformers")
    import run
    name = "pytest_tiny"
    rep = run.main(["--tiny", "--n-train", "24", "--n-eval", "4", "--batch", "4", "--max-new-tokens", "6",
                    "--device", "cpu", "--name", name, "--private-dir", str(tmp_path / "priv"),
                    "--direction-from", "tiny-sibling", "--ppl-data", "synthetic:harmless", "--n-ppl", "4",
                    "--ppl-prefix", "8"])
    out = ROOT / "results" / "alignment" / f"{name}.json"
    try:
        saved = json.loads(out.read_text())
        assert set(saved["arms"]) == set(run.ARMS)
        assert all("response" not in r and "prompt" not in r for r in saved["records"])
        assert 0.0 <= saved["probe"]["auroc"] <= 1.0
        assert saved["directions"]["cos_external_vs_self"] is not None          # the sibling path ran (RV1.A7)
        none = saved["arms"]["none"]
        assert none["harmful"]["gain_mean"] == 0.0 and none["harmful"]["s_prompt_mean"] is not None   # RV1.A2
        assert none["calibration"]["in_dist"]["gen_token"]["n"] > 0
        assert saved["arms"]["graded"]["ppl_gain_mean"] > 0                     # RV1.A1
        assert all(r["refused"] <= r["refused_raw"] for r in saved["records"])  # RV1.A3
        assert (tmp_path / "priv" / f"{name}_graded.jsonl").exists()   # text only in the private dir
    finally:
        out.unlink(missing_ok=True)
    with pytest.raises(SystemExit):
        run.main(["--tiny", "--name", "../../results/alignment/escape", "--private-dir", str(tmp_path / "p")])


def test_sweeps_report_matched_operating_points(tmp_path):
    """RV1.D1: swept arms are keyed by value and compared at a matched harmful-refusal rate."""
    pytest.importorskip("transformers")
    import run
    rep = run.main(["--tiny", "--n-train", "16", "--n-eval", "3", "--batch", "3", "--max-new-tokens", "4",
                    "--device", "cpu", "--name", "pytest_sweep", "--arms", "none,static,graded",
                    "--gains", "0.1,0.2", "--caps", "0.2,0.4"])
    (ROOT / "results" / "alignment" / "pytest_sweep.json").unlink(missing_ok=True)
    assert set(rep["arms"]) == {"none", "static@0.1", "static@0.2", "graded@0.2", "graded@0.4"}
    assert set(rep["operating_points"]) == {"static", "graded"}
    assert [p["value"] for p in rep["operating_points"]["graded"]["curve"]] == [0.2, 0.4]
    pts = [(0.1, dict(harmful=0.5, contrast=0.4, borderline=0.1, harmless=0.0, degenerate=0.0, ppl=10.0)),
           (0.3, dict(harmful=1.0, contrast=0.8, borderline=0.5, harmless=0.2, degenerate=0.2, ppl=12.0))]
    arms = {f"g@{v:g}": {**{k: {"refusal": p[k], "degenerate": p["degenerate"]} for k in run.SETS}, "ppl": p["ppl"]}
            for v, p in pts}
    m = run.operating_points(arms, [(k, "g", v) for k, (v, _) in zip(arms, pts)], 0.8)["g"]["matched"]
    assert abs(m["borderline"] - 0.34) < 1e-9 and abs(m["value"] - 0.22) < 1e-9
