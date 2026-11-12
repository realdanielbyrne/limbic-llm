"""M1a (experiments/ca2/adapters.py): the memory-free adapter's parameter match and memory-freeness, the
evaluation-time ablations of the trained loop (foreign, zero, in-block), determinism, the per-sequence
evaluation against ``pretrained_loop.evaluate``, the bootstrap summary and the runner end to end on a tiny model.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from ca2 import adapters as A  # noqa: E402

pl = A.pl
P5 = ROOT / "results" / "pretrained" / "pretrained_qwen3_lm.json"


def p5_args(**kw):
    args = pl.parse_args([])
    for k, v in json.loads(P5.read_text())["args"].items():
        setattr(args, k, v)
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def loop(**kw):
    torch.manual_seed(0)
    base = dict(tau=4, d_key=8)
    base.update(kw)
    return A.HippocampalLoop(16, **base)


# ------------------------------------------------------------------------------------- parameter match
@pytest.mark.parametrize("d", [64, 2048])
def test_mfa_matches_loop_parameters_within_one_percent(d):
    args = p5_args()
    target = A.loop_param_count(d, args)
    if d == 2048:
        assert target == 4_380_675                                  # P5's recorded trainable_params
    side = A.build_side("mfa", d, args, seed=0)
    assert A.n_trainable(side) == A.mfa_params(d, side.rank)
    assert abs(A.n_trainable(side) / target - 1) < 0.01
    assert A.n_trainable(A.build_side("loop", d, args, seed=0)) == target


def test_match_rank_refuses_impossible_target():
    with pytest.raises(ValueError):
        A.match_rank(64, 10)


# --------------------------------------------------------------------------------------- memory-free
def test_mfa_output_at_t_depends_only_on_h_t():
    torch.manual_seed(0)
    mfa = A.MemoryFreeAdapter(16, rank=12)
    h = torch.randn(2, 24, 16)
    out, _ = mfa(h)
    for t in range(24):
        h2 = h.clone()
        h2[:, :t] = torch.randn(2, t, 16) * 3                     # every earlier position changed
        h2[:, t + 1:] = torch.randn(2, 23 - t, 16)                 # and every later one
        out2, _ = mfa(h2)
        assert torch.equal(out2[:, t], out[:, t])
    single = torch.cat([mfa(h[:, t:t + 1])[0] for t in range(24)], 1)
    torch.testing.assert_close(single, out, rtol=1e-6, atol=1e-6)
    h3 = h.clone()
    h3[:, 5] += 1.0                                               # it does depend on h_t itself
    assert not torch.equal(mfa(h3)[0][:, 5], out[:, 5])


def test_mfa_uses_the_ca1_bound_and_projection():
    torch.manual_seed(0)
    mfa = A.MemoryFreeAdapter(16, rank=12)
    h = torch.randn(3, 10, 16) * 5
    out, st = mfa(h)
    inj = out - h
    torch.testing.assert_close(inj.norm(dim=-1), st.gain * h.norm(dim=-1), rtol=1e-4, atol=1e-5)
    assert (st.gain <= 0.9 + 1e-6).all() and (st.gain > 0).all()
    cos = (inj * h).sum(-1) / (inj.norm(dim=-1) * h.norm(dim=-1))
    assert cos.abs().max() < 1e-4                                 # error-only: orthogonal to h_t
    assert mfa.loop is mfa and mfa.loop.tau == 0


# ---------------------------------------------------------------------------------------- ablations
def test_zero_ablation_removes_exactly_the_memory_injection():
    lp = loop()
    h = torch.randn(2, 30, 16)
    with A.ablate(lp, "zero"):
        out, _ = lp(h)
    assert torch.equal(out, h)
    torch.manual_seed(1)
    side = pl.SideChannels(16, tau=4, d_key=8)
    with torch.no_grad():
        side.time.out.weight.normal_()                            # a trained-looking clock
        with A.ablate(side.loop, "zero"):
            out, _ = side(h)
        assert torch.equal(out, side.time(h)[0])                   # the side reduces to the TimeChannel


def test_foreign_ablation_reads_only_the_donor_memory():
    lp = loop(ca1_control=False)                                   # no energy feedback: memory is the only path
    h = torch.randn(3, 30, 16)
    with torch.no_grad(), A.ablate(lp, "foreign"):
        base, _ = lp(h)
        same, _ = lp(h[:1].expand(3, -1, -1).contiguous())
        t = 20
        h_own = h.clone()
        h_own[1, :t] = torch.randn(t, 16)                          # row 1's own past changed, h_t kept
        own, _ = lp(h_own)
        h_don = h.clone()
        h_don[0, :t] = torch.randn(t, 16)                          # row 1's donor (row 0) changed
        don, _ = lp(h_don)
    plain, _ = lp(h)
    assert torch.equal(same[0], same[1])                           # identical rows: the roll changes nothing
    torch.testing.assert_close(same[0], lp(h[:1])[0][0].detach(), rtol=1e-6, atol=1e-6)   # (batch-size rounding)
    assert torch.equal(own[1, t], base[1, t])
    assert not torch.equal(don[1, t], base[1, t])
    assert not torch.equal(base, plain)
    assert torch.equal(base[:, :4], h[:, :4])                     # before τ nothing returns, ablated or not
    assert A.donors(A.strided_groups(8, 4)) == {0: 6, 2: 0, 4: 2, 6: 4, 1: 7, 3: 1, 5: 3, 7: 5}


def test_in_block_ablation_masks_reads_to_the_current_block():
    lp = loop(ca1_control=False)
    h = torch.randn(2, 24, 16)
    with torch.no_grad():
        with A.ablate(lp, "in_block", block=24):
            one, _ = lp(h)
        assert torch.equal(one, lp(h)[0])                         # one block: identical to the intact loop
        h2 = h.clone()
        h2[:, :12] = torch.randn(2, 12, 16)                        # change block 0 (block = 12)
        with A.ablate(lp, "in_block", block=12):
            a, _ = lp(h)
            b, _ = lp(h2)
        assert torch.equal(a[:, 12:], b[:, 12:])
        assert not torch.equal(lp(h)[0][:, 12:], lp(h2)[0][:, 12:])


def test_ablations_restore_state_and_refuse_bad_inputs():
    lp = loop()
    h = torch.randn(2, 20, 16)
    ref = lp(h)[0]
    sd = {k: v.clone() for k, v in lp.state_dict().items()}
    orig = A.TL.loop_kernel
    for mode in A.ABLATIONS:
        with A.ablate(lp, mode, block=8):
            lp(h)
    assert A.TL.loop_kernel is orig and torch.equal(lp(h)[0], ref)
    assert all(torch.equal(v, lp.state_dict()[k]) for k, v in sd.items())
    with pytest.raises(TypeError):
        with A.ablate(A.MemoryFreeAdapter(16, 4), "zero"):
            pass
    with pytest.raises(ValueError):
        with A.ablate(loop(loop="closed"), "foreign"):
            pass


# ------------------------------------------------------------------------------- determinism, eval
def test_build_side_is_deterministic_given_a_seed():
    args = p5_args()
    for arm in A.ARMS:
        a, b, c = (A.build_side(arm, 64, args, s) for s in (0, 0, 1))
        assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
        assert any(not torch.equal(a.state_dict()[k], c.state_dict()[k]) for k in a.state_dict()
                   if a.state_dict()[k].numel() > 1)


def tiny_lm(side_kind="loop"):
    model, tok = pl.build_tiny("gpt2", seed=0)
    model.eval()                                                  # as pl.load_model: no dropout
    for p in model.parameters():
        p.requires_grad_(False)
    args = p5_args(tau=4)
    side = A.build_side(side_kind, 64, args, seed=0)
    return pl.HookedLM(model, side, window=8, mode="block", sink=1, pad_id=tok.eos_token_id), tok


def test_eval_rows_matches_pretrained_loop_evaluate():
    lm, tok = tiny_lm()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 256, (8, 33), generator=g)
    ids[:, 0] = 256
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 10:] = True
    batches = [(ids[i:i + 4], mask[i:i + 4], None) for i in (0, 4)]
    groups = A.strided_groups(8, 4)
    for w, s in ((8, False), (8, True), (0, False)):
        ref = pl.evaluate(lm, batches, w, s, "lm")
        got = A.eval_rows(lm, ids, mask, groups, w, s, block=8, buckets=4)
        assert got["loss"] == pytest.approx(ref["loss"], rel=1e-5)
        assert got["token_acc"] == pytest.approx(ref["token_acc"], rel=1e-6)
    zero = A.eval_rows(lm, ids, mask, groups, 8, True, ablation="zero", block=8)
    assert zero["loss"] != pytest.approx(A.eval_rows(lm, ids, mask, groups, 8, True, block=8)["loss"], rel=1e-9)


def test_summarise_closure_and_paired_difference():
    rng = np.random.default_rng(0)
    Lw = 3 + rng.random(40)
    Lf = Lw - 1 - 0.2 * rng.random(40)
    rec = {"base": {"windowed": {"per_seq": Lw.tolist(), "loss": Lw.mean()},
                    "full": {"per_seq": Lf.tolist(), "loss": Lf.mean()}}, "cells": {}}
    for arm, c in (("loop", 0.5), ("mfa", 0.4)):
        for s in (0, 1):
            La = Lw - c * (Lw - Lf)                                # exactly c of every sequence's gap
            rec["cells"][f"{arm}_s{s}"] = {"arm": arm, "seed": s, "windowed+side": {"per_seq": La.tolist()},
                                           "ablations": {"zero": {"per_seq": Lw.tolist()}} if arm == "loop" else {}}
    s = A.summarise(rec, n_boot=500)
    assert s["closure_points"]["loop/side/mean"]["point"] == pytest.approx(50)
    assert s["closure_points"]["loop/side/s0"]["lo"] == pytest.approx(50)
    d = s["loop_minus_mfa"]["mean"]
    assert d["point"] == pytest.approx(10) and d["lo"] == pytest.approx(10) and s["gate"]["passes"]
    assert s["ablation_drop"]["zero"]["mean"]["point"] == pytest.approx(50)


# ------------------------------------------------------------------------------------------- runner
def test_runner_end_to_end_tiny(tmp_path, capsys):
    rng = random.Random(0)
    words = "the a river city king war song ship old new north south year built which was is of in and".split()
    (tmp_path / "t.txt").write_text(" ".join(rng.choice(words) for _ in range(6000)))
    out = tmp_path / "m1a.json"
    argv = ["--replicate", str(P5), "--tiny", "--dtype", "float32", "--device", "cpu", "--window", "8",
            "--seq-len", "32", "--steps", "3", "--eval-n", "8", "--tau", "4", "--text-file", str(tmp_path / "t.txt"),
            "--out", str(out), "--ckpt-dir", str(tmp_path / "ck"), "--boot", "200"]
    rec = A.main(argv)
    assert rec["args"]["kernel"] == "osc" and rec["args"]["window_mode"] == "block"      # from P5
    assert rec["args"]["window"] == 8 and rec["args"]["batch"] == 4 and rec["args"]["grad_checkpointing"]
    assert set(rec["cells"]) == {"loop_s0", "mfa_s0", "loop_s1", "mfa_s1"}
    assert set(rec["cells"]["loop_s0"]["ablations"]) == set(A.ABLATIONS)
    assert abs(rec["model"]["param_mismatch"]) < 0.01
    for c in rec["cells"].values():
        assert Path(c["checkpoint"]).exists()
        assert c["params"] in (rec["model"]["loop_params"], rec["model"]["mfa_params"])
        assert c["p5_format"]["loss"] == pytest.approx(c["windowed+side"]["loss"], rel=1e-5)
    zero = rec["cells"]["loop_s0"]["ablations"]["zero"]
    assert zero["inj_ratio"] < rec["cells"]["loop_s0"]["windowed+side"]["inj_ratio"]
    assert {"loop_minus_mfa", "ablation_drop", "gate", "closure_by_offset", "no_harm"} <= set(rec["summary"])
    saved = json.loads(out.read_text())
    assert saved["summary"]["loop_minus_mfa"]["mean"]["point"] == rec["summary"]["loop_minus_mfa"]["mean"]["point"]

    again = A.main(argv[:-4] + ["--out", str(tmp_path / "m1a2.json"), "--ckpt-dir", str(tmp_path / "ck")])
    for k in rec["cells"]:                                         # same seeds: bitwise the same run
        assert again["cells"][k]["windowed+side"]["per_seq"] == rec["cells"][k]["windowed+side"]["per_seq"]

    capsys.readouterr()
    resumed = A.main(argv + ["--resume"])
    assert "resume: 4 finished cells" in capsys.readouterr().out
    assert resumed["cells"]["loop_s1"]["curve"] == rec["cells"]["loop_s1"]["curve"]


def test_replicate_matches_exact_flags_only():
    """A flag given on the command line overrides only itself: --window keeps P5's window_mode / window_layers."""
    q35 = ROOT / "results" / "pretrained" / "pretrained_qwen35_lm.json"
    src = json.loads(q35.read_text())["args"]
    assert src["window_layers"] == "attention"
    _, args = A.parse(["--replicate", str(q35), "--window", "256", "--no-sink"])
    assert args.window == 256 and args.sink is False
    assert args.window_layers == src["window_layers"] and args.window_mode == src["window_mode"]
    with pytest.raises(SystemExit):
        A.parse(["--replicate", str(q35), "--wind", "256"])


def test_runner_refuses_checkpoints_in_the_repo():
    with pytest.raises(SystemExit):
        A._ckpt_dir(str(ROOT / "results"))
