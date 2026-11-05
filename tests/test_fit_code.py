"""Gate G0.5's fit (experiments/evolve/fit_code.py): the KL between the teacher's and the student's
per-slot alignment, and the whole path on the tiny byte-tokenizer model (CPU, no downloads): train an
I1-like W1s for two steps with run.py, cache the teacher, fit W_z, save the fragment."""

import json
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from evolve import fit_code  # noqa: E402
from evolve import run as evolve_run  # noqa: E402

pl = evolve_run.pl
I1 = ["--w1s-taps", "", "--w1s-gate", "point", "--no-w1s-centre", "--w1s-pre", "0", "--w1s-key", "learned"]
TEXT = ("The river runs past the old mill, where the miller kept his records of every sack and every cart. "
        "In the spring the water rose and the wheel turned faster than anyone in the valley remembered. ") * 12


def _cache(S=3, T=40, D=12, N=4, n1=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    J = torch.randint(0, T - n1, (S, N, 1), generator=g) + torch.arange(n1)
    ok = torch.ones(S, N, n1, dtype=torch.bool)
    ok[:, -1, 3:] = False                                         # a clipped slot
    a = torch.rand(S, T, N, generator=g)
    valid = torch.rand(S, T, generator=g) > 0.2
    a = a * valid[..., None]
    return dict(x=torch.randn(S, T, D, generator=g), J=J, ok=ok, a=a, valid=valid,
                ans=torch.tensor([[T - 4, T - 3, T - 2]] * S))


def test_kl_zero_when_the_student_is_the_teacher():
    """A teacher made from the student's own code gives KL 0 and full argmax agreement; masking a
    position (ok = False) leaves the slot's softmax over the others."""
    c = _cache()
    wz, lb = torch.randn(6, 12), torch.tensor(math.log(8.0))
    z = F.normalize(c["x"] @ wz.T, dim=-1)
    c["logE"] = 8.0 * (z @ z.transpose(1, 2) - 1)
    assert fit_code.kl_loss(c, wz, lb).item() == pytest.approx(0.0, abs=1e-5)
    ev = fit_code.evaluate(c, wz, lb, "cpu")
    assert ev["kl"] == pytest.approx(0.0, abs=1e-5) and ev["argmax_agree_answer"] == 1.0
    other = fit_code.kl_loss(c, torch.randn(6, 12), lb).item()
    assert other > 1e-3
    c["logE"][0, :, c["J"][0, -1, 4]] += 50.0                     # a masked position is never read
    lT, _ = fit_code._slot_logits(c, wz, lb)
    assert lT[0, :, -1, 3:].exp().max().item() == 0.0


def test_init_wz_projects_out_the_mean():
    x = torch.randn(200, 16) + 3.0
    w = fit_code.init_wz(x, 8, "pca", torch.Generator().manual_seed(0))
    assert w.shape == (8, 16) and (w @ F.normalize(x.mean(0), dim=0)).abs().max() < 1e-5
    assert fit_code.init_wz(x, 24, "pca", torch.Generator().manual_seed(0)).shape == (24, 16)


def test_fit_code_end_to_end_tiny(tmp_path, monkeypatch):
    """run.py trains an I1-like W1s on the tiny model; fit_code.py caches its teacher on needle
    sequences (natural filler, held out from another file), fits a 16-d code, and saves a fragment
    with the eval's key names and shapes; the fit loss decreases."""
    for name in ("NeedleTask", "SideChannels", "HookedLM", "train", "evaluate", "scores"):
        monkeypatch.setattr(pl, name, getattr(pl, name))
    tr, te = tmp_path / "train.txt", tmp_path / "test.txt"
    tr.write_text(TEXT), te.write_text(TEXT[::-1])
    res = tmp_path / "i1.json"
    evolve_run.main(["--tiny", "--task", "needle", "--window", "64", "--seq-len", "512", "--pairs", "3",
                     "--steps", "2", "--batch", "2", "--eval-n", "2", "--device", "cpu", "--needle-scatter", "8",
                     "--needle-names", "train", "--layer", "3", "--text-file", str(tr),
                     "--candidate", "evolve.w1s_slots:W1sSide", "--out", str(res)] + I1)
    out = tmp_path / "frag" / "wz.pt"
    rep = fit_code.main([str(res), "--dc", "16", "--n-seq", "6", "--n-heldout", "3", "--steps", "40",
                         "--lr", "1e-2", "--fit-batch", "3", "--batch", "3", "--log-every", "10", "--tiny",
                         "--text-file", str(tr), "--heldout-text-file", str(te), "--device", "cpu",
                         "--out", str(out)])[16]
    frag = torch.load(out)
    assert set(frag) == {"loop.wz.weight", "loop.log_beta_align"}
    assert frag["loop.wz.weight"].shape == (16, 64) and frag["loop.log_beta_align"].shape == ()
    js = json.loads(out.with_suffix(".json").read_text())
    assert js["dc"] == 16 and js["query_positions_train"] > 0 and not js["missing_keys"]
    assert js["fit_loss"] < js["fit_loss_init"] and math.isfinite(js["heldout_kl"])
    for k in ("argmax_agree_answer", "argmax_agree_digit1"):
        v = js["final_eval"]["heldout"][k]
        assert v is None or 0.0 <= v <= 1.0
    assert rep["fragment"] == str(out)
    with pytest.raises(SystemExit, match="tiny"):                   # --tiny refuses a real-model result
        rec = json.loads(res.read_text())
        rec["args"]["tiny"] = False
        fake = tmp_path / "real.json"
        fake.write_text(json.dumps(rec))
        fit_code.main([str(fake), "--tiny", "--out", str(out)])
    # the fragment loads into a learned-code W1s, once that flag exists (round 11, w1s_slots.py)
    from evolve import w1s_slots
    if "w1s_code" not in vars(w1s_slots.split_args([])[0]):
        pytest.skip("--w1s-code is not in w1s_slots.py yet")
    rec = json.loads(res.read_text())
    flags = {**vars(w1s_slots.split_args([])[0]), **rec["candidate"]["args"], "w1s_code": "learned", "w1s_dc": 16}
    import argparse
    side = w1s_slots.W1sSide(64, argparse.Namespace(**rec["args"]), argparse.Namespace(**flags))
    side.load_state_dict(torch.load(res.with_suffix(".pt")), strict=False)
    missing, unexpected = side.load_state_dict(frag, strict=False)
    assert not unexpected and torch.equal(side.loop.wz.weight, frag["loop.wz.weight"])
    # the eval's learned alignment is the student fitted here (same code, β, clamp, no centring)
    assert fit_code.MAX_BETA == w1s_slots.MAX_BETA
    loop, h = side.loop, torch.randn(1, 96, 64)
    with torch.no_grad():
        f, st = loop._run_store(loop.norm(h), h, None, None)
        E = loop._alignment(f, st)
        z = F.normalize(f["x"] @ frag["loop.wz.weight"].T, dim=-1)
        beta = frag["loop.log_beta_align"].exp().clamp(max=fit_code.MAX_BETA)
    # equal up to the per-row shift m_t (the best held match), which the per-slot normalisation cancels
    d = torch.log(E) - beta * (z @ z.transpose(1, 2) - 1)
    torch.testing.assert_close(d, d[..., :1].expand_as(d), atol=1e-4, rtol=0)
