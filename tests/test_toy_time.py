"""Smoke tests of the G0.8 CPU toy (experiments/evolve/toy_time.py): the task's invariants, the time forms'
identities at initialisation, seed independence of the seed-batched training, and a tiny end-to-end run."""

import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from evolve import toy_time as tt  # noqa: E402

TINY = ["--seeds", "2", "--steps", "6", "--eval-every", "3", "--bank-n", "64", "--eval-n", "8", "--threads", "1"]


def test_streams_are_well_formed():
    c = tt.parse(TINY + ["--out", ""])
    rng = random.Random(0)
    for q in (0, 1, 2) * 20:
        for T in (192, 768):
            r = tt.make_seq(rng, T, q, c)
            toks, ons = r["toks"], [s for s in r["ons"] if s >= 0]
            assert len(toks) == T and toks[-1] == tt.MARK and ons == sorted(ons)
            stmt = [s for s in ons if toks[s - 1] == tt.MARK and tt.NAME0 <= toks[s - 2] < tt.NAME0 + tt.NN]
            assert len(stmt) in (3, 4) and len(ons) == len(stmt) + c.n_false
            d1 = [toks[s] - tt.DIG0 for s in stmt]
            assert len(set(d1)) == len(d1)                        # distinct first digits
            assert toks[r["ons"][r["tgt"]]] - tt.DIG0 == r["ans"]
            for s in set(ons) - set(stmt):                        # false windows: apart from every other window
                assert min(abs(s - o) for o in ons if o != s) >= c.win and s - c.pre >= 0
            if q == 1:
                assert r["stale"] >= 0 and toks[r["ons"][r["stale_slot"]]] - tt.DIG0 == r["stale"]
                assert r["ons"][r["stale_slot"]] < r["ons"][r["tgt"]] and toks[-2] == toks[r["ons"][r["tgt"]] - 2]
            if q == 2:
                assert r["tgt"] == (ons.index(stmt[0]) if r["osub"] == 0 else ons.index(stmt[-1]))


def test_rope_depends_on_lag_only():
    f = tt.rope_freqs(16, "64,65536")
    q, k = torch.randn(32), torch.randn(32)
    a = tt.rotate(q, torch.tensor(500), f) @ tt.rotate(k, torch.tensor(430), f)
    b = tt.rotate(q, torch.tensor(90), f) @ tt.rotate(k, torch.tensor(20), f)
    assert torch.allclose(a, b, atol=1e-4)
    assert torch.allclose(tt.rotate(q, torch.tensor(77), f).norm(), q.norm())
    assert torch.equal(tt.rotate(q, torch.tensor(77), f)[16:], q[16:])


def test_time_forms_start_as_content_only():
    """ordinal (E = 0) and supr at w_s = 0 read exactly as the content-only arm at initialisation."""
    c = tt.parse(TINY + ["--out", ""])
    E, M = tt.world()
    bk = tt.make_bank(16, 3, c.T, c)
    xw, xq, s, valid, T = tt.inputs(bk, None, 2, E, M, c)
    ref, _ = tt.SlotToy("none", [0, 1], c)(xw, xq, s, valid, T - 1)
    out, _ = tt.SlotToy("ordinal", [0, 1], c)(xw, xq, s, valid, T - 1)
    assert torch.equal(out, ref)
    m = tt.SlotToy("supr", [0, 1], c)
    with torch.no_grad():
        m.ws.zero_()
    assert torch.allclose(m(xw, xq, s, valid, T - 1)[0], ref, atol=1e-6)


def test_seeds_are_independent():
    """Seed 1 trained beside seed 0 equals seed 1 trained alone (per-seed loss, clip and Adam)."""
    E, M = tt.world()
    res = {}
    for seeds, seed0 in ((2, 0), (1, 1)):
        c = tt.parse(TINY + ["--seeds", str(seeds), "--seed0", str(seed0), "--arms", "rope", "--out", ""])
        trb = tt.make_bank(64, 1, c.T, c)
        evb = {1: tt.make_bank(24, 5, c.T, c)}
        res[seeds] = tt.train_arm("rope", c, trb, evb, E, M, log=lambda s: None)["runs"][-1]
    assert res[2]["seed"] == res[1]["seed"] == 1
    assert res[2]["final"] == res[1]["final"]
    for k in ("beta", "beta_align", "null"):
        assert abs(res[2]["params"][k] - res[1]["params"][k]) < 1e-5


def test_main_smoke(tmp_path):
    out = tmp_path / "g08.json"
    r = tt.main(TINY + ["--out", str(out)])
    j = json.loads(out.read_text())
    assert set(j["arms"]) == set(tt.ARMS) and set(j["summary"]) == set(tt.ARMS)
    for arm, sm in j["summary"].items():
        assert sm["seeds"] == 2 and 0 <= sm["learned"] <= 2
        for mult in (1, 2, 4):
            row = sm[f"x{mult}"]
            for k in ("plain", "update", "stale", "order", "order_first", "order_last"):
                assert 0.0 <= row[k] <= 1.0
    assert {"rope", "ordinal", "any_time_data_pass"} <= set(j["gate"]["literal"]) and "calibration" in j["gate"]
    assert r["arms"]["rope"]["secs_per_step"] > 0
