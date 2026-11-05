"""Round-11 teacher arms (experiments/evolve/teacher.py, wired by run.py): A14 self-distillation from full
attention and A15 the teacher attention-mass admission target, plus diag_teacher_attention.py. Tiny
byte-tokenizer model only (CPU, no downloads)."""

import json
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from evolve import diag_teacher_attention as diag  # noqa: E402
from evolve import run as evolve_run  # noqa: E402
from evolve import teacher  # noqa: E402
from evolve.taps import TapLM  # noqa: E402
from evolve.w1s_slots import W1sSide, split_args  # noqa: E402

pl = evolve_run.pl
BASE = ["--tiny", "--task", "needle", "--window", "64", "--seq-len", "512", "--pairs", "3", "--steps", "2",
        "--batch", "2", "--eval-n", "4", "--device", "cpu", "--needle-scatter", "8", "--layer", "3", "--seed", "3",
        "--candidate", "evolve.w1s_slots:W1sSide", "--w1s-taps", "1"]


def keep_pl(monkeypatch):
    """run.main leaves pretrained_loop's task, side, train and evaluate patched; undo after the test."""
    for name in ("NeedleTask", "SideChannels", "HookedLM", "train", "evaluate", "scores"):
        monkeypatch.setattr(pl, name, getattr(pl, name))


def corpus(tmp_path) -> str:
    rng = random.Random(0)
    words = ["the", "river", "north", "station", "was", "built", "in", "a", "later", "year", "and", "its", "name"]
    p = tmp_path / "text.txt"
    p.write_text(" ".join(rng.choice(words) for _ in range(6000)))
    return str(p)


def strip(rec: dict) -> dict:
    rec = dict(rec)
    rec.pop("seconds")
    rec["args"] = {k: v for k, v in rec["args"].items() if k != "out"}
    return rec


# ------------------------------------------------------------------- KL
@pytest.mark.parametrize("temp,chunk", [(1.0, 3), (2.0, 7), (0.5, 100)])
def test_chunked_kl_equals_the_direct_kl_and_its_gradient(temp, chunk):
    """The chunked, checkpointed T²·KL(teacher ‖ student) from hidden states equals F.kl_div on the full
    logits, in value and in the student's gradient (small vocabulary)."""
    torch.manual_seed(0)
    head = nn.Linear(8, 37, bias=True)
    s_h = torch.randn(2, 7, 8, requires_grad=True)
    t_h = torch.randn(2, 7, 8)
    kl, agree = teacher.chunked_kl(s_h, t_h, head, temp, chunk)
    ref = teacher.kl_reference(head(t_h), head(s_h), temp)
    assert kl.shape == (2, 7) and torch.allclose(kl, ref, atol=1e-6)
    assert torch.equal(agree, head(t_h).argmax(-1) == head(s_h).argmax(-1))
    g1, = torch.autograd.grad(kl.mean(), s_h)
    g2, = torch.autograd.grad(teacher.kl_reference(head(t_h), head(s_h), temp).mean(), s_h)
    assert torch.allclose(g1, g2, atol=1e-6)
    direct = (torch.softmax(head(t_h) / temp, -1) * (torch.log_softmax(head(t_h) / temp, -1)
                                                   - torch.log_softmax(head(s_h) / temp, -1))).sum(-1) * temp ** 2
    assert torch.allclose(ref, direct, atol=1e-6)
    assert teacher.chunked_kl(t_h, t_h, head, temp, chunk)[0].abs().max() < 1e-6     # same input: 0


def test_text_windows_are_contiguous_with_the_sink(tmp_path):
    tok = pl.ByteTokenizer()
    tw = teacher.TextWindows(tok, corpus(tmp_path), 64, 1)
    ids = tw.batch(3, random.Random(0))
    assert ids.shape == (3, 65) and (ids[:, 0] == tok.bos_token_id).all()
    full = tw.ids.tolist()
    for row in ids[:, 1:].tolist():                                 # a contiguous slice of the corpus
        assert any(full[i:i + 64] == row for i in range(len(full) - 63))


def sharp_lm(window=64):
    """The tiny GPT-2 with sharper attention and logits (attention maps and final LayerNorm scaled up),
    so that full and windowed outputs differ visibly (KL ≈ 0.0003 at random init), and a W1s side at
    the last block."""
    model, tok = pl.build_tiny("gpt2", seed=0)
    with torch.no_grad():
        for b in model.transformer.h:
            b.attn.c_attn.weight.mul_(3)
            b.attn.c_proj.weight.mul_(3)
        model.transformer.ln_f.weight.mul_(10)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(0)
    side = W1sSide(64, type("A", (), {"time": True})(), split_args(["--w1s-taps", "1"])[0])
    lm = TapLM(model, side, 3, window, "block", "all", 1, pad_id=tok.eos_token_id)
    return lm, tok, side


def test_distill_loss_is_finite_and_decreases(tmp_path):
    """A14 on a fixed text batch: the student (windowed + W1s) moves towards the full-attention teacher."""
    lm, tok, side = sharp_lm()
    ns = teacher.split_args(["--distill-frac", "0.5", "--distill-positions", "128", "--distill-chunk", "40"])[0]
    dist = teacher.Distiller(lm, ns)
    ids = teacher.TextWindows(tok, corpus(tmp_path), 256, 1).batch(2, random.Random(1))
    opt = torch.optim.Adam([p for p in side.parameters() if p.requires_grad], lr=3e-3)
    losses = []
    for _ in range(8):
        with lm.condition(64, True):
            kl, agree, m = dist(lm, ids)
        loss = (kl * m).sum() / m.sum()
        assert kl.shape == (2, 128) and torch.isfinite(kl).all() and m.all()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[0] > 0.01 and losses[-1] < losses[0]
    assert dist.losses == pytest.approx(losses)
    with torch.no_grad():                                           # the teacher is the full model, side off
        with lm.condition(0, False):
            t_h, t_lg = teacher.hidden_states(lm, ids)
        assert torch.allclose(lm.model.get_output_embeddings()(t_h[:, -1:]), t_lg, atol=1e-4)


# ------------------------------------------------------------- attention
def test_capture_attention_matches_an_eager_model_and_leaves_no_trace():
    """Eager weights captured at chosen layers of an sdpa model (config proxy + causal mask) equal those
    of the same weights loaded with eager attention; they are causal; logits are unchanged; and the
    configs and hooks are restored."""
    model, tok = pl.build_tiny("gpt2", seed=0, attn_impl="sdpa")
    ref, _ = pl.build_tiny("gpt2", seed=0, attn_impl="eager")
    for m in (model, ref):
        m.eval()
    lm = pl.HookedLM(model, None, 3, 64, "block", "all", 1, pad_id=tok.eos_token_id)
    lm.window, lm.side_on = 0, False                                # the plain model, for the reference logits
    ids = torch.randint(0, 256, (2, 40))
    ids[:, 0] = tok.bos_token_id
    mods = [pl._child(lm.blocks[i], pl.ATTN_MIXERS) for i in range(4)]
    cfgs = [m.config for m in mods]
    nhooks = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in mods]
    got, lg = teacher.capture_attention(lm, ids, [1, 3], rows=5, keep=3)
    with torch.no_grad():
        att = ref(input_ids=ids, output_attentions=True, use_cache=False).attentions
        plain = model(input_ids=ids, use_cache=False).logits[:, -3:]
    assert set(got) == {1, 3}
    for i in (1, 3):
        assert got[i].shape == (2, 5, 40)
        assert torch.allclose(got[i], att[i][:, :, -5:].float().mean(1), atol=1e-5)
        assert torch.allclose(got[i].sum(-1), torch.ones(2, 5), atol=1e-5)
        rows = torch.arange(35, 40)
        assert got[i][:, rows[:, None] < torch.arange(40)[None]].abs().max() == 0     # nothing from the future
    assert torch.allclose(lg, plain, atol=1e-5)
    assert [m.config for m in mods] == cfgs and all(type(c).__name__ != "_EagerConfig" for c in cfgs)
    assert [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in mods] == nhooks


def test_window_targets_on_a_constructed_pattern():
    """The target puts the query rows' attention mass on the candidate windows that hold it, counts
    overlapping windows each, ignores candidates not closed before the query, and normalises."""
    T, sink = 101, 1
    att = torch.zeros(1, T)
    att[0, sink + 20:sink + 24] = 0.125                             # 0.5 on side positions 20–23
    att[0, sink + 50] = 0.3
    att[0, sink + 90] = 0.2                                         # inside the query's own span: not a window
    s = torch.tensor([[22, 52, 80, 95, 24, 0]])
    valid = torch.tensor([[True, True, True, True, True, False]])
    P, n, delay = 2, 6, 3                                           # windows [s−2, s+3]
    q0 = torch.tensor([sink + 85])
    tg = teacher.window_targets(att, s, valid, P, n, delay, q0, sink)
    # masses: [20,25] 0.5; [50,55] 0.3; [78,83] 0; [93,98] not closed (95 + 3 ≥ 85); [22,27] 0.25; invalid
    assert tg["elig"].tolist() == [[True, True, True, False, True, False]]
    assert torch.allclose(tg["target"], torch.tensor([[0.5, 0.3, 0.0, 0.0, 0.25, 0.0]]) / 1.05)
    assert tg["has"].tolist() == [True] and torch.allclose(tg["covered"], torch.tensor([1.0]))
    # a student equal to the target has KL 0; a flat one Σ t log(t / q); the gradient is finite
    t = tg["target"]
    flat = torch.zeros(1, 6, requires_grad=True)
    kl = teacher.admit_kl(flat, t, tg["elig"], tg["has"])
    q = 1 / 4                                                        # four eligible candidates
    want = sum(x * torch.log(torch.tensor(x / q)) for x in (t[0, 0].item(), t[0, 1].item(), t[0, 4].item()))
    assert torch.allclose(kl, want.reshape(1), atol=1e-6)
    kl.sum().backward()
    assert torch.isfinite(flat.grad).all() and flat.grad[0, 3] == 0 and flat.grad[0, 5] == 0
    match = torch.log(t.clamp_min(1e-12))                            # the target itself (eligible ones)
    assert teacher.admit_kl(match, t, tg["elig"], tg["has"]).abs().item() < 1e-5
    none = teacher.window_targets(att, s, valid, P, n, delay, torch.tensor([sink + 10]), sink)   # nothing closed
    assert none["has"].tolist() == [False] and teacher.admit_kl(flat, none["target"], none["elig"], none["has"]) == 0


def test_admission_gradient_reaches_only_the_gate():
    """A15 on a needle batch: the added term is λ·KL, and its gradient reaches the gate and nothing else."""
    lm, tok, side = sharp_lm()
    ns = teacher.split_args(["--admit-target", "teacher", "--admit-weight", "2.0", "--admit-layers", "1-3"])[0]
    adm = teacher.Admitter(lm, ns)
    assert adm.layers == [1, 2, 3]
    task = evolve_run.ScatterNeedleTask(tok, 512, 3, 1, None, min_gap=8, tail=256, block=64)
    proxy = teacher._Batches(task, None, 0.0, 0, True)
    ids, mask, _ = proxy.batch(2, random.Random(0))
    assert len(ids.query_len) == 2 and "_record" not in vars(task)
    seen = {}

    def prev(lm_, ids_, mask_):
        ce, ok, m = pl.scores(lm_, ids_, mask_)
        seen["ce"] = ce
        return ce, ok, m
    with lm.condition(64, True):
        ce, ok, m = adm(lm, ids, mask, prev)
    term = (ce - seen["ce"])[0, 0]
    assert torch.allclose(ce - seen["ce"], term.expand_as(ce)) and term.item() > 0
    assert term.item() == pytest.approx(2.0 * adm.losses[-1], rel=1e-5)
    assert lm.last_stats is not None                                # the student ran last
    side.zero_grad()
    term.backward()
    got = {k for k, p in side.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    gate = {"loop.gate_a", "loop.gate_b", "loop.lam_logit", "loop.latch_c", "loop.gate_w.0.weight",
            "loop.gate_w.1.weight"}
    # the gate and its input streams only: tap 1 and the main tap h, which is the TimeChannel's output
    assert gate <= got and got <= gate | {"time.out.weight", "time.out.bias"}
    assert "_store" not in vars(side.loop)                          # the store tap is gone


# --------------------------------------------------------------- runner
def test_teachers_off_train_exactly_as_before(tmp_path, monkeypatch):
    """With A14/A15 off, run.py trains byte-identically to pl.train without the teacher wrapper (same final
    side state_dict and result JSON after 2 steps, fixed seed): pl.train gets the task itself and the
    unpatched pl.scores, and the JSON has no distill/admit records."""
    keep_pl(monkeypatch)
    seen = []
    real = evolve_run._train

    def spy(lm, task, args):
        seen.append((type(task).__name__, pl.scores))
        return real(lm, task, args)
    monkeypatch.setattr(evolve_run, "_train", spy)
    a = tmp_path / "a.json"
    evolve_run.main(BASE + ["--out", str(a)])
    b = tmp_path / "b.json"
    evolve_run.main(BASE + ["--distill-frac", "0", "--admit-target", "none", "--out", str(b)])
    monkeypatch.setattr(evolve_run, "train", real)                  # the runner without the wrapper at all
    c = tmp_path / "c.json"
    evolve_run.main(BASE + ["--out", str(c)])
    assert seen == [("ScatterNeedleTask", evolve_run._scores)] * 2
    sd = [torch.load(p.with_suffix(".pt")) for p in (a, b, c)]
    for other in sd[1:]:
        assert sd[0].keys() == other.keys() and all(torch.equal(sd[0][k], other[k]) for k in sd[0])
    recs = [strip(json.loads(p.read_text())) for p in (a, b, c)]
    assert recs[0] == recs[1] == recs[2] and "distill" not in recs[0] and "admit" not in recs[0]


def test_distill_runs_end_to_end_and_records(tmp_path, monkeypatch):
    keep_pl(monkeypatch)
    out = tmp_path / "d.json"
    evolve_run.main(BASE[:BASE.index("--steps") + 1] + ["6"] + BASE[BASE.index("--steps") + 2:]
                    + ["--distill-frac", "0.5", "--distill-text", corpus(tmp_path), "--distill-positions", "64",
                       "--distill-temp", "2", "--out", str(out)])
    rec = json.loads(out.read_text())
    d = rec["distill"]
    assert d["frac"] == 0.5 and d["positions"] == 64 and d["temp"] == 2.0 and 1 <= d["steps"] <= 5
    assert d["loss_mean"] is not None and d["loss_mean"] >= 0 and len(d["loss_by_tenth"]) == d["steps"]
    assert len(d["step_kinds"]) == 6 and d["step_kinds"].count("d") == d["steps"]
    assert "admit" not in rec and pl.scores is evolve_run._scores
    assert rec["results"]["windowed"]["exact"] == 0.0
    kinds = [c.get("kind") for c in rec.get("curve", []) if isinstance(c, dict)]   # review R5: every entry labelled
    assert kinds and all(k in ("d", "n") for k in kinds)
    with pytest.raises(SystemExit, match="oracle"):                 # review R4: no onsets on text batches
        evolve_run.main(BASE + ["--distill-frac", "0.5", "--distill-text", corpus(tmp_path), "--w1s-gate", "oracle",
                                "--out", str(tmp_path / "o.json")])


def test_admission_runs_end_to_end_and_records(tmp_path, monkeypatch):
    keep_pl(monkeypatch)
    out = tmp_path / "a.json"
    evolve_run.main(BASE + ["--admit-target", "teacher", "--admit-weight", "0.5", "--out", str(out)])
    a = json.loads(out.read_text())["admit"]
    assert a["steps"] == 2 and a["weight"] == 0.5 and a["layers"] == [0, 1, 2, 3]
    assert a["kl_mean"] >= 0 and 0 < a["covered_mean"] <= 1 and a["eligible_mean"] > 1
    assert pl.scores is evolve_run._scores


@pytest.mark.parametrize("flags,match", [
    (["--distill-positions", "64"], "distill-frac"),
    (["--admit-weight", "2"], "admit-target"),
    (["--distill-frac", "1.5"], r"\[0, 1\]"),
    (["--distill-frac", "0.5", "--distill-positions", "500"], "beyond the window"),
    (["--distill-frac", "0.5", "--distill-text", "/nonexistent/x.txt"], "no such file"),
    (["--admit-target", "teacher", "--w1s-gate", "oracle"], "oracle"),
])
def test_refusals(tmp_path, monkeypatch, flags, match):
    keep_pl(monkeypatch)
    with pytest.raises(SystemExit, match=match):
        evolve_run.main(BASE + flags + ["--out", str(tmp_path / "x.json")])


def test_admission_needs_the_scattered_layout_and_w1s(tmp_path, monkeypatch):
    keep_pl(monkeypatch)
    with pytest.raises(SystemExit, match="needle-scatter"):
        evolve_run.main(["--tiny", "--admit-target", "teacher", "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit, match="W1s"):
        evolve_run.main(["--tiny", "--needle-scatter", "8", "--admit-target", "teacher",
                         "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit, match="window"):
        evolve_run.main(["--tiny", "--window", "0", "--distill-frac", "0.5", "--out", str(tmp_path / "x.json")])


def test_diag_teacher_attention_runs_on_tiny(tmp_path, monkeypatch):
    """The Stage-0 diagnostic on a tiny trained W1s run: masses sum to 1, AUCs and per-layer readings are
    in range, and the A15 target is compared with the run's gate."""
    keep_pl(monkeypatch)
    out = tmp_path / "r.json"
    evolve_run.main(BASE + ["--out", str(out)])
    res = diag.main([str(out), "--n", "4", "--batch", "2", "--device", "cpu", "--layers", "1-3"])
    saved = json.loads(out.with_suffix(".teacher_attn.json").read_text())
    assert saved["n"] == 4 and saved["layers_mean"] == [1, 2, 3] and saved["layers_all"] == [0, 1, 2, 3]
    m = res["mean"]
    assert sum(m[k] for k in ("queried", "other", "filler", "sink", "query")) == pytest.approx(1.0, abs=1e-4)
    assert all(0 <= v <= 1 for v in res["auc"].values()) and set(res["per_layer"]) == {"0", "1", "2", "3"}
    t = res["a15_target"]
    assert t["n"] == 4 and 0 <= t["target_on_queried"] <= 1 and 0 <= t["gate_on_queried"] <= 1 and t["covered"] > 0
    assert res["missing_keys"] == [] and res["checkpoint"].endswith(".pt")
    assert diag.auc(torch.tensor([1.0, 2.0]), torch.tensor([0.0])) == 1.0
    assert diag.auc(torch.tensor([0.5]), torch.tensor([0.5, 0.5])) == 0.5
