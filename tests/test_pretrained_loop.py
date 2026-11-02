"""Hooks of experiments/pretrained_loop.py on tiny random Hugging Face models (offline)."""

import importlib.util
import random
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
_spec = importlib.util.spec_from_file_location(
    "pretrained_loop", Path(__file__).resolve().parents[1] / "experiments" / "pretrained_loop.py")
pl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pl)

ARCHS = ["gpt2"] + (["qwen35"] if hasattr(transformers, "Qwen3_5TextConfig") else [])
W, TAU, T = 8, 8, 49                                              # 1 sink + 48 real tokens


def hooked(arch="gpt2", window=W, mode="sliding", window_layers="all", **side_kw):
    model, tok = pl.build_tiny(arch, seed=0)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(1)
    side = pl.SideChannels(model.config.hidden_size, tau=TAU, **side_kw)
    return pl.HookedLM(model, side, window=window, mode=mode, window_layers=window_layers, sink=1,
                       pad_id=tok.eos_token_id), tok


def tokens(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 256, (B, T), generator=g)
    ids[:, 0] = 256                                               # BOS sink
    return ids


@pytest.mark.parametrize("arch", ARCHS)
def test_side_channels_off_match_base_model(arch):
    lm, _ = hooked(arch)
    ref, _ = pl.build_tiny(arch, seed=0)                          # same weights, no hooks
    ids = tokens()
    with torch.no_grad():
        base = ref.eval()(input_ids=ids, use_cache=False).logits
        with lm.condition(window=0, side=False):
            off = lm(ids)
        with lm.condition(window=0, side=True):
            on = lm(ids)
    assert torch.allclose(off, base, atol=1e-6)
    assert not torch.allclose(on, base, atol=1e-4)                # the hook is live when switched on


@pytest.mark.parametrize("arch", ARCHS)
def test_only_side_channel_parameters_train(arch):
    lm, _ = hooked(arch)
    side = {id(p) for p in lm.side.parameters()}
    trainable = {id(p) for p in list(lm.model.parameters()) + list(lm.side.parameters()) if p.requires_grad}
    assert trainable == side and len(side) > 0
    lm(tokens()).float().pow(2).mean().backward()
    assert all(p.grad is None for p in lm.model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in lm.side.parameters())


@pytest.mark.parametrize("arch,mode", [("gpt2", "sliding"), ("gpt2", "block"), ("gpt2", "segment")]
                         + [("qwen35", m) for m in ("sliding", "block") if "qwen35" in ARCHS])
def test_window_cuts_distant_tokens_unless_memory_carries_them(arch, mode):
    """Changing the first real token must not move logits beyond the windowed reach with the loop
    off (sliding: L·(W−1) through stacked windows; block/segment: the first block), but does with it on."""
    lm, _ = hooked(arch, mode=mode)
    check = pl.check_window(lm)                                   # raises if the mask is ignored
    assert check == "skipped" if mode == "segment" else check.startswith("ok")
    far = 1 + int(lm.reach()) + 1 if mode == "sliding" else 1 + W
    assert far < T - 4
    ids = tokens()
    ids2 = ids.clone()
    ids2[:, 1] = (ids2[:, 1] + 1) % 256
    with torch.no_grad():
        with lm.condition(W, side=False):
            a, b = lm(ids), lm(ids2)
        with lm.condition(W, side=True):
            c, d = lm(ids), lm(ids2)
    assert (a[:, 1:far] - b[:, 1:far]).abs().max() > 0            # nearby positions do see it
    assert torch.allclose(a[:, far:], b[:, far:], atol=1e-6)      # distant ones do not
    assert (c[:, far:] - d[:, far:]).abs().max() > 1e-6           # memory is a route past the window


def test_window_mask_and_fold():
    m = pl.window_mask(6, 2, "sliding", 1, "cpu", torch.float32)[0, 0] == 0
    assert m[5].tolist() == [True, False, False, False, True, True]
    m = pl.window_mask(6, 2, "block", 1, "cpu", torch.float32)[0, 0] == 0
    assert m[4].tolist() == [True, False, False, True, True, False]   # blocks {1,2} {3,4} {5}
    x = torch.randn(2, 11, 3)
    y, shape = pl.fold(x, 4, 1)
    assert y.shape == (2 * 4, 4, 3)
    assert torch.equal(pl.unfold(y, 4, 1, shape), x)


def test_needle_task_shapes_and_gap():
    tok = pl.ByteTokenizer()
    task = pl.NeedleTask(tok, seq_len=96, pairs=2, sink=1)
    ids, mask, gaps = task.batch(3, random.Random(0))
    assert ids.shape == (3, 97) and (ids[:, 0] == tok.bos_token_id).all()
    assert (mask.sum(1) == 5).all() and mask[:, -1].all()          # " 1234" in bytes
    assert all(g > W for g in gaps)


def test_side_channels_train_in_float32_on_a_bf16_model():
    """With bf16 side weights, Adam steps of ~lr round away near 1 (bf16 spacing 2^-7)."""
    lm, _ = hooked("gpt2")
    lm.model.to(torch.bfloat16)
    ids = tokens()
    out = lm(ids)
    assert out.dtype == torch.bfloat16                            # the base model still runs in bf16
    assert all(p.dtype == torch.float32 for p in lm.side.parameters())
    opt = torch.optim.AdamW(lm.side.parameters(), lr=1e-3)
    g0 = lm.side.loop.gain_logit.item()
    out.float().pow(2).mean().backward()
    opt.step()
    assert lm.side.loop.gain_logit.item() != g0
