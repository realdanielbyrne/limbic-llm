"""Multi-tap hooks for round 10 (experiments/evolve/taps.py): TapLM caches, reinstatement, E1, gradient.

Stub sides implement the W1s interface (``taps``, ``extra``, ``out_mode``, ``reinstate_L``,
``reinstate_bias``), so these tests do not depend on w1s_slots.py. Models are tiny and random:
GPT-2 from ``pl.build_tiny`` and a plain Qwen3 built here (4 attention layers, no downloads).
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import pretrained_loop as pl  # noqa: E402
from evolve.taps import NEG, TapLM  # noqa: E402

ARCHS = ["gpt2", "qwen3"]
W, T, LAYER = 8, 49, 2                                            # 1 sink + 48 real tokens, 6 blocks of W
Q0 = 41                                                           # query rows: the last block, side coordinates 40..47


def build(arch, attn="sdpa", seed=0):
    torch.manual_seed(seed)
    if arch == "gpt2":
        model, _ = pl.build_tiny("gpt2", seed=seed, attn_impl=attn)
    else:
        cfg = transformers.Qwen3Config(
            vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, max_position_embeddings=1024,
            bos_token_id=256, eos_token_id=256, pad_token_id=256)
        model = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation=attn)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def tokens(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 256, (B, T), generator=g)
    ids[:, 0] = 256                                               # BOS sink
    return ids


class TapSide(nn.Module):
    """Additive stub: injects a causal function of the tap states and records what it was given."""

    def __init__(self, d, taps=(0, 1)):
        super().__init__()
        self.taps, self.extra, self.seen = tuple(taps), None, []
        self.proj = nn.Linear(d, d)

    def forward(self, h):
        self.seen.append({i: x.clone() for i, x in (self.extra or {}).items()})
        e = sum(self.extra.get(i, h) for i in self.taps) if self.extra else h
        run = e.cumsum(1) / torch.arange(1, e.shape[1] + 1, device=e.device)[:, None]   # causal running mean
        return h + 0.5 * self.proj(run), None


class ReinstateSide(nn.Module):
    """Reinstatement stub: rows ≥ q0 see columns [c0, c0 + n) at log weight ``logw``; h passes unchanged."""

    def __init__(self, c0=10, n=4, q0=Q0 - 1, L=1, taps=(0,), logw=0.0, empty=False):
        super().__init__()
        self.taps, self.extra, self.out_mode, self.reinstate_L = tuple(taps), None, "reinstate", L
        self.c0, self.n, self.q0, self.empty = c0, n, q0, empty
        self.logw = nn.Parameter(torch.tensor(float(logw)))

    def forward(self, h):
        return h, None

    def reinstate_bias(self, extra):
        x = extra[self.taps[0]]
        B, L = x.shape[:2]
        rb = torch.full((B, L, L), NEG, device=x.device)
        if self.empty:
            return rb
        rows = torch.arange(L, device=x.device)[:, None] >= self.q0
        cols = (torch.arange(L, device=x.device) >= self.c0) & (torch.arange(L, device=x.device) < self.c0 + self.n)
        return torch.where(rows & cols, self.logw, rb)


class ContentSide(ReinstateSide):
    """Bias from the tap content, deliberately finite on future columns too (TapLM must cut them)."""

    def reinstate_bias(self, extra):
        x = extra[self.taps[0]]
        x = x / x.norm(dim=-1, keepdim=True)
        return 4.0 * torch.einsum("btd,bcd->btc", x, x) + self.logw   # every (t, c), future included


def pair(arch, side_fn, mode="block", attn="sdpa", **kw):
    """HookedLM and TapLM on identical copies of the model, with the same side."""
    side = side_fn()
    base = pl.HookedLM(build(arch, attn), side, LAYER, W, mode, "all", 1, pad_id=256)
    tap = TapLM(build(arch, attn), side, LAYER, W, mode, "all", 1, pad_id=256, **kw)
    return base, tap


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("mode", ["sliding", "block"])
def test_side_off_is_byte_identical_to_hookedlm(arch, mode):
    for side_fn in (lambda: TapSide(64), lambda: ReinstateSide(logw=3.0)):
        base, tap = pair(arch, side_fn, mode)
        ids = tokens()
        with torch.no_grad():
            for window in (W, 0):
                with base.condition(window, False), tap.condition(window, False):
                    assert torch.equal(base(ids), tap(ids))
                    assert tap.side.extra is None and not tap._cache


@pytest.mark.parametrize("arch", ARCHS)
def test_side_without_taps_matches_hookedlm_with_side_on(arch):
    torch.manual_seed(1)
    side = pl.SideChannels(64, tau=8)
    base = pl.HookedLM(build(arch), side, LAYER, W, "block", "all", 1, pad_id=256)
    tap = TapLM(build(arch), side, LAYER, W, "block", "all", 1, pad_id=256)
    assert tap.taps == () and len(tap.handles) == len(base.handles)
    ids = tokens()
    with torch.no_grad():
        assert torch.equal(base(ids), tap(ids))


@pytest.mark.parametrize("arch", ARCHS)
def test_caches_are_tap_outputs_without_the_sink_and_are_cleared(arch):
    _, lm = pair(arch, lambda: TapSide(64, taps=(1, 0)))
    ids = tokens()
    with torch.no_grad(), lm.condition(W, True):
        lm(ids)
    assert lm.taps == (0, 1) and len(lm.side.seen) == 1
    seen = lm.side.seen[0]
    with torch.no_grad(), lm.condition(W, False):                 # same masks, no side: blocks < LAYER agree
        hs = lm.model(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states
    for i in (0, 1):
        assert seen[i].dtype == torch.float32 and seen[i].shape == (2, T - 1, 64)
        assert torch.equal(seen[i], hs[i + 1][:, 1:].float())     # hidden_states[i + 1] is block i's output
    assert lm.side.extra is None and lm._cache == {} and lm._rbias is None

    looked = []                                                   # side off: the cache hooks stay idle
    hk = lm.blocks[LAYER].register_forward_hook(lambda m, a, o: looked.append(dict(lm._cache)))
    with torch.no_grad(), lm.condition(W, False):
        lm(ids)
    hk.remove()
    assert looked == [{}] and len(lm.side.seen) == 1


def test_bad_configurations_raise():
    with pytest.raises(ValueError, match="segment"):
        TapLM(build("gpt2"), TapSide(64), LAYER, W, "segment", "all", 1, pad_id=256)
    with pytest.raises(ValueError, match="below the main tap"):
        TapLM(build("gpt2"), TapSide(64, taps=(2,)), LAYER, W, "block", "all", 1, pad_id=256)
    with pytest.raises(ValueError, match="reinstatement"):
        TapLM(build("gpt2"), ReinstateSide(L=1, taps=(1,)), 3, W, "block", "all", 1, pad_id=256)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_empty_reinstatement_changes_nothing(arch, attn):
    _, lm = pair(arch, lambda: ReinstateSide(empty=True), attn=attn)
    ids = tokens()
    with torch.no_grad():
        with lm.condition(W, False):
            off = lm(ids)
        with lm.condition(W, True):
            on = lm(ids)
    assert torch.equal(on, off)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_reinstated_window_reaches_the_query_and_nothing_else_does(arch, attn):
    """Columns 10..13 (side coordinates; tokens 11..14, in block 9..16) are reinstated for the last block's
    rows. Changing token 12 moves the query logits; changing token 16 (same block, after the window)
    or token 30 (another block) does not, and with the side off nothing distant reaches the query."""
    _, lm = pair(arch, lambda: ReinstateSide(c0=10, n=4, L=1), attn=attn)
    ids = tokens()

    def query_logits(x, side):
        with torch.no_grad(), lm.condition(W, side):
            return lm(x)[:, Q0:]

    def bump(p):
        x = ids.clone()
        x[:, p] = (x[:, p] + 1) % 256
        return x

    ref_off, ref_on = query_logits(ids, False), query_logits(ids, True)
    assert (ref_on - ref_off).abs().max() > 1e-4                  # reinstatement is live
    assert (query_logits(bump(12), True) - ref_on).abs().max() > 1e-4
    for p in (16, 30):
        assert torch.equal(query_logits(bump(p), True), ref_on), p
    for p in (12, 16, 30):
        assert torch.equal(query_logits(bump(p), False), ref_off), p


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("side_fn", [lambda: TapSide(64), lambda: ContentSide(L=1, logw=1.0)],
                         ids=["additive", "reinstate"])
def test_lm_is_strictly_causal(arch, side_fn):
    """E1 at the LM level: changing tokens from t0 on leaves the logits before t0 bitwise unchanged,
    including the content-dependent reinstatement bias, whose future columns TapLM must cut."""
    _, lm = pair(arch, side_fn, mode="block")
    ids = tokens()
    with torch.no_grad(), lm.condition(W, True):
        ref = lm(ids)
        for t0 in (1, 7, 9, 30, 48):
            x = ids.clone()
            x[:, t0:] = torch.randint(0, 256, x[:, t0:].shape, generator=torch.Generator().manual_seed(t0))
            assert torch.equal(lm(x)[:, :t0], ref[:, :t0]), t0


def _bias_grad(lm):
    lm.side.logw.grad = None
    with lm.condition(W, True):
        lm(tokens())[:, Q0:].float().pow(2).mean().backward()
    return lm.side.logw.grad


@pytest.mark.parametrize("arch", ARCHS)
def test_bias_receives_gradient_under_eager_attention(arch):
    _, lm = pair(arch, lambda: ReinstateSide(L=1, logw=-1.0), attn="eager")
    g = _bias_grad(lm)
    assert g is not None and g.abs() > 0
    _, ck = pair(arch, lambda: ReinstateSide(L=1, logw=-1.0), attn="eager", grad_checkpointing=True)
    assert torch.allclose(_bias_grad(ck), g, rtol=1e-5, atol=1e-8)    # the recompute sees the same bias


@pytest.mark.parametrize("arch", ARCHS)
def test_bias_gradient_under_sdpa(arch):
    """Recorded, not required: whether this torch's sdpa differentiates attn_mask (skips if not)."""
    _, lm = pair(arch, lambda: ReinstateSide(L=1, logw=-1.0), attn="sdpa")
    try:
        g = _bias_grad(lm)
    except RuntimeError as e:
        pytest.skip(f"sdpa: no gradient to attn_mask ({e})")
    if g is None or g.abs() == 0:
        pytest.skip("sdpa: no gradient to attn_mask")
    _, ref = pair(arch, lambda: ReinstateSide(L=1, logw=-1.0), attn="eager")
    assert torch.allclose(g, _bias_grad(ref), rtol=1e-3)


# ============================================================ attention taps (round 11, A13 native keys)
class AttnSide(nn.Module):
    """Additive stub with ``attn_taps``: injects a causal function of the captured K and Q (a running mean of
    Q⊙K mapped to the width), and records what it was given. ``taps`` lists hidden-state taps (default none)."""

    def __init__(self, d, attn_taps=(LAYER,), taps=(), width=32):
        super().__init__()
        self.attn_taps, self.taps, self.extra, self.seen = tuple(attn_taps), tuple(taps), None, []
        self.proj = nn.Linear(width, d)

    def forward(self, h):
        self.seen.append({k: x.clone() for k, x in (self.extra or {}).items()})
        if not self.extra:
            return h, None
        L = self.attn_taps[0]
        e = self.extra[("q", L)] * self.extra[("k", L)]
        run = e.cumsum(1) / torch.arange(1, e.shape[1] + 1, device=e.device)[:, None]
        return h + 0.5 * self.proj(run), None


def _attn_reference(lm, ids, layers):
    """What each tapped attention module computes from its own input, by hand: (pre-rotary K, pre-rotary Q
    with each query group averaged, post-rotary K), sink included, from a pre-hook on the attention module."""
    got, hs = {}, []
    for L in layers:
        attn = pl._child(lm.blocks[L], pl.ATTN_MIXERS)

        def pre(mod, args, kwargs, L=L):
            x = args[0] if args else kwargs["hidden_states"]
            B, T_ = x.shape[:2]
            d = mod.head_dim
            k = mod.k_norm(mod.k_proj(x).view(B, T_, -1, d))      # (B, T, H_kv, d)
            q = mod.q_norm(mod.q_proj(x).view(B, T_, -1, d))      # (B, T, H_q, d)
            cos, sin = kwargs["position_embeddings"]
            _, k_rot = transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb(
                q.transpose(1, 2), k.transpose(1, 2), cos, sin)
            n_kv = k.shape[2]
            qg = q.float().reshape(B, T_, n_kv, q.shape[2] // n_kv, d).mean(3)
            got[L] = (k.float().flatten(2), qg.flatten(2), k_rot.transpose(1, 2).float().flatten(2))
        hs.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        with torch.no_grad():
            lm(ids)
    finally:
        for hk in hs:
            hk.remove()
    return got


def test_attn_taps_capture_pre_rotary_k_and_grouped_q_without_the_sink():
    """The captured K and Q of layers 0 and 2 (= the main tap) are what the attention module computes after
    k_proj/q_proj and k_norm/q_norm, before rotary: K equals it bitwise, Q is the mean of each group of
    H_q / H_kv = 2 query heads (width 2 KV heads × 16), the sink is stripped, and the post-rotary K differs."""
    side = AttnSide(64, attn_taps=(2, 0))
    lm = TapLM(build("qwen3"), side, LAYER, W, "block", "all", 1, pad_id=256)
    assert lm.attn_taps == (0, 2)
    ids = tokens()
    with lm.condition(W, True):
        ref = _attn_reference(lm, ids, (0, 2))
    assert len(side.seen) == 1
    seen = side.seen[0]
    assert set(seen) == {("k", 0), ("q", 0), ("k", 2), ("q", 2)}
    for L in (0, 2):
        k, q, k_rot = ref[L]
        assert seen[("k", L)].dtype == torch.float32 and seen[("k", L)].shape == (2, T - 1, 32)
        assert seen[("q", L)].shape == (2, T - 1, 32)
        assert torch.equal(seen[("k", L)], k[:, 1:])
        torch.testing.assert_close(seen[("q", L)], q[:, 1:], rtol=0, atol=1e-6)
        assert (seen[("k", L)] - k_rot[:, 1:]).abs().max() > 1e-3   # rotary would have changed it
    assert side.extra is None and lm._cache == {}


def test_attn_taps_carry_no_absolute_position():
    """At layer 0 (input: the token embeddings; Qwen3 has no position embedding) the same token at two
    positions has bitwise the same captured K and Q, which a post-rotary capture would not."""
    side = AttnSide(64, attn_taps=(0,))
    lm = TapLM(build("qwen3"), side, LAYER, W, "block", "all", 1, pad_id=256)
    ids = tokens(B=1)
    ids[0, 6] = ids[0, 33] = 77
    with torch.no_grad(), lm.condition(W, True):
        lm(ids)
    k, q = side.seen[0][("k", 0)], side.seen[0][("q", 0)]
    assert torch.equal(k[0, 5], k[0, 32]) and torch.equal(q[0, 5], q[0, 32])   # side coordinates: token − 1
    assert not torch.equal(k[0, 5], k[0, 6])


@pytest.mark.parametrize("mode", ["sliding", "block"])
def test_attn_taps_side_off_is_byte_identical_and_captures_nothing(mode):
    """The windowed control (side off) and the full condition match HookedLM bitwise and nothing is cached;
    with the side on, a side that ignores extra gives HookedLM's output too (the hooks are read-only)."""
    base, tap = pair("qwen3", lambda: AttnSide(64, attn_taps=(1, 2)), mode)
    ids = tokens()
    looked = []
    hk = tap.blocks[LAYER].register_forward_hook(lambda m, a, o: looked.append(dict(tap._cache)))
    with torch.no_grad():
        for window in (W, 0):
            with base.condition(window, False), tap.condition(window, False):
                assert torch.equal(base(ids), tap(ids))
                assert tap.side.extra is None and not tap._cache
    hk.remove()
    assert looked == [{}, {}] and tap.side.seen == []

    class Ignore(AttnSide):
        def forward(self, h):
            self.seen.append(set(self.extra or {}))
            return h, None
    base, tap = pair("qwen3", lambda: Ignore(64, attn_taps=(0, 2), taps=(1,)), mode)
    with torch.no_grad():
        assert torch.equal(base(ids), tap(ids))
    assert tap.side.seen[-1] == {1, ("k", 0), ("q", 0), ("k", 2), ("q", 2)}


def test_attn_taps_lm_is_strictly_causal():
    """E1 at the LM level with the captured K/Q feeding the side: future tokens leave earlier logits bitwise."""
    _, lm = pair("qwen3", lambda: AttnSide(64, attn_taps=(0, 2)), mode="block")
    ids = tokens()
    with torch.no_grad(), lm.condition(W, True):
        ref = lm(ids)
        for t0 in (1, 9, 30, 48):
            x = ids.clone()
            x[:, t0:] = torch.randint(0, 256, x[:, t0:].shape, generator=torch.Generator().manual_seed(t0))
            assert torch.equal(lm(x)[:, :t0], ref[:, :t0]), t0


def test_attn_tap_bad_configurations_raise():
    with pytest.raises(ValueError, match="main tap"):                 # layer 3 runs after the side at layer 2
        TapLM(build("qwen3"), AttnSide(64, attn_taps=(3,)), LAYER, W, "block", "all", 1, pad_id=256)
    with pytest.raises(ValueError, match="q_norm"):                   # GPT-2 has no separate q/k states
        TapLM(build("gpt2"), AttnSide(64, attn_taps=(1,)), LAYER, W, "block", "all", 1, pad_id=256)
    with pytest.raises(ValueError, match="segment"):
        TapLM(build("qwen3"), AttnSide(64), LAYER, W, "segment", "all", 1, pad_id=256)
    rs = ReinstateSide()
    rs.attn_taps = (1,)
    with pytest.raises(ValueError, match="reinstatement"):
        TapLM(build("qwen3"), rs, LAYER, W, "block", "all", 1, pad_id=256)
