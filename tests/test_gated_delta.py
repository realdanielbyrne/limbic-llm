import math

import pytest
import torch
import torch.nn.functional as F

from hippoformer import (GatedDeltaLM, GatedDeltaMemory, GatedDeltaNet, TitansHippocampalLM,
                         gated_delta_chunked, gated_delta_scan, kwta, lm_loss)

torch.manual_seed(0)
B, H, T, Dk, Dv, V = 2, 3, 12, 8, 5, 40


def _rand(normalise=True):
    q = torch.randn(B, H, T, Dk)
    k = torch.randn(B, H, T, Dk)
    if normalise:
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    v = torch.randn(B, H, T, Dv)
    alpha = torch.rand(B, H, T) * 0.5 + 0.5
    beta = torch.rand(B, H, T)
    return q, k, v, alpha, beta


# ------------------------------------------------------------------- the rule
def test_closed_form_matches_gated_deltanet_equation():
    q, k, v, a, b = _rand()
    _, S, _ = gated_delta_scan(q, k, v, a, b)
    S_ref = torch.zeros(B, H, Dv, Dk)
    I = torch.eye(Dk)
    for t in range(T):
        kt, vt = k[:, :, t], v[:, :, t]
        S_ref = a[:, :, t, None, None] * S_ref @ (I - b[:, :, t, None, None] * kt[..., :, None] * kt[..., None, :]) \
            + b[:, :, t, None, None] * vt[..., :, None] * kt[..., None, :]
    assert torch.allclose(S, S_ref, atol=1e-5)


def test_equals_one_sgd_step_on_titans_loss_at_decayed_state():
    """S_t = αS − β ∇_S ½‖(αS)k − v‖²  — Gated DeltaNet is a linear Titans memory."""
    q, k, v, a, b = _rand()
    S_prev = torch.randn(B, H, Dv, Dk)
    _, S_gdn, _ = gated_delta_scan(q[:, :, :1], k[:, :, :1], v[:, :, :1], a[:, :, :1], b[:, :, :1], S_prev)
    M = (a[:, :, 0, None, None] * S_prev).requires_grad_(True)
    loss = 0.5 * (torch.einsum("bhvk,bhk->bhv", M, k[:, :, 0]) - v[:, :, 0]).pow(2).sum()
    (g,) = torch.autograd.grad(loss, M)
    S_sgd = M.detach() - b[:, :, 0, None, None] * g
    assert torch.allclose(S_gdn, S_sgd, atol=1e-5)


def test_one_hot_keys_alpha_beta_one_is_exact_ram():
    """Qwenputer regime: each write replaces exactly one slot, others untouched."""
    n = 6
    keys = torch.eye(Dk)[[0, 3, 5, 3, 0, 7]].view(1, 1, n, Dk)
    vals = torch.randn(1, 1, n, Dv)
    ones = torch.ones(1, 1, n)
    _, S, _ = gated_delta_scan(keys, keys, vals, ones, ones)
    read = lambda i: S[0, 0] @ torch.eye(Dk)[i]
    assert torch.allclose(read(0), vals[0, 0, 4], atol=1e-6)   # overwritten by the later write
    assert torch.allclose(read(3), vals[0, 0, 3], atol=1e-6)
    assert torch.allclose(read(5), vals[0, 0, 2], atol=1e-6)
    assert torch.allclose(read(7), vals[0, 0, 5], atol=1e-6)
    assert torch.allclose(read(1), torch.zeros(Dv))            # never written


def test_delta_rule_overwrites_where_linear_attention_accumulates():
    k = F.normalize(torch.randn(1, 1, 1, Dk), dim=-1).expand(1, 1, 2, Dk)
    v = torch.randn(1, 1, 2, Dv)
    o, _, _ = gated_delta_scan(k, k, v, torch.ones(1, 1, 2), torch.ones(1, 1, 2))
    assert torch.allclose(o[0, 0, 1], v[0, 0, 1], atol=1e-5)    # latest value, not v0 + v1


@pytest.mark.parametrize("T_,chunk", [(12, 4), (37, 16), (5, 16), (64, 32)])
def test_chunked_kernel_equals_exact_scan(T_, chunk):
    torch.manual_seed(1)
    q = F.normalize(torch.randn(B, H, T_, Dk), dim=-1)
    k = F.normalize(torch.randn(B, H, T_, Dk), dim=-1)
    v = torch.randn(B, H, T_, Dv)
    a, b = torch.rand(B, H, T_) * 0.5 + 0.5, torch.rand(B, H, T_)
    S0 = torch.randn(B, H, Dv, Dk)
    for x, y in zip(gated_delta_scan(q, k, v, a, b, S0), gated_delta_chunked(q, k, v, a, b, S0, chunk)):
        assert torch.allclose(x, y, atol=1e-4)


def test_chunked_kernel_stable_under_strong_decay():
    q = F.normalize(torch.randn(1, 1, 64, Dk), dim=-1)
    v = torch.randn(1, 1, 64, Dv)
    a = torch.full((1, 1, 64), 1e-3)                            # Γ underflows within a chunk
    o1, S1, _ = gated_delta_scan(q, q, v, a, torch.ones(1, 1, 64))
    o2, S2, _ = gated_delta_chunked(q, q, v, a, torch.ones(1, 1, 64), chunk=32)
    assert torch.isfinite(o2).all() and torch.allclose(o1, o2, atol=1e-4) and torch.allclose(S1, S2, atol=1e-4)


def test_kwta_sparsity():
    x = torch.randn(4, 3, 7, 100)
    y = kwta(x, 0.05)
    assert ((y != 0).sum(-1) <= math.ceil(0.05 * 100)).all() and (y >= 0).all()


# ---------------------------------------------------------------- mixer / LM
@pytest.mark.parametrize("addressing", ["gdn", "tied", "dg"])
def test_streaming_matches_full_sequence(addressing):
    torch.manual_seed(0)
    m = GatedDeltaLM(V, 32, 4, n_layers=2, addressing=addressing, key_expansion=4).eval()
    tok = torch.randint(0, V, (B, 20))
    with torch.no_grad():
        full = m(tok).logits
        a = m(tok[:, :9])
        b = m(tok[:, 9:], state=a.state, offset=9)
    assert torch.allclose(full, torch.cat([a.logits, b.logits], 1), atol=1e-5)


@pytest.mark.parametrize("addressing,attn_every", [("gdn", None), ("tied", None), ("dg", None), ("gdn", 2)])
def test_gated_delta_lm_causal_and_trains(addressing, attn_every):
    torch.manual_seed(0)
    m = GatedDeltaLM(V, 32, 4, n_layers=2, addressing=addressing, attn_every=attn_every, key_expansion=4)
    tok = torch.randint(0, V, (B, 24))
    out = m(tok)
    lm_loss(out.logits[:, :-1], tok[:, 1:]).backward()
    assert out.logits.shape == (B, 24, V) and out.surprise.shape == (B, 24)
    m.eval()
    with torch.no_grad():
        x = m(tok).logits
        t2 = tok.clone(); t2[:, 12] = (t2[:, 12] + 1) % V
        y = m(t2).logits
    assert torch.allclose(x[:, :12], y[:, :12], atol=1e-5)


def test_dg_addresses_are_sparse_and_unit_norm():
    mix = GatedDeltaNet(32, 4, addressing="dg", key_expansion=8, dg_sparsity=0.05)
    a = mix.address(torch.randn(B, 10, 32))
    nnz = (a != 0).sum(-1)
    assert nnz.max() <= math.ceil(0.05 * mix.dk)
    norms = a.norm(dim=-1)
    assert torch.allclose(norms[nnz > 0], torch.ones_like(norms[nnz > 0]), atol=1e-5)


def test_tied_successor_binding_recalls_value_in_one_layer():
    """Hand-built tied memory: presenting a key reads back the token that followed it."""
    torch.manual_seed(0)
    d = 16
    mix = GatedDeltaNet(d, 1, d_head=d, addressing="tied", output_gate=False)
    with torch.no_grad():
        mix.wa.weight.copy_(torch.eye(d)); mix.wv.weight.copy_(torch.eye(d))
        mix.wo.weight.copy_(torch.eye(d)); mix.norm.weight.fill_(1.0)
        mix.w_alpha.weight.zero_(); mix.w_alpha.bias.fill_(-30.0)   # α ≈ 1
        mix.w_beta.weight.zero_(); mix.w_beta.bias.fill_(30.0)      # β ≈ 1
    e = torch.eye(d)
    seq = torch.stack([e[0], e[1], e[2], e[3], e[4], e[5], e[0]]).unsqueeze(0)  # pairs (0→1),(2→3),(4→5); query 0
    out, _, _ = mix(seq)
    assert out[0, -1].argmax().item() == 1


# -------------------------------------------------------- Titans integration
@pytest.mark.parametrize("use_dg", [False, True])
def test_titans_with_gated_delta_memory(use_dg):
    torch.manual_seed(0)
    m = TitansHippocampalLM(V, 32, 4, n_layers=1, variant="mac", chunk=8, window=8,
                            memory_type="gated_delta", use_dg=use_dg, dg_expansion=4)
    tok = torch.randint(0, V, (B, 24))
    out = m(tok)
    lm_loss(out.logits[:, :-1], tok[:, 1:]).backward()
    m.eval()
    with torch.no_grad():
        a = m(tok).logits
        t2 = tok.clone(); t2[:, -1] = (t2[:, -1] + 1) % V
        b = m(t2).logits
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)
    m.train(); c = m(tok).logits
    assert torch.allclose(a, c, atol=1e-4)                      # same update in train and eval


def test_gated_delta_memory_carries_transition_across_chunks():
    torch.manual_seed(0)
    mem = GatedDeltaMemory(16, 16, sequential=True)
    x = torch.randn(1, 8, 16)
    s_full, _ = mem.write(x, mem.init_state(1))
    s_a, _ = mem.write(x[:, :3], mem.init_state(1))
    s_b, _ = mem.write(x[:, 3:], s_a)
    assert torch.allclose(s_full.weights[0], s_b.weights[0], atol=1e-5)
