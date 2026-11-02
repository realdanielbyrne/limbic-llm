import pytest
import torch

from limbic_llm import (
    CA1Comparator, CA2Modulator, CA3Autoassociative, Consolidator, DentateGyrus, EpisodicStore,
    HippocampalBlock, HippocampalLM, NeuralMemory, PrioritizedReplayBuffer, TitansHippocampalLM,
    BaselineLM, lm_loss, novelty_alignment_loss,
)

torch.manual_seed(0)
B, T, D, V = 2, 40, 32, 50


# ---------------------------------------------------------------- dentate gyrus
def test_dg_sparsity_is_exact():
    dg = DentateGyrus(D, expansion=8, sparsity=0.05)
    code = dg(torch.randn(B, T, D))
    active = (code != 0).sum(-1)
    assert code.shape == (B, T, D * 8)
    assert active.max() <= dg.k


def test_dg_pattern_separation_reduces_overlap():
    dg = DentateGyrus(D, expansion=16, sparsity=0.04)
    a = torch.randn(1, 1, D)
    b = a + 0.35 * torch.randn(1, 1, D)
    cin, cout = DentateGyrus.pattern_separation(a, b, dg(a), dg(b))
    assert cout < cin


def test_neurogenesis_grow_and_young_plasticity():
    dg = DentateGyrus(D, expansion=4, sparsity=0.1, reserve_fraction=0.25, young_boost=4.0)
    n0 = dg.n_alive
    born = dg.grow(8)
    assert born == 8 and dg.n_alive == n0 + 8
    young = (dg.age == 0).nonzero().squeeze(-1)
    plast = dg._plasticity()
    assert torch.allclose(plast[young], torch.full_like(plast[young], 5.0))
    assert (plast[~dg.alive] == 0).all()
    dg.tick(10_000)
    assert dg._plasticity()[young].max() < 1.1


def test_decorrelation_loss_is_scalar_and_differentiable():
    dg = DentateGyrus(D, expansion=2, sparsity=0.25)
    loss = DentateGyrus.decorrelation_loss(dg(torch.randn(4, 16, D)))
    loss.backward()
    assert loss.dim() == 0 and dg.expand.weight.grad is not None


# ------------------------------------------------------------------------- CA3
def test_ca3_store_write_and_retrieval_mass():
    ca3 = CA3Autoassociative(D, D, n_heads=4, capacity=16, n_write=4)
    code = torch.randn(B, T, D)
    out, h, mass = ca3(code)
    assert out.shape == (B, T, D) and mass is None
    store = ca3.write(None, h, torch.rand(B, T))
    assert store.valid.sum(1).tolist() == [4, 4]
    _, _, mass = ca3(code, store)
    assert mass.shape == (B, 16) and mass[store.valid].sum() > 0
    assert (mass[~store.valid] == 0).all()


def test_ca3_eviction_keeps_most_salient():
    ca3 = CA3Autoassociative(D, D, capacity=4, n_write=2, salience_decay=0.5, sequential=False)
    h = lambda: torch.randn(1, 4, D)
    store = ca3.write(None, h(), torch.tensor([[0.9, 0.8, 0.1, 0.2]]))
    assert store.valid.sum() == 2                        # empty slots fill first
    store = ca3.write(store, h(), torch.tensor([[0.3, 0.05, 0.7, 0.01]]))
    assert store.valid.all()                             # {.45, .40, .70, .30}
    store = ca3.write(store, h(), torch.tensor([[0.6, 0.0, 0.0, 0.0]]))
    # decayed {.225, .20, .35, .15}: the two weakest (.15, .20) are evicted
    got = sorted(store.salience.squeeze(0).tolist())
    assert got == pytest.approx([1e-4, 0.225, 0.35, 0.6], abs=1e-6)


def test_ca3_sequential_store_recalls_successor():
    """A stored cue h_t retrieves h_{t+1} (heteroassociative CA3)."""
    torch.manual_seed(0)
    ca3 = CA3Autoassociative(D, D, n_heads=4, capacity=8, n_write=7, init_beta=30.0)
    with torch.no_grad():
        ca3.wv.weight.copy_(torch.eye(D)); ca3.wo.weight.copy_(torch.eye(D)); ca3.wo.bias.zero_()
        ca3.wq.weight.copy_(torch.eye(D)); ca3.wk.weight.copy_(torch.eye(D))
    code = torch.randn(1, 8, D)
    _, h, _ = ca3(code)
    store = ca3.write(None, h, torch.ones(1, 8))
    occupied = store.successors[0, store.valid[0]]
    assert torch.allclose(occupied.sort(0).values, h[0, 1:].sort(0).values)
    with torch.no_grad():
        out, q, _ = ca3(code[:, 2:3], store)  # re-present cue 2 as a one-token segment
    retrieved = out - q                       # strip the residual cue
    sim = torch.nn.functional.cosine_similarity(retrieved[0, 0], h[0], dim=-1)
    # output = cue (autoassociative, current segment) ⊕ its stored successor
    sim[2] = -1
    assert sim.argmax().item() == 3 and sim[3] > 0.5   # recalls what followed the cue


def test_ca3_is_causal():
    ca3 = CA3Autoassociative(D, D)
    x = torch.randn(1, T, D)
    y1, _, _ = ca3(x)
    x2 = x.clone(); x2[:, -1] += 5
    y2, _, _ = ca3(x2)
    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-5)


# ------------------------------------------------------------------- CA1 / CA2
@pytest.mark.parametrize("mode", ["pointwise", "cross_attention"])
def test_ca1_outputs(mode):
    ca1 = CA1Comparator(D, mode=mode)
    fused, nov = ca1(torch.randn(B, T, D), torch.randn(B, T, D))
    assert fused.shape == (B, T, D) and nov.shape == (B, T)
    assert ((nov > 0) & (nov < 1)).all()


def test_ca2_initialises_to_identity():
    ca2 = CA2Modulator(D)
    x = torch.randn(B, T, D)
    assert torch.allclose(ca2(x), x)
    ca2c = CA2Modulator(D, d_context=8)
    assert torch.allclose(ca2c(x, torch.randn(B, 8)), x)


# ----------------------------------------------------------------------- block
def test_hippocampal_block_backward():
    blk = HippocampalBlock(D, n_heads=4, ca3_capacity=32, ca3_write=8)
    x = torch.randn(B, T, D, requires_grad=True)
    out = blk(x)
    out2 = blk(x, out.store)
    out2.hidden.sum().backward()
    assert x.grad is not None and out2.store.valid.sum() == B * 16


# --------------------------------------------------------------- neural memory
@pytest.mark.parametrize("use_dg", [False, True])
def test_neural_memory_memorises_at_test_time(use_dg):
    dg = DentateGyrus(D, 4, 0.1) if use_dg else None
    mem = NeuralMemory(D, D, depth=2, dg=dg).eval()
    x = torch.randn(B, 16, D)
    state = mem.init_state(B)
    with torch.no_grad():
        losses = []
        for _ in range(6):
            state, sur = mem.write(x, state)
            losses.append(sur.mean().item())
    assert losses[-1] < losses[0] * 0.8


def test_neural_memory_meta_gradients_reach_key_projection():
    mem = NeuralMemory(D, D, depth=2, meta_grad=True).train()
    x = torch.randn(B, 16, D)
    state, _ = mem.write(x, mem.init_state(B))
    mem.read(x, state).sum().backward()
    assert mem.wk.weight.grad is not None and mem.wk.weight.grad.abs().sum() > 0
    assert mem.init_weights[0].grad is not None


# ---------------------------------------------------------------------- models
def _causal_check(model, tokens, atol=1e-5):
    model.eval()
    with torch.no_grad():
        a = model(tokens).logits
        t2 = tokens.clone(); t2[:, -1] = (t2[:, -1] + 1) % V
        b = model(t2).logits
        t3 = tokens.clone(); t3[:, T // 2] = (t3[:, T // 2] + 1) % V
        c = model(t3).logits
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=atol)
    assert torch.allclose(a[:, : T // 2], c[:, : T // 2], atol=atol)


@pytest.mark.parametrize("variant", ["mac", "mag", "mal"])
@pytest.mark.parametrize("hippo", [True, False])
def test_titans_variants_forward_backward_causal(variant, hippo):
    m = TitansHippocampalLM(V, D, 4, n_layers=1, variant=variant, chunk=8, window=8,
                            use_dg=hippo, use_ca1=hippo, novelty_lr=hippo)
    tok = torch.randint(0, V, (B, T))
    out = m(tok)
    assert out.logits.shape == (B, T, V) and out.surprise.shape == (B, T)
    assert (out.novelty is not None) == hippo
    loss = lm_loss(out.logits[:, :-1], tok[:, 1:])
    loss.backward()
    _causal_check(m, tok)


@pytest.mark.parametrize("variant", ["mac", "mag", "mal"])
def test_titans_train_and_eval_apply_same_memory_update(variant):
    """Regression: meta-grad mode must not differentiate through earlier reads."""
    torch.manual_seed(0)
    m = TitansHippocampalLM(V, D, 4, n_layers=2, variant=variant, chunk=8, window=8)
    tok = torch.randint(0, V, (B, T))
    m.train(); a = m(tok).logits
    m.eval()
    with torch.no_grad():
        b = m(tok).logits
    assert torch.allclose(a, b, atol=1e-4)


def test_hippocampal_lm_forward_backward_causal():
    m = HippocampalLM(V, D, 4, n_ec=1, n_hpc=1, n_neocortex=1, segment=8,
                      ca3_capacity=16, ca3_write=4)
    tok = torch.randint(0, V, (B, T))
    out = m(tok)
    tl = lm_loss(out.logits[:, :-1], tok[:, 1:], reduction="none")
    loss = tl.mean() + novelty_alignment_loss(out.novelty[:, :-1], tl)
    loss.backward()
    assert len(out.state) == 1 and out.state[0].valid.any()
    _causal_check(m, tok)


def test_hippocampal_lm_state_carries_across_calls():
    m = HippocampalLM(V, D, 4, segment=8, ca3_capacity=16, ca3_write=4).eval()
    tok = torch.randint(0, V, (1, 16))
    with torch.no_grad():
        o1 = m(tok)
        cold = m(tok).logits
        warm = m(tok, state=o1.state).logits
    assert not torch.allclose(cold, warm)


def test_baseline_window():
    m = BaselineLM(V, D, 4, 1, window=4)
    _causal_check(m, torch.randint(0, V, (B, T)))


# ---------------------------------------------------------------------- replay
def test_replay_buffer_priority_eviction():
    buf = PrioritizedReplayBuffer(capacity=3, seed=0)
    buf.add(torch.zeros(3, 5, dtype=torch.long), torch.tensor([0.1, 0.5, 0.9]))
    buf.add(torch.ones(1, 5, dtype=torch.long), torch.tensor([0.7]))
    assert len(buf) == 3 and min(buf.priority.tolist()) > 0.4
    toks, lg, idx = buf.sample(8)
    assert toks.shape == (8, 5) and lg is None


def test_consolidator_updates_only_slow_params():
    m = HippocampalLM(V, D, 4, segment=8, ca3_capacity=16, ca3_write=4)
    buf = PrioritizedReplayBuffer(16, seed=0)
    tok = torch.randint(0, V, (4, 17))
    with torch.no_grad():
        logits = m(tok[:, :-1]).logits
    buf.add(tok, torch.rand(4), logits)
    slow = list(m.neocortex.parameters())
    fast_before = [p.clone() for p in m.ec.parameters()]
    slow_before = [p.clone() for p in slow]
    cons = Consolidator(m, buf, slow, lr=1e-2)
    cons.sleep(steps=3, batch_size=4)
    assert all(torch.equal(a, b) for a, b in zip(fast_before, m.ec.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(slow_before, slow))
