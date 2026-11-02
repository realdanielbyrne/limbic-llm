"""Round-1 candidate W1 (experiments/evolve/w1_gdn.py): the anti-gaming rules E1-E3 and the runner."""

import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from evolve import run as evolve_run  # noqa: E402
from evolve.w1_gdn import GDNLoop, W1Side, split_args  # noqa: E402

D = 64


def side(**kw):
    torch.manual_seed(0)
    ns = split_args([f"--w1-{k.replace('_', '-')}={v}" for k, v in kw.items()])[0]
    return W1Side(D, type("A", (), {"time": True})(), ns).eval()


@pytest.mark.parametrize("addressing,ctx", [("tied", 1), ("dg", 1), ("tied", 4), ("dg", 4), ("gdn", 1)])
@pytest.mark.parametrize("ctx_init", ["identity", "uniform"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_strictly_causal(addressing, ctx, ctx_init, dtype):
    """E1: perturbing h[:, t0:] leaves the output before t0 bitwise unchanged, also for bf16 input."""
    m = side(addressing=addressing, heads=2, dhead=16, key_expansion=2, chunk=8, ctx=ctx, ctx_init=ctx_init)
    with torch.no_grad():
        m.loop.gain_logit.fill_(2.0)
        m.time.out.weight.normal_(0, 0.1)                         # the clock starts as identity
        if m.loop.ctx is not None:
            m.loop.ctx.conv.weight.normal_(0, 0.5)                # context taps active
    h = torch.randn(2, 70, D).to(dtype).float()
    y, _ = m(h)
    for t0 in (1, 7, 8, 9, 33, 69):
        h2 = h.clone()
        h2[:, t0:] += torch.randn_like(h2[:, t0:])
        y2, _ = m(h2)
        assert torch.equal(y[:, :t0], y2[:, :t0]), t0


def test_state_size_independent_of_length():
    """E2: the memory's state has the same size for any sequence length."""
    loop = GDNLoop(D, heads=2, d_head=16, chunk=8)
    sizes = {loop.mem(torch.randn(1, T, D))[1].S.numel() for T in (16, 64, 257)}
    assert sizes == {2 * 16 * 16}


def _successor_read(addressing: str, key_expansion: int = 1) -> torch.Tensor:
    """Cosine of the read at a repeat of x4 with each value stored in the first block x0..x11."""
    torch.manual_seed(0)
    loop = GDNLoop(D, addressing=addressing, heads=1, d_head=D, key_expansion=key_expansion,
                   sparsity=0.05, decay_bias=-20.0, beta_bias=20.0, chunk=8)          # α ≈ 1, β ≈ 1
    x = torch.randn(1, 12, D)
    seq = torch.cat([x, torch.randn(1, 20, D), x[:, :6]], 1)    # x0..x11, filler, x0..x5 again
    mem = loop.mem
    mem.norm, mem.wg = torch.nn.Identity(), None                  # expose the raw read S_t a_t
    with torch.no_grad():
        mem.wo.weight.copy_(torch.eye(D))
    o, _, _ = mem(loop.norm(seq))
    v = mem.wv(loop.norm(seq))                                    # (1, T, D) stored values
    return torch.stack([torch.cosine_similarity(o[0, 36], v[0, s], dim=0) for s in range(12)]).detach()


def test_successor_binding_retrieves_the_next_value():
    """Writing v under the previous address makes a repeat of a context read out what followed it."""
    dense = _successor_read("tied")
    top2 = dense.topk(2).values
    assert dense.argmax().item() == 5 and top2[0] - top2[1] > 0.1   # x5, not x4, despite crosstalk
    sparse = _successor_read("dg", key_expansion=8)               # DG codes: nearly orthogonal keys
    assert sparse.argmax().item() == 5 and sparse[5] > dense[5]


@pytest.mark.parametrize("argv", [[], ["--w1-addressing=dg", "--w1-heads=4", "--w1-key-expansion=4"],
                                  ["--w1-ctx=4"], ["--w1-addressing=gdn", "--w1-heads=6"]])
def test_parameter_budget_at_qwen_width(argv):
    """E3: at Qwen3-1.7B's width every round-1/2 configuration stays within the 4.4M budget."""
    m = W1Side(2048, type("A", (), {"time": True})(), split_args(argv)[0])
    assert sum(p.numel() for p in m.parameters()) <= 4_400_000


def test_context_address_starts_as_w1():
    """W1b: the context conv starts at the current position, so it reproduces W1 before training."""
    torch.manual_seed(0)
    a = GDNLoop(D, heads=2, d_head=16, chunk=8)
    b = GDNLoop(D, heads=2, d_head=16, chunk=8, ctx=4)
    b.load_state_dict({**a.state_dict(), **{k: v for k, v in b.state_dict().items() if k.startswith("ctx")}})
    x = torch.randn(2, 40, D)
    assert torch.allclose(a(x)[0], b(x)[0], atol=1e-5)


def test_runner_end_to_end(tmp_path):
    """The runner trains W1 on a tiny model and records per-position metrics and the candidate."""
    out = tmp_path / "w1.json"
    evolve_run.main(["--tiny", "--task", "needle", "--window", "16", "--seq-len", "64", "--steps", "3",
                     "--batch", "2", "--eval-n", "4", "--candidate", "evolve.w1_gdn:W1Side",
                     "--w1-heads", "2", "--w1-dhead", "16", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert rec["candidate"]["args"]["w1_heads"] == 2
    assert len(rec["results"]["windowed+side"]["per_pos"]) >= 2
    assert out.with_suffix(".pt").exists()


@pytest.mark.parametrize("conv_init", ["random", "shift"])
def test_gdn_conv_init_options(conv_init):
    """B12 control: gdn mode can start without identity convs; shift puts the key on t − 1."""
    m = side(addressing="gdn", heads=2, dhead=16, chunk=8, conv_init=conv_init)
    w = m.loop.mem.conv_k.conv.weight
    if conv_init == "shift":
        assert torch.all(w[:, 0, -2] == 1) and torch.all(w[:, 0, -1] == 0)
    else:
        assert not torch.all(w[:, 0, -1] == 1)
    h = torch.randn(1, 30, D)
    y, _ = m(h)
    h2 = h.clone()
    h2[:, 12:] += 1.0
    assert torch.equal(y[:, :12], m(h2)[0][:, :12])


def test_scattered_needle_breaks_the_fixed_layout():
    """--needle-scatter: right length, answer at the end, the queried fact beyond the tail, and the
    first fact's position varies (so position cannot identify the queried fact)."""
    import random
    _, tok = evolve_run.pl.build_tiny("gpt2", seed=0)
    task = evolve_run.ScatterNeedleTask(tok, 512, 3, 1, min_gap=8, tail=256)
    rng = random.Random(0)
    firsts = set()
    for _ in range(20):
        ids, n_ans, gap = task.sample(rng)
        assert len(ids) == 513 and n_ans == len(tok.encode(" 0000"))
        assert gap >= 256 + n_ans
        text = bytes(i for i in ids[1:] if i < 256).decode("latin-1")
        assert text.count(" The code for ") == 4                  # 3 facts and the query
        firsts.add(text.index(" The code for "))
    assert len(firsts) > 10


def test_scattered_facts_stay_inside_one_window_block():
    """With block = W, no fact spans two attention-window blocks (as in the fixed layout)."""
    import random
    _, tok = evolve_run.pl.build_tiny("gpt2", seed=0)
    task = evolve_run.ScatterNeedleTask(tok, 512, 3, 1, min_gap=8, tail=256, block=64)
    rng = random.Random(1)
    head = len(tok.encode(" The code for "))
    for _ in range(50):
        ids, _, _ = task.sample(rng)
        body = bytes(i for i in ids[1:] if i < 256).decode("latin-1")
        starts = [i for i in range(len(body)) if body.startswith(" The code for ", i)][:3]
        for st in starts:
            end = body.index(".", st)
            assert st // 64 == end // 64, (st, end)
    assert task.last_queried in (0, 1, 2)
