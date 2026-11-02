import math

import pytest
import torch
import torch.nn.functional as F

from limbic_llm.temporal_loop import HippocampalLoop, TemporalLoopLM, TimeChannel, decay_kernel, loop_kernel

torch.manual_seed(0)


# ------------------------------------------------------------------ kernels
def test_kernel_is_zero_within_one_circuit():
    for kind in ["osc", "exp", "flat"]:
        W = loop_kernel(40, 8, kind)
        i = torch.arange(40)
        d = i[:, None] - i[None, :]
        assert (W[d < 8] == 0).all()


def test_exp_kernel_attenuates_by_rho_each_circuit():
    W = loop_kernel(64, 8, "exp", rho=0.6)
    assert W[8, 0] == pytest.approx(1.0) and W[16, 0] == pytest.approx(0.6) and W[24, 0] == pytest.approx(0.36)


def test_osc_kernel_is_clamped_damped_sinusoid():
    tau, rho = 8, 0.5
    W = loop_kernel(64, tau, "osc", rho=rho, amp=3.0)              # amp is clamped to 1
    assert W.max() <= 1.0 + 1e-6 and W.min() >= 0
    assert W[8, 0] == pytest.approx(1.0) and W[16, 0] == pytest.approx(0.5)   # peaks once per circuit
    assert W[12, 0] == pytest.approx(0.0, abs=1e-6)                             # trough is clamped to zero


def test_flat_kernel_does_not_decay():
    W = loop_kernel(64, 8, "flat")
    assert W[63, 0] == W[8, 0] == 1.0


# ------------------------------------------------------------- time channel
def test_time_features_decode_elapsed_time_exactly():
    """A single salient event at s: each leaky integrator reads (1-ρ)ρ^(t-s), so t - s is recoverable."""
    T, s = 200, 30
    u = torch.zeros(1, T); u[0, s] = 1.0
    for tau in [8.0, 32.0, 128.0]:
        rho = math.exp(-1 / tau)
        kl, _ = decay_kernel(T, rho)
        L = (u @ kl.T)[0]
        assert (L[:s] == 0).all()
        t = torch.arange(s, min(s + int(4 * tau), T))
        dt = -torch.log(L[t] / (1 - rho)) * tau
        assert torch.allclose(dt, (t - s).float(), atol=1e-2)


def test_time_channel_bounded_causal_and_starts_as_identity():
    tc = TimeChannel(16)
    h = torch.randn(2, 50, 16)
    out, st = tc(h)
    assert torch.allclose(out, h)                                    # zero-initialised projection
    z = st["time_features"]
    assert z.abs().max() <= 1.0 + 1e-5
    h2 = h.clone(); h2[:, 30] += 5
    z2 = tc(h2)[1]["time_features"]
    assert torch.allclose(z[:, :30], z2[:, :30], atol=1e-6)


# ------------------------------------------------------------ the loop
def _naive_loop(m: HippocampalLoop, h: torch.Tensor) -> torch.Tensor:
    """Token-by-token reference for the chunked implementation."""
    B, T, D = h.shape
    tau = m.tau
    W = loop_kernel(T, tau, m.kernel, m.rho, m.amp)
    x_pre = m.norm(h)
    q = F.normalize(m.wq(x_pre), dim=-1)
    out = h.clone()
    energy = h.new_zeros(B, T)
    for t in range(tau, T):
        src = m.norm(out[:, :t]) if m.loop == "closed" else x_pre[:, :t]
        k, v = m._keys(src), m.wv(src)
        a = torch.exp(m.sharpness * (q[:, t:t + 1] @ k.transpose(1, 2) - 1))
        w = W[t:t + 1, :t]
        M = w.unsqueeze(0) * a
        Fb = M @ v
        if m.contrast:
            avail = (w > 0).float()
            prior = (avail.unsqueeze(0) @ v) / avail.sum().clamp_min(1)
            delta = Fb - M.sum(-1, keepdim=True) * prior
        else:
            delta = Fb
        d = (t - torch.arange(t)).float()
        kE = torch.where(d >= tau, (1 - m.energy_rho) * m.energy_rho ** (d - tau), torch.zeros_like(d))
        E = (energy[:, :t] * kE).sum(-1, keepdim=True) if m.ca1_control else h.new_zeros(B, 1)
        inj, _ = m._inject(h[:, t:t + 1], delta, E)
        out[:, t:t + 1] = h[:, t:t + 1] + inj
        energy[:, t] = (inj.norm(dim=-1) / h[:, t:t + 1].norm(dim=-1)).squeeze(-1) ** 2
    return out


@pytest.mark.parametrize("loop", ["open", "closed"])
@pytest.mark.parametrize("ctrl", [True, False])
@pytest.mark.parametrize("kernel", ["osc", "flat"])
def test_chunked_loop_matches_token_by_token_reference(loop, ctrl, kernel):
    torch.manual_seed(1)
    m = HippocampalLoop(16, d_key=8, tau=4, kernel=kernel, loop=loop, ca1_control=ctrl, contrast=True)
    with torch.no_grad():
        m.gain_logit.fill_(0.0)
    h = torch.randn(2, 23, 16)
    with torch.no_grad():
        fast, _ = m(h)
        ref = _naive_loop(m, h)
    assert torch.allclose(fast, ref, atol=1e-5)


@pytest.mark.parametrize("loop", ["open", "closed"])
def test_loop_is_causal(loop):
    m = HippocampalLoop(16, d_key=8, tau=4, loop=loop)
    h = torch.randn(1, 40, 16)
    with torch.no_grad():
        a, _ = m(h)
        h2 = h.clone(); h2[:, 25] += 3
        b, _ = m(h2)
    assert torch.allclose(a[:, :25], b[:, :25], atol=1e-6)


def _amplifier(kernel, ctrl, rho=0.7):
    """Worst case: every memory is fully relevant, values are the identity, gain near 1."""
    m = HippocampalLoop(8, d_key=4, tau=4, kernel=kernel, rho=rho, loop="closed",
                        ca1_control=ctrl, contrast=False, sharpness=0.0)
    with torch.no_grad():
        m.wv.weight.copy_(torch.eye(8)); m.gain_logit.fill_(4.0)
        m.norm.weight.fill_(1.0); m.norm.bias.zero_()
    return m


def test_undamped_closed_loop_runs_away():
    h = torch.ones(1, 96, 8) + 0.1 * torch.randn(1, 96, 8)
    with torch.no_grad():
        _, st = _amplifier("flat", ctrl=False)(h)
    assert st.inj_ratio[0, -8:].mean() > 10 * st.inj_ratio[0, 4:12].mean()


def test_ca1_control_clamps_injection_below_new_input():
    h = torch.ones(1, 96, 8) + 0.1 * torch.randn(1, 96, 8)
    for kernel in ["flat", "osc"]:
        m = _amplifier(kernel, ctrl=True)
        with torch.no_grad():
            _, st = m(h)
        assert st.inj_ratio.max() <= m.max_gain + 1e-5                # new input always dominates


def test_model_forward_backward_and_causal():
    m = TemporalLoopLM(30, 32, 4, window=8, memory=True, time=True, tau=8, loop="closed", d_key=8)
    tok = torch.randint(0, 30, (2, 60))
    out = m(tok)
    F.cross_entropy(out.logits.flatten(0, 1), tok.flatten()).backward()
    m.eval()
    with torch.no_grad():
        a = m(tok).logits
        t2 = tok.clone(); t2[:, 40] = (t2[:, 40] + 1) % 30
        b = m(t2).logits
    assert torch.allclose(a[:, :40], b[:, :40], atol=1e-5)
