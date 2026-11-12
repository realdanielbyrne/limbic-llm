"""Stage 0a: Gated DeltaNet (GDN) diagnostics on the frozen hybrid (Qwen3.5-2B-Base). Nothing is trained.

Each diagnostic (``--diag``, default all) writes one block of the results JSON (``--out``) and per-token /
per-item records through ``common.TraceWriter`` (``--trace-dir/<diag>/``, outside the repository). Notation
as in ``gdn.py``: per GDN layer l and value head h, write gate β_t = σ(b_t), log-decay g_t ≤ 0, retention
ρ_t = exp(g_t), state S_t = ρ_t S_{t−1} + k̂_t(β_t e_t)ᵀ with mismatch e_t = v_t − p̂_t, p̂_t = ρ_t S_{t−1}ᵀk̂_t,
familiarity fam = 1 − ‖e‖/(‖v‖ + ‖p̂‖) and its lagged variant fam_lag (the state from before the 4-token
conv footprint). Per-token signals come from ``GDNTaps`` (the exact per-token copy of the torch recurrent
rule); final states are read from the cache (``past_key_values.layers[l].recurrent_states[0]``, (B, H, K, V)
fp32, and ``conv_states[0]``, the last 4 pre-conv columns).

census          per layer and head, on ordinary text: mean, SD and quantiles of β, logit β, ρ, g and the
                half-life τ½ = ln2/|g_t|; cumulative retention over the next ℓ tokens
                    R_ℓ(t) = Π_{u=t+1..t+ℓ} ρ_u = exp(Σ g_u),   ℓ ∈ {64, 500, 2000},
                as log10 quantiles and the share above e^{−5}, plus the stationary extrapolation ℓ·E[g];
                the token-independent rate A = exp(A_log) and ρ at a = 0; β and ρ by position.
attenuation     two cached prefills identical except for one span (``insert``: [A, span, C] against [A, C];
                ``replace``: [A, span, C] against [A, span′, C], position-aligned). After d tokens of C,
                    a_{l,h}(d) = ‖S⁺_{t+d} − S⁻_{t+d}‖_F / ‖S⁺_t − S⁻_t‖_F,
                against the decay-only bound Π_{u≤d} ρ_u (which ignores delta-rule overwriting); the measured
                horizon is the smallest d with median a < e^{−1} (and < e^{−5}). Also the conv-state
                difference and KL(P⁺ ‖ P⁻) of the next-token distributions on C by offset (temporal reach).
recurrent_only  prefill a context with the cache, then score a continuation with only parts of the cache:
                full, kv_only (attention cache, GDN state empty), rec_only (GDN recurrent + conv state,
                attention cache empty, position ids continuing from the prefix length), rec_noconv (recurrent
                state only, conv tail zeroed), rec_foreign (another context's GDN state), none (empty cache).
                On a needle prompt (≥ 3 facts scattered in natural text; answer NLL, per-token and per-digit
                accuracy, exact, by queried rank) and on natural continuation loss by offset. The check that
                the restored state matters: KL(rec_only ‖ none) and KL(rec_only ‖ rec_foreign).
state_swap      at a boundary, A's continuation is scored with A's attention cache and the GDN state of
                another document B: swap_rec (B's recurrent state, A's conv tail), swap_gdn (B's recurrent and
                conv state), zero_rec (zero recurrent state); and, with the attention cache cut to the sink +
                the last W columns, own_w / swap_w / zero_w (the GDN state is then the only route beyond W).
                Δloss and KL against the own-state condition, by offset into the continuation.
signals         per head on natural text: SD of every GDN signal over tokens, within and between documents
                (and the head-mean SD, which averaging shrinks); Pearson and Spearman correlation with the
                token's own surprisal −log p(x_t | x_<t), with the next-token loss and with log position, and
                the surprisal correlation after removing each position's mean across documents. Probes:
                [pre, E, filler(d), P] with the earlier passage E = the probe P (exact repeat), a later
                passage of the same document (related), P's tokens permuted (shuffled: same bag of words), or
                another document (unrelated, a null), each against the same row with E replaced by an
                unrelated baseline passage; Δsignal on P's tokens at d ∈ {32, 128, 512}.
gate_shift      inject h ← h + g‖h‖d̂ at block L's output (d̂ the gist or a seeded random direction, several g,
                from position ``--gs-from``); at every GDN layer l > L the induced |Δβ|, |Δ logit β|, |Δg|,
                |Δρ|, |Δfam|, |Δfam_lag|, |Δstr| relative to each head's natural SD over the same positions
                (and the exactness check Δ = 0 at l ≤ L); KL and Δloss to the unsteered run; downstream norms.
impulse         inject for 1 token and for a 16-token burst, then g = 0: KL(steered ‖ unsteered), Δloss,
                top-1 agreement, the relative hidden-state change at several blocks and the GDN signal changes,
                by lag after the pulse's last token (log2 bins up to 4,000). ``impulse_cmp`` runs the same
                code on the attention-only comparator (Qwen3-1.7B-Base; no GDN signals).

Every injection arm runs through ``CA2Controller`` + ``CA2Attachment`` with every source and tap registered
(gist agreement, the attention tap and its two sources at the highest full-attention block ≤ L, the GDN tap
and all GDN sources, residual norms at every block), so every signal is traced in every arm; census and
signals use the same rig at gain 0 (byte-identical to the base model). The cache-surgery diagnostics
(attenuation, recurrent_only, state_swap) trace the GDN tap standalone plus the outcomes (the controller's
carried state, e.g. the attention tap's sink columns, would describe the unaltered cache). Outcomes per
token: loss, entropy, max probability, top-1, target rank, and KL / top-1 agreement / reference loss against
the paired reference. Logits are computed in chunks from the captured final-norm output (``logits_to_keep``
keeps the model's own logits to the last position), so long sequences fit.

Deterministic given ``--seed`` (documents are non-overlapping slots of ``--text-file``, chosen by seeded RNG).
Resumable: a diagnostic already complete in ``--out`` is skipped unless ``--force``; a partial trace
directory of a diagnostic that is run again is removed first. ``--tiny`` runs random tiny models (Qwen3.5
layout with spread gates, and a tiny Qwen3 comparator) end to end on CPU with the pure-torch GDN kernels.
No generated text is written anywhere (nothing is generated).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402
from ca2.attention import add_attention  # noqa: E402
from ca2.common import ROOT, TraceWriter, alignment, gain_stats, git_state, load_test_text, paired_boot  # noqa: E402
from ca2.controller import CA2Attachment, CA2Controller, NormTap  # noqa: E402
from ca2.gdn import GDNTaps, add_gdn_sources, gate_params  # noqa: E402
from ca2.sources import Source, StepContext  # noqa: E402

DIAGS = ("census", "attenuation", "recurrent_only", "state_swap", "signals", "gate_shift", "impulse", "impulse_cmp")
SIG_KEYS = ("beta", "ret", "logret", "vnorm", "pred", "err", "cos_vp", "ratio", "fam", "read", "str", "qk", "snorm",
            "pred_lag", "err_lag", "cos_vp_lag", "ratio_lag", "fam_lag")
GS_KEYS = ("beta", "blogit", "logret", "ret", "fam", "fam_lag", "str", "err", "cos_vp")
IMP_GDN_KEYS = ("beta", "logret", "fam", "fam_lag", "snorm")
LN10 = math.log(10.0)

PRESETS = {
    "real": dict(batch=8, census_n=64, census_len=2048, census_lags="64,500,2000", census_keep=65536,
                 att_n=16, att_pre=256, att_span=32, att_dists="0,1,2,4,8,16,32,64,128,256,512,1024,2000",
                 ro_n=32, ro_ctx=512, ro_pairs=3, ro_text_n=32, ro_cont=128,
                 swap_n=32, swap_ctx=512, swap_cont=256, swap_window=64,
                 sig_n=16, sig_len=512, probe_n=24, probe_len=32, probe_pre=64, probe_dists="32,128,512",
                 gs_n=16, gs_len=512, gs_from=32, gs_gains="0.05,0.1,0.2,0.5", gs_dirs="gist,random",
                 imp_n=8, imp_t0=128, imp_lag=4000, imp_bursts="1,16", imp_gains="0.1,0.5", imp_dirs="gist,random",
                 per_head_n=32),
    "tiny": dict(batch=4, census_n=4, census_len=96, census_lags="8,32,64", census_keep=4096,
                 att_n=2, att_pre=24, att_span=6, att_dists="0,1,2,4,8,16,32",
                 ro_n=2, ro_ctx=160, ro_pairs=3, ro_text_n=4, ro_cont=16,
                 swap_n=4, swap_ctx=48, swap_cont=24, swap_window=8,
                 sig_n=4, sig_len=64, probe_n=2, probe_len=8, probe_pre=8, probe_dists="8,16,32",
                 gs_n=4, gs_len=48, gs_from=8, gs_gains="0.1,0.5", gs_dirs="gist,random",
                 imp_n=2, imp_t0=16, imp_lag=96, imp_bursts="1,4", imp_gains="0.5", imp_dirs="gist,random",
                 per_head_n=2),
}


# ----------------------------------------------------------------------------------------- utilities
def _floats(s: str) -> list[float]:
    return [float(x) for x in str(s).split(",") if x.strip()]


def _ints(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if x.strip()]


def _clean(o):
    """JSON-safe: tensors / arrays to lists, NaN and inf to None, floats rounded to 6 significant digits."""
    if isinstance(o, torch.Tensor):
        o = o.detach().cpu().double().numpy()
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not math.isfinite(f) else float(f"{f:.6g}")
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=1))
    os.replace(tmp, path)


def _versions() -> dict:
    out = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
           "cuda": torch.version.cuda, "host": platform.node()}
    for pkg in ("transformers", "flash-linear-attention", "accelerate"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _chunks(xs: Sequence, n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def _chunks_min2(xs: Sequence, n: int) -> list:
    """Chunks of at most n (n ≥ 2) with no chunk of one (a lone last item joins the previous chunk)."""
    cs = [list(xs[i:i + n]) for i in range(0, len(xs), n)]
    if len(cs) > 1 and len(cs[-1]) == 1:
        cs[-2] += cs.pop()
    if cs and len(cs[0]) < 2:
        raise ValueError("the rolled-row foreign control needs at least two sequences")
    return cs


def _nanq(x: np.ndarray, qs: Sequence[float], axis: int = 0) -> np.ndarray:
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanquantile(x, qs, axis=axis)


def _nanmean(x, axis=None):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


def _nanstd(x, axis=None):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(x, axis=axis)


def lag_bins(max_lag: int) -> list[tuple[int, int]]:
    """[0, 1), [1, 2), [2, 4), [4, 8), ... covering 0..max_lag."""
    out, lo = [(0, 1)], 1
    while lo <= max_lag:
        out.append((lo, min(2 * lo, max_lag + 1)))
        lo *= 2
    return out


def _bin_label(b: tuple[int, int]) -> str:
    return str(b[0]) if b[1] - b[0] == 1 else f"{b[0]}-{b[1] - 1}"


def use_torch_gdn_kernels() -> str:
    """Point modeling_qwen3_5's two GDN kernel names at their pure-torch ``__wrapped__`` originals (the test
    fixture's mechanism); needed on CPU when flash-linear-attention is installed. Returns what was done."""
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as M
    except ImportError:
        return "no qwen3_5 module"
    done = []
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        fn = getattr(M, name, None)
        orig = getattr(fn, "__wrapped__", None)
        if fn is not None and orig is not None and getattr(orig, "__module__", "") == M.__name__:
            setattr(M, name, orig)
            done.append(name)
        elif fn is not None and getattr(fn, "__module__", "") == M.__name__:
            done.append(f"{name} (already torch)")
    return "torch: " + ", ".join(done) if done else "unchanged"


# ------------------------------------------------------------------------------------- tiny models
def _spread_gates(model, seed: int) -> None:
    """Random init leaves A_log / dt_bias at their defaults; spread them so heads differ (as the tests do)."""
    g = torch.Generator().manual_seed(seed + 7)
    for blk in pl.find_blocks(model):
        lin = getattr(blk, "linear_attn", None)
        if lin is not None:
            H = lin.A_log.numel()
            with torch.no_grad():
                lin.A_log.copy_(torch.empty(H).uniform_(-4.0, 0.5, generator=g).to(lin.A_log.dtype))
                lin.dt_bias.copy_(torch.empty(H).uniform_(-3.0, 1.0, generator=g).to(lin.dt_bias.dtype))


def build_tiny(arch: str, seed: int, dtype: torch.dtype, device: str):
    """Tiny random models as in the tests: 'qwen35' (pl.build_tiny + spread gates), 'qwen3' (4 blocks)."""
    import transformers
    if arch == "qwen35":
        model, tok = pl.build_tiny("qwen35", seed=seed)
        _spread_gates(model, seed)
    elif arch == "qwen3":
        torch.manual_seed(seed)
        cfg = transformers.Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                       num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                       max_position_embeddings=1024, bos_token_id=256, eos_token_id=256,
                                       pad_token_id=256)
        model, tok = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa"), pl.ByteTokenizer()
    else:
        raise ValueError(arch)
    model = model.to(device, dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def load_any(args, which: str):
    """(model, tok, blocks, kinds, facts) for the main model or the comparator."""
    name = args.model if which == "main" else args.compare_model
    dtype = pl.DTYPES[args.dtype]
    if args.tiny:
        dev = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        model, tok = build_tiny("qwen35" if which == "main" else "qwen3", args.seed, dtype, dev)
        name = f"tiny-{'qwen35' if which == 'main' else 'qwen3'}"
    else:
        ns = argparse.Namespace(model=name, tiny=False, tiny_arch="gpt2", seed=args.seed, dtype=args.dtype,
                                device=args.device, attn_impl=args.attn_impl)
        model, tok = pl.load_model(ns)
    blocks = pl.find_blocks(model)
    kinds = pl.layer_kinds(model, blocks)
    return model, tok, blocks, kinds, model_facts(model, blocks, kinds, name)


def model_facts(model, blocks, kinds, name: str) -> dict:
    cfg = pl.text_config(model)
    lin = [i for i, k in enumerate(kinds) if k == "linear"]
    facts = {"name": name, "class": type(model).__name__, "n_blocks": len(blocks), "kinds": kinds,
             "linear_layers": lin, "attention_layers": [i for i, k in enumerate(kinds) if k == "attention"],
             "hidden": int(cfg.hidden_size), "vocab": int(model.get_output_embeddings().weight.shape[0]),
             "n_params": int(sum(p.numel() for p in model.parameters())),
             "dtype": str(next(model.parameters()).dtype), "device": str(next(model.parameters()).device),
             "n_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}
    if lin:
        la = pl._child(blocks[lin[0]], pl.LINEAR_MIXERS)
        facts.update(gdn_value_heads=int(la.num_v_heads), gdn_key_heads=int(la.num_k_heads),
                     gdn_head_k_dim=int(la.head_k_dim), gdn_head_v_dim=int(la.head_v_dim),
                     conv_kernel=int(la.conv_kernel_size))
        gp = gate_params(model)
        facts["gate_params"] = {str(l): {"A": v["A"].tolist(), "dt_bias": v["dt_bias"].tolist(),
                                         "rho_at_a0": torch.exp(-v["A"] * F.softplus(v["dt_bias"])).tolist()}
                                for l, v in gp.items()}
    return facts


# ------------------------------------------------------------------------------------------ outcomes
def final_norm(model):
    dec = model.get_decoder() if hasattr(model, "get_decoder") else model
    for n in ("norm", "ln_f", "final_layernorm"):
        m = getattr(dec, n, None)
        if m is not None:
            return m
    raise ValueError("no final norm module found")


class Capture:
    """Forward hooks keeping the outputs of the given modules (detached) during a ``with`` block."""

    def __init__(self, mods: dict) -> None:
        self.mods, self.out, self.handles = mods, {}, []

    def __enter__(self) -> "Capture":
        for k, m in self.mods.items():
            self.handles.append(m.register_forward_hook(self._fn(k)))
        return self

    def _fn(self, k):
        def fn(mod, args, out):
            self.out[k] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()


def _keep_kw(model) -> dict:
    import inspect
    return {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}


@torch.no_grad()
def forward(model, ids: torch.Tensor, cache=None, position_ids=None, use_cache: bool = False, blocks=None,
            keep: Sequence[int] = ()) -> SimpleNamespace:
    """One call; returns the final-norm output ``hf`` (B, T, D), the cache and the outputs of blocks ``keep``."""
    mods = {"hf": final_norm(model)}
    mods.update({i: blocks[i] for i in keep})
    kw = _keep_kw(model)
    with Capture(mods) as cap:
        o = model(input_ids=ids, past_key_values=cache, position_ids=position_ids, use_cache=use_cache, **kw)
    return SimpleNamespace(hf=cap.out["hf"], cache=getattr(o, "past_key_values", None),
                           blocks={i: cap.out[i] for i in keep})


@torch.no_grad()
def outcomes(model, hf: torch.Tensor, targets: torch.Tensor, ref: torch.Tensor | None = None,
             budget: int = 1 << 27) -> dict[str, torch.Tensor]:
    """Per position (B, T) on CPU: loss, ent, maxp, top1, rank of ``targets`` (−1: no target → NaN); with
    ``ref`` (the paired run's final hidden) also kl = KL(P ‖ P_ref), agree (same argmax) and ref_loss."""
    lm = model.get_output_embeddings()
    B, T, _ = hf.shape
    V = lm.weight.shape[0]
    step = max(1, budget // max(1, B * V))
    keys = ["loss", "ent", "maxp", "top1", "rank"] + (["kl", "agree", "ref_loss"] if ref is not None else [])
    out = {k: torch.full((B, T), float("nan")) for k in keys}
    tg = targets.to(hf.device)
    for s in range(0, T, step):
        e = min(T, s + step)
        lp = lm(hf[:, s:e]).float().log_softmax(-1)
        p = lp.exp()
        out["ent"][:, s:e] = (-(p * lp).sum(-1)).cpu()
        out["maxp"][:, s:e] = p.amax(-1).cpu()
        t = tg[:, s:e]
        ok = t >= 0
        tc = t.clamp_min(0)
        lt = lp.gather(-1, tc[..., None])[..., 0]
        am = lp.argmax(-1)
        nan = torch.full_like(lt, float("nan"))
        out["loss"][:, s:e] = torch.where(ok, -lt, nan).cpu()
        out["rank"][:, s:e] = torch.where(ok, (lp > lt[..., None]).sum(-1).float(), nan).cpu()
        out["top1"][:, s:e] = torch.where(ok, (am == tc).float(), nan).cpu()
        if ref is not None:
            rlp = lm(ref[:, s:e]).float().log_softmax(-1)
            out["kl"][:, s:e] = (p * (lp - rlp)).sum(-1).cpu()
            out["agree"][:, s:e] = (am == rlp.argmax(-1)).float().cpu()
            out["ref_loss"][:, s:e] = torch.where(ok, -rlp.gather(-1, tc[..., None])[..., 0], nan).cpu()
            del rlp
        del lp, p
    return out


def next_targets(ids: torch.Tensor) -> torch.Tensor:
    t = torch.full_like(ids, -1)
    t[:, :-1] = ids[:, 1:]
    return t


def own_surprisal(loss: torch.Tensor) -> torch.Tensor:
    """−log p(x_t | x_<t) at column t (the loss of the previous position); NaN at column 0."""
    s = torch.full_like(loss, float("nan"))
    s[:, 1:] = loss[:, :-1]
    return s


# ---------------------------------------------------------------------------------- the CA2 rig
class PulseSource(Source):
    """Final source: r_t = 1 at real-token index t ∈ [t0, t0 + n), else 0 (the impulse)."""

    final = True

    def __init__(self, t0: int, n: int, name: str = "pulse") -> None:
        self.t0, self.n, self.name = int(t0), int(n), name

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        return {"s": ((ctx.idx >= self.t0) & (ctx.idx < self.t0 + self.n)).float()}


class Rig:
    """One arm: ``CA2Controller`` at block L's output (constant gain ``gain`` from ``steer_from``, or a pulse)
    with every source and tap registered (agree; attention tap + attn sources at the highest full-attention
    block ≤ L; GDN tap + GDN sources on a hybrid; NormTap). Attach, ``run`` one teacher-forced pass, remove."""

    def __init__(self, E, direction: str = "gist", gain: float = 0.0, steer_from: int = 1,
                 pulse: tuple[int, int] | None = None, L: int | None = None) -> None:
        a = E.args
        L = E.L if L is None else L
        D = int(pl.text_config(E.model).hidden_size)
        src = PulseSource(*pulse) if pulse is not None else "constant"
        g = float(gain)
        self.ctrl = CA2Controller(D, source=src, g_min=0.0, g_max=g, g_abs=max(0.3, g), smooth="none",
                                  direction=direction, steer_from=int(pulse[0] if pulse else steer_from),
                                  seed=a.seed, sink_skip=1)
        R = max([i for i in range(L + 1) if E.kinds[i] == "attention"], default=None)
        taps = [NormTap()]
        if R is not None:
            taps.append(add_attention(self.ctrl, layer=R, window=a.attn_window))
        self.gdn = None
        if E.lin:
            self.gdn = GDNTaps(mismatch=not a.no_mismatch)
            taps.append(self.gdn)
            add_gdn_sources(self.ctrl)
        self.att = CA2Attachment(E.model, self.ctrl, tap=L, relevance_layer=L if R is None else R, taps=taps)
        self.E, self.L, self.R, self.g = E, L, R, g

    @torch.no_grad()
    def run(self, ids: torch.Tensor, keep: Sequence[int] = ()) -> SimpleNamespace:
        ctrl = self.ctrl
        ctrl.reset()
        ctrl.record = True
        r = forward(self.E.model, ids.to(self.E.dev), blocks=self.E.blocks, keep=keep)
        tt = ctrl.trace_tensors()
        st = {l: dict(s) for l, s in self.gdn.stats.items()} if self.gdn is not None else {}
        return SimpleNamespace(hf=r.hf, blocks=r.blocks, trace=tt, gdn=st)

    def remove(self) -> None:
        self.att.remove()


def run_arm(E, ids, keep: Sequence[int] = (), **kw) -> SimpleNamespace:
    rig = Rig(E, **kw)
    try:
        return rig.run(ids, keep)
    finally:
        rig.remove()


def head_stack(st: dict, layers: Sequence[int], key: str) -> torch.Tensor | None:
    """(B, T, n_layers, H) fp32 from per-layer GDN stats, or None if the key is missing."""
    if not st or any(key not in st.get(l, {}) for l in layers):
        return None
    return torch.stack([st[l][key].float() for l in layers], 2)


def trace_tokens(tt: dict, b: int, extra: dict | None = None) -> dict:
    out = {k: v[b] for k, v in tt.items()}
    for k, v in (extra or {}).items():
        out[k] = v[b]
    return out


def add_head_items(E, W: TraceWriter, arm: str, item: dict, st: dict, b: int) -> None:
    """Per-head GDN signals (T, n_layers, H) of row b, for the per-head subsample."""
    if not st or E.n_head_items.get(arm, 0) >= E.args.per_head_n:
        return
    toks = {}
    for k in E.args.per_head_keys.split(","):
        x = head_stack(st, E.lin, k)
        if x is not None:
            toks[f"gdn_head/{k}"] = x[b].cpu()
    if toks:
        W.add(f"{arm}.head", item, toks)
        E.n_head_items[arm] = E.n_head_items.get(arm, 0) + 1


# ------------------------------------------------------------------------------------- cache surgery
def _is_lin(layer) -> bool:
    return hasattr(layer, "conv_states") and hasattr(layer, "recurrent_states")


def cache_parts(cache) -> dict:
    """Cloned parts of a hybrid cache: kv {layer: (keys, values)}, conv / rec {layer: tensor}."""
    kv, conv, rec = {}, {}, {}
    for i, layer in enumerate(cache.layers):
        if _is_lin(layer):
            if layer.has_previous_state[0]:
                conv[i] = layer.conv_states[0].detach().clone()
                if layer.recurrent_states[0] is not None:
                    rec[i] = layer.recurrent_states[0].detach().clone()
        elif getattr(layer, "keys", None) is not None and layer.keys.numel():
            kv[i] = (layer.keys.detach().clone(), layer.values.detach().clone())
    return {"kv": kv, "conv": conv, "rec": rec}


def make_cache(model, kv: dict | None = None, conv: dict | None = None, rec: dict | None = None):
    """A fresh ``DynamicCache`` holding exactly the given parts (missing parts: empty / no GDN state). A GDN
    layer with a conv tail but no recurrent state starts from S = 0 (the model's own no-state rule)."""
    from transformers import DynamicCache
    c = DynamicCache(config=model.config)
    for i, layer in enumerate(c.layers):
        if _is_lin(layer):
            cv, rc = (conv or {}).get(i), (rec or {}).get(i)
            if cv is None and rc is None:
                continue
            if cv is None:
                raise ValueError(f"layer {i}: a recurrent state needs a conv tail (pass zeros for none)")
            layer.lazy_initialization(conv_states=cv, recurrent_states=rc, conv_kernel_size=cv.shape[-1])
            layer.conv_states[0].copy_(cv)
            if rc is not None:
                layer.recurrent_states[0].copy_(rc)
            layer.has_previous_state[0] = True
        elif kv and i in kv:
            layer.update(kv[i][0].clone(), kv[i][1].clone())
    return c


def kv_window(kv: dict, W: int, sink: int = 1) -> dict:
    """Keep the sink columns and the last W columns of every attention layer's cache."""
    return {i: (torch.cat([k[..., :sink, :], k[..., -W:, :]], -2), torch.cat([v[..., :sink, :], v[..., -W:, :]], -2))
            for i, (k, v) in kv.items()}


def continue_call(E, ids: torch.Tensor, cache, P: int, tap: GDNTaps | None = None) -> SimpleNamespace:
    """Score ``ids`` after a prefix of P columns held (in part) by ``cache``; positions continue from P."""
    B, n = ids.shape
    pos = (torch.arange(n, device=E.dev) + P)[None].expand(B, n)
    if tap is not None:
        tap.reset()
        tap.begin_call(None)
    r = forward(E.model, ids.to(E.dev), cache=cache, position_ids=pos, use_cache=True)
    r.sig = tap.signals() if tap is not None else {}
    r.st = {l: dict(s) for l, s in tap.stats.items()} if tap is not None else {}
    return r


# ------------------------------------------------------------------------------------------ environment
class Env:
    def __init__(self, args, model, tok, blocks, kinds, facts, text: torch.Tensor) -> None:
        self.args, self.model, self.tok, self.blocks, self.kinds, self.facts = args, model, tok, blocks, kinds, facts
        self.text = text
        self.dev = next(model.parameters()).device
        self.lin = [i for i, k in enumerate(kinds) if k == "linear"]
        n = len(blocks)
        lay = args.layer if facts.get("which", "main") == "main" else args.cmp_layer
        self.L = n // 2 if lay is None else lay % n
        self.sink = int(tok.bos_token_id)
        self.n_head_items: dict[str, int] = {}

    def used(self) -> np.ndarray:
        """A fresh token-occupancy mask for ``pick(..., taken=...)`` (segments of one diagnostic never overlap)."""
        return np.zeros(len(self.text), dtype=bool)

    def pick(self, length: int, n: int, rng: random.Random, taken: np.ndarray | None = None) -> list[int]:
        """n starts of ``length``-token slots of the text (slot k starts at k·length), seeded; with ``taken`` (an
        occupancy mask from ``used()``) only slots overlapping no earlier pick, which are then marked."""
        slots = np.arange(len(self.text) // length) * length
        if taken is not None:
            cs = np.concatenate([[0], np.cumsum(taken)])
            slots = slots[cs[slots + length] - cs[slots] == 0]
        if len(slots) < n:
            raise ValueError(f"the text has {len(slots)} free slots of {length} tokens; need {n}")
        ch = [int(slots[i]) for i in rng.sample(range(len(slots)), n)]
        if taken is not None:
            for s in ch:
                taken[s:s + length] = True
        return ch

    def seg(self, start: int, length: int) -> torch.Tensor:
        return self.text[start:start + length]

    def rows(self, starts: Sequence[int], length: int) -> torch.Tensor:
        return torch.stack([torch.cat([torch.tensor([self.sink]), self.seg(s, length)]) for s in starts])

    def writer(self, diag: str) -> TraceWriter:
        d = Path(self.args.trace_dir).expanduser() / diag
        if d.exists():                                          # a partial run of this diagnostic: start over
            alignment().judge.assert_private(d, ROOT)
            shutil.rmtree(d)
        return TraceWriter(diag, root=self.args.trace_dir, meta={"diag": diag, "model": self.facts["name"],
                                                                 "args": _clean(vars(self.args))})


def _boot(a, b=None, seed=0, n=None) -> dict:
    return paired_boot(a, b, n=n or 4000, seed=seed)


# ---------------------------------------------------------------------------------------------- census
def d_census(E: Env) -> dict:
    a = E.args
    if not E.lin:
        raise ValueError("census needs GDN layers")
    T, lags = a.census_len, _ints(a.census_lags)
    rng = random.Random(a.seed * 1000 + 1)
    starts = E.pick(T, a.census_n, rng)
    W = E.writer("census")
    nL = len(E.lin)
    keep_stride = max(1, (a.census_n * T) // max(1, a.census_keep))
    pool = {k: [] for k in ("beta", "logret")}
    ret_pool = {lag: [] for lag in lags}
    meanG = []
    pos_edges = [b for b in lag_bins(T) if b[0] < T]
    pos_sum = {k: np.zeros((len(pos_edges), nL)) for k in ("beta", "ret")}
    pos_n = np.zeros(len(pos_edges))
    losses = []
    for bi, ch in enumerate(_chunks(starts, a.batch)):
        ids = E.rows(ch, T)
        res = run_arm(E, ids, gain=0.0)
        out = outcomes(E.model, res.hf, next_targets(ids))
        out["surp"] = own_surprisal(out["loss"])
        losses.append(_nanmean(out["loss"][:, 1:].numpy(), 1))
        beta = head_stack(res.gdn, E.lin, "beta")[:, 1:].cpu()          # sink column excluded
        lg = head_stack(res.gdn, E.lin, "logret")[:, 1:].cpu().double()
        B = beta.shape[0]
        pool["beta"].append(beta.reshape(-1, nL, beta.shape[-1])[::keep_stride])
        pool["logret"].append(lg.float().reshape(-1, nL, lg.shape[-1])[::keep_stride])
        meanG.append(lg.mean(1))                                       # (B, nL, H) per-doc mean log-decay
        cs = torch.cat([torch.zeros_like(lg[:, :1]), lg.cumsum(1)], 1)  # cs[t] = Σ_{u<t} g
        for lag in lags:
            if lag < lg.shape[1]:
                R = (cs[:, lag:] - cs[:, :-lag])[:, 1:] / LN10          # log10 Π over the next ℓ tokens
                stride = max(1, R.shape[1] // 256)
                ret_pool[lag].append(R[:, ::stride].reshape(-1, nL, R.shape[-1]).float())
        bm, rm = beta.mean(-1).numpy(), lg.exp().mean(-1).float().numpy()  # head means (B, T−1, nL)
        for j, (lo, hi) in enumerate(pos_edges):
            sl = slice(max(lo, 1) - 1, min(hi, bm.shape[1] + 1) - 1)    # columns [lo, hi) without the sink
            if sl.start < sl.stop:
                pos_sum["beta"][j] += bm[:, sl].mean(1).sum(0)
                pos_sum["ret"][j] += rm[:, sl].mean(1).sum(0)
                pos_n[j] += B
        for b in range(B):
            item = {"doc": bi * a.batch + b, "start": int(ch[b]), "len": T}
            W.add("base", item, trace_tokens(res.trace, b, {f"out/{k}": v for k, v in out.items()}))
            add_head_items(E, W, "base", item, res.gdn, b)
    beta = torch.cat(pool["beta"]).numpy()
    lg = torch.cat(pool["logret"]).double().numpy()
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    with np.errstate(all="ignore"):
        bl = np.log(np.clip(beta, 1e-7, 1 - 1e-7) / np.clip(1 - beta, 1e-7, None))
        half = np.log(2.0) / np.maximum(-lg, 1e-12)
    ret = np.exp(lg)
    per_head = {
        "beta": {"mean": beta.mean(0), "sd": beta.std(0), "q": dict(zip(map(str, qs), _nanq(beta, qs)))},
        "beta_logit_sd": bl.std(0),
        "ret": {"mean": ret.mean(0), "sd": ret.std(0), "q": dict(zip(map(str, qs), _nanq(ret, qs))),
                "frac_gt_0.99": (ret > 0.99).mean(0), "frac_gt_0.999": (ret > 0.999).mean(0)},
        "logret": {"mean": lg.mean(0), "sd": lg.std(0)},
        "halflife": {"q": dict(zip(["0.05", "0.25", "0.5", "0.75", "0.95"],
                                   _nanq(half, [0.05, 0.25, 0.5, 0.75, 0.95])))},
    }
    meanG = torch.cat(meanG).numpy()                                   # (docs, nL, H)
    cum = {}
    for lag in lags:
        ext = lag * meanG.mean(0) / LN10                                # stationary extrapolation (log10)
        if ret_pool[lag]:
            R = torch.cat(ret_pool[lag]).double().numpy()
            cum[str(lag)] = {"n": R.shape[0], "log10_q": dict(zip(["0.05", "0.25", "0.5", "0.75", "0.95", "0.99"],
                                                                  _nanq(R, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))),
                             "frac_gt_e-5": (R > -5.0 / LN10).mean(0), "log10_extrapolated": ext}
        else:
            cum[str(lag)] = {"n": 0, "log10_extrapolated": ext, "note": f"census_len {T} ≤ lag: extrapolation only"}
    per_head["cum_retention"] = cum
    layers = {}
    for j, l in enumerate(E.lin):
        row = {"beta_mean_med": float(np.median(beta.mean(0)[j])), "beta_sd_med": float(np.median(beta.std(0)[j])),
               "beta_logit_sd_min": float(bl.std(0)[j].min()), "ret_med_med": float(np.median(np.median(ret, 0)[j])),
               "halflife_med_max": float(np.median(half, 0)[j].max())}
        for lag in lags:
            c = cum[str(lag)]
            if c.get("n"):
                med = np.asarray(c["log10_q"]["0.5"])[j]
                row[f"R{lag}_log10_med_max"] = float(med.max())
                row[f"R{lag}_heads_gt_e-5"] = int((med > -5.0 / LN10).sum())
            row[f"R{lag}_extrap_log10_max"] = float(np.asarray(c["log10_extrapolated"])[j].max())
        layers[str(l)] = row
    pos = {"bins": [_bin_label(b) for b in pos_edges],
           "beta": pos_sum["beta"] / np.maximum(pos_n, 1)[:, None],
           "ret": pos_sum["ret"] / np.maximum(pos_n, 1)[:, None]}
    loss = np.concatenate(losses)
    summ = {"n_docs": len(starts), "n_tokens": int(len(starts) * T), "mean_loss": float(loss.mean()),
            "loss_boot": _boot(loss, seed=E.args.seed), "per_layer": layers}
    stop = {}
    for lag in lags:
        c = cum[str(lag)]
        if c.get("n"):
            stop[f"heads_median_R{lag}_gt_e-5"] = int((np.asarray(c["log10_q"]["0.5"]) > -5.0 / LN10).sum())
    summ["horizon_counts"] = stop
    print_table("census (per GDN layer; head medians / maxima)", layers)
    return {"settings": {"n": a.census_n, "len": T, "lags": lags, "keep_stride": keep_stride, "layers": E.lin},
            "summary": summ, "per_head": per_head, "by_position": pos, "trace": W.close()}


# ----------------------------------------------------------------------------------------- attenuation
def _rec_states(cache, layers) -> dict:
    return {l: cache.layers[l].recurrent_states[0].detach().float() for l in layers}


def _conv_states(cache, layers) -> dict:
    return {l: cache.layers[l].conv_states[0].detach().float() for l in layers}


def _diffnorm(a: dict, b: dict, layers, sl_a=slice(None), sl_b=slice(None)) -> torch.Tensor:
    """(B, n_layers, H) ‖a − b‖_F over the last two dims per head (rec) or per layer (conv: (B, n_layers))."""
    out = []
    for l in layers:
        d = a[l][sl_a] - b[l][sl_b]
        out.append(d.flatten(2).norm(dim=-1) if d.dim() == 4 else d.flatten(1).norm(dim=-1))
    return torch.stack(out, 1).cpu()


def d_attenuation(E: Env) -> dict:
    a = E.args
    if not E.lin:
        raise ValueError("attenuation needs GDN layers")
    dists = sorted(set([0] + _ints(a.att_dists)))
    maxd = dists[-1]
    rng = random.Random(a.seed * 1000 + 2)
    taken = E.used()
    base_starts = E.pick(a.att_pre + maxd + 1, a.att_n, rng, taken)
    span_starts = E.pick(a.att_span, 2 * a.att_n, rng, taken)
    W = E.writer("attenuation")
    lin = E.lin
    tap = GDNTaps(mismatch=False)
    k_items = max(1, a.batch // 2)
    rat = {"insert": [], "replace": []}
    decay, base_norm, snorm = [], {"insert": [], "replace": []}, []
    conv_r = {"insert": [], "replace": []}
    kls = {"insert": [], "replace": []}
    for bi, idx in enumerate(_chunks(list(range(a.att_n)), k_items)):
        k = len(idx)
        A = [E.seg(base_starts[i], a.att_pre) for i in idx]
        C = torch.stack([E.seg(base_starts[i] + a.att_pre, maxd + 1) for i in idx]).to(E.dev)
        sp = [E.seg(span_starts[2 * i], a.att_span) for i in idx]
        sp2 = [E.seg(span_starts[2 * i + 1], a.att_span) for i in idx]
        sink = torch.tensor([E.sink])
        plus = torch.stack([torch.cat([sink, A[j], sp[j]]) for j in range(k)] +
                           [torch.cat([sink, A[j], sp2[j]]) for j in range(k)]).to(E.dev)
        minus = torch.stack([torch.cat([sink, A[j]]) for j in range(k)]).to(E.dev)
        rp = forward(E.model, plus, use_cache=True)
        rm = forward(E.model, minus, use_cache=True)
        cp, cm = rp.cache, rm.cache
        Pp, Pm = plus.shape[1], minus.shape[1]
        hf_p, hf_m = [rp.hf[:, -1:]], [rm.hf[:, -1:]]
        tap.attach_model(E.model)
        G = torch.zeros(2 * k, len(lin), E.facts["gdn_value_heads"], dtype=torch.float64)
        per_d = {"insert": [], "replace": []}
        dec_d, conv_d, sn_d = [], {"insert": [], "replace": []}, []
        prev = 0
        for d in dists:
            if d > prev:
                r1 = continue_call(E, C.repeat(2, 1)[:, prev:d], cp, Pp + prev, tap)
                G += head_stack(r1.st, lin, "logret").double().sum(1).cpu()
                hf_p.append(r1.hf)
                r2 = continue_call(E, C[:, prev:d], cm, Pm + prev, None)
                hf_m.append(r2.hf)
                prev = d
            Sp, Sm = _rec_states(cp, lin), _rec_states(cm, lin)
            Cp, Cm = _conv_states(cp, lin), _conv_states(cm, lin)
            per_d["insert"].append(_diffnorm(Sp, Sm, lin, slice(0, k)))
            per_d["replace"].append(_diffnorm(Sp, Sp, lin, slice(0, k), slice(k, 2 * k)))
            conv_d["insert"].append(_diffnorm(Cp, Cm, lin, slice(0, k)))
            conv_d["replace"].append(_diffnorm(Cp, Cp, lin, slice(0, k), slice(k, 2 * k)))
            dec_d.append((G[:k] / LN10).float())
            sn_d.append(torch.stack([Sp[l][:k].flatten(2).norm(dim=-1) for l in lin], 1).cpu())
        tap.remove()
        hp, hm = torch.cat(hf_p, 1), torch.cat(hf_m, 1)
        tgt = C[:, : hp.shape[1]]
        o_ins = outcomes(E.model, hp[:k], tgt, ref=hm)
        o_rep = outcomes(E.model, hp[:k], tgt, ref=hp[k:])
        for kind in ("insert", "replace"):
            x = torch.stack(per_d[kind], 1)                              # (k, n_d, nL, H)
            base = x[:, :1].clamp_min(1e-30)
            rat[kind].append(x / base)
            base_norm[kind].append(x[:, 0])
            cv = torch.stack(conv_d[kind], 1)
            conv_r[kind].append(cv / cv[:, :1].clamp_min(1e-30))
            kls[kind].append((o_ins if kind == "insert" else o_rep)["kl"])
        decay.append(torch.stack(dec_d, 1))
        snorm.append(torch.stack(sn_d, 1))
        for j, i in enumerate(idx):
            item = {"pair": i, "start": int(base_starts[i]), "dists": dists,
                    "ratio_insert_log10": torch.log10(rat["insert"][-1][j].clamp_min(1e-30)),
                    "ratio_replace_log10": torch.log10(rat["replace"][-1][j].clamp_min(1e-30)),
                    "decay_log10": decay[-1][j], "dS0_insert": base_norm["insert"][-1][j],
                    "dS0_replace": base_norm["replace"][-1][j], "conv_ratio_insert": conv_r["insert"][-1][j]}
            W.add("pairs", item, {"kl_insert": o_ins["kl"][j], "kl_replace": o_rep["kl"][j],
                                  "loss_plus": o_ins["loss"][j], "loss_minus": o_ins["ref_loss"][j],
                                  "loss_sub": o_rep["ref_loss"][j], "agree_insert": o_ins["agree"][j]})
    out = {"settings": {"n": a.att_n, "pre": a.att_pre, "span": a.att_span, "dists": dists, "layers": lin}}
    dec = torch.cat(decay).double().numpy()                              # (N, n_d, nL, H) log10 decay-only
    out["decay_only_log10_median"] = np.median(dec, 0)
    summ = {}
    for kind in ("insert", "replace"):
        r = torch.log10(torch.cat(rat[kind]).clamp_min(1e-30)).double().numpy()   # (N, n_d, nL, H)
        med = np.median(r, 0)
        hz = {}
        for thr, nm in ((-1.0 / LN10, "e-1"), (-5.0 / LN10, "e-5")):
            below = med < thr
            first = np.where(below.any(0), np.argmax(below, 0), -1)
            hz[nm] = np.where(first >= 0, np.asarray(dists)[np.clip(first, 0, None)], -1)
        layer_mean = r.mean(-1)                                          # (N, n_d, nL) head-mean log10 ratio
        boots = {str(d): [_boot(layer_mean[:, i, j], seed=a.seed) for j in range(len(lin))]
                 for i, d in enumerate(dists)}
        kl = torch.cat(kls[kind]).double().numpy()
        kb = lag_bins(kl.shape[1] - 1)
        kl_bins = {_bin_label(b): _boot(_nanmean(kl[:, b[0]:b[1]], 1), seed=a.seed) for b in kb if b[0] < kl.shape[1]}
        conv = torch.log10(torch.cat(conv_r[kind]).clamp_min(1e-30)).double().numpy()
        summ[kind] = {"log10_ratio_median": med, "log10_ratio_q": dict(zip(["0.1", "0.9"], _nanq(r, [0.1, 0.9]))),
                      "horizon_tokens": hz, "layer_mean_log10_ratio_boot": boots,
                      "measured_minus_decay_log10_median": np.median(r - dec, 0),
                      "dS0_median": np.median(torch.cat(base_norm[kind]).double().numpy(), 0),
                      "conv_log10_ratio_median": np.median(conv, 0), "kl_by_offset": kl_bins}
    out["state_norm_median"] = np.median(torch.cat(snorm).double().numpy(), 0)
    out["arms"] = summ
    rows = {}
    for j, l in enumerate(lin):
        med = summ["insert"]["log10_ratio_median"]
        show = (dists[1], dists[len(dists) // 2], maxd)
        rows[str(l)] = {f"d{d}": float(np.median(med[i, j])) for i, d in enumerate(dists) if d in show}
        rows[str(l)]["horizon_e-1_max"] = int(summ["insert"]["horizon_tokens"]["e-1"][j].max())
        rows[str(l)]["never_e-1"] = int((summ["insert"]["horizon_tokens"]["e-1"][j] < 0).sum())
    print_table("attenuation, insert: head-median log10 ‖ΔS(d)‖/‖ΔS(0)‖", rows)
    out["trace"] = W.close()
    return out


# -------------------------------------------------------------------------------------- recurrent_only
RO_CONDS = ("full", "kv_only", "rec_only", "rec_noconv", "rec_foreign", "none")


def _conditions(parts: dict, foreign: dict) -> dict[str, dict]:
    zconv = {i: torch.zeros_like(c) for i, c in parts["conv"].items()}
    return {"full": dict(kv=parts["kv"], conv=parts["conv"], rec=parts["rec"]),
            "kv_only": dict(kv=parts["kv"]),
            "rec_only": dict(conv=parts["conv"], rec=parts["rec"]),
            "rec_noconv": dict(conv=zconv, rec=parts["rec"]),
            "rec_foreign": dict(conv=foreign["conv"], rec=foreign["rec"]),
            "none": {}}


def _needle_item(E: Env, rng: random.Random, ctx_len: int, pairs: int) -> dict:
    """Facts scattered at random positions in a natural-text filler; query + answer as the continuation."""
    enc = E.tok.encode
    names = rng.sample(pl.NAMES, pairs)
    vals = ["".join(rng.choice("0123456789") for _ in range(4)) for _ in names]
    facts = [enc(f" The code for {n} is") + enc(f" {v}") + enc(".") for n, v in zip(names, vals)]
    n_fill = ctx_len - 1 - sum(len(f) for f in facts)
    if n_fill < pairs:
        raise ValueError(f"--ro-ctx {ctx_len} too short for {pairs} facts")
    o = rng.randrange(0, len(E.text) - n_fill)
    filler = E.text[o:o + n_fill].tolist()
    cuts = sorted(rng.sample(range(n_fill + 1), pairs))
    order = list(range(pairs))
    rng.shuffle(order)
    ids, at, prev = [E.sink], {}, 0
    for c, j in zip(cuts, order):
        ids += filler[prev:c]
        at[j] = len(ids) + len(enc(f" The code for {names[j]} is"))
        ids += facts[j]
        prev = c
    ids += filler[prev:]
    q = rng.randrange(pairs)
    query, answer = enc(f" The code for {names[q]} is"), enc(f" {vals[q]}")
    rank = sorted(at.values()).index(at[q])
    return {"ctx": torch.tensor(ids), "query": query, "answer": answer, "rank": rank,
            "gap": len(ids) - at[q] + len(query), "pairs": pairs,
            "digit": [any(ch.isdigit() for ch in E.tok.decode([t])) for t in answer]}


def d_recurrent_only(E: Env) -> dict:
    a = E.args
    if not E.lin:
        raise ValueError("recurrent_only needs GDN layers")
    rng = random.Random(a.seed * 1000 + 3)
    W = E.writer("recurrent_only")
    tap = GDNTaps(mismatch=not a.no_mismatch)
    # ---- needle
    nd = {c: {"nll": [], "acc": [], "exact": [], "digit": [], "kl_none": [], "kl_foreign": [], "flip_none": []}
          for c in RO_CONDS}
    ranks = []
    for i in range(a.ro_n):
        it, fo = _needle_item(E, rng, a.ro_ctx, a.ro_pairs), _needle_item(E, rng, a.ro_ctx, a.ro_pairs)
        ctx = it["ctx"][None].to(E.dev)
        P = ctx.shape[1]
        parts = cache_parts(forward(E.model, ctx, use_cache=True).cache)
        fparts = cache_parts(forward(E.model, fo["ctx"][None].to(E.dev), use_cache=True).cache)
        cont = torch.tensor(it["query"] + it["answer"])[None].to(E.dev)
        tg = next_targets(cont)
        nq, na = len(it["query"]), len(it["answer"])
        ans = slice(nq - 1, nq - 1 + na)                                 # positions predicting the answer
        conds = _conditions(parts, fparts)
        tap.attach_model(E.model)
        res = {c: continue_call(E, cont, make_cache(E.model, **kw), P, tap) for c, kw in conds.items()}
        tap.remove()
        ranks.append(it["rank"])
        for c in RO_CONDS:
            o = outcomes(E.model, res[c].hf, tg, ref=res["none"].hf)
            of = outcomes(E.model, res[c].hf, tg, ref=res["rec_foreign"].hf)
            top1 = o["top1"][0, ans]
            dig = torch.tensor(it["digit"], dtype=torch.bool)
            nd[c]["nll"].append(float(o["loss"][0, ans].mean()))
            nd[c]["acc"].append(float(top1.mean()))
            nd[c]["exact"].append(float(top1.min()))
            nd[c]["digit"].append(float(top1[dig].mean()) if dig.any() else float("nan"))
            nd[c]["kl_none"].append(float(o["kl"][0, : nq - 1 + na].mean()))
            nd[c]["kl_foreign"].append(float(of["kl"][0, : nq - 1 + na].mean()))
            nd[c]["flip_none"].append(float(1 - o["agree"][0, : nq - 1 + na].mean()))
            item = {"item": i, "cond": c, "rank": it["rank"], "gap": it["gap"], "pairs": it["pairs"], "P": P,
                    "n_query": nq, "n_ans": na, **{k: v[-1] for k, v in nd[c].items()}}
            toks = {f"out/{k}": v[0] for k, v in o.items()}
            toks.update({f"gdn/{k}": v[0].cpu() for k, v in res[c].sig.items()})
            W.add(f"needle.{c}", item, toks)
    needle = {c: {k: float(_nanmean(np.asarray(v))) for k, v in d.items()} for c, d in nd.items()}
    rk = np.asarray(ranks)
    for c in RO_CONDS:
        needle[c]["digit_by_rank"] = {str(r): float(_nanmean(np.asarray(nd[c]["digit"])[rk == r]))
                                      for r in sorted(set(ranks))}
    comps = {}
    for x, y in (("rec_only", "none"), ("rec_only", "rec_foreign"), ("kv_only", "full"), ("full", "none"),
                 ("rec_noconv", "none"), ("rec_only", "rec_noconv"), ("kv_only", "none")):
        comps[f"{x}-{y}"] = {m: _boot(nd[x][m], nd[y][m], seed=a.seed) for m in ("nll", "acc", "digit")}
    needle_out = {"by_cond": needle, "comparisons": comps, "chance_digit": 0.1,
                  "check_restored_state": {"kl_rec_only_vs_none": _boot(nd["rec_only"]["kl_none"], seed=a.seed),
                                           "kl_rec_only_vs_foreign": _boot(nd["rec_only"]["kl_foreign"], seed=a.seed),
                                           "flip_rec_only_vs_none": _boot(nd["rec_only"]["flip_none"], seed=a.seed)}}
    # ---- natural continuation
    starts = E.pick(a.ro_ctx + a.ro_cont, a.ro_text_n, rng)
    lossb = {c: [] for c in RO_CONDS}
    klb = {c: [] for c in RO_CONDS}
    for bi, ch in enumerate(_chunks_min2(starts, max(2, a.batch))):
        ids = E.rows(ch, a.ro_ctx + a.ro_cont).to(E.dev)
        ctx, cont = ids[:, : a.ro_ctx + 1], ids[:, a.ro_ctx + 1:]
        parts = cache_parts(forward(E.model, ctx, use_cache=True).cache)
        foreign = {"conv": {i: t.roll(1, 0) for i, t in parts["conv"].items()},
                   "rec": {i: t.roll(1, 0) for i, t in parts["rec"].items()}}
        tap.attach_model(E.model)
        res = {c: continue_call(E, cont, make_cache(E.model, **kw), ctx.shape[1], tap)
               for c, kw in _conditions(parts, foreign).items()}
        tap.remove()
        tg = next_targets(cont)
        for c in RO_CONDS:
            o = outcomes(E.model, res[c].hf, tg, ref=res["full"].hf)
            lossb[c].append(o["loss"][:, :-1])
            klb[c].append(o["kl"][:, :-1])
            for b in range(len(ch)):
                toks = {f"out/{k}": v[b] for k, v in o.items()}
                toks.update({f"gdn/{k}": v[b].cpu() for k, v in res[c].sig.items()})
                W.add(f"text.{c}", {"batch": bi, "row": b, "start": int(ch[b]), "cond": c}, toks)
    L_ = {c: torch.cat(v).double().numpy() for c, v in lossb.items()}
    K_ = {c: torch.cat(v).double().numpy() for c, v in klb.items()}
    bins = [b for b in lag_bins(a.ro_cont) if b[0] < a.ro_cont - 1]
    text = {"mean_loss": {c: float(L_[c].mean()) for c in RO_CONDS},
            "loss_by_offset": {c: {_bin_label(b): float(L_[c][:, b[0]:b[1]].mean()) for b in bins} for c in RO_CONDS},
            "kl_to_full_by_offset": {c: {_bin_label(b): float(K_[c][:, b[0]:b[1]].mean()) for b in bins}
                                     for c in RO_CONDS}, "comparisons": {}}
    for x, y in (("rec_only", "none"), ("rec_only", "rec_foreign"), ("kv_only", "full"), ("rec_noconv", "none")):
        text["comparisons"][f"{x}-{y}"] = {_bin_label(b): _boot(L_[x][:, b[0]:b[1]].mean(1),
                                                                L_[y][:, b[0]:b[1]].mean(1), seed=a.seed) for b in bins}
        text["comparisons"][f"{x}-{y}"]["all"] = _boot(L_[x].mean(1), L_[y].mean(1), seed=a.seed)
    rows = {c: {"needle_nll": needle[c]["nll"], "digit_acc": needle[c]["digit"], "exact": needle[c]["exact"],
                "text_loss": text["mean_loss"][c]} for c in RO_CONDS}
    print_table("recurrent_only (needle answer, natural continuation)", rows)
    return {"settings": {"needle_n": a.ro_n, "ctx": a.ro_ctx, "pairs": a.ro_pairs, "text_n": len(starts),
                         "cont": a.ro_cont, "conditions": RO_CONDS}, "needle": needle_out, "text": text,
            "trace": W.close()}


# -------------------------------------------------------------------------------------------- state_swap
SW_CONDS = ("own", "swap_rec", "swap_gdn", "zero_rec", "own_w", "swap_w", "zero_w")


def d_state_swap(E: Env) -> dict:
    a = E.args
    if not E.lin:
        raise ValueError("state_swap needs GDN layers")
    if a.swap_window >= a.swap_ctx:
        raise ValueError("--swap-window must be shorter than --swap-ctx")
    rng = random.Random(a.seed * 1000 + 4)
    taken = E.used()
    A_st = E.pick(a.swap_ctx + a.swap_cont, a.swap_n, rng, taken)
    B_st = E.pick(a.swap_ctx + a.swap_cont, a.swap_n, rng, taken)
    W = E.writer("state_swap")
    tap = GDNTaps(mismatch=not a.no_mismatch)
    loss = {c: [] for c in SW_CONDS}
    kl = {c: [] for c in SW_CONDS}
    for bi, (ca, cb) in enumerate(zip(_chunks(A_st, a.batch), _chunks(B_st, a.batch))):
        ia = E.rows(ca, a.swap_ctx + a.swap_cont).to(E.dev)
        ib = E.rows(cb, a.swap_ctx).to(E.dev)
        ctx, cont = ia[:, : a.swap_ctx + 1], ia[:, a.swap_ctx + 1:]
        pa = cache_parts(forward(E.model, ctx, use_cache=True).cache)
        pb = cache_parts(forward(E.model, ib, use_cache=True).cache)
        zrec = {i: torch.zeros_like(t) for i, t in pa["rec"].items()}
        kvw = kv_window(pa["kv"], a.swap_window, 1)
        conds = {"own": dict(kv=pa["kv"], conv=pa["conv"], rec=pa["rec"]),
                 "swap_rec": dict(kv=pa["kv"], conv=pa["conv"], rec=pb["rec"]),
                 "swap_gdn": dict(kv=pa["kv"], conv=pb["conv"], rec=pb["rec"]),
                 "zero_rec": dict(kv=pa["kv"], conv=pa["conv"], rec=zrec),
                 "own_w": dict(kv=kvw, conv=pa["conv"], rec=pa["rec"]),
                 "swap_w": dict(kv=kvw, conv=pa["conv"], rec=pb["rec"]),
                 "zero_w": dict(kv=kvw, conv=pa["conv"], rec=zrec)}
        tap.attach_model(E.model)
        res = {c: continue_call(E, cont, make_cache(E.model, **kw), ctx.shape[1], tap) for c, kw in conds.items()}
        tap.remove()
        tg = next_targets(cont)
        for c in SW_CONDS:
            ref = res["own_w" if c.endswith("_w") else "own"].hf
            o = outcomes(E.model, res[c].hf, tg, ref=ref)
            loss[c].append(o["loss"][:, :-1])
            kl[c].append(o["kl"][:, :-1])
            for b in range(len(ca)):
                toks = {f"out/{k}": v[b] for k, v in o.items()}
                toks.update({f"gdn/{k}": v[b].cpu() for k, v in res[c].sig.items()})
                W.add(c, {"batch": bi, "row": b, "start_A": int(ca[b]), "start_B": int(cb[b])}, toks)
    L_ = {c: torch.cat(v).double().numpy() for c, v in loss.items()}
    K_ = {c: torch.cat(v).double().numpy() for c, v in kl.items()}
    bins = [b for b in lag_bins(a.swap_cont) if b[0] < a.swap_cont - 1]
    comps = {}
    for x, y in (("swap_rec", "own"), ("swap_gdn", "own"), ("zero_rec", "own"), ("swap_w", "own_w"),
                 ("zero_w", "own_w"), ("own_w", "own"), ("swap_rec", "zero_rec"), ("swap_w", "zero_w")):
        comps[f"{x}-{y}"] = {_bin_label(b): _boot(L_[x][:, b[0]:b[1]].mean(1), L_[y][:, b[0]:b[1]].mean(1),
                                                  seed=a.seed) for b in bins}
        comps[f"{x}-{y}"]["all"] = _boot(L_[x].mean(1), L_[y].mean(1), seed=a.seed)
    out = {"settings": {"n": a.swap_n, "ctx": a.swap_ctx, "cont": a.swap_cont, "window": a.swap_window,
                        "conditions": SW_CONDS},
           "mean_loss": {c: float(L_[c].mean()) for c in SW_CONDS},
           "kl_to_own_by_offset": {c: {_bin_label(b): float(K_[c][:, b[0]:b[1]].mean()) for b in bins}
                                   for c in SW_CONDS},
           "comparisons": comps, "trace": W.close()}
    rows = {k: {"dloss_all": v["all"]["diff"], "lo": v["all"]["lo"], "hi": v["all"]["hi"],
                "dloss_first_bin": v[_bin_label(bins[0])]["diff"]} for k, v in comps.items()}
    print_table("state_swap: Δloss on A's continuation", rows)
    return out


# ------------------------------------------------------------------------------------------------ signals
def _pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Column-wise Pearson r of x (N, K) with y (N,) or (N, K), over rows where both are finite."""
    if y.ndim == 1:
        y = np.broadcast_to(y[:, None], x.shape)
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(0)
    xs, ys = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    mx, my = xs.sum(0) / np.maximum(n, 1), ys.sum(0) / np.maximum(n, 1)
    xc, yc = np.where(ok, x - mx, 0.0), np.where(ok, y - my, 0.0)
    den = np.sqrt((xc * xc).sum(0) * (yc * yc).sum(0))
    with np.errstate(all="ignore"):
        return np.where((n > 2) & (den > 0), (xc * yc).sum(0) / den, np.nan)


def _rank(x: np.ndarray) -> np.ndarray:
    """Column-wise ranks (NaN kept as NaN; ties broken by order)."""
    r = np.full(x.shape, np.nan)
    for j in range(x.shape[1]):
        ok = np.isfinite(x[:, j])
        v = x[ok, j]
        rr = np.empty(v.size)
        rr[np.argsort(v, kind="stable")] = np.arange(v.size)
        r[ok, j] = rr
    return r


def _spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y)
    xx = np.where(ok[:, None], x, np.nan)
    return _pearson(_rank(xx), _rank(np.where(ok, y, np.nan)[:, None])[:, 0])


def d_signals(E: Env) -> dict:
    a = E.args
    if not E.lin:
        raise ValueError("signals needs GDN layers")
    if a.no_mismatch:
        raise ValueError("signals needs the per-token mismatch loop (drop --no-mismatch)")
    rng = random.Random(a.seed * 1000 + 5)
    W = E.writer("signals")
    lin, T = E.lin, a.sig_len
    taken = E.used()
    starts = E.pick(T, a.sig_n, rng, taken)
    xs = {k: [] for k in SIG_KEYS}
    surp, nxt = [], []
    for bi, ch in enumerate(_chunks(starts, a.batch)):
        ids = E.rows(ch, T)
        res = run_arm(E, ids, gain=0.0)
        out = outcomes(E.model, res.hf, next_targets(ids))
        out["surp"] = own_surprisal(out["loss"])
        surp.append(out["surp"][:, 1:])
        nxt.append(out["loss"][:, 1:])
        for k in SIG_KEYS:
            x = head_stack(res.gdn, lin, k)
            if x is not None:
                xs[k].append(x[:, 1:].cpu())
        for b in range(len(ch)):
            item = {"doc": bi * a.batch + b, "start": int(ch[b]), "len": T}
            W.add("natural", item, trace_tokens(res.trace, b, {f"out/{k}": v for k, v in out.items()}))
            add_head_items(E, W, "natural", item, res.gdn, b)
    S = torch.cat(surp).double().numpy()                                 # (N, T−1)
    Nx = torch.cat(nxt).double().numpy()
    N, Tm = S.shape
    logpos = np.broadcast_to(np.log1p(np.arange(1, Tm + 1, dtype=np.float64))[None], (N, Tm))
    S_res = S - _nanmean(S, 0)[None]
    per = {}
    for k in SIG_KEYS:
        if not xs[k]:
            continue
        X = torch.cat(xs[k]).double().numpy()                            # (N, T−1, nL, H)
        nL, H = X.shape[2:]
        flat = X.reshape(N * Tm, nL * H)
        Xr = (X - _nanmean(X, 0)[None]).reshape(N * Tm, nL * H)
        doc_means = _nanmean(X, 1)                                       # (N, nL, H)
        hm = _nanmean(X, 3)                                              # head mean (N, T−1, nL)
        sh = lambda v: np.asarray(v).reshape(nL, H)                      # noqa: E731
        per[k] = {"mean": sh(_nanmean(flat, 0)), "sd_tokens": sh(_nanstd(flat, 0)),
                  "sd_within_doc": _nanmean(_nanstd(X, 1), 0), "sd_between_docs": _nanstd(doc_means, 0),
                  "sd_head_mean": _nanstd(hm.reshape(-1, nL), 0),
                  "r_surprisal": sh(_pearson(flat, S.reshape(-1))),
                  "rho_surprisal": sh(_spearman(flat, S.reshape(-1))),
                  "r_next_loss": sh(_pearson(flat, Nx.reshape(-1))),
                  "r_logpos": sh(_pearson(flat, logpos.reshape(-1))),
                  "rho_logpos": sh(_spearman(flat, logpos.reshape(-1))),
                  "r_surprisal_pos_removed": sh(_pearson(Xr, S_res.reshape(-1))),
                  "frac_heads_sd_lt_0.02": float((_nanstd(flat, 0) < 0.02).mean())}
    rows = {}
    for j, l in enumerate(lin):
        rows[str(l)] = {}
        for k in ("beta", "ret", "fam", "fam_lag"):
            if k in per:
                rows[str(l)][f"{k}_sd"] = float(np.median(per[k]["sd_tokens"][j]))
                rows[str(l)][f"{k}_r_surp"] = float(np.nanmedian(per[k]["r_surprisal"][j]))
        if "fam" in per:
            rows[str(l)]["fam_r_pos"] = float(np.nanmedian(per["fam"]["r_logpos"][j]))
    print_table("signals: per-layer head medians (SD over tokens; r with own surprisal / log position)", rows)
    probes = d_probes(E, W, rng, taken)
    return {"settings": {"n": a.sig_n, "len": T, "keys": list(per), "layers": lin, "probe_n": a.probe_n,
                         "probe_len": a.probe_len, "probe_pre": a.probe_pre, "probe_dists": _ints(a.probe_dists)},
            "natural": {"n_tokens": int(N * Tm), "per_head": per}, "probes": probes, "trace": W.close()}


PROBE_CONDS = ("exact", "related", "shuffled", "unrelated")
PROBE_KEYS = ("fam", "fam_lag", "err", "err_lag", "cos_vp", "cos_vp_lag", "ratio", "ratio_lag", "str", "beta", "ret")


def d_probes(E: Env, W: TraceWriter, rng: random.Random, taken: np.ndarray) -> dict:
    """Mismatch on an exact repeat, a related passage, a shuffled passage and an unrelated one (see docstring)."""
    a = E.args
    n, lin = a.probe_len, E.lin
    dists = _ints(a.probe_dists)
    gap = max(4, n // 2)
    docs = E.pick(2 * n + gap, a.probe_n, rng, taken)                   # X and the related passage
    others = E.pick(max(dists) + a.probe_pre + 3 * n, a.probe_n, rng, taken)   # unrelated, baseline, filler, pre
    acc = {(c, d): {k: [] for k in PROBE_KEYS + ("loss",)} for c in PROBE_CONDS for d in dists}
    acc4 = {(c, d): {k: [] for k in PROBE_KEYS + ("loss",)} for c in PROBE_CONDS for d in dists}
    for i in range(a.probe_n):
        X = E.seg(docs[i], n)
        rel = E.seg(docs[i] + n + gap, n)
        g = torch.Generator().manual_seed(a.seed * 7919 + i)
        shuf = X[torch.randperm(n, generator=g)]
        o = others[i]
        unrel, base = E.seg(o, n), E.seg(o + n, n)
        pre = E.seg(o + 2 * n, a.probe_pre)
        for d in dists:
            fill = E.seg(o + 2 * n + a.probe_pre, d)
            probes = {"exact": X, "related": rel, "shuffled": shuf, "unrelated": unrel}
            rows = []
            for c in PROBE_CONDS:                                         # E = X; the unrelated probe never matches it
                rows.append(torch.cat([torch.tensor([E.sink]), pre, X, fill, probes[c]]))
                rows.append(torch.cat([torch.tensor([E.sink]), pre, base, fill, probes[c]]))
            ids = torch.stack(rows)
            res = run_arm(E, ids, gain=0.0)
            out = outcomes(E.model, res.hf, next_targets(ids))
            surp = own_surprisal(out["loss"])
            p0 = ids.shape[1] - n
            for j, c in enumerate(PROBE_CONDS):
                w, bse = 2 * j, 2 * j + 1
                for k in PROBE_KEYS:
                    x = head_stack(res.gdn, lin, k)
                    if x is None:
                        continue
                    pr = x[:, p0:].cpu().double()                         # (rows, n, nL, H)
                    dlt = (pr[w] - pr[bse])
                    acc[(c, d)][k].append(_nanmean(dlt.numpy(), 0))        # (nL, H)
                    acc4[(c, d)][k].append(_nanmean(dlt[4:].numpy(), 0) if n > 4 else _nanmean(dlt.numpy(), 0))
                dl = (surp[w, p0:] - surp[bse, p0:]).double().numpy()
                acc[(c, d)]["loss"].append(np.asarray(np.nanmean(dl)))
                acc4[(c, d)]["loss"].append(np.asarray(np.nanmean(dl[4:] if n > 4 else dl)))
                item = {"item": i, "cond": c, "dist": d, "probe_start": p0}
                for row, arm in ((w, f"probe.{c}.with"), (bse, f"probe.{c}.base")):
                    W.add(f"{arm}.d{d}", item, trace_tokens(res.trace, row, {f"out/{k}": v for k, v in out.items()}
                                                            | {"out/surp": surp}))
    out = {}
    for (c, d), dd in acc.items():
        key = f"{c}.d{d}"
        out[key] = {}
        for k, v in dd.items():
            if not v:
                continue
            arr = np.stack(v)                                            # (items, nL, H) or (items,)
            if arr.ndim == 1:
                out[key][k] = {"all": _boot(arr, seed=a.seed), "from4": _boot(np.stack(acc4[(c, d)][k]), seed=a.seed)}
                continue
            lm = _nanmean(arr, 2)                                        # (items, nL) head means
            lm4 = _nanmean(np.stack(acc4[(c, d)][k]), 2)
            out[key][k] = {"layer_boot": [_boot(lm[:, j], seed=a.seed) for j in range(len(lin))],
                           "layer_boot_from4": [_boot(lm4[:, j], seed=a.seed) for j in range(len(lin))],
                           "head_mean_delta": _nanmean(arr, 0), "all_layers": _boot(_nanmean(lm, 1), seed=a.seed)}
    rows = {key: {k: v["all_layers"]["diff"] for k, v in vv.items()
                  if k in ("fam", "fam_lag", "err") and "all_layers" in v} | {"dsurp": vv["loss"]["all"]["diff"]}
            for key, vv in out.items()}
    print_table("signals probes: Δ(with − baseline) over probe tokens, mean over layers", rows)
    return out


# --------------------------------------------------------------------------------------------- gate_shift
def _blogit(beta: torch.Tensor) -> torch.Tensor:
    b = beta.clamp(1e-6, 1 - 1e-6)
    return torch.log(b) - torch.log1p(-b)


def _gs_get(st: dict, lin, k: str) -> torch.Tensor | None:
    if k == "blogit":
        x = head_stack(st, lin, "beta")
        return None if x is None else _blogit(x)
    return head_stack(st, lin, k)


def d_gate_shift(E: Env) -> dict:
    a = E.args
    if not E.lin:
        raise ValueError("gate_shift needs GDN layers")
    rng = random.Random(a.seed * 1000 + 6)
    starts = E.pick(a.gs_len, a.gs_n, rng)
    W = E.writer("gate_shift")
    lin = E.lin
    above = [j for j, l in enumerate(lin) if l > E.L]
    below = [j for j, l in enumerate(lin) if l <= E.L]
    arms = [(d, g) for d in a.gs_dirs.split(",") for g in _floats(a.gs_gains)]
    nat = {k: [0.0, 0.0, 0.0] for k in GS_KEYS}                          # n, sum, sumsq per (nL, H)
    acc = {f"{d}_g{g}": {k: [0.0, 0.0, 0.0] for k in GS_KEYS} for d, g in arms}
    low = {f"{d}_g{g}": 0.0 for d, g in arms}
    per_item = {f"{d}_g{g}": {"dloss": [], "kl": [], "norm_ratio": [], "inj": [], "g": [], "steer": []}
                for d, g in arms}
    t0 = a.gs_from
    for bi, ch in enumerate(_chunks(starts, a.batch)):
        ids = E.rows(ch, a.gs_len)
        base = run_arm(E, ids, gain=0.0, steer_from=t0)
        tg = next_targets(ids)
        ob = outcomes(E.model, base.hf, tg)
        bst = {k: _gs_get(base.gdn, lin, k) for k in GS_KEYS}
        for k, x in bst.items():
            if x is None:
                continue
            v = x[:, t0:].double()
            ok = torch.isfinite(v)
            v0 = torch.where(ok, v, torch.zeros_like(v))
            nat[k][0] = nat[k][0] + ok.sum((0, 1)).cpu()
            nat[k][1] = nat[k][1] + v0.sum((0, 1)).cpu()
            nat[k][2] = nat[k][2] + (v0 * v0).sum((0, 1)).cpu()
        for b in range(len(ch)):
            W.add("base", {"doc": bi * a.batch + b, "start": int(ch[b])},
                  trace_tokens(base.trace, b, {f"out/{k}": v for k, v in ob.items()}))
        nb = base.trace["tap/norms/norm"]
        for d, g in arms:
            key = f"{d}_g{g}"
            res = run_arm(E, ids, direction=d, gain=g, steer_from=t0)
            o = outcomes(E.model, res.hf, tg, ref=base.hf)
            for k in GS_KEYS:
                x = _gs_get(res.gdn, lin, k)
                if x is None or bst[k] is None:
                    continue
                dlt = (x - bst[k]).double()
                if below:
                    low[key] = max(low[key], float(torch.nan_to_num(dlt[:, :, below]).abs().max()))
                v = dlt[:, t0:]
                ok = torch.isfinite(v)
                v0 = torch.where(ok, v, torch.zeros_like(v))
                acc[key][k][0] = acc[key][k][0] + ok.sum((0, 1)).cpu()
                acc[key][k][1] = acc[key][k][1] + v0.abs().sum((0, 1)).cpu()
                acc[key][k][2] = acc[key][k][2] + v0.sum((0, 1)).cpu()
            nr = (res.trace["tap/norms/norm"][:, t0:] / nb[:, t0:].clamp_min(1e-30)).mean(1)   # (B, n_blocks)
            for b in range(len(ch)):
                pi = per_item[key]
                pi["dloss"].append(float(_nanmean((o["loss"][b, t0:-1] - ob["loss"][b, t0:-1]).numpy())))
                pi["kl"].append(float(_nanmean(o["kl"][b, t0:-1].numpy())))
                pi["norm_ratio"].append(nr[b].numpy())
                pi["inj"].append(float(res.trace["inj_ratio"][b, t0:].mean()))
                pi["g"].append(res.trace["g_raw"][b])
                pi["steer"].append(res.trace["steer"][b])
                W.add(key, {"doc": bi * a.batch + b, "start": int(ch[b]), "direction": d, "gain": g},
                      trace_tokens(res.trace, b, {f"out/{k}": v for k, v in o.items()}))
    sd = {}
    for k, (n, s1, s2) in nat.items():
        if isinstance(n, float):
            continue
        n = n.clamp_min(1)
        mu = s1 / n
        sd[k] = (s2 / n - mu * mu).clamp_min(0).sqrt().numpy()             # (nL, H)
    out_arms, rows = {}, {}
    for d, g in arms:
        key = f"{d}_g{g}"
        pa = {"direction": d, "gain": g, "shift": {}}
        for k in GS_KEYS:
            n, sabs, ssig = acc[key][k]
            if isinstance(n, float) or k not in sd:
                continue
            n = n.clamp_min(1)
            z = (sabs / n).numpy() / np.maximum(sd[k], 1e-12)
            zs = (ssig / n).numpy() / np.maximum(sd[k], 1e-12)
            pa["shift"][k] = {"abs_over_sd": z, "signed_over_sd": zs,
                              "layer_head_median": [float(np.median(z[j])) for j in range(len(lin))],
                              "layer_head_max": [float(z[j].max()) for j in range(len(lin))],
                              "frac_heads_ge_0.5sd_above_L": float((z[above] >= 0.5).mean()) if above else None}
        pi = per_item[key]
        pa["max_abs_delta_at_or_below_L"] = low[key]
        pa["dloss"] = _boot(pi["dloss"], seed=a.seed)
        pa["kl"] = _boot(pi["kl"], seed=a.seed)
        pa["inj_ratio"] = float(np.mean(pi["inj"]))
        pa["norm_ratio_by_block"] = np.mean(np.stack(pi["norm_ratio"]), 0)
        pa["gain"] = gain_stats(torch.stack(pi["g"]), torch.stack(pi["steer"]), 0.0, g)
        out_arms[key] = pa
        if "beta" in pa["shift"]:
            med = lambda k: float(np.median(pa["shift"][k]["abs_over_sd"][above])) if above else None   # noqa: E731
            rows[key] = {"dbeta/sd_med": med("beta"), "dlogret/sd_med": med("logret"),
                         "frac>=.5sd": pa["shift"]["beta"]["frac_heads_ge_0.5sd_above_L"],
                         "kl": pa["kl"]["diff"], "dloss": pa["dloss"]["diff"], "low_max": low[key]}
    print_table(f"gate_shift at block {E.L} output (GDN layers above: median over heads of |Δ|/SD)", rows)
    return {"settings": {"n": a.gs_n, "len": a.gs_len, "from": t0, "layer": E.L, "arms": [f"{d}_g{g}" for d, g in arms],
                         "layers": lin, "layers_above": [lin[j] for j in above]},
            "natural_sd": sd, "arms": out_arms, "trace": W.close()}


# ------------------------------------------------------------------------------------------------ impulse
def d_impulse(E: Env, diag: str = "impulse") -> dict:
    a = E.args
    rng = random.Random(a.seed * 1000 + 7)
    bursts, gains, dirs = _ints(a.imp_bursts), _floats(a.imp_gains), a.imp_dirs.split(",")
    t0 = a.imp_t0
    T = t0 + max(bursts) + a.imp_lag + 1                                 # columns after the sink
    starts = E.pick(T, a.imp_n, rng)
    W = E.writer(diag)
    n = len(E.blocks)
    keep = sorted({E.L, min(E.L + 1, n - 1), (E.L + n) // 2, n - 1})
    arms = [(d, g, nb) for d in dirs for g in gains for nb in bursts]
    gk = IMP_GDN_KEYS if E.lin else ()
    rec = {f"{d}_g{g}_b{nb}": {"kl": [], "dloss": [], "agree": [], "pre_kl_max": [], "g": [], "steer": [],
                              **{f"h{i}": [] for i in keep}, **{f"gdn_{k}": [] for k in gk}} for d, g, nb in arms}
    for bi, ch in enumerate(_chunks(starts, a.batch)):
        ids = E.rows(ch, T)
        tg = next_targets(ids)
        base = run_arm(E, ids, keep=keep, gain=0.0, steer_from=t0)
        ob = outcomes(E.model, base.hf, tg)
        for b in range(len(ch)):
            W.add("base", {"doc": bi * a.batch + b, "start": int(ch[b])},
                  trace_tokens(base.trace, b, {f"out/{k}": v for k, v in ob.items()}))
        for d, g, nb in arms:
            key = f"{d}_g{g}_b{nb}"
            res = run_arm(E, ids, keep=keep, direction=d, pulse=(t0, nb), gain=g)
            o = outcomes(E.model, res.hf, tg, ref=base.hf)
            r = rec[key]
            r["kl"].append(o["kl"])
            r["dloss"].append(o["loss"] - ob["loss"])
            r["agree"].append(o["agree"])
            r["pre_kl_max"].append(o["kl"][:, :t0].abs().amax(1))
            r["g"].append(res.trace["g_raw"])
            r["steer"].append(res.trace["steer"])
            hd = {}
            for i in keep:
                hb = base.blocks[i].float()
                hd[i] = ((res.blocks[i].float() - hb).norm(dim=-1) / hb.norm(dim=-1).clamp_min(1e-30)).cpu()
                r[f"h{i}"].append(hd[i])
            for k in gk:
                x, y = res.trace.get(f"tap/gdn/{k}"), base.trace.get(f"tap/gdn/{k}")
                if x is not None and y is not None:
                    r[f"gdn_{k}"].append((x - y).abs())
            for b in range(len(ch)):
                extra = {f"out/{k}": v for k, v in o.items()}
                extra.update({f"hdiff/b{i}": hd[i] for i in keep})
                W.add(key, {"doc": bi * a.batch + b, "start": int(ch[b]), "direction": d, "gain": g, "burst": nb,
                            "t0": t0}, trace_tokens(res.trace, b, extra))
    out_arms, rows = {}, {}
    for d, g, nb in arms:
        key = f"{d}_g{g}_b{nb}"
        r = rec[key]
        t_end = t0 + nb - 1
        bins = lag_bins(T - t_end)
        res_ = {"direction": d, "gain": g, "burst": nb, "t0": t0, "t_end": t_end,
                "pre_pulse_kl_max": float(torch.cat(r["pre_kl_max"]).max()),
                "gain": gain_stats(torch.cat(r["g"]), torch.cat(r["steer"]), 0.0, g),
                "bins": [_bin_label(b) for b in bins]}
        series = {"kl": r["kl"], "dloss": r["dloss"], "agree": r["agree"], **{f"hdiff_b{i}": r[f"h{i}"] for i in keep}}
        for name, v in series.items():
            x = torch.cat(v).double().numpy()                            # (N, T+1) columns incl. the sink
            res_[name] = {_bin_label(b): _boot(_nanmean(x[:, t_end + b[0]: t_end + b[1]], 1), seed=a.seed)
                          for b in bins if t_end + b[0] < x.shape[1]}
            if nb > 1:
                res_[name]["in_pulse"] = _boot(_nanmean(x[:, t0:t_end], 1), seed=a.seed)
        for k in gk:
            if r[f"gdn_{k}"]:
                x = torch.cat(r[f"gdn_{k}"]).double().numpy()            # (N, T+1, nL)
                res_[f"gdn_abs_d_{k}"] = {_bin_label(b): _nanmean(x[:, t_end + b[0]: t_end + b[1]], (0, 1))
                                          for b in bins if t_end + b[0] < x.shape[1]}
        klm = {lb: v["diff"] for lb, v in res_["kl"].items() if lb != "in_pulse" and v["diff"] is not None}
        k0 = klm.get("0")
        for frac in (0.1, 0.01):
            res_[f"first_bin_below_{frac}_of_lag0"] = next((lb for lb, v in klm.items() if k0 and v < frac * k0), None)
        out_arms[key] = res_
        rows[key] = {f"kl@{lb}": klm.get(lb) for lb in ("0", "1", "4-7", "64-127", "1024-2047")}
        rows[key].update({"kl@last": klm[list(klm)[-1]] if klm else None,
                          "below_1%": res_["first_bin_below_0.01_of_lag0"], "pre_max": res_["pre_pulse_kl_max"]})
    print_table(f"{diag}: KL(steered ‖ base) by lag after the pulse ({E.facts['name']}, block {E.L})", rows)
    return {"settings": {"model": E.facts["name"], "n": a.imp_n, "t0": t0, "max_lag": a.imp_lag, "T": T + 1,
                         "layer": E.L, "keep_blocks": keep, "bursts": bursts, "gains": gains, "dirs": dirs,
                         "gdn_signals": list(gk)}, "arms": out_arms, "trace": W.close()}


# ---------------------------------------------------------------------------------------------- printing
def print_table(title: str, rows: dict) -> None:
    print(f"\n== {title}")
    if not rows:
        print("  (empty)")
        return
    cols = []
    for r in rows.values():
        cols += [c for c in r if c not in cols]
    fmt = lambda v: ("—" if v is None else f"{v:.4g}" if isinstance(v, float) else str(v))   # noqa: E731
    w0 = max(len(str(k)) for k in rows) + 2
    print(" " * w0 + "".join(f"{c:>18}" for c in cols))
    for k, r in rows.items():
        print(f"{str(k):<{w0}}" + "".join(f"{fmt(r.get(c)):>18}" for c in cols))


# --------------------------------------------------------------------------------------------------- main
RUNNERS = {"census": d_census, "attenuation": d_attenuation, "recurrent_only": d_recurrent_only,
           "state_swap": d_state_swap, "signals": d_signals, "gate_shift": d_gate_shift, "impulse": d_impulse}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    p.add_argument("--compare-model", default="Qwen/Qwen3-1.7B-Base", help="impulse_cmp model; '' skips it")
    p.add_argument("--dtype", default=None, help="default bfloat16 (float32 with --tiny)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--tiny", action="store_true", help="random tiny models on CPU (torch GDN kernels)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diag", default="all", help=f"all or a comma list of {DIAGS}")
    p.add_argument("--out", default=None, help="results JSON (default results/evolve/round12/stage0/hybrid_diag.json)")
    p.add_argument("--trace-dir", default=None, help="default ~/limbic-data/ca2_traces/<out stem>; outside the repo")
    p.add_argument("--force", action="store_true", help="rerun diagnostics already complete in --out")
    p.add_argument("--text-file", default="~/limbic-data/wikitext103_test.txt")
    p.add_argument("--layer", type=int, default=None, help="injection / gist block L of the main model (default n//2)")
    p.add_argument("--cmp-layer", type=int, default=None, help="block L of the comparator (default n//2)")
    p.add_argument("--no-mismatch", action="store_true", help="skip the per-token GDN loop (gates only)")
    p.add_argument("--attn-window", type=int, default=32)
    p.add_argument("--per-head-keys", default="beta,ret,fam,fam_lag,err,err_lag,str,cos_vp,read,snorm")
    for k, v in PRESETS["real"].items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=None, help=f"default {v} (tiny: "
                       f"{PRESETS['tiny'][k]})")
    a = p.parse_args(argv)
    pre = PRESETS["tiny" if a.tiny else "real"]
    for k, v in pre.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.dtype is None:
        a.dtype = "float32" if a.tiny else "bfloat16"
    if a.tiny and a.layer is None:
        a.layer = 1                          # tiny hybrid [GDN, attn, GDN, attn]: block 1 leaves GDN block 2 above it
    if a.out is None:
        a.out = str(ROOT / "results" / "evolve" / "round12" / "stage0" / ("hybrid_diag_tiny.json" if a.tiny
                                                                           else "hybrid_diag.json"))
    if a.trace_dir is None:
        a.trace_dir = str(Path("~/limbic-data/ca2_traces").expanduser() / Path(a.out).stem)
    a.diags = list(DIAGS) if a.diag == "all" else [d.strip() for d in a.diag.split(",") if d.strip()]
    bad = [d for d in a.diags if d not in DIAGS]
    if bad:
        p.error(f"unknown diagnostics {bad}; choose from {DIAGS}")
    if not a.compare_model and "impulse_cmp" in a.diags and a.diag == "all":
        a.diags.remove("impulse_cmp")
    return a


COMPAT = ("model", "compare_model", "dtype", "tiny", "seed", "text_file", "layer", "cmp_layer", "no_mismatch",
          "attn_window", "per_head_keys") + tuple(k for k in PRESETS["real"] if k != "batch")


def compat_of(a) -> dict:
    """The settings a diagnostic's numbers depend on (stored in every block; resume refuses a mismatch)."""
    return {k: getattr(a, k) for k in COMPAT}


def stale_diags(res: dict, compat: dict) -> dict:
    """Complete diagnostics in ``res`` produced with other settings: {diag: {key: (stored, now)}}. A block
    without its own ``run_settings`` (older files) is judged by the header ``args`` it was written under."""
    out = {}
    for d, blk in res.get("diags", {}).items():
        if not blk.get("complete"):
            continue
        old = blk.get("run_settings") or {k: v for k, v in res.get("args", {}).items() if k in compat}
        diff = {k: (old[k], v) for k, v in compat.items() if k in old and old[k] != v}
        if diff:
            out[d] = diff
    return out


def main(argv=None) -> dict:
    a = parse_args(argv)
    tdir = alignment().judge.assert_private(Path(a.trace_dir), ROOT)       # refuses a path inside the repo
    a.trace_dir = str(tdir)
    torch.manual_seed(a.seed)
    out_path = Path(a.out)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    res.setdefault("diags", {})
    compat = compat_of(a)
    stale = stale_diags(res, compat)
    if stale and not a.force:
        raise SystemExit(f"{out_path} holds diagnostics run with other settings {stale}; use another --out, "
                         "or --force (reruns --diag; each block keeps its own 'run_settings')")
    for blk in res["diags"].values():                    # pin older blocks to the header they were written under
        if blk.get("complete") and "run_settings" not in blk:
            blk["run_settings"] = {k: v for k, v in res.get("args", {}).items() if k in compat}
    res.update(args=vars(a), git=git_state(), versions=_versions(), command=" ".join(sys.argv))
    kern = None
    if a.tiny or a.device == "cpu" or not torch.cuda.is_available():
        kern = use_torch_gdn_kernels()
    res["gdn_kernels"] = kern or "default dispatch (FLA Triton on GPU)"
    todo = [d for d in a.diags if a.force or not res["diags"].get(d, {}).get("complete")]
    for d in a.diags:
        if d not in todo:
            print(f"[skip] {d}: complete in {out_path} (use --force to rerun)")
    groups = [("main", [d for d in todo if d != "impulse_cmp"]),
              ("comparator", [d for d in todo if d == "impulse_cmp"])]
    for which, diags in groups:
        if not diags:
            continue
        model, tok, blocks, kinds, facts = load_any(a, "main" if which == "main" else "cmp")
        facts["which"] = "main" if which == "main" else "comparator"
        res["model" if which == "main" else "comparator"] = facts
        text = load_test_text(a.text_file, tok) if Path(a.text_file).expanduser().exists() else _fallback_text(tok)
        E = Env(a, model, tok, blocks, kinds, facts, text)
        print(f"[{which}] {facts['name']}: {len(blocks)} blocks, GDN layers {E.lin}, L = {E.L}, "
              f"{len(text)} text tokens, device {E.dev}")
        for d in diags:
            t = time.time()
            print(f"\n[run] {d}")
            fn = d_impulse if d == "impulse_cmp" else RUNNERS[d]
            block = fn(E, "impulse_cmp") if d == "impulse_cmp" else fn(E)
            block.update(complete=True, seconds=round(time.time() - t, 2), model=facts["name"],
                         finished=time.strftime("%Y-%m-%d %H:%M:%S"), run_settings=compat)
            res["diags"][d] = block
            _write_json(out_path, res)
            print(f"[done] {d} in {block['seconds']} s -> {out_path}")
        del model, E
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_json(out_path, res)
    return res


def _fallback_text(tok) -> torch.Tensor:
    """When --text-file is missing: the repository's Markdown records as text (deterministic)."""
    parts = [p.read_text(encoding="utf-8", errors="replace") for p in sorted(ROOT.glob("*.md"))]
    parts += [p.read_text(encoding="utf-8", errors="replace") for p in sorted((ROOT / "research").glob("*.md"))]
    return torch.tensor(tok.encode("\n".join(parts)))


if __name__ == "__main__":
    main()
