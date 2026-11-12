"""Shared measurement pieces of the round-12 runners (``text.py``, ``ceiling.py``) and ``analyse.py``.

Nothing here is trained; everything is read from the frozen model's own forward passes.

Outcome per token (``token_metrics``; logits z_t over the vocabulary, target y = x_{t+1}, chunked over T so
that no (B, T, V) float32 copy is ever held):
    loss_t   = logsumexp(z_t) − z_t[y]                     next-token cross-entropy (nats)
    ent_t    = −Σ_v p_t(v) log p_t(v)                       predictive entropy
    rank_t   = #{v : z_t[v] > z_t[y]}                       0 ⇔ top-1 correct (ties count as correct)
    kl_<r>_t = Σ_v p_t(v) (log p_t(v) − log q_t(v))         KL(arm ‖ reference r), r = the unsteered model in
                                                            the same condition ("base") or the full-context
                                                            model ("full")
    lp_k     = log p_t(c_k)                                  log-probability of candidate ids c_k (contiguity)

Logit lens (``LensCapture``): the residual stream at a few depths, through the model's own final norm and
unembedding, gives the rank of the eventual (target) token at each depth.

Contiguity (``contiguity``; Kahana's lag-CRP transposed to teacher forcing). At position t whose token x_t
occurred before, j = its most recent earlier position. The candidates are the tokens around that earlier
occurrence, c_{+1} = x_{j+1}, c_{+2} = x_{j+2}, c_{−1} = x_{j−1}, c_{−2} = x_{j−2} (only from positions ≤ t),
and a control c_0 = x_{r+1} for a seeded random earlier r with x_r ≠ x_t. The contiguity boost in direction k
is lp_k − lp_0; lag = t − j; ``inblock`` says whether j lies in t's attention window block (windowed runs:
anything outside it can only arrive through the GDN state or the side channel); ``hit`` = [x_{j+1} = y].

Gist drift (``GistDriftSource``, a non-driving source, so it is traced in every arm): with the bank's raw
gist γ_t = A_fast(t) − A_slow(t), records ‖γ_t‖, cos(γ_t, h_t) and cos(γ_t, γ_{t−k}) for k in ``lags``
(a ring of the last max(lags) unit gists is carried; NaN where undefined).

Buckets: offset within the window block, o = (p − sink) mod W, in [0, W/8), [W/8, W/2), [W/2, W) (W = 64:
0–7, 8–31, 32–63; W = 512: 0–63, 64–255, 256–511); text position in log₂ bins (0–63, 64–127, 128–255, ...);
lag in log₂ bins (1, 2–3, 4–7, ...).

Records. ``ResultFile`` keeps the results JSON (atomic rewrite, resume by arm key, argument compatibility).
``ArmTraces`` writes one ``common.TraceWriter`` per (condition, arm) under the trace directory (a stale,
incomplete directory of an arm that is re-run is renamed, never deleted), and splits per-token keys into the
main stream (present in every item) and the ``__sub`` stream (the GDN per-token-loop keys and per-head keys,
present only on the subsample that runs the mismatch loop, whose cost is ~56 forward passes).
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
sys.path.insert(0, str(EXP))
import pretrained_loop as pl  # noqa: E402
from ca2 import common  # noqa: E402
from ca2.sources import Source, StepContext  # noqa: E402

RESULTS = ROOT / "results" / "evolve" / "round12"
SMOKE_ROOT = Path("~/limbic-data/ca2_smoke")
DIRECTION_NAMES = ("gist", "anti_gist", "mean", "self", "random", "foreign", "window_gist")
CONT_DIRS = ("p1", "p2", "m1", "m2", "ctrl")          # c_{+1}, c_{+2}, c_{−1}, c_{−2}, control
CONT_OFFSETS = (1, 2, -1, -2)


# ------------------------------------------------------------------------------------------ environment
def versions() -> dict:
    out = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}
    for mod in ("transformers", "fla", "datasets"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                                   # noqa: BLE001
            out[mod] = None
    out["cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    out["host"] = platform.node()
    out["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return out


def swap_torch_gdn() -> bool:
    """Point the Qwen3.5 GDN kernel names at their pure-torch ``__wrapped__`` originals (the Triton ones of
    flash-linear-attention fail on CPU). Returns True when a swap happened. Used for CPU (--tiny) runs only."""
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as M
    except ImportError:
        return False
    done = False
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        fn = getattr(M, name, None)
        orig = getattr(fn, "__wrapped__", None)
        if orig is not None:
            setattr(M, name, orig)
            done = True
    return done


def build_tiny_qwen3(seed: int = 0, attn_impl: str = "sdpa"):
    """A tiny random Qwen3 (all attention, q/k norms, 4:2 GQA) with the byte tokenizer: the comparator's layout."""
    import transformers
    torch.manual_seed(seed)
    cfg = transformers.Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                   max_position_embeddings=2048, bos_token_id=256, eos_token_id=256,
                                   pad_token_id=256)
    return transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation=attn_impl), pl.ByteTokenizer()


def load_model(args):
    """(model, tok, blocks, kinds). ``--tiny`` builds a random model (gpt2, qwen35, qwen3) on the CPU unless
    ``--device cuda``; the pure-torch GDN kernels are selected on the CPU."""
    if args.tiny:
        dev = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        if dev == "cpu" and args.tiny_arch == "qwen35":
            swap_torch_gdn()
        if args.tiny_arch == "qwen3":
            model, tok = build_tiny_qwen3(args.seed, args.attn_impl)
        else:
            model, tok = pl.build_tiny(args.tiny_arch, args.seed, args.attn_impl)
        model = model.to(dev, pl.DTYPES[args.dtype]).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        blocks = pl.find_blocks(model)
        return model, tok, blocks, pl.layer_kinds(model, blocks)
    return common.load(args.model, args.dtype, args.device, False, "gpt2", args.seed, args.attn_impl)


def model_facts(model: nn.Module, tok, blocks, kinds, name: str | None) -> dict:
    cfg = pl.text_config(model)
    hf = getattr(tok, "tok", None)
    eos = []
    if hf is not None and getattr(hf, "chat_template", None) and name and "base" not in name.lower():
        eos = [i for i in (hf.convert_tokens_to_ids("<|im_end|>"), hf.convert_tokens_to_ids("<|endoftext|>"))
               if isinstance(i, int) and i >= 0]
    if not eos and getattr(tok, "eos_token_id", None) is not None:
        eos = [int(tok.eos_token_id)]
    return {"name": name, "n_blocks": len(blocks), "kinds": list(kinds),
            "attention_blocks": [i for i, k in enumerate(kinds) if k == "attention"],
            "linear_blocks": [i for i, k in enumerate(kinds) if k == "linear"],
            "hidden": int(cfg.hidden_size), "vocab": int(model.get_output_embeddings().weight.shape[0]),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "dtype": str(next(model.parameters()).dtype), "device": str(next(model.parameters()).device),
            "bos": getattr(tok, "bos_token_id", None), "eos_ids": eos,
            "class": type(model).__name__}


def is_hybrid(kinds: Sequence[str]) -> bool:
    return "linear" in kinds


def default_name(args, prefix: str) -> str:
    tag = f"tiny-{args.tiny_arch}" if args.tiny else str(args.model).split("/")[-1]
    return f"{prefix}_{tag}"


def resolve_paths(args, prefix: str, stage: str) -> tuple[Path, Path]:
    """(--out, --trace-dir). Real runs write under results/evolve/round12/<stage>/; --tiny runs outside the
    repository (~/limbic-data/ca2_smoke/) unless --out is given. The trace directory must be outside the repo."""
    name = args.name or default_name(args, prefix)
    args.name = common.alignment().judge.safe_name(name)
    if args.out:
        out = Path(args.out).expanduser()
    elif args.tiny:
        out = SMOKE_ROOT.expanduser() / f"{args.name}.json"
    else:
        out = RESULTS / stage / f"{args.name}.json"
    trace = Path(args.trace_dir).expanduser() if args.trace_dir else common.TRACE_ROOT.expanduser() / args.name
    if pl_inside(trace, ROOT):
        raise SystemExit(f"--trace-dir {trace} is inside the repository; traces must live outside it")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out, trace


def pl_inside(p: Path, root: Path) -> bool:
    p, root = Path(p).expanduser().resolve(), Path(root).resolve()
    return p == root or root in p.parents


# ----------------------------------------------------------------------------------------- directions
def parse_direction(spec: str) -> tuple[str, int]:
    """'random:2' -> ('random', 2); others -> (name, 0)."""
    name, _, seed = spec.partition(":")
    if name not in DIRECTION_NAMES:
        raise ValueError(f"unknown direction {spec!r}; known: {DIRECTION_NAMES} (random:<seed>)")
    return name, int(seed or 0)


def safe_arm(name: str) -> str:
    """Arm name usable as a file stem (``judge.safe_name``): ':' and '@' become '-' and '_g'."""
    s = name.replace("@", "_g").replace(":", "-").replace("/", "__")
    return common.alignment().judge.safe_name(s)


# ----------------------------------------------------------------------------------------- model parts
def final_norm(model: nn.Module) -> nn.Module:
    """The final norm before the unembedding (Qwen: model.norm; GPT-2: transformer.ln_f)."""
    base = getattr(model, getattr(model, "base_model_prefix", "model"), None)
    for owner in (base, getattr(base, "language_model", None), model):
        if owner is None:
            continue
        for n in ("norm", "ln_f", "final_layernorm", "final_layer_norm"):
            m = getattr(owner, n, None)
            if isinstance(m, nn.Module):
                return m
    raise ValueError("no final norm found")


def attention_below(kinds: Sequence[str], layer: int) -> int | None:
    """The deepest full-attention block ≤ layer (the attention tap of a site), or None."""
    c = [i for i, k in enumerate(kinds) if k == "attention" and i <= layer]
    return c[-1] if c else None


def attention_above(kinds: Sequence[str], layer: int) -> int | None:
    c = [i for i, k in enumerate(kinds) if k == "attention" and i > layer]
    return c[0] if c else None


def has_qk_norm(blocks, layer: int) -> bool:
    a = pl._child(blocks[layer], pl.ATTN_MIXERS)
    return a is not None and hasattr(a, "q_norm") and hasattr(a, "k_norm")


class LensCapture:
    """Forward hooks on blocks at ``depths`` keeping each block's output (after any injection there)."""

    def __init__(self, blocks, depths: Sequence[int]) -> None:
        self.depths = [int(d) for d in depths]
        self.h: dict[int, torch.Tensor] = {}
        self.handles = [blocks[d].register_forward_hook(self._hook(d)) for d in self.depths]

    def _hook(self, d: int):
        def fn(mod, args, out):
            self.h[d] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    @torch.no_grad()
    def ranks(self, model: nn.Module, targets: torch.Tensor, valid: torch.Tensor,
              chunk_elems: int = 1 << 26) -> dict[int, torch.Tensor]:
        """Rank of ``targets`` (B, T) under the lens at each depth: (B, T) float, NaN where not ``valid``."""
        norm, head = final_norm(model), model.get_output_embeddings()
        out = {}
        for d in self.depths:
            h = self.h[d]
            B, T, _ = h.shape
            V = head.weight.shape[0]
            step = max(1, chunk_elems // max(1, B * V))
            r = torch.empty(B, T, device=h.device)
            for a in range(0, T, step):
                z = head(norm(h[:, a:a + step])).float()
                tl = z.gather(-1, targets[:, a:a + step, None].to(z.device))
                r[:, a:a + step] = (z > tl).sum(-1).float()
            out[d] = torch.where(valid.to(r.device), r, torch.full_like(r, float("nan")))
        return out


class NormCapture:
    """‖h‖ at every block output of a plain (reference) pass: ``norms()`` -> (B, T, n_blocks)."""

    def __init__(self, blocks) -> None:
        self.n = len(blocks)
        self.buf: dict[int, torch.Tensor] = {}
        self.handles = [b.register_forward_hook(self._hook(i)) for i, b in enumerate(blocks)]

    def _hook(self, i: int):
        def fn(mod, args, out):
            self.buf[i] = (out[0] if isinstance(out, tuple) else out).detach().float().norm(dim=-1)
        return fn

    def norms(self) -> torch.Tensor:
        return torch.stack([self.buf[i] for i in range(self.n)], -1)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


class BranchScale:
    """u ← (1 + g)·u on the attention (``self_attn``) or MLP (``mlp``) branch of the given blocks, at model
    columns ≥ ``first`` (the steered positions of an unpadded text run): the controls of the GDN branch gain.
    With g = 0 the hooks return None."""

    def __init__(self, blocks, layers: Sequence[int], target: str, g: float, first: int) -> None:
        if target not in ("attn", "mlp"):
            raise ValueError("target must be 'attn' or 'mlp'")
        if not -1.0 <= g:
            raise ValueError("g must be >= -1")
        self.g, self.first, self.handles = float(g), int(first), []
        for l in layers:
            mod = pl._child(blocks[l], pl.ATTN_MIXERS) if target == "attn" else getattr(blocks[l], "mlp", None)
            if mod is None:
                raise ValueError(f"block {l} has no {target} branch")
            self.handles.append(mod.register_forward_hook(self._hook))

    def _hook(self, mod, args, out):
        if self.g == 0.0:
            return None
        u = out[0] if isinstance(out, tuple) else out
        f = torch.ones(u.shape[1], device=u.device, dtype=torch.float32)
        f[self.first:] = 1.0 + self.g
        y = (u.float() * f[None, :, None]).to(u.dtype)
        return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def gdn_layer_sets(kinds: Sequence[str]) -> dict[str, list[int]]:
    """GDN layer sets of the branch sweep (gdn_design.md §3): ``pre`` (directly below a full-attention block),
    ``band`` (around the middle: 8–10, 12–14 on 24 blocks), ``all``."""
    n = len(kinds)
    lin = [i for i, k in enumerate(kinds) if k == "linear"]
    pre = [i for i in lin if i + 1 < n and kinds[i + 1] == "attention"]
    band = [8, 9, 10, 12, 13, 14] if n == 24 else [i for i in lin if n // 3 <= i <= (2 * n) // 3]
    return {"pre": pre, "band": [i for i in band if i in lin], "all": lin}


def neighbour_attention(kinds: Sequence[str], layers: Sequence[int]) -> list[int]:
    """For each GDN layer, the first full-attention block above it (deduplicated): the attention control."""
    return sorted({a for a in (attention_above(kinds, l) for l in layers) if a is not None})


# -------------------------------------------------------------------------------------------- metrics
@torch.no_grad()
def token_metrics(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor,
                  refs: dict[str, torch.Tensor] | None = None, cand: torch.Tensor | None = None,
                  chunk_elems: int = 1 << 26) -> dict[str, torch.Tensor]:
    """Per-token outcome measures (module docstring), each (B, T) float32 (NaN where not ``valid``); with
    ``cand`` (B, T, K) long (−1 = none) also ``lp`` (B, T, K) = log p(cand) (NaN where cand < 0)."""
    B, T, V = logits.shape
    dev = logits.device
    refs = refs or {}
    step = max(1, chunk_elems // max(1, B * V))
    keys = ["loss", "ent", "rank", "top1", "p_tgt", "p_max"] + [f"kl_{r}" for r in refs]
    out = {k: torch.empty(B, T, device=dev) for k in keys}
    lp_c = None if cand is None else torch.empty(cand.shape, device=dev)
    tg = targets.to(dev).clamp_min(0)
    for a in range(0, T, step):
        sl = slice(a, a + step)
        z = logits[:, sl].float()
        lse = torch.logsumexp(z, -1, keepdim=True)
        lp = z - lse
        p = lp.exp()
        tl = z.gather(-1, tg[:, sl, None])
        out["loss"][:, sl] = (lse - tl)[..., 0]
        out["ent"][:, sl] = -(p * lp).sum(-1)
        rk = (z > tl).sum(-1).float()
        out["rank"][:, sl] = rk
        out["top1"][:, sl] = (rk == 0).float()
        out["p_tgt"][:, sl] = (tl - lse)[..., 0].exp()
        out["p_max"][:, sl] = p.amax(-1)
        for r, ref in refs.items():
            zr = ref[:, sl].float()
            lq = zr - torch.logsumexp(zr, -1, keepdim=True)
            out[f"kl_{r}"][:, sl] = (p * (lp - lq)).sum(-1).clamp_min(0.0)
        if cand is not None:
            c = cand[:, sl].to(dev)
            g = lp.gather(-1, c.clamp_min(0))
            lp_c[:, sl] = torch.where(c >= 0, g, torch.full_like(g, float("nan")))
        del z, lse, lp, p
    v = valid.to(dev)
    nan = torch.tensor(float("nan"), device=dev)
    res = {k: torch.where(v, x, nan) for k, x in out.items()}
    if lp_c is not None:
        res["lp"] = torch.where(v[..., None], lp_c, nan)
    return res


def shifted_targets(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(targets, valid): targets[:, p] = ids[:, p + 1]; the last column has no target."""
    tg = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, :1])], 1)
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[:, -1] = False
    return tg, valid


def contiguity(ids: torch.Tensor, start: int, window: int, sink: int, seed: int, row_ids: Sequence[int]
               ) -> dict[str, torch.Tensor]:
    """Contiguity candidates per position (module docstring), in model coordinates.

    Returns ``cand`` (B, T, 5) long (CONT_DIRS order; −1 where undefined), ``lag`` (B, T) float (NaN: no
    earlier occurrence), ``inblock`` (B, T) float (1: the earlier occurrence lies in the same window block;
    always 1 without a window), ``hit`` (B, T) float ([x_{j+1} = x_{t+1}]; NaN where undefined)."""
    x = ids.cpu().numpy()
    B, T = x.shape
    cand = np.full((B, T, len(CONT_DIRS)), -1, dtype=np.int64)
    lag = np.full((B, T), np.nan, dtype=np.float32)
    inb = np.full((B, T), np.nan, dtype=np.float32)
    hit = np.full((B, T), np.nan, dtype=np.float32)
    for b in range(B):
        rng = np.random.default_rng([int(seed), int(row_ids[b])])
        last: dict[int, int] = {}
        row = x[b]
        for t in range(start, T):
            tok = int(row[t])
            j = last.get(tok)
            if j is not None:
                lag[b, t] = t - j
                for k, off in enumerate(CONT_OFFSETS):
                    q = j + off
                    if start <= q <= t:
                        cand[b, t, k] = row[q]
                for _ in range(8):                          # control: x_{r+1} for a random earlier r, x_r != x_t
                    r = int(rng.integers(start, t))
                    if row[r] != tok:
                        cand[b, t, 4] = row[r + 1]
                        break
                inb[b, t] = 1.0 if not window else float((j - sink) // window == (t - sink) // window)
                if t + 1 < T:
                    hit[b, t] = float(row[j + 1] == row[t + 1])
            last[tok] = t
    return {"cand": torch.from_numpy(cand), "lag": torch.from_numpy(lag), "inblock": torch.from_numpy(inb),
            "hit": torch.from_numpy(hit)}


# --------------------------------------------------------------------------------------------- buckets
def offset_edges(W: int) -> list[int]:
    return [0, max(1, W // 8), max(2, W // 2), W]


def log2_edges(n: int, first: int = 64) -> list[int]:
    """[0, first, 2·first, ...] up to ≥ n (position buckets)."""
    e = [0, first]
    while e[-1] < n:
        e.append(e[-1] * 2)
    return e


def lag_edges(n: int) -> list[int]:
    """[1, 2, 4, ...] up to > n: bin i covers lags [2^i, 2^{i+1})."""
    e = [1]
    while e[-1] <= n:
        e.append(e[-1] * 2)
    return e


def bucketize(x: torch.Tensor, edges: Sequence[int]) -> torch.Tensor:
    """Bucket index of x in [edges[i], edges[i+1]); −1 outside or NaN."""
    e = torch.tensor(list(edges), dtype=torch.float32)
    xf = x.float().contiguous()
    i = torch.bucketize(xf, e, right=True) - 1
    bad = ~torch.isfinite(xf) | (i < 0) | (i >= len(edges) - 1)
    return torch.where(bad, torch.full_like(i, -1), i)


def bucket_sums(v: torch.Tensor, idx: torch.Tensor, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row sums and counts of finite ``v`` (B, T) by bucket ``idx`` (B, T; −1 ignored): (B, n) each."""
    v, idx = v.detach().float().cpu(), idx.cpu()
    ok = torch.isfinite(v) & (idx >= 0)
    s = torch.zeros(v.shape[0], n, dtype=torch.float64)
    c = torch.zeros(v.shape[0], n, dtype=torch.float64)
    s.scatter_add_(1, idx.clamp_min(0), torch.where(ok, v, torch.zeros_like(v)).double())
    c.scatter_add_(1, idx.clamp_min(0), ok.double())
    return s.numpy(), c.numpy()


def masked_sum(v: torch.Tensor, m: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Per-row sum and count of finite v over mask m: (B,), (B,)."""
    v = v.detach().float().cpu()
    ok = torch.isfinite(v) & m.cpu().bool()
    return (torch.where(ok, v, torch.zeros_like(v)).double().sum(1).numpy(), ok.double().sum(1).numpy())


# --------------------------------------------------------------------------------------------- sources
class GistDriftSource(Source):
    """Non-driving source tracing the gist's size and drift (module docstring). ``s`` is 0."""

    def __init__(self, lags: Sequence[int] = (1, 4, 16, 64, 256), name: str = "gist_drift") -> None:
        self.lags, self.name = tuple(int(k) for k in lags), name
        self.reset()

    def reset(self) -> None:
        self.ring: torch.Tensor | None = None            # (B, K, D) earlier unit gists
        self.ring_ok: torch.Tensor | None = None         # (B, K) bool

    def state_numel(self) -> int:
        return sum(0 if t is None else t.numel() for t in (self.ring, self.ring_ok))

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        b = ctx.bank
        raw = b.avg(0) - b.avg(-1)
        gn = raw.norm(dim=-1)
        ok = (b.den[0] > 0) & (b.den[-1] > 0) & (gn > 0) & (ctx.mask > 0)
        g = F.normalize(raw, dim=-1)
        B, T, D = g.shape
        nan = torch.full((B, T), float("nan"), device=g.device)
        out = {"s": torch.zeros(B, T, device=g.device), "gnorm": torch.where(ok, gn, nan),
               "rel": torch.where(ok, gn / ctx.h.norm(dim=-1).clamp_min(1e-12), nan),
               "cos_h": torch.where(ok, F.cosine_similarity(g, ctx.h, dim=-1, eps=1e-12), nan)}
        K = max(self.lags) if self.lags else 0
        if K:
            ring = self.ring if self.ring is not None and self.ring.shape[0] == B else g.new_zeros(B, K, D)
            rok = (self.ring_ok if self.ring_ok is not None and self.ring_ok.shape[0] == B
                   else torch.zeros(B, K, dtype=torch.bool, device=g.device))
            seq, sok = torch.cat([ring, g], 1), torch.cat([rok, ok], 1)
            for k in self.lags:
                c = (g * seq[:, K - k: K - k + T]).sum(-1)
                out[f"lag{k}"] = torch.where(ok & sok[:, K - k: K - k + T], c, nan)
            self.ring, self.ring_ok = seq[:, -K:].detach(), sok[:, -K:]
        return out


class OptionalSource(Source):
    """Wraps a tap source whose tap signal may be absent in a call (the GDN mismatch keys when the per-token
    loop is off): then s = 0, x = NaN, defined = 0, so the trace keys stay the same in every call."""

    def __init__(self, inner: Source) -> None:
        self.inner, self.name = inner, inner.name
        self.needs, self.final = getattr(inner, "needs", ()), False

    def reset(self) -> None:
        self.inner.reset()

    def state_numel(self) -> int:
        return self.inner.state_numel()

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        try:
            return self.inner(ctx)
        except KeyError:
            z = torch.zeros(ctx.mask.shape, device=ctx.mask.device)
            return {"s": z, "x": torch.full_like(z, float("nan")), "defined": z.clone()}


# --------------------------------------------------------------------------------------- signal set-up
def gdn_mod():
    from ca2 import gdn
    return gdn


def attention_mod():
    from ca2 import attention
    return attention


def register_signals(ctrl, blocks, kinds, site: int, attn_window: int, gdn_mismatch: bool,
                     per_head: Sequence[str] = (), extra_taps: Sequence = (), branch_law=None,
                     norms: bool = True) -> list:
    """Every relevance signal in every arm (plan, "Measurement policy"): gist drift, the attention tap at the
    deepest full-attention block ≤ the site with both attention sources, the GDN taps on a hybrid with every
    GDN source (mismatch-based ones wrapped in ``OptionalSource``), and ``NormTap``. Returns the taps to
    attach (through ``SteerSide`` or ``CA2Attachment``)."""
    from ca2.controller import NormTap
    taps = []
    if "gist_drift" not in ctrl.sources:
        ctrl.add_source(GistDriftSource())
    a = attention_below(kinds, site)
    if a is not None and has_qk_norm(blocks, a):
        taps.append(attention_mod().add_attention(ctrl, layer=a, window=attn_window))
    if is_hybrid(kinds):
        G = gdn_mod()
        tap = G.GDNTaps(mismatch=gdn_mismatch, per_head=list(per_head), law=branch_law)
        taps.append(tap)
        for n, make in G.GDN_SOURCES.items():
            if n in ctrl.sources:
                continue
            src = make(tap="gdn")
            if not any(l < site + 1 for l, k in enumerate(kinds) if k == "linear"):
                continue
            ctrl.add_source(src if src.signal in G.GATE_KEYS else OptionalSource(src))
    if norms:
        taps.append(NormTap())
    taps.extend(extra_taps)
    return taps


def extra_attention_taps(blocks, kinds, site: int, window: int) -> list:
    """Attention statistics at the first full-attention block above the site (``attn_up``) and at the last
    one (``attn_last``): the ceiling's attention-entropy measures, traced in every arm."""
    A = attention_mod()
    out, seen = [], set()
    for name, l in (("attn_up", attention_above(kinds, site)),
                    ("attn_last", max([i for i, k in enumerate(kinds) if k == "attention"], default=None))):
        if l is None or l in seen or l <= site or not has_qk_norm(blocks, l):
            continue
        seen.add(l)
        out.append(A.AttnTap(layer=l, window=window, name=name))
    return out


def is_sub_key(key: str) -> bool:
    """Per-token keys that exist only on the GDN-loop subsample (the ``__sub`` trace stream)."""
    if key.endswith("_head"):
        return True
    if key.startswith("tap/gdn/"):
        G = gdn_mod()
        return key.split("/", 2)[2] not in set(G.GATE_KEYS) | set(G.CALL_KEYS)
    return False


# ---------------------------------------------------------------------------------------------- records
def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def clean(o):
    """Recursively replace NaN/inf floats by None (strict JSON) and arrays by lists."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, torch.Tensor):
        return clean(o.tolist())
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


class ResultFile:
    """The results JSON: header (args, git, versions, model, task) and ``arms`` keyed '<condition>/<arm>'.

    Resume: arms with ``complete`` are skipped unless ``force``; the header fields in ``compat`` must equal
    the stored ones (else SystemExit, so that a resumed file never mixes settings). Rewritten atomically."""

    def __init__(self, path: Path, header: dict, compat: dict, force: bool = False) -> None:
        self.path = Path(path)
        self.data = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            old = self.data.get("compat", {})
            diff = {k: (old.get(k), v) for k, v in clean(compat).items() if k in old and old.get(k) != v}
            if diff and not force:
                raise SystemExit(f"{self.path} was written with other settings {diff}; use --force or a new --out")
        self.data.update(clean(header))
        self.data["compat"] = clean(compat)
        self.data.setdefault("arms", {})
        if force:
            for v in self.data["arms"].values():
                v["complete"] = False
        self.force = force
        self.save()

    def done(self, key: str) -> bool:
        a = self.data["arms"].get(key)
        return bool(a and a.get("complete"))

    def put(self, key: str, rec: dict) -> None:
        self.data["arms"][key] = clean(rec)
        self.save()

    def save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(clean(self.data), indent=1, default=_jsonable))
        tmp.replace(self.path)


class ArmTraces:
    """One ``TraceWriter`` per (condition, arm) under ``root``; main and ``__sub`` streams (module docstring).
    ``max_items`` limits how many items get per-token arrays (−1: all); item records are always written."""

    def __init__(self, root: Path, cond: str, arm: str, meta: dict, max_items: int = -1) -> None:
        run = safe_arm(f"{cond}__{arm}")
        d = Path(root).expanduser() / run
        if d.exists():                                      # an incomplete earlier attempt: keep it, renamed
            k = 1
            while d.with_name(f"{run}.stale{k}").exists():
                k += 1
            d.rename(d.with_name(f"{run}.stale{k}"))
        self.w = common.TraceWriter(run, root=root, meta=meta)
        self.arm = "tokens"
        self.max_items, self.n = int(max_items), 0

    def add(self, item: dict, tokens: dict[str, torch.Tensor], sub: bool) -> None:
        keep = self.max_items < 0 or self.n < self.max_items
        main = {k: v for k, v in tokens.items() if not is_sub_key(k)} if keep else None
        self.w.add(self.arm, item, main)
        if sub and keep:
            extra = {k: v for k, v in tokens.items() if is_sub_key(k)}
            if extra:
                self.w.add(self.arm + "__sub", {"row": item.get("row")}, extra)
        self.n += 1

    def close(self) -> dict:
        return self.w.close()


def row_dict(tt: dict[str, torch.Tensor], b: int) -> dict[str, torch.Tensor]:
    return {k: v[b] for k, v in tt.items()}


def boot(a, b=None, n: int = 2000, seed: int = 0) -> dict:
    return common.paired_boot(a, b, n=n, seed=seed)


def timer() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()
