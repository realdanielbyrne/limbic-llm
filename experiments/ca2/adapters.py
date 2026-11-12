"""M1a: the trained hippocampal loop against a parameter-matched memory-free adapter (register P5, O4, L2).

P5 trained ``pretrained_loop.SideChannels`` (TimeChannel then HippocampalLoop) on a windowed frozen model and
closed 41% (Qwen3) / 74% (Qwen3.5) of the loss gap to full attention. Only the windowed+side condition trains,
so the gain may be memory (content carried across the window) or adaptation (a per-token correction of the
out-of-distribution windowed stream). M1a replicates P5's run exactly (``--replicate`` reads its ``args`` block)
and changes one thing: the side module.

Memory-free adapter (MFA)
-------------------------
The adapter sees only the current position's stream h_t:

    u_t = B · SiLU(A · LN(h_t)),           A ∈ R^{r×D}, B ∈ R^{D×r} (with biases)
    h_t ← h_t + g_t rms(h_t) ê_t,          e_t = u_t − proj_{h_t}(u_t),  g_t = G σ(γ) tanh(‖u_t‖/λ),  G = 0.9

i.e. the loop's own CA1 path (``HippocampalLoop._inject``, reused unbound: error-only projection, saturation,
clamp ‖injection‖ = g_t‖h_t‖ ≤ G‖h_t‖), at the same tap block, with the divisive energy feedback switched off
(that feedback integrates earlier injections and would be a cross-position path). There is no TimeChannel
(its leaky integrators of the salience are cross-position too). The rank r is set so that the trainable
parameter count equals the loop's side channels within 1% (``match_rank``): 2D + (Dr + r) + (rD + D) + 2
against 4,380,675 at D = 2048 gives r = 1068 (4,381,742, +0.02%).

Evaluation-time ablations of the trained loop (``ablate``; no retraining)
--------------------------------------------------------------------------
  foreign   the store's keys and values are those of another evaluation sequence at the same positions
            (forward hooks roll the outputs of ``wk`` and ``wv`` by one row of the batch; the read query
            uses ``wq`` and stays the sequence's own). Batches are laid out with a stride of n / B sequences
            (``strided_groups``), so the donor is 32+ sequences (≥ 32k tokens) away: another article.
            The contrast prior μ_t is computed from the same (foreign) values, so the whole memory term
            F_t − m_t μ_t is the donor's. The CA1 energy feedback and the TimeChannel stay the sequence's own.
  zero      the stored values are zero, so F_t = μ_t = 0, Δ_t = 0 and the injection is exactly 0; the
            side reduces to the trained TimeChannel alone.
  in_block  the re-entry kernel w(t − s) is masked to s in the same W-block as t (block index s // W in the
            side's sink-stripped coordinates, as ``pretrained_loop.window_mask``), so the memory reads only
            what block attention already sees. Done by swapping ``limbic_llm.temporal_loop.loop_kernel``
            (the module global that ``HippocampalLoop.forward`` calls) for the duration of the context.

Runner
------
One process per model: the frozen model is loaded once and wrapped once in ``pretrained_loop.HookedLM``;
each cell (arm × seed) swaps ``lm.side``, trains with ``pretrained_loop.train`` (unchanged) and is evaluated
with ``pretrained_loop.scores`` per sequence (``eval_rows``; its token-weighted mean equals
``pretrained_loop.evaluate``, which is also called for the P5-format row). Seed s sets both the side's
initialisation (``torch.manual_seed(s)`` just before it is built) and the training batches
(``random.Random(s)`` inside ``pl.train``), so loop and MFA at one seed see the same batches.

Measures (``summarise``), on the P5 evaluation set (the first ``eval_n`` WikiText-103 validation slices):
  closure      C = (L_win − L_arm) / (L_win − L_full), per cell and averaged over seeds, with a bootstrap over
               evaluation sequences (one index draw shared by every quantity, so all comparisons are paired)
  loop − MFA   difference of closures (points), per seed and seed-averaged; gate: ≥ 3 points (plan, M1a)
  ablations    closure of the loop under each ablation and its drop from the intact loop
  by offset    closure per within-block offset bucket (memory should help most just after a block boundary)
  no harm      full attention with the side switched on, minus full attention

Usage
    python experiments/ca2/adapters.py --replicate results/pretrained/pretrained_qwen3_lm.json \\
        --out results/evolve/round12/stage1/m1a_q3.json 2>&1 | tee results/evolve/round12/stage1/m1a_q3.log
    python experiments/ca2/adapters.py --analyse results/evolve/round12/stage1/m1a_q3.json
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
sys.path.insert(0, str(EXP))
import pretrained_loop as pl  # noqa: E402  (puts ROOT on sys.path)
import limbic_llm.temporal_loop as TL  # noqa: E402
from limbic_llm.temporal_loop import HippocampalLoop, LoopStats  # noqa: E402

ARMS = ("loop", "mfa")
ABLATIONS = ("foreign", "zero", "in_block")
CKPT_ROOT = Path("~/limbic-data/m1a")
GATE_POINTS = 3.0                                                  # plan: M1a stops the line if loop − MFA < 3


# ------------------------------------------------------------------------------------------ adapter
class MemoryFreeAdapter(nn.Module):
    """u_t = B·SiLU(A·LN(h_t)) through the loop's CA1 injection; no access to any position but t.

    Exposes ``loop`` (itself) with ``gain_logit`` and ``tau`` (0) so that ``pretrained_loop.evaluate`` /
    ``train`` and ``experiments/evolve/run.py`` treat it as a side with a loop.
    """

    def __init__(self, d_model: int, rank: int, max_gain: float = 0.9, ca1_control: bool = True) -> None:
        super().__init__()
        self.rank = rank
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, rank)
        self.up = nn.Linear(rank, d_model)
        self.gain_logit = nn.Parameter(torch.tensor(-1.0))         # as HippocampalLoop: σ(−1) ≈ 0.27
        self.log_lambda = nn.Parameter(torch.tensor(0.0))
        # attributes read by HippocampalLoop._inject (CA1 path); feedback off: no cross-position energy
        self.max_gain, self.forward_gain, self.feedback = max_gain, 1.0, "none"
        self.energy_kappa, self.fb_beta = 0.0, 0.0
        self.clamp = self.project = self.saturate = ca1_control
        self.tau = 0

    @property
    def loop(self) -> "MemoryFreeAdapter":
        return self

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, LoopStats]:
        u = self.up(F.silu(self.down(self.norm(h))))
        inj, gain = HippocampalLoop._inject(self, h, u, h.new_zeros(h.shape[:-1]))
        ratio = inj.norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-6)
        return h + inj, LoopStats(ratio, gain)


def mfa_params(d_model: int, rank: int) -> int:
    return 2 * d_model + (d_model * rank + rank) + (rank * d_model + d_model) + 2


def n_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_rank(d_model: int, target: int, tol: float = 0.01) -> int:
    """The rank whose ``mfa_params`` is closest to ``target``; refuses a mismatch above ``tol``."""
    r0 = max(1, round((target - 3 * d_model - 2) / (2 * d_model + 1)))
    r = min((x for x in (r0 - 1, r0, r0 + 1) if x >= 1), key=lambda x: abs(mfa_params(d_model, x) - target))
    if abs(mfa_params(d_model, r) - target) > tol * target:
        raise ValueError(f"no rank matches {target} parameters within {tol:.0%} at d_model {d_model}")
    return r


def loop_side(d_model: int, args) -> pl.SideChannels:
    """P5's side channels, built exactly as ``pretrained_loop.main`` builds them."""
    side = pl.SideChannels(d_model, memory=args.memory, time=args.time, **pl.loop_kwargs(args))
    if args.init_gain is not None and side.loop is not None:
        side.loop.gain_logit.data.fill_(math.log(args.init_gain / (1 - args.init_gain)))
    return side


def loop_param_count(d_model: int, args) -> int:
    with torch.device("meta"):                                     # no RNG draws, no memory
        return n_trainable(loop_side(d_model, args))


def build_side(arm: str, d_model: int, args, seed: int, rank: int | None = None) -> nn.Module:
    """The side of one cell, initialised from ``torch.manual_seed(seed)``."""
    if arm not in ARMS:
        raise ValueError(arm)
    target = loop_param_count(d_model, args)
    torch.manual_seed(seed)
    if arm == "loop":
        return loop_side(d_model, args)
    side = MemoryFreeAdapter(d_model, rank or match_rank(d_model, target),
                             max_gain=dict(pl.loop_kwargs(args)).get("max_gain", 0.9), ca1_control=args.ca1_control)
    if args.init_gain is not None:
        side.gain_logit.data.fill_(math.log(args.init_gain / (1 - args.init_gain)))
    return side


# ---------------------------------------------------------------------------------------- ablations
@contextlib.contextmanager
def ablate(loop: HippocampalLoop | None, mode: str = "none", block: int | None = None, shift: int = 1):
    """Evaluation-time memory ablation of a trained ``HippocampalLoop`` (see module docstring)."""
    if mode == "none":
        yield
        return
    if mode not in ABLATIONS:
        raise ValueError(mode)
    if not isinstance(loop, HippocampalLoop):
        raise TypeError(f"ablation {mode!r} needs a HippocampalLoop, got {type(loop).__name__}")
    if loop.loop != "open" or loop.read != "direct":
        raise ValueError("ablations assume loop='open' and read='direct' (P5's settings)")
    handles, orig = [], TL.loop_kernel
    if mode == "foreign":
        def roll(mod, args, out):
            if out.shape[0] < 2:
                raise ValueError("the foreign ablation needs at least 2 sequences per batch")
            return out.roll(shift, 0)
        handles = [loop.wk.register_forward_hook(roll), loop.wv.register_forward_hook(roll)]
    elif mode == "zero":
        handles = [loop.wv.register_forward_hook(lambda mod, args, out: torch.zeros_like(out))]
    else:
        if not block:
            raise ValueError("in_block needs the window size")

        def kernel(T, *a, **kw):
            w = orig(T, *a, **kw)
            i = torch.arange(T, device=w.device) // block
            return w * (i[:, None] == i[None, :]).to(w.dtype)
        TL.loop_kernel = kernel
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()
        TL.loop_kernel = orig


# --------------------------------------------------------------------------------------- evaluation
def strided_groups(n: int, B: int) -> list[list[int]]:
    """Batches of sequence indices {k, k + S, k + 2S, ...}, S = ceil(n / B): neighbours in a batch are far apart."""
    S = -(-n // B)
    return [[k + j * S for j in range(B) if k + j * S < n] for k in range(S)]


def donors(groups: list[list[int]], shift: int = 1) -> dict[int, int]:
    """Sequence → the sequence whose memory it reads under the foreign ablation (``roll(shift, 0)``)."""
    return {g[j]: g[(j - shift) % len(g)] for g in groups for j in range(len(g))}


@torch.no_grad()
def eval_rows(lm: pl.HookedLM, ids: torch.Tensor, mask: torch.Tensor, groups: list[list[int]], window: int,
              side: bool, ablation: str = "none", block: int = 0, buckets: int = 8) -> dict:
    """Per-sequence loss (mean CE over its scored targets) for one condition, plus pooled token accuracy, the
    mean injection ratio, and the loss per within-block offset bucket of the predicting position."""
    n = len(ids)
    loss = np.full(n, np.nan)
    n_ok = n_tok = 0.0
    nb = buckets if block and buckets and block % buckets == 0 else 1
    b_sum, b_cnt, inj = np.zeros(nb), np.zeros(nb), []
    loop = getattr(lm.side, "loop", None) if ablation != "none" else None
    with lm.condition(window, side), ablate(loop, ablation, block):
        for g in groups:
            x, m0 = ids[g], mask[g]
            ce, ok, m = pl.scores(lm, x, m0)
            mf = m.float()
            loss[g] = ((ce * mf).sum(1) / mf.sum(1)).double().cpu().numpy()
            n_ok += (ok & m).sum().item()
            n_tok += mf.sum().item()
            first = x.shape[1] - ce.shape[1]
            if nb > 1:
                pos = torch.arange(first, x.shape[1], device=ce.device) - 1 - lm.sink      # predicting position
                bk = (pos % block) // (block // nb)
                for j in range(nb):
                    sel = (bk == j)[None] & m
                    b_sum[j] += ce[sel].sum().item()
                    b_cnt[j] += sel.sum().item()
            if lm.last_stats is not None:
                inj.append(lm.last_stats.inj_ratio[:, lm.side.loop.tau:].mean().item())
    out = {"loss": float(loss.mean()), "token_acc": n_ok / n_tok, "per_seq": loss.tolist(),
           "by_offset": (b_sum / np.maximum(b_cnt, 1)).tolist() if nb > 1 else None}
    if inj:
        out["inj_ratio"] = float(np.mean(inj))
    return out


# ------------------------------------------------------------------------------------------ summary
def _ci(point: float, boots: np.ndarray) -> dict:
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "p_le_0": float((boots <= 0).mean())}


def summarise(record: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    """Closures, loop − MFA and ablation drops, all bootstrapped with one shared draw of sequence indices."""
    Lw = np.asarray(record["base"]["windowed"]["per_seq"])
    Lf = np.asarray(record["base"]["full"]["per_seq"])
    n = len(Lw)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    w_b, f_b = Lw[idx].mean(1), Lf[idx].mean(1)
    gap, gap_b = Lw.mean() - Lf.mean(), w_b - f_b

    def clo(La):
        La = np.asarray(La)
        return 100 * (Lw.mean() - La.mean()) / gap, 100 * (w_b - La[idx].mean(1)) / gap_b

    cells, out = record["cells"], {"n_seq": n, "n_boot": n_boot, "gap_loss": float(gap), "closure_points": {}}
    per = {}                                                        # (arm, cond) -> list over seeds of (pt, boots)
    for key, c in sorted(cells.items()):
        arm, s = c["arm"], c["seed"]
        conds = {"side": c["windowed+side"]} | {a: c["ablations"][a] for a in c.get("ablations", {})}
        for cond, r in conds.items():
            pt, bt = clo(r["per_seq"])
            per.setdefault((arm, cond), []).append((s, pt, bt))
            out["closure_points"][f"{arm}/{cond}/s{s}"] = _ci(pt, bt)
    means = {k: (np.mean([p for _, p, _ in v]), np.mean([b for _, _, b in v], 0)) for k, v in per.items()}
    for (arm, cond), (pt, bt) in means.items():
        out["closure_points"][f"{arm}/{cond}/mean"] = _ci(pt, bt)
    if ("loop", "side") in per and ("mfa", "side") in per:
        loop = {s: (p, b) for s, p, b in per["loop", "side"]}
        mfa = {s: (p, b) for s, p, b in per["mfa", "side"]}
        common = sorted(set(loop) & set(mfa))
        out["loop_minus_mfa"] = {f"s{s}": _ci(loop[s][0] - mfa[s][0], loop[s][1] - mfa[s][1]) for s in common}
        if common:
            d_pt = np.mean([loop[s][0] - mfa[s][0] for s in common])
            d_bt = np.mean([loop[s][1] - mfa[s][1] for s in common], 0)
            out["loop_minus_mfa"]["mean"] = _ci(d_pt, d_bt)
            out["gate"] = {"rule": f"loop − MFA ≥ {GATE_POINTS} closure points (seed mean)",
                           "passes": bool(d_pt >= GATE_POINTS), "seeds": common}
    if ("loop", "side") in per:
        loop = dict((s, (p, b)) for s, p, b in per["loop", "side"])
        out["ablation_drop"] = {}
        for a in ABLATIONS:
            if ("loop", a) not in per:
                continue
            ab = dict((s, (p, b)) for s, p, b in per["loop", a])
            common = sorted(set(loop) & set(ab))
            d = {f"s{s}": _ci(loop[s][0] - ab[s][0], loop[s][1] - ab[s][1]) for s in common}
            d["mean"] = _ci(np.mean([loop[s][0] - ab[s][0] for s in common]),
                            np.mean([loop[s][1] - ab[s][1] for s in common], 0))
            out["ablation_drop"][a] = d
    bw, bf = record["base"]["windowed"].get("by_offset"), record["base"]["full"].get("by_offset")
    if bw:
        bw, bf = np.asarray(bw), np.asarray(bf)
        out["closure_by_offset"] = {}
        for arm in ARMS:
            rows = [np.asarray(c["windowed+side"]["by_offset"]) for c in cells.values() if c["arm"] == arm]
            if rows:
                out["closure_by_offset"][arm] = (100 * (bw - np.mean(rows, 0)) / (bw - bf)).tolist()
    out["no_harm"] = {k: c["full+side"]["loss"] - record["base"]["full"]["loss"]
                      for k, c in cells.items() if "full+side" in c}
    return out


def print_summary(s: dict) -> None:
    print(f"\ngap closure (points; 95% bootstrap over {s['n_seq']} sequences)")
    for k, v in s["closure_points"].items():
        print(f"  {k:24s} {v['point']:7.2f}  [{v['lo']:7.2f}, {v['hi']:7.2f}]")
    for name in ("loop_minus_mfa",):
        for k, v in s.get(name, {}).items():
            print(f"  loop − MFA {k:12s} {v['point']:7.2f}  [{v['lo']:7.2f}, {v['hi']:7.2f}]  p≤0 {v['p_le_0']:.4f}")
    for a, d in s.get("ablation_drop", {}).items():
        for k, v in d.items():
            print(f"  drop {a:9s} {k:8s} {v['point']:7.2f}  [{v['lo']:7.2f}, {v['hi']:7.2f}]")
    if "gate" in s:
        print(f"  gate ({s['gate']['rule']}): {'passes' if s['gate']['passes'] else 'fails'}")


# -------------------------------------------------------------------------------------------- runner
def parse(argv=None) -> tuple[argparse.Namespace, argparse.Namespace]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False)
    ap.add_argument("--replicate", default=None,
                    help="results JSON whose 'args' block sets every pretrained_loop flag not given here (P5)")
    ap.add_argument("--arms", default="loop,mfa")
    ap.add_argument("--seeds", default="0,1", help="init and data seeds; one cell per arm × seed")
    ap.add_argument("--ablations", default=",".join(ABLATIONS), help="evaluated on the loop arm; '' for none")
    ap.add_argument("--mfa-rank", type=int, default=None, help="default: matched to the loop's parameter count")
    ap.add_argument("--buckets", type=int, default=8, help="within-block offset buckets")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--ckpt-dir", default=str(CKPT_ROOT), help="side-channel state dicts (outside the repo)")
    ap.add_argument("--resume", action="store_true", help="keep the base rows and finished cells of --out")
    ap.add_argument("--analyse", default=None, metavar="JSON", help="recompute and print the summary of a run")
    own, rest = ap.parse_known_args(argv)
    if any(t in ("-h", "--help") for t in rest):
        ap.print_help()
    args = pl.parse_args(rest)
    if own.replicate:
        src = json.loads(Path(own.replicate).read_text())["args"]
        given = {t.split("=")[0] for t in rest if t.startswith("--")}
        known = {f"--{p}{k.replace('_', '-')}" for k in vars(args) for p in ("", "no-")}
        if given - known:                  # an abbreviation would be matched by prefix and hide its neighbours
            raise SystemExit(f"--replicate: give pretrained_loop flags in full: {sorted(given - known)}")
        for k, v in src.items():
            if k in ("out", "seed"):
                continue
            flag, neg = "--" + k.replace("_", "-"), "--no-" + k.replace("_", "-")
            if flag in given or neg in given:
                continue                                           # given on the command line: it wins
            if not hasattr(args, k):
                raise SystemExit(f"--replicate: unknown pretrained_loop argument {k!r}")
            setattr(args, k, v)
    return own, args


def _ckpt_dir(path: str) -> Path:
    d = Path(path).expanduser().resolve()
    if d.is_relative_to(ROOT.resolve()):
        raise SystemExit(f"--ckpt-dir {d} is inside the repository; checkpoints go under ~/limbic-data/")
    return d


def main(argv=None) -> dict:
    own, args = parse(argv)
    if own.analyse:
        rec = json.loads(Path(own.analyse).read_text())
        s = summarise(rec, own.boot)
        print_summary(s)
        return s
    if args.task != "lm" or not args.window:
        raise SystemExit("M1a is P5's LM task on a windowed model: --task lm and --window > 0")
    if args.window_mode in ("block", "segment") and args.seq_len % args.window:
        raise SystemExit("--seq-len must be a multiple of --window for block/segment windows")
    arms = [a for a in own.arms.split(",") if a]
    seeds = [int(s) for s in own.seeds.split(",") if s]
    abls = [a for a in own.ablations.split(",") if a]
    if set(arms) - set(ARMS) or set(abls) - set(ABLATIONS):
        raise SystemExit(f"arms ⊆ {ARMS}, ablations ⊆ {ABLATIONS}")
    out = Path(args.out or f"results/evolve/round12/stage1/m1a_{Path(args.model).name}.json")
    ckdir = _ckpt_dir(own.ckpt_dir) / out.stem
    t0 = time.time()

    model, tok = pl.load_model(args)
    tc = pl.text_config(model)
    sink, d = int(args.sink), tc.hidden_size
    lm = pl.HookedLM(model, pl.SideChannels(d, memory=False, time=False), args.layer, args.window, args.window_mode,
                     args.window_layers, sink, pad_id=tok.eos_token_id, grad_checkpointing=args.grad_checkpointing)
    dev = next(lm.blocks[lm.layer].parameters()).device
    check = pl.check_window(lm)
    tr, ev = pl.load_text(args, tok)
    task = pl.TextTask(torch.tensor(tr), torch.tensor(ev), args.seq_len, sink, tok.bos_token_id, args.window)
    batches = task.eval_batches(args.eval_n, args.batch)
    ids, mask = torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches])
    groups = strided_groups(len(ids), args.batch)
    if "foreign" in abls and min(len(g) for g in groups) < 2:
        raise SystemExit("the foreign ablation needs --batch ≥ 2 and --eval-n ≥ 2 · batch")
    target = loop_param_count(d, args)
    rank = own.mfa_rank or match_rank(d, target)
    info = {"blocks": len(lm.blocks), "linear_layers": lm.kinds.count("linear"), "tap_layer": lm.layer,
            "windowed_layers": len(lm.windowed), "window_check": check, "d_model": d,
            "loop_params": target, "mfa_rank": rank, "mfa_params": mfa_params(d, rank),
            "param_mismatch": mfa_params(d, rank) / target - 1,
            "base_params": sum(p.numel() for p in model.parameters()), "train_tokens": len(tr), "eval_tokens": len(ev),
            "eval_seqs": len(ids), "transformers": pl.transformers.__version__}
    print(f"{args.model if not args.tiny else 'tiny-' + args.tiny_arch}: {info['blocks']} blocks, tap {lm.layer}, "
          f"{info['windowed_layers']} windowed ({args.window_mode} {args.window}), window check {check}")
    print(f"loop side {target:,} params; MFA rank {rank}: {info['mfa_params']:,} ({info['param_mismatch']:+.3%})")

    rec = {"args": vars(args), "m1a": vars(own), "model": info, "base": None, "cells": {},
           "eval_layout": {"groups": groups, "foreign_donor": donors(groups)}}
    if own.resume and out.exists():
        old = json.loads(out.read_text())
        rec["base"], rec["cells"] = old.get("base"), old.get("cells", {})
        print(f"resume: {len(rec['cells'])} finished cells from {out}")

    def save():
        rec["seconds"] = time.time() - t0
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1))

    kw = dict(block=args.window, buckets=own.buckets)
    if rec["base"] is None:
        rec["base"] = {"windowed": eval_rows(lm, ids, mask, groups, args.window, False, **kw),
                       "full": eval_rows(lm, ids, mask, groups, 0, False, **kw)}
        save()
    print(f"base: windowed {rec['base']['windowed']['loss']:.4f}, full {rec['base']['full']['loss']:.4f}")

    for seed in seeds:
        for arm in arms:
            key = f"{arm}_s{seed}"
            if key in rec["cells"]:
                continue
            tc0 = time.time()
            side = build_side(arm, d, args, seed, rank).to(dev)       # float32 master weights (pl._tap)
            lm.side = side
            a = copy.copy(args)
            a.seed = seed
            print(f"\n== {key}: {n_trainable(side):,} trainable parameters", flush=True)
            curve = pl.train(lm, task, a) if args.steps else []
            ckdir.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in side.state_dict().items()}, ckdir / f"{key}.pt")
            cell = {"arm": arm, "seed": seed, "params": n_trainable(side), "curve": curve,
                    "gain": torch.sigmoid(side.loop.gain_logit.detach().float()).item(),
                    "p5_format": pl.evaluate(lm, batches, args.window, True, "lm"),
                    "windowed+side": eval_rows(lm, ids, mask, groups, args.window, True, **kw),
                    "full+side": eval_rows(lm, ids, mask, groups, 0, True, **kw),
                    "checkpoint": str(ckdir / f"{key}.pt")}
            if arm == "loop":
                cell["ablations"] = {ab: eval_rows(lm, ids, mask, groups, args.window, True, ablation=ab, **kw)
                                     for ab in abls}
            cell["seconds"] = time.time() - tc0
            rec["cells"][key] = cell
            msg = ", ".join(f"{ab} {r['loss']:.4f}" for ab, r in cell.get("ablations", {}).items())
            print(f"{key}: windowed+side {cell['windowed+side']['loss']:.4f} "
                  f"(P5 format {cell['p5_format']['loss']:.4f}), full+side {cell['full+side']['loss']:.4f}"
                  f"{'; ' + msg if msg else ''} ({cell['seconds']:.0f}s)")
            save()
            del side
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    rec["summary"] = summarise(rec, own.boot)
    print_summary(rec["summary"])
    if dev.type == "cuda":
        rec["model"]["max_memory_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    save()
    print(f"saved {out} ({time.time() - t0:.0f}s)")
    return rec


if __name__ == "__main__":
    main()
