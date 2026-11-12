"""Stage 0c text (and the later windowed-text stage): teacher-forced CA2 injection on a frozen LM, three
conditions on the same rows, every signal recorded in every arm.

Task. The evaluation set is the whole test file (``common.text_task``: ``pl.TextTask(ev, ev, ...)``; nothing
is trained, so train = eval). Rows are consecutive ``--seq-len`` chunks with ``--sink`` sink tokens in front;
batches are interleaved (batch k holds rows k, k + n_b, k + 2n_b, ...) so that the ``foreign`` direction
(another row's gist at the same column) comes from a distant part of the file, not the same article.
Targets predicted from positions ≥ sink + W are "scored" (``TextTask.first``), in every condition, so the
conditions are paired on identical tokens.

Conditions (one ``pl.HookedLM`` per condition, block window mode with a sink):
    full   full attention (window 0)
    attn   attention windowed to W, GDN layers intact (``--window-layers attention``): GDN state is unbounded
    all    every token mixer windowed (``--window-layers all``): the GDN state restarts every block, the
           control for retained state. On a model without linear layers ``all`` equals ``attn`` and is skipped.

Arms (per condition; ``SteerSide`` hosts a ``CA2Controller`` at block L's output, h ← h + g‖h‖d̂ from
text position ``--steer-from`` on, fixed g):
    none                 g = 0 with every tap and source attached (byte-identical to the base, checked)
    <dir>@<g>            gist, anti_gist, mean, self, random:<seed>, foreign, window_gist (windowed only)
    impulse:<dir>@<g>    g on text positions [start, start + len) only, then 0 (impulse response)

Per token (trace directory, ``ArmTraces``; text-position coordinates, the sink column dropped): the whole
controller trace (gain, injection, every source including the gist drift, regulators, bookkeeping, the
attention tap at the deepest full-attention block ≤ L and two more above the site, ``attn_up`` and
``attn_last``, the GDN taps, residual norms at every block), plus
    out/loss, ent, rank, top1, p_tgt, p_max, kl_base (KL to the unsteered model in this condition),
    kl_full (KL to the full-context model; windowed conditions), loss_base, loss_full, ent_base,
    norm_ratio (‖h_l‖ arm / ‖h_l‖ unsteered at every block, (T, n_blocks)),
    lens/rank_d<k> (logit-lens rank of the target at depth k), cont/* (contiguity, ``measure.contiguity``),
    pos/text, pos/offset, pos/block, pos/scored, imp/lag (impulse arms).
Per item and per arm (results JSON): scored sums, offset-bucket and position-bucket sums (loss, base and
full loss, KL) over the scored tokens (``off_*``, ``pos_*``: closure, loss and Δloss by bucket) and over every
token (``offall_*``, ``posall_*``: descriptive), contiguity boost by lag bin × in-block, impulse KL and Δloss
by lag bin, logit-lens log-rank, gain distribution, gap closure C = (L_win − L_arm)/(L_win − L_full) with numerator and
denominator bootstrapped jointly over sequences, paired Δloss against none and against full, |Δ entropy|,
the downstream norm ratio by block, the gist drift by lag, and the pooled mean of every scalar signal.
GDN per-token-loop keys (mismatch, read, lagged prediction, state norm) and per-head keys are computed on the
first ``--gdn-mismatch-n`` rows of each arm only (the loop costs ~56 forward passes) and stored in the
``tokens__sub`` stream; the GDN sources built on them are NaN (``defined`` = 0) on the other rows.

    python experiments/ca2/text.py --model Qwen/Qwen3.5-2B-Base --n-seqs 256 --seq-len 1024 --window 64 \\
        --conditions full,attn,all --gains 0.05,0.1,0.2 2>&1 | tee results/evolve/round12/stage0/text_q35.log
    python experiments/ca2/text.py --tiny --tiny-arch qwen35          # CPU smoke, < 1 min
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402
from ca2 import common  # noqa: E402
from ca2 import measure as ms  # noqa: E402
from ca2.controller import CA2Controller  # noqa: E402
from ca2.sources import ConstantSource, PositionMeanSource  # noqa: E402

CONDITIONS = {"full": "attention", "attn": "attention", "all": "all"}
OUT_KEYS = ("loss", "ent", "rank", "top1", "p_tgt", "p_max")


# ------------------------------------------------------------------------------------------------ CLI
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a = p.add_argument
    a("--model", default="Qwen/Qwen3.5-2B-Base")
    a("--dtype", default="bfloat16", choices=list(pl.DTYPES))
    a("--device", default="cuda")
    a("--attn-impl", default="sdpa")
    a("--tiny", action="store_true", help="tiny random model + byte tokenizer, CPU, small defaults")
    a("--tiny-arch", default="qwen35", choices=["qwen35", "qwen3", "gpt2"])
    a("--seed", type=int, default=0)
    a("--text-file", default="~/limbic-data/wikitext103_test.txt")
    a("--max-chars", type=int, default=None, help="read only the first N characters (default: all; tiny: 200k)")
    a("--seq-len", type=int, default=None, help="text tokens per row (default 1024; tiny 96)")
    a("--n-seqs", type=int, default=None, help="rows (default 256; tiny 4)")
    a("--batch", type=int, default=None, help="rows per forward (default 2)")
    a("--sink", type=int, default=1)
    a("--window", type=int, default=None, help="attention window W (default 64; tiny 16)")
    a("--conditions", default="full,attn,all")
    a("--site", type=int, default=None, help="tap block L (inject at its output); default n_blocks // 2")
    a("--from-ceiling", default=None, help="ceiling JSON: site and gains (g_max/4, g_max/2, g_max) from it")
    a("--directions", default="gist,anti_gist,mean,self,random:0,random:1,random:2,foreign,window_gist")
    a("--gains", default=None, help="fixed gains (default 0.05,0.1,0.2)")
    a("--g-abs", type=float, default=2.0)
    a("--betas", default="0.9,0.995")
    a("--steer-from", type=int, default=None, help="first steered text position (default 32; tiny 8)")
    a("--impulse-directions", default="gist,random:0", help="'' for none")
    a("--impulse-gains", default=None, help="default: the largest fixed gain")
    a("--impulse-start", type=int, default=None, help="text position of the burst (default 128; tiny 32)")
    a("--impulse-len", type=int, default=None, help="burst length (default 16; tiny 4)")
    a("--attn-window", type=int, default=32, help="W_r of the attention tap (capped at W)")
    a("--lens-depths", default=None, help="block indices for the logit lens (default 4 depths)")
    a("--no-attn-extra", dest="attn_extra", action="store_false",
      help="skip the attention taps above the site (attn_up, attn_last)")
    a("--gdn-mismatch-n", type=int, default=32, help="rows per arm running the GDN per-token loop (-1: all)")
    a("--gdn-per-head", default="beta,ret,fam,fam_lag,err,str,cos_vp", help="per-head keys on that subsample")
    a("--trace-tokens", type=int, default=-1, help="rows per arm with per-token arrays (-1: all)")
    a("--arms", default=None, help="comma list: run only these arm names (e.g. none,gist@0.1)")
    a("--boot", type=int, default=2000)
    a("--name", default=None)
    a("--stage", default="stage0")
    a("--out", default=None)
    a("--trace-dir", default=None)
    a("--force", action="store_true")
    args = p.parse_args(argv)
    tiny = dict(seq_len=96, n_seqs=4, batch=2, window=16, steer_from=8, impulse_start=32, impulse_len=4,
                max_chars=200_000, gains="0.1,0.3")
    real = dict(seq_len=1024, n_seqs=256, batch=2, window=64, steer_from=32, impulse_start=128, impulse_len=16,
                max_chars=None, gains="0.05,0.1,0.2")
    for k, v in (tiny if args.tiny else real).items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.tiny:
        args.attn_window = min(args.attn_window, args.window)
        if args.dtype == "bfloat16" and args.device != "cuda":
            args.dtype = "float32"
    return args


def floats(s: str | None) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()] if s else []


# ----------------------------------------------------------------------------------------------- arms
def make_arms(args, cond: str) -> list[dict]:
    win = cond != "full"
    arms = [{"name": "none", "kind": "none", "direction": "gist", "seed": 0, "gain": 0.0}]
    for g in floats(args.gains):
        for spec in [d for d in args.directions.split(",") if d]:
            name, seed = ms.parse_direction(spec)
            if name == "window_gist" and not win:
                continue
            arms.append({"name": f"{spec}@{g:g}", "kind": "fixed", "direction": name, "seed": seed, "gain": g})
    ig = floats(args.impulse_gains) or ([max(floats(args.gains))] if floats(args.gains) else [])
    for g in ig:
        for spec in [d for d in args.impulse_directions.split(",") if d]:
            name, seed = ms.parse_direction(spec)
            if name == "window_gist" and not win:
                continue
            arms.append({"name": f"impulse:{spec}@{g:g}", "kind": "impulse", "direction": name, "seed": seed,
                         "gain": g})
    if args.arms:
        keep = set(args.arms.split(","))
        arms = [a for a in arms if a["name"] in keep]
    return arms


def build_ctrl(arm: dict, d_model: int, args, T_text: int) -> CA2Controller:
    g = float(arm["gain"])
    if arm["kind"] == "impulse":
        table = torch.zeros(T_text)
        table[args.impulse_start: args.impulse_start + args.impulse_len] = 1.0
        source, steer_from = PositionMeanSource(table, index="position", name="impulse"), 0
    else:
        source, steer_from = ConstantSource(1.0), args.steer_from
    betas = tuple(floats(args.betas))
    return CA2Controller(d_model, source=source, g_min=0.0, g_max=g, g_abs=max(args.g_abs, g), smooth="none",
                         direction=arm["direction"], betas=betas, sink_skip=0, steer_from=steer_from,
                         window=args.window if arm["direction"] == "window_gist" else None, seed=arm["seed"],
                         massive_window=max(64, args.attn_window), sources=("agree",))


# ----------------------------------------------------------------------------------------------- data
def make_batches(args, tok) -> tuple[list, dict]:
    path = Path(args.text_file).expanduser()
    text = path.read_text(encoding="utf-8", errors="replace")
    if args.max_chars:
        text = text[: args.max_chars]
    ids = torch.tensor(tok.encode(text))
    task = common.text_task(ids, args.seq_len, args.sink, tok.bos_token_id, args.window)
    rows = task.eval_batches(args.n_seqs, 1)
    n = len(rows)
    has_foreign = "foreign" in args.directions or "foreign" in args.impulse_directions
    if has_foreign and n % args.batch:
        n -= n % args.batch
    if n < 1 or (has_foreign and args.batch < 2):
        raise SystemExit("need at least one full batch, and --batch >= 2 for the foreign direction")
    rows = rows[:n]
    nb = -(-n // args.batch)
    batches = []
    for k in range(nb):
        idx = list(range(k, n, nb))
        batches.append((torch.cat([rows[i][0] for i in idx]), torch.cat([rows[i][1] for i in idx]), idx))
    info = {"file": str(path), "n_tokens_file": int(len(ids)), "max_chars": args.max_chars, "n_rows": n,
            "seq_len": args.seq_len, "sink": args.sink, "bos": tok.bos_token_id, "first_scored": task.first,
            "row_starts": [i * args.seq_len for i in range(n)], "batches": [b[2] for b in batches],
            "layout": "interleaved: batch k = rows k, k + n_b, ..."}
    return batches, info


# ------------------------------------------------------------------------------------------- one arm
class Acc:
    """Per-item sums of one arm (results JSON ``per_item``) and the pooled gain samples."""

    def __init__(self, n_off: int, n_pos: int, n_lag: int, depths: list[int]) -> None:
        self.items: dict[str, list] = {}
        self.g: list[torch.Tensor] = []
        self.inj: list[torch.Tensor] = []
        self.shapes = (n_off, n_pos, n_lag)
        self.depths = depths
        self.checks = {"gain0_equal": None, "kl_base_max_none": None, "imp_pre_kl_max": None}
        self.sig: dict[str, list[float]] = {}          # pooled sums and counts of every scalar signal
        self.time = {"base_s": 0.0, "arm_s": 0.0, "metrics_s": 0.0, "tokens": 0}

    def put(self, key: str, rows) -> None:
        self.items.setdefault(key, []).extend(np.asarray(rows).tolist())


def run_arm(arm: dict, cond: str, lm: pl.HookedLM, model, blocks, kinds, batches, args, d_model: int,
            cont_cache: dict, writer: ms.ArmTraces, depths: list[int]) -> dict:
    s, win = args.sink, (args.window if cond != "full" else 0)
    T_text = args.seq_len
    off_e, pos_e = ms.offset_edges(args.window), ms.log2_edges(T_text)
    lag_e = ms.lag_edges(T_text)
    acc = Acc(len(off_e) - 1, len(pos_e) - 1, len(lag_e) - 1, depths)
    per_head = [k for k in args.gdn_per_head.split(",") if k]
    n_sub_b = (len(batches) if args.gdn_mismatch_n < 0 else -(-args.gdn_mismatch_n // args.batch))
    pos = torch.arange(T_text + s)
    text_pos = (pos - s).float()
    for bi, (ids, scored, rows) in enumerate(batches):
        B, T = ids.shape
        dev = lm.input_device
        scored = torch.cat([scored[:, 1:], torch.zeros_like(scored[:, :1])], 1)   # target mask → predicting position
        tg, valid = ms.shifted_targets(ids)
        valid[:, :s] = False
        t0 = ms.timer()
        refs = {}
        ncap = ms.NormCapture(blocks)
        try:
            with lm.condition(win, False):
                refs["base"] = lm(ids)
        finally:
            ncap.remove()
        base_norm = ncap.norms()
        if win:
            with lm.condition(0, False):
                refs["full"] = lm(ids)
        acc.time["base_s"] += ms.timer() - t0
        sub = bi < n_sub_b
        ctrl = build_ctrl(arm, d_model, args, T_text)
        ctrl.record = True
        br = arm.get("branch")
        law = None
        if br and br["target"] == "gdn":
            law = ms.gdn_mod().BranchLaw(br["layers"], mode="fixed", gain=float(br["g"]), g_abs=max(2.0, br["g"]))
        extra = ms.extra_attention_taps(blocks, kinds, lm.layer, args.attn_window) if args.attn_extra else []
        taps = ms.register_signals(ctrl, blocks, kinds, lm.layer, args.attn_window, sub, per_head,
                                   extra_taps=extra, branch_law=law)
        side = common.SteerSide(ctrl, model, sink=s, taps=taps, tap_layer=lm.layer)
        lm.side = side
        lens = ms.LensCapture(blocks, depths)
        bscale = (ms.BranchScale(blocks, br["layers"], br["target"], float(br["g"]), s + args.steer_from)
                  if br and br["target"] != "gdn" else None)
        t0 = ms.timer()
        try:
            with lm.condition(win, True):
                lg = lm(ids)
        finally:
            lens.remove()
            side.remove()
            if bscale is not None:
                bscale.remove()
        acc.time["arm_s"] += ms.timer() - t0
        t0 = ms.timer()
        if arm["kind"] == "none":
            eq = bool(torch.equal(lg, refs["base"]))
            acc.checks["gain0_equal"] = eq if acc.checks["gain0_equal"] is None else (acc.checks["gain0_equal"] and eq)
        ck = (cond, bi)
        if ck not in cont_cache:
            cont_cache[ck] = ms.contiguity(ids, s, win, s, args.seed, rows)
        cont = cont_cache[ck]
        m = ms.token_metrics(lg, tg, valid, refs, cand=cont["cand"])
        for r, ref in refs.items():
            mr = ms.token_metrics(ref, tg, valid)
            m[f"loss_{r}"] = mr["loss"]
            if r == "base":
                m["ent_base"] = mr["ent"]
        ranks = lens.ranks(model, tg.to(dev), valid)
        del lg, refs
        tt = ctrl.trace_tensors()
        acc.time["metrics_s"] += ms.timer() - t0
        acc.time["tokens"] += int(B * (T - s))
        # ---- per-token arrays in text coordinates (sink dropped), (B, T_text, ...)
        cut = lambda x: x[:, s:].detach().float().cpu()                         # noqa: E731
        tok = dict(tt)
        for k in OUT_KEYS + tuple(f"kl_{r}" for r in (["base", "full"] if win else ["base"])):
            tok[f"out/{k}"] = cut(m[k])
        for r in (["base", "full"] if win else ["base"]):
            tok[f"out/loss_{r}"] = cut(m[f"loss_{r}"])
        tok["out/ent_base"] = cut(m["ent_base"])
        if "tap/norms/norm" in tok:
            tok["out/norm_ratio"] = tok["tap/norms/norm"] / base_norm[:, s:].cpu().clamp_min(1e-12)
        for d, r in ranks.items():
            tok[f"lens/rank_d{d}"] = cut(r)
        lp = m["lp"][:, s:].float().cpu()
        for k, name in enumerate(ms.CONT_DIRS):
            tok[f"cont/lp_{name}"] = lp[..., k]
        for k in ("lag", "inblock", "hit"):
            tok[f"cont/{k}"] = cont[k][:, s:]
        tp = text_pos[s:T].expand(B, -1)
        tok["pos/text"] = tp
        tok["pos/offset"] = (tp % args.window) if args.window else tp
        tok["pos/block"] = (tp // args.window) if args.window else torch.zeros_like(tp)
        tok["pos/scored"] = scored[:, s:].float()
        imp_lag = None
        if arm["kind"] == "impulse":
            t_end = args.impulse_start + args.impulse_len - 1
            imp_lag = tp - t_end
            tok["imp/lag"] = imp_lag
        # ---- per-item sums
        sc = scored[:, s:].bool() & torch.isfinite(tok["out/loss"])
        nll, n = ms.masked_sum(tok["out/loss"], sc)
        acc.put("row", rows)
        acc.put("start", [r * args.seq_len for r in rows])
        acc.put("nll_sum", nll)
        acc.put("n_tok", n)
        for k in ("top1", "ent", "kl_base", "loss_base") + (("kl_full", "loss_full") if win else ()):
            acc.put(f"{k}_sum", ms.masked_sum(tok[f"out/{k}"], sc)[0])
        dent = tok["out/ent"] - tok["out/ent_base"]
        acc.put("dent_sum", ms.masked_sum(dent, sc)[0])
        acc.put("absdent_sum", ms.masked_sum(dent.abs(), sc)[0])
        if "out/norm_ratio" in tok:
            nr = tok["out/norm_ratio"]
            w = sc.float()[..., None] * torch.isfinite(nr).float()
            acc.put("normratio_sum", (torch.nan_to_num(nr) * w).sum(1).double().numpy())
            acc.put("normratio_n", w.sum(1).double().numpy())
        for tname in ("attn_up", "attn_last"):
            if f"tap/{tname}/ent" in tok:
                acc.put(f"{tname}_ent_sum", ms.masked_sum(tok[f"tap/{tname}/ent"], sc)[0])
                acc.put(f"{tname}_ent_n", ms.masked_sum(tok[f"tap/{tname}/ent"], sc)[1])
        for d in depths:
            acc.put(f"lens_logrank_d{d}_sum", ms.masked_sum(torch.log1p(tok[f"lens/rank_d{d}"]), sc)[0])
            acc.put(f"lens_top1_d{d}_sum", ms.masked_sum((tok[f"lens/rank_d{d}"] == 0).float(), sc)[0])
        # offset / position buckets: off_*, pos_* over the scored set ``sc`` (the tokens loss, KL and closure use;
        # in a windowed condition the first block is steered but has no window gap, so it must not enter closure);
        # offall_*, posall_* over every valid token (descriptive only)
        allv = torch.isfinite(tok["out/loss"])
        bo, bp = ms.bucketize(tok["pos/offset"], off_e), ms.bucketize(tp, pos_e)
        for bname, bidx, keep in (("off", bo, sc), ("pos", bp, sc), ("offall", bo, allv), ("posall", bp, allv)):
            bidx = torch.where(keep, bidx, torch.full_like(bidx, -1))
            nb_ = len(off_e) - 1 if bname.startswith("off") else len(pos_e) - 1
            for k in ("loss", "loss_base", "kl_base") + (("loss_full", "kl_full") if win else ()):
                sm, ct = ms.bucket_sums(tok[f"out/{k}"], bidx, nb_)
                acc.put(f"{bname}_{k}_sum", sm)
                if k == "loss":
                    acc.put(f"{bname}_n", ct)
        # contiguity: boost in each direction against the control, by lag bin × in/out of block
        boost = {d: tok[f"cont/lp_{d}"] - tok["cont/lp_ctrl"] for d in ms.CONT_DIRS[:4]}
        lag_b = ms.bucketize(tok["cont/lag"], lag_e)
        for inb in (1.0, 0.0):
            sel = torch.where(tok["cont/inblock"] == inb, lag_b, torch.full_like(lag_b, -1))
            for d in ("p1",):
                sm, ct = ms.bucket_sums(boost[d], sel, len(lag_e) - 1)
                tag = "in" if inb else "out"
                acc.put(f"cont_{d}_{tag}_sum", sm)
                acc.put(f"cont_{d}_{tag}_n", ct)
        for d in ms.CONT_DIRS[:4]:
            sm, ct = ms.masked_sum(boost[d], torch.ones_like(sc))
            acc.put(f"cont_{d}_all_sum", sm)
            acc.put(f"cont_{d}_all_n", ct)
        hs, hn = ms.masked_sum(tok["cont/hit"], torch.ones_like(sc))
        acc.put("cont_hit_sum", hs)
        acc.put("cont_hit_n", hn)
        if imp_lag is not None:
            dl = tok["out/loss"] - tok["out/loss_base"]
            pre = imp_lag < -(args.impulse_len - 1)
            kb = tok["out/kl_base"]
            mx = float(torch.nan_to_num(kb[pre], nan=0.0).abs().max()) if pre.any() else 0.0
            acc.checks["imp_pre_kl_max"] = max(acc.checks["imp_pre_kl_max"] or 0.0, mx)
            burst = (imp_lag <= 0) & ~pre
            acc.put("imp_burst_kl_sum", ms.masked_sum(kb, burst)[0])
            acc.put("imp_burst_n", ms.masked_sum(kb, burst)[1])
            li = ms.bucketize(imp_lag, lag_e)
            for k, v in (("kl", kb), ("dloss", dl)) + ((("klfull", tok["out/kl_full"]),) if win else ()):
                sm, ct = ms.bucket_sums(v, li, len(lag_e) - 1)
                acc.put(f"imp_{k}_sum", sm)
                if k == "kl":
                    acc.put("imp_n", ct)
        if arm["kind"] == "none":
            kmax = float(torch.nan_to_num(tok["out/kl_base"], nan=0.0).max())
            acc.checks["kl_base_max_none"] = max(acc.checks["kl_base_max_none"] or 0.0, kmax)
        for k, v in tok.items():                      # pooled mean of every scalar float signal over scored tokens
            if v.dim() == 2 and v.is_floating_point() and not k.startswith(("pos/", "cont/lp")):
                sm, ct = ms.masked_sum(v, sc)
                cur = acc.sig.setdefault(k, [0.0, 0.0])
                cur[0] += float(sm.sum())
                cur[1] += float(ct.sum())
        for k in [k for k in tok if k.startswith("src/gist_drift/") and k != "src/gist_drift/s"]:
            acc.put(k.replace("/", "_") + "_sum", ms.masked_sum(tok[k], sc)[0])
            acc.put(k.replace("/", "_") + "_n", ms.masked_sum(tok[k], sc)[1])
        st = tok["steer"].bool()
        acc.g.append(tok["g_raw"][st])
        acc.inj.append(tok["inj_ratio"][st])
        # ---- traces
        for b in range(B):
            item = {"row": rows[b], "start": rows[b] * args.seq_len, "cond": cond, "arm": arm["name"],
                    "nll_sum": float(nll[b]), "n_tok": float(n[b]), "batch": bi, "gdn_loop": sub}
            writer.add(item, ms.row_dict(tok, b), sub)
    return finish_arm(arm, cond, acc, args, win, (off_e, pos_e, lag_e))


def finish_arm(arm: dict, cond: str, acc: Acc, args, win: int, edges) -> dict:
    it = {k: np.asarray(v, dtype=np.float64) for k, v in acc.items.items()}
    n = it["n_tok"]
    seq = lambda k: it[f"{k}_sum"] / np.maximum(n, 1)                         # noqa: E731
    tot = lambda k: float(it[f"{k}_sum"].sum() / max(n.sum(), 1))             # noqa: E731
    agg = {"loss": float(it["nll_sum"].sum() / max(n.sum(), 1)), "n_tok": int(n.sum()), "n_seq": int(len(n))}
    agg["ppl"] = math.exp(agg["loss"])
    for k in ("top1", "ent", "kl_base", "loss_base", "dent", "absdent") + (("kl_full", "loss_full") if win else ()):
        agg[k] = tot(k)
    if "normratio_n" in it:
        agg["norm_ratio_by_block"] = (it["normratio_sum"].sum(0) / np.maximum(it["normratio_n"].sum(0), 1)).tolist()
        agg["norm_ratio_final"] = agg["norm_ratio_by_block"][-1]
    for tname in ("attn_up", "attn_last"):
        if f"{tname}_ent_n" in it:
            agg[f"{tname}_ent"] = float(it[f"{tname}_ent_sum"].sum() / max(it[f"{tname}_ent_n"].sum(), 1))
    for d in acc.depths:
        agg[f"lens_logrank_d{d}"] = tot(f"lens_logrank_d{d}")
        agg[f"lens_top1_d{d}"] = tot(f"lens_top1_d{d}")
    L = it["nll_sum"] / np.maximum(n, 1)
    agg["d_loss_vs_base"] = ms.boot(L, seq("loss_base"), n=args.boot, seed=args.seed)
    if win:
        agg["d_loss_vs_full"] = ms.boot(L, seq("loss_full"), n=args.boot, seed=args.seed)
        agg["closure"] = closure_boot(it["nll_sum"], it["loss_base_sum"], it["loss_full_sum"], n, args.boot, args.seed)
        for b in ("off", "pos"):
            nb = it[f"{b}_n"]
            agg[f"closure_by_{b}"] = [closure_boot(it[f"{b}_loss_sum"][:, i], it[f"{b}_loss_base_sum"][:, i],
                                                   it[f"{b}_loss_full_sum"][:, i], nb[:, i], args.boot, args.seed)
                                      for i in range(nb.shape[1])]
    for b in ("off", "pos", "offall", "posall"):
        nb = it[f"{b}_n"]
        agg[f"loss_by_{b}"] = (it[f"{b}_loss_sum"].sum(0) / np.maximum(nb.sum(0), 1)).tolist()
        agg[f"dloss_by_{b}"] = ((it[f"{b}_loss_sum"] - it[f"{b}_loss_base_sum"]).sum(0)
                                / np.maximum(nb.sum(0), 1)).tolist()
        agg[f"n_by_{b}"] = nb.sum(0).tolist()
    cont = {}
    for tag in ("in", "out"):
        c = it[f"cont_p1_{tag}_n"].sum(0)
        cont[f"p1_{tag}_by_lag"] = (it[f"cont_p1_{tag}_sum"].sum(0) / np.maximum(c, 1)).tolist()
        cont[f"p1_{tag}_n_by_lag"] = c.tolist()
    for d in ms.CONT_DIRS[:4]:
        cont[f"{d}_all"] = float(it[f"cont_{d}_all_sum"].sum() / max(it[f"cont_{d}_all_n"].sum(), 1))
    cont["hit_rate"] = float(it["cont_hit_sum"].sum() / max(it["cont_hit_n"].sum(), 1))
    agg["contiguity"] = cont
    if "imp_n" in it:
        c = it["imp_n"].sum(0)
        imp = {"kl_by_lag": (it["imp_kl_sum"].sum(0) / np.maximum(c, 1)).tolist(),
               "dloss_by_lag": (it["imp_dloss_sum"].sum(0) / np.maximum(c, 1)).tolist(), "n_by_lag": c.tolist(),
               "burst_kl": float(it["imp_burst_kl_sum"].sum() / max(it["imp_burst_n"].sum(), 1))}
        if "imp_klfull_sum" in it:
            imp["klfull_by_lag"] = (it["imp_klfull_sum"].sum(0) / np.maximum(c, 1)).tolist()
        agg["impulse"] = imp
    g = torch.cat(acc.g) if acc.g else torch.zeros(0)
    gstats = common.gain_stats(g, torch.ones_like(g), 0.0, float(arm["gain"])) if g.numel() else {"n": 0}
    inj = torch.cat(acc.inj) if acc.inj else torch.zeros(0)
    gstats["inj_ratio_mean"] = float(inj.double().mean()) if inj.numel() else None
    agg["signal_means"] = {k: (v[0] / v[1] if v[1] else None) for k, v in sorted(acc.sig.items())}
    agg["gist_drift"] = {k[len("src_gist_drift_"):-4]: (float(it[k].sum() / it[k[:-4] + "_n"].sum())
                                                        if it[k[:-4] + "_n"].sum() > 0 else None)
                         for k in it if k.startswith("src_gist_drift_") and k.endswith("_sum")}
    tm = dict(acc.time)
    tm["overhead_ratio"] = tm["arm_s"] / max(tm["base_s"] / (2 if win else 1), 1e-9)
    tm["tok_per_s_arm"] = tm["tokens"] / max(tm["arm_s"], 1e-9)
    return {"condition": cond, "window": win, "arm": arm["name"], **{k: arm[k] for k in ("kind", "direction",
            "seed", "gain")}, "branch": arm.get("branch"), "complete": True, "agg": agg, "gain_dist": gstats,
            "checks": acc.checks,
            "time": tm, "per_item": it, "edges": {"offset": edges[0], "position": edges[1], "lag": edges[2]}}


def closure_boot(nll_arm, nll_win, nll_full, n, n_boot: int, seed: int) -> dict:
    """C = (L_win − L_arm)/(L_win − L_full), L = Σnll/Σn over sequences; numerator and denominator resampled
    jointly. ``gap_excludes_0`` says whether the denominator's interval excludes 0 (else read the numerator)."""
    a, w, f, n = (np.asarray(x, dtype=np.float64) for x in (nll_arm, nll_win, nll_full, n))
    ok = n > 0
    a, w, f, n = a[ok], w[ok], f[ok], n[ok]
    if n.size == 0:
        return {"C": None, "n_units": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n.size, size=(n_boot, n.size))
    N = n[idx].sum(1)
    num = (w[idx].sum(1) - a[idx].sum(1)) / N
    den = (w[idx].sum(1) - f[idx].sum(1)) / N
    with np.errstate(divide="ignore", invalid="ignore"):
        C = num / den
    def q(x):
        return np.quantile(x[np.isfinite(x)], [0.025, 0.975]).tolist() if np.isfinite(x).any() else [None, None]

    num0 = (w.sum() - a.sum()) / n.sum()
    den0 = (w.sum() - f.sum()) / n.sum()
    gq = q(den)
    return {"C": float(num0 / den0) if den0 != 0 else None, "C_ci": q(C), "num": float(num0), "num_ci": q(num),
            "gap": float(den0), "gap_ci": gq, "gap_excludes_0": bool(gq[0] is not None and (gq[0] > 0 or gq[1] < 0)),
            "n_units": int(n.size)}


# ---------------------------------------------------------------------------------------------- main
def lens_depths(args, n: int) -> list[int]:
    if args.lens_depths:
        return [int(x) % n for x in args.lens_depths.split(",")]
    return sorted({max(0, round(n * f) - 1) for f in (0.25, 0.5, 0.75)} | {max(0, n - 2)})


def compare_to_none(res: ms.ResultFile, cond: str, args) -> None:
    """Paired per-sequence Δloss of every complete arm against none in the same condition (stored in place)."""
    none = res.data["arms"].get(f"{cond}/none")
    if not none or not none.get("complete"):
        return
    pn = none["per_item"]
    Ln = np.asarray(pn["nll_sum"]) / np.maximum(np.asarray(pn["n_tok"]), 1)
    for key, a in res.data["arms"].items():
        if a.get("condition") != cond or not a.get("complete") or a["arm"] == "none":
            continue
        pi = a["per_item"]
        if pi["row"] != pn["row"]:
            continue
        La = np.asarray(pi["nll_sum"]) / np.maximum(np.asarray(pi["n_tok"]), 1)
        a["agg"]["d_loss_vs_none"] = ms.clean(ms.boot(La, Ln, n=args.boot, seed=args.seed))
    res.save()


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    out, trace_root = ms.resolve_paths(args, "text", args.stage)
    model, tok, blocks, kinds = ms.load_model(args)
    n = len(blocks)
    hybrid = ms.is_hybrid(kinds)
    if args.from_ceiling:
        rec = json.loads(Path(args.from_ceiling).expanduser().read_text()).get("recommend", {})
        if args.site is None and rec.get("site") is not None:
            args.site = int(rec["site"])
        gm = rec.get("g_max")
        if gm:
            args.gains = ",".join(f"{x:g}" for x in (gm / 4, gm / 2, gm))
    site = (n // 2 if args.site is None else args.site) % n
    if site >= n - 1:
        raise SystemExit("the site must be below the last block")
    conds = [c for c in args.conditions.split(",") if c]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"unknown conditions {bad}; known {list(CONDITIONS)}")
    skipped = []
    if not hybrid and "all" in conds and "attn" in conds:
        conds.remove("all")
        skipped.append("all: identical to attn on a model without linear-attention layers")
    batches, task = make_batches(args, tok)
    depths = lens_depths(args, n)
    d_model = int(pl.text_config(model).hidden_size)
    facts = ms.model_facts(model, tok, blocks, kinds, args.model if not args.tiny else f"tiny-{args.tiny_arch}")
    header = {"runner": "text", "stage": args.stage, "name": args.name, "args": vars(args), "git": common.git_state(),
              "versions": ms.versions(), "model": facts, "task": task, "site": site,
              "attn_tap_layer": ms.attention_below(kinds, site), "lens_depths": depths,
              "conditions_skipped": skipped, "trace_root": str(trace_root)}
    compat = {k: getattr(args, k) for k in ("model", "tiny", "tiny_arch", "dtype", "seq_len", "sink", "window",
                                            "steer_from", "betas", "seed", "text_file", "max_chars", "batch",
                                            "impulse_start", "impulse_len", "attn_window")}
    compat.update(site=site, n_rows=task["n_rows"])
    res = ms.ResultFile(out, header, compat, force=args.force)
    print(f"[text] {facts['name']} site={site} conds={conds} rows={task['n_rows']} out={out}", flush=True)
    for cond in conds:
        lm = pl.HookedLM(model, side=None, layer=site, window=args.window, mode="block",
                         window_layers=CONDITIONS[cond], sink=args.sink, pad_id=tok.bos_token_id)
        try:
            cinfo = {"window": 0 if cond == "full" else args.window, "window_layers": CONDITIONS[cond],
                     "check_window": pl.check_window(lm) if cond != "full" else "full attention",
                     "windowed_blocks": lm.windowed}
            res.data.setdefault("conditions", {})[cond] = cinfo
            res.save()
            cont_cache: dict = {}
            for arm in make_arms(args, cond):
                key = f"{cond}/{arm['name']}"
                if res.done(key) and not args.force:
                    print(f"[text] skip {key} (complete)", flush=True)
                    continue
                meta = {"runner": "text", "cond": cond, "arm": arm, "site": site, "model": facts["name"],
                        "args": vars(args)}
                w = ms.ArmTraces(trace_root, cond, arm["name"], meta, args.trace_tokens)
                rec = run_arm(arm, cond, lm, model, blocks, kinds, batches, args, d_model, cont_cache, w, depths)
                rec["traces"] = w.close()
                res.put(key, rec)
                agg = rec["agg"]
                print(f"[text] {key}: loss {agg['loss']:.4f} (base {agg['loss_base']:.4f}) "
                      f"kl {agg['kl_base']:.4f} g {rec['gain_dist'].get('mean')} "
                      f"t {rec['time']['arm_s']:.1f}s", flush=True)
            compare_to_none(res, cond, args)
        finally:
            for h in lm.handles:
                h.remove()
    return res.data


if __name__ == "__main__":
    main()
