"""Stage 0b: the gain ceiling of a fixed CA2 injection per direction and site, and the GDN branch-gain sweep.

Residual sweep (``residual``). At each site L (block output; M = L + 1), full attention, teacher-forced text
(the test file, ``--n-seqs`` rows of ``--seq-len`` tokens, steering from text position ``--steer-from``):
    h_t ← h_t + g‖h_t‖d̂_t,   g on the grid 0, 0.025, ..., 2;   d̂ ∈ gist, anti_gist, mean, self, random:0..2.
Each arm is ``text.run_arm`` (every signal traced; outcome per token), so it records Δloss, KL(arm ‖ base),
Δ entropy and |Δ entropy|, the downstream norm ratio ‖h_l(g)‖/‖h_l(0)‖ at every block, attention entropy
at the first full-attention block above the site (``attn_up``) and at the last one (``attn_last``), the
logit lens, contiguity and the gist drift.

Free generation (``gen``). 32-token prefixes of the test file (plus the sink), 512 new tokens sampled at
T = 0.7 with the same seeded noise in every arm (paired), through ``CA2Attachment`` with a KV cache and
``steer_from = "last_prompt"``; for the gen directions at gains ≥ ``--gen-min-gain`` and g = 0. Per item:
collapse (``judge.repetition_onset`` is not None), the onset, distinct-4 (1 − ``judge.degeneration``),
entropy and the model's own NLL of its sample; per token: the whole controller trace and the sampled ids
(trace directory only).

Knee (per site and direction; design_protocol.md "0a. Knee, operationally"; recorded, not a gate):
    g_col  = the smallest gain whose collapse rate exceeds the g = 0 rate by more than ``--collapse-delta``;
    g_sat  = the smallest gain whose final-block norm ratio exceeds ``--sat-norm`` or whose mean |Δ entropy|
             exceeds ``--sat-dent`` nats;
    g_knee = min(g_col, g_sat) (Δloss is not used: the gist may lower loss);
    g_on   = the smallest gain whose mean KL exceeds ``--kl-on``.
``recommend``: g_max = the grid point one step below the gist's knee at the chosen site (the site with the
largest g_knee/g_on; ties to the middle block), g_min = 0. ``text.py --from-ceiling`` reads it.

GDN branch sweep (``branch``; hybrids only; gdn_design.md D4 with the controls of §5(a)). At the GDN layer
sets ``pre`` (directly below an attention block), ``band``, ``all`` and each single layer of ``pre``:
    u_t ← (1 + g)u_t on the GDN branch (``gdn.BranchLaw`` on out_proj), g ∈ {−1, −0.5, −0.25, 0.25, 0.5, 1, 2},
with the same factor on the neighbouring attention branch (``self_attn`` output of the first attention block
above each layer) and on the MLP branch of the same blocks as controls, in full attention and with attention
windowed to ``--branch-window`` (GDN intact). g = −1 is the layer ablation.

    python experiments/ca2/ceiling.py --model Qwen/Qwen3.5-2B-Base \\
        2>&1 | tee results/evolve/round12/stage0/ceiling_q35.log
    python experiments/ca2/ceiling.py --tiny --tiny-arch qwen35          # CPU smoke, < 1 min
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402
from ca2 import common  # noqa: E402
from ca2 import measure as ms  # noqa: E402
from ca2 import text as tx  # noqa: E402
from ca2.controller import CA2Attachment, CA2Controller  # noqa: E402
from ca2.sources import ConstantSource  # noqa: E402

MODES = ("residual", "gen", "branch")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a = p.add_argument
    a("--model", default="Qwen/Qwen3.5-2B-Base")
    a("--dtype", default="bfloat16", choices=list(pl.DTYPES))
    a("--device", default="cuda")
    a("--attn-impl", default="sdpa")
    a("--tiny", action="store_true")
    a("--tiny-arch", default="qwen35", choices=["qwen35", "qwen3", "gpt2"])
    a("--seed", type=int, default=0)
    a("--text-file", default="~/limbic-data/wikitext103_test.txt")
    a("--max-chars", type=int, default=None)
    a("--seq-len", type=int, default=None, help="default 1024; tiny 96")
    a("--n-seqs", type=int, default=None, help="default 64; tiny 4")
    a("--batch", type=int, default=None, help="default 2")
    a("--sink", type=int, default=1)
    a("--steer-from", type=int, default=None, help="first steered text position (default 32; tiny 8)")
    a("--modes", default="residual,gen,branch")
    a("--sites", default=None, help="block outputs (default: hybrid 7,11,12,14,19; Qwen3 7,14,21)")
    a("--directions", default="gist,anti_gist,mean,self,random:0,random:1,random:2")
    a("--gains", default=None, help="default 0,0.025,0.05,0.1,0.2,0.3,0.5,0.75,1,2")
    a("--g-abs", type=float, default=2.0)
    a("--betas", default="0.9,0.995")
    a("--gen-directions", default="gist,random:0,mean,anti_gist")
    a("--gen-min-gain", type=float, default=0.1)
    a("--gen-n", type=int, default=None, help="prompts (default 64; tiny 4)")
    a("--gen-prompt-len", type=int, default=None, help="default 32; tiny 8")
    a("--gen-new", type=int, default=None, help="new tokens (default 512; tiny 16)")
    a("--gen-temp", type=float, default=0.7)
    a("--gen-batch", type=int, default=None, help="default 16; tiny 2")
    a("--branch-gains", default="-1,-0.5,-0.25,0.25,0.5,1,2")
    a("--branch-sets", default="pre,band,all")
    a("--branch-singles", default="pre", help="'pre': each layer of the pre set; a comma list of blocks; '' none")
    a("--branch-controls", default="sets", choices=["sets", "all", "none"],
      help="run the attention and MLP controls for the sets, for sets and singles, or not at all")
    a("--branch-conditions", default="full,attn")
    a("--branch-window", type=int, default=None, help="attention window of the branch sweep (default 256; tiny 16)")
    a("--collapse-delta", type=float, default=0.10)
    a("--sat-norm", type=float, default=1.10)
    a("--sat-dent", type=float, default=0.25)
    a("--kl-on", type=float, default=0.01)
    a("--attn-window", type=int, default=32)
    a("--lens-depths", default=None)
    a("--no-attn-extra", dest="attn_extra", action="store_false")
    a("--gdn-mismatch-n", type=int, default=8, help="rows per arm running the GDN per-token loop (-1: all)")
    a("--gdn-per-head", default="beta,ret,fam,fam_lag,err,str,cos_vp")
    a("--trace-tokens", type=int, default=-1)
    a("--arms", default=None, help="comma list of arm names to run (others skipped)")
    a("--boot", type=int, default=2000)
    a("--name", default=None)
    a("--stage", default="stage0")
    a("--out", default=None)
    a("--trace-dir", default=None)
    a("--force", action="store_true")
    args = p.parse_args(argv)
    tiny = dict(seq_len=96, n_seqs=4, batch=2, steer_from=8, gen_n=4, gen_prompt_len=8, gen_new=16, gen_batch=2,
                branch_window=16, max_chars=200_000, gains="0,0.1,0.5")
    real = dict(seq_len=1024, n_seqs=64, batch=2, steer_from=32, gen_n=64, gen_prompt_len=32, gen_new=512,
                gen_batch=16, branch_window=256, max_chars=None, gains="0,0.025,0.05,0.1,0.2,0.3,0.5,0.75,1,2")
    for k, v in (tiny if args.tiny else real).items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.tiny:
        args.attn_window = min(args.attn_window, args.branch_window, args.steer_from)
        if args.dtype == "bfloat16" and args.device != "cuda":
            args.dtype = "float32"
    bad = [c for c in args.branch_conditions.split(",") if c and c not in tx.CONDITIONS]
    if bad:
        p.error(f"--branch-conditions: unknown {bad}; known {list(tx.CONDITIONS)}")
    # fields text.run_arm reads
    args.impulse_start, args.impulse_len, args.impulse_directions = 0, 1, ""
    return args


def default_sites(kinds) -> list[int]:
    n = len(kinds)
    if ms.is_hybrid(kinds) and n == 24:
        return [7, 11, 12, 14, 19]
    if n == 28:
        return [7, 14, 21]
    return sorted({max(0, n // 4), n // 2, min(n - 2, (3 * n) // 4)})


def task_args(args, window: int) -> argparse.Namespace:
    """A copy of ``args`` for ``text.make_batches`` / ``text.run_arm`` with the given scoring window W (scored
    targets start at sink + W + 1; for the residual sweep W = steer_from, i.e. the first steered target)."""
    t = copy.copy(args)
    t.window = int(window)
    t.directions = args.directions
    return t


# ------------------------------------------------------------------------------------------------ arms
def residual_arms(args) -> list[dict]:
    arms = [{"name": "none", "kind": "none", "direction": "gist", "seed": 0, "gain": 0.0}]
    for spec in [d for d in args.directions.split(",") if d]:
        name, seed = ms.parse_direction(spec)
        for g in tx.floats(args.gains):
            if g > 0:
                arms.append({"name": f"{spec}@{g:g}", "kind": "fixed", "direction": name, "seed": seed, "gain": g})
    return arms


def branch_arms(args, kinds) -> list[dict]:
    sets = ms.gdn_layer_sets(kinds)
    chosen = {s: sets[s] for s in args.branch_sets.split(",") if s and sets.get(s)}
    singles = []
    if args.branch_singles == "pre":
        singles = sets["pre"]
    elif args.branch_singles:
        singles = [int(x) for x in args.branch_singles.split(",")]
    groups = [(k, v, args.branch_controls in ("sets", "all")) for k, v in chosen.items()]
    groups += [(f"L{l}", [l], args.branch_controls == "all") for l in singles]
    arms = [{"name": "none", "kind": "none", "direction": "gist", "seed": 0, "gain": 0.0}]
    for gname, layers, controls in groups:
        targets = [("gdn", layers)]
        if controls:
            targets += [("attn", ms.neighbour_attention(kinds, layers)), ("mlp", layers)]
        for g in tx.floats(args.branch_gains):
            for tgt, ls in targets:
                if not ls:
                    continue
                arms.append({"name": f"branch:{gname}:{tgt}@{g:g}", "kind": "branch", "direction": "gist", "seed": 0,
                             "gain": 0.0, "branch": {"set": gname, "target": tgt, "layers": list(ls), "g": g}})
    return arms


def keep(args, arms: list[dict]) -> list[dict]:
    if not args.arms:
        return arms
    k = set(args.arms.split(","))
    return [a for a in arms if a["name"] in k]


# ------------------------------------------------------------------------------------------ generation
@torch.no_grad()
def run_gen(arm: dict, site: int, model, blocks, kinds, prompts: torch.Tensor, args, d_model: int,
            writer: ms.ArmTraces) -> dict:
    """Sampled continuation with a KV cache; the same seeded noise per batch in every arm (paired)."""
    aj = common.alignment().judge
    dev = model.get_input_embeddings().weight.device
    per_head = [k for k in args.gdn_per_head.split(",") if k]
    n_sub_b = (10 ** 9 if args.gdn_mismatch_n < 0 else -(-args.gdn_mismatch_n // args.gen_batch))
    items: dict[str, list] = {}
    tm = {"gen_s": 0.0, "tokens": 0}
    for bi, a in enumerate(range(0, len(prompts), args.gen_batch)):
        ids = prompts[a:a + args.gen_batch].to(dev)
        B, P = ids.shape
        ctrl = CA2Controller(d_model, source=ConstantSource(1.0), g_min=0.0, g_max=float(arm["gain"]),
                             g_abs=max(args.g_abs, float(arm["gain"])), smooth="none", direction=arm["direction"],
                             betas=tuple(tx.floats(args.betas)), sink_skip=args.sink, steer_from="last_prompt",
                             seed=arm["seed"], massive_window=max(64, args.attn_window), sources=("agree",))
        taps = ms.register_signals(ctrl, blocks, kinds, site, args.attn_window, bi < n_sub_b, per_head)
        att = CA2Attachment(model, ctrl, tap=site, taps=taps)
        gen = torch.Generator(device=dev).manual_seed(args.seed * 1_000_003 + bi)
        ctrl.reset()
        ctrl.record = True
        t0 = ms.timer()
        ent, lp, pmax, new = [], [], [], []
        try:
            out = model(input_ids=ids, use_cache=True)
            cache = out.past_key_values
            for step in range(args.gen_new):
                z = out.logits[:, -1].float()
                lps = torch.log_softmax(z, -1)
                nxt = torch.multinomial(torch.softmax(z / args.gen_temp, -1), 1, generator=gen)
                ent.append(-(lps.exp() * lps).sum(-1))
                lp.append(lps.gather(-1, nxt)[:, 0])
                pmax.append(lps.amax(-1).exp())
                new.append(nxt)
                out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
        finally:
            att.remove()
        tm["gen_s"] += ms.timer() - t0
        tm["tokens"] += B * args.gen_new
        gen_ids = torch.cat(new, 1).cpu()
        ent, lp, pmax = (torch.stack(x, 1).cpu() for x in (ent, lp, pmax))
        tt = ctrl.trace_tensors()
        Ttot = P + args.gen_new
        def pad(x):
            return torch.cat([torch.full((B, P - 1), float("nan")), x, torch.full((B, 1), float("nan"))], 1)

        tok = dict(tt)
        tok["gen/ids"] = torch.cat([ids.cpu(), gen_ids], 1)
        tok["gen/is_gen"] = torch.cat([torch.zeros(B, P), torch.ones(B, args.gen_new)], 1)
        tok["gen/ent"], tok["gen/lp_next"], tok["gen/pmax"] = pad(ent), pad(lp), pad(pmax)
        tok = {k: v for k, v in tok.items() if v.shape[1] == Ttot}
        for b in range(B):
            g_ids = gen_ids[b].tolist()
            onset = aj.repetition_onset(g_ids)
            rec = {"prompt": a + b, "collapse": float(onset is not None), "onset": float("nan") if onset is None
                   else float(onset), "distinct4": 1.0 - aj.degeneration(g_ids, 4), "ent": float(ent[b].mean()),
                   "nll": float(-lp[b].mean()), "n_gen": len(g_ids), "gain_mean": float(tt["g_raw"][b, P - 1:].mean())}
            for k, v in rec.items():
                items.setdefault(k, []).append(v)
            writer.add({**rec, "site": site, "arm": arm["name"]}, ms.row_dict(tok, b), bi < n_sub_b)
    it = {k: np.asarray(v, dtype=np.float64) for k, v in items.items()}
    agg = {"collapse_rate": float(it["collapse"].mean()), "distinct4": float(it["distinct4"].mean()),
           "onset_median": float(np.nanmedian(it["onset"])) if np.isfinite(it["onset"]).any() else None,
           "ent": float(it["ent"].mean()), "nll": float(it["nll"].mean()), "n": int(len(it["collapse"]))}
    tm["tok_per_s"] = tm["tokens"] / max(tm["gen_s"], 1e-9)
    return {"mode": "gen", "site": site, "arm": arm["name"], **{k: arm[k] for k in ("kind", "direction", "seed",
            "gain")}, "complete": True, "agg": agg, "per_item": it, "time": tm}


def gen_prompts(args, tok) -> torch.Tensor:
    """The first ``--gen-prompt-len`` tokens of each of the first ``--gen-n`` rows, after the sink."""
    t = task_args(args, args.steer_from)
    t.n_seqs = args.gen_n
    batches, _ = tx.make_batches(t, tok)
    rows = sorted([(r, ids[i]) for ids, _, rs in batches for i, r in enumerate(rs)], key=lambda x: x[0])
    return torch.stack([x[1][: args.sink + args.gen_prompt_len] for x in rows])


# ------------------------------------------------------------------------------------------------ knee
def first_gain(pairs: list[tuple[float, bool]]) -> float | None:
    for g, hit in sorted(pairs):
        if hit:
            return g
    return None


def knees(res: dict, args) -> dict:
    """Knee per site and direction from the stored aggregates (module docstring)."""
    arms = res["arms"]
    grid = sorted(g for g in tx.floats(args.gains) if g > 0)
    out = {}
    for site in [int(s) for s in res.get("sites", [])]:
        per = {}
        base_gen = arms.get(f"gen/L{site}/none", {}).get("agg", {}).get("collapse_rate")
        for spec in [d for d in args.directions.split(",") if d]:
            sat, on, col = [], [], []
            for g in grid:
                a = arms.get(f"L{site}/full/{spec}@{g:g}")
                if a and a.get("complete"):
                    ag = a["agg"]
                    sat.append((g, (ag.get("norm_ratio_final") or 0) > args.sat_norm or ag["absdent"] > args.sat_dent))
                    on.append((g, ag["kl_base"] > args.kl_on))
                gr = arms.get(f"gen/L{site}/{spec}@{g:g}")
                if gr and gr.get("complete") and base_gen is not None:
                    col.append((g, gr["agg"]["collapse_rate"] - base_gen > args.collapse_delta))
            g_sat, g_on, g_col = first_gain(sat), first_gain(on), first_gain(col)
            cand = [x for x in (g_sat, g_col) if x is not None]
            per[spec] = {"g_sat": g_sat, "g_col": g_col, "g_knee": min(cand) if cand else None, "g_on": g_on,
                         "gen_measured": bool(col), "grid_max": grid[-1] if grid else None}
        out[str(site)] = per
    rec = {"g_min": 0.0}
    best = None
    for site, per in out.items():
        k = per.get("gist", {})
        kn, on = k.get("g_knee"), k.get("g_on")
        if kn is None and on:                        # no knee within the grid: at least grid top / g_on
            ratio = (grid[-1] / on) if grid else None
            ratio = float("inf") if ratio is None else ratio * (1.0 + 1e-9)   # strictly above any in-grid knee
        else:
            ratio = (kn / on) if (kn and on) else (float("inf") if (kn and not on) else None)
        key = (ratio if ratio is not None else -1.0, -abs(int(site) - len(res["model"]["kinds"]) // 2))
        if best is None or key > best[0]:
            best = (key, int(site), kn)
    if best is not None:
        rec["site"] = best[1]
        kn = best[2]
        if kn is None:
            rec["g_max"], rec["note"] = (grid[-1] if grid else None), "no gist knee within the grid: g_max = grid top"
        else:
            below = [g for g in grid if g < kn]
            rec["g_max"] = below[-1] if below else None
            if not below:
                rec["note"] = "the gist's knee is the first grid point; no g_max below it"
        rec["rule"] = "g_max one grid step below the gist knee at the site with the largest g_knee / g_on"
    return {"per_site": out, "recommend": rec}


def deltas_vs_none(res: ms.ResultFile, prefix: str, args) -> None:
    """Paired per-sequence Δloss, Δ|ent|, Δ attention entropy against the prefix's none arm (stored in place)."""
    arms = res.data["arms"]
    none = arms.get(f"{prefix}/none")
    if not none or not none.get("complete"):
        return
    pn = none["per_item"]
    nn_ = np.maximum(np.asarray(pn["n_tok"]), 1)
    for key, a in arms.items():
        if not key.startswith(prefix + "/") or key.endswith("/none") or not a.get("complete"):
            continue
        pi = a["per_item"]
        if pi.get("row") != pn.get("row"):
            continue
        na = np.maximum(np.asarray(pi["n_tok"]), 1)
        d = {"d_loss_vs_none": ms.boot(np.asarray(pi["nll_sum"]) / na, np.asarray(pn["nll_sum"]) / nn_,
                                      n=args.boot, seed=args.seed)}
        for t in ("attn_up", "attn_last"):
            k, c = f"{t}_ent_sum", f"{t}_ent_n"
            if k in pi and k in pn:
                d[f"d_{t}_ent"] = ms.boot(np.asarray(pi[k]) / np.maximum(np.asarray(pi[c]), 1),
                                          np.asarray(pn[k]) / np.maximum(np.asarray(pn[c]), 1), n=args.boot,
                                          seed=args.seed)
        a["agg"].update(ms.clean(d))
    res.save()


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    out, trace_root = ms.resolve_paths(args, "ceiling", args.stage)
    model, tok, blocks, kinds = ms.load_model(args)
    n = len(blocks)
    hybrid = ms.is_hybrid(kinds)
    sites = [int(s) % n for s in args.sites.split(",")] if args.sites else default_sites(kinds)
    if any(s >= n - 1 for s in sites):
        raise SystemExit("sites must be below the last block")
    modes = [m for m in args.modes.split(",") if m]
    notes = []
    if "branch" in modes and not hybrid:
        modes.remove("branch")
        notes.append("branch sweep skipped: the model has no GDN layers")
    d_model = int(pl.text_config(model).hidden_size)
    depths = tx.lens_depths(args, n)
    facts = ms.model_facts(model, tok, blocks, kinds, args.model if not args.tiny else f"tiny-{args.tiny_arch}")
    res_args = task_args(args, args.steer_from)
    batches, task = tx.make_batches(res_args, tok)
    header = {"runner": "ceiling", "stage": args.stage, "name": args.name, "args": vars(args),
              "git": common.git_state(), "versions": ms.versions(), "model": facts, "task": task, "sites": sites,
              "modes": modes, "notes": notes, "lens_depths": depths, "trace_root": str(trace_root),
              "branch_sets": ms.gdn_layer_sets(kinds) if hybrid else None}
    compat = {k: getattr(args, k) for k in ("model", "tiny", "tiny_arch", "dtype", "seq_len", "sink", "steer_from",
                                            "betas", "seed", "text_file", "max_chars", "batch", "attn_window",
                                            "gen_prompt_len", "gen_new", "gen_temp", "gen_batch", "branch_window")}
    compat["n_rows"] = task["n_rows"]
    res = ms.ResultFile(out, header, compat, force=args.force)
    print(f"[ceiling] {facts['name']} sites={sites} modes={modes} rows={task['n_rows']} out={out}", flush=True)

    def run_text_arms(prefix: str, cond: str, lm, arms, targs, bt) -> None:
        cont_cache: dict = {}
        for arm in arms:
            key = f"{prefix}/{cond}/{arm['name']}"
            if res.done(key) and not args.force:
                continue
            meta = {"runner": "ceiling", "key": key, "arm": arm, "model": facts["name"]}
            w = ms.ArmTraces(trace_root, f"{prefix}-{cond}", arm["name"], meta, args.trace_tokens)
            rec = tx.run_arm(arm, cond, lm, model, blocks, kinds, bt, targs, d_model, cont_cache, w, depths)
            rec.update(site=lm.layer, mode=prefix.split("/")[0] if prefix == "branch" else "residual",
                       traces=w.close())
            res.put(key, rec)
            ag = rec["agg"]
            print(f"[ceiling] {key}: dloss {ag['loss'] - ag['loss_base']:+.4f} kl {ag['kl_base']:.4f} "
                  f"|dent| {ag['absdent']:.3f} norm {ag.get('norm_ratio_final')}", flush=True)
        deltas_vs_none(res, f"{prefix}/{cond}", args)

    if "residual" in modes:
        for site in sites:
            lm = pl.HookedLM(model, side=None, layer=site, window=args.steer_from, mode="block",
                             window_layers="attention", sink=args.sink, pad_id=tok.bos_token_id)
            try:
                run_text_arms(f"L{site}", "full", lm, keep(args, residual_arms(args)), res_args, batches)
            finally:
                for h in lm.handles:
                    h.remove()
    if "gen" in modes:
        prompts = gen_prompts(args, tok)
        res.data["gen_prompts"] = {"n": int(len(prompts)), "len": int(prompts.shape[1]), "rows": "first rows"}
        gen_dirs = [d for d in args.gen_directions.split(",") if d]
        for site in sites:
            garms = [{"name": "none", "kind": "none", "direction": "gist", "seed": 0, "gain": 0.0}]
            for spec in gen_dirs:
                name, seed = ms.parse_direction(spec)
                garms += [{"name": f"{spec}@{g:g}", "kind": "fixed", "direction": name, "seed": seed, "gain": g}
                          for g in tx.floats(args.gains) if g >= args.gen_min_gain]
            for arm in keep(args, garms):
                key = f"gen/L{site}/{arm['name']}"
                if res.done(key) and not args.force:
                    continue
                w = ms.ArmTraces(trace_root, f"gen-L{site}", arm["name"], {"runner": "ceiling", "key": key,
                                                                         "arm": arm}, args.trace_tokens)
                rec = run_gen(arm, site, model, blocks, kinds, prompts, args, d_model, w)
                rec["traces"] = w.close()
                res.put(key, rec)
                print(f"[ceiling] {key}: collapse {rec['agg']['collapse_rate']:.3f} "
                      f"distinct4 {rec['agg']['distinct4']:.3f} tok/s {rec['time']['tok_per_s']:.0f}", flush=True)
    if "branch" in modes:
        bargs = task_args(args, args.branch_window)
        bb, btask = tx.make_batches(bargs, tok)
        res.data["branch_task"] = {k: btask[k] for k in ("first_scored", "n_rows")}
        site = n // 2
        for cond in [c for c in args.branch_conditions.split(",") if c]:
            lm = pl.HookedLM(model, side=None, layer=site, window=args.branch_window, mode="block",
                             window_layers=tx.CONDITIONS[cond], sink=args.sink, pad_id=tok.bos_token_id)
            try:
                if cond != "full":
                    res.data.setdefault("branch_conditions", {})[cond] = pl.check_window(lm)
                run_text_arms("branch", cond, lm, keep(args, branch_arms(args, kinds)), bargs, bb)
            finally:
                for h in lm.handles:
                    h.remove()
    res.data.update(ms.clean(knees(res.data, args)))
    res.save()
    print(f"[ceiling] recommend {res.data['recommend']}", flush=True)
    return res.data


if __name__ == "__main__":
    main()
