"""Round-8 diagnostic: does W1j's CA1 novelty mark the facts, and what does it do to the event gate?

W1j (``w1j_ca1.py``) adds κ_h n_t to W1i's event logit, where n_t is the chain store's relative
prediction error against its recent mean. It learned selection in 0 of 5 seeds without the
curriculum and 1 of 5 with it (P40, P41), and κ stayed near its initial 4 in every seed. This
rebuilds checkpoints (or a fresh initialisation) and, on fresh scattered needle batches, reports by
token class:

  classes   d1 (each fact's first digit, where the start must be written), d234 (the other code
            digits), space (the token before d1), head (" The code for NAME is"), filler, query
  per class the relative error rel, novelty n, learned detector logit w_e·x + b, the novelty term
            κ n, the write strength ε as run, and ε with the novelty term removed (κ = 0)
  selectivity  AUC of each signal for d1 against filler (0.5 = no information), mean over heads,
            and d1 novelty by fact rank (the first fact has nothing stored under " is"; later
            facts find the previous fact's code there)

A W1i checkpoint (no κ) gets the same report with the novelty computed from its own chain store,
so the signal can be compared where it was never used.

    python experiments/evolve/diag_novelty.py results/evolve/round8/r8_w1j_s0.json \
        --init results/evolve/round8/r8_w1j_s0.json --out results/evolve/round8/diag_novelty.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402
from evolve.run import ScatterNeedleTask  # noqa: E402
from evolve.w1j_ca1 import CA1Loop  # noqa: E402
from limbic_llm.gated_delta import _gates, _run  # noqa: E402

CLASSES = ("d1", "d234", "space", "head", "filler", "query")


def build_side(rec: dict, d: int, ckpt: Path | None, seed: int):
    mod_name, cls = rec["candidate"]["class"].split(":")
    mod = importlib.import_module(mod_name)
    flags = {**vars(mod.split_args([])[0]), **rec["candidate"]["args"]}
    torch.manual_seed(seed)
    side = getattr(mod, cls)(d, argparse.Namespace(**rec["args"]), argparse.Namespace(**flags))
    if ckpt is not None:
        side.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    return side.cuda().eval()


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """P(pos > neg) by ranks (ties count half)."""
    allv = torch.cat([pos, neg])
    r = allv.argsort().argsort().float() + 1
    rp = r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(rp / (len(pos) * len(neg)))


@torch.no_grad()
def signals(side, h: torch.Tensor) -> dict:
    loop = side.loop
    m = loop.mem
    x = loop.norm(side.time(h)[0] if side.time is not None else h)
    a = m.address(x)
    v = m._heads(m.wv(x), m.dv)
    alpha, beta = _gates(m.w_alpha, m.w_beta, x)
    _, _, err = _run(a, loop._shift(a), v, alpha, beta, None, m.chunk_size)
    rel = (err / (v.pow(2).sum(-1) + 1e-6)).float()
    n = CA1Loop.novelty(loop, err, v) if isinstance(loop, CA1Loop) else \
        CA1Loop.novelty(argparse.Namespace(baseline=0.9), err, v)
    kappa = loop.kappa[None, :, None] if isinstance(loop, CA1Loop) else torch.zeros(1, 1, 1, device=h.device)
    logit = loop.event(x).transpose(1, 2).float()
    return {"rel": rel, "n": n, "logit": logit, "kn": kappa * n,
            "eps": loop.event_strength(x, kappa * n), "eps_k0": loop.event_strength(x)}


def labels(T: int, sink: int, vats: list[int], heads: list[int], n_q: int) -> torch.Tensor:
    lab = torch.full((T,), CLASSES.index("filler"))
    lab[:sink] = -1
    for vat, hd in zip(vats, heads):
        lab[vat - hd:vat] = CLASSES.index("head")
        lab[vat] = CLASSES.index("space")
        lab[vat + 1] = CLASSES.index("d1")
        lab[vat + 2:vat + 5] = CLASSES.index("d234")
        lab[vat + 5] = CLASSES.index("head")                            # the period
    lab[T - n_q:] = CLASSES.index("query")
    return lab


def analyse(lm, side, task, tok, args, natural: bool) -> dict:
    rng = random.Random(args.seed)
    acc = {k: {c: [] for c in CLASSES} for k in ("rel", "n", "logit", "kn", "eps", "eps_k0")}
    d1_by_rank = [[] for _ in range(task.pairs)]
    for _ in range(args.n // args.batch):
        rows, labs = [], []
        for _ in range(args.batch):
            ids, n_ans, _ = task.sample(rng)
            q_len = task.last_query_len + n_ans
            rows.append(ids), labs.append(labels(len(ids), task.sink, task.last_value_at, task.last_heads, q_len))
        ids = torch.tensor(rows)
        hcap = {}
        hook = lm.blocks[lm.layer].register_forward_hook(
            lambda mod, i, o: hcap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).float()))
        with torch.no_grad(), lm.condition(args.window, False):
            lm(ids, keep=1)
        hook.remove()
        s = signals(side, hcap["h"])
        lab = torch.stack(labs).cuda()                                   # (B, T)
        for k, val in s.items():
            for ci, c in enumerate(CLASSES):
                acc[k][c].append(val.permute(1, 0, 2)[:, lab == ci].cpu())                 # (H, N)
        nn_ = s["n"].permute(1, 0, 2)                                    # (H, B, T)
        for i in range(len(rows)):
            vats = [int(v) for v in (lab[i] == CLASSES.index("d1")).nonzero().flatten()]
            for r, t in enumerate(vats):
                d1_by_rank[r].append(nn_[:, i, t].mean().item())
    out = {}
    for k, per in acc.items():
        cat = {c: torch.cat(v, 1) for c, v in per.items()}               # (H, N_c)
        out[k] = {c: round(t.mean().item(), 4) for c, t in cat.items()}
        out[k]["std_filler"] = round(cat["filler"].std().item(), 4)
        H = cat["d1"].shape[0]
        neg = cat["filler"][:, torch.randperm(cat["filler"].shape[1])[:20000]]
        per_head = [auc(cat["d1"][hh], neg[hh]) for hh in range(H)]
        out[k]["auc_d1_vs_filler"] = round(sum(per_head) / H, 3)
        out[k]["auc_d1_vs_filler_by_head"] = [round(x, 3) for x in per_head]
        out[k]["d1_by_head"] = [round(x, 4) for x in cat["d1"].mean(1).tolist()]
        out[k]["filler_by_head"] = [round(x, 4) for x in cat["filler"].mean(1).tolist()]
    out["n_d1_by_fact_rank"] = [round(sum(r) / max(1, len(r)), 4) for r in d1_by_rank]
    out["natural"] = natural
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="*", help="round result JSONs with .pt checkpoints next to them")
    ap.add_argument("--init", nargs="*", default=[], help="result JSONs whose candidate to analyse at initialisation")
    ap.add_argument("--text-file", required=True, help="natural filler (held-out)")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    first = json.loads(Path((args.results + args.init)[0]).read_text())["args"]
    pa = pl.parse_args(["--model", first["model"], "--dtype", "bfloat16", "--device", "cuda"])
    model, tok = pl.load_model(pa)
    d = pl.text_config(model).hidden_size
    corpus = tok.encode(Path(args.text_file).read_text(encoding="utf-8", errors="replace"))
    report = {}
    jobs = [(p, Path(p).with_suffix(".pt"), f"{Path(p).stem}") for p in args.results] + \
           [(p, None, f"{Path(p).stem}@init") for p in args.init]
    for path, ckpt, name in jobs:
        rec = json.loads(Path(path).read_text())
        ra = rec["args"]
        side = build_side(rec, d, ckpt, ra.get("seed", 0))
        args.window = ra["window"]
        lm = pl.HookedLM(model, side, ra["layer"], ra["window"], ra["window_mode"], ra["window_layers"], 1,
                         pad_id=tok.eos_token_id)
        report[name] = {}
        for filler, corp in (("natural", corpus), ("random", None)):
            task = ScatterNeedleTask(tok, ra["seq_len"], ra["pairs"], 1, corp, min_gap=rec["needle_scatter"],
                                     block=ra["window"])
            report[name][filler] = analyse(lm, side, task, tok, args, corp is not None)
        for hd in lm.handles:
            hd.remove()
        r = report[name]["natural"]
        print(name, json.dumps({k: {c: r[k][c] for c in ("d1", "filler", "auc_d1_vs_filler")}
                                for k in ("n", "logit", "kn", "eps", "eps_k0")}), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
