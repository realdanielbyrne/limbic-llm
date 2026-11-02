"""Round-3 diagnostic: does the cue for the first digit carry the queried name?

Rounds 1–2 found that with 2–3 stored codes W1 picks the first digit at chance (digit-1 loss ≈
ln(pairs), register P11/P13), while the successor chain after it works. RV2 (F1) reads this as
Samsonovich & Ascoli's missing-key problem: start selection needs a cue that identifies the
queried fact. This script measures, on the frozen windowed model and without training:

  (a) name decodability: a linear probe for *which name* from the hidden state at
      - the query's space token (the position that predicts digit 1: W1's read cue),
      - each fact's space token (W1's write key for that fact's first digit),
      - the query's name token (the upper bound: the name itself),
      at the given blocks;
  (b) address collapse: with a trained W1 checkpoint, the cosine between the two facts' write
      addresses at the space token, and whether the query's read address is closer to the
      queried fact's address than to the other fact's.

    python experiments/evolve/probe_cue.py --blocks 14 21 --pairs 2 3 \
        --ckpt 14:results/evolve/round1/g1_w1_tied_pairs2.pt \
        --ckpt 21:results/evolve/round2/r2_w1b_late21_pairs2.pt \
        --out results/evolve/round3/probe_cue.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402
from evolve.w1_gdn import W1Side, split_args  # noqa: E402


def sample(task: pl.NeedleTask, rng: random.Random):
    """One needle sequence plus the positions the probe needs (mirrors NeedleTask.sample)."""
    enc, s = task.tok.encode, task.sink
    names = rng.sample(pl.NAMES, task.pairs)
    values = ["".join(rng.choice("0123456789") for _ in range(4)) for _ in names]
    facts, space_at = [], []
    for name, v in zip(names, values):
        head = enc(f" The code for {name} is")
        space_at.append(s + len(facts) + len(head))              # the space token before digit 1
        facts += head + enc(f" {v}") + enc(".")
    j = rng.randrange(task.pairs)
    query, answer = enc(f" The code for {names[j]} is"), enc(f" {values[j]}")
    n_fill = task.seq_len - len(facts) - len(query) - len(answer)
    filler = [task.pool[rng.randrange(len(task.pool))] for _ in range(n_fill)]
    ids = [task.bos] * s + facts + filler + query + answer
    q_space = len(ids) - len(answer)                              # answer's space token
    q_name = len(ids) - len(answer) - len(query) + len(enc(" The code for"))
    return ids, [pl.NAMES.index(n) for n in names], j, space_at, q_space, q_name


def logistic_probe(X: torch.Tensor, y: torch.Tensor, n_cls: int, steps: int = 300, wd: float = 1e-3) -> float:
    """Held-out accuracy of a standardised multinomial logistic probe (80/20 split)."""
    n = len(X)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-5
    Z = (X - mu) / sd
    W = torch.zeros(X.shape[1], n_cls, device=X.device, requires_grad=True)
    b = torch.zeros(n_cls, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Z[tr] @ W + b, y[tr]) + wd * W.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return ((Z[te] @ W + b).argmax(-1) == y[te]).float().mean().item()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--blocks", type=int, nargs="+", default=[14, 21])
    ap.add_argument("--pairs", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--n", type=int, default=2048, help="sequences per pairs setting")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--ckpt", action="append", default=[], metavar="BLOCK:PATH",
                    help="trained W1 side channels tapping BLOCK (address-collapse check)")
    ap.add_argument("--out", default="results/evolve/round3/probe_cue.json")
    args = ap.parse_args(argv)
    torch.manual_seed(0)

    pa = pl.parse_args(["--model", args.model, "--dtype", "bfloat16", "--device", "cuda"])
    model, tok = pl.load_model(pa)
    d = pl.text_config(model).hidden_size
    lm = pl.HookedLM(model, pl.SideChannels(d, memory=False, time=False), None, args.window, "block",
                     "all", 1, pad_id=tok.eos_token_id)
    grab: dict[int, torch.Tensor] = {}
    hooks = [lm.blocks[b].register_forward_hook(
        lambda m, i, o, b=b: grab.__setitem__(b, (o[0] if isinstance(o, tuple) else o).float()))
        for b in args.blocks]
    ckpts = {}
    for spec in args.ckpt:
        blk, path = spec.split(":", 1)
        side = W1Side(d, type("A", (), {"time": True})(), split_args([])[0]).cuda().eval()
        side.load_state_dict(torch.load(path, map_location="cuda"))
        ckpts[int(blk)] = (path, side)

    out = {"args": vars(args), "results": {}}
    for P in args.pairs:
        task = pl.NeedleTask(tok, args.seq_len, P, 1)
        rng = random.Random(1234 + P)
        feats = {b: {"q_space": [], "fact_space": [], "q_name": []} for b in args.blocks}
        labels, fact_labels, coll = [], [], {b: {"fact_cos": [], "query_prefers_target": []} for b in ckpts}
        for _ in range(args.n // args.batch):
            rows = [sample(task, rng) for _ in range(args.batch)]
            ids = torch.tensor([r[0] for r in rows])
            with torch.no_grad(), lm.condition(args.window, False):
                lm(ids)
            ar = torch.arange(len(rows))
            qs = torch.tensor([r[4] for r in rows])
            qn = torch.tensor([r[5] for r in rows])
            fs = torch.tensor([r[3] for r in rows])                # (B, P)
            for b in args.blocks:
                h = grab[b]
                feats[b]["q_space"].append(h[ar, qs].cpu())
                feats[b]["q_name"].append(h[ar, qn].cpu())
                feats[b]["fact_space"].append(h[ar[:, None], fs].reshape(-1, d).cpu())
            labels += [r[1][r[2]] for r in rows]
            fact_labels += [n for r in rows for n in r[1]]
            for b, (_, side) in ckpts.items():
                with torch.no_grad():
                    x = side.loop.norm(side.time(grab[b])[0]) if side.time is not None else side.loop.norm(grab[b])
                    a = side.loop.mem.address(x)                   # (B, H, T, Dk)
                    af = a[ar[:, None], :, fs]                     # (B, P, H, Dk)
                    aq = a[ar, :, qs]                              # (B, H, Dk)
                    tgt = torch.tensor([r[2] for r in rows], device=a.device)
                    coll[b]["fact_cos"] += (af[:, 0] * af[:, 1]).sum(-1).mean(-1).tolist()
                    sim = (af * aq[:, None]).sum(-1).mean(-1)      # (B, P)
                    best_other = sim.masked_fill(F.one_hot(tgt, P).bool(), -9).max(-1).values
                    coll[b]["query_prefers_target"] += (sim[ar.to(a.device), tgt] > best_other).float().tolist()
        y = torch.tensor(labels).cuda()
        yf = torch.tensor(fact_labels).cuda()
        res = {}
        for b in args.blocks:
            X = {k: torch.cat(v).cuda() for k, v in feats[b].items()}
            res[f"block{b}"] = {
                "probe_name_at_query_space": logistic_probe(X["q_space"], y, len(pl.NAMES)),
                "probe_name_at_fact_space": logistic_probe(X["fact_space"], yf, len(pl.NAMES)),
                "probe_name_at_query_name": logistic_probe(X["q_name"], y, len(pl.NAMES)),
            }
        for b, (path, _) in ckpts.items():
            res[f"w1_addresses_block{b}"] = {
                "checkpoint": path,
                "mean_cos_between_fact_addresses": sum(coll[b]["fact_cos"]) / len(coll[b]["fact_cos"]),
                "query_address_closer_to_queried_fact": sum(coll[b]["query_prefers_target"]) / len(coll[b]["query_prefers_target"]),
            }
        res["chance"] = {"name": 1 / len(pl.NAMES), "queried_fact": 1 / P}
        out["results"][f"pairs{P}"] = res
        print(f"pairs {P}: {json.dumps(res, indent=1)}", flush=True)

    for hk in hooks:
        hk.remove()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
