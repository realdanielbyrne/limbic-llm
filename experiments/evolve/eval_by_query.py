"""Evaluate a trained candidate by *which* fact was queried, and report W1i's learned events.

Mean digit-1 accuracy cannot tell content addressing from positional shortcuts: a memory of only
the first and the last fact scores 2/3 with 3 pairs. This rebuilds a run's candidate from its
result JSON (class, flags, tap, pairs, scattered or fixed layout), loads its ``.pt`` checkpoint,
and on fresh batches reports, by the queried fact's rank (first, middle, last):

  digit-1 accuracy and exact match of the 4-digit code, with the side channels on;
  for an EventLoop candidate (W1i): the learned write strength ε (mean over all tokens, and at
  each fact's first digit), the evidence on whether the events became sparse (review w1i-3).

    python experiments/evolve/eval_by_query.py results/evolve/round4/r4_w1i_t21_pairs3_scatter.json
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("result", help="a round result JSON with a .pt checkpoint next to it")
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--pairs", type=int, default=None, help="evaluate with this many stored pairs (default: as trained)")
    ap.add_argument("--gap-bins", type=int, nargs="+", default=[300, 350, 400, 450],
                    help="bin edges for accuracy against the queried fact's gap")
    ap.add_argument("--out", default=None, help="default: <result>.by_query.json")
    args = ap.parse_args(argv)
    rec = json.loads(Path(args.result).read_text())
    ra, cand = rec["args"], rec.get("candidate")
    pa = pl.parse_args(["--model", ra["model"], "--dtype", "bfloat16", "--device", "cuda"])
    model, tok = pl.load_model(pa)
    d = pl.text_config(model).hidden_size
    if cand:
        mod_name, cls = cand["class"].split(":")
        Cand = getattr(importlib.import_module(mod_name), cls)
        side = Cand(d, argparse.Namespace(**ra), argparse.Namespace(**cand["args"]))
    else:
        side = pl.SideChannels(d, memory=ra["memory"], time=ra["time"], **pl.loop_kwargs(argparse.Namespace(**ra)))
    state = torch.load(Path(args.result).with_suffix(".pt"), map_location="cpu")
    missing, unexpected = side.load_state_dict(state, strict=False)
    side = side.cuda().eval()
    lm = pl.HookedLM(model, side, ra["layer"], ra["window"], ra["window_mode"], ra["window_layers"], 1,
                     pad_id=tok.eos_token_id)
    P = args.pairs or ra["pairs"]
    scatter = rec.get("needle_scatter", 0)
    task = (ScatterNeedleTask(tok, ra["seq_len"], P, 1, min_gap=scatter, block=ra["window"]) if scatter
            else pl.NeedleTask(tok, ra["seq_len"], P, 1))
    rng = random.Random(args.seed)
    stats = torch.zeros(P, 3)                                      # per queried rank: digit-1 ok, exact ok, n
    eps_all, eps_d1 = [], []
    per_seq = []                                                   # (rank, gap, digit-1 ok, exact ok)
    events = hasattr(side.loop, "event_strength")
    for _ in range(args.n // args.batch):
        rows, js, vals, gaps = [], [], [], []
        for _ in range(args.batch):
            ids, n_ans, gap = task.sample(rng)
            gaps.append(gap)
            if scatter:
                j, vat = task.last_queried, task.last_value_at
            else:                                                  # fixed layout: recover j from the text
                text = tok.tok.decode(ids)
                names = [n for n in pl.NAMES if f"for {n} is" in text]
                q = text.rsplit("The code for ", 1)[1].split(" is")[0]
                order = sorted(names, key=lambda n: text.index(f"for {n} is"))
                j, vat = order.index(q), None
            rows.append(ids), js.append(j), vals.append(vat)
        ids = torch.tensor(rows)
        with torch.no_grad(), lm.condition(ra["window"], True):
            lg = lm(ids)
        pred = lg.argmax(-1).cpu()
        T = ids.shape[1]
        for i, j in enumerate(js):
            first = T - n_ans                                     # position of the answer's first token
            ok = [pred[i, p - 1].item() == ids[i, p].item() for p in range(first, T)]
            stats[j, 0] += float(ok[1])                           # ok[0] is the space; ok[1] digit 1
            stats[j, 1] += float(all(ok))
            stats[j, 2] += 1
            per_seq.append((j, gaps[i], float(ok[1]), float(all(ok))))
        if events:
            with torch.no_grad():
                h_tap = {}
                hook = lm.blocks[lm.layer].register_forward_hook(
                    lambda m, i, o: h_tap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).float()))
                with lm.condition(ra["window"], False):
                    lm(ids)
                hook.remove()
                x = side.loop.norm(side.time(h_tap["h"])[0] if side.time is not None else h_tap["h"])
                eps = side.loop.event_strength(x).cpu()           # (B, H, T)
            eps_all.append(eps.mean().item())
            if scatter:
                d1 = [[v + 1 for v in vat] for vat in vals]       # value_at is the space token; +1 → digit 1
                eps_d1.append(torch.stack([eps[i, :, d1[i]].mean() for i in range(len(d1))]).mean().item())
    out = {"result": args.result, "pairs": P, "scatter": scatter, "n": int(stats[:, 2].sum()),
           "digit1_acc_by_queried_rank": (stats[:, 0] / stats[:, 2]).tolist(),
           "exact_by_queried_rank": (stats[:, 1] / stats[:, 2]).tolist(),
           "count_by_queried_rank": stats[:, 2].tolist(),
           "missing_keys": list(missing), "unexpected_keys": list(unexpected)}
    edges = [-1] + args.gap_bins + [10 ** 9]
    bins = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in per_seq if lo < r[1] <= hi]
        if sel:
            key = f"{lo + 1 if lo >= 0 else 0}-{hi if hi < 10 ** 9 else 'max'}"
            bins[key] = {"n": len(sel), "digit1": sum(r[2] for r in sel) / len(sel), "exact": sum(r[3] for r in sel) / len(sel),
                         "by_rank": {str(k): {"n": sum(1 for r in sel if r[0] == k),
                                              "exact": (sum(r[3] for r in sel if r[0] == k) / max(1, sum(1 for r in sel if r[0] == k)))}
                                     for k in range(P)}}
    out["by_gap"] = bins
    out["per_sequence"] = per_seq
    if events:
        out["eps_mean_all_tokens"] = sum(eps_all) / len(eps_all)
        if eps_d1:
            out["eps_mean_at_fact_digit1"] = sum(eps_d1) / len(eps_d1)
    path = Path(args.out or Path(args.result).with_suffix(".by_query.json"))
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
