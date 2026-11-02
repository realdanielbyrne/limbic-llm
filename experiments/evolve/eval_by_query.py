"""Evaluate a trained candidate by *which* fact was queried, and report W1i's learned events.

Mean digit-1 accuracy cannot tell content addressing from positional shortcuts: a memory of only
the first and the last fact scores 2/3 with 3 pairs. This rebuilds a run's candidate from its
result JSON (class, flags, tap, pairs, scattered or fixed layout), loads its ``.pt`` checkpoint,
and on fresh batches reports, by the queried fact's rank (first, middle, last):

  digit-1 accuracy and exact match of the 4-digit code, with the side channels on, and the same for
  the windowed base (side channels off) and full attention as controls (``--controls``);
  for an EventLoop candidate (W1i): the learned write strength ε (mean over all tokens, and at
  each fact's first digit), the evidence on whether the events became sparse (review w1i-3).

Evaluation can differ from training (rule E6): ``--pairs``, ``--seq-len`` (a multiple of the window)
and ``--text-file`` (natural filler) change the task at evaluation only. Only the logits of the
answer positions are computed, so long sequences fit in memory.

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


def _score(lg: torch.Tensor, ids: torch.Tensor, n_ans: int) -> list[list[bool]]:
    """Per sequence, whether each answer token is predicted. lg: logits of the last n_ans + 1 positions."""
    pred = lg.argmax(-1).cpu()                                     # (B, n_ans + 1)
    T = ids.shape[1]
    return [[pred[i, k].item() == ids[i, T - n_ans + k].item() for k in range(n_ans)] for i in range(len(ids))]


def _summarise(per_seq: list, P: int, edges: list[int]) -> dict:
    """Accuracy by queried rank and by gap bin from (rank, gap, digit-1 ok, exact ok) tuples."""
    by_rank = {"digit1": [], "exact": [], "n": []}
    for k in range(P):
        sel = [r for r in per_seq if r[0] == k]
        by_rank["n"].append(len(sel))
        by_rank["digit1"].append(sum(r[2] for r in sel) / max(1, len(sel)))
        by_rank["exact"].append(sum(r[3] for r in sel) / max(1, len(sel)))
    bins = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in per_seq if lo < r[1] <= hi]
        if sel:
            key = f"{lo + 1 if lo >= 0 else 0}-{hi if hi < 10 ** 9 else 'max'}"
            bins[key] = {"n": len(sel), "digit1": sum(r[2] for r in sel) / len(sel),
                         "exact": sum(r[3] for r in sel) / len(sel),
                         "by_rank": {str(k): {"n": sum(1 for r in sel if r[0] == k),
                                              "exact": sum(r[3] for r in sel if r[0] == k)
                                              / max(1, sum(1 for r in sel if r[0] == k))} for k in range(P)}}
    n = max(1, len(per_seq))
    return {"digit1": sum(r[2] for r in per_seq) / n, "exact": sum(r[3] for r in per_seq) / n,
            "digit1_by_queried_rank": by_rank["digit1"], "exact_by_queried_rank": by_rank["exact"],
            "count_by_queried_rank": by_rank["n"], "by_gap": bins}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("result", help="a round result JSON with a .pt checkpoint next to it")
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--pairs", type=int, default=None, help="evaluate with this many stored pairs (default: as trained)")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="evaluation sequence length, a multiple of the window (default: as trained)")
    ap.add_argument("--text-file", default=None, help="natural-text filler at evaluation only (default: as trained)")
    ap.add_argument("--random-filler", action="store_true",
                    help="random-token filler even if the run was trained with --text-file")
    ap.add_argument("--controls", action=argparse.BooleanOptionalAction, default=True,
                    help="also score the windowed base (side channels off) and full attention")
    ap.add_argument("--gap-bins", type=int, nargs="+", default=None,
                    help="bin edges for accuracy against the queried fact's gap (default: 4 even bins above 256)")
    ap.add_argument("--out", default=None, help="default: <result>.by_query.json")
    args = ap.parse_args(argv)
    rec = json.loads(Path(args.result).read_text())
    ra, cand = rec["args"], rec.get("candidate")
    pa = pl.parse_args(["--model", ra["model"], "--dtype", "bfloat16", "--device", "cuda"])
    model, tok = pl.load_model(pa)
    d = pl.text_config(model).hidden_size
    if cand:
        mod_name, cls = cand["class"].split(":")
        mod = importlib.import_module(mod_name)
        Cand = getattr(mod, cls)
        flags = {**vars(mod.split_args([])[0]), **cand["args"]} if hasattr(mod, "split_args") else cand["args"]
        side = Cand(d, argparse.Namespace(**ra), argparse.Namespace(**flags))   # flags added later: defaults
    else:
        side = pl.SideChannels(d, memory=ra["memory"], time=ra["time"], **pl.loop_kwargs(argparse.Namespace(**ra)))
    state = torch.load(Path(args.result).with_suffix(".pt"), map_location="cpu")
    missing, unexpected = side.load_state_dict(state, strict=False)
    side = side.cuda().eval()
    W = ra["window"]
    lm = pl.HookedLM(model, side, ra["layer"], W, ra["window_mode"], ra["window_layers"], 1, pad_id=tok.eos_token_id)
    P = args.pairs or ra["pairs"]
    L = args.seq_len or ra["seq_len"]
    if ra["window_mode"] in ("block", "segment") and L % W:
        raise SystemExit(f"--seq-len {L} must be a multiple of the window {W}")
    corpus = None
    text_file = None if args.random_filler else (args.text_file or ra.get("text_file"))
    if text_file:
        corpus = tok.encode(Path(text_file).read_text(encoding="utf-8", errors="replace"))
    scatter = rec.get("needle_scatter", 0)
    task = (ScatterNeedleTask(tok, L, P, 1, corpus, min_gap=scatter, block=W) if scatter
            else pl.NeedleTask(tok, L, P, 1, corpus))
    edges = [-1] + (args.gap_bins or [256 + i * (L - 256) // 5 for i in range(1, 5)]) + [10 ** 9]
    conds = {"windowed+side": (W, True)}
    if args.controls:
        conds.update({"windowed": (W, False), "full": (0, False)})
    rng = random.Random(args.seed)
    per_seq = {c: [] for c in conds}                               # (rank, gap, digit-1 ok, exact ok)
    eps_all, eps_d1 = [], []
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
        for c, (w, s_on) in conds.items():
            with torch.no_grad(), lm.condition(w, s_on):
                ok = _score(lm(ids, keep=n_ans + 1), ids, n_ans)  # only the answer positions' logits
            for i, j in enumerate(js):                             # ok[i][0] is the space; [1] is digit 1
                per_seq[c].append((j, gaps[i], float(ok[i][1]), float(all(ok[i]))))
        if events:
            with torch.no_grad():
                h_tap = {}
                hook = lm.blocks[lm.layer].register_forward_hook(
                    lambda m, i, o: h_tap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).float()))
                with lm.condition(W, False):
                    lm(ids, keep=1)
                hook.remove()
                x = side.loop.norm(side.time(h_tap["h"])[0] if side.time is not None else h_tap["h"])
                eps = side.loop.event_strength(x).cpu()           # (B, H, T)
            eps_all.append(eps.mean().item())
            if scatter:
                d1 = [[v + 1 for v in vat] for vat in vals]       # value_at is the space token; +1 → digit 1
                eps_d1.append(torch.stack([eps[i, :, d1[i]].mean() for i in range(len(d1))]).mean().item())
    main_ = _summarise(per_seq["windowed+side"], P, edges)
    out = {"result": args.result, "pairs": P, "seq_len": L, "text_file": text_file, "scatter": scatter,
           "n": len(per_seq["windowed+side"]),
           # the side-channel model (kept at the top level, as in earlier rounds)
           "digit1_acc_by_queried_rank": main_["digit1_by_queried_rank"],
           "exact_by_queried_rank": main_["exact_by_queried_rank"],
           "count_by_queried_rank": main_["count_by_queried_rank"],
           "digit1": main_["digit1"], "exact": main_["exact"], "by_gap": main_["by_gap"],
           "controls": {c: _summarise(per_seq[c], P, edges) for c in conds if c != "windowed+side"},
           "per_sequence": per_seq["windowed+side"],
           "missing_keys": list(missing), "unexpected_keys": list(unexpected)}
    if events:
        out["eps_mean_all_tokens"] = sum(eps_all) / len(eps_all)
        if eps_d1:
            out["eps_mean_at_fact_digit1"] = sum(eps_d1) / len(eps_d1)
    path = Path(args.out or Path(args.result).with_suffix(".by_query.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    summary = {k: out[k] for k in ("pairs", "seq_len", "exact", "digit1", "digit1_acc_by_queried_rank")}
    summary["controls_exact"] = {c: round(v["exact"], 3) for c, v in out["controls"].items()}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
