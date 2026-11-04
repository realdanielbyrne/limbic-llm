"""Evaluate a trained candidate by *which* fact was queried, and report what its gate learned.

Mean digit-1 accuracy cannot tell content addressing from positional shortcuts: a memory of only
the first and the last fact scores 2/3 with 3 pairs. This rebuilds a run's candidate from its
result JSON (class, flags, tap, pairs, scattered or fixed layout), loads its ``.pt`` checkpoint,
and on fresh batches reports, by the queried fact's rank (first, middle, last):

  digit-1 accuracy and exact match of the 4-digit code, with the side channels on, and the same for
  the windowed base (side channels off) and full attention as controls (``--controls``);
  for an EventLoop candidate (W1i, W1j): the learned write strength ε (mean over all tokens, and at
  each fact's first digit), the evidence on whether the events became sparse (review w1i-3). ε is
  the gate as run (B17): on the side's input (the sink dropped), and for W1j with its CA1 novelty
  term, recorded in the side's own forward;
  for a slot candidate (W1s), from ``side.loop.last_diag`` recorded in its own forward: fact
  coverage (the queried fact has a retained slot at the query: an onset inside the fact), slot hit
  (the most-read slot is that one; null for the superposed store, which reads no slot), alignment
  hit (the aligned read lands on the queried first digit; 0 by construction for reinstatement and
  ``--w1s-align none``, which align nothing), the null slot's weight, and the gate's AUC for first
  digits against filler tokens.

Variants of the task test generalisation (all need a scattered-layout run):
  --jitter         facts take one of three phrasings; the query keeps the plain one
  --update FORM    one name is stated again with a new code; the answer is the new code, and the
                   stale (old) code is scored too: digit 1 and exact, NEW against OLD
  --heldout-names  names from a list disjoint from ``pl.NAMES`` (never seen in training)
  --w1s-ctrl       W1s controls: ``empty`` forces the store empty (must score 0.00), ``shuffle``
                   permutes the retained slots across the batch's rows (about chance; additive
                   output only, and --batch ≥ 2)

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
from evolve.run import TRAIN_NAMES, ScatterNeedleTask, UpdateNeedleTask, oracle_gate, oracle_onsets  # noqa: E402

HELDOUT_NAMES = ["Adam", "Beth", "Chris", "Diana", "Ethan", "Fiona", "George", "Hannah", "Ivan", "Julia",
                 "Kevin", "Laura", "Mark", "Nina", "Oscar", "Paula", "Ryan", "Sarah", "Tom", "Uma"]


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


def _auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """P(pos > neg), ties counted half: the gate's ranking of first digits above filler."""
    neg = neg.sort().values
    below = torch.searchsorted(neg, pos, right=False) + torch.searchsorted(neg, pos, right=True)
    return (below.double() / 2).mean().item() / max(1, len(neg))


def _slot_diag(diag: dict, js, vals, spans, qlens, T: int, n_ans: int, sink: int, acc: dict) -> None:
    """Accumulate W1s's diagnostics for one batch. The query is the answer's space (it predicts
    digit 1); positions are side coordinates (the sink dropped), as in ``last_diag``."""
    p, on = diag["p"].float().cpu(), diag["onsets"].cpu()
    ret, rw, al = diag["retained"].cpu(), diag["read_w"].float().cpu(), diag["align"].cpu()
    K, tq = ret.shape[-1], T - n_ans - sink
    for b, j in enumerate(js):
        sp = [(a - sink, e - sink) for a, e in spans[b]]
        d1 = [v + 1 - sink for v in vals[b]]
        r = ret[b, tq]
        in_q = (r >= sp[j][0]) & (r <= sp[j][1])                   # retained onsets inside the queried fact
        k, a = int(rw[b, tq].argmax()), int(al[b, tq])
        hit = float(k < K and bool(in_q[k])) if rw[b, tq].sum() > 0 else None   # None: no slot read (superposed)
        acc["seq"].append((j, float(in_q.any()), hit, float(a == d1[j])))
        acc["align_within1"].append(float(a >= 0 and abs(a - d1[j]) <= 1))
        acc["null_weight"].append(rw[b, tq, -1].item() if hit is not None else None)
        fill = torch.ones(p.shape[1], dtype=torch.bool)
        for lo, hi in sp:
            fill[lo:hi + 1] = False
        fill[T - sink - qlens[b] - n_ans:] = False                  # the query and its answer
        acc["p_d1"].append(p[b, d1]), acc["p_fill"].append(p[b, fill])
        acc["onset_in_fact"] += [float(on[b, lo:hi + 1].any()) for lo, hi in sp]
        acc["density"].append(on[b].float().mean().item())


def _slot_summary(acc: dict, R: int) -> dict:
    def mean(v):                                                   # None entries are not measured
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None
    out = {}
    for i, name in ((1, "fact_coverage"), (2, "slot_hit"), (3, "align_hit")):
        out[name] = mean([r[i] for r in acc["seq"]])
        out[f"{name}_by_queried_rank"] = [mean([r[i] for r in acc["seq"] if r[0] == k]) for k in range(R)]
    pos, neg = torch.cat(acc["p_d1"]), torch.cat(acc["p_fill"])
    out.update({"align_within1": mean(acc["align_within1"]), "null_weight_at_query": mean(acc["null_weight"]),
                "onset_density": mean(acc["density"]), "onset_in_fact": mean(acc["onset_in_fact"]),
                "p_mean_digit1": pos.mean().item(), "p_mean_filler": neg.mean().item(),
                "gate_auc_digit1_vs_filler": _auc(pos, neg)})
    return out


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
    ap.add_argument("--jitter", action="store_true", help="facts in one of three phrasings (query plain)")
    ap.add_argument("--update", choices=["same", "now", "new"], default=None,
                    help="a name restated with a new code (phrasing: same template, ' is now', 'The new code')")
    ap.add_argument("--heldout-names", action="store_true", help="names never used in training (not in pl.NAMES)")
    ap.add_argument("--w1s-ctrl", choices=["none", "empty", "shuffle"], default="none",
                    help="W1s control: an empty store, or retained slots shuffled across rows")
    ap.add_argument("--out", default=None,
                    help="default: <result>.by_query.json (a variant adds its name, e.g. .by_query.jitter.json)")
    args = ap.parse_args(argv)
    rec = json.loads(Path(args.result).read_text())
    ra, cand = rec["args"], rec.get("candidate")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if ra.get("tiny"):                                             # random small model, rebuilt from its seed
        pa = pl.parse_args(["--tiny", "--tiny-arch", ra["tiny_arch"], "--seed", str(ra["seed"]),
                            "--dtype", ra["dtype"], "--device", dev, "--attn-impl", ra["attn_impl"]])
    else:
        pa = pl.parse_args(["--model", ra["model"], "--dtype", "bfloat16", "--device", dev,
                            "--attn-impl", ra.get("attn_impl", "sdpa")])
    model, tok = pl.load_model(pa)
    d = pl.text_config(model).hidden_size
    flags = {}
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
    side = side.to(dev).eval()
    oracle = oracle_gate(argparse.Namespace(**flags))
    if args.w1s_ctrl != "none":
        if not hasattr(side, "ctrl"):
            raise SystemExit("--w1s-ctrl needs a candidate with a ctrl attribute (W1s)")
        if args.w1s_ctrl == "shuffle" and args.batch < 2:
            raise SystemExit("--w1s-ctrl shuffle needs --batch ≥ 2 (a row of its own reads its own store)")
        try:
            side.ctrl = args.w1s_ctrl
        except ValueError as e:                                    # shuffle under reinstatement
            raise SystemExit(str(e))
    W = ra["window"]
    LM = pl.HookedLM
    if hasattr(side, "taps"):                                      # W1s reads earlier blocks too
        from evolve.taps import TapLM
        LM = TapLM
    lm = LM(model, side, ra["layer"], W, ra["window_mode"], ra["window_layers"], 1, pad_id=tok.eos_token_id)
    P = args.pairs or ra["pairs"]
    L = args.seq_len or ra["seq_len"]
    if ra["window_mode"] in ("block", "segment") and L % W:
        raise SystemExit(f"--seq-len {L} must be a multiple of the window {W}")
    corpus = None
    text_file = None if args.random_filler else (args.text_file or ra.get("text_file"))
    if text_file:
        corpus = tok.encode(Path(text_file).read_text(encoding="utf-8", errors="replace"))
    scatter = rec.get("needle_scatter", 0)
    if (args.jitter or args.update or args.heldout_names or oracle) and not scatter:
        raise SystemExit("--jitter, --update, --heldout-names and an oracle gate need a scattered-layout run")
    if args.jitter and args.update:
        raise SystemExit("--jitter and --update are separate tests")
    pool = HELDOUT_NAMES if args.heldout_names else TRAIN_NAMES if rec.get("needle_names") == "train" else None
    if args.update:
        task = UpdateNeedleTask(tok, L, P, 1, corpus, min_gap=scatter, block=W, names=pool, form=args.update)
    elif scatter:
        task = ScatterNeedleTask(tok, L, P, 1, corpus, min_gap=scatter, block=W, names=pool, jitter=args.jitter)
    else:
        task = pl.NeedleTask(tok, L, P, 1, corpus)
    R = P + 1 if args.update else P                                # statements, so queried ranks
    edges = [-1] + (args.gap_bins or [256 + i * (L - 256) // 5 for i in range(1, 5)]) + [10 ** 9]
    conds = {"windowed+side": (W, True)}
    if args.controls:
        conds.update({"windowed": (W, False), "full": (0, False)})
    rng = random.Random(args.seed)
    per_seq = {c: [] for c in conds}                               # (rank, gap, digit-1 ok, exact ok)
    upd = {c: [] for c in conds}                                   # (new digit 1, old digit 1, new exact, old exact)
    eps_all, eps_d1 = [], []
    slots = {"seq": [], "align_within1": [], "null_weight": [], "p_d1": [], "p_fill": [], "onset_in_fact": [],
             "density": []}
    events = hasattr(side.loop, "event_strength") and not hasattr(side, "taps")   # never W1s (B17)
    for _ in range(args.n // args.batch):
        rows, js, vals, gaps, spans, qlens, olds = [], [], [], [], [], [], []
        for _ in range(args.batch):
            ids, n_ans, gap = task.sample(rng)
            gaps.append(gap)
            if scatter:
                j, vat = task.last_queried, task.last_value_at
                spans.append(task.last_spans), qlens.append(task.last_query_len)
                if args.update:                                    # the same sequence answered with the old code
                    if len(task.last_old_answer) != n_ans:
                        raise SystemExit("old and new codes tokenise to different lengths")
                    olds.append(ids[:-n_ans] + task.last_old_answer)
            else:                                                  # fixed layout: recover j from the text
                text = tok.tok.decode(ids)
                names = [n for n in pl.NAMES if f"for {n} is" in text]
                q = text.rsplit("The code for ", 1)[1].split(" is")[0]
                order = sorted(names, key=lambda n: text.index(f"for {n} is"))
                j, vat = order.index(q), None
            rows.append(ids), js.append(j), vals.append(vat)
        ids = torch.tensor(rows)
        if oracle:                                                 # diagnostic gate: the true first digits
            side.oracle_onsets = oracle_onsets(vals, ids.shape[1], 1, dev)
        diag = nov = None
        for c, (w, s_on) in conds.items():
            with torch.no_grad(), lm.condition(w, s_on):
                ok = _score(lm(ids, keep=n_ans + 1), ids, n_ans)  # only the answer positions' logits
            if s_on:                                               # recorded in the side's own forward
                diag = getattr(side.loop, "last_diag", None)
                nov = getattr(side.loop, "last_novelty", None)
            for i, j in enumerate(js):                             # ok[i][0] is the space; [1] is digit 1
                per_seq[c].append((j, gaps[i], float(ok[i][1]), float(all(ok[i]))))
            if args.update:                                        # digit 1 is predicted before either code
                old = torch.tensor(olds)
                with torch.no_grad(), lm.condition(w, s_on):
                    ok_old = _score(lm(old, keep=n_ans + 1), old, n_ans)
                upd[c] += [(float(ok[i][1]), float(ok_old[i][1]), float(all(ok[i])), float(all(ok_old[i])))
                           for i in range(len(js))]
        if isinstance(diag, dict) and scatter:
            _slot_diag(diag, js, vals, spans, qlens, ids.shape[1], n_ans, 1, slots)
        if events:
            with torch.no_grad():
                h_tap = {}
                hook = lm.blocks[lm.layer].register_forward_hook(
                    lambda m, i, o: h_tap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).float()))
                with lm.condition(W, False):
                    lm(ids, keep=1)
                hook.remove()
                h = h_tap["h"][:, 1:]                              # the sink dropped, as the side sees it (B17)
                x = side.loop.norm(side.time(h)[0] if side.time is not None else h)
                extra = None                                       # W1j: the CA1 novelty in the logit (B17)
                if nov is not None and hasattr(side.loop, "kappa"):
                    extra = side.loop.kappa[None, :, None] * nov
                eps = side.loop.event_strength(x, extra).cpu()    # (B, H, T), side coordinates
            eps_all.append(eps.mean().item())
            if scatter:
                d1 = [list(vat) for vat in vals]                  # value_at is the space (with the sink): − 1 + 1
                eps_d1.append(torch.stack([eps[i, :, d1[i]].mean() for i in range(len(d1))]).mean().item())
    main_ = _summarise(per_seq["windowed+side"], R, edges)
    out = {"result": args.result, "pairs": P, "seq_len": L, "text_file": text_file, "scatter": scatter,
           "n": len(per_seq["windowed+side"]),
           # the side-channel model (kept at the top level, as in earlier rounds)
           "digit1_acc_by_queried_rank": main_["digit1_by_queried_rank"],
           "exact_by_queried_rank": main_["exact_by_queried_rank"],
           "count_by_queried_rank": main_["count_by_queried_rank"],
           "digit1": main_["digit1"], "exact": main_["exact"], "by_gap": main_["by_gap"],
           "controls": {c: _summarise(per_seq[c], R, edges) for c in conds if c != "windowed+side"},
           "per_sequence": per_seq["windowed+side"],
           "missing_keys": list(missing), "unexpected_keys": list(unexpected)}
    if args.jitter or args.update or pool:
        out["task"] = {"jitter": args.jitter, "update": args.update,
                       "names": "heldout" if args.heldout_names else "train" if pool else "all"}
    if args.update:                                                # new = the current code; old = stale
        out["update"] = {c: dict(zip(("new_digit1", "old_digit1", "new_exact", "old_exact"),
                                     (sum(r[i] for r in v) / max(1, len(v)) for i in range(4))))
                         for c, v in upd.items()}
    if args.w1s_ctrl != "none":
        out["w1s_ctrl"] = args.w1s_ctrl
    if oracle:
        out["oracle_gate"] = True
    if slots["seq"]:
        out["w1s"] = _slot_summary(slots, R)
    if events:
        out["eps_mean_all_tokens"] = sum(eps_all) / len(eps_all)
        if eps_d1:
            out["eps_mean_at_fact_digit1"] = sum(eps_d1) / len(eps_d1)
    variant = "".join(f".{name}" for name, on in (("jitter", args.jitter), (f"update-{args.update}", args.update),
                                                    ("heldout", args.heldout_names),
                                                    (f"ctrl-{args.w1s_ctrl}", args.w1s_ctrl != "none")) if on)
    path = Path(args.out or Path(args.result).with_suffix(f".by_query{variant}.json"))   # variants kept apart
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    summary = {k: out[k] for k in ("pairs", "seq_len", "exact", "digit1", "digit1_acc_by_queried_rank")}
    summary["controls_exact"] = {c: round(v["exact"], 3) for c, v in out["controls"].items()}
    if "update" in out:
        summary["update"] = {k: round(v, 3) for k, v in out["update"]["windowed+side"].items()}
    if "w1s" in out:
        summary["w1s"] = {k: v if v is None else round(v, 3) for k, v in out["w1s"].items() if not isinstance(v, list)}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
