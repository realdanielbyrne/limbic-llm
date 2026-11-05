"""Stage-0 check for A15 (RV4.R2.1): does the FULL model's attention from the answer mark the queried fact?

A15 trains W1s's gate towards the full-attention model's attention mass per candidate window. That only
makes sense if full attention, read from the query rows, concentrates on the queried fact. Round 11's
"received attention does not mark the code" was about the *windowed* model's attention inside a slot;
this is the full model's attention from the answer, which must mark the fact if that is how full
attention answers (RV4 §5, R2.1: "check it as a Stage-0 gate").

On fresh batches of a trained run's task (scattered layout, its pairs, filler, tail and names, rebuilt
from its result JSON like ``eval_by_query.py``), one pass of the frozen model under the full condition
records the attention of the query rows (the query span " The code for NAME is" and the answer's space,
which predicts digit 1) at every attention layer, with eager attention at those layers only
(``teacher.capture_attention``), averaged over heads and the query rows. Per sequence it reports:

  masses        the share of the query rows' attention on the queried fact's tokens, on the other facts'
                tokens, on the filler before the query, on the sink and on the query itself (they sum
                to 1), with the queried fact's share of the pre-query, non-sink attention
                (``queried_share``), the mass on the queried code's digits, and per-token densities
                (a fact has ~12 tokens against ~450 of filler)
  top fact      whether the queried fact gets the most mass among the facts
  AUC           per-token attention (pre-query, sink excluded) of fact tokens against filler tokens, of the
                queried fact's tokens against filler, and of the queried fact's against the other facts'
  full model    exact and digit-1 accuracy of the full model on the same rows, and the masses split by
                whether it answered (concentration where full attention succeeds is what A15 needs)
  per layer     queried_share, top-fact rate and AUC (queried vs filler) for each layer: the evidence
                for choosing ``--admit-layers``
  A15 target    with a W1s checkpoint, the run's own gate (``windowed+side`` pass, its store): the A15
                target over its candidates closed before the query (mass per window, normalised), how
                much of it falls on windows holding the queried fact's first digit, the same for the
                trained gate's normalised p, KL(target ‖ gate), and ``covered``: the share of the
                pre-query attention inside the candidates' windows

The layer set of the "mean" readings and the A15 target is ``--layers`` (default all attention layers).
Nothing is trained; the checkpoint is used only for the gate comparison.

    python experiments/evolve/diag_teacher_attention.py results/evolve/round10/r10_I1_s1.json --n 128 \\
        --device cuda --out results/evolve/round11/diag_teacher_attention_I1_s1.json
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
from evolve import teacher  # noqa: E402
from evolve.run import TRAIN_NAMES, ScatterNeedleTask  # noqa: E402

FALLBACK_TEXT = teacher.DEFAULT_TEXT                               # when the run's filler file is gone


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float | None:
    """P(pos > neg) + ½ P(pos = neg) by ranks (ties averaged); None if either side is empty."""
    if not len(pos) or not len(neg):
        return None
    x = torch.cat([pos, neg]).double()
    order = x.argsort()
    _, inv, cnt = torch.unique_consecutive(x[order], return_inverse=True, return_counts=True)
    ends = cnt.cumsum(0).double()
    r = torch.empty_like(x)
    r[order] = ((ends - cnt + 1 + ends) / 2)[inv]
    n1, n0 = len(pos), len(neg)
    return ((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()


def row_masks(T: int, sink: int, spans: list, j: int, q0: int, value_at: int, n_dig: int) -> dict:
    """Boolean (T,) masks of one sequence, full coordinates: queried fact, other facts, filler before the
    query, sink, query rows onwards, and the queried code's digits."""
    m = {k: torch.zeros(T, dtype=torch.bool) for k in ("queried", "other", "filler", "sink", "query", "digits")}
    m["sink"][:sink] = True
    m["query"][q0:] = True
    for k, (a, b) in enumerate(spans):
        m["queried" if k == j else "other"][a:b + 1] = True
    m["filler"][sink:q0] = True
    m["filler"] &= ~(m["queried"] | m["other"])
    m["digits"][value_at + 1:value_at + 1 + n_dig] = True
    return m


def _mean(v: list) -> float | None:
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def rebuild(rec: dict, dev: str, attn_impl: str | None):
    """The run's model, side (with its checkpoint, if any) and hooked LM, as ``eval_by_query.py`` builds them."""
    ra, cand = rec["args"], rec.get("candidate")
    impl = attn_impl or ra.get("attn_impl", "sdpa")
    if ra.get("tiny"):
        pa = pl.parse_args(["--tiny", "--tiny-arch", ra["tiny_arch"], "--seed", str(ra["seed"]), "--dtype", ra["dtype"],
                            "--device", dev, "--attn-impl", impl])
    else:
        pa = pl.parse_args(["--model", ra["model"], "--dtype", "bfloat16", "--device", dev, "--attn-impl", impl])
    model, tok = pl.load_model(pa)
    d = pl.text_config(model).hidden_size
    if cand:
        mod = importlib.import_module(cand["class"].split(":")[0])
        Cand = getattr(mod, cand["class"].split(":")[1])
        flags = {**vars(mod.split_args([])[0]), **cand["args"]} if hasattr(mod, "split_args") else cand["args"]
        side = Cand(d, argparse.Namespace(**ra), argparse.Namespace(**flags))
    else:
        side = pl.SideChannels(d, memory=ra["memory"], time=ra["time"], **pl.loop_kwargs(argparse.Namespace(**ra)))
    pt = Path(rec["_path"]).with_suffix(".pt")
    missing = unexpected = None
    if pt.exists():
        missing, unexpected = side.load_state_dict(torch.load(pt, map_location="cpu"), strict=False)
    side = side.to(next(model.parameters()).device).eval()
    LM = pl.HookedLM
    if hasattr(side, "taps"):
        from evolve.taps import TapLM
        LM = TapLM
    lm = LM(model, side, ra["layer"], ra["window"], ra["window_mode"], ra["window_layers"], 1, pad_id=tok.eos_token_id)
    return lm, tok, side, (str(pt) if pt.exists() else None, missing, unexpected)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", help="a scattered-layout round result JSON (its .pt next to it for the gate readings)")
    ap.add_argument("--n", type=int, default=128, help="sequences")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--layers", default="", metavar="LO-HI",
                    help="layers of the mean readings and the A15 target ('' = all)")
    ap.add_argument("--text-file", default=None, help="natural filler (default: as trained; if that file is gone, "
                    f"{FALLBACK_TEXT})")
    ap.add_argument("--random-filler", action="store_true", help="random-token filler")
    ap.add_argument("--attn-impl", choices=["sdpa", "eager"], default=None,
                    help="loaded implementation (default: as trained)")
    ap.add_argument("--out", default=None, help="default <result>.teacher_attn.json")
    args = ap.parse_args(argv)
    rec = json.loads(Path(args.result).read_text())
    rec["_path"] = args.result
    ra = rec["args"]
    scatter = rec.get("needle_scatter", 0)
    if not scatter:
        raise SystemExit("needs a scattered-layout run (the facts' spans come from its layout)")
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    lm, tok, side, (pt, missing, unexpected) = rebuild(rec, dev, args.attn_impl)
    all_layers = teacher.admit_layers("", lm)
    mean_layers = teacher.admit_layers(args.layers, lm)
    W, P, L = ra["window"], ra["pairs"], ra["seq_len"]
    text_file = None
    if not args.random_filler:
        text_file = args.text_file or ra.get("text_file")
        if text_file and not Path(text_file).is_file() and not args.text_file:
            text_file = FALLBACK_TEXT
    corpus = tok.encode(Path(text_file).read_text(encoding="utf-8", errors="replace")) if text_file else None
    names = TRAIN_NAMES if rec.get("needle_names") in ("train", "pool") else None
    task = ScatterNeedleTask(tok, L, P, 1, corpus, min_gap=scatter, tail=rec.get("needle_tail", 256), block=W,
                             names=names)
    loop = getattr(side, "loop", None)
    gate = loop is not None and hasattr(loop, "_store") and getattr(loop, "gate", None) not in ("oracle",)
    rng = random.Random(args.seed)
    keys = ("queried", "other", "filler", "sink", "query", "digits")
    seq = []                                                        # per-sequence records
    per_layer = {i: {"share": [], "top": [], "pos": [], "neg": []} for i in all_layers}
    tok_lists = {"fact": [], "queried": [], "other": [], "filler": []}
    for _ in range(max(1, args.n // args.batch)):
        rows, js, spans, vats, qlens = [], [], [], [], []
        for _ in range(args.batch):
            ids, n_ans, _ = task.sample(rng)
            rows.append(ids), js.append(task.last_queried), spans.append(task.last_spans)
            vats.append(task.last_value_at), qlens.append(task.last_query_len)
        ids = torch.tensor(rows)
        B, T = ids.shape
        q0, a0, R = teacher.query_rows(T, torch.full((B,), n_ans), torch.tensor(qlens))
        got, lg = teacher.capture_attention(lm, ids, all_layers, R, keep=n_ans + 1)
        pred = lg.argmax(-1).cpu()                                  # pred[:, k] predicts ids[T − n_ans + k]
        ok = pred[:, :n_ans] == ids[:, T - n_ans:]
        rowatt = {i: teacher.row_mean(got[i], q0, a0).cpu() for i in all_layers}
        mean_att = torch.stack([rowatt[i] for i in mean_layers]).mean(0)
        tgt = None
        if gate:                                                    # the run's gate on the same rows
            with teacher.store_tap(loop) as box, torch.no_grad(), lm.condition(W, True):
                lm(ids, keep=1)
            st = box["st"]
            tgt = teacher.window_targets(mean_att, st["s"], st["valid"], loop.P, loop.n, loop.delay, q0, lm.sink)
            lq = torch.where(tgt["elig"], st["lp"], torch.full_like(st["lp"], float("-inf"))).log_softmax(-1)
            lq = torch.where(tgt["has"][:, None], lq, torch.zeros_like(lq))
            kl = teacher.admit_kl(st["lp"], tgt["target"], tgt["elig"], tgt["has"])
        for b in range(B):
            m = row_masks(T, lm.sink, spans[b], js[b], int(q0[b]), vats[b][js[b]], n_ans - 1)
            a = mean_att[b]
            r = {k: a[m[k]].sum().item() for k in keys}
            pre = a[lm.sink:int(q0[b])].sum().item()
            r.update(queried_share=r["queried"] / pre if pre > 0 else None,
                     density_queried=r["queried"] / max(1, int(m["queried"].sum())),
                     density_other=r["other"] / max(1, int(m["other"].sum())),
                     density_filler=r["filler"] / max(1, int(m["filler"].sum())),
                     full_exact=bool(ok[b].all()), full_digit1=bool(ok[b, 1]), rank=js[b])
            fm = [a[spans[b][k][0]:spans[b][k][1] + 1].sum().item() for k in range(len(spans[b]))]
            r["top_fact"] = max(range(len(fm)), key=fm.__getitem__) == js[b]
            tok_lists["fact"].append(a[m["queried"] | m["other"]])
            tok_lists["queried"].append(a[m["queried"]])
            tok_lists["other"].append(a[m["other"]])
            tok_lists["filler"].append(a[m["filler"]])
            for i in all_layers:
                x = rowatt[i][b]
                pre_i = x[lm.sink:int(q0[b])].sum().item()
                per_layer[i]["share"].append(x[m["queried"]].sum().item() / pre_i if pre_i > 0 else None)
                fm_i = [x[s0:s1 + 1].sum().item() for s0, s1 in spans[b]]
                per_layer[i]["top"].append(float(max(range(len(fm_i)), key=fm_i.__getitem__) == js[b]))
                per_layer[i]["pos"].append(x[m["queried"]])
                per_layer[i]["neg"].append(x[m["filler"]])
            if tgt is not None and bool(tgt["has"][b]):
                d1 = vats[b][js[b]] + 1 - lm.sink                   # the queried first digit, side coordinates
                s_ = st["s"][b].cpu()
                holds = ((s_ - loop.P <= d1) & (d1 <= s_ - loop.P + loop.n - 1)) & tgt["elig"][b].cpu()
                r.update(target_on_queried=tgt["target"][b].cpu()[holds].sum().item(),
                         gate_on_queried=lq[b].exp().cpu()[holds].sum().item(),
                         queried_in_candidates=bool(holds.any()), kl_target_gate=kl[b].item(),
                         covered=tgt["covered"][b].item(), n_eligible=int(tgt["elig"][b].sum()))
            seq.append(r)

    def means(rs: list) -> dict:
        out = {k: _mean([x.get(k) for x in rs]) for k in keys + ("queried_share", "density_queried", "density_other",
                                                               "density_filler")}
        out["top_fact_rate"] = _mean([float(x["top_fact"]) for x in rs])
        out["n"] = len(rs)
        return out
    cat = {k: torch.cat(v) if v else torch.zeros(0) for k, v in tok_lists.items()}
    out = {"result": args.result, "checkpoint": pt, "n": len(seq), "pairs": P, "seq_len": L, "window": W,
           "text_file": text_file, "layers_all": all_layers, "layers_mean": mean_layers, "seed": args.seed,
           "full_exact": _mean([float(r["full_exact"]) for r in seq]),
           "full_digit1": _mean([float(r["full_digit1"]) for r in seq]),
           "mean": means(seq),
           "by_full_correct": {"correct": means([r for r in seq if r["full_exact"]]),
                               "wrong": means([r for r in seq if not r["full_exact"]])},
           "by_queried_rank": [means([r for r in seq if r["rank"] == k]) for k in range(P)],
           "auc": {"fact_vs_filler": auc(cat["fact"], cat["filler"]),
                   "queried_vs_filler": auc(cat["queried"], cat["filler"]),
                   "queried_vs_other_facts": auc(cat["queried"], cat["other"])},
           "per_layer": {str(i): {"queried_share": _mean(v["share"]), "top_fact_rate": _mean(v["top"]),
                                  "auc_queried_vs_filler": auc(torch.cat(v["pos"]), torch.cat(v["neg"]))}
                         for i, v in per_layer.items()},
           "missing_keys": list(missing) if missing is not None else None,
           "unexpected_keys": list(unexpected) if unexpected is not None else None}
    if gate:
        g = [r for r in seq if "target_on_queried" in r]
        out["a15_target"] = {k: _mean([float(r[k]) for r in g]) for k in
                             ("target_on_queried", "gate_on_queried", "queried_in_candidates", "kl_target_gate",
                              "covered", "n_eligible")}
        out["a15_target"]["n"] = len(g)
    path = Path(args.out or Path(args.result).with_suffix(".teacher_attn.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    summ = {"n": out["n"], "full_exact": out["full_exact"], "auc": out["auc"],
            **{k: out["mean"][k] for k in ("queried", "other", "filler", "sink", "queried_share", "top_fact_rate")}}
    if gate:
        summ["a15_target"] = out["a15_target"]
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()}))
    print(f"saved {path}")
    return out


if __name__ == "__main__":
    main()
