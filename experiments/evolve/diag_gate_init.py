"""Gate track, Phase 0 (round 11, Stage A2): where do UNTRAINED W1s gates start on the frozen model's states?

research/plans/round11_combination_plan.md, "Gate track". The point gate's failures look like facts that are
never written or never kept, and ``gate_bias`` calibrates the initial density for a centred, unit-variance
drive. Without centring (I1's ``--no-w1s-centre``), w·x_t on LayerNormed LM states has a mean far from 0,
with the seed's sign. On synthetic states that leaves the hysteresis latch and the Schmitt trigger always
on or always off from step 0 (``tests/test_evolve_w1s.py``). This measures it on real needle sequences, for
the point, hyst and schmitt gates, uncentred (``none``, as I1), with the calibrated drive (``init``,
``--w1s-gate-norm init``) and, if asked, with the round-10 running-mean centring (``centre``).

For each gate × norm an untrained ``W1sSide`` is built with the same torch seed (``--side-seed``; the gate
weights are the same across arms). The recipe's other W1s flags come from a run JSON, or from W1s flags on
the command line. The side runs inside ``taps.TapLM`` on ``--n`` scattered-layout needle sequences
(``run.ScatterNeedleTask``, as ``run.py`` builds it). Each filler is run on the same sequences for every arm:
natural (``--text-file``, or the run's) and random tokens. Calibration (``init``) is one forward pass on a
batch of its own (``--calib-filler``, default natural when there is a text file, like a training run's first
batch), never on the measured sequences. Per gate × norm × filler, from the side's ``last_diag`` (side
coordinates, as ``eval_by_query._slot_diag`` reads it):

  onset_density           candidate onsets per token (strict local maxima of p). For point and hyst
                          this is about 1/(r + 1) whatever p's level; ``p_onset_*`` gives the level
  on_fraction             the mean on-state g over positions ≥ the window (for point, the mean p);
                          ``on_fraction_first_window`` is the start-up transient (t < window), and
                          ``latch_saturated`` the share of sequences with an on-fraction < 0.03 or > 0.97
  p_onset_mean / p_onset_median_log   the onset strength at candidates (it enters the read as log p)
  onset_in_fact           the share of facts with an onset inside their span (``_slot_diag``)
  coverage_all_facts      the share of facts with a retained slot at the query whose onset lies in the fact
  fact_coverage           the same for the queried fact only, with ``_slot_summary``'s other readings (gate
                          AUC of first digits against filler, p means, null weight; slot and alignment hits
                          are untrained here)

    python experiments/evolve/diag_gate_init.py results/evolve/round10/r10_I1_s1.json --device cuda --n 128 \\
        --out results/evolve/round11/diag_gate_init_I1.json
    python experiments/evolve/diag_gate_init.py --model Qwen/Qwen3-1.7B-Base --dtype bfloat16 --device cuda \\
        --layer 21 --window 64 --seq-len 512 --pairs 3 --text-file FILE --w1s-pre 0 --w1s-key learned \\
        --w1s-taps '' --out results/evolve/round11/diag_gate_init_recipe.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402
from evolve.eval_by_query import _slot_diag, _slot_summary  # noqa: E402
from evolve.run import TRAIN_NAMES, ScatterNeedleTask  # noqa: E402
from evolve.taps import TapLM  # noqa: E402
from evolve.w1s_slots import W1sSide, split_args  # noqa: E402

NORMS = {"none": dict(w1s_centre=False, w1s_gate_norm="none"),     # I1: uncentred
         "init": dict(w1s_centre=False, w1s_gate_norm="init"),     # the calibrated drive
         "centre": dict(w1s_centre=True, w1s_gate_norm="none")}    # round 10's running-mean centring
GATES = ("point", "hyst", "schmitt")
SATURATED = (0.03, 0.97)


def _csv(s: str, allowed) -> list[str]:
    out = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in out if x not in allowed]
    if bad or not out:
        raise SystemExit(f"{bad or s!r}: choose from {list(allowed)}")
    return out


def recipe(result: str | None, rest: list[str]) -> tuple[argparse.Namespace, dict, dict, dict]:
    """(model and task args for ``pl``, the W1s recipe, the run's layout {scatter, tail, names}, the source).

    From a run JSON: its model, tap, window, length, pairs, filler and W1s flags; command-line flags override
    (``pl`` flags by coming last, W1s flags where they differ from the defaults). Without one: everything from
    the command line."""
    w_cli, rest = split_args(rest)
    defaults = vars(split_args([])[0])
    explicit = {k: v for k, v in vars(w_cli).items() if v != defaults[k]}
    if result is None:
        return pl.parse_args(rest), vars(w_cli), {}, {"result": None}
    rec = json.loads(Path(result).read_text())
    ra, cand = rec["args"], rec.get("candidate") or {}
    if "w1s_slots" not in cand.get("class", ""):
        raise SystemExit(f"{result}: not a W1s run (candidate {cand.get('class')!r})")
    base = (["--tiny", "--tiny-arch", ra["tiny_arch"], "--seed", str(ra["seed"])] if ra.get("tiny")
            else ["--model", ra["model"]])
    base += ["--dtype", ra["dtype"], "--layer", str(ra["layer"]), "--window", str(ra["window"]),
             "--window-mode", ra["window_mode"], "--window-layers", ra["window_layers"], "--seq-len", str(ra["seq_len"]),
             "--pairs", str(ra["pairs"]), "--attn-impl", ra.get("attn_impl", "sdpa")]
    if ra.get("text_file"):
        base += ["--text-file", ra["text_file"]]
    if not ra.get("time", True):
        base += ["--no-time"]
    flags = {**defaults, **cand["args"], **explicit}
    layout = {"scatter": rec.get("needle_scatter", 0), "tail": rec.get("needle_tail", 256),
              "names": rec.get("needle_names", "all")}
    return pl.parse_args(base + rest), flags, layout, {"result": result}


def arm_flags(base: dict, gate: str, norm: str) -> argparse.Namespace:
    return argparse.Namespace(**{**base, "w1s_gate": gate, **NORMS[norm]})


def _batches(task, rng, n: int, batch: int):
    """``n // batch`` batches of (ids, n_ans, query rank, value_at, spans, query lengths), as eval_by_query draws them."""
    for _ in range(max(1, n // batch)):
        rows, js, vals, spans, qlens = [], [], [], [], []
        n_ans = None
        for _ in range(batch):
            ids, n_ans, _ = task.sample(rng)
            rows.append(ids), js.append(task.last_queried), vals.append(task.last_value_at)
            spans.append(task.last_spans), qlens.append(task.last_query_len)
        yield torch.tensor(rows), n_ans, js, vals, spans, qlens


def measure(lm, side, task, seed: int, n: int, batch: int, pairs: int) -> dict:
    """The gate's readings on ``n`` sequences of ``task`` (drawn from ``random.Random(seed)``)."""
    W, sink = lm.window, lm.sink
    acc = {"seq": [], "align_within1": [], "null_weight": [], "p_d1": [], "p_fill": [], "onset_in_fact": [],
           "density": [], "fire_false": [], "fire_filler": [], "fire_true": []}
    on_frac, on_first, p_mean, p_on, cov_all, held, n_on = [], [], [], [], [], [], []
    loop = side.loop
    dev = lm.input_device
    for ids, n_ans, js, vals, spans, qlens in _batches(task, random.Random(seed), n, batch):
        with torch.no_grad(), lm.condition(W, True):
            lm(ids.to(dev), keep=n_ans + 1)
        diag = loop.last_diag
        T = ids.shape[1]
        _slot_diag(diag, js, vals, spans, qlens, T, n_ans, sink, acc, None, 0.9, loop.P, loop.pool)
        g, p = diag["g"].float().cpu(), diag["p"].float().cpu()
        on, ret = diag["onsets"].cpu(), diag["retained"].cpu()
        tq = T - n_ans - sink
        for b in range(len(js)):
            on_frac.append(g[b, W:].mean().item())
            on_first.append(g[b, :W].mean().item())
            p_mean.append(p[b].mean().item())
            p_on.append(p[b, on[b]])
            n_on.append(int(on[b].sum()))
            r = ret[b, tq]
            held.append(int((r >= 0).sum()))
            for lo, hi in spans[b]:
                lo, hi = lo - sink, hi - sink
                cov_all.append(float(((r >= lo) & (r <= hi)).any()))
    pon = torch.cat(p_on) if p_on else torch.zeros(0)
    out = _slot_summary(acc, pairs)
    mean = lambda v: sum(v) / len(v) if v else None   # noqa: E731
    out.update({
        "n": len(on_frac),
        "on_fraction": mean(on_frac),
        "on_fraction_first_window": mean(on_first),
        "latch_saturated": mean([float(x < SATURATED[0] or x > SATURATED[1]) for x in on_frac]),
        "on_fraction_by_sequence": [round(x, 4) for x in on_frac],
        "p_mean": mean(p_mean),
        "p_onset_mean": pon.mean().item() if len(pon) else None,
        "p_onset_median_log": torch.log(pon.clamp_min(1e-30)).median().item() if len(pon) else None,
        "onsets_per_sequence": mean(n_on),
        "retained_at_query": mean(held),
        "coverage_all_facts": mean(cov_all),
    })
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    result = argv.pop(0) if argv and not argv[0].startswith("-") else None   # first, so flags' values are not taken
    ap = argparse.ArgumentParser(usage="%(prog)s [RESULT.json] [options] [pretrained_loop and --w1s-* flags]",
                                 description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates", default=",".join(GATES), help="comma list of point, hyst, schmitt")
    ap.add_argument("--norms", default="none,init", help="comma list of none (uncentred, I1), init "
                    "(--w1s-gate-norm init), centre (round-10 running-mean centring)")
    ap.add_argument("--fillers", default=None, help="natural,random (default: both when a text file is given)")
    ap.add_argument("--calib-filler", choices=["natural", "random"], default=None,
                    help="the calibration batch's filler (init; default natural when a text file is given)")
    ap.add_argument("--n", type=int, default=64, help="sequences per filler")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--data-seed", type=int, default=2024, help="the sequences (the calibration batch uses + 1)")
    ap.add_argument("--side-seed", type=int, default=0, help="torch seed of the untrained side")
    ap.add_argument("--needle-scatter", type=int, default=None, help="minimum gap (default: the run's, else 8)")
    ap.add_argument("--needle-tail", type=int, default=None, help="default: the run's, else 256")
    ap.add_argument("--needle-names", choices=["all", "train"], default=None, help="default: the run's, else all")
    ap.add_argument("--out", default=None, help="default <result>.gate_init.json (required without a result)")
    own, rest = ap.parse_known_args(argv)
    gates, norms = _csv(own.gates, GATES), _csv(own.norms, NORMS)
    pa, base, layout, src = recipe(result, rest)
    out_path = Path(own.out) if own.out else (Path(result).with_suffix(".gate_init.json") if result else None)
    if out_path is None:
        raise SystemExit("--out is required without a result JSON")
    scatter = own.needle_scatter if own.needle_scatter is not None else (layout.get("scatter") or 8)
    tail = own.needle_tail if own.needle_tail is not None else layout.get("tail", 256)
    names = own.needle_names or ("train" if layout.get("names") in ("train", "pool") else "all")
    if not scatter:
        raise SystemExit("the gate diagnostic needs the scattered layout (--needle-scatter > 0)")
    text_file = pa.text_file
    fillers = _csv(own.fillers, ("natural", "random")) if own.fillers else (["natural", "random"] if text_file
                                                                              else ["random"])
    if "natural" in fillers and not text_file:
        raise SystemExit("natural filler needs --text-file (or a run trained with one)")
    calib = own.calib_filler or ("natural" if text_file else "random")
    if calib == "natural" and not text_file:
        raise SystemExit("--calib-filler natural needs --text-file")
    if pa.window_mode in ("block", "segment") and pa.seq_len % pa.window:
        raise SystemExit(f"--seq-len {pa.seq_len} must be a multiple of --window {pa.window}")
    model, tok = pl.load_model(pa)
    d = pl.text_config(model).hidden_size
    corpus = tok.encode(Path(text_file).read_text(encoding="utf-8", errors="replace")) if text_file else None
    kw = dict(min_gap=scatter, tail=tail, block=pa.window, names=TRAIN_NAMES if names == "train" else None)
    tasks = {f: ScatterNeedleTask(tok, pa.seq_len, pa.pairs, 1, corpus if f == "natural" else None, **kw)
             for f in set(fillers) | {calib}}
    arms, layer = {}, pa.layer
    for gate in gates:
        for norm in norms:
            torch.manual_seed(own.side_seed)
            side = W1sSide(d, pa, arm_flags(base, gate, norm))
            lm = TapLM(model, side, pa.layer, pa.window, pa.window_mode, pa.window_layers, 1, pad_id=tok.eos_token_id)
            side.to(next(lm.blocks[lm.layer].parameters()).device).eval()
            layer = lm.layer
            rec = {"flags": {"w1s_gate": gate, **NORMS[norm]}}
            try:
                if norm == "init":                                # the first batch the gate sees sets μ̂
                    ids = next(_batches(tasks[calib], random.Random(own.data_seed + 1), own.batch, own.batch))[0]
                    with torch.no_grad(), lm.condition(pa.window, True):
                        lm(ids.to(lm.input_device), keep=1)
                    mu = side.loop.gate_mu
                    if not torch.isfinite(mu).all():
                        raise RuntimeError("the calibration pass did not set μ̂")
                    rec["calibration"] = {"filler": calib, "sequences": own.batch,
                                          "gate_mu_norm": [round(x, 4) for x in mu.norm(dim=-1).tolist()]}
                rec["fillers"] = {f: measure(lm, side, tasks[f], own.data_seed, own.n, own.batch, pa.pairs)
                                  for f in fillers}
            finally:
                for hd in lm.handles:                             # the next arm hooks the same model
                    hd.remove()
            arms[f"{gate}/{norm}"] = rec
            for f, r in rec["fillers"].items():
                print(f"{gate:7s} {norm:6s} {f:7s} onsets/token {r['onset_density']:.3f}  on {r['on_fraction']:.3f} "
                      f"(first window {r['on_fraction_first_window']:.3f}, saturated {r['latch_saturated']:.2f})  "
                      f"p@onset {r['p_onset_mean'] if r['p_onset_mean'] is not None else math.nan:.3f}  "
                      f"onset-in-fact {r['onset_in_fact']:.2f}  coverage {r['coverage_all_facts']:.2f} "
                      f"(queried {r['fact_coverage']:.2f})  AUC {r['gate_auc_digit1_vs_filler']:.2f}")
    out = {**src, "model": "tiny-" + pa.tiny_arch if pa.tiny else pa.model, "layer": layer, "window": pa.window,
           "seq_len": pa.seq_len, "pairs": pa.pairs, "scatter": scatter, "tail": tail, "names": names,
           "text_file": text_file, "n": own.n, "data_seed": own.data_seed, "side_seed": own.side_seed,
           "recipe": {k: v for k, v in base.items() if k not in ("w1s_gate", "w1s_centre", "w1s_gate_norm")},
           "arms": arms}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved {out_path}")
    return out


if __name__ == "__main__":
    main()
