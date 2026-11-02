"""Round runner for the evolution search: ``experiments/pretrained_loop.py`` plus per-position logging.

Runs ``pretrained_loop.main`` unchanged, with three additions:

  per-position  For the needle task every evaluated condition also records loss and accuracy at each
                answer position (``results[cond]["per_pos"]``, first answer token first). The
                plateau at (4 − k)/5 · ln 10 reads as "k digits known", and this shows which ones.
  checkpoint    The trained side channels' ``state_dict`` is saved next to the JSON (``.pt``), so
                later rounds can ablate a trained candidate at evaluation time.
  candidate     ``--candidate module:Class`` replaces ``SideChannels`` with another side-channel
                module, built as ``Class(d_model, args)``. It must return ``(h, stats)`` like
                ``SideChannels``, where ``stats`` is ``None`` or has an ``inj_ratio`` (B, T) field,
                and expose ``loop`` with ``gain_logit`` and ``tau`` (``loop = None`` is allowed).
                A module-level ``split_args(argv) -> (namespace, rest)`` takes the candidate's own
                flags, and the namespace is passed as ``Class(d_model, args, namespace)``.

    python experiments/evolve/run.py --model Qwen/Qwen3-1.7B-Base --dtype bfloat16 --device cuda \
        --task needle --kernel exp --rho 0.95 --tau 32 --steps 3000 --out results/evolve/round1/x.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402

_last_lm = None
_evaluate = pl.evaluate


@torch.no_grad()
def per_position(lm, batches: list, window: int, side: bool) -> list[dict]:
    """Loss and accuracy at each answer position, aligned from the end of the sequence."""
    ce_sum = ok_sum = n = None
    with lm.condition(window, side):
        for ids, mask, _ in batches:
            ce, ok, m = pl.scores(lm, ids, mask)
            k = int(m.sum(1).min())                               # answer length (same for every row)
            ce, ok = ce[:, -k:], (ok & m)[:, -k:]
            ce_sum = ce.sum(0) if ce_sum is None else ce_sum + ce.sum(0)
            ok_sum = ok.float().sum(0) if ok_sum is None else ok_sum + ok.float().sum(0)
            n = len(ids) if n is None else n + len(ids)
    return [{"loss": (c / n).item(), "acc": (a / n).item()} for c, a in zip(ce_sum, ok_sum)]


def evaluate(lm, batches, window, side, task):
    global _last_lm
    _last_lm = lm
    res = _evaluate(lm, batches, window, side, task)
    if task == "needle":
        res["per_pos"] = per_position(lm, batches, window, side)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--candidate", default=None, help="module:Class replacing SideChannels")
    own, rest = ap.parse_known_args(argv)
    cand_ns = None
    pl.evaluate = evaluate
    if own.candidate:
        mod_name, cls = own.candidate.split(":")
        mod = importlib.import_module(mod_name)
        Cand = getattr(mod, cls)
        if hasattr(mod, "split_args"):
            cand_ns, rest = mod.split_args(rest)
        parsed = pl.parse_args(rest)
        pl.SideChannels = lambda d_model, **_: Cand(d_model, parsed, cand_ns)
    results = pl.main(rest)
    args = pl.parse_args(rest)
    out = Path(args.out or f"results/pretrained/pretrained_loop_{args.task}.json")
    if own.candidate:                                             # record the candidate and its flags
        rec = json.loads(out.read_text())
        rec["candidate"] = {"class": own.candidate, "args": vars(cand_ns) if cand_ns else {}}
        out.write_text(json.dumps(rec, indent=2))
    if _last_lm is not None:
        torch.save({k: v.cpu() for k, v in _last_lm.side.state_dict().items()}, out.with_suffix(".pt"))
        print(f"saved {out.with_suffix('.pt')}")
    return results


if __name__ == "__main__":
    main()
