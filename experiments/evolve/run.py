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


class ScatterNeedleTask(pl.NeedleTask):
    """The needle task with the facts scattered: random filler before, between and after them.

    In ``pl.NeedleTask`` the facts sit right after the sink, in order, which lets a memory solve
    two pairs by position alone. W1g did: it retrieved the first and the last fact, never the
    middle one (register P18). Here each fact is preceded by a random stretch of filler (at least
    ``min_gap`` tokens between facts), and at least ``tail`` filler tokens separate the last fact
    from the query. This removes absolute-position cues but keeps the facts' order, so a memory of
    the first and the last fact still covers 2 pairs; use 3 or more pairs, and read the accuracy by
    queried rank (``eval_by_query.py``). With ``block`` (the attention window), facts that would
    straddle a window-block boundary are resampled, as in the fixed layout, where none do.
    """

    def __init__(self, *a, min_gap: int = 8, tail: int = 256, block: int = 0, **kw) -> None:
        super().__init__(*a, **kw)
        self.min_gap, self.tail, self.block = min_gap, tail, block

    def sample(self, rng):
        enc = self.tok.encode
        names = rng.sample(pl.NAMES, self.pairs)
        values = ["".join(rng.choice("0123456789") for _ in range(4)) for _ in names]
        facts = [enc(f" The code for {n} is") + enc(f" {v}") + enc(".") for n, v in zip(names, values)]
        heads = [len(enc(f" The code for {n} is")) for n in names]
        j = rng.randrange(self.pairs)
        query, answer = enc(f" The code for {names[j]} is"), enc(f" {values[j]}")
        n_fill = self.seq_len - sum(map(len, facts)) - len(query) - len(answer)
        spare = n_fill - self.tail - self.min_gap * (self.pairs - 1)
        if spare < 0:
            raise ValueError(f"--seq-len {self.seq_len} too short for {self.pairs} scattered pairs")
        for _ in range(1000):                                              # filler before each fact
            cuts = sorted(rng.randint(0, spare) for _ in range(self.pairs))
            pre = [cuts[0]] + [cuts[i] - cuts[i - 1] + self.min_gap for i in range(1, self.pairs)]
            if not self.block or not self._straddles(pre, facts):
                break
        def fill(k):
            if self.corpus:
                o = rng.randrange(len(self.corpus) - k)
                return self.corpus[o:o + k]
            return [self.pool[rng.randrange(len(self.pool))] for _ in range(k)]
        ids, value_at = [self.bos] * self.sink, []
        for f, h, k in zip(facts, heads, pre):
            ids += fill(k)
            value_at.append(len(ids) + h)
            ids += f
        ids += fill(self.seq_len + self.sink - len(ids) - len(query) - len(answer)) + query + answer
        gap = (len(ids) - len(answer)) - value_at[j]
        self.last_queried, self.last_value_at = j, value_at                # read by eval_by_query.py
        return ids, len(answer), gap

    def _straddles(self, pre: list[int], facts: list[list[int]]) -> bool:
        """True if any fact would span two window blocks (positions counted after the sink)."""
        p = 0
        for k, f in zip(pre, facts):
            p += k
            if p // self.block != (p + len(f) - 1) // self.block:
                return True
            p += len(f)
        return False


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
    ap.add_argument("--needle-scatter", type=int, default=0, metavar="MIN_GAP",
                    help="scatter the facts with random filler (at least MIN_GAP tokens apart); 0 = fixed layout")
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
    if own.needle_scatter:                                         # after the candidate took its flags
        gap, window = own.needle_scatter, pl.parse_args(rest).window
        pl.NeedleTask = lambda *a, **kw: ScatterNeedleTask(*a, min_gap=gap, block=window, **kw)
    results = pl.main(rest)
    args = pl.parse_args(rest)
    out = Path(args.out or f"results/pretrained/pretrained_loop_{args.task}.json")
    if own.candidate or own.needle_scatter:
        rec = json.loads(out.read_text())
    if own.candidate:                                             # record the candidate and its flags
        rec["candidate"] = {"class": own.candidate, "args": vars(cand_ns) if cand_ns else {}}
    if own.needle_scatter:
        rec["needle_scatter"] = own.needle_scatter
    if own.candidate or own.needle_scatter:
        out.write_text(json.dumps(rec, indent=2))
    if _last_lm is not None:
        torch.save({k: v.cpu() for k, v in _last_lm.side.state_dict().items()}, out.with_suffix(".pt"))
        print(f"saved {out.with_suffix('.pt')}")
    return results


if __name__ == "__main__":
    main()
