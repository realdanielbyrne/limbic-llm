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
                A module that sets ``TAPS_REQUIRED`` reads hidden states of earlier blocks: the
                runner then builds ``taps.TapLM`` in place of ``HookedLM`` (``experiments/evolve/taps.py``).
  names         ``--needle-names train`` samples the facts' names from the first 20 of ``NAMES``
                only, so that ``eval_by_query.py`` has names the run never trained on.
  oracle gate   A candidate flag ``--<prefix>-gate oracle`` gets each batch's fact onsets (the first
                digits) as ``side.oracle_onsets``. Diagnostic only: it breaks rule E4.

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


PLAIN = " The code for {n} is"
JITTER_STEMS = [PLAIN, " The code for {n} is now", " The code for {n}, as noted, is"]
UPDATE_STEMS = {"same": PLAIN, "now": " The code for {n} is now", "new": " The new code for {n} is"}
TRAIN_NAMES = pl.NAMES[:20]                                        # --needle-names train


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

    ``names`` restricts the names sampled (default ``pl.NAMES``). With ``jitter`` each fact takes
    one of three phrasings (" is", " is now", ", as noted, is") while the query keeps the plain
    one, so a fixed lag from the name to the digits no longer finds the value. The defaults draw
    the same random numbers as before both options existed, so the sequences are unchanged.
    """

    def __init__(self, *a, min_gap: int = 8, tail: int = 256, block: int = 0, mix: float = 1.0,
                 curriculum: list[tuple[int, int]] | None = None, names: list[str] | None = None,
                 jitter: bool = False, **kw) -> None:
        super().__init__(*a, **kw)
        self.min_gap, self.tail, self.block, self.mix = min_gap, tail, block, mix
        self.curriculum, self._active, self._calls = curriculum or [], False, 0
        self.names, self.jitter, self._rows = names or pl.NAMES, jitter, None
        if self.corpus and mix < 1.0 and self.pool is None:              # random filler for the mixed case
            self.pool = self.tok.filler_pool()

    def sample(self, rng):
        enc = self.tok.encode
        names = rng.sample(self.names, self.pairs)
        values = ["".join(rng.choice("0123456789") for _ in range(4)) for _ in names]
        stems = [rng.choice(JITTER_STEMS) if self.jitter else PLAIN for _ in names]
        facts = [enc(s.format(n=n)) + enc(f" {v}") + enc(".") for n, v, s in zip(names, values, stems)]
        heads = [len(enc(s.format(n=n))) for n, s in zip(names, stems)]
        j = rng.randrange(self.pairs)
        query, answer = enc(PLAIN.format(n=names[j])), enc(f" {values[j]}")
        ids, value_at, natural = self._place(rng, facts, heads, query, answer)
        gap = (len(ids) - len(answer)) - value_at[j]
        self._record(j, value_at, natural, heads, len(query), facts)
        return ids, len(answer), gap

    def _place(self, rng, facts: list[list[int]], heads: list[int], query: list[int],
               answer: list[int]) -> tuple[list[int], list[int], bool]:
        """Scatter ``facts`` in filler before ``query + answer``: (ids, value_at, natural filler).

        ``value_at`` is each fact's position of the space before its digits (``heads`` tokens in)."""
        n = len(facts)
        n_fill = self.seq_len - sum(map(len, facts)) - len(query) - len(answer)
        spare = n_fill - self.tail - self.min_gap * (n - 1)
        if spare < 0:
            raise ValueError(f"--seq-len {self.seq_len} too short for {n} scattered pairs")
        for _ in range(1000):                                              # filler before each fact
            cuts = sorted(rng.randint(0, spare) for _ in range(n))
            pre = [cuts[0]] + [cuts[i] - cuts[i - 1] + self.min_gap for i in range(1, n)]
            if not self.block or not self._straddles(pre, facts):
                break
        natural = bool(self.corpus) and (self.mix >= 1.0 or rng.random() < self.mix)   # per sequence

        def fill(k):
            if natural:
                o = rng.randrange(len(self.corpus) - k)
                return self.corpus[o:o + k]
            return [self.pool[rng.randrange(len(self.pool))] for _ in range(k)]
        ids, value_at = [self.bos] * self.sink, []
        for f, h, k in zip(facts, heads, pre):
            ids += fill(k)
            value_at.append(len(ids) + h)
            ids += f
        ids += fill(self.seq_len + self.sink - len(ids) - len(query) - len(answer)) + query + answer
        return ids, value_at, natural

    def _record(self, j: int, value_at: list[int], natural: bool, heads: list[int], query_len: int,
                facts: list[list[int]]) -> None:
        """Keep the last sample's layout for the analysis tools, and each row's value_at in a batch."""
        self.last_queried, self.last_value_at, self.last_natural = j, value_at, natural   # for eval_by_query.py
        self.last_heads, self.last_query_len = heads, query_len           # for diag_novelty.py
        self.last_spans = [(v - h, v - h + len(f) - 1) for v, h, f in zip(value_at, heads, facts)]   # first, last token
        if self._rows is not None:
            self._rows.append(value_at)

    def batch(self, B: int, rng):
        """``pl.NeedleTask.batch``, recording each row's ``value_at`` list in ``last_batch_value_at``.

        The same lists ride on the returned ids as ``ids.value_at``: the evaluation batches are all
        built before training, so an oracle gate must find a batch's onsets on the batch itself."""
        self._rows = []
        try:
            ids, mask, gaps = self._batch(B, rng)
        finally:
            rows, self._rows = self._rows, None
        self.last_batch_value_at = ids.value_at = rows
        return ids, mask, gaps

    def _batch(self, B: int, rng):
        """With a curriculum, training batches use fewer pairs early ((pairs, steps) stages, then
        the full count). Only active inside training (see ``train`` below), so the evaluation
        batches, built before training, always use the full number of pairs."""
        if not (self.curriculum and self._active):
            return super().batch(B, rng)
        step, self._calls = self._calls, self._calls + 1
        pairs = self.pairs
        for p, n in self.curriculum:
            if step < n:
                pairs = p
                break
            step -= n
        full, self.pairs = self.pairs, pairs
        try:
            return super().batch(B, rng)
        finally:
            self.pairs = full

    def _straddles(self, pre: list[int], facts: list[list[int]]) -> bool:
        """True if any fact would span two window blocks (positions counted after the sink)."""
        p = 0
        for k, f in zip(pre, facts):
            p += k
            if p // self.block != (p + len(f) - 1) // self.block:
                return True
            p += len(f)
        return False


class UpdateNeedleTask(ScatterNeedleTask):
    """A fact is updated: one name is stated twice, with an old and then a new code; the query asks
    for the current (new) code.

    Each of the ``pairs`` names is stated once, in a random order; one of them, U, is stated again
    later with a new code whose first digit differs from the old one. ``form`` phrases the update:
    ``same`` repeats the template (pure recency), ``now`` adds " now", ``new`` says "The new code".
    The answer is the new code. The layout records cover all ``pairs + 1`` statements, with
    ``last_queried`` the update's rank, ``last_old_rank`` the old statement's and
    ``last_old_answer`` the old code's tokens (to score the stale answer).
    """

    def __init__(self, *a, form: str = "same", **kw) -> None:
        super().__init__(*a, **kw)
        if form not in UPDATE_STEMS:
            raise ValueError(form)
        self.form = form

    def sample(self, rng):
        enc = self.tok.encode
        names = rng.sample(self.names, self.pairs)
        codes = ["".join(rng.choice("0123456789") for _ in range(4)) for _ in names]
        u = rng.randrange(self.pairs)                                      # the updated name
        new = codes[u]
        while new[0] == codes[u][0]:                                       # digit 1 tells new from old
            new = "".join(rng.choice("0123456789") for _ in range(4))
        order = list(range(self.pairs))
        rng.shuffle(order)
        ins = rng.randint(order.index(u) + 1, self.pairs)                  # the update comes after the old
        stmts = [(names[i], codes[i], PLAIN) for i in order]
        stmts.insert(ins, (names[u], new, UPDATE_STEMS[self.form]))
        facts = [enc(s.format(n=n)) + enc(f" {v}") + enc(".") for n, v, s in stmts]
        heads = [len(enc(s.format(n=n))) for n, _, s in stmts]
        query, answer = enc(PLAIN.format(n=names[u])), enc(f" {new}")
        ids, value_at, natural = self._place(rng, facts, heads, query, answer)
        gap = (len(ids) - len(answer)) - value_at[ins]
        self.last_old_rank, self.last_old_answer = order.index(u), enc(f" {codes[u]}")
        self._record(ins, value_at, natural, heads, len(query), facts)
        return ids, len(answer), gap


_train = pl.train


def train(lm, task, args):
    """Activate a task curriculum for the duration of training only."""
    task._active = bool(getattr(task, "curriculum", None))
    try:
        return _train(lm, task, args)
    finally:
        task._active = False


def oracle_gate(ns) -> bool:
    """True if a candidate's flags select the oracle gate (``--<prefix>-gate oracle``)."""
    return ns is not None and any(k.endswith("_gate") and v == "oracle" for k, v in vars(ns).items())


def oracle_onsets(value_at: list[list[int]], T: int, sink: int, device=None) -> torch.Tensor:
    """(B, T − sink) bool: True at each fact's first digit, in side coordinates (the sink dropped).

    ``value_at`` holds each row's positions of the space before a fact's digits, counted with the sink."""
    on = torch.zeros(len(value_at), T - sink, dtype=torch.bool)
    for b, vat in enumerate(value_at):
        on[b, [v + 1 - sink for v in vat]] = True
    return on.to(device)


_scores = pl.scores


def scores(lm, ids, mask):
    """``pl.scores``, after handing an oracle gate this batch's fact onsets (from ``ids.value_at``)."""
    vat = getattr(ids, "value_at", None)
    if vat is not None:
        dev = next(lm.side.parameters()).device
        lm.side.oracle_onsets = oracle_onsets(vat, ids.shape[1], lm.sink, dev)
    return _scores(lm, ids, mask)


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
    ap.add_argument("--needle-curriculum", default="", metavar="P:STEPS,...",
                    help="with --needle-scatter: train with P pairs for STEPS steps per stage, then the full count")
    ap.add_argument("--needle-mix", type=float, default=1.0,
                    help="with --text-file and --needle-scatter: share of sequences with natural filler (rest random)")
    ap.add_argument("--needle-names", choices=["all", "train"], default="all",
                    help="with --needle-scatter: train = the first 20 names only (the rest are held out)")
    own, rest = ap.parse_known_args(argv)
    cand_ns = None
    pl.evaluate = evaluate
    pl.train = train
    base_lm, base_scores = pl.HookedLM, pl.scores                  # restored after the run
    if own.candidate:
        mod_name, cls = own.candidate.split(":")
        mod = importlib.import_module(mod_name)
        Cand = getattr(mod, cls)
        if hasattr(mod, "split_args"):
            cand_ns, rest = mod.split_args(rest)
        parsed = pl.parse_args(rest)
        pl.SideChannels = lambda d_model, **_: Cand(d_model, parsed, cand_ns)
        if getattr(mod, "TAPS_REQUIRED", False):                   # the side reads earlier blocks too
            from evolve.taps import TapLM
            pl.HookedLM = TapLM
        if oracle_gate(cand_ns):
            if not own.needle_scatter:
                raise SystemExit("an oracle gate needs --needle-scatter (the onsets come from its layout)")
            pl.scores = scores
    if own.needle_names != "all" and not own.needle_scatter:
        raise SystemExit("--needle-names needs --needle-scatter")
    if own.needle_scatter:                                         # after the candidate took its flags
        gap, window, mix = own.needle_scatter, pl.parse_args(rest).window, own.needle_mix
        cur = [tuple(int(x) for x in st.split(":")) for st in own.needle_curriculum.split(",") if st]
        names = TRAIN_NAMES if own.needle_names == "train" else None
        pl.NeedleTask = lambda *a, **kw: ScatterNeedleTask(*a, min_gap=gap, block=window, mix=mix,
                                                           curriculum=cur, names=names, **kw)
    try:
        results = pl.main(rest)
    finally:
        pl.HookedLM, pl.scores = base_lm, base_scores
    args = pl.parse_args(rest)
    out = Path(args.out or f"results/pretrained/pretrained_loop_{args.task}.json")
    if own.candidate or own.needle_scatter:
        rec = json.loads(out.read_text())
    if own.candidate:                                             # record the candidate and its flags
        rec["candidate"] = {"class": own.candidate, "args": vars(cand_ns) if cand_ns else {}}
    if own.needle_scatter:
        rec["needle_scatter"] = own.needle_scatter
        rec["needle_mix"] = own.needle_mix
        rec["needle_curriculum"] = own.needle_curriculum
        if own.needle_names != "all":                             # absent = all (older runs)
            rec["needle_names"] = own.needle_names
    if own.candidate or own.needle_scatter:
        out.write_text(json.dumps(rec, indent=2))
    if _last_lm is not None:
        torch.save({k: v.cpu() for k, v in _last_lm.side.state_dict().items()}, out.with_suffix(".pt"))
        print(f"saved {out.with_suffix('.pt')}")
    return results


if __name__ == "__main__":
    main()
