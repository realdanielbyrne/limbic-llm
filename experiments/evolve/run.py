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
                ``--needle-names pool --needle-name-pool N`` trains on N synthetic syllable names
                (``names.syllable_pool(N, 0)``, disjoint from ``NAMES`` and the held-out list).
  tail          ``--needle-tail T`` (default 256): the minimum filler between the last fact and the
                query; it must exceed ``--window``, or the base model reads the last fact itself.
  oracle gate   A candidate flag ``--<prefix>-gate oracle`` gets each batch's fact onsets (the first
                digits) as ``side.oracle_onsets``. Diagnostic only: it breaks rule E4. A candidate
                flag ``--<prefix>-keep-by external`` (eval-time keep oracles) is refused here.
  state         A side with ``state_report()`` has its streaming state size recorded as ``w1s_state``.
  teachers      Round-11 arms trained against the frozen model's own full-attention pass (training only;
                ``experiments/evolve/teacher.py``), both off by default: ``--distill-frac F`` (A14) puts a
                share F of the steps on general text with the loss T²·KL(full ‖ windowed+side) over the last
                ``--distill-positions`` positions; ``--admit-target teacher`` (A15, W1s, scattered layout)
                adds λ·KL(full attention's mass per candidate window ‖ the gate's p). ``train`` below
                installs them around ``pl.train``; the result JSON records ``distill`` / ``admit``.
  init-from     ``--init-from PT`` (repeatable; default none) loads a partial state dict (a fragment: a dict of
                tensors named as in the side's ``state_dict``) into the side channels before the first step,
                e.g. G0.5's fitted ``{loop.wz.weight, loop.log_beta_align}`` for C5r. Keys not in the side, a
                key in two fragments, a wrong shape and ``--steps 0`` are refused; the loaded values replace
                the constructor's (and ``--init-gain``). The result JSON records ``init_from`` (each file and
                its keys). Without the flag nothing is loaded and training is exactly as before.
  init-seed     ``--init-seed N`` (default off) seeds torch's CPU generator with N just before the side channels
                are constructed, and afterwards leaves the generator where the unseeded construction would
                (``seeded_init``). Only the side's initialisation changes: the base model, the data
                (``random.Random(--seed)``) and every later torch draw are those of the run without the flag.
                It makes same-data-seed replicates (s_seed from within-seed differences in analyse_round.py).
                The result JSON records ``init_seed``. Without the flag nothing is wrapped and the run is
                exactly as before.

The task variants that ``eval_by_query.py`` evaluates live here too: ``UpdateNeedleTask`` (forms
same/now/new/changed/nowold, ``k`` restatements, decoys), ``OrderNeedleTask`` (before/next/first/last) and
the paraphrased queries (``PARAPHRASE_QUERIES``, the ``query`` template of the scattered tasks).

    python experiments/evolve/run.py --model Qwen/Qwen3-1.7B-Base --dtype bfloat16 --device cuda \
        --task needle --kernel exp --rho 0.95 --tau 32 --steps 3000 --out results/evolve/round1/x.json
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
# an update statement is HEAD + " NEW" + TAIL; value_at is the space before NEW's digits (``heads`` tokens in)
UPDATE_STEMS = {"same": PLAIN, "now": " The code for {n} is now", "new": " The new code for {n} is",
                "changed": " {n}'s code has changed from {old} to", "nowold": " The code for {n} is now"}
UPDATE_TAILS = {"nowold": " (it used to be {old})."}               # the others end with "."
ORDER_QUERIES = {"before": " The code stated just before {n}'s is", "next": " The code stated just after {n}'s is",
                 "first": " The first code stated is", "last": " The last code stated is"}
# paraphrased queries (RV4.E3, eval_by_query.py --paraphrase K): the facts keep PLAIN, the query takes one of these, and
# the answer stays " DDDD". Only 1 does not end in " is" (the token the offset read lines up with the stored " is")
PARAPHRASE_QUERIES = {1: " Q: What is the code for {n}? A:", 2: " {n}'s code is", 3: " Remember the code for {n}. It is",
                      4: " The number assigned to {n} is"}
TRAIN_NAMES = pl.NAMES[:20]                                        # --needle-names train
OVERFLOW_POOL = 2000                                               # syllable_pool(2000, 0): the name rule's extra names


def code4(rng) -> str:
    return "".join(rng.choice("0123456789") for _ in range(4))


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

    ``tail`` (``--needle-tail``) must exceed ``block`` (the window), or the base model could read
    the last fact itself. ``decoys`` adds that many code statements for names never queried, in a
    random statement order with the pairs; the query is one of the ``pairs`` and ``last_queried``
    is its rank among all statements.

    Name rule: with more distinct names in a sequence than ``names`` holds (e.g. 32 codes on the
    20 training names), the queried name still comes from ``names`` and the other statements draw
    from ``names`` ∪ ``syllable_pool(2000, 0)`` minus it (``names_rule`` says so; None otherwise).

    ``query`` (eval_by_query.py ``--paraphrase``, RV4.E3) is the query's template, with ``{n}`` for the
    name (default ``PLAIN``, the facts' own prefix); the facts and the answer " DDDD" are unchanged.
    The sequence is laid out with the plain query and then re-queried (``_requery``), so it is the
    plain cell's sequence, draw for draw, except the end of the last filler: a query d tokens longer
    takes d filler tokens before it, a shorter one adds them (natural filler continues the corpus;
    random filler draws from a generator of its own). value_at, spans, the gap and the distance from
    the last fact to the answer are those of the plain cell; the filler between the last fact and
    the query is ``tail`` − d at the least.
    """

    def __init__(self, *a, min_gap: int = 8, tail: int = 256, block: int = 0, mix: float = 1.0,
                 curriculum: list[tuple[int, int]] | None = None, names: list[str] | None = None,
                 jitter: bool = False, decoys: int = 0, query: str | None = None, **kw) -> None:
        super().__init__(*a, **kw)
        if block and tail <= block:
            raise ValueError(f"--needle-tail {tail} must exceed the window {block} (else the base reads the last fact)")
        if decoys < 0:
            raise ValueError(decoys)
        if query is not None and "{n}" not in query:
            raise ValueError(f"a query template needs {{n}} for the name: {query!r}")
        self.min_gap, self.tail, self.block, self.mix = min_gap, tail, block, mix
        self.curriculum, self._active, self._calls = curriculum or [], False, 0
        self.names, self.jitter, self._rows = names or pl.NAMES, jitter, None
        self.decoys, self._extra = decoys, None
        self.query = query or PLAIN
        self._pad_rng, self._fill_end = random.Random(0), None          # _requery's own draws (never the task's rng)
        if self.corpus and mix < 1.0 and self.pool is None:              # random filler for the mixed case
            self.pool = self.tok.filler_pool()

    # ------------------------------------------------------------------ names
    def n_names(self) -> int:
        """Distinct names in one sequence (restatements reuse a name)."""
        return self.pairs + self.decoys

    @property
    def names_rule(self) -> str | None:
        if self.n_names() <= len(self.names):
            return None
        return (f"{self.n_names()} names > {len(self.names)} listed: the queried name from the configured list, the "
                f"other statements from that list ∪ syllable_pool({OVERFLOW_POOL}, seed 0) minus the queried name")

    def _draw_names(self, rng, n: int) -> tuple[list[str], bool]:
        """``n`` distinct names, and whether the name rule applied. Within the list: ``rng.sample`` as
        before. Beyond it: names[0] is the queried name (from the list), the rest from the list ∪
        the syllable pool; the caller swaps names[0] to the queried index once it is drawn."""
        if n <= len(self.names):
            return rng.sample(self.names, n), False
        if self._extra is None:
            from evolve.names import syllable_pool
            self._extra = list(dict.fromkeys(list(self.names) + syllable_pool(OVERFLOW_POOL, 0)))
        # every list name is used (familiarity then says nothing about which fact is queried: review RV
        # round 11, F8); the queried name is one of them and the rest are syllable names
        q = rng.choice(self.names)
        fill = rng.sample([x for x in self._extra if x not in self.names], n - len(self.names))
        return [q] + rng.sample([x for x in self.names if x != q] + fill, n - 1), True

    # ----------------------------------------------------------------- sample
    def sample(self, rng):
        enc = self.tok.encode
        n = self.pairs + self.decoys
        names, rule = self._draw_names(rng, n)
        values = [code4(rng) for _ in names]
        stems = [rng.choice(JITTER_STEMS) if self.jitter else PLAIN for _ in names]
        j = rng.randrange(self.pairs)                                    # queried: one of the pairs
        if rule:
            names[0], names[j] = names[j], names[0]
        order = list(range(n))
        if self.decoys:                                                  # the decoys (names[pairs:]) among the facts
            rng.shuffle(order)
        facts = [enc(stems[i].format(n=names[i])) + enc(f" {values[i]}") + enc(".") for i in order]
        heads = [len(enc(stems[i].format(n=names[i]))) for i in order]
        r = order.index(j)
        query, answer = enc(PLAIN.format(n=names[j])), enc(f" {values[j]}")
        ids, value_at, natural = self._place(rng, facts, heads, query, answer)
        ids, q_len = self._requery(ids, query, answer, names[j], natural)
        gap = (len(ids) - len(answer)) - value_at[r]
        self.last_decoy_ranks = [r_ for r_, i in enumerate(order) if i >= self.pairs]
        self._record(r, value_at, natural, heads, q_len, facts)
        return ids, len(answer), gap

    def _requery(self, ids: list[int], query: list[int], answer: list[int], name: str,
                 natural: bool) -> tuple[list[int], int]:
        """``ids`` (laid out with the plain ``query``) with ``self.query`` in its place: (ids, query length).

        The plain template returns ``ids`` untouched and draws nothing. Otherwise the last filler loses
        (or gains) the length difference just before the query, so nothing before it moves and the
        task's rng is never drawn from: the next sample is the plain cell's next sample."""
        if self.query == PLAIN:
            return ids, len(query)
        para = self.tok.encode(self.query.format(n=name))
        d = len(para) - len(query)
        if d >= self.tail or (self.block and self.tail - d <= self.block):   # the last fact stays out of reach
            raise ValueError(f"the paraphrased query is {d} tokens longer than the plain one: --needle-tail "
                             f"{self.tail} leaves {self.tail - d} filler tokens, not more than the window {self.block}")
        head = ids[:len(ids) - len(query) - len(answer)]
        if d > 0:
            head = head[:-d]
        elif d < 0:
            if natural:                                                  # the corpus is whole repeats: wrap around
                head = head + [self.corpus[(self._fill_end + i) % len(self.corpus)] for i in range(-d)]
            else:
                head = head + [self.pool[self._pad_rng.randrange(len(self.pool))] for _ in range(-d)]
        return head + para + answer, len(para)

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
                self._fill_end = o + k                                     # where the last filler stops (_requery)
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
    ``same`` repeats the template (pure recency), ``now`` adds " now", ``new`` says "The new code",
    ``changed`` says " U's code has changed from OLD to NEW." and ``nowold`` " The code for U is now
    NEW (it used to be OLD)." (both name the stale code, so recency alone does not decide). In every
    form ``value_at`` is the space before NEW's digits. The answer is the new code. The layout
    records cover all ``pairs + k + decoys`` statements, with ``last_queried`` the last update's
    rank, ``last_old_rank`` the previous statement of U and ``last_old_answer`` its code's tokens
    (to score the stale answer), and ``last_update_ranks`` every statement of U in order.

    ``k`` (``--update-k``) restates U k times, each code's digit 1 differing from the previous
    code's; the last wins. ``k = 1`` with no decoys and the forms same/now/new draws the same random
    numbers as before these options existed. A ``query`` template (``--paraphrase``) phrases the query
    as in the scattered task; the stale answer is scored on the same paraphrased query.
    """

    def __init__(self, *a, form: str = "same", k: int = 1, **kw) -> None:
        super().__init__(*a, **kw)
        if form not in UPDATE_STEMS:
            raise ValueError(form)
        if k < 1:
            raise ValueError(f"update k must be ≥ 1, got {k}")
        if not 1 <= k <= 9:
            raise ValueError("update k must be in 1..9 (each restated code's digit 1 differs from all earlier ones)")
        self.form, self.k = form, k

    def sample(self, rng):
        enc = self.tok.encode
        n = self.pairs + self.decoys
        names, rule = self._draw_names(rng, n)
        codes = [code4(rng) for _ in names]
        u = rng.randrange(self.pairs)                                      # the updated (queried) name
        if rule:
            names[0], names[u] = names[u], names[0]
        news, used = [], {codes[u][0]}
        for _ in range(self.k):                                            # digit 1 tells each from every earlier code
            new = codes[u]
            while new[0] in used:
                new = code4(rng)
            news.append(new)
            used.add(new[0])
        order = list(range(n))
        rng.shuffle(order)
        stmts = [(i, codes[i], PLAIN.format(n=names[i]), ".") for i in order]   # (name index, code, head, tail)
        ranks, olds = [order.index(u)], [codes[u]]
        head, tail = UPDATE_STEMS[self.form], UPDATE_TAILS.get(self.form, ".")
        for new in news:                                                   # each update after the previous one
            at = rng.randint(ranks[-1] + 1, len(stmts))                    # (inserting later shifts no earlier rank)
            stmts.insert(at, (u, new, head.format(n=names[u], old=olds[-1]), tail.format(old=olds[-1])))
            ranks.append(at)
            olds.append(new)
        facts = [enc(h) + enc(f" {v}") + enc(t) for _, v, h, t in stmts]
        heads = [len(enc(h)) for _, _, h, _ in stmts]
        query, answer = enc(PLAIN.format(n=names[u])), enc(f" {news[-1]}")
        ids, value_at, natural = self._place(rng, facts, heads, query, answer)
        ids, q_len = self._requery(ids, query, answer, names[u], natural)
        gap = (len(ids) - len(answer)) - value_at[ranks[-1]]
        self.last_old_rank, self.last_old_answer = ranks[-2], enc(f" {olds[-2]}")
        self.last_update_ranks = ranks
        self.last_decoy_ranks = [r_ for r_, st in enumerate(stmts) if st[0] >= self.pairs]
        self._record(ranks[-1], value_at, natural, heads, q_len, facts)
        return ids, len(answer), gap


class OrderNeedleTask(ScatterNeedleTask):
    """Order queries over the scattered facts (G0.7): the answer is a fact chosen by its place in
    the order of statement, not by its name.

    ``kind``: ``before`` " The code stated just before NAME's is", ``next`` " The code stated just
    after NAME's is", ``first`` " The first code stated is", ``last`` " The last code stated is".
    The facts are the ``pairs`` plain statements in random order. The answer is the chosen fact's
    code, ``last_queried`` its rank (so ``value_at`` and the slot diagnostics stay valid) and
    ``last_anchor_rank`` the rank of the fact named in the query (= the answer for first/last).
    Under the name rule the anchor's name comes from the configured list.
    """

    def __init__(self, *a, kind: str = "before", **kw) -> None:
        super().__init__(*a, **kw)
        if kind not in ORDER_QUERIES:
            raise ValueError(kind)
        if self.query != PLAIN:
            raise ValueError("the order task has its own queries: no paraphrased query template")
        if self.decoys:
            raise ValueError("the order task has no decoys: every statement can be the answer")
        if kind in ("before", "next") and self.pairs < 2:
            raise ValueError(f"--order {kind} needs ≥ 2 pairs")
        self.kind = kind

    def sample(self, rng):
        enc = self.tok.encode
        P = self.pairs
        names, rule = self._draw_names(rng, P)
        values = [code4(rng) for _ in names]
        if self.kind == "before":
            a = rng.randrange(1, P)
            r = a - 1
        elif self.kind == "next":
            a = rng.randrange(P - 1)
            r = a + 1
        else:
            a = r = 0 if self.kind == "first" else P - 1
        if rule:
            names[0], names[a] = names[a], names[0]
        facts = [enc(PLAIN.format(n=nm)) + enc(f" {v}") + enc(".") for nm, v in zip(names, values)]
        heads = [len(enc(PLAIN.format(n=nm))) for nm in names]
        query, answer = enc(ORDER_QUERIES[self.kind].format(n=names[a])), enc(f" {values[r]}")
        ids, value_at, natural = self._place(rng, facts, heads, query, answer)
        gap = (len(ids) - len(answer)) - value_at[r]
        self.last_anchor_rank = a
        self._record(r, value_at, natural, heads, len(query), facts)
        return ids, len(answer), gap


_train = pl.train
_teacher_ns = None                                                 # teacher.split_args namespace (set by main)
_teacher_stats = None                                              # what teacher.training recorded
_init_from = []                                                    # --init-from: [(path, fragment)], set by main
_init_loaded = None                                                # what load_init loaded (recorded in the JSON)


def read_fragments(paths: list[str]) -> list[tuple[str, dict]]:
    """``--init-from`` files → [(path, {name: tensor})], refusing what cannot be a state-dict fragment: a
    missing file, a non-dict, a non-tensor value, an empty fragment, or a key that two fragments share."""
    out, seen = [], {}
    for p in paths:
        if not Path(p).is_file():
            raise SystemExit(f"--init-from {p}: no such file")
        frag = torch.load(p, map_location="cpu")
        if not isinstance(frag, dict) or not frag:
            raise SystemExit(f"--init-from {p}: expected a non-empty dict of tensors (a state-dict fragment)")
        bad = sorted(k for k, v in frag.items() if not isinstance(k, str) or not torch.is_tensor(v))
        if bad:
            raise SystemExit(f"--init-from {p}: not tensors: {bad}")
        twice = sorted(k for k in frag if k in seen)
        if twice:
            raise SystemExit(f"--init-from {p}: {twice} also in {seen[twice[0]]} (one source per key)")
        seen.update({k: p for k in frag})
        out.append((p, frag))
    return out


def load_init(side, fragments: list[tuple[str, dict]]) -> list[dict]:
    """Load each fragment into ``side`` (strict=False), refusing keys the side does not have and wrong
    shapes; values are cast to the side's dtype and device by ``load_state_dict``. Returns
    [{"path", "keys"}] for the result JSON."""
    sd = side.state_dict()
    done = []
    for p, frag in fragments:
        foreign = sorted(set(frag) - set(sd))
        if foreign:
            raise SystemExit(f"--init-from {p}: keys not in the side: {foreign}")
        shape = sorted(k for k, v in frag.items() if tuple(v.shape) != tuple(sd[k].shape))
        if shape:
            raise SystemExit(f"--init-from {p}: shapes differ from the side's: "
                             + ", ".join(f"{k} {tuple(frag[k].shape)} vs {tuple(sd[k].shape)}" for k in shape))
        side.load_state_dict(frag, strict=False)
        done.append({"path": str(p), "keys": sorted(frag)})
    return done


def seeded_init(build, seed: int):
    """``build`` (the side's constructor) wrapped so that the side it returns is built with torch's CPU generator
    seeded at ``seed``, and the generator is left where the unseeded construction leaves it: a throwaway side is
    built first from the current state (it advances the generator exactly as a run without the flag does), then
    the returned one from ``seed``, then the throwaway's end state is restored. Every later torch draw (the
    window check, anything drawn in training) is therefore that of the run without the flag."""
    def wrapped(*a, **kw):
        throwaway = build(*a, **kw)
        after = torch.random.get_rng_state()
        del throwaway
        torch.random.default_generator.manual_seed(seed)
        try:
            return build(*a, **kw)
        finally:
            torch.random.set_rng_state(after)
    return wrapped


def train(lm, task, args):
    """Activate a task curriculum for the duration of training only. With an A14/A15 flag on, train
    through ``teacher.training`` (a task proxy and a patched ``pl.scores``); with both off, exactly as
    before: ``pl.train`` gets the task itself and the unpatched ``pl.scores``. With ``--init-from`` the
    fragments are loaded into the side first (before the optimiser exists); without it nothing is."""
    global _teacher_stats, _init_loaded
    if _init_from:
        _init_loaded = load_init(lm.side, _init_from)
    task._active = bool(getattr(task, "curriculum", None))
    try:
        if _teacher_ns is None or not teacher.active(_teacher_ns):
            return _train(lm, task, args)
        with teacher.training(lm, task, args, _teacher_ns) as (proxy, stats):
            _teacher_stats = stats
            curve = _train(lm, proxy, args)
            kinds = stats.get("kinds") or []
            for c in curve if isinstance(curve, list) else []:   # "d": KL / agreement; "n": needle CE / accuracy
                if isinstance(c, dict) and "step" in c and 0 < c["step"] <= len(kinds):
                    c["kind"] = kinds[c["step"] - 1]
            return curve
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
    ap.add_argument("--needle-names", choices=["all", "train", "pool"], default="all",
                    help="with --needle-scatter: train = the first 20 names only (the rest are held out); "
                         "pool = a synthetic syllable-name pool (--needle-name-pool), training only")
    ap.add_argument("--needle-name-pool", type=int, default=2000, metavar="N",
                    help="with --needle-names pool: the pool size (names.syllable_pool(N, 0))")
    ap.add_argument("--needle-tail", type=int, default=256, metavar="T",
                    help="with --needle-scatter: minimum filler between the last fact and the query (> --window)")
    ap.add_argument("--init-from", action="append", default=[], metavar="PT",
                    help="a state-dict fragment loaded into the side channels before training (repeatable)")
    ap.add_argument("--init-seed", type=int, default=None, metavar="N",
                    help="seed torch with N for the side's construction only (data order and the base unchanged)")
    own, rest = ap.parse_known_args(argv)
    cand_ns = None
    if own.candidate:                                              # refuse before patching anything
        mod_name, cls = own.candidate.split(":")
        mod = importlib.import_module(mod_name)
        Cand = getattr(mod, cls)
        if hasattr(mod, "split_args"):
            cand_ns, rest = mod.split_args(rest)
        if cand_ns is not None and any(k.endswith("_keep_by") and v == "external" for k, v in vars(cand_ns).items()):
            raise SystemExit("--w1s-keep-by external is an evaluation-time diagnostic (eval_by_query.py "
                             "--keep-oracle); training must not use it")
    global _teacher_ns, _teacher_stats
    tns, rest = teacher.split_args(rest)                           # A14 / A15 (default off)
    teacher.check(tns, pl.parse_args(rest), own.needle_scatter, cand_ns)
    _teacher_ns, _teacher_stats = (tns if teacher.active(tns) else None), None
    global _init_from, _init_loaded
    _init_from, _init_loaded = [], None
    if own.init_from:                                              # refuse before loading the model
        if pl.parse_args(rest).steps == 0:
            raise SystemExit("--init-from initialises training: it needs --steps > 0 "
                             "(evaluate a fragment with eval_by_query.py --w1s-load-extra)")
        _init_from = read_fragments(own.init_from)
    if own.needle_names != "all" and not own.needle_scatter:
        raise SystemExit("--needle-names needs --needle-scatter")
    if own.needle_tail != 256 and not own.needle_scatter:
        raise SystemExit("--needle-tail needs --needle-scatter")
    pl.evaluate = evaluate
    pl.train = train
    base_lm, base_scores = pl.HookedLM, pl.scores                  # restored after the run
    if own.candidate:
        parsed = pl.parse_args(rest)
        pl.SideChannels = lambda d_model, **_: Cand(d_model, parsed, cand_ns)
        if getattr(mod, "TAPS_REQUIRED", False):                   # the side reads earlier blocks too
            from evolve.taps import TapLM
            pl.HookedLM = TapLM
        if oracle_gate(cand_ns):
            if not own.needle_scatter:
                raise SystemExit("an oracle gate needs --needle-scatter (the onsets come from its layout)")
            pl.scores = scores
    if own.needle_scatter:                                         # after the candidate took its flags
        gap, window, mix, tail = own.needle_scatter, pl.parse_args(rest).window, own.needle_mix, own.needle_tail
        cur = [tuple(int(x) for x in st.split(":")) for st in own.needle_curriculum.split(",") if st]
        names = None
        if own.needle_names == "train":
            names = TRAIN_NAMES
        elif own.needle_names == "pool":
            from evolve.names import syllable_pool
            names = syllable_pool(own.needle_name_pool, 0)
        if window and tail <= window:                              # before loading the model
            raise SystemExit(f"--needle-tail {tail} must exceed --window {window}")
        pl.NeedleTask = lambda *a, **kw: ScatterNeedleTask(*a, min_gap=gap, tail=tail, block=window, mix=mix,
                                                           curriculum=cur, names=names, **kw)
    base_side = pl.SideChannels
    if own.init_seed is not None:                                  # the side's initialisation only
        pl.SideChannels = seeded_init(base_side, own.init_seed)
    try:
        results = pl.main(rest)
    finally:
        pl.HookedLM, pl.scores, pl.SideChannels = base_lm, base_scores, base_side
    args = pl.parse_args(rest)
    out = Path(args.out or f"results/pretrained/pretrained_loop_{args.task}.json")
    side = _last_lm.side if _last_lm is not None else None
    state = side.state_report() if hasattr(side, "state_report") else None
    teach = teacher.active(tns)
    record = own.candidate or own.needle_scatter or teach or _init_loaded or own.init_seed is not None
    if record:
        rec = json.loads(out.read_text())
    if own.candidate:                                             # record the candidate and its flags
        rec["candidate"] = {"class": own.candidate, "args": vars(cand_ns) if cand_ns else {}}
        if state is not None:                                     # streaming state size (T-independent)
            rec["w1s_state"] = state
    if own.needle_scatter:
        rec["needle_scatter"] = own.needle_scatter
        rec["needle_mix"] = own.needle_mix
        rec["needle_curriculum"] = own.needle_curriculum
        rec["needle_tail"] = own.needle_tail
        if own.needle_names != "all":                             # absent = all (older runs)
            rec["needle_names"] = own.needle_names
        if own.needle_names == "pool":
            rec["needle_name_pool"] = own.needle_name_pool
    if teach:                                                     # settings and mean losses of A14 / A15
        rec.update(teacher.summary(tns, _teacher_stats))
    if _init_loaded:                                              # --init-from: each fragment and its keys
        rec["init_from"] = _init_loaded
    if own.init_seed is not None:                                 # --init-seed: the side's construction seed
        rec["init_seed"] = own.init_seed
    if record:
        out.write_text(json.dumps(rec, indent=2))
    if _last_lm is not None:
        torch.save({k: v.cpu() for k, v in _last_lm.side.state_dict().items()}, out.with_suffix(".pt"))
        print(f"saved {out.with_suffix('.pt')}")
    return results


if __name__ == "__main__":
    main()
