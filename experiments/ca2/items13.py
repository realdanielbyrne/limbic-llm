"""Round 13 items (plan §3): decoys, jittered positions, paraphrased questions, updates, three distance cells.

Every item is one prompt of exactly ``seq_len`` tokens: ``pre`` (BOS, or the chat template's head) + a document of
filler passages with statements inserted between sentences + the question (+ the chat template's foot) + an
optional answer prefix. Filler comes from the corpus the round-12 screen used (``~/limbic-data/
wikitext103_test.txt``, one paragraph per line): each passage is a run of whole sentences of one paragraph, of
``passage_min`` … ``passage_max`` tokens, starting at a random sentence of a random paragraph; passages follow each
other until the filler is long enough, and the last one is cut to the exact length. Statements go only at sentence
boundaries, so each statement is one candidate span of the span builder (checked).

Types (``TYPES``)
  plain       the round-12 form " The code for NAME is DDDD." with 4 statements (positions random), one queried;
              the question "What is the code for NAME? Answer with the 4-digit code.", prefix "The code for NAME is".
  decoy       24 statements of that form, one queried, on a regular grid of sentence boundaries (positions fixed by
              the layout, the queried slot random). Near / inside: ``band_others`` decoys also go in the queried band
              (step 0b; step 0 ignored it there), the rest on the grid. Beyond items are unchanged (pinned by test).
  jitter      decoy with the statement positions drawn at random (sentence boundaries, never in the sink).
  paraphrase  decoy + jitter, and the question names the entity in other words: the statement says "NAME, the ROLE"
              ("Ruiz, the pharmacist"), the question asks for "the person who dispenses medicine". Half the items use
              the code form, half another fact form (a date, a city, a colour; rotating).
  update      decoy + jitter; the queried name is stated twice with different codes (the second "… is now DDDD.",
              later); both spans are recorded, the later one is correct (round-12 ``now`` form).

Cells (``cells(W)``; W the attention window; d = first question row − last token of the queried statement)
  beyond      d > W; every statement of the item is beyond the window too (round-12 continuity).
  near        0.78125·W ≤ d ≤ W − 8 (1,600–2,040 at W = 2,048); the others anywhere at least 1,600 back, with
              min(3, n − 1) of them also in the band (so the queried fact is not simply the most recent).
  inside      the queried statement within the last W/2 tokens before the question (1,024); others anywhere, with
              min(3, n − 1) also in that band.

Per item: ``uid`` (type/cell/seed/index), ``split`` (dev / held; balanced in blocks of 8 together with the answer
prefix flag and, for paraphrase, the fact form, so both halves are balanced), ``prefix`` (False on half: no answer
prefix; the question tokens are identical in both halves), token spans of every statement and which is queried
(``queried``; ``old`` for update), ``passages`` (start, end, passage index, corpus line) of every filler piece in
final positions (the context-tag check), ``q_rows`` (the question's own tokens, the rows that drive the qk
selectors), ``q0`` (first row after the document). Deterministic in (seed, type, cell, index); text is kept in
memory only (never written to results).
"""

from __future__ import annotations

import random
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]

TYPES = ("plain", "decoy", "jitter", "paraphrase", "update")
CELLS = ("beyond", "near", "inside")
FORMS = ("code", "date", "city", "colour")
DEFAULT_FILLER = "~/limbic-data/wikitext103_test.txt"

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack", "Karen",
         "Liam", "Maria", "Noah", "Olivia", "Peter", "Quinn", "Rosa", "Sam", "Tara", "Victor", "Wendy",
         "Xavier", "Yara", "Zoe"]                                  # = pretrained_loop.NAMES (round 12)
SURNAMES = ["Ruiz", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Kowalski", "Haddad", "Fischer", "Novak", "Silva",
            "Brennan", "Castillo", "Dubois", "Eriksen", "Ferreira", "Gallagher", "Horvath", "Ivanova", "Jansen",
            "Kaplan", "Larsen", "Mendez", "Nakamura", "Oyelaran", "Petrov", "Quintero", "Rahman", "Schulz",
            "Takahashi", "Underwood", "Varga", "Whitaker", "Yilmaz", "Zielinski"]
ROLES = [("pharmacist", "the person who dispenses medicine"),
         ("surgeon", "the doctor who operates on patients"),
         ("librarian", "the person in charge of the lending shelves"),
         ("gardener", "the person who looks after the plants"),
         ("plumber", "the person who fixes the pipes"),
         ("electrician", "the person who repairs the wiring"),
         ("baker", "the person who makes the bread"),
         ("pilot", "the person who flies the plane"),
         ("lawyer", "our legal counsel"),
         ("accountant", "the person who handles the finances"),
         ("janitor", "the person who cleans the building"),
         ("dentist", "the person who looks after our teeth"),
         ("veterinarian", "the animal doctor"),
         ("mechanic", "the person who repairs the cars"),
         ("architect", "the person who designs the buildings"),
         ("cashier", "the person at the till"),
         ("translator", "the person who interprets foreign languages"),
         ("photographer", "the person who takes the pictures"),
         ("carpenter", "the person who builds the furniture"),
         ("nurse", "the person who cares for patients on the ward"),
         ("teacher", "the person who instructs the class"),
         ("chef", "the head cook"),
         ("firefighter", "the person who puts out fires"),
         ("tailor", "the person who alters the clothes"),
         ("receptionist", "the person at the front desk"),
         ("courier", "the person who delivers the parcels"),
         ("butcher", "the person who sells the meat"),
         ("programmer", "the person who writes the software"),
         ("journalist", "the reporter"),
         ("farmer", "the person who grows the crops")]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
CITIES = ["Lisbon", "Nairobi", "Osaka", "Toronto", "Krakow", "Valparaiso", "Tbilisi", "Dublin", "Hanoi", "Bergen",
          "Marseille", "Adelaide", "Cusco", "Tallinn", "Accra", "Seville", "Winnipeg", "Busan", "Porto", "Lyon",
          "Medellin", "Gdansk", "Zagreb", "Leipzig", "Bilbao", "Kyoto", "Halifax", "Durban", "Utrecht", "Tampere"]
COLOURS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "grey", "silver",
           "gold", "teal", "navy", "maroon", "olive", "lime", "cyan", "magenta", "beige", "ivory", "violet", "indigo",
           "turquoise", "crimson", "amber", "lavender"]

PLAIN_STMT = " The code for {n} is {v}."
UPDATE_STMT = " The code for {n} is now {v}."
PLAIN_Q = "What is the code for {n}? Answer with the 4-digit code."
UPDATE_Q = "What is the current code for {n}? Answer with the 4-digit code."
PLAIN_PREFIX = "The code for {n} is"
PARA = {   # statement (name, role, value), question (description), answer prefix (description)
    "code": (" The access code of {n}, the {r}, is {v}.", "What is the access code of {d}?",
             "The access code of {d} is"),
    "date": (" {n}, the {r}, was born on {v}.", "On what date was {d} born?", "{D} was born on"),
    "city": (" {n}, the {r}, lives in {v}.", "In which city does {d} live?", "{D} lives in"),
    "colour": (" The favourite colour of {n}, the {r}, is {v}.", "What is the favourite colour of {d}?",
               "The favourite colour of {d} is"),
}


def cells(W: int) -> dict[str, tuple[int, int]]:
    """Distance bands (lo, hi) of d = q0 − last token of the queried statement (inside: q0 − first token)."""
    return {"beyond": (W + 1, 10 ** 9), "near": (int(round(0.78125 * W)), W - 8), "inside": (1, W // 2)}


@dataclass
class Item13:
    uid: str
    type: str
    cell: str
    index: int
    split: str
    prefix: bool
    form: str
    ids: list[int]
    q0: int                                                        # first row after the document
    q_rows: tuple[int, int]                                        # the question's own tokens [a, b)
    statements: list[tuple[int, int]]                              # (first, last) inclusive, position order
    queried: int                                                   # index into statements
    old: int | None                                                # update: the earlier statement of that name
    passages: list[tuple[int, int, int, int]]                      # (start, end excl, passage idx, corpus line)
    answer: str
    stale: str | None
    question: str = field(repr=False, default="")                  # text: memory only
    seed: int = 0
    n_pre: int = 0                                                 # tokens before the document (BOS / chat head)
    n_foot: int = 0                                                # template foot after the question (before prefix)

    @property
    def foot_rows(self) -> tuple[int, int]:
        return (self.q_rows[1], self.q_rows[1] + self.n_foot)

    def meta(self) -> dict:
        """JSON-safe labels (no text)."""
        q = self.statements[self.queried]
        out = {"uid": self.uid, "type": self.type, "cell": self.cell, "split": self.split, "prefix": self.prefix,
               "form": self.form, "n_statements": len(self.statements), "queried_rank": self.queried,
               "gap": self.q0 - q[1], "q0": self.q0, "n_passages": len({p[2] for p in self.passages})}
        if self.old is not None:
            out["old_rank"] = self.old
            out["old_gap"] = self.q0 - self.statements[self.old][1]
        return out


# ----------------------------------------------------------------------------------------------- corpus
class Corpus:
    """The filler corpus, one paragraph per line, tokenized lazily and split into sentences."""

    def __init__(self, tok, tables, path: str | Path = DEFAULT_FILLER, min_tokens: int = 40,
                 lines: list[str] | None = None) -> None:
        from ca2.selectors13 import cut_spans
        self._cut = cut_spans
        self.tok, self.tables = tok, tables
        if lines is None:
            lines = Path(path).expanduser().read_text(encoding="utf-8", errors="replace").split("\n")
        self.lines = [ln for ln in lines if ln.strip()]
        self.min_tokens = min_tokens
        self._cache: dict[int, list[list[int]]] = {}

    def sentences(self, i: int) -> list[list[int]]:
        if i not in self._cache:
            ids = self.tok.encode(self.lines[i].rstrip())
            sp = self._cut(ids, self.tables, 0, len(ids), cap=10 ** 9)
            self._cache[i] = [ids[a:b] for a, b in sp]
        return self._cache[i]

    def passage(self, rng: random.Random, lo: int, hi: int) -> tuple[int, list[list[int]]]:
        """(line, sentences): whole sentences from one paragraph, about ``lo`` … ``hi`` tokens."""
        for _ in range(1000):
            i = rng.randrange(len(self.lines))
            ss = self.sentences(i)
            if sum(map(len, ss)) < self.min_tokens:
                continue
            want = rng.randint(lo, hi)
            j = rng.randrange(len(ss))
            out = []
            while j < len(ss) and sum(map(len, out)) < want:
                out.append(ss[j])
                j += 1
            if sum(map(len, out)) >= min(lo, self.min_tokens):
                return i, out
        raise RuntimeError("the corpus has no usable paragraphs")


# ----------------------------------------------------------------------------------------------- prompt
class Prompt:
    """pre (BOS or chat head), and the question tail: q_ids (the question), foot (chat), prefix ids."""

    def __init__(self, tok, chat: bool) -> None:
        self.tok, self.chat = tok, chat
        if chat:
            hf = tok.tok
            mark = "XQXDOCXQX"
            text = hf.apply_chat_template([{"role": "user", "content": mark}], tokenize=False,
                                          add_generation_prompt=True)
            head, foot = text.split(mark)
            self.pre = tok.encode(head)                            # the template spells <bos> out
            self.foot = tok.encode(foot)
        else:
            self.pre = [tok.bos_token_id]
            self.foot = tok.encode("\nAnswer:")

    def question(self, q: str) -> list[int]:
        return self.tok.encode("\n\n" + q) if self.chat else self.tok.encode("\n\nQuestion: " + q)

    def prefix(self, p: str) -> list[int]:
        return self.tok.encode(p if self.chat else " " + p)


# -------------------------------------------------------------------------------------------------- builder
@dataclass
class ItemConfig:
    seq_len: int = 8192
    window: int = 2048
    sink: int = 4
    n_statements: int = 24
    n_plain: int = 4
    passage_min: int = 120
    passage_max: int = 480
    cap: int = 48
    seed: int = 0
    band_others: int = 3                                           # near / inside: other statements in the band
    margin: int = 16                                               # statements start ≥ max(sink, len(pre)) + margin


def _design(seed: int, typ: str, cell: str, i: int) -> tuple[str, bool, str]:
    """(split, prefix, form) for item i: balanced in blocks of 8 (2 splits × 2 prefix × 2 form classes)."""
    blk = i // 8
    combos = [(s, p, f) for s in ("dev", "held") for p in (True, False) for f in (0, 1)]
    random.Random(f"{seed}:{typ}:{cell}:design:{blk}").shuffle(combos)
    s, p, f = combos[i % 8]
    if typ == "paraphrase":
        form = "code" if f == 0 else ("date", "city", "colour")[(i // 8) % 3]
    else:
        form = "code"
    return s, p, form


def _code(rng) -> str:
    return "".join(rng.choice("0123456789") for _ in range(4))


def _distinct(rng, fn, n: int) -> list[str]:
    out: list[str] = []
    while len(out) < n:
        v = fn(rng)
        if v not in out:
            out.append(v)
    return out


class ItemBuilder:
    def __init__(self, tok, tables, corpus: Corpus, prompt: Prompt, cfg: ItemConfig) -> None:
        self.tok, self.tables, self.corpus, self.prompt, self.cfg = tok, tables, corpus, prompt, cfg

    # ---------------------------------------------------------------------------------- texts
    def _texts(self, rng, typ: str, form: str):
        """Statement texts (in a random statement order), the queried index (and old), question, prefix, answer."""
        cfg = self.cfg
        n = cfg.n_plain if typ == "plain" else cfg.n_statements
        if typ in ("plain", "decoy", "jitter", "update"):
            if n > len(NAMES):
                raise ValueError(f"{n} statements > {len(NAMES)} names")
            names = rng.sample(NAMES, n)
            vals = _distinct(rng, _code, n + 1)
            stm = [PLAIN_STMT.format(n=names[i], v=vals[i]) for i in range(n)]
            q = rng.randrange(n)
            old = None
            if typ == "update":
                new = vals[n]
                while new[0] == vals[q][0]:                       # digit 1 tells the new code from the old
                    new = _code(rng)
                stm.append(UPDATE_STMT.format(n=names[q], v=new))
                old, q = q, n                                     # the update is the queried statement
                question = UPDATE_Q.format(n=names[old])
                prefix = PLAIN_PREFIX.format(n=names[old])
                return stm, q, old, question, prefix, new, vals[old]
            return stm, q, None, PLAIN_Q.format(n=names[q]), PLAIN_PREFIX.format(n=names[q]), vals[q], None
        # paraphrase
        if n > len(ROLES):
            raise ValueError(f"{n} statements > {len(ROLES)} roles")
        roles = rng.sample(ROLES, n)
        sur = rng.sample(SURNAMES, n)
        if form == "code":
            vals = _distinct(rng, _code, n)
        elif form == "date":
            vals = _distinct(rng, lambda r: f"{r.choice(MONTHS)} {r.randint(1, 28)}", n)
        elif form == "city":
            vals = rng.sample(CITIES, n)
        else:
            vals = rng.sample(COLOURS, n)
        st, qt, pt = PARA[form]
        stm = [st.format(n=sur[i], r=roles[i][0], v=vals[i]) for i in range(n)]
        q = rng.randrange(n)
        desc = roles[q][1]
        D = desc[0].upper() + desc[1:]
        return stm, q, None, qt.format(d=desc), pt.format(d=desc, D=D), vals[q], None

    # ---------------------------------------------------------------------------------- layout
    def build(self, typ: str, cell: str, i: int) -> Item13:
        cfg = self.cfg
        if typ not in TYPES or cell not in CELLS:
            raise ValueError((typ, cell))
        split, has_prefix, form = _design(cfg.seed, typ, cell, i)
        rng = random.Random(f"{cfg.seed}:{typ}:{cell}:{i}")
        stm_txt, q, old, question, prefix_txt, answer, stale = self._texts(rng, typ, form)
        enc = self.tok.encode
        stm = [enc(s) for s in stm_txt]
        for s in stm:
            if len(s) > cfg.cap:
                raise ValueError(f"a statement takes {len(s)} tokens > the span cap {cfg.cap}")
        pre = list(self.prompt.pre)
        q_ids = self.prompt.question(question)
        foot = list(self.prompt.foot)
        pfx = self.prompt.prefix(prefix_txt) if has_prefix else []
        tail_len = len(q_ids) + len(foot) + len(pfx)
        body = cfg.seq_len - len(pre) - tail_len
        F = body - sum(map(len, stm))
        if F < cfg.seq_len // 4:
            raise ValueError(f"seq_len {cfg.seq_len} too short for {len(stm)} statements")
        # filler: passages of whole sentences, cut to exactly F tokens
        sents, src = [], []                                        # sentence tokens; (passage idx, line)
        tot, pi = 0, 0
        while tot < F:
            line, ss = self.corpus.passage(rng, cfg.passage_min, cfg.passage_max)
            for s in ss:
                if tot >= F:
                    break
                s = s[:F - tot]
                sents.append(s)
                src.append((pi, line))
                tot += len(s)
            pi += 1
        q0 = len(pre) + body
        bounds = [0]                                               # filler offsets of sentence boundaries
        for s in sents:
            bounds.append(bounds[-1] + len(s))
        nb = len(sents)                                            # boundary j: before sentence j (j = 0 … nb − 1)
        lo_b = next(j for j in range(nb) if len(pre) + bounds[j] >= max(cfg.sink, len(pre)) + cfg.margin)
        tb = self.tables
        # a statement may go at boundary j only if the span builder cuts on both sides of it: the sentence before
        # ends a sentence, and the sentence after starts with whitespace
        okb = {j for j in range(lo_b, nb) if bool(tb.end[sents[j - 1][-1]] or tb.newline[sents[j - 1][-1]])
               and bool(tb.space[sents[j][0]])}
        place = self._layout(rng, typ, cell, stm, q, old, bounds, okb, nb, len(pre), q0)
        # assemble
        at: dict[int, int] = {b: k for k, b in enumerate(place)}  # boundary → statement index (one per boundary)
        ids, spans_by_stmt, passages = list(pre), {}, []
        for j, s in enumerate(sents):
            if j in at:
                k = at[j]
                spans_by_stmt[k] = (len(ids), len(ids) + len(stm[k]) - 1)
                ids += stm[k]
            passages.append((len(ids), len(ids) + len(s), src[j][0], src[j][1]))
            ids += s
        ids += q_ids + foot + pfx
        assert len(ids) == cfg.seq_len, (len(ids), cfg.seq_len)
        order = sorted(spans_by_stmt, key=lambda k: spans_by_stmt[k][0])
        statements = [spans_by_stmt[k] for k in order]
        qi = order.index(q)
        oi = order.index(old) if old is not None else None
        uid = f"{typ}/{cell}/{cfg.seed}/{i}"
        return Item13(uid, typ, cell, i, split, has_prefix, form, ids, q0, (q0, q0 + len(q_ids)), statements, qi, oi,
                      passages, answer, stale, question, zlib.crc32(uid.encode()), len(pre), len(foot))

    def _finals(self, place: list[int], stm: list[list[int]], bounds: list[int], npre: int) -> list[tuple[int, int]]:
        """(first, last) final positions of each statement for boundary choices ``place`` (one per statement)."""
        order = sorted(range(len(place)), key=lambda k: place[k])
        out = [None] * len(place)
        shift = 0
        for k in order:
            a = npre + bounds[place[k]] + shift
            out[k] = (a, a + len(stm[k]) - 1)
            shift += len(stm[k])
        return out

    def _ok(self, fin, cell, q, old, q0, band, others_band: set[int]) -> bool:
        lo, hi = band
        d = q0 - fin[q][1] if cell != "inside" else q0 - fin[q][0]
        if not lo <= d <= hi:
            return False
        if old is not None and not fin[old][1] < fin[q][0]:
            return False
        W = self.cfg.window
        for k, (a, b) in enumerate(fin):
            if k == q:
                continue
            if cell == "beyond" and not q0 - b > W:
                return False
            if cell == "near" and q0 - b < cells(W)["near"][0]:
                return False
            if k in others_band:
                dk = q0 - b if cell != "inside" else q0 - a
                if not lo <= dk <= hi:
                    return False
        return True

    def _layout(self, rng, typ, cell, stm, q, old, bounds, okb, nb, npre, q0) -> list[int]:
        cfg = self.cfg
        W = cfg.window
        band = cells(W)[cell]
        n = len(stm)
        others = [k for k in range(n) if k != q and k != old]
        nband = 0 if cell == "beyond" else min(cfg.band_others, len(others))
        for attempt in range(4000):
            # allowed boundary ranges in filler offsets (approximate; checked exactly below)
            S = sum(map(len, stm))
            if cell == "beyond":
                hi_off = q0 - W - S - npre - 2
                pool_q = pool_o = [j for j in sorted(okb) if bounds[j] <= hi_off]
            else:
                lo, hi = band
                pool_q = [j for j in sorted(okb) if lo - 2 * S <= q0 - npre - bounds[j] <= hi + 2 * S]
                min_d = cells(W)["near"][0] if cell == "near" else 0
                pool_o = [j for j in sorted(okb) if q0 - npre - bounds[j] - S >= min_d]
            if len(pool_o) < n or not pool_q:
                raise ValueError(f"{typ}/{cell}: too few sentence boundaries for {n} statements")
            place = [None] * n
            lo, hi = band

            def free(pool, used):
                return [j for j in pool if j not in used]

            def banded(pool, k, used):
                """Boundaries in ``pool`` where statement k would land in the band, given the placements so far."""
                out = []
                for j in free(pool, used):
                    sh = sum(len(stm[m]) for m, bj in enumerate(place) if bj is not None and bj < j)
                    a0 = npre + bounds[j] + sh
                    d = q0 - (a0 + len(stm[k]) - 1) if cell != "inside" else q0 - a0
                    if lo <= d <= hi:
                        out.append(j)
                return out
            band_set: set[int] = set()
            if typ == "decoy":
                if cell == "beyond":
                    grid = [pool_o[int(round(x * (len(pool_o) - 1) / max(1, n - 1)))] for x in range(n)]
                    s_q = rng.randrange(n)
                    place[q] = grid[s_q]
                    rest = [g for s_, g in enumerate(grid) if s_ != s_q]
                    for k, g in zip([k for k in range(n) if k != q], rest):
                        place[k] = g
                else:
                    # near / inside: ``band_others`` decoys go in the queried band too (as for the random layouts),
                    # the rest on a regular grid over the others' region
                    band_set = set(rng.sample(others, nband)) if nband else set()
                    grid_k = [k for k in others if k not in band_set]
                    m = len(grid_k)
                    rest = [pool_o[int(round(x * (len(pool_o) - 1) / max(1, m - 1)))] for x in range(m)]
                    for k, g in zip(grid_k, rest):
                        place[k] = g
                    used = set(rest)
                    cq = banded(sorted(okb), q, used)
                    if not cq:
                        continue
                    place[q] = rng.choice(cq)
                    used.add(place[q])
                    miss = False
                    for k in sorted(band_set):
                        cands = banded(sorted(okb), k, used)
                        if not cands:
                            miss = True
                            break
                        place[k] = rng.choice(cands)
                        used.add(place[k])
                    if miss:
                        continue
            else:
                band_set = set(rng.sample(others, nband)) if nband else set()
                used: set[int] = set()
                for k in others:
                    if k in band_set:
                        continue
                    place[k] = rng.choice(free(pool_o, used))
                    used.add(place[k])
                cq = banded(sorted(okb) if cell != "beyond" else pool_q, q, used)
                if not cq:
                    continue
                place[q] = rng.choice(cq)
                used.add(place[q])
                if old is not None:
                    cands = [j for j in free(pool_o, used) if j < place[q]]
                    if not cands:
                        continue
                    place[old] = rng.choice(cands)
                    used.add(place[old])
                miss = False
                for k in sorted(band_set):
                    cands = banded(sorted(okb), k, used)
                    if not cands:
                        miss = True
                        break
                    place[k] = rng.choice(cands)
                    used.add(place[k])
                if miss:
                    continue
            if len(set(place)) < n:
                if typ == "decoy" and cell == "beyond":
                    raise ValueError("grid slots collide; raise seq_len or lower n_statements")
                continue                                           # the queried slot hit a grid slot: draw again
            fin = self._finals(place, stm, bounds, npre)
            if self._ok(fin, cell, q, old, q0, band, band_set):
                return place
            if typ == "decoy" and cell == "beyond":
                raise ValueError("the decoy grid does not satisfy the beyond cell")
        raise ValueError(f"{typ}/{cell}: no layout satisfies the cell after 4000 attempts")

    def items(self, typ: str, cell: str, n: int, start: int = 0) -> list[Item13]:
        return [self.build(typ, cell, i) for i in range(start, start + n)]


def answer_digits(text: str) -> str:
    return re.sub(r"\D", "", text)
