"""CPU tests for experiments/ca2/items13.py (round 13 items): deterministic, statements never in the sink and always
one candidate span, cells within their distance bounds, dev / held-out halves disjoint and balanced, the question
identical between the prefix halves, update and paraphrase labels. Byte tokenizer, a synthetic filler of short
sentences in passages (no downloads)."""

import random
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import pretrained_loop as pl  # noqa: E402
from ca2 import items13 as I  # noqa: E402
from ca2 import selectors13 as S  # noqa: E402

TOK = pl.ByteTokenizer()
TB = S.TokenTables(TOK)
WORDS = "river mill town market church railway station bridge harbour castle valley forest garden road".split()
VERBS = "ran grew stood opened closed crossed faced joined".split()


def lines(n=200, seed=0):
    r = random.Random(seed)
    return ["".join(" the %s %s the %s ." % (r.choice(WORDS), r.choice(VERBS), r.choice(WORDS))
                    for _ in range(r.randint(4, 12))) for _ in range(n)]


W = 1024
CFG = I.ItemConfig(seq_len=4096, window=W, sink=2, n_statements=8, passage_min=150, passage_max=400, cap=96,
                   band_others=1)


@pytest.fixture(scope="module")
def builder():
    return I.ItemBuilder(TOK, TB, I.Corpus(TOK, TB, lines=lines()), I.Prompt(TOK, chat=False), CFG)


@pytest.mark.parametrize("typ", I.TYPES)
@pytest.mark.parametrize("cell", I.CELLS)
def test_item_invariants(builder, typ, cell):
    lo, hi = I.cells(W)[cell]
    for i in range(8):
        it = builder.build(typ, cell, i)
        assert len(it.ids) == CFG.seq_len
        n = CFG.n_plain if typ == "plain" else CFG.n_statements
        assert len(it.statements) == n + (typ == "update")
        spans = set(S.cut_spans(it.ids, TB, max(CFG.sink, it.n_pre), it.q0, CFG.cap))
        for f, l in it.statements:
            assert f >= max(CFG.sink, it.n_pre) + CFG.margin           # never in the sink
            assert (f, l + 1) in spans                                 # each statement is one candidate span
            assert l < it.q0
        assert it.statements == sorted(it.statements)
        qf, ql = it.statements[it.queried]
        d = it.q0 - ql if cell != "inside" else it.q0 - qf
        assert lo <= d <= hi, (it.uid, d)
        if cell == "beyond":
            assert all(it.q0 - l > W for _, l in it.statements)
        if cell == "near":
            assert all(it.q0 - l >= lo for _, l in it.statements)
        if cell != "beyond":
            inb = sum(lo <= (it.q0 - l if cell != "inside" else it.q0 - f) <= hi for f, l in it.statements)
            assert inb >= 1 + min(CFG.band_others, n - 1)
        txt = TOK.decode(it.ids[qf:ql + 1])
        assert it.answer in txt
        if typ == "update":
            of, ol = it.statements[it.old]
            assert ol < qf and it.stale and it.stale != it.answer and it.stale in TOK.decode(it.ids[of:ol + 1])
            assert " now " in txt
        else:
            assert it.old is None and it.stale is None
        # the question's own rows, then the foot, then the prefix (or nothing)
        assert TOK.decode(it.ids[it.q_rows[0]:it.q_rows[1]]) == "\n\nQuestion: " + it.question
        tail = TOK.decode(it.ids[it.q_rows[1]:])
        assert tail.startswith("\nAnswer:") and (len(tail) > len("\nAnswer:")) == it.prefix
        # passages cover the filler, statements lie between filler pieces
        filler = sum(b - a for a, b, _, _ in it.passages)
        assert filler + sum(l - f + 1 for f, l in it.statements) == it.q0 - it.n_pre
        m = it.meta()
        assert "question" not in m and all(not isinstance(v, str) or len(v) < 40 for v in m.values())


def test_deterministic_and_independent_of_n(builder):
    a = builder.items("jitter", "beyond", 6)
    b = builder.items("jitter", "beyond", 6)
    assert [x.ids for x in a] == [x.ids for x in b]
    c = builder.build("jitter", "beyond", 4)
    assert c.ids == a[4].ids and c.uid == a[4].uid == "jitter/beyond/0/4"
    other = I.ItemBuilder(TOK, TB, I.Corpus(TOK, TB, lines=lines()), I.Prompt(TOK, False),
                          I.ItemConfig(**{**CFG.__dict__, "seed": 1}))
    assert other.build("jitter", "beyond", 4).ids != c.ids
    assert len({x.uid for x in a}) == 6


def test_splits_disjoint_and_balanced(builder):
    for typ in ("decoy", "paraphrase"):
        its = builder.items(typ, "beyond", 16)
        dev = {x.uid for x in its if x.split == "dev"}
        held = {x.uid for x in its if x.split == "held"}
        assert not dev & held and len(dev) == len(held) == 8
        for sp in ("dev", "held"):
            pf = Counter(x.prefix for x in its if x.split == sp)
            assert pf[True] == pf[False] == 4
        if typ == "paraphrase":
            forms = Counter(x.form for x in its)
            assert forms["code"] == 8 and sum(forms[f] for f in ("date", "city", "colour")) == 8
            for sp in ("dev", "held"):
                assert Counter(x.form == "code" for x in its if x.split == sp)[True] == 4


def test_question_identical_between_prefix_halves(builder):
    """Same question text → the same question tokens, whatever the prefix flag."""
    pr = I.Prompt(TOK, chat=False)
    its = builder.items("decoy", "beyond", 8)
    for x in its:
        q = pr.question(x.question)
        assert x.ids[x.q_rows[0]:x.q_rows[1]] == q
    assert {x.prefix for x in its} == {True, False}


def test_paraphrase_names_the_entity_in_other_words(builder):
    for i in range(16):
        it = builder.build("paraphrase", "beyond", i)
        qf, ql = it.statements[it.queried]
        stmt = TOK.decode(it.ids[qf:ql + 1])
        name = stmt.split(",")[0].split()[-1]
        role = stmt.split(", the ")[1].split(",")[0]
        assert name not in it.question and role not in it.question
        assert dict(I.ROLES)[role] in it.question
        if it.form == "code":
            assert it.answer.isdigit() and len(it.answer) == 4
        # every statement has a different role, so the description picks one
        roles = [TOK.decode(it.ids[f:l + 1]).split(", the ")[1].split(",")[0] for f, l in it.statements]
        assert len(set(roles)) == len(roles)


def test_decoy_grid_is_regular(builder):
    it = builder.build("decoy", "beyond", 0)
    starts = [f for f, _ in it.statements]
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert max(gaps) < 3 * min(gaps) + 200                         # evenly spread (sentence boundaries snap)


def test_cells_bounds_at_the_real_window():
    c = I.cells(2048)
    assert c["near"] == (1600, 2040) and c["inside"] == (1, 1024) and c["beyond"][0] == 2049


# ------------------------------------------------------------------------- pinned beyond items (step 0 → 0b)
def _digest(b, typ, n):
    import hashlib
    import json
    h = hashlib.sha256()
    for i in range(n):
        it = b.build(typ, "beyond", i)
        h.update(json.dumps([it.ids, it.statements, it.queried, it.old, it.passages, it.q_rows, it.split, it.prefix,
                             it.form, it.answer, it.stale]).encode())
    return h.hexdigest()


PINNED_BYTE = {   # the step-0 generator's beyond items (byte tokenizer, synthetic filler), computed before step 0b
    "plain": "c5104844407c7aaae60d75b71ff3057f0ad5ec2be2a312dc73117e6418b985dc",
    "decoy": "7dd801bc04d912e30c10d42419ff5d47bbf0d06360afedd182ab1379776717f1",
    "jitter": "051e8521087b764e39a1a62b64683485c9d91eb523db3823a3ac4a01e1a9f143",
    "paraphrase": "26e720763cce8852195f618b6f5c988aa205827ad78f1449e1410ea99daf3a16",
    "update": "2dd62392aa7a7cf1117bae48bd70501ca9f9232932a4d2e2f55d12b755ba44ae",
}
PINNED_GEMMA = {  # the real tokenizers and corpus (default ItemConfig, seed 0, items 0-1)
    "google/recurrentgemma-2b": {
        "plain": "455b15cb42dfbee10e21566dec95011c1ee310e6b1a4efbd2446facb83d894de",
        "decoy": "453a5772e5ce766aed4f92b7b98502938bcffdead806deed30397c393bb343f3",
        "jitter": "7531363a30fd0091465dd5243c91a087a6cf34d937c37d75da8232c3aa5caa57",
        "paraphrase": "26cedd30e5d9182f4ac5632bc3beca01a3034c71ff719ee10000b958e5a0695b",
        "update": "a1c50cb4c429adccc8c18f8e5c844bbcba2e2dfd2beded056e4b18c118fa41ab"},
    "google/recurrentgemma-2b-it": {
        "plain": "a608d99a959c8215813f6b75f52a7b8ae68ae21e1c9d81779369da1b515984ce",
        "decoy": "763444cbaef03b2615e582f84b89bd82af0530b34829da1cc8d2df24d56ce695",
        "jitter": "bd0e8b20766428b358b5174745523b667c7c6fb919b76af0f75e471067df5810",
        "paraphrase": "6f8aafdd88c5c04da82867be642a1b4d0295ab615a5ea89a5a67f92e394c02ae",
        "update": "0655e23fc59ff91ca2b28b6943f461c2193244e2a8ac14db6bb6e5bd2e6ce3ad"},
}


@pytest.mark.parametrize("typ", I.TYPES)
def test_beyond_items_pinned(builder, typ):
    assert _digest(builder, typ, 4) == PINNED_BYTE[typ]


def _gemma_ready(m):
    import os
    root = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    return (root / ("models--" + m.replace("/", "--"))).exists() and Path(I.DEFAULT_FILLER).expanduser().exists()


@pytest.mark.parametrize("model", list(PINNED_GEMMA))
def test_beyond_items_pinned_real_tokenizer(model):
    if not _gemma_ready(model):
        pytest.skip("tokenizer or filler corpus not available")
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = pl.HFTokenizer(AutoTokenizer.from_pretrained(model))
    tb = S.TokenTables(tok)
    b = I.ItemBuilder(tok, tb, I.Corpus(tok, tb), I.Prompt(tok, chat=bool(tok.tok.chat_template)), I.ItemConfig())
    for typ in I.TYPES:
        assert _digest(b, typ, 2) == PINNED_GEMMA[model][typ], typ
