"""Reasoning-speed runner (round 12, Stage 0c reasoning and Stage 1c): does the CA2 injection reach answers sooner?

Generation. Each problem is rendered with the model's chat template, thinking switched on explicitly where the
model has the switch (``enable_thinking`` is always passed: Qwen3.5 defaults it off, Qwen3 on), and the prompt
always ends in ``<think>\\n`` (as the notebook's ``render``), so the generated text starts inside the reasoning.
Rows are sampled (greedy thinking loops; T = 0.6, top-p 0.95, top-k 20 by default) in left-padded batches with
the KV cache, with a ``CA2Controller`` attached at block L (``CA2Attachment``) and every tap the model supports
(``NormTap``; ``AttnTap`` at the last full-attention block R ≤ L; ``GDNTaps`` on a hybrid). Stop ids are the
tokenizer's eos plus ``<|im_end|>`` and ``<|endoftext|>`` for chat models (Qwen3.5-2B ships no
generation_config, so ``run.eos_ids`` alone would miss ``<|im_end|>``).

Steering span. The injection runs from the last prompt token (N1) and only inside the reasoning span:
``SpanController`` (a ``CA2Controller`` whose steer mask is also zero from the first ``</think>`` input on,
per row; one (B,) bool of state) sets g_t = 0 at ``</think>`` and after it. Without a thinking span (no-think
render, non-thinking model) the whole response is steered.

Per item (one problem × one sample); in the results JSON as per-item arrays, and with the text in the trace:
  correct            the final answer (``\\boxed{}`` first, else the last number; MATH-500: normalised LaTeX, or
                     ``math_verify`` when installed) equals the gold; a truncated row counts as wrong
  n_reason           reasoning tokens (before ``</think>``; the whole response without a span); n_gen; truncated
  forced-answer probe every ``--probe-every`` reasoning tokens c ∈ {E, 2E, …} ∪ {n_reason}: the prefix
                     prompt + response[:c] runs WITHOUT steering, followed by ``</think>\\n\\nThe final answer is
                     \\boxed{`` and ``--probe-tokens`` greedy tokens; the answer up to the matching brace is scored.
    ttc              tokens until correct: the first checkpoint c_i with ok_j = 1 for every j ≥ i (censored, None,
                     when the last checkpoint is wrong); the arm reports the Kaplan–Meier median and the restricted
                     mean (up to the longest span) with the censoring
    acc@b            accuracy at budget b: the natural answer if the span closed by b, else the forced answer at b;
                     defined only on probed rows (samples < ``--probe-samples``; None elsewhere, so the mean is not
                     restricted to early closers), ``acc_b_n`` rows per budget
    flips            changes between consecutive parsed forced answers
  backtracking       "wait" / "alternatively" / "hmm" per 1,000 span tokens (plus "actually", "let me check",
                     "double-check" counts in the trace)
  repetition         ``judge.repetition_onset`` of the response ids; distinct-4; self-loop rate (share of
                     sentences whose word 4-gram Jaccard with an earlier sentence exceeds 0.8)
  agreement          across samples of one problem: the share of samples giving the modal answer
  wall clock         batch time, its share by tokens, and that share net of the side channel's measured overhead
                     (bare against attached greedy decoding of ``--calib-tokens`` on the arm's first batch)
  gain               mean / SD / share at g_min, g_max of the raw gain over steered positions
  internal           first reasoning position at which the gold answer's first token is top-1 under the logit lens
                     at each lens depth and at the output
Per arm against ``none`` on the same (problem, sample): wrong→right, right→wrong and paired bootstrap intervals
(clusters = problems) for accuracy, log reasoning tokens, accuracy at each budget, TTC where both arms are
uncensored, flips, backtracking, distinct-4, self-loop, repetition.

Per token (``common.TraceWriter``, fp16, outside the repository; measurement policy: every registered signal in
every arm). Over prompt + response (columns of real tokens):
  the controller trace  (g, g_raw, g_law, s, p, E, r, release, dose, injection ratio, cos(d̂, h), cos(d̂_t, d̂_{t−k}),
                        ‖h‖, the repetition signals, massive flag, every ``src/<source>/<key>`` and every
                        ``tap/<tap>/<key>``: norms at every block, the attention statistics, the GDN gates and
                        mismatch per layer); NaN for the last generated token (never fed back) and for forced tokens
  ``out/*``             teacher-forced, steered model (a second pass with the same controller and steer schedule,
                        the gain checked against generation as ``tf_gain_dev``): log p of the next token, entropy,
                        rank, top-1; the unsteered model's log p, entropy and KL(steered ‖ unsteered) (T = 1
                        distributions, not the sampling ones)
  ``lens/*``            logit-lens rank of the next token at ``--lens-layers`` (4 depths) and of the gold answer's
                        first token at those depths and at the output
  ``bk/*``              ids, position, prompt / reasoning / answer / forced flags, response index, paragraph
                        boundary ("\\n\\n"), ordinal event rank (boundaries before), tokens since the last boundary,
                        backtracking-marker flag, probe checkpoints and their outcome
Per-head GDN signals (``--per-head-keys``) on the first ``--per-head-batches`` batches of sample 0, under
``<arm>/perhead``. Each arm also saves ``steered_r.npz`` (r at steered positions per row), the donor of the
yoked and position-mean controls.

Arms (``--arms``, comma separated; ``--preset stage0|stage1`` gives the protocol lists; ``{g}`` = ``--g-max``):
  none                          the controller attached at g = 0 (byte-identical to the base; signals traced)
  nothink                       thinking switched off in the template, unsteered (models with the switch)
  fixed:<dir>@<g>               g_t = g on the span; dir ∈ gist, anti_gist, mean, self, foreign, random<k>
                                (seeded unit vector k), window_gist (needs ``--window``)
  law:<source>:<polarity>[:<dir>][@<g_max>]   g = g_min + (g_max − g_min) r_t from a relevance source
                                (agree, attn_window, attn_agree, gdn_write, gdn_decay, gdn_mismatch,
                                gdn_mismatch_lag, gdn_familiar, gdn_recall, constant), polarity reinforce / restore
  matched:<arm>                 fixed gain at <arm>'s realised mean raw gain, <arm>'s direction
  yoked:<arm>                   <arm>'s r trajectory replayed from another problem (same sample index; problem
                                j reads problem j + 1), <arm>'s range and direction
  posmean:<arm>                 r_t = mean of <arm>'s r at steered step t across the OTHER problems (leave-one-
                                problem-out, every sample of the row's own problem excluded, so the control never
                                replays the row's own content-driven trajectory); past the last step that another
                                problem reaches, the last value is held. N_k (traces at step k) is stored as
                                ``setup.posmean_n``
  trunc:<arm> | trunc@<B>       unsteered, thinking cut at B tokens (or <arm>'s mean reasoning length): the
                                span is closed with "\\n</think>\\n\\n" and the answer generated
Dependent arms read the referenced arm from ``--out`` and its ``steered_r.npz`` from the trace directory.

The results JSON (``--out``) holds args, git, library versions, model facts, data spec and ids, and per arm the
specification, aggregates, the gain distribution, the comparison with ``none``, per-item arrays (no text) and
pointers + SHA-256 to the trace files. It is rewritten after every arm; arms already complete are skipped
unless ``--force`` (resumable). Generated text goes only to ``--trace-dir``, which must lie outside the repo.

Usage
  # offline smoke test (tiny random hybrid, CPU)
  CUDA_VISIBLE_DEVICES="" python experiments/ca2/reason.py --tiny --preset stage1 \\
      --out /tmp/x/tiny.json --trace-dir /tmp/x/traces
  # Stage 0c reasoning on the hybrid
  CUDA_VISIBLE_DEVICES=1 python experiments/ca2/reason.py --model Qwen/Qwen3.5-2B --data gsm8k:train:400 \\
      --preset stage0 --g-max 0.2 --name s0c_q35 2>&1 | tee results/evolve/round12/stage0/s0c_q35.log
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
sys.path.insert(0, str(EXP))
import pretrained_loop as pl  # noqa: E402
from ca2 import attention as ca2_attention  # noqa: E402  (registers the attn_* sources)
from ca2 import gdn as ca2_gdn  # noqa: E402  (registers the gdn_* sources)
from ca2.common import TraceWriter, alignment, gain_stats, git_state, paired_boot  # noqa: E402
from ca2.controller import DIRECTIONS, POLARITY, CA2Attachment, CA2Controller, NormTap  # noqa: E402
from ca2.sources import SOURCES, PositionMeanSource, YokedSource  # noqa: E402

BUDGETS = (256, 512, 1024, 2048)
CUE_ANSWER = "\n\nThe final answer is \\boxed{"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
TINY_THINK_END = 257                                     # byte tokenizer: a non-byte id stands for </think>
BACKTRACK_CORE = ("wait", "alternatively", "hmm")
BACKTRACK_RE = {"wait": r"\bwait\b", "alternatively": r"\balternatively\b", "hmm": r"\bhmm+\b",
                "actually": r"\bactually\b", "let_me_check": r"\blet me (?:re-?)?(?:check|verify|recheck)\b",
                "double_check": r"\bdouble[- ]check"}
MARKER_TOKENS = {"wait", "hmm", "alternatively", "actually"}
LAW_SOURCES = ("agree", "attn_window", "attn_agree", "gdn_write", "gdn_decay", "gdn_mismatch", "gdn_mismatch_lag",
               "gdn_familiar", "gdn_recall", "constant")
KINDS = ("none", "nothink", "fixed", "law", "matched", "yoked", "posmean", "trunc")

# The notebook's 20 problems (docs/notebooks/activation_additions_qwen3.ipynb, COT_PROBLEMS).
NOTEBOOK20 = [
    ("A baker makes 12 trays of 18 cookies and sells 150 cookies. How many cookies are left?", 66),
    ("Tom has 3 times as many marbles as Ann. Together they have 48. How many marbles does Tom have?", 36),
    ("A train travels 240 km in 3 hours, then 150 km in 2 hours. What is its average speed in km/h?", 78),
    ("A shirt costs $40. It is discounted 25%, then a 10% tax is added. What is the final price in dollars?", 33),
    ("Sara reads 15 pages a day for 6 days and 22 pages on the seventh. How many pages does she read?", 112),
    ("A rectangle has perimeter 50 and length 15. What is its area?", 150),
    ("There are 7 rows of 13 chairs. 19 chairs are removed. How many remain?", 72),
    ("A tank holds 120 liters and leaks 8 liters per hour. After how many hours is it a quarter full?", 11.25),
    ("Mia earns $14 an hour and works 37 hours. She spends $210. How much does she keep?", 308),
    ("If 5 pens cost $8.50, how much do 12 pens cost?", 20.4),
    ("A class has 32 students; 3/8 are boys. How many girls are there?", 20),
    ("Jake is 4 years older than twice Lily's age. Lily is 9. How old is Jake?", 22),
    ("A recipe needs 3 eggs per 2 cakes. How many eggs for 14 cakes?", 21),
    ("A car uses 6 liters per 100 km. How many liters for 350 km?", 21),
    ("Two numbers sum to 91 and differ by 17. What is the larger number?", 54),
    ("A farmer has 45 sheep and 3 times as many chickens. How many legs are there in total?", 450),
    ("A store sells 3 apples for $2. How much do 27 apples cost?", 18),
    ("A movie starts at 7:45 pm and lasts 2 hours 50 minutes. At how many minutes past 10 pm does it end?", 35),
    ("What is 17% of 600, plus 23?", 125),
    ("A worker packs 48 boxes in 6 hours. How many boxes in 15 hours at the same rate?", 120),
]


# ------------------------------------------------------------------------------------------------- data
@dataclass
class Item:
    uid: str                    # dataset and index, e.g. "gsm8k/train/123"
    question: str
    gold: str                   # gold answer as a string (numbers formatted plainly)
    kind: str                   # "numeric" or "math"


def _fmt_num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def load_items(spec: str, data_seed: int = 0, limit: int | None = None) -> list[Item]:
    """Problems by spec: ``gsm8k:<split>[:N]``, ``math500[:N]``, ``notebook20``, ``synthetic[:N]``.

    N draws a seeded subset (``data_seed``), kept in dataset order."""
    parts = spec.split(":")
    name = parts[0]
    pick = lambda n_all, n: sorted(random.Random(data_seed).sample(range(n_all), n)) if n and n < n_all \
        else list(range(n_all))
    if name == "gsm8k":
        split = parts[1] if len(parts) > 1 else "test"
        n = int(parts[2]) if len(parts) > 2 else None
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=split)
        items = [Item(f"gsm8k/{split}/{i}", ds[i]["question"],
                      _fmt_num(float(ds[i]["answer"].split("####")[-1].strip().replace(",", ""))), "numeric")
                 for i in pick(len(ds), n)]
    elif name == "math500":
        n = int(parts[1]) if len(parts) > 1 else None
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [Item(f"math500/{i}", ds[i]["problem"], str(ds[i]["answer"]), "math") for i in pick(len(ds), n)]
    elif name == "notebook20":
        items = [Item(f"notebook20/{i}", q, _fmt_num(a), "numeric") for i, (q, a) in enumerate(NOTEBOOK20)]
    elif name == "synthetic":
        n = int(parts[1]) if len(parts) > 1 else 8
        rng = random.Random(data_seed)
        items = []
        for i in range(n):
            a, b = rng.randrange(10, 99), rng.randrange(10, 99)
            items.append(Item(f"synthetic/{i}", f"What is {a} + {b}?", str(a + b), "numeric"))
    else:
        raise ValueError(f"unknown data spec {spec!r}: gsm8k:<split>[:N], math500[:N], notebook20, synthetic[:N]")
    return items[:limit] if limit else items


# ---------------------------------------------------------------------------------------------- scoring
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text: str) -> float | None:
    """The notebook's ``last_number``: the last number in the text (commas removed)."""
    m = NUM_RE.findall(text)
    try:
        return float(m[-1].replace(",", "")) if m else None
    except ValueError:
        return None


def brace_body(text: str, start: int) -> str | None:
    """The text from ``start`` (just after an opening brace) to its matching closing brace, or None."""
    depth = 1
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` (or ``\\fbox{...}``), braces matched."""
    best = None
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        body = brace_body(text, m.end())
        if body is not None:
            best = body
    return best


def extract_answer(text: str) -> str | None:
    """``\\boxed{}`` first, then the last number."""
    b = boxed(text)
    if b is not None:
        return b.strip()
    x = last_number(text)
    return None if x is None else _fmt_num(x)


def _unwrap_text(s: str) -> str:
    return re.sub(r"\\(?:text|textbf|mathrm|mbox)\s*\{([^{}]*)\}", r"\1", s)


def norm_tex(s: str) -> str:
    """A light LaTeX normalisation for exact-match scoring (used when ``math_verify`` is absent)."""
    s = _unwrap_text(s.strip())
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\:", ""), ("$", ""),
                 ("\\displaystyle", ""), ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""),
                 ("\\%", ""), ("%", ""), ("\\$", ""), (" ", ""), ("\n", "")):
        s = s.replace(a, b)
    s = re.sub(r"^[a-zA-Z]\s*=", "", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    return s.rstrip(".")


def to_float(s: str | None) -> float | None:
    """A number from an answer string: plain, with commas or units text, or a simple \\frac{a}{b} / a/b."""
    if s is None:
        return None
    t = norm_tex(s).replace(",", "")
    m = re.fullmatch(r"(-?)\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", t)
    try:
        if m:
            v = float(m.group(2)) / float(m.group(3))
            return -v if m.group(1) else v
        m = re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", t)
        if m:
            return float(m.group(1)) / float(m.group(2))
        return float(t)
    except (ValueError, ZeroDivisionError):
        x = last_number(t) if re.search(r"\d", t) and not re.search(r"[a-zA-Z\\]", t) else None
        return x


def _math_verify():
    try:
        import math_verify  # noqa: F401
        return math_verify
    except ImportError:
        return None


def is_correct(ans: str | None, gold: str, kind: str) -> bool:
    """Numeric: |a − g| ≤ 1e-6·max(1, |g|). Math: math_verify if installed, else normalised LaTeX or numbers."""
    if ans is None:
        return False
    g = to_float(gold)
    a = to_float(ans)
    if a is not None and g is not None and abs(a - g) <= 1e-6 * max(1.0, abs(g)):
        return True
    if kind == "numeric":
        return False
    mv = _math_verify()
    if mv is not None:
        try:
            return bool(mv.verify(mv.parse(f"${gold}$"), mv.parse(f"${ans}$")))
        except Exception:                                   # noqa: BLE001  (parser failures count as wrong)
            pass
    return norm_tex(ans) == norm_tex(gold)


def forced_answer(text: str) -> str | None:
    """The forced probe's answer: the text up to the brace closing the cue's ``\\boxed{``."""
    body = brace_body(text, 0)
    s = (body if body is not None else text).strip()
    return s or None


# ------------------------------------------------------------------------------------------ text measures
def backtrack_counts(text: str) -> dict[str, int]:
    t = text.lower()
    return {k: len(re.findall(p, t)) for k, p in BACKTRACK_RE.items()}


def distinct_n(ids: Sequence[int], n: int = 4) -> float | None:
    ids = [int(i) for i in ids]
    if len(ids) < n:
        return None
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def self_loop_rate(text: str, n: int = 4, thr: float = 0.8) -> float | None:
    """Share of sentences (≥ n words) whose word n-gram Jaccard with some earlier sentence exceeds ``thr``."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    grams = []
    for s in sents:
        w = re.findall(r"\w+", s.lower())
        if len(w) >= n:
            grams.append({tuple(w[i:i + n]) for i in range(len(w) - n + 1)})
    if not grams:
        return None
    loops = 0
    for i in range(1, len(grams)):
        a = grams[i]
        if any(len(a & b) / len(a | b) > thr for b in grams[:i]):
            loops += 1
    return loops / len(grams)


# ------------------------------------------------------------------------------------------ probe reading
def checkpoints(n_span: int, every: int) -> list[int]:
    """Probe points in span tokens: every, 2·every, … below n_span, then n_span itself (if > 0)."""
    if every <= 0 or n_span <= 0:
        return []
    cks = list(range(every, n_span, every))
    return cks + [n_span]


def ttc_from(cks: Sequence[int], oks: Sequence[bool]) -> tuple[int | None, bool]:
    """(tokens until correct, censored): the first checkpoint from which every forced answer is correct."""
    if not cks or not oks[-1]:
        return None, True
    i = len(oks) - 1
    while i > 0 and oks[i - 1]:
        i -= 1
    return int(cks[i]), False


def count_flips(answers: Sequence[str | None]) -> int:
    """Changes between consecutive parsed forced answers (unparsed ones skipped); numbers compared as numbers."""
    seq = []
    for a in answers:
        if a is None:
            continue
        x = to_float(a)
        seq.append(("n", round(x, 9)) if x is not None else ("s", norm_tex(a)))
    return sum(1 for u, v in zip(seq, seq[1:]) if u != v)


def budget_acc(budgets: Sequence[int], cks: Sequence[int], oks: Sequence[bool], n_span: int, closed: bool,
               final_ok: bool) -> dict[int, float | None]:
    """Accuracy at budget b: the natural answer when the span closed within b tokens, else the forced answer
    at checkpoint b (None when b was not probed)."""
    out = {}
    at = dict(zip(cks, oks))
    for b in budgets:
        if closed and n_span <= b:
            out[b] = float(final_ok)
        elif b in at:
            out[b] = float(at[b])
        elif not closed and n_span <= b and n_span in at:      # truncated (or ended in the span) before b
            out[b] = float(at[n_span])
        else:
            out[b] = None
    return out


def kaplan_meier(times: Sequence[float], events: Sequence[bool], tmax: float | None = None) -> dict:
    """Kaplan–Meier survival of 'not yet correct': median and restricted mean up to ``tmax``."""
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(events, dtype=bool)
    ok = np.isfinite(t)
    t, e = t[ok], e[ok]
    if t.size == 0:
        return {"n": 0, "events": 0, "median": None, "rmst": None, "tmax": tmax}
    tmax = float(t.max()) if tmax is None else float(tmax)
    S, prev, area, median = 1.0, 0.0, 0.0, None
    for u in np.unique(t[e]):
        if u > tmax:
            break
        area += S * (u - prev)
        at_risk = (t >= u).sum()
        d = ((t == u) & e).sum()
        S *= 1.0 - d / at_risk
        prev = u
        if median is None and S <= 0.5:
            median = float(u)
    area += S * (tmax - prev)
    return {"n": int(t.size), "events": int(e.sum()), "median": median, "rmst": float(area), "tmax": tmax,
            "surv_end": float(S)}


# ------------------------------------------------------------------------------------------------- arms
def safe_key(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", spec).strip("-")


@dataclass
class Arm:
    spec: str
    kind: str
    key: str = ""
    direction: str = "gist"
    seed: int | None = None
    gain: float | None = None
    source: str | None = None
    polarity: str = "reinforce"
    g_max: float | None = None
    ref: str | None = None
    budget: int | None = None

    def __post_init__(self) -> None:
        self.key = safe_key(self.spec)


def _direction(tok: str) -> tuple[str, int | None]:
    m = re.fullmatch(r"random(\d*)", tok)
    if m:
        return "random", (int(m.group(1)) if m.group(1) else None)
    if tok not in DIRECTIONS:
        raise ValueError(f"unknown direction {tok!r}; use {DIRECTIONS[:-1]} or random<k>")
    return tok, None


def parse_arm(spec: str) -> Arm:
    """One arm from its spec (module docstring, "Arms")."""
    s = spec.strip()
    if s in ("none", "nothink"):
        return Arm(s, s)
    if s.startswith("trunc@"):
        return Arm(s, "trunc", budget=int(s.split("@", 1)[1]))
    kind, _, rest = s.partition(":")
    if kind == "fixed":
        d, _, g = rest.partition("@")
        if not g:
            raise ValueError(f"{spec!r}: fixed arms need @<gain>")
        direction, seed = _direction(d)
        return Arm(s, "fixed", direction=direction, seed=seed, gain=float(g))
    if kind == "law":
        body, _, gmax = rest.partition("@")
        p = body.split(":")
        if len(p) not in (2, 3) or p[0] not in LAW_SOURCES or p[1] not in POLARITY:
            raise ValueError(f"{spec!r}: law:<source>:<polarity>[:<direction>][@g_max], source in {LAW_SOURCES}, "
                             f"polarity in {POLARITY}")
        direction, seed = _direction(p[2]) if len(p) == 3 else ("gist", None)
        return Arm(s, "law", direction=direction, seed=seed, source=p[0], polarity=p[1],
                   g_max=float(gmax) if gmax else None)
    if kind in ("matched", "yoked", "posmean", "trunc"):
        if not rest:
            raise ValueError(f"{spec!r}: {kind}:<arm spec>")
        parse_arm(rest)
        return Arm(s, kind, ref=rest)
    raise ValueError(f"unknown arm {spec!r}; kinds: {KINDS}")


def preset_arms(name: str, g: float, sources: Sequence[str], has_switch: bool) -> list[str]:
    """Protocol arm lists: ``stage0`` (design 0b(ii), plus self and foreign) and ``stage1`` (1c)."""
    f = lambda x: f"{x:g}"
    base = ["none"]
    if name == "stage0":
        base += [f"fixed:gist@{f(g / 2)}", f"fixed:gist@{f(g)}"]
        base += [f"fixed:{d}@{f(g)}" for d in ("anti_gist", "mean", "self", "random0", "random1", "random2",
                                                "foreign")]
        return base + (["nothink"] if has_switch else [])
    if name == "stage1":
        base += [f"fixed:gist@{f(g / 4)}", f"fixed:gist@{f(g / 2)}", f"fixed:gist@{f(g)}"]
        laws = [f"law:{s}:{p}" for s in sources for p in POLARITY]
        base += laws
        best = laws[0]
        base += [f"matched:{best}", f"yoked:{best}", f"posmean:{best}", f"law:{sources[0]}:reinforce:random0",
                 f"fixed:random0@{f(g)}", f"trunc:{best}"]
        return base + (["nothink"] if has_switch else [])
    raise ValueError(f"unknown preset {name!r}")


# ------------------------------------------------------------------------------------------- controller
class SpanController(CA2Controller):
    """``CA2Controller`` that also stops steering at the end of the reasoning span.

    The steer mask of ``CA2Controller`` is multiplied by open_t = 1 − [``span_end_id`` occurred at an input
    position ≤ t in this row] (so ``</think>`` itself and everything after it get g = 0), and the steered-step
    index is recounted from the gated mask. State: ``span_closed`` (B,) bool, carried across KV-cached calls.
    ``span_end_id=None`` leaves the controller unchanged.
    """

    def __init__(self, *args, span_end_id: int | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        self.span_end_id = None if span_end_id is None else int(span_end_id)

    def reset(self) -> None:
        super().reset()
        self.span_closed: torch.Tensor | None = None

    def state_numel(self) -> int:
        return super().state_numel() + (0 if self.span_closed is None else self.span_closed.numel())

    def _positions(self, B: int, T: int, dev) -> dict:
        if self._pos is not None:
            return self._pos
        pos = super()._positions(B, T, dev)
        if getattr(self, "span_end_id", None) is None:
            return pos
        ids = self._ids if self._ids is not None and tuple(self._ids.shape) == (B, T) else None
        hit = (ids.to(dev) == self.span_end_id) if ids is not None else torch.zeros(B, T, dtype=torch.bool,
                                                                                     device=dev)
        before = self.span_closed if self.span_closed is not None else torch.zeros(B, dtype=torch.bool, device=dev)
        closed = before.to(dev)[:, None] | (hit.long().cumsum(1) > 0)
        steer = pos["steer"] * (~closed).float()
        ns = self.n_steer if self.n_steer is not None else torch.zeros(B, device=dev)
        pos["steer"] = steer
        pos["step"] = (ns[:, None] + steer.cumsum(1) - steer).long()
        pos["span_open"] = (~closed).float()
        self.span_closed = closed[:, -1].detach()
        return pos


# -------------------------------------------------------------------------------------------- environment
@dataclass
class Env:
    model: torch.nn.Module
    tok: object
    blocks: torch.nn.ModuleList
    kinds: list
    dev: torch.device
    d_model: int
    L: int
    M: int
    R: int | None                     # attention-tap block (None: no attention tap on this model)
    hybrid: bool
    thinking: bool
    has_switch: bool
    chat: bool
    span_end: int | None
    stops: list[int]
    pad: int
    lens_layers: list[int]
    tiny: bool
    release_ids: list[int] = field(default_factory=list)
    facts: dict = field(default_factory=dict)


def _hf(tok):
    return getattr(tok, "tok", None)


def _special_id(tok, s: str) -> int | None:
    hf = _hf(tok)
    if hf is None:
        return None
    i = hf.convert_tokens_to_ids(s)
    if not isinstance(i, int) or i == hf.unk_token_id or i < 0:
        return None
    return i


def build_tiny(arch: str, seed: int):
    """Random tiny models with the byte tokenizer: gpt2 / qwen35 from ``pretrained_loop``, qwen3 as the tests."""
    import transformers
    if arch != "qwen3":
        return pl.build_tiny(arch, seed)
    torch.manual_seed(seed)
    cfg = transformers.Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                   max_position_embeddings=2048, bos_token_id=256, eos_token_id=256,
                                   pad_token_id=256)
    return transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa"), pl.ByteTokenizer()


def torch_gdn_kernels() -> bool:
    """Swap Qwen3.5's hub-dispatched GDN kernels (Triton under flash-linear-attention, which fails on CPU) for
    their pure-torch originals; returns whether a swap happened. For CPU runs only."""
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as M
    except ImportError:
        return False
    done = False
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        fn = getattr(M, name, None)
        orig = getattr(fn, "__wrapped__", None)
        if orig is not None:
            setattr(M, name, orig)
            done = True
    return done


def load_env(args) -> Env:
    if args.tiny:
        model, tok = build_tiny(args.tiny_arch, args.seed)
        dtype = pl.DTYPES[args.dtype]
        dev = "cuda" if args.device in ("cuda", "auto") and torch.cuda.is_available() else "cpu"
        model = model.to(dev, dtype).eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        ns = argparse.Namespace(model=args.model, tiny=False, tiny_arch="gpt2", seed=args.seed, dtype=args.dtype,
                                device=args.device, attn_impl=args.attn_impl)
        model, tok = pl.load_model(ns)
    dev = model.get_input_embeddings().weight.device
    if dev.type == "cpu" and getattr(pl.text_config(model), "layer_types", None):
        torch_gdn_kernels()
    blocks = pl.find_blocks(model)
    kinds = pl.layer_kinds(model, blocks)
    n = len(blocks)
    L = n // 2 if args.tap is None else args.tap % n
    M = L + 1 if args.inject_before is None else int(args.inject_before)
    R = None
    if not args.no_attn_tap:
        cand = [i for i in range(min(L, M - 1), -1, -1) if kinds[i] == "attention"] if args.attn_layer is None \
            else [int(args.attn_layer)]
        for i in cand:
            a = pl._child(blocks[i], pl.ATTN_MIXERS)
            if kinds[i] == "attention" and a is not None and hasattr(a, "q_norm") and hasattr(a, "k_norm") and i < M:
                R = i
                break
        if args.attn_layer is not None and R is None:
            raise SystemExit(f"--attn-layer {args.attn_layer}: not a full-attention block (Qwen layout) below {M}")
    hybrid = ca2_gdn.is_hybrid(model)
    name = (args.model or "").lower()
    base_model = "base" in name.split("/")[-1]
    hf = _hf(tok)
    template = getattr(hf, "chat_template", None) if hf is not None else None
    chat = bool(template) and not base_model and args.render != "plain"
    think_end = TINY_THINK_END if args.tiny else (_special_id(tok, "</think>") if chat else None)
    has_switch = (args.tiny or (chat and "enable_thinking" in str(template))) and think_end is not None
    thinking = {"on": True, "off": False, "auto": think_end is not None}[args.thinking]
    if thinking and think_end is None:
        raise SystemExit("--thinking on: this model has no </think> token (or no chat template)")
    eos = alignment().run.eos_ids(model, tok)
    stops = list(dict.fromkeys(eos + ([i for i in (_special_id(tok, "<|im_end|>"), _special_id(tok, "<|endoftext|>"))
                                       if i is not None] if chat else [])))
    pad = alignment().run.pad_id(tok)
    lens = sorted({int(x) % n for x in args.lens_layers.split(",")}) if args.lens_layers else \
        sorted({n // 4, n // 2, (3 * n) // 4, (7 * n) // 8})
    rel = []
    for w in [x for x in args.release_words.split(",") if x]:
        ids = tok.encode(w)
        if len(ids) == 1:
            rel.append(ids[0])
    env = Env(model, tok, blocks, kinds, dev, model.get_input_embeddings().weight.shape[1], L, M, R, hybrid,
              thinking, has_switch, chat, think_end if thinking else None, stops, pad, lens, args.tiny,
              sorted(set(rel)))
    env.facts = {"name": "tiny-" + args.tiny_arch if args.tiny else args.model, "n_blocks": n, "kinds": kinds,
                 "d_model": env.d_model, "vocab": int(model.get_output_embeddings().weight.shape[0]),
                 "tap": L, "inject_before": M, "attn_tap_layer": R, "hybrid": hybrid,
                 "gdn_layers": ca2_gdn.linear_layers(model, blocks) if hybrid else [],
                 "chat_template": bool(template), "chat": chat, "thinking_switch": has_switch, "thinking": thinking,
                 "think_end_id": think_end, "stop_ids": stops, "pad_id": pad, "lens_layers": lens,
                 "release_ids": env.release_ids, "dtype": args.dtype, "device": str(dev),
                 "n_params": int(sum(p.numel() for p in model.parameters()))}
    return env


def render(env: Env, question: str, thinking: bool, instruction: str) -> list[int]:
    """Prompt ids; ``enable_thinking`` always explicit; with thinking the prompt ends in ``<think>\\n``."""
    text = f"{question}\n{instruction}" if instruction else question
    tok = env.tok
    if env.tiny:
        ids = tok.encode(f"User: {text}\nAssistant: <think>\n")
        if not thinking:
            ids += [TINY_THINK_END] + tok.encode("\n\n")
        return ids
    if env.chat:
        s = _hf(tok).apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=thinking)
        if thinking and not s.rstrip().endswith("<think>"):
            s += "<think>\n"
        return tok.encode(s)
    return tok.encode(f"Question: {text}\nAnswer:")


# ------------------------------------------------------------------------------------------ controllers
@dataclass
class ArmSetup:
    """Resolved arm: everything needed to build a controller for a batch of rows."""
    arm: Arm
    kind: str
    direction: str
    seed: int
    g_min: float
    g_max: float
    source: str
    polarity: str
    smooth: str
    thinking: bool
    steered: bool
    budget: int | None = None
    donor: dict | None = None             # (uid, sample) -> r at steered steps (yoked)
    table: dict | None = None             # posmean: uid -> leave-one-problem-out table
    table_n: list | None = None           # posmean: N_k, traces reaching steered step k (all problems)


def controller(env: Env, st: ArmSetup, args, rows: Sequence[tuple[Item, int]], taps_for: str = "all",
               per_head: bool = False):
    """A fresh ``SpanController`` for ``rows`` with every tap the model supports (``taps_for='all'``) or only
    the taps the driving source needs (``'drive'``, the teacher-forced pass); returns (ctrl, taps)."""
    B = len(rows)
    if st.kind == "yoked":
        seqs = [st.donor.get((it.uid, k)) for it, k in rows]
        seqs = [np.asarray(s if s is not None and len(s) else [0.0], dtype=np.float32) for s in seqs]
        N = max(len(s) for s in seqs)
        donor = torch.zeros(B, N)
        for b, s in enumerate(seqs):
            donor[b, : len(s)] = torch.from_numpy(np.clip(s, 0.0, 1.0))
        src = YokedSource(donor, torch.tensor([len(s) for s in seqs]), index="step", name="yoked")
    elif st.kind == "posmean":
        tabs = [st.table[it.uid] for it, _ in rows]
        N = max(len(t) for t in tabs)
        tab = torch.zeros(B, N)
        for b, t in enumerate(tabs):                        # hold each row's last value (the source clamps at N − 1)
            tab[b] = torch.from_numpy(np.clip(np.pad(t, (0, N - len(t)), mode="edge"), 0.0, 1.0))
        src = PositionMeanSource(tab, index="step", name="position_mean")
    else:
        src = st.source
    g_abs = max(args.g_abs, st.g_max)
    ctrl = SpanController(env.d_model, source=src, g_min=st.g_min, g_max=st.g_max, g_abs=g_abs, smooth=st.smooth,
                          leak=args.leak, kappa=args.kappa, polarity=st.polarity, restore_kappa=args.restore_kappa,
                          restore_target=args.restore_target, release_rho=args.release_rho,
                          release_ids=env.release_ids, direction=st.direction,
                          betas=tuple(float(x) for x in args.betas.split(",")), sink_skip=args.sink_skip,
                          regulator=args.regulator, window=args.window, seed=st.seed,
                          span_end_id=env.span_end if st.thinking else None, sources=("agree",))
    drv = ctrl.drive
    taps: list = []
    want_attn = env.R is not None and (taps_for == "all" or drv.startswith("attn"))
    want_gdn = env.hybrid and (taps_for == "all" or drv.startswith("gdn"))
    if taps_for == "all":
        taps.append(NormTap())
    if want_attn:
        taps.append(ca2_attention.add_attention(ctrl, layer=env.R, window=args.attn_window,
                                                sources=("attn_window", "attn_agree") if taps_for == "all"
                                                else ()))
    if want_gdn:
        mism = args.gdn_mismatch if taps_for == "all" else _gdn_signal(drv) not in ca2_gdn.GATE_KEYS
        keys = [k for k in args.per_head_keys.split(",") if k] if per_head else False
        taps.append(ca2_gdn.GDNTaps(mismatch=mism, per_head=keys))
        if taps_for == "all":
            ca2_gdn.add_gdn_sources(ctrl, names=[n for n in ca2_gdn.GDN_SOURCES
                                                 if mism or _gdn_signal(n) in ca2_gdn.GATE_KEYS])
    return ctrl, taps


def _gdn_signal(name: str) -> str:
    p = ca2_gdn.GDN_SOURCES[name]
    return p.keywords.get("signal", p.args[0] if p.args else "fam")


def attach(env: Env, ctrl, taps) -> CA2Attachment:
    return CA2Attachment(env.model, ctrl, tap=env.L, inject_before=env.M,
                         relevance_layer=env.R if env.R is not None else None, taps=taps)


# ------------------------------------------------------------------------------------------- generation
def _sync(dev) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _gen_kwargs(env: Env, args, n_new: int, greedy: bool = False) -> dict:
    kw = dict(max_new_tokens=n_new, pad_token_id=env.pad, eos_token_id=env.stops)
    if greedy or args.temperature <= 0:
        kw.update(do_sample=False)
    else:
        kw.update(do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    return kw


def pad_left(seqs: Sequence[Sequence[int]], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    return alignment().data.pad_left([list(s) for s in seqs], pad)


@torch.no_grad()
def generate_plain(env: Env, seqs: Sequence[Sequence[int]], n_new: int, args, greedy: bool) -> list[list[int]]:
    """Unsteered generation (no hooks) from left-padded prefixes; returns the new ids per row (pads kept)."""
    ids, mask = pad_left(seqs, env.pad)
    out = env.model.generate(input_ids=ids.to(env.dev), attention_mask=mask.to(env.dev),
                             **_gen_kwargs(env, args, n_new, greedy))
    return out[:, ids.shape[1]:].cpu().tolist()


def cut_stop(ids: Sequence[int], stops: Sequence[int]) -> tuple[list[int], bool]:
    """Ids up to and including the first stop id; whether a stop was hit."""
    s = set(stops)
    for j, t in enumerate(ids):
        if t in s:
            return list(ids[: j + 1]), True
    return list(ids), False


def probe_rows(env: Env, jobs: list[tuple[list[int], str, str]], args) -> list[tuple[str | None, bool]]:
    """Forced-answer probe, unsteered and greedy: jobs are (prefix ids incl. the cue, gold, kind)."""
    out: list[tuple[str | None, bool]] = [None] * len(jobs)
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i][0]))
    for i in range(0, len(order), args.probe_batch):
        idx = order[i:i + args.probe_batch]
        new = generate_plain(env, [jobs[j][0] for j in idx], args.probe_tokens, args, greedy=True)
        for j, row in zip(idx, new):
            ids, _ = cut_stop(row, env.stops)
            ids = [t for t in ids if t not in env.stops]
            a = forced_answer(env.tok.decode(ids))
            out[j] = (a, is_correct(a, jobs[j][1], jobs[j][2]))
    return out


# --------------------------------------------------------------------------------------- teacher forcing
def _final_norm(model):
    base = getattr(model, "base_model", model)
    for n in ("norm", "ln_f", "final_layernorm"):
        m = getattr(base, n, None)
        if isinstance(m, torch.nn.Module):
            return m
    raise RuntimeError("no final norm found for the logit lens")


@torch.no_grad()
def tf_pass(env: Env, ids: torch.Tensor, mask: torch.Tensor, lens: bool, ctrl=None, taps=(),
            steer_from: torch.Tensor | None = None) -> dict:
    """One teacher-forced pass; returns the final-norm hidden states (and block outputs at the lens depths),
    plus the controller trace when a controller is given."""
    model = env.model
    norm = _final_norm(model)
    cap: dict = {}
    att = attach(env, ctrl, list(taps)) if ctrl is not None else None   # first: lens hooks then see the injection
    hs = [norm.register_forward_hook(lambda m, a, o: cap.__setitem__("final", o))]
    if lens:
        for d in env.lens_layers:
            hs.append(env.blocks[d].register_forward_hook(
                lambda m, a, o, d=d: cap.__setitem__(d, o[0] if isinstance(o, tuple) else o)))
    try:
        if ctrl is not None:
            ctrl.reset()
            ctrl.record = True
            ctrl.pending_mask = mask
            ctrl.steer_from = steer_from
        pos = alignment().probe.position_ids(mask)
        model(input_ids=ids.to(env.dev), attention_mask=mask.to(env.dev), position_ids=pos.to(env.dev),
              use_cache=False, logits_to_keep=1)
    finally:
        for h in hs:
            h.remove()
        if att is not None:
            att.remove()
    if ctrl is not None:
        cap["trace"] = ctrl.trace_tensors()
    return cap


@torch.no_grad()
def token_stats(env: Env, ids: torch.Tensor, mask: torch.Tensor, steered: dict, base: dict | None,
                gold_first: Sequence[int | None], chunk: int) -> list[dict[str, np.ndarray]]:
    """Per row, over real columns: next-token log p / entropy / rank / top-1 of the steered pass, the base pass's
    log p / entropy and KL(steered ‖ base), logit-lens ranks of the next token and of the gold's first token."""
    head = env.model.get_output_embeddings()
    norm = _final_norm(env.model)
    out = []
    B = ids.shape[0]
    for b in range(B):
        cols = torch.nonzero(mask[b] > 0).flatten()
        n = cols.numel()
        tgt = torch.full((n,), -1, dtype=torch.long)
        tgt[:-1] = ids[b, cols[1:]]
        res = {k: np.full(n, np.nan, dtype=np.float32) for k in
               ("out/logp_next", "out/ent", "out/rank_next", "out/top1_next", "out/logp_next_base", "out/ent_base",
                "out/kl", "lens/gold_rank_out")}
        nl = len(env.lens_layers) if any(d in steered for d in env.lens_layers) else 0
        res["lens/rank_next"] = np.full((n, nl), np.nan, dtype=np.float32)
        res["lens/gold_rank"] = np.full((n, nl), np.nan, dtype=np.float32)
        gf = gold_first[b]
        for a in range(0, n, chunk):
            c = cols[a:a + chunk].to(env.dev)
            t = tgt[a:a + chunk].to(env.dev)
            has = t >= 0
            tt = t.clamp_min(0)
            ls = head(steered["final"][b, c]).float()
            lp = F.log_softmax(ls, -1)
            p = lp.exp()
            sl = slice(a, a + c.numel())
            res["out/ent"][sl] = (-(p * lp).sum(-1)).cpu().numpy()
            lt = ls.gather(1, tt[:, None])
            rank = (ls > lt).sum(-1).float()
            res["out/logp_next"][sl] = torch.where(has, lp.gather(1, tt[:, None])[:, 0], torch.nan).cpu().numpy()
            res["out/rank_next"][sl] = torch.where(has, rank, torch.nan).cpu().numpy()
            res["out/top1_next"][sl] = torch.where(has, (rank == 0).float(), torch.nan).cpu().numpy()
            if gf is not None:
                res["lens/gold_rank_out"][sl] = (ls > ls[:, gf:gf + 1]).sum(-1).float().cpu().numpy()
            if base is not None:
                lb = F.log_softmax(head(base["final"][b, c]).float(), -1)
                res["out/logp_next_base"][sl] = torch.where(has, lb.gather(1, tt[:, None])[:, 0],
                                                            torch.nan).cpu().numpy()
                res["out/ent_base"][sl] = (-(lb.exp() * lb).sum(-1)).cpu().numpy()
                res["out/kl"][sl] = (p * (lp - lb)).sum(-1).clamp_min(0.0).cpu().numpy()
            else:
                res["out/logp_next_base"][sl] = res["out/logp_next"][sl]
                res["out/ent_base"][sl] = res["out/ent"][sl]
                res["out/kl"][sl] = 0.0
            del ls, lp, p
            for j, d in enumerate(env.lens_layers[:nl]):
                ld = head(norm(steered[d][b, c])).float()
                lt = ld.gather(1, tt[:, None])
                res["lens/rank_next"][sl, j] = torch.where(has, (ld > lt).sum(-1).float(), torch.nan).cpu().numpy()
                if gf is not None:
                    res["lens/gold_rank"][sl, j] = (ld > ld[:, gf:gf + 1]).sum(-1).float().cpu().numpy()
                del ld
        out.append(res)
    return out


# ------------------------------------------------------------------------------------------ trace rows
def take_cols(tt: dict[str, torch.Tensor], b: int, cols: Sequence[int], total: int) -> dict[str, np.ndarray]:
    """Row b of a controller trace at ``cols`` (model columns), padded to ``total`` with NaN / −1."""
    idx = torch.as_tensor(list(cols), dtype=torch.long)
    out = {}
    for k, v in tt.items():
        r = v[b].index_select(0, idx) if idx.numel() else v[b][:0]
        if r.shape[0] < total:
            fill = float("nan") if r.is_floating_point() else -1
            r = torch.cat([r, torch.full((total - r.shape[0],) + tuple(r.shape[1:]), fill, dtype=r.dtype)], 0)
        out[k] = r.numpy() if r.dtype != torch.bfloat16 else r.float().numpy()
    return out


class TokText:
    """Cached single-token decoding for the bookkeeping flags."""

    def __init__(self, tok) -> None:
        self.tok, self.cache = tok, {}

    def __call__(self, i: int) -> str:
        s = self.cache.get(i)
        if s is None:
            s = self.cache[i] = self.tok.decode([int(i)])
        return s


def bookkeeping(seq: Sequence[int], n_prompt: int, n_span: int, span_closed_at: int | None, forced_from: int | None,
                cks: Sequence[int], oks: Sequence[bool], text_of: TokText) -> dict[str, np.ndarray]:
    """Per-token flags: positions, spans, boundaries ("\\n\\n"), ordinal event rank, time since boundary,
    backtracking-marker tokens, probe checkpoints and outcomes."""
    N = len(seq)
    j = np.arange(N) - n_prompt
    resp = j >= 0
    reason = resp & (j < n_span)
    answer = resp & (span_closed_at is not None) & (j > (span_closed_at if span_closed_at is not None else N))
    forced = resp & (forced_from is not None) & (j >= (forced_from if forced_from is not None else N))
    bnd = np.zeros(N, dtype=np.float32)
    nl = np.zeros(N, dtype=np.float32)
    mark = np.zeros(N, dtype=np.float32)
    for i, t in enumerate(seq):
        s = text_of(t)
        nl[i] = float("\n" in s)
        bnd[i] = float("\n\n" in s)
        mark[i] = float(re.sub(r"[^a-z]", "", s.lower()) in MARKER_TOKENS)
    rank = np.concatenate([[0], np.cumsum(bnd)[:-1]]).astype(np.float32)
    since = np.zeros(N, dtype=np.float32)
    last = -1
    for i in range(N):
        since[i] = i - last - 1 if last >= 0 else i
        if bnd[i]:
            last = i
    ck = np.zeros(N, dtype=np.float32)
    ok = np.full(N, np.nan, dtype=np.float32)
    for c, o in zip(cks, oks):
        p = n_prompt + c - 1                                # last span token before the probe's cue
        if 0 <= p < N:
            ck[p], ok[p] = 1.0, float(o)
    return {"bk/ids": np.asarray(seq, dtype=np.int64), "bk/pos": np.arange(N), "bk/prompt": (~resp).astype(np.float32),
            "bk/resp_idx": np.where(resp, j, -1), "bk/reason": reason.astype(np.float32),
            "bk/answer": answer.astype(np.float32), "bk/forced": forced.astype(np.float32), "bk/newline": nl,
            "bk/boundary": bnd, "bk/event_rank": rank, "bk/since_boundary": since, "bk/backtrack": mark,
            "bk/probe_ck": ck, "bk/probe_ok": ok}


# ------------------------------------------------------------------------------------------------ batch
def _num(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else x


def _first_top1(rank: np.ndarray, reason: np.ndarray) -> int | None:
    hit = np.nonzero((rank == 0) & (reason > 0))[0]
    return None if hit.size == 0 else int(hit[0])


def run_batch(env: Env, st: ArmSetup, rows: list[tuple[Item, int]], bi: int, args, writer: TraceWriter,
              per_head: bool, probe: bool, acc: dict, n_keep: int | None = None) -> list[dict]:
    """Generate one batch for one arm; probe, teacher-force, write the traces; returns per-row records of the
    first ``n_keep`` rows (the rest are filler rows, e.g. the foreign direction's second row)."""
    n_keep = len(rows) if n_keep is None else n_keep
    model, tok = env.model, env.tok
    prompts = [render(env, it.question, st.thinking, args.instruction) for it, _ in rows]
    ids, mask = pad_left(prompts, env.pad)
    P = ids.shape[1]
    ctrl, taps = controller(env, st, args, rows, "all", per_head)
    att = attach(env, ctrl, taps)
    n_new = args.max_new_tokens if st.kind != "trunc" else min(st.budget, args.max_new_tokens)
    seed = args.seed * 1_000_003 + rows[0][1] * 7919 + bi
    try:
        ctrl.reset()
        ctrl.record = True
        ctrl.pending_mask = mask
        torch.manual_seed(seed)
        _sync(env.dev)
        t0 = time.time()
        out = model.generate(input_ids=ids.to(env.dev), attention_mask=mask.to(env.dev),
                             **_gen_kwargs(env, args, n_new))
        _sync(env.dev)
        t_gen = time.time() - t0
        tt = ctrl.trace_tensors()
    finally:
        att.remove()
    new = out[:, P:].cpu()
    n_steps = new.shape[1]
    resp1 = [cut_stop(new[b].tolist(), env.stops) for b in range(len(rows))]
    # budget truncation: close the span and generate the answer, unsteered
    forced_from: list[int | None] = [None] * len(rows)
    resp = [r for r, _ in resp1]
    stopped = [h for _, h in resp1]
    t_cont = 0.0
    if st.kind == "trunc":
        todo = [b for b in range(len(rows)) if not stopped[b]]
        if todo:
            close = tok.encode("\n") + [env.span_end] + tok.encode("\n\n") if env.span_end is not None \
                else tok.encode("\n\n")
            pref, adds = [], {}
            for b in todo:
                is_open = env.span_end is None or env.span_end not in resp[b]
                adds[b] = close if is_open else []
                pref.append(prompts[b] + resp[b] + adds[b])
            torch.manual_seed(seed + 1)
            t1 = time.time()
            cont = generate_plain(env, pref, args.answer_tokens, args, greedy=False)
            t_cont = time.time() - t1
            for b, c in zip(todo, cont):
                c, hit = cut_stop(c, env.stops)
                forced_from[b] = len(resp[b])
                resp[b] = resp[b] + adds[b] + c
                stopped[b] = hit
    # per-row reading
    recs = []
    probe_jobs: list[tuple[list[int], str, str]] = []
    job_of: list[list[int]] = []
    span_cue = ([env.span_end] if env.span_end is not None and st.thinking else []) + tok.encode(CUE_ANSWER)
    for b, (it, k) in enumerate(rows):
        r = resp[b]
        close_at = r.index(env.span_end) if (st.thinking and env.span_end is not None and env.span_end in r) else None
        if close_at is not None:
            n_span = close_at
        else:
            n_span = len(r) - (1 if stopped[b] else 0)
        if forced_from[b] is not None:
            n_span = min(n_span, forced_from[b]) if close_at is None or close_at > forced_from[b] else n_span
        body = [t for t in (r[close_at + 1:] if close_at is not None else r) if t not in env.stops]
        ans_text = tok.decode(body)
        ans = extract_answer(ans_text)
        truncated = not stopped[b]
        ok = is_correct(ans, it.gold, it.kind) and not truncated
        probed_row = bool(probe and k < args.probe_samples)
        cks = checkpoints(n_span, args.probe_every) if probed_row else []
        job_of.append(list(range(len(probe_jobs), len(probe_jobs) + len(cks))))
        for c in cks:
            probe_jobs.append((prompts[b] + r[:c] + span_cue, it.gold, it.kind))
        span_ids = r[:n_span]
        span_text = tok.decode([t for t in span_ids if t not in env.stops])
        bt = backtrack_counts(span_text)
        resp_ids = [t for t in r if t not in env.stops]
        onset = alignment().judge.repetition_onset(resp_ids)
        recs.append(dict(uid=it.uid, sample=k, gold=it.gold, answer=ans, correct=bool(ok),
                         raw_correct=bool(is_correct(ans, it.gold, it.kind)), truncated=truncated,
                         closed=close_at is not None, n_reason=int(n_span), n_gen=len(r), n_prompt=len(prompts[b]),
                         forced_from=forced_from[b], cks=cks, probed=probed_row, backtrack=bt,
                         backtrack_rate=1000.0 * sum(bt[x] for x in BACKTRACK_CORE) / max(1, n_span),
                         rep_onset=onset, distinct4=distinct_n(resp_ids, 4), self_loop=self_loop_rate(span_text),
                         text=tok.decode(resp_ids), seed=seed, batch=bi, row=b))
    t2 = time.time()
    probed = probe_rows(env, probe_jobs, args) if probe_jobs else []
    t_probe = time.time() - t2
    for rec, js in zip(recs, job_of):
        res = [probed[j] for j in js]
        rec["probe_answers"] = [a for a, _ in res]
        rec["probe_ok"] = [bool(o) for _, o in res]
        if js:
            rec["ttc"], rec["censored"] = ttc_from(rec["cks"], rec["probe_ok"])
            rec["flips"] = count_flips(rec["probe_answers"])
        else:
            rec["ttc"], rec["censored"], rec["flips"] = None, None, None
        # acc@b on probed rows only: on an unprobed row it would exist only for early closers (outcome-selected)
        rec["acc_budget"] = (budget_acc(args.budget_list, rec["cks"], rec["probe_ok"], rec["n_reason"],
                                        rec["closed"] and not rec["truncated"], rec["correct"])
                             if rec["probed"] else {bb: None for bb in args.budget_list})
    # teacher-forced outcome measures
    seqs = [prompts[b] + resp[b] for b in range(len(rows))]
    stats = None
    gold_first = []
    for it, _ in rows:
        g = tok.encode(it.gold)
        gold_first.append(int(g[0]) if g and int(g[0]) < env.facts["vocab"] else None)
    t3 = time.time()
    tf_dev = [None] * len(rows)
    tg = None
    if args.tf:
        fids, fmask = pad_left(seqs, env.pad)
        if st.steered:
            c2, t2s = controller(env, st, args, rows, "drive")
            sf = torch.tensor([len(p) - 1 for p in prompts], dtype=torch.float32)
            S = tf_pass(env, fids, fmask, True, c2, t2s, sf)
            Bp = tf_pass(env, fids, fmask, False)
            tg = S.pop("trace")
        else:
            S, Bp = tf_pass(env, fids, fmask, True), None
        stats = token_stats(env, fids, fmask, S, Bp, gold_first, args.stat_chunk)
        del S, Bp
    t_tf = time.time() - t3
    # assemble and write
    text_of = TokText(tok)
    head_keys = [k for k in tt if k.endswith("_head")]
    for b, rec in enumerate(recs[:n_keep]):
        Np = len(prompts[b])
        cols = torch.nonzero(mask[b] > 0).flatten().tolist()
        n_fed = min(len(resp1[b][0]), n_steps - 1)
        cols += [P + j for j in range(n_fed)]
        total = Np + len(resp[b])
        main = {k: v for k, v in take_cols(tt, b, cols, total).items() if not k.endswith("_head")}
        bk = bookkeeping(seqs[b], Np, rec["n_reason"], (rec["n_reason"] if rec["closed"] else None),
                         rec["forced_from"], rec["cks"], rec["probe_ok"], text_of)
        toks = {**main, **bk}
        if stats is not None:
            toks.update(stats[b])
        # gain on steered response positions
        steer = np.nan_to_num(main["steer"], nan=0.0) > 0
        graw = main["g_raw"][steer].astype(np.float64)
        rvals = main["r"][steer].astype(np.float32)
        rec["gain_mean"] = float(graw.mean()) if graw.size else None
        rec["gain_sd"] = float(graw.std()) if graw.size > 1 else (0.0 if graw.size else None)
        rec["n_steered"] = int(graw.size)
        acc["g"].append(graw)
        acc["r"][(rec["uid"], rec["sample"])] = rvals
        if tg is not None and "g_raw" in tg:
            gt = take_cols(tg, b, torch.nonzero(fmask[b] > 0).flatten().tolist(), total)["g_raw"]
            m = steer & np.isfinite(gt) & np.isfinite(main["g_raw"])
            rec["tf_gain_dev"] = float(np.abs(gt[m] - main["g_raw"][m]).max()) if m.any() else 0.0
        if stats is not None:
            sb = stats[b]
            rs = bk["bk/reason"] > 0
            rsp = bk["bk/resp_idx"] >= 0
            mean = lambda x, m: _num(np.nanmean(x[m])) if m.any() and np.isfinite(x[m]).any() else None
            rec.update(kl_reason=mean(sb["out/kl"], rs), ent_reason=mean(sb["out/ent"], rs),
                       logp_reason=mean(sb["out/logp_next"], rs), logp_base_reason=mean(sb["out/logp_next_base"], rs),
                       kl_resp=mean(sb["out/kl"], rsp), top1_reason=mean(sb["out/top1_next"], rs))
            firsts = [_first_top1(sb["lens/gold_rank"][:, j], bk["bk/reason"]) for j in
                      range(sb["lens/gold_rank"].shape[1])]
            firsts.append(_first_top1(sb["lens/gold_rank_out"], bk["bk/reason"]))
            rec["gold_top1_first"] = [None if f is None else f - Np for f in firsts]
        share = (len(resp1[b][0]) / max(1, sum(len(x[0]) for x in resp1)))
        rec["wall_batch_s"] = t_gen + t_cont
        rec["wall_s"] = (t_gen + t_cont) * share
        rec["wall_net_s"] = rec["wall_s"] / acc.get("overhead_ratio", 1.0)
        rec["probe_s"], rec["tf_s"] = t_probe * share, t_tf * share
        writer.add("tokens", rec, toks)
        if per_head and head_keys:
            hk = take_cols({k: tt[k] for k in head_keys}, b, cols, total)
            writer.add("perhead", {"uid": rec["uid"], "sample": rec["sample"]}, hk)
    return recs[:n_keep]


@torch.no_grad()
def measure_overhead(env: Env, st: ArmSetup, rows: list[tuple[Item, int]], args) -> dict:
    """Seconds per greedy decode step without hooks and with this arm's controller and every tap."""
    if args.calib_tokens <= 0:
        return {"ratio": 1.0}
    prompts = [render(env, it.question, st.thinking, args.instruction) for it, _ in rows]
    ids, mask = pad_left(prompts, env.pad)

    def once(with_side: bool) -> float:
        att = None
        if with_side:
            ctrl, taps = controller(env, st, args, rows, "all")
            att = attach(env, ctrl, taps)
            ctrl.reset()
            ctrl.record = True
            ctrl.pending_mask = mask
        try:
            _sync(env.dev)
            t0 = time.time()
            env.model.generate(input_ids=ids.to(env.dev), attention_mask=mask.to(env.dev),
                               **{**_gen_kwargs(env, args, args.calib_tokens, greedy=True), "eos_token_id": None,
                                  "min_new_tokens": args.calib_tokens})
            _sync(env.dev)
            return time.time() - t0
        finally:
            if att is not None:
                att.remove()
    once(False)                                                    # warm-up
    bare, side = once(False), once(True)
    return {"bare_s": bare, "side_s": side, "ratio": side / bare if bare > 0 else 1.0, "tokens": args.calib_tokens,
            "batch": len(rows)}


# ---------------------------------------------------------------------------------------------- arm loop
def resolve(arm: Arm, report: dict, args, env: Env) -> ArmSetup:
    """Turn an arm into an ``ArmSetup``, reading referenced arms from the report and their r traces."""
    g_max = args.g_max if arm.g_max is None else arm.g_max
    seed = args.seed if arm.seed is None else arm.seed
    think = env.thinking
    if arm.kind == "none":
        return ArmSetup(arm, "none", "gist", seed, 0.0, 0.0, "constant", "reinforce", "none", think, False)
    if arm.kind == "nothink":
        if not env.has_switch:
            raise ValueError("nothink: this model has no thinking switch")
        return ArmSetup(arm, "nothink", "gist", seed, 0.0, 0.0, "constant", "reinforce", "none", False, False)
    if arm.kind == "fixed":
        return ArmSetup(arm, "fixed", arm.direction, seed, arm.gain, arm.gain, "constant", "reinforce", "none",
                        think, arm.gain > 0)
    if arm.kind == "law":
        return ArmSetup(arm, "law", arm.direction, seed, args.g_min, g_max, arm.source, arm.polarity, args.smooth,
                        think, g_max > 0)
    ref = report["arms"].get(arm.ref)
    if ref is None or not ref.get("complete"):
        raise ValueError(f"{arm.spec}: arm {arm.ref!r} must be complete first (run it earlier in --arms)")
    rs = ref["setup"]
    if arm.kind == "matched":
        g = ref["gain"].get("mean") or 0.0
        return ArmSetup(arm, "matched", rs["direction"], rs["seed"], g, g, "constant", "reinforce", "none", think,
                        g > 0)
    if arm.kind == "trunc":
        b = int(round(ref["agg"]["n_reason_mean"])) if arm.budget is None else arm.budget
        return ArmSetup(arm, "trunc", "gist", seed, 0.0, 0.0, "constant", "reinforce", "none", think, False,
                        budget=max(1, b))
    r = load_steered_r(Path(args.trace_dir), safe_key(arm.ref))
    base = ArmSetup(arm, arm.kind, rs["direction"], rs["seed"], rs["g_min"], rs["g_max"], arm.kind, "reinforce",
                    "none", think, rs["g_max"] > 0)
    if arm.kind == "yoked":
        keys = sorted({u for u, _ in r}, key=lambda u: (u.split("/")[:-1], int(u.split("/")[-1])))
        nxt = {u: keys[(i + 1) % len(keys)] for i, u in enumerate(keys)}
        base.donor = {(u, k): r.get((nxt[u], k), np.zeros(1, np.float32)) for (u, k) in r}
        return base
    if arm.kind == "posmean":
        base.table, n_k = posmean_tables(r)
        base.table_n = n_k.astype(int).tolist()
        return base
    raise ValueError(arm.kind)


def posmean_tables(r: dict[tuple[str, int], np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Leave-one-problem-out position means of a referenced arm's r. For problem u,
    table_u[k] = Σ_{(u', s): u' ≠ u, len > k} r_{u',s}[k] / #{(u', s): u' ≠ u, len > k}, over k < K_u, the first step
    no other problem reaches (the controller holds table_u[K_u − 1] beyond it). Returns ({uid: table}, N_k) with
    N_k the traces (all problems) reaching step k. Raises when a problem has no other problem to average."""
    L = max((len(v) for v in r.values()), default=0)
    tot, cnt = np.zeros(L, np.float64), np.zeros(L, np.float64)
    own: dict[str, list[np.ndarray]] = {}
    for (u, _), v in r.items():
        v = np.asarray(v, np.float64)
        tot[: len(v)] += v
        cnt[: len(v)] += 1
        o = own.setdefault(u, [np.zeros(L, np.float64), np.zeros(L, np.float64)])
        o[0][: len(v)] += v
        o[1][: len(v)] += 1
    out = {}
    for u, (s_u, c_u) in own.items():
        c = cnt - c_u                                       # non-increasing in k
        K = int((c > 0).sum())
        if K == 0:
            raise ValueError(f"posmean: problem {u!r} has no other problem's r trajectory to average")
        out[u] = ((tot[:K] - s_u[:K]) / c[:K]).astype(np.float32)
    return out, cnt


def save_steered_r(path: Path, r: dict[tuple[str, int], np.ndarray]) -> dict:
    keys = sorted(r)
    off = np.cumsum([0] + [len(r[k]) for k in keys])
    np.savez(path, uid=np.array([k[0] for k in keys]), sample=np.array([k[1] for k in keys]), offset=off,
             r=np.concatenate([r[k] for k in keys]) if keys else np.zeros(0, np.float32))
    from ca2.common import _sha256
    return {"path": str(path), "sha256": _sha256(path)}


def load_steered_r(trace_dir: Path, key: str) -> dict[tuple[str, int], np.ndarray]:
    p = trace_dir.expanduser() / key / "steered_r.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p}: the referenced arm's steered-r file is missing (rerun it with --force)")
    z = np.load(p)
    off = z["offset"]
    return {(str(u), int(s)): z["r"][off[i]:off[i + 1]] for i, (u, s) in enumerate(zip(z["uid"], z["sample"]))}


def _mean(xs) -> float | None:
    v = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    return float(np.mean(v)) if v else None


PER_ITEM = ("correct", "raw_correct", "truncated", "closed", "n_reason", "n_gen", "ttc", "censored", "flips",
            "backtrack_rate", "rep_onset", "distinct4", "self_loop", "gain_mean", "gain_sd", "n_steered",
            "kl_reason", "kl_resp", "ent_reason", "logp_reason", "logp_base_reason", "top1_reason", "tf_gain_dev",
            "wall_s", "wall_net_s")


def per_item_arrays(recs: list[dict], budgets: Sequence[int]) -> dict:
    out = {"uid": [r["uid"] for r in recs], "sample": [r["sample"] for r in recs]}
    for k in PER_ITEM:
        out[k] = [(_num(r.get(k)) if not isinstance(r.get(k), bool) else r.get(k)) for r in recs]
    for b in budgets:
        out[f"acc_b{b}"] = [r["acc_budget"].get(b) for r in recs]
    out["probed"] = [bool(r.get("probed")) for r in recs]
    out["gold_top1_first"] = [r.get("gold_top1_first") for r in recs]
    # agreement across samples of one problem (from answers; the answers themselves stay in the trace)
    by: dict[str, list] = {}
    for i, r in enumerate(recs):                              # an unparsed answer agrees with nothing
        a = r["answer"]
        x = to_float(a) if a is not None else None
        by.setdefault(r["uid"], []).append(("n", round(x, 9)) if x is not None else
                                           (("s", norm_tex(a)) if a is not None else ("none", i)))
    agree = {u: (Counter(v).most_common(1)[0][1] / len(v)) if len(v) > 1 else None for u, v in by.items()}
    out["agreement"] = [agree[r["uid"]] for r in recs]
    return out


def aggregate(pi: dict, budgets: Sequence[int], tmax: float) -> dict:
    n = len(pi["uid"])
    agg = {"n_rows": n, "n_problems": len(set(pi["uid"]))}
    for k in ("correct", "raw_correct", "truncated", "closed", "flips", "backtrack_rate", "distinct4", "self_loop",
              "gain_mean", "kl_reason", "kl_resp", "ent_reason", "logp_reason", "logp_base_reason", "top1_reason",
              "agreement", "tf_gain_dev"):
        agg[f"{k}_mean"] = _mean([float(x) if isinstance(x, bool) else x for x in pi[k]])
    nr = np.asarray([x for x in pi["n_reason"]], dtype=np.float64)
    agg["n_reason_mean"] = float(nr.mean()) if n else None
    agg["n_reason_median"] = float(np.median(nr)) if n else None
    agg["n_gen_mean"] = _mean(pi["n_gen"])
    unt = [c for c, t in zip(pi["correct"], pi["truncated"]) if not t]
    agg["acc"] = agg.pop("correct_mean")
    agg["acc_untruncated"] = _mean([float(c) for c in unt])
    agg["trunc_rate"] = agg.pop("truncated_mean")
    agg["rep_rate"] = _mean([float(x is not None) for x in pi["rep_onset"]])
    agg["rep_onset_mean"] = _mean([x for x in pi["rep_onset"] if x is not None])
    for b in budgets:
        agg[f"acc_b{b}"] = _mean(pi[f"acc_b{b}"])
        agg[f"acc_b{b}_n"] = sum(1 for x in pi[f"acc_b{b}"] if x is not None)
    probed = [i for i in range(n) if pi["censored"][i] is not None]
    times = [pi["ttc"][i] if not pi["censored"][i] else pi["n_reason"][i] for i in probed]
    agg["ttc_km"] = kaplan_meier(times, [not pi["censored"][i] for i in probed], tmax)
    agg["ttc_uncensored_mean"] = _mean([pi["ttc"][i] for i in probed if not pi["censored"][i]])
    agg["wall_s_sum"] = float(np.nansum([x or 0.0 for x in pi["wall_s"]]))
    agg["wall_net_s_sum"] = float(np.nansum([x or 0.0 for x in pi["wall_net_s"]]))
    return agg


def compare(pi: dict, none: dict, budgets: Sequence[int], seed: int) -> dict:
    """Paired against the none arm on (problem, sample); clusters = problems."""
    key = lambda p, i: (p["uid"][i], p["sample"][i])
    ref = {key(none, i): i for i in range(len(none["uid"]))}
    pairs = [(i, ref[key(pi, i)]) for i in range(len(pi["uid"])) if key(pi, i) in ref]
    if not pairs:
        return {"n_pairs": 0}
    cl = [pi["uid"][i] for i, _ in pairs]
    f = lambda v: float("nan") if v is None else float(v)

    def boot(name, a_of, b_of, mask=None):
        a = [f(a_of(i)) for i, _ in pairs]
        b = [f(b_of(j)) for _, j in pairs]
        if mask is not None:
            keep = [mask(i, j) for i, j in pairs]
            a = [x if k else float("nan") for x, k in zip(a, keep)]
        return paired_boot(a, b, clusters=cl, seed=seed)

    out = {"n_pairs": len(pairs)}
    c1 = [bool(pi["correct"][i]) for i, _ in pairs]
    c0 = [bool(none["correct"][j]) for _, j in pairs]
    out["wrong_to_right"] = sum(1 for a, b in zip(c1, c0) if a and not b)
    out["right_to_wrong"] = sum(1 for a, b in zip(c1, c0) if b and not a)
    out["d_acc"] = boot("acc", lambda i: pi["correct"][i], lambda j: none["correct"][j])
    lg = lambda v: math.log(max(1.0, float(v)))
    out["d_log_n_reason"] = boot("lt", lambda i: lg(pi["n_reason"][i]), lambda j: lg(none["n_reason"][j]))
    for b in budgets:
        out[f"d_acc_b{b}"] = boot("b", lambda i: pi[f"acc_b{b}"][i], lambda j: none[f"acc_b{b}"][j])
    both = lambda i, j: (pi["censored"][i] is False) and (none["censored"][j] is False)
    out["d_ttc_both_uncensored"] = boot("ttc", lambda i: pi["ttc"][i], lambda j: none["ttc"][j], both)
    for k in ("flips", "backtrack_rate", "distinct4", "self_loop", "kl_reason"):
        out[f"d_{k}"] = boot(k, lambda i, k=k: pi[k][i], lambda j, k=k: none[k][j])
    out["d_rep"] = boot("rep", lambda i: pi["rep_onset"][i] is not None, lambda j: none["rep_onset"][j] is not None)
    return out


def run_arm(env: Env, arm: Arm, st: ArmSetup, items: list[Item], args, report: dict) -> dict:
    trace_dir = Path(args.trace_dir).expanduser()
    adir = trace_dir / arm.key
    if adir.exists():
        shutil.rmtree(adir)                                  # an incomplete or forced arm starts clean
    writer = TraceWriter(arm.key, root=trace_dir, meta={"arm": arm.spec, "run": args.name})
    rows = [(it, k) for k in range(args.samples) for it in items]
    batches = []
    for k in range(args.samples):
        rk = [r for r in rows if r[1] == k]
        batches += [rk[i:i + args.batch] for i in range(0, len(rk), args.batch)]
    acc: dict = {"g": [], "r": {}}
    t0 = time.time()
    first = batches[0] if batches else []
    if st.direction == "foreign" and len(first) < 2:
        first = first * 2
    ov = measure_overhead(env, st, first, args) if first else {"ratio": 1.0}
    acc["overhead_ratio"] = ov.get("ratio", 1.0)
    recs = []
    for bi, br in enumerate(batches):
        dummy = 0
        if st.direction == "foreign" and len(br) < 2:        # the foreign direction needs another row
            other = next(it for it in items if it.uid != br[0][0].uid) if len(items) > 1 else br[0][0]
            br = br + [(other, br[0][1])]
            dummy = 1
        per_head = bool(env.hybrid and args.per_head_batches > bi and br[0][1] == 0)
        got = run_batch(env, st, br, bi, args, writer, per_head, args.probe_every > 0, acc, len(br) - dummy)
        recs += got
        print(f"  [{arm.key}] batch {bi + 1}/{len(batches)}: acc {np.mean([r['correct'] for r in got]):.2f} "
              f"reason {np.mean([r['n_reason'] for r in got]):.0f} tok  ({time.time() - t0:.0f}s)", flush=True)
    pointer = writer.close()
    rfile = save_steered_r(writer.dir / "steered_r.npz", acc["r"])
    pi = per_item_arrays(recs, args.budget_list)
    g = np.concatenate(acc["g"]) if acc["g"] else np.zeros(0)
    gs = gain_stats(torch.from_numpy(g), torch.ones(len(g)), st.g_min, st.g_max)
    return {"spec": arm.spec, "key": arm.key, "kind": st.kind, "complete": True,
            "setup": {"direction": st.direction, "seed": st.seed, "g_min": st.g_min, "g_max": st.g_max,
                      "source": st.source, "polarity": st.polarity, "smooth": st.smooth, "thinking": st.thinking,
                      "steered": st.steered, "budget": st.budget, "ref": arm.ref,
                      **({"posmean_n": st.table_n, "posmean_rule": "leave-one-problem-out"}
                         if st.kind == "posmean" else {})},
            "agg": aggregate(pi, args.budget_list, float(args.max_new_tokens)), "gain": gs, "overhead": ov,
            "per_item": pi, "trace": {**pointer, "steered_r": rfile}, "seconds": time.time() - t0}


# ------------------------------------------------------------------------------------------------- main
def versions() -> dict:
    import transformers
    out = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
           "numpy": np.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "math_verify": getattr(_math_verify(), "__version__", None) if _math_verify() else None}
    try:
        import datasets
        out["datasets"] = datasets.__version__
    except ImportError:
        pass
    return out


def write_json(path: Path, report: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = ap.add_argument
    a("--model", default="Qwen/Qwen3.5-2B")
    a("--tiny", action="store_true", help="random tiny model + byte tokenizer, offline (CPU smoke test)")
    a("--tiny-arch", default="qwen35", choices=["gpt2", "qwen3", "qwen35"])
    a("--dtype", choices=list(pl.DTYPES), default="bfloat16")
    a("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    a("--attn-impl", default="sdpa")
    a("--data", default="gsm8k:train:400", help="gsm8k:<split>[:N] | math500[:N] | notebook20 | synthetic[:N]")
    a("--data-seed", type=int, default=0, help="seed of the problem subset (fixed across runs)")
    a("--limit", type=int, default=None, help="first N problems only (smoke runs)")
    a("--instruction", default=INSTRUCTION, help="appended to each question ('' for none)")
    a("--thinking", choices=["auto", "on", "off"], default="auto")
    a("--render", choices=["auto", "plain"], default="auto", help="plain: no chat template")
    a("--samples", type=int, default=1, help="samples per problem (sample k is paired across arms)")
    a("--seed", type=int, default=0)
    a("--batch", type=int, default=32)
    a("--max-new-tokens", type=int, default=4096)
    a("--temperature", type=float, default=0.6)
    a("--top-p", type=float, default=0.95)
    a("--top-k", type=int, default=20)
    # placement and law
    a("--tap", type=int, default=None, help="block L whose output is read (default n // 2)")
    a("--inject-before", type=int, default=None, help="block M whose input is steered (default L + 1)")
    a("--attn-layer", type=int, default=None, help="attention-tap block (default: last full-attention block <= L)")
    a("--no-attn-tap", action="store_true")
    a("--arms", default=None, help="comma-separated arm specs (module docstring)")
    a("--preset", choices=["stage0", "stage1"], default="stage0")
    a("--g-min", type=float, default=0.0)
    a("--g-max", type=float, default=0.2)
    a("--g-abs", type=float, default=2.0)
    a("--smooth", choices=["none", "ema", "integral"], default="ema")
    a("--leak", type=float, default=0.9)
    a("--kappa", type=float, default=0.5)
    a("--restore-kappa", type=float, default=None)
    a("--restore-target", type=float, default=0.5)
    a("--release-rho", type=float, default=1.0)
    a("--release-words", default="Wait, Wait,wait,Hmm, Hmm,Alternatively, Alternatively")
    a("--regulator", choices=["none", "ngram", "hidden"], default="none")
    a("--betas", default="0.9,0.995")
    a("--sink-skip", type=int, default=1)
    a("--window", type=int, default=None, help="window of the window_gist direction")
    a("--attn-window", type=int, default=32)
    a("--gdn-mismatch", action=argparse.BooleanOptionalAction, default=True)
    a("--per-head-batches", type=int, default=1, help="batches of sample 0 with per-head GDN traces")
    a("--per-head-keys", default="beta,ret,fam,fam_lag,str,err")
    # measures
    a("--probe-every", type=int, default=128, help="forced-answer probe interval in span tokens (0: off)")
    a("--probe-tokens", type=int, default=16)
    a("--probe-samples", type=int, default=1, help="samples (from 0) that are probed")
    a("--probe-batch", type=int, default=32)
    a("--budgets", default=",".join(str(b) for b in BUDGETS))
    a("--answer-tokens", type=int, default=256, help="answer tokens after a forced </think> (trunc arms)")
    a("--tf", action=argparse.BooleanOptionalAction, default=True, help="teacher-forced outcome measures")
    a("--lens-layers", default=None, help="logit-lens blocks (default n/4, n/2, 3n/4, 7n/8)")
    a("--stat-chunk", type=int, default=512, help="positions per logits chunk")
    a("--calib-tokens", type=int, default=16, help="decode steps of the overhead calibration (0: off)")
    # output
    a("--name", default=None)
    a("--stage", default="stage0")
    a("--out", default=None, help="results JSON (default results/evolve/round12/<stage>/<name>.json)")
    a("--trace-dir", default=None, help="per-token traces and text (default ~/limbic-data/ca2_traces/<name>)")
    a("--force", action="store_true", help="rerun arms already complete in --out")
    return ap


# every argument that changes generation, the controller, the probes or the scoring: a resumed --out must match
RESUME_KEYS = ("model", "tiny", "tiny_arch", "dtype", "attn_impl", "data", "instruction", "thinking", "render",
               "samples", "seed", "batch", "max_new_tokens", "temperature", "top_p", "top_k", "tap", "inject_before",
               "attn_layer", "no_attn_tap", "g_min", "g_max", "g_abs", "smooth", "leak", "kappa", "restore_kappa",
               "restore_target", "release_rho", "release_words", "regulator", "betas", "sink_skip", "window",
               "attn_window", "gdn_mismatch", "per_head_batches", "per_head_keys", "probe_every", "probe_tokens",
               "probe_samples", "budgets", "answer_tokens", "tf", "lens_layers")


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    aj = alignment().judge
    if args.tiny:
        args.dtype = "float32" if args.dtype == "bfloat16" else args.dtype
    short = "tiny-" + args.tiny_arch if args.tiny else args.model.split("/")[-1]
    args.name = aj.safe_name(args.name or safe_key(f"reason_{short}_{args.data}"))
    args.out = Path(args.out or ROOT / "results" / "evolve" / "round12" / args.stage / f"{args.name}.json")
    args.trace_dir = str(aj.assert_private(args.trace_dir or Path("~/limbic-data/ca2_traces") / args.name, ROOT))
    args.budget_list = [int(b) for b in args.budgets.split(",") if b]
    if args.probe_every > 0 and any(b % args.probe_every for b in args.budget_list):
        raise SystemExit("--budgets must be multiples of --probe-every")
    torch.manual_seed(args.seed)
    t0 = time.time()
    env = load_env(args)
    items = load_items("synthetic:6" if args.tiny and args.data == "gsm8k:train:400" else args.data, args.data_seed,
                       args.limit)
    srcs = ["agree"] + (["attn_window", "attn_agree"] if env.R is not None else []) + \
        (["gdn_mismatch_lag", "gdn_mismatch", "gdn_write", "gdn_decay"] if env.hybrid else [])
    specs = [s for s in args.arms.split(",") if s.strip()] if args.arms else \
        preset_arms(args.preset, args.g_max, srcs, env.has_switch)
    arms = [parse_arm(s) for s in specs]
    for a_ in arms:
        need = a_.source if a_.kind == "law" else None
        if need and (need.startswith("attn") and env.R is None or need.startswith("gdn") and not env.hybrid):
            raise SystemExit(f"{a_.spec}: source {need} is not available on this model")
        if need and need not in SOURCES:
            raise SystemExit(f"{a_.spec}: unknown source {need}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.out.read_text()) if args.out.exists() else {}
    report.setdefault("arms", {})
    old = report.get("data", {}).get("uids")
    keep = RESUME_KEYS
    prev = report.get("args", {})
    diff = [k for k in keep if k in prev and prev[k] != getattr(args, k)]
    stale = bool(report["arms"]) and bool(diff or (old is not None and old != [it.uid for it in items]))
    if stale and not args.force:
        raise SystemExit(f"{args.out} holds a run with different settings ({diff or 'problems'}); "
                         "use another --out, or --force to start it again")
    if stale:
        report["arms"] = {}                                  # --force over different settings: start again
    report.update({"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                   "git": git_state(), "versions": versions(), "model": env.facts,
                   "data": {"spec": args.data, "n_items": len(items), "uids": [it.uid for it in items],
                            "kinds": sorted({it.kind for it in items}),
                            "scorer": "math_verify" if _math_verify() else "numeric+normalised-latex"},
                   "arm_order": [a_.spec for a_ in arms]})
    print(f"{env.facts['name']}: {env.facts['n_blocks']} blocks, tap {env.L} -> inject {env.M}, attn tap {env.R}, "
          f"hybrid {env.hybrid}, thinking {env.thinking}; {len(items)} problems x {args.samples} samples; "
          f"{len(arms)} arms", flush=True)
    for a_ in arms:
        if report["arms"].get(a_.spec, {}).get("complete") and not args.force:
            print(f"[{a_.key}] complete in {args.out.name}; skipped (--force reruns)", flush=True)
            continue
        st = resolve(a_, report, args, env)
        print(f"[{a_.key}] {st.kind} dir {st.direction} g [{st.g_min:g}, {st.g_max:g}] source {st.source} "
              f"{st.polarity}" + (f" budget {st.budget}" if st.budget else ""), flush=True)
        report["arms"][a_.spec] = run_arm(env, a_, st, items, args, report)
        none = report["arms"].get("none")
        for spec, res in report["arms"].items():
            if none and none.get("complete") and spec != "none" and res.get("complete"):
                res["vs_none"] = compare(res["per_item"], none["per_item"], args.budget_list, args.seed)
        report["seconds"] = time.time() - t0
        write_json(args.out, report)
        ag = report["arms"][a_.spec]["agg"]
        print(f"[{a_.key}] acc {ag['acc']:.3f}  reason {ag['n_reason_mean']:.0f}  trunc {ag['trunc_rate']:.2f}  "
              f"gain {report['arms'][a_.spec]['gain'].get('mean', 0) or 0:.3f}  TTC median "
              f"{ag['ttc_km']['median']}", flush=True)
    report["seconds"] = time.time() - t0
    write_json(args.out, report)
    print(f"saved {args.out} (no generated text); traces in {args.trace_dir}", flush=True)
    return report


if __name__ == "__main__":
    main()
