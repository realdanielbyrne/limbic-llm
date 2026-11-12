"""Decoding control for the round-12 thinking experiments (C1, C3): logits processors, stop controllers,
forced-answer probes and segmented generation with a forked probe.

Everything here runs inside Hugging Face ``generate(logits_processor=[...])``: a processor is called once per
decode step, between forwards, with the whole (B, t) ``input_ids`` (left-padded prompt of width P, then the
t − P generated tokens) and the (B, V) float scores of the next token. ``generate`` puts custom processors after
its own non-sampling processors and before its sampling warpers (temperature, top-k, top-p), so every penalty
and bias below acts on the logits before temperature, as in vLLM. ``build_processors`` fixes the order.

Reasoning span (``SpanTracker``). Per row, from the ids alone (recomputed at every call, nothing carried): with
events e_j = +1 at the span-open marker (``<think>``), −1 at the close marker (``</think>``), 0 elsewhere over
the generated columns, the state after column j is the sign of the last non-zero event up to j, or the prompt's
state when there is none (the prompt's state is computed the same way over the prompt, or given). This covers
both layouts: Qwen3.5 (and reason.py's ``render`` for every model) end the prompt in ``<think>\\n``, so the span is
open from the first generated token; Qwen3's own template does not, and the model emits ``<think>`` itself.
n_span = #{generated j : state before j open, e_j = 0} counts reasoning tokens (markers excluded). With the stop
ids given (``stops``), a row is finished from its first generated stop id on: its span counts as closed there
and nothing after it counts, so no detector, controller or span-gated processor acts on a finished row (it is
fed pad ids, and Qwen's pad id is itself a stop; ``generate`` resets its own finished mask at every call of a
segmented generation, so the gate must not rely on it).

Processors (all batched; "stateless" means the output is a function of (input_ids, scores) alone):
  PresencePenalty   OpenAI / vLLM semantics. With c_b[v] = #{generated j of row b : x_j = v} (prompt columns
                    excluded), z_b[v] ← z_b[v] − α_presence·1[c_b[v] > 0] − α_frequency·c_b[v]. The penalty is
                    subtracted once per token type whatever its count (presence) and is additive in logit space.
                    This is NOT transformers' ``repetition_penalty`` (CTRL, Keskar et al. 2019): that one is
                    multiplicative (z/θ for z > 0, z·θ for z < 0), so its effect depends on the logit's sign and
                    size, and it counts the prompt's tokens too. Qwen's card value "presence_penalty 1.5" means the
                    former. Stateless; by default the whole generated response is penalised (vendor semantics);
                    ``span`` restricts it to rows whose reasoning span is open.
  ThinkCloseBias    z_b[close] += β_b while row b's span is open; β (B,) is set between steps (``set``).
  ForceTokens       per-row FIFO of ids; while row b's queue is non-empty the step's scores become −inf except
                    the head id (0), which is popped, so sampling (any T, top-k, top-p) and greedy emit exactly the
                    queue; an empty queue leaves the row untouched. Stateful (the queues).
  SuppressTokens    z_b[v] = −inf for v in a set while row b's span is open (the NoWait arm).
  NgramLoopRule     a detector: at a call with t generated tokens (t ≥ W), with the last W generated tokens, fires
                    when 1 − distinct n-grams / (W − n + 1) > ρ, i.e. ``judge.degeneration`` of that window,
                    and only
                    while the span is open (the window may still hold tokens from before a span the model
                    opened itself, Qwen3's own layout). ``fired_at`` = the first such t, which equals
                    ``judge.repetition_onset(generated ids, n, W, ρ)`` when the span covers the generation
                    (tests/test_ca2_decode.py). Sticky; the scores pass through unchanged.
  BudgetRule        a detector that fires when n_span ≥ B (real truncation at B reasoning tokens).
  ProbeRule         a detector fed by forced-answer probes (``observe``): ``consist`` fires when the last k parsed
                    probe answers agree (numbers as numbers, as ``reason.count_flips``); ``conf`` fires when the
                    probe's answer confidence q ≥ τ (DEER: q = geometric mean of the answer tokens' probabilities).
  AnyOf             fires when any of its detectors has fired; ``why`` names the first one.
  StopController    detector + action, only inside the span: at the first step where the detector has fired and
                    the row's span is open it acts once and records (step, n_span, why). Actions: ``stop`` pushes
                    the canonical close (below) into ``ForceTokens``; ``bias<β>`` sets β on ``ThinkCloseBias``;
                    ``none`` only records (the detector-alone arm).
  TokenStats        records, per step and row, the entropy H = −Σ p log p of the raw model distribution (T = 1,
                    before any penalty; as reason.py's teacher-forced ``out/ent``), log p of the token that was
                    then chosen and p(close).
  FinishedRows      forces the pad id on rows that already emitted a stop id (what ``generate`` does internally;
                    needed when generation is split over several ``generate`` calls).

Canonical close. Qwen3's thinking budget (Qwen3 technical report, arXiv 2505.09388) stops the reasoning by
inserting "Considering the limited time by the user, I have to give the solution based on the thinking directly
now." followed by the end of thinking. The chat template's own closing is "\\n</think>\\n\\n" (Qwen3 and Qwen3.5:
``'\\n</think>\\n\\n' + content``), so the forced sequence is
    CANONICAL_CLOSE = "\\n\\nConsidering the limited time by the user, I have to give the solution based on the
                       thinking directly now.\\n</think>\\n\\n"
(the paragraph break first, as in the design note). ``close_sequence`` tokenizes the text on each side of the
marker and places the marker's id between them.

Forced-answer probe. As reason.py: after a prefix the cue ``[</think>] + "\\n\\nThe final answer is \\boxed{"``
is appended and ``probe_tokens`` tokens are decoded greedily without any processor; the answer is the text up to
the brace closing the cue's ``\\boxed{`` (``reason.forced_answer``). Added here: the answer's confidence q (the
geometric mean of p over the answer tokens, those before the token that closes the brace; DEER) and the entropy
of the first answer token. ``probe_prefixes`` re-prefills each prefix (reason.py's method); ``forked_probe`` reads
the open generation's cache instead: full-attention layers (``DynamicLayer``) are cropped back after the probe,
every other cache layer (linear-attention / GDN conv and recurrent states, sliding windows) is deep-copied before
and put back after, so the probe leaves the cache exactly as it found it.

Segmented generation (``generate_segmented``). ``generate`` is called for E tokens at a time on the same cache
(input = everything so far; the cache holds all but the last token, which the next call feeds); between calls an
optional forked probe runs. The probe is greedy and draws no random numbers, the cache is restored, and finished
rows are fed the pad id as ``generate`` would, so the main continuation is token for token the one an
unsegmented ``generate`` with the same seed produces (tested on tiny Qwen3 and Qwen3.5, fp32, CPU).
"""

from __future__ import annotations

import copy
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList, RepetitionPenaltyLogitsProcessor

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from ca2 import reason as R  # noqa: E402

CLOSE_SENTENCE = ("Considering the limited time by the user, I have to give the solution based on the thinking "
                  "directly now.")
CANONICAL_CLOSE = "\n\n" + CLOSE_SENTENCE + "\n</think>\n\n"
CLOSE_MARKER = "</think>"
WAIT_WORDS = ("wait", "hmm", "alternatively")             # NoWait (arXiv 2506.08343): "Wait", "Hmm", "Alternatively"
NEG = float("-inf")


# ------------------------------------------------------------------------------------------- token sequences
def close_sequence(tok, close_id: int, text: str = CANONICAL_CLOSE, marker: str = CLOSE_MARKER) -> list[int]:
    """Ids of ``text`` with ``marker`` replaced by ``close_id`` (the text on each side tokenized alone)."""
    if text.count(marker) != 1:
        raise ValueError(f"the close text must contain {marker!r} exactly once")
    pre, post = text.split(marker)
    return list(tok.encode(pre)) + [int(close_id)] + list(tok.encode(post))


def cue_sequence(tok, close_id: int | None) -> list[int]:
    """The forced-answer cue of reason.py: ``[close] + "\\n\\nThe final answer is \\boxed{"``."""
    return ([int(close_id)] if close_id is not None else []) + list(tok.encode(R.CUE_ANSWER))


def wait_token_ids(tok, words: Sequence[str] = WAIT_WORDS) -> list[int]:
    """Single-token ids of the words, capitalised or not, with and without a leading space."""
    out = set()
    for w in words:
        for v in (w, w.capitalize(), " " + w, " " + w.capitalize()):
            ids = tok.encode(v)
            if len(ids) == 1:
                out.add(int(ids[0]))
    return sorted(out)


def answer_key(a: str | None):
    """Comparison key of a parsed answer (numbers as numbers, rounded to 1e-9; else normalised LaTeX), or None."""
    if a is None:
        return None
    x = R.to_float(a)
    return ("n", round(x, 9)) if x is not None else ("s", R.norm_tex(a))


def _prompt_len(prompt_len: int | torch.Tensor, B: int, dev) -> torch.Tensor:
    if isinstance(prompt_len, torch.Tensor):
        return prompt_len.to(dev).long().reshape(B)
    return torch.full((B,), int(prompt_len), dtype=torch.long, device=dev)


# ------------------------------------------------------------------------------------------- span tracking
class SpanTracker:
    """Per-row reasoning-span state from the ids (module docstring). ``prompt_len``: width of the left-padded
    prompt; ``prompt_open`` (B,) bool or bool: the state at the end of the prompt (computed from the prompt's own
    markers when None, which needs ``open_id``); after the first ``state`` call it is a (B,) bool tensor."""

    def __init__(self, close_id: int | None, prompt_len: int, open_id: int | None = None,
                 prompt_open: torch.Tensor | bool | None = None, stops: Sequence[int] = ()) -> None:
        self.close_id = None if close_id is None else int(close_id)
        self.open_id = None if open_id is None else int(open_id)
        self.P = int(prompt_len)
        self.prompt_open = prompt_open
        self.stops = torch.as_tensor(sorted({int(s) for s in stops}), dtype=torch.long)
        self._key = None
        self._val: tuple[torch.Tensor, torch.Tensor] | None = None

    def _initial(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, dev = input_ids.shape[0], input_ids.device
        if isinstance(self.prompt_open, torch.Tensor):
            return self.prompt_open.to(dev).bool().reshape(B)
        if isinstance(self.prompt_open, bool):
            self.prompt_open = torch.full((B,), self.prompt_open, dtype=torch.bool, device=dev)
            return self.prompt_open
        if self.open_id is None or self.close_id is None:
            raise ValueError("SpanTracker: give prompt_open, or both marker ids")
        self.prompt_open = self._last_state(input_ids[:, : self.P], torch.zeros(B, dtype=torch.bool, device=dev))[0]
        return self.prompt_open

    def _last_state(self, ids: torch.Tensor, init: torch.Tensor, gate_stops: bool = False
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(state after the last column, number of in-span non-marker columns) of ``ids`` starting from ``init``.
        ``gate_stops`` (generated columns only): a row is finished from its first stop id on, so its span is
        closed there and nothing after it (the pad ids a finished row is fed) counts as reasoning."""
        B, T = ids.shape
        if T == 0:
            return init, torch.zeros(B, dtype=torch.long, device=ids.device)
        ev = torch.zeros_like(ids)
        if self.open_id is not None:
            ev = ev + (ids == self.open_id).long()
        if self.close_id is not None:
            ev = ev - (ids == self.close_id).long()
        pos = torch.arange(T, device=ids.device).expand(B, T)
        last = torch.where(ev != 0, pos, torch.full_like(pos, -1)).cummax(1).values      # last event index ≤ j
        after = torch.where(last >= 0, ev.gather(1, last.clamp_min(0)) > 0, init[:, None])
        before = torch.cat([init[:, None], after[:, :-1]], 1)
        live = torch.ones_like(ids, dtype=torch.bool)
        if gate_stops and self.stops.numel():
            live = torch.isin(ids, self.stops.to(ids.device)).long().cumsum(1) == 0          # before the first stop
        n_span = (before & (ev == 0) & live).sum(1)
        return after[:, -1] & live[:, -1], n_span

    def state(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(open (B,) bool, n_span (B,) long) after the last column of ``input_ids``; memoised per call."""
        if self._key is input_ids and self._val is not None:
            return self._val
        init = self._initial(input_ids)
        self._val = self._last_state(input_ids[:, self.P:], init, gate_stops=True)
        self._key = input_ids
        return self._val


class _Gate:
    """Mixin: rows on which a processor acts (span open, when a tracker is given)."""
    span: SpanTracker | None = None

    def _rows(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.span is None:
            return torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        return self.span.state(input_ids)[0]


# ------------------------------------------------------------------------------------------- processors
class PresencePenalty(LogitsProcessor, _Gate):
    """OpenAI / vLLM presence (and frequency) penalty over the generated columns only (module docstring)."""

    def __init__(self, presence: float, prompt_len: int | torch.Tensor, frequency: float = 0.0,
                 span: SpanTracker | None = None) -> None:
        self.presence, self.frequency = float(presence), float(frequency)
        self.prompt_len, self.span = prompt_len, span

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        B, T = input_ids.shape
        P = _prompt_len(self.prompt_len, B, input_ids.device)
        valid = torch.arange(T, device=input_ids.device)[None, :] >= P[:, None]
        counts = torch.zeros_like(scores).scatter_add_(1, input_ids, valid.to(scores.dtype))
        pen = self.presence * (counts > 0).to(scores.dtype) + self.frequency * counts
        rows = self._rows(input_ids)
        return scores - pen * rows[:, None].to(scores.dtype)


class ThinkCloseBias(LogitsProcessor, _Gate):
    """z_b[close] += β_b while row b's span is open; ``set(rows, β)`` between steps."""

    def __init__(self, close_id: int, batch: int, span: SpanTracker | None = None) -> None:
        self.close_id, self.span = int(close_id), span
        self.bias = torch.zeros(batch)

    def set(self, rows, value: float) -> None:
        self.bias[torch.as_tensor(rows, dtype=torch.long)] = float(value)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        b = self.bias.to(scores.device, scores.dtype) * self._rows(input_ids).to(scores.dtype)
        if not torch.any(b != 0):
            return scores
        scores = scores.clone()
        scores[:, self.close_id] += b
        return scores


class ForceTokens(LogitsProcessor):
    """Per-row FIFO of ids to emit next (module docstring). ``emitted[b]`` logs (generated step, id)."""

    def __init__(self, batch: int, prompt_len: int = 0) -> None:
        self.queues: list[deque] = [deque() for _ in range(batch)]
        self.emitted: list[list[tuple[int, int]]] = [[] for _ in range(batch)]
        self.P = int(prompt_len)

    def push(self, row: int, ids: Sequence[int]) -> None:
        self.queues[row].extend(int(i) for i in ids)

    def pending(self, row: int) -> int:
        return len(self.queues[row])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        rows = [b for b, q in enumerate(self.queues) if q]
        if not rows:
            return scores
        scores = scores.clone()
        step = input_ids.shape[1] - self.P
        for b in rows:
            v = self.queues[b].popleft()
            scores[b] = NEG
            scores[b, v] = 0.0
            self.emitted[b].append((step, v))
        return scores


class SuppressTokens(LogitsProcessor, _Gate):
    """z_b[v] = −inf for v in ``ids`` while row b's span is open (and b is in ``rows``, default all)."""

    def __init__(self, ids: Sequence[int], span: SpanTracker | None = None, rows: torch.Tensor | None = None) -> None:
        self.ids = torch.as_tensor(sorted({int(i) for i in ids}), dtype=torch.long)
        self.span, self.only = span, rows

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.ids.numel() == 0:
            return scores
        rows = self._rows(input_ids)
        if self.only is not None:
            rows = rows & self.only.to(rows.device).bool()
        if not rows.any():
            return scores
        scores = scores.clone()
        sub = scores[:, self.ids.to(scores.device)]
        scores[:, self.ids.to(scores.device)] = torch.where(rows[:, None], torch.full_like(sub, NEG), sub)
        return scores


class FinishedRows(LogitsProcessor):
    """Forces ``pad`` on rows whose generated columns already hold a stop id (as ``generate`` does)."""

    def __init__(self, stops: Sequence[int], pad: int, prompt_len: int) -> None:
        self.stops = torch.as_tensor(list(stops), dtype=torch.long)
        self.pad, self.P = int(pad), int(prompt_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        G = input_ids[:, self.P:]
        if G.shape[1] == 0:
            return scores
        done = torch.isin(G, self.stops.to(G.device)).any(1)
        if not done.any():
            return scores
        scores = scores.clone()
        scores[done] = NEG
        scores[done, self.pad] = 0.0
        return scores


class TokenStats(LogitsProcessor):
    """Per step: entropy of the raw model distribution (T = 1), log p of the token then chosen, p(close).

    ``arrays()`` returns (B, steps) float32 numpy arrays; column s describes the distribution of generated token
    s (0-based) and the log p of the token actually emitted there (NaN for the last step until ``finish``)."""

    def __init__(self, close_id: int | None = None) -> None:
        self.close_id = close_id
        self.ent: list[torch.Tensor] = []
        self.p_close: list[torch.Tensor] = []
        self.logp: list[torch.Tensor] = []
        self._prev: torch.Tensor | None = None

    def _take_prev(self, chosen: torch.Tensor) -> None:
        if self._prev is not None:
            self.logp.append(self._prev.gather(1, chosen[:, None].to(self._prev.device))[:, 0].float())
            self._prev = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._take_prev(input_ids[:, -1])
        lp = F.log_softmax(scores.float(), -1)
        self.ent.append(-(lp.exp() * lp).sum(-1))
        if self.close_id is not None:
            self.p_close.append(lp[:, self.close_id].exp())
        self._prev = lp
        return scores

    def finish(self, last_ids: torch.Tensor) -> None:
        """Record log p of the last emitted tokens (the final step is never fed back to a processor)."""
        self._take_prev(last_ids)

    def arrays(self) -> dict[str, np.ndarray]:
        n = len(self.ent)
        if n == 0:
            return {"ent": np.zeros((0, 0), np.float32), "logp": np.zeros((0, 0), np.float32),
                    "p_close": np.zeros((0, 0), np.float32)}
        B = self.ent[0].shape[0]
        ent = torch.stack(self.ent, 1).cpu().numpy()
        lp = np.full((B, n), np.nan, np.float32)
        if self.logp:
            lp[:, : len(self.logp)] = torch.stack(self.logp, 1).cpu().numpy()
        pc = torch.stack(self.p_close, 1).cpu().numpy() if self.p_close else np.full((B, n), np.nan, np.float32)
        return {"ent": ent.astype(np.float32), "logp": lp, "p_close": pc.astype(np.float32)}


# ------------------------------------------------------------------------------------------- detectors
class Detector(LogitsProcessor):
    """Sticky per-row detector; ``update`` returns the (B,) fired mask. As a processor it only observes."""

    name = "detector"

    def __init__(self, batch: int) -> None:
        self.fired_at = torch.full((batch,), -1, dtype=torch.long)       # generated tokens when it first fired
        self.fired_span = torch.full((batch,), -1, dtype=torch.long)     # n_span then
        self.span: SpanTracker | None = None

    @property
    def fired(self) -> torch.Tensor:
        return self.fired_at >= 0

    def _mark(self, now: torch.Tensor, t: int, n_span: torch.Tensor) -> None:
        now = now.cpu() & ~self.fired
        self.fired_at[now] = int(t)
        self.fired_span[now] = n_span.cpu()[now]

    def check(self, input_ids: torch.Tensor, open_: torch.Tensor, n_span: torch.Tensor, t: int) -> torch.Tensor:
        raise NotImplementedError

    def update(self, input_ids: torch.Tensor, open_: torch.Tensor, n_span: torch.Tensor, t: int) -> torch.Tensor:
        now = self.check(input_ids, open_, n_span, t)
        if now is not None:
            self._mark(now.cpu() & open_.cpu(), t, n_span)
        return self.fired

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.span is None:
            raise ValueError(f"{self.name}: used as a processor, the detector needs a SpanTracker (.span)")
        open_, n_span = self.span.state(input_ids)
        self.update(input_ids, open_, n_span, input_ids.shape[1] - self.span.P)
        return scores


def ngram_repeat_share(win: torch.Tensor, n: int) -> torch.Tensor:
    """``judge.degeneration`` of each row of ``win`` (B, W): 1 − distinct n-grams / (W − n + 1); 0 if W < 2n."""
    B, W = win.shape
    if W < 2 * n:
        return torch.zeros(B, device=win.device)
    g = win.unfold(1, n, 1)                                                # (B, G, n)
    G = g.shape[1]
    eq = (g[:, :, None, :] == g[:, None, :, :]).all(-1)                    # (B, G, G)
    seen = torch.tril(eq, diagonal=-1).any(-1)                             # equal to an earlier n-gram
    return seen.float().sum(1) / G


class NgramLoopRule(Detector):
    """Fires when the last ``window`` generated tokens have a repeated-n-gram share above ``threshold``
    (``judge.repetition_onset`` semantics; module docstring)."""

    def __init__(self, batch: int, prompt_len: int, window: int = 32, n: int = 4, threshold: float = 0.5,
                 every: int = 1) -> None:
        super().__init__(batch)
        self.P, self.W, self.n, self.thr, self.every = int(prompt_len), int(window), int(n), float(threshold), every
        self.name = f"ngram-{self.W}-{self.n}-{self.thr:g}"

    def check(self, input_ids, open_, n_span, t):
        if t < self.W or (t - self.W) % self.every:
            return None
        win = input_ids[:, input_ids.shape[1] - self.W:]
        return (ngram_repeat_share(win, self.n) > self.thr).cpu()


class BudgetRule(Detector):
    """Fires when n_span ≥ ``budget`` (real truncation at a reasoning budget)."""

    def __init__(self, batch: int, budget: int) -> None:
        super().__init__(batch)
        self.budget = int(budget)
        self.name = f"budget-{self.budget}"

    def check(self, input_ids, open_, n_span, t):
        return (n_span >= self.budget).cpu()


class ProbeRule(Detector):
    """Fed by forced-answer probes: ``consist`` (last k answers agree) or ``conf`` (confidence ≥ τ)."""

    def __init__(self, batch: int, kind: str, k: int = 3, tau: float = 0.95) -> None:
        super().__init__(batch)
        if kind not in ("consist", "conf"):
            raise ValueError(kind)
        self.kind, self.k, self.tau = kind, int(k), float(tau)
        self.hist: list[list] = [[] for _ in range(batch)]
        self._pending = torch.zeros(batch, dtype=torch.bool)
        self._at: list[tuple[int, int] | None] = [None] * batch
        self.name = f"consist-{self.k}" if kind == "consist" else f"conf-{self.tau:g}"

    def observe(self, rows: Sequence[int], answers: Sequence[str | None], conf: Sequence[float | None],
                t: int, n_span: Sequence[int]) -> None:
        """Record one probe per listed row (taken at t generated tokens, n_span reasoning tokens)."""
        for b, a, q, ns in zip(rows, answers, conf, n_span):
            self.hist[b].append(answer_key(a))
            if self.kind == "consist":
                h = self.hist[b][-self.k:]
                hit = len(h) == self.k and h[0] is not None and all(x == h[0] for x in h)
            else:
                hit = q is not None and q >= self.tau
            if hit and not self.fired[b] and not self._pending[b]:
                self._pending[b] = True
                self._at[b] = (int(t), int(ns))

    def check(self, input_ids, open_, n_span, t):
        return None

    def update(self, input_ids, open_, n_span, t):
        for b in torch.nonzero(self._pending & open_.cpu()).flatten().tolist():
            self.fired_at[b], self.fired_span[b] = self._at[b]
        self._pending &= ~self.fired
        return self.fired


class AnyOf(Detector):
    """Fires when any detector has fired; ``why[b]`` is the name of the first one (ties: list order)."""

    def __init__(self, batch: int, detectors: Sequence[Detector]) -> None:
        super().__init__(batch)
        self.parts = list(detectors)
        self.why: list[str | None] = [None] * batch
        self.name = "any(" + ",".join(d.name for d in self.parts) + ")"

    def update(self, input_ids, open_, n_span, t):
        for d in self.parts:
            d.update(input_ids, open_, n_span, t)
            new = d.fired & ~self.fired
            for b in torch.nonzero(new).flatten().tolist():
                self.why[b] = d.name
            self.fired_at[new] = d.fired_at[new]
            self.fired_span[new] = d.fired_span[new]
        return self.fired


class StopController(LogitsProcessor):
    """Detector + action inside the reasoning span (module docstring); records per row what it did and when."""

    def __init__(self, detector: Detector, action: str, span: SpanTracker, batch: int,
                 force: ForceTokens | None = None, bias: ThinkCloseBias | None = None,
                 close_ids: Sequence[int] = (), bias_value: float = 0.0) -> None:
        if action == "stop" and (force is None or not close_ids):
            raise ValueError("action 'stop' needs ForceTokens and the close ids")
        if action == "bias" and bias is None:
            raise ValueError("action 'bias' needs ThinkCloseBias")
        if action not in ("stop", "bias", "none"):
            raise ValueError(action)
        self.det, self.action, self.span = detector, action, span
        self.force, self.bias, self.close_ids, self.bias_value = force, bias, list(close_ids), float(bias_value)
        self.act_step = torch.full((batch,), -1, dtype=torch.long)
        self.act_span = torch.full((batch,), -1, dtype=torch.long)
        self.why: list[str | None] = [None] * batch

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        open_, n_span = self.span.state(input_ids)
        t = input_ids.shape[1] - self.span.P
        fired = self.det.update(input_ids, open_, n_span, t)
        new = fired & open_.cpu() & (self.act_step < 0)
        for b in torch.nonzero(new).flatten().tolist():
            self.act_step[b], self.act_span[b] = t, int(n_span[b])
            self.why[b] = self.det.why[b] if isinstance(self.det, AnyOf) else self.det.name
            if self.action == "stop":
                self.force.push(b, self.close_ids)
            elif self.action == "bias":
                self.bias.set([b], self.bias_value)
        return scores

    def record(self, b: int) -> dict:
        return {"detector": self.det.name, "action": self.action + (f"{self.bias_value:g}" if self.action == "bias"
                                                                    else ""),
                "detected_step": int(self.det.fired_at[b]), "detected_span": int(self.det.fired_span[b]),
                "act_step": int(self.act_step[b]), "act_span": int(self.act_span[b]), "why": self.why[b]}


def build_processors(*, prompt_len: int, batch: int, span: SpanTracker | None, presence: float = 0.0,
                     frequency: float = 0.0, repetition: float = 1.0, suppress: Sequence[int] = (),
                     controller: tuple[Detector, str, float] | None = None, close_id: int | None = None,
                     close_ids: Sequence[int] = (), finished: tuple[Sequence[int], int] | None = None,
                     stats: bool = True, penalty_span_only: bool = False) -> tuple[LogitsProcessorList, dict]:
    """The processor list in its fixed order, and the parts by name.

    Order: TokenStats (raw logits) → StopController (decides; no score change) → repetition penalty (HF, CTRL
    semantics, prompt included) → PresencePenalty → SuppressTokens → ThinkCloseBias → FinishedRows →
    ForceTokens.
    ``controller`` = (detector, action ∈ stop|bias|none, β)."""
    parts: dict = {}
    lst = LogitsProcessorList()
    if stats:
        parts["stats"] = TokenStats(close_id)
        lst.append(parts["stats"])
    force = ForceTokens(batch, prompt_len)
    bias = ThinkCloseBias(close_id, batch, span) if close_id is not None else None
    if span is not None and finished is not None and not span.stops.numel():
        span.stops = torch.as_tensor(sorted({int(s) for s in finished[0]}), dtype=torch.long)   # finished rows gated
    if controller is not None:
        det, action, beta = controller
        if span is None:
            raise ValueError("a controller needs the SpanTracker")
        det.span = span
        parts["controller"] = StopController(det, action, span, batch, force=force, bias=bias, close_ids=close_ids,
                                             bias_value=beta)
        lst.append(parts["controller"])
    if repetition != 1.0:
        parts["repetition"] = RepetitionPenaltyLogitsProcessor(repetition)
        lst.append(parts["repetition"])
    if presence or frequency:
        parts["presence"] = PresencePenalty(presence, prompt_len, frequency, span if penalty_span_only else None)
        lst.append(parts["presence"])
    if suppress:
        parts["suppress"] = SuppressTokens(suppress, span)
        lst.append(parts["suppress"])
    if bias is not None and controller is not None and controller[1] == "bias":
        parts["bias"] = bias
        lst.append(bias)
    if finished is not None:
        parts["finished"] = FinishedRows(finished[0], finished[1], prompt_len)
        lst.append(parts["finished"])
    parts["force"] = force
    lst.append(force)
    return lst, parts


# ------------------------------------------------------------------------------------------- probes
@dataclass
class ProbeResult:
    answer: str | None
    conf: float | None            # geometric mean p over the answer tokens (DEER)
    ent_first: float | None       # entropy of the first answer token's distribution
    n_answer_tokens: int


def read_probe(new_ids: Sequence[int], logp_rows: torch.Tensor | None, tok, stops: Sequence[int]) -> ProbeResult:
    """Answer and confidence of one probe row: ``new_ids`` the greedy tokens, ``logp_rows`` (k, V) log-softmax of
    the step distributions (raw logits)."""
    ids, _ = R.cut_stop(list(new_ids), stops)
    ids = [t for t in ids if t not in set(stops)]
    text = tok.decode(ids)
    ans = R.forced_answer(text)
    close_at = None
    for j in range(len(ids)):
        if R.brace_body(tok.decode(ids[: j + 1]), 0) is not None:
            close_at = j
            break
    span = list(range(close_at if close_at is not None else len(ids)))
    if not span and ids:
        span = [0]
    conf = ent = None
    if logp_rows is not None and span:
        lp = logp_rows[span].gather(1, torch.as_tensor([ids[j] for j in span])[:, None].to(logp_rows.device))
        conf = float(lp.float().mean().exp())
        l0 = logp_rows[0].float()
        ent = float(-(l0.exp() * l0).sum())
    return ProbeResult(ans, conf, ent, len(span))


@torch.no_grad()
def _greedy(model, ids: torch.Tensor, mask: torch.Tensor, n_tokens: int, stops: Sequence[int], pad: int,
            cache=None) -> tuple[torch.Tensor, torch.Tensor]:
    kw = dict(max_new_tokens=n_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
              eos_token_id=list(stops), pad_token_id=pad, output_logits=True, return_dict_in_generate=True)
    if cache is not None:
        kw["past_key_values"] = cache
    out = model.generate(input_ids=ids, attention_mask=mask, **kw)
    new = out.sequences[:, ids.shape[1]:]
    lps = torch.stack([F.log_softmax(x.float(), -1) for x in out.logits], 1)            # (B, k, V)
    return new, lps


@torch.no_grad()
def probe_prefixes(model, tok, prefixes: Sequence[Sequence[int]], cue: Sequence[int], n_tokens: int,
                   stops: Sequence[int], pad: int, batch: int, dev) -> list[ProbeResult]:
    """reason.py's re-prefill probe (prefix + cue, greedy, sorted by length), with the confidence added."""
    out: list[ProbeResult | None] = [None] * len(prefixes)
    order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        ids, mask = R.pad_left([list(prefixes[i]) + list(cue) for i in idx], pad)
        new, lps = _greedy(model, ids.to(dev), mask.to(dev), n_tokens, stops, pad)
        for r, i in enumerate(idx):
            out[i] = read_probe(new[r].tolist(), lps[r], tok, stops)
    return out


class CacheFork:
    """Snapshot of a cache before a probe; ``restore`` crops full-attention layers back and puts every other
    layer (linear-attention states, sliding windows) back from a deep copy."""

    def __init__(self, cache) -> None:
        from transformers.cache_utils import DynamicLayer
        self.whole = None
        layers = getattr(cache, "layers", None)
        if layers is None:
            self.whole = copy.deepcopy(cache)
            return
        self.lengths: dict[int, int] = {}
        self.saved: dict[int, object] = {}
        for i, L in enumerate(layers):
            if type(L) is DynamicLayer:
                self.lengths[i] = L.get_seq_length()
            else:
                self.saved[i] = copy.deepcopy(L)

    def restore(self, cache):
        if self.whole is not None:
            return self.whole
        for i, n0 in self.lengths.items():
            L = cache.layers[i]
            extra = L.get_seq_length() - n0
            if extra > 0:
                L.crop(-extra)
        for i, L in self.saved.items():
            cache.layers[i] = L
        return cache


@torch.no_grad()
def forked_probe(model, tok, cache, seq: torch.Tensor, mask: torch.Tensor, cue: Sequence[int], n_tokens: int,
                 stops: Sequence[int], pad: int):
    """Forced-answer probe of every row on the open cache (which holds all of ``seq`` but its last column);
    returns (results, cache) with the cache restored to its state before the call."""
    B = seq.shape[0]
    cue_t = torch.as_tensor(list(cue), dtype=seq.dtype, device=seq.device).expand(B, -1)
    ids = torch.cat([seq, cue_t], 1)
    m = torch.cat([mask, torch.ones(B, cue_t.shape[1], dtype=mask.dtype, device=mask.device)], 1)
    fork = CacheFork(cache)
    try:
        new, lps = _greedy(model, ids, m, n_tokens, stops, pad, cache=cache)
    finally:
        cache = fork.restore(cache)
    return [read_probe(new[b].tolist(), lps[b], tok, stops) for b in range(B)], cache


# ------------------------------------------------------------------------------------------- segmented
@dataclass
class SegmentedOutput:
    new: torch.Tensor                                        # (B, n) generated ids (pads after a stop)
    probes: list[dict] = field(default_factory=list)          # per probe: t, n_span (B,), open (B,), results
    seconds_gen: float = 0.0
    seconds_probe: float = 0.0
    calls: int = 0
    seq: torch.Tensor | None = None                          # prompt + generated, the mask and the open cache
    mask: torch.Tensor | None = None
    cache: object = None


@torch.no_grad()
def generate_segmented(model, ids: torch.Tensor, mask: torch.Tensor, max_new_tokens: int, every: int,
                       gen_kwargs: dict, processors: LogitsProcessorList, stops: Sequence[int], pad: int,
                       probe: Callable | None = None, span: SpanTracker | None = None,
                       on_probe: Callable | None = None, sync: Callable[[], None] | None = None) -> SegmentedOutput:
    """``generate`` E = ``every`` tokens at a time on one cache; after each segment (while rows remain and
    tokens remain) ``probe(cache, seq, mask) -> (results, cache)`` runs on rows whose span is open, and
    ``on_probe(record)`` sees each probe (e.g. ``ProbeRule.observe``). ``processors`` should include
    ``FinishedRows`` (``build_processors(finished=...)``) so that finished rows are fed the pad id.
    ``gen_kwargs``: sampling arguments of ``generate`` (do_sample, temperature, top_p, top_k)."""
    import time
    sync = sync or (lambda: None)
    P = ids.shape[1]
    seq, m, cache = ids, mask, None
    st = torch.as_tensor(list(stops), device=ids.device)
    out = SegmentedOutput(new=ids[:, :0])
    n = 0
    while n < max_new_tokens:
        k = min(every, max_new_tokens - n)
        sync()
        t0 = time.time()
        res = model.generate(input_ids=seq, attention_mask=m, past_key_values=cache, max_new_tokens=k,
                             logits_processor=processors, eos_token_id=list(stops), pad_token_id=pad,
                             return_dict_in_generate=True, **gen_kwargs)
        sync()
        out.seconds_gen += time.time() - t0
        out.calls += 1
        cache = res.past_key_values
        got = res.sequences.shape[1] - seq.shape[1]
        seq = res.sequences
        m = torch.cat([m, torch.ones(m.shape[0], got, dtype=m.dtype, device=m.device)], 1)
        n += got
        done = torch.isin(seq[:, P:], st).any(1)
        if got < k or bool(done.all()) or n >= max_new_tokens:
            break
        if probe is not None:
            open_, n_span = span.state(seq) if span is not None else (~done, torch.full_like(done, n).long())
            open_ = open_ & ~done
            if bool(open_.any()):
                t1 = time.time()
                results, cache = probe(cache, seq, m)
                sync()
                out.seconds_probe += time.time() - t1
                rec = {"t": n, "n_span": n_span.cpu().tolist(), "open": open_.cpu().tolist(), "results": results}
                out.probes.append(rec)
                if on_probe is not None:
                    on_probe(rec)
    out.new, out.seq, out.mask, out.cache = seq[:, P:], seq, m, cache
    return out
