"""Round 13 step 0: the shared span builder and the selectors' scores (plan §2.3, §3, §4 step 0).

Span builder (every selector uses it, so selectors differ only in their scores)
  ``TokenTables``  per vocabulary id: does the token end a sentence (its text, without trailing quotes or brackets,
                   ends in . ! ?), is it only closing quotes / brackets, does it hold a newline, does it start with
                   whitespace.
  ``cut_spans``    candidate spans over positions [lo, hi): a span ends after a sentence-ending token (or a closing
                   quote right after one) whose next token starts with whitespace (so "3 @.@ 5" and "U.S." inside a
                   word do not cut), or after a newline token; spans longer than ``cap`` (48) are split into pieces of at most ``cap`` tokens.
  ``span_scores``  a selector's per-position scores → span score = mean of the span's top three positions (all of
                   them when the span is shorter); positions scored −inf are ignored.
  ``take``         spans in score order (ties broken by a fixed per-item random order) are taken while they fit;
                   the first span that does not fit ends the selection (a strict prefix, so recall@budget is a
                   function of the rank and the lengths only).
  ``rank_of``      1-based rank of a span in that order.

Scores (per position, −inf where not a candidate; the caller passes the tensors one forward pass captured)
  ``qk_scores``        the model's own pre-softmax q·k at its attention layers, queries = the question's rows only.
                       Family members (``QK_VARIANTS``): ``pos`` nope (no rotary: both un-rotated), rope (the true
                       positions, distance t − p) or ord (every key at the ordinal read-back distance D, as read-back
                       would place it); ``agg`` mean (mean over layers, heads, rows of q·k/√d), max (max over rows),
                       soft (softmax over candidate positions per layer, head, row, then summed: the attention mass a
                       full-context layer would give); ``layers`` all or late (the later half of the attention layers).
  ``infllm_scores``    repaired InfLLM: in every span, the ``n_rep`` representatives are the positions that received
                       the most attention from the whole later context (``attention_received``, the model's own
                       windowed attention, mean over layers); score of a representative = mean over layers, heads and
                       question rows of q̃·k̃ / (√d ‖k̃‖) (un-rotated, norm-corrected: invariant to rescaling a key);
                       other positions −inf.
  ``cos_scores``       cosine of centred early-tap states, question rows against positions; max (or mean) over rows.
  ``dentate_scores``   cosine of the dentate codes: the question's pooled key against each position's code.
  ``store_scores``     the store's recall per span (pooled key overlap + λ · context-tag cosine), on its positions.
  ``retriever``        ``Retriever``: a sentence embedder on span text against the question text (bge-small with its
                       query instruction; Qwen3-Embedding-4B as an upper reference), cosine per span, offline only.
  ``with_salience``    score × write-time salience, as the product of within-item percentile ranks over candidate
                       positions (scale-free and safe for scores of either sign; −inf off the candidates).
  row sets             qk and infllm read the question's rows (``""``), the template's foot (``.foot``) or both
                       (``.qfoot``); ``qk_layer`` / ``qk_head`` members score one attention layer / one query head
                       (``QK_UNIT`` aggregation, rows = qfoot).
  ``position``         reference: the token position (pure recency).
  ``rank_many``        batched ranking: ranks and the columns taken by the spans ranked above each target.
  references           ``oracle_queried`` (the queried span first), ``oracle_all`` (every statement span first, in a
                       random order), ``random`` (a random score per span).
  ``attention_received``  per key position, the attention mass it receives from later rows in the model's own
                       sliding-window attention, summed over heads, divided by the number of rows that can see it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

CAP = 48
TOP = 3
QUOTES = "\"'”’»)]}"


# ------------------------------------------------------------------------------------------------ span builder
class TokenTables:
    """Per-id flags: ``end`` (sentence-ending token), ``space`` (starts with whitespace), ``newline``."""

    def __init__(self, tok, vocab_size: int | None = None) -> None:
        hf = getattr(tok, "tok", None)
        if hf is not None:
            n = vocab_size or len(hf)
            pieces = hf.convert_ids_to_tokens(list(range(n)))
            texts = [(p or "").replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\n") for p in pieces]
            texts = ["\n" if p == "<0x0A>" else t for p, t in zip(pieces, texts)]
        else:
            n = vocab_size or getattr(tok, "vocab_size", 257)
            texts = [tok.decode([i]) for i in range(n)]
        self.n = n
        end, space, nl, close = [], [], [], []
        for t in texts:
            core = t.strip().rstrip(QUOTES)
            nl.append("\n" in t)
            end.append(bool(core) and core[-1] in ".!?")
            space.append(bool(t) and t[0].isspace())
            close.append(bool(t.strip()) and all(c in QUOTES for c in t.strip()))
        self.end = torch.tensor(end)
        self.close = torch.tensor(close)
        self.space = torch.tensor(space)
        self.newline = torch.tensor(nl)


def cut_spans(ids, tables: TokenTables, lo: int, hi: int, cap: int = CAP) -> list[tuple[int, int]]:
    """Candidate spans [(a, b)] (b exclusive) covering positions lo … hi − 1 (module docstring)."""
    ids = list(ids)
    spans, a = [], lo
    for p in range(lo, hi):
        t = ids[p]
        nxt = ids[p + 1] if p + 1 < len(ids) else None
        ends = bool(tables.end[t]) or (bool(tables.close[t]) and p > 0 and bool(tables.end[ids[p - 1]]))
        cut = bool(tables.newline[t]) or (ends and (nxt is None or bool(tables.space[nxt]) or bool(tables.newline[nxt])))
        if cut or p == hi - 1:
            spans.append((a, p + 1))
            a = p + 1
    out = []
    for a, b in spans:
        while b - a > cap:
            out.append((a, a + cap))
            a += cap
        if b > a:
            out.append((a, b))
    return out


def span_index(spans: list[tuple[int, int]], cap: int, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """(n, cap) positions (0-padded) and a validity mask."""
    n = len(spans)
    pos = torch.zeros(n, cap, dtype=torch.long)
    ok = torch.zeros(n, cap, dtype=torch.bool)
    for j, (a, b) in enumerate(spans):
        pos[j, :b - a] = torch.arange(a, b)
        ok[j, :b - a] = True
    return pos.to(device), ok.to(device)


def span_scores(pos_scores: torch.Tensor, spans: list[tuple[int, int]], cap: int = CAP, top: int = TOP,
                index: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    """Mean of each span's top ``top`` finite position scores (−inf if none). pos_scores (T,)."""
    pos, ok = index if index is not None else span_index(spans, cap, pos_scores.device)
    v = pos_scores.float()[..., pos]                                # (..., n, cap): batched over leading dims
    ok = ok & torch.isfinite(v)
    v = v.masked_fill(~ok, float("-inf"))
    k = min(top, v.shape[1])
    tv = v.topk(k, -1).values
    fin = torch.isfinite(tv)
    s = torch.where(fin, tv, torch.zeros_like(tv)).sum(-1) / fin.sum(-1).clamp_min(1)
    return torch.where(fin.any(-1), s, torch.full_like(s, float("-inf")))


def order(scores: torch.Tensor, tiebreak: torch.Tensor) -> torch.Tensor:
    """Span indices by descending score; ties (and −inf) by the fixed random order ``tiebreak`` (a permutation)."""
    s = scores.detach().double().cpu()
    tb = tiebreak.cpu()
    keys = sorted(range(len(s)), key=lambda j: (-(s[j].item() if math.isfinite(s[j].item()) else -1e300), int(tb[j])))
    return torch.tensor(keys, dtype=torch.long)


def take(ordered: torch.Tensor, lengths: list[int], budget: int) -> list[int]:
    """Strict prefix of ``ordered`` whose total length fits in ``budget``."""
    out, used = [], 0
    for j in ordered.tolist():
        if used + lengths[j] > budget:
            break
        out.append(j)
        used += lengths[j]
    return out


def rank_of(ordered: torch.Tensor, j: int) -> int:
    return int((ordered == j).nonzero()[0, 0]) + 1


@dataclass
class Ranked:
    rank: int
    in_budget: bool
    rr: float


def evaluate(scores: torch.Tensor, spans: list[tuple[int, int]], targets: list[int], budget: int,
             tiebreak: torch.Tensor) -> list[Ranked]:
    od = order(scores, tiebreak)
    sel = set(take(od, [b - a for a, b in spans], budget))
    out = []
    for t in targets:
        r = rank_of(od, t)
        out.append(Ranked(r, t in sel, 1.0 / r))
    return out


# ------------------------------------------------------------------------------------------------ salience
def percentile(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Within-mask percentile ranks in (0, 1] (ties averaged); −inf outside the mask or where x is −inf."""
    out = torch.full_like(x, float("-inf"), dtype=torch.float32)
    ok = mask & torch.isfinite(x)
    v = x[ok].double()
    if v.numel() == 0:
        return out
    srt, inv = torch.unique(v, sorted=True, return_inverse=True)
    cnt = torch.bincount(inv, minlength=srt.numel()).double()
    upto = cnt.cumsum(0)
    mid = (upto - (cnt - 1) / 2) / v.numel()
    out[ok] = mid[inv].float()
    return out


def pct_product(p_score: torch.Tensor, p_sal: torch.Tensor) -> torch.Tensor:
    """Product of two percentile maps; −inf wherever either is −inf (outside the mask)."""
    ok = torch.isfinite(p_score) & torch.isfinite(p_sal)
    return torch.where(ok, p_score * p_sal, torch.full_like(p_score, float("-inf")))


def with_salience(pos_scores: torch.Tensor, salience: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """score × salience as the product of within-item percentile ranks over the candidate positions (−inf outside
    the mask: the step-0 version gave −inf × −inf = +inf there)."""
    return pct_product(percentile(pos_scores, mask), percentile(salience, mask))


def position_scores(cand: torch.Tensor) -> torch.Tensor:
    """Reference ``position``: the token position itself (pure recency), −inf off the candidates."""
    return torch.arange(cand.shape[0], device=cand.device, dtype=torch.float32).masked_fill(~cand, float("-inf"))


def rank_many(span_sc: torch.Tensor, lengths: torch.Tensor, targets: list[int],
              tiebreak: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched ``order``: span_sc (m, n) → for every scorer and target, the 1-based rank and the columns taken by the
    spans ranked above it (descending score; ties and −inf/NaN by the fixed random order ``tiebreak``).
    Returns ranks (m, len(targets)) and cols_before (m, len(targets)), both long."""
    s = span_sc.detach().double().cpu()
    s = torch.where(torch.isfinite(s), s, torch.full_like(s, -1e300))
    tb = tiebreak.cpu()
    m, n = s.shape
    o1 = torch.argsort(tb)                                          # tiebreak order first ...
    o2 = torch.argsort(-s[:, o1], dim=1, stable=True)               # ... then a stable sort by score
    od = o1[o2]                                                     # (m, n) span indices, best first
    L = lengths.cpu().long()[od]
    cum = torch.cumsum(L, 1) - L                                    # columns before each place
    ranks, cols = [], []
    for t in targets:
        place = (od == t).long().argmax(1)
        ranks.append(place + 1)
        cols.append(cum.gather(1, place[:, None])[:, 0])
    return torch.stack(ranks, 1), torch.stack(cols, 1)


def broadcast(span_vals: torch.Tensor, spans: list[tuple[int, int]], T: int) -> torch.Tensor:
    """Span-level scores → per-position scores (each position gets its span's score; −inf elsewhere)."""
    out = torch.full((T,), float("-inf"), device=span_vals.device)
    for j, (a, b) in enumerate(spans):
        out[a:b] = span_vals[j]
    return out


# ------------------------------------------------------------------------------------------------ rotary
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


def rope(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Rotate the rotary part of x (..., N, d) by positions pos (N,) or broadcastable (float32)."""
    rd = 2 * inv_freq.numel()
    f = pos.float()[..., None] * inv_freq.float().to(x.device)
    emb = torch.cat((f, f), -1)
    xr, xp = x[..., :rd].float(), x[..., rd:].float()
    return torch.cat((xr * emb.cos() + _rotate_half(xr) * emb.sin(), xp), -1)


# -------------------------------------------------------------------------------------------- qk family
QK_VARIANTS = {
    "qk.nope.mean": ("nope", "mean", "all"),
    "qk.nope.max": ("nope", "max", "all"),
    "qk.nope.soft": ("nope", "soft", "all"),
    "qk.nope.soft.late": ("nope", "soft", "late"),
    "qk.rope.soft": ("rope", "soft", "all"),
    "qk.ord.mean": ("ord", "mean", "all"),
    "qk.ord.soft": ("ord", "soft", "all"),
}
# row sets: "" the question's own tokens (step 0), ".foot" the template's foot only ("\nAnswer:" or the chat foot;
# never the answer prefix, so both prefix halves rank the same), ".qfoot" question + foot
ROW_SETS = ("", ".foot", ".qfoot")
QK_UNIT = ("nope", "max", "all")                                    # qk_layer / qk_head members (step-0 smoke's best)


def qk_scores(q: list[torch.Tensor], k: list[torch.Tensor], q_pos: torch.Tensor, cand: torch.Tensor,
              inv_freq: torch.Tensor | None, variant: tuple[str, str, str], ord_delta: int = 0) -> torch.Tensor:
    """q[l] (H, R, d) pre-rotary question rows, k[l] (T, d) pre-rotary keys (one KV head), q_pos (R,) their
    positions, cand (T,) bool candidate positions. Returns per-position scores (T,), −inf off the candidates."""
    pos_mode, agg, layers = variant
    L = len(q)
    use = list(range(L)) if layers == "all" else list(range(L // 2, L))
    T = k[0].shape[0]
    tot = torch.zeros(T, device=k[0].device)
    for li in use:
        ql, kl = q[li].float(), k[li].float()
        d = ql.shape[-1]
        if pos_mode == "rope":
            ql = rope(ql, q_pos, inv_freq)
            kl = rope(kl, torch.arange(T, device=kl.device), inv_freq)
        elif pos_mode == "ord":
            kl = rope(kl, torch.full((T,), -int(ord_delta), device=kl.device), inv_freq)
        s = torch.einsum("hrd,td->hrt", ql, kl) / math.sqrt(d)            # (H, R, T)
        if agg == "mean":
            v = s.mean((0, 1))
        elif agg == "max":
            v = s.amax(1).mean(0)
        elif agg == "soft":
            v = torch.softmax(s.masked_fill(~cand, float("-inf")), -1).sum((0, 1))
        else:
            raise ValueError(agg)
        tot += v
    tot = tot / len(use)
    return tot.masked_fill(~cand, float("-inf"))


# ---------------------------------------------------------------------------------------- attention received
def attention_received(q: torch.Tensor, k: torch.Tensor, window: int, inv_freq: torch.Tensor | None,
                       scaling: float | None = None, chunk: int = 1024) -> torch.Tensor:
    """q (H, T, d), k (T, d) pre-rotary (one KV head). The model's sliding-window causal attention (key p visible to
    row t when t − window < p ≤ t); returns, per key, the mass received from all rows summed over heads, divided by
    the number of rows that see it (float32)."""
    H, T, d = q.shape
    sc = scaling if scaling is not None else d ** -0.5
    dev = q.device
    pos = torch.arange(T, device=dev)
    qr = rope(q.float(), pos, inv_freq) if inv_freq is not None else q.float()
    kr = rope(k.float(), pos, inv_freq) if inv_freq is not None else k.float()
    recv = torch.zeros(T, device=dev)
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        lo = max(0, t0 - window + 1)
        s = torch.einsum("htd,pd->htp", qr[:, t0:t1], kr[lo:t1]) * sc
        rows, cols = pos[t0:t1, None], pos[None, lo:t1]
        vis = (cols <= rows) & (cols > rows - window)
        p = torch.softmax(s.masked_fill(~vis, float("-inf")), -1)
        recv[lo:t1] += p.sum((0, 1))
    seen = (torch.clamp(pos + window, max=T) - pos).float()
    return recv / seen


# ---------------------------------------------------------------------------------------------- infllm
def infllm_scores(q: list[torch.Tensor], k: list[torch.Tensor], received: torch.Tensor,
                  spans: list[tuple[int, int]], n_rep: int = TOP) -> torch.Tensor:
    """Repaired InfLLM (module docstring). q[l] (H, R, d), k[l] (T, d) pre-rotary; received (T,)."""
    T = k[0].shape[0]
    rep = torch.zeros(T, dtype=torch.bool, device=received.device)
    for a, b in spans:
        r = received[a:b]
        idx = r.topk(min(n_rep, b - a)).indices + a
        rep[idx] = True
    tot = torch.zeros(T, device=k[0].device)
    for ql, kl in zip(q, k):
        ql, kl = ql.float(), kl.float()
        d = ql.shape[-1]
        s = torch.einsum("hrd,td->hrt", ql, kl).mean((0, 1)) / (math.sqrt(d) * kl.norm(dim=-1).clamp_min(1e-12))
        tot += s
    tot = tot / len(q)
    return tot.masked_fill(~rep, float("-inf"))


# ------------------------------------------------------------------------------------------ cos, dentate
def cos_scores(early: torch.Tensor, q_rows: torch.Tensor, cand: torch.Tensor, centre: torch.Tensor,
               agg: str = "max") -> torch.Tensor:
    """early (T, D); q_rows (R,) long; centre (D,) the document mean of the early states."""
    x = F.normalize(early.float() - centre, dim=-1)
    s = x[q_rows] @ x.T                                                     # (R, T)
    v = s.amax(0) if agg == "max" else s.mean(0)
    return v.masked_fill(~cand, float("-inf"))


def dentate_scores(idx: torch.Tensor, val: torch.Tensor, q_rows: torch.Tensor, cand: torch.Tensor,
                   U: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position code overlap with the question's pooled key. Returns (scores (T,), the key (U,))."""
    from ca2.circuit import code_overlap
    key = torch.zeros(U, device=idx.device)
    key.index_put_((idx[q_rows].reshape(-1),), val[q_rows].reshape(-1), accumulate=True)
    key = key / len(q_rows)
    return code_overlap(key, idx, val).masked_fill(~cand, float("-inf")), key


def store_scores(store, key: torch.Tensor, ctx: torch.Tensor, lam: float, spans: list[tuple[int, int]],
                 T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The store's recall per span (entries = ``spans``, in order) broadcast to positions; also the span scores."""
    s = store.recall(key, ctx, lam)
    return broadcast(s, spans, T), s


# ---------------------------------------------------------------------------------------------- references
def oracle_scores(spans: list[tuple[int, int]], targets: list[int], T: int, device="cpu") -> torch.Tensor:
    v = torch.zeros(len(spans), device=device)
    v[targets] = 1.0
    return broadcast(v, spans, T)


def random_scores(spans: list[tuple[int, int]], T: int, seed: int, device="cpu") -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    return broadcast(torch.rand(len(spans), generator=g).to(device), spans, T)


# ---------------------------------------------------------------------------------------------- retriever
BGE = "unsloth/bge-small-en-v1.5"
QWEN_EMB = "unsloth/Qwen3-Embedding-4B"
BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "
QWEN_INSTRUCTION = "Instruct: Given a question, retrieve the sentence that answers it\nQuery:"


class Retriever:
    """Sentence embedder, offline (``local_files_only``): bge-small (CLS pooling) or Qwen3-Embedding (last token)."""

    def __init__(self, name: str = BGE, device="cpu", dtype=torch.float32, embed_fn=None) -> None:
        self.name = name
        self.embed_fn = embed_fn                                    # tests: a stand-in (texts, is_query) → (n, d)
        if embed_fn is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.qwen = "qwen3-embedding" in name.lower()
        self.tok = AutoTokenizer.from_pretrained(name, local_files_only=True,
                                                 padding_side="left" if self.qwen else "right")
        self.model = AutoModel.from_pretrained(name, local_files_only=True, dtype=dtype).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, texts: list[str], query: bool, batch: int = 64) -> torch.Tensor:
        if self.embed_fn is not None:
            return F.normalize(self.embed_fn(texts, query).float(), dim=-1)
        if query:
            texts = [(QWEN_INSTRUCTION if self.qwen else BGE_INSTRUCTION) + t for t in texts]
        outs = []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            v = h[:, -1] if self.qwen else h[:, 0]
            outs.append(F.normalize(v.float(), dim=-1))
        return torch.cat(outs, 0)

    def score(self, question: str, span_texts: list[str]) -> torch.Tensor:
        q = self.embed([question], True)
        d = self.embed(span_texts, False)
        return (d @ q[0]).cpu()
