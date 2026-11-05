"""Gate G0.8 (round 11, ``research/plans/round11_runbook.md`` §4): does time *as data* in the W1s slot read keep
content selection and solve order and update queries, where a time code concatenated into the key failed (0/10)?

A CPU toy, ported from the round-10 merit toy (``$DATA/scratch/r10_merit/b_hysteresis/toy.py`` and its seed-batched
``toy2.py``; NOTEBOOK "Round-10 merit tests"). It keeps that toy's world, filler and statement grammar, and its
seed-batched training (parameters with a leading seed dimension, the sum of per-seed mean losses, elementwise Adam,
per-seed gradient clipping to norm 1, so seeds are independent runs), but replaces its superposed delta-rule store
with W1s's episodic slots, because round 10 showed the slots are the memory that works (P49–P58).

World (fixed, like a frozen LM): random token embeddings, d = 64, with LLM-like anisotropy (a shared direction of
weight 3, so random tokens have cosine ≈ 0.9) and a token-class direction. Vocabulary as the source (200 filler, 6
template, "is", ".", 10 digits, 20 names), plus two order words FIRST and LAST. Stream of T tokens (192 in training):
filler (10% contaminated by template / "is" / "." / digit tokens), then, scattered,

    3 facts      [PREFIX(2–5)] NAME is D1 D2 D3 D4 .        three distinct names
    restatement  [PREFIX(2–5)] NAME is D1' D2' D3' D4' .     one of the names again, later, with a new code
                                                              (always for an update query, else with prob. 0.5)
    query        [PREFIX(2–5)] NAME is   or   [PREFIX(2–5)] FIRST|LAST is      → the answer is a first digit

The statements' first digits are distinct, so an answer names one statement. Query types:
    plain   NAME of a fact that is not restated                → its D1
    update  NAME of the restated fact                          → the newest D1; answering the old D1 is *stale*
    order   FIRST / LAST (half each)                           → D1 of the first / last statement in the stream
Only an answer loss at the last position (as the source). Training mixes plain/update/order 0.4/0.3/0.3.

Store (shared by every arm; W1s with oracle onsets, ``--w1s-gate oracle``): an onset s at each statement's D1, plus
``--n-false`` false onsets in filler (2 by default); slot k keeps the window x_{s−P} … x_{s−P+n−1} (P = 4, n = 8:
"… NAME is D1 D2 D3 D4"); K = 8 slots hold the newest closed windows, so nothing is evicted at the default sizes and
the read is what is tested. Read at the last position t (the query's "is"):

    key      k_k = W_k k̄_k,  k̄_k = mean of the window's first ``pool`` (4) entries;  query q_t = W_k q̄_t,
             q̄_t = mean of the last 4 tokens (one map for keys and queries, as W1s's learned key)
    select   a_t = softmax over the retained slots + a null (logit b_∅) of  ℓ_k = β_L·cos(q_t, k_k) + time_k
    align    inside slot k, α_kj ∝ exp(β_a (cos(z_t, z_j) − 1)), z = x − μ (μ: the mean of the retained windows'
             entries, fit-free, as W1s's centred lexical alignment); the value read is v_{j+1} = W_v x_{j+1}
    output   logits = W_o Σ_k a_k Σ_j α_kj v_{j+1} + b_o  (10 digits)

Arms (the time term; age_k = t − s_k, rank r_k = #{retained slots newer than k}, 0 = newest):

    none      time_k = 0                                        (control: content only)
    linear    time_k = −ω·age_k,  ω = exp(log ω) from 0.002      (round 10's I1 recency penalty, the reference)
    concat    an absolute-time code concatenated into the slot key (a reference arm; NOT a port of the merit
              arm that failed 0/10, which multiplied a time-since-onset code into the key of a superposed
              delta-rule store, from an unfinished run; review F10):
              k_k ← [k̂_k ; φ(s_k)],  q_t ← [q̂_t ; W_t q̄_t],  ℓ_k = β_L·cos(q_t, k_k);  φ(s) = sinusoids of the
              absolute onset at the rope periods, unit norm, so the query must ask for an absolute time. W_t starts
              at 0 (``--concat-init zero``, the most favourable start: the read begins as ``none``); with the
              source's init (``kaiming``) the time half swamps the content cosine at the start
    rope      partial rotary time: the first ``rope_dims`` dims of k_k turn by ω_f·s_k and of q_t by ω_f·t (pairs
              (2f, 2f+1), periods log-spaced over ``--rope-periods``), so cos(q_t, k_k) depends on content and t − s
              jointly; norms and the other dims are untouched
    ordinal   time_k = u_t·E[r_k],  u_t = W_u q̄_t,  E ∈ ℝ^{K×16} learned (from 0): the query chooses the rank it
              wants; bounded by K, so the same at any length
    supr      content-gated read supersession, no age term: a¹ = softmax(ℓ⁰ + null) with ℓ⁰ the content logits,
              m_k = Σ_{k′ newer than k} a¹_k′,  time_k = −w_s·m_k,  w_s learned from 3.0

The toy's raw tokens carry no context, so the query-side vectors of ``concat`` and ``ordinal`` (W1s: W·x_t at the
read position) read the pooled query q̄_t, which is where FIRST / LAST / NAME are.

Learned (as the source toy): final plain-query accuracy ≥ 0.8 at the training length. Every run is also evaluated
at 2× and 4× the training length (longer filler gaps, the same statements). Pre-registered (runbook G0.8): ``rope``
or ``ordinal`` passes if it learns ≥ 8/10 seeds and its order-query accuracy beats ``linear``'s (seed means, at the
training length); a time-data form whose plain-query accuracy is < 0.9 (seed mean) is dropped.

    python experiments/evolve/toy_time.py --seeds 10 --out results/evolve/round11/stage0/g08_toy_time.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------- world (the source toy's, plus 2 order words)
D = 64
NF, NT, NN, ND = 200, 6, 20, 10
TPL0 = NF
MARK = NF + NT                                                    # "is"
PER = MARK + 1                                                    # "."
DIG0 = PER + 1
NAME0 = DIG0 + ND
FIRST, LAST = NAME0 + NN, NAME0 + NN + 1
V = NAME0 + NN + 2
ANISO, CLS = 3.0, 1.0
ARMS = ("concat", "rope", "ordinal", "supr", "linear", "none")
QTYPES = ("plain", "update", "order")


def tok_class(i: int) -> int:
    return (0 if i < NF else 1 if i < MARK else 2 if i == MARK else 3 if i == PER else 4 if i < NAME0
            else 5 if i < FIRST else 6)


def world(seed: int = 1234, regime: str = "raw") -> tuple[torch.Tensor, list]:
    """Embeddings E (V, D) and the frozen local mixer M (ctx regime). The source's draws come first, so the source
    tokens have the source's embeddings; the order words' rows (and their class direction) are drawn after."""
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(D, generator=g)
    cls = torch.randn(6, D, generator=g)
    own = torch.randn(V - 2, D, generator=g)
    M = [torch.randn(D, D, generator=g) / math.sqrt(D) for _ in range(4)]
    own = torch.cat([own, torch.randn(2, D, generator=g)])
    cls = torch.cat([cls, torch.randn(1, D, generator=g)])
    E = own + CLS * cls[torch.tensor([tok_class(i) for i in range(V)])] + ANISO * m
    return E, (M if regime == "ctx" else [])


def feats(toks: torch.Tensor, pos: torch.Tensor, E: torch.Tensor, M: list) -> torch.Tensor:
    """x at positions ``pos`` (..., P) of ``toks`` (..., T) → (..., P, D). raw: x = E[tok]; ctx (the source's frozen
    mixer): h_p = x_p + Σ_{j=1..4} 0.6^j M_j x_{p−j} (zero before the start). Only the needed positions are built."""
    x = E[toks.gather(-1, pos)]
    for j, Mj in enumerate(M, 1):
        pj = pos - j
        xj = E[toks.gather(-1, pj.clamp_min(0))] * (pj >= 0)[..., None]
        x = x + (0.6 ** j) * xj @ Mj.T
    return x


# ----------------------------------------------------------------------- task
def make_seq(rng: random.Random, T: int, qtype: int, c: argparse.Namespace) -> dict:
    """One stream (see the module docstring). Onsets (the statements' D1 positions and the false onsets) are sorted in
    time order; ``tgt`` / ``stale`` index the target / superseded statement's slot (−1 if none)."""
    names = rng.sample(range(NN), 3)
    restate = qtype == 1 or rng.random() < c.p_restate
    u = rng.randrange(3)                                          # the restated fact
    order = [0, 1, 2]
    if restate:
        order.insert(rng.randint(u + 1, 3), 3)
    nm = {0: names[0], 1: names[1], 2: names[2], 3: names[u]}
    ns = len(order)
    d1 = rng.sample(range(ND), ns)                                # distinct first digits, by statement (time order)
    stmts = []
    for i, f in enumerate(order):
        pre = [TPL0 + rng.randrange(NT) for _ in range(rng.randint(2, 5))]
        digs = [DIG0 + d1[i]] + [DIG0 + rng.randrange(ND) for _ in range(3)]
        stmts.append((pre + [NAME0 + nm[f], MARK] + digs + [PER], len(pre) + 2))
    stale_i, osub = -1, -1
    if qtype == 0:
        f = rng.choice([i for i in range(3) if not (restate and i == u)])
        tgt_i, word = order.index(f), NAME0 + nm[f]
    elif qtype == 1:
        tgt_i, stale_i, word = order.index(3), order.index(u), NAME0 + nm[u]
    else:
        osub = rng.randrange(2)
        tgt_i, word = (0, FIRST) if osub == 0 else (ns - 1, LAST)
    query = [TPL0 + rng.randrange(NT) for _ in range(rng.randint(2, 5))] + [word, MARK]
    nfill = T - sum(len(s) for s, _ in stmts) - len(query)
    if nfill < (c.n_false + 1) * c.win:
        raise ValueError(f"T={T} too short for the statements and {c.n_false} false windows")
    cuts = sorted(rng.randint(0, nfill) for _ in range(ns))
    gaps = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, ns)] + [nfill - cuts[-1]]
    fill = [rng.randrange(NF) if rng.random() >= c.contam else TPL0 + rng.randrange(NT + 2 + ND) for _ in range(nfill)]
    seq, isfill, st_on, fi = [], [], [], 0
    for i in range(ns):
        seq += fill[fi:fi + gaps[i]]; isfill += [True] * gaps[i]; fi += gaps[i]
        st_on.append(len(seq) + stmts[i][1])
        seq += stmts[i][0]; isfill += [False] * len(stmts[i][0])
    seq += fill[fi:]; isfill += [True] * (nfill - fi)
    q0 = len(seq)
    seq += query
    assert len(seq) == T
    # false onsets: a whole window of filler before the query, apart from every other window by ≥ n
    cand = [s for s in range(c.pre, q0 - c.win + c.pre + 1) if all(isfill[s - c.pre:s - c.pre + c.win])]
    rng.shuffle(cand)
    fal = []
    for s in cand:
        if len(fal) == c.n_false:
            break
        if all(abs(s - o) >= c.win for o in st_on + fal):
            fal.append(s)
    ons = sorted(st_on + fal)
    ons += [-1] * (3 + 1 + c.n_false - len(ons))
    return dict(toks=seq, ons=ons, qtype=qtype, osub=osub, ans=d1[tgt_i], tgt=ons.index(st_on[tgt_i]),
                stale=d1[stale_i] if stale_i >= 0 else -1, stale_slot=ons.index(st_on[stale_i]) if stale_i >= 0 else -1)


def make_bank(n: int, seed: int, T: int, c: argparse.Namespace, qtype: int | None = None) -> dict:
    """n streams as tensors; query types drawn from ``c.mix`` unless ``qtype`` is fixed."""
    rng = random.Random(seed)
    rows = [make_seq(rng, T, qtype if qtype is not None else rng.choices(range(3), weights=c.mix)[0], c)
            for _ in range(n)]
    return {k: torch.tensor([r[k] for r in rows]) for k in rows[0]}


# ----------------------------------------------------------------------- model (seed-batched)
def kaiming(S: int, out: int, inp: int, seeds: list, salt: int) -> torch.Tensor:
    """The source's per-seed init: U(±1/√inp) from generator seed·1000 + salt."""
    w = torch.empty(S, out, inp)
    for i, s in enumerate(seeds):
        w[i].uniform_(-1 / math.sqrt(inp), 1 / math.sqrt(inp), generator=torch.Generator().manual_seed(s * 1000 + salt))
    return w


def rope_freqs(dims: int, periods: str) -> torch.Tensor:
    """Angular frequencies of the rotated pairs, periods log-spaced from the shortest to the longest (as W1s)."""
    lo, hi = (float(x) for x in periods.split(","))
    if dims % 2 or dims < 2:
        raise ValueError("--rope-dims must be even and ≥ 2")
    return (2 * math.pi / torch.logspace(math.log10(lo), math.log10(hi), dims // 2, dtype=torch.float64)).float()


def rotate(x: torch.Tensor, pos: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Pairs (2f, 2f+1) of the first 2·len(freqs) dims of x (..., d) turn by pos·ω_f (pos broadcast to x[..., 0])."""
    R = 2 * len(freqs)
    ang = pos.to(torch.float64)[..., None] * freqs.double()
    c, s = ang.cos().float(), ang.sin().float()
    x0, x1 = x[..., 0:R:2], x[..., 1:R:2]
    rot = torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).flatten(-2)
    return torch.cat([rot, x[..., R:]], -1)


def sincos(pos: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """φ(pos): [cos ω_f pos, sin ω_f pos] over the frequencies, unit norm (concat arm)."""
    ang = pos.to(torch.float64)[..., None] * freqs.double()
    return torch.stack([ang.cos(), ang.sin()], -1).flatten(-2).float() / math.sqrt(len(freqs))


class SlotToy(torch.nn.Module):
    """S independent seeds of one arm (parameters carry a leading seed dimension)."""

    def __init__(self, arm: str, seeds: list, c: argparse.Namespace) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        S, P = len(seeds), torch.nn.Parameter
        self.arm, self.S, self.c = arm, S, c
        self.wk = P(kaiming(S, c.dk, D, seeds, 1))
        self.wv = P(kaiming(S, c.dv, D, seeds, 2))
        self.wo = P(kaiming(S, ND, c.dv, seeds, 3))
        self.bo = P(torch.zeros(S, ND))
        self.log_beta = P(torch.full((S,), math.log(10.0)))      # W1s's learned-key β init
        self.log_beta_a = P(torch.full((S,), math.log(30.0)))    # W1s's lexical alignment β init
        self.null = P(torch.full((S,), 0.3 * 10.0))               # W1s's null: a cosine of 0.3
        self.register_buffer("freqs", rope_freqs(c.rope_dims, c.rope_periods))
        if arm == "linear":
            self.log_omega = P(torch.full((S,), math.log(c.recency)))
        if arm == "concat":                                       # zero: starts as the content-only read
            self.wt = P(torch.zeros(S, c.rope_dims, D) if c.concat_init == "zero"
                        else kaiming(S, c.rope_dims, D, seeds, 6))
        if arm == "ordinal":
            self.wu = P(kaiming(S, c.ord_dim, D, seeds, 7))
            self.Eo = P(torch.zeros(S, c.K, c.ord_dim))
        if arm == "supr":
            self.ws = P(torch.full((S,), c.sup_w))

    def forward(self, xw: torch.Tensor, xq: torch.Tensor, s: torch.Tensor, valid: torch.Tensor, t: int):
        """xw (S, B, N, n, D) window entries, xq (S, B, pool, D) the last tokens (xq[..., -1, :] = x_t), s (S, B, N)
        onsets, valid (S, B, N), t the read position. Returns logits (S, B, ND) and the read weights (S, B, N + 1)."""
        c = self.c
        S, B, N = s.shape
        newer = valid[..., None, :] & (s[..., None, :] > s[..., :, None])          # [k, k'] k' newer than k
        rank = newer.sum(-1)
        ret = valid & (s - c.pre + c.win - 1 <= t) & (rank < c.K)                 # closed, newest K
        kb, qb = xw[..., :c.pool, :].mean(-2), xq.mean(-2)                       # (S,B,N,D), (S,B,D)
        k = torch.einsum("sbnd,skd->sbnk", kb, self.wk)
        q = torch.einsum("sbd,skd->sbk", qb, self.wk)
        if self.arm == "rope":
            k, q = rotate(k, s, self.freqs), rotate(q, torch.full((S, B), t), self.freqs)
        if self.arm == "concat":
            k = torch.cat([F.normalize(k, dim=-1), sincos(s, self.freqs)], -1)
            q = torch.cat([F.normalize(q, dim=-1), torch.einsum("sbd,skd->sbk", qb, self.wt)], -1)
        lg = self.log_beta.exp()[:, None, None] * torch.einsum("sbk,sbnk->sbn", F.normalize(q, dim=-1),
                                                               F.normalize(k, dim=-1))
        null = self.null[:, None, None].expand(S, B, 1)
        if self.arm == "linear":
            lg = lg - self.log_omega.exp()[:, None, None] * (t - s).float()
        if self.arm == "ordinal":
            u = torch.einsum("sbd,skd->sbk", qb, self.wu)
            Er = self.Eo[torch.arange(S)[:, None, None], rank.clamp(max=c.K - 1)]  # (S,B,N,ord)
            lg = lg + torch.einsum("sbk,sbnk->sbn", u, Er)
        if self.arm == "supr":
            a1 = torch.softmax(torch.cat([lg.masked_fill(~ret, float("-inf")), null], -1), -1)[..., :N]
            m = (newer.float() * a1[..., None, :]).sum(-1)                         # read mass of newer slots
            lg = lg - self.ws[:, None, None] * m
        a = torch.softmax(torch.cat([lg.masked_fill(~ret, float("-inf")), null], -1), -1)
        # offset alignment inside each slot, centred on the retained windows' mean (fit-free)
        with torch.no_grad():
            rw = ret.float()[..., None, None]
            mu = (xw * rw).sum((2, 3)) / (rw.sum((2, 3)) * c.win).clamp_min(1)    # (S,B,D)
            zw = F.normalize(xw[..., :-1, :] - mu[:, :, None, None], dim=-1)
            zt = F.normalize(xq[..., -1, :] - mu, dim=-1)
            cs = torch.einsum("sbnjd,sbd->sbnj", zw, zt)
        E = torch.exp(self.log_beta_a.exp().clamp(max=40.0)[:, None, None, None] * (cs - 1))
        al = E / E.sum(-1, keepdim=True)
        v = torch.einsum("sbnjd,svd->sbnjv", xw[..., 1:, :], self.wv)            # the value stored at j + 1
        r = torch.einsum("sbn,sbnj,sbnjv->sbv", a[..., :N], al, v)
        return torch.einsum("sbv,sov->sbo", r, self.wo) + self.bo[:, None], a

    @torch.no_grad()
    def params_report(self) -> list:
        """Per-seed learned scalars (β's, null, ω, w_s, ordinal rank preference norms)."""
        out = []
        for i in range(self.S):
            d = dict(beta=float(self.log_beta[i].exp()), beta_align=float(self.log_beta_a[i].exp()),
                     null=float(self.null[i]))
            if self.arm == "linear":
                d["omega"] = float(self.log_omega[i].exp())
            if self.arm == "supr":
                d["w_s"] = float(self.ws[i])
            if self.arm == "ordinal":
                d["rank_emb_norm"] = self.Eo[i].norm(dim=-1).tolist()
            out.append(d)
        return out


def inputs(bk: dict, idx: torch.Tensor | None, S: int, E: torch.Tensor, M: list, c: argparse.Namespace):
    """Gather the window entries and the query tokens for streams ``idx`` (S, B) of a bank (or the whole bank,
    shared by every seed, when idx is None)."""
    if idx is None:
        tk, ons = bk["toks"][None], bk["ons"][None]
    else:
        tk, ons = bk["toks"][idx], bk["ons"][idx]
    T = tk.shape[-1]
    valid = ons >= 0
    s = torch.where(valid, ons, torch.full_like(ons, c.pre))
    wpos = (s[..., None] - c.pre + torch.arange(c.win)).clamp(0, T - 1)          # (., B, N, n)
    B, N = s.shape[-2:]
    xw = feats(tk, wpos.flatten(-2), E, M).view(*wpos.shape, D)
    qpos = torch.arange(T - c.pool, T).expand(*tk.shape[:-1], c.pool)
    xq = feats(tk, qpos, E, M)
    if idx is None:
        xw, xq, s, valid = (z.expand(S, *z.shape[1:]) for z in (xw, xq, s, valid))
    return xw, xq, s, valid, T


@torch.no_grad()
def evaluate(m: SlotToy, bk: dict, E: torch.Tensor, M: list, c: argparse.Namespace, chunk: int = 256) -> list:
    """Per-seed metrics on a bank: accuracy per query type, the update's stale rate, order first/last, and the
    selection accuracy (the target slot has the largest read weight)."""
    n = bk["toks"].shape[0]
    preds, sel = [], []
    for i in range(0, n, chunk):
        sub = {k: v[i:i + chunk] for k, v in bk.items()}
        xw, xq, s, valid, T = inputs(sub, None, m.S, E, M, c)
        lo, a = m(xw, xq, s, valid, T - 1)
        preds.append(lo.argmax(-1)); sel.append(a.argmax(-1))
    pred, sel = torch.cat(preds, 1), torch.cat(sel, 1)                              # (S, n)
    ok, sok = (pred == bk["ans"]).float(), (sel == bk["tgt"]).float()
    qt, osub = bk["qtype"], bk["osub"]
    res = []
    for j in range(m.S):
        d = {}
        for q, name in enumerate(QTYPES):
            mk = qt == q
            if mk.any():
                d[name], d["sel_" + name] = float(ok[j][mk].mean()), float(sok[j][mk].mean())
        mk = qt == 1
        if mk.any():
            d["stale"] = float((pred[j][mk] == bk["stale"][mk]).float().mean())
        for o, name in enumerate(("order_first", "order_last")):
            mk = (qt == 2) & (osub == o)
            if mk.any():
                d[name] = float(ok[j][mk].mean())
        res.append(d)
    return res


def train_arm(arm: str, c: argparse.Namespace, trb: dict, evb: dict, E: torch.Tensor, M: list, log=print) -> dict:
    """Train S seeds of one arm on the bank ``trb``; evaluate on ``evb`` ({mult: bank}) at the end."""
    seeds = list(range(c.seed0, c.seed0 + c.seeds))
    S = len(seeds)
    m = SlotToy(arm, seeds, c)
    opt = torch.optim.Adam(m.parameters(), lr=c.lr)
    gens = [torch.Generator().manual_seed(s) for s in seeds]
    base = evb[1]
    curves, learned_at = [[] for _ in range(S)], [None] * S
    t0 = time.time()
    for step in range(1, c.steps + 1):
        idx = torch.stack([torch.randint(0, trb["toks"].shape[0], (c.bs,), generator=g) for g in gens])
        xw, xq, s, valid, T = inputs(trb, idx, S, E, M, c)
        lo, _ = m(xw, xq, s, valid, T - 1)
        ce = F.cross_entropy(lo.reshape(-1, ND), trb["ans"][idx].reshape(-1), reduction="none")
        loss = ce.view(S, -1).mean(1).sum()
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():                                                     # per-seed clip to norm 1
            sq = sum(p.grad.pow(2).flatten(1).sum(1) if p.dim() > 1 else p.grad.pow(2)
                     for p in m.parameters() if p.grad is not None)
            sc = (1.0 / (sq.sqrt() + 1e-6)).clamp(max=1.0)
            for p in m.parameters():
                if p.grad is not None:
                    p.grad.mul_(sc.view(-1, *[1] * (p.dim() - 1)))
        opt.step()
        if step % c.eval_every == 0 or step == c.steps:
            st = evaluate(m, base, E, M, c)
            for j in range(S):
                curves[j].append([step] + [round(st[j].get(q, float("nan")), 4) for q in QTYPES])
                if learned_at[j] is None and st[j].get("plain", 0.0) >= 0.8:
                    learned_at[j] = step
            log(f"  {arm:8s} {step:5d} {time.time() - t0:6.0f}s  plain/update/order " +
                " ".join(f"{x.get('plain', 0):.2f}/{x.get('update', 0):.2f}/{x.get('order', 0):.2f}" for x in st))
    secs = time.time() - t0
    fin = {mult: evaluate(m, bk, E, M, c) for mult, bk in evb.items()}
    pr = m.params_report()
    runs = [dict(arm=arm, seed=seeds[j], final={f"x{mult}": fin[mult][j] for mult in fin}, curve=curves[j],
                 learned_at=learned_at[j], learned=fin[1][j].get("plain", 0.0) >= 0.8, params=pr[j])
            for j in range(S)]
    return dict(runs=runs, secs=secs, secs_per_step=secs / max(c.steps, 1),
                n_params=sum(p[0].numel() for p in m.parameters()))


def summarize(arm_res: dict, mults: list) -> dict:
    """Per arm: seeds learned, and seed means of every metric at each length."""
    out = {}
    for arm, r in arm_res.items():
        runs = r["runs"]
        d = dict(learned=sum(x["learned"] for x in runs), seeds=len(runs),
                 plain_ge_0_9=sum(x["final"]["x1"].get("plain", 0) >= 0.9 for x in runs))
        for mult in mults:
            keys = sorted({k for x in runs for k in x["final"][f"x{mult}"]})
            d[f"x{mult}"] = {k: sum(x["final"][f"x{mult}"].get(k, 0.0) for x in runs) / len(runs) for k in keys}
        out[arm] = d
    return out


def gate(summary: dict) -> dict:
    """Runbook G0.8: rope / ordinal pass with ≥ 8/10 learned (≥ 0.8 of the seeds) and order accuracy above
    linear's; any time-data form with mean plain accuracy < 0.9 is dropped.

    Two readings are reported (review of 4 Nov 2026, F9). ``literal`` applies the runbook's words at the
    training length (x1). ``strict`` applies the drop rule at every evaluated length (the DEC2 grid runs to 4×
    the training length) and asks the order query to beat both linear and the content-only ``none``. The
    ``calibration`` block says whether the toy separates the arms at all: whether the concatenated-time
    reference fails, and whether ``none`` (no time information) learns. Which reading decides is Daniel's
    call; a change to the pre-registered rule is a DEC."""
    mults = sorted(int(k[1:]) for k in next(iter(summary.values())) if k.startswith("x") and k[1:].isdigit())
    lin1 = summary.get("linear", {}).get("x1", {}).get("order")
    none1 = summary.get("none", {}).get("x1", {}).get("order")
    out = {"literal": {}, "strict": {}}
    for arm in ("rope", "ordinal", "concat", "supr", "mult"):
        if arm not in summary:
            continue
        sm = summary[arm]
        plain, order = sm["x1"].get("plain", 0.0), sm["x1"].get("order", 0.0)
        learned = sm["learned"] >= math.ceil(0.8 * sm["seeds"])
        plains = {f"x{m}": sm[f"x{m}"].get("plain", 0.0) for m in mults}
        orders = {f"x{m}": sm[f"x{m}"].get("order", 0.0) for m in mults}
        out["literal"][arm] = dict(learned=f"{sm['learned']}/{sm['seeds']}", plain=plain, order=order,
                                   linear_order=lin1, drop_plain=plain < 0.9,
                                   **{"pass": bool(learned and lin1 is not None and order > lin1 and plain >= 0.9)})
        floor = {f"x{m}": max(summary.get(a, {}).get(f"x{m}", {}).get("order", 0.0) for a in ("linear", "none"))
                 for m in mults}
        out["strict"][arm] = dict(plain_by_length=plains, order_by_length=orders, order_floor_by_length=floor,
                                  drop_plain=min(plains.values()) < 0.9,
                                  **{"pass": bool(learned and min(plains.values()) >= 0.9
                                                  and all(orders[k] > floor[k] for k in orders))})
    for reading in ("literal", "strict"):
        out[reading]["any_time_data_pass"] = any(out[reading].get(a, {}).get("pass", False) for a in ("rope", "ordinal"))
    out["calibration"] = {
        "concat_passes_literal": out["literal"].get("concat", {}).get("pass"),
        "concat_passes_strict": out["strict"].get("concat", {}).get("pass"),
        "none_learned": f"{summary['none']['learned']}/{summary['none']['seeds']}" if "none" in summary else None,
        "none_order_x1": none1,
        "note": "a reading in which the concatenated-time reference passes cannot show that time as data avoids "
                "its failure",
    }
    out["any_time_data_pass"] = out["literal"]["any_time_data_pass"]   # the runbook's words (kept for readers)
    return out


def table(summary: dict, mults: list) -> str:
    cols = ("plain", "update", "stale", "order", "order_first", "order_last")
    lines = [f"{'arm':8s} {'learned':>7s} {'len':>4s} " + " ".join(f"{k:>11s}" for k in cols)]
    for arm, sm in summary.items():
        for i, mult in enumerate(mults):
            row = sm[f"x{mult}"]
            lrn = f"{sm['learned']}/{sm['seeds']}" if i == 0 else ""
            lines.append(f"{arm if i == 0 else '':8s} {lrn:>7s} {str(mult) + 'x':>4s} "
                         + " ".join(f"{row.get(k, float('nan')):11.3f}" for k in cols))
    return "\n".join(lines)


def parse(argv: list | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default=",".join(ARMS), help="comma list of " + ", ".join(ARMS))
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds (seed0 … seed0 + seeds − 1)")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000, help="the source toy's default")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--T", type=int, default=192, help="training length (the source's TOY_T)")
    ap.add_argument("--eval-mults", default="1,2,4", help="evaluation lengths as multiples of T (must include 1)")
    ap.add_argument("--eval-n", type=int, default=256, help="evaluation streams per query type and length")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--bank-n", type=int, default=16384, help="training streams (the source's bank size)")
    ap.add_argument("--mix", default="0.4,0.3,0.3", help="training mix of plain,update,order queries")
    ap.add_argument("--p-restate", type=float, default=0.5, help="P(restatement) in plain and order streams")
    ap.add_argument("--contam", type=float, default=0.10)
    ap.add_argument("--n-false", type=int, default=2, help="false onsets (filler windows) per stream")
    ap.add_argument("--regime", choices=["raw", "ctx"], default="raw")
    ap.add_argument("--K", type=int, default=8, help="slots")
    ap.add_argument("--pre", type=int, default=4, help="P: entries before the onset")
    ap.add_argument("--win", type=int, default=8, help="n: entries per window")
    ap.add_argument("--pool", type=int, default=4, help="entries pooled into a key, and tokens into the query")
    ap.add_argument("--dk", type=int, default=64)
    ap.add_argument("--dv", type=int, default=16)
    ap.add_argument("--rope-dims", type=int, default=16, help="rope: rotated dims (concat: width of φ)")
    ap.add_argument("--rope-periods", default="64,65536", help="shortest,longest period (log-spaced; W1s default)")
    ap.add_argument("--concat-init", choices=["zero", "kaiming"], default="zero", help="concat: init of W_t")
    ap.add_argument("--ord-dim", type=int, default=16)
    ap.add_argument("--sup-w", type=float, default=3.0)
    ap.add_argument("--recency", type=float, default=0.002, help="linear: initial ω per token (W1s default)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/evolve/round11/stage0/g08_toy_time.json")
    c = ap.parse_args(argv)
    c.arms = [a for a in c.arms.split(",") if a]
    c.mults = [int(x) for x in c.eval_mults.split(",")]
    c.mix = [float(x) for x in c.mix.split(",")]
    if any(a not in ARMS for a in c.arms) or 1 not in c.mults or not 0 <= c.pre < c.win or c.pool > c.win:
        raise SystemExit("bad --arms / --eval-mults / --pre / --win / --pool")
    if c.rope_dims > c.dk:
        raise SystemExit("--rope-dims must be ≤ --dk")
    return c


def main(argv: list | None = None) -> dict:
    c = parse(argv)
    torch.set_num_threads(c.threads)
    E, M = world(regime=c.regime)
    t0 = time.time()
    trb = make_bank(c.bank_n, 1, c.T, c)
    evb = {}
    for mult in c.mults:                                          # eval_n streams per query type at each length
        parts = [make_bank(c.eval_n, 10_000 + 10 * mult + q, c.T * mult, c, qtype=q) for q in range(3)]
        evb[mult] = {k: torch.cat([b[k] for b in parts]) for k in trb}
    print(f"banks built in {time.time() - t0:.1f}s", flush=True)
    res = {}
    for arm in c.arms:
        res[arm] = train_arm(arm, c, trb, evb, E, M, log=lambda s: print(s, flush=True))
        print(f"{arm}: {res[arm]['secs']:.0f}s, {1000 * res[arm]['secs_per_step']:.1f} ms/step for {c.seeds} seeds",
              flush=True)
    summary = summarize(res, c.mults)
    g = gate(summary)
    print(table(summary, c.mults))
    print("G0.8:", json.dumps(g))
    cfg = {k: v for k, v in vars(c).items()}
    out = dict(config=cfg, secs=time.time() - t0, summary=summary, gate=g, arms=res)
    if c.out:
        Path(c.out).parent.mkdir(parents=True, exist_ok=True)
        Path(c.out).write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
