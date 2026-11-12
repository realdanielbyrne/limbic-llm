"""Round 13 step 0: ranking without generation (plan §4, step 0).

One forward pass per item on RecurrentGemma (``google/recurrentgemma-2b`` or ``-2b-it``; 8,192-token inputs, the
model's native 2,048-token window, stock sdpa). No read-back, no decoding. The pass captures, by forward hooks:
the early-tap and deep-tap block outputs, the pre-rotary queries and keys of every attention layer, and the RG-LRU
gates of the recurrent layers up to the deep tap. From these, for every scorer (``selectors13.py``) and item: the
rank of the queried span among the item's candidate spans, whether it falls within the read budget (512 columns),
and the reciprocal rank; for ``update`` items the ranks of both the new (queried) and the old statement.

Candidates. The shared span builder cuts the document (positions max(sink, len(pre)) … q0 − 1) into sentences of at
most 48 tokens. The candidates are the spans read-back could offer when the question is read: those that end at or
before q0 − D, D = W − sink − budget (the recent band), as ``readback.py`` enforces. When the queried statement is
itself in the recent band (the inside cell) the candidates are all the document's spans (``cand_mode``: all).

Circuit (``circuit.py``; fixed, float32): the dentate gate on the early tap, the filter bank on the deep tap, the
store over the candidate spans, the comparator at the question's last row (d RMS-normalised, as the store's states
and the running mean are). The slow bank is gated by one salience above a fixed threshold: ``--gate-signal``
(auto: the causal salience with the highest token AUC, statement against filler, on the first ``--calib-items``
dev items) and ``--gate-threshold`` (auto: Youden's J on those tokens); both stored in ``calibration``, reused on
resume, dropped by ``--force``.

Write-time saliences (per position; each is a multiplier, a fact-vs-filler AUC and a candidate gate signal):
  dentate        the dentate code's total strength (step 0; nearly constant by construction, kept for the record)
  rglru          mean over units and recurrent layers (≤ deep tap) of i_t · √(1 − a_t²), the RG-LRU's write
  attn           attention received (reads later rows: not causal, so never a gate signal)
  novelty@τ      1 − cos(centred deep state at t, fast band τ at t), τ = 4, 16, 64
  key_novelty    1 − overlap(sparse code at t, exponential average of earlier codes, τ = 64)
  surprisal      the model's own −log p(x_t | x_<t) (soft-capped logits from the captured final hidden states)
  early_norm     the early-tap norm over its running median (previous 4,096 non-sink positions)

Scorers (``selectors13.py``): the qk family for the question rows, the foot rows and both (``""``, ``.foot``,
``.qfoot``); ``qk_layer`` / ``qk_head`` (one layer / one query head, rows = qfoot; own families, plain selectors);
``infllm`` and ``infllm.foot``; ``cos_early``; ``dentate``; ``dentate_ctx@λ`` and ``dentate_ctx_z@λ`` (both
terms z-scored within the candidates); the retrievers; each main scorer × every salience; references
``position`` (pure recency), ``oracle_queried``, ``oracle_all``, ``random``.

Rows (step 0b format): ``rk`` / ``cb`` lists aligned to the file's ``scorer_names`` (rank of the queried span, and
the columns taken by the spans ranked above it), ``q_len``; on update items ``rk_old`` / ``cb_old`` / ``old_len``.
in_budget = cb + q_len ≤ budget. ``--summarise-only`` also reads step-0 files (``ranks`` dicts; column budgets
skipped).

Circuit-part checks in the same pass (stored per item in the trace directory, outside the repo):
  salience     per document span, its mean of every salience; AUC for "the span states a fact" against filler.
  context tag  cosine of the tags of two filler spans of one item; AUC for "same passage" against "same document,
               other passage", all pairs (raw) and distance-matched (each same-passage pair against the cross-passage
               pair of the closest gap).
  comparator   per λ: a_t with the top recalled entry; AUC for "the top entry is the queried statement".

Output (``--out``; ``ResultFile``: header with args, git, versions, model, compat): ``items`` {uid: row}, rows hold
labels, ranks and scores only (no text); ``summary``: recall@budget and MRR per scorer × type × cell × split with
bootstrap 95% intervals; the dev winner per family (qk, retriever, infllm, circuit) on decoy + paraphrase items and
its held-out numbers; paired differences of every circuit scorer against the better of the qk and retriever
winners; Holm-corrected comparisons between the family winners (held out); the AUCs with intervals; and the stop
rule evaluated (a part with AUC < 0.6; a plain selector with recall@budget ≥ 0.9 on decoy + paraphrase items).

    python experiments/ca2/rank13.py --model google/recurrentgemma-2b --device cuda --n 400 \\
        --out results/evolve/round13/step0/rank_rg2b.json 2>&1 | tee results/evolve/round13/step0/rank_rg2b.log
    CUDA_VISIBLE_DEVICES="" python experiments/ca2/rank13.py --tiny --n 4 --out /tmp/x/rank.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
sys.path.insert(0, str(EXP))
from ca2 import selectors13 as S  # noqa: E402
from ca2.circuit import (CircuitConfig, DentateGate, EpisodicStore, FilterBank, comparator,  # noqa: E402
                         context_tag, key_novelty, norm_ratio, novelty, rms_normalise)
from ca2.common import alignment, git_state, paired_boot  # noqa: E402
from ca2.items13 import CELLS, TYPES, Corpus, ItemBuilder, ItemConfig, Prompt  # noqa: E402
from ca2.measure import ResultFile, clean, versions  # noqa: E402

FAMILIES = ("qk", "retriever", "infllm", "circuit")
UNIT_FAMILIES = ("qk_layer", "qk_head")                           # plain selectors, reported apart from qk
PLAIN = ("qk", "qk_layer", "qk_head", "retriever")
SALIENCES = ("dentate", "rglru", "attn")                           # step-0 files; step 0b reads the trace keys
GATE_SIGNALS = ("dentate", "rglru", "novelty@4", "novelty@16", "novelty@64", "key_novelty", "surprisal",
                "early_norm")                                       # causal saliences (attn reads later rows)
PRIMARY_TYPES = ("decoy", "paraphrase")
REFS = ("oracle_queried", "oracle_all", "random")
TOPK = (1, 2, 4, 8)


class StopForward(Exception):
    pass


def family(name: str) -> str | None:
    base = name.split("*")[0]
    if base.startswith("qk_layer."):
        return "qk_layer"
    if base.startswith("qk_head."):
        return "qk_head"
    if base == "position":
        return "reference"
    if base.startswith("qk."):
        return "qk"
    if base == "retriever.bge" or base == "retriever.bow":
        return "retriever"
    if base.startswith("retriever."):
        return "retriever_upper"
    if base.startswith("infllm"):
        return "infllm"
    if base.startswith(("cos_early", "dentate")):
        return "circuit"
    return None


# --------------------------------------------------------------------------------------------- model + hooks
def load(args):
    from ca2 import readback_screen as scr
    ns = argparse.Namespace(tiny=args.tiny, tiny_window=args.tiny_window, seed=args.seed, device=args.device,
                            model=args.model, dtype=args.dtype)
    model, tok = scr.load(ns)
    return model, tok, scr.model_facts(model)


class Capture:
    """Forward hooks: early/deep block outputs, pre-rotary q/k per attention layer, RG-LRU write strength, and the
    final normed hidden states (for the model's own surprisal)."""

    def __init__(self, model, early: int, deep: int, stop_after: int | None = None) -> None:
        self.model = model
        layers = model.model.layers
        kinds = list(model.config.layers_block_type)[:len(layers)]
        self.attn = [m for m in model.modules() if hasattr(m, "rotary_emb") and hasattr(m, "layer_idx")
                     and hasattr(m, "k_proj")]
        self.attn_layers = [m.layer_idx for m in self.attn]
        self.early, self.deep = early, deep
        self.rec = [i for i, k in enumerate(kinds) if k == "recurrent" and i <= deep]
        self.out: dict = {}
        self.handles = [layers[early].register_forward_hook(self._grab("early")),
                        layers[deep].register_forward_hook(self._grab("deep"))]
        if stop_after is not None:
            self.handles.append(layers[stop_after].register_forward_hook(self._stop))
        else:                                                      # final hidden states (surprisal)
            self.handles.append(model.model.final_norm.register_forward_hook(self._grab("final", raw=True)))
        for m in self.attn:
            li = m.layer_idx
            self.handles.append(m.q_proj.register_forward_hook(self._grab(("q", li), raw=True)))
            self.handles.append(m.k_proj.register_forward_hook(self._grab(("k", li), raw=True)))
        for i in self.rec:
            self.handles.append(layers[i].temporal_block.rg_lru.register_forward_pre_hook(self._lru))

    def _grab(self, key, raw: bool = False):
        def hook(mod, args, out):
            x = out[0] if isinstance(out, tuple) else out
            self.out[key] = x.detach() if raw else x.detach().float()
        return hook

    @staticmethod
    def _stop(mod, args, out):
        raise StopForward

    def _lru(self, mod, args):
        act = args[0]
        B, T, Wd = act.shape
        H, bw = mod.num_attention_heads, mod.block_width
        x = act.float().reshape(B * T, H, bw).permute(1, 0, 2)
        ig = torch.sigmoid(torch.baddbmm(mod.input_gate_bias.float()[:, None], x, mod.input_gate_weight.float()))
        rg = torch.sigmoid(torch.baddbmm(mod.recurrent_gate_bias.float()[:, None], x,
                                         mod.recurrent_gate_weight.float()))
        log_a = -8.0 * rg * torch.nn.functional.softplus(mod.recurrent_param.float()).reshape(H, 1, bw)
        w = ig * torch.sqrt((1 - torch.exp(2 * log_a)).clamp_min(0))
        v = w.permute(1, 0, 2).reshape(B, T, Wd).mean(-1)
        self.out["lru_sum"] = self.out.get("lru_sum", 0) + v
        self.out["lru_n"] = self.out.get("lru_n", 0) + 1
        return None

    @torch.no_grad()
    def run(self, ids: torch.Tensor) -> dict:
        self.out = {}
        try:
            self.model(ids, use_cache=False, logits_to_keep=1)
        except StopForward:
            pass
        return self.out

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------------------------------------- statistics
def auc(scores, labels) -> float | None:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.all() or not y.any():
        return None
    return _wauc(s, y, np.ones_like(s))


def _wauc(s, y, w) -> float | None:
    u, inv = np.unique(s, return_inverse=True)
    wp = np.bincount(inv, weights=w * y, minlength=u.size)
    wn = np.bincount(inv, weights=w * ~y, minlength=u.size)
    P, N = wp.sum(), wn.sum()
    if P <= 0 or N <= 0:
        return None
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (P * N))


def auc_boot(scores, labels, clusters, n: int = 500, seed: int = 0) -> dict:
    """AUC with a percentile interval from a cluster (item) bootstrap."""
    s, y, c = np.asarray(scores, float), np.asarray(labels, bool), np.asarray(clusters)
    ok = np.isfinite(s)
    s, y, c = s[ok], y[ok], c[ok]
    if s.size == 0 or y.all() or not y.any():
        return {"auc": None, "lo": None, "hi": None, "n_pos": int(y.sum()), "n_neg": int((~y).sum()), "n_items": 0}
    keys, ci = np.unique(c, return_inverse=True)
    u, inv = np.unique(s, return_inverse=True)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, keys.size, keys.size), minlength=keys.size).astype(float)[ci]
        wp = np.bincount(inv, weights=w * y, minlength=u.size)
        wn = np.bincount(inv, weights=w * ~y, minlength=u.size)
        P, N = wp.sum(), wn.sum()
        if P > 0 and N > 0:
            bs.append((wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum() / (P * N))
    lo, hi = (np.quantile(bs, [0.025, 0.975]).tolist() if bs else (None, None))
    return {"auc": _wauc(s, y, np.ones_like(s)), "lo": lo, "hi": hi, "n_pos": int(y.sum()),
            "n_neg": int((~y).sum()), "n_items": int(keys.size)}


def holm(ps: dict) -> dict:
    keys = sorted([k for k, p in ps.items() if p is not None], key=lambda k: ps[k])
    m, out, run = len(keys), {k: None for k in ps}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def youden(sal: np.ndarray, lab: np.ndarray) -> tuple[float, float]:
    """Threshold maximising TPR − FPR (salience > threshold predicts a statement token)."""
    qs = np.unique(np.quantile(sal, np.linspace(0.0, 0.999, 400)))
    P, N = lab.sum(), (~lab).sum()
    best = (-1.0, float(qs[0]))
    for th in qs:
        pr = sal > th
        j = (pr & lab).sum() / max(P, 1) - (pr & ~lab).sum() / max(N, 1)
        if j > best[0]:
            best = (float(j), float(th))
    return best[1], best[0]


# ---------------------------------------------------------------------------------------------- per item
class Ranker:
    def __init__(self, model, tok, facts, args) -> None:
        self.model, self.tok, self.args = model, tok, args
        self.W = facts["window"]
        self.dev = next(model.parameters()).device
        self.D = model.config.hidden_size
        self.recent = self.W - args.sink - args.budget
        if self.recent < 1:
            raise ValueError("sink + budget leave no recent band")
        self.tables = S.TokenTables(tok)
        self.gate = DentateGate(self.D, args.expand, args.k_frac, args.circuit_seed, device=self.dev)
        self.lams = [float(x) for x in args.lambdas.split(",")]
        self.retrievers: dict[str, S.Retriever] = {}
        rdev = self.dev
        if args.retriever == "bge":
            self.retrievers["retriever.bge"] = S.Retriever(S.BGE, device=rdev)
        elif args.retriever == "bow":
            self.retrievers["retriever.bow"] = S.Retriever("bow", embed_fn=bow_embed)
        if args.upper_retriever:
            self.retrievers["retriever.qwen3_4b"] = S.Retriever(S.QWEN_EMB, device=rdev, dtype=torch.bfloat16)
        self.cap = Capture(model, args.early_tap, args.deep_tap)
        self.inv = {m.layer_idx: m.rotary_emb.inv_freq for m in self.cap.attn}
        self.scaling = {m.layer_idx: m.scaling for m in self.cap.attn}
        self.H = model.config.num_attention_heads
        self.hd = model.config.head_dim
        self.bank = None
        self.gate_signal = None
        self.names: list[str] | None = None
        self.set_threshold(math.inf, "dentate")                    # replaced after calibration

    def set_threshold(self, th: float, signal: str = "dentate") -> None:
        cfg = CircuitConfig(sink=self.args.sink, ring=self.args.ring, mean_tau=self.args.mean_tau,
                            gate_threshold=th, seed=self.args.circuit_seed, centre=not self.args.no_centre,
                            lag0=self.args.lag0, slow_gated=not self.args.slow_ungated,
                            band_diff=not self.args.raw_bands)
        self.ccfg = cfg
        self.bank = FilterBank(cfg)
        self.gate_signal = signal

    # --------------------------------------------------------------------- spans
    def spans(self, it) -> dict:
        a = self.args
        lo = max(a.sink, it.n_pre)
        allsp = S.cut_spans(it.ids, self.tables, lo, it.q0, a.cap)
        where = {sp: j for j, sp in enumerate(allsp)}
        st = []
        for f, l in it.statements:
            j = where.get((f, l + 1))
            if j is None:
                raise AssertionError(f"{it.uid}: statement ({f}, {l}) is not one candidate span")
            st.append(j)
        last_ok = it.q0 - self.recent
        elig = [j for j, (s0, s1) in enumerate(allsp) if s1 - 1 <= last_ok]
        targets_all = [st[it.queried]] + ([st[it.old]] if it.old is not None else [])
        if all(t in set(elig) for t in targets_all):
            cand, mode = elig, "eligible"
        else:
            cand, mode = list(range(len(allsp))), "all"
        cmap = {j: i for i, j in enumerate(cand)}
        return {"all": allsp, "stmt_all": st, "cand": [allsp[j] for j in cand], "mode": mode,
                "targets": [cmap[t] for t in targets_all], "stmts": [cmap[j] for j in st if j in cmap]}

    # --------------------------------------------------------------------- saliences
    def surprisal(self, final: torch.Tensor, ids: list[int], chunk: int = 512) -> torch.Tensor:
        """−log p(x_t | x_<t) from the model's own final hidden states (soft-capped logits as the model makes them);
        0 at t = 0. final (T, D) in the base dtype."""
        lm = self.model.lm_head
        cap = getattr(self.model.config, "logits_soft_cap", None)
        T = final.shape[0]
        tgt = torch.tensor(ids, device=final.device)
        out = torch.zeros(T, device=final.device)
        for c0 in range(0, T - 1, chunk):
            c1 = min(T - 1, c0 + chunk)
            z = lm(final[c0:c1]).float()
            if cap:
                z = torch.tanh(z / cap) * cap
            lp = torch.log_softmax(z, -1)
            out[c0 + 1:c1 + 1] = -lp.gather(1, tgt[c0 + 1:c1 + 1, None])[:, 0]
        return out

    def saliences(self, out: dict, b: int, it) -> tuple[dict, dict]:
        """Every write-time salience of row b (T,), and the shared pieces (codes, fast bank)."""
        early, deep = out["early"][b], out["deep"][b]
        T = early.shape[0]
        idx, val, dsal = self.gate(early)
        fast = self.bank(deep[None], None, only_fast=True)
        sal = {"dentate": dsal}
        sal["rglru"] = (out["lru_sum"][b] / out["lru_n"]).float() if "lru_sum" in out else torch.ones(T,
                                                                                                        device=self.dev)
        for k, tau in enumerate(self.ccfg.taus[:self.ccfg.n_fast]):
            sal[f"novelty@{tau:g}"] = novelty(fast.x[0], fast.mean[0], fast.bands[0, :, k])
        sal["key_novelty"] = key_novelty(self.gate, idx, val, tau=64.0)
        if "final" in out:
            sal["surprisal"] = self.surprisal(out["final"][b], it.ids)
        sal["early_norm"] = norm_ratio(early[None], self.args.sink, self.args.ring)[0]
        return sal, {"idx": idx, "val": val, "fast": fast}

    # --------------------------------------------------------------------- calibration
    @torch.no_grad()
    def calibrate(self, items, batch: int) -> dict:
        """Token AUC (statement vs filler, document positions) of every causal salience on the calibration items;
        the gate signal (``--gate-signal``, auto: the highest AUC) and its Youden threshold."""
        vals: dict[str, list] = {}
        labs = []
        for k in range(0, len(items), batch):
            its = items[k:k + batch]
            out = self.cap.run(torch.tensor([it.ids for it in its], device=self.dev))
            for b, it in enumerate(its):
                sal, _ = self.saliences(out, b, it)
                lab = np.zeros(len(it.ids), bool)
                for f, l in it.statements:
                    lab[f:l + 1] = True
                lo = max(self.args.sink, it.n_pre)
                labs.append(lab[lo:it.q0])
                for n, v in sal.items():
                    vals.setdefault(n, []).append(v[lo:it.q0].float().cpu().numpy())
        lab = np.concatenate(labs)
        aucs = {n: auc(np.concatenate(v), lab) for n, v in vals.items() if n in GATE_SIGNALS}
        sig = self.args.gate_signal
        if sig == "auto":
            sig = max((n for n in aucs if aucs[n] is not None), key=lambda n: aucs[n])
        if sig not in vals:
            raise SystemExit(f"--gate-signal {sig}: not available (have {sorted(vals)})")
        x = np.concatenate(vals[sig])
        th, j = youden(x, lab)
        return {"signal": sig, "threshold": th, "youden_j": j, "token_auc": aucs.get(sig), "token_aucs": aucs,
                "gate_rate": float((x > th).mean()), "uids": [it.uid for it in items]}

    # --------------------------------------------------------------------- one batch
    @torch.no_grad()
    def batch(self, items) -> list[tuple[dict, dict]]:
        ids = torch.tensor([it.ids for it in items], device=self.dev)
        out = self.cap.run(ids)
        res = []
        for b, it in enumerate(items):
            t0 = time.time()
            res.append(self.item(it, out, b))
            res[-1][0]["seconds_post"] = time.time() - t0
        return res

    def item(self, it, out, b) -> tuple[dict, dict]:
        a, dev = self.args, self.dev
        sp = self.spans(it)
        cand = sp["cand"]
        T = len(it.ids)
        cmask = torch.zeros(T, dtype=torch.bool, device=dev)
        for s0, s1 in cand:
            cmask[s0:s1] = True
        rowsets = {"": torch.arange(*it.q_rows, device=dev), ".foot": torch.arange(*it.foot_rows, device=dev),
                   ".qfoot": torch.arange(it.q_rows[0], it.foot_rows[1], device=dev)}
        early, deep = out["early"][b], out["deep"][b]
        L = self.cap.attn_layers
        qall = {li: out[("q", li)][b].float().view(T, self.H, self.hd).permute(1, 0, 2) for li in L}
        ks = [out[("k", li)][b].float().view(T, -1, self.hd)[:, 0] for li in L]
        inv = self.inv[L[0]]
        base: dict[str, torch.Tensor] = {}
        unit_scores: dict[str, torch.Tensor] = {}
        for rs, rows in rowsets.items():
            qs = [qall[li][:, rows] for li in L]
            for name, var in S.QK_VARIANTS.items():
                base[name + rs] = S.qk_scores(qs, ks, rows, cmask, inv, var, ord_delta=self.recent)
        qf = rowsets[".qfoot"]
        for i, li in enumerate(L):                                 # qk_layer / qk_head: one layer / one head
            ql = qall[li][:, qf]
            unit_scores[f"qk_layer.L{li}"] = S.qk_scores([ql], [ks[i]], qf, cmask, inv, S.QK_UNIT)
            for h in range(self.H):
                unit_scores[f"qk_head.L{li}.H{h}"] = S.qk_scores([ql[h:h + 1]], [ks[i]], qf, cmask, inv, S.QK_UNIT)
        recv = torch.zeros(T, device=dev)
        for li, k in zip(L, ks):
            recv += S.attention_received(qall[li], k, self.W, self.inv[li], self.scaling[li])
        recv /= len(L)
        base["infllm"] = S.infllm_scores([qall[li][:, rowsets[""]] for li in L], ks, recv, cand, n_rep=a.n_rep)
        base["infllm.foot"] = S.infllm_scores([qall[li][:, rowsets[".foot"]] for li in L], ks, recv, cand,
                                              n_rep=a.n_rep)
        qr = rowsets[""]
        lo = max(a.sink, it.n_pre)
        centre = early[lo:it.q0].mean(0)
        base["cos_early.max"] = S.cos_scores(early, qr, cmask, centre, "max")
        base["cos_early.mean"] = S.cos_scores(early, qr, cmask, centre, "mean")
        sal, sh = self.saliences(out, b, it)
        sal["attn"] = recv
        idx, val = sh["idx"], sh["val"]
        base["dentate"], qkey = S.dentate_scores(idx, val, qr, cmask, self.gate.U)
        bo = self.bank(deep[None], sal[self.gate_signal][None])
        outputs = bo.outputs[0]
        store = EpisodicStore.from_spans(cand, self.gate, idx, val, outputs, deep)
        tq = it.q_rows[1] - 1
        ctx_q = context_tag(outputs[tq])
        ov, cc = store.terms(qkey, ctx_q)
        zs = lambda v: (v - v.mean()) / v.std().clamp_min(1e-12)  # noqa: E731
        store_span = {}
        for lam in self.lams:
            store_span[lam] = ov + lam * cc
            base[f"dentate_ctx@{lam:g}"] = S.broadcast(store_span[lam], cand, T)
            base[f"dentate_ctx_z@{lam:g}"] = S.broadcast(zs(ov) + lam * zs(cc), cand, T)
        texts = None
        for name, r in self.retrievers.items():
            if texts is None:
                texts = [self.tok.decode(it.ids[s0:s1]) for s0, s1 in cand]
            base[name] = S.broadcast(r.score(it.question, texts).to(dev), cand, T)
        # ---- salience products (percentile ranks, computed once each)
        p_sal = {n: S.percentile(v, cmask) for n, v in sal.items()}
        scores = dict(base)
        for n, v in base.items():
            pb = S.percentile(v, cmask)
            for sn, ps in p_sal.items():
                scores[f"{n}*{sn}"] = S.pct_product(pb, ps)
        scores.update(unit_scores)
        scores["position"] = S.position_scores(cmask)
        scores["oracle_queried"] = S.oracle_scores(cand, sp["targets"][:1], T, dev)
        scores["oracle_all"] = S.oracle_scores(cand, sp["stmts"], T, dev)
        scores["random"] = S.random_scores(cand, T, it.seed, dev)
        # ---- ranks (batched)
        names = list(scores)
        P = torch.stack([scores[n] for n in names])
        nan_scores = [names[i] for i in torch.isnan(P).any(1).nonzero()[:, 0].tolist()]
        index = S.span_index(cand, a.cap, dev)
        SS = S.span_scores(P, cand, a.cap, index=index)
        g = torch.Generator().manual_seed(it.seed + 1)
        tieb = torch.randperm(len(cand), generator=g)
        lengths = torch.tensor([s1 - s0 for s0, s1 in cand])
        rk, cb = S.rank_many(SS, lengths, sp["targets"], tieb)
        q_len = int(lengths[sp["targets"][0]])
        nsel = len(S.take(S.order(SS[names.index("random")], tieb), lengths.tolist(), a.budget))
        # ---- comparator (d RMS-normalised: the states and the mean are on that scale)
        comp = {}
        dmean = bo.mean[0, tq]
        c = outputs[tq, -2] if outputs.shape[1] > 1 else outputs[tq, -1]
        d = rms_normalise(deep[tq])
        for lam in self.lams:
            top = int(store_span[lam].argmax())
            cm = comparator(d, store.states, top, c, dmean)
            comp[f"{lam:g}"] = [cm["a"], int(top == sp["targets"][0])]
        # ---- parts (trace)
        allsp = sp["all"]
        isst = np.zeros(len(allsp), bool)
        isst[sp["stmt_all"]] = True
        ai = S.span_index(allsp, a.cap, dev)

        def span_mean(v):
            x = v.float()[ai[0]] * ai[1]
            return (x.sum(-1) / ai[1].sum(-1).clamp_min(1)).cpu().numpy()
        parts = {"sal_label": isst, **{f"sal_{k}": span_mean(v) for k, v in sal.items()}}
        parts.update(self.ctx_pairs(it, allsp, isst, outputs))
        row = {**it.meta(), "cand_mode": sp["mode"], "n_cand": len(cand), "n_cand_statements": len(sp["stmts"]),
               "n_spans": len(allsp), "random_selected": nsel, "rk": rk[:, 0].tolist(), "cb": cb[:, 0].tolist(),
               "q_len": q_len, "comparator": comp, "excluded": int(bo.excluded[0].sum()),
               "slow_gate_rate": float(bo.gate[0, lo:it.q0].mean()), "nan_scores": nan_scores}
        if rk.shape[1] > 1:
            row.update(rk_old=rk[:, 1].tolist(), cb_old=cb[:, 1].tolist(), old_len=int(lengths[sp["targets"][1]]))
        if self.names is None:
            self.names = names
        elif names != self.names:
            raise AssertionError("the scorer list changed between items")
        return row, parts

    def ctx_pairs(self, it, allsp, isst, outputs) -> dict:
        """Filler spans inside one passage piece: same-passage and other-passage tag cosines (with gaps)."""
        a = self.args
        pid = np.full(len(it.ids), -1)
        for s0, s1, p, _ in it.passages:
            pid[s0:s1] = p
        js = [j for j, (s0, s1) in enumerate(allsp) if not isst[j] and pid[s0] >= 0 and pid[s0] == pid[s1 - 1]]
        if len(js) < 3:
            return {"ctx_cos": np.zeros(0), "ctx_same": np.zeros(0, bool), "ctx_gap": np.zeros(0)}
        last = torch.tensor([allsp[j][1] - 1 for j in js], device=outputs.device)
        tags = context_tag(outputs[last])
        cos = (tags @ tags.T).cpu().numpy()
        P = np.array([pid[allsp[j][0]] for j in js])
        endp = np.array([allsp[j][1] - 1 for j in js])
        iu = np.triu_indices(len(js), 1)
        same = P[iu[0]] == P[iu[1]]
        gap = np.abs(endp[iu[0]] - endp[iu[1]])
        rng = np.random.default_rng(it.seed)
        sel_s = np.flatnonzero(same)
        sel_o = np.flatnonzero(~same)
        if sel_s.size > a.ctx_pairs:
            sel_s = rng.choice(sel_s, a.ctx_pairs, replace=False)
        # distance-matched: for each same pair, the cross pair with the closest gap
        match = np.zeros(0, int)
        if sel_o.size and sel_s.size:
            og = gap[sel_o]
            srt = np.argsort(og, kind="stable")
            pos = np.searchsorted(og[srt], gap[sel_s]).clip(0, srt.size - 1)
            pos2 = (pos - 1).clip(0, srt.size - 1)
            pick = np.where(np.abs(og[srt][pos] - gap[sel_s]) <= np.abs(og[srt][pos2] - gap[sel_s]), pos, pos2)
            match = sel_o[srt[pick]]
        if sel_o.size > a.ctx_pairs:
            sel_o = rng.choice(sel_o, a.ctx_pairs, replace=False)
        keep = np.concatenate([sel_s, sel_o])
        return {"ctx_cos": cos[iu][keep], "ctx_same": same[keep], "ctx_gap": gap[keep],
                "ctxm_cos": np.concatenate([cos[iu][sel_s], cos[iu][match]]),
                "ctxm_same": np.concatenate([np.ones(sel_s.size, bool), np.zeros(match.size, bool)])}


def bow_embed(texts: list[str], query: bool) -> torch.Tensor:
    """Hashed bag of words (a deterministic offline stand-in for tests)."""
    import re
    import zlib
    out = torch.zeros(len(texts), 512)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, zlib.crc32(w.encode()) % 512] += 1.0
    return out + 1e-6


# ---------------------------------------------------------------------------------------------- summary
COL_BUDGETS = (32, 64, 128)


class RankTable:
    """All rows as matrices (items × scorers). Reads both row formats: step 0 (``ranks``: {scorer: [rank, in_budget]
    (+ [rank_old, in_old])}) and step 0b (``rk``/``cb`` lists aligned to ``scorer_names``, ``q_len``, and
    ``rk_old``/``cb_old``/``old_len`` on update items; in_budget = cb + q_len ≤ budget)."""

    def __init__(self, data: dict) -> None:
        rows = list(data.get("items", {}).values())
        self.rows = rows
        n = len(rows)
        budget = int(data.get("args", {}).get("budget", 512))
        nan = float("nan")
        if rows and "rk" in rows[0]:
            self.names = list(data["scorer_names"])
            m = len(self.names)
            self.R = np.array([r["rk"] for r in rows], float)
            self.CB = np.array([r["cb"] for r in rows], float)
            self.QL = np.array([r["q_len"] for r in rows], float)
            self.IN = (self.CB + self.QL[:, None] <= budget).astype(float)
            self.Ro = np.array([r.get("rk_old", [nan] * m) for r in rows], float)
            self.CBo = np.array([r.get("cb_old", [nan] * m) for r in rows], float)
            self.OL = np.array([r.get("old_len", nan) for r in rows], float)
            self.INo = np.where(np.isfinite(self.CBo), (self.CBo + self.OL[:, None] <= budget), nan)
            self.has_cols = True
        else:
            self.names = list(rows[0]["ranks"]) if rows else []
            m = len(self.names)
            get = lambda r, k: [r["ranks"][s][k] if len(r["ranks"][s]) > k else nan for s in self.names]  # noqa
            self.R = np.array([get(r, 0) for r in rows], float).reshape(n, m)
            self.IN = np.array([get(r, 1) for r in rows], float).reshape(n, m)
            self.Ro = np.array([get(r, 2) for r in rows], float).reshape(n, m)
            self.INo = np.array([get(r, 3) for r in rows], float).reshape(n, m)
            self.CB = self.QL = None
            self.has_cols = False
        self.col = {s: j for j, s in enumerate(self.names)}

    def metric(self, name: str) -> np.ndarray | None:
        """(items × scorers) matrix of a per-item measure."""
        if name == "recall":
            return self.IN
        if name == "mrr":
            return 1.0 / self.R
        if name.startswith("recall@cols"):
            if not self.has_cols:
                return None
            return (self.CB + self.QL[:, None] <= int(name[len("recall@cols"):])).astype(float)
        if name.startswith("recall@"):
            return (self.R <= int(name[len("recall@"):])).astype(float)
        if name == "top1":
            return (self.R <= 1).astype(float)
        if name == "new_above_old":
            return np.where(np.isfinite(self.Ro), (self.R < self.Ro).astype(float), np.nan)
        raise KeyError(name)


_BOOT: dict = {}


def boot_cols(X: np.ndarray, n_boot: int, seed: int) -> dict:
    """Bootstrap over items (rows) of every column's mean at once: mean, 2.5/97.5% quantiles, p(≤ 0), p(≥ 0)."""
    n = X.shape[0]
    key = (n, n_boot, seed)
    if key not in _BOOT:
        _BOOT[key] = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)
    B = _BOOT[key] @ np.nan_to_num(X) / n
    lo, hi = np.quantile(B, [0.025, 0.975], axis=0)
    return {"mean": np.nanmean(X, 0), "lo": lo, "hi": hi, "p_le_0": (B <= 0).mean(0), "p_ge_0": (B >= 0).mean(0),
            "n": n}


def _ent(bc: dict, j: int) -> dict:
    r = lambda x: round(float(x), 5)  # noqa: E731  (the table is large: five decimals are plenty)
    return {"mean": r(bc["mean"][j]), "lo": r(bc["lo"][j]), "hi": r(bc["hi"][j]), "n": int(bc["n"])}


def summarise(data: dict, trace: Path, seed: int = 0, n_boot: int = 2000) -> dict:
    rt = RankTable(data)
    rows = rt.rows
    if not rows:
        return {}
    scorers = rt.names
    types = sorted({r["type"] for r in rows}, key=TYPES.index)
    cellset = sorted({r["cell"] for r in rows}, key=CELLS.index)
    out: dict = {"n_items": len(rows), "scorers": scorers}
    groups: dict[str, np.ndarray] = {}
    for c in cellset:
        for sp in ("dev", "held", "all"):
            for t in types + ["primary"]:
                ok = [i for i, r in enumerate(rows) if r["cell"] == c and (sp == "all" or r["split"] == sp)
                      and (r["type"] in PRIMARY_TYPES if t == "primary" else r["type"] == t)]
                if ok:
                    groups[f"{t}/{c}/{sp}"] = np.array(ok)
    metrics = ["recall", "mrr"] + [f"recall@{k}" for k in TOPK] + \
        ([f"recall@cols{b}" for b in COL_BUDGETS] if rt.has_cols else [])
    mats = {m: rt.metric(m) for m in metrics}
    # ---- per scorer x type x cell x split
    tab: dict = {s: {} for s in scorers}
    for gk, gi in groups.items():
        bcs = {m: boot_cols(mats[m][gi], n_boot, seed) for m in metrics}
        upd = rows[gi[0]]["type"] == "update" and np.isfinite(rt.Ro[gi]).any()
        if upd:
            nao = boot_cols(rt.metric("new_above_old")[gi], n_boot, seed)
        for j, s in enumerate(scorers):
            ent = {m: _ent(bcs[m], j) for m in metrics}
            if upd and np.isfinite(rt.Ro[gi, j]).all():
                ent["recall_old"] = float(np.nanmean(rt.INo[gi, j]))
                ent["mrr_old"] = float(np.mean(1.0 / rt.Ro[gi, j]))
                ent["new_above_old"] = float(nao["mean"][j])
            tab[s][gk] = ent
    out["table"] = tab
    out["metrics"] = {"recall": "recall@budget (queried span within --budget columns)", "mrr": "mean reciprocal rank",
                      **{f"recall@{k}": f"queried span ranked <= {k} (spans)" for k in TOPK},
                      "column_budgets": ({f"recall@cols{b}": f"the queried span fits in the first {b} columns"
                                          for b in COL_BUDGETS} if rt.has_cols else
                                         "not computable from these rows (step-0 format: no span lengths); skipped")}
    # ---- update items: the new statement ranked above the old one, with an interval
    nao_out = {}
    for gk, gi in groups.items():
        if rows[gi[0]]["type"] != "update" or not np.isfinite(rt.Ro[gi]).any():
            continue
        bc = boot_cols(rt.metric("new_above_old")[gi], n_boot, seed)
        for j, s in enumerate(scorers):
            if np.isfinite(rt.Ro[gi, j]).all():
                nao_out.setdefault(s, {})[gk] = _ent(bc, j)
    out["update_new_above_old"] = nao_out
    if "position" in nao_out:
        out["update_new_above_old_position"] = nao_out["position"]   # pure recency: 1 by construction
    # ---- winners per family on the primary pool (the beyond cell if present)
    pc = "beyond" if "beyond" in cellset else cellset[0]
    dev_key, held_key = f"primary/{pc}/dev", f"primary/{pc}/held"
    out["primary_pool"] = {"types": list(PRIMARY_TYPES), "cell": pc}
    fams_all = FAMILIES + UNIT_FAMILIES + ("retriever_upper",)
    ranked_fams = [f for f in FAMILIES + UNIT_FAMILIES]

    def pick(fam, key):
        cands = [s for s in scorers if family(s) == fam]
        if not cands or dev_key not in groups:
            return None
        return max(cands, key=lambda s: key(tab[s][dev_key]))

    def winners_by(key, keys):
        w = {}
        for fam in fams_all:
            best = pick(fam, key)
            if best is None:
                continue
            e = {"scorer": best, "dev": {k: tab[best][dev_key][k] for k in keys}}
            if held_key in groups:
                e["held"] = {k: tab[best][held_key][k] for k in keys}
                e["held_by_type"] = {t: {k: tab[best][f"{t}/{pc}/held"][k]["mean"] for k in keys}
                                     for t in types if f"{t}/{pc}/held" in groups}
            w[fam] = e
        return w

    def paired(gi, a_, b_, m):
        X = mats[m] if m != "top1" else mats["recall@1"]
        return boot_cols(X[gi][:, [rt.col[a_]]] - X[gi][:, [rt.col[b_]]], n_boot, seed)

    def vs_baseline(bb, ms):
        diffs = {}
        for name in [s for s in scorers if family(s) == "circuit"]:
            diffs[name] = {}
            for gk, gi in groups.items():
                if gk.endswith("/all"):
                    continue
                if len(ms) == 1:
                    d = paired(gi, name, bb, ms[0])
                    diffs[name][gk] = {"recall_diff": float(d["mean"][0]), "lo": float(d["lo"][0]),
                                       "hi": float(d["hi"][0])}
                else:
                    diffs[name][gk] = {}
                    for m in ms:
                        d = paired(gi, name, bb, m)
                        diffs[name][gk][m] = {"diff": float(d["mean"][0]), "lo": float(d["lo"][0]),
                                              "hi": float(d["hi"][0])}
        return diffs

    def holm_set(win, m):
        gi = groups[held_key]
        fams = [f for f in ranked_fams if f in win]
        comps, ps = {}, {}
        for i in range(len(fams)):
            for j in range(i + 1, len(fams)):
                a_, b_ = win[fams[i]]["scorer"], win[fams[j]]["scorer"]
                d = paired(gi, a_, b_, m)
                key = f"{fams[i]} - {fams[j]}"
                comps[key] = {"a": a_, "b": b_, "diff": float(d["mean"][0]), "lo": float(d["lo"][0]),
                              "hi": float(d["hi"][0])}
                ps[key] = min(1.0, 2 * min(float(d["p_le_0"][0]), float(d["p_ge_0"][0])))
        adj = holm(ps)
        for k in comps:
            comps[k]["p"], comps[k]["p_holm"] = ps[k], adj[k]
        return comps

    # step-0 versions (chosen on recall@budget), kept
    winners = winners_by(lambda e: (e["recall"]["mean"], e["mrr"]["mean"]), ("recall", "mrr"))
    out["winners"] = winners
    base = [winners[f]["scorer"] for f in PLAIN if f in winners]
    if base:
        bb = max(base, key=lambda s: (tab[s][dev_key]["recall"]["mean"], tab[s][dev_key]["mrr"]["mean"]))
        out["best_baseline"] = bb
        out["circuit_vs_best_baseline"] = vs_baseline(bb, ["recall"])
    if held_key in groups and len(winners) >= 2:
        out["holm_families"] = holm_set(winners, "recall")
    # DEC18: chosen on dev MRR; compared on top-1, MRR and recall@budget
    keys2 = ("recall@1", "mrr", "recall") + tuple(f"recall@{k}" for k in TOPK[1:]) + \
        tuple(f"recall@cols{b}" for b in COL_BUDGETS if rt.has_cols)
    win2 = winners_by(lambda e: (e["mrr"]["mean"], e["recall@1"]["mean"]), keys2)
    out["winners_mrr"] = win2
    base = [win2[f]["scorer"] for f in PLAIN if f in win2]
    if base:
        bb = max(base, key=lambda s: (tab[s][dev_key]["mrr"]["mean"], tab[s][dev_key]["recall@1"]["mean"]))
        out["best_baseline_mrr"] = bb
        out["circuit_vs_best_baseline_mrr"] = vs_baseline(bb, ["top1", "mrr", "recall"])
    if held_key in groups and len(win2) >= 2:
        out["holm_families_mrr"] = {m: holm_set(win2, m) for m in ("top1", "mrr", "recall")}
    # ---- AUCs of the circuit's parts
    parts = load_parts(trace, [r["uid"] for r in rows])
    aucs = {}
    sal_names: list[str] = []
    if parts:
        sal_names = [k[4:] for k in parts[0] if k.startswith("sal_") and k != "sal_label"]
        cl = np.concatenate([[i] * len(p["sal_label"]) for i, p in enumerate(parts)])
        lab = np.concatenate([p["sal_label"] for p in parts])
        for sn in sal_names:
            if all(f"sal_{sn}" in p for p in parts):
                aucs[f"salience.{sn}"] = auc_boot(np.concatenate([p[f"sal_{sn}"] for p in parts]), lab, cl,
                                                  seed=seed)
        for key, (sk, lk) in {"context_tag": ("ctx_cos", "ctx_same"),
                              "context_tag.matched": ("ctxm_cos", "ctxm_same")}.items():
            ps_ = [p for p in parts if sk in p and len(p[sk])]
            if ps_:
                cl2 = np.concatenate([[i] * len(p[sk]) for i, p in enumerate(ps_)])
                aucs[key] = auc_boot(np.concatenate([p[sk] for p in ps_]), np.concatenate([p[lk] for p in ps_]),
                                     cl2, seed=seed)
    lams = list(rows[0].get("comparator", {}))
    for lam in lams:
        a_ = [r["comparator"][lam][0] for r in rows]
        y_ = [r["comparator"][lam][1] for r in rows]
        aucs[f"comparator@{lam}"] = auc_boot(a_, y_, list(range(len(rows))), seed=seed)
        aucs[f"comparator@{lam}"]["top_is_queried"] = float(np.mean(y_))
    out["auc"] = aucs
    if data.get("calibration"):
        out["gate"] = {k: data["calibration"].get(k) for k in ("signal", "threshold", "youden_j", "token_auc",
                                                               "token_aucs", "gate_rate")}

    def flag(k):
        e = aucs.get(k, {})
        return {"auc": e.get("auc"), "lo": e.get("lo"), "hi": e.get("hi"),
                "below_0.6": None if e.get("auc") is None else bool(e["auc"] < 0.6),
                "upper_below_0.6": None if e.get("hi") is None else bool(e["hi"] < 0.6)}

    def lam_of(w):
        sc = w.get("circuit", {}).get("scorer", "")
        if "dentate_ctx" in sc and "@" in sc:
            return sc.split("@")[1].split("*")[0]
        return lams[0] if lams else None
    # ---- stop rule (step 0, kept)
    lam_star = lam_of(winners)
    low = {k: {kk: v for kk, v in flag(k).items() if kk != "lo"}
           for k in ["salience.dentate", "context_tag.matched", f"comparator@{lam_star}"]}
    plain = {}
    for fam in ("qk", "retriever"):
        if fam in winners and "held" in winners[fam]:
            r = winners[fam]["held"]["recall"]
            plain[fam] = {"scorer": winners[fam]["scorer"], "held_recall": r["mean"], "lo": r["lo"],
                          "ge_0.9": bool(r["mean"] is not None and r["mean"] >= 0.9)}
    dev_any = None
    if dev_key in groups:
        dev_any = {s: tab[s][dev_key]["recall"]["mean"] for s in scorers if family(s) in ("qk", "retriever")
                   and tab[s][dev_key]["recall"]["mean"] >= 0.9}
    out["stop_rule"] = {"parts": low, "parts_below_0.6": [k for k, v in low.items() if v["below_0.6"]],
                        "comparator_lambda": lam_star, "plain_selector": plain,
                        "plain_dev_variants_ge_0.9": dev_any,
                        "selection_not_missing": bool(any(v["ge_0.9"] for v in plain.values()))}
    # ---- stop rule, DEC18: held-out top-1 of the dev-chosen (MRR) plain winners; every part's AUC
    plain2 = {}
    for fam in PLAIN:
        if fam in win2 and "held" in win2[fam]:
            r = win2[fam]["held"]["recall@1"]
            plain2[fam] = {"scorer": win2[fam]["scorer"], "held_top1": r["mean"], "lo": r["lo"], "hi": r["hi"],
                           "held_mrr": win2[fam]["held"]["mrr"]["mean"],
                           "ge_0.9": bool(r["mean"] is not None and r["mean"] >= 0.9)}
    lam2 = lam_of(win2)
    keys_p = [f"salience.{sn}" for sn in sal_names] + ["context_tag.matched", f"comparator@{lam2}"]
    low2 = {k: flag(k) for k in keys_p}
    out["stop_rule_dec18"] = {"pool": f"decoy+paraphrase/{pc}/held", "plain_selector": plain2,
                              "selection_not_missing": bool(any(v["ge_0.9"] for v in plain2.values())),
                              "parts": low2, "parts_below_0.6": [k for k, v in low2.items() if v["below_0.6"]],
                              "comparator_lambda": lam2}
    # ---- sanity
    j_or, j_rd = rt.col.get("oracle_queried"), rt.col.get("random")
    out["sanity"] = {
        "oracle_queried_recall": float(np.mean(rt.IN[:, j_or])) if j_or is not None else None,
        "random_recall": float(np.mean(rt.IN[:, j_rd])) if j_rd is not None else None,
        "random_chance": float(np.mean([r["random_selected"] / r["n_cand"] for r in rows])),
        "nan_items": sum(bool(r.get("nan_scores")) for r in rows),
        "cand_mode_all": sum(r.get("cand_mode") == "all" for r in rows),
    }
    return out


def part_path(trace: Path, uid: str) -> Path:
    return trace / (uid.replace("/", "__") + ".npz")


def load_parts(trace: Path, uids: list[str]) -> list[dict]:
    out = []
    for u in uids:
        p = part_path(trace, u)
        if p.exists():
            with np.load(p) as z:
                out.append({k: z[k] for k in z.files})
    return out


# ------------------------------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a = ap.add_argument
    a("--model", default="google/recurrentgemma-2b")
    a("--dtype", default="bfloat16")
    a("--device", default="cuda")
    a("--tiny", action="store_true", help="random 6-layer RecurrentGemma + byte tokenizer (CPU smoke)")
    a("--tiny-window", type=int, default=384)
    a("--types", default=",".join(TYPES))
    a("--cell", default="beyond", help="comma list of beyond, near, inside")
    a("--n", type=int, default=400, help="items per type x cell")
    a("--seed", type=int, default=0)
    a("--batch", type=int, default=4)
    a("--seq-len", type=int, default=8192)
    a("--sink", type=int, default=4)
    a("--budget", type=int, default=512)
    a("--cap", type=int, default=48)
    a("--n-statements", type=int, default=24)
    a("--passage-min", type=int, default=120)
    a("--passage-max", type=int, default=480)
    a("--band-others", type=int, default=3, help="near / inside: other statements placed in the queried band")
    a("--early-tap", type=int, default=5)
    a("--deep-tap", type=int, default=20)
    a("--lambda", dest="lambdas", default="0,0.03,0.1,0.25,0.5,1")
    a("--n-rep", type=int, default=3, help="repaired infllm: representatives per span")
    a("--expand", type=int, default=4)
    a("--k-frac", type=float, default=0.05)
    a("--circuit-seed", type=int, default=0)
    a("--ring", type=int, default=4096)
    a("--mean-tau", type=float, default=4096.0)
    a("--gate-threshold", default="auto", help="slow-bank gate on dentate salience: a number or auto")
    a("--calib-items", type=int, default=16)
    a("--gate-signal", default="auto", choices=("auto",) + GATE_SIGNALS,
      help="salience gating the slow bank (auto: highest token AUC on the calibration dev items)")
    a("--no-centre", action="store_true")
    a("--lag0", action="store_true")
    a("--slow-ungated", action="store_true")
    a("--raw-bands", action="store_true")
    a("--retriever", default="bge", choices=["bge", "bow", "none"])
    a("--upper-retriever", action="store_true", help="also Qwen3-Embedding-4B (upper reference)")
    a("--ctx-pairs", type=int, default=200)
    a("--chat", default="auto", choices=["auto", "on", "off"])
    a("--filler", default="~/limbic-data/wikitext103_test.txt")
    a("--out", required=True)
    a("--trace-dir", default=None, help="per-item part arrays (default ~/limbic-data/round13_traces/<out stem>)")
    a("--resume", action="store_true", help="skip items already in --out")
    a("--force", action="store_true")
    a("--save-every", type=int, default=4, help="batches between saves")
    a("--boot", type=int, default=2000)
    a("--summarise-only", action="store_true",
      help="recompute the summary of --out from its rows and traces (no model; run after the run has finished)")
    return ap


COMPAT = ("model", "dtype", "tiny", "tiny_window", "seed", "seq_len", "sink", "budget", "cap", "n_statements",
          "band_others", "passage_min", "passage_max", "early_tap", "deep_tap", "lambdas", "n_rep", "expand", "k_frac",
          "circuit_seed", "ring", "mean_tau", "gate_threshold", "gate_signal", "no_centre", "lag0", "slow_ungated",
          "raw_bands", "retriever", "upper_retriever", "chat", "filler", "ctx_pairs")


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.tiny:
        args.dtype = "float32"
        if args.seq_len == 8192:
            args.seq_len, args.budget, args.cap, args.sink, args.n_statements = 4096, 128, 96, 2, 8
            args.passage_min, args.passage_max, args.ring, args.early_tap, args.deep_tap = 200, 600, 512, 1, 4
            args.calib_items, args.band_others = min(args.calib_items, 2), 0
            if args.tiny_window == 384:
                args.tiny_window = 1024                            # byte sentences: the near band needs room
    out = Path(args.out).expanduser()
    if args.summarise_only:                                        # rows + traces only: no model, nothing else changes
        if not out.exists():
            raise SystemExit(f"{out} does not exist")
        res = ResultFile.__new__(ResultFile)
        res.path, res.data, res.force = out, json.loads(out.read_text()), False
        tr = Path(args.trace_dir or res.data.get("trace_dir") or Path("~/limbic-data/round13_traces") / out.stem)
        seed = res.data.get("args", {}).get("seed", args.seed)
        res.data["summary"] = summarise(res.data, tr.expanduser(), seed, args.boot)
        res.save()
        print(f"summary rewritten in {out} ({res.data['summary'].get('n_items', 0)} items)", flush=True)
        return res.data
    out.parent.mkdir(parents=True, exist_ok=True)
    aj = alignment().judge
    trace = Path(args.trace_dir or Path("~/limbic-data/round13_traces") / out.stem).expanduser()
    trace = aj.assert_private(trace, ROOT)
    trace.mkdir(parents=True, exist_ok=True)
    if out.exists() and not (args.resume or args.force or args.summarise_only):
        raise SystemExit(f"{out} exists; pass --resume to continue it or --force to start again")
    t_start = time.time()
    model, tok, facts = load(args)
    W = facts["window"]
    chat = args.chat == "on" or (args.chat == "auto" and bool(getattr(getattr(tok, "tok", None), "chat_template",
                                                                       None)))
    args.chat = "on" if chat else "off"
    compat = {k: getattr(args, k) for k in COMPAT}
    header = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "git": git_state(),
              "versions": versions(), "model": facts, "trace_dir": str(trace),
              "plan": "research/plans/round13_gated_memory_plan.md §4 step 0"}
    res = ResultFile(out, header, compat, force=args.force)
    if args.force:
        res.data["items"] = {}
        res.data.pop("calibration", None)                          # a stale gate threshold must not survive
        res.data.pop("summary", None)
    res.data.setdefault("items", {})
    if any("ranks" in r for r in res.data["items"].values()):
        raise SystemExit(f"{out} holds step-0 rows (another row format); use a new --out (or only --summarise-only)")
    rk = Ranker(model, tok, facts, args)
    prompt = Prompt(tok, chat)
    corpus = Corpus(tok, rk.tables, args.filler)
    icfg = ItemConfig(seq_len=args.seq_len, window=W, sink=args.sink, n_statements=args.n_statements,
                      passage_min=args.passage_min, passage_max=args.passage_max, cap=args.cap, seed=args.seed,
                      band_others=args.band_others)
    builder = ItemBuilder(tok, rk.tables, corpus, prompt, icfg)
    types = [t for t in args.types.split(",") if t]
    cellv = [c for c in args.cell.split(",") if c]
    # ---- gate signal and threshold (stored; reused on resume; dropped by --force)
    if args.gate_threshold == "auto" or args.gate_signal == "auto":
        cal = res.data.get("calibration")
        if cal is None or "signal" not in cal:
            cal_items = []
            for i in range(10 ** 6):
                if len(cal_items) >= args.calib_items:
                    break
                for t in types:
                    it = builder.build(t, cellv[0], i)
                    if it.split == "dev" and len(cal_items) < args.calib_items:
                        cal_items.append(it)
            cal = rk.calibrate(cal_items, args.batch)
            if args.gate_threshold != "auto":                      # a fixed threshold on the chosen signal
                cal.update(threshold=float(args.gate_threshold), fixed=True)
            res.data["calibration"] = cal
            res.save()
        th, sig = float(cal["threshold"]), cal["signal"]
    else:
        th, sig = float(args.gate_threshold), args.gate_signal
        res.data["calibration"] = {"signal": sig, "threshold": th, "fixed": True}
    rk.set_threshold(th, sig)
    print(f"{'tiny' if args.tiny else args.model}: W={W}, attention layers {rk.cap.attn_layers}, L={args.seq_len}, "
          f"types {types}, cells {cellv}, n={args.n}, chat={args.chat}, gate {sig} > {th:.4g}", flush=True)
    todo = []
    for t in types:
        for c in cellv:
            for i in range(args.n):
                uid = f"{t}/{c}/{args.seed}/{i}"
                if uid in res.data["items"] and part_path(trace, uid).exists():
                    continue
                todo.append((t, c, i))
    print(f"{len(todo)} items to run ({len(res.data['items'])} already in {out.name})", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_run, n_done, nb = time.time(), 0, 0
    for k in range(0, len(todo), args.batch):
        items = [builder.build(*x) for x in todo[k:k + args.batch]]
        t0 = time.time()
        recs = rk.batch(items)
        dt = time.time() - t0
        for it, (row, parts) in zip(items, recs):
            row["seconds"] = dt / len(items)
            if res.data.get("scorer_names") not in (None, rk.names):
                raise SystemExit(f"{out} holds rows with another scorer list; use a new --out or --force")
            res.data["items"][it.uid] = clean(row)
            res.data["scorer_names"] = rk.names
            np.savez(part_path(trace, it.uid), **parts)
        n_done += len(items)
        nb += 1
        if nb % args.save_every == 0 or k + args.batch >= len(todo):
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
            res.data["run"] = {"peak_gb": peak, "seconds_per_item": (time.time() - t_run) / max(n_done, 1)}
            res.save()
            print(f"  {n_done}/{len(todo)} items, {(time.time() - t_run) / n_done:.2f} s/item"
                  + (f", peak {peak:.1f} GB" if peak else ""), flush=True)
    rk.cap.remove()
    res.data["summary"] = summarise(res.data, trace, args.seed, args.boot)
    res.data["seconds"] = time.time() - t_start
    res.save()
    sm = res.data["summary"]
    if sm:
        print("sanity:", sm["sanity"], flush=True)
        for fam, w in sm.get("winners", {}).items():
            print(f"  {fam}: {w['scorer']} dev recall {w['dev']['recall']['mean']:.3f}", flush=True)
        print("stop rule:", {k: v for k, v in sm["stop_rule"].items() if k != "plain_dev_variants_ge_0.9"},
              flush=True)
    print(f"saved {out} (labels, ranks and scores only); parts in {trace}", flush=True)
    return res.data


if __name__ == "__main__":
    main()
