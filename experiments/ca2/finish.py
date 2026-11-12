"""C2 and C3-inject (round 12): a residual "finish thinking" direction injected through the CA2 side channel,
screened teacher-forced, then driven by a training-free gain law inside the reasoning span of sampled traces.

Directions (``directions``; forward passes only; block outputs h^L in float32; stored raw, used unit-normalised):
  think      d = mean_i h^L_off(i)[−1] − mean_i h^L_on(i)[−1]: the residual at the last prompt position of one
             question rendered with thinking off (the prompt ends "<think>\\n\\n</think>\\n\\n": about to answer)
             minus thinking on (it ends "<think>\\n": about to reason), over problems held out from the evaluation
             set (ActAdd with a prompt set: the notebook's mean-difference CoT vector with the sign that points
             toward answering; N5 used the opposite sign in no-think mode).
  conclude   d = mean_r [mean_{p∈C_r} h^L_p − mean_{p∈M_r} h^L_p] over held-out traces r whose span the model closed
             itself: C_r = the K positions just before the close marker (c − K … c − 1), M_r = K positions centred
             on the middle of the same span (P + ⌊n/2⌋ − ⌊K/2⌋ …), n the span length; a row needs C_r and M_r in the
             span and at least K positions apart. Sequences are cut after the last position used (causal model: the
             states are those of the whole trace).
  controls   random: the CA2 controller's own seeded unit vector (``CA2Controller.d_rand``, ``random_direction``);
             negated: −d (``inj=-think``).
  Reported per block: raw norms, rows used, split-half cosine (even against odd rows), the mean cosine of the
  per-row differences with d, and cosines between the directions.

Injection (``FinishController``: ``reason.SpanController``, i.e. the CA2 controller; zero parameters):
  h_t ← h_t + g_t ‖h_t‖ d̂ at the output of block L (``CA2Attachment(tap=L, inject_before=L + 1)``), d̂ = ±unit(d).
  Steered positions: the last prompt column and every input position inside the reasoning span (``SpanController``:
  the close marker and everything after it get g = 0), never at or after a row's first stop id (a finished row is
  fed pad ids). Forced-answer probes run with the controller paused: they read the steered cache but are not
  steered. g_t comes from ``controller.GainPipeline`` with smoothing off: g_t = g_min + (g_max − g_min)·r_t at steered
  positions, 0 elsewhere, r_t = s_t of the driving source:
    fixed     s ≡ 1, g_min = 0, g_max = g; ``g=match:<arm>`` takes g = that arm's realised mean gain (Σ g / steered
              steps, pooled over its rows), the fixed-at-matched-mean control.
    law       s_t = max over the listed signals, each in [0, 1] (``make_law``):
                elapsed-N[-δ]  clip((k_t − δ)/N, 0, 1), k_t the steered-step index (reasoning tokens so far)
                consist-k      clip((run − 1)/(k − 1), 0, 1), run = length of the trailing run of equal parsed
                               forced-probe answers (``decode.answer_key``; one probe every --probe-every span
                               tokens, held in between): the state of the consistency detector that fires at run = k
                rep-n-W[-λ]    E_t = λE_{t−1} + (1 − λ)·rep_t over steered positions, rep_t = 1 when the n-gram ending
                               at t occurred in the previous W ids (``regulators.NgramRepeat``)
    yoked     r_t replayed from a law arm's recorded r trajectory of another problem (derangement: item i reads item
              (i + 1) mod N of the problem list, same sample), indexed by the steered step and held at its last value
              (``sources.PositionMeanSource``), with that arm's g_min and g_max: the same gain profile, timing unrelated
              to this trace.
  With g ≡ 0 the hooks return None, so generation is token for token the unsteered arm's under the same seed.

C2 screen (``screen``, teacher-forced). On recorded traces (think.py / reason.py layout), the prompt plus the span up
to and excluding the close marker is fed with the injection at every steered position, as in generation (fixed g),
and at every span position p (the distribution of token p + 1):
  Δ_p = log p_steer(close | x_≤p) − log p_base(close | x_≤p),  KL_p = KL(p_steer ‖ p_base),  ΔNLL_p of the next token.
  Labels from the trace's forced-answer probes, position p holding m = p − P + 1 span tokens:
    correct  formed if m ≥ ttc (the first checkpoint from which every forced answer is correct), not formed if m ≤ the
             last checkpoint before ttc (wrong there); rows never correct are left out
    stable   the same with ttc replaced by the first checkpoint from which the parsed answer equals the final one
  Positions between the two checkpoints are unlabelled. Per row, the mean Δ over formed (F) and not-formed (U)
  positions; E_F, E_U = means over rows; selectivity = E_F / E_U (undefined when E_U ≤ 0; then "sign-selective" when
  E_F > 0); E_F − E_U with a paired bootstrap over rows; the same on paragraph boundaries ("\\n\\n" tokens); the
  design's sel_kl = mean Δ at boundaries / mean KL at non-boundary span positions.
  Gate (pre-registered, design C2): a direction passes at (block, gain) when E_F > 0, the CI of E_F − E_U lies above 0,
  and its selectivity ≥ 2·max(1, best random selectivity) (sign-selective counts as passing).
  Position-matched contrast (review RV, finish.py:321): within a row every formed position comes after every
  not-formed one, and the injection acts from the last prompt column on, so E_F − E_U also carries "more tokens
  elapsed, more steered context". Span offsets m are binned ((m − 1) // ``--pos-bin``; the width divides the probe
  spacing, so labels change on bin edges); per bin b, F_b and U_b are the means over rows of the row's mean Δ on its
  formed / not-formed positions in that bin (a row counts in a bin when the label holds ≥ half of it), and
    E_match = Σ_b w_b (F_b − U_b) / Σ_b w_b,   w_b = n_F,b · n_U,b / (n_F,b + n_U,b)   (rows with F / U in bin b),
  i.e. formed against not-formed at the same offset, compared across rows; row bootstrap. ``pass_matched`` = ``pass``
  and the CI of E_match above 0 and E_match above every random direction's.
  Rows: the screen refuses the evaluation problems. It skips the problems of the direction file's ``heldout_from``
  (the C3 evaluation set) and of ``--exclude-from``, so it runs on held-out traces (reason_base_*/none, problems
  200–399); ``--allow-eval-rows`` keeps them and marks the report as a dev screen. The rows the conclude direction
  was built from are counted, and conclude's selectivity is also reported on the rows not used to build it
  (``fresh``).

C3-inject (``run``). think.py's runner (``think.main``) with injection arms, by wrapping it, not editing it (as
evolve/run.py wraps pretrained_loop): ``think.parse_arm``, ``think.run_arm``, ``think.run_batch`` and, inside an
injection arm, ``decode.forked_probe`` / ``decode.probe_prefixes`` (to pause the controller and feed the consistency
state) are replaced for the call. An arm is a think.py arm plus the keys
    inj=[-]think|[-]conclude|[-]random   block=<L>   rseed=<s> (random direction seed, default 0)
    g=<g> | g=match:<label>               fixed gain
    gmin=<a>;gmax=<b>;law=<sig>[+<sig>]   dynamic gain
    yoke=<label>                          the yoked control of a law arm run earlier in the same results JSON
Per item, think.py's record plus inj_n_steered, inj_gain_mean / max / last, inj_r_mean, inj_t_gmax (first steered step
at g_max); per-token g and r in ``<trace dir>/<arm>/gain.npz`` (outside the repository). An arm whose law reads the
consistency state is charged the probes it consulted: tokens_total = n_gen + probes while the span was open ×
--probe-cost (the items.jsonl copy in the trace keeps think.py's value; the results JSON is authoritative).
After the arms: arms imported from earlier think.py runs (``--import file.json:label,...``; same problems, seeds,
batches and caps, checked), the truncation frontier and the pre-registered comparison:
  tokens at matched accuracy, H = T_frontier(acc_arm) − tokens_arm (positive: fewer tokens than the frontier at that
  accuracy), with T_frontier the inverse of the monotone envelope of the truncation arms' (mean tokens_total, acc),
  linear between points; and the vertical distance acc_arm − A_frontier(tokens_arm); bootstrap over problems with
  the frontier recomputed on each resample; for the dynamic arm against the logit-bias arm and against truncation at
  256, with the random-direction arm as the attribution control; plus think.py's paired comparisons.

Usage
  CUDA_VISIBLE_DEVICES=1 python experiments/ca2/finish.py directions --model Qwen/Qwen3-1.7B --name c2dirs_q3 \\
      --heldout-from results/evolve/round12/stage1/c3dev_q3.json \\
      --conclude-traces ~/limbic-data/ca2_traces/reason_base_q3/none
  CUDA_VISIBLE_DEVICES=1 python experiments/ca2/finish.py screen --model Qwen/Qwen3-1.7B --name c2screen_q3 \\
      --dirs ~/limbic-data/ca2_dirs/c2dirs_q3.pt --traces ~/limbic-data/ca2_traces/reason_base_q3/none
  CUDA_VISIBLE_DEVICES=1 python experiments/ca2/finish.py run --model Qwen/Qwen3-1.7B --name c3inj_q3 --batch 16 \\
      --probe-batch 8 --dirs ~/limbic-data/ca2_dirs/c2dirs_q3.pt --preset c3inj --inj-dir think --inj-block 14 \\
      --g-max 0.2 --law consist-3 --import results/evolve/round12/stage1/c3dev_q3.json:t06,f256,f512,f1024,f2048,biasc3
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
sys.path.insert(0, str(EXP))
from ca2 import decode as D  # noqa: E402
from ca2 import reason as R  # noqa: E402
from ca2 import think as T  # noqa: E402
from ca2.common import _sha256, alignment, git_state, paired_boot  # noqa: E402
from ca2.controller import CA2Attachment  # noqa: E402
from ca2.regulators import NgramRepeat, leaky_scan  # noqa: E402
from ca2.sources import ConstantSource, PositionMeanSource, Source, StepContext  # noqa: E402

DIR_ROOT = Path("~/limbic-data/ca2_dirs")
STAGE = ROOT / "results" / "evolve" / "round12" / "stage1"
KINDS = ("think", "conclude")
INJ_KEYS = ("inj", "block", "g", "gmin", "gmax", "law", "yoke", "rseed")
_THINK_PARSE_ARM = T.parse_arm                        # think.py's own parser (T.parse_arm is wrapped during a run)


# ------------------------------------------------------------------------------------------- directions
def random_direction(d_model: int, seed: int) -> torch.Tensor:
    """The CA2 controller's random control: ``F.normalize(randn(d, Generator(seed)))`` (== ``ctrl.d_rand``)."""
    g = torch.Generator().manual_seed(int(seed))
    return F.normalize(torch.randn(int(d_model), generator=g), dim=0)


def default_blocks(n: int) -> list[int]:
    return sorted({n // 2, (5 * n) // 8, (3 * n) // 4})


@torch.no_grad()
def states_at(model, blocks, seqs: Sequence[Sequence[int]], positions: Sequence[Sequence[int]], layers: Sequence[int],
              pad: int, batch: int = 8) -> dict[int, list[torch.Tensor]]:
    """Block outputs (float32, CPU) of each sequence at the given indices: {L: [(len(positions[i]), D)]}.

    Each sequence is cut after its last listed index (the model is causal, so the states are those of the whole
    sequence), batches are left-padded (positions counted over real tokens), sorted by length."""
    dev = model.get_input_embeddings().weight.device
    out: dict[int, list] = {L: [None] * len(seqs) for L in layers}
    cut = [list(s[: max(p) + 1]) for s, p in zip(seqs, positions)]
    order = sorted(range(len(cut)), key=lambda i: len(cut[i]))
    cap: dict[int, torch.Tensor] = {}
    hs = [blocks[L].register_forward_hook(lambda m, a, o, L=L: cap.__setitem__(L, o[0] if isinstance(o, tuple) else o))
          for L in layers]
    try:
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            ids, mask = R.pad_left([cut[i] for i in idx], pad)
            pos = alignment().probe.position_ids(mask)
            model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev), use_cache=False,
                  logits_to_keep=1)
            W = ids.shape[1]
            for r, i in enumerate(idx):
                off = W - len(cut[i])
                at = torch.as_tensor([off + p for p in positions[i]], device=dev)
                for L in layers:
                    out[L][i] = cap[L][r].index_select(0, at).float().cpu()
    finally:
        for h in hs:
            h.remove()
    return out


def conclude_positions(n_prompt: int, n_reason: int, close_idx: int | None, k: int
                       ) -> tuple[list[int], list[int]] | None:
    """(C, M): the k sequence indices just before the close marker and k indices centred on the span's middle;
    None when the row has no close or the windows do not fit k apart inside the span."""
    if close_idx is None or k <= 0:
        return None
    pre = list(range(close_idx - k, close_idx))
    c = n_prompt + n_reason // 2
    mid = list(range(c - k // 2, c - k // 2 + k))
    if pre[0] < n_prompt or mid[0] < n_prompt or mid[-1] + k >= pre[0] or close_idx > n_prompt + n_reason:
        return None
    return pre, mid


def _dir_stats(diffs: torch.Tensor) -> dict:
    """Per-row differences (N, D) -> mean d and reliability statistics."""
    d = diffs.mean(0)
    n = diffs.shape[0]
    st = {"n_rows": int(n), "raw_norm": float(d.norm()),
          "row_cos_mean": float(F.cosine_similarity(diffs, d[None], dim=-1).mean()) if n else None,
          "split_half_cos": (float(F.cosine_similarity(diffs[0::2].mean(0), diffs[1::2].mean(0), dim=0))
                             if n >= 4 else None)}
    return {"d": d, "stats": st}


def think_direction(model, blocks, prompts_on: Sequence[Sequence[int]], prompts_off: Sequence[Sequence[int]],
                    layers: Sequence[int], pad: int, batch: int = 8) -> dict[int, dict]:
    """{L: {"d": mean h_off[−1] − mean h_on[−1], "stats"}} (row i of both lists is the same question)."""
    if len(prompts_on) != len(prompts_off) or not prompts_on:
        raise ValueError("think direction: the same non-empty problem list rendered both ways")
    on = states_at(model, blocks, prompts_on, [[len(p) - 1] for p in prompts_on], layers, pad, batch)
    off = states_at(model, blocks, prompts_off, [[len(p) - 1] for p in prompts_off], layers, pad, batch)
    return {L: _dir_stats(torch.cat(off[L], 0) - torch.cat(on[L], 0)) for L in layers}


def conclude_direction(model, blocks, rows: Sequence["TraceRow"], layers: Sequence[int], pad: int, k: int = 16,
                       batch: int = 1) -> dict[int, dict]:
    """{L: {"d": mean_r (mean h at C_r − mean h at M_r), "stats"}} over rows that closed the span themselves."""
    use, seqs, pos = [], [], []
    for r in rows:
        cm = conclude_positions(r.n_prompt, r.n_reason, r.close_idx, k) if r.natural else None
        if cm is None:
            continue
        use.append(r)
        seqs.append(r.ids.tolist())
        pos.append(cm[0] + cm[1])
    if not use:
        return {L: {"d": None, "stats": {"n_rows": 0}} for L in layers}
    st = states_at(model, blocks, seqs, pos, layers, pad, batch)
    out = {}
    for L in layers:
        diffs = torch.stack([h[:k].mean(0) - h[k:].mean(0) for h in st[L]])
        out[L] = _dir_stats(diffs)
    return out


class Directions:
    """A saved direction file: raw directions per kind and block, unit-normalised on use."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        blob = torch.load(self.path, map_location="cpu", weights_only=False)
        self.blob = blob
        self.d_model, self.blocks, self.model = int(blob["d_model"]), [int(x) for x in blob["blocks"]], blob["model"]
        self.sha256 = _sha256(self.path)

    def get(self, kind: str, block: int) -> torch.Tensor:
        d = self.blob["dirs"].get(kind, {}).get(int(block))
        if d is None:
            raise KeyError(f"{self.path.name}: no {kind!r} direction at block {block} (blocks {self.blocks})")
        return F.normalize(d.float(), dim=0)

    def kinds(self) -> list[str]:
        return [k for k in KINDS if any(v is not None for v in self.blob["dirs"].get(k, {}).values())]


# ------------------------------------------------------------------------------------------- traces
@dataclass
class TraceRow:
    uid: str
    sample: int
    ids: np.ndarray                 # prompt + response ids
    n_prompt: int
    n_reason: int
    close_idx: int | None           # sequence index of the first close marker after the prompt
    natural: bool                   # the model closed the span itself (no forced close, not cut by the cap)
    correct: bool
    cks: list
    probe_ok: list
    probe_answers: list
    boundary: np.ndarray


def _arm_items_dir(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if (p / "items.jsonl").exists() else p / "tokens"


def read_trace(path: str | Path, close_id: int | None = None, uids: set | None = None,
               exclude: set | None = None) -> list[TraceRow]:
    """Rows of one arm's trace (``<trace dir>/<arm>`` or its ``tokens`` folder): ids, span, close and probes."""
    d = _arm_items_dir(path)
    items = [json.loads(line) for line in (d / "items.jsonl").read_text().splitlines() if line.strip()]
    shards: dict[int, dict] = {}
    out = []
    for it in items:
        if (uids is not None and it["uid"] not in uids) or (exclude is not None and it["uid"] in exclude):
            continue
        if "tok_shard" not in it:
            continue
        k = int(it["tok_shard"])
        if k not in shards:
            z = np.load(d / f"tokens_{k:04d}.npz")
            shards[k] = {"ids": z["bk__ids"], "bnd": z["bk__boundary"]}
        a, n = int(it["tok_offset"]), int(it["n_tok"])
        ids = shards[k]["ids"][a:a + n].astype(np.int64)
        P, nr = int(it["n_prompt"]), int(it["n_reason"])
        ci = None
        if close_id is not None:
            hit = np.nonzero(ids[P:] == close_id)[0]
            ci = int(P + hit[0]) if hit.size else None
        elif it.get("closed"):
            ci = P + nr
        natural = bool(it.get("closed") and ci is not None and it.get("forced_from") is None
                       and not it.get("forced") and not it.get("truncated"))
        out.append(TraceRow(it["uid"], int(it.get("sample", 0)), ids, P, nr, ci, natural, bool(it.get("correct")),
                            list(it.get("cks") or []), list(it.get("probe_ok") or []),
                            list(it.get("probe_answers") or []), shards[k]["bnd"][a:a + n].astype(np.float32)))
    return out


def formation_labels(row: TraceRow, rule: str = "correct", end: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(formed, unformed) bool masks over sequence indices [0, end) (module docstring); span positions only."""
    end = len(row.ids) if end is None else end
    formed, unformed = np.zeros(end, bool), np.zeros(end, bool)
    cks = row.cks
    if not cks:
        return formed, unformed
    if rule == "correct":
        ttc, cens = R.ttc_from(cks, row.probe_ok)
        if cens or ttc is None:
            return formed, unformed
        i = cks.index(ttc)
    elif rule == "stable":
        keys = [D.answer_key(a) for a in row.probe_answers]
        if not keys or keys[-1] is None:
            return formed, unformed
        i = len(keys) - 1
        while i > 0 and keys[i - 1] == keys[-1]:
            i -= 1
    else:
        raise ValueError("rule: correct | stable")
    lo = cks[i - 1] if i > 0 else None
    hi = cks[i]
    m = np.arange(end) - row.n_prompt + 1                  # span tokens held at index p
    span = (m >= 1) & (m <= row.n_reason)
    formed = span & (m >= hi)
    if lo is not None:
        unformed = span & (m <= lo)
    return formed, unformed


# ------------------------------------------------------------------------------------------- screen measures
def row_effects(delta: np.ndarray, kl: np.ndarray, dnll: np.ndarray, formed: np.ndarray, unformed: np.ndarray,
                boundary: np.ndarray, span: np.ndarray) -> dict:
    """Per-row means of Δ log p(close) on the label masks (NaN where a mask is empty), KL and ΔNLL."""
    m = lambda x, k: float(np.mean(x[k])) if k.any() else float("nan")
    bnd = boundary > 0
    return {"F": m(delta, formed), "U": m(delta, unformed), "Fb": m(delta, formed & bnd),
            "Ub": m(delta, unformed & bnd),
            "all": m(delta, span), "bnd": m(delta, span & bnd), "kl": m(kl, span), "kl_nonbnd": m(kl, span & ~bnd),
            "dnll": m(dnll, span & np.isfinite(dnll)), "nF": int(formed.sum()), "nU": int(unformed.sum())}


def _ratio_ci(f: np.ndarray, u: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float | None, float | None]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, f.size, size=(n, f.size))
    ef, eu = f[idx].mean(1), u[idx].mean(1)
    ok = eu > 0
    if ok.mean() < 0.95:
        return None, None
    lo, hi = np.quantile(ef[ok] / eu[ok], [0.025, 0.975])
    return float(lo), float(hi)


def selectivity(rows: Sequence[dict], seed: int = 0) -> dict:
    """E_F, E_U, selectivity E_F / E_U and E_F − E_U over rows with both labels (module docstring)."""
    use = [r for r in rows if np.isfinite(r["F"]) and np.isfinite(r["U"])]
    f = np.asarray([r["F"] for r in use], np.float64)
    u = np.asarray([r["U"] for r in use], np.float64)
    out = {"n_rows": len(use)}
    if not use:
        return {**out, "E_F": None, "E_U": None, "ratio": None, "sign_selective": False, "diff": paired_boot([], [])}
    EF, EU = float(f.mean()), float(u.mean())
    out.update(E_F=EF, E_U=EU, ratio=EF / EU if EU > 0 else None, sign_selective=bool(EU <= 0 < EF),
               diff=paired_boot(f, u, seed=seed))
    out["ratio_lo"], out["ratio_hi"] = _ratio_ci(f, u, seed=seed) if EU > 0 else (None, None)
    fb = np.asarray([r["Fb"] for r in use], np.float64)
    ub = np.asarray([r["Ub"] for r in use], np.float64)
    ok = np.isfinite(fb) & np.isfinite(ub)
    out["E_Fb"] = float(fb[ok].mean()) if ok.any() else None
    out["E_Ub"] = float(ub[ok].mean()) if ok.any() else None
    out["n_rows_b"] = int(ok.sum())
    return out


def binned_effects(delta: np.ndarray, formed: np.ndarray, unformed: np.ndarray, m: np.ndarray, width: int,
                   nbins: int) -> tuple[list[float], list[float]]:
    """Per offset bin (m − 1) // width: the row's mean Δ on its formed and its not-formed positions; NaN when the
    label holds fewer than width / 2 positions of the bin (a checkpoint's stray position or the span's ragged end),
    so the positions a row contributes cover most of the bin and F and U are compared at the same offsets."""
    b = np.clip((np.asarray(m) - 1) // int(width), 0, nbins - 1)
    need = max(1, int(width) // 2)
    out_f, out_u = [], []
    for k in range(nbins):
        kf, ku = formed & (b == k), unformed & (b == k)
        out_f.append(float(delta[kf].mean()) if kf.sum() >= need else float("nan"))
        out_u.append(float(delta[ku].mean()) if ku.sum() >= need else float("nan"))
    return out_f, out_u


def _matched(fb: np.ndarray, ub: np.ndarray) -> tuple[float | None, int]:
    """E_match of (rows, bins) arrays (module docstring) and the number of bins holding both labels."""
    nf, nu = np.isfinite(fb).sum(0), np.isfinite(ub).sum(0)
    ok = (nf > 0) & (nu > 0)
    if not ok.any():
        return None, 0
    with np.errstate(invalid="ignore"):
        F_ = np.nansum(fb[:, ok], 0) / nf[ok]
        U_ = np.nansum(ub[:, ok], 0) / nu[ok]
    w = nf[ok] * nu[ok] / (nf[ok] + nu[ok])
    return float((w * (F_ - U_)).sum() / w.sum()), int(ok.sum())


def matched_contrast(rows: Sequence[dict], n_boot: int = 4000, seed: int = 0) -> dict:
    """Formed − not-formed Δ at matched span offsets, across rows (module docstring), with a row bootstrap."""
    use = [r for r in rows if r.get("Fbin") is not None]
    if not use:
        return {"diff": None, "lo": None, "hi": None, "n_rows": 0, "n_bins": 0}
    fb = np.asarray([r["Fbin"] for r in use], np.float64)
    ub = np.asarray([r["Ubin"] for r in use], np.float64)
    est, nb = _matched(fb, ub)
    if est is None:
        return {"diff": None, "lo": None, "hi": None, "n_rows": len(use), "n_bins": 0}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(use), len(use))
        v, _n = _matched(fb[i], ub[i])
        if v is not None:
            boots.append(v)
    lo, hi = (np.quantile(boots, [0.025, 0.975]) if boots else (None, None))
    return {"diff": est, "lo": None if lo is None else float(lo), "hi": None if hi is None else float(hi),
            "n_rows": len(use), "n_bins": nb, "n_boot_ok": len(boots)}


def screen_gate(sel: dict, random_sels: Sequence[dict], factor: float = 2.0) -> dict:
    """The pre-registered C2 gate of one direction at one (block, gain) against the random directions there."""
    rr = [s["ratio"] if s.get("ratio") is not None else (math.inf if s.get("sign_selective") else 1.0)
          for s in random_sels if s.get("E_F") is not None]
    best = max(rr) if rr else 1.0
    mine = sel.get("ratio")
    mine = math.inf if sel.get("sign_selective") else mine
    ok = (sel.get("E_F") is not None and sel["E_F"] > 0 and sel["diff"].get("lo") is not None
          and sel["diff"]["lo"] > 0 and mine is not None and mine >= factor * max(1.0, best))
    if ok and math.isinf(best):                   # a random direction is sign-selective too: compare E_F instead
        ef = max(s["E_F"] for s in random_sels if s.get("sign_selective"))
        ok = sel["E_F"] >= factor * ef
    ef_rand = max([s["E_F"] for s in random_sels if s.get("E_F") is not None], default=None)
    strict = ok and (ef_rand is None or sel["E_F"] > ef_rand)
    mt = sel.get("matched") or {}
    mt_rand = max([(s.get("matched") or {}).get("diff") for s in random_sels
                   if (s.get("matched") or {}).get("diff") is not None], default=None)
    matched = (ok and mt.get("lo") is not None and mt["lo"] > 0
               and (mt_rand is None or mt["diff"] > mt_rand))
    return {"pass": bool(ok), "pass_strict": bool(strict), "pass_matched": bool(matched),
            "matched_diff": mt.get("diff"), "matched_lo": mt.get("lo"), "best_random_matched": mt_rand,
            "selectivity": None if mine is None or math.isinf(mine)
            else mine, "best_random": None if math.isinf(best) else best, "best_random_E_F": ef_rand,
            "factor": factor}


# ------------------------------------------------------------------------------------------- controller
class _Off:
    """Stand-in for the controller's trace-only repetition signals (zeros; nothing carried)."""

    def __call__(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], x.shape[1], device=x.device)

    def reset(self) -> None:
        pass

    def state_numel(self) -> int:
        return 0


class FinishController(R.SpanController):
    """CA2 injection along a fixed direction, gated to the reasoning span and to unfinished rows (module docstring).

    ``direction`` (D,) is unit-normalised; None takes the controller's seeded random vector (``d_rand``, seed
    ``seed``). ``stops``: ids that finish a row. ``paused``: every hook is a no-op (probes). ``log_gain``: keeps
    (g, r, steer) of the last column of every call (one (3, B) device tensor per call). ``light``: the trace-only
    repetition signals (n-gram, hidden) are not computed (they drive nothing here)."""

    def __init__(self, d_model: int, direction: torch.Tensor | None = None, sign: float = 1.0,
                 stops: Sequence[int] = (), light: bool = True, log_gain: bool = False, **kw) -> None:
        self.paused, self.log_gain = False, bool(log_gain)
        kw.setdefault("sources", ())
        super().__init__(d_model, direction="random", **kw)
        d = self.d_rand.clone() if direction is None else F.normalize(direction.detach().float().flatten(), dim=0)
        if d.numel() != self.d_model:
            raise ValueError(f"direction has {d.numel()} entries, the model {self.d_model}")
        self.register_buffer("d_fixed", d * float(sign))
        self.register_buffer("stop_ids", torch.as_tensor(sorted({int(s) for s in stops}), dtype=torch.long))
        if light:
            self.ngram, self.hrep = _Off(), _Off()

    def reset(self) -> None:
        super().reset()
        self.finished: torch.Tensor | None = None
        self.glog: list[torch.Tensor] = []

    def state_numel(self) -> int:
        return super().state_numel() + (0 if self.finished is None else self.finished.numel())

    def _direction(self, x32, bstep, wstep, gist) -> torch.Tensor:
        return self.d_fixed

    def _positions(self, B: int, T: int, dev) -> dict:
        if self._pos is not None:
            return self._pos
        pos = super()._positions(B, T, dev)
        if not self.stop_ids.numel():
            return pos
        ids = self._ids if self._ids is not None and tuple(self._ids.shape) == (B, T) else None
        # stop ids count from the first generated token on: the prompt (first call) holds the template's own
        # turn ends (Qwen's <|im_end|> after the user turn) and left padding
        hit = (torch.isin(ids.to(dev), self.stop_ids.to(dev)) & (pos["mask"] > 0)) \
            if ids is not None and not pos["first"] else torch.zeros(B, T, dtype=torch.bool, device=dev)
        before = self.finished if self.finished is not None else torch.zeros(B, dtype=torch.bool, device=dev)
        fin = before.to(dev)[:, None] | (hit.long().cumsum(1) > 0)
        steer = pos["steer"] * (~fin).float()
        ns = self.n_steer if self.n_steer is not None else torch.zeros(B, device=dev)
        pos["steer"] = steer
        pos["step"] = (ns[:, None] + steer.cumsum(1) - steer).long()
        self.finished = fin[:, -1].detach()
        return pos

    # the hooks are no-ops while paused
    def begin_call(self, ids: torch.Tensor | None = None) -> None:
        if not self.paused:
            super().begin_call(ids)

    def end_call(self) -> None:
        if not self.paused:
            super().end_call()

    def observe(self, h: torch.Tensor) -> None:
        if not self.paused:
            super().observe(h)

    def inject(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.paused:
            return None
        y = super().inject(x)
        if self.log_gain:
            c = self.current
            self.glog.append(torch.stack([c["g"][:, -1], c["r"][:, -1], c["steer"][:, -1]]).detach())
        return y

    def open_rows(self) -> torch.Tensor:
        """(B,) bool on CPU: rows whose span is open and that have not finished, after the last fed token."""
        n = self.n_real.shape[0] if self.n_real is not None else 0
        closed = self.span_closed.cpu() if self.span_closed is not None else torch.zeros(n, dtype=torch.bool)
        fin = self.finished.cpu() if self.finished is not None else torch.zeros(n, dtype=torch.bool)
        return ~(closed | fin)

    def gain_log(self) -> torch.Tensor:
        """(3, B, calls) float32 CPU: g, r and steer of the last column of every call since reset."""
        if not self.glog:
            return torch.zeros(3, 0, 0)
        return torch.stack(self.glog, -1).float().cpu()


# ------------------------------------------------------------------------------------------- gain laws
class ElapsedSource(Source):
    """s_t = clip((k_t − δ)/N, 0, 1), k_t the steered-step index (reasoning tokens so far)."""

    def __init__(self, ramp: int, delay: int = 0, name: str = "elapsed") -> None:
        if ramp <= 0 or delay < 0:
            raise ValueError("elapsed: need N > 0 and δ >= 0")
        self.ramp, self.delay, self.name = int(ramp), int(delay), name

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        return {"s": ((ctx.step.float() - self.delay) / self.ramp).clamp(0.0, 1.0)}


class ConsistSource(Source):
    """s = clip((run − 1)/(k − 1), 0, 1) per row, run = trailing run of equal parsed probe answers (``observe``)."""

    def __init__(self, k: int = 3, name: str = "consist") -> None:
        if k < 2:
            raise ValueError("consist: k >= 2")
        self.k, self.name = int(k), name
        self.reset()

    def reset(self) -> None:
        self.hist: list[list] | None = None
        self.level: torch.Tensor | None = None

    def set_rows(self, B: int) -> None:
        self.hist = [[] for _ in range(B)]
        self.level = torch.zeros(B)

    def run_length(self, b: int) -> int:
        h = self.hist[b]
        if not h or h[-1] is None:
            return 0
        n = 1
        while n < len(h) and h[-1 - n] == h[-1]:
            n += 1
        return n

    def observe(self, rows: Sequence[int], answers: Sequence[str | None]) -> None:
        for b, a in zip(rows, answers):
            self.hist[b].append(D.answer_key(a))
            self.level[b] = min(1.0, max(0.0, (self.run_length(b) - 1) / (self.k - 1)))

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        B, T = ctx.mask.shape
        lv = self.level if self.level is not None and self.level.numel() == B else torch.zeros(B)
        return {"s": lv.to(ctx.mask.device)[:, None].expand(B, T).clone()}


class RepeatSource(Source):
    """E_t = λE_{t−1} + (1 − λ) rep_t over steered positions (held elsewhere), rep_t from ``NgramRepeat(n, W)``."""

    def __init__(self, n: int = 4, window: int = 64, leak: float = 0.98, name: str = "repeat") -> None:
        if not 0.0 <= leak < 1.0:
            raise ValueError("rep: λ in [0, 1)")
        self.det, self.leak, self.name = NgramRepeat(n, window), float(leak), name
        self.reset()

    def reset(self) -> None:
        self.det.reset()
        self.E: torch.Tensor | None = None

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        B, T = ctx.mask.shape
        rep = self.det(ctx.ids, ctx.mask) if ctx.ids is not None else torch.zeros(B, T, device=ctx.mask.device)
        st = ctx.steer
        coef = torch.where(st > 0, torch.full_like(st, self.leak), torch.ones_like(st))
        E = leaky_scan(coef, (1.0 - self.leak) * rep * st, self.E)
        self.E = E[:, -1].detach()
        return {"s": E.clamp(0.0, 1.0), "rep": rep}


class MaxSource(Source):
    """s_t = max over the parts' s_t; each part's s is kept as an extra signal."""

    def __init__(self, parts: Sequence[Source], name: str = "law") -> None:
        self.parts, self.name = list(parts), name

    def reset(self) -> None:
        for p in self.parts:
            p.reset()

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        outs = [p(ctx) for p in self.parts]
        out = {"s": torch.stack([o["s"] for o in outs]).amax(0)}
        for p, o in zip(self.parts, outs):
            out[p.name] = o["s"]
        return out


def make_law(spec: str) -> MaxSource:
    """``elapsed-N[-δ]`` | ``consist-k`` | ``rep-n-W[-λ]``, joined by '+' (module docstring)."""
    parts: list[Source] = []
    for i, p in enumerate(spec.split("+")):
        a = p.split("-")
        try:
            if a[0] == "elapsed" and len(a) in (2, 3):
                parts.append(ElapsedSource(int(a[1]), int(a[2]) if len(a) == 3 else 0, name=f"elapsed{i}"))
            elif a[0] == "consist" and len(a) == 2:
                parts.append(ConsistSource(int(a[1]), name=f"consist{i}"))
            elif a[0] == "rep" and len(a) in (3, 4):
                parts.append(RepeatSource(int(a[1]), int(a[2]), float(a[3]) if len(a) == 4 else 0.98,
                                          name=f"repeat{i}"))
            else:
                raise ValueError
        except ValueError:
            raise ValueError(f"law part {p!r}: elapsed-N[-delay] | consist-k | rep-n-W[-leak]") from None
    return MaxSource(parts)


def consist_parts(src: Source) -> list[ConsistSource]:
    if isinstance(src, ConsistSource):
        return [src]
    return [p for p in getattr(src, "parts", []) if isinstance(p, ConsistSource)]


class YokeSource(Source):
    """A replayed r trajectory per row (final): ``PositionMeanSource`` over a (B, N) table set per batch."""

    final = True
    name = "yoked"

    def __init__(self) -> None:
        self.inner: PositionMeanSource | None = None

    def set_table(self, table: torch.Tensor) -> None:
        self.inner = PositionMeanSource(table, index="step", name="yoked")

    def __call__(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        if self.inner is None:
            raise RuntimeError("yoked source: no donor table for this batch")
        return self.inner(ctx)


# ------------------------------------------------------------------------------------------- arms
@dataclass
class InjectSpec:
    kind: str                       # think | conclude | random
    sign: float
    block: int | None
    mode: str                       # fixed | law | yoke
    g: float | None = None
    match: str | None = None
    g_min: float | None = None
    g_max: float | None = None
    law: str | None = None
    yoke: str | None = None
    rseed: int = 0

    def as_dict(self) -> dict:
        return dict(vars(self))


def parse_inject(kv: dict[str, str], spec: str) -> InjectSpec:
    if "inj" not in kv:
        raise ValueError(f"{spec!r}: injection keys need inj=<direction>")
    name = kv["inj"]
    sign = -1.0 if name.startswith("-") else 1.0
    name = name.lstrip("-+")
    if name not in KINDS + ("random",):
        raise ValueError(f"{spec!r}: inj=[-]think | [-]conclude | [-]random")
    s = InjectSpec(name, sign, int(kv["block"]) if "block" in kv else None, "fixed", rseed=int(kv.get("rseed", 0)))
    modes = [m for m, keys in (("fixed", ("g",)), ("law", ("law", "gmin", "gmax")), ("yoke", ("yoke",)))
             if any(k in kv for k in keys)]
    if len(modes) != 1:
        raise ValueError(f"{spec!r}: give exactly one of g=..., law=...;gmin=...;gmax=..., yoke=<label>")
    s.mode = modes[0]
    if s.mode == "fixed":
        if kv["g"].startswith("match:"):
            s.match = kv["g"][6:]
        else:
            s.g = float(kv["g"])
            if not 0.0 <= s.g <= 1.0:
                raise ValueError(f"{spec!r}: g in [0, 1]")
    elif s.mode == "law":
        if not all(k in kv for k in ("law", "gmin", "gmax")):
            raise ValueError(f"{spec!r}: a law arm needs law, gmin and gmax")
        s.law, s.g_min, s.g_max = kv["law"], float(kv["gmin"]), float(kv["gmax"])
        if not 0.0 <= s.g_min <= s.g_max <= 1.0:
            raise ValueError(f"{spec!r}: need 0 <= gmin <= gmax <= 1")
        make_law(s.law)
    else:
        s.yoke = kv["yoke"]
        if "gmin" in kv or "gmax" in kv:
            raise ValueError(f"{spec!r}: a yoked arm takes its donor's gain range")
    return s


def parse_finish_arm(spec: str) -> T.ThinkArm:
    """A think.py arm, plus ``.inject`` (an ``InjectSpec`` or None) from the injection keys."""
    label, _, body = spec.strip().partition("@")
    mine, rest = {}, []
    for kv in [x for x in body.split(";") if x.strip()]:
        k, _, v = kv.partition("=")
        if k.strip() in INJ_KEYS:
            mine[k.strip()] = v.strip()
        else:
            rest.append(kv)
    arm = _THINK_PARSE_ARM(f"{label}@{';'.join(rest)}")
    arm.spec = spec.strip()
    arm.inject = parse_inject(mine, spec) if mine else None
    if arm.inject is not None and not arm.thinking:
        raise ValueError(f"{spec!r}: injection acts on the reasoning span (think=on)")
    return arm


def check_arm_order(arms: Sequence[T.ThinkArm]) -> None:
    """yoke= and g=match: must name an earlier law / injection arm."""
    seen: dict[str, T.ThinkArm] = {}
    for a in arms:
        s = getattr(a, "inject", None)
        if s is not None and s.yoke is not None:
            d = seen.get(R.safe_key(s.yoke))
            if d is None or d.inject is None or d.inject.mode != "law":
                raise ValueError(f"{a.spec!r}: yoke={s.yoke} must name an earlier law arm")
        if s is not None and s.match is not None:
            d = seen.get(R.safe_key(s.match))
            if d is None or d.inject is None:
                raise ValueError(f"{a.spec!r}: g=match:{s.match} must name an earlier injection arm")
        seen[a.key] = a


def preset_arms(base: str, direction: str, block: int, g_max: float, law: str, g_fixed: float | None = None
                ) -> list[str]:
    """The C3-inject arm set: dynamic (primary), fixed at its realised mean, yoked, random direction, negated
    direction, an elapsed-only dynamic, and optionally a fixed gain."""
    neg = direction[1:] if direction.startswith("-") else "-" + direction
    inj = f"inj={direction};block={block}"
    dyn = f"gmin=0;gmax={g_max:g};law={law}"
    arms = [f"dyn@{base};{inj};{dyn}", f"fixm@{base};{inj};g=match:dyn", f"yoke@{base};{inj};yoke=dyn",
            f"rand@{base};inj=random;block={block};{dyn}", f"neg@{base};inj={neg};block={block};{dyn}",
            f"el@{base};{inj};gmin=0;gmax={g_max:g};law=elapsed-1024"]
    if g_fixed is not None:
        arms.append(f"fix@{base};{inj};g={g_fixed:g}")
    return arms


# ------------------------------------------------------------------------------------------- injection session
def _read_report(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"arms": {}}


def _arm_by_label(report: dict, label: str) -> dict | None:
    for res in report.get("arms", {}).values():
        if res.get("key") == R.safe_key(label) and res.get("complete"):
            return res
    return None


def load_gain(path: Path) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    z = np.load(path)
    out = {}
    for i, key in enumerate(z["keys"].tolist()):
        uid, _, k = key.rpartition("|")
        a, b = int(z["offsets"][i]), int(z["offsets"][i + 1])
        out[(uid, int(k))] = (z["g"][a:b].astype(np.float32), z["r"][a:b].astype(np.float32))
    return out


def save_gain(path: Path, traj: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]) -> dict:
    keys = list(traj)
    off = np.cumsum([0] + [len(traj[k][0]) for k in keys])
    g = np.concatenate([traj[k][0] for k in keys]) if keys else np.zeros(0, np.float32)
    r = np.concatenate([traj[k][1] for k in keys]) if keys else np.zeros(0, np.float32)
    np.savez(path, keys=np.asarray([f"{u}|{k}" for u, k in keys]), offsets=off, g=g.astype(np.float32),
             r=r.astype(np.float32))
    return {"path": str(path), "sha256": _sha256(path), "n_rows": len(keys)}


class Injection:
    """One injection arm's controller, its per-batch state and the gain record (module docstring)."""

    def __init__(self, mdl: T.Model, arm: T.ThinkArm, args, items: Sequence[R.Item], dirs: Directions | None) -> None:
        env, s = mdl.env, arm.inject
        self.mdl, self.arm, self.args, self.spec = mdl, arm, args, s
        self.items_idx = {it.uid: i for i, it in enumerate(items)}
        self.items = list(items)
        n = len(env.blocks)
        if s.block is not None:
            L = s.block
        elif dirs is not None:
            L = dirs.blocks[0]
        else:
            L = n // 2
        if not 0 <= L < n - 1:
            raise SystemExit(f"{arm.spec}: block {L} must be in [0, {n - 2}] (the injection is at its output)")
        if s.kind == "random":
            d = None
        else:
            if dirs is None:
                raise SystemExit(f"{arm.spec}: --dirs is needed for inj={s.kind}")
            if dirs.d_model != env.d_model:
                raise SystemExit(f"{dirs.path}: d_model {dirs.d_model}, the model {env.d_model}")
            d = dirs.get(s.kind, L)
        report = _read_report(Path(args.out))
        g_min, g_max, self.donor = 0.0, 0.0, None
        if s.mode == "fixed":
            if s.match is not None:
                donor = _arm_by_label(report, s.match)
                if donor is None:
                    raise SystemExit(f"{arm.spec}: arm {s.match} is not complete in {args.out}")
                g_max = float(donor["agg"]["inj_gain_mean_pooled"])
            else:
                g_max = float(s.g)
            src: Source = ConstantSource(1.0)
        elif s.mode == "law":
            g_min, g_max = float(s.g_min), float(s.g_max)
            src = make_law(s.law)
        else:
            donor = _arm_by_label(report, s.yoke)
            if donor is None:
                raise SystemExit(f"{arm.spec}: arm {s.yoke} is not complete in {args.out}")
            di = donor["setup"]["inject"]
            g_min, g_max = float(di["g_min_used"]), float(di["g_max_used"])
            self.donor = load_gain(Path(donor["setup"]["inject"]["gain_file"]["path"]))
            src = YokeSource()
        self.L, self.g_min, self.g_max, self.src = L, g_min, g_max, src
        self.ctrl = FinishController(env.d_model, d, s.sign, stops=env.stops, source=src, g_min=g_min, g_max=g_max,
                                     g_abs=max(0.3, g_max), smooth="none", span_end_id=env.span_end,
                                     steer_from="last_prompt", seed=s.rseed, log_gain=True)
        self.ctrl.to(env.dev)
        self.cos = {}
        if dirs is not None:
            for k in dirs.kinds():
                try:
                    self.cos[k] = float(F.cosine_similarity(self.ctrl.d_fixed.cpu(), dirs.get(k, L), dim=0))
                except KeyError:
                    pass
        self.att: CA2Attachment | None = None
        self.traj: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        self.recs: list[dict] = []
        self.consist = consist_parts(src)

    # -- batch cycle
    def begin_batch(self, rows: Sequence[tuple[R.Item, int]]) -> None:
        env, c = self.mdl.env, self.ctrl
        c.reset()
        c.record = False
        prompts = [T.render(self.mdl, it.question, True, self.args.instruction) for it, _ in rows]
        _, mask = R.pad_left(prompts, env.pad)
        c.pending_mask = mask
        for p in self.consist:
            p.set_rows(len(rows))
        if self.spec.mode == "yoke":
            N = len(self.items)
            seqs = []
            for it, k in rows:
                du = self.items[(self.items_idx[it.uid] + 1) % N].uid
                g_r = self.donor.get((du, k))
                seqs.append(np.clip(g_r[1], 0.0, 1.0) if g_r is not None and len(g_r[1]) else np.zeros(1, np.float32))
            W = max(len(x) for x in seqs)
            table = torch.zeros(len(rows), W)
            for b, x in enumerate(seqs):
                table[b] = torch.from_numpy(np.pad(x, (0, W - len(x)), mode="edge").astype(np.float32))
            self.src.set_table(table)
            self.donor_uid = [self.items[(self.items_idx[it.uid] + 1) % N].uid for it, _ in rows]
        else:
            self.donor_uid = [None] * len(rows)

    def end_batch(self, rows: Sequence[tuple[R.Item, int]], recs: list[dict]) -> None:
        G = self.ctrl.gain_log()                              # (3, B, calls)
        tol = 1e-6 * max(1.0, self.g_max)
        for b, ((it, k), rec) in enumerate(zip(rows, recs)):
            if G.shape[-1]:
                st = G[2, b] > 0
                g, r = G[0, b][st].numpy(), G[1, b][st].numpy()
            else:
                g = r = np.zeros(0, np.float32)
            self.traj[(it.uid, k)] = (g.astype(np.float32), r.astype(np.float32))
            at = np.nonzero(g >= self.g_max - tol)[0] if self.g_max > self.g_min else np.zeros(0, int)
            rec.update(inj_n_steered=int(g.size), inj_gain_mean=float(g.mean()) if g.size else None,
                       inj_gain_max=float(g.max()) if g.size else None, inj_gain_last=float(g[-1]) if g.size else None,
                       inj_r_mean=float(r.mean()) if r.size else None,
                       inj_t_gmax=int(at[0]) if at.size else None, inj_donor=self.donor_uid[b])
            if self.consist:                                  # the law consulted every probe of the open span
                rec["n_ctrl_probes"] = int(rec.get("n_online_probes") or 0)
                rec["tokens_total"] = rec["n_gen"] + rec["n_ctrl_probes"] * self.args.probe_cost
        self.recs += recs

    # -- probes
    def forked_probe(self, orig):
        def fn(model, tok, cache, seq, mask, cue, n_tokens, stops, pad):
            self.ctrl.paused = True
            try:
                res, cache = orig(model, tok, cache, seq, mask, cue, n_tokens, stops, pad)
            finally:
                self.ctrl.paused = False
            if self.consist:
                op = self.ctrl.open_rows()
                last = seq[:, -1].cpu().tolist()
                ends = set(int(x) for x in stops) | ({int(self.mdl.env.span_end)} if self.mdl.env.span_end is not None
                                                      else set())
                rows = [b for b in range(len(res)) if b < op.numel() and bool(op[b]) and last[b] not in ends]
                for p in self.consist:
                    p.observe(rows, [res[b].answer for b in rows])
            return res, cache
        return fn

    def paused(self, orig):
        def fn(*a, **kw):
            self.ctrl.paused = True
            try:
                return orig(*a, **kw)
            finally:
                self.ctrl.paused = False
        return fn

    def finalize(self, res: dict, trace_dir: Path) -> None:
        pi = res["per_item"]
        for k in ("inj_n_steered", "inj_gain_mean", "inj_gain_max", "inj_gain_last", "inj_r_mean", "inj_t_gmax",
                  "inj_donor"):
            pi[k] = [r.get(k) for r in self.recs]
        gs = [v[0] for v in self.traj.values()]
        n = sum(len(g) for g in gs)
        res["agg"]["inj_gain_mean_pooled"] = float(sum(float(g.sum()) for g in gs) / n) if n else 0.0
        res["agg"]["inj_gain_mean"] = R._mean(pi["inj_gain_mean"])
        res["agg"]["inj_n_steered_mean"] = R._mean(pi["inj_n_steered"])
        res["agg"]["inj_t_gmax_rate"] = R._mean([float(x is not None) for x in pi["inj_t_gmax"]])
        path = Path(trace_dir) / self.arm.key / "gain.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        res["setup"]["inject"] = {**self.spec.as_dict(), "block": self.L, "g_min_used": self.g_min,
                                  "g_max_used": self.g_max, "direction_cos": self.cos,
                                  "dirs": None if DIRS[0] is None else {"path": str(DIRS[0].path),
                                                                         "sha256": DIRS[0].sha256},
                                  "gain_file": save_gain(path, self.traj),
                                  "gate": "last prompt column + span (close and after: 0), never after a stop id; "
                                          "probes not steered"}


DIRS: list[Directions | None] = [None]                       # the run's direction file (set by run_main)


def run_batch_wrapped(orig_run_batch, sess: Injection):
    def fn(mdl, arm, rows, bi, args, writer):
        sess.begin_batch(rows)
        recs = orig_run_batch(mdl, arm, rows, bi, args, writer)
        sess.end_batch(rows, recs)
        return recs
    return fn


def make_run_arm(orig_run_arm):
    def run_arm(mdl, arm, items, args):
        if getattr(arm, "inject", None) is None:
            return orig_run_arm(mdl, arm, items, args)
        sess = Injection(mdl, arm, args, items, DIRS[0])
        saved = (T.run_batch, D.forked_probe, D.probe_prefixes)
        sess.att = CA2Attachment(mdl.env.model, sess.ctrl, tap=sess.L, inject_before=sess.L + 1)
        T.run_batch = run_batch_wrapped(saved[0], sess)
        D.forked_probe, D.probe_prefixes = sess.forked_probe(saved[1]), sess.paused(saved[2])
        try:
            res = orig_run_arm(mdl, arm, items, args)
        finally:
            T.run_batch, D.forked_probe, D.probe_prefixes = saved
            sess.att.remove()
        sess.finalize(res, Path(args.trace_dir))
        ag = res["agg"]
        print(f"[{arm.key}] injection block {sess.L}, g in [{sess.g_min:g}, {sess.g_max:g}]: realised mean gain "
              f"{ag['inj_gain_mean_pooled']:.4f}, tokens_total {ag['tokens_total_mean']:.0f}", flush=True)
        return res
    return run_arm


@contextlib.contextmanager
def wrapped_think():
    saved = (T.parse_arm, T.run_arm)
    T.parse_arm, T.run_arm = parse_finish_arm, make_run_arm(saved[1])
    try:
        yield
    finally:
        T.parse_arm, T.run_arm = saved


# ------------------------------------------------------------------------------------------- frontier
def envelope(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Monotone frontier from (tokens, acc) points: sorted by tokens, the best accuracy reachable at ≤ tokens."""
    pts = sorted((float(t), float(a)) for t, a in points)
    out, best = [], -math.inf
    for t, a in pts:
        best = max(best, a)
        out.append((t, best))
    return out


def env_acc(env: Sequence[tuple[float, float]], t: float) -> float | None:
    """Envelope accuracy at t tokens (linear between points, flat after the last); None below the first point."""
    if not env or t < env[0][0]:
        return None
    for (t0, a0), (t1, a1) in zip(env, env[1:]):
        if t0 <= t <= t1:
            return a0 if t1 == t0 else a0 + (a1 - a0) * (t - t0) / (t1 - t0)
    return env[-1][1]


def env_tokens(env: Sequence[tuple[float, float]], a: float) -> float | None:
    """Fewest tokens at which the envelope reaches accuracy a (linear between points); None outside its range."""
    if not env or a > env[-1][1] or a < env[0][1]:
        return None
    if a == env[0][1]:
        return env[0][0]
    for (t0, a0), (t1, a1) in zip(env, env[1:]):
        if a0 < a <= a1:
            return t0 + (t1 - t0) * (a - a0) / (a1 - a0)
    return None


def _by_uid(pi: dict, key: str) -> dict[str, float]:
    acc: dict[str, list] = {}
    for u, v in zip(pi["uid"], pi[key]):
        if v is not None:
            acc.setdefault(u, []).append(float(v))
    return {u: float(np.mean(v)) for u, v in acc.items()}


def frontier_distance(arm_pi: dict, front: dict[str, dict], n_boot: int = 2000, seed: int = 0,
                      other_pi: dict | None = None) -> dict:
    """Vertical (Δacc at the arm's tokens) and horizontal (frontier tokens at the arm's accuracy − arm tokens)
    distances from the truncation envelope, with problem-bootstrap intervals (the frontier is resampled too).
    With ``other_pi`` also the difference of the horizontal distances (arm − other) on the same resamples."""
    uids = sorted(set(arm_pi["uid"]).intersection(*[set(p["uid"]) for p in front.values()])
                  .intersection(set(other_pi["uid"]) if other_pi is not None else set(arm_pi["uid"])))
    if not uids or not front:
        return {"n_problems": len(uids)}
    mats = []
    for pi in [arm_pi] + list(front.values()) + ([other_pi] if other_pi is not None else []):
        a, t = _by_uid(pi, "correct"), _by_uid(pi, "tokens_total")
        mats.append((np.asarray([a[u] for u in uids]), np.asarray([t[u] for u in uids])))
    nf = len(front)

    def dist(idx):
        means = [(t[idx].mean(), a[idx].mean()) for a, t in mats]
        env = envelope(means[1:1 + nf])
        t0, a0 = means[0]
        fa, ft = env_acc(env, t0), env_tokens(env, a0)
        v = None if fa is None else a0 - fa
        h = None if ft is None else ft - t0
        hd = None
        if other_pi is not None:
            t1, a1 = means[-1]
            f1 = env_tokens(env, a1)
            hd = None if (ft is None or f1 is None) else (ft - t0) - (f1 - t1)
        return v, h, hd, env

    full = np.arange(len(uids))
    v, h, hd, env = dist(full)
    rng = np.random.default_rng(seed)
    bs = [dist(rng.integers(0, len(uids), len(uids)))[:3] for _ in range(n_boot)]

    def ci(j, val):
        xs = np.asarray([b[j] for b in bs if b[j] is not None], np.float64)
        if xs.size < 0.95 * n_boot:
            return {"value": val, "lo": None, "hi": None, "defined_frac": float(xs.size / n_boot)}
        lo, hi = np.quantile(xs, [0.025, 0.975])
        return {"value": val, "lo": float(lo), "hi": float(hi), "defined_frac": float(xs.size / n_boot)}

    out = {"n_problems": len(uids), "acc": float(mats[0][0].mean()), "tokens_total": float(mats[0][1].mean()),
           "envelope": env, "vertical": ci(0, v), "horizontal": ci(1, h)}
    if other_pi is not None:
        out["horizontal_minus_other"] = ci(2, hd)
    return out


def import_arms(report: dict, specs: Sequence[str], args_now: dict, uids: list[str]) -> dict:
    """Arms of earlier think.py runs: ``file.json:label,label``; refused unless the problems, seeds, batches, caps
    and probe settings agree with this run."""
    out = {}
    keys = ("model", "data", "data_seed", "limit", "samples", "seed", "batch", "max_new_tokens", "probe_every",
            "probe_tokens", "probe_mode", "probe_cost", "budgets", "instruction", "dtype", "tiny", "tiny_arch")
    for spec in specs:
        path, _, labels = spec.rpartition(":")
        src = json.loads(Path(path).read_text())
        diff = [k for k in keys if k in src.get("args", {}) and src["args"][k] != args_now.get(k)]
        if diff or src.get("data", {}).get("uids") != uids:
            raise SystemExit(f"--import {path}: settings differ from this run ({diff or 'problem list'})")
        for lab in [x for x in labels.split(",") if x]:
            res = _arm_by_label(src, lab)
            if res is None:
                raise SystemExit(f"--import {path}: no complete arm {lab!r}")
            out[R.safe_key(lab)] = {"source": str(path), "spec": res["spec"], "key": res["key"], "agg": res["agg"],
                                    "per_item": res["per_item"], "setup": res.get("setup")}
    return out


def analyse(report: dict, front_labels: Sequence[str], primary: str, bias: str, trunc: str, random_: str,
            budgets: Sequence[int], seed: int, n_boot: int) -> dict:
    """The frontier, every arm's distance to it, and the pre-registered C2/C3-inject comparison."""
    pool = {res["key"]: res for res in report.get("imported", {}).values()}
    pool.update({res["key"]: res for res in report.get("arms", {}).values() if res.get("complete")})
    front = {R.safe_key(x): pool[R.safe_key(x)]["per_item"] for x in front_labels if R.safe_key(x) in pool}
    out = {"frontier_arms": list(front), "points": {k: (pool[k]["agg"]["tokens_total_mean"], pool[k]["agg"]["acc"])
                                                    for k in front}}
    if not front:
        out["note"] = "no truncation arms: give --frontier labels present in the run or imported"
        return out
    out["distances"] = {k: frontier_distance(res["per_item"], front, n_boot, seed) for k, res in pool.items()}
    pk, bk, tk, rk = (R.safe_key(x) for x in (primary, bias, trunc, random_))
    pre = {"rule": "tokens at matched accuracy (H = frontier tokens at the arm's accuracy − arm tokens_total) for the "
                   f"dynamic injection ({primary}) against the logit bias ({bias}) and against truncation at "
                   f"{trunc}; random-direction arm ({random_}) as the attribution control"}
    if pk in pool:
        P = pool[pk]["per_item"]
        for name, k in (("vs_bias", bk), ("vs_trunc", tk), ("vs_random", rk)):
            if k in pool and k != pk:
                Q = pool[k]["per_item"]
                pre[name] = {"arm": k, "H_minus_other": frontier_distance(P, front, n_boot, seed, other_pi=Q)
                             .get("horizontal_minus_other"), "paired": T.compare(P, Q, budgets, seed)}
    out["preregistered"] = pre
    return out


# ------------------------------------------------------------------------------------------- shared loading
def _model_args(a) -> argparse.Namespace:
    return argparse.Namespace(tiny=a.tiny, tiny_arch=a.tiny_arch, seed=a.seed, dtype="float32" if a.tiny else a.dtype,
                              device=a.device, model=a.model, attn_impl=a.attn_impl, span_open=None, span_close=None)


def _common(ap: argparse.ArgumentParser) -> None:
    a = ap.add_argument
    a("--model", default="Qwen/Qwen3.5-2B")
    a("--tiny", action="store_true")
    a("--tiny-arch", default="qwen3", choices=["qwen3", "qwen35"])
    a("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    a("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    a("--attn-impl", default="sdpa")
    a("--seed", type=int, default=0)
    a("--name", required=True)
    a("--out", default=None, help="results JSON (default results/evolve/round12/stage1/<name>.json)")


def _blocks(spec: str | None, n: int) -> list[int]:
    if not spec:
        return default_blocks(n)
    return sorted({int(x) % n for x in spec.split(",") if x.strip()})


def _heldout(path: str | None) -> set:
    if not path:
        return set()
    return set(json.loads(Path(path).read_text())["data"]["uids"])


def _resolve(path: str) -> str:
    """A results path as recorded (relative to the working directory or the repository root)."""
    p = Path(path).expanduser()
    return str(p if p.exists() or p.is_absolute() else ROOT / p)


# ------------------------------------------------------------------------------------------- directions CLI
def directions_main(argv: Sequence[str]) -> dict:
    ap = argparse.ArgumentParser(prog="finish.py directions")
    _common(ap)
    a = ap.add_argument
    a("--blocks", default=None, help="blocks (default n/2, 5n/8, 3n/4)")
    a("--think-data", default="gsm8k:train:400", help="reason.py data spec of the think prompt set")
    a("--data-seed", type=int, default=0)
    a("--heldout-from", default=None, help="a results JSON whose problems (data.uids) are excluded everywhere")
    a("--n-think", type=int, default=0, help="first N held-out problems for the think direction (0: all)")
    a("--conclude-traces", default=None, help="trace arm dir of unsteered runs (closed rows build 'conclude')")
    a("--window", type=int, default=16, help="K positions per conclude window")
    a("--min-conclude", type=int, default=24, help="fewer closed rows than this: no conclude direction")
    a("--instruction", default=R.INSTRUCTION)
    a("--batch", type=int, default=8)
    a("--dir-out", default=None, help="direction file (default ~/limbic-data/ca2_dirs/<name>.pt)")
    args = ap.parse_args(argv)
    aj = alignment().judge
    name = aj.safe_name(args.name)
    out = Path(args.out or STAGE / f"{name}.json")
    dpath = Path(args.dir_out).expanduser() if args.dir_out else aj.assert_private(DIR_ROOT, ROOT) / f"{name}.pt"
    aj.assert_private(dpath.parent, ROOT)
    t0 = time.time()
    mdl = T.load_model(_model_args(args))
    env = mdl.env
    n = len(env.blocks)
    layers = _blocks(args.blocks, n)
    excl = _heldout(args.heldout_from)
    spec = "synthetic:16" if args.tiny and args.think_data == "gsm8k:train:400" else args.think_data
    items = [it for it in R.load_items(spec, args.data_seed) if it.uid not in excl]
    if args.n_think:
        items = items[: args.n_think]
    on = [T.render(mdl, it.question, True, args.instruction) for it in items]
    off = [T.render(mdl, it.question, False, args.instruction) for it in items]
    th = think_direction(env.model, env.blocks, on, off, layers, env.pad, args.batch)
    dirs: dict[str, dict[int, torch.Tensor | None]] = {"think": {L: th[L]["d"] for L in layers}, "conclude": {}}
    stats: dict = {"think": {L: th[L]["stats"] for L in layers}, "conclude": {}}
    con_rows, blob_uids = [], []
    if args.conclude_traces:
        con_rows = read_trace(args.conclude_traces, close_id=env.span_end, exclude=excl)
        nat = [r for r in con_rows if r.natural]
        fit = [r for r in nat if conclude_positions(r.n_prompt, r.n_reason, r.close_idx, args.window) is not None]
        stats["conclude_rows"] = {"rows": len(con_rows), "natural_close": len(nat), "fit_windows": len(fit),
                                  "min_rows": args.min_conclude, "enough": len(fit) >= args.min_conclude}
        blob_uids = [r.uid for r in fit]
        if len(fit) >= args.min_conclude:
            co = conclude_direction(env.model, env.blocks, fit, layers, env.pad, args.window)
            dirs["conclude"] = {L: co[L]["d"] for L in layers}
            stats["conclude"] = {L: co[L]["stats"] for L in layers}
    cos = {}
    for L in layers:
        vs = {"think": dirs["think"].get(L), "conclude": dirs["conclude"].get(L),
              "random0": random_direction(env.d_model, 0)}
        vs = {k: v for k, v in vs.items() if v is not None}
        cos[L] = {f"{a_}|{b_}": float(F.cosine_similarity(vs[a_], vs[b_], dim=0))
                  for i, a_ in enumerate(vs) for b_ in list(vs)[i + 1:]}
    blob = {"model": mdl.facts["name"], "d_model": env.d_model, "blocks": layers, "dirs": dirs,
            "meta": {"think_uids": [it.uid for it in items], "think_data": spec, "heldout_from": args.heldout_from,
                     "conclude_traces": args.conclude_traces, "conclude_uids": blob_uids, "window": args.window,
                     "instruction": args.instruction,
                     "git": git_state(), "created": time.strftime("%Y-%m-%d %H:%M:%S")}}
    torch.save(blob, dpath)
    report = {"kind": "c2_directions",
              "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "git": git_state(), "versions": R.versions(), "model": mdl.facts, "blocks": layers,
              "dir_file": {"path": str(dpath), "sha256": _sha256(dpath)},
              "think": {"n_problems": len(items), "rule": "mean h_off[-1] - mean h_on[-1] (last prompt position)",
                        "stats": {str(L): v for L, v in stats["think"].items()}},
              "conclude": {"rule": f"mean over closed rows of (mean h at the {args.window} positions before the close "
                                   f"- mean h at {args.window} positions centred on the span middle)",
                           "rows": stats.get("conclude_rows"), "uids": blob_uids,
                           "stats": {str(L): v for L, v in stats["conclude"].items()}},
              "cosines": {str(L): v for L, v in cos.items()}, "gpu_peak_gb": _peak_gb(), "seconds": time.time() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    R.write_json(out, report)
    con = f"from {stats['conclude_rows']['fit_windows']} closed rows" if stats["conclude"] else "not built"
    print(f"directions at blocks {layers}: think from {len(items)} problems; conclude {con} "
          f"({stats.get('conclude_rows')}); saved {dpath} and {out}", flush=True)
    for L in layers:
        print(f"  block {L}: think {stats['think'][L]}  conclude {stats['conclude'].get(L)}  cos {cos[L]}", flush=True)
    return report


# ------------------------------------------------------------------------------------------- screen CLI
@torch.no_grad()
def final_hidden(model, ids: torch.Tensor) -> torch.Tensor:
    """Final-norm hidden states (T, D) of one unpadded sequence (1, T); the head is applied to one column only."""
    norm, cap = R._final_norm(model), {}
    h = norm.register_forward_hook(lambda m, a, o: cap.__setitem__("h", o))
    try:
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
    finally:
        h.remove()
    return cap["h"][0]


@torch.no_grad()
def tf_effects(head, hs: torch.Tensor, hb: torch.Tensor, ids: torch.Tensor, close_id: int, cols: torch.Tensor,
               chunk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """At sequence columns ``cols``: Δ log p(close), KL(steer ‖ base) and ΔNLL of the next token (NaN at the end)."""
    dl, kl, dn = [], [], []
    T_ = ids.shape[0]
    for a in range(0, cols.numel(), chunk):
        c = cols[a:a + chunk]
        ls = F.log_softmax(head(hs[c]).float(), -1)
        lb = F.log_softmax(head(hb[c]).float(), -1)
        dl.append((ls[:, close_id] - lb[:, close_id]).cpu())
        kl.append((ls.exp() * (ls - lb)).sum(-1).clamp_min(0.0).cpu())
        nxt = (c + 1).clamp_max(T_ - 1)
        tg = ids[nxt]
        d = -(ls.gather(1, tg[:, None])[:, 0] - lb.gather(1, tg[:, None])[:, 0])
        dn.append(torch.where(c + 1 < T_, d, torch.full_like(d, float("nan"))).cpu())
        del ls, lb
    return torch.cat(dl).numpy(), torch.cat(kl).numpy(), torch.cat(dn).numpy()


@torch.no_grad()
def steered_hidden(env: R.Env, ids: torch.Tensor, n_prompt: int, block: int, d: torch.Tensor | None, sign: float,
                   g: float, rseed: int, close_id: int | None) -> torch.Tensor:
    """Final hidden states of one sequence with the fixed-gain injection at block ``block`` from the last prompt
    position on (the generation gate: the last prompt column and the span)."""
    ctrl = FinishController(env.d_model, d, sign, stops=env.stops, source=ConstantSource(1.0), g_min=0.0, g_max=g,
                            g_abs=max(0.3, g), smooth="none", span_end_id=close_id, seed=rseed,
                            steer_from=torch.tensor([n_prompt - 1]))
    ctrl.to(env.dev)
    att = CA2Attachment(env.model, ctrl, tap=block, inject_before=block + 1)
    try:
        ctrl.reset()
        ctrl.pending_mask = torch.ones(1, ids.shape[1])
        return final_hidden(env.model, ids)
    finally:
        att.remove()


def _peak_gb() -> float | None:
    return round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None


def _nanmean(rows: Sequence[dict], key: str) -> float | None:
    x = np.asarray([r[key] for r in rows], np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else None


def screen_main(argv: Sequence[str]) -> dict:
    ap = argparse.ArgumentParser(prog="finish.py screen")
    _common(ap)
    a = ap.add_argument
    a("--dirs", required=True)
    a("--traces", required=True, help="held-out trace arm dir with forced-answer probes (.../reason_base_q35/none)")
    a("--blocks", default=None, help="default: the direction file's blocks")
    a("--gains", default="0.05,0.1,0.2,0.3")
    a("--kinds", default=None, help="directions (default every kind in the file)")
    a("--random-seeds", default="0,1")
    a("--rows", type=int, default=64, help="rows with both labels under --label (first N in trace order)")
    a("--label", choices=["correct", "stable"], default="correct", help="primary label (both are reported)")
    a("--exclude-from", default=None, help="a results JSON whose problems are skipped (besides the direction "
                                           "file's heldout_from, which is always skipped)")
    a("--allow-eval-rows", action="store_true", help="keep the evaluation problems (a dev screen, marked as such)")
    a("--pos-bin", type=int, default=128, help="span-offset bin width of the position-matched contrast (a divisor "
                                               "of the traces' probe spacing)")
    a("--max-len", type=int, default=4608)
    a("--chunk", type=int, default=512)
    a("--n-boot", type=int, default=4000)
    args = ap.parse_args(argv)
    aj = alignment().judge
    name = aj.safe_name(args.name)
    out = Path(args.out or STAGE / f"{name}.json")
    t0 = time.time()
    dirs = Directions(args.dirs)
    mdl = T.load_model(_model_args(args))
    env = mdl.env
    if dirs.d_model != env.d_model:
        raise SystemExit(f"{dirs.path}: d_model {dirs.d_model}, the model {env.d_model}")
    blocks = [int(x) for x in args.blocks.split(",")] if args.blocks else dirs.blocks
    kinds = [k for k in (args.kinds.split(",") if args.kinds else dirs.kinds())]
    gains = [float(x) for x in args.gains.split(",")]
    cfg_dirs = [(f"{'-' if s < 0 else ''}{k}", k, s, 0) for k in kinds for s in (1.0, -1.0)] + \
               [(f"random{r}", "random", 1.0, int(r)) for r in args.random_seeds.split(",") if r.strip()]
    meta = dirs.blob.get("meta", {})
    eval_from = meta.get("heldout_from")
    eval_uids = _heldout(_resolve(eval_from)) if eval_from else set()
    excl = _heldout(args.exclude_from) | (set() if args.allow_eval_rows else eval_uids)
    rows_all = read_trace(args.traces, close_id=env.span_end, exclude=excl)
    if not rows_all:
        raise SystemExit(f"{args.traces}: no rows outside the evaluation problems ({eval_from}); screen held-out "
                         "traces (e.g. reason_base_*/none) or pass --allow-eval-rows for a dev screen")
    con_uids = set(meta.get("conclude_uids") or [])
    aligned = all(int(c) % args.pos_bin == 0 for r in rows_all for c in r.cks[:-1])   # the last is the span end
    nbins = max(1, -(-args.max_len // args.pos_bin))
    rows = []
    for r in rows_all:
        f, u = formation_labels(r, args.label, min(len(r.ids), args.max_len))
        if f.any() and u.any():
            rows.append(r)
        if len(rows) >= args.rows:
            break
    head = env.model.get_output_embeddings()
    close = int(env.span_end)
    per_row: dict[str, list] = {}
    print(f"{mdl.facts['name']}: {len(rows)} rows of {len(rows_all)} with both labels; blocks {blocks}, gains {gains}, "
          f"directions {[c[0] for c in cfg_dirs]}", flush=True)
    for i, r in enumerate(rows):
        end = min(r.n_prompt + r.n_reason, r.close_idx if r.close_idx is not None else len(r.ids), args.max_len)
        ids = torch.as_tensor(r.ids[:end], device=env.dev)[None]
        cols = torch.arange(r.n_prompt, end, device=env.dev)
        lab = {k: formation_labels(r, k, end) for k in ("correct", "stable")}
        mspan = np.arange(r.n_prompt, end) - r.n_prompt + 1
        span = np.zeros(end, bool)
        span[r.n_prompt:end] = True
        sel = slice(r.n_prompt, end)
        hb = final_hidden(env.model, ids)
        for L in blocks:
            for lab_d, kind, sign, rs in cfg_dirs:
                d = None if kind == "random" else dirs.get(kind, L)
                for g in gains:
                    hs = steered_hidden(env, ids, r.n_prompt, L, d, sign, g, rs, close)
                    dl, kl, dn = tf_effects(head, hs, hb, ids[0], close, cols, args.chunk)
                    for lk, (fm, um) in lab.items():
                        e = row_effects(dl, kl, dn, fm[sel], um[sel], r.boundary[:end][sel], span[sel])
                        e["Fbin"], e["Ubin"] = binned_effects(dl, fm[sel], um[sel], mspan, args.pos_bin, nbins)
                        per_row.setdefault(f"{L}|{lab_d}|{g:g}|{lk}", []).append({"uid": r.uid, **e})
                    del hs
        print(f"  row {i + 1}/{len(rows)} ({end} tokens)  {time.time() - t0:.0f}s", flush=True)
    results, gates = {}, {}
    for L in blocks:
        for g in gains:
            for lk in ("correct", "stable"):
                sels = {lab_d: selectivity(per_row.get(f"{L}|{lab_d}|{g:g}|{lk}", []), args.seed)
                        for lab_d, *_ in cfg_dirs}
                for lab_d, s in sels.items():
                    rr = per_row.get(f"{L}|{lab_d}|{g:g}|{lk}", [])
                    kl_nb, bn = _nanmean(rr, "kl_nonbnd"), _nanmean(rr, "bnd")
                    s["sel_kl"] = float(bn / kl_nb) if bn is not None and kl_nb is not None and kl_nb > 0 else None
                    s["kl_mean"], s["dnll_mean"] = _nanmean(rr, "kl"), _nanmean(rr, "dnll")
                    s["delta_all_mean"] = _nanmean(rr, "all")
                    s["matched"] = matched_contrast(rr, args.n_boot, args.seed)
                    if lab_d.lstrip("-") == "conclude":
                        fr = [x for x in rr if x["uid"] not in con_uids]
                        s["fresh"] = {**selectivity(fr, args.seed), "matched": matched_contrast(fr, args.n_boot,
                                                                                                 args.seed)}
                    results[f"{L}|{lab_d}|{g:g}|{lk}"] = {**s, "rows": rr}
                rand = [sels[c[0]] for c in cfg_dirs if c[1] == "random"]
                for lab_d, kind, *_ in cfg_dirs:
                    if kind != "random":
                        gates[f"{L}|{lab_d}|{g:g}|{lk}"] = screen_gate(sels[lab_d], rand)
    passing = sorted([k for k, v in gates.items() if v["pass"] and k.endswith("|" + args.label)],
                     key=lambda k: -(gates[k]["selectivity"] or math.inf))
    used = [r.uid for r in rows]
    report = {"kind": "c2_screen", "args": vars(args), "git": git_state(), "versions": R.versions(),
              "model": mdl.facts, "dirs": {"path": str(dirs.path), "sha256": dirs.sha256, "blocks": dirs.blocks},
              "traces": str(Path(args.traces).expanduser()), "n_rows": len(rows), "uids": [r.uid for r in rows],
              "blocks": blocks, "gains": gains, "directions": [c[0] for c in cfg_dirs],
              "measure": "delta = log p_steer(close) - log p_base(close) at span positions; F/U = formed / not formed",
              "results": results, "gates": gates, "passing": passing, "injection_alive": bool(passing),
              "passing_strict": [k for k in passing if gates[k]["pass_strict"]],
              "passing_matched": [k for k in passing if gates[k]["pass_matched"]],
              "rows_policy": {"eval_from": eval_from, "eval_rows_excluded": not args.allow_eval_rows,
                              "dev_screen": bool(args.allow_eval_rows and set(used) & eval_uids),
                              "eval_overlap": len(set(used) & eval_uids), "exclude_from": args.exclude_from,
                              "conclude_built_from_screen_rows": len(set(used) & con_uids)},
              "pos_bin": args.pos_bin, "pos_bin_aligned_with_probes": aligned,
              "gate_rule": "pass: E_F > 0, CI(E_F - E_U) > 0, selectivity >= 2 max(1, best random); pass_strict "
                           "also E_F > the best random direction's E_F",
              "gpu_peak_gb": _peak_gb(), "seconds": time.time() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    R.write_json(out, report)
    print(f"{'passing' if passing else 'no direction passes the gate'}: {passing[:6]}; position-matched: "
          f"{report['passing_matched'][:6]}; saved {out}", flush=True)
    for L in blocks:
        for g in gains:
            line = []
            for lab_d, *_ in cfg_dirs:
                s = results[f"{L}|{lab_d}|{g:g}|{args.label}"]
                rt = "sign" if s["sign_selective"] else ("-" if s["ratio"] is None else f"{s['ratio']:.2f}")
                ef = "-" if s["E_F"] is None else f"{s['E_F']:+.3f}"
                line.append(f"{lab_d} {ef}/{rt}")
            print(f"  L{L} g{g:g}: " + "  ".join(line), flush=True)
    return report


# ------------------------------------------------------------------------------------------- run CLI
def run_main(argv: Sequence[str]) -> dict:
    ap = argparse.ArgumentParser(prog="finish.py run", add_help=False, allow_abbrev=False,
                                 description="think.py's runner with injection arms; every think.py flag applies")
    a = ap.add_argument
    a("--dirs", default=None, help="direction file from 'directions'")
    a("--preset-inj", "--preset", dest="preset_inj", default=None, choices=["c3inj"])
    a("--base", default="T=0.6;pp=0", help="decoding keys of every preset arm (as the c3dev arms)")
    a("--inj-dir", default="think", help="direction of the preset arms ([-]think | [-]conclude)")
    a("--inj-block", type=int, default=None)
    a("--g-max", type=float, default=0.2)
    a("--g-fixed", type=float, default=None, help="add a fixed-gain arm at this g")
    a("--law", default="consist-3")
    a("--import", dest="imports", action="append", default=[], help="file.json:label,label (repeatable)")
    a("--frontier", default="nothink,f256,f512,f1024,f2048,t06", help="labels of the truncation frontier")
    a("--primary", default="dyn")
    a("--bias", default="biasc3")
    a("--trunc-ref", default="f256")
    a("--random-arm", default="rand")
    a("--n-boot", type=int, default=2000)
    mine, rest = ap.parse_known_args(argv)
    if "--name" not in rest and "--out" not in rest:
        raise SystemExit("run: give --name (or --out)")
    if mine.preset_inj:
        if "--arms" in rest:
            raise SystemExit("run: --preset c3inj builds the arms; do not also give --arms")
        if mine.inj_block is None:
            raise SystemExit("run: --preset c3inj needs --inj-block (from the C2 screen)")
        rest += ["--arms", ",".join(preset_arms(mine.base, mine.inj_dir, mine.inj_block, mine.g_max, mine.law,
                                                mine.g_fixed))]
    targs = T.build_parser().parse_args(rest)
    DIRS[0] = Directions(mine.dirs) if mine.dirs else None
    arms = [parse_finish_arm(s) for s in (targs.arms.split(",") if targs.arms else T.PRESETS[targs.preset])
            if s.strip()]
    check_arm_order(arms)
    name = targs.name
    out = Path(targs.out or ROOT / "results" / "evolve" / "round12" / targs.stage / f"{name}.json")
    prev = _read_report(out)
    for spec, res in prev.get("arms", {}).items():           # a direction file changed under a complete arm
        di = (res.get("setup") or {}).get("inject") or {}
        if di.get("dirs") and DIRS[0] is not None and di["dirs"]["sha256"] != DIRS[0].sha256 \
                and "--force" not in rest:
            raise SystemExit(f"{out}: arm {spec!r} used another direction file ({di['dirs']['path']}); "
                             "use another --name, or --force")
    with wrapped_think():
        report = T.main(rest)
    out = Path(report["args"]["out"])
    report = _read_report(out)
    if mine.imports:
        report["imported"] = import_arms(report, mine.imports, report["args"], report["data"]["uids"])
    report["finish"] = {"dirs": None if DIRS[0] is None else {"path": str(DIRS[0].path), "sha256": DIRS[0].sha256},
                        "preset": mine.preset_inj, "inj_args": vars(mine), "gpu_peak_gb": _peak_gb()}
    budgets = [int(b) for b in report["args"]["budgets"].split(",") if b]
    report["analysis"] = analyse(report, mine.frontier.split(","), mine.primary, mine.bias, mine.trunc_ref,
                                 mine.random_arm, budgets, report["args"]["seed"], mine.n_boot)
    R.write_json(out, report)
    an = report["analysis"]
    for k, d in (an.get("distances") or {}).items():
        if "vertical" in d:
            v, h = d["vertical"], d["horizontal"]
            fmt = lambda x: "  -  " if x["value"] is None else f"{x['value']:+.3f}"
            print(f"  {k:>12}: acc {d['acc']:.3f} tokens {d['tokens_total']:.0f}  vertical {fmt(v)}  "
                  f"tokens saved at matched acc {'-' if h['value'] is None else round(h['value'])}", flush=True)
    print(f"analysis saved in {out}", flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("directions", "screen", "run"):
        raise SystemExit("usage: finish.py directions|screen|run ... (module docstring)")
    return {"directions": directions_main, "screen": screen_main, "run": run_main}[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
