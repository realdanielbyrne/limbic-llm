# Round 11 plan: compressed episodes and a usable sense of time (W1s on I1)

*Proposal, 4 Nov 2026, from a planning workflow (biology, ML and repo-design planners, plus a synthesis). Not yet approved. Its diagnostics are eval-only CPU checks on the round-10 I1 checkpoints (seeds 1–2); their scripts are in the GPU session's scratchpad, and they are not register rows. Open decision 1 (attention weights) is already settled by DEC8: they are admissible, but they fail empirically here.*

**Evidence.** Eval-only CPU diagnostics on I1 s1/s2 by three planners (scratchpad r11_plan/{biology,ml_compression_time,design}/); n = 24-96, fp32 base. They are diagnoses, not register rows. Stage B is still running (logs only).

They agree on five points:
1. **Length fails because of time.** The linear age penalty learned at 512 tokens costs about 1.5 logits at age ~1800, against a cosine margin of 0.3-0.45. 90-98% of misses lose to a younger filler window.
2. **omega = 0 fixes length but breaks updates.** Random 2048x5 rises 0.71 to 0.84-0.90 and 4096x8 rises 0.38-0.46 to 0.83-0.92, but the stale code then wins 46-50% of updates.
3. **The index dominates the state.** 79-92% of the 0.89M-float state is the 2048-d alignment code, and only about 5 of 24 entries are ever read.
4. **More slots hurt** under linear recency.
5. **Received attention does not mark the code.** It ranks the digits 17-24 of 24, so the PI's literal idea fails.

## 1. Hypotheses

| ID | Claim | Biology | ML precedent |
|---|---|---|---|
| H-T | A scale-free or content-driven recency prior fixes length and keeps updates | Reconsolidation, CA1 match/mismatch (Nader 2000; Kumaran & Maguire 2007); ratio recency (SIMPLE); interference over decay | ALiBi does not extrapolate; T5 buckets |
| H-C1 | A 64-d index is enough to address a position | Indexing theory (Teyler & DiScenna 1986) | JL / low-dimensional keys |
| H-C2 | An episode is its tagged transitions, kept in order | Selective, compressed replay (Davidson 2009; Singer & Frank 2009); tagging with later capture (Frey & Morris 1997) | H2O/SnapKV, but a learned hidden-state score at write time |
| H-K | Compression pays only where codes > K | DG/CA3 capacity | Compressive Transformer (by selection) |
| H-0 | The TimeChannel is dead weight | Time cells are event-driven and log-scaled; this clock saturates | none |

## 2. Stage 0: no training

Run on CPU caches now and as GPU evals after Stage B. Use eval_by_query with n ≥ 96 and the windowed, empty and shuffle controls on every override. Eval-time overrides are diagnostics only (E6).

| Gate | Test | Pass, and what follows |
|---|---|---|
| G0.1 | Stage B: learned omega against rand2048 exact (5 seeds); overrides re-run on s0/s3/s4 | omega = 0 or sup helps on ≥ 4/5 seeds. Otherwise stop and checkpoint. |
| G0.2 | Forms {linear, none, log, ord, supr, supw} on 2048x5 and 4096x8, and on update same/now/new at 512 and 2048 | supr or supw: 4096x8 ≥ 0.80 and stale ≤ 0.10 on both seeds. The better of log/ord on update becomes A1. |
| G0.3 | supw false fires (cos > theta between distinct facts) on held-out names and jitter | ≤ 2%, else drop supw |
| G0.4 | Selection oracle on natural, random and jitter: first-12; read offsets only (m = 4 ceiling); value-norm top-8; random-8; attention top-8 (diagnostic) | Settles the planners' natural-filler disagreement (ML: fine; design: -0.06 to -0.09). Value-norm within 0.03 means A6 is initialised from it. |
| G0.5 | Random projection to 64/128 on natural; W_z fitted on CPU from train caches, scored held-out | Learned 64-d ≥ 2048-d - 0.02 sets dc = 64; otherwise dc = 128 |
| G0.6 | 4096x{16,24} and 8192x32 codes; K16 vs K64 under the G0.2 winner | K16 coverage ≤ 0.8 and K64 ≥ 0.95 keeps K = 64 in A8 |
| G0.7 | Ceilings, full vs windowed at 512: update-k latest-wins; "code stated just before NAME's"; first/last; next-fact | full ≥ 0.80 and windowed ≤ 0.10 admits the task; any admission runs A9 |

## 3. Code changes and tests

**`experiments/evolve/w1s_slots.py`**
- `--w1s-recency-form {linear,none,log,ord,supr,supw}`, in `SlotLoop._select` (l.426) and the `stream()` read (l.693).
  - supw: `sup_i += sigmoid(gamma(cos(KL_i,KL_j) - theta))` in `_store` when window j closes (c_j ≤ t). The read subtracts `omega_s·sup_i`.
  - supr: penalises old slots by the read mass of newer slots that match the query.
  - All new terms are learned scalars.
- `--w1s-code {lex,learned,proj} --w1s-dc`: the alignment code becomes `W_z LN(h)`, in `_features`, the ring and `_alignment`.
- `--w1s-keep M --w1s-keep-by {first,value,learned}`: `mem &= keep` in `_store`.
  - Transitions are stored as (z_j, v_{j+1}, offset).
  - `sigmoid(score)` scales the kept values, so the scorer gets a gradient.
- `_stream_size` reports the state per component in the JSON.

**`experiments/evolve/eval_by_query.py`**
- New flags: `--w1s-set KEY=VAL`, `--update-k`, `--decoys N`, codes up to 32, 4096/8192 tokens (batch 1-2), and the admitted order tasks.

**`tests/test_evolve_w1s.py`**
- E1 bitwise with fp32 and bf16-rounded h, covering an update after a 2000-token gap, an eviction burst, and top-m at a window edge.
- stream() == parallel for every flag.
- State floats identical at T = 512/2048/4096.
- supw is monotone.
- The empty store returns 0.

**Before any GPU time**
- Cached oracles re-run through the new code must reproduce Stage 0 within 0.02.
- `--tiny` smoke test and the CPU suite.
- A CPU toy where the learned scorer finds the digits within 300 steps.

## 4. Stage A GPU arms (seeds 1-2, 6000 steps, I1 recipe)

The I1 reference is the existing s1/s2 checkpoints (no new runs).

| Arm | Single change | Runs |
|---|---|---|
| A1 T-rope | rotary slot time in the learned-key score, no age penalty (revision 1) | 2 |
| A1b T-ordinal | query-chosen ordinal rank feature, no age penalty (revision 1) | 2 |
| A2 T-supr | content-gated read supersession, no age term | 2 |
| A3 T-supw | write-time supersession tag, no age term | 2 |
| A4 T0 | `--no-time`. Eval-time evidence conflicts: ML -0.11/-0.02 slot hit, design ≈ 0. | 2 |
| A5 C-code | learned W_z, dc from G0.5 | 2 |
| A6 C-keep | learned scorer keeps m = 8 of n = 24 (the PI's idea, E4-safe) | 2 |
| A7 C-n12 | `--w1s-win 12` (fixed truncation; the control for A6) | 2 |
| A8 COMB | best of A2/A3 + A5 + n = 12 + K = 64 (about 0.17M floats) | 2 |
| A9 (if G0.7) | ordinal stamps phi(k) in the read; 25% order-task training | 2 |

Total: 16 runs (18 with A9).

**Dropped or deferred**
- Theta pointer: no eval-time gain.
- Gist, Perceiver and ToMe summaries: superposition, which failed in round 10.
- Attention salience: falsified, and an E4 question.
- Compressed reinstatement: waits for the RI gate.
- Two-tier store: A8 and G0.6 come first.
- Boundary windows: G0.4 jitter comes first.
- Retention threshold: P50-style filler-adaptation risk.
- Representative keys: wait for I1K.
- Laplace time cells: only if ord saturates.

## 5. Evaluations

- **DEC2 grid**, plus jitter and held-out names.
- **Stress:** 4096x8; 4096x16/24 and 8192x32 (codes > K); `--decoys 20`.
- **Update:** same/now/new at 512/2048/4096, plus update-k (k = 1-3). Report new-exact, stale rate and old_digit1.
- **Order/time:** admitted tasks only, each with full and windowed rows.
- **Per arm:** exact by rank and gap, slot hit, coverage, state floats and params.
- **E8 LM run:** Stage B only.

## 6. Pre-registered decision rules

- **Void:** windowed ≠ 0.00, empty ≠ 0, shuffle not near chance, or any E1 failure.
- **T arms (A1-A3):** win if all three hold:
  - mean exact on rand2048x5 and 4096x8 is ≥ +0.05 over I1;
  - no DEC2 cell drops more than 0.02;
  - update new-exact ≥ I1 - 0.02 and stale ≤ I1.
  The tie-break is the lower stale rate.
- **A4:** remove the TimeChannel if every DEC2 cell and the update task are within -0.02.
- **A5:** adopt if every cell is within 0.02 and the state falls ≥ 5x.
- **A6 and A7:** both must be within 0.02 of I1.
  - A6 also needs tag hit ≥ 0.95.
  - Learned tags are adopted only if A6 beats A7 by ≥ 0.05 on jitter; otherwise n = 12.
  - Tags settling on frame tokens are a falsifier.
- **A8:** adopt if ≥ I1 - 0.02 on every DEC2 cell and ≥ +0.10 on the codes > K cells. If A8 fails, the fallback recipe is the best T arm plus the passing C arms.
- **A9:** judged against its own DEC bar, with at most a 0.02 cost on the content task.
- **Stage B** (seeds 0-4, 1-2 recipes) needs all of:
  - ≥ 4/5 seeds at ≥ 0.85 exact on the full grid;
  - no update regression;
  - E8 and the CPU suite.

  Then a human checkpoint.

## 7. Open decisions for the PI

1. **E4:** attention weights are not hidden states, and they fail empirically. Recommendation: diagnostic only, no DEC.
2. **E3 buffer scope:** a seeded random projection is a 131k-float fixed buffer. Does the 1M limit cover buffers, streaming state, or both? A learned W_z avoids the question.
3. **DEC5:** supw's theta is initialised (0.88) from a measured cosine distribution, then learned. Is that admissible?
4. **New cells:** add codes > K and 4096/8192 as robustness cells, leaving the DEC2 bar unchanged.
5. **Order-task bar:** needs a DEC, for example ≥ 0.8 of the full-attention ceiling.
6. **Recipe changes:** removing the TimeChannel or the age penalty needs a DEC if adopted.

## Revision 1 (4 Nov 2026): time as data, not as a penalty

Daniel: "time as a penalty might not be the most appropriate use of time. LLMs went away from distance-based position encoding to RoPE … time as a data point, ordinal encoding, position relative to other tokens I think still matters."

- **The diagnosis agrees.** The failure is a fixed penalty (ω·age subtracted whatever the query), which over-penalises at lengths unseen in training. That is the ALiBi pattern.
- **The fix is time that interacts with content and that the query selects**, as RoPE puts relative position into q·k. (RoPE itself needs rescaling, such as NTK or YaRN, for long contexts. The lesson is "multiplicative, relative, query-selected", not that RoPE extrapolates for free.)
- **Arms.** A1 (log/ord penalty) is replaced by:
  - **A1 T-rope:** partial rotary time in the slot key and query, with log-spaced periods from 64 to 65,536 tokens;
  - **A1b T-ordinal:** a learned rank embedding with a query-dependent preference vector. Its range is bounded by K, so it is the same at any length.
- **What stays.** Supersession (A2, A3) stays, because it is already ordinal and content-gated.
- **Gate first.** G0.8, a CPU toy, checks that time as data does not repeat the merit-test failure of time codes concatenated into keys. That failure was 0/10; rotation preserves the norm and the content dims, but the claim is checked before GPU time.
- **Also worth testing.** If an order task has a full-attention ceiling (G0.7), the time-data arms are also the ones that can answer "which came first"; a penalty cannot.

## Revision 2 (4 Nov 2026): what round 10 Stage B changed

Daniel: "modify round 11 if results from 10 dictate a change". Stage B (P54–P58) dictates five changes.

1. **The base is I1, and the Stage-A seeds are 1 and 4.** I1 learned 3 of 5 seeds (seed 1 at 0.94, seed 4 at 0.87), with grid means 0.93–0.96 on the learned seeds. A2, I1K and RI did not beat it. Seeds 1 (borderline) and 4 (failing) discriminate better than seeds 1 and 2.
2. **Name generalisation becomes a goal.** Held-out names give 0.58–0.74: the learned key memorised the 20 training names, although facts are still retained. Two new arms:
   - **A10:** a large seeded synthetic name pool for training, disjoint from the held-out names;
   - **A11:** lexical keys behind a gate that starts near zero. I1K seed 3 generalised at 0.88, but lexical keys at full strength collapsed 3 of 5 seeds.

   Held-out names are a deciding cell for every arm.
3. **Updates need a task with a ceiling.** In the current form, full attention prefers the old code (new 0.39, old 0.57), so it cannot score updates. G0.7 now searches for phrasings with an explicit change ("has changed from OLD to NEW"), and the update rules apply only to an admitted form.
4. **G0.1 is reframed.** The planners' premise (a larger learned recency penalty means long-range failure) is refuted: the failing seed has a mid-range ω, and the best seeds the largest. What separates the failures is blunter alignment (log β_align 2.39 against 2.65–2.77 in I1; 1.55–1.72 against 2.58 in I1K). G0.1 now also tests sharper alignment at evaluation, and arm A12 (alignment starting sharper) runs if it helps. A5, the learned alignment code, may help the same way.
5. **Reinstatement stays deferred** (RI 0 of 5, grid means 0.16–0.63).

Arms are now A1/A1b (time as data), A2/A3 (supersession), A4 (no TimeChannel), A5–A8 (compression), A9 (order, optional), A10–A11 (names) and A12 (alignment, conditional): 22–26 runs.

