# Register of results, conclusions and experiments

This file is the single source of truth for the project's current state: every result with its status and result file, every correction, every decision, every open review finding and every planned experiment. Update it in place when a result lands, a decision is made or an experiment changes status, and cite the result file.

**Related documents** (links are relative to `research/`; evidence paths in the tables are relative to the repository root)

| Document | Role | Editing rule |
|---|---|---|
| `REGISTER.md` (this file) | Current state | Edit in place |
| [`NOTEBOOK.md`](NOTEBOOK.md) | Dated lab record: how each entry got here | Do not rewrite entries; add bracketed, dated corrections |
| [`../RESULTS.md`](../RESULTS.md) | Write-up of the CPU results (§1–4.6), the first pretrained results (§4.7) and round 1 (§4.8) | Its numbers must match this file |
| [`HANDOFF.md`](HANDOFF.md) | Note from the GPU session that runs the pretrained jobs | Owned by the GPU session, as are `results/pretrained/*`, `results/evolve/round*/` and `experiments/evolve/` |
| [`plans/round0_plan.md`](plans/round0_plan.md) | Round-0 diagnosis (D1–D5), runs g0–g7, diagnostics C1–C6, evaluator rules E1–E9, round-1 directions | Plan; statuses are tracked here |
| [`reviews/2026-11-02_alignment_cot_ca1.md`](reviews/2026-11-02_alignment_cot_ca1.md) | Review RV1 of the alignment, CoT and CA1 decision tracks | Dated record, not edited. Its `file:line` anchors refer to commits `73259fc` and `241a13c` |
| [`reviews/2026-11-02_literature/`](reviews/2026-11-02_literature/README.md) | Review RV2: literature review of the four tracks and the Samsonovich & Ascoli (2005) model, code audit, bibliography audit. Start with its `HANDOFF.md` | Dated record, not edited. Track reports are snapshots at `3799b36` |
| [`../docs/genesis.md`](../docs/genesis.md) | History of the ideas | |

**Result files.** `results/synthetic/` (CPU; M and T rows), `results/pretrained/` (GPU; P1–P6), `results/evolve/round1/` (GPU round 1; P7–P12; the files keep their `g1_*` names, which refer to the round, not to arm g1), `results/evolve/round0/` (planned g0–g7 outputs; not created yet), `results/evolve/round2/` (round 2, in progress), `results/alignment/` (written by `experiments/alignment/run.py`; labels and aggregates only).

**Status of results:** **Supported** · **Refuted** · **Mixed** (seeds or settings disagree) · **Not supported** (the predicted effect did not appear; weak evidence) · **Partly** · **Open** (untested, inconclusive or confounded) · **Fixed** (a bug, now corrected).
**Status of experiments:** **Planned** · **Ready** (built, can run) · **Running** · **Blocked** (by a review finding or a dependency) · **Done** (with result ID) · **Superseded** (answered by another result or replaced by another item) · **Dropped** · **Duplicate** (see the named item).
**Priority:** **High** next round · **Med** soon · **Low** later.
**Seeds:** results are from small CPU models (0.1–0.4M parameters) or frozen 2B models, mostly single seeds. Treat a difference of a few points as noise.

### ID scheme

| Prefix | Meaning | Defined in |
|---|---|---|
| M1–M8 | Results: memory systems (CPU) | §1 |
| T1–T21 | Results: temporal loop and clock (CPU) | §1 |
| P1–P12 | Results: pretrained models (GPU) | §1 |
| N1–N5 | Steering spot checks on Qwen3 (not experiments) | §1 |
| B1–B15 | Corrections and fixed bugs. B1–B8 were C1–C8 before 2 Nov 2026 | §1 |
| DEC1 | Decisions. DEC1 was O3 | §2 |
| RV1.A1–A8, RV1.C1–C8, RV1.D1–D4 | Findings of review RV1 (the review itself numbers them A, C, D) | §3; the review file |
| RV2.X1–X8, RV2.K1–K3, RV2.S1–S3, RV2.D1, RV2.F1 | Findings of review RV2, the literature review: code and experiments (X), CA3–CA1 circuit (K), CA2 steering (S), DG (D), pathfinding (F) | §3; `research/reviews/2026-11-02_literature/` |
| D1–D5 | Round-0 diagnoses | `round0_plan.md` §1 |
| C1–C6, C1a | Round-0 diagnostics that need code | `round0_plan.md` §2; §4.1 |
| E1–E9 | Evaluator rules for candidates | `round0_plan.md` §3 |
| g0–g7 | Round-0 diagnostic runs: the control and the `flat` / `exp 0.98` arms. g2b and g5h are later additions on the horizon kernel | `round0_plan.md` §2; §4.1 |
| W1–W11 | Candidates and scaling experiments on pretrained models. W5–W7 were D-a, D-b, D-c | §4.2 |
| S1–S9 | Feedback stability programme | NOTEBOOK; §4.3 |
| A1–A8 | Alignment track. A6.H1–A6.H4 are A6's hypotheses | NOTEBOOK; §4.4 |
| R1–R6 | Reasoning: CA2 steering of chain-of-thought | NOTEBOOK; §4.5 |
| J1–J6 | CA1 as a calibrated decision model | NOTEBOOK; §4.6 |
| O1–O13 | Other open questions | §4.7 |
| L1–L16 | Experiments proposed by the literature review (RV2) | §4.8 |
| DR1–DR5 | Findings of the NOTEBOOK design review of 2 Nov 2026 (R1–R5 there before the rename) | NOTEBOOK |

Sub-IDs: a letter suffix marks a variant or follow-up (P4a, C1a, g2b, g5h, W1b); a dot marks an arm or part inside an item (W1b.a, A6.H1, J3.a, W6.a, RV1.A1). Not IDs: $K = Ag\beta$ is the loop gain, and "capacity H4" and similar name round-0 lens hypotheses. The NOTEBOOK's stability findings K1–K6 map to T8, T6, T7, T9/T10/T3, T20 and a gap (S1, S2).

## 1. Results and conclusions

### Memory systems (RESULTS.md §1–3)

| ID | Claim | Evidence | Result file | Seeds | Status |
|---|---|---|---|---|---|
| M1 | Hippocampal memories recall bindings beyond a 16-token window; windowed attention alone stays at chance | Strategy I 0.41, Titans 0.48, full attention 0.49, windowed 0.06 | `results/synthetic/assoc_recall.json` | 1 | **Open** (RV2.X1): the task admits a binding-free strategy scoring 0.476 (remember the studied values, strike each as revealed). Full attention 0.485 and Titans 0.483–0.490 sit at it, Strategy I below it, so this shows set memory, not binding. Only successor-tied memories bind (M5). Re-test under L1 |
| M2 | Stacked DG writes and CA1 novelty-lr on Titans help | Neutral alone (DG writes 0.478, CA1 + novelty-lr 0.440, plain Titans 0.483); 0.17 together, peak 0.183 at step 2,500 | `results/synthetic/assoc_recall_ablation.json`, `results/synthetic/assoc_recall.json` | 1 | **Open** (RV2.X1): only tied addressing solves this task, so neither DG writes nor novelty-lr could show a benefit on it |
| M3 | Novelty-prioritised replay with sleep prevents forgetting | Backward transfer −0.73 → 0.00 | `results/synthetic/continual.json` | 1 | Supported; no uniform-replay or sleep-only control (O1). RV2.X5: the 256-sequence buffer holds nearly every prior fact; novelty priority and sleep are untested (O1) |
| M4 | Standard Gated DeltaNet loses recall once stored pairs exceed its key size | 0.43 → 0.13 at 8 → 16 pairs | `results/synthetic/gdn_capacity.json` | 1 | **Refuted** (RV2.X1): not a capacity limit. `gdn_x8` (d_k = 128) scores 0.428 / 0.320 / 0.222 at 8 / 16 / 32 pairs against a binding-free ceiling of 0.423 / 0.319 / 0.243, far below d_k. Standard GDN never learned binding, likely the identity-initialised short convolutions (B12) and a single learning rate. Re-test under L1 |
| M5 | Successor-bound addressing makes the GDN state exact RAM | 1.00 up to 32 pairs, dense (`tied_x8`) and DG (`dg_x8`) addresses. The counterfactual quoted in RESULTS §3 (100% new / 0% old after swapping values) has no result file or test | `results/synthetic/gdn_capacity.json` | 1 | Supported (exact recall); counterfactual unbacked. RV2: loads never exceed d_k, so 1.00 is the expected linear-memory result (Kohonen 1972; Schlag et al. 2021), and 100% new / 0% old is the delta rule's defining overwrite. Capacity above d_k: O9 (extended by L3) |
| M6 | An exact delta-rule memory beats Titans' deep memory | 1.00 against 0.48 | `results/synthetic/assoc_recall_gdn.json` | 1 | **Open** (RV2.X4): confounded. `titans_gdn` ties the query and key projections (a hard-wired previous-token address), while `NeuralMemory` keeps them separate and is a chunk-level Titans variant. Factorial ablation in L1 |
| M7 | DG addresses beat dense addresses | Same final recall; DG faster early (0.905 vs 0.680 at step 500) | `results/synthetic/assoc_recall_gdn.json` | 1 | Open (task does not separate them; O9). RV2.D1: under GDN's linear readout sparse keys cannot raise the d_k bound; test above d_k with overlapping keys (O9) |
| M8 | Wider keys alone fix standard GDN's capacity | `gdn_x8` (128-dim keys): 0.43 / 0.32 / 0.22 at 8 / 16 / 32 pairs, against 1.00 with successor binding (M5) | `results/synthetic/gdn_capacity.json` | 1 | Refuted (wider keys help, but do not fix it) |

### Temporal loop and clock (RESULTS.md §4.1–4.6)

| ID | Claim | Evidence | Result file | Seeds | Status |
|---|---|---|---|---|---|
| T1 | The loop carries content and the clock carries elapsed time | What 1.00 / when 0.85 (loop); when 0.88 (clock); both 1.00 / 0.92 | `results/synthetic/temporal_loop_all.json`, `results/synthetic/temporal_loop_long.json` | 1 per config | Supported |
| T2 | "What" is learned in a sudden late jump; 1,500 steps is too short for clamped-sinusoid loops | Full design crosses 0.5 at step 1,000 of 3,000 (0.537 and 0.527 on two seeds) and never within 1,500 (peak 0.158). Other designs: note below | `results/synthetic/temporal_loop_long.json`, `results/synthetic/temporal_loop_long_seed1.json`, `results/synthetic/temporal_loop_all.json` | 2 | Supported |
| T3 | The full design keeps closed-loop echo bounded and stationary | 0.29–0.35× the input, peak ≤ 0.53 | `results/synthetic/temporal_loop_long.json`, `results/synthetic/temporal_loop_long_seed1.json` | 2 | Supported (the bound comes from the clamp) |
| T4 | Undamped closed loops run away; damping alone overshoots early; the clamped sinusoid without control resonates | 4 → 14× (peak 63×); 1.0× early (peak 10.6×); about 3× | `results/synthetic/temporal_loop_probe.json` (undamped, damped), `results/synthetic/temporal_loop_all.json` (no control) | 1 | Supported |
| T5 | Damping trades long recall for recency | Recall at gaps of 300–500: 0.40 undamped, 0.00 damped, 0.04–0.10 full design. Open loop at 1,500 steps: 1.00 / 0.61 / 0.17 at gaps 20–140 / 140–300 / 300–500 (note below) | `results/synthetic/temporal_loop_probe.json`, `results/synthetic/temporal_loop_long.json`, `results/synthetic/temporal_loop_long_seed1.json`, `results/synthetic/temporal_loop_all.json` | 1–2 | Supported |
| T6 | Subtractive inverted-copy feedback obeys the op-amp law on persistent echo | $y_\text{sub}/y_\text{none}$ = 0.787, 0.647, 0.475, 0.373 against $1/(1+K)$ = 0.788, 0.650, 0.482, 0.383 | `results/synthetic/loop_systems.json` | 3 (untrained) | Supported. RV2: an analytic property of the linear delayed loop, so untrained seeds add no evidence; Bode's sensitivity integral applies (RV2.K3, L7) |
| T7 | Delayed subtractive feedback rings above $K = 1.68$ | Stable at 1.08, growing at 2.15 and above; period about 44 tokens | `results/synthetic/loop_systems.json` | 3 (untrained) | Supported. RV2: reproduced independently, K = 1.680, period 44.1 tokens |
| T8 | Divisive feedback only compresses | Undamped kernel, $A$ = 32: 0.69 against 2.39 | `results/synthetic/loop_systems.json` | 3 (untrained) | Supported |
| T9 | Without the clamp, only subtractive feedback learns content | What 0.99 / 1.00 against 0.13 (none) and 0.15 (divisive) | `results/synthetic/temporal_loop_feedback.json`, `results/synthetic/temporal_loop_open.json` | 2 | Supported |
| T10 | Subtractive feedback bounds the echo | 1.7–2.6× the input | `results/synthetic/temporal_loop_feedback.json`, `results/synthetic/temporal_loop_open.json` | 2 | Refuted. RV2.K3: expected from Bode's sensitivity integral. At K = 1 (τ 16, ρ 0.9) DC echo halves but content near a 46-token period is amplified about 2.5× (L7) |
| T11 | Closed loops favour the first event, by self-rehearsal | Older named 0.61 (undamped), 0.84 (damped), 0.63 (full design without contrast) | `results/synthetic/temporal_loop_probe.json`, `results/synthetic/temporal_loop_ablations.json` | 1 each | Supported (mechanism is a hypothesis). RV2: one seed per arm with a zero-shot probe; treat as Open until O2 |
| T12 | The contrast against the prior reduces primacy | Older 0.63 → 0.18 | `results/synthetic/temporal_loop_ablations.json`, `results/synthetic/temporal_loop_probe.json` | 1 | Supported, weakly; it also halves learning speed (crosses at step 1,000 vs 500). RV2: one seed per arm with a zero-shot probe; treat as Open until O2 |
| T13 | Sparse DG write keys favour the newer event | Newer 0.68, older 0.29–0.31 | `results/synthetic/temporal_loop_ablations.json` | 1 (two arms) | Supported, weakly. RV2: one seed per arm with a zero-shot probe; treat as Open until O2 |
| T14 | The open loop shows little primacy | Newer 0.45 / older 0.38 | `results/synthetic/temporal_loop_open.json` | 1 | Supported, weakly (different seed). RV2: one seed per arm with a zero-shot probe; treat as Open until O2 |
| T15 | Subtractive feedback breaks primacy | Older 0.54 (seed 0), 0.38 (seed 1) | `results/synthetic/temporal_loop_feedback.json`, `results/synthetic/temporal_loop_open.json` | 2 | Mixed |
| T16 | A direct EC → CA3 read completes degraded cues better than a DG read (Treves–Rolls) | At σ = 2: 0.84 direct, 0.94 DG read; dense keys without contrast 0.97 | `results/synthetic/temporal_loop_ablations.json` | 1 | **Open** (RV2.K2): not a test of Treves–Rolls. There is no CA3 recurrent completion stage, and `read=direct` vs `read=dg` also switches untied vs tied projections. Proper test: L9 |
| T17 | A trained recency task ranks the loops | No model learned it (what 0.14–0.15) | `results/synthetic/temporal_loop_recency.json` | 1 | Open (inconclusive) |
| T18 | The clock removed a curriculum, so memory + time failed "what" | Learned by step 500 with a longer schedule (0.998) | `results/synthetic/temporal_loop_long.json` | 1 | Refuted |
| T19 | Training is deterministic | A rerun reproduced every seed-0 number exactly: the full design's curves and rollouts are identical in the two files | `results/synthetic/temporal_loop_long.json`, `results/synthetic/temporal_loop_probe.json` | — | Supported |
| T20 | No trained closed loop diverges numerically; the next LayerNorm rescales the stream, so runaway shows as echo outweighing input, not as overflow | `diverged_at` is null in every trained run, including echo peaks of 63× (undamped) and 189× (divisive). The untrained sweep, which has no LayerNorm, does overflow above $K_\text{crit}$ (T7) | `results/synthetic/temporal_loop_*.json` | 1–2 | Supported |
| T21 | Subtractive feedback is a high-pass filter: at a stable loop gain it removes only about a quarter of sustained echo | Varied stream, $A$ = 4: echo 0.012 with and without it (`osc`), 0.289 against 0.278 (`flat`). Trained gains $K$ = 0.33 (seed 0) and 0.31 (seed 1), so $1/(1+K)$ ≈ 0.75 on persistent echo (T6) | `results/synthetic/loop_systems.json`, `results/synthetic/temporal_loop_feedback.json`, `results/synthetic/temporal_loop_open.json` | 2 | Supported (the quarter is T6's law at the trained gain, not a direct measurement) |

Notes:
- **T2, learning speed** (RESULTS §4.6). Steps to cross 0.5: damped 500; undamped 750 (1,500-step run) and 1,000 (3,000-step run); full design 1,000 on both seeds; full design without contrast 500; DG keys 500; subtractive 2,000 (seed 0) and 1,000 (seed 1); open loop 1,250 (1,500-step run) and 1,000 (3,000-step run). The unregulated clamped sinusoids (no feedback, divisive only) did not cross in 3,000 steps. Files: `temporal_loop_probe.json`, `_all.json`, `_ablations.json`, `_feedback.json`, `_open.json` in `results/synthetic/`.
- **T5, open loop.** The 1.00 / 0.61 / 0.17 fall-off is from the seed-0 run at 1,500 steps (`temporal_loop_all.json`). The seed-1 open loop at 3,000 steps keeps 1.00 / 1.00 / 0.82 at the same gaps (`temporal_loop_open.json`). One seed each, different seeds.

### Pretrained models (RESULTS.md §4.7–4.8, [`../results/README.md`](../results/README.md), NOTEBOOK "Round 1")

| ID | Claim | Evidence | Result file | Seeds | Status |
|---|---|---|---|---|---|
| P1 | Qwen3.5's GDN layers alone do not recall a needle 500 tokens back | Exact 0.00 with attention windowed and GDN intact | `results/pretrained/pretrained_qwen35_needle.json` (`windowed`) | 1 | Supported. RV2.X3: the windowed baseline is below the answer-format floor (loss 4.03 against 1.84), so the frozen model is out of distribution; needs the memory-free adapter control (L2) |
| P2 | The side channels restore exact recall on Qwen3.5, through the GDN layers above the tap | Exact 1.00 tapping block 12, 0.00 tapping block 22 | `results/pretrained/pretrained_qwen35_needle.json`, `results/pretrained/pretrained_qwen35_needle_layer22.json` | 1 | Supported (the route is a hypothesis; C4). RV2.X3: two readings remain, re-injection through later GDN layers or adapter repair of the windowing shift with GDN carrying the needle (the loop kernel is 0 at the gap; gain 0.2689 → 0.2697). Controls L2, probes L4 |
| P3 | With the default kernel the loop cannot reach the needle | Horizon τ/ln(1/ρ) ≈ 45 tokens; Qwen3 at the format floor, 1.842. The gain did train (0.2689 → 0.2797), so bf16 rounding (B3) was not the only cause | `results/pretrained/pretrained_qwen3_needle.json` | 1 | Supported (round-0 D1, D2) |
| P4 | With a horizon that covers the gap, the loop alone carries needle information | Digit loss 1.38 (Qwen3) and 1.24 (Qwen3.5), below the 1.842 floor; token accuracy 0.46–0.55; exact 0.00–0.02 | `results/pretrained/pretrained_qwen3_needle_horizon.json`, `results/pretrained/pretrained_qwen35_needle_allwin_horizon.json` (Qwen3.5 run superseded by P9) | 1 | Supported (partial recall at 3,000 steps) |
| P4a | The loop recovers whole digits, not partial information about each one | Qwen3 1.381 / 0.461 equals one digit known and three at chance, $(3/5)\ln 10$ / 0.46; Qwen3.5 lies between one and two digits | `results/pretrained/pretrained_qwen3_needle_horizon.json`, `results/pretrained/pretrained_qwen35_needle_allwin_horizon.json` | 1 | Supported (P7 confirms the digit). Mechanism: key and value come from the same position, so nothing moves the lookup to the next digit. This motivates W1 |
| P5 | The side channels improve WikiText perplexity on windowed models | Qwen3, all windowed: 14.6 → 12.0 (full 9.0; 41% of the loss gap). Qwen3.5, attention windowed: 16.3 → 11.2 (full 9.8; 74%) | `results/pretrained/pretrained_qwen3_lm.json`, `results/pretrained/pretrained_qwen35_lm.json` | 1 each | Open (confounded with domain adaptation, since only that condition trains; O4) |
| P6 | The default kernel helps Qwen3.5 when every layer is windowed | Float32 rerun at the floor, 1.843 / 0.289. The earlier 1.69 was the bf16 leak (B4) | `results/pretrained/pretrained_qwen35_needle_allwin.json` | 1 | Refuted |
| P7 | The horizon loop recovers the *first* digit | Per-digit loss 0.001 / 2.30 / 2.30 / 2.30. The run reproduces `results/pretrained/pretrained_qwen3_needle_horizon.json` (P4, P4a) exactly: loss 1.3809, token accuracy 0.4609, gain 0.3264 | `results/evolve/round1/g1_qwen3_horizon_perpos.json` | 1 | Supported |
| P8 | The horizon loop's one-pair recall uses the query | With 2 stored pairs, digit 1 goes back to near chance (loss 2.17, acc 0.23). Needle gap 490–502 tokens (`model.gap_range`), not 501 | `results/evolve/round1/g1_qwen3_horizon_pairs2.json` | 1 | Refuted: query-independent |
| P9 | With enough steps the horizon loop alone gives exact recall on Qwen3.5 with every layer windowed | 6000 steps: exact 1.000, loss 0.001. Supersedes the 3,000-step run (`results/pretrained/pretrained_qwen35_needle_allwin_horizon.json`, undertrained), the Qwen3.5 evidence for P4 and P4a | `results/evolve/round1/g1_qwen35_allwin_horizon_6k.json` | 1 | Supported (1 pair only; see P8, W1f). RV2.X2: one pair, and P8 shows one-pair recall is query-independent, so this carries one digit string rather than binding. It does rule out adapter repair when every layer is windowed |
| P10 | **W1 (GDN side memory, successor-bound keys) gives exact recall on Qwen3 with every layer windowed** | Dense keys: exact 1.000, loss 0.0001, solved by step 1,200. DG keys: exact 1.000, loss 0.0001, 0.008 by step 600, 12.6% fewer parameters (3,742,795 against 4,282,451) | `results/evolve/round1/g1_w1_tied.json`, `results/evolve/round1/g1_w1_dg.json` | 1 | Supported (1 pair). RV2.X2: one pair, as P9. Closest published system: δ-mem (Lei et al. 2026; L5). The loss below full attention's compares a trained side module with an untrained frozen base |
| P11 | W1 selects the queried pair | 2 pairs: exact 0.55 (dense) / 0.43 (DG). Digits 2–4 at 0.94–0.99, but digit 1 at 0.57 / 0.51 with loss 0.74 / 0.76 ≈ ln 2: a coin flip between the two stored codes. The chain is retrieved; the start is not selected. Needle gap 490–502 tokens, not 501 | `results/evolve/round1/g1_w1_tied_pairs2.json`, `results/evolve/round1/g1_w1_dg_pairs2.json` | 1 | Partly: sequence recall yes, start selection no (W1b). RV2.F1: the missing-key / start-selection problem of Samsonovich & Ascoli (2005). The successor chain (their J) works; a query-conditioned read cue or a goal field (their W) must pick the start (L6) |
| P12 | A horizon kernel improves LM perplexity over the default kernel | Qwen3 WikiText 12.0 → 11.5 (41% → 49% of the gap) | `results/evolve/round1/g1_qwen3_lm_horizon.json` (against `results/pretrained/pretrained_qwen3_lm.json`) | 1 | Supported, weakly; still confounded (O4) |
| P13 | Context-bearing addresses let the queried name select the first digit (W1b) | No variant does. Digit-1 loss ≈ ln(pairs) throughout: 2 pairs, W1 0.74, context conv K = 4 0.74, + DG keys 0.80, gdn mode 0.80, tap 21 0.65 (ln 2 = 0.69); 3 pairs, W1 1.20, context conv 1.22, gdn mode 1.34, tap 21 0.97 (ln 3 = 1.10). The read at the space after "is" returns an even mix of all stored first digits | `results/evolve/round2/r2_*.json` | 1 each | Refuted for these variants at 3,000 steps. RV2.F1: as predicted, more history in the key *at the answer position* cannot select the start. The read must be conditioned on the query span (dense Treves–Rolls cue) or scored by a goal field (L6) |
| P14 | Explicit successor binding carries the chain after the first digit | Digits 2–4 at 0.91–1.00 for W1, context conv and tap 21; at chance for gdn mode, which has no explicit successor binding (`r2_w1b_gdn_pairs2/3`) | `results/evolve/round2/r2_w1b_gdn_pairs2.json`, `r2_w1b_gdn_pairs3.json` | 1 | **Revised (round 3):** what carries the chain is the token shift, not explicit binding as such. With the key conv started on the previous token (`--w1-conv-init shift`), gdn mode learns the chain (digits 2–4 at 0.88 / 0.73 / 0.42, still improving at 3,000 steps); with a random start it does not (chance), as with the identity start (B12). Explicit binding learns faster. `results/evolve/round3/r3_w1b_gdn_shift_pairs2.json`, `r3_w1b_gdn_random_pairs2.json` |
| P15 | A later tap (block 21) helps | Fastest learner (2 pairs: loss 0.19 by step 300) and the only slight selection with 3 pairs (digit 1 loss 0.97, acc 0.48 against chance 0.33) | `results/evolve/round2/r2_w1b_late21_pairs2.json`, `r2_w1b_late21_pairs3.json` | 1 | Supported, weakly |
| P16 | Starting the context conv as an even average puts the name in the address from step 0 | It blurs the successor chain instead: 2 pairs digits 2–4 fall to 0.52–0.65 (exact 0.11), 3 pairs near chance; digit 1 still ≈ coin flip (0.84 / 0.58) | `results/evolve/round2/r2_w1b_ctx4u_pairs2.json`, `r2_w1b_ctx4u_pairs3.json` | 1 | Refuted |
| P17 | The queried name is available at the start cue, but W1's addresses discard it | Frozen windowed Qwen3-1.7B, logistic probe for which name (chance 0.04): 0.95–0.96 at the query's space token at block 14, 0.98–1.00 at block 21; 1.00 at the facts' space tokens. Trained W1 addresses there: cosine between the two facts 0.785 (block 14) / 0.88 (block 21); the query address is closest to the queried fact at chance (0.51 / 0.34 with 2 / 3 pairs) | `results/evolve/round3/probe_cue.json` | 1 | Supported. The start-selection failure (P11, P13) is a key-projection problem, not missing information |
| P18 | **A DG-keyed start store (W1g) selects the queried code** (corrected 2 Nov: it does not; see the status) | Tap 21, 2 pairs: exact 0.984, digit 1 loss 0.002 / acc 1.00 (W1 at tap 21: 0.49, 0.65 / 0.53; round-2 ablation, 8 heads against W1g's 5). Tap 14, 2 pairs: exact 0.43, digit 1 0.855 / 0.48, no better than its no-goal ablation (0.42, 0.892 / 0.46). 1 pair at tap 14: exact 1.000 (no regression) | `results/evolve/round3/r3_w1g_t21_pairs2.json`, `r3_w1g_t14_pairs2.json`, `r3_w1g_nogoal_t14_pairs2.json`, `r3_w1g_t14_pairs1.json` | 1 | **Refuted as selection (round-4 diagnostic, `results/evolve/round4/diag_goal.json`):** the 2-pair model retrieves the *first* and *last* fact only. On 3-pair data its digit-1 accuracy by queried fact is 1.00 / 0.08 / 0.92 (first / middle / last). G's query key is closest to the first fact in 256 of 256 sequences whichever fact is queried; the last fact comes through the delta-rule overwrite. With 2 pairs first + last covers every query. A positional shortcut of the fixed task layout (facts right after BOS, in order), not content addressing |
| P19 | W1g extends to 3 pairs | Tap 21: exact 0.28, digit 1 1.004 / 0.38 (ln 3 = 1.10; chance 0.33); tap 14: 0.20, 1.374 / 0.38. Training loss plateaus at ~0.25 from step 900 | `results/evolve/round3/r3_w1g_t21_pairs3.json`, `r3_w1g_t14_pairs3.json` | 1 | Refuted at 3,000 steps. Explained by P18's shortcut: the middle fact is in neither store. The 3-pair-trained model shows no structure (0.49 / 0.35 / 0.48 by queried fact) |
| P20 | CA3 settling (W1h, K = 3, gain 4, k-WTA 5%) sharpens the start cue | It undoes W1g's 2-pair success at tap 21 (exact 0.45, digit 1 0.66 / 0.49) and does not help elsewhere (tap 14, 2 pairs 0.48; 3 pairs 0.24 / 0.27, digit 1 at ≈ ln 3). Likely cause: the autoassociative store holds every earlier position, including the query's own "<name> is" tokens, which share the name and are the most recent, so the cue settles onto them instead of the stored fact | `results/evolve/round3/r3_w1h_*.json` | 1 | Refuted as built; the store's contents need gating |

Notes:
- **Learned gain and injection ratio** (every run starts at gain 0.2689). Default-kernel needle runs end at gain 0.270–0.293 with injection ratio 0.20–0.22. Horizon runs: gain 0.326 (Qwen3) and 0.277 (Qwen3.5), ratio 0.23 and 0.21. LM runs: gain 0.368 and 0.374, ratio 0.25 and 0.26. W1 (P10, P11): ratio 0.31–0.33, gain 0.34–0.37, under the same CA1 clamp. All from `results.gain` and `results["windowed+side"].inj_ratio` in the files above.
- **W1 CPU check (P10).** A repeated context reads out the value that followed it, at cos 0.57 with dense keys and 0.85 with DG keys (`tests/test_evolve_w1.py`; numbers from NOTEBOOK "Round 1"; no result file).
- **Round-0 refuted hypotheses** (read sharpness, contrast or the CA1 gate as the bottleneck; "the loop structurally cannot reach the needle", a CPU check at 97%) are listed in `round0_plan.md` §1. They rest on code reading and an unrecorded CPU check, so they have no P rows.
- **Provenance of the GPU files.** Every `results/pretrained/*` and `results/evolve/round1/*` JSON records `args.dtype: "bfloat16"`, the base model's dtype. The side channels ran in float32 (commit `3c90bde`, B3); only `results/README.md` (Provenance) and RESULTS §4.7–4.8 record this. The bf16 versions are in git history (`0f5cc8c`, `775e0f2`, `a741b80`, `2297571`). The `g1_w1_*` files record the default loop arguments; the configuration actually used is under `candidate`. `results/synthetic/temporal_loop_recency.json` records `args.models: "closed_osc_ca1"` but holds three models. `args.out` and the logs' "saved …" lines give the paths from before the 2 Nov reorganisation.

### Steering spot checks on Qwen3-1.7B (NOTEBOOK, 2 Nov)

Abliterated target, block 14 of 28 unless stated, 4 samples × 3–4 prompts. Spot checks, not experiments.

| ID | Claim | Evidence | Result file | Seeds | Status |
|---|---|---|---|---|---|
| N1 | Prompt-position ActAdd (the GPT-2-XL recipe) steers a chat model | No refusals at COEFF 5–20, blocks 6–14. Only the first ~10 chat-template tokens are steered; content shifts slightly, behaviour does not | `docs/notebooks/activation_additions_qwen3.ipynb` | 1 | Refuted (a failure of position, not gain) |
| N2 | The abliterated model's own harmful − harmless direction restores refusal | 0/12 harmful samples refuse at g = 0.1–0.3 (‖v‖ 26 against ‖h‖ 180) | `docs/notebooks/activation_additions_qwen3.ipynb` | 1 | Refuted, as A6 expects (a failure of direction) |
| N3 | The aligned sibling's direction restores refusal selectively | g ≤ 0.2: no refusals. g = 0.3: harmful 2/4, borderline 0/4, benign 0/4, degeneration ≤ 0.05 | `docs/notebooks/activation_additions_qwen3.ipynb` | 1 | Supported, weakly. Onset between 0.2 and 0.3, not saturated: gain-limited |
| N4 | The saturation knee depends on the direction | At block 14, g = 0.3, the thinking direction degenerates on 4/4 problems (degeneration 1.00), while the refusal direction stays coherent | `docs/notebooks/activation_additions_qwen3.ipynb` | 1 | Supported, weakly. S1 must sweep the directions in use |
| N5 | A thinking-on − thinking-off direction shortens no-think CoT at equal accuracy | 4/4 correct throughout. g = 0.2: tokens −4 to −30% on all four problems. g = 0.1: mixed, −8% to +4%. The baseline is also 4/4, so the problems are too easy | `docs/notebooks/activation_additions_qwen3.ipynb` | 1 | Open |

### Corrections and fixed bugs

| ID | What was wrong | Fix | Status |
|---|---|---|---|
| B1 | The "when" baseline was taken as 0.25 | The bins are unequal; the prior scores 0.333 | Fixed |
| B2 | Sliding windows were assumed to make memory the only route | They leak $L(W-1)$ tokens through depth; block windows are now the default | Fixed |
| B3 | bf16 side channels: Adam updates rounded away | Side channels train in float32 | Fixed |
| B4 | bf16 index arithmetic: positions merged, an anti-causal leak | Float32 indices, plus a regression test (round-0 D5) | Fixed |
| B5 | The design note claimed an $O(1)$-state form for the loop | Only true for factorised relevance; the loop is $O(T)$ | Fixed |
| B6 | Rollout length off by one | Gaps stop at 497 | Fixed |
| B7 | CI depended on numpy | Torch companion-matrix roots | Fixed |
| B8 | Interim commits rewrote the logs of runs still in progress | Logs rebuilt from JSON; in-progress logs are no longer committed | Fixed |
| B9 | The Titans meta-gradient flowed through the read path, so the training-mode update did not match the inference-mode update | The meta-gradient is taken as a partial derivative | Fixed |
| B10 | The memory loss was averaged over features, so writes were scaled down by $1/d$ | Summed loss with a per-sample clip (a trust region) | Fixed |
| B11 | CA3 was purely autoassociative, so a cue returned itself instead of its successor | CA3 binds each pattern to its successor | Fixed |
| B12 | `gated_delta.py` `CausalConv1d` initialises the short convolutions to the identity (current position only; fla uses random init), so standard `gdn` addressing, W1b.a included, starts with no token shift | Random or shift-biased init; re-run M4–M7 (L1) and W1b.a | Open |
| B13 | Biology in docstrings: CA2 "lacks mossy-fibre input"; mossy cells as "feedback inhibition"; `TimeChannel` as time cells; the exact-RAM limit stated as reachable | Docstrings corrected (Kohara et al. 2014; Scharfman 2016; Shankar & Howard 2012 Laplace stage; α, β < 1 under the parametrisation). RV2 | Fixed |
| B14 | Bibliography: no DOIs, stale venues, placeholder author lists, five misdescribed entries | `docs/references.bib` replaced by the audited version (all keys kept, 49 DOIs); REFERENCES.md corrected; 78 entries added. RV2 | Fixed |
| B15 | genesis §3 presented "GDN = one gradient step on the Titans loss" as derived in the session | It is stated in Yang, Kautz & Hatamizadeh 2025 §3, DeltaNet and Longhorn; credited in genesis §9 and README. RV2 | Fixed |
| B16 | W1 was described as modelling CA3 | W1 is a linear fast-weight sequence store: it has CA3's one-shot heteroassociative storage rule but no recurrence in activity, no sparse activity gating and no high gain (Daniel, 2 Nov). The CA3 dynamics are W1h (`--w1g-settle`) | Corrected (NOTEBOOK, round 3) |

B9–B11 are the three bugs in RESULTS.md, "Bugs the experiments caught", and README, "Lessons from the experiments". Each is covered by a test.

## 2. Decisions

| ID | Decision | Approved | Consequences | Was |
|---|---|---|---|---|
| DEC1 | The pretrained track requires a fixed-size state (rule E2). The O(T) loop with the horizon kernel stays as the reference arm | Daniel, 2 Nov 2026, when he approved the round-1 plan. Recommended by both sessions (NOTEBOOK; HANDOFF at `53caa77`) | W1 meets it (P10). The `flat` and `exp` round-0 arms fail E2 and are diagnostics only (§4.1) | O3 |

## 3. Open review findings (RV1)

RV1 is [`reviews/2026-11-02_alignment_cot_ca1.md`](reviews/2026-11-02_alignment_cot_ca1.md), a review of commits `73259fc` and `241a13c`. Severity is the review's. None is fixed: since the review, `experiments/alignment/` has changed only in docstrings.

| ID | Finding | Severity | Blocks | Smallest fix (from the review) | Status |
|---|---|---|---|---|---|
| RV1.A1 | The perplexity check never activates the conditional, graded and graded_self arms (they inject only at the last position, whose logits are dropped), so A6's capability metric is vacuous for them | High | A2, A6 | Prefill a prefix, teacher-force the continuation through the KV cache, score only the continuation | Open |
| RV1.A2 | The probe is fit and calibrated on the last prompt token, but the graded arm integrates it over generated tokens (same as RV1.D2) | Med | A1 (caveat), J1; distorts A2 and A6 graded-vs-conditional | Report probe score and ECE on generated-token activations; aggregate s over generated tokens per set | Open |
| RV1.A3 | Degenerate loops count as refusals | Med | A2, A6 | Count a refusal only when the output is non-degenerate; print degeneration for every set | Open |
| RV1.A4 | The single-pass equivalence test passes for conditional only by its seed; the `"last"` path that `run.py` uses is untested | Med | — (test gap under A2, A6) | Test `score_positions="last"` with a padded prefill and decode steps | Open |
| RV1.A5 | Gain and score diagnostics include pad positions | Low | — | Average over real positions; carry an EOS mask through decode | Open |
| RV1.A6 | Data-rule bypass: an absolute or `../` value of `--name` writes raw completions inside the repository | Low (data rule) | A2, A6 (any run with `--name`) | Validate the final resolved path | Open |
| RV1.A7 | `--direction-from` keeps the first model's layers in memory; layer counts are never compared | Low | A6 with `--direction-from`, A7 | Free the first model; assert matching layer counts and widths | Open |
| RV1.A8 | Missing `position_ids` on left-padded passes | Low | — (absolute-position models only, not Qwen) | Pass `position_ids` derived from the mask | Open |
| RV1.C1 | The gist direction is dominated by the attention-sink token and the stream mean, not the current thought | High | R2 (R4 via R2); the S1 gist arm | Exclude the sink from both EMAs; initialise from the first included token or subtract a corpus mean; test that the background component is near zero | Open |
| RV1.C2 | R2/R4 cannot attribute a shorter CoT to the gist: no matched-norm random, mean or anti-gist controls, no sampling protocol, no non-inferiority margin, no truncation rate; `/no_think` is not a baseline for R1-Distill | High | R2, R4 | Add those arms; k ≥ 4 samples at T ≈ 0.6 with fixed seeds; pre-register a margin; report truncation | Open |
| RV1.C3 | Graded gist with no probe is silently a no-op; R3 has no gain law | Med | R3 | Raise on graded mode without a probe; define R3's gain law | Open |
| RV1.C4 | STU-PID, R3's baseline, was withdrawn on 28 Sep 2026, and its quoted figures are stale | Med | R3 | Use arXiv 2510.04309 as the published baseline; reimplement decode-time PID as an R3 ablation | Open |
| RV1.C5 | The novelty claim misses ERSS (arXiv 2606.06188); the over-commitment risk misses STARS (arXiv 2601.22484) | Med | — (NOTEBOOK prior art) | Cite both; narrow the claim to the EMA form with negative feedback | Open |
| RV1.C6 | "Open loop" holds only within one forward pass; sampled tokens feed the injection back | Low | S1 gist arm | Add a repetition or distinct-n metric on long gist generations | Open |
| RV1.C7 | R1's thinking direction is confounded by template tokens | Low | R1 | Take the difference at matched content positions | Open |
| RV1.C8 | DSEM is mis-described | Low | — | Correct the description | Open |
| RV1.D1 | J1's hypothesis ("at equal AUROC, better calibration gives higher selectivity") is ill-posed for a gain of cap·tanh(κE) | High | J1 | Compare sources at a matched operating point (fixed harmful-refusal rate) | Open |
| RV1.D2 | J1 measures ECE where the probe was trained, not on the signal that drives the gain (see RV1.A2) | Med | J1 | As RV1.A2, at the positions the controller uses | Open |
| RV1.D3 | J3's premise is misstated (the harm came from stacking novelty-lr with DG writes, M2), it bundles warm-up with calibration, and ECE is undefined for the novelty target | Med | J3 | Correct the premise; run warm-up alone (J3.a); define a calibration measure | Open |
| RV1.D4 | J1 is not a cheap add-on: two of its four arms cannot plug into `score()` | Low | J1 | Add an external score callback or make those arms conditional-only; temperature-fit LAYA | Open |

The review also cites lens findings "F2" (under RV1.C1) and "F4" (under RV1.C4) that are recorded nowhere else. Its `REGISTER.md:` anchors refer to this file at commits `73259fc` and `241a13c`.

### Open review findings (RV2, literature review)

Report: [`reviews/2026-11-02_literature/`](reviews/2026-11-02_literature/README.md) (synthesis; track reports `code_review.md`, `ca3ca1.md`, `ca2.md`, `dg_moe.md`, `gdn.md`, `pathfinding.md`, `bib_audit.md`). Bugs it found are B12–B15.

| ID | Finding | Severity | Blocks | Smallest fix (from the review) | Status |
|---|---|---|---|---|---|
| RV2.X1 | `AssociativeRecall` (teacher-forced shuffled queries, values with replacement from 16) admits a binding-free strategy: 0.476 at 6 pairs, 0.423 / 0.319 / 0.243 at 8 / 16 / 32 (`reviews/2026-11-02_literature/scripts/binding_free_baseline.py`) | High | Claims M1, M2, M4, M6, M7 | Report against the ceiling and first-query accuracy; move to MQAR (L1) | Open |
| RV2.X2 | The one-pair needle is passkey retrieval (one digit string in digit-free filler); P8 confirms that one-pair recall ignores the query | Med | Claims P9, P10 | Score 2–3 pairs (W1f) and RULER multi-key / multi-value variants (L2) | Open |
| RV2.X3 | P2 cannot be attributed to loop memory: windowed baseline below the format floor, kernel 0 at the gap, gain barely moves | High | Write-up of P2, O4 | Parameter-matched memory-free adapter, LoRA and state-offset arms (L2); GDN state probes (L4) | Open |
| RV2.X4 | M6 is confounded: `titans_gdn` ties q and k (a previous-token address); `NeuralMemory` is a chunk-level Titans variant (η, α averaged per chunk; extra update clip) | Med | Claim M6 | Factorial ablation (L1); describe the Titans module as a variant | Open |
| RV2.X5 | M3's buffer holds nearly every prior fact; no uniform-replay, sleep-only or awake-only arm | Med | Claim M3, W3 | O1 | Open |
| RV2.X6 | The CPU windowed baseline still uses sliding windows: 3 layers × (16 − 1) = 45-token reach exceeds the 41-token gap (B2 was applied only to the pretrained runs) | Low | — | Block windows in `experiments/assoc_recall.py` | Open |
| RV2.X7 | `DentateGyrus.grow()` is never called, and the age-dependent gradient boost is normalised away by Adam | Low | O11 | Boost the learning rate or initial gain instead; exercise `grow()` in a test | Open |
| RV2.X8 | `ca3.py` calls `n_iter` "recurrent settling", but ξ = h + retrieved re-anchors to the cue and neither chains successors nor converges; sequential mode mixes successor and autoassociative values in one softmax | Low | L6 (chained reads) | Fix the docstring; implement chained reads explicitly | Open |
| RV2.K1 | The CA1 comparator is mistimed (compares the stored successor of t with x_t), and its novelty output is distilled token loss, not a comparator signal | Med | J2, J3 | Time-aligned comparator (L8) | Open |
| RV2.K2 | `HippocampalBlock` has no EC-II → CA3 read and puts CA2 in series between CA3 and CA1; anatomically CA2 is a parallel route (Kohara et al. 2014; Chevaleyre & Siegelbaum 2010) | Med | T16, L9, O12 | Add a direct read query; wire CA2 in parallel | Open |
| RV2.K3 | Subtractive feedback is not a clean high-pass: by Bode's integral, at K = 1 (τ 16, ρ 0.9) it amplifies content near a 46-token period about 2.5× | Med | S4 | Frequency-domain test (L7) | Open |
| RV2.S1 | The graded controller at defaults (λ 0.9, κ 0.5) reaches 0.875 of the cap after 3 tokens, with a time constant of about 10 tokens, and resets per sequence, so multi-turn escalation cannot occur (analytic) | High | A3, A4 | Turn-level integrator; λ ≈ 0.998 for token-level escalation | Open |
| RV2.S2 | NOTEBOOK's A6 formula h ← h − G·tanh(κE)‖h‖d̂ has the wrong sign; `steering.py` correctly adds +g‖h‖d̂ | Low | — | Bracketed correction in NOTEBOOK | Fixed (NOTEBOOK, A6 arms, bracketed note) |
| RV2.S3 | Novelty claims: the graded controller is largely anticipated by AdaSteer and PID steering (only the temporal integral is distinct); A6 is close to Herbster et al. 2026; selectivity must come from the gate (Joad et al. 2026) | Low | Write-up of A2, A6 | Cite; add AdaSteer and PID arms to A2 | Open |
| RV2.D1 | DG as MoE gate runs backwards for coarse load-balanced routers (they cluster similar inputs); under GDN's linear readout DG sparsity cannot raise the d_k bound | Med | Framing of the DG track; O9, W4 | Measure router-code separation (L11); test above d_k (O9) | Open |
| RV2.F1 | P11 and P13 are the missing-key problem of Samsonovich & Ascoli (2005): successor chains work, but no read cue or goal field selects the start | High | W1b follow-ups | L6 | Open |

## 4. Experiment backlog

Every table has the same columns. "Script" names the entry point, or says "not built". The round-1 mutation directions of `round0_plan.md` §4 map to: 1 → W1, 2 → W8, 3 → W9, 4 → W5, 5 (address-shift keys) → realised by W1's successor binding, 6 → W10.

### 4.1 Round 0 diagnostics ([`plans/round0_plan.md`](plans/round0_plan.md))

g0–g7: 1,000 steps on Qwen3, every layer windowed; pass means eval loss ≤ 1.82 and token accuracy ≥ 0.315 (floor 1.842 / 0.28); output `results/evolve/round0/<name>.json`.

| ID | Experiment | Hypothesis → metric | Script | Depends on / blocked by | Priority | Status |
|---|---|---|---|---|---|---|
| g0 | Control with the matched schedule | Stays at the floor → loss, token accuracy | `experiments/pretrained_loop.py` | Useful only beside a flat arm | Low | Ready; candidate to drop (its arms are superseded) |
| g1 | `flat` kernel | D2 against D3: a pass means the kernel was the barrier → loss, token accuracy | `experiments/pretrained_loop.py --kernel flat` | — | — | Superseded by P4 (the horizon kernel left the floor, so the kernel binds). Fails E2 (DEC1) |
| g2 | `exp` kernel, ρ 0.98, τ 16 (w(500) ≈ 0.54) | A slow decay is enough → loss, token accuracy | `experiments/pretrained_loop.py --kernel exp --rho 0.98` | — | Low | Ready. Not covered: the horizon runs (P4, P9) used ρ 0.95, τ 32, a close variant. Fails E2 |
| g3 | `flat --no-time` | The loop is the route; the clock is not needed → loss | `experiments/pretrained_loop.py` | — | — | Superseded by W1e (the same question on W1) |
| g4 | `flat --no-contrast` | The value arrives through the content-addressed read, not the uniform prior → loss | `experiments/pretrained_loop.py` | — | — | Superseded by P8 and P10 (the prior path is query-independent; W1 has no prior). Fails E2 |
| g5 | `flat --pairs 2` | Forces key addressing → loss with 2 pairs | `experiments/pretrained_loop.py` | — | Low | Ready; never run (P8 ran 2 pairs on the horizon kernel: g5h). Candidate to drop: fails E2 |
| g6 | `flat` at tap 6 | D3a/b: the readout depends on depth → loss | `experiments/pretrained_loop.py --layer 6` | — | — | Superseded by P10 (W1 decodes the value at tap 14). Depth is carried by W1b.c and W5. Fails E2 |
| g7 | `flat` at tap 22 | As g6 | `experiments/pretrained_loop.py --layer 22` | — | — | Superseded, as g6 |
| g2b | Rerun `pretrained_qwen35_needle_allwin_horizon` with more steps | It was undertrained → exact | `experiments/evolve/run.py` | — | — | Done (P9) |
| g5h | Horizon kernel (exp, ρ 0.95, τ 32) with 2 pairs. Was "g5" in this register; not the round-0 flat arm | One-pair recall uses the query → digit-1 loss | `experiments/evolve/run.py --pairs 2` | — | — | Done (P8) |
| C1 | Per-position loss and accuracy; save the side-channel `state_dict` | Which digits arrive → per-digit CE (rule E9) | `experiments/evolve/run.py` | — | — | Done (`per_pos` in each result, `.pt` beside each JSON; P7). Same as E9 |
| C1a | Per-digit loss and accuracy, the first half of C1 | Which digit the loop recovers (P4a) | `experiments/evolve/run.py` | — | — | Done (P7) |
| C2 | Linear probe for the value digits on frozen Qwen3 states at layers 6, 14 and 22 | D3a: the tap encodes the value → probe accuracy | not built (GPU script) | — | — | Superseded by P10 (W1 reads the value exactly at tap 14) |
| C3 | Loop instrumentation: M's share on block 0 and cos(Δ, wv(x_needle) − prior) at the answer rows | Where the O(T) loop loses the needle | not built | — | High | Planned |
| C4 | Eval-time ablations on a trained Qwen3.5 attention-only checkpoint: zero the injection at positions < 128 or ≥ 400; batch-mean and swapped injections | D4: local write carried by GDN, or readout at the query → exact, loss | not built | C1 (done); a Qwen3.5 checkpoint, which needs a rerun of `pretrained_qwen35_needle` through `experiments/evolve/run.py` (no `.pt` exists) | High | Planned. O5 is a duplicate |
| C5 | `check_window` on every windowed layer, plus an end-to-end side-off perturbation test | The base does not leak by itself (rule E7) → bitwise-equal logits | not built (`pretrained_loop.py` checks only the first windowed layer) | — | High | Planned. Same as E7 |
| C6 | Held-out evaluation: `--eval-seq-len` (gaps 373, 757, 1013), random needle offset, natural-text filler at eval only | Generalisation (rule E6) → min over conditions of ln 10 − digit CE | not built | — | High | Planned. Blocks W1's E6 score. Same as the E6 harness |
| — | taskeval-H2 phase test at seq-len 508 | The osc phase hole | cannot run (block mode needs seq-len divisible by the window) | — | — | Dropped (`round0_plan.md` §2) |

### 4.2 Pretrained candidates and scaling (W)

| ID | Experiment | Hypothesis → metric | Script | Depends on / blocked by | Priority | Status |
|---|---|---|---|---|---|---|
| W1 | GDN side memory with successor-bound keys, plus the clock, on Qwen3-1.7B, all layers windowed | M5 transfers to a frozen 2B model with a fixed-size state, and the address shift lifts recall from one digit (P4a) to all four → needle digit CE and exact; stability; recency | `experiments/evolve/w1_gdn.py` via `experiments/evolve/run.py` | — | High | Done on the needle (P10, P11). Stability and recency: W1c. E6: C6. LM: W1d |
| W1b | Context-bearing addresses so the queried name selects the *first* digit; 2 and 3 pairs | P11's coin flip comes from addresses that encode mainly the current token (the space after "is" is the same in both facts) → digit-1 accuracy with 2–3 pairs; exact | see W1b.a–c | — | High | Done (P13–P16): start selection not achieved; see round 3. RV2.F1: read-side remedies remain untried, a dense read cue from the query span (Treves–Rolls) and a goal field (L6) |
| W1b.a | GDN short-conv keys (`addressing="gdn"`) | As W1b | `experiments/evolve/w1_gdn.py --w1-addressing gdn` (6 heads for the budget) | — | High | Done (P13, P14): coin flip on digit 1 and no chain |
| W1b.b | Addresses from a learned causal conv over the last K positions' hidden states (the n-gram idea; rule E4 forbids token ids) | As W1b | `experiments/evolve/w1_gdn.py --w1-ctx K` | — | High | Done (P13, P16): coin flip with identity init; even-average init blurs the chain |
| W1b.c | A tap at a later block, where the residual at "is" carries more of the name | As W1b | `experiments/evolve/run.py --layer` | — | High | Done (P15): fastest, slight selection with 3 pairs. Overlaps W5 |
| W1g | **Start store** (option 2 of the round-3 checkpoint): a second successor store G with its own dentate-gyrus key projection (2× expansion, 5% k-WTA), beside W1's chain store S; 2 and 3 pairs at taps 14 and 21, a no-goal ablation and a 1-pair check | P17: the name is decodable at the cue, so a key projection not shaped by the chain, with conjunctive (DG) codes, selects the start → digit-1 loss against ln(pairs); exact | `experiments/evolve/w1g_goal.py` | RV2.F1 (online version of L6's goal field, no hindsight writes) | High | Done (P18, P19): solves 2 pairs at tap 21; not 3 pairs, not tap 14 |
| W1h | **CA3 settling** (option 3): W1g's cue settled in an autoassociative store of earlier positions, K = 3 iterations, k-WTA, recurrent gain 4, divisive normalisation; 2 and 3 pairs at taps 14 and 21 | Recurrent pattern completion turns a cue biased toward the queried start into a clean choice → digit-1 loss and accuracy against W1g | `experiments/evolve/w1g_goal.py --w1g-settle 3` | W1g; B16 | High | Done (P20): hurts as built |
| W1c | W1 adapter for the CPU stability and recency suites | W1 passes the two must-pass fitness criteria → closed-loop echo, zero-shot older/newer named | not built (`experiments/temporal_loop.py` cannot host W1) | — | High | Planned |
| W1d | W1 on WikiText-103 | No LM regression (rule E8) → perplexity against the default-kernel loop (12.01, P5) and the horizon kernel (11.53, P12) | `experiments/evolve/run.py --task lm --candidate evolve.w1_gdn:W1Side` | — | Med | Ready. `round0_plan.md`'s E8 threshold (11.9) is the superseded bf16 figure |
| W1e | W1 without the clock (`--no-time`), the live form of g3 | The memory alone is the route → needle exact, digit CE | `experiments/evolve/run.py --candidate evolve.w1_gdn:W1Side --no-time` | — | Med | Ready |
| W1f | Two-pair runs on Qwen3.5: P9's horizon kernel and W1 | P9's exact recall uses the query (P8's caveat) → digit-1 accuracy, exact | `experiments/evolve/run.py --pairs 2` | — | High | Ready |
| W2 | The clock on a GDN hybrid (Qwen3.5) with elapsed-time questions in long documents | T1 transfers → "when" accuracy | not built (needs a new `--task`) | — | Med | Planned |
| W3 | Replay consolidation into LoRA adapters on the MLPs of a frozen model, against uniform replay and no replay | M3 holds, and novelty priority beats uniform → backward transfer, capability | not built | Rule E3's budget must cover the adapters; needs a continual-learning fitness term. Its uniform-replay control: see O1 | Med | Planned |
| W4 | `--dg-keys` (and `--read dg`) with the horizon kernel on Qwen | T13 transfers; less interference → needle with 2+ pairs; recency probe | `experiments/pretrained_loop.py --dg-keys --loop-arg read=dg` | — | Med | Ready. Not covered by W1's DG keys (a different module); T13's recency transfer is untested |
| W5 | Read at about L/3, inject at L/2 (round-0 direction 4). Was D-a | The readout depends on depth → needle digit CE | not built (tap = inject today) | `round0_plan.md` made it conditional on g6/g7 or C2, now superseded. Overlaps W1b.c | Med | Planned |
| W6 | Depth loop: re-run blocks L+1 … N with a CA1-novelty exit. Was D-b | Big-loop recurrence helps multi-step recall → accuracy against extra compute | not built | A task: W6.a. J5 is its exit arm | Low | Planned |
| W6.a | Multi-hop recall task for the depth loop | — | not built | — | Low | Superseded by L6 (missing-key k-hop task) |
| W7 | Multi-layer injection. Was D-c | — | not built | — | Low | Planned |
| W8 | Fixed-state multi-timescale φ-feature loop: S ← ρS + φ(k)vᵀ, per-head learnable decay, one head initialised at ρ ≥ 0.995 (round-0 direction 2) | Keeps the loop within rule E2 → needle digit CE; stability | not built | High value only if g2 passes. Overlaps S8 | Med | Planned |
| W9 | Write selectivity: gate writes on novelty or salience, and replace the unweighted prior with a decayed running mean (round-0 direction 3) | Removes the uniform shift that washes the needle out (D3c) → needle digit CE | not built | — | Low | Planned |
| W10 | Gap curriculum: short gaps first, held-out E6 gaps fixed (round-0 direction 6) | → held-out digit CE | not built (task-code change) | C6 | Low | Planned |
| W11 | Joint light fine-tuning of the base and the side channels (training-plan step 2, `docs/design/temporal-loop.md` §4) | → needle, LM perplexity | not built | — | Low | Planned |

### 4.3 Feedback stability (S) (NOTEBOOK, stability programme)

| ID | Experiment | Hypothesis → metric | Script | Depends on / blocked by | Priority | Status |
|---|---|---|---|---|---|---|
| S1 | Saturation curve on Qwen: fixed gain g ∈ {0.05, 0.1, 0.2, 0.5, 1, 2} × ‖h‖, open and closed loops, plus a matched-norm random-direction control | Additive injection saturates the frozen model past a knee → downstream residual norms, attention and logit entropy, WikiText perplexity, needle accuracy | not built | Must include the gist arm before R2 runs at high gain (RV1.C1; RV1.C6 adds distinct-n) | High | Planned |
| S2 | Closed loop on Qwen up to 4k tokens, for each regulator and for none | → echo against sequence length | not built | — | High | Planned |
| S3 | Regulator ablation with ≥ 3 seeds on the CPU stability suite: clamp, divisive, subtractive, projection and saturation, alone and combined | Current stability numbers (T3–T15) hold across seeds → echo, what, older/newer named | `experiments/temporal_loop.py` | Overlaps O2 | High | Ready |
| S4 | Compensated subtractive feedback: a lead–lag filter, or a shorter or distributed delay | Raising the critical gain above 1.68 (T7) suppresses more echo and breaks closed-loop primacy → $K_\text{crit}$, echo, older named | `experiments/loop_systems.py` for the analysis; the kernel is not built | — | Med | Planned |
| S5 | Homeostatic gain: adapt log g slowly towards a target echo ratio r* | → echo tracking, stability margin | not built | — | Med | Planned |
| S6 | Novelty-gated gain, the acetylcholine switch (also the blueprint's gate in `docs/design/temporal-loop.md` §4) | Suppressing injection while input is novel → echo, recall | not built | — | Med | Planned |
| S7 | Norm-preserving injection against additive injection, at matched effect sizes | It moves the S1 knee → S1 metrics | not built | S1 | Med | Planned |
| S8 | Multi-band damping: several ρ bands (spans from about 45 to 1,000 tokens) | One design gets both recency and long reach without losing stability → recall by gap, older named, echo | not built | Part of W8; same as the README roadmap item "multi-band horizons" | Med | Planned |
| S9 | S1 for steering directions: refusal, thinking and random directions at g ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.75} | The knee depends on the direction (N4) → degeneration, perplexity, downstream norms | not built (sweep) | — | High | Planned |

### 4.4 Alignment: CA2 as a norm-steering controller (A) (NOTEBOOK, proposal)

Data rule: commit only judge scores, refusal labels and aggregates. Raw completions to harmful prompts stay off the repository.

| ID | Experiment | Hypothesis → metric | Script | Depends on / blocked by | Priority | Status |
|---|---|---|---|---|---|---|
| A1 | Harm probe at the tap, with a layer sweep | Harmfulness is linearly decodable (A6.H1) → AUROC, ECE | `experiments/alignment/run.py --layer-sweep` | Caveat RV1.A2 (ECE on the wrong positions) | High | Ready |
| A2 | Escalating controller against static, conditional (CAST) and no steering, on harmful prompts, XSTest and perplexity (HarmBench and MMLU later) | Graded steering is more selective at lower capability cost → attack success, over-refusal, perplexity | `experiments/alignment/run.py --arms` | Blocked by RV1.A1, RV1.A3, RV1.A6. Cap from S1. Its arms are a subset of A6's; J1 sweeps its decision source | Med | Blocked. RV2.S3: add AdaSteer and PID-steering arms at matched mean injected norm, a StrongREJECT judge and a token-level probe (was L13) |
| A3 | Escalation and recovery under multi-turn pressure | E_t rises on persistent attempts and falls on benign turns (A6.H4) → attack success over turns, recovery | not built (multi-turn) | — | Med | Planned. RV2.S1: needs a turn-level integrator; the token-level one resets per sequence (was L14) |
| A4 | Controller stability: ringing between refusing and complying | The delayed integral controller stays below the S1 knee and the T7 limit → oscillation of the gain | not built | S1, T7 | Med | Planned |
| A5 | Robustness to adaptive attacks on the probe | → attack success under probe-aware attacks | not built | A1 | Low | Planned |
| A6 | **Re-align an unaligned model with runtime hooks** (Daniel). Abliterated Qwen3-1.7B and the never-aligned base, against the aligned reference. Arms: none, static refusal direction, conditional, graded controller, controller without the aligned sibling | A6.H1–A6.H4 in the notebook → attack success, over-refusal, capability, degeneration, controller dynamics | `experiments/alignment/run.py` (`--direction-from`, arm `graded_self`) | Blocked by RV1.A1, RV1.A3, RV1.A6 (and RV1.A7 with `--direction-from`). Depends on A1 and S1. Spot checks N2, N3: the sibling direction needs g ≥ 0.3, so sweep past 0.3; `graded_self` needs a direction estimate other than the self difference of means | High (standalone alignment track) | Blocked |
| A7 | Scale and hybrid check of A6 on `Huihui-Qwen3.5-9B-abliterated` | → as A6 | not built (it loads as `AutoModelForMultimodalLM`, so `pretrained_loop.py` model loading and `find_blocks` need changes) | A6; RV1.A7 | Med | Planned |
| A8 | **Gain law** (Daniel): additive-constant, tanh $G\tanh(\kappa E)$, squared $G\tanh(\kappa E^2)$ and exponential $\min(G, g_0(e^{\kappa E}-1))$ schedules at matched maximum gain. Gain on the existing refusal component, $g(h\cdot\hat d)\hat d$, is excluded: $h\cdot\hat d\approx 0$ after abliteration | Superlinear schedules stay near zero on weak evidence and escalate fast on strong evidence → selectivity (harmful − borderline refusal), degeneration, perplexity | not built (needs new modes in `steering.py`) | Run with A6 | High | Planned |

### 4.5 Reasoning: CA2 steering of chain-of-thought (R) (NOTEBOOK, proposal)

The gist arm is positive feedback. S1 must include it before R2 runs at high gain. A reasoning runner is not built yet.

| ID | Experiment | Hypothesis → metric | Script | Depends on / blocked by | Priority | Status |
|---|---|---|---|---|---|---|
| R1 | Thinking-minus-non-thinking direction on Qwen3 hybrid checkpoints, injected in no-think mode | → accuracy against tokens | `experiments/alignment/steering.py` (fixed direction); runner not built | RV1.C7 (take the difference at matched content positions) | Med | Planned. Spot-checked (N4, N5): collapse at g = 0.3. Needs harder problems (MATH-500, GSM8K hard) and n ≥ 100 |
| R2 | Gist (momentum) steering during thinking on GSM8K, MATH-500 and AIME, against full thinking, `/no_think`, budget truncation, SEAL and the concise vector | Gist steering cuts reasoning tokens at equal accuracy → accuracy against reasoning tokens | `experiments/alignment/steering.py` (gist mode); runner not built | Blocked by RV1.C1, RV1.C2. S1 (gist arm) | High | Blocked. RV2: add a self-loop-rate metric (Yu et al. 2026) to RV1.C2's controls (was L15) |
| R3 | Regulated gist: gain from agreement between the gist and the current token, released on backtracking. Baseline: PID steering (arXiv 2510.04309) and a decode-time PID ablation, not the withdrawn STU-PID | Negative feedback keeps the gain while on track → accuracy against tokens | not built (no gain law) | Blocked by RV1.C3, RV1.C4. Run with J4 | Med | Blocked |
| R4 | Over-commitment checks: flip analysis, "wait" rate, GPQA-Diamond, wall-clock time and tokens-to-correct | The measurement set of R2 → flips, backtracking rate | as R2 | R2; RV1.C2 | Med | Blocked (with R2) |
| R5 | Memory-retrieved gist vectors from successful traces (hippocampal version of STIR and DSEM) | → accuracy against tokens | not built | R2 | Low | Planned |
| R6 | **Repetition-regulated gain** (Daniel): negative feedback from a repetition signal, $g_t = g_0 e^{-\gamma R_t}$ with $R_t = \lambda_r R_{t-1} + r_t$. Candidate $r_t$: n-gram repeat rate, $\cos(h_t, h_{t-k})$, logit-entropy drop. Also decode-only injection (no prefill) | Regulation delays collapse at fixed accuracy → collapse onset, accuracy against tokens, against a fixed gain (N4) | not built (needs a regulator in `steering.py`) | Related to S5, R3 and J4; uses RV1.C6's repetition metric | High | Planned |

### 4.6 CA1 as a calibrated decision model (J) (NOTEBOOK, proposal: JEV, LAYA)

| ID | Experiment | Hypothesis → metric | Script | Depends on / blocked by | Priority | Status |
|---|---|---|---|---|---|---|
| J1 | Decision source for the alignment controller: linear probe, calibrated head, LAYA-style external encoder, guard model | To be reworded per RV1.D1: compare sources at a matched operating point → borderline refusal at a fixed harmful-refusal rate; AUROC; ECE in and out of distribution, on generated tokens | `experiments/alignment/run.py` (linear probe only; no external score callback) | Blocked by RV1.D1, RV1.D4, RV1.A2/RV1.D2. A decision-source sweep inside A2 | High | Blocked |
| J2 | LAYA-style CA1 comparator for the loop: joint encoder over the stream and the CA3 retrieval, outputting injection strength, write/no-write and replay priority, trained on the counterfactual benefit of injecting | → what-when, recency probe, stability suite, then the Qwen needle | not built | — | Med | Planned |
| J3 | Revisit M2 with a calibrated CA1 novelty signal for Titans | Premise corrected (RV1.D3): the harm came from stacking novelty-lr with DG writes, not from CA1 novelty-lr alone → recall; a calibration measure for the novelty target (a binarised event or a reliability curve of z) | `experiments/assoc_recall.py` (no calibrated-CA1 option) | Blocked by RV1.D3. J3.a first | Low | Blocked |
| J3.a | Warm up `novelty_lr`, or stop gradients through novelty, as its own arm | M2's harm is early noise from an untrained novelty signal (RESULTS §1) → recall of the stacked model | not built (no warm-up option in `experiments/assoc_recall.py`) | — | Low | Planned |
| J4 | Calibrated "on track" decision gating the gist gain and early stopping of thinking | → accuracy against tokens, against PID control | not built | Run with R3 (Blocked) | Med | Planned |
| J5 | Calibrated exit for a depth loop, against Ouro's gate | → accuracy against compute | not built | W6 | Low | Planned |
| J6 | Calibrated CA1 with a proper scoring rule and reliability diagrams; JEPA-style target with SIGReg | → ECE, reliability, downstream recall | not built | Overlaps J2 and J3 | Low | Planned |

### 4.7 Other open questions (O)

| ID | Experiment | Hypothesis → metric | Script | Depends on / blocked by | Priority | Status |
|---|---|---|---|---|---|---|
| O1 | Replay controls: uniform replay; sleep only against awake replay | M3's effect needs novelty priority and sleep → backward transfer | `experiments/continual.py` (new conditions needed) | Also W3's uniform-replay control | Med | Planned |
| O2 | Multi-seed replication of T11–T15 | → older/newer named, echo | `experiments/temporal_loop.py --seed` | Overlaps S3 | High | Ready |
| O3 | Fixed-size state for the pretrained track | — | — | — | — | Moved: see DEC1 |
| O4 | Separate memory from domain adaptation in P5: horizon kernel, and an arm with the loop's memory ablated | → perplexity | `experiments/evolve/run.py --task lm`; memory ablation not built | — | Med | Horizon arm Done (P12); memory-ablated arm Planned |
| O5 | Qwen3.5 eval-time ablations: zero the injection early or late to locate the GDN route | — | — | — | — | Duplicate: see C4 |
| O6 | Multi-seed replication of the P results (rule E6 requires ≥ 2 seeds; O2 covers only T11–T15) | → the P metrics | `experiments/evolve/run.py --seed` | — | Med | Ready |
| O7 | Replicate T16 (Treves–Rolls noisy cue) | → recall under cue noise | `experiments/temporal_loop.py` | — | Low | Superseded by L9 (T16 is not a test of Treves–Rolls; RV2.K2) |
| O8 | Compensate the delayed CA1 feedback through theta timing or GABA_B | → $K_\text{crit}$, echo | not built | S4 covers only lead–lag or delay shaping | Low | Planned |
| O9 | Pattern separation on similar keys, plus a DG-sparsity sweep from 1% to 25% | DG addresses beat dense ones once keys overlap (would resolve M7) → recall | not built (needs a task with overlapping keys) | — | Med | Planned. RV2: extend to loads above d_k and FlyHash keys (was L3); under linear readout sparsity is not expected to move the d_k bound |
| O10 | Deep against linear memory on nonlinear associations; the cost of chunk-start gradients | Does M6 hold beyond linear associations? → recall | not built | — | Low | Planned |
| O11 | Neurogenesis and consolidation scheduling: fixed, surprise-triggered, adaptive | → backward transfer, recall | not built | — | Low | Planned |
| O12 | CA2 identity steering: per-speaker vectors on multi-speaker recall, plus a post-hoc pretrained version. First use of `CA2Modulator` | → multi-speaker recall | not built | — | Low | Planned |
| O13 | HippocampalBlock (Strategy I) inside a pretrained LLM (W1 covers only the GatedDeltaMemory half) | → needle, LM perplexity | not built | — | Low | Planned |

### 4.8 Literature review (L) (RV2)

Proposed by RV2; details and literature are in the track reports. Items that duplicate an existing row point to it.

| ID | Experiment | Hypothesis → metric | Script | Depends on / blocked by | Priority | Status |
|---|---|---|---|---|---|---|
| L1 | Zoology-style MQAR re-benchmark: binding-free baseline and first-query accuracy; LR sweep per width; ≥ 5 seeds; random short-conv init; Titans factorial (tied/untied q-k, per-token vs chunk updates) | Properly trained attention, standard GDN and deep Titans all bind; tied/DG addresses help only in sample efficiency or near load = d_k → MQAR accuracy, first-query accuracy, gap to the ceiling | not built (replace `experiments/tasks.py::AssociativeRecall`) | B12; RV2.X1, RV2.X4 | High | Planned |
| L2 | Qwen controls: parameter-matched memory-free adapter, LoRA and state-offset tuning, with per-digit loss; RULER S-NIAH, multi-key and multi-value at 2k–8k | A memory-free adapter restores the needle on Qwen3.5 with GDN intact (P2 is adaptation); only memory helps when every layer is windowed → exact, per-digit CE against the format floor | `experiments/evolve/run.py` (new `--candidate` arms; RULER task not built) | RV2.X2, RV2.X3; overlaps O4 (LM side) | High | Planned |
| L3 | Capacity above d_k with controlled key overlap | — | — | — | — | Duplicate: see O9 (add loads above d_k and FlyHash keys) |
| L4 | Linear probes for the needle in each GDN layer's state; causal tracing of the route | With attention windowed, the needle stays linearly decodable in some GDN states (P1 is a read failure) → probe digit accuracy by layer; patching recovery | not built | Extends C2, C4 | High | Planned |
| L5 | State-space injection: write the side memory into a downstream GDN state in that layer's key space; W1 head-to-head with δ-mem at equal state size | Injecting into the state gives recall with less perplexity damage than residual injection → needle exact at 500 / 2k / 8k; perplexity change | not built | DEC1 | High | Planned |
| L6 | **Missing-key k-hop recall beyond the window**: chained successor reads, then a hindsight goal field (a second delta matrix written at replay events, Samsonovich & Ascoli's CA3 → CA1 W) to select the start and route the path | Single reads solve only k = 1; chained reads give ≈ p^k; a goal field selects the queried start where P13's addresses cannot → exact vs k; digit-1 accuracy with 2–3 pairs; path length / shortest; invalid-edge rate, against `reviews/2026-11-02_literature/scripts/sa2005_replicate.py` | not built | RV2.F1, RV2.X8; supersedes W6.a as the multi-hop task | High | Planned |
| L7 | Frequency-domain (waterbed) test of subtractive feedback | Measured \|S(ω)\| matches the delayed-loop prediction; content near the amplified band is over-recalled → \|S(ω)\|, echo and what-accuracy by event spacing | `experiments/loop_systems.py` (new sweep) | RV2.K3 | Med | Planned |
| L8 | Time-aligned CA1 comparator: CA3's prediction from t − 1 against EC at t | Its novelty tracks held-out surprisal better than the distilled head → Spearman with surprisal; recall under novelty-gated writes | not built | RV2.K1 | Med | Planned |
| L9 | Trisynaptic vs monosynaptic double dissociation (one-shot binding vs statistics, after Nakashiba 2008), and a Treves–Rolls test with a CA3 completion stage | CA3 path needed for one-shot binding, temporoammonic path suffices for statistics; dense direct read beats DG read under cue noise only with completion → crossover interaction; recall vs cue noise | not built | RV2.K2; supersedes O7 | Med | Planned |
| L10 | Skip-track baselines at matched parameters: value residuals, DenseFormer, hyper-connections, Feedback Transformer | A temporoammonic skip does not beat these → perplexity, MQAR, RULER 4k | not built | Track 1 has no experiment yet | Med | Planned |
| L11 | Do Qwen MoE router codes separate or cluster near-duplicates? Router codes vs FlyHash as side-memory keys | Router codes cluster (load balancing) and lose to FlyHash for similar items → separation curve; recall with template-sharing pairs | not built | RV2.D1 | Med | Planned |
| L12 | Expert neurogenesis for continual fine-tuning: newborn experts with decaying router bias, updates confined to them, against ESFT, sparse memory finetuning and LoRA | Matches ESFT and beats LoRA on forgetting → new-task accuracy, forgetting, router drift | not built | Replay arms: see O1 | Med | Planned |
| L13 | Matched-norm controller comparison (AdaSteer and PID arms, StrongREJECT judge, token-level probe) | — | — | — | — | Duplicate: see A2 (arms added) |
| L14 | Turn-level harm integrator under multi-turn attacks | — | — | — | — | Duplicate: see A3 (RV2.S1) |
| L15 | Gist controls (matched-norm random, mean, anti-gist; self-loop, "wait" and flip rates) | — | — | — | — | Duplicate: see R2, RV1.C2 |
| L16 | Alternating-candidate loop: cycle top-k CA3 retrievals across loop periods with a CA1 goal selector (Kay et al. 2020) | Alternation plus goal selection beats gist steering per reasoning token → accuracy vs tokens; invalid-step rate | not built | L6; J4 | Low | Planned |
