# Register of results, conclusions and experiments

This file holds the current state of the project: every result with its status and evidence, every correction, and every planned experiment. [`NOTEBOOK.md`](NOTEBOOK.md) is the dated record of how each entry got here, and [`docs/genesis.md`](../../docs/genesis.md) is the history of the ideas. Update this file in place when a result lands or an experiment changes status, and cite the result file.

**Status:** **Supported** · **Refuted** · **Mixed** (seeds or settings disagree) · **Open** (untested or confounded) · **Fixed** (a bug, now corrected).
**Seeds:** results are from small CPU models (0.1–0.4M parameters) or frozen 2B models, mostly single seeds. Treat a difference of a few points as noise.

## 1. Results and conclusions

### Memory systems (RESULTS.md §1–3)

| ID | Claim | Evidence | Seeds | Status |
|---|---|---|---|---|
| M1 | Hippocampal memories recall bindings beyond a 16-token window; windowed attention alone stays at chance | Strategy I 0.41, Titans 0.48, full attention 0.49, windowed 0.06 (`results/assoc_recall.json`) | 1 | Supported |
| M2 | Stacked DG writes and CA1 novelty-lr on Titans help | Neutral alone, 0.17 together | 1 | Refuted (warm-up untested) |
| M3 | Novelty-prioritised replay with sleep prevents forgetting | Backward transfer −0.73 → 0.00 (`results/continual.json`) | 1 | Supported; no uniform-replay or sleep-only control |
| M4 | Standard Gated DeltaNet loses recall once stored pairs exceed its key size | 0.43 → 0.13 at 8 → 16 pairs (`results/gdn_capacity.json`) | 1 | Supported |
| M5 | Successor-bound addressing makes the GDN state exact RAM | 1.00 up to 32 pairs; counterfactual 100% new / 0% old | 1 | Supported |
| M6 | An exact delta-rule memory beats Titans' deep memory | 1.00 against 0.48 (`results/assoc_recall_gdn.json`) | 1 | Supported |
| M7 | DG addresses beat dense addresses | Same final recall; DG faster early (0.905 vs 0.680 at step 500) | 1 | Open (task does not separate them) |

### Temporal loop and clock (RESULTS.md §4)

| ID | Claim | Evidence | Seeds | Status |
|---|---|---|---|---|
| T1 | The loop carries content and the clock carries elapsed time | What 1.00 / when 0.85 (loop); when 0.88 (clock); both 1.00 / 0.92 (`temporal_loop_all.json`, `_long.json`) | 1 per config | Supported |
| T2 | "What" is learned in a sudden late jump; 1,500 steps is too short for clamped-sinusoid loops | Full design crosses 0.5 at step 1,000 of 3,000 and never within 1,500 | 2 | Supported |
| T3 | The full design keeps closed-loop echo bounded and stationary | 0.29–0.35× the input, peak ≤ 0.53 (`_long.json`, `_long_seed1.json`) | 2 | Supported (the bound comes from the clamp) |
| T4 | Undamped closed loops run away; damping alone overshoots early; the clamped sinusoid without control resonates | 4 → 14× (peak 63×); 1.0× early (peak 10.6×); about 3× (`_probe.json`, `_all.json`) | 1 | Supported |
| T5 | Damping trades long recall for recency | Recall at gaps of 300–500: 0.40 undamped, 0.00 damped, 0.04–0.10 full design | 1–2 | Supported |
| T6 | Subtractive inverted-copy feedback obeys the op-amp law on persistent echo | $y_\text{sub}/y_\text{none}$ = 0.787, 0.647, 0.475, 0.373 against $1/(1+K)$ = 0.788, 0.650, 0.482, 0.383 (`loop_systems.json`) | 3 (untrained) | Supported |
| T7 | Delayed subtractive feedback rings above $K = 1.68$ | Stable at 1.08, growing at 2.15 and above; period about 44 tokens | 3 (untrained) | Supported |
| T8 | Divisive feedback only compresses | Undamped kernel, $A$ = 32: 0.69 against 2.39 | 3 (untrained) | Supported |
| T9 | Without the clamp, only subtractive feedback learns content | What 0.99 / 1.00 against 0.13 (none) and 0.15 (divisive) (`_feedback.json`, `_open.json`) | 2 | Supported |
| T10 | Subtractive feedback bounds the echo | 1.7–2.6× the input | 2 | Refuted |
| T11 | Closed loops favour the first event, by self-rehearsal | Older named 0.61 (undamped), 0.84 (damped), 0.63 (full design without contrast) | 1 each | Supported (mechanism is a hypothesis) |
| T12 | The contrast against the prior reduces primacy | Older 0.63 → 0.18 | 1 | Supported, weakly; it also halves learning speed (crosses at step 1,000 vs 500) |
| T13 | Sparse DG write keys favour the newer event | Newer 0.68, older 0.29–0.31 (`_ablations.json`) | 1 (two arms) | Supported, weakly |
| T14 | The open loop shows little primacy | Newer 0.45 / older 0.38 (`_open.json`) | 1 | Supported, weakly (different seed) |
| T15 | Subtractive feedback breaks primacy | Older 0.54 (seed 0), 0.38 (seed 1) | 2 | Mixed |
| T16 | A direct EC → CA3 read completes degraded cues better than a DG read (Treves–Rolls) | At σ = 2: 0.84 direct, 0.94 DG read | 1 | Not supported |
| T17 | A trained recency task ranks the loops | No model learned it (what 0.14–0.15) | 1 | Open (inconclusive) |
| T18 | The clock removed a curriculum, so memory + time failed "what" | Learned by step 500 with a longer schedule | 1 | Refuted |
| T19 | Training is deterministic | A rerun reproduced every seed-0 number exactly | — | Supported |

### Pretrained models (RESULTS.md §4.7, `results/pretrained_HANDOFF.md`)

| ID | Claim | Evidence | Seeds | Status |
|---|---|---|---|---|
| P1 | Qwen3.5's GDN layers alone do not recall a needle 500 tokens back | Exact 0.00 with attention windowed and GDN intact | 1 | Supported |
| P2 | The side channels restore exact recall on Qwen3.5, through the GDN layers above the tap | Exact 1.00 tapping block 12, 0.00 tapping block 22 | 1 | Supported (the route is a hypothesis) |
| P3 | With the default kernel the loop cannot reach the needle | Horizon τ/ln(1/ρ) ≈ 45 tokens; Qwen3 at the format floor, 1.842 | 1 | Supported (round-0 D1, D2) |
| P4 | With a horizon that covers the gap, the loop alone carries needle information | Digit loss 1.38 (Qwen3) and 1.24 (Qwen3.5), below the 1.842 floor; token accuracy 0.46–0.55; exact 0.00–0.02 | 1 | Supported (partial recall) |
| P4a | The loop recovers whole digits, not partial information about each one | Qwen3 1.381 / 0.461 equals one digit known and three at chance, $(3/5)\ln 10$ / 0.46; Qwen3.5 lies between one and two digits | 1 | Supported. Mechanism: key and value come from the same position, so nothing moves the lookup to the next digit. This motivates W1 |
| P5 | The side channels improve WikiText perplexity on windowed models | Qwen3, all windowed: 14.6 → 12.0 (full 9.0; 41% of the loss gap). Qwen3.5, attention windowed: 16.3 → 11.2 (full 9.8; 74%) | 1 each | Open (confounded with domain adaptation, since only that condition trains) |
| P6 | The default kernel helps Qwen3.5 when every layer is windowed | Float32 rerun at the floor, 1.843 / 0.289. The earlier 1.69 was the bf16 leak | 1 | Refuted |

### Corrections and fixed bugs

| ID | What was wrong | Fix | Status |
|---|---|---|---|
| C1 | The "when" baseline was taken as 0.25 | The bins are unequal; the prior scores 0.333 | Fixed |
| C2 | Sliding windows were assumed to make memory the only route | They leak $L(W-1)$ tokens through depth; block windows are now the default | Fixed |
| C3 | bf16 side channels: Adam updates rounded away | Side channels train in float32 | Fixed |
| C4 | bf16 index arithmetic: positions merged, an anti-causal leak | Float32 indices, plus a regression test (round-0 D5) | Fixed |
| C5 | The design note claimed an $O(1)$-state form for the loop | Only true for factorised relevance; the loop is $O(T)$ | Fixed |
| C6 | Rollout length off by one | Gaps stop at 497 | Fixed |
| C7 | CI depended on numpy | Torch companion-matrix roots | Fixed |
| C8 | Interim commits rewrote the logs of runs still in progress | Logs rebuilt from JSON; in-progress logs are no longer committed | Fixed |

## 2. Experiment backlog

**Priority:** **H** next round · **M** soon · **L** later. **Status:** Planned · Running · Done (with result ID) · Dropped.

### Round 0 diagnostics (`round0_plan.md`)

| ID | Experiment | Discriminates | Priority | Status |
|---|---|---|---|---|
| g0–g7 | Control and the `flat` kernel arms (no time, no contrast, 2 pairs, taps 6 and 22) | The kernel (D2) against a downstream block (D3) | H | g2 covered by the horizon runs (P4); the rest Planned |
| C1–C6 | Per-position logging, value probes at layers 6/14/22, loop instrumentation, eval-time ablations, window checks, held-out gaps | D3, D4, E6, E7 | H | Planned |
| C1a | Per-digit loss and accuracy logging (the first piece of C1) | Which digit the loop recovers (P4a) | **H** (cheap) | Planned |
| g2b | Rerun `pretrained_qwen35_needle_allwin_horizon` with more steps | Undertraining: it left the floor only near step 2,700 of 3,000 | **H** (cheap) | Planned |

### Round 1 shortlist (scaling the small-model wins)

| ID | Experiment | Hypothesis | Metric | Priority | Status |
|---|---|---|---|---|---|
| W1 | GDN side memory with successor-bound keys, plus the clock, on Qwen3-1.7B, all layers windowed | M5 transfers to a frozen 2B model with a fixed-size state, and the address shift lifts recall from one digit (P4a) to all four | Needle digit CE and exact match; stability; recency probe | H | Planned |
| W2 | The clock on a GDN hybrid (Qwen3.5) with elapsed-time questions in long documents | T1 transfers | "When" accuracy | M | Planned |
| W3 | Replay consolidation into LoRA adapters on the MLPs of a frozen model, against uniform replay and no replay | M3 holds, and novelty priority beats uniform | Backward transfer, capability | M | Planned |
| W4 | `--dg-keys` (and `--read dg`) with the horizon kernel on Qwen | T13 transfers; less interference | Needle with 2+ pairs; recency probe | M | Planned |

### Design gaps (design review, 2 Nov)

| ID | Experiment | Priority | Status |
|---|---|---|---|
| D-a | Read at about L/3, inject at L/2 (round-0 direction 4) | M | Planned |
| D-b | Depth loop: re-run blocks L+1 … N with a CA1-novelty exit | L | Planned |
| D-c | Multi-layer injection | L | Planned |

### Feedback stability (NOTEBOOK, stability programme)

| ID | Experiment | Priority | Status |
|---|---|---|---|
| S1 | Saturation curve on Qwen: gain sweep, open and closed loops, with downstream norms, entropy, perplexity and needle accuracy, plus a random-direction control | **H** | Planned |
| S2 | Closed loop on Qwen up to 4k tokens, for each regulator | H | Planned |
| S3 | Regulator ablation with ≥ 3 seeds on the CPU stability suite | H | Planned |
| S4 | Compensated subtractive feedback to raise the critical gain | M | Planned |
| S5 | Homeostatic gain adaptation towards a target echo | M | Planned |
| S6 | Novelty-gated gain (the acetylcholine switch) | M | Planned |
| S7 | Norm-preserving injection against additive injection | M | Planned |
| S8 | Multi-band damping | M | Planned |

### Alignment: CA2 as a norm-steering controller (NOTEBOOK, proposal)

| ID | Experiment | Priority | Status |
|---|---|---|---|
| A1 | Harm probe at the tap: AUROC and calibration | M | Planned |
| A2 | Escalating controller against static, conditional (CAST) and no steering, on HarmBench, XSTest and capability benchmarks | M | Planned (after S1) |
| A3 | Escalation and recovery under multi-turn pressure | M | Planned |
| A4 | Controller stability: ringing between refusing and complying | M | Planned |
| A5 | Robustness to adaptive attacks on the probe | L | Planned |
| A6 | **Re-align an unaligned model with runtime hooks** (Daniel). Abliterated Qwen3-1.7B and the never-aligned base, against the aligned reference. Arms: none, static refusal direction, conditional, graded controller, controller without the aligned sibling. Hypotheses H1–H4 in the notebook | **H** (standalone alignment track) | Planned (after A1, S1) |
| A7 | Scale and hybrid check of A6 on `Huihui-Qwen3.5-9B-abliterated`. It loads as `AutoModelForMultimodalLM`, so `pretrained_loop.py` model loading and `find_blocks` need changes | M | Planned |

Alignment track data rule: commit only judge scores, refusal labels and aggregates. Raw completions to harmful prompts stay off the repository.

### Other open questions

| ID | Experiment | Priority | Status |
|---|---|---|---|
| O1 | Replay controls: uniform replay; sleep only against awake replay | M | Planned |
| O2 | Multi-seed replication of T11–T15 | H | Planned |
| O3 | Decide whether the pretrained track requires a fixed-size state (round-0 decision). Recommended yes by both sessions, keeping the O(T) loop with the horizon kernel as the reference arm | H | Awaiting Daniel |
| O4 | Separate memory from domain adaptation in P5: horizon kernel, and an arm with the loop's memory ablated | M | Planned |
| O5 | Qwen3.5 eval-time ablations: zero the injection early or late to locate the GDN route (round-0 C4) | M | Planned |
