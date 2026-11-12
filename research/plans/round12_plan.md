# Round 12, re-planned (10 Nov 2026): control of thinking, and memory on natively windowed models

## Context

Stage 0 of the training-free CA2 side channel is done. Under full attention nothing helped on any tracked metric. The gist direction hurt everywhere. The only sizeable effect, the branch rebalance on the windowed hybrid, is partly repair of a condition we created.

Daniel's directives for the re-plan:
- Incorporate the improvement suggestions, reorder by importance, cancel what cannot show a benefit.
- Windowing is acceptable when cast correctly: operation of models with recurrent attention modules, and tests on models **trained** with sliding windows.
- **Training on text is allowed** if it gives a scientifically reportable result.
- Task-level outcomes matter: accuracy, tokens, speed.

Sources: Stage 0 digest and the re-plan research, two designs and the critic's merged programme, in the session scratchpad (`stage0_digest.md`, `replan_research.md`, `replan_design_control.md`, `replan_design_content.md`, `replan_critique.md`). They move to `research/plans/` with the records.

## What the re-plan research changed

- **The hybrid's thinking loops are confounded with decoding.** Our runs sample at temperature 0.6 with no presence penalty. Qwen's card recommends temperature 1.0 with presence penalty 1.5 for thinking, and warns this model loops. The first GPU experiment tests that.
- **Small models trained with windows exist and load in our transformers.**
  - `google/recurrentgemma-2b(-it)`: recurrent layers plus local attention, window 2048. It is a natively windowed recurrent hybrid.
  - `google/gemma-4-E2B-it`: a thinking model with a 512 window on local layers.
  - Gemma 3 (512 or 1024 windows).
- **The closest published system, δ-mem, has no parameter-matched memory-free control and one seed.** A trained result with those controls is reportable.
- **HF `generate` accepts a logits processor at every step.** A presence penalty, a `</think>` bias, forced tokens and a loop rule need no custom decode loop.

## Cancelled

- Fixed-gain gist, every gist or anti-gist reasoning law, multi-site dose splitting, skip-ahead injection: refuted in Stage 0.
- GDN state as an exact store; GDN-familiarity loop detector: the state holds no items, and an n-gram rule sees loops directly.
- Lag-grid screens and confirmations under an artificial window on full-attention-trained models.
- gpt-oss-20b, Phi-4-mini-flash, Hymba, KronLSTM-SWA rows: each needs its own environment or does not fit.
- Bounding the global layers of Gemma or Granite as a headline: the same induced-condition repair.
- Speed engineering beyond what the experiments need.

## Ranked programme

Models: `Qwen/Qwen3.5-2B` (hybrid, thinks, loops), `Qwen/Qwen3-1.7B` (comparator, thinking terminates), `google/gemma-4-E2B-it` (thinker trained with windows), `google/recurrentgemma-2b(-it)` (natively windowed recurrent hybrid).

| Rank | ID | Experiment | GPU-h | Runs when |
|---|---|---|---|---|
| 1 | C0 | Offline stop-rule study on the unsteered traces: n-gram loop rule, answer consistency, confidence; probe cost charged in tokens | 0 | the `none` arms finish |
| 2 | C1 | Decoding grid on the hybrid and Qwen3: (T, presence penalty) ∈ {(0.6, 0), (1.0, 0), (1.0, 1.5)}; no-think; one real truncation; optional Gemma-4 row | 2–3 | presence-penalty processor built and tested |
| 3 | M1a | Trained loop against a parameter-matched memory-free adapter, replicating P5's setup exactly; foreign-state and zeroed-state ablations | 1.5 | adapter code built and tested |
| 4 | M2+M3a | RecurrentGemma at 8K: native, sink + window, InfLLM-style retrieval, random, foreign, and oracle read-back with ordinal or true position | 2 | licence accepted; read-back code built and tested |
| 5 | C3-core (dev) | Hard stop and `</think>` logit bias driven by the best C0 rules, against the truncation frontier; no-think, repetition penalty, suppression of "wait" tokens | 2 | C0 and C1 |
| 6 | B-a | Constant-memory thinking: hybrid with sink + window 256/512 on its attention layers, Qwen3 under the same windows, RecurrentGemma as trained | 1.5 | C1 decoding fixed |
| 7 | C3-core (test) | GSM8K test and MATH-500, 2 seeds | 4 | dev beats the frontier |
| 8 | M3b | Selectors × time code (ordinal, rotary, none) at 4K/8K/16K, plus BABILong | 4 | M2/M3a gate |
| 9 | C2, C3-inject | Residual "finish thinking" direction with dynamic gain, against the logit bias; fixed, yoked and random-direction controls | 0.7 + 3 | a detector already beats the frontier |
| 10 | M4 | Trained retrieval key on RecurrentGemma: generic text only, 3 seeds, memory-free adapter, LoRA at equal parameters, foreign store, held-out tasks | 8 | oracle − best training-free ≥ 15 points |
| 11 | B0, B1 | Branch gain (fixed and one dynamic law) on bounded-cache thinking | 1 + 2.5 | B-a shows the hybrid loses ≥ 5 points |
| 12 | M1b | Loop against adapter with distillation, LoRA, FineWeb → WikiText, 3 seeds | 3 | M1a: loop − adapter ≥ 3 points |
| 13 | M5 | Lost-in-the-middle base-dip check, then the method | 0.3 (+1.7) | dip ≥ 8 points |

Week 1 is about 17 GPU-hours. M4 is week 2.

## Pre-registered primary comparisons

- **C0:** best stop rule's accuracy minus the truncation frontier at matched mean tokens (probe cost included), on held-out problems, paired bootstrap.
- **C1:** hybrid non-closure rate and accuracy at (1.0, 1.5) against (0.6, 0) on the same 200 problems.
- **M1a:** loop's gap closure minus the memory-free adapter's, 2 seeds per cell; the foreign-state drop is the attribution check.
- **M2/M3a:** beyond-window exact match at 8K, oracle-ordinal read-back against native. Gate: oracle ≥ 0.5 and native ≤ 0.8.
- **C3-core:** vertical distance of each controller from the real-truncation frontier (Δ accuracy at matched total tokens), per model, 2 seeds.
- **B-a:** accuracy at a 2,048-token budget, windowed minus full cache, hybrid against Qwen3.
- **M3b:** best own selector against InfLLM at equal KV columns; ordinal against true time on order and update.
- **C2/C3-inject:** injection's tokens at matched accuracy against the logit bias's, with a random direction as the attribution control.
- **M4:** learned key minus InfLLM, and minus the memory-free adapter, 3 seeds, held-out BABILong; the gain must vanish with a foreign store.

## Stop rules for the direction

- **Thinking control closes** if vendor decoding brings non-closure to ≤ 10% and no stop rule beats the frontier by ≥ 2 points or ≥ 15% tokens on held-out problems.
- **Memory closes** if RecurrentGemma shows no headroom (native ≥ 0.8) or an unusable read path (oracle < 0.5), and M1a gives loop ≈ adapter.
- **If both close,** write up the negatives and stop; do not design round 13.

## Code (new files under `experiments/ca2/`; existing files unchanged while jobs import them)

- `decode.py`: logits processors (presence penalty, `</think>` bias, forced close, n-gram loop rule); segmented generation with cache reuse and a forked probe.
- `stoprules.py`: C0 offline analysis from the reasoning traces.
- `think.py`: C1 and C3 runner (decoding grid, controllers, frontier), reusing `reason.py` helpers by import.
- `adapters.py`: parameter-matched memory-free adapter and the M1 trainer around `pretrained_loop`.
- `readback.py`: attention read-back for models with native sliding layers; selectors; time codes.
- `bounded.py`: sink + window mask for generation with a KV cache (B-a).
- Tests for each before any GPU time.

## Needs Daniel

- Accept the RecurrentGemma licence on Hugging Face (gated).
- Approve installing `math_verify` (MATH-500 grading) and `causal_conv1d` (faster hybrid decoding) into the venv.
- Approve downloads: `google/recurrentgemma-2b`, `-2b-it`, `google/gemma-4-E2B-it`, BABILong.

## Records (overdue; no commit unless asked)

- REGISTER: DEC for the Stage 0 outcome (gist direction dropped; training on text permitted for reportable results; windowing only on natively windowed or bounded-cache hybrids), P rows for Stage 0, backlog rows for C0–C3, B-a, M1–M5.
- NOTEBOOK entry; HANDOFF branch and status; `results/README.md` provenance for `results/evolve/round12/`.
- `research/plans/round12_plan.md` from this file and the scratchpad sources.

## Verification

- CPU tests for each new module; the full suite stays green.
- C0 reported with held-out intervals; C1 with paired differences; human checkpoint after C0 + C1 + M1a.
