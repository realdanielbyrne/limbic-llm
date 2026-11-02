# Handoff: pretrained-model side-channel runs (from the geof session)

Written by the Claude session on geof (2× RTX 5090) that runs
`experiments/pretrained_loop.py`. It is meant for the session that combines
results into one report. This file is the only one this session edits by hand.
It is updated each time a run lands.

## Branches (2 Nov 2026)

At Daniel's request, PR #2 (`feat/temporal-loop` → `main`) has been merged. **From now on this session works on `feat/round1-side-memory`**, branched from `main` after the merge. It pushes round-1 results, code under `experiments/evolve/`, and updates to this note there. `feat/temporal-loop` is kept, not deleted. Pull `feat/round1-side-memory` to see round-1 results.

## Status (rerun after the bf16 fix)

Every run before commit `3c90bde` had a bug: the side channels were cast to bf16. Their Adam updates
(about 1e-3) were smaller than bf16's spacing near 1 (2^-7), so they rounded away. As a result the
loop gain stayed at its initial value of 0.2689 in every run, and several training curves were flat.
All earlier pretrained numbers are **superseded**, and the runs below are being redone with float32 side channels.

| run | status |
|---|---|
| `pretrained_qwen3_needle` (3000 steps) | **done** (fixed) |
| `pretrained_qwen35_needle` (attention only, 3000 steps) | **done** (fixed) |
| `pretrained_qwen35_needle_allwin` (all layers, 3000 steps) | **done** (fixed) |
| `pretrained_qwen35_needle_layer22` (attention only, tap at block 22, 3000 steps) | **done** (fixed) |
| `pretrained_qwen3_lm` | **done** (fixed rerun) |
| `pretrained_qwen35_lm` | **done** (fixed rerun). The bf16 run was **stopped** at step 600 of 1000 and never wrote a JSON. |
| `pretrained_qwen3_needle_horizon` (`--kernel exp --rho 0.95 --tau 32`, 3000 steps) | **done** |
| `pretrained_qwen35_needle_allwin_horizon` (same kernel, all layers windowed, 3000 steps) | **done** |

The horizon runs use a loop kernel that reaches the needle: w(Δ) = ρ^(Δ/τ−1), which is 0.472 at a lag of 500 and 0.470 at 503 (`limbic_llm/temporal_loop.py:100`). The default `osc` kernel's weight is 0 at that gap.

The rerun JSONs overwrite the old files in place. Git history keeps the bf16 versions (commits
`0f5cc8c`, `775e0f2`, `a741b80`, `2297571`).

## Round-0 diagnosis and evolution notebook

See [`research/NOTEBOOK.md`](NOTEBOOK.md) and [`research/plans/round0_plan.md`](plans/round0_plan.md). For the report:

- The Qwen3 windowed+side eval loss, 1.8421, equals 0.8·ln10, the answer-format floor exactly. The side channels learned the leading space and the digit distribution, and carry no information about the value.
- **Caution:** the bf16 run's all-windowed drop (eval 1.692) was an **anti-causal leak**, not long-range transfer. `torch.arange(512, dtype=bfloat16)` has only 385 distinct values; positions 505–511 become [504, 506, 508, 508, 508, 510, 512]. So the `d >= 0` masks in `loop_kernel` and `decay_kernel` (`limbic_llm/temporal_loop.py:94`, `:109-111`) let some answer positions see their own targets. This session verified it on CPU. The kernels take their dtype from the activations, so any bf16 use of `HippocampalLoop` or `TimeChannel` hits this. The float32 fix (`3c90bde`) avoids it in `pretrained_loop.py`.

## Round 1 results (`results/evolve/round1/`, branch `feat/round1-side-memory`)

Needle task (gap about 500 tokens), seq 512, window 64, every layer windowed unless noted, 3000 steps unless noted. "per digit" lists the loss on digits 1–4; chance is ln 10 = 2.30.

| run | side channels | exact | loss | per digit | notes |
|---|---|---|---|---|---|
| `g1_qwen3_horizon_perpos` | O(T) loop, exp ρ 0.95 τ 32 | 0.00 | 1.381 | **0.001**, 2.30, 2.30, 2.30 | Recovers exactly the first digit; reproduces the earlier run to 4 decimals |
| `g1_qwen3_horizon_pairs2` | same, 2 stored pairs | 0.00 | 1.809 | 2.17, 2.29, 2.28, 2.30 | Loses even the first digit. The 1-pair success did not use the query |
| `g1_qwen35_allwin_horizon_6k` | same loop, Qwen3.5, 6000 steps | **1.00** | 0.001 | 0.002, 0.001, 0.001, 0.001 | Loss broke at ~2400 and reached 0.006 by 3600. The 3000-step run was undertrained |
| **`g1_w1_tied`** | **W1**: GDN side memory, successor-bound dense keys + clock (4.28M params) | **1.00** | **0.0001** | 0.000 × 4 | Solved by step 1200. Fixed-size state (rule E2) |
| **`g1_w1_dg`** | **W1**, DG k-WTA keys (4 heads, key expansion 4; 3.74M params) | **1.00** | **0.0001** | 0.000 × 4 | Learns faster: loss 0.73 at step 300, 0.008 at 600 |
| `g1_w1_tied_pairs2` | W1 dense, 2 stored pairs | 0.55 | 0.187 | **0.74**, 0.05, 0.11, 0.03 | Digit 1 is a coin flip (≈ ln 2); the chain after it is retrieved |
| `g1_w1_dg_pairs2` | W1 DG, 2 stored pairs | 0.43 | 0.231 | **0.76**, 0.11, 0.15, 0.14 | Same; DG keys do not help choose the start |
| full attention (reference) | — | 1.00 | 0.011 | — | |

LM, `g1_qwen3_lm_horizon`: Qwen3 WikiText with the horizon kernel. Perplexity 11.53 against 12.01 with the default kernel and 9.01 with full attention, so 49% of the loss gap is closed against 41%.

The round-1 write-up is in `research/NOTEBOOK.md` ("Round 1"). The register now has entries P7–P12 and the new item W1b.

## Review of the alignment, CoT and CA1 decision tracks (2 Nov, round 1)

Full report: [`research/reviews/2026-11-02_alignment_cot_ca1.md`](reviews/2026-11-02_alignment_cot_ca1.md). It covers commits `73259fc` and `241a13c`. Three track reviewers worked independently, an adversarial verifier checked each finding, and every bug was reproduced on CPU. Items to fix **before running A6, R2 or J1**:

- **A1 (high).** The perplexity check never activates the conditional, graded or graded_self arms. With `score_positions="last"` they inject only at the last position, and those logits are dropped. Perplexity came out bit-identical to the no-hook arm, so the capability metric is vacuous for three of the four hooked arms.
- **C1 (high).** The gist is dominated by the attention-sink token and the stream mean: the EMAs are zero-initialised and position 0 is included. On Qwen3-1.7B-Base block 14, ‖h₀‖ = 13,587 against a median of 167, and cos(gist, h₀) = 0.998 at t = 5. The fix is to exclude the sink, and to initialise from the first included token or subtract a corpus mean. Bias correction alone does not fix it.
- **C2 (high).** R2/R4 lack matched-norm random, mean and anti-gist controls, a sampling or seed protocol, and a non-inferiority margin. `/no_think` is not a baseline for R1-Distill.
- **D1 (high).** J1's hypothesis ("better calibration gives better selectivity at equal AUROC") is ill-posed for a gain of cap·tanh(κE). Compare at a matched operating point instead.
- **Medium:**
  - The probe is fit and calibrated only on the last prompt token, yet the controller integrates it over generated tokens (A2/D2).
  - Degenerate loops count as refusals (A3).
  - The equivalence test passes for conditional only by its seed (A4).
  - Graded gist without a probe is silently a no-op (C3).
  - The STU-PID baseline was withdrawn on 28 Sep 2026 (C4).
  - ERSS (2606.06188) and STARS (2601.22484) are missing from the prior-art discussion (C5).
  - J3's premise is misstated (D3).
- **Data rule (A6, low but important).** An absolute or `../` value of `--name` bypasses `assert_private` and writes raw completions inside the repo. Validate the final resolved path.

## Review of the 2 Nov additions (genesis §5.6–5.7, NOTEBOOK, REGISTER, references), for the coordinating session

These files belong to the coordinating session, so this session did not edit them. Corrections and gaps:

1. **RESULTS.md §4.7, horizon table, Qwen3.5 "Side, default kernel" = 1.69.** This is the superseded bf16 number. The fp32 rerun exists (`pretrained_qwen35_needle_allwin.json`, 3000 steps): **1.84 / 0.29 / 0.00**, at the format floor like Qwen3. With 1.69 in the table, the default kernel looks as if it partly worked on Qwen3.5. It did not.
2. **REGISTER has no entry for the Qwen3.5 LM result.** `pretrained_qwen35_lm` (attention layers only windowed): perplexity 16.28 → 11.22 with the side channels, full attention 9.83, which closes **74%** of the loss gap against 41% for Qwen3. P5 should cover both models, with the same domain-adaptation caveat.
3. **The "one digit known" decomposition of P4 is missing from REGISTER and RESULTS.** The Qwen3 horizon run's 1.381 / 0.461 equals exactly (3/5)·ln10 and 0.46, the level for one of the four digits known perfectly and three at chance. This is the most useful mechanistic clue so far. The loop stores key and value from the same position, so nothing moves the lookup on to the next digit. It also links directly to **W1**: successor-bound addressing is exactly that shift.
4. **REGISTER lacks two cheap experiments:**
   - (a) Repeat `pretrained_qwen35_needle_allwin_horizon` with more steps. It left the floor only at step ~2700 of 3000, as the cosine learning rate reached zero, so it is undertrained.
   - (b) Per-digit loss logging (round-0 C1), to confirm which digit the loop recovers.
5. **References checked:** Zhao et al. 2025 (arXiv 2507.11878; Zhao, Huang, Wu, Bau, Shi) exists and says what the notebook claims. The four A6/A7 models exist. `huihui-ai/Huihui-Qwen3.5-9B-abliterated` is tagged image-text-to-text and loads as `AutoModelForMultimodalLM`, so `pretrained_loop.py`'s model loading and `find_blocks` will probably need changes for A7.
6. **Agree with O3:** a fixed-size state for the pretrained track, keeping the O(T) loop with the horizon kernel as the reference arm. W1 also targets the observed one-digit failure.

## Files this session owns

Every `results/pretrained/pretrained_*.json` and `results/pretrained/pretrained_*.log` file, plus this note. Each JSON holds
`args`, `model` (layer counts, `windowed_layers`, `window_check`), `curve` (training points every 100
steps), and `results` with three conditions:

- `windowed`: the frozen base model under the window, side channels off
- `windowed+side`: the same model with the hippocampal loop and the clock on (4.38M trainable params)
- `full`: full attention, the upper bound

`results.gain` is the loop's final sigmoid gain. A value of exactly 0.2689 means the gain never trained.

## Fixed-run results

### Needle task: float32 side channels, 3000 steps, seq 512, batch 8, window 64, needle gap about 501 tokens

Each cell is exact / token_acc / loss. "gain" is the final loop gain; the initial value is 0.2689.

| run | windowed layers | tap | windowed | windowed+side | full | gain |
|---|---|---|---|---|---|---|
| `pretrained_qwen3_needle` | 28/28 | 14 | 0.00 / 0.106 / 3.069 | 0.00 / 0.287 / 1.842 | 1.00 / 1.000 / 0.011 | 0.2797 |
| `pretrained_qwen35_needle` | 6/24 attention | 12 | 0.00 / 0.056 / 4.028 | **1.00 / 1.000 / 0.0000** | 1.00 / 1.000 / 0.034 | 0.2697 |
| `pretrained_qwen35_needle_allwin` | 24/24 | 12 | 0.00 / 0.130 / 3.039 | 0.00 / 0.289 / 1.843 | 1.00 / 1.000 / 0.034 | 0.2718 |
| `pretrained_qwen35_needle_layer22` | 6/24 attention | **22** | 0.00 / 0.056 / 4.028 | 0.00 / 0.345 / 1.734 | 1.00 / 1.000 / 0.034 | 0.2933 |

- **Qwen3:** the fix did not help. The training loss stayed flat at about 1.84 for all 3000 steps (1.840 at the end), and exact match is still 0. The gain did train (0.2689 → 0.2797), so bf16 rounding was not the only reason this run was stuck. A diagnosis is in progress in this session.
- **Qwen3.5 with only the attention layers windowed:** reproduces the bf16-era result. Exact is 1.0 with near-zero loss, and the gain moved only slightly (0.2697). Whether the loop reads a trace kept by the Gated DeltaNet layers or replays the needle into them is what the allwin and layer-22 runs test.
- **Qwen3.5 with all layers windowed:** with every Gated DeltaNet layer windowed too, the loop on its own fails exactly as on Qwen3. The training loss stays flat at about 1.84 for all 3000 steps (1.831 at the end), and exact match is 0. The late drop to 1.60 in the bf16 1000-step run did not reproduce. Combined with the kernel calculation below, this is expected: no route remains to a needle about 500 tokens back.
- **Qwen3.5 with the tap moved to block 22:** exact match drops from 1.0 to 0.0, and training loss plateaus around 1.7–1.8, like the failing runs. With the tap at 12, the loop's output passes through 9 more Gated DeltaNet layers. At 22 only blocks 22–23 come after it, and block 23 is windowed attention. So the success depends on Gated DeltaNet layers after the tap, not just on a trace that already exists when the loop reads its input. This supports the "the loop feeds the needle into the GDN layers after it" reading, not "the loop reads a trace the GDN layers already hold".
- **Why the loop cannot recall alone (Qwen3), confirmed by calculation:** with the default `--kernel osc --rho 0.7 --tau 16`, the loop weight is w(Δ) = ρ^(Δ/τ−1)·[cos(2πΔ/τ)]₊ (`limbic_llm/temporal_loop.py:100-102`). That gives 1.0 at Δ=16, 0.08 at 128, 0.005 at 256, about 4e-19 at 500, and **exactly 0 at Δ = 501–503** (the needle gap falls on the clamped half of the cosine). The loop cannot see the needle on this task, whatever the training. Its kernel limits it to roughly the last 100–200 tokens. On Qwen3.5 with tap 12, it can only be passing along recent content that the earlier Gated DeltaNet layers have already carried forward.

### Needle task with a loop horizon that covers the gap (`--kernel exp --rho 0.95 --tau 32`, w(500) ≈ 0.47)

| run | windowed layers | windowed | windowed+side | full | gain |
|---|---|---|---|---|---|
| `pretrained_qwen3_needle_horizon` | 28/28 | 0.00 / 0.106 / 3.069 | 0.00 / **0.461 / 1.381** | 1.00 / 1.000 / 0.011 | 0.3264 |
| `pretrained_qwen35_needle_allwin_horizon` | 24/24 (GDN windowed too) | 0.00 / 0.130 / 3.039 | 0.023 / **0.550 / 1.239** | 1.00 / 1.000 / 0.034 | 0.2768 |

- **Qwen3.5, all layers windowed:** the loop also crosses the gap here. Training loss stayed at the floor (about 1.84) until step 2400, dropped to 1.36 by step 2700, and was 1.395 at step 3000.
  - Eval (1.239 / 0.550) falls between the k = 1 and k = 2 levels in the table below. That means one digit plus partial knowledge of a second, and 3 of 128 sequences fully correct.
  - The breakthrough came late, when the cosine learning-rate schedule was already near zero, so this run is very likely **undertrained**. A longer run would probably improve it.

- **Training curve (loss / token_acc):** 300: 1.844 / 0.25 → 600: 1.683 / 0.32 → 900: 1.372 / 0.47, then flat at 1.38 to step 3000.
- **First real long-range transfer through the loop on a pure-attention model.** Qwen3 leaves the format floor (1.842) for the first time. With all 28 layers windowed, the loop is the only route to the needle.
- **The plateau is exactly "one digit known".** Mark k of the 4 value digits as known exactly and the rest at chance. Then loss = (4−k)/5·ln10 and token_acc = 0.2 + 0.2k + 0.02(4−k):

  | k | loss | token_acc |
  |---|---|---|
  | 0 | 1.842 | 0.28 |
  | **1** | **1.382** | **0.46** |
  | 2 | 0.921 | 0.64 |

  The measured 1.381 / 0.461 matches k = 1. So the loop carries one of the four digits perfectly and the other three not at all. Exact match therefore stays 0.
- **Likely reason** (round-0 hypothesis "capacity H4"): the loop stores key and value from the *same* position (`limbic_llm/temporal_loop.py:262-263`). A query can therefore fetch the content at the position it matches, but nothing shifts the address to the *next* digit. Which digit is recovered is unknown: per-digit logging (round-0 item C1) is needed to confirm it, but it is most likely the first.
- **The fix is a round-1 direction:** address-shift keys, meaning keys taken from the previous token (or the TimeChannel's lag features), so that each answer digit retrieves the next stored one.

### Language modelling: float32 side channels, WikiText-103, seq 1024, window 256, batch 4, 1000 steps, grad checkpointing

Training set: 1,155,524 tokens. Eval set: 116,893 tokens in 114 sequences. Each cell is perplexity / loss. Gap closed = (windowed − side) / (windowed − full), measured on loss.

| run | windowed layers | windowed | windowed+side | full | gap closed | gain |
|---|---|---|---|---|---|---|
| `pretrained_qwen3_lm` | 28/28 | 14.62 / 2.683 | 12.01 / 2.486 | 9.01 / 2.199 | 41% | 0.3681 |
| `pretrained_qwen35_lm` | 6/24 attention (GDN intact) | 16.28 / 2.790 | 11.22 / 2.417 | 9.83 / 2.286 | **74%** | 0.3744 |

- **Qwen3.5 (GDN intact):** windowing only the 6 attention layers costs more perplexity (9.83 → 16.28) than windowing all of Qwen3 (9.01 → 14.62), even though GDN still reaches the whole sequence. The side channels recover 74% of that loss gap, against 41% on Qwen3. The same domain-adaptation caveat applies. The larger recovery fits the needle result: with the tap at block 12 and GDN layers after it, the loop's re-injection can be carried forward and used.

- The gain trained (0.2689 → 0.368), and the injection ratio rose to 0.254 from about 0.20 in the needle runs.
- The result is close to the superseded bf16 run (11.91, 43%).
- **Interpretation caveat:** with the default `osc` kernel (ρ 0.7, τ 16), the loop's weight falls by e every ~45 tokens and is about 0.005 at 256. So it reaches little beyond the 256-token window. Much of this 41% is probably local or domain adaptation (only this condition trains on WikiText), not memory from beyond the window. A run with the horizon kernel, or one with the loop's memory ablated, would separate the two.

## Superseded results: bf16 bug (do not cite)

These were measured with bf16 side channels whose updates rounded away. Gain was 0.2689 in every run.
They are kept only for comparison with the reruns.

Needle task: seq 512, batch 8, window 64, 1000 steps, needle gap about 501 tokens. Each cell is exact / token_acc / loss.

| run | windowed layers | windowed | windowed+side | full |
|---|---|---|---|---|
| qwen3_needle (Qwen3-1.7B-Base) | 28/28 | 0.00 / 0.106 / 3.07 | 0.00 / 0.280 / 1.84 | 1.00 / 1.000 / 0.011 |
| qwen35_needle (Qwen3.5-2B-Base) | 6/24 attention; 18 GDN intact | 0.00 / 0.056 / 4.03 | 1.00 / 1.000 / 0.0006 | 1.00 / 1.000 / 0.034 |
| qwen35_needle_allwin | 24/24 | 0.00 / 0.130 / 3.04 | 0.00 / 0.320 / 1.69 | 1.00 / 1.000 / 0.034 |

Language modelling: WikiText-103, seq 1024, window 256, 1000 steps. Each cell is perplexity / loss.

| run | windowed | windowed+side | full |
|---|---|---|---|
| qwen3_lm | 14.62 / 2.683 | 11.91 / 2.477 | 9.01 / 2.199 |
| qwen35_lm | stopped, no result | | |

Caveat that still applies to the reruns: in the LM task only `windowed+side` trains anything on WikiText
(4.38M parameters). The base model stays frozen in the other conditions. Part of any LM gain may
therefore be adaptation to the WikiText domain rather than long-range recall.
