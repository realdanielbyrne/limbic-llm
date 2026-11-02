# Handoff: pretrained-model side-channel runs (from the geof session)

Written by the Claude session on geof (2× RTX 5090) that runs
`experiments/pretrained_loop.py`. It is meant for the session that combines
results into one report. This file is the only one this session edits by hand.
It is updated each time a run lands.

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
| `pretrained_qwen35_lm` | running (rerun). The bf16 run was **stopped** at step 600 of 1000 and never wrote a JSON. |
| `pretrained_qwen3_needle_horizon` (`--kernel exp --rho 0.95 --tau 32`, 3000 steps) | queued |
| `pretrained_qwen35_needle_allwin_horizon` (same kernel, all layers windowed, 3000 steps) | queued |

The horizon runs use a loop kernel that reaches the needle: w(Δ) = ρ^(Δ/τ−1), which is 0.472 at a lag of 500 and 0.470 at 503 (`limbic_llm/temporal_loop.py:100`). The default `osc` kernel's weight is 0 at that gap.

The rerun JSONs overwrite the old files in place. Git history keeps the bf16 versions (commits
`0f5cc8c`, `775e0f2`, `a741b80`, `2297571`).

## Round-0 diagnosis and evolution notebook

See [`results/evolve/NOTEBOOK.md`](evolve/NOTEBOOK.md) and [`results/evolve/round0_plan.md`](evolve/round0_plan.md). For the report:

- The Qwen3 windowed+side eval loss, 1.8421, equals 0.8·ln10, the answer-format floor exactly. The side channels learned the leading space and the digit distribution, and carry no information about the value.
- **Caution:** the bf16 run's all-windowed drop (eval 1.692) was an **anti-causal leak**, not long-range transfer. `torch.arange(512, dtype=bfloat16)` has only 385 distinct values; positions 505–511 become [504, 506, 508, 508, 508, 510, 512]. So the `d >= 0` masks in `loop_kernel` and `decay_kernel` (`limbic_llm/temporal_loop.py:94`, `:109-111`) let some answer positions see their own targets. This session verified it on CPU. The kernels take their dtype from the activations, so any bf16 use of `HippocampalLoop` or `TimeChannel` hits this. The float32 fix (`3c90bde`) avoids it in `pretrained_loop.py`.

## Files this session owns

Every `results/pretrained_*.json` and `results/pretrained_*.log` file, plus this note. Each JSON holds
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

### Language modelling: float32 side channels, WikiText-103, seq 1024, window 256, batch 4, 1000 steps, grad checkpointing

Training set: 1,155,524 tokens. Eval set: 116,893 tokens in 114 sequences. Each cell is perplexity / loss. Gap closed = (windowed − side) / (windowed − full), measured on loss.

| run | windowed layers | windowed | windowed+side | full | gap closed | gain |
|---|---|---|---|---|---|---|
| `pretrained_qwen3_lm` | 28/28 | 14.62 / 2.683 | 12.01 / 2.486 | 9.01 / 2.199 | 41% | 0.3681 |

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
