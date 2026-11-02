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
| `pretrained_qwen35_needle_allwin` (all layers, 3000 steps) | running, GPU 0 |
| `pretrained_qwen35_needle_layer22` (attention only, tap at block 22, 3000 steps) | running, GPU 1 |
| `pretrained_qwen3_lm` | queued (rerun) |
| `pretrained_qwen35_lm` | queued (rerun). The bf16 run was **stopped** at step 600 of 1000 and never wrote a JSON. |

The rerun JSONs overwrite the old files in place. Git history keeps the bf16 versions (commits
`0f5cc8c`, `775e0f2`, `a741b80`, `2297571`).

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

- **Qwen3:** the fix did not help. The training loss stayed flat at about 1.84 for all 3000 steps (1.840 at the end), and exact match is still 0. The gain did train (0.2689 → 0.2797), so bf16 rounding was not the only reason this run was stuck. A diagnosis is in progress in this session.
- **Qwen3.5 with only the attention layers windowed:** reproduces the bf16-era result. Exact is 1.0 with near-zero loss, and the gain moved only slightly (0.2697). Whether the loop reads a trace kept by the Gated DeltaNet layers or replays the needle into them is what the allwin and layer-22 runs test.

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
