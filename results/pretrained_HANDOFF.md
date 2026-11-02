# Handoff: pretrained-model side-channel runs (from the geof session)

Written by the Claude session on geof (2× RTX 5090) that runs
`experiments/pretrained_loop.py`. It is meant for the session that combines
results into one report. This file is the only one this session edits by hand.
It is updated each time a run lands.

## Files this session owns

Every `results/pretrained_*.json` and `results/pretrained_*.log` file, plus this
note. Each JSON holds `args`, `model` (layer counts, `windowed_layers`,
`window_check`), `curve` (training points every 100 steps), and `results`
with three conditions:

- `windowed`: the frozen base model under the window, side channels off
- `windowed+side`: the same model with the hippocampal loop and the clock on (4.38M trainable params)
- `full`: full attention, the upper bound

All runs: bfloat16, seed 0, 1000 steps, lr 1e-3, eval_n 128.
The needle runs use seq 512, batch 8, window 64, with a needle gap of about 501 tokens.

## Needle results (exact match / token_acc / loss)

| run | model | windowed layers | windowed | windowed+side | full |
|---|---|---|---|---|---|
| `pretrained_qwen3_needle` | Qwen3-1.7B-Base | 28/28 attention | 0.00 / 0.106 / 3.07 | 0.00 / 0.280 / 1.84 | 1.00 / 1.000 / 0.011 |
| `pretrained_qwen35_needle` | Qwen3.5-2B-Base | 6/24 (attention only; 18 GDN intact) | 0.00 / 0.056 / 4.03 | **1.00 / 1.000 / 0.0006** | 1.00 / 1.000 / 0.034 |
| `pretrained_qwen35_needle_allwin` | Qwen3.5-2B-Base | 24/24 (GDN windowed too) | 0.00 / 0.130 / 3.04 | 0.00 / 0.320 / 1.69 | 1.00 / 1.000 / 0.034 |

How to read the table. Please keep these caveats in the combined report.

1. **The loop alone does not recall the needle at a gap of about 500 tokens on either model.** This holds for Qwen3, and for Qwen3.5 with every layer windowed. In both cases the side channels lower loss (3.0 to 1.7–1.8) and raise token_acc (0.11–0.13 to 0.28–0.32), but exact match stays 0.
2. **The Qwen3.5 attention-only success needs both parts.** The GDN layers alone give exact 0 (the windowed row), and the loop alone gives exact 0 (the allwin row). Together they give exact 1.0. Do not credit the loop alone with that result. The likely reading is that the GDN state carries a weak trace of the needle and the side channel learns to read it out. That reading is a hypothesis; we have no probe for it.
3. The allwin training loss was flat at about 1.85 until step 700, then fell to 1.60 by step 1000. The Qwen3 curve stayed flat at about 1.84. The allwin run may be undertrained, and a longer run (`--steps 3000`) is worth trying before calling the loop-only result a hard negative.
4. `gain` is 0.269 in every run (sigmoid(-1), its starting value). The injection ratio sits at about 0.20 in every run.

## Language-modelling runs (WikiText-103, seq 1024, window 256, batch 4, grad checkpointing)

The training set is 1,155,524 WikiText-103 tokens. The eval set is 116,893 tokens in 114 sequences.

| run | model | windowed layers | windowed ppl / loss | windowed+side ppl / loss | full ppl / loss | gap closed |
|---|---|---|---|---|---|---|
| `pretrained_qwen3_lm` | Qwen3-1.7B-Base | 28/28 | 14.62 / 2.683 | 11.91 / 2.477 | 9.01 / 2.199 | 43% of loss gap |
| `pretrained_qwen35_lm` | Qwen3.5-2B-Base | attention only | *pending* | *pending* | *pending* | |

"Gap closed" is (windowed − side) / (windowed − full), measured on eval loss.

Caveat for the LM rows: in the `windowed` and `full` conditions the base model is frozen and never trained. Only `windowed+side` trains 4.38M parameters on WikiText. Some of its gain could therefore be adaptation to the WikiText domain rather than long-range recall. A proper control would train side channels with the memory read-out ablated (or measure loss by token position beyond the window). This session has not run that control.

This section gets filled in as each run lands. Check `git log -- results/pretrained_*` for the latest.
