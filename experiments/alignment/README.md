# Alignment track: CA2 as a norm-steering controller

A runtime side channel that steers a model away from harmful outputs, more strongly the more a harm signal persists. The design and its motivation are in [`research/NOTEBOOK.md`](../../research/NOTEBOOK.md) ("CA2 as a norm-steering controller" and "re-aligning an unaligned model"). Experiment IDs (A1–A7) and status are in [`research/REGISTER.md`](../../research/REGISTER.md).

## Pieces

| File | What it does |
|---|---|
| `steering.py` | `SteeringController`: score $s_t$, leaky integral $E_t = \lambda E_{t-1} + s_t$, and injection $h \leftarrow h + g_t\lVert h\rVert\hat d$, with modes `none`, `static`, `conditional` and `graded` ($g_t = G\tanh(\kappa E_t)$). The score comes from a probe dict, a per-token `score_fn`, or (conditional only) a per-prompt `external_decision`; a mode without the score it needs raises. Directions: `fixed` (e.g. the refusal direction); `gist` (fast minus slow bias-corrected running average of the stream, skipping the attention-sink token, i.e. momentum); and the matched-norm controls `anti_gist`, `mean` and `random`. Carries state across KV-cached decoding steps and runs in float32. |
| `probe.py` | Activations at the last prompt token (position ids from the padding mask), a logistic harm probe with held-out AUROC and calibration error, and the difference-of-means refusal direction. |
| `data.py` | Prompt sets from Hub specs, local files or synthetic stand-ins; chat formatting with thinking turned off; left-padded batches. |
| `judge.py` | Refusal detection by substring, counted only for non-degenerate responses; degeneration score (the Asimov "freeze") and repetition onset; an optional guard model; and the data rule (`assert_private`, `safe_name`, `private_file`). |
| `run.py` | The experiment: probe (A1), refusal direction, then six arms on four evaluation sets, optionally swept over gain or cap and compared at a matched harmful-refusal rate. |

## Measurements

| Metric | Set | Better |
|---|---|---|
| Refusal rate (refusal markers in a non-degenerate response; `refusal_raw` without that condition) | harmful (`mlabonne/harmful_behaviors` test), contrast (XSTest unsafe) | higher |
| Refusal rate | borderline (XSTest safe), harmless (`mlabonne/harmless_alpaca` test) | lower |
| Selectivity | harmful − borderline | higher |
| Degeneration, repetition onset, token-limit rate | share of repeated 4-grams (flagged above 0.5); first token at which a 32-token window turns repetitive; share of responses that hit `--max-new-tokens` | lower |
| Perplexity | `--ppl-data` text: the first `--ppl-prefix` tokens are prefilled as a prompt and the rest teacher-forced through the KV cache, so the controller acts on every scored token (`ppl_gain_mean` shows how much) | close to the no-hook arm |
| Guard unsafe rate | optional, `--guard` | lower |
| Probe AUROC and calibration | held-out 20% at the last prompt token, optionally at every layer; per arm, `calibration` at the last prompt token and on generated tokens (the positions that drive the graded gain), in distribution and on XSTest | — |
| Controller statistics | mean and max gain over response positions (last prompt token and generated tokens before EOS), mean gain over prompt tokens, share at the cap, final $E$, mean score at the prompt and on generated tokens | — |
| Operating points (with `--gains` / `--caps`) | each family's curve, and borderline and harmless refusal interpolated at `--target-refusal` | lower at the same harmful refusal |

## Arms

`none` (a controller that records scores and never injects), `static` (activation addition), `static_random` (the same along a seeded random direction: the matched-norm control), `conditional` (CAST-style switch), `graded` (the escalating controller) and `graded_self` (graded, with the direction estimated from the target model alone). With `--direction-from`, the refusal direction comes from an aligned sibling, as experiment A6 needs; its layer count and width must match the target's.

`--gains 0.1,0.2,0.3` sweeps the static and conditional gain, and `--caps 0.2,0.3,0.4` the graded cap; swept arms are keyed `static@0.2`, `graded@0.3` and so on. Compare arms at a matched harmful-refusal rate (`operating_points`), not at a fixed cap: at a fixed cap, selectivity depends on where the scores sit on the gain curve (review RV1.D1).

## Data rule

The results JSON (`results/alignment/<name>.json`) holds labels, scores and aggregates only. Response text is written only when `--private-dir` is given, and that directory must lie outside the repository (`judge.assert_private` refuses otherwise). `--name` may contain only letters, digits, `.`, `_` and `-`, and each private file is checked after resolution (`judge.private_file`). Never commit completions to harmful prompts.

## Interactive notebook

[`docs/notebooks/activation_additions_qwen3.ipynb`](../../docs/notebooks/activation_additions_qwen3.ipynb) runs the same controller and probe interactively. Its config cell switches alignment and CoT steering, a prompt-pair or difference-of-means vector, and ActAdd or controller injection. First spot checks on the abliterated model (register N1–N5, NOTEBOOK "First steering runs on real models"):
- The aligned sibling's direction (`DIRECTION_FROM="Qwen/Qwen3-1.7B"`) starts to restore refusal selectively only at a gain of 0.3 (block 14). Sweep `--gain` and `--cap` past 0.3 in A6.
- The thinking direction collapses into repetition at that same gain. Regulating the CoT gain against repetition is R6; the gain-law comparison for alignment is A8.

## Running

```bash
# offline smoke test (random tiny model, synthetic prompts)
OMP_NUM_THREADS=1 python experiments/alignment/run.py --tiny --n-train 32 --n-eval 8 --max-new-tokens 8 --device cpu

# A1: is harmfulness still decodable after abliteration? (probe at every layer)
python experiments/alignment/run.py --model mlabonne/Qwen3-1.7B-abliterated --dtype bfloat16 --device cuda \
    --layer-sweep --arms none --name a1_abliterated

# A6: re-align the abliterated model with the aligned sibling's refusal direction
python experiments/alignment/run.py --model mlabonne/Qwen3-1.7B-abliterated --direction-from Qwen/Qwen3-1.7B \
    --dtype bfloat16 --device cuda --n-eval 100 --max-new-tokens 96 \
    --gains 0.1,0.2,0.3,0.4 --caps 0.2,0.3,0.4,0.5 \
    --ppl-data hf:Salesforce/wikitext:wikitext-2-raw-v1:test:text \
    --private-dir ~/limbic_private/a6 --name a6_qwen3_abliterated

# references: the aligned model itself, and the never-aligned base
python experiments/alignment/run.py --model Qwen/Qwen3-1.7B --arms none --name ref_qwen3_aligned ...
python experiments/alignment/run.py --model Qwen/Qwen3-1.7B-Base --direction-from Qwen/Qwen3-1.7B --name a6_qwen3_base ...
```

Gains are fractions of $\lVert h\rVert$ (`--gain` for static and conditional, `--cap` for graded). Keep them below the saturation knee from stability experiment S1 once it is measured. The fixes for review RV1 are listed in [`research/NOTEBOOK.md`](../../research/NOTEBOOK.md), "RV1 fixes".
