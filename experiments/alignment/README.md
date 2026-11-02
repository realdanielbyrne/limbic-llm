# Alignment track: CA2 as a norm-steering controller

A runtime side channel that steers a model away from harmful outputs, more strongly the more a harm signal persists. The design and its motivation are in [`research/NOTEBOOK.md`](../../research/NOTEBOOK.md) ("CA2 as a norm-steering controller" and "re-aligning an unaligned model"). Experiment IDs (A1–A7) and status are in [`research/REGISTER.md`](../../research/REGISTER.md).

## Pieces

| File | What it does |
|---|---|
| `steering.py` | `SteeringController`: probe score $s_t$, leaky integral $E_t = \lambda E_{t-1} + s_t$, and injection $h \leftarrow h + g_t\lVert h\rVert\hat d$, with modes `none`, `static`, `conditional` and `graded` ($g_t = G\tanh(\kappa E_t)$). Directions are `fixed` (e.g. the refusal direction) or `gist` (fast minus slow running average of the stream, i.e. momentum, open loop). Carries state across KV-cached decoding steps and runs in float32. |
| `probe.py` | Activations at the last prompt token, a logistic harm probe with held-out AUROC and calibration error, and the difference-of-means refusal direction. |
| `data.py` | Prompt sets from Hub specs, local files or synthetic stand-ins; chat formatting with thinking turned off; left-padded batches. |
| `judge.py` | Refusal detection by substring, degeneration score (the Asimov "freeze"), an optional guard model, and the data rule. |
| `run.py` | The experiment: probe (A1), refusal direction, then five arms on four evaluation sets. |

## Measurements

| Metric | Set | Better |
|---|---|---|
| Refusal rate | harmful (`mlabonne/harmful_behaviors` test), contrast (XSTest unsafe) | higher |
| Refusal rate | borderline (XSTest safe), harmless (`mlabonne/harmless_alpaca` test) | lower |
| Selectivity | harmful − borderline | higher |
| Degeneration | share of repeated 4-grams; flagged above 0.5 | lower |
| Perplexity | `--ppl-data` text, with the controller active | close to the no-hook arm |
| Guard unsafe rate | optional, `--guard` | lower |
| Probe AUROC and calibration | held-out 20%, optionally at every layer | — |
| Controller statistics | mean and max gain, share of tokens at the cap, final $E$ | — |

## Arms

`none`, `static` (activation addition), `conditional` (CAST-style switch), `graded` (the escalating controller) and `graded_self` (graded, with the direction estimated from the target model alone). With `--direction-from`, the refusal direction comes from an aligned sibling, as experiment A6 needs.

## Data rule

The results JSON (`results/alignment/<name>.json`) holds labels, scores and aggregates only. Response text is written only when `--private-dir` is given, and that directory must lie outside the repository (`judge.assert_private` refuses otherwise). Never commit completions to harmful prompts.

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
    --ppl-data hf:Salesforce/wikitext:wikitext-2-raw-v1:test:text \
    --private-dir ~/limbic_private/a6 --name a6_qwen3_abliterated

# references: the aligned model itself, and the never-aligned base
python experiments/alignment/run.py --model Qwen/Qwen3-1.7B --arms none --name ref_qwen3_aligned ...
python experiments/alignment/run.py --model Qwen/Qwen3-1.7B-Base --direction-from Qwen/Qwen3-1.7B --name a6_qwen3_base ...
```

Gains are fractions of $\lVert h\rVert$ (`--gain` for static and conditional, `--cap` for graded). Keep them below the saturation knee from stability experiment S1 once it is measured.
