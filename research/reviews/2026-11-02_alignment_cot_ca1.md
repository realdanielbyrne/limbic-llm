# Review of the alignment, CoT and decision tracks (commits 73259fc and 241a13c, compared with 8314d5a)

These are the findings that held up under verification. Where a verdict corrected a finding, the corrected form is used. Where two reviewers found the same problem, it is listed once and cross-referenced. Some reviewer line numbers were off, and the ones below are fixed: `refused()` is at `judge.py:29-31`, `assert_private` is at `judge.py:43-49`, the README perplexity criterion is at `experiments/alignment/README.md:23`, and the incremental-decode test is at `tests/test_alignment.py:28-35`.

---

## 1. Alignment code (`experiments/alignment/`, `tests/test_alignment.py`)

### High
**A1. The perplexity check never measures the conditional, graded or graded_self arms.** (`run.py:124-129`, `run.py:229-230`, `steering.py:105-107,123-124`)
- **Problem:** `perplexity()` makes one forward pass with `calls==0`, and the controller defaults to `score_positions="last"`. As a result, s, E and g are non-zero only at the last position. Those logits are then dropped by `lg[:, :-1]` (`run.py:129`).
- **Failure scenario:** In A6 the graded arm can inject up to 0.3·‖h‖ and still meet "close to the no-hook arm" (`README.md:23`). The CPU run gave bit-identical perplexity for none, conditional, graded and graded_self (539.97); only static differed (537.99).
- **Smallest fix:** Split each perplexity text into a prefix and a continuation. Prefill the prefix (calls=0), then teacher-force the continuation through the KV cache (calls≥1, so every position is scored). Score only the continuation logits.

### Medium
**A2. The probe is fit and calibrated only on the last prompt token, but the graded arm integrates it over generated tokens.** (`probe.py:29,35,104-111`; `steering.py:32-33,103-114,126`)
- **Problem:** The mu/sd standardisation, AUROC and ECE all come from the last-prompt-token (template) position. The docstring at `steering.py:32-33` wrongly says the probe was trained on generated tokens. The decision lens reported the same gap for J1 ECE (D2).
- **Failure scenario:** On benign answers, s_t drifts off-distribution, E climbs and the graded arm over-refuses. The graded-vs-conditional comparison in A2/A6 then reflects probe extrapolation, not controller design. Only the per-record `s_mean`/`E_final` (`run.py:245-248`) would hint at it, and those mix prompt and generated tokens and are not aggregated per set.
- **Smallest fix:**
  - Report the probe score and ECE on generated-token activations, keeping the activations in the private dir under the data rule.
  - Aggregate s over generated tokens only, per set.
  - Correct the docstring.

**A3. Headline refusal and selectivity count degenerate loops as refusals.** (`run.py:243-244,260,269`; `judge.py:29-31`)
- **Problem:** `refused()` matches markers in the first 300 characters regardless of the degeneration score. Degeneration is printed for the harmful set only (`run.py:275`).
- **Failure scenario:** In a gain sweep past the saturation knee, "I cannot I cannot …" on every prompt gives a harmful refusal rate of about 1.0.
- **Smallest fix:** Count a refusal only when the output is also non-degenerate (`refused and not degenerate`). Print degeneration for every set.

**A4. The "matches a single pass for every mode" test passes for conditional only because of the seed, and the run.py configuration is untested.** (`tests/test_alignment.py:28-35`, seed at `:29`; `steering.py:120-121`; `NOTEBOOK.md:280`)
- **Problem:** With `score_positions="all"`, the conditional decision is taken at the last position of the first call. That is position 11 in the one-pass run and position 6 in the incremental run.
- **Failure scenario:** Of seeds 1-8, only seeds 1 and 7 pass. A regression in the `"last"` path (padded prefill, then decode) that run.py actually uses would go unnoticed.
- **Smallest fix:**
  - Run the test with `score_positions="last"` and a padded prefill followed by decode steps.
  - Exclude conditional from the `"all"` equivalence check.
  - Correct `NOTEBOOK.md:280`.

### Low
**A5. Diagnostic gain and score statistics depend on batch padding.** (`steering.py:127,155-163`)
- **Problem:** The means include pad positions. Post-EOS pad tokens count as real, because the mask is `None` during decode.
- **Failure scenario:** The same prompt reports `gain_mean` 0.064 instead of 0.150 when batched next to a long prompt.
- **Smallest fix:** Average over real positions only, and carry an EOS mask through decode.

**A6. The data-rule bypass through `--name`.** (`run.py:264`; `judge.py:43-49`)
- **Problem:** `assert_private` validates only the directory. An absolute or `../..` value of `--name` makes pathlib discard the private dir.
- **Failure scenario:** `--name /…/limbic-llm/results/alignment/a6` writes raw harmful completions inside the repo.
- **Smallest fix:** Run `assert_private` on the final resolved file path, or reject a `--name` that contains a path separator.

**A7. `--direction-from` does not free the first model.** (`run.py:189-198`)
- **Problem:** `dblocks` keeps the layers referenced. The path is skipped under `--tiny`, so no test covers it, and the two models' layer counts are never checked against each other.
- **Failure scenario:** Two 7-8B bf16 stacks on one 24 GB GPU run out of memory. A layer-count mismatch silently compares different relative depths.
- **Smallest fix:** `del dm, dblocks, Xd; gc.collect()`, and assert the two models' layer counts and hidden sizes match.

**A8. Missing `position_ids` on left-padded passes.** (`probe.py:35`, `run.py:127`)
- **Problem:** Absolute-position models (GPT-2, OPT) shift positions with padding. The tiny GPT-2 showed a max |diff| of 0.084 at the tap layer. Qwen3 and other RoPE models are unaffected.
- **Smallest fix:** Pass `position_ids = (mask.cumsum(-1) - 1).clamp(min=0)`.

**Refuted:** none in this track.

**Strengths:**
- The masked leaky-integral and EMA kernels handle left padding and carried state correctly.
- `probe.auroc` matches a brute-force AUROC to 6 decimals on tie-heavy data.
- The results JSON carries labels and scores only (`tests/test_alignment.py:109-122`).

**Verdict: sound with fixes.**
The controller kernels and the data rule are solid. Close the `--name` gap (A6).
A1 makes the A6 capability metric vacuous for three arms and must be fixed before A6 runs. A2 and A3 can distort the graded-vs-conditional and selectivity conclusions.

---

## 2. CoT / gist (R1-R5, `steering.py` gist mode)

### High
**C1. The gist direction is dominated by the attention-sink token and the stream mean, not by "the current thought".** (`steering.py:139-153`; claims at `steering.py:25-28` and `NOTEBOOK.md:300-304`; weak test at `tests/test_alignment.py:77-84`)
- **Problem:** Both EMAs start at zero with no bias correction, and position 0 is not excluded. For constant input, m−b = (β_s^{t+1} − β^{t+1})·h, which is 0.37h at t=200 with the defaults. The alignment lens found the same issue (its F2).
- **Measured on CPU:** On Qwen3-1.7B-Base, block 14, ‖h₀‖ was 13587 against a median of 167. cos(gist, h₀) was 0.998 at t=5 and 0.816 at t=20. cos(gist, sequence mean) was about 0.7 at t=50-186.
- **Failure scenario:** R2 static gist injects at every position, including the prefill (`steering.py:117-118`). The effect is roughly "add the sink or mean direction, rescale h by (1+g)", which confounds every comparison against the gist baseline. The unit test asserts only `delta[0] > 0` (`:84`), even though the background axis carries 0.63 of the unit direction.
- **Smallest fix:**
  - Exclude the first k tokens (the sink) from both EMAs.
  - Initialise the EMAs from the first included token, or subtract a fixed precomputed corpus mean.
  - Make the test assert that the background component is near zero.

  Bias correction alone does not fix it: the corrected gist points along −h₀ for hundreds of tokens.

**C2. The R2/R4 design cannot attribute a shorter CoT to the gist.** (`NOTEBOOK.md:328,330,333`; `REGISTER.md:133`)
- **Problem:**
  - There is no matched-norm random-direction, mean-direction or sign-flipped-gist control. Only S1 has a random control (`REGISTER.md:105`).
  - There is no sampling or seed protocol, no non-inferiority margin and no accounting for traces that hit max tokens.
  - `/no_think` is not a valid baseline for R1-Distill-Qwen.
- **Failure scenario:** Any perturbation at g·‖h‖ can raise the end-of-think probability, or cause repetition that is truncated and scored wrong. With 60 AIME problems and greedy decoding (the SE is about 6 pp at 30% accuracy), "equal accuracy" cannot be separated from a 5-10 pp loss. The reasoning runner is not built yet (`:333`), so this is a protocol gap, not broken code.
- **Smallest fix:**
  - Add the random, mean and anti-gist arms at matched norm.
  - Use k≥4 samples at T≈0.6 with fixed seeds.
  - Pre-register a non-inferiority margin.
  - Report the truncation rate.
  - Restrict `/no_think` to Qwen3.

### Medium
**C3. Graded gist is silently a no-op, and "R1-R4 can reuse steering.py" is wrong for R3.** (`steering.py:59-63,76,125-126`; `NOTEBOOK.md:329,333`)
- **Problem:** With no probe, `b = -30` gives g = 0. No gain source exists for agreement between the gist and the current token.
- **Failure scenario:** `SteeringController(None, None, mode="graded", direction_mode="gist", d_model=D)` gives max|out−h| = 0.0, and R3 reports a false null.
- **Smallest fix:**
  - Raise an error in the constructor for graded mode without a probe.
  - Mark R3 as needing a new gain law in `NOTEBOOK.md:333`.

**C4. STU-PID, the R3 baseline, was withdrawn on 28 Sep 2026, and its quoted figures are stale.** (`NOTEBOOK.md:306,318,329`; `REGISTER.md:134`)
- **Problem:**
  - The notebook's "81→87%, 32% fewer tokens" matches the v1-era abstract. The latest substantive version reports 85.7→89.6% and −23% on a GSM8K subset, as "a small-scale proof of concept".
  - The withdrawal notice defers to arXiv 2510.04309, which integrates across layers, not across decode steps.
  - The decision lens found the same issue (its F4).
- **Smallest fix:**
  - Mark STU-PID as withdrawn and cite its current figures.
  - Use 2510.04309 as the published feedback baseline.
  - Reimplement decode-time PID as an R3 ablation.

**C5. The novelty claim misses ERSS, and the over-commitment risk misses STARS.** (`NOTEBOOK.md:308-323`)
- **Problem:**
  - ERSS (arXiv 2606.06188, Jun 2026) already steers with the current generation's own hidden states: a high-norm cache, cosine retrieval, h_t + λh*.
  - STARS (arXiv 2601.22484) documents "inertia of direction" as a failure mode. That directly supports the risk named at `:323`.
- **Smallest fix:**
  - Cite both papers.
  - Narrow the claim at `:323` to the EMA form plus negative-feedback regulation.

### Low
**C6. The "open loop, never feeds back" claim holds only within one forward pass.** (`steering.py:27-30`; `NOTEBOOK.md:304`)
- **Problem:** During generation, the injection changes the sampled token, and that token sets every later h_t the EMAs read.
- **Failure scenario:** A repetition attractor at high gain goes undetected. S1 (`REGISTER.md:105`) does not measure long free-running repetition.
- **Smallest fix:**
  - Soften the docstring.
  - Add a repetition or distinct-n metric on long gist generations.

**C7. R1's thinking direction is confounded by template tokens.** (`NOTEBOOK.md:327`)
- **Problem:** With `enable_thinking=False`, Qwen3 appends `<think>\n\n</think>\n\n`, so the two conditions differ in their last tokens and positions.
- **Failure scenario:** The difference-of-means encodes "after an empty think block" versus "assistant start". Injecting it just opens `<think>`.
- **Smallest fix:** Take the difference at matched content positions, for example the user-prompt tokens or the generated response tokens, not at the last template token.

**C8. DSEM is slightly mis-described.** (`NOTEBOOK.md:319`)
- **Problem:** DSEM's memory holds steering vectors from alignment demonstrations, not past reasoning runs.
- **Smallest fix:** Correct the description.

**Refuted:** The ThinkEdit figures at `NOTEBOOK.md:315` (~4% of heads, +6.39%) are correct; the reviewer's figures came from stale HF metadata. The claim that R5 overstates its novelty relative to DSEM was also refuted, because `:331` already frames R5 relative to STIR and DSEM.

**Strengths:**
- The gist EMAs are causal, computed in float32, and carry state across KV-cached decode calls.
- The gist reads the pre-injection stream within a pass, and pads are zeroed out of both averages.
- The proposal names over-commitment as the main risk and plans flip analysis, a "wait"-rate check, a budget-truncation baseline and tokens-to-correct (`NOTEBOOK.md:323,328,330`).

**Verdict: needs rework.**
The gist signal as implemented is mostly sink/mean amplification, so R2/R3 would test the wrong thing (C1).
R2 lacks the controls and the sampling protocol needed to credit any effect to the gist (C2). R3 has no gain law and no live published baseline (C3, C4).

---

## 3. CA1 decision track (J1-J5), references and register

### High
**D1. J1's hypothesis, "at equal AUROC, better calibration gives higher selectivity", is ill-posed for this controller.** (`NOTEBOOK.md:355,365`; `REGISTER.md:144`; `steering.py:108-126`; `run.py:146-148`; `probe.py:59`)
- **Problem:** The gain is cap·tanh(κ·E), where E is the leaky sum of the raw score s (steady state ≈ 10s). At a fixed cap and κ, selectivity depends on where benign and borderline scores sit on that curve, not on ECE. The baseline probe is already trained with BCE, and temperature scaling is monotone, so AUROC is equal by construction.
- **Failure scenario:** If the probe is overconfident, T>1 raises benign scores. The simulated final gain is 0.030 at s=0.02 and 0.138 at s=0.1, so the "better-calibrated" arm lowers selectivity. J1 would then "refute" a hypothesis that was never well posed.
- **Smallest fix:** Compare the sources at a matched operating point, by tuning κ or the threshold per source to a fixed harmful-refusal rate and comparing borderline refusal. Alternatively, drive the gain from a thresholded decision instead of a raw probability. Reword `:365` accordingly.

### Medium
**D2. J1 measures ECE where the probe was trained, not on the signal that drives the gain.** See A2. ECE at the prompt position says nothing about per-token s_t on generated text (`probe.py:29`; `run.py:206`; `NOTEBOOK.md:365`).
- **Smallest fix:** Same as A2, plus report ECE at the positions the controller actually uses.

**D3. J3's motivation is misstated, it confounds calibration with warm-up, and it has no calibration metric.** (`NOTEBOOK.md:355,367`; `RESULTS.md:14,17,18,24`; `REGISTER.md:15`; `limbic_llm/models.py:239-241`)
- **Problem:**
  - (a) CA1 + novelty-lr alone was near-neutral (0.440 vs 0.483). The harm came from stacking it with DG writes (0.172, still rising at 3k steps).
  - (b) "proper scoring, warm-up" bundles the fix that `RESULTS.md:24` already names as untested.
  - (c) The novelty target is `sigmoid(z)`, a regression target with no binary event, so ECE is undefined for it.
- **Failure scenario:** Recall improves with warm-up and is credited to calibration. Alternatively, J3 runs on `titans_seq_ca1`, where there is nothing to fix.
- **Smallest fix:**
  - Correct `:355`.
  - Run warm-up alone as its own arm.
  - Define a calibration measure for the novelty target, either a binarised event (loss above a quantile) or a reliability curve of predicted against realised z.

### Low
**D4. J1 is labelled a "cheap add-on", but two of its four arms cannot plug in.** (`REGISTER.md:144`; `NOTEBOOK.md:362-363`; `steering.py:73-76,90-93`; `run.py:229-230`)
- **Problem:** `score()` is hard-coded to the linear probe. The LAYA-style and guard-model arms give one score per text, so they need either a new score interface with per-step re-encoding or a downgrade to a prompt-level, conditional-only decision. LAYA also ships over-confident with no fitted temperatures.
- **Smallest fix:**
  - Add an external score callback to `SteeringController`, or declare those arms conditional-only.
  - Temperature-fit LAYA before comparing.
  - Drop the "cheap" label.

**Refuted:** The claim that the README test count is wrong. 114 (`README.md:73,245`) is correct for the committed tree; the extra 8 tests come from the untracked `tests/test_evolve_w1.py`.

**Strengths:**
- All 10 new bibliography entries (`REFERENCES.md:130-139`, `docs/references.bib:671-764`) resolve to real papers with correct authors, years, IDs and venues. The quoted results for SEAL, ASC, Manifold Steering, CoT-Valve, ThinkEdit, STIR, RISER and LOTUS match their abstracts. STU-PID is the exception (C4).
- The JEV and LAYA descriptions are accurate and hedged, including LAYA's Khmer failure (0.000 accuracy at 0.952 confidence, `NOTEBOOK.md:355`) and the flag that jevai.net is unofficial (`REFERENCES.md:147`).
- J5's Ouro baseline (static exit, hidden-state-difference threshold, learned Q-exit gate) is described correctly.

**Verdict: sound with fixes (references and register); needs rework (J1 and J3).**
The references and register entries are accurate apart from STU-PID.
J1 needs a matched-operating-point comparison and generated-token calibration before its hypothesis can be tested. J3 needs a corrected premise, a separate warm-up arm and a defined calibration measure.