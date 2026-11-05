# Round 11, Stage A2: combinations and decompositions (revision 2, for Daniel's approval, 5 Nov 2026)

Revision 1 (a nested chain C1 → C2 → C3) was reviewed adversarially: 30 findings, 28 standing. This revision takes in the review and Daniel's directions:
- "there is promise in all of the top 6 techniques";
- untested replay variants;
- the hysteresis and Schmitt gates;
- compression rescue arms.

## What the review changed

1. **Seed 4 measures rescue, not mechanism.** I1's seed-4 run failed (random 2048 × 5: 0.56, against 0.83–0.87 on seeds 0–3). Almost any arm that trains cleanly recovers it: A7, a compression control, gains +0.34. Against I1's learned seeds 0, 2 and 3 (round-10 Stage-B files, same sequences):
   - A11, A1b and A15 are still better (+0.04 to +0.12);
   - A12 and A1 are not.
2. **The references are on old code.** The I1 references were trained before the round-11 alignment fix (`2d20975`), so every arm-vs-I1 difference also contains a code-version difference.
3. **A1b, and A1, lose updates heavily.** New-code exact at 2048, paired with Stage-0's I1 linear cells (n = 96): −0.42 to −0.48 on seed 1 and −0.27 to −0.33 on seed 4. Stale rate 0.48–0.66 against 0.11–0.26. A1b's length gain on seed 1 matches what switching off the age penalty gives at evaluation (P60), and the ordinal rank has never been isolated.
4. **A11's lexical key is near zero after training** (weight 0.21–0.26 against 11–12). A11 also moves the alignment code to block 3, which there bypasses the TimeChannel clock, and makes the gate and key read two blocks. Revision 1's "A11n" isolated only the near-zero part.
5. **The per-sequence paired interval compares two checkpoints, not two recipes.** I1 against itself on different seeds "differs" on up to 13 of 20 cells. A recipe-level verdict needs seed-level variance.
6. **Winner's curse.** The components were selected on seeds 1 and 4 with eval seed 2024. Comparing combinations with them on the same draws is biased against the combinations.
7. **Removing the linear age penalty needs a DEC** (DEC9 item 6).
8. **Cost:** an evaluation takes about 43 minutes per checkpoint, not 30.

## Phase 0: evaluation-only, no training (about 1.5 GPU hours)

| Item | What | Answers |
|---|---|---|
| E-lex | A11 seeds 1 and 4 with `--w1s-param loop.log_beta_lex=-30`, on the deciding cells | Does the trained A11 use its lexical key? |
| E-ord | A1b seeds 1 and 4 with `--w1s-param loop.ord_E=0`, on the deciding cells and updates | Does the ordinal rank do anything beyond "no age penalty"? |
| E-rescue | Every seed-4 arm paired against I1's learned seeds 0, 2 and 3 (round-10 Stage-B files; grid, jitter, held-out) | Separates rescue from mechanism; no GPU |
| E-upd | The I1 update references (in the queue), then a per-sequence stale rate | Updates scored properly |

## Gate track (Daniel, 5 Nov: "add in the hysteresis and Schmitt gate fix tests as well. Point gates seem fragile")

The point gate's failures (I1 seed 4, A5, A6, A14 collapses) look like facts that are never written or never kept. The hysteresis and Schmitt gates have never had a single-change test: round 10 ran them only inside combinations with gate centring (P50, P52, P57).

- **Code fixes and tests (CPU, before any gate GPU time).**
  - E1 bitwise (fp32 and bf16-rounded) and stream = parallel for `--w1s-gate hyst` and `schmitt` with every round-11 option (time features, recency forms, keep, code, native keys, A15's target).
  - The initial firing density and latch on-fraction with `--no-w1s-centre`. `gate_bias` is calibrated for a centred, unit-variance drive; an uncentred drive can start the latch saturated (always on, one onset per sequence: P35's starved-gate failure).
  - If it does, fix the calibration (centre only the initial bias estimate, or calibrate b on the drive's running statistics) without bringing back the filler-adaptive centring that failed in P50. Regression-test the fix.
- **Phase 0 (evaluation only).** A diagnostic script reports each untrained gate's onset density, on-fraction and onset-in-fact rate on real Qwen3 block-21 states, natural and random filler, for point, hyst and schmitt, with and without the fix.
- **Phase 2.** G-hyst and G-schmitt (I1 with only the gate swapped, no centring, fixed calibration) on seeds 1, 2 and 4.
- **Phase 3.** Gate type is a factor (point against the better of hyst and schmitt from Phase 2).
- **Reading.** A "gate robustness" group for every arm: fact coverage, onset-in-fact and gate AUC on natural against random filler, with jitter and paraphrased queries; plus the training step at which selection is learned (late-drop step). A gate is judged more robust if coverage holds across fillers and learning is earlier and less seed-dependent.

## Phase 1: noise floor and references under the current code (5 runs)

Retrain I1, BASE flags unchanged, under the round-11 code: seed 1 once, seed 2 once, seed 4 twice (the first two repeat with a different torch seed; data order fixed). Plus I1 seed 3 if the budget allows.
- These become the I1 references (round-10 references stay as a second column).
- They measure training noise at a fixed recipe: run-to-run spread at the same data seed, and whether seed 4's failure reproduces.

## Phase 2: single-change decompositions (on seeds 1, 2 and 4)

| Arm | Change | Question |
|---|---|---|
| A0 | `--w1s-recency-form none` | Is A1b's gain just "no age penalty"? |
| A1b+lin | ordinal time **with** the linear penalty kept | Can ordinal time keep updates? |
| A11a | alignment code on block 3 only (new flag `--w1s-align-stream 3`; gate and key on block 21) | The block-3 alignment code alone (with the clock removed from it) |
| A11k | gate and key on blocks 3 + 21, alignment on block 21 | The two-block gate and key alone |
| G-hyst | the hysteresis gate (no centring) | First single-change test of the gate |
| G-schmitt | the Schmitt gate (no centring) | First single-change test of the gate |
| R-ptr | pointer replay (new code): after the first match, step through the slot instead of re-matching | Robust to codes with repeated digits? |
| R-learned | `--w1s-align learned`: matching in the learned-key space | Alternative match code |
| C5r | A5 started from G0.5's fitted 64-d code (new `--init-from`) | Compression that trains? |
| C6v | fixed value-norm keep, 8 of 24 (`--w1s-keep 8 --w1s-keep-by value`) | Compression with nothing to learn |

That is 10 arms on 3 seeds = 30 runs. The single arms A11, A1b, A12, A15, A1 and A7 also run on seed 2 (6 runs), so every comparison has an unselected seed.

## Phase 3: factorial over the top six

Factors, each on or off against I1:
- K: A11's block-3 taps and lexical key;
- O: the ordinal rank, with the linear penalty kept unless P is off;
- R: rotary time (new code allows rotary and ordinal together);
- P: the linear age penalty (on = I1);
- S: A12's sharp alignment start;
- W: A7's 12-token windows;
- T: A15's teacher admission target.

The factorial uses a 2^(7−3) fraction of resolution IV: 16 combinations. Main effects are clear of all pairwise interactions. Seeds 1, 2 and 4 give 48 runs.

The factor set may be trimmed by Phase 0–2 results (e.g. drop O if E-ord shows the rank does nothing; split K per A11a/A11k). Phase 3 starts only after Phases 0–2 and Daniel's go-ahead on the final factor list.

## How results are read (pre-registered; DEC11, refined by the review)

**Groups.** The statistic is the mean of per-cell paired differences. Cells sharing sequence rows (held-out, nat512p3, para1–4; updnow and updnew) are averaged row by row.
- **Grid:** 18 DEC2 cells.
- **Names:** held-out.
- **Query:** para1, 2 and 4. para3 is excluded (full attention 0.44 < 0.8). Jitter is reported alongside, with full attention at 0.57.
- **Updates:** new-exact and stale rate on update same, now and new at 512 and 2048, and update-k 2 and 3.
- **Robustness:** reported only (n = 96).

**Two levels of interval:**
- **Checkpoint level:** the per-sequence paired interval, as now, describes two trained checkpoints.
- **Recipe level:** d̄ ± t·s_seed·√(1/k_A + 1/k_B), where s_seed is the between-run SD of the group statistic for a fixed recipe (from Phase 1 and I1's learned seeds 0–3). A recipe is better or worse only where this interval clears 0 and the sign holds on every seed.

**Seed 4** is reported twice: against I1 seed 4 (rescue, a reliability fact) and against I1's learned seeds (mechanism). Only the second and the other seeds count as mechanism evidence.

**Factorial.** Main effects and their intervals come from the 16 runs per seed (8 on, 8 off for every factor), averaged over seeds. That is the main evidence for which techniques to keep.

**Stage-B pick, a default for Daniel:**
- Rank recipes by the recipe-level mean difference against I1 on Grid + Names + Query.
- The top recipe goes to Stage B. If it is worse on Updates, the best recipe that is not worse on Updates goes too.
- Ties go to the simpler recipe.
- A recipe without the linear age penalty needs a DEC (DEC9 item 6).

## Code needed before Phases 2–3 (tested and reviewed like the rest)

1. `--w1s-time-feat` accepts rotary and ordinal together.
2. Pointer replay (`--w1s-replay pointer`).
3. `--w1s-align-stream` decouples the alignment code from the lowest tap.
4. `--init-from PT` loads a partial state dict before training (C5r).
5. Analysis: arm against arm; groups; recipe-level intervals; the rescue and mechanism columns; a per-sequence stale rate; a stronger pairing check.
6. Gate: hyst/schmitt E1 and parity tests with the round-11 options; initial-density check without centring, with a calibration fix if needed; a gate diagnostic script for real states (Phase 0).

## Budget and order

Measured: training 13–25 minutes (A15-based about 25), evaluation about 43 minutes per checkpoint with the full cell set.

| Phase | Runs | GPU hours on 2 GPUs |
|---|---|---|
| 0 | eval only | about 1.5 |
| 1 | 5 | about 2.5 |
| 2 | 36 | about 18 |
| 3 | 48 | about 24 (about 15 with a reduced evaluation set: grid 8 cells, names, query, updates) |

**Order:** Phase 0 now, inside the running queue. Phases 1–2 after Stage A and the tap sweep (or before the tap sweep, at Daniel's choice). Phase 3 after a checkpoint with Daniel. Stage B (10 seeds) follows.
