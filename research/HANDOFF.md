# Handoff: coordination between sessions

This note coordinates the Claude sessions working on this repository: branches, who owns which files, where results go, data rules, and warnings. It holds no results. Results are in [`REGISTER.md`](REGISTER.md) and [`RESULTS.md`](../RESULTS.md), and the raw files are described in [`results/README.md`](../results/README.md). How the records fit together is in [`README.md`](README.md).

## Sessions and branches (2 Nov 2026)

- **GPU session** on geof (2× RTX 5090). Runs `experiments/pretrained_loop.py` and `experiments/evolve/run.py`.
- **Coordinating session.** Combines results into the register, the notebook, RESULTS.md and the docs.

At Daniel's request, PR #2 (`feat/temporal-loop` → `main`) was merged. Since then the GPU session works on **`feat/round1-side-memory`**, branched from `main` after the merge. It pushes round results, code under `experiments/evolve/`, and updates to this note there. PR #3 (`feat/qwen3-actadd-notebook`, the Qwen3 activation-additions notebook) was merged into this branch (`82b8fca`). `feat/temporal-loop` is kept, not deleted. Pull `feat/round1-side-memory` to see round-1 results.

## File ownership

| Session | Files |
|---|---|
| All sessions | This note and any other handoff document (Daniel, 2 Nov). Pull first and edit only the parts you are updating |
| GPU session | `results/pretrained/`, `results/evolve/round*/` (including the local `.pt` checkpoints), `experiments/evolve/` |
| Coordinating session | Everything else: `RESULTS.md`, `README.md`, `docs/`, `research/REGISTER.md`, `research/NOTEBOOK.md`, `research/plans/`, `research/reviews/`, `research/README.md`, `results/synthetic/`, `results/README.md`, `experiments/*.py`, `experiments/alignment/`, `limbic_llm/`, `tests/` |

So far the GPU session has also written its round checkpoints into the coordinating session's files: the round-1 NOTEBOOK entry and the P7–P12 register rows (`494ab17`), and the RV1 review report (`9e3f507`). When doing that, pull first and touch only those entries.

`results/alignment/` is written on the GPU machine by `experiments/alignment/run.py`; the session that runs the alignment track owns its output files.

## Layout (since `1b9dc42`)

```
research/            REGISTER.md, NOTEBOOK.md, HANDOFF.md, README.md, plans/, reviews/
results/synthetic/   CPU runs (assoc_recall*, continual, gdn_capacity, loop_systems, temporal_loop_*)
results/pretrained/  pretrained_*.json|log
results/evolve/round0/   round-0 arms g0–g7 (planned, empty)
results/evolve/round1/   round-1 runs (was results/evolve/gen01/; files keep their g1_* names)
results/evolve/round2/   round-2 runs (W1b, in progress; r2_* names)
results/alignment/   alignment track output (labels and scores only)
```

The scripts now default to this layout: the CPU scripts write to `results/synthetic/`, `pretrained_loop.py` and `evolve/run.py` to `results/pretrained/pretrained_loop_<task>.json` unless `--out` is given, and `alignment/run.py` to `results/alignment/<name>.json`. Evolve runs must pass `--out results/evolve/round<N>/<name>.json`.

**Pull `feat/round1-side-memory` before writing any result.** A checkout from before `1b9dc42` writes to the old flat paths (`results/*.json`, `results/evolve/gen01/`), and `gdn_capacity.py` and `temporal_loop.py` would not find their existing outputs to resume from. The old → new path map is in `results/README.md` (Provenance); `args.out` inside the existing JSONs still shows the old paths. The round-1 `.pt` files are git-ignored: they exist only on geof, already moved to `results/evolve/round1/`.

## Data rules

- **Alignment track.** Commit only judge scores, refusal labels and aggregate metrics. Raw completions to harmful prompts go to `--private-dir`, outside the repository, and are never committed. `--name` may hold only letters, digits, `.`, `_` and `-`, and each private file is checked after resolution (RV1.A6, fixed).
- **Result files.** Do not edit JSON or logs by hand. Commit logs only for finished runs (B8). A rerun that replaces a file in place must be recorded in the Provenance section of `results/README.md`.
- **Checkpoints** (`*.pt`) are not committed.

## Side-memory status for other sessions (GPU session, 4 Nov 2026)

- **Current best: the W1s episodic-slot memory, minimal form I1** (`experiments/evolve/w1s_slots.py`; flags `--w1s-gate point --no-w1s-centre --w1s-pre 0 --w1s-key learned --w1s-taps ''`).
  - On seeds 1–2, which no earlier recipe solved, exact is 0.86–0.98 in every cell, natural and random filler, 512 and 2048 tokens (register P51).
  - The full combination fails because gate centring adapts to the filler (P50). Leave-one-out and isolation arms: P52. Reinstatement: P53. Stage 0: P49.
- **Where to look:**
  - Round 10 results: `results/evolve/round10/` (`r10_*`, `e10_*`; Stage 0 in `stage0/`).
  - Narrative: NOTEBOOK "Round 10 Stage A".
  - Decisions: DEC4–DEC7 (DEC5: no pre-fit directions).
  - Plan: the round-10 plan entry in the NOTEBOOK.
- **Stage B done (P54–P58):** I1 learns 3 of 5 seeds (seed 1 at 0.94) with grid means 0.93–0.96 on the learned seeds, fails on held-out names (0.58–0.74), and A2, I1K and RI do not beat it. Round 11 (runbook revision 2) follows.
- **New tools:**
  - `experiments/evolve/taps.py` (`TapLM`: extra taps, reinstatement);
  - `eval_by_query.py` flags `--jitter`, `--update`, `--heldout-names` and `--w1s-ctrl`;
  - diagnostics `diag_novelty.py`, `diag_surprise.py`, `diag_declarative.py` and `diag_varlen.py`.

## Round 11 is ready for another session (4 Nov 2026)

**Plain-language status of the side-memory track:** [`side_memory_summary.md`](side_memory_summary.md). It covers what works, what doesn't, why the 64-token window, and production relevance. Read it first.

Daniel approved the round-11 plan (DEC9) and asked for it to be executed by another session once round 10 Stage B completes. **Start from [`plans/round11_runbook.md`](plans/round11_runbook.md).** It lists:
- the preconditions (Stage B committed, the base recipe chosen);
- the environment (filler text and scratch references in `~/limbic-data` on geof);
- the code changes and tests;
- the Stage-0 no-training gates;
- the Stage-A arms with exact flags;
- the evaluations, the pre-registered rules, the record-keeping, and Daniel's checkpoints.

The design and its evidence are in [`plans/round11_plan.md`](plans/round11_plan.md).

## Side-memory review RV4 (4 Nov 2026)

A Claude Science session reviewed rounds 1–10 of the side-memory track against the literature, at Daniel's request: [`reviews/2026-11-04_side_memory/README.md`](reviews/2026-11-04_side_memory/README.md); open findings RV4.E1–E12 in the register. **Before running round 11, read §6 of the review.** It recommends:
- closing the Stage-B gaps first: E8 and the CPU suite for I1, I1-superposed and I1-no-align, and eval-only paraphrase and repeated-digit cells;
- 10 seeds for finalists;
- putting the name-generalisation and training-signal arms (A10, A11, new A13–A15) ahead of compression and time;
- scoring updates against ground truth;
- a 128-token and a partially windowed hybrid configuration in Stage C;
- W16 on an equal-byte axis (the W1s state equals about 31 KV tokens).

It added 25 bibliography entries (16 new, 9 promoted from RV2's `new_refs.bib`) to `docs/references.bib` and REFERENCES.md.

## How to run a round

Follow [`evolution_protocol.md`](evolution_protocol.md): the round loop, controls, the candidate interface, `experiments/evolve/queue.sh` (GPU job queue) and `experiments/evolve/sync.sh` (rebase-safe push), and the pitfalls met in rounds 0–5.

## Review RV1 (fixed, 2 Nov)

All twenty findings of the review of the alignment, CoT and CA1 decision tracks are addressed (RV1 table of [`REGISTER.md`](REGISTER.md); NOTEBOOK, "RV1 fixes"; report: [`reviews/2026-11-02_alignment_cot_ca1.md`](reviews/2026-11-02_alignment_cot_ca1.md)). **A1, A2 and A6 are ready for geof.** What changed for whoever runs them:
- Perplexity now prefills `--ppl-prefix` tokens and teacher-forces the rest, so every arm is measured; `ppl_gain_mean` shows the arm acted.
- A refusal counts only in a non-degenerate response (`refusal_raw` keeps the old count).
- Every arm, including `none`, reports the probe's AUROC and ECE at the prompt and on generated tokens (`calibration`). Check the generated-token numbers before reading graded-vs-conditional differences.
- New arm `static_random` (matched-norm control). `--gains` and `--caps` sweep the gain and cap, and `operating_points` compares arms at a matched harmful-refusal rate (`--target-refusal`). For A6 sweep past 0.3 (N3).
- `--direction-from` frees the sibling before loading the target and exits if layer count or width differ.
- The gist direction skips the sink token and is bias-corrected, with `anti_gist`, `mean` and `random` controls; R2 and R3 still need a reasoning runner (protocols in the NOTEBOOK).

## Literature review RV2 (2 Nov)

A Claude Science session reviewed the four research tracks against the literature, at Daniel's request. It also reviewed the Samsonovich & Ascoli (2005) pathfinding model, audited the bibliography and reviewed the code. **Agents should start with [`reviews/2026-11-02_literature/HANDOFF.md`](reviews/2026-11-02_literature/HANDOFF.md)**; machine-readable verdicts are in `findings.json`. It went in as a PR from `docs/literature-review` into this branch. Points that affect running work:

- **GPU session, W1b.**
  - Round 2 changed only the write-side address. The review reads P11 and P13 as needing a *read-side* selector, a cue from the query span or a goal field (RV2.F1, L6).
  - W1b.a (gdn mode, P14) is confounded by **B12**: `CausalConv1d` starts as the identity, so gdn mode begins with no token shift.
- **Before writing up P2:** run the memory-free adapter, LoRA and state-offset controls (RV2.X3, L2).
- **CPU recall numbers** (M1, M4, M6, M7) sit at a binding-free ceiling of 0.476 (RV2.X1). Do not cite them as binding results.
- **Bibliography.** `docs/references.bib` is the audited version (160 entries, DOIs). `docs/src/bib_to_json.py` requires REFERENCES.md and the bib to share keys, so add a source to both. Further verified sources are in `reviews/2026-11-02_literature/new_refs.bib`.

## Warning: bf16 leaks

- **bf16 index arithmetic leaks the future.** `torch.arange(512, dtype=bfloat16)` has only 385 distinct values; positions 505–511 become [504, 506, 508, 508, 508, 510, 512]. Position masks such as `d >= 0` then let some answer positions see their own targets. This anti-causal leak, not long-range transfer, produced the bf16 all-windowed Qwen3.5 loss of 1.69 (round-0 D5, register B4). The kernels used to take their index dtype from the activations, so every bf16 use of `HippocampalLoop` or `TimeChannel` was exposed. `loop_kernel` and `decay_kernel` now build indices in float32 (`51aecae`; `limbic_llm/temporal_loop.py`, `test_bf16_stays_causal_over_long_sequences`), and `pretrained_loop.py` runs the side channels in float32 (`3c90bde`). **Any new script or candidate** must keep index arithmetic in float32 or integers (rule E5) and test causality on bf16 input (rule E1, as `tests/test_evolve_w1.py` does).
- **bf16 parameters do not train.** Side-channel Adam updates (about 1e-3) are smaller than bf16's spacing near 1 (2⁻⁷), so they round away (register B3). Every pretrained run before `3c90bde` was affected and has been redone; the superseded numbers are in `results/README.md` (Provenance). A final `results.gain` of exactly 0.2689 means the gain never trained.

## Steering notebook and new steering items (2 Nov)

`docs/notebooks/activation_additions_qwen3.ipynb` (PR #3) runs the alignment and CoT steering interactively on Qwen3-1.7B. Its config cell sets `STEER`, `VECTOR_SOURCE` and `INJECTION`, and it reuses `SteeringController`. The first spot checks are N1–N5 in the register (NOTEBOOK, "First steering runs on real models"). They added three backlog items that affect whoever runs A6 and R1:
- **S9.** The saturation knee depends on the direction. The thinking direction collapses at g = 0.3 at block 14, where the refusal direction is still coherent. Sweep the directions actually in use.
- **A8.** The gain law: tanh against squared and exponential schedules in the harm evidence. The A6 gain sweep should extend past 0.3, since the sibling direction first refuses there.
- **R6.** A repetition-regulated CoT gain.

A8 and R6 need new modes in `steering.py`, and neither is built yet. The notebook reads `summary()`'s new keys (`s_gen_mean` replaces `s_mean`). **Do not commit this notebook with outputs from the alignment preset** (data rule): clear them with `jupyter nbconvert --clear-output --inplace`.

## Earlier contents

Earlier versions of this note carried the result tables of the pretrained and round-1 runs. Every number from them is now in `results/README.md` (per-file tables), the register (P1–P12) or RESULTS.md (§4.7–4.8). The GPU session's review of the coordinating session's 2 Nov additions was applied (NOTEBOOK, "Corrections from the GPU session's review"); that review also checked that Zhao et al. 2025 (arXiv 2507.11878) and the four A6/A7 models exist. The full earlier text is in git history (`53caa77`).
