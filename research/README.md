# Research records

How the project keeps track of what it has found, what it plans, and who is doing what. The scripts are described in [`experiments/README.md`](../experiments/README.md) and the raw output in [`results/README.md`](../results/README.md).

## What each record is for

| Record | Purpose | How it changes |
|---|---|---|
| [`REGISTER.md`](REGISTER.md) | **Current state.** Every result and conclusion with its status, numbers and evidence file; every correction; decisions; open review findings; the backlog of planned experiments | Edited in place whenever a result lands or a status changes |
| [`NOTEBOOK.md`](NOTEBOOK.md) | **Dated log.** What was tried, why, the scores, surprises and the proposed next step, one entry per event (a round, a proposal, a review) | Append only. Old entries are not rewritten: a correction is a bracketed note, "[Correction, 2 Nov 2026: …]", and ID renames are applied |
| [`plans/`](plans/) | Plans written before a round: arms, commands, pass criteria and the anti-gaming rules E1–E9 ([`round0_plan.md`](plans/round0_plan.md)) | Fixed once the round starts; outcomes go to the register and notebook |
| [`reviews/`](reviews/) | Dated review reports, kept exactly as written. Cite a finding as `RV<n>.<ID>`; RV1 is [`2026-11-02_alignment_cot_ca1.md`](reviews/2026-11-02_alignment_cot_ca1.md), whose `file:line` anchors refer to commits `73259fc` and `241a13c` | Never edited. Open findings are tracked in the register's RV table |
| [`HANDOFF.md`](HANDOFF.md) | Coordination between sessions: branches, file ownership, layout, data rules, warnings | Updated when any of those change. Not a results document |

Outside this folder: [`RESULTS.md`](../RESULTS.md) is the stable write-up of results, [`results/`](../results/) holds the raw files, and [`docs/genesis.md`](../docs/genesis.md) is the history of the ideas.

## IDs

| Prefix | Meaning | Where |
|---|---|---|
| M, T, P | Results: memory systems, temporal loop, pretrained models | REGISTER §1 |
| N | Steering spot checks from a notebook | REGISTER §1 |
| B | Corrections and fixed bugs | REGISTER §1 |
| DEC | Decisions, with date and approver | REGISTER |
| RV1.A1 … | Review findings, by report | REGISTER (open findings); the report itself |
| g0–g7, g2b, g5h | Round-0 arms (g0–g7 are the flat and exp098 arms of the round-0 plan), and later runs of the same kind | REGISTER §4.1, round-0 plan |
| C1–C6 | Round-0 diagnostics that need code | REGISTER §4.1, round-0 plan |
| D1–D5 | Round-0 diagnoses | NOTEBOOK, "Round 0" |
| DR1–DR5 | Findings of the design review of 2 Nov | NOTEBOOK |
| E1–E9 | Rules a candidate must pass | round-0 plan §3 |
| W, S, A, R, J, O | Backlog: side-channel candidates and design gaps, feedback stability, alignment, reasoning, CA1 decision models, other | REGISTER §4.2–4.7 |

A letter suffix marks a variant or sub-finding (P4a, C1a, W1b). A dot marks an arm or a hypothesis within an item (W1b.a, A6.H1, J3.a). Priorities are High, Med and Low.

## Adding a result

1. **Run.** The script writes `results/<track>/<name>.json`; capture its console output as a `.log` of the same stem (`2>&1 | tee results/<track>/<name>.log`). Scripts default to the right folder; evolve runs need `--out results/evolve/round<N>/…`. Commit the JSON and log together once the run has finished; logs of runs in progress are not committed (B8).
2. **Register.** Add or update the row: claim, numbers, evidence path written from the repository root (for example `results/evolve/round1/g1_w1_tied.json`), seeds and status. Mark the backlog item Done with the result ID.
3. **Notebook.** A dated entry: setup, scores, what it shows, surprises, next step.
4. **RESULTS.md**, once the result is stable: final for its round and unlikely to change with more steps or seeds. Cite the register ID.
5. **Supersessions.** If the run replaces or reproduces an earlier file, say so in the old register row and in the Provenance section of `results/README.md`. Never overwrite a result file silently; if a rerun must replace one in place, record it there.

## Adding an experiment

1. **Register backlog row**: ID, experiment, hypothesis, metric, priority, status, script (path, or "not built"), and what it depends on or is blocked by.
2. **Notebook entry** if it needs motivation, prior art or a design: a dated proposal.
3. **Script** in `experiments/` (or `experiments/<track>/`), writing to `results/<track>/`. Add it to [`experiments/README.md`](../experiments/README.md) with its register IDs and status.
4. **New track**: a new folder under `results/`, described in `results/README.md`. Do not name it after a git-ignored folder (`runs`, `checkpoints`, `build`, `dist`, `env`, `wandb`).

## Corrections and decisions

- **A wrong number or claim**: fix the register row in place; if a bug caused it, add a B row. Add a bracketed correction note to the notebook entry that reported it. Do not edit reviews.
- **A decision** that changes what is allowed (for example DEC1, a fixed-size state for the pretrained track): a DEC row with the date and who approved it, and a notebook entry.

## Branches and sessions

- `main`: merged work.
- `feat/round1-side-memory`: round 1 of the evolution search and the reorganisation of these records. The GPU session works here.
- `feat/temporal-loop`: merged into `main` as PR #2 and kept.

Two kinds of Claude session work on the repository: the GPU session on geof (2× RTX 5090), which runs the pretrained and evolve experiments, and the coordinating session, which writes the register, notebook, RESULTS and docs. Each file has one owner (with the exceptions listed in HANDOFF). The ownership list, and the rule to pull before writing results, are in [`HANDOFF.md`](HANDOFF.md).
