# Results

Raw output of every run, one folder per track. Each run writes a JSON (its arguments, curves and final scores); the `.log` next to it is its console output, captured by redirection (`2>&1 | tee`), since the scripts do not save stdout themselves. Interpretation lives elsewhere: [`RESULTS.md`](../RESULTS.md) for the write-up, and [`research/REGISTER.md`](../research/REGISTER.md) for every claim with its status and the file it rests on. Do not edit result files by hand. [`research/README.md`](../research/README.md) describes how a result gets from here into the register.

```
results/
  synthetic/        CPU, small models trained from scratch: memory systems (M rows) and temporal loop (T rows)
  pretrained/       GPU, side channels on frozen Qwen3-1.7B-Base and Qwen3.5-2B-Base (P1–P6)
  evolve/
    round0/         round-0 diagnostic arms g0–g7 (planned; not created yet)
    round1/         round-1 runs through experiments/evolve/run.py (P7–P12)
    round2/         round-2 runs (W1b), in progress
  alignment/        alignment track (A1, A2, A6); labels and scores only (empty so far)
```

## synthetic/

CPU runs, $d_\text{model} = 64$, seed 0 unless noted. Scripts are in `experiments/`; the command is recorded in each JSON's `args`.

| File | Script and arguments | RESULTS.md | Register |
|---|---|---|---|
| `assoc_recall` | `assoc_recall.py --steps 3000 --models all` | §1 | M1, M2 (`titans_hippo`) |
| `assoc_recall_ablation` | `assoc_recall.py --models titans_seq_dg,titans_seq_ca1` | §1 (the two ablation rows) | M2 |
| `assoc_recall_gdn` | `assoc_recall.py --models titans_gdn,titans_gdn_dg` | §3 (Titans with a delta-rule memory) | M6, M7 |
| `continual` | `continual.py --steps 600` | §2 | M3 |
| `gdn_capacity` | `gdn_capacity.py --pairs 8,16,32 --steps 1500` | §3 (capacity sweep) | M4, M5, M8 |
| `loop_systems` | `loop_systems.py` (untrained, 3 seeds) | §4.3 | T6, T7, T8 |
| `temporal_loop_all` | `temporal_loop.py --suite all --steps 1500` | §4.1, §4.2, §4.6 | T1, T2, T4; the open-loop recall against distance in §4.1 (`memory`: 1.00 / 0.61 / 0.17). Partly superseded, see Provenance |
| `temporal_loop_long` | `--models closed_osc_ca1,memory_time --steps 3000`, seed 0 | §4.1, §4.2 | T1, T2, T3, T5, T18, T19 |
| `temporal_loop_long_seed1` | `--models closed_osc_ca1`, seed 1 | §4.2 | T2, T3, T5. No `.log` (B8) |
| `temporal_loop_probe` | `--models closed_flat,closed_exp,closed_osc_ca1 --steps 3000` | §4.2, §4.4 | T4, T5, T11, T12, T19 |
| `temporal_loop_feedback` | `--models closed_osc,closed_osc_div,closed_osc_sub --steps 3000`, seed 0 | §4.3, §4.4 | T9, T10, T15, T21 |
| `temporal_loop_open` | `--models memory,closed_osc_sub --steps 3000`, seed 1 | §4.3, §4.4 | T9, T10, T14, T15 |
| `temporal_loop_ablations` | `temporal_loop.py --suite ablations --steps 3000` | §4.4, §4.5 | T11, T12, T13, T16 |
| `temporal_loop_recency` | `--task recent`, three closed loops | §4.4 (trained recency task) | T17 |

- `closed_osc_ca1` in `temporal_loop_long` and `temporal_loop_probe` is the same run: identical curves and rollouts (T19).
- `diverged_at` is `null` for every model in every `temporal_loop_*` file (T20).
- `temporal_loop_recency` records `args.models: "closed_osc_ca1"` but holds three models. It was assembled from several invocations.

## pretrained/

Written by `experiments/pretrained_loop.py` on geof (2× RTX 5090). Needle task: seq 512, batch 8, block window 64, 3000 steps, one stored pair, needle gap 501–502 tokens (`model.gap_range`). Default loop kernel `osc`, ρ 0.7, τ 16; "horizon" means `--kernel exp --rho 0.95 --tau 32`. 4,380,675 trainable side-channel parameters in every run. All files are float32 reruns (see Provenance). RESULTS.md §4.7.

Each cell is exact / token_acc / loss. "gain" is the loop's final sigmoid gain (initial 0.2689); "inj" is the injection ratio ‖inj‖/‖h‖ at evaluation.

| File | Windowed layers | Tap | Kernel | Windowed | Windowed + side | Full | gain | inj | Register |
|---|---|---|---|---|---|---|---|---|---|
| `pretrained_qwen3_needle` | 28/28 | 14 | osc | 0.00 / 0.106 / 3.069 | 0.00 / 0.287 / 1.842 | 1.00 / 1.000 / 0.011 | 0.2797 | 0.207 | P3 |
| `pretrained_qwen35_needle` | 6/24 attention, GDN intact | 12 | osc | 0.00 / 0.056 / 4.028 | 1.00 / 1.000 / 0.0000 | 1.00 / 1.000 / 0.034 | 0.2697 | 0.208 | P1, P2 |
| `pretrained_qwen35_needle_allwin` | 24/24 | 12 | osc | 0.00 / 0.130 / 3.039 | 0.00 / 0.289 / 1.843 | 1.00 / 1.000 / 0.034 | 0.2718 | 0.204 | P6 |
| `pretrained_qwen35_needle_layer22` | 6/24 attention, GDN intact | 22 | osc | 0.00 / 0.056 / 4.028 | 0.00 / 0.345 / 1.734 | 1.00 / 1.000 / 0.034 | 0.2933 | 0.218 | P2 |
| `pretrained_qwen3_needle_horizon` | 28/28 | 14 | horizon | 0.00 / 0.106 / 3.069 | 0.00 / 0.461 / 1.381 | 1.00 / 1.000 / 0.011 | 0.3264 | 0.233 | P4, P4a |
| `pretrained_qwen35_needle_allwin_horizon` | 24/24 | 12 | horizon | 0.00 / 0.130 / 3.039 | 0.023 / 0.550 / 1.239 | 1.00 / 1.000 / 0.034 | 0.2768 | 0.213 | P4, P4a. Superseded by P9 |

Language modelling: WikiText-103, seq 1024, block window 256, batch 4, 1000 steps, grad checkpointing, default kernel. Each cell is perplexity / loss. Gap closed = (windowed − side) / (windowed − full), on loss.

| File | Windowed layers | Train / eval tokens | Windowed | Windowed + side | Full | Gap closed | gain | inj | Register |
|---|---|---|---|---|---|---|---|---|---|
| `pretrained_qwen3_lm` | 28/28 | 1,155,524 / 116,893 (114 sequences) | 14.62 / 2.683 | 12.01 / 2.486 | 9.01 / 2.199 | 41% | 0.3681 | 0.254 | P5 |
| `pretrained_qwen35_lm` | 6/24 attention, GDN intact | 1,150,251 / 116,393 (113 sequences) | 16.28 / 2.790 | 11.22 / 2.417 | 9.83 / 2.286 | 74% | 0.3744 | 0.263 | P5 |

Notes from the training curves and logs:
- `pretrained_qwen3_needle`: training loss flat at about 1.84 for all 3000 steps (1.840 at the end).
- `pretrained_qwen35_needle_allwin`: flat at about 1.84 (1.831 at the end).
- `pretrained_qwen35_needle_layer22`: plateaus around 1.7–1.8.
- `pretrained_qwen3_needle_horizon`: 300: 1.844 / 0.25 → 600: 1.683 / 0.32 → 900: 1.372 / 0.47, then flat at 1.38 to step 3000.
- `pretrained_qwen35_needle_allwin_horizon`: at the floor (about 1.84) until step 2400, 1.36 at 2700, 1.395 at 3000. Exact 0.023 is 3 of 128 eval sequences.
- Kernel weights at the needle gap. Default `osc` (ρ 0.7, τ 16), w(Δ) = ρ^(Δ/τ−1)·[cos(2πΔ/τ)]₊: 1.0 at Δ = 16, 0.08 at 128, 0.005 at 256, about 4e-19 at 500, exactly 0 at 501–503. Horizon `exp` (ρ 0.95, τ 32), w(Δ) = ρ^(Δ/τ−1): 0.472 at 500, 0.470 at 503.
- Qwen3.5-2B-Base has 24 blocks; 3, 7, 11, 15, 19 and 23 are attention, the other 18 Gated DeltaNet. The side channels hook the output of the tap block, so with tap 12 the injection passes through blocks 13–23 (8 GDN, 3 attention); with tap 22 only block 23 (windowed attention) follows. [Correction, 2 Nov 2026: the handoff note had said 9 GDN layers follow tap 12 and that blocks 22–23 follow tap 22; it counted the tap block itself.]
- "k digits known" levels (five scored tokens, k digits exact, the rest at chance): k = 0 gives 1.842 / 0.28, k = 1 gives 1.382 / 0.46, k = 2 gives 0.921 / 0.64 (loss / token_acc).

## evolve/round1/

Round 1 of the evolution search (research/NOTEBOOK.md, "Round 1"). Written by `experiments/evolve/run.py`, which runs `pretrained_loop.main` unchanged and adds per-position scores (`results[cond]["per_pos"]`) and a `.pt` checkpoint of the trained side channels. W1 runs use `--candidate evolve.w1_gdn:W1Side`. Qwen3-1.7B-Base unless noted, every layer windowed, needle settings as in `pretrained/`. RESULTS.md §4.8.

Per-digit loss is for digits 1–4 (`per_pos[1:]`; `per_pos[0]` is the leading space). Chance is ln 10 = 2.30. Full attention is 1.00 / 0.011 on Qwen3 with one pair, 1.00 / 0.025 with two, and 1.00 / 0.034 on Qwen3.5.

| File | Side channels | Pairs | Steps | Exact | Loss | Digits 1–4 | gain | inj | Params | Register |
|---|---|---|---|---|---|---|---|---|---|---|
| `g1_qwen3_horizon_perpos` | O(T) loop, horizon kernel | 1 | 3000 | 0.00 | 1.381 | 0.001, 2.30, 2.30, 2.30 | 0.3264 | 0.233 | 4,380,675 | P7 |
| `g1_qwen3_horizon_pairs2` | same | 2 | 3000 | 0.00 | 1.809 | 2.17, 2.29, 2.28, 2.30 | 0.2862 | 0.219 | 4,380,675 | P8 (g5h) |
| `g1_qwen35_allwin_horizon_6k` | same, Qwen3.5-2B-Base, 24/24 windowed | 1 | 6000 | 1.00 | 0.001 | 0.002, 0.001, 0.001, 0.001 | 0.3277 | 0.241 | 4,380,675 | P9 (g2b) |
| `g1_w1_tied` | W1, dense successor-bound keys, 8 heads | 1 | 3000 | 1.00 | 0.0001 | 0.000 × 4 | 0.3447 | 0.310 | 4,282,451 | P10 |
| `g1_w1_dg` | W1, DG k-WTA keys, 4 heads, key expansion 4 | 1 | 3000 | 1.00 | 0.0001 | 0.000 × 4 | 0.3487 | 0.314 | 3,742,795 | P10 |
| `g1_w1_tied_pairs2` | W1 dense | 2 | 3000 | 0.55 | 0.186 | 0.74, 0.05, 0.11, 0.03 | 0.3685 | 0.332 | 4,282,451 | P11 |
| `g1_w1_dg_pairs2` | W1 DG | 2 | 3000 | 0.43 | 0.231 | 0.76, 0.11, 0.15, 0.14 | 0.3634 | 0.327 | 3,742,795 | P11 |

LM: `g1_qwen3_lm_horizon`, Qwen3 WikiText-103 with the horizon kernel, LM settings as in `pretrained/`. Perplexity 11.53 (loss 2.445), against 12.01 with the default kernel and 9.01 with full attention: 49% of the loss gap against 41%. gain 0.3592, inj 0.192. Register P12.

Notes from the training curves:
- `g1_qwen3_horizon_perpos` reproduces `pretrained_qwen3_needle_horizon` exactly (same eval numbers and curve).
- `g1_qwen35_allwin_horizon_6k`: loss at the floor to step 1800, 1.393 at 2400, 0.228 at 3000, 0.006 at 3600.
- `g1_w1_tied`: 1.842 at step 600, 0.178 at 900, 0.000 by 1200.
- `g1_w1_dg`: 0.73 at step 300, 0.008 at 600.
- With two pairs the needle gap is 490–502 tokens (`model.gap_range`), not 501–502.
- W1 has 12.6% fewer parameters with DG keys than with dense keys (3,742,795 against 4,282,451).

## evolve/round2/

Round 2, candidates W1b (W1 with context-bearing addresses, `evolve/w1_gdn.py` from `26c51e3`), two stored pairs. Runs were in progress when this file was written (`r2_w1b_gdn_pairs2`: `--w1-addressing gdn --w1-heads 6`; `r2_w1b_ctx4_pairs2`: `--w1-ctx 4`). Results go to the register when they land.

## evolve/round0/

Planned outputs of the round-0 diagnostic arms g0–g7 (`research/plans/round0_plan.md` §2: `g0_ctrl`, `g1_flat`, `g2_exp098`, `g3_flat_notime`, `g4_flat_nocontrast`, `g5_flat_pairs2`, `g6_flat_l6`, `g7_flat_l22`). None has run; their status is in the register's round-0 table.

## alignment/

Written by `experiments/alignment/run.py` as `results/alignment/<--name>.json` (`tiny.json` for `--tiny`). For A1, A2 and A6 (register). `tests/test_alignment.py` writes `pytest_tiny.json` here and deletes it at the end; do not commit it if a test crashes first.

**Data rule.** Only judge scores, refusal labels and aggregate metrics may be committed here. Raw completions to harmful prompts go to `--private-dir`, outside the repository. Review finding RV1.A6 (an absolute or `../` `--name` bypasses the check) is open, so check the output path by hand until it is fixed.

## Naming conventions

- **synthetic/**: `<script>[_<suite or variant>][_seed<N>]`. `temporal_loop.py` names its default output `temporal_loop_<suite>.json` (`temporal_loop_custom.json` for a model built from flags); other variants were named with `--out`.
- **pretrained/**: `pretrained_<model>_<task>[_<variant>]`. Model `qwen3` = Qwen3-1.7B-Base, `qwen35` = Qwen3.5-2B-Base. Task `needle` or `lm`. Variants: `allwin` (every layer windowed, GDN included), `layer22` (tap at block 22), `horizon` (exp kernel, ρ 0.95, τ 32). Without `--out` the script writes `pretrained_loop_<task>.json`.
- **evolve/round\<N\>/**: one folder per round. Round-1 files are `g1_<model or candidate>_<variant>`; the `g1_` prefix means round 1 (the folder was `gen01/`) and is unrelated to round-0 arm g1. From round 2 the prefix is `r<N>_` (`r2_w1b_gdn_pairs2`), which avoids that clash. Round-0 arms are named `g<arm>_<variant>` as in the plan.
- Every `.json` has a `.log` with the same stem (redirected console output), except `temporal_loop_long_seed1`. Evolve runs also write a `.pt` with the same stem.
- Do not name a new subfolder `runs/`, `checkpoints/`, `build/`, `dist/`, `env/` or `wandb/`: `.gitignore` matches those at any depth.

### JSON fields (pretrained/ and evolve/)

| Key | Content |
|---|---|
| `args` | Every command-line argument of `pretrained_loop.py` |
| `model` | Block counts, `linear_layers` (GDN), `tap_layer`, `windowed_layers`, `reach`, `trainable_params`, `base_params`, `transformers` version, `gap_range` (needle), `window_check` |
| `curve` | Training loss, token_acc and inj_ratio every few hundred steps |
| `results` | `windowed` (frozen model under the window, side channels off), `windowed+side` (side channels on), `full` (full attention, the upper bound); `gain`; `per_pos` (evolve, needle only) |
| `candidate` | Evolve runs with `--candidate` only: the class and its own flags |
| `seconds` | Wall time |

`results.gain` of exactly 0.2689 means the gain never trained (the bf16 bug below).

## Provenance

**Float32 reruns.** Before commit `3c90bde` (1 Nov 2026) the side channels were cast to the base model's dtype, bf16. Their Adam updates (about 1e-3) were smaller than bf16's spacing near 1 (2⁻⁷), so they rounded away and the loop gain stayed at 0.2689 in every run (register B3). Every GPU run was redone with float32 side channels. The reruns overwrote the files in `pretrained/` in place. The bf16 versions exist only in git history, at their old paths: `0f5cc8c` (`pretrained_qwen3_needle`), `775e0f2` (`pretrained_qwen35_needle`), `a741b80` (`pretrained_qwen35_needle_allwin`) and `2297571` (`pretrained_qwen3_lm`). The bf16 Qwen3.5 LM run was stopped at step 600 of 1000 and never wrote a JSON. All files in `evolve/round1/` were run after the fix.

**`args.dtype` is the base model's dtype**, not the side channels'. Every file in `pretrained/` and `evolve/round1/` records `bfloat16`; the side channels were float32 in all of them. No file records the side-channel dtype or the commit it ran at.

**The bf16 index leak.** In bf16, `torch.arange(512)` has only 385 distinct values, so the `d >= 0` masks in the loop kernels let some answer positions see their own targets (register B4, round-0 D5). That is why the bf16 all-windowed Qwen3.5 run scored 1.69. See the warning in [`research/HANDOFF.md`](../research/HANDOFF.md).

**Superseded results: bf16 bug (do not cite).** Kept for comparison with the reruns only. Gain was 0.2689 in every run.

Needle task: seq 512, batch 8, window 64, 1000 steps, needle gap about 501 tokens. Each cell is exact / token_acc / loss.

| Run (commit) | Windowed layers | Windowed | Windowed + side | Full |
|---|---|---|---|---|
| qwen3_needle, Qwen3-1.7B-Base (`0f5cc8c`) | 28/28 | 0.00 / 0.106 / 3.07 | 0.00 / 0.280 / 1.84 | 1.00 / 1.000 / 0.011 |
| qwen35_needle, Qwen3.5-2B-Base (`775e0f2`) | 6/24 attention; 18 GDN intact | 0.00 / 0.056 / 4.03 | 1.00 / 1.000 / 0.0006 | 1.00 / 1.000 / 0.034 |
| qwen35_needle_allwin (`a741b80`) | 24/24 | 0.00 / 0.130 / 3.04 | 0.00 / 0.320 / 1.69 | 1.00 / 1.000 / 0.034 |

The bf16 allwin run's training loss dropped late, to 1.60 at steps 800 and 1000. The float32 rerun did not reproduce the drop (P6).

Language modelling: WikiText-103, seq 1024, window 256, 1000 steps. Each cell is perplexity / loss.

| Run (commit) | Windowed | Windowed + side | Full |
|---|---|---|---|
| qwen3_lm (`2297571`) | 14.62 / 2.683 | 11.91 / 2.477 (43% of the gap) | 9.01 / 2.199 |
| qwen35_lm | stopped, no result | | |

The domain-adaptation caveat applies to the reruns too: in the LM task only `windowed+side` trains anything on WikiText (4.38M parameters), so part of any LM gain may be adaptation to the domain rather than long-range recall (P5, O4).

**Superseded and duplicated files**

| File | Status |
|---|---|
| `pretrained/pretrained_qwen35_needle_allwin_horizon` | Superseded by `evolve/round1/g1_qwen35_allwin_horizon_6k` (P9). The 3000-step run was undertrained |
| `pretrained/pretrained_qwen3_needle_horizon` | Reproduced exactly by `evolve/round1/g1_qwen3_horizon_perpos` (P7), which adds per-digit scores |
| `synthetic/temporal_loop_all` | Partly superseded by `temporal_loop_long`: its `closed_osc_ca1` and `memory_time` runs (1500 steps) are too short; the 3000-step versions are in `_long`. Its other models stand |

**W1 arguments.** The four `g1_w1_*` JSONs record the default loop arguments in `args` (`kernel osc`, `rho 0.7`, `tau 16`, `dg_keys false`). W1 ignores them. Its real configuration is the top-level `candidate` key. The `.log` files do not name the candidate.

**Old paths.** The files moved in commit `1b9dc42`. `args.out` inside the JSONs and the `saved …` lines in the logs still show the old paths; re-running from recorded args writes to the old location unless `--out` is changed.

| Old | New |
|---|---|
| `results/{assoc_recall*,continual,gdn_capacity,loop_systems,temporal_loop_*}.{json,log}` | `results/synthetic/` |
| `results/pretrained_*.{json,log}` | `results/pretrained/` |
| `results/evolve/gen01/` | `results/evolve/round1/` (file names unchanged) |
| `results/evolve/gen00/` (planned, never created) | `results/evolve/round0/` |
| `results/pretrained_HANDOFF.md` | `research/HANDOFF.md` |
| `results/evolve/REGISTER.md`, `results/evolve/NOTEBOOK.md` | `research/REGISTER.md`, `research/NOTEBOOK.md` |
| `results/evolve/round0_plan.md` | `research/plans/round0_plan.md` |
| `results/evolve/reviews/` | `research/reviews/` |

**Checkpoints.** The eight `evolve/round1/*.pt` files (about 134 MB, the trained side channels' `state_dict`) are git-ignored by `*.pt` and exist only on geof. `pretrained_loop.py` saves no checkpoint, so nothing in `pretrained/` can be ablated at evaluation time without a rerun through `evolve/run.py`.
