# Experiments

Every script that produces a result, with the track it belongs to, where its output goes, and which register rows rest on it. IDs are those of [`research/REGISTER.md`](../research/REGISTER.md); result files are described in [`results/README.md`](../results/README.md). Status: **done** (its register results exist), **ready** (runs as is for the planned items named), **blocked** (open review findings must be fixed first).

## Scripts

| Script | Track | What it tests | Output | RESULTS.md | Register | Status |
|---|---|---|---|---|---|---|
| `assoc_recall.py` | synthetic (CPU) | One-shot associative recall across a gap longer than a 16-token window: full and windowed attention, Strategy I, Titans and its add-on ablations, Titans with a Gated DeltaNet memory | `results/synthetic/assoc_recall*.json` | §1, §3 | M1, M2, M6, M7 | Done |
| `continual.py` | synthetic | Three fact tasks in sequence: naive fine-tuning against novelty-prioritised replay with sleep | `results/synthetic/continual.json` | §2 | M3 | Done. O1 (uniform replay, sleep only) needs new conditions |
| `gdn_capacity.py` | synthetic | Pure Gated DeltaNet recall against the number of stored pairs; standard, wide, dense successor-bound and DG successor-bound addresses | `results/synthetic/gdn_capacity.json` (resumes from it) | §3 | M4, M5, M8 | Done. O9 needs a similar-key task |
| `temporal_loop.py` | synthetic | `TemporalLoopLM` on the what-when task: which channel carries what, closed-loop stability and long rollouts, recency probe, feedback forms, contrast and DG ablations, cue noise | `results/synthetic/temporal_loop_<suite>.json` (resumes from it) | §4.1–4.6 | T1–T5, T9–T21 | Done. Ready for S3, O2 and O7 (more seeds with `--seed`) |
| `loop_systems.py` | synthetic | The op-amp test: an untrained closed loop with the clamp removed, swept over forward gain for no, divisive and subtractive feedback | `results/synthetic/loop_systems.json` | §4.3 | T6, T7, T8 | Done |
| `tasks.py` | synthetic | Shared tasks (`AssociativeRecall`, `FactTask`); writes nothing | — | — | — | Library |
| `pretrained_loop.py` | pretrained (GPU) | The loop and the clock as a forward hook on a frozen Hugging Face model, with windowed attention (`block`, `sliding`, `segment`); needle task and WikiText LM | `results/pretrained/` (always pass `--out`; the default is `pretrained_loop_<task>.json`) | §4.7 | P1–P6 | Done. Ready for W4 (`--dg-keys`, `--loop-arg read=dg`) |
| `evolve/run.py` | evolve (GPU) | `pretrained_loop.main` unchanged, plus per-position scores, a `.pt` checkpoint, and `--candidate` to swap the side-channel module | `results/evolve/round<N>/` (pass `--out`) | §4.8 | P7, P8, P9, P12; C1a | Done. Ready for g0–g7, W1b.c (`--layer`), W1d (`--task lm`), W1e (`--no-time`), W1f (Qwen3.5, `--pairs 2`), O6 (`--seed`) |
| `evolve/w1_gdn.py` | evolve candidate | W1: the clock, then a Gated DeltaNet side memory at the tap with successor-bound keys (dense or DG), injected through CA1 control; fixed-size state | through `evolve/run.py` | §4.8 | W1; P10, P11 | Done (W1). W1b.a (`--w1-addressing gdn`) and W1b.b (`--w1-ctx N`, context positions in the address) built in `26c51e3`; round-2 runs in progress |
| `alignment/run.py` | alignment (GPU) | A1 harm probe (AUROC, calibration, layer sweep), refusal direction, then the steering arms `none`, `static`, `conditional`, `graded`, `graded_self` on four prompt sets | `results/alignment/<name>.json`; completions only to `--private-dir` outside the repo | — | A1, A2, A6, J1 | A1 ready (caveat RV1.A2). A2 and A6 blocked (RV1.A1, RV1.A3, RV1.A6). J1 blocked (RV1.D1, RV1.D4) |
| `alignment/steering.py` | alignment, reasoning | `SteeringController`: modes `none` / `static` / `conditional` / `graded`, directions `fixed` / `gist` | used by `run.py` | — | A2, A6; R2, R3 | R2 blocked (RV1.C1, RV1.C2). R3 blocked (RV1.C3, RV1.C4) |
| `alignment/probe.py`, `data.py`, `judge.py` | alignment | Activations and the harm probe; prompt sets and chat formatting; refusal and degeneration judges, and the data rule (`assert_private`) | — | — | A1, A2, A6 | Library |

The steering spot checks N1–N5 come from a notebook, [`docs/notebooks/activation_additions_qwen3.ipynb`](../docs/notebooks/activation_additions_qwen3.ipynb), not from a script. They are recorded in the register only.

Tests are in `tests/`: `test_temporal_loop.py`, `test_pretrained_loop.py` (CPU, `--tiny` models), `test_evolve_w1.py` (W1's rules E1–E3 and an end-to-end runner check), `test_alignment.py` (skipped unless `transformers` is installed), plus the module tests.

## Running

The CPU scripts default to `results/synthetic/`:

```bash
python experiments/assoc_recall.py --steps 3000
python experiments/continual.py --steps 600
python experiments/gdn_capacity.py --pairs 8,16,32 --steps 1500
python experiments/temporal_loop.py --suite all          # also: what_when, stability, recency, feedback, ablations
python experiments/loop_systems.py
```

`gdn_capacity.py` and `temporal_loop.py` skip any model already in their output file. Delete the file, or pass a new `--out`, to rerun.

The GPU scripts need `pip install -e ".[pretrained]"`. The commands for the alignment track are in [`alignment/README.md`](alignment/README.md).

## Running a round of the evolution search

A round is one folder, `results/evolve/round<N>/`, plus a dated entry in [`research/NOTEBOOK.md`](../research/NOTEBOOK.md). Round 0's plan is [`research/plans/round0_plan.md`](../research/plans/round0_plan.md); its arms g0–g7 write to `results/evolve/round0/`.

Run every arm through `evolve/run.py`, so it gets per-digit scores and a checkpoint, and always pass `--out` into the round's folder. (The round-0 plan's commands call `pretrained_loop.py` directly; that works, but records neither.)

```bash
# the O(T) loop with the horizon kernel (P7)
python experiments/evolve/run.py --model Qwen/Qwen3-1.7B-Base --dtype bfloat16 --device cuda \
    --task needle --window 64 --seq-len 512 --batch 8 --steps 3000 --log-every 300 \
    --kernel exp --rho 0.95 --tau 32 --out results/evolve/round1/g1_qwen3_horizon_perpos.json

# a candidate module in place of the loop (W1 with DG keys, P10)
python experiments/evolve/run.py --candidate evolve.w1_gdn:W1Side \
    --w1-addressing dg --w1-heads 4 --w1-key-expansion 4 \
    --model Qwen/Qwen3-1.7B-Base --dtype bfloat16 --device cuda \
    --task needle --window 64 --seq-len 512 --batch 8 --steps 3000 --log-every 300 \
    --out results/evolve/round1/g1_w1_dg.json
```

- `--dtype` sets the base model's dtype. The side channels always train in float32.
- `evolve/run.py` reads back the JSON that `pretrained_loop.main` wrote, so its default output path must stay identical to `pretrained_loop.py`'s.
- A candidate is `--candidate module:Class`, importable from `experiments/`. It is built as `Class(d_model, args, ns)` and returns `(h, stats)`, where `stats` is `None` or has an `inj_ratio` (B, T) field. It exposes `loop` with `gain_logit` and `tau` (`loop = None` is allowed). An optional module-level `split_args(argv) -> (namespace, rest)` takes the candidate's own flags; they are saved under the JSON's `candidate` key, while `args` keeps the unused loop defaults.
- A candidate must pass rules E1–E3 of the round-0 plan (causality, fixed-size state, parameter budget). `tests/test_evolve_w1.py` is the template.
- After the runs: the round's NOTEBOOK entry, register rows citing `results/evolve/round<N>/…`, and RESULTS.md once the result is stable ([`research/README.md`](../research/README.md)).

## Planned experiments that need new code

Planned items that run with the scripts above are marked "ready" in the table. Everything else needs code first.

| ID | What has to be built |
|---|---|
| W1c | A W1 adapter for `temporal_loop.py`, so W1 can run the stability and recency suites |
| W2 | A new `--task` in `pretrained_loop.py`: elapsed-time questions in long documents |
| W3 | Replay consolidation into LoRA adapters on a frozen model |
| W5 | Separate read (about L/3) and inject (L/2) blocks in `pretrained_loop.py` |
| W6, W6.a | A depth loop over blocks L+1 … N with a CA1-novelty exit, and a multi-hop recall task |
| W7 | Multi-layer injection |
| W8 | A fixed-state multi-timescale φ-feature loop (round-0 direction 2) |
| W9 | Write selectivity with a decayed running-mean prior (round-0 direction 3) |
| W10 | A gap curriculum in the needle task (round-0 direction 6) |
| W11 | Joint light fine-tuning of the base and the side channels |
| C2, C3, C5, C6 | Round-0 diagnostics: value probes at layers 6/14/22, loop instrumentation, window checks on every layer, held-out evaluation (`--eval-seq-len`, random needle offset, natural filler). C6 blocks W1's E6 score |
| C4 (= O5) | An evaluation-time ablation script for a trained checkpoint. It also needs a Qwen3.5 checkpoint from `evolve/run.py`: `pretrained_qwen35_needle` has no `.pt` |
| O4 | A memory-ablated arm in `pretrained_loop.py` |
| S1, S2, S9 | A gain sweep and closed-loop runs on Qwen with downstream norms, entropy and a random-direction control; S9 sweeps the steering directions |
| S4–S8 | Compensated subtractive feedback, homeostatic gain, novelty-gated gain, norm-preserving injection, multi-band damping |
| A3, A4, A5 | Multi-turn pressure, controller ringing, adaptive attacks on the probe |
| A7 | Loading `AutoModelForMultimodalLM` and `find_blocks` for the 9B hybrid |
| A8 | Gain-law modes in `alignment/steering.py` |
| R1–R5 | A reasoning runner (R1 has only notebook spot checks) |
| R6 | A repetition regulator in `alignment/steering.py` |
| J1–J6, J3.a | Calibrated decision heads, a LAYA-style comparator, a calibrated CA1 for Titans, a novelty-lr warm-up arm |
| O1 | Uniform-replay and sleep-only conditions in `continual.py` |
| O8–O13 | Delay compensation by theta timing, pattern separation on similar keys, deep against linear memory, neurogenesis and consolidation scheduling, CA2 identity steering, HippocampalBlock inside a pretrained LLM |
