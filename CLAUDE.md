# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A research library of hippocampus-inspired memory modules for sequence models (PyTorch), plus the experiments that test them and the lab records that track every result. It has three layers:
- `limbic_llm/`: the modules;
- `experiments/`: CPU experiments on small models, and GPU experiments that attach "side channels" to a frozen Hugging Face LLM;
- `research/` and `results/`: the records and the raw outputs.

Several Claude sessions work on this repo concurrently, on different machines: a GPU session on geof, a coordinating session, and others. Treat `research/HANDOFF.md` as the coordination note.

## Commands

```bash
source .venv/bin/activate                 # the repo venv (pip install -e ".[dev]"; add ".[pretrained]" for HF models)
pytest -q                                 # full suite (CI: CPU torch, Python 3.11, without the [pretrained] extra)
pytest -q tests/test_evolve_w1i.py        # one file
pytest -q tests/test_gated_delta.py -k chunked     # one test by name
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 pytest -q   # when the GPUs are busy with runs
```

**Known failures.** With `flash-linear-attention` installed (as on geof), the four `qwen35` cases in `tests/test_pretrained_loop.py` fail on CPU ("Pointer argument cannot be accessed from Triton"). That comes from the environment; CI does not install fla.

**Experiments**
- **CPU suites:**
  ```bash
  python experiments/temporal_loop.py --suite all|feedback|ablations|recency
  python experiments/assoc_recall.py | continual.py | gdn_capacity.py | loop_systems.py
  ```
- **Frozen-LLM side channels** (GPU):
  ```bash
  python experiments/pretrained_loop.py --model Qwen/Qwen3-1.7B-Base --dtype bfloat16 --device cuda --task needle --window 64 --seq-len 512 --steps 3000 --out results/pretrained/<name>.json
  ```
- **Evolution search** (GPU): `python experiments/evolve/run.py <pretrained_loop flags> --candidate evolve.<module>:<Class> [--needle-scatter 8] --out results/evolve/round<N>/<name>.json`. Batches go through `experiments/evolve/queue.sh JOBS OUT_DIR 0 1`. Push with `experiments/evolve/sync.sh`.
- **Docs:** `python docs/src/build.py` (architecture HTML, needs mermaid-cli). For the design .docx: `python docs/src/bib_to_json.py > docs/src/references.json && node docs/src/build_architectures_docx.js`.

Several scripts skip work already present in their default output file, or overwrite it (`temporal_loop.py`, `gdn_capacity.py`, `cpu_suite.py`, `probe_cue.py`, `eval_by_query.py`). Pass `--out` into the right results folder. Console output is not saved; capture `.log` files with `2>&1 | tee`.

## Architecture (the parts that span files)

**`limbic_llm/`: one module per circuit stage** (re-exported in `__init__.py`):
- **Strategy I** (`hippocampal_block.py`, `models.HippocampalLM`): DG → CA3 (episodic store) → CA2 → CA1 → Subiculum.
- **Strategy II** (`titans_hybrid.py` over `neural_memory.py`): Titans.
- **`gated_delta.py`**: the delta-rule memory. `gated_delta_scan` is the reference per-token loop; `gated_delta_chunked` is the exact chunkwise solve used by default. `_run` dispatches between them. Addressing modes: `gdn` (short convs, which start as the identity), `tied` (successor-bound: v_t is filed under a_{t−1}) and `dg` (k-WTA codes).
- **`temporal_loop.py`**: `HippocampalLoop`, the O(T) kernel-weighted loop with contrast and CA1 control; `TimeChannel`, the clock; and `TemporalLoopLM`, the small CPU model.

**`experiments/pretrained_loop.py`: hooks on a frozen HF model.**
- `HookedLM` registers a forward hook on one block (`--layer`, default the middle one). The hook (`_tap`) replaces that block's output `h` with `side(h)`.
- Pre-hooks on every attention submodule (`_mask_hook`) overwrite the attention mask with a window (`block` / `sliding` / `segment`). `--window-layers attention` windows only the attention layers of Gated DeltaNet hybrids (Qwen3.5).
- `condition(window, side)` switches one model between the three evaluated conditions: `windowed` (side off), `windowed+side`, and `full`.
- **The side channels run in float32 whatever the base dtype.** bf16 rounds away the Adam updates, and bf16 `arange` merged positions into an anti-causal leak (register B3, B4).

**`experiments/evolve/`: the evolution search.** `run.py` wraps `pretrained_loop.main` *without editing it*, by monkeypatching three things:
- `pl.evaluate`, for per-digit metrics;
- `pl.SideChannels`, so `--candidate` swaps in another module;
- `pl.NeedleTask`, so `--needle-scatter` uses `ScatterNeedleTask`.

It also saves the side channels as a `.pt` next to the JSON. A candidate module must provide:
- `split_args(argv)`, taking its own prefixed flags;
- `Class(d_model, args, ns)`, whose `forward(h)` returns `(h + injection, LoopStats)`;
- a `.loop` with `gain_logit` and `tau`.

Candidates so far:
- `w1_gdn.py`: W1, a successor-bound GDN side memory;
- `w1g_goal.py`: W1g, a DG-keyed start store, plus W1h settling;
- `w1i_events.py`: W1i, an event-gated goal field.

`cpu_suite.py` runs candidates through `temporal_loop.py`'s training by monkeypatching `temporal_loop.build`. `eval_by_query.py`, `probe_cue.py` and `diag_goal.py` are the analysis tools.

**`experiments/alignment/`**: a CA2 norm-steering controller (probe, steering, judge, run). **Data rule:** results JSONs hold labels and scores only. Response text goes only to `--private-dir`, outside the repo (`judge.assert_private`). Never commit completions to harmful prompts.

## Research records (keep them current)

- **`research/REGISTER.md`** is the single source of truth.
  - Every result has an ID (M/T/P for results, B for corrections, DEC for decisions, RV for review findings) and a repo-root evidence path.
  - Every planned experiment is a backlog row (W, S, A, R, J, O, L) with its status.
  - Update rows in place when a result lands or overturns an earlier one.
- **`research/NOTEBOOK.md`** is the dated log. Do not rewrite old entries; add bracketed "[Correction, <date>: …]" notes.
- **`research/evolution_protocol.md`** is how rounds of the side-memory search are run. Read it before starting a round. In brief:
  - Diagnose before building, and toy-check the mechanism.
  - Test before any GPU time, at least: E1, strict causality, bitwise, including bf16-rounded input; E2, a fixed-size state; E3, at most 4.4M parameters at width 2048; and the mechanism through the module's real code path.
  - Use controls: the `windowed` row must be 0.00.
  - Use 3 or more pairs on the scattered layout, and read accuracy by queried rank (one pair is passkey retrieval; the fixed layout allows position shortcuts).
  - Use ≥ 2 seeds before claiming.
  - Stop for a human checkpoint after every round.
- **`results/<track>/`** holds the raw output: `synthetic/` (CPU), `pretrained/`, `evolve/round<N>/` and `alignment/`.
  - `results/README.md` records provenance, including which runs predate the bf16 fix and which files are superseded.
  - Commit the finished `.json` with its `.log`. Never commit `.pt`, the queue files, or logs of runs still in progress.
  - Stage files by name: other sessions' uncommitted edits can be in the working tree.
- **`RESULTS.md`** is the narrative write-up and must agree with the register. **`docs/genesis.md`** is the history of the ideas.

## Working conventions

- Work on feature branches; `main` advances by PR.
- Before pushing, bring in the other sessions' commits: `experiments/evolve/sync.sh` does a fast-forward, or a rebase-equivalent replay that tolerates logs still being written, then pushes and never force-pushes.
- The search's fitness and the anti-gaming rules E1–E9 are in `research/plans/round0_plan.md` §3 and in `evolution_protocol.md` §2. Don't relax them silently. A change of rule is a decision (DEC) recorded in the register.
