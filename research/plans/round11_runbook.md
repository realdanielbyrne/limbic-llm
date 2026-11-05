# Round 11 runbook: compression and time in the W1s slot memory

**For:** the session that executes round 11 on geof (2× RTX 5090), after round 10 Stage B completes.
**Design and rationale:** [`round11_plan.md`](round11_plan.md). Read it first; this runbook says how to execute it.
**Approved:** Daniel, 4 Nov 2026 ("Looks good. Write up the plan for another session to execute once the current round completes"). The defaults for the plan's open decisions are in §2; they are recorded as DEC9 in the register.

**Revision 1 (Daniel, 4 Nov 2026): time as data, not as a penalty.** "Time as a penalty might not be the most appropriate use of time. LLMs went away from distance-based position encoding to RoPE … time as a data point, ordinal encoding, position relative to other tokens I think still matters."
- The log/ordinal *penalty* arm is replaced by two **time-as-data** arms: rotary slot time (`rope`) and a query-chosen ordinal rank (`ordinal`).
- Supersession (supr/supw) stays: it is already ordinal and content-gated.
- The linear penalty remains only as the I1 reference.
- A new CPU-toy gate (G0.8) checks first that time as data does not repeat the merit failure of time codes concatenated into keys (0/10).

**Revision 2 (after round 10 Stage B, P54–P58; Daniel: "modify round 11 if results from 10 dictate a change").**
- **The base recipe is I1.** It learns 3 of 5 seeds (seed 1 at 0.94, seed 4 at 0.87), with grid means 0.93–0.96 on the learned seeds. A2, I1K and RI do not beat it.
- **Stage-A seeds are now 1 and 4,** the borderline and the failing I1 seed, which are more informative than 1 and 2. The references are the committed I1 seeds 1 and 4.
- **Held-out names are a deciding cell** (I1 gets 0.58–0.74 on names never seen in training, P55). New arms A10 (a large training name pool) and A11 (lexical keys behind a gate that starts near zero; I1K seed 3 generalised at 0.88, but lexical keys collapsed 3 of 5 seeds).
- **The update task is ill-posed as it stands:** full attention prefers the old code (P56). G0.7 must find an update form with a ceiling, and update rules apply only to such a form.
- **G0.1 is reframed.** Its recency premise was refuted (P58): the failing seeds have blunter alignment (log β_align 2.39 against 2.65–2.77). New arm A12 (sharper alignment), conditional on G0.1.
- **Reinstatement stays deferred** (RI 0 of 5).

**Revision 3 (4 Nov 2026).** Adds Stage C, a window-size sweep (§5b): the 64-token window was never swept. Also adds the plain-language summary [`../side_memory_summary.md`](../side_memory_summary.md) to the reading list and to the records kept per stage.

Read before starting, in this order:
1. `CLAUDE.md`, then [`research/side_memory_summary.md`](../side_memory_summary.md), the plain-language status;
2. `research/HANDOFF.md`;
3. `research/evolution_protocol.md` (the round loop, rules E1–E9, pitfalls);
4. the NOTEBOOK from "Round 10 plan" to the end;
5. REGISTER rows P49–P53, DEC4–DEC9 and W13;
6. `experiments/evolve/w1s_slots.py`, which is the code you will extend.

---

## 0. Preconditions

1. **Round 10 Stage B is finished and committed.** Check:
   - `results/evolve/round10/stageB/queue.status` and `stageB/eval/queue.status` both end with `all done`;
   - `git log` shows a Stage-B commit with register rows from P54 on.

   If the GPU session did not commit it:
   - Training JSONs are `results/evolve/round10/r10_{I1,A2,I1K,RI}_s{0..4}.json`. Logs are in `results/evolve/round10/stageB/`.
   - Evaluations are `results/evolve/round10/stageB/e10b_*.json`, with logs in `stageB/eval/`.
   - Commit the JSONs and logs by name, never the `.pt` files. Add the register rows and a NOTEBOOK entry using the Stage-B rules in `round11_plan.md` §6 and the round-10 plan entry in the NOTEBOOK.
2. **Base recipe: I1 (decided by Stage B, P54–P57).** The rest of this item is kept for reference.
   - If I1 meets DEC2 (≥ 4 of 5 seeds; ≥ 0.85 exact across the full grid), round 11 builds on I1.
   - If another Stage-B arm (A2, I1K or RI) is clearly better on the full grid, build on that instead. Note it in the NOTEBOOK; all I1 flags below then change accordingly.
3. **Sync:** `git fetch && experiments/evolve/sync.sh`. Other sessions push to `feat/round1-side-memory` too. Stage files by name.
4. **GPUs free:** `nvidia-smi` shows no training jobs.

## 1. Environment and data

```bash
cd ~/limbic-llm && source .venv/bin/activate
DATA=~/limbic-data          # on geof; outside the repo
#   $DATA/wikitext103_train.txt   filler for training and fitting (first ~5 MB of WikiText-103 raw train)
#   $DATA/wikitext103_test.txt    held-out filler for evaluation (WikiText-103 raw test)
#   $DATA/scratch/r11_plan/       the planners' diagnostic scripts and results (no .pt caches)
#   $DATA/scratch/r10_merit/      the round-10 merit-test scripts and results
```

If `~/limbic-data` is missing, regenerate the text files with `datasets`:
- `load_dataset("wikitext", "wikitext-103-raw-v1")`;
- train: the first 5,009,653 characters of `"".join(train["text"])`;
- test: all of `test["text"]`.

**The I1 base command.** This is the exact recipe of the round-10 arm I1:
```bash
BASE="python experiments/evolve/run.py --dtype bfloat16 --device cuda --model Qwen/Qwen3-1.7B-Base \
  --task needle --window 64 --seq-len 512 --batch 8 --steps 6000 --needle-scatter 8 --pairs 3 --layer 21 \
  --needle-names train --text-file $DATA/wikitext103_train.txt --needle-curriculum 1:1500,2:1500 \
  --candidate evolve.w1s_slots:W1sSide \
  --w1s-gate point --no-w1s-centre --w1s-pre 0 --w1s-key learned --w1s-taps ''"
# a run:  $BASE <arm flags> --seed S --out results/evolve/round11/r11_<ARM>_s<S>.json
```

**Queues:**
- Write job lines as `name|command` into a file outside the repo.
- Run `experiments/evolve/queue.sh JOBS OUT_DIR 0 1` in the background, then the evaluation queue after it, chained with `&&`.
- Run one worker per GPU: a second worker per GPU gives no speed-up.
- Additive 6000-step runs take about 13 minutes.
- `queue.*` files are git-ignored.

## 2. Defaults for the plan's open decisions (DEC9)

1. **Attention weights.** They are admissible (DEC8), but they fail as a write-time salience signal (the digits rank 17–24 of 24). Use them only as a diagnostic arm (G0.4).
2. **The alignment code.** Use a **learned** W_z (`--w1s-code learned`), not a seeded random projection. That avoids the question of whether the 1M-float buffer limit covers fixed buffers, so no ruling is needed.
3. **The supw threshold θ.** Initialise it at a fixed a-priori 0.9 and learn it. Do not take it from measured cosines, which would be a DEC5 question.
4. **New evaluation cells.** Add 4096 × 8, codes greater than K (4096 × 16/24, 8192 × 32) and `--decoys 20` as **reported robustness cells**. The DEC2 bar is unchanged.
5. **The order task.** Admit it only if full ≥ 0.80 and windowed ≤ 0.10. Its pass bar is ≥ 0.8 of the full-attention ceiling; if adopted, record a DEC.
6. **Recipe changes.** If round 11 adopts removing the TimeChannel or the linear age penalty, record a DEC.

## 3. Code changes (plan §3), then tests

Edit `experiments/evolve/w1s_slots.py`. Every new flag defaults to today's behaviour, so the round-10 checkpoints still load and reproduce.

| Flag | Where | What |
|---|---|---|
| `--w1s-recency-form {linear,none,log,ord,supr,supw}` | `SlotLoop._select` (the `log_omega` block) and the `stream()` read | `log`: penalty on log(1+age). `ord`: count of newer retained slots. `supr`: subtract w_s × the read mass that newer, query-matching slots carry (two-pass softmax, no age term). `supw`: at a window's close, add σ(γ(cos(K_i,K_j) − θ)) to every older slot i's supersession tag; the read subtracts ω_s·sup_i. All scalars are learned. |
| `--w1s-time-feat {none,rope,ordinal}` with `--w1s-rope-dims 32`, `--w1s-rope-periods 64,65536` | `SlotLoop._select` (key and query of the learned-key score) and the `stream()` read | **Time as data, not a penalty (revision 1).** `rope`: rotate the first rope-dims dims of the slot key by ω_f·s (the slot's onset) and of the query by ω_f·t (the read time), at log-spaced periods. The score then depends on content and t − s jointly, and the norm and other dims are untouched (partial rotary). `ordinal`: a learned embedding E[r_k] of the slot's rank among retained slots (0 = newest … K−1), plus a query-dependent vector u_t = W_u·x_t; the score adds u_t·E[r_k], so the query chooses which rank it wants. Its range is bounded by K, so it is the same at any length. Both default to no linear age penalty (`--w1s-recency-form none`) unless combined explicitly. |
| `--needle-names pool --needle-name-pool N` (in `run.py`) | `ScatterNeedleTask` name sampling | A seeded synthetic name pool, e.g. 2–3 random syllables, N = 2000. It must be disjoint from `eval_by_query.HELDOUT_NAMES` and from `pl.NAMES`; assert both. Training only. |
| `--w1s-lex-init` (log β_lex at init, e.g. −2), `--w1s-align-init` (log β_align at init) | `SlotLoop.__init__` | Lets lexical keys start near-off and grow (A11), and alignment start sharper (A12) |
| `--w1s-code {lex,learned} --w1s-dc 64` | `_features`, the ring buffer, `_alignment` | The alignment code becomes W_z·LN(h) (learned, dc dimensions) in place of the 2048-d state |
| `--w1s-keep M --w1s-keep-by {first,value,learned}` | `_store` (`mem &= keep`) | Keep M of n window entries: the first M; the top M by value norm; or the top M by a learned score from h. Store each as (z_j, v_{j+1}, offset j), so order survives the gaps. Scale the kept values by σ(score) so the scorer gets gradient. |
| (report) | `_stream_size` | State floats per component, written into the result JSON |

Edit `experiments/evolve/eval_by_query.py`:
- `--w1s-set KEY=VAL`: an override applied at evaluation only, for the Stage-0 diagnostics. Its output file name gets a suffix.
- `--update-k K`: a fact restated K times.
- `--decoys N`: N extra non-queried code statements.
- Support up to 32 codes and seq-len 4096/8192, at batch 1–2.
- The order tasks of G0.7, admitted only if they pass.

Tests in `tests/test_evolve_w1s.py` (CPU, `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 pytest -q`):
- **E1 bitwise** (fp32 and bf16-rounded h) for every new flag, including an update after a 2000-token gap, an eviction burst, and top-M at a window's edge.
- **`stream()` equals the parallel forward** for every flag. `stream()` must stay an independent sequential reference: do not call the parallel code from it.
- **State floats** identical at T = 512, 2048 and 4096.
- **supw** is monotone in the match.
- **The empty store** gives 0.
- **Round-10 compatibility:** with all-default new flags, the round-10 I1 checkpoints load (`strict=False`, no missing keys) and evaluate within 0.01 of their committed results.

Then:
- run the full suite (the 4 known qwen35 CPU failures are expected);
- a `run.py --tiny` smoke test for each new flag;
- a CPU toy in which the learned keep-scorer finds the code digits within 300 steps.

Before any GPU time, launch two adversarial reviewers, one on E1/E2 and one on the hooks and controls, as round 10 did. Fix what they confirm.

## 4. Stage 0: no-training gates (plan §2)

Run these on the round-10 checkpoints (I1 seeds 0–4 after Stage B) as evaluation-time overrides through `eval_by_query.py --w1s-set ...`, with n ≥ 96 and the windowed, empty and shuffle controls on every override. They are diagnostics, not results (E6). The planners' scripts in `$DATA/scratch/r11_plan/` are references; re-run the gates through the repo code.

| Gate | Test | Pass, and what follows |
|---|---|---|
| G0.1 | (revised by P58) On I1 seed 4, and on I1K's collapsed seeds 0–2, evaluation-time overrides: recency `none` and `supr`; the alignment temperature sharpened (log β_align + 0.3, + 0.6) | If sharper alignment restores seed 4's oldest ranks (random 2048 × 5 rank 0–2 ≥ 0.75), run A12. If recency overrides do, prioritise A1–A3. If neither does, report to Daniel before Stage A. |
| G0.2 | Forms {linear, none, log, ord, supr, supw} on random 2048 × 5, 4096 × 8, and update same/now/new at 512 and 2048 | supr or supw: 4096 × 8 ≥ 0.80 with stale ≤ 0.10 on both seeds. The better of log/ord on updates becomes arm A1. |
| G0.3 | supw false fires (cos > θ between different facts) on held-out names and jitter | ≤ 2%, otherwise drop supw (A3) |
| G0.4 | Selection oracles: first-12; read offsets only (M = 4, ceiling); value-norm top-8; random-8; attention top-8 (diagnostic) | Value-norm within 0.03 of the ceiling means A6's scorer starts from value norm |
| G0.5 | A W_z fitted on CPU from train caches, scored on held-out | Learned 64-d ≥ 2048-d − 0.02 gives dc = 64; otherwise dc = 128 |
| G0.6 | 4096 × {16, 24} and 8192 × 32 codes, K = 16 vs K = 64, under the G0.2 winner | K = 16 coverage ≤ 0.8 and K = 64 ≥ 0.95 keeps K = 64 in A8 |
| G0.8 | CPU toy (port `$DATA/scratch/r10_merit/b_hysteresis/toy2.py`, ≥ 10 seeds) with an update and an order query: time as an additive/concatenated key code (the merit arm that failed, 0/10) vs `rope` vs `ordinal` vs `supr` vs the linear penalty | `rope` or `ordinal` must learn ≥ 8/10 and beat the linear penalty on the order query. Otherwise drop that arm before GPU time; a time-data form that loses content selection (< 0.9 on the plain query) is dropped too. |
| G0.7 | Ceilings, full vs windowed at 512: latest-wins update-k; "the code stated just before NAME's"; first/last; next fact. **Also update phrasings** (e.g. "NAME's code has changed from OLD to NEW", "… is now NEW (it used to be OLD)"), because the current `--update new` form has full attention preferring the OLD code (P56) | full ≥ 0.80 and windowed ≤ 0.10 admits a task. The update decision rules use only an admitted update form; any admission runs A9. |

**Commit the Stage-0 outputs** (JSON + logs) under `results/evolve/round11/stage0/`, with a short NOTEBOOK entry.

## 5. Stage A: GPU arms (seeds 1 and 4; plan §4, revision 2)

The reference is the committed round-10 I1 seeds 1 and 4 (`results/evolve/round10/r10_I1_s{1,4}.json`, Stage-B evaluations in `stageB/`); do not rerun them. Each arm is a single change to the base recipe (§1).

| Arm | Flags added to `$BASE` |
|---|---|
| A1 T-rope | `--w1s-recency-form none --w1s-time-feat rope` (if G0.8 passes) |
| A1b T-ordinal | `--w1s-recency-form none --w1s-time-feat ordinal` (if G0.8 passes) |
| A2 T-supr | `--w1s-recency-form supr` |
| A3 T-supw | `--w1s-recency-form supw` (dropped if G0.3 fails) |
| A4 T0 | `--no-time` |
| A5 C-code | `--w1s-code learned --w1s-dc 64` (or 128, per G0.5) |
| A6 C-keep | `--w1s-keep 8 --w1s-keep-by learned` |
| A7 C-n12 | `--w1s-win 12` (fixed truncation, the control for A6) |
| A8 COMB | best time arm of A1/A1b/A2/A3 + A5 + `--w1s-win 12 --w1s-K 64` |
| A9 (only if G0.7 admits a task) | ordinal stamps in the read + 25% order-task training |
| A10 N-pool | `--needle-names pool --needle-name-pool 2000` |
| A11 K-lexgate | `--w1s-key both --w1s-taps 3 --w1s-lex-init -2` |
| A12 align-sharp (only if G0.1 points to alignment) | `--w1s-align-init <G0.1's best>` |

That is 22 runs (24–26 with A9 and A12), about 3–3.5 hours on 2 GPUs. The penalty-form arm (log/ord penalty) is dropped by revision 1. The log/ord penalty forms stay only as Stage-0 diagnostics (G0.2). A8's components depend on G0; run it last in the queue.

**Evaluations per checkpoint** (`eval_by_query.py`, output `results/evolve/round11/e11_<run>_<cell>.json`):
- the DEC2 grid: {512, 1024, 2048} × {1, 3, 5} codes × {natural `--text-file $DATA/wikitext103_test.txt`, `--random-filler`};
- `--jitter`, `--heldout-names` (a deciding cell from revision 2);
- `--update same|now|new` at 512 and 2048, and `--update-k 2,3`;
- the robustness cells: 4096 × 8, 4096 × 16/24, 8192 × 32 (batch 1–2), `--decoys 20`;
- any admitted order task.

Report per arm:
- exact by rank and gap;
- slot hit, coverage, null weight;
- the stale rate on updates;
- state floats and parameters.

## 5b. Stage C: window-size sweep (revision 3; after Stage B, with Daniel's approval)

**Why.** The 64-token window was inherited, not swept. Production local-attention windows are 1,000–8,000 tokens. The plain-language summary ([`../side_memory_summary.md`](../side_memory_summary.md)) lists this as the main untested gap for real-model relevance (backlog W15).

**What.** The adopted round-11 recipe (I1 if nothing is adopted), seeds 1 and 4, at windows {256, 1024}. Window 64 is the reference.
- **Keep the facts out of reach.** Scale the sequence so facts stay several windows away:
  - window 256: train at seq-len 2048, evaluate at 2048 and 8192;
  - window 1024: train at 4096–8192, evaluate at 8192 and 16384 (as memory allows).
- **The last fact must be outside the window.** `ScatterNeedleTask`'s `tail` (the minimum filler between the last fact and the query) defaults to 256 and is not a flag. Add `--needle-tail` to `run.py`, set it to ≥ 2 × window, and assert in the task that tail > window, or the base model can read the last fact itself.
- **Measure cost first.** W1s's parallel training form builds (B, T, T) matrices. Run a 100-step smoke at each window and seq-len, and record step time and peak memory. If one does not fit in 32 GB:
  - lower `--batch`, or add gradient accumulation to `run.py`;
  - record the change.

  The streaming form (`stream()`) is O(T·K) and is the inference path. Report both costs.
- **Controls at every window:** windowed 0.00, full ≈ 1.00, the empty store 0.00.
- **Rule (reported, not a DEC2 criterion):** the design "holds at real window sizes" if exact at window 1024 is ≥ 0.85 on the natural and random cells, on both seeds. Report how exact, slot hit and coverage change with the window, and the cost against the KV cache at each length.

## 6. Decision rules (pre-registered; plan §6)

> **Superseded for Stage A and B by DEC11 (4 Nov 2026):** arms are judged by relative difference against I1, full attention and windowed, with 95% intervals; the fixed margins below are not used, trade-offs are reported rather than vetoed, and updates are scored against ground truth. DEC10 also runs A1 and A12 (G0.8 read literally; A12 starts at log β_align 3.68). The rules below are kept for the record.

- **Void:** windowed ≠ 0.00, empty-store ≠ 0, shuffle not near chance, or any E1 failure.
- **Every arm (revision 2):** it is not adopted if it lowers held-out-name exact by more than 0.02 against I1 on the same seeds.
- **A10 and A11:** adopt if held-out-name exact rises by ≥ 0.10 over I1 and no DEC2 cell drops by more than 0.02. A11 must also not collapse a seed (grid mean < 0.5 counts as a collapse).
- **A12:** adopt if seed 4's random 2048 × 5 exact rises by ≥ 0.10 and no other cell drops by more than 0.02.
- **The update rules** below apply only to an update form admitted by G0.7; otherwise update scores are reported, not used.
- **T arms (A1, A1b, A2, A3):** win only if all three hold:
  - mean exact on random 2048 × 5 and 4096 × 8 is ≥ +0.05 over I1;
  - no DEC2 cell drops more than 0.02;
  - update new-exact ≥ I1 − 0.02, with stale ≤ I1.
  The tie-break is the lower stale rate. If an admitted order task exists (G0.7), a time-data arm (A1/A1b) must also reach 0.8 of its full-attention ceiling to be adopted for "when" questions.
- **A4:** drop the TimeChannel if every DEC2 cell and the update task are within −0.02.
- **A5:** adopt if every cell is within 0.02 and the state falls ≥ 5×.
- **A6 and A7:** both must be within 0.02 of I1.
  - A6 also needs tag hit ≥ 0.95 on the code digits.
  - Learned tags are adopted only if A6 beats A7 by ≥ 0.05 on jitter; otherwise use n = 12.
  - Tags settling on frame tokens falsify A6.
- **A8:** adopt if ≥ I1 − 0.02 on every DEC2 cell and ≥ +0.10 on the codes-greater-than-K cells. If A8 fails, the recipe is the best T arm plus the C arms that passed.
- **Then Stage B** (5 seeds, 1–2 recipes): ≥ 4 of 5 seeds learned (DEC2 criterion 1, which I1 misses by one seed), ≥ 0.85 exact on the full grid, held-out names reported and ≥ I1, no update regression on an admitted form, the E8 LM run (`--task lm`) and the CPU suite.

## 7. Records and checkpoints

- **REGISTER:**
  - backlog row W14 (round 11, linked to W13);
  - result rows from the next free P;
  - a DEC for any recipe change adopted (§2, items 5–6).
- **NOTEBOOK:** a dated entry per stage, each with a results table and what it shows.
- **Protocol** §7: a round-11 row.
- **RESULTS.md** §4.9: keep it consistent with the register.
- **Commits:** JSON + `.log` only, by name; never `.pt`, the queue files, or logs of runs still in progress. Sync with `experiments/evolve/sync.sh`. On a conflict: `git rebase --abort` and report.
- **HANDOFF:** update the "Side-memory status" section when a stage lands.
- **Checkpoints with Daniel:** after Stage 0, if G0.1 fails, after Stage A, and after Stage B. Do not start Stage B or Stage C without his approval.
- **Summary:** update [`../side_memory_summary.md`](../side_memory_summary.md) when a stage changes the picture. It is the plain-language status that other sessions and readers start from.
