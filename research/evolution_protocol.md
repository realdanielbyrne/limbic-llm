# Evolution protocol: how the side-memory search is run

This is the procedure the GPU session used for rounds 0–5 of the side-memory search (2 Nov 2026). Future sessions should follow it, or change it here, so that rounds stay comparable and every result can be traced.

The search is AlphaEvolve-style:
- propose candidate side-channel modules;
- score them on fixed tasks on the GPUs;
- keep what works;
- record everything;
- **stop at a human checkpoint after every round**, where Daniel approves, redirects or vetoes the next one.

It differs from AlphaEvolve in one deliberate way. Most rounds begin with a *diagnosis* of why the best candidate fails, not with a blind batch of mutations. Most of the progress came from diagnoses.

Related records: [`REGISTER.md`](REGISTER.md) is the current state and IDs, [`NOTEBOOK.md`](NOTEBOOK.md) the dated log, [`plans/round0_plan.md`](plans/round0_plan.md) holds the anti-gaming rules E1–E9, and [`HANDOFF.md`](HANDOFF.md) covers coordination between sessions.

---

## 1. The loop (one round)

| Step | What | Tools |
|---|---|---|
| 0. Read the state | REGISTER (results P*, backlog W*), the latest NOTEBOOK entry, HANDOFF, any new review in `reviews/`. Pull first: other sessions push to the same branch | `git fetch`, `experiments/evolve/sync.sh` |
| 1. Question and plan | One question per round, framed as hypothesis → metric. List the arms *and their controls*. Propose at the checkpoint; do not start the GPU work before Daniel approves | NOTEBOOK "Proposed round N" |
| 2. Diagnose first | Before building, look for the cause with cheap, training-free measurements on the frozen model or on saved checkpoints. Then check the proposed mechanism numerically on a one-hot toy | `probe_cue.py`, `diag_goal.py`, `eval_by_query.py` |
| 3. Build the candidate | A new module in `experiments/evolve/` with the candidate interface (§3). Reuse `limbic_llm/gated_delta.py` and earlier candidates. Stay within the budget | e.g. `w1i_events.py` |
| 4. Test before any GPU time | Causality, state size, budget, and a mechanism test **through the module's real code path**. Then a tiny end-to-end run and the full suite | `tests/test_evolve_*.py` |
| 5. Review (for large changes) | An adversarial review: independent reviewers try to break the code (leaks, wiring, task bugs); a verifier tries to refute each finding. Fix what survives | Workflow; reports in `reviews/` |
| 6. Run | A jobs file through `queue.sh`, one worker per GPU. Each run writes JSON, `.log` and `.pt` to `results/evolve/round<N>/` | `queue.sh`, `run.py` |
| 7. Evaluate | Per-digit loss against the chance level ln(pairs); accuracy by queried rank; accuracy by gap; training curves (late jumps); controls | `run.py` output, `eval_by_query.py` |
| 8. Record as results land | Commit each finished JSON and `.log`, then push. Add a REGISTER result row with its evidence path and backlog status. Correct earlier rows when a result overturns them | `sync.sh`, REGISTER |
| 9. Checkpoint | A NOTEBOOK entry: setup, results table, what it shows, surprises, corrections, proposed next round. Report to Daniel and **stop** | NOTEBOOK |

## 2. Fitness and rules

- **Reach.** The needle task on Qwen3-1.7B-Base with every layer windowed (block window 64, gaps of about 270–500 tokens).
  - Score the per-digit loss against chance (ln 10 per digit; ln(pairs) for choosing the start), then exact match.
  - Use **2 pairs or more**: one pair is passkey retrieval, not binding.
  - Prefer **3 pairs or more on the scattered layout** (`--needle-scatter 8`). The fixed layout lets a memory solve 2 pairs by position (P18).
- **Must pass.** Two checks on the CPU suite (`experiments/evolve/cpu_suite.py`), against the full design as reference (echo 0.29–0.35, peak 0.53; older event named 18%):
  - **bounded echo:** the injection ratio does not grow over the 520-token rollout, and its maximum stays at or below the reference peak of 0.53;
  - **new tokens dominate:** on the zero-shot recency probe the newer event is named more often than the older one. Report the older share against the reference 0.18.
- **Anti-gaming rules E1–E9** (`plans/round0_plan.md` §3). In practice:
  - **E1:** strict causality, bitwise, for fp32 and for bf16-rounded input.
  - **E2:** a fixed-size state (decision DEC1).
  - **E3:** at most 4.4M trainable parameters at width 2048.
  - **E4:** the side channel sees only `h` (no token ids).
  - **E6:** held-out layouts and seeds.
- **Controls in every comparison.**
  - The `windowed` row of every JSON (side channels off) must be 0.00 exact. If not, the window leaks.
  - `full` is the ceiling.
  - Include an ablation that removes the new component (e.g. `--w1g-no-goal`).
  - Keep the same tap, pairs, layout, steps and seed across the arms being compared.

## 3. Candidate interface

A candidate is a module run by `experiments/evolve/run.py --candidate evolve.<module>:<Class>`:
- `split_args(argv) -> (namespace, rest)` takes the candidate's own flags (prefix them, e.g. `--w1i-*`).
- `Class(d_model, args, namespace)` builds it. Its `forward(h)` returns `(h + injection, LoopStats(inj_ratio, gain))`.
- It exposes `.loop` with `gain_logit` and `tau`.
- Inject through the CA1 control used by every candidate so far: error-only (the component along `h` removed), norm ≤ 0.9·σ(gain)·‖h‖, saturating.
- `run.py` records the candidate and its flags in the JSON (`candidate`), plus per-digit metrics (`results[cond]["per_pos"]`), and saves the side-channel checkpoint as `.pt` next to the JSON.

## 4. Commands

```bash
source .venv/bin/activate
# a round's jobs file: one "name|command" per line
cat > /tmp/jobs.txt <<'EOF'
r6_example|python experiments/evolve/run.py --dtype bfloat16 --device cuda --model Qwen/Qwen3-1.7B-Base --task needle --window 64 --seq-len 512 --batch 8 --steps 3000 --needle-scatter 8 --pairs 3 --layer 21 --candidate evolve.w1i_events:W1iSide --out results/evolve/round6/r6_example.json
EOF
experiments/evolve/queue.sh /tmp/jobs.txt results/evolve/round6 0 1 &   # one worker per GPU
# progress: results/evolve/round6/queue.status; append jobs to results/evolve/round6/queue.pending
python experiments/evolve/eval_by_query.py results/evolve/round6/r6_example.json            # accuracy by queried rank
python experiments/evolve/eval_by_query.py results/evolve/round6/r6_example.json --pairs 1 \
    --out results/evolve/round6/r6_example.by_query_pairs1.json                            # distance without interference
python experiments/evolve/probe_cue.py --blocks 14 21 --pairs 2 3 --out results/evolve/round6/probe_cue.json   # is the cue in h?
python experiments/evolve/cpu_suite.py --models w1i_time --steps 3000 --out results/evolve/round6/cpu_suite.json  # must-pass (CPU)
python -m pytest -q tests/test_evolve_w1.py tests/test_evolve_w1g.py tests/test_evolve_w1i.py
git add results/evolve/round6/r6_example.json results/evolve/round6/r6_example.log && git commit -m "..."
experiments/evolve/sync.sh                                                                # rebase-safe push
```

Every script has a default output path, and some skip work already in that file (`cpu_suite.py`) or overwrite it (`probe_cue.py`, a second `eval_by_query.py` run). Always pass `--out` into the current round's folder.

Commit only finished runs: the JSON and its `.log`. Never commit:
- `.pt` checkpoints (git-ignored, about 13–18 MB each, kept on geof);
- the queue files (`queue.pending`, `queue.pending.lock`, `queue.status`; not ignored, so stage files by name);
- logs of runs still in progress (B8).

## 5. Pitfalls met so far (with register IDs)

| Pitfall | What happened | Guard |
|---|---|---|
| bf16 side channels | Adam updates rounded away; the gain stuck at its initial value (B3) | Side channels train in float32 |
| bf16 index arithmetic | `arange` in bf16 merges positions, an anti-causal leak (B4, round-0 D5) | Float32 indices; E1 tested with bf16-rounded input |
| A horizon too short | The default kernel gave weight 0 at the needle's gap (P3) | Compute the kernel's reach before a run |
| One pair | Solvable without the query (P8); passkey retrieval | 2–3+ pairs; per-digit loss against ln(pairs) |
| Positional shortcut | W1g "solved" 2 pairs by remembering the first and the last fact (P18) | Scattered layout; 3+ pairs; accuracy by queried rank |
| Mean accuracy hides structure | 2/3 accuracy can be first + last | `eval_by_query.py` by rank |
| A result taken for decay | The long-chain loss was interference between stored codes (P24; one checkpoint, small per-bin counts) | Evaluate with `--pairs 1`, and by gap at a fixed rank |
| One seed, late jumps | Seed 0 learned selection in a sudden late drop (near step 1,800 with 3 pairs, 2,100 with 5). Seeds 1–2 had not by step 3,000, and did with a 6,000-step schedule (P25) | ≥ 2 seeds before claiming; read the curves; budget 6,000 steps for selection |
| Tests that rebuild the mechanism | A mechanism test reimplemented the path it claimed to test (RV3.w1i-1) | Test through the module's real `_read`, or spy on its calls |
| Unreliable attribution | Cosine-argmax "which fact did the read return" disagreed with behaviour (RV3.F3) | Base conclusions on behavioural accuracy |
| Designs that fail on paper | Dense context-keyed writes overwrite each other; a linear key shares the token component across facts | One-hot toy check before building |
| Scattered-layout window artefacts | Scattered facts could straddle a 64-token window block (RV3.F2) | `ScatterNeedleTask` resamples them (`block=`), fixed in `672406d`. The round-4 scattered runs predate the fix (≈4% straddled), so P21 seed 0 and the round-5 seed reruns ran on slightly different task versions |
| Training-format policies | W1i's event detector learned what is rare in random-token filler; natural text triggers far more events and overwrites the goal field (P32) | Evaluate on natural filler (`eval_by_query.py --text-file`); train on the target text distribution |
| Constraints that bind at initialisation | A write budget below the detector's initial firing density (≈ 0.12 per token) starved every write from step 0, and the detector died (P35) | Check every gate and constraint at its initial values before running; start budgets above the initial density or anneal them |
| Shared working tree | Another session's uncommitted edits were in the same files | Check `git status` before committing; name others' work in the message |
| Command chains | A failed step was hidden by a later one in a pipe | `set -o pipefail`; check `rc=` in `queue.status` |

## 6. Exit criteria for the side-memory track (DEC2, approved by Daniel 2 Nov 2026; stop rule applied 3 Nov 2026; reopened for rounds 9–10 by DEC3)

The track is **done** when one fixed recipe meets all of these:
1. **Reliability.** At least 4 of 5 seeds learn content selection within 6,000 steps: digit-1 accuracy ≥ 0.95 at every queried rank, with 3 pairs on the scattered layout.
2. **Robustness.** Those seeds average ≥ 0.85 exact on held-out natural filler *and* on random filler, at sequence lengths 512–2048 and with 1–5 stored codes (`eval_by_query.py`).
3. **Must-pass.** Bounded echo and newer-over-older on the CPU suite. WikiText perplexity no worse than the O(T) loop at the same tap, or the trade-off stated explicitly.
4. **Written up.** Register rows, a RESULTS.md section, and this protocol updated.

**Stop rule.** If two consecutive rounds bring no recipe to criterion 1, close the track with a "capable but not reliably learnable" conclusion and move on to other experiments.

**Extension (DEC3, approved by Daniel 3 Nov 2026).** After round 8 closed the track, Daniel extended it by two rounds (9 and 10), because the curriculum had moved reliability from 1 of 5 to 3 of 5. The criteria are unchanged. If neither round meets criterion 1, the closure stands. Any further extension is a new decision.

**New territory (DEC4, approved by Daniel 3 Nov 2026).** DEC3's two-round limit covers iterating the W1i line. New mechanisms (a meaning-based trigger, an integrate-and-fire write gate, episodic slots, reinstatement into attention) are not counted against it. DEC2's criteria remain the bar.

## 7. Round history

| Round | Question | Candidates and runs | Outcome (register) |
|---|---|---|---|
| 0 | Why does the loop fail on Qwen3? | Diagnosis workflow (21 agents) | Kernel reach (D2), the format floor (D1), the bf16 leak (D5) |
| 1 | Does successor binding transfer to a frozen 2B model? | W1 (GDN side memory) | Exact recall with 1 pair; digit 1 a coin flip with 2 (P10, P11) |
| 2 | Can context in the address select the start? | W1b.a/b/c | No. Digit-1 loss ≈ ln(pairs) in the identity-initialised arms, higher with the even-average init (0.84 / 1.90, P16); slight selection only at tap 21 with 3 pairs (0.97, P15) (P13–P16) |
| 3 | Is the cue present? Do a start store and CA3 settling help? | Probe; W1g (DG start store); W1h (settling) | The name is decodable but W1's addresses discard it (P17). W1g "won" 2 pairs, later shown to be a shortcut (P18). Settling hurt as built (P20). W1 is not CA3 (B16) |
| 4 | Why do 3 pairs fail? Do event-gated writes work? | Diagnostic; scattered task; W1i (S&A goal field) | A positional shortcut explains the failure. W1i selects by content at every rank, on one seed (P21, P22) |
| 5 | Does W1i replicate and scale; is the chain loss decay? | Seeds 1–2, 4–5 pairs, DG chain keys, tap 14 at 6,000 steps, LM (W1d), CPU suite (W1c) | The chain loss is interference, not decay (P24). Seeds 1–2 did not learn selection by step 3,000 but did at 6,000 (seed 1 fully, seed 2 partly; P25), so P21 holds across seeds with a longer schedule. 5 pairs selected at every rank and 4 pairs only partly (seed 0, P26). DG chain keys did not help (P27). W1i with the clock passes the CPU must-pass checks, one seed (P28). LM perplexity 12.29, a narrow E8 miss, confounded by the tap (P29). Tap 14 learns partial selection at 6,000 steps (P23) |

| 6a | Does W1i generalise beyond its training format (rule E6)? | Evaluation only: seq 512–2048, 1–5 codes, natural filler; 3 checkpoints with windowed and full controls | Yes for length (gaps up to about 2,000, digit 1 0.98–1.00; P30) and for code count (P31). No for natural filler: exact 0.09–0.45, because the event detector fires about 4× more and the goal field is overwritten (P32) |

| 6b | Does training on natural filler fix P32; does a write budget help? | W1i trained on natural, mixed, natural + refractory 0.95, natural + budget, random + budget (6,000 steps); evaluated on random and natural | When selection is learned, natural training gives robustness to both fillers (0.93 / 0.89; with refractory 0.95, 0.94 / 0.91; P33). But only 2 of 4 natural-type runs learned it (P34). The write budget starved the detector from initialisation and killed selection (P35) |

| 7 | Is the best recipe reliable (DEC2 criterion 1)? | Recipe (natural, refractory 0.95, 6,000 steps) seeds 1–4; lr 2e-3; event bias −5 with and without a budget | No: 1 of 5 seeds (P36). lr 2e-3 (P37), a sparse-start detector and a non-binding budget (P38) did not help. First round under DEC2 to miss criterion 1 |

| 8 | Can the barrier be crossed: explicit CA1 novelty, a curriculum, or both? | W1j (CA1 comparator drives the event gate), curriculum (1 → 2 → 3 codes), both; 5 seeds each, natural recipe | Curriculum 3 of 5 (P40), W1j 0 of 5 (P39), both 1 of 5 (P41). Second round under DEC2 to miss criterion 1: **the track is closed by the stop rule** |

| 9 | Does a different curriculum schedule reach 4 of 5? (DEC3) | W1i natural recipe with `1:1500,2:3000`, `2:3000` and `1:1000,2:2000`; seeds 0–3 (seed 4 cut once no arm could reach 4 of 5). Before it: why W1j failed (P42); after it, the meaning-trigger diagnoses (P43–P45) | No: 1, 2 and 1 of 4 seeds (P46–P48). Schedules reshuffle which seeds learn. First round under DEC3 to miss criterion 1; DEC4 opens new mechanisms for round 10 |

| 10 | Does an episodic-slot memory that combines the merit ideas learn reliably, and does the combination beat each idea alone? (DEC4, DEC6, DEC7) | W1s: Stage 0 smoke, Stage A (C, leave-one-out A1–A9, isolation I1–I3, B0, reinstatement R, Ro) on seeds 1–2, Stage B on seeds 0–4 | Stage A: **the minimal slot arm I1 reaches exact 0.86–0.98 on every cell on seeds 1–2** (P51). The combination C fails because gate centring adapts to the filler (P50). The extras never help on top of slots (P52). Reinstatement needs a better gate (P53). Stage B: I1 reaches 3 of 5 seeds by the strict bar (seed 1 at 0.94), with grid means 0.93–0.96 on the learned seeds; held-out names 0.58–0.74; A2, I1K and RI do not beat it (P54–P58) |

| 11 | Does time as data, supersession, compression or a name pool fix I1's weaknesses (length, updates, held-out names)? (DEC9; runbook revision 3) | Code and an adversarial review (16 findings, 14 fixed); Stage 0, evaluation-time gates G0.1–G0.8 on the round-10 checkpoints, plus a keep-scorer toy | Stage 0 (P59–P67): the age penalty, not alignment, limits long-range recall at evaluation time; every penalty form trades length against updates; no update or order task has a full-attention ceiling; supw dropped (false matches 3–4%); value-norm keep and a 64-d code are free; ordinal time generalises in the CPU toy and rotary time does not. Waiting for Daniel before Stage A (RV4 §6 proposes reordering it) Stage A (P68–P76): A15 (teacher attention target) is the only arm better or tied on every group on both seeds; the seed-4 gains are mostly rescue; dropping the age penalty costs updates; compression fails when learned from scratch. Stage A2 (DEC14) follows |

Each round's entry in NOTEBOOK has the full tables. Update this table at each checkpoint.
