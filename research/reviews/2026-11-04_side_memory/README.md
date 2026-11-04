# RV4: review of the side-memory track through round 10 (W1 → W1s)

*Review of `main` at `4ff90a4` (rounds 1–10 snapshot, PR #5), 4 Nov 2026.*

- *Sources read:* `research/REGISTER.md` (P1–P58, DEC1, B1–B18, W/L backlog), `research/side_memory_summary.md`, `research/HANDOFF.md`, `research/plans/round11_plan.md` (revisions 1–2), `experiments/evolve/w1s_slots.py`, `experiments/evolve/eval_by_query.py`, `experiments/pretrained_loop.py`.
- *Computation:* no GPU runs; the numbers computed here are arithmetic (E4, E5, E12).
- *Status:* frozen once committed, like the other reviews.
- *Finding IDs:* RV4.E (errors and overstatements), RV4.N (novelty), RV4.R (refinements from the literature), RV4.P (plan changes). The open ones are in the RV3 table of the register.

## 1. Verdict

W1s/I1 is a real advance. A frozen, windowed Qwen3-1.7B gets 0.93–0.96 exact on 1–5 facts up to 2,048 tokens (about 30 windows back) on 3 of 5 seeds. The controls are clean (P54), and the state is fixed at 0.89M floats. Four parts of the current reading need correcting before round 11:

1. **"Superposition is the barrier" is not isolated** inside I1 (E1).
2. **The evaluation measures verbatim-prefix copying over 20 names.** The held-out-name failure follows from the design, and paraphrased queries are untested (E2, E3).
3. **The 4-of-5-seed bar cannot separate the recipes being compared** (E4).
4. **The production memory comparison uses the wrong baseline.** The W1s state equals **about 31 tokens** of this model's KV cache. The real comparison is a ~31-token learned-admission cache, not the full 235 MB cache (E5).

The most defensible novel results are not the slot memory itself, which is anticipated by Landmark Attention, InfLLM and HOLA. They are:
- the learnability result: superposed stores learn selection on 1–3 seeds of 5, slots on 3–5 (N3);
- the evidence that surprise and novelty are *anti*-selective write signals for templated declarative facts, against the write rules of EM-LLM and HOLA (N2).

## 2. What the record establishes (short)

| Stage | Result | Rows |
|---|---|---|
| Loop / horizon kernel | Carries one digit string, query-independent | P3–P9 |
| W1 successor store | Chain yes, start selection no (coin flip at ln pairs) | P10–P16 |
| Diagnosis | Name decodable at the cue (0.95–1.00); W1's addresses discard it | P17 |
| W1g / W1h | Positional shortcut; settling hurts | P18–P20 |
| W1i event-gated goal field | Content selection 3–5 codes, length to 2,048; brittle to filler; 1–3 of 5 seeds | P21–P48 |
| Write-trigger diagnostics | Novelty, LM surprisal and CA1 error do not mark facts at init (AUC 0.24–0.48); a linear "declarative binding" direction does (AUC ≈ 1.00) | P42–P45 |
| W1s / I1 | 0.93–0.96 grid means on 3 of 5 seeds; held-out names 0.58–0.74; update task unscorable; combination C fails via gate centring | P49–P58 |

**Checked and sound.**
- *Teacher-forced exact.* Exact match is scored teacher-forced (`pretrained_loop.py`, `_score`), but it requires every answer argmax to be right. In that case greedy decoding produces the same string, so exact equals free-generation exact.
- *Empty-store control.* The empty-store control is 0, so unlike P2 (RV2.X3) W1s's recall is not adapter repair.

## 3. Errors and overstatements (RV4.E)

**E1. The superposition diagnosis is not isolated within I1.**
- *The claim.* The summary says "separate slots remove the overwriting by design".
- *What was tested.* The only superposed-store arm (A4, P52) ran inside combination C, which carried the harmful gate centring (P50). P52's own status says the leave-one-out reading is confounded.
- *Everything else that changed.* I1 differs from W1i in more than the store: hard top-K retention by gate strength, an offset-aligned read (an induction-style "match the current token, emit its stored successor"), windows of 24 entries, and a point gate.
- *Needed.* I1 with `--w1s-store superposed` and I1 with `--w1s-align none`, 2 seeds each, before the slot claim is written up.
- *Supporting theory, scoped.* Jelassi et al. (2024) give a theoretical reason why fixed-state stores fail exact copying. But at a load of 3–5 facts this is a learnability question, not a capacity one (N3).

**E2. The held-out-name failure follows from the design.**
- *The setup.* `TRAIN_NAMES` has 20 names, and the learned key is a 128-d map trained only on queries about those 20. It is in effect a 20-way name classifier.
- *What P55 shows.* Coverage stays 0.95–1.00 while the slot hit falls to 0.58–0.73: storage is name-agnostic, but addressing is not.
- *Consequence.* The DEC2 grid (lengths × counts × fillers) varies everything except the key distribution. Held-out names, and the paraphrase cell of E3, belong in the pass criterion, not in a robustness appendix. Revision 2 of the round-11 plan already makes held-out names a deciding cell; that is right.

**E3. The query restates the fact's own prefix.**
- *The setup.* Every query is the literal "The code for NAME is" (`pretrained_loop.py`, line 55). The read pools the last 8 query tokens against the first 8 entries of each window. The offset alignment then lines the query's "is" up with the stored "is".
- *What is measured.* This is exact-prefix completion from a retrieved window. Real queries are paraphrased ("What was Emma's code?") or indirect.
- *Missing from the summary.* The plain-language summary's "What is not shown yet" omits this, and no evaluation varies the query form.
- *Needed.* An eval-only cell with 3–4 paraphrase templates, admitted only where full attention scores ≥ 0.8.

**E4. The 4-of-5-seed criterion cannot separate recipes.**
- *The intervals.* Clopper–Pearson 95% intervals are [0.15, 0.95] for 3/5 and [0.28, 1.00] for 4/5.
- *Error rates.* A recipe that learns on 80% of seeds fails the bar 26% of the time. One that learns on 60% passes 34% of the time.
- *Readings that rest on it.* "One seed short" (P54) and "schedules change which seeds learn, not how many" (P46–P48) are within this noise. So is the 0.94 against 0.95 seed-1 call.
- *Recommendation.*
  - Run finalists on 10 seeds. At ≥ 8/10, p = 0.6 passes with probability 0.17 and p = 0.9 with 0.93.
  - Report the success rate with its interval.
  - Add a continuous reliability measure: the distribution of steps-to-selection, the "late drop" step.

**E5. The production memory claim uses the wrong comparison.**
- *Full-cache arithmetic.* Qwen3-1.7B's bf16 KV cache is 2 × 28 layers × 8 KV heads × 128 × 2 B = 114,688 B per token, so 235 MB at 2,048 tokens and 3.76 GB at 32k. The summary's figures check out.
- *Equal-memory arithmetic.* W1s's 0.89M fp32 floats (3.56 MB) equal **31 tokens** of that cache. That is less than half of the 64-token window's own KV (7.3 MB).
- *Why that matters.* Any bounded cache is constant-memory: H2O, SnapKV, learned eviction (LKV, KVP), KV Admission (Huang et al. 2025). The task needs about 5 × (name + 4 digits) ≈ 25–30 tokens, so a learned ~31-token admission cache with reinstatement could in principle solve it.
- *Evidence for that.* The oracle-gate reinstatement arm Ro already reaches 0.92–0.94 at 512 tokens and 0.84–0.90 at 2,048 (P53).
- *What would be distinctive.* Three things, and none is yet shown against that baseline:
  - one layer's residual windows instead of all-layer KV;
  - a write-time admission policy learned through a frozen model;
  - successor replay.

  W16 must use the equal-byte budget (≈ 31 KV tokens) as its axis.

**E6. "No change to the base model" is shown only when the memory is off.**
- The memory leaves the output bit-for-bit unchanged when switched off.
- Round 10 has no LM-regression run (rule E8) and no CPU-suite result for I1, although Stage B listed both (NOTEBOOK, round-10 Stage B plan). No such files exist under `results/evolve/round10/`.
- Whether the always-on additive path hurts ordinary text is untested. P29 was W1i, with a narrow E8 miss.
- Run E8 for I1 before round 11.

**E7. P56: full attention preferring the old code is expected, and the planned rule may throw away the interesting result.**
- *Prior evidence.* Wang & Sun (2025) show proactive interference in LLMs: accuracy on the latest value falls log-linearly as updates accumulate, and prompting does not fix it. A 1.7B base model preferring the old code (new 0.39, old 0.57) is that effect, not only a wording problem.
- *The planned rule.* Round 11 scores update rules only on a form with a full-attention ceiling. That would discard the case where a supersession-aware memory **beats** full attention.
- *Recommendation.* Score updates against ground truth (new-exact and stale rate), report full attention as a reference rather than a ceiling, and use PI-LLM-style update streams.

**E8. The novelty result needs its scope stated, which also makes it a contribution.**
- *What P42–P43 show.* LM surprisal, change in h and untrained delta-store error mark the facts' first digits *less* than filler (AUC 0.00–0.48).
- *Scope.* This holds for templated declarative facts, where the value after "is" is expected and the head ("The code for NAME") is the surprising part.
- *Wording.* The register's "for this task" is correct. The summary and any write-up should keep that scope and set it against HOLA (Cui 2026, writes by delta-rule surprise β‖e‖) and EM-LLM (surprise segmentation), which are not cited anywhere in the side-memory track. See N2.

**E9. DEC5's rationale applies equally to the learned gate.**
- *The objection to P44.* DEC5 rejected P44's fixed direction because another dialect or domain "would break it".
- *The same objection to the gate.* The learned gate, trained on one template and 20 names, has the same exposure. P32 shows a learned write policy breaking on a filler shift, and P50 shows an adaptive one doing so.
- *A fair test.* Both on held-out templates and domains.
- *A middle route.* Initialise `w_g` from the P44 contrast and train it. That is a prior, not a frozen component, and is worth a DEC revisit, since it may also fix seed reliability (P42: "an optimisation problem").

**E10. The window note omits the closest production setting.**
- *What the summary says.* "Production local-attention windows are 1,000–8,000".
- *The omission.* gpt-oss alternates 128-token sliding-window layers with full layers (OpenAI 2025), so a 64-token window is only 2× from a shipping model. Gemma 3 uses 1,024 with a 5:1 local:global ratio.
- *What production actually looks like.* The production pattern is a hybrid: a subset of layers windowed (or GDN), with the global layers' KV being the cost. Every W1s experiment windows *all* layers. Stage C should include "windowed layers only, global layers KV-capped" (P6 below).

**E11. The compression claims rest on a near-zero-entropy value.**
- A 4-digit code is 13.3 bits.
- The round-11 diagnostics ("index dominates", "about 5 of 24 entries are ever read", "39× compression at no cost") are measured on that value.
- Compression arms A5–A8 should not be read as production compression until values are multi-token and open-ended.

**E12. An eval-only check on repeated digits.**
- 49.6% of 4-digit codes repeat a digit.
- The offset read aligns the current token with a stored one and returns its successor, so a repeated digit could create two candidates.
- The alignment code is the contextual tap-21 state (`window_mean(x, …)` in `_features`), not the token, so this is probably resolved. Still, exact split by "code has a repeat" is free to compute.
- It may explain the blunter alignment of the failing seeds (P58: log β_align 2.39 against 2.65–2.77).

## 4. Novel contributions (RV4.N)

**N1. A retrofit exact episodic memory for a frozen windowed LLM, learned through the frozen model.** Write-time admission (gate) + K ordered windows from one layer's residual + content key + successor replay into the residual, < 1M parameters, bit-identical when off. Relation to prior art:

| System | Trained | Base model frozen | Write selection | Memory | Read |
|---|---|---|---|---|---|
| Memorizing Transformers (Wu 2022) | Yes | No | None (all) | Growing | kNN on (k, v) |
| Landmark Attention (Mohtashami & Jaggi 2023) | Yes | Fine-tuned | None (blocks) | Growing | Landmark-gated blocks |
| InfLLM (Xiao 2024) | No | Yes | None (all blocks, offloaded) | Growing | Representative tokens |
| EM-LLM (Fountas 2025) | No | Yes | Surprise segmentation (all kept) | Growing | Similarity + contiguity |
| AHN (Fang 2025) | Yes (self-distillation) | Yes | None; compresses every out-of-window token | Fixed, lossy | RNN state |
| HOLA (Cui 2026) | Yes, from scratch | — | Surprise β‖e‖, every layer | Fixed, exact | Sharpened KV read |
| NSA / DSA (Yuan 2025; DeepSeek-AI 2025) | Yes | No | None (read-time top-k) | Full cache | Learned indexer |
| KV Admission (Huang 2025) | Yes | — | Learned admission | Bounded | Attention |
| **W1s / I1** | Yes (side only) | **Yes** | **Learned admission at events** | **Fixed (≈ 31 KV tokens)** | **Content key + successor replay, one layer** |

The combination "frozen base, learned admission, fixed and exact, successor replay" is not in this list. Each part has precedent, so the claim is the combination and the measurements, not any single mechanism.

**N2. Surprise and novelty are anti-selective write triggers for declarative facts (P42–P43), while a declarative-binding direction is near-perfect (P44–P45).**
- *Why it matters.* This contradicts the write rules of EM-LLM and HOLA for one important content class: IDs, codes, values bound to names.
- *Its scope.* HOLA's residual is computed in a state co-trained from scratch. P42 shows novelty becomes selective only *after* the store is trained, so HOLA may be self-consistent where a frozen retrofit is not.
- *Testable prediction.* In a frozen retrofit, a HOLA-style surprise-admitted cache under-stores templated facts. This is one W16 arm (R8).

**N3. Superposition hurts the learnability of selection far below capacity.**
- W1i solved 3–5 codes on some seeds (P21, P26, P33), so capacity was not the limit. Slots changed the seed success rate.
- This "learnability, not capacity" reading is less studied than the Zoology and Jelassi capacity results.
- It is publishable once E1's isolation arms are run.

**N4. Adaptive write thresholds lose facts under distribution shift (P50).**
- Causal centring of the gate adapted to the filler distribution and stopped retaining facts on random filler.
- This is a general caution for admission and eviction policies that normalise scores online: a running mean or deviation is part of the policy and shifts with the stream.

**N5. A length failure from an additive age penalty in a memory read (round-11 diagnostics).**
- It is the ALiBi failure moved to episodic retrieval: about 1.5 logits at age ~1,800 against a 0.3–0.45 margin.
- Revision 1's move to time as data matches CoPE (Golovneva et al. 2024), which counts gated events rather than tokens. That supports A1b (ordinal rank) over A1 (rotary token time).

Still open from RV2: the S&A goal-field pathfinding over a successor memory (L6). W1s's explicit within-slot successor structure now makes the cross-slot k-hop version feasible (G0.7's next-fact task).

## 5. Refinements from the literature (RV4.R)

**R1. Train on self-distillation from full attention, not only on the needle task.**
- *Precedent.* AHN trains only its memory module on a frozen open LLM, minimising divergence from the full-attention model's outputs on long text, and randomises the window size for generalisation (Fang et al. 2025).
- *For W1s.* The loss becomes KL(full-attention logits ‖ windowed+memory logits) on long general text, with the needle task as a fraction of batches.
- *What it addresses at once:*
  - held-out names and paraphrase (E2, E3), with diverse keys;
  - E8 (E6), with dense supervision at every token where full attention beats windowed;
  - open content (E11);
  - the gate's sparse gradient (R2).
- *Rule check.* It respects rule E4, since inputs stay hidden states and the teacher is used only in training.

**R2. Give the gate a dense training target.** Today the gate gets gradient only through log p_k of slots that were retained and read. That is the chicken-and-egg P42 describes. Options, all training-time only:
1. **Teacher attention mass.** The full-attention model's attention from answer and query positions onto past tokens, summed per candidate window, as a KL target for p. DSA trains its lightning indexer by KL to dense attention before sparse training (DeepSeek-AI 2025); ForesightKV and LookaheadKV supervise eviction with future attention. DEC8 already admits attention weights as inputs, and as a training target they cannot leak at inference.
   - *Distinct from round 11's falsified idea.* "Received attention does not mark the code" (round-11 diagnostics) concerns the *windowed* model's attention inside a slot. The full model's attention from the answer must mark the fact, because that is how full attention answers. Check it as a Stage-0 gate.
2. **Dense-to-sparse annealing of K.** Start with K large (store every candidate window, InfLLM-like) so keys and read learn first, then anneal K to 16. Admission then learns from read utility (Moschella et al. 2026 rank by future utility).
3. **A relaxed top-K** (Gumbel top-k or a straight-through estimator) on retention, so evicted windows also get gradient.

**R3. Use the base model's own attention projections as keys (new arm A13).**
- InfLLM reads its memory with the frozen model's own attention keys from representative tokens, with no training, and works on arbitrary content.
- For W1s: key = a frozen tap-21 head's K-projection, pooled over the window; query = the same head's Q at the answer position.
- Name generalisation then holds by construction, and paraphrase robustness is inherited from the base model.
- *Biology.* The barcode work separates a sparse, episode-unique index from content (Chettih et al. 2024; Fang et al. 2024). In that reading the slot ID is the index, and the content key should be the cortex's own (frozen) representation, not a new map fitted to 20 names.

**R4. Make reinstatement a two-stage method.**
- The additive path's 32-d values suffice for digits but not for open content.
- Reinstatement works with an oracle gate (Ro 0.92–0.94 at 512) and fails with a learned one (R 0.54–0.66, RI 0/5).
- Train I1's gate and keys with the additive path, freeze them, then train reinstatement on top. That separates "where to write" from "how to read back".
- This is the route to multi-token values (Landmark, InfLLM and HOLA all read memory through attention).

**R5. Time and updates.**
- CoPE supports event-counted (ordinal) time over token time (A1b).
- For updates: PI-LLM streams (Wang & Sun 2025) as the task. Sleep-inspired curation of stale associations (Xie 2026) is the closest precedent for A2/A3 supersession.
- For order: SORT and temporal-context reinstatement (Pink et al. 2026) as the A9 task, which has published LLM and human numbers.

**R6. Biology that fits the current design** (useful for the write-up, not new arms):
- awake sharp-wave ripples select which experiences are kept (Yang et al. 2024): write-time admission;
- behavioural-timescale plasticity potentiates inputs over a seconds-long window around a plateau (Bittner et al. 2017): the gate onset plus a window of n entries, including the pre-record P;
- indexing theory, updated (Teyler & Rudy 2007): the slot as an index into replayable cortical states.

**R7. Evaluate on public benchmarks.** RULER multi-key, multi-value and multi-query with paraphrased questions, and PI-LLM, before any production claim.

**R8. W16 baselines at equal bytes (about 31 KV tokens fp32, 15.5 in bf16):**
1. windowed base plus an H2O, SnapKV or learned-admission cache (LKV, KV Admission) of the same bytes, reinstated into the windowed layers;
2. a HOLA-style surprise-admitted exact cache with W1s's read: the N2 test;
3. AHN-GDN on the same frozen Qwen3 at equal state;
4. InfLLM at its actual (growing) memory, reported in bytes.

## 6. How round 11 and the backlog should change (RV4.P)

1. **Close the Stage-B gaps first** (no new mechanisms):
   - the E8 LM run and the CPU suite for I1 (E6);
   - I1-superposed and I1-no-align, 2 seeds each (E1);
   - eval-only cells: paraphrased queries (E3) and the repeated-digit split (E12).
2. **Change the reliability bar** to ≥ 8 of 10 seeds for finalists, reported with an interval (E4). This is a DEC, since DEC2 is fixed at 4 of 5.
3. **Reorder the arms:**
   - *First, generalisation and training signal.* A10 (name pool), A11 (gated lexical keys), new A13 (native keys, R3), new A14 (self-distillation, R1) and new A15 (teacher attention-mass admission target, R2.1).
   - *Promote A12 (sharper alignment) from conditional to core.* P58 puts the long-range failure of seed 4 in the alignment, not in ω.
   - *Second, compression* (A5–A8). Treat it as provisional until values are open-ended (E11).
   - *Third, time* (A1, A1b, A2, A3), with A1b favoured (N5).
4. **Updates:** drop "update rules only on a form with a full-attention ceiling". Score new-exact and stale rate against ground truth, with full attention as a reference (E7).
5. **Stage C window sweep (W15):**
   - add 128 (gpt-oss) beside 256 and 1,024;
   - add a hybrid configuration: only alternate (or 5 of 6) layers windowed, the rest full with KV capped at the window plus W1s;
   - randomise the window during training (AHN).

   The parallel form's (B, T, T) weight matrices make training quadratic, so the longest Stage-C lengths need the streaming form or chunking.
6. **W16:** use the equal-byte axis and the four arms of R8. It is the experiment that decides whether the production section of the summary holds.
7. **Backlog consequences of W1/W1s:**
   - *Close* W9 (novelty- or salience-gated writes), refuted by P42–P43, and W1e and W4, superseded by W1s.
   - *Deprioritise* W8 (a superposed multi-timescale loop) given N3.
   - *Reframe* L5 (state-space injection; W1 against δ-mem) as W1s against δ-mem at equal bytes.
   - *Keep* L6 high: the remaining novelty gap, now buildable on W1s's successor structure.
   - L2's adapter controls remain open for P2, but W1s is not exposed to that confound (§2).
   - W2 (the clock on Qwen3.5) should wait for A4's verdict on the TimeChannel.

## 7. How the side memory could help production models

- **Targets.** Hybrids whose cost is the global layers' KV:
  - gpt-oss (alternating 128-token windows);
  - Gemma 3 (5:1 local:global, 1,024);
  - GDN and linear hybrids such as Qwen3-Next, Qwen3.5 and Kimi Linear (3:1).

  A retrofit like W1s sits beside the windowed or linear layers and supplies exact recall of selected moments, so the global layers' KV can be capped. HOLA shows the from-scratch version of the idea; W1s is the frozen-model retrofit, which a deployment can add without retraining the base.
- **What it adds over compressive memory (AHN, GDN state, δ-mem): exact, ordered recall.** Successor replay returns the stored token sequence, not a blend. That matters for IDs, file paths, account numbers, numeric constraints and quoted strings in agent sessions, support chats and log streams. These are exactly the items proactive interference and compression corrupt.
- **What it adds over read-time sparse attention (NSA, DSA): bounded memory.** Those keep the full cache and save compute. W1s bounds the cache with a learned admission policy (the "admission" primitive of Huang et al. 2025).
- **Operational properties:**
  - detachable and bit-identical when off;
  - under 1M parameters;
  - a state of about 31 KV-token equivalents, and roughly one if the 40× compression survives open content;
  - per-session, so it suits on-device and multi-tenant serving.
- **Policy insight for existing eviction systems.** Surprise- and attention-score heuristics under-weight templated declarative facts (N2). A cheap "declarative binding" probe (P44) could serve as an admission feature in production KV eviction, including Daniel's KV-trimming project for never-ending chats.
- **What a production version still needs:**
  - training on diverse data by self-distillation (R1);
  - native keys (R3);
  - a reinstatement read for open content (R4);
  - E8 with the memory on;
  - wins at equal bytes over learned eviction (R8);
  - measured latency.

## References

Every key below is in `docs/references.bib` and `REFERENCES.md` (section "Side-memory review RV4"): `fang2025artificial`, `cui2026hippocampus`, `xiao2024infllm`, `mohtashami2023landmark`, `wu2022`, `fountas2025human`, `yuan2025native`, `deepseek2025v32`, `huang2025kv`, `zhou2026lkv`, `moschella2026learning`, `zhang2023h`, `xiao2023efficient`, `he2025alleviating`, `kimi2025kimi`, `openai2025gptoss`, `gemma2025gemma3`, `wang2025unable`, `xie2026learning`, `pink2026temporal`, `fallahpour2026memorilla`, `chettih2024barcoding`, `fang2024barcode`, `teyler2007hippocampal`, `yang2024selection`, `bittner2017behavioral`, `jelassi2024repeat`, `golovneva2024`, `olsson2022context`, `lei2026mem`. ForesightKV (arXiv:2602.03203) and LookaheadKV (arXiv:2603.10899, ICLR 2026) are cited in R2 by arXiv ID only.
