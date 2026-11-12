# Round 14: the circuit under a bounded store, on a cached bench (draft 2, 12 Nov 2026)

Status: **draft 2, revised after an adversarial review (Opus, 12 Nov 2026; register RV7). Proposed by the GPU session at Daniel's request ("structure the next round with appropriate experiments and metrics"). Not approved; nothing built or run.** The review's verdict on draft 1 was "run after changes"; §10 lists its findings and where each is answered.

## 1. Where round 13 left us

- Ranking the queried fact beyond the window (P102): `bge-small` 0.89 top-1, the model's best attention head 0.43–0.52, the circuit's best key 0.32–0.47, the dentate key 0.07–0.09. The model's query-key scores retrieve only on repeated wording (layers 5 and 11) and are at chance on paraphrase at every layer.
- Parts that carried signal in isolation: the comparator (AUC 0.74–0.84 for "this recall is right"), the context tag (0.75–0.77, distance-matched), a novelty salience (0.93–0.95, confounded by insertion).
- The literature review (`research/reviews/2026-11-12_hippocampus_models/`): the dentate code is for separating memories at write time in a capacity-limited store, not for matching cues; cue matching is the cortical encoder's job; hippocampal models are expected to win on updating, time and order, near-duplicate events, novelty-gated storage and rejecting wrong recalls.
- Process: a 90-second smoke had the answer that five GPU-hours confirmed.

## 2. The lesson from the review of draft 1, and the one question

Draft 1 kept an exact store of every span with its position. On such a store, a retriever plus a stored integer is unbeatable by construction: "find the entity, take its latest statement" solves updating, and a wider retrieval window solves context binding. No time code can beat a stored position, and no sparse code can improve an exact lookup. **The circuit's parts are answers to a problem that an unlimited exact store does not have.** They can only earn their place when memory is bounded, which is also the programme's oldest rule (E2, a fixed-size state).

**Question:** under a fixed store budget, does the circuit's write policy (novelty gate, supersede-at-write, comparator) keep and return the right things better than one-line policies, at equal budget, on a model reading beyond its window?

Each circuit part is credited only if it beats its one-line stand-in with everything else fixed:

| Circuit part | One-line stand-in it must beat |
|---|---|
| Supersede-at-write (key match high, value mismatch high → replace the older entry) | replace a stored span when its `bge` cosine to the new span exceeds a threshold; FIFO; keep-latest-k |
| Novelty write gate | token rarity (inverse document frequency); surprisal; "this span repeats an entity already seen" |
| Context tag as a second-stage cue | a retriever on a wider window; "find the scene sentence, take the next statement" |
| Comparator as a reject gate | the retriever's top-1 score and margin; lexical presence of the question's words in the top span |
| Integrator time tags | the stored ordinal index (expected to tie at best; tested once, then dropped unless it wins) |

## 3. How the round runs (AlphaEvolve loop with a DS-STAR verifier)

**3.1 Evaluator: a cached bench.**
- One forward pass per item, once. Cached per item, in float16: hidden states at an early and a deep tap for every token; span-pooled states at every block; attention keys at the eight attention layers; queries at the question and answer-position rows; per-token surprisal and the mean RG-LRU gate per layer; `bge-small` embeddings of every span, of ±2-sentence windows and of the question. Labels live in a separate file.
- A **candidate** is a small function over a label-free view. It has two entry points: `write(stream) → store` (sees the document only, in order, under the budget) and `read(store, question) → span scores, gate score`.
- **Automatic leak checks on every candidate:** the store must be bitwise unchanged when the question and all later tokens are changed; outputs at position t must not change when tokens after t change.
- The bench computes every metric, baseline and control, so candidates cannot differ in how they are measured.

**3.2 Archive.** `results/evolve/round14/archive.jsonl`: one line per candidate with its code hash, parent, one-sentence hypothesis, screen metrics and verdict. Proposals start from the best and from the most different survivors.

**3.3 Cascade.**

| Stage | Items | Cost | Purpose |
|---|---|---|---|
| A. Screen | 200 dev items per test | seconds per candidate after a cache build of about 25 minutes | kill or promote |
| B. Confirm | 200 held-out items per test, both models, features computed on the fly | about 30 minutes per model, once per family winner | the reported number |
| C. Generation | 100 held-out items per test | 1–2 GPU-hours in all, winners only | does the gain reach the answer |

Dev and held-out items use **disjoint templates, name pools and value formats**, so a candidate tuned on dev failures cannot key on template strings.

**3.4 One generation of the loop** (at most four per family before stage B):
1. **Diagnose:** which screen items the best candidate fails, by item attribute.
2. **Propose** at most four mutations, each one hypothesis.
3. **Verify:** the leak checks, a toy check of the mechanism, tests through the real code path.
4. **Score** on the screen set.
5. **Route:** promote, mutate again, fix a bug, or kill.
6. **Archive** the verdict.

Before stage B an independent reviewer tries to break each promoted candidate and its test. Roles: Fable proposes and orchestrates; Opus sub-agents implement, verify and review.

**3.5 Bars** (sized for n = 200, where a paired difference has a 95% interval of about ±0.09):
- **Promote:** point gain over the best baseline of that test ≥ 0.20 top-1 (or ≥ 0.10 AUC on T4) and the paired lower bound above 0.
- **Kill:** two successive mutations of a line with gain below 0.05.
- **Regression (T6):** on all 400 held-out round-13 items, the paired upper bound of the loss below 0.05.
- Stage B corrects the selection among up to 16 candidates: one pre-registered winner per family, fresh items, Holm across families.

**3.6 Rules carried over.** Strict causality; float32 side computations; labels and scores only in results; intervals on every reported number. New protocol rule: **no full-size run of a candidate that has not cleared a screen.**

## 4. Tests and metrics

All tests: 8,192-token inputs on `recurrentgemma-2b` (and `-it` at stage B), relevant facts beyond the 2,048 window, and **a store budget** stated per test. Chance is measured on the content-only row, never assumed.

**T1 Update under a budget** (reconsolidation).
- Items: 12 entities; each has its value stated 2–6 times, every statement of an entity in the same template and token length, templates counterbalanced; statements of different entities interleaved.
- Budget: the store holds 25% and 50% of the statement spans.
- Metric: top-1 = the statement of the queried entity's *current* value is in the store and ranked first; stale-answer rate.
- Baselines (bench rows): FIFO; keep-latest-k; `bge` dedup at write; random eviction; and, as the unbudgeted reference, retriever + "take the latest of the entity's statements" on the full store.
- Published companion (needs a download, §9): FactConsolidation, 6K single-hop (BM25 48, embedding retrieval 18–29, HippoRAG 2 54).

**T3 Context binding.**
- Items: the same statement form occurs in 3–4 scenes with different values. Scenes are **interleaved and revisited** (A, B, A, C, B); the statement sits 50–1,000 tokens after its scene is set, with filler between that does not name the scene; the question describes the scene **in other words**.
- Budget: the store holds every statement but not the filler (the binding is the question here, not eviction).
- Metric: top-1 of the statement from the described scene; hit rate minus false-alarm rate on lures.
- Baselines: sentence retriever (chance among lures); retriever on a ±2-sentence window; "retrieve the scene sentence, take the next statement"; oracle.

**T4 Rejecting recalls.**
- Items: answerable items and twins, half each. Twins: the same entity and attribute stated only in another scene; or a near-duplicate name (Marla Penn / Marla Fenn). The selector is fixed to the retriever's top-1 for every gate.
- Metric: AUC for "the top-1 is the answer"; risk at 80% coverage reported against the oracle gate.
- Baselines: the retriever's top-1 score; its top-1 − top-2 margin; presence of the question's content words in the top span; BM25 maximum.

**T5 Write gate on natural text.**
- Items: WikiText-103 articles of at least 8,192 tokens (cached; count at build). "Needed" spans hold the first mention of an entity that is mentioned again more than 2,048 tokens later; entities are runs of capitalised tokens not at a sentence start (stated extractor). Only spans with at least 2,048 tokens of future are scored.
- Budget: 5% and 10% of spans.
- Metric: fraction of needed spans kept, by position stratum.
- Baselines: random; earliest-first; uniform stride; token rarity; surprisal; "repeats an entity already seen".
- Stage C metric: loss on the re-mention tokens with the gated store read back, against the rarity-gated store at equal budget.

**T6 Read-path regression.** Round 13's held-out decoy and paraphrase items; `bge-small` (0.89) is the reference, not a target.

**T2 Order** (stage C only; needs a download, §9): Book-SORT (CC0, natural text; published: plain RAG 55–67%, no-text 0.49–0.57, order-preserving oracle about 90%). A home-made order test is dropped: `readback.py` already keeps the original order of what it reads back, so it would only retest P66.

Fairness rows on every test: a form detector; a position-only rule; lexical overlap; equal statement lengths within an item; items pinned by hash.

## 5. Candidate families

Bench baselines (never candidates): everything in the right-hand column of §2, plus content + ordinal, latest among near-ties, neighbours i ± m, the RG-LRU gate.

| Family | Seeds for generation 0 | Target |
|---|---|---|
| **Supersede** | at write, an incoming span whose key matches a stored entry and whose pooled value state mismatches it replaces that entry; key and value taken at different taps; thresholds from dev | T1 |
| **Write gate** | novelty of the current state against recent context (bank at τ = 4, 16, 64), alone and combined with supersede as the eviction rule | T5, T1 |
| **Context** | context-tag cosine as a second-stage cue after a content hit; tag from different bands and longer time constants | T3 |
| **Comparator** | round 13's agreement score on top of the retriever's top-1; variants of the contrast set | T4 |
| **Separation** (one generation only) | the sparse dentate code as the stored key when entries are superposed in a fixed number of slots, against the dense key in the same slots | T1 at the 25% budget |
| **What** (secondary) | re-rank the retriever's top 20 by a few of the model's own heads chosen on dev, calibrated against a null query (QRHead) | T6 |

Evidence for a part = an ablation with all else fixed, and a win over its one-line stand-in, under the budget.

## 6. Stage C: generation

Winners only, 100 held-out items per test, at 128 and 512 read columns.

Arms: native; **plain RAG as text** (the selected spans prepended in original order, same selector); retriever-only read-back through attention; the circuit's store read back; oracle.

Metrics: exact match; stale-answer rate on T1; false-answer rate on T4's twins and exact match on answerable items; order accuracy on Book-SORT; loss on re-mentions for T5; columns used.

Prediction stated in advance: at 128 columns an unbudgeted read-back holds all of an entity's statements, so on T1 that arm measures whether the model prefers the later column. The budgeted arms are the test of the store.

**The claim the round is trying to earn:** under a fixed store budget, the circuit's write policy returns current, correctly bound facts and declines when it has none, better than one-line policies at equal budget, and the gain reaches the model's answers.

## 7. Exit and stop rules

- **Success:** on at least one of T1, T3, T4, T5, a circuit part beats its one-line stand-in at stage B with a paired lower bound above 0.10 (0.05 AUC on T4), T6 passes, and the stage-C lower bound of the same comparison is above 0.
- **Stop:** if no family clears its screen bar within four generations, the gated-circuit line closes with a negative result: under a bounded store, one-line policies are as good. No round 15 is designed in this direction.
- **Human checkpoints** after stage A and after stage C.

## 8. Cost

- Bench, item generators and baselines: one build session, CPU tests first.
- Screen cache: 4 tests × 200 items ≈ 25 minutes of GPU and about 100 GB on `/mnt/data` (two per-token taps at 84 MB per item plus keys at 33 MB).
- Stage A: seconds per candidate. Stage B: about 30 minutes of GPU per model, no cache. Stage C: 1–2 GPU-hours.

## 9. Needs Daniel

- Approval of the round.
- Downloads: `ai-hyz/MemoryAgentBench` (FactConsolidation 6K) and `memari/booksort` (Book-SORT, CC0). Without them T1 has no published companion and the order test is dropped.

## 10. Review findings (RV7) and where they are answered

| # | Finding | Answer |
|---|---|---|
| 1 | T1 is solved by "find the entity, take the latest"; the time seeds were that baseline; the stop rule could not fire | §2 bounded store; trivial rules are bench baselines; §4 T1 budget; success is against the stand-in |
| 2 | T3's bar was met by the windowed retriever itself | §4 T3: distance 50–1,000 tokens, interleaved and revisited scenes, paraphrased scene, scene-hop baseline |
| 3 | Most seeds were baselines; the dentate ID cannot help an exact store | §5 baselines moved to the bench; separation tested only with superposed slots |
| 4 | T4 twins were solved by string presence; AUC depended on the selector | §4 T4 twins, fixed selector, lexical baselines, oracle-gate reference |
| 5 | T5 favoured position; few documents reach 8K; extractor unstated | §4 T5 long articles, future-length rule, earliest-first and repeated-entity baselines, stated extractor, stage-C metric |
| 6 | Bars did not fit n = 50 or each other | §3.3 screen at 200; §3.5 bars |
| 7 | Leaks through write-time features, shared templates, missing span embeddings | §3.1 leak checks and cached embeddings; §3.3 disjoint pools |
| 8 | The home-made order test only retests P66 | §4 T2 Book-SORT or dropped |
| 9 | Plain RAG as text was missing | §6 arms |
| 10 | Published benchmarks belong in this round | §4 T1 companion, T2; §9 |
| 11 | Cost understated | §3.1 dtype and tensors; §3.3 stage B on the fly; §8 |

## 11. Risks that remain

- One-line policies may simply be as good under a budget. That is a reportable negative and the stop rule covers it.
- T1, T3 and T4 statements are still inserted template sentences; T5, Book-SORT and FactConsolidation are the natural-text checks.
- The comparator may not beat the retriever's own margin.
- The supersede rule needs a key that matches restatements of the same entity; round 13 showed internal keys do that on repeated wording, which is the case here, but it will not extend to paraphrased restatements.
