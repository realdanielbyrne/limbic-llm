# Round 14: time, binding and gating on a cached bench (draft 1, 12 Nov 2026)

Status: **draft 1, proposed by the GPU session on 12 Nov 2026 at Daniel's request ("structure the next round with appropriate experiments and metrics"). Not approved; nothing built or run.**

## 1. Where round 13 left us

- Ranking the queried fact beyond the window (P102): `bge-small` 0.89 top-1, the model's best attention head 0.43–0.52, the circuit's best key 0.32–0.47, the dentate key 0.07–0.09. Model-internal keys are lexical (0.70–0.97 on repeated wording, 0.05–0.20 on paraphrase).
- Parts that carried signal in isolation: the comparator (AUC 0.74–0.84 for "this recall is right"), the context tag (0.75–0.77, distance-matched), a novelty salience (0.93–0.95, confounded by insertion).
- The literature review (`research/reviews/2026-11-12_hippocampus_models/`) says the same: the dentate code is for separating memories at write time, not for matching cues; cue matching is the cortical encoder's job; hippocampal models are expected to win on time and order, updating, near-duplicate events, novelty-gated storage and rejecting wrong recalls. Published retrieval pipelines sit at or near chance on those.
- Process: a 90-second smoke had the answer that five GPU-hours confirmed.

**So the round changes two things: what is asked, and how fast each answer comes.**

## 2. The one question

With content matching ("what") handed to a text encoder, do the circuit's **time code, context binding, comparator and write gate** beat content-only selection on tests where content matching is at chance by construction, at equal read budget, on a model reading beyond its window?

Hypothesis → metric: each part has one target test (§4) with a content-only baseline that sits at chance by design, a ceiling, and a pre-set bar.

## 3. How the round runs (AlphaEvolve loop with a DS-STAR verifier)

**3.1 Evaluator: a cached bench.**
- One forward pass per item, once. The cache holds, per item: hidden states at an early and a deep tap for every token, span-pooled states at every block, attention keys at the eight attention layers, queries at the question and answer-position rows, per-token surprisal and RG-LRU gate values, span text, and retriever scores (`bge-small`). Labels live in a separate file.
- A **candidate** is a small function over a label-free, causal view of one item. It returns span scores, and optionally a gate score and a write mask. Scoring a candidate on the screen set takes seconds; no model is run.
- The bench computes every metric, the fixed baselines and the controls itself, so candidates cannot differ in how they are measured.

**3.2 Archive.** `results/evolve/round14/archive.jsonl`: one line per candidate with its code hash, parent, hypothesis in one sentence, screen metrics, and verdict. This is the programme database: proposals start from the best and from the most different survivors.

**3.3 Cascade.**

| Stage | Items | Cost | Purpose |
|---|---|---|---|
| A. Screen | 50 dev items per test | seconds per candidate | kill or promote |
| B. Confirm | 200 held-out items per test, both models | minutes, once per family winner | the number that is reported |
| C. Generation | 100 held-out items per test | tens of minutes, winners only | does the gain reach the model's answer |

**3.4 One generation of the loop** (at most four per family before stage B):
1. **Diagnose.** Read the archive; list which screen items the best candidate fails and why (the bench prints per-item failures by item attribute).
2. **Propose** at most four mutations, each one hypothesis.
3. **Verify** (DS-STAR's verifier): causality and no-label checks run automatically; a toy check of the mechanism; tests through the real code path.
4. **Score** on the screen set.
5. **Route:** promote (clears the bar), mutate again (within 0.05 of the bar), fix (a bug found), or kill (below the kill line twice).
6. **Archive** the verdict.

Before stage B, an independent reviewer tries to break each promoted candidate and its test (item artefacts, leaks, a weak baseline). Roles: Fable proposes and orchestrates; Opus sub-agents implement, verify and review.

**3.5 Rules carried over.** Strict causality; fixed-size state for the bank; float32 side computations; results hold labels and scores only; dev for every choice, held-out once; intervals on every reported number; Holm across families. New rule for the protocol: **no full-size run of a candidate that has not cleared a screen.**

## 4. Tests and metrics

Every test uses 8,192-token inputs on `recurrentgemma-2b` (and `-it` at stage B), with the relevant facts beyond the 2,048 window. Chance, ceiling and the bar are fixed before any run.

| Test | Function | Items | Primary metric | Content-only baseline | Reference rows | Promote at screen | Kill |
|---|---|---|---|---|---|---|---|
| **T1 Update** | what × when | An entity's value is stated 2–5 times with different values; **every statement of an entity has the same template and token length**; templates counterbalanced; 6 such entities per item; lag between updates varied | top-1 = the statement of the *current* value of the queried entity | 1 / (statements of that entity) by construction; measured, not assumed | latest-mention oracle; position-only (fails: other entities are stated later); form detector | ≥ 0.15 above content-only and ≥ 0.80 current-above-stale | < 0.05 above content-only twice |
| **T3 Context binding** | what-where-when, lures | The same statement form occurs in 3–4 passages with different values; the question names the passage's scene, never words of the statement; the scene is set in the sentences before the statement | top-1 of the statement from the named scene; lure discrimination = hit rate − false-alarm rate on lures | sentence-level retriever: chance among lures | **retriever on a ±2-sentence window** (the plain fix, must be beaten or matched at equal read columns); oracle | ≥ 0.15 above the sentence retriever and within 0.05 of the windowed retriever at ≤ its columns | as above |
| **T4 Rejecting recalls** | match/mismatch | Answerable items and twins in which the queried fact is absent (entity never stated, or stated for another attribute); same form | AUC for "the selector's top-1 is the answer"; risk at 80% coverage | the retriever's own top-1 score and its top-1 − top-2 margin | oracle gate | AUC ≥ 0.05 above the better retriever-confidence baseline | not above it twice |
| **T5 Write gate** | novelty-gated storage | Natural text only (WikiText-103, FineWeb; both cached): spans whose entity is re-mentioned beyond the window are the "needed" spans; store budget 5% and 10% of spans | fraction of needed spans kept | random; most recent; uniform stride | **rarity of the span's tokens (inverse document frequency)** as the cheap strong baseline | ≥ 0.10 above rarity at either budget | not above rarity twice |
| **T6 Read-path regression** | "what" | Round 13's decoy and paraphrase items | top-1 | `bge-small` (0.89) is the reference, not a target | — | every promoted candidate must lose ≤ 0.03 here | — |

Order and recency (T2) has no ranking-stage metric, because a stored ordinal index answers it by construction. It is tested at stage C: questions of the form "which was stated first", read back at ordinal positions against a shuffled-ordinal control and against retriever-only read-back.

Fairness controls on every test: a form-detector row; a position-only row; statement lengths equal within an item; question wording never overlapping the distinguishing words; items deterministic and pinned by hash; chance measured on the content-only row.

## 5. Candidate families (the search space)

| Family | Seeds for generation 0 | Target |
|---|---|---|
| **Time** | content + λ·ordinal; latest among near-ties in content; supersede at write (key match high and value mismatch high marks the older entry, which is then down-weighted); integrator tags with time constants up to 8,192 | T1 |
| **Context** | context-tag cosine as a second-stage cue after a content hit; tag from different bands; reinstating neighbours i ± m; the dentate sparse code as an ID that separates lures (its role in the literature) | T3 |
| **Comparator** | round 13's agreement score on top of the retriever; variants of what it contrasts against | T4 |
| **Write** | novelty against recent context (τ = 4, 16, 64); RG-LRU gate; surprisal; combinations | T5 |
| **What** (secondary) | a second-stage re-rank of the retriever's top 20 by a few of the model's own attention heads chosen on dev, calibrated against a null query (QRHead); deep-layer state cosine | T6 |

Dropped from round 13: the dentate key as an address, the dentate-strength salience, the InfLLM-style arm, sweeps of hundreds of scorers.

## 6. Stage C: generation

Winners only, 100 held-out items per test, read back through attention (`readback.py`) at 128 and 512 columns.

Arms: native; retriever-only read-back (the plain baseline); retriever + the winning time / context / gate parts; oracle.

Metrics: exact match; on T1 the rate of answering with a stale value; on T4 the false-answer rate on unanswerable twins and exact match on answerable ones; on T2 order accuracy; columns used.

**The claim the round is trying to earn:** at equal read budget, adding the circuit's time, binding and gate to content selection improves answers on update, context-bound and unanswerable questions over retriever-only read-back, with no loss on plain recall.

## 7. Exit and stop rules

- **Success:** on at least one of T1, T3, T4, the stage-B lower bound of (circuit − best content-only baseline) is above 0.10, T6 loss is within 0.03, and the stage-C lower bound of the same comparison is above 0.
- **Stop:** if no family clears its screen bar within four generations, the gated-circuit line closes with a negative result and no round 15 is designed in this direction.
- **Human checkpoint** after stage A (which families survive), and again after stage C.

## 8. Cost

- Bench build and item generators: one build session, CPU tests first.
- Screen cache: 250 items, about 6 minutes of GPU, about 35 GB on `/mnt/data`.
- Stage A: seconds per candidate; the whole stage within a working session.
- Stage B: about 30 minutes of GPU per model. Stage C: about 1–2 GPU-hours in all.

## 9. Risks

- T1, T3 and T4 statements are still inserted template sentences. Every arm sees the same items, and T5 uses natural text only, but a published benchmark row would be stronger. Candidates for a later round, each needing a download: Book-SORT (CC0), FactConsolidation at 6K, BABILong qa2/qa3 at 8K.
- A windowed retriever may solve T3 outright. The bar is then "match it at fewer columns", which is a smaller claim.
- The comparator may not beat the retriever's own margin on T4.
- Round 13's update result (retrievers preferring the stale statement) was probably an item artefact; T1's equal-form rule is the fix, and its content-only row must measure at chance before anything else is read.
