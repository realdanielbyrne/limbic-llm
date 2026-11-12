## Five-line summary

1. Beyond-window cued retrieval of one stated fact is a valid test of the read path, but it is the test a dense retriever is built for. It measures "what" only, so it cannot separate a hippocampus-like circuit from bge.
2. The published episodic benchmarks put the difficulty elsewhere: order, "latest state", several near-identical events, and abstention. Retrieval pipelines there are weak or at chance (SORT vanilla RAG 55–67% on a two-way choice; BABILong RAG below chance on qa2/qa3; Huet et al. latest-state exact match at most 36%).
3. Published retrievers do not systematically prefer the superseded fact; they are roughly indifferent to currency (HoH: bge-base hits current 0.70 and outdated 0.53 at top-5). Your 13–25% new-above-old is worse than indifference, which points to an item confound in form, length or wording.
4. Fixed-state models do lose exact recall beyond the window (Griffin phone-book lookup fails past its 1,024 window), and the RecurrentGemma paper reports no long-range test at all. The needle task is therefore worth keeping as a regression check and "what" baseline.
5. Best separating tests, in order: update under interference, sequence-order recall, lure discrimination among near-duplicate events, abstention on absent memories, and ordered multi-fact state (BABILong qa2/qa3).

**Verification.** Numbers were read from the arXiv HTML or abstract pages through a page-reading tool; I did not view the raw tables or figures. Items I could not verify are marked [unverified].

## 1. Candidate tests

| Test | Function isolated | Plain dense retriever: expectation and why | Published reference numbers | Data, licence, size | Cost on a frozen 2B model, 8K input |
|---|---|---|---|---|---|
| **SORT / Book-SORT** (Pink et al. 2024) | Temporal order and recency judgement | About chance: similarity carries no order | Humans 0.64 (50-word segments) and 0.56 (20-word), about 7.5 days after reading. No-text baseline 0.49–0.57. In-context up to 0.95 on short excerpts, 0.56–0.79 at 10–20K words. Vanilla RAG 55–67%; oracle order-preserved retrieval about 90%. Fine-tuning on the book stays near chance | HF `memari/booksort`, CC0-1.0; 9 books, about 36K pairs | Ranking-only two-way choice. The 2,500-word excerpts fit; 10,000-word ones exceed 8K, so cut custom excerpts from the same books |
| **PI-LLM** (Wang & Sun 2025) | Updating; resisting proactive interference | Chance between old and new if forms match | Accuracy declines log-linearly toward zero with update count; errors are overwritten values; "forget" prompts change accuracy by under 10 points | GitHub `zhuangziGiantfish/Unable-to-Forget`; synthetic, 46 keys, 3–400 updates per key; licence not checked | Trivial to generate at 8K; ranking or one-word generation |
| **FactConsolidation** (MemoryAgentBench, Hu et al. 2025) | Selective forgetting, "which is current" | Low | Single-hop: GPT-4o 60.0, BM25 48.0, HippoRAG-v2 54.0, embedding RAG 18–29, Zep 7.0. Multi-hop: 7.0 or less for all of these. o4-mini at 6K: 100 single-hop, 80 multi-hop | HF `ai-hyz/MemoryAgentBench`; paper states CC BY 4.0 (intended); 6K, 32K, 64K and 262K versions | The 6K version fits; about 50 generated answers |
| **Episodic Memories benchmark** (Huet et al., ICLR 2025) | What-where-when binding, several events per cue, latest state, zero-event abstention | Good on single-event cues; flat as matching events grow | Long book, F1 by matching events (0 / 1 / 2 / 3–5 / 6+): GPT-4o in-context 0.84 / 0.81 / 0.60 / 0.57 / 0.53; Claude 3.5 Sonnet RAG 0.91 / 0.59 / 0.59 / 0.59 / 0.62. Latest-state exact match at most 36%; all-states at most 18%. Repo README: gemini-2.5-pro 0.968 simple recall, 0.796 chronological; gpt-4o 0.670 and 0.204 | GitHub `ahstat/episodic-memory-benchmark`, MIT; 20 chapters (about 10K tokens, 456 QA) and 200 chapters (about 100K, 686 QA) | The short book slightly exceeds 8K; generation with judged F1, so the most expensive here |
| **LongMemEval** (Wu et al., ICLR 2025) | Extraction, multi-session, temporal reasoning, knowledge update, abstention | Session recall is fine (Contriever R@5 0.723, BM25 0.634) | GPT-4o: oracle 0.870, full 115K history 0.606. Zep paper per type, GPT-4o full-context then Zep: knowledge-update 78.2 / 83.3, temporal 45.1 / 62.4, multi-session 44.3 / 57.9 | GitHub and HF, MIT; 500 questions; 115K (S) and 1.5M (M) tokens; turn-level `has_answer` labels and session dates. Per-type counts [unverified] | S does not fit; build 8K items from oracle sessions. Ranking-only is possible from the turn labels |
| **LoCoMo** (Maharana et al. 2024) | Temporal questions and adversarial (unanswerable) questions in dialogue | Helps temporal, weak on adversarial | Human F1 87.9. gpt-3.5-16K long-context: 37.8 overall, temporal 20.3, adversarial 2.1. RAG on observations: temporal 41.8, adversarial 27.7 | CC BY-NC 4.0; 50 conversations averaging 9.2K tokens; 7,512 QA (1,547 temporal, 1,871 adversarial) | Conversations are about 8K; generation |
| **BABILong** qa1–qa3 (Kuratov et al. 2024) | qa1: "what". qa2 and qa3: ordered state across 2–3 facts | qa1 about 60%; qa2 and qa3 below random guessing | LLMs use 10–20% of their context; all LLMs under 85% on qa2/qa3 with background text; best qa3 under 80%. Fine-tuned Mamba-130M and RMT hold on qa1 | HF `RMT-team/babilong`; code Apache 2.0; splits from 0K to 10M, 100 samples per task and length | The 8K split fits; one-word answers |
| **NoLiMa** (Modarressi et al. 2025) | "What" without lexical overlap | Degrades: literal matching is the shortcut | GPT-4o 99.3 base, 89.2 at 8K, 69.7 at 32K. Llama 3.3 70B at 32K: direct question 98.5, one-hop without literal match 56.2, two-hop 25.9 | GitHub `adobe-research/NoLiMa`, CC BY-NC-SA 4.0; 58 question–needle pairs, 7,540 tests per length | Haystack length is configurable; ranking-only |
| **Lure discrimination** (Mnemonic Similarity Task style) | Pattern separation | Should false-alarm on lures | No LLM port found in two searches. Nearest: DRM false memory in GPT-4, critical-lure endorsement 0.114 against 0.053 false alarms (Cao et al. 2025) | Must be built | Ranking-only |

Other benchmarks covered:
- **MemBench** (Tan et al. 2025): knowledge-update accuracy at 10K, participation scenario: FullMemory 0.75, RecentMemory 0.725, RetrievalMemory 0.675. At 100K retrieval leads (0.833 against 0.489 for full memory).
- **PerLTQA**: 8,593 questions for 30 characters; mixes semantic and episodic memory.
- **TimeQA**: best model (FiD) 46% against human 87%.
- **Vendor LongMemEval scores of 85–97%** (Exabase, Supermemory, Mem0 and others) are self-reported and conflict with each other [unverified].
- **EMemBench, MemGround, REMem** (2026) were seen only in search snippets [unverified].

**Paradigms run on models**
- **Contiguity (lag-CRP).** Induction heads fit the CMR model of free recall, with CMR distance 0.05–0.12 for the top heads across 12 models (Ji-An et al. 2024). In 7–9B models the profile is a sharp +1 lag, closer to serial recall than to the graded human curve, and ablating induction heads removes it (Bajaj et al. 2026).
- **Serial position.** GPT-4 showed no primacy or recency curve on lists; fan and list-length effects did replicate (Cao et al. 2025).
- **Associative learning.** Kolling et al. 2025 report non-monotonic representational change in six LLMs (abstract only).
- **Not found.** Transitive inference, source memory, reversal learning and oddball paradigms run on memory-augmented LLMs.

**Bounded-window and recurrent models**
- **RecurrentGemma.** Local window 2,048, trained on 8,192-token sequences, no long-range recall evaluation in the paper.
- **Griffin.** Phone-book lookup is perfect up to its 1,024 window and degrades beyond; Hawk fails as the phone book grows.
- **Mamba versus Pythia.** Pythia-410M beats Mamba-2.8B on phone books of 70 or more entries (Jelassi et al. 2024).
- **Gated DeltaNet.** Uses RULER's S-NIAH: pass-key 91.8 at 8K, number-in-haystack falls from 92.2 at 4K to 29.6 at 8K, UUID 27.6 at 4K.
- **State tracking.** State-space models cannot express permutation composition (Merrill et al. 2024).
- **xLSTM.** Appendix uses formal languages, MQAR and Long Range Arena; numbers [unverified].

## 2. Is beyond-window cued retrieval of one stated fact a fair or informative test?

**Not as the headline.**

- **What it measures.** Content-addressable cued recall after one exposure, beyond the window. That covers two of the five properties Pink et al. (2025) give for episodic memory (long-term, single-shot) and neither of the two that distinguish it from semantic memory (instance-specific, contextual relations of when and where).
- **Why a retriever wins.** It is the retriever's training objective. NoLiMa shows how much lexical overlap carries: 98.5% with a literal match against 56.2% and 25.9% without. RULER reports near-perfect vanilla needle scores for models that fail its harder tasks. Your bge result of about 1.0 on repeated wording is the same effect.
- **What it does not measure.** Order, recency, currency, discrimination among near-duplicates, and absence. In the literature these are where retrievers fall to chance.
- **Value in keeping it.** Yes, as a regression check of the read path and as the "what" baseline, with bge reported as a reference rather than a target. It is a real deficit of the base model class (Griffin, Jelassi, Gated DeltaNet above), and RecurrentGemma has no published number for it. A low-overlap variant in the NoLiMa style makes it informative.

**On your update result.** LongMemEval is not evidence that retrievers fail on updates. Knowledge-update is one of its easier categories (78.2% for full-context GPT-4o), because sessions carry dates and a reader LLM resolves the conflict. Its authors note retrievers tend to find the updated turn and miss the older one. HoH shows near-indifference. The closest published analogue to your failure is Huet et al.'s latest-state score of at most 36%. A systematic preference for the old statement suggests a wording, length or literal-match asymmetry between the first and the updating statement; this is my inference, not a published finding. Fayyaz et al. (2025) measured exactly those biases: Contriever and Dragon+ pick the answer-bearing document in under 10% of cases when brevity, literal-match, position and repetition biases combine.

## 3. Ranked shortlist

| Rank | Test | Control that makes it fair | Number to beat |
|---|---|---|---|
| 1 | **Update under interference** (PI-LLM design, FactConsolidation 6K, LongMemEval knowledge-update turns) | Old and new statements identical in form and length, templates counterbalanced, so a content-only selector sits at 50% by construction. Vary the number of intervening updates and the distance. Add a "latest mention" oracle as ceiling | Retriever 50% by construction; HoH outdated hit at top-5 above 50%; FactConsolidation single-hop: BM25 48, HippoRAG-v2 54, GPT-4o 60; Huet latest-state at most 36% |
| 2 | **Sequence order recall** (SORT) | No-text baseline (0.49–0.57) to remove order guessable from content; distance bins; counterbalanced presentation order (the dataset field `present_seg1_first`) | Vanilla RAG 55–67%; humans 0.64; in-context 0.56–0.79 at 10–20K words |
| 3 | **Lure discrimination among near-duplicate events** (Huet multi-event cues, or an MST-style old/similar/new set) | Score "similar" on lures minus "similar" on foils, to remove response bias. Lures share entity and template and differ in one attribute | Huet F1 at 2 or more matching events: 0.53–0.62; no LLM MST baseline exists |
| 4 | **Abstention, and knowing recall is wrong** | Each unanswerable item has an answerable twin of the same form; report AUC | LoCoMo adversarial: long-context 2.1 F1, human 89.4. Huet zero-event F1: 0.51–0.97. Your comparator AUC 0.74–0.84 is the internal reference |
| 5 | **Ordered multi-fact state** (BABILong qa2/qa3 at 8K) | 0K split and windowed-only rows. bAbI facts are templated, so include a form-detector baseline | RAG below random guessing; best LLM under 80% on qa3 |

**Novelty-gated storage under a fixed budget.** I found no published benchmark that isolates it. The nearest are MemBench's capacity metric and surprise-based writing in Titans and EM-LLM, both read from abstracts only. The control that matters is whether the detector merely finds inserted text:
- LongBioBench argues that needle–haystack non-coherence is a shortcut.
- The Chroma report (a vendor report, not peer-reviewed) found needles that do not blend with the haystack are found more easily in one of two haystacks, and that shuffled haystacks raise scores across all 18 models.
- So the items to store should be drawn from the source text itself, or the templated fillers should be matched in form.

## Sources

- LongMemEval: https://arxiv.org/abs/2410.10813 and https://github.com/xiaowu0162/LongMemEval
- Zep on LongMemEval: https://arxiv.org/abs/2501.13956
- LoCoMo: https://arxiv.org/abs/2402.17753
- Huet et al.: https://arxiv.org/abs/2501.13121 and https://github.com/ahstat/episodic-memory-benchmark
- SORT: https://arxiv.org/abs/2410.08133 and https://huggingface.co/datasets/memari/booksort
- Pink et al. position paper: https://arxiv.org/abs/2502.06975
- BABILong: https://arxiv.org/abs/2406.10149
- MemoryAgentBench: https://arxiv.org/abs/2507.05257
- MemBench: https://arxiv.org/abs/2506.21605
- PerLTQA: https://arxiv.org/abs/2402.16288
- TimeQA: https://arxiv.org/abs/2108.06314
- NoLiMa: https://arxiv.org/abs/2502.05167
- PI-LLM: https://arxiv.org/abs/2506.08184
- HoH: https://arxiv.org/abs/2503.04800
- Collapse of Dense Retrievers: https://arxiv.org/abs/2503.05037
- RULER: https://arxiv.org/abs/2404.06654
- Michelangelo: https://arxiv.org/abs/2409.12640
- LongBioBench: https://arxiv.org/abs/2506.02921
- Chroma Context Rot: https://www.trychroma.com/research/context-rot
- Ji-An et al.: https://arxiv.org/abs/2405.14992
- Bajaj et al.: https://arxiv.org/abs/2604.01094
- Mistry et al.: https://arxiv.org/abs/2502.06902
- Cao et al.: https://arxiv.org/abs/2509.17138
- Kolling et al.: https://arxiv.org/abs/2510.21408
- RecurrentGemma: https://arxiv.org/abs/2404.07839
- Griffin: https://arxiv.org/abs/2402.19427
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Jelassi et al.: https://arxiv.org/abs/2402.01032
- Merrill et al.: https://arxiv.org/abs/2404.08819
- xLSTM: https://arxiv.org/abs/2405.04517
- Titans: https://arxiv.org/abs/2501.00663
- EM-LLM: https://arxiv.org/abs/2407.09450
- Entity tracking (Kim & Schuster): https://arxiv.org/abs/2305.02363
- Minerva: https://arxiv.org/abs/2502.03358

**Caveats on specific numbers**
- The Zep per-type table has internal inconsistencies (identical single-session-user rows for both models; deltas that do not match the scores).
- MemoryAgentBench renamed "Conflict Resolution" to "Selective Forgetting" across versions; figures are from Table 3 of v4, and I did not confirm which context length that table uses.
- The SORT page text was garbled on the book count; the abstract says 9 books.
- Kim & Schuster and Minerva were read from abstracts only and gave no numbers.
