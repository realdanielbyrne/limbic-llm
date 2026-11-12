# Hippocampus-inspired memory for LLMs: what works, against what, and why

## Summary
1. Three mechanisms beat a plain dense retriever with ablation support: graph spreading activation (multi-hop only), the reader model's own query-conditioned attention over candidates, and write-time restructuring of what is indexed. None beats dense retrieval on simple single-fact questions; HippoRAG 2 reaches parity there.
2. I found no evidence that a training-free, model-internal key computed independently of the query matches a trained encoder on paraphrase. Untrained LLM states score 10–16 on MTEB retrieval against 38 after unsupervised contrastive training (LLM2Vec). Internal keys do win on exact-match lookups and when the query is in the same forward pass as the candidates.
3. In the ablations, the "hippocampal" parts (surprise segmentation, contiguity buffer, recognition filter, consolidation) are worth about ±1 point. The gain comes from the lookup itself, the graph walk, or the trained embedder.
4. Updates are mostly unsolved and rarely ablated. On an independent benchmark, memory products that invalidate or edit facts scored below BM25 on conflict resolution (Zep 7, Mem0 18, BM25 48).
5. Of the trained side memories, δ-mem, MemoryLLM, M+, Larimar and Hippoformer report no parameter-matched no-memory control. Only Memoria has a comparable control (random wiring), and Titans compares at matched model size.

**Verification:** numbers were read from arXiv full text through a page reader, not from PDFs by eye. Items I could not verify are listed at the end.

## Systems with usable evidence

| System | Mechanism and hippocampal claim | Trained? | Headline vs baseline | What carried it | For our design, and cheapest test |
|---|---|---|---|---|---|
| **HippoRAG** (2024, 2405.14831) | LLM-extracted triple graph plus personalised PageRank (PPR) from query entities; "hippocampal index" | No (prompted LLM, off-the-shelf retriever) | Recall@5 vs ColBERTv2: 2Wiki 89.1 vs 68.2, MuSiQue 51.9 vs 49.2, HotpotQA 77.7 vs 79.3 (loses) | PPR: query-nodes-only drops MuSiQue R@5 51.9 to 41.0 | Does not transfer to single-fact ranking. Test: count how many of 100 items need two stored sentences; if near zero, skip |
| **HippoRAG 2** (2025, 2502.14802) | Adds passage nodes, query-to-triple linking, LLM "recognition" filter | No (uses NV-Embed-v2) | F1 avg 59.8 vs NV-Embed-v2 57.0. 2Wiki 71.0 vs 61.5, MuSiQue 48.6 vs 45.7; NQ 63.3 vs 61.9, PopQA 56.2 vs 55.7. GraphRAG, RAPTOR and HippoRAG 1 are below dense on NQ | Query-to-triple linking (avg R@5 87.1 vs 74.6 with NER linking) and passage nodes (81.0 without). The filter adds 0.7 | Same as above. It needs the trained embedder as its base |
| **InfLLM** (2024, 2402.04617) | Evicted context kept as key/value blocks; lookup by current queries · keys of "representative tokens"; read back into attention | No | ∞-Bench Mistral avg 57.7 vs 21.5 streaming. Retrieve.KV 96.8 vs 13.2 for E5-Mistral RAG. Llama-3 Retrieve.KV only 5.0 | Lookup: R.KV 96.8 to 0.4 without it. Mean-pooled block key 84.6 | Adopt. This is our read-back with multi-token, multi-head keys. Test: score Σq·k over 4 representative tokens per span on cached states, split same-wording vs paraphrase |
| **EM-LLM** (2025, 2407.09450) | InfLLM plus surprise-based event boundaries, graph refinement and a contiguity buffer | No | LongBench (Llama-3.1-8B) 51.58 vs NV-Embed-v2 RAG 36.44 vs full context 39.30. Vs InfLLM: +1.8 on Mistral, ≤1 elsewhere, and lower on ∞-Bench (48.81 vs 50.31) | Inherited lookup. Segmentation about +1. Contiguity buffer between −0.26 and +0.06 on the two ablation tables | Adapt segmentation as the span boundary rule only. Test: 50-item generation check with and without the ±1 neighbouring span; expect about zero |
| **ICR** (2024, 2410.02642) and **QRHead** (2025, 2506.09944) | Rank candidates by attention mass from query tokens, calibrated with a null query; QRHead uses 16 heads chosen on 70 labelled examples | No | QRHead on LongMemEval recall@5: 85.5 vs Stella 63.9, Contriever 62.7; accuracy 59.8 vs 50.9. ICR on BEIR over ColBERTv2 top-100: 63.7 vs 57.6 | Head selection and calibration. Copy-defined retrieval heads give 81.5; only 8 of the top 32 heads overlap | Adapt as a second-stage re-ranker: top-20 candidates and the question fit in the 2,048 window. Test on 100 items: 16 heads chosen on held-out items, null-query calibration, compare with our single-head 0.43–0.52, split by paraphrase |
| **Memory³** (2024, 2407.01178) | Retrieved chunks injected as sparse attention key/values; selection by BGE-M3 | Trained from scratch (2.4B) | 63.31 with memory vs 60.80 without. Some RAG baselines score higher on legal and medical | Not isolated | Supports the split we already see: trained text encoder selects, key/value read-back delivers. Test: 50-item generation with bge-small as selector against the oracle 0.70 |
| **Larimar** (2024, 2403.11901) | Trained encoder, 512×768 episodic memory and decoder; one-shot write, edit and forget | Trained jointly; no gradients at edit time | CounterFact paraphrase 88.4 vs ROME 99.1; 93.6 if two rephrases are also written. Retention 0.92–0.97 | Scope detector: neighbourhood 80.6 to 15.7 without it (detector F1 0.974) | Adapt two ideas. Write rephrased keys: generate two questions per span at write time, re-score paraphrase top-1 on 100 items. Use our comparator as the scope gate (see abstention) |
| **LongMemEval** (2024, 2410.10813) | Benchmark plus design study on a dense retriever (Stella) | n/a | Fact-augmented keys: +9.4% recall, +5.4% QA. Time-aware query expansion: +6.8–11.3% temporal recall. Oracle retrieval still only 0.870–0.924 | Key expansion and round-level granularity | Same write-time key augmentation test as Larimar |
| **CAMELoT** (2024, 2402.13449) | Slots merged by similarity above 0.93, novelty opens a slot, oldest is evicted | No (frozen Llama-2-7B) | Perplexity only: PG-19 8.58 to 7.16. No QA, no retriever baseline | Recency eviction (9.25 without). Consolidation 7.00 vs 6.85; novelty 7.23 | Adapt the eviction rule. Evidence is perplexity, not recall |
| **Memoria** (2023, 2310.03052) | Hebbian co-retrieval graph over engrams, lifespans | Trained | WikiText-103 perplexity 23.47 vs Transformer-XL 24.54 | Hebbian links: long-term-memory gain +8.39 falls to +1.26 with random wiring (sorting task) | The random-wiring control is the right shape for our memory-vs-adapter worry |
| **HiCL** (2025, 2508.16651) | Learned DG layer with top-k sparsity routes to experts | Trained | Split CIFAR-10 Class-IL 65.58 vs DER++ 66.49 (not ahead) | Top-k sparsity (Task-IL 92.67 to 55.85 without), DG gating and contrastive loss (Class-IL 57.99 to about 19). CA3, CA1 and EWC are about zero | The sparse code carries separation, and it is learned, not random. Use our DG key to separate near-duplicates, not to match queries. Test: AUC of old-vs-new value separation under DG key vs dense key on update items |

## Systems with weak or missing controls

- **Titans** (2501.00663): meta-trained, so not training-free. Ablations at matched size cost 0.6–2 perplexity each; weight decay (forgetting) is the largest. An independent reimplementation (2510.09551) reports it does not consistently beat baselines. The surprise gate maps to our novelty signal.
- **δ-mem** (2605.12357): 4.87M trained parameters beside frozen Qwen3-4B, average 46.79 to 51.66. The only retrieval baseline is BM25 (44.56, below the backbone). No plain LoRA control and no state-reset ablation.
- **MemoryLLM** (2402.04624) and **M+** (2502.00592): trained, BM25 baseline only. M+ is below its own backbone on LongBench (32.78 vs 35.81). Its co-trained retriever "substantially outperforms" attention-score retrieval, which is direct evidence against internal keys.
- **kNN-LM** (1911.00172) and **Unlimiformer** (2305.01625): training-free internal keys. kNN-LM reaches perplexity 15.79 (a 2.9 gain); Unlimiformer at test time moves GovReport ROUGE-1 from 48.7 to 49.7. These are continuation lookups, not paraphrased questions.
- **MemGPT** (2310.08560): Deep Memory Retrieval 35.3% to 93.4% against a summary baseline; it uses ada-002 retrieval internally.
- **Generative Agents** (2304.03442): score = recency (0.995 decay) + importance + relevance, equal weights. No ablation separates the three terms.
- **A-MEM** (2502.12110): no dense baseline; Mem0's re-run puts it far lower.
- **Mem0** (2504.19413): LoCoMo J 66.88 vs best dense RAG 60.97, but full context scores 72.90.
- **Zep** (2501.13956): LongMemEval 71.2 vs full context 60.2 (gpt-4o). No dense-only baseline and no invalidation ablation.
- **LightMem** (2510.18866): topic segmentation is worth about 6 points. It loses single-session-assistant 32.14 vs naive RAG 98.21.
- **BMAM** (2601.20465): the "hippocampus" ablation (−24.6) removes the episodic store itself.
- **Human-Inspired Memory Architecture** (2605.08538): 70.1 vs 71.2 for raw retrieval, so no gain.
- **HEMA** (2504.16754): self-reported 41% to 87% against summarisation only.
- **Hippoformer** (ICLR 2026) and **TEM-transformer** (2112.04035): trained, spatial tasks only. Hippoformer's multi-step error is 9.71 vs Transformer 36.13, with no matched control and no language task.
- **MemoryAgentBench** (2507.05257), independent: BM25 overall 41.5 beats MemGPT 28.3, MIRIX 26.2, Zep 24.0, Mem0 21.1. HippoRAG-v2 is the best retrieval system on accurate retrieval (65.1 vs BM25 60.5).

## What transformers already do
- **Retrieval heads** (Wu et al. 2024, 2404.15574): 3–6% of heads, defined by copying. Masking them cuts extractive QA F1 by 9.2% and 23.1%.
- **Episodic-memory link** (Ji-An et al. 2024, 2405.14992): induction heads fit the CMR model, with contiguity and forward asymmetry. This was shown on repeated random tokens, which are exact repeats.
- **Associative memory views** (Ramsauer 2021, 2008.02217; Bricken and Pehlevan 2021, 2111.05498): attention is an associative read in a learned key space.
- All three match our pattern: same-wording recall works and paraphrase collapses. QRHead explicitly says copy-defined heads miss paraphrase.

## Answers

**1. What beats plain dense retrieval, and on what?**
- **Graph walk (PPR) over extracted entities and triples** wins on multi-hop and associative questions: +9.5 F1 on 2Wiki and +2.9 on MuSiQue over NV-Embed-v2. It is at parity on NQ and PopQA.
- **Query-conditioned attention in the reader** (QRHead, ICR, InfLLM) wins on multi-session and exact-match lookups: recall@5 85.5 vs 63.9; Retrieve.KV 96.8 vs 13.2.
- **Write-time restructuring** (fact or observation keys, topic segments) wins on temporal and multi-session conversational questions. LoCoMo observation-RAG scores 41.4 F1 vs 31.7 for raw dialog, and 41.9 vs 21.3 on temporal.

**2. Does a training-free internal key match a trained encoder on paraphrase?**
No published evidence that it does.
- LLM2Vec (2404.05961): untrained Mistral-7B MTEB retrieval is 10.43 (causal) or 15.84 (bidirectional), 38.05 after unsupervised training, about 56 supervised. Echo embeddings reach 22.68.
- M+ and Larimar point the same way (above).
- The only wins for internal signals are exact-match tasks and the cross-attention setting, where the query is in context with the candidates. QRHead and ICR do not report a paraphrase split.
- PromptEOL (2307.16645) claims parity with unsupervised contrastive methods on sentence similarity, not retrieval.
- I found nothing showing that sparse random codes improve semantic matching.

**3. What do they do for time and abstention that a retriever does not?**
- **Time mechanisms:**
  - Timestamped validity with LLM-judged invalidation (Zep). Temporal 45.1 to 62.4; knowledge-update 78.2 to 83.3 with gpt-4o but 76.9 to 74.4 with gpt-4o-mini.
  - LLM edit operations, add/update/delete (Mem0). Temporal J 55.51; the operation is not ablated.
  - Time-aware query filtering (LongMemEval): +6.8–11.3% temporal recall, and only with a strong LLM.
  - Recency eviction or scoring (CAMELoT, Generative Agents): not isolated on updates.
  - On MemoryAgentBench conflict resolution, all of these lose to BM25, and every method is at or below 28% on multi-hop. The decision to supersede is made by an LLM at write time, not by the ranker.
  - Cheapest test for us: on update items, add λ·write-order to any content score. Measure the new-above-old rate against the current 13–25% and the damage on non-update items.
- **Abstention mechanisms:**
  - Larimar's scope detector: removing it drops neighbourhood accuracy from 80.6 to 15.7.
  - Self-RAG (2310.11511, trained critic): without the critic ASQA falls 32.1 to 18.1; forcing a hard retrieve decision drops PopQA 45.5 to 28.3.
  - Adaptive-RAG (2403.14403): the router is only 54.5% accurate, while an oracle router lifts F1 46.9 to 56.3. The headroom is real and learned gates are weak.
  - Sufficient Context (2411.06037): retrieval reduces abstention; with insufficient context models hallucinate 15–40% of the time. A sufficiency rater plus confidence adds 2–10% selective accuracy.
  - Cheapest test for us: use the comparator (AUC 0.74–0.84) as an inject/skip gate in a 50-item generation check, and report exact match separately on correct and wrong retrievals.

## Not verified
- Titans' BABILong scores, M+'s attention-retrieval ablation and Wu et al.'s needle-masking curves appear only in figures.
- Hippoformer was read from a secondary summary (OpenReview blocked the fetch).
- HEMA, PromptEOL, TEM-transformer, Bricken and Pehlevan, and the Titans reimplementation were checked at abstract level only.
- Zep's table repeats 81.4 to 92.9 for single-session-user under both models, which looks like a copy error.
- The Zep/Mem0 LoCoMo dispute (84% claimed, 58.44% re-run, 75.14% corrected) and Letta's 74% filesystem result come from vendor blogs only.
- No per-system abstention accuracy was found in LongMemEval.
- The Omidi et al. review was not re-read.

## Sources
arXiv ids are inline. Non-arXiv pages:
- [Hippoformer ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10008006)
- [Hippoformer summary (papernotes)](https://en.papernotes.org/ICLR2026/others/hippoformer_integrating_hippocampus-inspired_spatial_memory_with_transformers/)
- [QRHead, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1214)
- [Zep's reply to Mem0 on LoCoMo](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- [Mem0 benchmark page](https://mem0.ai/blog/ai-memory-benchmarks)

Bibliography read (not edited): /home/realdanielbyrne/limbic-llm/docs/references.bib
