# Hippocampal models: what demonstrably works, and what it means for the side circuit

## Five-line summary
1. The literature assigns the dentate code to **writing** (decorrelating patterns stored in a capacity-limited, superposed autoassociator), not to ranking stored items against a cue. Cues reach the store by a separate, learned associative path. Your dentate-key result (0.02–0.19) is what the theory predicts.
2. The mechanisms with ablation or capacity evidence are: a fixed index/scaffold with learned heteroassociation (MESH, Vector-HaSH, barcodes); drifting temporal context used as a second-stage cue (TCM/CMR); a CA1-style comparator; and surprise-gated writing.
3. No hippocampal model is expected to beat a trained semantic encoder at "find one stated fact from a paraphrased question". Hippocampus-inspired retrieval systems only reach parity there.
4. They are expected to win on recency/order, update (AB–AC) interference, lure discrimination, contiguity, associative inference, novelty-gated storage and recall-to-reject.
5. Your three parts that worked in isolation (comparator, context tag, novelty) are the ones the literature supports. The update failure is the classic AB–AC problem, which the models solve with context and recency rather than with better content keys.

Verification marks: **[T]** full text read, **[A]** abstract read, **[M]** metadata only, **[U]** not verified. "Inference" marks my own deduction.

## 1. Model families

| Family | Mechanism | Validation and headline result | Load-bearing part | Translation and cheap test |
|---|---|---|---|---|
| **CLS** (McClelland 1995; Kumaran 2016; Norman & O'Reilly 2003; Schapiro 2017) | Fast sparse hippocampal store plus slow interleaved cortical learner; replay interleaves. | Retrograde-amnesia gradients and catastrophic interference (1995 [A]). The recognition model separates hippocampal recall from cortical familiarity, which respond differently to target–lure similarity [A]. Simulation-based. | Schapiro [A]: the trisynaptic path memorises episodes; the monosynaptic EC–CA1 path does statistical learning. | Supports an exact episodic store beside a frozen "cortex". **Test:** use the comparator as recall-to-reject on the text retriever's top-1 (risk–coverage on 100 items, including unanswerable ones). |
| **Treves–Rolls CA3 / dentate** | CA3 autoassociator with capacity p_max ≈ k·C_RC/(a·ln(1/a)). Mossy fibres force a new CA3 pattern at learning; the perforant path carries the retrieval cue. | p_max ≈ 36,000 for C_RC = 12,000, a = 0.02 [T] (so k ≈ 0.23). Analytical. | Rolls 2013 [T]: mossy fibres "may be necessary for new learning… but may not be necessary for the recall". Lee & Kesner 2004 [A]: dentate lesion impairs encoding, not retrieval; perforant-path lesion the reverse. O'Reilly & McClelland 1994 [A]: separation trades off against completion; the two-stage path serves separation, the direct path completion. | **Contradicts the dentate code as retrieval key.** Inference: an exact slot store has no crosstalk to reduce, so separation buys nothing for ranking. A random expansion with top-k is a similarity-preserving hash (Dasgupta 2017 [A]), so at best it reproduces input-similarity ranks. **Test:** score the dentate code on what it is for: overlap of near-duplicate statements differing in one value, against raw-key cosine, plus completion from truncated verbatim cues. |
| **TCM / CMR / CMR2 / Laplace time** | Context drifts: t_i = ρ·t_{i−1} + β·t_i^IN. Items bind to context; a recalled item reinstates its context, which cues its neighbours. Shankar–Howard: leaky integrators form a Laplace transform of history, giving scale-invariant time cells. | Recency and contiguity across timescales in free recall [T]. Forward recalls about twice backward (Kahana 1996 [A]). Contiguity across 34 benchmark findings (Healey 2019 [A]). Neural "jump back in time" (Folkerts 2018 [A]); human time cells predict temporal organisation (Umbach 2020 [A]). DeepSITH claims state of the art against LSTMs on time-series tasks [A]. | Retrieved pre-experimental context gives the forward asymmetry [T]. CMR2 [A]: context focuses retrieval on the target list and **censors** items whose context mismatches. | Supports the integrator bank, but as a **second-stage cue and filter, not a content key**. Inference: with τ_max = 1,024, a tag at lag 8,192 has decayed to e^−8, so tags cannot order old entries; extend τ or use the ordinal. **Tests:** (a) after a content hit, add tag-nearest spans and measure answer recall@k; (b) recency-judgement accuracy against log lag ratio; (c) on update items, take the latest among near-tied content candidates. |
| **SR, TEM, TEM-t, CSCG** | SR: predictive map; a TCM variant estimates the SR by TD learning (Gershman 2012 [A]). TEM: a structural code g bound to sensory x in fast Hebbian memory. TEM-t [T]: **queries and keys come from the position code, values from content**. CSCG: clones of one observation per context. | SR: place-cell phenomena, qualitative [A]. TEM: grid/place cells and zero-shot structural transfer on graph walks [A]. TEM-t: "dramatic" sample-efficiency gains, no numbers; capacity gain "not shown" [T]. CSCG: transitive stitching of disjoint episodes; perfect next-symbol prediction on one aliased layout [T]. | Learned structure code (TEM); cloning by context (CSCG). All trained. | Supports addressing by where/when (ordinal, context) with content as value. CSCG matches the update case: same entity–attribute, different context, separate clone. No training-free transfer. **Test:** for repeated entity–attribute statements, does key plus context tag separate occurrences where the key alone cannot (AUC, 50 pairs)? |
| **MESH / Vector-HaSH** | Fixed, pre-structured scaffold of attractor states; content attached by heteroassociation. Cue → hippocampal state → scaffold fixed point → reconstruction. | No memory cliff; information per synapse stays non-zero where Hopfield variants drop to zero. Perfect recall up to N_h patterns, graceful beyond [T]. Sequences: asymmetric Hopfield fails within about 50 steps, the scaffold recalls 1.4×10^4 [T]. | Factoring index from content; sequences via low-dimensional transitions on the scaffold. A random nonlinear projection suffices for the scaffold; the **content↔scaffold weights are learned** (pseudoinverse or Hebbian) [T]. | Supports ordinal index plus stored value; your slot store already has "no cliff". Says nothing about paraphrase cues: content addressing relies on a learned map from the same sensory pattern. **Test:** after a hit at slot i, read i±m and score multi-sentence answers. |
| **Index theory / barcodes** (Teyler & DiScenna 1986 [M]; Teyler & Rudy 2007 [A]; Chettih 2024 [A]; Fang 2026 [T]) | Hippocampus stores a pointer to cortical patterns. Barcodes are sparse, event-unique codes, uncorrelated even for nearby caches. | Chickadee recordings: the barcode reactivates at retrieval of that cache. | Fang [T]: with barcodes ablated, the model cannot distinguish memories at nearby locations; with content ablated, flexible retrieval is lost. | Directly matches your finding: the sparse code is an **ID, reactivated by content input**, not the cue. Storing pooled deep state with a pointer is consistent. Same lure test as the dentate row. |
| **Hasselmo; comparator; novelty** | Theta phases separate encoding (EC input strong, CA3 retrieval weak) from retrieval, with depotentiation of mismatching old associations. CA1 compares recalled against actual. Novelty gates plasticity via a hippocampus–VTA loop. | Hasselmo 2002 [A]: a T-maze **reversal** model. CA1 response tracks the number of changes (Duncan 2012 [A]). Prediction errors shift CA1 connectivity toward entorhinal input and away from CA3 (Bein 2020 [A]). Mismatch signal in fMRI (Kumaran & Maguire 2006 [A]). Lisman & Grace and Vinogradova are conceptual or physiological, with no quantitative model [A]. | Phase separation is load-bearing for reversal in the model. | Supports the comparator and the write gate, and gives the mechanism for **updates**: at write, high key match plus value mismatch marks the old entry superseded. **Tests:** (a) AUC of that flag on old/new update pairs against restatements, then update top-1 after down-weighting; (b) is comparator score monotone in number of changed attributes (0–4)? |
| **SDM / modern Hopfield** | SDM writes to all addresses within a Hamming radius. Softmax attention approximates the SDM read, and equals the modern Hopfield update. | Capacity exponential in dimension, one-step retrieval (Ramsauer [A]). Trained GPT-2 β values lie in SDM-plausible ranges [T]. Bricken & Pehlevan map SDM to cerebellum, not hippocampus [T]. | Separation between stored patterns controls retrieval error. | Your store read by softmax over keys already is this memory. The limit is key geometry: the query must lie near the right key. Nothing here fixes paraphrase. |
| **Key-value memory** (Gershman, Fiete, Irie 2025 [A/T]) | Keys optimised for discriminability, values for fidelity; forgetting is mostly retrieval failure. | Review plus simulations. | The query→key map is learned, or partly fixed and partly Hebbian [T]. | Supports separating key from value. Implies a training-free random key is the weak link. |
| **REMERGE / big loop** | Recirculating retrieved output as new input yields inference over separated codes. | Transitive inference, paired-associate inference, acquired equivalence [T]. Laminar EC connectivity correlates with inference performance (Koster 2018 [A]). | Recurrence. | Not in your design. **Test:** 50 A–B/B–C items; re-cue with the retrieved value and compare A→C top-1 with single pass and with the text retriever. |
| **Replay / consolidation** | Hippocampus trains cortex offline. | Alternating NREM/REM replay supports continual learning (Singh 2022 [A]). Generative replay: state of the art on class-incremental CIFAR-100 without stored data (van de Ven 2020 [A]). Consolidate only what aids generalisation (Sun 2023 [A]). Error-driven CA3 beats Hebbian on AB–AC lists (Zheng 2022 [A/T]). | Interleaving. | Not applicable to a frozen model. AB–AC is the right benchmark framing for your update items. |

## 2. Advances from 2023–2026

- **Induction heads resemble CMR** (Ji-An et al., NeurIPS 2024 [T]). Attention-by-lag shows contiguity and forward asymmetry in GPT-2, Pythia, Qwen-7B, Mistral-7B and Llama3-8B. Ablating the top 10% of CMR-like heads hurts in-context learning more than random heads; no effect sizes in the text. Bajaj et al. 2025 [A] find similar temporal biases in state-space models. This matches your result that model-internal keys work on verbatim repetition and collapse on paraphrase: those heads are token-match plus context machines. **Test:** select heads by lag curve on a repeated-token prompt and compare their paraphrase top-1 with your current best head.
- **EM-LLM** (ICLR 2025 [T]), training-free: surprise-based segmentation plus similarity retrieval plus a contiguity buffer. It beats an NV-Embed-v2 retriever by 30.5% on LongBench; results against InfLLM are mixed by model; passkey is saturated; the contiguity buffer is best on 44% of tasks. A non-archival abstract (Pink et al., CCN 2025 [T]) reports it is **at chance on recency judgements** over a novel, so segmentation and contiguity without a time code do not give order. That is the gap your context tag and ordinal fill.
- **HippoRAG 2** (ICML 2025 [T]): on simple fact QA it only matches the embedding retriever (NQ F1 63.3 against 61.9; PopQA 56.2 against 55.7), while the original HippoRAG is far below on NQ recall@5 (44.4 against 75.4). It gains on multi-hop (2Wiki F1 71.0 against 61.5). The largest ablation drop is query-to-triple linking (−27.5 recall@5).
- **HOLA** (single-author preprint, 2026 [A; numbers from the paper page, not double-checked]): a Gated DeltaNet state plus a bounded exact cache that keeps tokens with the largest delta-rule residual. Needle recall at 32k is 0.58, against 0.24 for a matched recency cache and 0.14 for the base model. It is trained end to end, but it is direct evidence that surprise-gated writing beats recency. **Test:** fraction of answer-bearing spans retained at a 5–10% write budget under novelty, recency and random gating.
- **Benchmarks:** SORT [A] (order recall), LongMemEval [A] (knowledge updates, temporal reasoning) and the Huet et al. episodic benchmark [A] (models fail with multiple related events) are the standard tasks in the regime where these mechanisms should matter.

## 3. What the system is for, and the standard tests

| Function | Standard test and signature | Source |
|---|---|---|
| Separating similar events | Mnemonic Similarity Task; lure discrimination index = p("similar"\|lure) − p("similar"\|foil); dentate-specific | Stark 2019 [A]; index definition from a task catalogue, not the original paper |
| Completion from partial cues | Partial-cue recall; CA3 NMDA knockout impairs it only under partial cues | Nakazawa 2002 [A]; Neunuebel & Knierim 2014 [A] |
| Temporal order and recency | Recency judgement; lag-CRP with forward asymmetry | DuBrow & Davachi 2014 [A]; Kahana 1996 [A] |
| What–where–when binding | Cache recovery by time since caching | Clayton & Dickinson 1998 [A]; Tsao 2018 [A] |
| Update / reversal | T-maze reversal; AB–AC lists | Hasselmo 2002 [A]; Zheng 2022 [T] |
| Associative inference | A–B, B–C → A–C | Zeithamova 2012 [A]; Kumaran & McClelland 2012 [T] |
| Novelty / mismatch | Changed-probe designs; response scales with number of changes | Duncan 2012 [A]; Bein 2020 [A] |
| Familiarity vs recollection | Dual-process dissociations | Norman & O'Reilly 2003 [A]; Yonelinas 2002 [M] |

## 4. Is single-fact retrieval from a verbal question a test they would win?

No. In every model above, a cue reaches a stored item either by overlap with the original input pattern (perforant path, Vector-HaSH heteroassociation, barcode reactivation by content) or by a learned query→key map (key-value memory, TEM). Paraphrase invariance is the cortical encoder's job, and the trained retriever is that encoder. HippoRAG 2 gives the empirical ceiling: parity on simple facts.

Tests where they should win, each scoreable on 50–100 items from cached states:
- update items using recency plus write-time supersession;
- recency and order judgements;
- lure discrimination among near-duplicate statements;
- neighbour recovery after a hit;
- two-hop inference by re-cueing;
- answer-span retention under a write budget;
- rejection of wrong recalls.

## References
- McClelland, McNaughton, O'Reilly 1995, Psychol Rev 102:419, 10.1037/0033-295X.102.3.419
- Kumaran, Hassabis, McClelland 2016, TiCS 20:512, 10.1016/j.tics.2016.05.004
- Norman & O'Reilly 2003, Psychol Rev 110:611, 10.1037/0033-295X.110.4.611
- Schapiro et al. 2017, Phil Trans B 372:20160049, 10.1098/rstb.2016.0049
- Treves & Rolls 1992, Hippocampus 2:189, 10.1002/hipo.450020209
- Treves & Rolls 1994, Hippocampus 4:374, 10.1002/hipo.450040319
- Rolls 2013, Front Syst Neurosci 7:74, 10.3389/fnsys.2013.00074
- O'Reilly & McClelland 1994, Hippocampus 4:661, 10.1002/hipo.450040605
- Lee & Kesner 2004, Hippocampus 14:66, 10.1002/hipo.10167
- McHugh et al. 2007, Science 317:94, 10.1126/science.1140263
- Dasgupta, Stevens, Navlakha 2017, Science 358:793, 10.1126/science.aam9868
- Howard & Kahana 2002, J Math Psychol 46:269, 10.1006/jmps.2001.1388
- Kahana 1996, Mem Cognit 24:103, 10.3758/BF03197276
- Sederberg, Howard, Kahana 2008, Psychol Rev 115:893, 10.1037/a0013396
- Polyn, Norman, Kahana 2009, Psychol Rev 116:129, 10.1037/a0014420
- Lohnas, Polyn, Kahana 2015, Psychol Rev 122:337, 10.1037/a0039036
- Healey, Long, Kahana 2019, Psychon Bull Rev 26:699, 10.3758/s13423-018-1537-3
- Shankar & Howard 2012, Neural Comput 24:134, 10.1162/NECO_a_00212
- Howard et al. 2014, J Neurosci 34:4692, 10.1523/JNEUROSCI.5808-12.2014
- MacDonald et al. 2011, Neuron 71:737, 10.1016/j.neuron.2011.07.012
- Umbach et al. 2020, PNAS 117:28463, 10.1073/pnas.2013250117
- Folkerts, Rutishauser, Howard 2018, J Neurosci 38:4200, 10.1523/JNEUROSCI.2312-17.2018
- Tsao et al. 2018, Nature 561:57, 10.1038/s41586-018-0459-6
- Jacques et al. 2021 (DeepSITH), arXiv:2104.04646
- Stachenfeld, Botvinick, Gershman 2017, Nat Neurosci 20:1643, 10.1038/nn.4650
- Gershman et al. 2012, Neural Comput 24:1553, 10.1162/NECO_a_00282
- Whittington et al. 2020, Cell 183:1249, 10.1016/j.cell.2020.10.024
- Whittington, Warren, Behrens 2022 (ICLR), arXiv:2112.04035
- George et al. 2021, Nat Commun 12:2392, 10.1038/s41467-021-22559-5
- Sharma, Chandra, Fiete 2022 (ICML), arXiv:2202.00159
- Chandra, Sharma, Chaudhuri, Fiete 2025, Nature 638:739, 10.1038/s41586-024-08392-y
- Teyler & DiScenna 1986, Behav Neurosci 100:147, 10.1037/0735-7044.100.2.147
- Teyler & Rudy 2007, Hippocampus 17:1158, 10.1002/hipo.20350
- Chettih et al. 2024, Cell 187:1922, 10.1016/j.cell.2024.02.032
- Fang et al. 2026, eLife 14:RP103512, 10.7554/eLife.103512
- Hasselmo, Bodelón, Wyble 2002, Neural Comput 14:793, 10.1162/089976602317318965
- Hasselmo 2006, Curr Opin Neurobiol 16:710, 10.1016/j.conb.2006.09.002
- Lisman & Grace 2005, Neuron 46:703, 10.1016/j.neuron.2005.05.002
- Vinogradova 2001, Hippocampus 11:578, 10.1002/hipo.1073
- Kumaran & Maguire 2006, PLoS Biol 4:e424, 10.1371/journal.pbio.0040424
- Kumaran & Maguire 2007, Hippocampus 17:735, 10.1002/hipo.20326
- Duncan et al. 2012, Hippocampus 22:389, 10.1002/hipo.20933
- Bein, Duncan, Davachi 2020, Nat Commun 11:3451, 10.1038/s41467-020-17287-1
- Kanerva 1988, Sparse Distributed Memory, MIT Press [U; described via Bricken & Pehlevan]
- Krotov & Hopfield 2016, arXiv:1606.01164
- Ramsauer et al. 2021 (ICLR), arXiv:2008.02217
- Bricken & Pehlevan 2021 (NeurIPS), arXiv:2111.05498
- Gershman, Fiete, Irie 2025 (accepted, Neuron), arXiv:2501.02950
- Kumaran & McClelland 2012, Psychol Rev 119:573, 10.1037/a0028681
- Koster et al. 2018, Neuron 99:1342, 10.1016/j.neuron.2018.08.009
- Singh, Norman, Schapiro 2022, PNAS 119:e2123432119, 10.1073/pnas.2123432119
- van de Ven, Siegelmann, Tolias 2020, Nat Commun 11:4069, 10.1038/s41467-020-17866-2
- Sun et al. 2023, Nat Neurosci 26:1438, 10.1038/s41593-023-01382-9
- Zheng et al. 2022, PLoS Comput Biol 18:e1010589, 10.1371/journal.pcbi.1010589
- Ji-An et al. 2024 (NeurIPS), arXiv:2405.14992
- Bajaj et al. 2025, arXiv:2510.22752
- Fountas et al. 2025 (ICLR), arXiv:2407.09450
- Pink, Saxena, Toneva 2025, CCN abstract "Evaluating Models of Naturalistic Episodic Memory" (no DOI)
- Gutiérrez et al. 2024 (NeurIPS), arXiv:2405.14831
- Gutiérrez et al. 2025 (ICML), arXiv:2502.14802
- Cui 2026, arXiv:2607.02303
- Pink et al. 2024 (SORT), arXiv:2410.08133
- Wu et al. 2025 (ICLR), arXiv:2410.10813
- Huet et al. 2025, arXiv:2501.13121
- Stark, Kirwan, Stark 2019, TiCS 23:938, 10.1016/j.tics.2019.08.003
- Nakazawa et al. 2002, Science 297:211, 10.1126/science.1071795
- Neunuebel & Knierim 2014, Neuron 81:416, 10.1016/j.neuron.2013.11.017
- DuBrow & Davachi 2014, J Neurosci 34:13998, 10.1523/JNEUROSCI.2535-14.2014
- Clayton & Dickinson 1998, Nature 395:272, 10.1038/26216
- Zeithamova, Dominick, Preston 2012, Neuron 75:168, 10.1016/j.neuron.2012.05.010
- Yonelinas 2002, J Mem Lang 46:441, 10.1006/jmla.2002.2864

## Verification caveats
- PubMed and the Nature site blocked fetches. Abstracts came from Europe PMC and arXiv. Full texts came from Europe PMC, arXiv HTML, or author- or course-hosted PDFs.
- Schapiro 2017 is abstract only: task details and pathway learning rates were not checked.
- The HOLA needle numbers and the EM-LLM and HippoRAG 2 table numbers were read through a summarising fetch, not by me line by line.
- Teyler & DiScenna pages are 147–154 per Crossref; Teyler & Rudy cite 147–152.
- The statements that separation buys nothing in an exact slot store, and that τ_max = 1,024 cannot tag at lag 8,192, are my inferences from the cited mechanisms, not claims made by any paper.
- No files in the repo were touched; working files are only in the session scratchpad.
