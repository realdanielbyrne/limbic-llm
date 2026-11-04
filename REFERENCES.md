# References

The bibliography for limbic-llm, grouped by the part of the hippocampal mapping each work informs. Every entry has a BibTeX key in [`docs/references.bib`](docs/references.bib).

## Memory architectures implemented here

| Key | Reference | Used for |
|---|---|---|
| `behrouz2024titans` | Behrouz, Zhong & Mirrokni (2025). *Titans: Learning to Memorize at Test Time.* NeurIPS 2025, arXiv:2501.00663 | `NeuralMemory`, Titans layers (Strategy II). Its appendix states that the memory generalises Gated DeltaNet and Longhorn |
| `yang2025gated` | Yang, Kautz & Hatamizadeh (2025). *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR | `gated_delta.py`. §3 already states the delta rule as test-time SGD on $\tfrac12\lVert Sk-v\rVert^2$ with the gate as adaptive weight decay: the identity used in this repo is theirs, not derived here |
| `yang2024parallelizing` | Yang, Wang, Zhang, Shen & Kim (2024). *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS | Chunkwise WY/UT algorithm that `gated_delta_chunked` re-derives; DeltaNet's online-regression view |
| `liu2024longhorn` | Liu et al. (2024). *Longhorn: State Space Models are Amortized Online Learners.* arXiv:2407.14207 | State-space layers as online learners on $\lVert Sk-v\rVert^2$: the GDN = one SGD step identity |
| `behrouz2025miras` | Behrouz et al. (2025). *It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization* (Miras). arXiv:2504.13173 | Unifies GDN, Titans and TTT as associative memories |
| `ramsauer2021hopfield` | Ramsauer et al. (2021). *Hopfield Networks is All You Need.* ICLR | CA3 retrieval |
| `krotov2016dense` | Krotov & Hopfield (2016). *Dense Associative Memory for Pattern Recognition.* NeurIPS | CA3 retrieval |
| `schlag2021linear` | Schlag, Irie & Schmidhuber (2021). *Linear Transformers Are Secretly Fast Weight Programmers.* ICML | Delta rule as fast weights |
| `vaswani2017attention` | Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS | EC encoder, cortex layers |
| `shazeer2017outrageously` | Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR | DG as an expert gate |
| `perez2018film` | Perez et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI | `CA2Modulator` |
| `qwenputer2026` | Agrawal, D. (2026). *Qwenputer.* Blog post and code (divagr.com), not peer reviewed | Gated DeltaNet state as RAM. The underlying principle is Kohonen (1972) and Schlag et al. (2021) |
| `qwen2025qwen3` | Qwen Team (2025). *Qwen3 Technical Report.* arXiv:2505.09388 | Attention-only pretrained target |
| `qwen2026qwen35` | Qwen Team (2026). *Qwen3.5: Towards Native Multimodal Agents.* Model release | Gated DeltaNet hybrid pretrained target (Qwen3.5-2B: 24 layers, 3:1 GDN:attention) |

## Hippocampal anatomy, physiology and theory

| Key | Reference | Relevance |
|---|---|---|
| `andersen2007hippocampus` | Andersen, Morris, Amaral, Bliss & O'Keefe, eds. (2007). *The Hippocampus Book.* Oxford University Press | Trisynaptic circuit |
| `bliss1993synaptic` | Bliss & Collingridge (1993). *A Synaptic Model of Memory: Long-Term Potentiation in the Hippocampus.* Nature 361 | Plasticity at CA3 → CA1 |
| `okeefe1978hippocampus` | O'Keefe & Nadel (1978). *The Hippocampus as a Cognitive Map.* Clarendon Press | Cognitive-map theory |
| `squire1992memory` | Squire (1992). *Memory and the Hippocampus.* Psychological Review 99(2) | Declarative memory |
| `scoville1957loss` | Scoville & Milner (1957). *Loss of Recent Memory after Bilateral Hippocampal Lesions.* JNNP 20(1) | Patient H.M. |
| `hafting2005microstructure` | Hafting, Fyhn, Molden, Moser & Moser (2005). *Microstructure of a Spatial Map in the Entorhinal Cortex.* Nature 436 | Grid cells (EC) |
| `eichenbaum2017integration` | Eichenbaum (2017). *On the Integration of Space, Time, and Memory.* Neuron 95(5) | Relational memory |
| `spalding2013dynamics` | Spalding et al. (2013). *Dynamics of Hippocampal Neurogenesis in Adult Humans.* Cell 153(6) | DG neurogenesis. Human adult neurogenesis is contested; cite the opposing evidence alongside |
| `levy1996sequence` | Levy (1996). *A Sequence Predicting CA3 Is a Flexible Associator.* Hippocampus 6(6) | Sequential CA3 store |
| `jensen1996hippocampal` | Jensen & Lisman (1996). *Hippocampal CA3 Region Predicts Memory Sequences.* Learning & Memory 3 | Sequential CA3 store |
| `samsonovich2005simple` | Samsonovich & Ascoli (2005). *A Simple Neural Network Model of the Hippocampus Suggesting Its Pathfinding Role in Episodic Memory Retrieval.* Learning & Memory 12(2) | CA3 transition store (the successor-bound store) plus a CA3 → CA1 goal field written at sharp waves; retrieval as path search. See `research/reviews/2026-11-02_literature/pathfinding.md` |
| `hasselmo2006role` | Hasselmo (2006). *The Role of Acetylcholine in Learning and Memory.* Curr. Opin. Neurobiol. 16(6) | Novelty-gated encoding (`novelty_lr`) |
| `lisman2005hippocampal` | Lisman & Grace (2005). *The Hippocampal-VTA Loop.* Neuron 46(5) | CA1 novelty → learning and decisions |
| `bein2020mnemonic` | Bein, Duncan & Davachi (2020). *Mnemonic Prediction Errors Bias Hippocampal States.* Nature Communications 11 | CA1 mismatch drives encoding |
| `hitti2014hippocampal` | Hitti & Siegelbaum (2014). *The Hippocampal CA2 Region Is Essential for Social Memory.* Nature 508 | CA2 function |
| `leroy2018circuit` | Leroy et al. (2018). *A Circuit from Hippocampal CA2 to Lateral Septum Disinhibits Social Aggression.* Nature 564 | CA2 also promotes aggression: a social-context selector, not a norm enforcer |
| `smith2016targeted` | Smith, Williams Avram, Cymerblit-Sabba, Song & Young (2016). *Targeted Activation of the Hippocampal CA2 Area Strongly Enhances Social Memory.* Mol. Psychiatry 21(8) | CA2 amplification |
| `boyle2024tuned` | Boyle, Posani, Irfan, Siegelbaum & Fusi (2024). *Tuned Geometries of Hippocampal Representations Meet the Computational Demands of Social Memory.* Neuron 112(8) | CA2 identity coding |
| `howard2002distributed` | Howard & Kahana (2002). *A Distributed Representation of Temporal Context.* J. Math. Psych. 46(3) | Temporal context model of free recall. It makes no claim about CA2; the CA2 link is this project's inference, and temporal-context signals are documented in LEC (`tsao2018integrating`) |
| `kumaran2012generalization` | Kumaran & McClelland (2012). *Generalization through the Recurrent Interaction of Episodic Memories.* Psychological Review 119(3) | Big-loop recurrence (REMERGE) |
| `koster2018bigloop` | Koster et al. (2018). *Big-Loop Recurrence within the Hippocampal System Supports Integration of Information across Episodes.* Neuron 99(6) | Big-loop recurrence |
| `mcclelland1995why` | McClelland, McNaughton & O'Reilly (1995). *Why There Are Complementary Learning Systems.* Psychological Review 102(3) | Replay consolidation |
| `wilson1994reactivation` | Wilson & McNaughton (1994). *Reactivation of Hippocampal Ensemble Memories during Sleep.* Science 265 | Sleep replay |
| `mcewen1999stress` | McEwen (1999). *Stress and Hippocampal Plasticity.* Annu. Rev. Neurosci. 22 | Background report |

## Hippocampus-inspired machine learning (prior art)

| Key | Reference | Overlap with this package |
|---|---|---|
| `li2026hippoformer` | Li, Cao, Wang, Yang, Zou & Hong (2026). *Hippoformer: Integrating Hippocampus-Inspired Spatial Memory with Transformers.* ICLR | mm-TEM (a Tolman–Eichenbaum Machine variant with a meta-MLP relational memory) coupled to a transformer for spatial prediction. Not a Titans-style memory. Its name prompted the rename |
| `whittington2020tolman` | Whittington et al. (2020). *The Tolman-Eichenbaum Machine.* Cell 183(5) | EC structural vs. sensory codes |
| `whittington2022relating` | Whittington, Warren & Behrens (2022). *Relating Transformers to Models and Neural Representations of the Hippocampal Formation.* ICLR | Position encodings as grid cells |
| `kapoor2025hicl` | Kapoor, Mackey, Aloimonos & Lin (2026). *HiCL: Hippocampal-Inspired Continual Learning.* AAAI 2026, arXiv:2508.16651 | DG top-k codes gate experts, with grid-cell input, a CA3 autoassociator, EWC and prioritised replay. Anticipates most of the DG → CA3 MoE track |
| `das2024larimar` | Das et al. (2024). *Larimar: Large Language Models with Episodic Memory Control.* ICML 2024 | Episodic memory for LLMs |
| `gutierrez2024hipporag` | Gutiérrez, Shu, Gu, Yasunaga & Su (2024). *HippoRAG.* NeurIPS | Hippocampal indexing for retrieval |
| `fountas2025human` | Fountas et al. (2025). *Human-Inspired Episodic Memory for Infinite Context LLMs* (EM-LLM). ICLR, arXiv:2407.09450 | Surprise-based event segmentation |
| `ahn2025hema` | Ahn (2025). *HEMA: A Hippocampus-Inspired Extended Memory Architecture for Long-Context AI Conversations.* arXiv:2504.16754 | Conversational memory |
| `omidi2025memory` | Omidi et al. (2025). *Memory-Augmented Transformers: A Systematic Review from Neuroscience Principles to Enhanced Model Architectures.* arXiv:2508.10824 | Survey |

## CA2 candidates: activation steering

| Key | Reference | Relevance |
|---|---|---|
| `turner2023activation` | Turner, Thiergart, Leech, Udell, Vazquez, Mini & MacDiarmid (2023). *Steering Language Models With Activation Engineering* (first released as *Activation Addition: Steering Language Models Without Optimization*). arXiv:2308.10248 | Side-channel steering vector at a middle layer, $h^l \leftarrow h^l + c\,\Delta h$; the FiLM special case $\gamma = 0$. Notebook in `docs/notebooks/` |

## CA1 candidates: calibrated decisions and predictive comparison

| Key | Reference | Relevance |
|---|---|---|
| `typesafe2026jev` | TypeSafe AI (2026). *Jev.* Launch post (primary) and jevai.net (third-party summary) | Single-pass structured decisions with calibrated confidence. Its method (RLCD) is unpublished, and independently measured calibration is mixed, so it cannot yet support claims of a calibrated CA1 |
| `nandhakishor2026laya` | Mukkunnoth, N. (2026). *LAYA.* github.com/NandhaKishorM/laya, Apache-2.0 | Open encoder decision engine: choice, score and P(true) in one forward pass |
| `huang2026llmjepa` | Huang, LeCun & Balestriero (2026). *LLM-JEPA.* ICLR | Final-layer embedding comparison between views |
| `balestriero2025lejepa` | Balestriero & LeCun (2025). *LeJEPA.* arXiv:2511.08544 | SIGReg: collapse-free learned comparators |
| `gorganmohammadi2026understanding` | Gorgan Mohammadi, Halvagal & Zenke (2026). *Understanding Neural Circuit Principles for Representation Learning through JEPAs.* eLife reviewed preprint | JEPA mapped to cortical circuits |

## Looped and recurrent-depth transformers

| Key | Reference | Relevance |
|---|---|---|
| `zhu2025scaling` | Zhu et al. (2025). *Scaling Latent Reasoning via Looped Language Models* (Ouro). arXiv:2510.25741 | Big-loop recurrence; learned exit gate |
| `dehghani2019universal` | Dehghani, Gouws, Vinyals, Uszkoreit & Kaiser (2019). *Universal Transformers.* ICLR | Weight-shared recurrence in depth |

## Feedback control and timing in the hippocampus

| Key | Reference | Relevance |
|---|---|---|
| `hasselmo2002proposed` | Hasselmo, Bodelón & Wyble (2002). *A Proposed Function for Hippocampal Theta Rhythm: Separate Phases of Encoding and Retrieval.* Neural Computation 14(4) | Theta-phase separation of writes and reads; loop period τ |
| `sik1994inhibitory` | Sik, Ylinen, Penttonen & Buzsáki (1994). *Inhibitory CA1-CA3-Hilar Region Feedback in the Hippocampus.* Science 265 | CA1 → CA3 inhibitory back-projection. The paper proposes synchronisation and spike timing; reading it as negative-feedback gain control is this project's analogy. An excitatory vCA1 → dCA3 projection also exists (`lin2021noncanonical`) |
| `treves1992computational` | Treves & Rolls (1992). *Computational Constraints Suggest the Need for Two Distinct Input Systems to the Hippocampal CA3 Network.* Hippocampus 2(2) | DG for storage, direct EC → CA3 path for retrieval cues |
| `buzsaki2002theta` | Buzsáki (2002). *Theta Oscillations in the Hippocampus.* Neuron 33(3) | Oscillatory timing of the loop |
| `macdonald2011hippocampal` | MacDonald, Lepage, Eden & Eichenbaum (2011). *Hippocampal "Time Cells" Bridge the Gap in Memory for Discontiguous Events.* Neuron 71(4) | Time cells; the time channel |
| `shankar2012scale` | Shankar & Howard (2012). *A Scale-Invariant Internal Representation of Time.* Neural Computation 24(1) | Log-spaced leaky integrators compute a Laplace transform of the past. Scale-invariant time cells need the inverse step, which `TimeChannel` does not implement |
| `kabir2024deep` | Kabir, Mochizuki-Freeman & Tiganj (2024). *Deep Reinforcement Learning with Time-Scale Invariant Memory.* | Scale-invariant (Laplace/inverse-Laplace) memory in deep RL agents; precedent for the time channel |
| `sapolsky1986neuroendocrinology` | Sapolsky, Krey & McEwen (1986). *The Neuroendocrinology of Stress and Aging: The Glucocorticoid Cascade Hypothesis.* Endocrine Reviews 7(3) | A loop without negative feedback: the cautionary case |

## Dynamic steering and oscillatory sequence models

| Key | Reference | Relevance |
|---|---|---|
| `rimsky2024steering` | Rimsky, Gabrieli, Schulz, Tong, Hubinger & Turner (2024). *Steering Llama 2 via Contrastive Activation Addition.* ACL | Contrast vectors averaged over many pairs |
| `lee2024programming` | Lee et al. (2025). *Programming Refusal with Conditional Activation Steering* (CAST). ICLR 2025, arXiv:2409.05907 | Steering gated by hidden-state patterns |
| `zhao2025adasteer` | Zhao et al. (2025). *AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender.* EMNLP 2025 | Input-adaptive steering coefficients from harmfulness and rejection directions; anticipates the graded controller except for its temporal integral |
| `do2025dynamic` | Do, Tran, Venkatesh & Le (2025). *Dynamic Steering With Episodic Memory For Large Language Models.* Findings of ACL | Closest prior work: memory-retrieved steering vectors |
| `sun2023retentive` | Sun et al. (2023). *Retentive Network.* arXiv:2307.08621 | Damped complex decay; recurrent form of the loop kernel |
| `orvieto2023resurrecting` | Orvieto et al. (2023). *Resurrecting Recurrent Neural Networks for Long Sequences.* ICML | Linear recurrences with complex decays |

## Knowledge in transformer weights (the neocortex analogue)

| Key | Reference | Relevance |
|---|---|---|
| `geva2021transformer` | Geva, Schuster, Berant & Levy (2021). *Transformer Feed-Forward Layers Are Key-Value Memories.* EMNLP | Facts stored in MLP blocks |
| `meng2022locating` | Meng, Bau, Andonian & Belinkov (2022). *Locating and Editing Factual Associations in GPT.* NeurIPS | Mid-layer MLPs as the consolidation target |

## Safety steering and evaluation (CA2 as a norm controller)

| Key | Reference | Relevance |
|---|---|---|
| `arditi2024refusal` | Arditi et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS | Static refusal direction |
| `zou2023representation` | Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405 | Reading and steering concepts in activations |
| `zou2024improving` | Zou et al. (2024). *Improving Alignment and Robustness with Circuit Breakers.* NeurIPS | Rerouting harmful representations towards incoherent or refusal states |
| `mazeika2024harmbench` | Mazeika et al. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML | Attack-success benchmark |
| `zhao2025llms` | Zhao, Huang, Wu, Bau & Shi (2025). *LLMs Encode Harmfulness and Refusal Separately.* NeurIPS | Certain jailbreaks reduce refusal without reversing the harmfulness judgement, which makes harmfulness a candidate controller input. Abliterated models were not tested |
| `rottger2024xstest` | Röttger et al. (2024). *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.* NAACL | Over-refusal benchmark |

## Reasoning steering (CA2 beyond alignment)

| Key | Reference | Relevance |
|---|---|---|
| `chen2025seal` | Chen et al. (2025). *SEAL: Steerable Reasoning Calibration of Large Language Models for Free.* COLM 2025, arXiv:2504.07986 | Fewer reasoning tokens with higher accuracy |
| `azizi2025activation` | Azizi et al. (2026). *Activation Steering for Chain-of-Thought Compression.* Findings of ACL 2026, arXiv:2507.04742 | Concise-vs-verbose steering vector |
| `huang2025mitigating` | Huang et al. (2025). *Mitigating Overthinking in Large Reasoning Models via Manifold Steering.* NeurIPS | Overthinking direction on the activation manifold |
| `lin2025controlling` | Lin et al. (2025). *Controlling Thinking Speed in Reasoning Models.* NeurIPS | Fast/slow-thinking vector, adaptive strength |
| `ma2025cotvalve` | Ma et al. (2025). *CoT-Valve: Length-Compressible Chain-of-Thought Tuning.* ACL | Parameter-space length control |
| `sun2025thinkedit` | Sun et al. (2025). *ThinkEdit: Interpretable Weight Editing to Mitigate Overly Short Thinking in Reasoning Models.* EMNLP | Shortening can hurt; over-commitment risk |
| `venhoff2025understanding` | Venhoff et al. (2025). *Understanding Reasoning in Thinking Language Models via Steering Vectors.* arXiv:2506.18167 | Backtracking and uncertainty directions |
| `hao2024training` | Hao et al. (2025). *Training Large Language Models to Reason in a Continuous Latent Space.* COLM 2025, arXiv:2412.06769 | Coconut: latent reasoning in place of tokens |
| `nguyen2025activation` | Nguyen et al. (2026). *Activation Steering with a Feedback Controller.* ICLR 2026, arXiv:2510.04309 | PID steering. Its integral term accumulates across layers within one forward pass, not across tokens or turns |
| `shi2026internalizing` | Shi et al. (2026). *STIR: Self-Distilled Latent Tools for Dynamic Control of LLM Reasoning.* EMNLP 2026 (arXiv:2602.04925, first titled *Internalizing LLM Reasoning via Discovery and Replay of Latent Actions*) | Memory-retrieved reasoning steering |

## Continual learning and replay

| Key | Reference | Relevance |
|---|---|---|
| `buzzega2020dark` | Buzzega, Boschini, Porrello, Abati & Calderara (2020). *Dark Experience for General Continual Learning.* NeurIPS | Logit-distillation replay in `Consolidator` |

## Literature review (Nov 2026)

The sections below were added by the literature review in [`research/reviews/2026-11-02_literature/`](research/reviews/2026-11-02_literature/README.md). They hold the closest prior work for each track, with how it relates to limbic-llm. A further 250+ verified sources cited in the track reports are in [`new_refs.bib`](research/reviews/2026-11-02_literature/new_refs.bib) and are not yet part of this bibliography.

## Prior art identified by the literature review (Nov 2026): memory, recall and hybrids

| Key | Reference | Relation to limbic-llm |
|---|---|---|
| `kohonen1972correlation` | Kohonen (1972). *Correlation Matrix Memories.* IEEE Transactions on Computers | Linear matrix memory, exact for orthonormal keys up to $d_k$: the 'one-hot keys = exact RAM' result |
| `sompolinsky1986temporal` | Sompolinsky & Kanter (1986). *Temporal Association in Asymmetric Neural Networks.* Physical Review Letters | Heteroassociative successor binding for sequence recall: the principle behind successor-bound CA3/GDN addressing |
| `olsson2022context` | Olsson et al. (2022). *In-context Learning and Induction Heads.* arXiv:2209.11895 | Previous-token shift plus content match: successor-bound lookup inside attention; the mechanism behind P4a's fix |
| `fu2022hungry` | Fu et al. (2022). *Hungry Hungry Hippos: Towards Language Modeling with State Space Models.* arXiv:2212.14052 | H3: a shift SSM supplies the token shift that associative recall needs; same role as tied addressing |
| `zhang2024memory` | Zhang et al. (2024). *Memory Mosaics.* arXiv:2405.06394 | Associative memories keyed on the past with next-token values: successor-bound addressing as an architecture |
| `wang2025test` | Wang et al. (2025). *Test-time regression: a unifying framework for designing sequence models with associative memory.* arXiv:2501.12352 | Test-time regression framework covering GDN, Mamba and Titans; the GDN-Titans identity is a special case |
| `lei2026mem` | Lei et al. (2026). *$\delta$-mem: Efficient Online Memory for Large Language Models.* arXiv:2605.12357 | δ-mem: a delta-rule state attached to a frozen Qwen3 backbone. Closest published system to W1; required baseline |
| `munkhdalai2024leave` | Munkhdalai et al. (2024). *Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention.* arXiv:2404.07143 | Infini-attention: delta-rule compressive memory added to the attention of pretrained LLMs |
| `okpekpe2025revisiting` | Okpekpe & Orvieto (2025). *Revisiting associative recall in modern recurrent models.* arXiv:2508.19029 | Associative recall in recurrent models is highly learning-rate sensitive; single-LR failures are not capacity evidence (M4) |
| `arora2023zoology` | Arora et al. (2023). *Zoology: Measuring and Improving Recall in Efficient Language Models.* arXiv:2312.04927 | MQAR benchmark; the standard replacement for the repo's AssociativeRecall task |
| `hsieh2024ruler` | Hsieh et al. (2024). *RULER: What's the Real Context Size of Your Long-Context Language Models?.* arXiv:2404.06654 | Multi-key and multi-value needle variants; single-needle passkey hides failures |
| `chen2026hla` | Chen et al. (2026). *HLA: Expressive Hybrid Linear Attention via Chunk-Wise Dynamic Mixing.* arXiv:2610.05842 | Pretrained Qwen3.5 GDN layers attenuate early chunks by orders of magnitude: expected background for P1 |
| `cabannes2025short` | Cabannes et al. (2025). *Short window attention enables long-term memorization.* arXiv:2509.24552 | Recurrent layers learn long-range memory only when local attention cannot do the retrieval (SWAX); bears on P1/P2 |
| `bick2025understanding` | Bick et al. (2025). *Understanding the Skill Gap in Recurrent Language Models: The Role of the Gather-and-Aggregate Mechanism.* arXiv:2504.18574 | Retrieval concentrates in a few gather-and-aggregate heads in Transformers and SSMs; read-bottleneck hypothesis for P1 |
| `kang2025state` | Kang et al. (2025). *State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models.* arXiv:2503.03499 | State-offset tuning: learned offsets into SSM state; the natural control and alternative injection site for P2 |
| `lee2026language` | Lee et al. (2026). *Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference.* arXiv:2605.26099 | Offline recurrent consolidation of context into SSM fast weights before clearing the KV cache; anticipates W3 and GDN 'sleep' |
| `behrouz2026language` | Behrouz et al. (2026). *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories.* arXiv:2606.03979 | Sleep as distillation-based consolidation plus dreaming; overlaps replay.py's sleep phase |
| `mcdermott2025lola` | McDermott et al. (2025). *LoLA: Low-Rank Linear Attention With Sparse Caching.* arXiv:2505.23666 | Window + sparse exact cache admitted by self-recall error + linear state: the three-tier design for KV trimming (O3) |
| `allenzhu2025physics` | Allen-Zhu (2025). *Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers.* arXiv:2512.17351 | Canon layers: short horizontal mixing lifts linear attention and GDN; supports the identity-initialised short-conv diagnosis |
| `jelassi2024repeat` | Jelassi et al. (2024). *Repeat After Me: Transformers are Better than State Space Models at Copying.* arXiv:2402.01032 | Fixed-state models cannot copy beyond their state information; bounds what a GDN-only memory can promise |
| `fan2020` | Fan et al. (2020). *Addressing Some Limitations of Transformers with Feedback Memory.* arXiv:2002.09402 | Feedback Transformer: higher-layer past states written into memory read by later steps; the closed loop of lagged self-states |
| `hatamizadeh2026gated` | Hatamizadeh et al. (2026). *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention.* arXiv:2605.22791 | Gated DeltaNet-2: decoupled erase and write gates, which true slot-overwrite RAM semantics need |
| `grazzi2024unlocking` | Grazzi et al. (2024). *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* arXiv:2411.12537 | β ∈ (0, 1) rules out parity-like state tracking; relevant if the RAM framing is extended to computation |

## Prior art identified by the literature review (Nov 2026): CA3 → CA1 circuit, skips and timing

| Key | Reference | Relation to limbic-llm |
|---|---|---|
| `conneau2017` | Conneau et al. (2017). *Supervised Learning of Universal Sentence Representations from Natural Language Inference Data.* arXiv:1705.02364 | InferSent matching features (u, v, |u−v|, u⊙v), identical to the CA1 comparator fusion vector |
| `chen2024predictive` | Chen et al. (2024). *Predictive sequence learning in the hippocampal formation.* Neuron | CA3 predicts the next EC input and CA1 computes the prediction error: the time-aligned comparator the repo lacks |
| `nakashiba2008transgenic` | Nakashiba et al. (2008). *Transgenic Inhibition of Synaptic Transmission Reveals Role of CA3 Output in Hippocampal Learning.* Science | CA3 output needed for one-trial learning; temporoammonic path suffices for incremental learning |
| `lin2021noncanonical` | Lin et al. (2021). *Noncanonical projections to the hippocampal CA3 regulate spatial learning and memory by modulating the feedforward hippocampal trisynaptic pathway.* PLOS Biology | Excitatory vCA1 → dCA3 back-projection; corrects 'CA1 pyramidal cells do not project to CA3' |
| `zhou2024value` | Zhou et al. (2024). *Value Residual Learning.* arXiv:2410.17897 | Value residual learning: first-layer values re-injected deep; closest ML analogue of the EC-III → CA1 skip |
| `pagliardini2024` | Pagliardini et al. (2024). *DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging.* arXiv:2402.02622 | DenseFormer: depth-weighted averaging of earlier blocks; long-skip baseline |
| `zhu2024hyper` | Zhu et al. (2024). *Hyper-Connections.* arXiv:2409.19606 | Hyper-connections: learned cross-depth connection strengths; skip-track baseline |
| `golovneva2024` | Golovneva et al. (2024). *Contextual Position Encoding: Learning to Count What's Important.* arXiv:2405.18719 | CoPE: learned gated counting of salient tokens; close to the salience-driven time channel |
| `tsao2018integrating` | Tsao et al. (2018). *Integrating time from experience in the lateral entorhinal cortex.* Nature | LEC temporal-context cells: what TimeChannel's leaky integrators resemble |
| `brunel1999fast` | Brunel & Hakim (1999). *Fast Global Oscillations in Networks of Integrate-and-Fire Neurons with Low Firing Rates.* Neural Computation | Delayed inhibitory feedback produces oscillations: the mechanism behind ringing above K ≈ 1.68 |
| `schapiro2017complementary` | Schapiro et al. (2017). *Complementary learning systems within the hippocampus: a neural network modelling approach to reconciling episodic memory with statistical learning.* Philosophical Transactions of the Royal Society B: Biological Sciences | Monosynaptic path learns statistics, trisynaptic path stores episodes |
| `grienberger2022entorhinal` | Grienberger & Magee (2022). *Entorhinal cortex directs learning-related changes in CA1 representations.* Nature | EC-III input instructs CA1 plasticity, beyond a summed skip |

## Prior art identified by the literature review (Nov 2026): dentate gyrus, sparse codes and MoE

| Key | Reference | Relation to limbic-llm |
|---|---|---|
| `bricken2023sparse` | Bricken et al. (2023). *Sparse Distributed Memory is a Continual Learner.* arXiv:2303.11934 | Top-k SDM-MLP learns continually without replay; closest analogue of DentateGyrus as a continual learner |
| `pan2025scaling` | Pan et al. (2025). *Scaling Linear Attention with Sparse State Expansion.* arXiv:2507.16577 | Row-sparse top-k state updates in linear attention: k-WTA addressing of a delta-style state |
| `du2025mom` | Du et al. (2025). *MoM: Linear Sequence Modeling with Mixture-of-Memories.* arXiv:2502.13685 | Mixture-of-Memories: a router sends tokens to separate linear memory states; the published form of a DG → CA3 MoE |
| `he2024mixture` | He (2024). *Mixture of A Million Experts.* arXiv:2407.04153 | PEER: product-key top-k over a million single-neuron experts; the only regime where DG-as-MoE holds formally |
| `lample2019large` | Lample et al. (2019). *Large Memory Layers with Product Keys.* arXiv:1907.05242 | Product-key memory: sparse top-k addressing into a large key table |
| `chi2022representation` | Chi et al. (2022). *On the Representation Collapse of Sparse Mixture of Experts.* arXiv:2204.09179 | MoE routing clusters tokens around expert centroids: completion-like, the opposite of DG separation |
| `qiu2025demons` | Qiu et al. (2025). *Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models.* arXiv:2501.11873 | Global-batch load balancing (as in Qwen) yields domain-specialised experts that cluster similar inputs |
| `wang2024let` | Wang et al. (2024). *Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models.* arXiv:2407.01906 | ESFT: tune only task-routed experts. Published form of subnetwork-restricted fine-tuning in MoE; baseline for expert neurogenesis |
| `lin2025continual` | Lin et al. (2025). *Continual Learning via Sparse Memory Finetuning.* arXiv:2510.15103 | Sparse memory finetuning: update only the slots new data activates; strong forgetting baseline for W3 |
| `dasgupta2017neural` | Dasgupta et al. (2017). *A neural algorithm for a fundamental computing problem.* Science | FlyHash: random sparse expansion + WTA is locality-sensitive hashing; the right baseline for DG keys |
| `scharfman2016enigmatic` | Scharfman (2016). *The enigmatic mossy cell of the dentate gyrus.* Nature Reviews Neuroscience | Mossy cells are excitatory; corrects dentate_gyrus.py |
| `ahmad2019how` | Ahmad & Scheinkman (2019). *How Can We Be So Dense? The Benefits of Using Highly Sparse Representations.* arXiv:1903.11257 | k-winners sparse layers with winner-only gradients, as in dentate_gyrus.py |

## Prior art identified by the literature review (Nov 2026): CA2 and steering

| Key | Reference | Relation to limbic-llm |
|---|---|---|
| `kohara2014cell` | Kohara et al. (2014). *Cell type–specific genetic and optogenetic tools reveal hippocampal CA2 circuits.* Nature Neuroscience | DG → CA2 input and a parallel CA2 → deep-CA1 circuit; corrects ca2.py and the serial CA2 wiring |
| `mankin2015hippocampal` | Mankin et al. (2015). *Hippocampal CA2 Activity Patterns Change over Time to a Larger Extent than between Spatial Contexts.* Neuron | CA2 population code drifts over hours to days: a slow episode tag, not a 10-token context |
| `alexander2018ca` | Alexander et al. (2018). *CA2 neuronal activity controls hippocampal low gamma and ripple oscillations.* eLife | Raising CA2 activity suppresses ripples: CA2 as a gate and stabiliser |
| `joad2026there` | Joad et al. (2026). *There Is More to Refusal in Large Language Models than a Single Direction.* arXiv:2602.02132 | Any refusal direction traces the same refusal versus over-refusal curve; selectivity must come from the gate (A1, J1) |
| `herbster2026activation` | Herbster et al. (2026). *Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence.* arXiv:2604.08169 | Aligned-model direction with a token-level gate restores alignment at runtime; closest precedent for A6 |
| `scalena2024multiproperty` | Scalena et al. (2024). *Multi-property Steering of Large Language Models with Dynamic Activation Composition.* arXiv:2406.17563 | Dynamic activation composition: steering intensity modulated per generation step |
| `jorgensen2023improving` | Jorgensen et al. (2023). *Improving Activation Steering in Language Models with Mean-Centring.* arXiv:2312.03813 | Mean-centring: static target-minus-mean steering, ancestor of the gist contrast |
| `yu2026break` | Yu et al. (2026). *Can We Break LLMs Out of Self-Loops? Fine-Grained Reasoning Control with Activation Steering.* arXiv:2607.18100 | Failed reasoning traces get stuck in self-loops; the predicted failure mode of open-loop gist steering |
| `sun2024massive` | Sun et al. (2024). *Massive Activations in Large Language Models.* arXiv:2402.17762 | Massive activations at delimiter and first tokens; a confound for EMA-based gist directions |

## Prior art identified by the literature review (Nov 2026): pathfinding, replay and cognitive maps

| Key | Reference | Relation to limbic-llm |
|---|---|---|
| `muller1996hippocampus` | Muller et al. (1996). *The hippocampus as a cognitive graph.* The Journal of general physiology | Hippocampus as a cognitive graph: place cells as nodes, CA3 synapses as edges; lineage of S&A 2005 |
| `teyler1986hippocampal` | Teyler & DiScenna (1986). *The hippocampal memory indexing theory.* Behavioral Neuroscience | Memory indexing theory: the hippocampus stores pointers, not content |
| `stachenfeld2017hippocampus` | Stachenfeld et al. (2017). *The hippocampus as a predictive map.* Nature Neuroscience | Successor representation as a hippocampal predictive map; policy-averaged relative of the S&A goal field |
| `pfeiffer2013hippocampal` | Pfeiffer & Foster (2013). *Hippocampal place-cell sequences depict future paths to remembered goals.* Nature | Replay sequences depict future paths to goals: replay as planning, absent from replay.py |
| `mattar2018prioritized` | Mattar & Daw (2018). *Prioritized memory access explains planning and hippocampal replay.* Nature Neuroscience | Gain × need replay priority; novelty priority approximates only the gain term |
| `kay2020constant` | Kay et al. (2020). *Constant Sub-second Cycling between Representations of Possible Futures in the Hippocampus.* Cell | Theta cycles alternate between possible futures; confirms the S&A theta-probe prediction |
| `graves2016hybrid` | Graves et al. (2016). *Hybrid computing using a neural network with dynamic external memory.* Nature | DNC temporal link matrix: successor-bound memory used for graph traversal |
| `eysenbach2019search` | Eysenbach et al. (2019). *Search on the Replay Buffer: Bridging Planning and Reinforcement Learning.* arXiv:1906.05253 | Graph search over stored states with learned goal distances: S&A pathfinding in ML form |
| `andrychowicz2017hindsight` | Andrychowicz et al. (2017). *Hindsight Experience Replay.* arXiv:1707.01495 | Hindsight relabelling; the S&A sharp-wave goal-field write is a form of it |
| `momennejad2023evaluating` | Momennejad et al. (2023). *Evaluating Cognitive Maps and Planning in Large Language Models with CogEval.* arXiv:2309.15129 | LLMs hallucinate invalid trajectories on graph planning (CogEval) |
| `saparov2025transformers` | Saparov et al. (2025). *Transformers Struggle to Learn to Search.* International Conference on Learning Representations (ICLR), arXiv:2412.04703 | Learned search degrades with graph size, and CoT does not fix it |
| `balesni2024lessons` | Balesni et al. (2024). *Lessons from Studying Two-Hop Latent Reasoning.* arXiv:2411.16353 | Latent two-hop composition of synthetic facts fails: the missing-key problem in LLMs |
| `chandra2025episodic` | Chandra et al. (2025). *Episodic and associative memory from spatial scaffolds in the hippocampus.* Nature | Vector-HaSH: a fixed scaffold index separated from content, with sequence memory along the scaffold |
| `george2021clonestructured` | George et al. (2021). *Clone-structured graph representations enable flexible learning and vicarious evaluation of cognitive maps.* Nature Communications | Clone-structured cognitive graphs: learned graphs with planning |
| `duvelle2021hippocampal` | Duvelle et al. (2021). *Hippocampal place cells encode global location but not connectivity in a complex space.* Current Biology | Place cells encode global location but not connectivity; against an S&A prediction |
| `widloski2022flexible` | Widloski & Foster (2022). *Flexible rerouting of hippocampal replay sequences around changing barriers in the absence of global place field remapping.* Neuron | Replay reroutes around changing barriers: path knowledge in replay, not place-field gradients |

Citations to web resources (Qwenputer, Jev, LAYA) use the year accessed. Jev's details come from jevai.net, a third-party summary; TypeSafe AI's own launch post is the primary source, and its training method (RLCD) is unpublished. The bibliography was audited on 2 Nov 2026 against Crossref, arXiv and OpenAlex (see [`bib_audit.md`](research/reviews/2026-11-02_literature/bib_audit.md)); every entry with a DOI now carries it.
