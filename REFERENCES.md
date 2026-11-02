# References

The bibliography for limbic-llm, grouped by the part of the hippocampal mapping each work informs. Every entry has a BibTeX key in [`docs/references.bib`](docs/references.bib).

## Memory architectures implemented here

| Key | Reference | Used for |
|---|---|---|
| `behrouz2024titans` | Behrouz, Zhong & Mirrokni (2024). *Titans: Learning to Memorize at Test Time.* arXiv:2501.00663 | `NeuralMemory`, Titans layers (Strategy II) |
| `yang2025gated` | Yang, Kautz & Hatamizadeh (2025). *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR | `gated_delta.py` |
| `ramsauer2021hopfield` | Ramsauer et al. (2021). *Hopfield Networks is All You Need.* ICLR | CA3 retrieval |
| `krotov2016dense` | Krotov & Hopfield (2016). *Dense Associative Memory for Pattern Recognition.* NeurIPS | CA3 retrieval |
| `schlag2021linear` | Schlag, Irie & Schmidhuber (2021). *Linear Transformers Are Secretly Fast Weight Programmers.* ICML | Delta rule as fast weights |
| `vaswani2017attention` | Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS | EC encoder, cortex layers |
| `shazeer2017outrageously` | Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR | DG as an expert gate |
| `perez2018film` | Perez et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI | `CA2Modulator` |
| `qwenputer2026` | divagr.com (2026). *Qwenputer.* Blog post | Gated DeltaNet state as RAM |
| `qwen2025qwen3` | Qwen Team (2025). *Qwen3 Technical Report.* arXiv:2505.09388 | Attention-only pretrained target |
| `qwen2026qwen35` | Qwen Team (2026). *Qwen3.5.* Model release | Gated DeltaNet hybrid pretrained target |

## Hippocampal anatomy, physiology and theory

| Key | Reference | Relevance |
|---|---|---|
| `andersen2007hippocampus` | Andersen, Morris, Amaral, Bliss & O'Keefe (2007). *The Hippocampus Book.* Oxford University Press | Trisynaptic circuit |
| `bliss1993synaptic` | Bliss & Collingridge (1993). *A Synaptic Model of Memory: Long-Term Potentiation in the Hippocampus.* Nature 361 | Plasticity at CA3 → CA1 |
| `okeefe1978hippocampus` | O'Keefe & Nadel (1978). *The Hippocampus as a Cognitive Map.* Clarendon Press | Cognitive-map theory |
| `squire1992memory` | Squire (1992). *Memory and the Hippocampus.* Psychological Review 99(2) | Declarative memory |
| `scoville1957loss` | Scoville & Milner (1957). *Loss of Recent Memory after Bilateral Hippocampal Lesions.* JNNP 20(1) | Patient H.M. |
| `hafting2005microstructure` | Hafting, Fyhn, Molden, Moser & Moser (2005). *Microstructure of a Spatial Map in the Entorhinal Cortex.* Nature 436 | Grid cells (EC) |
| `eichenbaum2017integration` | Eichenbaum (2017). *On the Integration of Space, Time, and Memory.* Neuron 95(5) | Relational memory |
| `spalding2013dynamics` | Spalding et al. (2013). *Dynamics of Hippocampal Neurogenesis in Adult Humans.* Cell 153(6) | DG neurogenesis |
| `levy1996sequence` | Levy (1996). *A Sequence Predicting CA3 Is a Flexible Associator.* Hippocampus 6(6) | Sequential CA3 store |
| `jensen1996hippocampal` | Jensen & Lisman (1996). *Hippocampal CA3 Region Predicts Memory Sequences.* Learning & Memory 3 | Sequential CA3 store |
| `hasselmo2006role` | Hasselmo (2006). *The Role of Acetylcholine in Learning and Memory.* Curr. Opin. Neurobiol. 16(6) | Novelty-gated encoding (`novelty_lr`) |
| `lisman2005hippocampal` | Lisman & Grace (2005). *The Hippocampal-VTA Loop.* Neuron 46(5) | CA1 novelty → learning and decisions |
| `bein2020mnemonic` | Bein, Duncan & Davachi (2020). *Mnemonic Prediction Errors Bias Hippocampal States.* Nature Communications 11 | CA1 mismatch drives encoding |
| `hitti2014hippocampal` | Hitti & Siegelbaum (2014). *The Hippocampal CA2 Region Is Essential for Social Memory.* Nature 508 | CA2 function |
| `leroy2018circuit` | Leroy et al. (2018). *A Circuit from Hippocampal CA2 to Lateral Septum Disinhibits Social Aggression.* Nature 564 | CA2 also promotes aggression: a social-context selector, not a norm enforcer |
| `smith2016targeted` | Smith, Williams Avram, Cymerblit-Sabba, Song & Young (2016). *Targeted Activation of the Hippocampal CA2 Area Strongly Enhances Social Memory.* Mol. Psychiatry 21(8) | CA2 amplification |
| `boyle2024tuned` | Boyle, Posani, Irfan, Siegelbaum & Fusi (2024). *Tuned Geometries of Hippocampal Representations Meet the Computational Demands of Social Memory.* Neuron 112(8) | CA2 identity coding |
| `howard2002distributed` | Howard & Kahana (2002). *A Distributed Representation of Temporal Context.* J. Math. Psych. 46(3) | CA2 temporal context |
| `kumaran2012generalization` | Kumaran & McClelland (2012). *Generalization through the Recurrent Interaction of Episodic Memories.* Psychological Review 119(3) | Big-loop recurrence (REMERGE) |
| `koster2018bigloop` | Koster et al. (2018). *Big-Loop Recurrence within the Hippocampal System Supports Integration of Information across Episodes.* Neuron 99(6) | Big-loop recurrence |
| `mcclelland1995why` | McClelland, McNaughton & O'Reilly (1995). *Why There Are Complementary Learning Systems.* Psychological Review 102(3) | Replay consolidation |
| `wilson1994reactivation` | Wilson & McNaughton (1994). *Reactivation of Hippocampal Ensemble Memories during Sleep.* Science 265 | Sleep replay |
| `mcewen1999stress` | McEwen (1999). *Stress and Hippocampal Plasticity.* Annu. Rev. Neurosci. 22 | Background report |

## Hippocampus-inspired machine learning (prior art)

| Key | Reference | Overlap with this package |
|---|---|---|
| `li2026hippoformer` | Li, Cao, Wang, Yang, Zou & Hong (2026). *Hippoformer: Integrating Hippocampus-Inspired Spatial Memory with Transformers.* ICLR | EC + Titans-style memory; its name prompted the rename |
| `whittington2020tolman` | Whittington et al. (2020). *The Tolman-Eichenbaum Machine.* Cell 183(5) | EC structural vs. sensory codes |
| `whittington2022relating` | Whittington, Warren & Behrens (2022). *Relating Transformers to Models and Neural Representations of the Hippocampal Formation.* ICLR | Position encodings as grid cells |
| `kapoor2025hicl` | Kapoor, Mackey, Aloimonos & Lin (2025). *HiCL: Hippocampal-Inspired Continual Learning.* arXiv:2508.16651 | DG-gated mixture of experts |
| `das2024larimar` | Das et al. (2024). *Larimar: Large Language Models with Episodic Memory Control.* arXiv:2403.11901 | Episodic memory for LLMs |
| `gutierrez2024hipporag` | Gutiérrez, Shu, Gu, Yasunaga & Su (2024). *HippoRAG.* NeurIPS | Hippocampal indexing for retrieval |
| `fountas2025human` | Fountas et al. (2025). *Human-Inspired Episodic Memory for Infinite Context LLMs* (EM-LLM). ICLR | Surprise-based event segmentation |
| `ahn2025hema` | Ahn (2025). *HEMA: A Hippocampus-Inspired Extended Memory Architecture for Long-Context AI Conversations.* arXiv:2504.16754 | Conversational memory |
| `omidi2025memory` | Omidi et al. (2025). *Memory-Augmented Transformers: A Systematic Review from Neuroscience Principles to Technical Solutions.* arXiv:2508.10824 | Survey |

## CA2 candidates: activation steering

| Key | Reference | Relevance |
|---|---|---|
| `turner2023activation` | Turner, Thiergart, Leech, Udell, Vazquez, Mini & MacDiarmid (2023). *Activation Addition: Steering Language Models Without Optimization.* arXiv:2308.10248 | Side-channel steering vector at a middle layer, $h^l \leftarrow h^l + c\,\Delta h$; the FiLM special case $\gamma = 0$. Notebook in `docs/notebooks/` |

## CA1 candidates: calibrated decisions and predictive comparison

| Key | Reference | Relevance |
|---|---|---|
| `typesafe2026jev` | TypeSafe AI (2026). *Jev.* jevai.net | Single-pass structured decisions with calibrated confidence (RLCD) |
| `nandhakishor2026laya` | NandhaKishorM (2026). *LAYA.* github.com/NandhaKishorM/laya, Apache-2.0 | Open encoder decision engine: choice, score and P(true) in one forward pass |
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
| `sik1994inhibitory` | Sik, Ylinen, Penttonen & Buzsáki (1994). *Inhibitory CA1-CA3-Hilar Region Feedback in the Hippocampus.* Science 265 | CA1 → CA3 inhibitory back-projection; negative-feedback gain control |
| `treves1992computational` | Treves & Rolls (1992). *Computational Constraints Suggest the Need for Two Distinct Input Systems to the Hippocampal CA3 Network.* Hippocampus 2(2) | DG for storage, direct EC → CA3 path for retrieval cues |
| `buzsaki2002theta` | Buzsáki (2002). *Theta Oscillations in the Hippocampus.* Neuron 33(3) | Oscillatory timing of the loop |
| `macdonald2011hippocampal` | MacDonald, Lepage, Eden & Eichenbaum (2011). *Hippocampal "Time Cells" Bridge the Gap in Memory for Discontiguous Events.* Neuron 71(4) | Time cells; the time channel |
| `shankar2012scale` | Shankar & Howard (2012). *A Scale-Invariant Internal Representation of Time.* Neural Computation 24(1) | Log-spaced leaky integrators encode elapsed time |
| `sapolsky1986neuroendocrinology` | Sapolsky, Krey & McEwen (1986). *The Neuroendocrinology of Stress and Aging: The Glucocorticoid Cascade Hypothesis.* Endocrine Reviews 7(3) | A loop without negative feedback: the cautionary case |

## Dynamic steering and oscillatory sequence models

| Key | Reference | Relevance |
|---|---|---|
| `rimsky2024steering` | Rimsky, Gabrieli, Schulz, Tong, Hubinger & Turner (2024). *Steering Llama 2 via Contrastive Activation Addition.* ACL | Contrast vectors averaged over many pairs |
| `lee2024programming` | Lee et al. (2024). *Programming Refusal with Conditional Activation Steering.* arXiv:2409.05907 | Steering gated by hidden-state patterns |
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
| `rottger2024xstest` | Röttger et al. (2024). *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.* NAACL | Over-refusal benchmark |

## Continual learning and replay

| Key | Reference | Relevance |
|---|---|---|
| `buzzega2020dark` | Buzzega, Boschini, Porrello, Abati & Calderara (2020). *Dark Experience for General Continual Learning.* NeurIPS | Logit-distillation replay in `Consolidator` |

Citations to web resources (Qwenputer, Jev, LAYA) use the year accessed. Jev's details come from jevai.net, which describes itself as an independent community resource; check TypeSafe AI's own documentation before relying on them.
