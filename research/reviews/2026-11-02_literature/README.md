# limbic-llm in context: literature review and research programme

*Review **RV2**. Reviews limbic-llm at commit 3799b36 (2 November 2026); committed on top of `feat/round1-side-memory` at 1620784. §3a reads the round-1 results (P7–P12) and §3b the round-2 results (P13–P16) against it. Findings are tracked in the register as RV2.X1–F1 and bugs as B12–B15; proposed experiments are L1–L16. Covers the four research tracks (CA3→CA1 skip paths, CA2 side-channel injection, DG→CA3 as mixture of experts, Gated DeltaNet as recurrent memory), the Samsonovich & Ascoli (2005) pathfinding model, the bibliography, and the code and results. Seven track reports with full argument and citations are linked in the final section.*

## Bottom line

Taken one at a time, almost every mechanism in limbic-llm is already published, usually without the hippocampal labels. CA3 retrieval is modern-Hopfield attention. The successor-bound store is Sompolinsky–Kanter sequence memory, which in transformers is the induction-head mechanism. The identity "Gated DeltaNet = one gradient step on the Titans loss" is stated in the GDN paper itself. "One-hot keys = exact RAM" is Kohonen's correlation-matrix memory. A dentate-gyrus-gated mixture of experts with CA3 storage and replay is HiCL. The CA1 comparator features are the InferSent matching vector. Replay into slow weights during "sleep" was published twice in 2026.

The defensible contributions lie elsewhere, in four areas:

1. **Loop control.** The control-theoretic treatment of activation re-injection loops: subtractive versus divisive feedback, a delay-limited critical gain, and primacy by self-rehearsal in closed loops.
2. **Two untested CA2 constructions.** Gist momentum steering, and a harm controller that integrates over time.
3. **One empirical observation on Qwen3.5.** A small mid-depth side module lets a frozen GDN hybrid recall a needle 500 tokens back with attention windowed. As run, this is confounded with plain adapter repair.
4. **The pathfinding gap.** A clear opening the attached Samsonovich & Ascoli paper points to: goal-directed, multi-hop path search through a successor memory, which neither the repo nor the LLM-memory literature implements.

The headline small-model results need restating before any of this can be built on. The CPU associative-recall task can be solved to 0.476 without any key–value binding (simulated independently here; see §3). Full attention (0.485) and Titans (0.483–0.490) sit at that ceiling, and the "capacity" curve in M4 follows it point for point. So the experiments show that successor-tied memories bind, and little else yet.

The most valuable next steps are:

1. **Measurement fixes**, which are cheap: an MQAR-style benchmark with a binding-free baseline, multiple seeds, and learning-rate sweeps.
2. **Adapter and state-probe controls on Qwen3.5**, to decide what P2 is.
3. **A missing-key multi-hop task with chained successor reads**, the experiment the attached paper suggests and the one most likely to give a result that is new to ML.

## 1. Where limbic-llm sits

Hippocampus-inspired machine learning has developed along three lines, and limbic-llm draws on all three without yet being anchored in any one.

The first line treats transformers *as* models of the hippocampal formation. In TEM-t, attention keys come from position-like structural codes and values from sensory codes, and modern Hopfield retrieval is formally attention ([Whittington et al. 2021](https://arxiv.org/abs/2112.04035); [Ramsauer et al. 2020](https://arxiv.org/abs/2008.02217)). The ICLR 2026 Hippoformer, whose name prompted the repo's rename, belongs here. It couples a TEM descendant with a meta-MLP relational memory to a transformer for spatial prediction, which is a different thing from the "EC + Titans-style memory" description in REFERENCES.md.

The second line builds memory modules for LLMs with hippocampal motivation: surprise-segmented episodic memory ([EM-LLM](https://arxiv.org/abs/2407.09450)), Larimar, HippoRAG and [HippoRAG 2](https://arxiv.org/abs/2502.14802), and the delta-rule side memories attached to frozen backbones such as Infini-attention ([Munkhdalai et al. 2024](https://arxiv.org/abs/2404.07143)) and δ-mem ([Lei et al. 2026](https://arxiv.org/abs/2605.12357)).

The third line imports hippocampal principles into continual learning: complementary learning systems, brain-inspired replay ([van de Ven et al. 2020](https://doi.org/10.1038/s41467-020-17866-2)), sparse distributed memory as a continual learner ([Bricken et al. 2023](https://arxiv.org/abs/2303.11934)), and HiCL's DG-gated experts ([Kapoor et al. 2026](https://doi.org/10.1609/aaai.v40i27.39411)).

limbic-llm is unusual because it keeps the whole circuit (EC, DG, CA3, CA2, CA1, subiculum, replay) in one codebase and treats the stability of feedback as a first-class problem. That breadth is also its main exposure. Each component has a closer-fitting precedent in one of the three lines, and the documents often credit a blog post or claim derivation where a primary source exists. The bibliography audit found all 82 existing entries real (none fabricated): 22 need corrections, 5 are misdescribed, and none carried a DOI. The most consequential misattributions:

- The GDN–Titans identity, presented in genesis.md §3 as derived here, is in [Yang, Kautz & Hatamizadeh 2025](https://arxiv.org/abs/2412.06464) §3 and in Longhorn ([Liu et al. 2024](https://arxiv.org/abs/2407.14207)).
- The "graded runtime controller" is largely anticipated by AdaSteer ([Zhao et al. 2025](https://doi.org/10.18653/v1/2025.emnlp-main.1248)) and PID steering ([Nguyen et al. 2025](https://arxiv.org/abs/2510.04309)).
- Samsonovich & Ascoli, the closest classical model of the successor store, is absent from the bibliography.

## 2. The four tracks against the literature

### 2.1 CA3→CA1 and the skip paths

The CA3→CA1 pathway is the serial path, and the skip-connection question is really about the two monosynaptic entorhinal inputs. Biology gives a sharp, testable division of labour that the repo has not yet used. Blocking CA3 output abolishes one-trial learning and pattern completion, while the temporoammonic (EC-III→CA1) route supports incremental learning ([Nakashiba et al. 2008](https://doi.org/10.1126/science.1151120)). Computational work recasts this as the monosynaptic path learning statistics and the trisynaptic path storing episodes ([Schapiro et al. 2017](https://doi.org/10.1098/rstb.2016.0049)). EC-III input to CA1 acts as an *instructive* signal that directs plasticity, more than a summed residual ([Grienberger & Magee 2022](https://doi.org/10.1038/s41586-022-05378-6)). CA1 is best modelled as computing the error between CA3's prediction of the next entorhinal state and the state that arrives ([Chen et al. 2024](https://doi.org/10.1016/j.neuron.2024.05.024); [Ketz et al. 2013](https://doi.org/10.1371/journal.pcbi.1003067)).

The repo's comparator falls short of this in two ways. With `sequential=True`, the query at *t* retrieves a stored *successor*, a prediction of *t*+1, and compares it with *x_t*. The comparison is therefore mistimed by one step. Separately, the "novelty" output is supervised to match token loss, so it is distilled surprisal and not a comparator output. The fusion vector [a; r; a⊙r; |a−r|] is the heuristic-matching feature of [Mou et al. 2015](https://arxiv.org/abs/1512.08422) and [Conneau et al. 2017](https://arxiv.org/abs/1705.02364) and should be credited.

On the ML side, a temporoammonic skip is a long residual into a late comparator. Its direct competitors are value residual learning ([Zhou et al. 2024](https://arxiv.org/abs/2410.17897)), DenseFormer ([Pagliardini et al. 2024](https://arxiv.org/abs/2402.02622)) and hyper-connections ([Zhu et al. 2024](https://arxiv.org/abs/2409.19606)). No experiment in the repo compares against them, so the track has no result yet.

The closed temporal loop of lagged self-states is the Feedback Transformer's organisation ([Fan et al. 2020](https://arxiv.org/abs/2002.09402)), and TransformerFAM ([Hwang et al. 2024](https://arxiv.org/abs/2404.09173)) applies a similar loop to pretrained models. The age kernel belongs to the ALiBi and Forgetting-Transformer family ([Lin et al. 2025](https://arxiv.org/abs/2503.02130)), and the time channel is the Laplace stage of [Shankar & Howard 2012](https://doi.org/10.1162/neco_a_00212). That stage matches LEC temporal-context cells ([Tsao et al. 2018](https://doi.org/10.1038/s41586-018-0459-6)) better than hippocampal time cells, which need the inverse transform. Contextual position encoding ([Golovneva et al. 2024](https://arxiv.org/abs/2405.18719)) already gives transformers salience-gated counts, so the claim that LLMs have no sense of elapsed time is overstated.

**What is new in this track** is the loop-gain analysis. The closed-loop gain law and delay-induced instability are textbook ([Black 1934](https://doi.org/10.1002/j.1538-7305.1934.tb00652.x); [Brunel & Hakim 1999](https://doi.org/10.1162/089976699300016179)). Applying them to a memory re-injection channel, with a confirmed prediction, was not found in prior work. Both the code-review and CA3–CA1 tracks reproduced the critical gain *K* = 1.680 (period 44.1 tokens) independently.

That analysis carries a correction the design note misses. The CA3–CA1 track computed the Bode sensitivity integral for *τ* = 16, *ρ* = 0.9, *K* = 1. Subtractive feedback halves DC echo (|S| = 0.50), but it amplifies content near a 46-token period 2.5-fold, and about half of all frequencies are amplified. "High-pass: cuts echo, passes new content" is therefore only half true, and this may explain T10 (subtractive feedback did not bound the echo).

The anatomy needs fixing in four places:

- An excitatory ventral-CA1→dorsal-CA3 back-projection exists ([Lin et al. 2021](https://doi.org/10.1371/journal.pbio.3001127)), contrary to temporal-loop.md.
- The theta justification is lag-locked, whereas [Hasselmo et al. 2002](https://doi.org/10.1162/089976602317318965) use a global encoding/retrieval phase.
- `HippocampalBlock` lacks the EC-II→CA3 read.
- The block puts CA2 in series between CA3 and CA1.

### 2.2 CA2 as a side channel for alignment and chain-of-thought

The biological premise needs narrowing before the engineering. CA2 is required for social *identity* memory ([Hitti & Siegelbaum 2014](https://doi.org/10.1038/nature13028)). It can promote aggression through the lateral septum ([Leroy et al. 2018](https://doi.org/10.1038/s41586-018-0772-0)), and its population code drifts over hours to days ([Mankin et al. 2015](https://doi.org/10.1016/j.neuron.2014.12.001)). Raising CA2 activity *suppresses* ripples ([Alexander et al. 2018](https://doi.org/10.7554/eLife.38052)). The defensible reading is CA2 as a context selector and gate. Nothing supports "societal norms drive CA2." Two docstring claims are outdated: that CA2 lacks mossy-fibre input, and that it is serially downstream of CA3. DG→CA2 input and a parallel DG→CA2→deep-CA1 circuit were shown by [Kohara et al. 2014](https://doi.org/10.1038/nn.3614).

In ML, the alignment controller is assembled from published parts:

- static refusal directions and abliteration ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717));
- probe-gated steering (CAST, [Lee et al. 2024](https://arxiv.org/abs/2409.05907));
- input-adaptive coefficients (AdaSteer);
- per-step intensity ([Scalena et al. 2024](https://arxiv.org/abs/2406.17563));
- PID control, whose integral runs across *layers* within one forward pass ([Nguyen et al. 2025](https://arxiv.org/abs/2510.04309));
- token-level harm accumulation on the monitoring side ([Li et al. 2025](https://arxiv.org/abs/2506.09996)).

Re-aligning a misaligned model at runtime with a direction taken from its aligned sibling, gated per token, has been shown for open-ended generation ([Herbster et al. 2026](https://arxiv.org/abs/2604.08169)). A recent result sharpens what A6 can show. Any refusal direction traces the same refusal versus over-refusal curve ([Joad et al. 2026](https://arxiv.org/abs/2602.02132)), so selectivity has to come from the *gate*, the harm probe, and not from the direction. That makes the probe experiments (A1, J1) decisive, not preliminary.

The controller as coded is weaker than described. At the defaults (λ = 0.9, κ = 0.5) the gain reaches 0.875 of the cap after three harmful tokens, with a time constant of about ten tokens. This figure is an analytic calculation by the CA2 track, not a model run. Because the integrator resets between sequences, it behaves as a smoothed CAST switch, and the multi-turn escalation of A3/H4 cannot occur. Its feedback also closes only through sampled tokens, so the op-amp stability analysis does not transfer to A4.

For chain-of-thought, fixed steering vectors already shorten reasoning substantially:

- SEAL;
- concise-vector steering ([Azizi et al. 2025](https://arxiv.org/abs/2507.04742));
- manifold steering;
- thinking-speed control.

Memory-retrieved steering is published as DSEM ([Do et al. 2025](https://doi.org/10.18653/v1/2025.findings-acl.706)) and STIR ([Shi et al. 2026](https://arxiv.org/abs/2602.04925)).

**The one construction with no precedent found is gist momentum**: a capped, normalised fast-minus-slow running average of the model's *own* residual stream over time. Its nearest relatives are static mean-centring ([Jorgensen et al. 2023](https://arxiv.org/abs/2312.03813)) and cross-layer momentum in vision-language-action models. It is also open-loop positive feedback. Failed reasoning traces get stuck in self-loops ([Yu et al. 2026](https://arxiv.org/abs/2607.18100)), and over-shortening hurts accuracy ([Sun et al. 2025](https://arxiv.org/abs/2503.22048)). The predicted failure mode is therefore specific: self-loop rate rises with gain. Massive activations at delimiter tokens ([Sun et al. 2024](https://arxiv.org/abs/2402.17762)) are a predicted confound for any EMA difference. The "under negative feedback" version (R3) is not implemented yet, and the gist arm is not wired into `run.py`.

### 2.3 DG→CA3 as a mixture of experts

The dentate gyrus is an expansion followed by a sparse code. It is not a router choosing among sub-networks, and this distinction decides the track. The `DentateGyrus` module (learned expansion, k-WTA, per-unit gain) is a top-k sparse layer of a well-known kind ([Ahmad & Scheinkman 2019](https://arxiv.org/abs/1903.11257); [Bricken et al. 2023](https://arxiv.org/abs/2303.11934)). Used as GDN write keys, it is a Kanerva-style sparse-address memory, close to Sparse State Expansion ([Pan et al. 2025](https://arxiv.org/abs/2507.16577)).

"DG as MoE gate" holds formally only in the fine-grained limit, where each expert is a single neuron selected by top-k, as in product keys ([Lample et al. 2019](https://arxiv.org/abs/1907.05242)) and PEER ([He 2024](https://arxiv.org/abs/2407.04153)). The published form of an MoE over CA3-like memory states is Mixture-of-Memories ([Du et al. 2025](https://arxiv.org/abs/2502.13685)). HiCL already implements a DG-gated expert model with CA3 storage and prioritised replay.

The analogy runs *backwards* for coarse routers like Qwen's. Load-balanced routers send similar tokens to the same experts and collapse representations toward expert centroids ([Chi et al. 2022](https://arxiv.org/abs/2204.09179); [Qiu et al. 2025](https://arxiv.org/abs/2501.11873)). That is clustering, closer to completion than separation. Even random sparse expansion preserves similarity, as locality-sensitive hashing does ([Dasgupta et al. 2017](https://doi.org/10.1126/science.aam9868)), and it decorrelates only in specific regimes ([Babadi & Sompolinsky 2014](https://doi.org/10.1016/j.neuron.2014.07.035)).

There is also a capacity limit. Under GDN's linear readout, sparsity cannot raise the *d_k* exact-capacity bound, because the gains of sparse coding require a non-linear readout ([Tsodyks & Feigel'man 1988](https://doi.org/10.1209/0295-5075/6/2/002)). The repo's own data agree: `tied_x8` and `dg_x8` both score 1.00 at every load tested. M7 is a ceiling effect and T13 is a single seed. T16 does not test Treves–Rolls, because no recurrent CA3 completion stage exists ([Neunuebel & Knierim 2014](https://doi.org/10.1016/j.neuron.2013.11.017)). Neurogenesis is untested: `grow()` is never called, and the code-review track showed that the age-dependent gradient boost is normalised away by Adam.

The DG/MoE track has one novel and useful opening. Measure whether Qwen MoE router codes separate or cluster near-duplicate inputs. Then use them, or FlyHash codes, as keys for a side memory. The prediction is that router codes lose for similar items. A second opening connects to your subnetwork-restricted fine-tuning project. "Expert neurogenesis" for continual fine-tuning sits next to ESFT ([Wang et al. 2024](https://arxiv.org/abs/2407.01906)) and sparse memory finetuning ([Lin et al. 2025](https://arxiv.org/abs/2510.15103)). Both of those are the published forms of restricting updates to a functional sub-network, and both are required baselines.

### 2.4 Gated DeltaNet as a recurrent memory

The formal claims are known. GDN as online regression is in the GDN and DeltaNet papers ([Yang et al. 2024](https://arxiv.org/abs/2406.06484)), in Longhorn, in Miras ([Behrouz et al. 2025](https://arxiv.org/abs/2504.13173)), and in test-time regression ([Wang, Shi & Fox 2025](https://arxiv.org/abs/2501.12352)). The d_k-bounded exact recall with orthogonal keys is [Kohonen 1972](https://doi.org/10.1109/TC.1972.5008975) and [Schlag et al. 2021](https://arxiv.org/abs/2102.11174). Successor-bound addressing is the shift-then-bind of H3 ([Fu et al. 2022](https://arxiv.org/abs/2212.14052)) and induction heads ([Olsson et al. 2022](https://arxiv.org/abs/2209.11895)), and it is the design principle of Memory Mosaics ([Zhang et al. 2024](https://arxiv.org/abs/2405.06394)).

"GDN state = CA3" is biologically weak. CA3 is a Hebbian, autoassociative attractor. GDN is a linear, error-corrective heteroassociative readout. One-shot instructive plasticity is documented in CA1 (BTSP, [Bittner et al. 2017](https://doi.org/10.1126/science.aan3846)). Key–value accounts of brain memory separate the retrieval code from the stored content ([Gershman, Fiete & Irie 2025](https://doi.org/10.1016/j.neuron.2025.02.029)).

The pretrained results line up with the hybrid-architecture literature. P1 (Qwen3.5's GDN layers alone do not recall the needle) is expected: pretrained Qwen3.5 GDN layers attenuate early chunks by orders of magnitude ([Chen et al. 2026, HLA](https://arxiv.org/abs/2610.05842)). Retrieval in hybrids concentrates in a few gather-and-aggregate heads ([Bick et al. 2025](https://arxiv.org/abs/2504.18574)), and recurrent layers learn long-range memory only when local attention cannot do the retrieval ([Cabannes et al. 2025](https://arxiv.org/abs/2509.24552)).

P2 is the most original result in the repo, and it is unexplained. With the default kernel, the loop's weight is exactly zero at the 500-token gap, and the loop gain moved only from 0.2689 to 0.2697. The 4.38 M-parameter side module is therefore behaving as an adapter. The windowed baseline sits *below* the answer-format floor (digit loss 4.03 against 1.84), so part of the 0→1.00 jump may be repair of a windowing distribution shift. State-offset tuning ([Kang et al. 2025](https://arxiv.org/abs/2503.03499)) is the natural control, and also an alternative injection site.

Two backlog items are already anticipated:

- **W1** (a fixed-size delta side memory on frozen Qwen3) is close to δ-mem, which attaches a delta-rule state to frozen Qwen3-4B.
- **W3** (sleep consolidation into adapters) sits beside two 2026 papers that consolidate context into fast weights offline: [Lee et al. 2026](https://arxiv.org/abs/2605.26099), which also reports gains on multi-hop graph retrieval, and [Behrouz et al. 2026](https://arxiv.org/abs/2606.03979).

For your never-ending-chat KV-trimming project, LoLA's three-tier design is the direct precedent: sliding window, a sparse exact cache admitted by self-recall error, and a linear state ([McDermott et al. 2025](https://arxiv.org/abs/2505.23666)). Fixed-state copying limits ([Jelassi et al. 2024](https://arxiv.org/abs/2402.01032)) bound what any GDN-only memory can promise there. Two newer results also matter: decoupled erase/write gates ([Hatamizadeh et al. 2026](https://arxiv.org/abs/2605.22791)) are what true slot-overwrite RAM needs, and β ∈ (0,1) rules out parity-like state tracking ([Grazzi et al. 2024](https://arxiv.org/abs/2411.12537)).

### 2.5 The attached paper: Samsonovich & Ascoli (2005)

[Samsonovich & Ascoli 2005](https://doi.org/10.1101/lm.85205) treat episodic retrieval as *search for a path* through a graph of remembered contexts. The model has three parts:

- CA3→CA3 weights (*J*) store traversed transitions.
- CA3→CA1 weights (*W*) store a goal field written during sharp waves by the max-recency rule W_ij = max(W_ij, 1/(t − t_j)). In modern terms this is hindsight goal relabelling of a shortest-observed-distance table ([Andrychowicz et al. 2017](https://arxiv.org/abs/1707.01495)).
- Theta-timed probes of candidate successors, from which a prefrontal selector picks the one that most excites the goal unit.

At 10⁴ contexts, the paper's greedy paths average 6.75 steps against 4.22 optimal. The pathfinding track re-implemented the simulation and obtained 6.69 ± 3.29 against 4.24 ± 0.66. Removing the two-hop tail of the CA3 activity packet lengthened paths about four-fold. These are single-graph, single-seed results and are provisional.

The paper's lineage is the cognitive graph ([Muller et al. 1996](https://doi.org/10.1085/jgp.107.6.663)) and memory-indexing theory ([Teyler & DiScenna 1986](https://doi.org/10.1037/0735-7044.100.2.147)). Its descendants mostly do not cite it (128 OpenAlex citers):

- the successor representation ([Stachenfeld et al. 2017](https://doi.org/10.1038/nn.4650));
- replay as planning ([Pfeiffer & Foster 2013](https://doi.org/10.1038/nature12112); [Mattar & Daw 2018](https://doi.org/10.1038/s41593-018-0232-z));
- clone-structured cognitive graphs ([George et al. 2021](https://doi.org/10.1038/s41467-021-22559-5));
- Vector-HaSH ([Chandra et al. 2025](https://doi.org/10.1038/s41586-024-08392-y));
- in ML, the DNC's temporal link matrix ([Graves et al. 2016](https://doi.org/10.1038/nature20101)) and search on the replay buffer ([Eysenbach et al. 2019](https://arxiv.org/abs/1906.05253)).

Later physiology confirmed one of its predictions and contradicted another. Theta cycles do alternate between possible futures ([Kay et al. 2020](https://doi.org/10.1016/j.cell.2020.01.014)). Barrier experiments, however, place path knowledge in replay and not in place-field gradients ([Duvelle et al. 2021](https://doi.org/10.1016/j.cub.2021.01.005); [Widloski & Foster 2022](https://doi.org/10.1016/j.neuron.2022.02.002)).

For limbic-llm, the paper matters more than its citation count suggests. The repo's successor-bound CA3/GDN store *is* the *J* matrix. What the repo lacks is the *W* goal field and any multi-step use of the store: `ca3.py`'s `n_iter` re-anchors the query to the cue and does not chain successors.

The ML gap this exposes is real. LLMs produce invalid trajectories on graph planning ([Momennejad et al. 2023](https://arxiv.org/abs/2309.15129)). Transformers struggle to learn search as graphs grow ([Saparov et al. 2024](https://arxiv.org/abs/2412.04703)). Latent two-hop composition of synthetic facts fails ([Balesni et al. 2024](https://arxiv.org/abs/2411.16353)), which is the paper's "missing contextual key" problem in LLM form. A side memory that chains successor reads, guided by a hindsight goal field written during replay, combines all three missing pieces, and no paper was found that does it beyond the attention window of a frozen hybrid. P4a is a direct instance of the problem. The loop recovers about one digit of a four-digit needle because key and value share a position, and chained successor reads are the S&A answer.

## 3. What the current results show

A model that never binds keys to values can score well above chance on the CPU associative-recall task. Queries are teacher-forced and shuffled, and values are drawn with replacement from 16. A model that remembers only the multiset of studied values, and strikes each one as it is revealed, scores 0.476 at 6 pairs. I simulated this independently of the code-review track ([`scripts/binding_free_baseline.py`](scripts/binding_free_baseline.py)) and obtained the same values: 0.476, 0.423, 0.319 and 0.243 at 6, 8, 16 and 32 pairs.

| Model (results file) | Recall | Binding-free ceiling |
|---|---:|---:|
| Full attention (`assoc_recall.json`) | 0.485 | 0.476 |
| Titans MAC / sequential | 0.483 / 0.490 | 0.476 |
| Strategy I (HippocampalLM) | 0.414 | 0.476 |
| `gdn_x8` at 8 / 16 / 32 pairs (`gdn_capacity.json`) | 0.428 / 0.320 / 0.222 | 0.423 / 0.319 / 0.243 |
| `tied_x8`, `dg_x8` at 8 / 16 / 32 pairs | 1.00 | — |

M1 therefore shows set memory, not binding. M4 is not a capacity curve: standard GDN never learned binding, even at loads far below *d_k*. Recall in recurrent models is highly sensitive to learning rate ([Okpekpe & Orvieto 2025](https://arxiv.org/abs/2508.19029)). The code-review track found that `gated_delta.py` initialises the short convolutions to the identity, so standard GDN starts with no token shift, the ingredient [Allen-Zhu 2025](https://arxiv.org/abs/2512.17351) identifies as decisive. M6 is confounded because `titans_gdn` ties the query and key projections, which hard-wires a previous-token address.

Five further problems from the code review:

1. The pretrained needle task, with one digit string in digit-free filler, is passkey retrieval ([Mohtashami & Jaggi 2023](https://arxiv.org/abs/2305.16300)), not binding. RULER's multi-key and multi-value variants are needed ([Hsieh et al. 2024](https://arxiv.org/abs/2404.06654)).
2. T11–T15 rest on one seed per arm, and T15's two seeds already disagree.
3. T6–T7 are analytic properties of a linear delayed loop, so "three untrained seeds" adds no evidence.
4. M3's 256-sequence replay buffer holds almost every prior fact, and there is no uniform-replay arm.
5. The CPU windowed baseline still uses sliding windows whose reach through depth (45 tokens) exceeds the 41-token gap.

The test suite runs clean in a CPU environment: 101 passed, 2 skipped because `transformers` was absent. The GDN scan and chunkwise kernels match a literal recurrence to 1e-15. The Titans module is a chunk-level variant, so its numbers describe that variant.

## 3a. Round-1 results read against the review

The GPU session's round-1 runs (`results/evolve/round1/`, register P7–P12) landed while this review was being written. They confirm the review's two central methodological points and sharpen the pathfinding proposal.

**The one-pair needle is passkey retrieval, and round 1 shows it.** The horizon loop's one-pair recall falls back to chance on digit 1 when a second pair is stored (P8: loss 2.17, against 2.30 at chance), so the one-pair success never used the query. P9 (exact recall by the O(T) loop on Qwen3.5 with every layer windowed, after 6,000 steps) and P10 (exact recall by W1 on Qwen3, every layer windowed) are therefore results about carrying *one* digit string across 500 tokens. They are not yet results about binding. P9 does weaken the adapter-repair reading of P2 for the all-windowed case: with every GDN layer windowed, nothing but the side channel can carry the string. P2 itself, with GDN intact, still needs the memory-free adapter control. W1's loss being "below full attention's" (0.0001 against 0.011) compares a trained side module with an untrained frozen base, so it is not a like-for-like comparison.

**P11 is the missing-key problem of Samsonovich & Ascoli, in LLM form.** With two stored pairs, W1 retrieves digits 2–4 at 0.94–0.99 but digit 1 is a coin flip (loss ≈ ln 2). The successor chain, the S&A *J* matrix, works. What fails is *choosing the start*, because the address at the space after "is" is the same token in both facts and carries little of the name three tokens back. The literature points to three remedies, and W1b already contains the first two:

1. **Context-bearing addresses.** The short convolutions in GDN, H3's shift SSM ([Fu et al. 2022](https://arxiv.org/abs/2212.14052)) and Canon layers ([Allen-Zhu 2025](https://arxiv.org/abs/2512.17351)) all supply a few tokens of history to the key. Note that `gated_delta.py` initialises the short convolutions to the identity, which removes that history at the start of training. Induction heads ([Olsson et al. 2022](https://arxiv.org/abs/2209.11895)) get the name into the key through a previous-token head.
2. **A separate dense read cue.** In the Treves–Rolls split ([Treves & Rolls 1992](https://doi.org/10.1002/hipo.450020209)), DG codes are write keys and the retrieval cue comes from a dense entorhinal path. Here the read address for the first digit should be computed from the query span (the name), not from the token before the answer. This is a test the repo has not yet run cleanly (T16 does not test it).
3. **A goal field that selects among candidate starts.** S&A's CA3→CA1 *W* matrix, written in hindsight at replay, scores candidate successors against the goal. In LLM terms, a second delta-rule matrix maps the query context to the stored start. The 2-pair and 3-pair needle in W1b is the smallest instance of the k-hop missing-key task proposed in §6, Phase 2.

DG keys learned faster (P10) but did not help start selection (P11), as the DG review predicted. Sparsity sharpens retrieval of what the address already encodes, and under a linear readout it does not raise capacity. It cannot add context the address lacks.

**Evaluation for round 2.** Two pairs and three pairs are a start. RULER's multi-key and multi-value needles ([Hsieh et al. 2024](https://arxiv.org/abs/2404.06654)) and an MQAR-style set with first-query accuracy should follow, so that "selects the queried pair" is measured against a binding-free floor. W1 should also be run head-to-head against δ-mem ([Lei et al. 2026](https://arxiv.org/abs/2605.12357)), the closest published system.

## 3b. Round-2 results read against the review

Round 2 tested W1b's three context-bearing address variants with 2 and 3 pairs (register P13–P16). **No variant selected the start.** Digit-1 loss stayed at about ln(number of pairs) in every arm (0.65–0.80 with 2 pairs, 0.97–1.34 with 3), so the read at the space after "is" returns an even mix of the stored first digits. The block-21 tap gave the only slight selection (digit-1 accuracy 0.48 against 0.33 chance with 3 pairs).

This is what the review's reading of P11 predicts, for a specific reason. Every W1b arm changes the *write-side address*, the history folded into the key at the answer position. In the Samsonovich & Ascoli model and in the Treves–Rolls split, start selection is a *read-side* operation: either a dense cue computed from the query (the name), or a goal field that scores candidate successors. Neither has been tried. The next arms are therefore:

1. a read query computed from the query span, for example attention-pooled hidden states over the name, used only for the first lookup;
2. the goal-field matrix of L6.

Two details matter for interpreting round 2:

- **P14 is confounded.** gdn mode (W1b.a) lost the chain as well as the start, but `CausalConv1d` initialises the short convolutions to the identity (B12). That arm therefore starts with no token shift and must learn one in 3,000 steps. Rerun it with a shift-biased or random initialisation before concluding that explicit successor binding is required.
- **P16 is the expected cost of an even-average address.** It blurs successor binding, the same trade-off seen in sparse-versus-dense keys: context in the address helps selection only if it is read at query time, not mixed into every write.

## 4. Biological mapping: corrections

| Repo statement | Evidence | Suggested restatement |
|---|---|---|
| CA1 pyramidal cells do not project back to CA3 | Excitatory vCA1→dCA3 projections ([Lin et al. 2021](https://doi.org/10.1371/journal.pbio.3001127)) | Back-projection is mixed-sign; let feedback sign be learned (CA3–CA1 E8) |
| CA2 lacks mossy-fibre input; lies in series CA3→CA2→CA1 | DG→CA2 input; parallel CA2→deep-CA1 circuit ([Kohara et al. 2014](https://doi.org/10.1038/nn.3614)) | Wire CA2 as a parallel, EC-II-driven route |
| Societal norms shape CA2 | Social identity memory; can promote aggression ([Hitti & Siegelbaum 2014](https://doi.org/10.1038/nature13028); [Leroy et al. 2018](https://doi.org/10.1038/s41586-018-0772-0)) | CA2 as context selector and gate |
| TemporalContext is "slowly drifting" | CA2 drift spans hours to days ([Mankin et al. 2015](https://doi.org/10.1016/j.neuron.2014.12.001)); ρ = 0.9 gives about 10 tokens | Rename or lengthen the time constant |
| Mossy cells provide feedback inhibition | Mossy cells are excitatory ([Scharfman 2016](https://doi.org/10.1038/nrn.2016.87)) | Inhibition from basket and other GABAergic cells |
| Static per-unit mossy-fibre gain | Detonation is frequency-dependent | Gate write strength by recent DG activity |
| Leaky integrators are Shankar–Howard time cells | Time cells need the inverse-Laplace stage | Call it a Laplace or temporal-context layer (LEC-like) |
| Sik et al. 1994 shows gain control | The paper proposes synchronisation and spike timing ([Sik et al. 1994](https://doi.org/10.1126/science.8085161)) | Present gain control as the project's analogy |
| The CA3 store is hippocampal indexing | Index theory stores pointers; the repo stores content | Describe it as a selective KV/episodic cache |
| Theta kernel follows Hasselmo 2002 | Hasselmo uses global phase; theta sequences are forward sweeps ([Foster & Wilson 2007](https://doi.org/10.1002/hipo.20345)) | Test a global phase gate (CA3–CA1 E5) |
| Human adult neurogenesis is established (Spalding 2013) | Contested in the literature | Cite both sides |

## 5. Where the contribution lies

| Idea | Verdict | Closest prior work |
|---|---|---|
| GDN = one SGD step on Titans loss; one-hot keys = exact RAM | Anticipated | GDN §3, Longhorn, Miras; Kohonen 1972, Schlag 2021 |
| Successor-bound (h_t→h_{t+1}) CA3 / tied GDN addressing | Anticipated | Sompolinsky–Kanter 1986, Levy 1996, induction heads, H3, S&A *J* matrix |
| DG k-WTA as MoE gate / sparse GDN keys | Anticipated | HiCL, PEER, SDM-MLP, Sparse State Expansion |
| CA1 comparator features and novelty head | Anticipated | InferSent; Chen 2024 prediction-error CA1 |
| CA2 as FiLM / activation addition | Anticipated | FiLM, ActAdd, CAA |
| Closed loop of lagged self-states | Anticipated | Feedback Transformer, TransformerFAM |
| Time channel | Anticipated (new only as a side channel on a frozen LLM) | Shankar & Howard 2012, DeepSITH, CoPE |
| Replay + sleep into slow weights | Anticipated | CLS, DER, Lee 2026, Behrouz 2026 |
| Fixed-size delta side memory on frozen Qwen (W1) | Anticipated | δ-mem 2026 |
| Loop-gain analysis of re-injection: delay-limited subtractive feedback, self-rehearsal primacy | **Partially novel** | Textbook control; no LLM precedent found. Needs multi-seed runs and the waterbed caveat |
| Leaky temporal integral of a harm score → capped gain | **Partially novel, incremental** | AdaSteer, CAST, PID steering, THRD; behaves as a smoothed switch at defaults |
| Gist momentum steering (own fast-minus-slow EMA) | **Novel construction, untested** | Mean-centring; cross-layer momentum in VLA models |
| Side module recruits frozen GDN layers for exact recall (P2) | **Partially novel, confounded** | HLA, SWAX, state-offset tuning |
| Goal-field pathfinding over a GDN successor memory, beyond the window | **Open gap** | S&A 2005, DNC, SoRB; not in LLMs |
| Qwen MoE router codes as DG keys | **No precedent found** | Predicted to fail on near-duplicates |

The safest framing for a paper is not "a hippocampus-inspired architecture." It is one of three narrower claims. (a) Stability and control of activation re-injection loops in language models, a systems paper with sharp predictions. (b) Hippocampal pathfinding over a recurrent side memory for multi-hop recall beyond the window. (c) The mechanism by which hybrid GDN models can be recruited for long-range recall. Each can be decided with a few weeks of GPU time.

## 6. Research programme

The programme is ordered by what unblocks the rest. IDs in parentheses refer to `research/REGISTER.md` items. The L-series is in register §4.8, where L3, L13, L14 and L15 point to the existing items O9, A2, A3 and R2 and to experiment numbers in the track reports.

**Phase 0: measurement, on CPU (days).** Fix the measurements before running anything new.

1. Replace `AssociativeRecall` with Zoology-style MQAR ([Arora et al. 2023](https://arxiv.org/abs/2312.04927)). Report first-query accuracy, where elimination is impossible, and the gap to the binding-free ceiling. Sweep learning rate per width and run at least five seeds.
2. Initialise GDN short convolutions randomly, as in fla, and re-run M4–M7, including loads above *d_k*. If sparse keys degrade more gracefully, that will show only above *d_k*.
3. Re-run the temporal-loop suite (T2–T16) on five seeds with bootstrap confidence intervals.
4. Use block windows in CPU baselines.
5. Add uniform-replay, sleep-only and awake-only arms to M3.
6. Run the cheap CA2 analytics: a token-level harm probe, and a fix for the controller's time constant.

**Phase 1: decisive tests on frozen Qwen (the two RTX 5090s).**

1. **P2 controls.** A parameter-matched memory-free adapter, LoRA and state-offset tuning, each with per-digit loss.
2. **State probes.** Linear probes for needle digits in each GDN layer's state, plus causal tracing to locate the route.
3. **Harder evaluation.** RULER S-NIAH, multi-key and multi-value variants at 2k–8k.
4. **State-space injection.** Write the side memory into a downstream GDN state, in that layer's key space, instead of the residual stream.
5. **W1 as a head-to-head against δ-mem.** P10 already shows the address shift lifts one-pair recall from one digit to four. The open questions are start selection with 2–3 pairs (P11, W1b) and whether W1 beats δ-mem at equal state size.

Together these settle whether P2 is memory or adaptation, and where the memory lives.

**Phase 2: the pathfinding experiment (the paper's contribution).**

1. Build a missing-key k-hop recall task beyond the window. A single read solves only k = 1, k chained successor reads should give roughly p^k accuracy, and attention should degrade with k.
2. Add a second delta-rule matrix written at hindsight "sharp-wave" events, the S&A goal field, and measure path success, length relative to optimal, and invalid-edge rate against the 2005 reference.
3. Compare reverse, goal-anchored replay with forward novelty replay, and gain × need priority ([Mattar & Daw 2018](https://doi.org/10.1038/s41593-018-0232-z)) with novelty priority.
4. Replace gist steering with an alternating-candidate version: top-k CA3 retrievals cycled across loop periods, with a CA1 goal selector, after [Kay et al. 2020](https://doi.org/10.1016/j.cell.2020.01.014).

**Phase 3: CA2 control.**

1. Compare controllers at matched mean injected norm (graded vs CAST vs AdaSteer vs PID vs static) on a Pareto of HarmBench ASR, over-refusal (XSTest, OR-Bench-Hard) and capability. Judge refusals with StrongREJECT ([Souly et al. 2024](https://arxiv.org/abs/2402.10260)), not substrings.
2. Test a turn-level integrator under Crescendo-style multi-turn attacks.
3. Run the three-model re-alignment: aligned, abliterated, and never-aligned base.
4. Measure gist steering against matched controls, recording self-loop rate, "wait" rate and answer flips. Include budget forcing ([Muennighoff et al. 2025](https://arxiv.org/abs/2501.19393)) and concise vectors as baselines. Only then build the negative-feedback gist.
5. Include the gist arm in the S1 saturation curve first.

**Phase 4: circuit and DG questions.**

1. A time-aligned comparator, so CA3's prediction from *t*−1 is compared with EC at *t*.
2. A trisynaptic/monosynaptic double dissociation (after Nakashiba 2008).
3. A frequency-domain waterbed test of subtractive feedback.
4. Value-residual, DenseFormer, hyper-connection and Feedback-Transformer baselines for the skip track.
5. Router-code separation curves on a Qwen MoE.
6. Expert neurogenesis against ESFT and sparse memory finetuning. This is the cleanest bridge to your functional-subnetwork fine-tuning work.

## 7. Track reports and bibliography

The track reports are dated snapshots at commit 3799b36, so their file paths use the pre-reorganisation layout (`results/evolve/REGISTER.md` is now `research/REGISTER.md`; result files are under `results/synthetic/` and `results/pretrained/`). For agents, start with [HANDOFF.md](HANDOFF.md) and [findings.json](findings.json).

Each track report gives the full argument, the closest-prior-work analysis, novelty verdicts and experiment designs:

- [ca3ca1.md](ca3ca1.md): CA3→CA1, skip paths, comparator, temporal loop and feedback control
- [ca2.md](ca2.md): CA2 biology, activation steering, alignment controller, chain-of-thought steering
- [dg_moe.md](dg_moe.md): dentate gyrus, sparse coding, MoE routing, neurogenesis, continual learning
- [gdn.md](gdn.md): Gated DeltaNet, fast weights, hybrids, pretrained results, consolidation
- [pathfinding.md](pathfinding.md): Samsonovich & Ascoli 2005, its lineage and descendants, mapping to limbic-llm
- [code_review.md](code_review.md): implementation fidelity, experimental validity, minimal publishable set
- [bib_audit.md](bib_audit.md): entry-by-entry audit of the existing bibliography

Bibliography files:

- [docs/references.bib](../../../docs/references.bib): replaced in this commit by the audited version, with all 82 keys kept, corrections applied, 49 DOIs added, six recommended additions, and the 72 promoted prior-art entries (160 in total).
- [new_refs.bib](new_refs.bib): 260 further verified sources cited in the track reports. The 72 closest prior-art works were promoted into `docs/references.bib` and REFERENCES.md ("Literature review" sections) in the same commit.

Coverage notes:

- The track agents verified every DOI and arXiv id against Crossref, arXiv or OpenAlex. A few candidates were dropped because they could not be verified (HippoTune as an arXiv record, DAMO, Wintzer 2014).
- Venues for some arXiv-first ML papers were not re-checked, so the new-references file lists them as preprints.
- Abstracts for several neuroscience papers were checked through publisher pages or web search because PubMed was not reachable.
- The Qwen3.5 MoE configuration (expert counts) comes from secondary sources and is provisional.
- No GPU experiment was re-run. The binding-free ceiling, the critical gain *K* = 1.680 and the S&A path lengths were recomputed; all other repo numbers come from the result files at commit 3799b36.
