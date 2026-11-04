# Bibliography audit of limbic-llm (commit 3799b36)

## Scope

The audit covers all 82 entries in `docs/references.bib` and the 82 rows of `REFERENCES.md`. The two files carry the same key set, so no entry is missing from either. Each entry was checked against its primary record: Crossref for journal and proceedings DOIs, the arXiv API for preprints, OpenAlex for venue records, and publisher or conference pages (ICLR, ACL Anthology, mlanthology, GitHub, Hugging Face) where neither registry holds the item. Every DOI in the corrected file resolves (49 of 49), and none is flagged as retracted. For the fifteen most load-bearing works and the three web resources, the repo's "Used for / Relevance" text was also compared with the papers' abstracts, full text or project pages, together with the sentences in `README.md`, `docs/genesis.md`, `docs/design/temporal-loop.md` and `limbic_llm/temporal_loop.py` that attribute findings to them.

The outcome is 54 entries verified as listed, 22 needing bibliographic correction, 5 whose citations are bibliographically right but cited for something the paper does not claim, and 1 not machine-verifiable (a 1978 monograph with no DOI). No entry is fabricated: every cited work exists, and every listed author list is correct where given. The problems are of three kinds. Venue records are stale: arXiv-only listings now have venues, as do papers renamed on publication. Author lists are shortened to "Surname and others". And a few sentences in the design documents credit to themselves, or to the cited paper, more than the paper supports. The corrected replacement file is `bib_audit_refs.bib`. It keeps every key unchanged so existing `\cite` commands still work, adds DOIs or arXiv ids throughout, and ends with six verified additions that the documents should cite.

## Systemic issues

**Stale "arXiv preprint" venues.** Eight entries cite a preprint for work that has since appeared at a venue. Titans is in NeurIPS 2025 ([Behrouz 2025](https://doi.org/10.52202/085713-3786)), not a 2024 preprint, and the README's "Behrouz et al., 2024" should change with it. HiCL is in AAAI 2026 ([Kapoor 2026](https://doi.org/10.1609/aaai.v40i27.39411)). Larimar went to ICML 2024 and CAST to ICLR 2025 ([Lee 2025](https://arxiv.org/abs/2409.05907)). Coconut and SEAL went to COLM 2025. Activation-Steered Compression appeared in Findings of ACL 2026 with an added fourth author ([Azizi 2026](https://doi.org/10.18653/v1/2026.findings-acl.1828)). The arXiv-only STIR paper was published at EMNLP 2026 under a new title ([Shi 2026](https://doi.org/10.18653/2026.emnlp-main.2278)). PID steering is ICLR 2026, but its key says 2025 and it is typed as `misc` with the venue hidden in `howpublished` ([Nguyen 2026](https://arxiv.org/abs/2510.04309)). Keys were left unchanged to avoid breaking citations, so several keys now carry a year that differs from the `year` field. This is cosmetic but worth knowing when grepping.

**Truncated author lists.** The ten reasoning-steering entries added in the CA2 track (`chen2025seal` through `shi2026internalizing`) list only "Surname and others". They now carry full author lists taken from Crossref or arXiv. `ramsauer2021hopfield` is expanded to its 16 authors. The EM-LLM entry splits the last author's name wrongly ("Ammar, Haitham Bou"), and *The Hippocampus Book* lists its five editors as authors.

**Retitled works.** Two arXiv records changed title after first posting. Turner et al. is now "Steering Language Models With Activation Engineering" ([Turner 2023](https://arxiv.org/abs/2308.10248)), and Omidi et al.'s survey now ends "…to Enhanced Model Architectures". STIR changed title on publication. The corrected file uses current titles and records the original title in `note`.

**Missing DOIs.** None of the 82 original entries had a `doi` field, although 49 of them have resolvable DOIs (journal articles, AAAI, ACL Anthology, and the Curran-registered NeurIPS 2022–2025 proceedings). All 49 are added. ICLR, ICML and COLM papers have no DOIs; they carry arXiv ids instead.

**Web resources with weak provenance.** The three web citations all exist, but two are attributed to a handle rather than a person. Qwenputer is by Divyansh Agrawal (blog at divagr.com, code at github.com/divagr18/qwenputer, Qwen3.5-4B models on Hugging Face). It was announced in late October 2026 and is a non-peer-reviewed engineering demonstration. The claim the repo takes from it is that one-hot keys with α = β = 1 turn a Gated DeltaNet head into an exact row-addressed memory. That claim is stated in the model card and follows directly from the gated delta rule, so the citation is sound but should be marked non-archival. LAYA is by Nandakishor Mukkunnoth; its benchmarks are self-reported. The Jev entry is the weakest. It cites jevai.net, a third-party site, rather than TypeSafe AI's own launch post (15 September 2026). The training method (RLCD) is described but not published. The one independent evaluation found during this audit reported good calibration on moderation (ECE 0.032) but poor calibration on answer rating (ECE 0.284). The repo should not cite Jev as evidence that calibrated single-pass decisions work as a CA1 comparator; that would need its own test (backlog J1–J5).

**Qwen3.5.** The bib cites the release with the 397B flagship's URL. The official citation in the Qwen model cards is "Qwen3.5: Towards Native Multimodal Agents" (Qwen Team, February 2026). The repo's pretrained experiments use Qwen3.5-2B: 24 layers in a 6 × (3 × Gated DeltaNet + 1 × gated attention) layout, released 2 March 2026. The corrected entry names that checkpoint.

## Where the documents misstate the cited work

**Gated DeltaNet as one gradient step on the Titans loss.** `genesis.md` §3 says the user proposed the analogy and the assistant "made it exact": the GDN update is one gradient step on the Titans memory loss at the decayed state. The identity is correct, but it is published. Yang, Kautz & Hatamizadeh write the delta rule as test-time SGD on ½‖S k − v‖² and the gate as an adaptive weight decay, building on the online-learning framework of Longhorn ([Yang 2025](https://arxiv.org/abs/2412.06464); [Liu 2024](https://arxiv.org/abs/2407.14207)). DeltaNet's own paper makes the same reading ([Yang 2024](https://doi.org/10.52202/079017-3668)). Titans' appendix states that its memory "is generalizing" Gated DeltaNet, Longhorn and RWKV-7, and its main text calls the decay "the generalization of forgetting mechanism in modern recurrent models" ([Behrouz 2025](https://doi.org/10.52202/085713-3786)). The README sentence "Gated DeltaNet is the linear special case of the Titans memory" is therefore a correct restatement of these papers and should cite them for it. The Miras framework later unifies all of these as associative memories with an internal objective ([Behrouz 2025b](https://arxiv.org/abs/2504.13173)), which is the right citation for the general claim. The idea goes back to linear attention as fast-weight programming ([Schlag 2021](https://arxiv.org/abs/2102.11174)).

**Scale-invariant time cells.** `temporal-loop.md`, `genesis.md` §5.3 and the `TimeChannel` docstring say a bank of leaky integrators at log-spaced time constants gives a scale-invariant code for elapsed time, "after Shankar & Howard's scale-invariant time cells". In the paper the leaky integrators compute the Laplace transform of the input history. The sequentially firing, scale-invariant "time cells" appear only after an approximate inverse Laplace operator (Post's formula, "TILT") is applied ([Shankar 2012](https://doi.org/10.1162/neco_a_00212)). `TimeChannel` implements the Laplace layer (seven integrators and damped oscillators, τ = 4…256) plus an unconstrained learned linear readout. It is a Laplace-domain clock, not a time-cell code, and the docstring should say so. A deep network that does include the inverse transform exists ([Kabir 2024](https://arxiv.org/abs/2412.15292), preprint), so the comparison is testable (experiment E2 below).

**CA1 → CA3 back-projection as gain control.** Sik et al. describe in vivo-labelled CA1 interneurons whose axons return, longitudinally, to CA3 and the hilus. They interpret this long-range inhibition as a means of synchronizing population activity and timing spikes for coordinated plasticity ([Sik 1994](https://doi.org/10.1126/science.8085161)). The anatomy in `temporal-loop.md` is reported correctly. But the REFERENCES.md gloss "negative-feedback gain control", and the op-amp law G = A/(1+Aβ) ≈ 1/β, are the project's own engineering analogy. They are not a finding of the paper and should be labelled as such. The analogy is also a stretch for a sparse interneuron population whose CA3 targets include interneurons as well as pyramidal cells.

**Harmfulness versus refusal.** `genesis.md` §5.7 and REFERENCES.md say jailbreaks and harmful fine-tuning "leave the harmfulness judgement intact". Zhao et al. show that *certain* jailbreak methods lower the refusal signal without reversing the internal harmfulness belief, and that adversarial fine-tuning has minimal effect on that belief ([Zhao 2025](https://doi.org/10.52202/085713-4688)). The paper does not test abliterated models, which is the A6 setting. The claim supports using a harmfulness probe as controller input, but the wording should keep "certain", and A6 should measure whether the harmfulness direction survives abliteration rather than assume it does.

**Novelty claim for the escalating controller.** `genesis.md` §5.6 says that what is new is "a runtime, graded controller". This is anticipated in large part. AdaSteer already sets per-input steering coefficients from an input's position along a harmfulness direction kept separate from the refusal direction, at inference time and without training ([Zhao 2025b](https://doi.org/10.18653/v1/2025.emnlp-main.1248)). PID steering already gives activation steering a closed-loop controller with an integral term ([Nguyen 2026](https://arxiv.org/abs/2510.04309)), but its integral accumulates error *across layers* within one forward pass. CAST gates steering on hidden-state conditions ([Lee 2025](https://arxiv.org/abs/2409.05907)). What remains distinct in limbic-llm is narrower. A *leaky temporal* integral of the harm score across tokens and turns drives a *capped* gain, so pressure accumulates over a conversation and decays when it stops. That is a real but incremental contribution, and the A2–A4 comparisons should include AdaSteer and PID steering as baselines, not only static steering and CAST.

**Other relabellings.** Hippoformer is not "EC + Titans-style memory". It couples mm-TEM, a Tolman–Eichenbaum Machine with a meta-MLP relational memory that forms grid-like codes, to a Transformer for 2D/3D spatial prediction ([Li 2026](https://mlanthology.org/iclr/2026/li2026iclr-hippoformer/); ICLR 2026, no DOI or arXiv id). Howard & Kahana's Temporal Context Model is filed under CA2 as "CA2 temporal context", but it is a behavioural model of recall that says nothing about CA2 ([Howard 2002](https://doi.org/10.1006/jmps.2001.1388)); the CA2 link is the project's hypothesis. Treves & Rolls is cited correctly in substance ([Treves 1992](https://doi.org/10.1002/hipo.450020209)): strong, non-associative mossy-fibre input drives storage, and the many associatively modifiable perforant-path synapses relay the retrieval cue. However, `temporal-loop.md`'s paraphrase ("a pattern-separated cue maps to a different code than the stored one") is the project's gloss, not the paper's information-theoretic argument. The REGISTER correctly records T16 as "Not supported" for a single seed.

## Closest prior work for the four tracks

The bibliography already contains most of the work a referee would name, but it does not always say how close that work is. On the DG → CA3 mixture-of-experts track, HiCL is closer than REFERENCES.md implies. It combines a grid-cell-like input layer, top-k dentate sparsity, a CA3 autoassociator, routing to experts by cosine similarity between sparse DG codes and per-task DG prototypes, EWC consolidation and prioritized replay ([Kapoor 2026](https://doi.org/10.1609/aaai.v40i27.39411)). The repo's k-WTA expansion gate and novelty-prioritized replay re-describe this design. Its distinct parts are neurogenesis reserve units and DG addressing of a GDN state (M5, M7).

On the CA2 side-channel track, memory-retrieved steering is published twice: DSEM retrieves steering vectors from a token-keyed episodic memory ([Do 2025](https://doi.org/10.18653/v1/2025.findings-acl.706)), and STIR retrieves reasoning-steering primitives from a library with gated injection ([Shi 2026](https://doi.org/10.18653/2026.emnlp-main.2278)). Static activation addition itself is from Turner et al. and Rimsky et al. ([Rimsky 2024](https://doi.org/10.18653/v1/2024.acl-long.828)). Steering to shorten chain-of-thought is well covered by SEAL ([Chen 2025](https://arxiv.org/abs/2504.07986)) and ASC ([Azizi 2026](https://doi.org/10.18653/v1/2026.findings-acl.1828)). That leaves the "gist" fast-minus-slow residual momentum vector as the CoT idea not found in the cited set. On alignment, circuit breakers reroute harmful representations through fine-tuning ([Zou 2024](https://doi.org/10.52202/079017-2651)), and the refusal direction that abliteration removes is from Arditi et al. ([Arditi 2024](https://doi.org/10.52202/079017-4322)).

On the GDN-as-memory track, the closest prior work is the DeltaNet/GDN/Titans/Miras line above, together with Qwenputer's exact one-hot addressing. Successor binding in CA3 has its own lineage in Levy's sequence-predicting CA3 and in Jensen & Lisman ([Jensen 1996](https://doi.org/10.1101/lm.3.2-3.279)), which the README sentence "as in models of CA3 sequence memory" should cite explicitly. The attached Samsonovich & Ascoli paper belongs in this group ([Samsonovich 2005](https://doi.org/10.1101/lm.85205)). It models CA3 as storing transitions between successive EC states and uses retrieval as pathfinding through that transition graph, which is close to the repo's h_t → h_{t+1} successor store. It is not in the bibliography and should be added. On the skip-connection track, the EC-III → CA1 temporoammonic path and the EC-II → CA3 direct path are standard anatomy ([Treves 1992](https://doi.org/10.1002/hipo.450020209); Andersen et al. 2007); the novelty claims are architectural, not anatomical.

## Where limbic-llm is novel and where it is anticipated

Stated plainly: the GDN–Titans equivalence is published, exact RAM via one-hot GDN keys is Qwenputer's, DG-gated experts with replay are HiCL's, and memory-retrieved steering is DSEM's and STIR's. Graded, input-adaptive safety steering is AdaSteer's, and closed-loop steering control is PID steering's. The parts that appear new are the closed, damped, regulated temporal loop on a frozen hybrid model, with clamped-sinusoid re-entry, subtractive feedback that rings above a measured loop gain (T6, T7), and an episode-minus-prior contrast. Two empirical results also stand out: that this side channel restores exact needle recall on Qwen3.5 only through the GDN layers above the tap (P1, P2), and the temporal leaky-integral escalation of the safety controller. These are modest, mostly single-seed CPU results plus one pretrained model family, and they should be reported as such.

## Biological accuracy

The anatomical corrections already in `genesis.md` are right: there is no DG → CA1 projection, the skip into CA1 is from EC layer III, and CA1 pyramidal cells do not project back to CA3. Three points need softening. The Sik back-projection is a timing and synchronization pathway in its authors' reading, not an established gain controller. The time channel is a Laplace-domain code, not time cells. CA2's role as a "norm controller" is metaphor: the cited biology shows a social-memory region that can *promote* aggression via the lateral septum ([Hitti 2014](https://doi.org/10.1038/nature13028); [Leroy 2018](https://doi.org/10.1038/s41586-018-0772-0)), as `genesis.md` already notes.

## Proposed experiments

E1 (extends A2–A4, priority high). If the repo's novelty is the temporal integral, a leaky-integral controller should beat AdaSteer's per-input coefficient on multi-turn escalation, at equal single-turn performance. Compare static steering, CAST, AdaSteer, PID steering and the escalating controller on the same Qwen3 models. Measure HarmBench attack success per turn, XSTest over-refusal and recovery time after benign turns.

E2 (extends W2 and T1, priority medium). If elapsed-time information needs the inverse transform, a TILT readout should answer elapsed-time questions more accurately and more scale-invariantly than the current learned readout. Replace `TimeChannel.out` with a fixed Post-approximation inverse over log-spaced s, and compare elapsed-time error across a 4× rescaling of gaps.

E3 (extends M7 and T13, priority medium). Run HiCL's prototype-cosine DG router as a baseline against the repo's k-WTA DG gate on the overlapping-key recall task in the roadmap. Measure recall and forgetting across key-similarity levels with at least three seeds.

E4 (extends R5, priority low). Run DSEM- and STIR-style retrieved steering against gist momentum steering on GSM8K/MATH-500, matched for reasoning tokens.

## Open questions

Does the harmfulness direction survive abliteration of Qwen3-1.7B, given that Zhao et al. tested jailbreaks and fine-tuning but not direction ablation? Is the ICML 2024 venue of HarmBench and Larimar consistent with the PMLR records? Their arXiv records were verified here, but PMLR was not queried. Should the Qwenputer and LAYA citations be pinned to commit hashes or releases, since both projects changed substantially within weeks of being cited?

## Entry-level table

Status codes: *verified* means the bibliographic data are correct as listed (DOIs or ids may still be added); *corrected* means the bib needed a fix; *misdescribed* means the data are right but the repo cites the work for something it does not claim; *unverifiable* means no registry record could be retrieved.

### Memory architectures implemented in this package

| key | status | issue | fix |
|---|---|---|---|
| `behrouz2024titans` | corrected | Listed as 2024 arXiv preprint; it is published in NeurIPS 2025 (Adv. NeurIPS 38, pp. 125925–125962). README cites "Behrouz et al., 2024". | Change to @inproceedings NeurIPS 2025, add DOI 10.52202/085713-3786 and eprint; key retained. |
| `yang2025gated` | verified | Correct (ICLR 2025 camera-ready per arXiv). Its §3 already states the delta rule as test-time SGD on ½‖Sk−v‖² and the gate as adaptive weight decay, which genesis.md presents as a derivation made in this project. | Add eprint 2412.06464; credit the gradient-step identity to this paper and to Titans App. in genesis.md. |
| `ramsauer2021hopfield` | verified | Correct; author list truncated to "and others" although all 16 authors are retrievable. | Full author list, eprint. |
| `krotov2016dense` | verified | Correct (NeurIPS 2016); no URL or identifier. | Add eprint 1606.01164. |
| `schlag2021linear` | verified | Correct (ICML 2021). | Add eprint. |
| `vaswani2017attention` | verified | Correct (NeurIPS 2017); no identifier. | Add eprint 1706.03762. |
| `shazeer2017outrageously` | verified | Correct (ICLR 2017). | Add eprint. |
| `perez2018film` | verified | Correct; missing DOI. | Add volume 32(1) and DOI 10.1609/aaai.v32i1.11671. |
| `qwenputer2026` | corrected | Author given as "{divagr.com}". The work is by Divyansh Agrawal (blog divagr.com/blog/qwenputer, code github.com/divagr18/qwenputer, models on Hugging Face), announced late Oct 2026, built on Qwen3.5-4B. It is a non-peer-reviewed engineering demonstration. The second URL (divagr.com/attention-lab/delta) was not checked. | Author = Agrawal, Divyansh; @software with GitHub URL; mark as non-archival; keep the claim limited to "one-hot keys with α=β=1 make a GDN head an exact RAM row" (stated in the model card). |

### Hippocampal anatomy, physiology and theory

| key | status | issue | fix |
|---|---|---|---|
| `andersen2007hippocampus` | corrected | The five names are editors, not authors (edited volume). Missing DOI. | Use editor field; add DOI 10.1093/acprof:oso/9780195100273.001.0001 (Oxford Academic edition; Crossref dates the record 2006). |
| `bliss1993synaptic` | verified | Correct; missing issue and DOI. | Add number 6407, DOI 10.1038/361031a0. |
| `okeefe1978hippocampus` | unverifiable | Classic monograph with no DOI; Crossref returned no record, so metadata were not machine-verified here (no evidence of error). | None; keep as is. |
| `squire1992memory` | verified | Correct; missing DOI. | Add DOI 10.1037/0033-295x.99.2.195. |
| `scoville1957loss` | verified | Correct; missing DOI. | Add DOI 10.1136/jnnp.20.1.11. |
| `hafting2005microstructure` | verified | Correct; missing issue/DOI. | Add number 7052, DOI 10.1038/nature03721. |
| `eichenbaum2017integration` | verified | Correct; missing DOI. | Add DOI 10.1016/j.neuron.2017.06.036. |
| `spalding2013dynamics` | verified | Correct (14 authors, truncated). | Add DOI 10.1016/j.cell.2013.05.002. |
| `levy1996sequence` | verified | Correct; missing DOI. | Add DOI. |
| `jensen1996hippocampal` | verified | Correct; missing DOI. | Add DOI 10.1101/lm.3.2-3.279. |
| `hasselmo2006role` | verified | Correct; description (ACh suppresses recurrent more than afferent transmission during encoding) matches the review. | Add DOI 10.1016/j.conb.2006.09.002. |
| `lisman2005hippocampal` | verified | Correct; missing DOI. | Add DOI 10.1016/j.neuron.2005.05.002. |
| `bein2020mnemonic` | verified | Correct; "pages = 3451" is the article number (confirmed by Crossref). | Add number 1, DOI 10.1038/s41467-020-17287-1. |
| `hitti2014hippocampal` | verified | Correct; missing issue/DOI. | Add number 7494, DOI 10.1038/nature13028. |
| `smith2016targeted` | verified | Correct; missing DOI. | Add DOI 10.1038/mp.2015.189. |
| `boyle2024tuned` | verified | Correct; page range ends at e9. | Add DOI; pages 1358--1371.e9. |
| `howard2002distributed` | misdescribed | Bibliographic data correct. REFERENCES.md lists it under CA2 as "CA2 temporal context", but the Temporal Context Model is a behavioural model of free recall that makes no claim about CA2; the CA2 link is the repo's own inference. | Add DOI 10.1006/jmps.2001.1388; relabel relevance as "drifting temporal context (TCM); CA2 link is our hypothesis". |
| `kumaran2012generalization` | verified | Correct (REMERGE). | Add DOI 10.1037/a0028681. |
| `koster2018bigloop` | verified | Correct; page range ends at e6. | Add DOI; pages 1342--1354.e6. |
| `mcclelland1995why` | verified | Correct; missing DOI. | Add DOI 10.1037/0033-295x.102.3.419. |
| `wilson1994reactivation` | verified | Correct; missing DOI. | Add DOI 10.1126/science.8036517. |
| `mcewen1999stress` | verified | Correct; missing DOI. | Add DOI 10.1146/annurev.neuro.22.1.105. |

### Hippocampus-inspired machine learning (prior art)

| key | status | issue | fix |
|---|---|---|---|
| `li2026hippoformer` | misdescribed | Bibliographic data correct (ICLR 2026; six authors match the ICLR listing). REFERENCES.md says "EC + Titans-style memory"; the paper builds mm-TEM, a Tolman–Eichenbaum-Machine variant with a meta-MLP relational memory that forms grid-like codes, and couples it to a Transformer for 2D/3D spatial prediction. The meta-MLP memory resembles Titans, but the contribution is structural spatial memory. | Relabel: "TEM-style structural spatial memory (mm-TEM, meta-MLP relational memory) + Transformer". |
| `whittington2020tolman` | verified | Correct; page range ends at e23. | Add DOI 10.1016/j.cell.2020.10.024. |
| `whittington2022relating` | verified | Correct (ICLR 2022). | Add eprint. |
| `kapoor2025hicl` | corrected | Listed as arXiv preprint; published in AAAI 2026 (Proc. AAAI 40(27):22518–22526). Description "DG-gated mixture of experts" is accurate. HiCL also includes grid-cell input, top-k DG sparsity, a CA3 autoassociator, EWC and prioritized replay, so it anticipates most of the repo's DG→CA3 MoE track. | @inproceedings AAAI 2026 with DOI 10.1609/aaai.v40i27.39411. |
| `das2024larimar` | corrected | Listed as arXiv preprint; arXiv record states ICML 2024. | @inproceedings ICML 2024 (PMLR volume not re-checked). |
| `gutierrez2024hipporag` | verified | Correct (NeurIPS 2024). | Add DOI 10.52202/079017-1902, pages 59532--59569. |
| `fountas2025human` | corrected | Venue correct (ICLR 2025). Last author name mis-split as "Ammar, Haitham Bou"; the ICLR listing gives "Haitham Bou Ammar" (arXiv: Bou-Ammar). No arXiv id. | Author "{Bou Ammar}, Haitham"; add eprint 2407.09450. |
| `ahn2025hema` | verified | Correct (single-author arXiv preprint). | Mark as preprint (already). |
| `omidi2025memory` | corrected | Title changed on arXiv: now "…from Neuroscience Principles to Enhanced Model Architectures" (bib has "…to Technical Solutions"). Still an arXiv preprint. | Update title. |

### CA2 candidates: activation steering

| key | status | issue | fix |
|---|---|---|---|
| `turner2023activation` | corrected | Bib/REFERENCES.md use the v1 title. The current arXiv version (v5) is titled "Steering Language Models With Activation Engineering"; the method is still ActAdd. Still arXiv-only. | Use current title; keep a note giving the original title. |

### CA1 candidates: calibrated decision readouts and predictive comparison

| key | status | issue | fix |
|---|---|---|---|
| `typesafe2026jev` | corrected | Weak provenance. jevai.net is a third-party site; the primary source is TypeSafe AI's launch post (Diogo Almeida, founder; 15 Sep 2026). RLCD's reward, architecture and calibration method are unpublished. An independent test reported good calibration on moderation (ECE 0.032) but poor calibration on answer rating (ECE 0.284). The "calibrated confidence" relevance rests on vendor claims. | Cite typesafe.ai launch post as primary; state that the method is undisclosed; do not cite Jev as evidence that calibrated single-pass CA1 decisions work. |
| `nandhakishor2026laya` | corrected | Author handle only. The repository owner is Nandakishor Mukkunnoth (Convai Innovations). README describes ModernBERT-large (421M) and mmBERT-base (322M) checkpoints, trained with RL against strictly proper scoring rules, using the same name RLCD. The Apache-2.0 licence field was not re-checked. | Author = Mukkunnoth, Nandakishor; keep URL; add PyPI note. |
| `huang2026llmjepa` | verified | Correct (ICLR 2026 poster); missing arXiv id; code moved to galilai-group/llm-jepa. | Add eprint 2509.14252. |
| `balestriero2025lejepa` | verified | Correct (arXiv preprint). | None. |
| `gorganmohammadi2026understanding` | verified | Correct (eLife reviewed preprint); missing DOI. | Add DOI 10.7554/eLife.110895; mark as reviewed preprint. |

### Looped / recurrent-depth transformers (big-loop recurrence)

| key | status | issue | fix |
|---|---|---|---|
| `zhu2025scaling` | verified | Correct; still an arXiv preprint (v6, 2026). "Learned exit gate" matches the entropy-regularized learned depth allocation. | None. |
| `dehghani2019universal` | verified | Correct (ICLR 2019). | Add eprint. |

### Continual learning and replay

| key | status | issue | fix |
|---|---|---|---|
| `buzzega2020dark` | verified | Correct (NeurIPS 2020). | Add eprint 2004.07211. |

### Feedback control and timing in the hippocampus

| key | status | issue | fix |
|---|---|---|---|
| `hasselmo2002proposed` | verified | Correct; the design table's use (separate encoding and retrieval phases within theta) matches the paper. | Add DOI 10.1162/089976602317318965. |
| `sik1994inhibitory` | misdescribed | Bibliographic data correct. The anatomy (in vivo-labelled CA1 interneurons with axons back to CA3 and hilus, longitudinally widespread) is accurately reported. Sik et al. propose that this long-range inhibition synchronizes population activity and times spikes for coordinated plasticity. "Negative-feedback gain control" and the op-amp law A/(1+Aβ) are the repo's own analogy, not the paper's claim. | Add DOI 10.1126/science.8085161; relabel relevance as "anatomical basis for CA1→CA3 inhibitory feedback; gain-control reading is ours". |
| `treves1992computational` | verified | Correct. Mossy fibres force storage; the associatively modifiable perforant path relays the retrieval cue. The temporal-loop.md gloss ("a pattern-separated cue maps to a different code") is a paraphrase, not T&R's information-theoretic argument. | Add DOI 10.1002/hipo.450020209. |
| `buzsaki2002theta` | verified | Correct. | Add DOI. |
| `macdonald2011hippocampal` | verified | Correct. | Add DOI 10.1016/j.neuron.2011.07.012. |
| `shankar2012scale` | misdescribed | Bibliographic data correct. The repo (temporal-loop.md, genesis.md, temporal_loop.py docstring) says leaky integrators with log-spaced time constants give scale-invariant time cells. In the paper the integrators compute a Laplace transform; scale-invariant sequential time cells come only after an approximate inverse Laplace operator (TILT/Post). TimeChannel implements only the Laplace layer plus a learned linear readout. | Add DOI 10.1162/neco_a_00212; reword as "Laplace-domain (leaky integrator) layer of Shankar & Howard; no inverse transform". |
| `sapolsky1986neuroendocrinology` | verified | Correct; missing DOI. | Add DOI 10.1210/edrv-7-3-284. |

### Dynamic steering and oscillatory sequence models

| key | status | issue | fix |
|---|---|---|---|
| `rimsky2024steering` | verified | Correct (ACL 2024). | Add DOI 10.18653/v1/2024.acl-long.828, pages 15504--15522. |
| `lee2024programming` | corrected | Listed as arXiv preprint; arXiv record states ICLR 2025 (spotlight). | @inproceedings ICLR 2025, year 2025 (key retained). |
| `do2025dynamic` | verified | Correct; "memory-retrieved steering vectors" matches DSEM (key–value memory from generated tokens to steering vectors). | Add DOI 10.18653/v1/2025.findings-acl.706. |
| `sun2023retentive` | verified | Correct (arXiv preprint). | None. |
| `orvieto2023resurrecting` | verified | Correct (ICML 2023). | Add eprint. |
| `qwen2025qwen3` | verified | Correct (arXiv 2505.09388, first author An Yang, 60 authors; corporate author acceptable). | None required. |
| `qwen2026qwen35` | corrected | Title "Qwen3.5" plus URL of the 397B flagship. The official citation in the Qwen model cards is "Qwen3.5: Towards Native Multimodal Agents" (Qwen Team, Feb 2026, qwen.ai/blog?id=qwen3.5). The model actually used, Qwen3.5-2B, came out on 2 Mar 2026; it has 24 layers, 6×(3×GDN→FFN→1×Gated Attention→FFN). | Use the official title and URL; add a note naming the Qwen3.5-2B checkpoint. |
| `leroy2018circuit` | verified | Correct; URL points to PubMed rather than DOI. | Add number 7735, DOI 10.1038/s41586-018-0772-0. |
| `geva2021transformer` | verified | Correct; booktitle abbreviated. | Full booktitle, pages 5484--5495, DOI 10.18653/v1/2021.emnlp-main.446. |
| `meng2022locating` | verified | Correct (NeurIPS 2022). | Add DOI 10.52202/068431-1262. |
| `arditi2024refusal` | verified | Correct (NeurIPS 2024). | Add DOI 10.52202/079017-4322. |
| `zou2023representation` | verified | Correct (arXiv preprint; 21 authors, truncated). | None. |
| `zou2024improving` | verified | Correct (NeurIPS 2024). | Add DOI 10.52202/079017-2651. |
| `mazeika2024harmbench` | verified | arXiv record verified (12 authors). The ICML 2024 venue was not re-checked here. | None. |
| `rottger2024xstest` | verified | Correct (NAACL 2024). | Full booktitle, DOI 10.18653/v1/2024.naacl-long.301, pages 5377--5400. |
| `zhao2025llms` | misdescribed | Bibliographic data correct (NeurIPS 2025). The REFERENCES.md and genesis.md wording ("harmfulness survives jailbreaks and harmful fine-tuning") over-generalises. The paper shows certain jailbreak methods reduce refusal without reversing the harmfulness belief, and adversarial fine-tuning has minimal impact on it. It did not test abliterated models. | Add DOI 10.52202/085713-4688; reword as "certain jailbreaks". |
| `chen2025seal` | corrected | Author "Chen and others"; venue omitted (COLM 2025). | Full authors; @inproceedings COLM 2025. |
| `azizi2025activation` | corrected | Author "Azizi and others"; published in Findings of ACL 2026, where Souvik Kundu was added as a fourth author (arXiv v2 lists three). | Full authors; @inproceedings Findings ACL 2026 with DOI 10.18653/v1/2026.findings-acl.1828; key retained. |
| `huang2025mitigating` | corrected | Author "Huang and others"; NeurIPS 2025 only in howpublished. | Full authors; @inproceedings with DOI 10.52202/085713-3430. |
| `lin2025controlling` | corrected | Author "Lin and others". | Full authors; @inproceedings NeurIPS 2025 (spotlight), DOI 10.52202/085713-2623. |
| `ma2025cotvalve` | corrected | Author "Ma and others". | Full authors; ACL 2025 long, DOI 10.18653/v1/2025.acl-long.300. |
| `sun2025thinkedit` | corrected | Author "Sun and others". | Full authors; EMNLP 2025, DOI 10.18653/v1/2025.emnlp-main.861. |
| `venhoff2025understanding` | corrected | Author "Venhoff and others"; venue omitted (ICLR 2025 Workshop on Reasoning and Planning for LLMs). | Full authors; note workshop. |
| `hao2024training` | corrected | Author "Hao and others"; published at COLM 2025. | Full authors; @inproceedings COLM 2025, year 2025. |
| `nguyen2025activation` | corrected | Author "Nguyen and others"; ICLR 2026 poster held in howpublished with year 2025. Description "steering as PID control" is fair. Its integral term accumulates error across layers within one forward pass, not across tokens or turns as in the repo's escalating controller. | Full authors; @inproceedings ICLR 2026, year 2026 (key retained). |
| `shi2026internalizing` | corrected | Author "Shi and others". The work appeared at EMNLP 2026 under a new title, "STIR: Self-Distilled Latent Tools for Dynamic Control of LLM Reasoning" (pp. 51814–51829); its Crossref DOI 10.18653/2026.emnlp-main.2278 lacks the usual "v1/" segment and resolves as such. "Memory-retrieved reasoning steering" fits: a library of steering primitives with anchor-based gated injection. | Full authors; cite EMNLP 2026 version, keep arXiv as eprint. |

### Verified additions recommended

| key | why |
|---|---|
| `yang2024parallelizing` | DeltaNet as online regression; source of the gradient-step reading (NeurIPS 2024, 10.52202/079017-3668) |
| `liu2024longhorn` | Online-learning framework that Gated DeltaNet's §3 builds on (arXiv 2407.14207, preprint) |
| `behrouz2025miras` | Unifies Titans, GDN and attention as associative memories with an internal objective (arXiv 2504.13173, preprint) |
| `zhao2025adasteer` | Input-adaptive safety steering from harmfulness and refusal directions; baseline for A2–A4 (EMNLP 2025, 10.18653/v1/2025.emnlp-main.1248) |
| `samsonovich2005simple` | CA3 storing EC-state transitions with pathfinding retrieval; closest classical model of the successor store (Learn. Mem. 12:193–208, 10.1101/lm.85205) |
| `kabir2024deep` | Laplace plus inverse-Laplace scale-invariant memory in a deep network; baseline for the time channel (arXiv 2412.15292, preprint) |
