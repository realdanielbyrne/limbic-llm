# Genesis: how limbic-llm came about

This is a record of the working session between **Daniel Byrne** and **Claude** (Claude Code) on 1 November 2026 that produced this repository. It keeps the order in which the ideas came, who proposed each one, and the corrections made along the way, so the reasoning behind the code is not lost.

**Full session:** <https://claude.ai/code/session_0156jLUgov1W1j9W1RUt6QqH>. It is private to the repository owner's Claude account. Every commit made in the session carries the same link in its trailer.

Daniel's prompts are quoted verbatim. "Daniel" marks ideas he proposed and "Claude" marks ideas Claude proposed. Equations use the notation of [`design/temporal-loop.md`](design/temporal-loop.md).

---

## 1. From parallels to a design document

> "Create a document describing the parallels between machine learning models and the hippocampus as we've discussed here. Include both our initial discussion about parallels between structures and also the discussion about how Titans and the hippocampus compares. I want potential implementation details as well for both potential modeling strategies"

The session began from an earlier discussion that mapped hippocampal regions onto neural-network components. The first artifact was the design document [`hippocampal_ml_architectures.docx`](hippocampal_ml_architectures.docx), version 1. It mapped entorhinal cortex to an encoder, dentate gyrus to k-WTA pattern separation (an expert gate), CA3 to associative memory, CA2 to a context side channel, CA1 to a comparator, and subiculum to output gain. It described two strategies: a direct circuit (Strategy I) and a Titans hybrid (Strategy II).

## 2. Implementation

> "Help me implement this idea into code"
> "save this into my github repo"

Claude built the package (first named *hippoformer*): `HippocampalBlock`, the Titans hybrid, and replay consolidation, together with recall and continual-learning experiments. The first benchmark had every memory model at chance, and three design rules came out of debugging (README, "Lessons from the experiments"):

- CA3 must bind each pattern to its successor.
- The Titans meta-gradient is a partial derivative.
- Writes need a trust region.

## 3. Gated DeltaNet as CA3

> "https://divagr.com/blog/qwenputer and https://divagr.com/attention-lab/delta describe a way to use architectures that incorporate Gated DeltaNet linear transformers as RAM. Could this be analogous to the LMM and/or the CA3. The decay gate is similar to the weight decay/ associative memory loss function. That is they serve similar functions."

**Daniel** proposed the analogy. **Claude** made it exact: the Gated DeltaNet update is one gradient step on the Titans memory loss, taken at the decayed state. *(Later found to be published: the identity is stated in the Gated DeltaNet paper itself, §3, and in DeltaNet and Longhorn; see §9.)*

$$S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$$

So the decay gate $\alpha_t$ is the forgetting term, and one-hot keys turn the state into exact RAM, the limit that dentate-gyrus pattern separation approaches. Daniel asked for the work to go on a branch, after cleaning up the folders and adding a `.gitignore`. He then asked:

> "Explain if you implemented the soft addressing rule or one-hot? Seems the softer approach, slightly overlapping might be better to try first."

This led to the `gdn`, `tied` and `dg` addressing modes. Swapping the deep Titans memory for Gated DeltaNet took one-shot recall from 0.48 to 1.00, and successor-bound addresses made the state exact RAM up to 32 pairs.

## 4. Diagrams, related models, a new name

> "what is hippoformer implementing? A full llm? Can you create a mermaid diagram?" · "this diagram should be an HTML artifact saved in the repo"

> "The inspiration covered several components of a traditional LLM 1. EC -> Embedding Layer 2. DG -> MoE Gate 3. CA3 -> Memory Constructs 4. CA2 -> Not implemented yet 5. CA1 -> combined with CA3, but possible connections with decision models like Jev What associated models have been attempted?"

> "...Comment on using Laya or Jev comparing output vs input in the final layer as a model of CA1. Also for CA2 what about Activation Additions, ... which uses a side channel amplification by hooking into an internal layer and boosts activations. This is an analog of CA2. Likewise Loop style transformers ... can be seen as a CA2 style construct, looping to strengthen, amplify, latent reasoning."

**Daniel** proposed three mappings:

- Activation Addition as CA2.
- Looped transformers (Ouro) as CA2-style amplification.
- Calibrated decision models ([JEV](https://jevai.net), [LAYA](https://github.com/NandhaKishorM/laya)) as CA1.

**Claude** agreed that Activation Addition is the closest existing CA2 mechanism. Claude argued that Ouro fits better as *big-loop recurrence* (EC → … → EC), with CA1 novelty as its exit gate.

An ICLR 2026 paper already uses the name "Hippoformer". Daniel rejected *mnemoformer* as hard to pronounce and chose **limbic-llm**. The PDFs were removed in favour of a bibliography ("What matters is the references for the bibliography"), the design document went to version 2, a README was written, and PR #1 was merged.

## 5. The temporal loop

This part of the session produced the main new idea, which emerged over four exchanges.

### 5.1 Dynamic activation addition (Daniel)

> "Activation addition can be made dynamic if we tap into an inner layer to capture a time lagged sliding window sample that we then use as a GDN, CA3 memory to return related context and negated context, take the difference and then re-inject. windows with strong differences between positive and negative contexts could produce stronger amplified side channel signals. CA3 analogies also need some sort of recurrent or autoregressive structure which maybe CA2, or our additive activations can use to feed temporal knowledge into the token stream. ..."

**Claude** pointed out that one mechanism then covers three regions: the memory stores (CA3), the contrast measures novelty (CA1), and the re-injection is the side channel (CA2). Two problems followed:

- A linear memory cannot supply the negative by negating the query, since $S(-q) = -Sq$. Claude proposed using the model's prior as the negative: $\Delta_t = F_t - m_t\mu_t$, this episode against what the model would expect anyway.
- Feeding the injection back creates positive feedback that can run away.

### 5.2 How biology keeps the loop stable (Daniel's question, Claude's survey)

> "How is the feedback loop mitigated in the biological system? ... Also have we been neglecting the skip connections from DG to CA1? Also CA1 feeds back into CA3 producing a negative feedback that could mitigate over saturation like the feedback sink in an opamp circuit."

**Claude** listed the safeguards: feedback inhibition, the acetylcholine encoding/retrieval switch, theta-phase separation, attenuated re-entry, self-terminating bursts, and sparse dentate input. Claude also corrected the anatomy:

- There is no projection from the dentate gyrus to CA1. The skip into CA1 comes from entorhinal layer III.
- The neglected path was the direct one from entorhinal layer II to CA3 (Treves & Rolls), which supplies the retrieval cue.

**Daniel's** op-amp analogy holds through CA1's GABAergic back-projection interneurons (Sik et al., 1994):

$$G = \frac{A}{1 + A\beta} \approx \frac{1}{\beta}$$

### 5.3 Clamped, damped sinusoids and a sense of time (Daniel)

> "temporal feedback should be clamped sinusoids that decay with every circuit around the loop. New tokens should dominate and feedback tokens should be attenuated. ... There is also a temporal relation to this feedback loop that again injects a sense of time passing into the token stream, something LLMs do not currently do. ... Would these pieces plug into different pieces of existing architectures or is this turning into a novel architecture? the long term memory could, white matter, could be the multi layer depth of a traditional model. Come up with an experiment we can test and run it."

**Daniel's** principle became the re-entry kernel:

$$w(\Delta) = A\,\rho^{\Delta/\tau - 1}\,[\cos(2\pi\Delta/\tau)]_+, \qquad \Delta \ge \tau,\ A \le 1,\ \rho < 1$$

**Daniel's** "sense of time passing" became the time channel, a bank of leaky integrators and damped oscillators at log-spaced timescales after Shankar & Howard's scale-invariant time cells.

**Claude** made one correction: long-term memory lives in the *layer weights* (cortical synapses). The *residual stream* is the white matter, the bus that every layer reads and writes, and it is where the side channels attach.

**Claude's** answer to the architecture question: every piece has a precedent and plugs into existing models. The closed, regulated, timed loop is a new organisation.

The experiment is the *what-when* task: what happened and how long ago, beyond a 16-token attention window. It is paired with closed-loop stability runs.

### 5.4 What the experiments did and did not test (Daniel), and the inverted copy

> "So did these experiments test the EC → CA3 direct path skip connections and the CA1 → CA3 inhibitory feedback gain? What about the Memory Steering or Additive Activations? If CA1 → CA3 is naturally an inverted feedback signal then isn't that the analog of negative query used in the additive activations notebook? Do these experiments already test all of this concepts or do we need additional experiments ?"

**Claude** answered that they were only partly tested. The CA1 switch bundled a hard clamp that bounds the injection *by construction*, so the stability result did not show what the feedback itself does.

**Daniel's** question led to a distinction:

- Activation Addition's negative is an independent reference, which corresponds to the prior $\mu_t$.
- An inverted CA1 copy subtracts the loop's *own* recent output, a self-negative.

Claude derived the consequences:

- Energy-*divisive* feedback only compresses.
- *Subtractive* feedback obeys the op-amp law, $y = A\Delta/(1+A\beta)$.
- The subtractive form is a high-pass filter: sustained echo is cut, new content passes.
- Because the copy arrives a circuit late, the loop rings above $K = A g\beta \approx 1.68$, the reason op-amps need compensation.

Daniel then said "fan out as necessary", and the switches and experiments were built in parallel.

### 5.5 Pretrained models (Daniel)

> "I have a local machine with two 5090s which we can host a model over API. Also Should we consider Qwen3.5Next which uses the GDN circuit already?"
> "So is GPT-2 or Qwen3.5 the better experiment or should both be a target or neither"

**Claude** explained that an inference API cannot expose hooks or gradients, and proposed a matched pair:

- Qwen3, pure attention: does the loop carry memory when nothing else can?
- Qwen3.5, a 3:1 Gated DeltaNet hybrid: is GDN the CA3, and does the loop add anything on top?

A second Claude session on Daniel's GPU machine (geof) ran these and pushed its results with a handoff note, [`research/HANDOFF.md`](../research/HANDOFF.md). It then began an evolution-style search over the side channels, with fitness criteria agreed with Daniel (reach on the Qwen3 needle, bounded echo, new tokens dominating) and anti-gaming rules such as strict causality and a fixed-size state. Its notebook is [`research/NOTEBOOK.md`](../research/NOTEBOOK.md), and its round-0 diagnosis found two of the corrections below.

### 5.6 Design review, stability and alignment (Daniel)

The next morning Daniel asked how the Qwen implementation relates to the circuit: whether DG reuses a mixture-of-experts router, how skip paths and feedback are wired, whether the CA2 injection is a single layer, and what plays the neocortex. **Claude's** answers, taken from the code and run settings, are in the [lab notebook's design review](../research/NOTEBOOK.md#design-review-how-the-qwen-side-channels-map-onto-the-circuit-2-nov-2026). In short:

- Both Qwen models are dense, and DG keys were off in every Qwen run.
- The skips and the feedback live inside one hook at one block, with feedback running over time rather than depth.
- The injection is single-layer and switchable.
- The frozen stack is the neocortex, and consolidation is not yet wired in.

Two of Daniel's premises were adjusted: CA3 → CA1 is the main serial path rather than a skip, and the Titans LMM is hippocampal rather than neocortical. Daniel then set a direction:

> "All results and conclusions should be tracked and potential future experiments noted. Feedback stability mechanisms need to be explored in more detail as our additive activations could likely cause saturation without mediation. Additive activations also seem to be analogous to llm alignment. There is the biological similarity in that societal norms seems to drive CA2 and in the world of deep learning that maps directly onto alignment efforts. CA2 could be a way to prevent models from producing harmful content if we can find a way to dynamically steer models away from harmful outputs. For instance I can imagine a scenario similar to the books of Isaac Asimov's robot books. In Asimov's books when a robot (AI) is asked to harm a human an internal feedback loop ever more aggressively blocks this behavior until the model's outputs become nonsensical. Dynamic Additive Activations can be used to push a model towards non-harmful outputs the more the "novelty" signal increases, and in this case we could model novelty as a proxy for a harmfullness score"

This became three records:

- the register of results and experiments, [`research/REGISTER.md`](../research/REGISTER.md);
- a feedback-stability programme, S1–S8, led by a saturation curve on Qwen;
- a proposal for CA2 as a norm-steering controller, A1–A5.

**Claude** added two points of context:
- Biology: CA2 supports social recognition memory and in mice also *promotes* aggression through the lateral septum, so it acts as a context selector rather than an enforcer of norms.
- Prior art: circuit breakers already reroute harmful representations towards incoherent or refusal states, but by fine-tuning. What is new here is a runtime, graded controller *(§9: input-adaptive coefficients are AdaSteer's and closed-loop control is PID steering's; only the leaky temporal integral is distinct)*: a leaky integral of a harm score drives a capped steering gain. Plain novelty is not the same as harm, so the score should come from a harm probe, or from "norm novelty", a mismatch against a memory of aligned behaviour.

### 5.7 Re-aligning an unaligned model (Daniel)

> "Yes on an open model these protections can be dropped, but on closed models that would be impossible. This might be a whole separate area of research that could be used in conjunction with the other hippocampus model constructs being explored in this research or as a standalone alignment research. One potential experiment could be to start with a purposefully unaligned model (ablated models on huggingface) and then add in the hooks to see if an unaligned model can be made to act like an aligned model."

**Claude** found suitable models on the Hub: abliterated Qwen3-1.7B variants next to the aligned and base models already in use. It also found supporting evidence: LLMs encode harmfulness separately from refusal, and jailbreaks or harmful fine-tuning leave the harmfulness judgement intact (Zhao et al. 2025). So the harmfulness representation can drive the controller while the hook supplies the refusal direction. Restoring refusal by adding the direction back is close to a tautology for an abliterated model. The informative tests are selectivity (over-refusal), capability cost, multi-turn persistence, and the never-aligned base model. Recorded as A6–A7 in the register, with a data rule: raw completions to harmful prompts are never committed.

### 5.8 CA2 beyond alignment: chain-of-thought and a windowed view (Daniel)

> "build a base alignment track which targets results needed in such research. The CA2 piece might also be helpful in non-alignment situations as well however. Consider CA2 side channel correlation's to COT reasoning. Chain of thought is essentially used to steer a model towards a correct answer by providing extra model generated context. Research has shown significant improvements in model responses when COT is activated. Additive Activations might be used as a shortcut, almost like momentum in training, to get a model to converge on a better answer more quickly my selectively boosting thoughts on the current thought process. So our CA2 model could be used to reduce COT token count while achieving the similar results. If true that could cut inference costs and improve speed."

> "I believe our CA2 model should work on a windowed view (exponentially averaged, autoregressive, etc) so that a general idea is boosted. Is that how it is modeled currently?"

**Claude** built the alignment track (`experiments/alignment/`). On the second question it answered "partly": the loop's injection averages *older* content, excluding the current circuit, and subtracts the long-run mean, while the alignment controller steers along a fixed direction. Claude added a *gist* direction, a fast-minus-slow running average of the stream before injection (momentum, open loop, capped), and flagged it as positive feedback for the stability programme. A literature survey found fixed reasoning-steering vectors already cutting chain-of-thought tokens by 10–70%, plus feedback-controlled and memory-retrieved steering. It found nothing that steers with the model's own running average of its current thought under negative feedback. Recorded as R1–R5, with over-commitment, the loss of useful backtracking, as the main risk.

### 5.9 First steering runs: regulate the CoT gain, strengthen the alignment gain (Daniel)

Daniel asked for a Qwen version of the GPT-2-XL ActAdd notebook that could test both the alignment and the CoT additions from one config switch. Claude built it (`docs/notebooks/activation_additions_qwen3.ipynb`) and spot-checked it on the abliterated Qwen3-1.7B. The results, N1–N5 in the register, were:
- Prompt-position ActAdd does nothing to a chat model.
- The abliterated model's own refusal direction does nothing.
- The aligned sibling's direction makes half the harmful samples refuse at gain 0.3, while benign prompts are still answered.
- The thinking direction collapses into repetition at that same gain.

> "The Cot experiment points to a need for a feedback regularization to prevent repeats. The alignment tests point to a need for possibly stronger gain, maybe multipliciative or exponential (squared) instead of additive?"

**Claude** agreed on both and narrowed each.

For CoT, the collapse is a cliff between gain 0.2 and 0.3. A repetition signal, integrated like the harm score and fed back negatively onto the gain, is the regulator that is missing. It joins homeostatic gain (S5) and the regulated gist (R3), and is recorded as R6.

For alignment, only the sibling-direction case is limited by gain; the other two failures are of position and direction. The injection is already multiplicative in the stream norm, so the superlinearity belongs in the gain *schedule*, as a function of harm evidence (squared or exponential in $E_t$). Gain on the existing refusal component would amplify nothing, because abliteration leaves $h\cdot\hat d \approx 0$. A superlinear schedule may help selectivity as well as strength, and the refusal direction's own saturation knee has to be measured before stronger gain is trusted. Recorded as A8 and S9.

## 6. Findings so far

Small models (about 0.1M parameters, CPU), mostly single seeds; final numbers are in [RESULTS.md](../RESULTS.md).

| Question | Result |
|---|---|
| Does each channel carry its answer? | The memory loop gives *what* 1.00 and *when* 0.85. The clock alone gives *when* 0.88 with *what* at chance. Windowed attention alone is at chance on *what* (0.11) and at the prior on *when* (0.34). |
| Does the clamped, damped loop with CA1 control stay bounded? | Yes. The echo is 0.29–0.35 of the input across 520 tokens on two seeds (peak 0.53). Without control it reaches about 3× (clamped sinusoid), 1.0× early with a peak of 10.6× (damped), and 4× rising to 14× with a peak of 63× (undamped). |
| Does the full design still learn? | Yes, given 3000 steps. *What* 1.00 on two seeds, *when* 0.98–1.00. "What" is learned in a sudden late jump whose timing varies. |
| Do new tokens dominate in a closed loop? | Not by damping alone. Probed with an older and a newer event, the undamped loop names the older event 61% of the time and the damped loop 84%. A closed loop re-stores its own echo every circuit, so per-trace damping does not damp rehearsed content (primacy by self-rehearsal). The full design with the contrast drops to 18%, sparse dentate-gyrus write keys flip it (68% newer), and an open loop that does not re-store its echo shows little primacy. |
| Is the CA1 inverted copy an op-amp? | Yes, for persistent echo. On an untrained loop, subtractive/none = 0.787, 0.647, 0.475, 0.373 against $1/(1+K)$ = 0.788, 0.650, 0.482, 0.383. It rings above $K = 1.68$, as predicted. |
| Pretrained models | On Qwen3.5-2B with only attention windowed, the side channels restore exact needle recall at about 500 tokens (1.00 vs 0.00), reproduced after the bf16 fix. Moving the tap from block 12 to block 22 drops it to 0.00, so the success depends on the Gated DeltaNet layers after the tap. The loop failed on Qwen3, but its kernel ($\rho = 0.7$, $\tau = 16$) gives a weight of about $10^{-19}$ at a 500-token lag: the needle was out of the loop's reach. With a horizon that covers the gap, the loop alone carries needle information 500 tokens back in both models (digit loss 1.84 → 1.38 and 1.24, below the format floor), though not exact recall in those runs. |
| Round 1 on the pretrained models | Exact recall with every layer windowed. On Qwen3, W1, a Gated DeltaNet side memory whose keys bind each value to the previous position's address (the successor rule of §§2–3), recalls all four digits exactly (1.00), with a fixed-size state and fewer parameters than the loop. The horizon loop on Qwen3 had recovered only the first digit. On Qwen3.5 the horizon loop also reaches 1.00 given 6,000 steps. Both results hold for one stored fact: with two, W1 retrieves the digit chain but picks which fact to start from by a coin flip (exact 0.43–0.55). Register P7–P12. |

## 7. Corrections made along the way

| What was assumed | What turned out to be true |
|---|---|
| The *when* baseline is 0.25 | The bins are unequal, so the prior alone scores 0.333 |
| Memory + time failed *what* because the clock removed a curriculum | It learned *what* by step 500 with a longer schedule; the timing of the jump varies |
| A trained recency task would test "new tokens dominate" | No model learned it in 3000 steps; a zero-shot probe replaced it |
| Damping each trace makes new input win | In a closed loop, rehearsal re-stores old content and it wins |
| CA1 control proves the feedback is stabilising | The hard clamp guarantees the bound by construction; the feedback had to be tested separately |
| A sliding window makes memory the only route | Information hops $W-1$ tokens per layer, reaching $L(W-1)$; block windows are strict |
| The first GPU runs trained the side channels | Side channels in bf16 lost Adam updates near 1; they now train in float32 |
| The loop's kernels were causal in any precision | In bf16, `arange(512)` has only 385 distinct values, so positions merged and the clock's $d \ge 0$ mask let a position see the next token. Found by the GPU session; index arithmetic now stays in float32 |
| The loop has an $O(1)$-state recurrent form | Only if relevance factorises as $\varphi(q)^\top\varphi(k)$. The exponential relevance used here does not, so the store grows as $O(T)$; a fixed-state version leads back to linear attention and Gated DeltaNet |
| The pretrained needle runs tested the loop's memory | With $\rho = 0.7$ and $\tau = 16$, the kernel's e-folding span is $\tau/\ln(1/\rho) \approx 45$ tokens, and its weight is exactly 0 for lags of 501–503 (the clamped half of the cosine). The horizon has to be set for the gap. |
| The GDN = Titans-step identity was derived in this session | It is stated in the Gated DeltaNet paper (§3), DeltaNet and Longhorn (§9) |
| The CPU recall task measures key–value binding | A binding-free strategy scores 0.476 at 6 pairs; full attention and Titans sit at it. Only successor-tied memories bind (§9) |
| Standard GDN's drop with more pairs is a capacity limit | It follows the binding-free curve at loads far below d_k; standard GDN never learned binding (§9) |
| CA1 pyramidal cells do not project back to CA3; CA2 lacks mossy-fibre input and sits between CA3 and CA1 | Excitatory vCA1 → dCA3 projections exist (Lin et al. 2021); DG → CA2 input and a parallel CA2 → CA1 route exist (Kohara et al. 2014) (§9) |
| Subtractive feedback is a high-pass filter that cuts echo and passes new content | By Bode's sensitivity integral it must amplify some band: at K = 1, about 2.5× near a 46-token period (§9) |

## 8. Open questions

The current list, with status and priority, is kept in [`research/REGISTER.md`](../research/REGISTER.md). The main threads:

- **How does Qwen3.5 recall the needle?** The late tap shows the success needs Gated DeltaNet layers after the tap. The loop cannot see 500 tokens back with its default kernel, so the needle must reach the tap through the GDN layers before it. The working hypothesis is a chain: GDN layers before the tap carry a weak trace, the loop re-presents it, and GDN layers after the tap integrate it for the attention layers to read.
- **How long a memory should the loop have?** Damping makes new input dominate but sets a horizon of about $\tau/\ln(1/\rho)$ tokens. Several bands with different $\rho$, as in the blueprint, may be needed to have both.
- Does subtractive feedback break closed-loop primacy, as the high-pass argument predicts?
- Does the direct entorhinal → CA3 read beat a dentate-gyrus read when the cue is noisy (Treves & Rolls)?
- What is the compensation for the delayed CA1 feedback: theta timing, slow GABA$_\text{B}$ inhibition, or the clamp?
- Consolidation: replaying high-novelty episodes into adapters on the frozen model.

## 9. Literature review (Daniel, 2 Nov 2026)

A separate session (Claude, in Claude Science) reviewed the project against the literature.

> "https://github.com/realdanielbyrne/limbic-llm describes a research path on modeling various aspects of the human hippocampus into deep learning constructs.  The repository lays out research tracks for CA3-CA1 style skip connections, CA2 style side channel injections for CoT reasoning and alignment, DG-CA3 style MoE, and using GDN modules in Qwen models as a form of soft Rnn style autoregressive memory. In addition I've added a paper I found on a previous modeling attempt to this chat.  Review the bibliography, the genesis document, the proposed experiments and existing code. Help me flesh out the research with a review of existing literature and how it relates to this or how my research extends existing research.  Fan out as necessary."

**Daniel** supplied Samsonovich & Ascoli (2005), a hippocampal model of episodic retrieval as path search. **Claude** split the work into seven parallel tracks: one per research track, one for the paper, a bibliography audit, and a code review. Every citation was verified against Crossref, arXiv or OpenAlex. The output is in [`research/reviews/2026-11-02_literature/`](../research/reviews/2026-11-02_literature/README.md).

The verdict was blunt. Most components are published under other names:

- the GDN–Titans identity (in the GDN paper itself);
- exact RAM from orthogonal keys (Kohonen 1972);
- successor binding (Sompolinsky & Kanter 1986; induction heads);
- the DG-gated MoE (HiCL);
- the CA1 fusion vector (InferSent);
- the closed loop (Feedback Transformer);
- the time channel (Shankar & Howard's Laplace stage).

What looks new is the control-theoretic treatment of re-injection loops, gist momentum steering, the temporal integral in the harm controller, the P2 observation on Qwen3.5, and an opening the paper points to. That opening is goal-directed path search through a successor memory beyond the attention window, which neither this repo nor the LLM-memory literature implements.

The code review found that the CPU recall task can be solved to 0.476 without binding, which restates M1 and M4. Claude reproduced the ceiling independently.

The review landed after the GPU session's round 1. P11 (W1 retrieves the digit chain but cannot choose which stored pair to start from) turned out to be the paper's missing-key problem, so the paper's goal field became backlog item L6.

> "Make a commit to the repo with your findings.  Update the lab notebook, bib, genesis, and other related documents.  Some experiments are ongoing.  Add a handoff comment for other agents so that they can consume your research."

The commit went on a new branch from the tip of `feat/round1-side-memory`, leaving the GPU session's `research/HANDOFF.md` untouched. It:

- replaced `docs/references.bib` with the audited version;
- added 72 prior-art entries to REFERENCES.md;
- updated the register (B12–B15, the RV2 findings table, backlog L1–L16, status notes) and the notebook;
- corrected four docstrings;
- added a handoff note for agents, [`research/reviews/2026-11-02_literature/HANDOFF.md`](../research/reviews/2026-11-02_literature/HANDOFF.md).

## Where to find things

| | |
|---|---|
| [`design/temporal-loop.md`](design/temporal-loop.md) | The temporal-loop design, its derivations and the full Limbic LLM blueprint |
| [`hippocampal_ml_architectures.docx`](hippocampal_ml_architectures.docx) | The design document (version 3) |
| [`../RESULTS.md`](../RESULTS.md) | All experimental results |
| [`../limbic_llm/temporal_loop.py`](../limbic_llm/temporal_loop.py) | `HippocampalLoop`, `TimeChannel` |
| [`../experiments/`](../experiments/) | `temporal_loop.py` (CLI), `loop_systems.py` (op-amp test), `pretrained_loop.py` (frozen HF models) |
| [`../research/REGISTER.md`](../research/REGISTER.md) | Every result, correction and planned experiment, with status and evidence |
| [`../research/NOTEBOOK.md`](../research/NOTEBOOK.md) | The dated lab notebook: rounds, design review, proposals |
| [`../research/HANDOFF.md`](../research/HANDOFF.md) | Notes from the GPU session |
| [`../research/plans/`](../research/plans/), [`../research/reviews/`](../research/reviews/) | Round plans (round 0) and dated reviews (RV1) |
| [`../results/`](../results/) | Raw JSON and logs: `synthetic/` (CPU), `pretrained/` (GPU), `evolve/round1/`, `alignment/` |
| [`../REFERENCES.md`](../REFERENCES.md) | Bibliography |
| [`../research/reviews/2026-11-02_literature/`](../research/reviews/2026-11-02_literature/README.md) | Literature review, track reports, handoff for agents |
