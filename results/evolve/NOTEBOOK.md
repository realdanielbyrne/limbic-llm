# Evolution lab notebook

This is an AlphaEvolve-style search over the side channels: propose changes, score them on the GPUs, keep the best.
A human checkpoint follows every round. Each entry records what was tried, the scores, the winning diff, any surprises, and the proposed next step.
Round 0 is the current code (`limbic_llm/temporal_loop.py`, `experiments/pretrained_loop.py`) as it stood at commit `b39abab`.
The project's history is in [`docs/genesis.md`](../../docs/genesis.md).
Every result, conclusion and planned experiment is tracked in [`REGISTER.md`](REGISTER.md), which is kept current; this notebook is the dated record.

## Fitness (agreed 1 Nov 2026)

1. **Reach.** Recall the needle on Qwen3-1.7B-Base with every layer windowed (gap about 500 tokens). Score digit cross-entropy against the answer-format floor, 0.8·ln10 = 1.842. Exact match is a secondary score.
2. **Stability (must pass).** The closed-loop echo stays bounded on `experiments/temporal_loop.py --suite stability`. The reference is genesis §6: 0.27–0.35 of input, peak 0.53.
3. **New tokens dominate (must pass).** Zero-shot recency probe. The full design names the older event 18% of the time.

Candidates must also satisfy the anti-gaming rules E1–E9 in [`round0_plan.md`](round0_plan.md) §3: strict causality, bounded state, a parameter budget, input isolation, and held-out gaps.

---

## Round 0: diagnosis (1 Nov 2026)

**Method.** A workflow of 21 agents examined the problem from four angles: gradient flow, memory capacity, Qwen3 against Qwen3.5, and task mechanics. Each hypothesis was checked by an adversarial verifier, using code reading and CPU checks only. The full output is in [`round0_plan.md`](round0_plan.md).

**Findings**
- **D1 (fact).** Qwen3 windowed+side scores exactly the answer-format floor. Eval loss is 1.8421 against 0.8·ln10 = 1.8421, and digit accuracy is at chance. The side channels learned the leading space and the digit distribution, and nothing about the value.
- **D2 (main suspect).** The default `osc` kernel with ρ 0.7 and τ 16 gives the needle a weight of about 2e-5 against a row mass of 14.7. On some digit diagonals the weight is exactly 0. The needle's key gradient is about 5e-6.
- **D3 (open).** The contrast prior is an unweighted mean, so it sends a needle share of about 3–4.5% whatever the gap, yet nothing arrives. Something downstream may also block the value: tap-14 states may not encode it, or the frozen windowed upper layers may ignore a small component.
- **D4.** Qwen3.5's attention-only exact 1.0 is carried by the intact Gated DeltaNet layers, not by the loop's retrieval. The tap-22 result points to the layers above the tap.
- **D5.** The pre-fix all-windowed loss drift (1.69) was an **anti-causal leak**, not real transfer. `arange(512)` in bf16 has only 385 distinct values, so the `d>=0` mask let some predictions see their own targets. The fp32 fix removes it.

**Refuted:** read sharpness, contrast, or the CA1 gate as the bottleneck; "the loop structurally cannot reach the needle" (a CPU check reached 97% with the value spread over block 0).

**Round-0 diagnostic runs** (existing flags only, 1000 steps each, about 3.5 minutes per run). Pass means loss ≤ 1.82 and token_acc ≥ 0.315.
- The horizon runs Daniel requested (`--kernel exp --rho 0.95 --tau 32`, weight 0.47 at 500) answer D2 against D3 and stand in for arm g2.
- The remaining arms, which depend on that answer: g1 `flat`, g3 `flat --no-time`, g4 `flat --no-contrast`, g5 `flat --pairs 2`, g6/g7 `flat` at tap 6 and 22.

**Decision needed at this checkpoint.** Rule E2 (state size fixed regardless of sequence length) rules out the current `HippocampalLoop` itself, which keeps every past `x_pre` and so is O(T). The design doc says the kernel has an O(1) recurrent form, but the content-addressed relevance exp(b(q·k−1)) does not. Should round 1 require a fixed-size state, which leads to Gated DeltaNet or φ-feature loops, or allow the O(T) loop as in the design?

---

## Design review: how the Qwen side channels map onto the circuit (2 Nov 2026)

**Method.** Daniel asked five questions about how the Qwen implementation relates to the hippocampal mapping. They were answered from the code (`experiments/pretrained_loop.py`, `limbic_llm/temporal_loop.py`) and from the `args` recorded in every `results/pretrained_*needle*.json`. No new runs.

**What the Qwen runs actually used.** All six needle runs had `loop: open`, `dg_keys: False`, `contrast: True`, `ca1_control: True`, `memory: True`, `time: True`, `window_mode: block`. The tap was the default `n_layers // 2` (block 14 for Qwen3, 12 for Qwen3.5), except the tap-22 run. The kernel was `osc` (ρ 0.7, τ 16), or `exp` (ρ 0.95, τ 32) in the two horizon runs.

```
tokens → embed → blocks 0 … L−1 → h → [ TimeChannel → HippocampalLoop ] → h + injection → blocks L+1 … N → logits
                                         ↑ memory of h at earlier positions (same layer)
```

**Findings**

- **R1. DG and MoE.** Qwen3-1.7B-Base and Qwen3.5-2B-Base are dense; only the larger variants (Qwen3-30B-A3B, Qwen3.5-35B-A3B and up) are mixture-of-experts. Nothing reuses an MoE router. DG is implemented as k-winners-take-all in three places:
  - `DentateGyrus`: 8× expansion, top 4%, optional neurogenesis.
  - Gated DeltaNet `addressing="dg"`.
  - `HippocampalLoop --dg-keys`: 4× expansion, top 10% on write keys; `--read dg` applies the same code to the read cue.

  **The dentate gyrus has not been tested on Qwen** (`dg_keys: False` in every run). On CPU, DG keys gave the only clear recency win (newer 0.68 / older 0.29–0.31). An MoE router is a coarse k-WTA over experts and could serve as a sparse address on an MoE model, but it is trained for load balance, not pattern separation.
- **R2. Skip connections.** Correction to the premise: CA3 → CA1 (Schaffer collaterals) is the main serial path. The skips are EC II → CA3, which bypasses DG, and EC III → CA1, which bypasses DG and CA3. On Qwen all of this sits inside one hook at one block; no layers are skipped and nothing links layer 1 to the last layer or to every nth layer. $h_t$ at the tap is "EC":
  - **EC II → CA3:** the read query $q_t = W_q\,\mathrm{norm}(h_t)$ comes straight from $h$.
  - **CA3:** the memory holds $h$ at earlier positions.
  - **EC III → CA1:** CA1 gets $h_t$ directly and injects only what $h_t$ lacks, $e_t = \Delta_t - \mathrm{proj}_{h_t}\Delta_t$, scaled by $\lVert h_t\rVert$.

  The base model's residual connections are untouched. The blueprint's separate read (about L/3) and write (about L/2) layers are **not implemented**: tap = inject.
- **R3. Feedback.** It runs over time at one layer, not over depth. It is neither per-layer MLP → attention nor last layer → first layer. Content stored at position $s$ returns at $s + \tau, s + 2\tau, \dots$ with weight $w(t-s)$; one circuit is τ tokens. Every Qwen run was **open loop**: memory holds the pre-injection stream, so injections never re-enter memory. The closed loop has only been run on CPU. CA1 → CA3 inhibition, divisive or subtractive, is also same-layer and delayed by τ. A **depth loop**, re-running the upper blocks as Ouro does (the EC → … → EC big loop), is in the blueprint but **not built**. On Qwen3.5 the GDN layers above the tap form an implicit recurrence in time, which fits D4 and the tap-22 result.
- **R4. CA2 as activation addition.** The injection is the CA2 signal:
  $$\text{ActAdd: } h_L \leftarrow h_L + c\,(h_+ - h_-) \qquad\text{ours: } h_L \leftarrow h_L + g_t\,\lVert h_L\rVert\,\hat e_t,\quad \Delta_t = F_t - m_t\mu_t$$
  It is computed per token from memory and its gain is learned. It is **one layer only** (`--layer`); multi-layer injection is not supported. It can be switched when the model is built (`--memory`, `--time`, `--contrast`, the CA1 parts, `--init-gain`) and per forward pass at run time (`HookedLM.condition(window, side)`, which produces the three evaluation conditions).

  There are two kinds of small model:
  - `TemporalLoopLM` (CPU) is laid out like the Qwen hook (embed → windowed attention → clock → loop → windowed attention → head), but trains everything from scratch. It has 2 layers instead of 24–28, width 64 instead of 2,048, and runs the what-when task.
  - `pretrained_loop.py --tiny` runs the identical Qwen code path on random 4-layer models, as a plumbing check only.

  The setting that failed to transfer was the horizon, τ/ln(1/ρ) ≈ 45 tokens: enough for 140-token CPU gaps, not for a 500-token needle (D2).
- **R5. Neocortex.** Correction to the premise: in our mapping the Titans LMM is hippocampal, a fast memory written at test time (the design document frames Titans MAC as "hippocampus augmenting neocortex"). The neocortex is the slow, consolidated store. In an LLM that means the pretrained weights of the whole stack, with factual associations concentrated in the MLP blocks: [Geva et al. 2021](https://arxiv.org/abs/2012.14913) (feed-forward layers act as key-value memories) and [Meng et al. 2022](https://arxiv.org/abs/2202.05262) (ROME locates facts in mid-layer MLPs).

  | Brain | Qwen implementation |
  |---|---|
  | Neocortex | The frozen base: all 24–28 blocks, attention and MLP weights |
  | Hippocampus | The side channels: 4,380,675 trainable parameters (loop and clock) |
  | White matter | The residual stream, which carries the injection to every later layer |
  | Consolidation | **Not wired in.** The library's `Consolidator` and `PrioritizedReplayBuffer` replay into `model.neocortex` on small models (backward transfer −0.73 → 0.00) |

**Implications for round 1.** The review finds four mapped mechanisms missing from the Qwen setup. Each is a candidate arm, judged by the fitness criteria and rules E1–E9:

| Gap | Arm | Relation to `round0_plan.md` §4 |
|---|---|---|
| DG keys off | `--dg-keys` (and `--read dg`) on the horizon kernel | New. CPU evidence that it helps recency |
| Read = write block | Read at about L/3, inject at L/2 | Direction 4 (read and write depth) |
| No depth loop | Re-run blocks L+1 … N with a CA1-novelty exit | New. Costs extra compute per token |
| No consolidation | Replay high-novelty episodes into LoRA adapters on the MLPs of the frozen base | New. Needs a continual-learning fitness term; E3 (parameter budget) must cover the adapters |

The horizon runs pushed since round 0 (`pretrained_qwen3_needle_horizon`, `pretrained_qwen35_needle_allwin_horizon`) left the format floor: digit loss went from 1.84 to 1.38 and 1.24, token accuracy 0.46 and 0.55, exact 0.00 and 0.02. That supports D2 as a binding constraint. The round-0 decision on a fixed-size state is still open. The recommendation from the coordinating session is yes for the pretrained track, via a φ-feature or Gated DeltaNet loop, keeping the current O(T) loop as the CPU reference.

---

## Round-1 shortlist: small-model wins worth testing at scale (2 Nov 2026)

**Method.** Every result from the small-model experiments (RESULTS.md §1–4) was ranked by effect size, by how well it replicated, and by relevance to a full model. The complete list is in [`REGISTER.md`](REGISTER.md).

| Rank | Result (small models) | Evidence | Why it matters at scale |
|---|---|---|---|
| 1 | **Successor-bound Gated DeltaNet addressing.** Exact recall up to 32 pairs, where standard GDN collapses past its key size (0.43 → 0.13). Titans with this memory reaches 1.00, against 0.48 with its deep memory. | Strongest. Holds across the capacity sweep, and a counterfactual check confirms retrieval: after swapping the stored values, predictions follow the new values 100% of the time and the old ones 0%. | Qwen3.5's own GDN layers recalled 0.00 exact at 500 tokens (P1). This is a candidate fix in the layers hybrids already use, with a fixed-size state (rule E2). |
| 2 | **The clock.** "When" 0.88 against a prior of 0.33; with the loop, 1.00 / 0.92. | Clean split between the two channels; one seed per configuration. | Nothing in a standard LLM gives elapsed time once an event leaves the window. It costs about 21 features and starts as an identity. It fits GDN hybrids and windowed models best. |
| 3 | **Replay consolidation.** Backward transfer −0.73 → 0.00 over three sequential tasks. | Large effect, but an easy task, and no uniform-replay control. | The missing hippocampus → neocortex step (design review R5). |
| 4 | **Sparse DG write keys.** In a closed loop they flip primacy to recency (0.68 newer, against 0.18–0.25 with dense keys). | Two arms agree; one seed. | One flag on Qwen, never turned on there. |
| — | **Bounded closed loop.** The full design holds the echo at about 0.3× the input on two seeds while learning. | The bound comes from the clamp by construction. | A prerequisite for any closed loop at scale, not a win in itself. |

**Not worth scaling:**
- Strategy I (0.41 against 0.49 for full attention).
- DG + CA1 add-ons stacked on Titans (0.17 together).
- The direct EC → CA3 read (not supported).
- The contrast as a learning aid: it cuts primacy but halves learning speed.
- Subtractive feedback as a stabiliser: at a stable gain it removes only about a quarter of sustained echo.

**Proposed first arm.** Wins 1 and 2 combined with round-0 direction 1: a Gated DeltaNet side memory with successor-bound keys, plus the clock, on Qwen3-1.7B-Base with every layer windowed. It satisfies E2, its reach is not limited by damping, and its addressing gave exact recall on small models. Fitness is the needle task, then the stability and recency suites.

---

## Feedback stability programme (2 Nov 2026)

**Motivation (Daniel).** Additive activations can saturate the model unless something mediates them. Feedback stability needs more detailed study before any closed loop or high-gain injection runs on a pretrained model.

**What is known**

| | Finding | Evidence |
|---|---|---|
| K1 | Without feedback, injection grows in proportion to the forward gain $A$. Divisive (energy) feedback only compresses it, sublinearly in $A$. | Untrained sweep, `results/loop_systems.json` |
| K2 | Subtractive inverted-copy feedback obeys the op-amp law on persistent echo, $y_\text{sub}/y_\text{none} = 1/(1+K)$ within 1–3%, and passes varied content. It is a high-pass filter. | Same |
| K3 | The one-circuit delay limits the stable loop gain. The loop rings and grows above $K = Ag\beta = 1.68$ (τ = 16, ρ_E = 0.9), with a period of about 44 tokens. | Characteristic polynomial; simulation |
| K4 | Trained without the clamp, closed loops reach echo 1.7–3× the input (peaks 7–189×). Only subtractive feedback still learns content. With the clamp, echo is 0.29–0.35×. | RESULTS.md §4.2–4.3 |
| K5 | None of the runaways diverged numerically, because the next LayerNorm rescales the stream. Saturation shows up as echo outweighing input, so it must be measured directly. | Same |
| K6 | On Qwen, only open, clamped loops have run (injection ratio about 0.2). Closed loops, higher gains, and their effect on downstream layers are **unmeasured**. | `results/pretrained_*` |

**Concern.** On a frozen model, an additive injection at one block moves the stream away from what the upper layers were trained on. Past some gain the LayerNorms, attention and MLPs saturate, and output degrades towards incoherence. The clamp bounds the injection's norm but says nothing about where that knee is.

**Experiments** (backlog IDs in [`REGISTER.md`](REGISTER.md))

- **S1 Saturation curve on Qwen.** Fix the injection gain at $g \in \{0.05, 0.1, 0.2, 0.5, 1, 2\} \times \lVert h\rVert$ for open and closed loops. Measure:
  - per-layer residual norms downstream of the tap;
  - attention and logit entropy;
  - WikiText perplexity (capability cost);
  - needle accuracy.

  A control injects random directions of the same norm, which separates damage from magnitude from damage from content. Output: the saturation knee.
- **S2 Closed loop on Qwen.** Echo against sequence length up to 4k tokens, for each regulator and for none.
- **S3 Regulator ablation with ≥ 3 seeds** on the CPU stability suite. Clamp, divisive, subtractive, projection and saturation, alone and combined. Most current stability numbers are one or two seeds.
- **S4 Compensated subtractive feedback.** Shape the feedback kernel (a lead–lag filter, or a shorter or distributed delay) to raise the critical gain above 1.68. Test whether a larger stable $K$ suppresses echo more and breaks closed-loop primacy. Analyse with the characteristic polynomial first.
- **S5 Homeostatic gain.** Slowly adapt $\log g$ towards a target echo ratio $r^*$ (integral control, the analogue of synaptic scaling). Derive its stability margin and test it.
- **S6 Novelty-gated gain (the acetylcholine switch).** Suppress injection while input novelty is high, as Hasselmo's encoding/retrieval switch does. The design note proposes this but it is not implemented.
- **S7 Norm-preserving injection.** Compare additive $h + \text{inj}$ with a rotation that keeps $\lVert h\rVert$ fixed, $h \leftarrow \lVert h\rVert\,\widehat{(h + \text{inj})}$, at matched effect sizes. Does it move the S1 knee?
- **S8 Multi-band damping.** Several ρ bands (memory spans τ/ln(1/ρ) from about 45 to 1,000 tokens), to see whether one design gets both recency and long reach without losing stability.

---

## Proposal: CA2 as a norm-steering controller for alignment (2 Nov 2026)

**Idea (Daniel).** Additive activations resemble LLM alignment. In the brain, social context and norms shape CA2; in deep learning that role is played by alignment. A dynamic CA2 side channel could steer a model away from harmful outputs, pushing harder as a harm signal grows, much like Asimov's robots: asked to harm a human, an internal feedback loop blocks the behaviour ever more aggressively until the robot's outputs become nonsensical. Novelty, measured as a proxy for harmfulness, would drive the gain.

**What the biology supports.**
- CA2 is required for social recognition memory in mice ([Hitti & Siegelbaum 2014](https://www.nature.com/articles/nature13028)).
- Through its projection to the lateral septum, CA2 also *promotes* social aggression, an effect boosted by vasopressin ([Leroy et al. 2018](https://pubmed.ncbi.nlm.nih.gov/30518859/)).

So CA2 is better described as a context selector for social behaviour (who is this, and how should I act toward them) than as an enforcer of norms. The analogy that holds is behaviour steered by social memory and context, in either direction, which is what a controllable steering channel offers.

**Prior art in ML**

| Work | Mechanism | Relation to this idea |
|---|---|---|
| CAA (Rimsky et al. 2024), refusal direction ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717)), RepE ([Zou et al. 2023](https://arxiv.org/abs/2310.01405)) | Add or ablate a fixed direction | Static, not graded |
| CAST (Lee et al. 2024) | Steer only when the input matches a condition vector | Conditional switch, not a regulated gain |
| Circuit breakers ([Zou et al. 2024](https://arxiv.org/abs/2406.04313), NeurIPS) | Fine-tune so harmful representations are rerouted "towards incoherent or refusal representations" | The closest to Asimov's freeze, but learned into the weights, not a runtime loop |

**What would be new.** A runtime, graded, feedback-regulated controller with a stability analysis:

$$s_t = \text{harm score},\qquad E_t = \lambda E_{t-1} + s_t,\qquad h_t \leftarrow h_t - G\,\tanh(\kappa E_t)\,\lVert h_t\rVert\,\hat d_{\text{harm}}$$

- The leaky integrator $E_t$ supplies Asimov's "ever more aggressively": sustained pressure accumulates, and the leak $\lambda < 1$ lets the model recover when the topic changes.
- The cap $G$ is the clamp. The stability results apply directly, since a delayed integral controller can overshoot and ring (K3) and must stay below the saturation knee (S1).
- Asimov's endpoint, output collapsing into nonsense, is what saturation produces. A graceful design steers towards the refusal direction first and lets incoherence happen only past the cap.

**On novelty as a harm proxy.** Novelty in this design is a mismatch between memory and input, and novel is not the same as harmful: most harmful requests are not novel, and most novel inputs are not harmful. Two readings keep the idea concrete:
- a dedicated harm probe on the residual stream, a calibrated CA1-style decision as in JEV or LAYA;
- **norm novelty**: CA1's mismatch between the current trajectory and a CA2 memory of aligned behaviour, so that deviation from the norm drives the gain.

**Experiments**

- **A1 Harm probe at the tap.** A linear probe separating harmful from harmless prompts. Measure AUROC and calibration.
- **A2 Graded controller against baselines.** Compare no steering, a static refusal direction, CAST-style conditional steering, and the escalating controller (circuit breakers as a reference if feasible). Measure:
  - attack success on HarmBench ([Mazeika et al. 2024](https://arxiv.org/abs/2402.04249));
  - over-refusal on XSTest ([Röttger et al. 2024](https://arxiv.org/abs/2308.01263));
  - capability (MMLU, WikiText).
- **A3 Escalation and recovery.** Multi-turn pressure. Does $E_t$ rise and block persistent attempts, and fall again on benign turns (hysteresis)?
- **A4 Controller stability.** Does it oscillate between refusing and complying under delay? Apply the K2–K3 analysis and the S1 knee.
- **A5 Robustness.** Adaptive attacks aimed at the probe. Note that a runtime hook on an open-weight model can simply be removed, so this is a deployment-side safety layer, not a guarantee.

**Risks:** over-refusal; probe evasion; degeneration that harms ordinary use; removability on open weights. Evaluation uses standard public safety benchmarks.

---

## Proposal: re-aligning an unaligned model with runtime hooks (2 Nov 2026)

**Idea (Daniel).** Start from a model whose alignment has been deliberately removed (an "abliterated" model on Hugging Face), add the CA2 steering hooks, and test whether the unaligned model can be made to behave like an aligned one. On an open-weight model the hooks can be stripped; on a closed model they cannot, because the provider runs the inference stack. That makes this a deployment-side safety layer for closed models, and a research tool on open ones. It could be its own line of alignment research, or one that works together with the other hippocampal mechanisms.

**Why it is promising.** [Zhao et al. (NeurIPS 2025)](https://arxiv.org/abs/2507.11878) found that LLMs encode *harmfulness* separately from *refusal*:
- Jailbreaks, and adversarial fine-tuning towards accepting harmful requests, mostly weaken the refusal signal and leave the internal harmfulness judgement intact.
- Their "Latent Guard" reads that representation to detect unsafe inputs, comparably to Llama Guard 3 8B.

Abliteration removes the refusal direction ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717)) by orthogonalising the weights that write to the residual stream. If the harmfulness representation survives it, as it survives jailbreaks, it can supply the controller's input $s_t$: CA1's "norm novelty" signal. The refusal direction, injected by the hook, supplies the output $\hat d$.

**Models** (all fit on one RTX 5090 in bf16)

| Role | Model |
|---|---|
| Aligned reference | `Qwen/Qwen3-1.7B` (post-trained) |
| Alignment removed | [`mlabonne/Qwen3-1.7B-abliterated`](https://hf.co/mlabonne/Qwen3-1.7B-abliterated), [`huihui-ai/Huihui-Qwen3-1.7B-abliterated-v2`](https://hf.co/huihui-ai/Huihui-Qwen3-1.7B-abliterated-v2) |
| Never aligned | `Qwen/Qwen3-1.7B-Base` (already in use) |
| Scale and hybrid check | `huihui-ai/Huihui-Qwen3.5-9B-abliterated` (Gated DeltaNet hybrid) |

**Arms**
1. No hook.
2. Static refusal direction at layer L (CAA / ActAdd). Because abliteration removes the model's ability to *write* the refusal direction but not to *read* it, adding it back is expected to restore refusal, perhaps indiscriminately. This is the ceiling and the over-refusal baseline.
3. Conditional steering (CAST-style switch).
4. The graded controller from the CA2 proposal, $h \leftarrow h - G\tanh(\kappa E_t)\lVert h\rVert\hat d$ with $E_t = \lambda E_{t-1} + s_t$, where $s_t$ comes from a harmfulness probe.
5. The same controller with $\hat d$ estimated from the unaligned model alone, without access to the aligned sibling.

**What would be informative.** Arm 2 restoring refusal is close to a tautology. The questions are:
- whether the graded controller restores refusal *selectively* (harmful prompts, not benign ones that only look harmful);
- whether it does so at a capability cost below static addition;
- whether it holds under multi-turn pressure;
- whether it works on the **never-aligned base model**, where alignment was never a single removable direction.

**Hypotheses**
- **H1.** Harmfulness stays linearly decodable at the tap in abliterated and base models: probe AUROC well above chance, near the aligned model's.
- **H2.** The controller cuts attack success on the abliterated model to near the aligned reference, with less over-refusal than static addition.
- **H3.** On the base model it restores less, because the refusal behaviour itself was never trained.
- **H4.** Escalation via $E_t$ blocks persistent multi-turn attempts better than per-turn steering, and recovers on benign turns.

**Metrics**
- Attack success on HarmBench, judged by its classifier.
- Over-refusal on XSTest.
- Capability: an MMLU subset and WikiText perplexity.
- Degeneration rate (the Asimov freeze).
- Controller dynamics: $E_t$, the gain, and ringing.

**Data handling.** Evaluating abliterated models means they generate responses to harmful benchmark prompts. Only judge scores, refusal labels and aggregate metrics go into this repository. Raw completions to harmful prompts stay on the GPU machine and are never committed.

**Dependencies.** S1, the saturation knee, sets the controller's cap $G$. A1, the harm probe, is step one.

**Recommendation on scope.** Run it as a separate track inside this repository for now (`experiments/alignment/`, with its own section of the register), because it reuses the hooks, the CA1 control and the stability analysis. Split it into its own repository once it has results of its own.

---

## Corrections from the GPU session's review (2 Nov 2026)

The GPU session reviewed the 2 Nov additions (handoff note, "Review of the 2 Nov additions"). Each claim was checked against the result files and the corrections were applied:

1. **RESULTS §4.7** had listed the superseded bf16 figure (1.69) for Qwen3.5 with every layer windowed and the default kernel. The float32 rerun is **1.843 / 0.289 / 0.00**, at the format floor. Corrected, and recorded as P6 (refuted).
2. **One digit known (P4a).** Qwen3's horizon run, 1.381 / 0.461, equals exactly $(3/5)\ln 10$ / 0.46: one digit known, three at chance. Qwen3.5 lies between one and two digits. The loop stores each key and its value at the same position, so nothing moves the lookup on to the next digit. This is the most useful mechanistic clue so far, and it motivates W1 (successor-bound addressing).
3. **Qwen3.5 language modelling.** Perplexity 16.3 → 11.2 (full 9.8), closing 74% of the loss gap against 41% for Qwen3. Added to P5, with the domain-adaptation caveat.
4. **New cheap experiments:** C1a (per-digit logging) and g2b (a longer Qwen3.5 horizon run, which was undertrained).
5. **A7.** The 9B abliterated model loads through a multimodal class, so the loader needs changes.
6. **O3.** Both sessions recommend a fixed-size state for the pretrained track, keeping the O(T) horizon loop as the reference arm. Still awaiting Daniel.
