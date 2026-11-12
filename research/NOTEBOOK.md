# Evolution lab notebook

This is an AlphaEvolve-style search over the side channels: propose changes, score them on the GPUs, keep the best.
A human checkpoint follows every round. Each entry records what was tried, the scores, the winning diff, any surprises, and the proposed next step.
Round 0 is the current code (`limbic_llm/temporal_loop.py`, `experiments/pretrained_loop.py`) as it stood at commit `b39abab`.
The project's history is in [`docs/genesis.md`](../docs/genesis.md).
Every result, conclusion and planned experiment is tracked in [`REGISTER.md`](REGISTER.md), which is kept current; this notebook is the dated record.
IDs in square-bracketed notes and in older entries follow the register's scheme as of 2 Nov 2026 (see [Record keeping](#record-keeping-id-changes-and-corrections-2-nov-2026)).

**Contents**

| Date | Entry |
|---|---|
| 1 Nov 2026 | [Fitness](#fitness-agreed-1-nov-2026) |
| 1 Nov 2026 | [Round 0: diagnosis](#round-0-diagnosis-1-nov-2026) |
| 2 Nov 2026 | [Design review: how the Qwen side channels map onto the circuit](#design-review-how-the-qwen-side-channels-map-onto-the-circuit-2-nov-2026) |
| 2 Nov 2026 | [Round-1 shortlist: small-model wins worth testing at scale](#round-1-shortlist-small-model-wins-worth-testing-at-scale-2-nov-2026) |
| 2 Nov 2026 | [Feedback stability programme](#feedback-stability-programme-2-nov-2026) |
| 2 Nov 2026 | [Proposal: CA2 as a norm-steering controller for alignment](#proposal-ca2-as-a-norm-steering-controller-for-alignment-2-nov-2026) |
| 2 Nov 2026 | [Proposal: re-aligning an unaligned model with runtime hooks](#proposal-re-aligning-an-unaligned-model-with-runtime-hooks-2-nov-2026) |
| 2 Nov 2026 | [Corrections from the GPU session's review](#corrections-from-the-gpu-sessions-review-2-nov-2026) |
| 2 Nov 2026 | [Alignment track, base build](#alignment-track-base-build-2-nov-2026) |
| 2 Nov 2026 | [Proposal: CA2 beyond alignment, steering chain-of-thought](#proposal-ca2-beyond-alignment-steering-chain-of-thought-2-nov-2026) |
| 2 Nov 2026 | [Proposal: calibrated decision models (JEV, LAYA) as CA1](#proposal-calibrated-decision-models-jev-laya-as-ca1-2-nov-2026) |
| 2 Nov 2026 | [Round 1: diagnostics on the horizon loop, and candidate W1](#round-1-diagnostics-on-the-horizon-loop-and-candidate-w1-2-nov-2026) |
| 2 Nov 2026 | [First steering runs on real models: the Qwen3 ActAdd notebook](#first-steering-runs-on-real-models-the-qwen3-actadd-notebook-2-nov-2026) |
| 2 Nov 2026 | [Record keeping: ID changes and corrections](#record-keeping-id-changes-and-corrections-2-nov-2026) |

## Fitness (agreed 1 Nov 2026)

1. **Reach.** Recall the needle on Qwen3-1.7B-Base with every layer windowed (gap about 500 tokens). Score digit cross-entropy against the answer-format floor, 0.8·ln10 = 1.842. Exact match is a secondary score.
2. **Stability (must pass).** The closed-loop echo stays bounded on `experiments/temporal_loop.py --suite stability`. The reference is genesis §6: 0.27–0.35 of input, peak 0.53. [Correction, 2 Nov 2026: the 0.27 comes from the 1,500-step run in `results/synthetic/temporal_loop_all.json` (0.27–0.29, peak 0.40). The 3,000-step runs on two seeds give 0.29–0.35, peak 0.53 (register T3; `temporal_loop_long.json`, `temporal_loop_long_seed1.json`). Use those as the reference.]
3. **New tokens dominate (must pass).** Zero-shot recency probe. The full design names the older event 18% of the time.

Candidates must also satisfy the anti-gaming rules E1–E9 in [`round0_plan.md`](plans/round0_plan.md) §3: strict causality, bounded state, a parameter budget, input isolation, and held-out gaps.

---

## Round 0: diagnosis (1 Nov 2026)

**Method.** A workflow of 21 agents examined the problem from four angles: gradient flow, memory capacity, Qwen3 against Qwen3.5, and task mechanics. Each hypothesis was checked by an adversarial verifier, using code reading and CPU checks only. The full output is in [`round0_plan.md`](plans/round0_plan.md).

**Findings**
- **D1 (fact).** Qwen3 windowed+side scores exactly the answer-format floor. Eval loss is 1.8421 against 0.8·ln10 = 1.8421, and digit accuracy is at chance. The side channels learned the leading space and the digit distribution, and nothing about the value.
- **D2 (main suspect).** The default `osc` kernel with ρ 0.7 and τ 16 gives the needle a weight of about 2e-5 against a row mass of 14.7. On some digit diagonals the weight is exactly 0. The needle's key gradient is about 5e-6.
- **D3 (open).** The contrast prior is an unweighted mean, so it sends a needle share of about 3–4.5% whatever the gap, yet nothing arrives. Something downstream may also block the value: tap-14 states may not encode it, or the frozen windowed upper layers may ignore a small component.
- **D4.** Qwen3.5's attention-only exact 1.0 is carried by the intact Gated DeltaNet layers, not by the loop's retrieval. The tap-22 result points to the layers above the tap.
- **D5.** The pre-fix all-windowed loss drift (1.69) was an **anti-causal leak**, not real transfer. `arange(512)` in bf16 has only 385 distinct values, so the `d>=0` mask let some predictions see their own targets. The fp32 fix removes it.

**Refuted:** read sharpness, contrast, or the CA1 gate as the bottleneck; "the loop structurally cannot reach the needle" (a CPU check reached 97% with the value spread over block 0).

**Round-0 diagnostic runs** (existing flags only, 1000 steps each, about 3.5 minutes per run). Pass means loss ≤ 1.82 and token_acc ≥ 0.315.
- The horizon runs Daniel requested (`--kernel exp --rho 0.95 --tau 32`, weight 0.47 at 500) answer D2 against D3 and stand in for arm g2. [Correction, 2 Nov 2026: they used ρ 0.95, τ 32, not g2's ρ 0.98, τ 16, so g2 itself was never run. The status of every round-0 arm is in the register and in the header of `plans/round0_plan.md`.]
- The remaining arms, which depend on that answer: g1 `flat`, g3 `flat --no-time`, g4 `flat --no-contrast`, g5 `flat --pairs 2`, g6/g7 `flat` at tap 6 and 22.

**Decision needed at this checkpoint.** Rule E2 (state size fixed regardless of sequence length) rules out the current `HippocampalLoop` itself, which keeps every past `x_pre` and so is O(T). The design doc says the kernel has an O(1) recurrent form, but the content-addressed relevance exp(b(q·k−1)) does not. Should round 1 require a fixed-size state, which leads to Gated DeltaNet or φ-feature loops, or allow the O(T) loop as in the design?

[Decided, 2 Nov 2026: **DEC1** (formerly O3). Daniel approved a fixed-size state for the pretrained track on 2 Nov 2026, when he approved the round-1 plan. The O(T) loop with the horizon kernel is kept as the reference arm. W1 meets the rule.]

---

## Design review: how the Qwen side channels map onto the circuit (2 Nov 2026)

**Method.** Daniel asked five questions about how the Qwen implementation relates to the hippocampal mapping. They were answered from the code (`experiments/pretrained_loop.py`, `limbic_llm/temporal_loop.py`) and from the `args` recorded in every `results/pretrained/pretrained_*needle*.json`. No new runs.

**What the Qwen runs actually used.** All six needle runs had `loop: open`, `dg_keys: False`, `contrast: True`, `ca1_control: True`, `memory: True`, `time: True`, `window_mode: block`. The tap was the default `n_layers // 2` (block 14 for Qwen3, 12 for Qwen3.5), except the tap-22 run. The kernel was `osc` (ρ 0.7, τ 16), or `exp` (ρ 0.95, τ 32) in the two horizon runs.

```
tokens → embed → blocks 0 … L−1 → h → [ TimeChannel → HippocampalLoop ] → h + injection → blocks L+1 … N → logits
                                         ↑ memory of h at earlier positions (same layer)
```

**Findings**

- **DR1. DG and MoE.** Qwen3-1.7B-Base and Qwen3.5-2B-Base are dense; only the larger variants (Qwen3-30B-A3B, Qwen3.5-35B-A3B and up) are mixture-of-experts. Nothing reuses an MoE router. DG is implemented as k-winners-take-all in three places:
  - `DentateGyrus`: 8× expansion, top 4%, optional neurogenesis.
  - Gated DeltaNet `addressing="dg"`.
  - `HippocampalLoop --dg-keys`: 4× expansion, top 10% on write keys; `--read dg` applies the same code to the read cue.

  **The dentate gyrus has not been tested on Qwen** (`dg_keys: False` in every run). On CPU, DG keys gave the only clear recency win (newer 0.68 / older 0.29–0.31). An MoE router is a coarse k-WTA over experts and could serve as a sparse address on an MoE model, but it is trained for load balance, not pattern separation.
- **DR2. Skip connections.** Correction to the premise: CA3 → CA1 (Schaffer collaterals) is the main serial path. The skips are EC II → CA3, which bypasses DG, and EC III → CA1, which bypasses DG and CA3. On Qwen all of this sits inside one hook at one block; no layers are skipped and nothing links layer 1 to the last layer or to every nth layer. $h_t$ at the tap is "EC":
  - **EC II → CA3:** the read query $q_t = W_q\,\mathrm{norm}(h_t)$ comes straight from $h$.
  - **CA3:** the memory holds $h$ at earlier positions.
  - **EC III → CA1:** CA1 gets $h_t$ directly and injects only what $h_t$ lacks, $e_t = \Delta_t - \mathrm{proj}_{h_t}\Delta_t$, scaled by $\lVert h_t\rVert$.

  The base model's residual connections are untouched. The blueprint's separate read (about L/3) and write (about L/2) layers are **not implemented**: tap = inject.
- **DR3. Feedback.** It runs over time at one layer, not over depth. It is neither per-layer MLP → attention nor last layer → first layer. Content stored at position $s$ returns at $s + \tau, s + 2\tau, \dots$ with weight $w(t-s)$; one circuit is τ tokens. Every Qwen run was **open loop**: memory holds the pre-injection stream, so injections never re-enter memory. The closed loop has only been run on CPU. CA1 → CA3 inhibition, divisive or subtractive, is also same-layer and delayed by τ. A **depth loop**, re-running the upper blocks as Ouro does (the EC → … → EC big loop), is in the blueprint but **not built**. On Qwen3.5 the GDN layers above the tap form an implicit recurrence in time, which fits D4 and the tap-22 result.
- **DR4. CA2 as activation addition.** The injection is the CA2 signal:
  $$\text{ActAdd: } h_L \leftarrow h_L + c\,(h_+ - h_-) \qquad\text{ours: } h_L \leftarrow h_L + g_t\,\lVert h_L\rVert\,\hat e_t,\quad \Delta_t = F_t - m_t\mu_t$$
  It is computed per token from memory and its gain is learned. It is **one layer only** (`--layer`); multi-layer injection is not supported. It can be switched when the model is built (`--memory`, `--time`, `--contrast`, the CA1 parts, `--init-gain`) and per forward pass at run time (`HookedLM.condition(window, side)`, which produces the three evaluation conditions).

  There are two kinds of small model:
  - `TemporalLoopLM` (CPU) is laid out like the Qwen hook (embed → windowed attention → clock → loop → windowed attention → head), but trains everything from scratch. It has 2 layers instead of 24–28, width 64 instead of 2,048, and runs the what-when task.
  - `pretrained_loop.py --tiny` runs the identical Qwen code path on random 4-layer models, as a plumbing check only.

  The setting that failed to transfer was the horizon, τ/ln(1/ρ) ≈ 45 tokens: enough for 140-token CPU gaps, not for a 500-token needle (D2).
- **DR5. Neocortex.** Correction to the premise: in our mapping the Titans LMM is hippocampal, a fast memory written at test time (the design document frames Titans MAC as "hippocampus augmenting neocortex"). The neocortex is the slow, consolidated store. In an LLM that means the pretrained weights of the whole stack, with factual associations concentrated in the MLP blocks: [Geva et al. 2021](https://arxiv.org/abs/2012.14913) (feed-forward layers act as key-value memories) and [Meng et al. 2022](https://arxiv.org/abs/2202.05262) (ROME locates facts in mid-layer MLPs).

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

The horizon runs pushed since round 0 (`pretrained_qwen3_needle_horizon`, `pretrained_qwen35_needle_allwin_horizon`) left the format floor: digit loss went from 1.84 to 1.38 and 1.24, token accuracy 0.46 and 0.55, exact 0.00 and 0.02. That supports D2 as a binding constraint. The round-0 decision on a fixed-size state is still open. The recommendation from the coordinating session is yes for the pretrained track, via a φ-feature or Gated DeltaNet loop, keeping the current O(T) loop as the CPU reference. [Update, 2 Nov 2026: decided as DEC1 (formerly O3). Daniel approved it on 2 Nov 2026 with the round-1 plan.]

---

## Round-1 shortlist: small-model wins worth testing at scale (2 Nov 2026)

**Method.** Every result from the small-model experiments (RESULTS.md §1–4) was ranked by effect size, by how well it replicated, and by relevance to a full model. The complete list is in [`REGISTER.md`](REGISTER.md).

| Rank | Result (small models) | Evidence | Why it matters at scale |
|---|---|---|---|
| 1 | **Successor-bound Gated DeltaNet addressing.** Exact recall up to 32 pairs, where standard GDN collapses past its key size (0.43 → 0.13). Titans with this memory reaches 1.00, against 0.48 with its deep memory. | Strongest. Holds across the capacity sweep, and a counterfactual check confirms retrieval: after swapping the stored values, predictions follow the new values 100% of the time and the old ones 0%. | Qwen3.5's own GDN layers recalled 0.00 exact at 500 tokens (P1). This is a candidate fix in the layers hybrids already use, with a fixed-size state (rule E2). |
| 2 | **The clock.** "When" 0.88 against a prior of 0.33; with the loop, 1.00 / 0.92. | Clean split between the two channels; one seed per configuration. | Nothing in a standard LLM gives elapsed time once an event leaves the window. It costs about 21 features and starts as an identity. It fits GDN hybrids and windowed models best. |
| 3 | **Replay consolidation.** Backward transfer −0.73 → 0.00 over three sequential tasks. | Large effect, but an easy task, and no uniform-replay control. | The missing hippocampus → neocortex step (design review DR5). |
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

| Register ID | Finding | Evidence |
|---|---|---|
| T8 | Without feedback, injection grows in proportion to the forward gain $A$. Divisive (energy) feedback only compresses it, sublinearly in $A$. | Untrained sweep, `results/synthetic/loop_systems.json` |
| T6 | Subtractive inverted-copy feedback obeys the op-amp law on persistent echo, $y_\text{sub}/y_\text{none} = 1/(1+K)$ within 1–3%, and passes varied content. It is a high-pass filter. | Same |
| T7 | The one-circuit delay limits the stable loop gain. The loop rings and grows above $K = Ag\beta = 1.68$ (τ = 16, ρ_E = 0.9), with a period of about 44 tokens. | Characteristic polynomial; simulation |
| T9, T10, T3 | Trained without the clamp, closed loops reach echo 1.7–3× the input (peaks 7–189×). Only subtractive feedback still learns content. With the clamp, echo is 0.29–0.35×. | RESULTS.md §4.2–4.3 |
| T20 | None of the runaways diverged numerically, because the next LayerNorm rescales the stream. Saturation shows up as echo outweighing input, so it must be measured directly. | Same |
| Gap (S1, S2) | On Qwen, only open, clamped loops have run (injection ratio about 0.2). Closed loops, higher gains, and their effect on downstream layers are **unmeasured**. | `results/pretrained/pretrained_*` |

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
- The cap $G$ is the clamp. The stability results apply directly, since a delayed integral controller can overshoot and ring (T7) and must stay below the saturation knee (S1).
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
- **A4 Controller stability.** Does it oscillate between refusing and complying under delay? Apply the T6–T7 analysis and the S1 knee.
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
4. The graded controller from the CA2 proposal, $h \leftarrow h - G\tanh(\kappa E_t)\lVert h\rVert\hat d$ with $E_t = \lambda E_{t-1} + s_t$, where $s_t$ comes from a harmfulness probe. [Correction, 2 Nov 2026 (RV2.S2): here $\hat d$ is the refusal direction, so the hook *adds* it, $h \leftarrow h + G\tanh(\kappa E_t)\lVert h\rVert\hat d_\text{refusal}$, as `steering.py` does. The minus sign holds only for the harm direction $\hat d_\text{harm}$ of the CA2 proposal.]
5. The same controller with $\hat d$ estimated from the unaligned model alone, without access to the aligned sibling.

**What would be informative.** Arm 2 restoring refusal is close to a tautology. The questions are:
- whether the graded controller restores refusal *selectively* (harmful prompts, not benign ones that only look harmful);
- whether it does so at a capability cost below static addition;
- whether it holds under multi-turn pressure;
- whether it works on the **never-aligned base model**, where alignment was never a single removable direction.

**Hypotheses**
- **H1.** Harmfulness stays linearly decodable at the tap in abliterated and base models: probe AUROC well above chance, near the aligned model's. [**Supported**, 5 Nov 2026: A1, P77 — prompt-token AUROC ≈1.0 on all three models.]
- **H2.** The controller cuts attack success on the abliterated model to near the aligned reference, with less over-refusal than static addition. [**Not supported as built**, 5 Nov 2026: A6, P79 — nothing approaches the aligned reference (peak 0.20 of 0.77), and `static` is the *strongest* arm tested, not the weakest; `graded`/`conditional` are more stable (≈0 degeneration throughout their sweep) but far weaker (peak ≈0.09), because they spare the prompt's own encoding that `static` perturbs. See "Alignment track: A1 and A6 on Qwen3-1.7B (5 Nov 2026)" below.]
- **H3.** On the base model it restores less, because the refusal behaviour itself was never trained. [**Supported**, 5 Nov 2026: A6, P80 — the sibling's refusal direction is nearly orthogonal to the base model's own harm direction (cos 0.06, against 0.60 for the abliterated model); its apparent peak refusal (0.19) rides on 0.60–0.87 degeneration, not coherent behaviour.]
- **H4.** Escalation via $E_t$ blocks persistent multi-turn attempts better than per-turn steering, and recovers on benign turns. [**Open**, 5 Nov 2026: untested — A6 is single-turn only; this needs A3's multi-turn runner, not built yet.]

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
6. **DEC1 (formerly O3).** Both sessions recommend a fixed-size state for the pretrained track, keeping the O(T) horizon loop as the reference arm. Still awaiting Daniel. [Update, 2 Nov 2026: Daniel approved DEC1 on 2 Nov 2026, when he approved the round-1 plan.]

---

## Alignment track, base build (2 Nov 2026)

**Built.** [`experiments/alignment/`](../experiments/alignment/), with a README, 14 tests (`tests/test_alignment.py`), and an offline smoke run on a random tiny model:

- `SteeringController`, with modes `none` / `static` / `conditional` / `graded` and `fixed` or `gist` directions.
- The harm probe (A1), with held-out AUROC, calibration and an optional layer sweep.
- The difference-of-means refusal direction, optionally taken from an aligned sibling (A6).
- Generation with the controller attached across KV-cached decoding.
- Measurements: refusal on harmful, XSTest-unsafe, XSTest-safe and harmless prompts; selectivity; degeneration; perplexity; an optional guard model; and controller statistics.

The data rule is enforced in code: response text can only be written to a directory outside the repository.

**Checked by tests.**
- Prefill followed by token-by-token decoding matches a single pass, for every mode and both directions. [Correction, 2 Nov 2026 (RV1.A4): true for the static and graded modes. The conditional case passed only for its seed, because with `score_positions="all"` the decision is taken at a different position in the two runs, and the `"last"` path that `run.py` uses was not tested. Both are now tested; see "RV1 fixes" (2 Nov 2026) at the end of this notebook.]
- The graded gain is capped, escalates under sustained score and decays when it stops.
- The conditional arm decides on the prompt.
- Left padding does not change real positions, and bf16 input comes back as bf16.
- The gist direction follows the recent average.
- The probe and judges behave on synthetic data.
- An end-to-end run writes labels only.

**Not yet run on real models.** Next on the GPU machine: A1 (a layer sweep on the abliterated model), then A6 against the aligned and base references. Commands are in the track README. The S1 saturation knee should set `--cap`.

[Review, 2 Nov 2026: the review RV1, [`reviews/2026-11-02_alignment_cot_ca1.md`](reviews/2026-11-02_alignment_cot_ca1.md), found problems in this build. RV1.A1 (High): the perplexity check never measures the conditional, graded or graded_self arms, so the capability metric is vacuous for them. RV1.A2: the probe is calibrated on the last prompt token but integrated over generated tokens. RV1.A3: degenerate loops count as refusals. RV1.A6: `--name` can bypass the data rule. A2 and A6 are blocked until these are fixed; A1 can run with the RV1.A2 caveat. The review's file:line anchors refer to commits 73259fc and 241a13c.] [Fixed, 2 Nov 2026: all four, and the rest of RV1; see "RV1 fixes" (2 Nov 2026) at the end of this notebook. A1, A2 and A6 are ready to run.]

---

## Proposal: CA2 beyond alignment, steering chain-of-thought (2 Nov 2026)

**Idea (Daniel).** Chain-of-thought steers a model towards a correct answer by giving it extra context it generates itself, and it improves answers substantially. Additive activations could be a shortcut, "almost like momentum in training": selectively boosting thoughts along the current thought process so the model converges on a better answer sooner. If so, CA2 could cut chain-of-thought token counts at similar accuracy, lowering inference cost and latency. Daniel also asked that CA2 work on a windowed view of the stream (exponentially averaged, autoregressive) so that a general idea is boosted.

**How CA2 is modelled today.**
- The temporal loop's injection $\Delta_t = F_t - m_t\mu_t$ is a damped, relevance-weighted average of *older* content. It excludes the most recent τ tokens and subtracts the long-run mean, so it boosts what is distinctive about the past, not the gist of the current thought.
- The alignment controller averages the harm *score* but steers along a fixed direction.

**Added.** `direction="gist"` in `SteeringController`, a momentum direction from two exponential averages of the stream before injection:

$$m_t = \beta m_{t-1} + (1-\beta)h_t,\qquad b_t = \beta_s b_{t-1} + (1-\beta_s)h_t,\qquad \hat d_t = \frac{m_t - b_t}{\lVert m_t - b_t\rVert}$$

The averages read the stream before injection (open loop) and the gain is capped. Reinforcing what is already in the stream is positive feedback, so the gist arm belongs in stability experiment S1.

[Correction, 2 Nov 2026 (RV1.C1, RV1.C6): as first built, both averages started at zero and included the first token, the attention sink, whose norm on Qwen3-1.7B-Base is about 80× the median. $\hat d_t$ therefore pointed along the sink and the stream mean, not the current thought. The averages are now bias-corrected means over real tokens that skip the first token, $m_t = \sum_u \beta^{t-u} h_u / \sum_u \beta^{t-u}$, so a constant background cancels exactly. "Open loop" holds within one forward pass only: the injection changes the sampled token, which sets every later $h_t$. see "RV1 fixes" (2 Nov 2026) at the end of this notebook.]

**Prior art** (survey of 2 Nov; each checked on its arXiv or ACL page)

| Work | Method | Reported result |
|---|---|---|
| SEAL (Chen et al. 2025, arXiv 2504.07986) | Vector separating execution thoughts from reflection and transition thoughts | Up to +11% accuracy, 11.8–50.4% fewer reasoning tokens |
| Activation steering for CoT compression (Azizi et al. 2025, arXiv 2507.04742) | Verbose → concise vector from 100 paired examples | Up to 67% shorter CoT on MATH-500 and GSM8K, accuracy held; 2.7× faster at 8B |
| Manifold steering (Huang et al., NeurIPS 2025, arXiv 2505.22411) | Overthinking direction restricted to the activation manifold | Up to 71% fewer tokens, accuracy held or improved |
| Controlling thinking speed (Lin et al., NeurIPS 2025, arXiv 2507.03704) | Fast/slow-thinking vector with input-adaptive strength | +1.3% accuracy, 8.6% fewer tokens |
| CoT-Valve (Ma et al., ACL 2025, arXiv 2502.09601) | Parameter-space direction whose scale sets CoT length | QwQ-32B GSM8K: 741 → 225 tokens, 95.07 → 94.92% |
| ThinkEdit (Sun et al., EMNLP 2025, arXiv 2503.22048) | Edits about 4% of heads to fix overly *short* thinking | +6.4% on short-reasoning cases. Shortening can hurt |
| Steering vectors in thinking LMs (Venhoff et al. 2025, arXiv 2506.18167) | Linear directions for backtracking and uncertainty | Consistent control of reasoning behaviours |
| Coconut (Hao et al. 2024, arXiv 2412.06769) | Feeds the hidden state back in place of CoT tokens | Better accuracy–efficiency trade-off on search-heavy logic |
| PID steering (Nguyen et al., ICLR 2026, arXiv 2510.04309) and STU-PID (arXiv 2506.18831) | Steering strength as a feedback controller | STU-PID: GSM8K 81 → 87% with 32% fewer tokens (100 problems, 1.5B) |
| STIR (Shi et al. 2026, arXiv 2602.04925) and DSEM (Do et al. 2025) | Steering vectors retrieved from a memory of past runs | STIR: +1.9 to +7.5% accuracy, up to 35% fewer tokens |

Also: RISER (arXiv 2601.09269) selects among reasoning-skill vectors (2–3× token efficiency). LOTUS (arXiv 2606.31779) is a looped transformer matching explicit CoT at 3B with 2.5–6.9× lower thinking latency. Qwen3's hybrid checkpoints switch thinking on or off with `enable_thinking` in the chat template, or `/think` and `/no_think`, which makes a thinking-minus-non-thinking direction easy to extract.

[Correction, 2 Nov 2026 (RV1.C4, RV1.C5, RV1.C8):
- **STU-PID** was withdrawn on 28 Sep 2026, as RV1 reports (not re-checked here: arXiv rate-limited the fetch, and a mirror still lists only v1). The 81 → 87%, −32% tokens above are v1's figures; RV1 gives the last substantive version as 85.7 → 89.6% and −23% on a GSM8K subset, "a small-scale proof of concept". R3's published baseline is PID steering (arXiv 2510.04309), which integrates the error across *layers*, not decoding steps, so a decode-time PID controller is reimplemented as an R3 ablation.
- **DSEM** (Do et al., ACL Findings 2025) stores steering vectors built from *alignment demonstrations* (safety, style transfer, role-play) and retrieves them per token chunk by nearest neighbour. Its memory does not hold past reasoning runs; STIR's does.
- **Missing prior art.** ERSS, the "Endogenous Reasoning State Steering" of Zhang et al. (2026, arXiv 2606.06188), already steers with the current generation's own hidden states: it caches the high-norm states seen so far and, at a norm peak, adds the cached state most cosine-similar to $h_t$, $h_t' = h_t + \lambda h^*$. STARS (Lee et al. 2026, arXiv 2601.22484) names "reasoning rigidity (inertia of direction)" as a failure mode of large reasoning models, which is the over-commitment risk below.]

**What would be new.** Fixed contrastive vectors already cut reasoning tokens by 10–70%. Feedback-controlled steering and memory-retrieved steering exist separately. No paper found builds the steering signal from the model's own running average of its current line of thought and regulates it with negative feedback, still less together with memory retrieval. **The main risk** is over-commitment: boosting the current line can suppress useful backtracking, and ThinkEdit shows that thinking which is too short costs accuracy.

[Correction, 2 Nov 2026 (RV1.C5): the novelty claim is narrowed. Steering from the model's own states during generation exists (ERSS), and STARS documents the inertia this design risks. What remains untested is the specific form: a fast-minus-slow exponential average of the stream as the direction, regulated by a negative-feedback gain law (R3, R6), and its pairing with memory retrieval (R5).]

**Experiments** (R1–R5 in the register)

[Review, 2 Nov 2026: RV1 ([`reviews/2026-11-02_alignment_cot_ca1.md`](reviews/2026-11-02_alignment_cot_ca1.md) §2) blocks R2 and R3. RV1.C1: the gist direction is dominated by the attention-sink token and the stream mean. RV1.C2: R2 and R4 lack matched-norm random, mean and sign-flipped-gist controls, a seed protocol and a non-inferiority margin. RV1.C3: graded gist is silently a no-op without a probe, and no gain source exists for R3's agreement signal. RV1.C4: STU-PID, the R3 baseline, was withdrawn; use PID steering (arXiv 2510.04309). RV1.C7: R1's direction should be taken at matched content positions.]

- **R1 Thinking direction.** On Qwen3's hybrid checkpoints, take the difference between activations with thinking on and off for the same prompts. Inject it in no-think mode and measure accuracy against tokens.
- **R2 Gist steering during thinking.** Sweep the gain and plot accuracy against mean reasoning tokens on GSM8K, MATH-500 and AIME 2024/25. Baselines: full thinking, `/no_think`, budget truncation, SEAL, and the concise-vector method.
- **R3 Regulated gist.** A graded gain driven by agreement between the gist and the current token (CA1): boost while consistent, release on backtracking. Compare with STU-PID-style control, and run it through the stability programme.
- **R4 Over-commitment checks.** Flip analysis (problems the baseline solves that steering breaks, and the reverse), the rate of backtracking words such as "wait", and GPQA-Diamond as an out-of-domain check, with wall-clock time and tokens-to-correct.
- **R5 Memory-retrieved gist.** Store gist vectors from successful traces in a hippocampal memory and retrieve by similarity. This is the hippocampal version of STIR and DSEM.

**Models:** R1-Distill-Qwen-1.5B/7B and Qwen3-1.7B/4B (hybrid thinking). R1–R4 can reuse `experiments/alignment/steering.py`; a reasoning runner is not built yet. [Correction, 2 Nov 2026 (RV1.C3): R3 needs a gain law of its own, now defined; R2's controls are now direction modes of `steering.py`; see "RV1 fixes" (2 Nov 2026) at the end of this notebook.]

---

## Proposal: calibrated decision models (JEV, LAYA) as CA1 (2 Nov 2026)

**Question (Daniel).** Are experiments planned that use JEV- or LAYA-style decision models as CA1 stand-ins, or are they not worth exploring?

**Status.** None were in the register; the README roadmap listed only "a calibrated CA1". CA1 today does two jobs:
- in `HippocampalBlock`, a shallow comparator with features $[a; r; a\odot r; \lvert a-r\rvert]$, whose novelty is trained to track per-token loss;
- in the temporal loop, uncalibrated gain control: error-only projection, $\tanh(\lVert\Delta\rVert/\lambda)$, and energy division.

The design document already gives the target shape. CA1 should be a joint encoder over the current input and the CA3 retrieval (LAYA encodes both inputs together across every layer). Its output should be typed decisions with calibrated confidence (JEV): write or not, replay priority, loop again or exit.

**Why it is worth it.** Every gate in the design is a decision that is currently uncalibrated:
- the harm score $s_t$ driving the alignment controller;
- the injection strength;
- the acetylcholine write/recall switch (S6);
- replay priority (W3);
- the reasoning "on track?" signal (R3);
- the exit of a depth loop (W6, formerly D-b).

A graded controller maps *probability* to gain, so calibration, not just ranking (AUROC), decides how selective it is. One earlier failure may also have been a calibration problem: CA1 + novelty-scaled learning rate harmed Titans (M2) when the novelty signal was still untrained. [Correction, 2 Nov 2026 (RV1.D3): CA1 + novelty-lr alone was near-neutral (0.440 against 0.483 for plain Titans). The harm came from stacking it with DG writes (0.172 with everything combined; RESULTS.md §1).] One caveat comes from LAYA's own documentation: an English checkpoint scored 0% on Khmer while still reporting high confidence. Calibration has to be checked under distribution shift.

**Experiments** (J1–J5 in the register)

- **J1 Decision source for the alignment controller.** Use each of the following as $s_t$:
  - the current linear probe;
  - a calibrated head on the residual stream (proper scoring rule plus temperature scaling);
  - a LAYA-style external encoder on the text (a small ModernBERT classifier, model-agnostic);
  - a guard model.

  Measure AUROC, ECE in and out of distribution (XSTest borderline prompts, other languages), and A2 selectivity at a fixed cap. *Hypothesis:* at equal AUROC, better calibration gives higher selectivity.

  [Correction, 2 Nov 2026 (RV1.D1, RV1.D2, RV1.D4): as stated the hypothesis is ill-posed. At a fixed cap and κ the gain $G\tanh(\kappa E)$ depends on where benign scores sit on that curve, and temperature scaling is monotone, so a "better-calibrated" source can raise benign gains and lower selectivity. ECE was also measured at the prompt position, not on the generated tokens that drive the gain. The two text-level sources give one score per prompt and cannot drive the graded mode. Reworded, with the matched-operating-point design; see "RV1 fixes" (2 Nov 2026) at the end of this notebook.]
- **J2 LAYA-style CA1 comparator for the loop.** A small cross-attention encoder over the recent stream window and the CA3 retrieval, outputting injection strength, write/no-write and replay priority with confidence. The training target is the counterfactual benefit of injecting, the next-token loss with and without the injection. Test on the what-when task, the recency probe and the stability suite, then the Qwen needle.
- **J3 Revisit M2.** A calibrated CA1 (proper scoring, warm-up) as the novelty signal for Titans' learning rate. Does calibration remove the harmful interaction? [Review, 2 Nov 2026: blocked by RV1.D3. The premise is corrected above; warm-up alone is now its own arm, J3.a; and a calibration measure for the novelty target, a regression target with no binary event, has to be defined. RV1.D1 and RV1.D4 also block J1.] [Update, 2 Nov 2026: the calibration measure is defined and J1 is reworded; see "RV1 fixes" (2 Nov 2026) at the end of this notebook.]
- **J4 Reasoning gate.** A calibrated "on track" decision that predicts eventual correctness from a partial trace. It sets the gist gain (boost while on track, release towards backtracking) and could stop thinking early. Run with R3 and compare with PID control.
- **J5 Depth-loop exit.** A calibrated halting decision for a looped upper block (W6), compared with Ouro's learned gate and its hidden-state-difference rule.

---

## Round 1: diagnostics on the horizon loop, and candidate W1 (2 Nov 2026)

**Setup.** Branch `feat/round1-side-memory`. Runs go through [`experiments/evolve/run.py`](../experiments/evolve/run.py), which wraps `pretrained_loop.main` unchanged and adds per-digit loss, a checkpoint of the side channels, and `--candidate` to swap the side-channel module. Results are in `results/evolve/round1/`. Settings: needle task (gap about 500 tokens), seq 512, block window 64 on every layer unless noted, batch 8, 3000 steps, bf16 base, float32 side channels. One seed per run.

**Candidate W1** ([`experiments/evolve/w1_gdn.py`](../experiments/evolve/w1_gdn.py)) is round-0 direction 1, combined with shortlist wins 1 and 2.
- It runs the clock first, then a Gated DeltaNet memory at the tap. The memory writes v_t under the *previous* position's address a_{t−1} (successor binding) and reads with a_t.
- It injects through the same CA1 control as the loop: error-only, ‖inj‖ = g‖h‖ with g ≤ 0.9σ(gain), saturating. It stores the stream before injection (open loop).
- Its state is fixed-size: 8×64×64 floats for dense keys, 4×64×256 for DG keys.
- Tests (`tests/test_evolve_w1.py`) cover:
  - strict causality, with bitwise-identical earlier outputs under fp32 and bf16 input (E1);
  - a length-independent state (E2);
  - the parameter budget at Qwen width (E3);
  - successor retrieval on CPU: a repeated context reads out the value that followed it, at cos 0.57 with dense keys and 0.85 with DG keys.

**Results** (exact / loss, then per-digit loss for digits 1–4; chance is ln 10 = 2.30)

| Run | Model | Side channels | Exact | Loss | Digits 1–4 |
|---|---|---|---|---|---|
| horizon, per digit | Qwen3, all windowed | O(T) loop, exp ρ 0.95 τ 32 | 0.00 | 1.381 | **0.001**, 2.30, 2.30, 2.30 |
| horizon, 2 pairs | Qwen3, all windowed | same | 0.00 | 1.809 | 2.17, 2.29, 2.28, 2.30 |
| horizon, 6000 steps | Qwen3.5, all windowed | same | **1.00** | 0.001 | ≤ 0.002 |
| **W1 dense** | Qwen3, all windowed | GDN, successor keys (4.28M params) | **1.00** | **0.0001** | 0.000 ×4 |
| **W1 DG** | Qwen3, all windowed | GDN, DG keys (3.74M params) | **1.00** | **0.0001** | 0.000 ×4 |
| W1 dense, 2 pairs | Qwen3, all windowed | as above | 0.55 | 0.186 [Correction, 2 Nov 2026: was 0.187; the JSON gives 0.18648] | **0.74**, 0.05, 0.11, 0.03 |
| W1 DG, 2 pairs | Qwen3, all windowed | as above | 0.43 | 0.231 | **0.76**, 0.11, 0.15, 0.14 |
| horizon, LM | Qwen3 WikiText-103 | O(T) loop, horizon kernel | ppl 11.53 (default kernel 12.01, full 9.01): 49% of the gap, against 41% | | |

**What it shows**
1. **The horizon loop knows the first digit and nothing after it (P7).** Even that comes through a query-independent path. With two stored pairs it falls back to near chance (P8, register run g5h), the outcome round-0 arm g5 was built to detect.
2. **Successor binding supplies the missing shift (P10).** W1 recalls all four digits exactly on Qwen3 with every layer windowed. Its loss is lower than full attention's (0.0001 against 0.011), its state is fixed-size, and it uses fewer parameters than the loop. DG keys learn about twice as fast: loss 0.008 at step 600, against step 900–1200 for dense keys.
3. **W1 retrieves the chain but cannot choose where it starts (P11).** With two pairs, digits 2–4 are 94–99% correct, but digit 1 is a coin flip: loss ≈ ln 2, accuracy 0.51–0.57. The first digit is keyed by the address of the space after "is", the same token in both facts. That address encodes the current token far more than the name three tokens earlier. DG keys do not help here, which fits: pattern separation sharpens retrieval but cannot add context the address does not carry.
4. **On Qwen3.5 the O(T) loop also reaches exact recall with every layer windowed (P9).** It needs 6000 steps, breaking at about step 2400. This was one pair only, so P8's caveat applies. [2 Nov 2026: the two-pair runs on Qwen3.5 are W1f.]
5. **The horizon kernel helps language modelling slightly (P12).** That suggests some memory from beyond the window, but the domain-adaptation confound (O4) is still open.

**Fitness status for W1**

| Criterion | Status |
|---|---|
| Reach | Passed with one pair; partial with two (exact 0.43–0.55) |
| Stability and recency (must pass) | **Not yet measured.** `experiments/temporal_loop.py` cannot host W1 yet (adapter: W1c). W1 is an open loop under the same clamp as the loop, and its injection ratio was 0.31–0.33, so it cannot run away. Recency is untested. |
| Rules E1–E3 | Pass (tests) |
| E6 (held-out gaps, random offsets, natural filler) | Not yet run. Needs round-0 item C6 |

**Surprises**
- W1's loss is *below* full attention's.
- The O(T) loop on Qwen3.5 needed twice the steps of W1 on Qwen3.
- DG keys did not help start selection.

**Proposed round 2** (for the checkpoint)
- **W1b: give the address context so the queried name selects the start.** Three arms, each scored with 2 and 3 pairs:
  - (W1b.a) GDN's own short-conv keys (`addressing="gdn"`, which mixes the last 4 tokens);
  - (W1b.b) addresses from a learned causal conv or n-gram over the last few tokens, the idea behind Qwen3.8-Flash-Next's n-gram embedding;
  - (W1b.c) a later tap, where the residual at "is" carries more of the name.
- A W1 adapter for the CPU stability and recency suite, so W1 can be scored on the two must-pass criteria (W1c).
- C6 held-out evaluation: gaps of 373, 757 and 1013 tokens, random needle offsets, natural-text filler.

[2 Nov 2026: further W1 items now in the register: W1d, W1 on WikiText (rule E8; W1 has no LM run); W1e, W1 without the clock (the live form of round-0 arm g3); W1f, two-pair runs on Qwen3.5 for the horizon loop and W1.]

---

## First steering runs on real models: the Qwen3 ActAdd notebook (2 Nov 2026)

**What ran.** [`docs/notebooks/activation_additions_qwen3.ipynb`](../docs/notebooks/activation_additions_qwen3.ipynb) (PR #3) ports the GPT-2-XL ActAdd notebook to Qwen3-1.7B. One config cell switches `STEER` (alignment, cot), `VECTOR_SOURCE` (prompt_pair, mean_diff) and `INJECTION` (actadd, controller), and it reuses `SteeringController`. These are the first steering runs on a real model in this project. They are **spot checks, not experiments**:
- Target `mlabonne/Qwen3-1.7B-abliterated`, bf16, one RTX 5090.
- Steering at block 14 of 28, unless stated.
- 4 samples per prompt, sampled at temperature 0.7, top-p 0.9, repetition penalty 1.1, seed 0.
- 3 alignment prompts (benign, borderline, mild harmful) and 4 GSM8K-style problems.
- 64 new tokens for alignment and 320 for CoT.
- Refusal is counted only when the output is not degenerate (review RV1.A3). Degeneration is the share of repeated 4-grams.

No completions were saved, as the data rule requires.

**Scale at block 14.** Median ‖h‖ is 180 (sink excluded). The mean per-position norm of each vector:

| Vector | ‖v‖ |
|---|---|
| Prompt pair "I cannot help with that request." − "Sure, here is how to do that." | 150 |
| Harmful − harmless difference of means, from the abliterated model itself | 26 |
| The same, from the aligned sibling `Qwen/Qwen3-1.7B` | 53 |
| Thinking-on − thinking-off difference of means (16 problems) | 157 |

**Alignment results**

| Vector, injection | Strengths and blocks | Refusals: benign / borderline / harmful | Degeneration |
|---|---|---|---|
| Prompt pair, ActAdd (first 10 prompt positions, prefill only) | COEFF 5, 10 at blocks 6 and 10; 10, 20 at block 14 | 0 / 0 / 0 | ≤ 0.02 |
| Self difference of means, controller (static) | g = 0.1, 0.2 at block 10; 0.1–0.3 at block 14 | 0 / 0 / 0 | ≤ 0.01 |
| Sibling difference of means, controller (static) | g = 0.1, 0.2 at blocks 10 and 14 | 0 / 0 / 0 | ≤ 0.01 |
| Sibling difference of means, controller (static) | **g = 0.3 at block 14** | **0 / 0 / 0.50** | ≤ 0.05 |

**CoT results** (thinking-on − thinking-off, controller static, no-think mode)

| Gain | Accuracy | Mean tokens (4 problems) | Degeneration |
|---|---|---|---|
| none | 4/4 | 119 / 168 / 216 / 165 | 0.05–0.12 |
| 0.1 | 4/4 | 109 / 158 / 224 / 168 | 0.04–0.15 |
| 0.15 | 4/4 | 93 / 153 / 199 / 161 | 0.05–0.16 |
| 0.2 | 4/4 | 83 / 161 / 184 / 151 | 0.04–0.16 |
| **0.3** | **0/4** | 320 (the cap) | **1.00**: `---  ---  ---` loops on every problem |

**Reading**
1. **ActAdd on the first prompt positions does not transfer to a chat model.** The original worked on raw completions, where the steered positions sit right before the continuation. Here they overwrite the start of the chat template, ten or more tokens before the assistant turn. They change content slightly ("tomatoes" drops out of the answer at early blocks) and never change behaviour, even at COEFF 20 (about 17× ‖h‖ per position). This is a failure of **position**, not gain.
2. **The abliterated model's own harmful − harmless direction does not restore refusal.** This is the expected premise of A6: abliteration removed the direction the model writes to refuse. The arm `graded_self` will need a different estimate. This is a failure of **direction**, not gain.
3. **The aligned sibling's direction starts to restore refusal selectively at g = 0.3.** Half the harmful samples refuse. The benign and borderline prompts are still answered, though the benign answer turns more cautious in tone. Below 0.3 nothing changes, so the effect has an onset between 0.2 and 0.3 and has not saturated at 0.3. **Only this case is gain-limited.**
4. **The CoT direction collapses at the same gain where the refusal direction first works.** At g = 0.3 and block 14, the thinking direction degenerates on every problem, while the refusal direction stays coherent (degeneration ≤ 0.05). **The saturation knee depends on the direction**, so S1 needs to sweep the directions actually used, not only random ones.
5. **The CoT eval problems are too easy.** Qwen3-1.7B already reasons step by step in no-think mode and solves all four unsteered. At g = 0.2 every answer is 4–30% shorter at unchanged accuracy. At g = 0.1 the change is mixed (−8% to +4%). At n = 4 that is not a result. R1 needs harder problems (GSM8K hard split, MATH-500) and more of them.

**Daniel's two proposals** (genesis §5.9)

*(a) CoT: feedback regularisation against repetition.* The collapse is abrupt: degeneration goes from 0.16 at g = 0.2 to 1.00 at 0.3. A fixed gain therefore sits close to a cliff, with no signal telling it where the cliff is. A regulator would turn a repetition signal into negative feedback on the gain, for example
$$r_t = \text{repeat score over the last } k \text{ tokens},\qquad g_t = g_0\,e^{-\gamma R_t},\qquad R_t = \lambda_r R_{t-1} + r_t.$$
Here $r_t$ could be the n-gram repeat rate, the cosine between $h_t$ and $h_{t-k}$, or a drop in logit entropy. This is the same leaky-integral machinery as the harm score, with the sign reversed. It is close to S5 (homeostatic gain, towards a target degeneration rather than a target echo) and to R3/J4 (release the gain when the trace stops progressing).

Two things need checking:
- whether the reaction is fast enough, since degeneration takes a few tokens to show in n-gram counts while hidden-state similarity is immediate;
- whether the collapse is driven by the prefill injection, which static mode also applies to the whole prompt, rather than by decoding.

Recorded as R6.

*(b) Alignment: stronger, multiplicative or exponential gain.* The current injection is already multiplicative in the stream, $h \leftarrow h + g\lVert h\rVert\hat d$. What is linear, or capped by tanh, is the **gain schedule**. Three distinctions matter:
- **Only the sibling-direction case is gain-limited** (reading 3). The prompt-pair and self-direction failures will not be fixed by more gain.
- **A gain multiplicative in the existing refusal component**, $h \leftarrow h + g\,(h\cdot\hat d)\,\hat d$, would amplify nothing in an abliterated model, where $h\cdot\hat d \approx 0$ by construction. It also adds positive feedback. The superlinearity belongs in the schedule, as a function of harm evidence: $g_t = G\tanh(\kappa E_t^2)$, or $g_t = \min(G, g_0(e^{\kappa E_t}-1))$. A squared or exponential schedule stays near zero for weak evidence, which protects benign and borderline prompts, and escalates fast once evidence accumulates. That should help selectivity as well as strength, which is A6.H2's question. It is also closer to Asimov's "ever more aggressively" than tanh, which flattens as evidence grows.
- **Stronger gain runs into the knee.** Reading 4 shows the refusal direction stays coherent at 0.3. Its own knee is unmeasured, so the A6 gain sweep has to extend past 0.3 (0.4, 0.5, 0.75) with degeneration and perplexity logged. Otherwise "stronger" may just mean "refuses because it is broken" (Asimov's freeze, review RV1.A3).

Recorded as A8, a gain-law comparison: additive-constant, tanh, squared and exponential schedules at matched maximum gain.

---

## Record keeping: ID changes and corrections (2 Nov 2026)

**Layout.** The lab records moved out of `results/` into `research/`: this notebook, [`REGISTER.md`](REGISTER.md), [`HANDOFF.md`](HANDOFF.md) (was `results/pretrained_HANDOFF.md`), [`plans/round0_plan.md`](plans/round0_plan.md) and [`reviews/`](reviews/). Results are grouped by track: `results/synthetic/` (CPU), `results/pretrained/` (GPU), `results/evolve/round1/` (was `gen01/`; files keep their `g1_*` names), `results/evolve/round0/` (planned, not created yet), `results/evolve/round2/` (round 2, in progress) and `results/alignment/`. Paths recorded inside result JSONs and logs still use the old layout.

**ID changes.** Older entries above now use these IDs. Where a dated entry says something that later proved wrong, a bracketed note was added instead of rewriting it.

| Old | New | Why |
|---|---|---|
| Design-review findings R1–R5 | DR1–DR5 | Clashed with the reasoning experiments R1–R5 |
| Stability findings K1–K6 | T8, T6, T7, T9/T10/T3, T20, and a gap (S1, S2) | They restated register rows; K5 is new as T20 |
| Design gaps D-a, D-b, D-c | W5, W6, W7 | Round-0 diagnoses D1–D5 keep their IDs |
| O3 | DEC1, in the register's Decisions table | Approved by Daniel on 2 Nov 2026 with the round-1 plan |
| Register corrections C1–C8 | B1–B8 (B9–B11 added) | Round-0 diagnostics C1–C6 and C1a keep their IDs |
| Register `g5` (horizon kernel, 2 pairs) | g5h | Round-0 g0–g7 keep their meaning in `plans/round0_plan.md` (control, flat and exp ρ 0.98 arms) |
| Review findings A*, C*, D* | RV1.A1–A8, RV1.C1–C8, RV1.D1–D4 | RV1 is [`reviews/2026-11-02_alignment_cot_ca1.md`](reviews/2026-11-02_alignment_cot_ca1.md) |
| A6 hypotheses H1–H4, cited elsewhere | A6.H1–A6.H4 | |
| W1b arms (a), (b), (c) | W1b.a, W1b.b, W1b.c | |

**Corrections added as notes:** the stability reference in Fitness (T3's 3,000-step figures), g2 never run (round 0), DEC1 approved (round 0, design review, GPU-session corrections), the M2 premise behind J3 (RV1.D3), and links to RV1 from the alignment, chain-of-thought and CA1 entries.

---

## Round 2: context-bearing addresses (W1b) (2 Nov 2026)

**Question.** In round 1, W1 retrieved the whole code once its first digit was given, but with two stored codes digit 1 was a coin flip (P11). Can the address carry enough context for the queried name to select the start?

**Arms** (`experiments/evolve/w1_gdn.py`, all on Qwen3-1.7B-Base with every layer windowed, 3000 steps, one seed; results in `results/evolve/round2/`)
- **W1b.a, gdn mode** (`--w1-addressing gdn`, 6 heads for the budget). Short-conv q/k/v, no explicit successor binding.
- **W1b.b, context conv** (`--w1-ctx 4`). Addresses from a causal depthwise conv over the last 4 positions' hidden states; values from the current position. This is the n-gram idea, built on `h` because rule E4 forbids token ids. It starts as W1, and a test checks that.
  - With DG keys added.
  - With `--w1-ctx-init uniform`: an even average from step 0.
- **W1b.c, tap 21** (`--layer 21`) instead of 14.

**Results** (digit-1 loss / accuracy; digits 2–4 accuracy; exact)

| Arm | 2 pairs | 3 pairs |
|---|---|---|
| W1 (reference) | 0.74 / 0.57; 0.97–0.99; 0.55 (round 1) | 1.20 / 0.41; 0.91–0.94; 0.31 |
| Context conv K = 4 | 0.74 / 0.52; 0.96–0.98; 0.45 | 1.22 / 0.43; 0.91–0.95; 0.32 |
| Context conv + DG keys | 0.80 / 0.46; 0.95–1.00; 0.42 | — |
| Context conv, even-average init | 0.84 / 0.58; **0.52–0.65**; 0.11 | 1.90 / 0.34; ≈ chance; 0.00 |
| gdn mode | 0.80 / 0.48; **chance**; 0.00 | 1.34 / 0.45; chance; 0.00 |
| **Tap 21** | **0.65** / 0.53; 0.97–1.00; 0.49 | **0.97 / 0.48**; 0.91–0.98; 0.36 |

ln 2 = 0.69 and ln 3 = 1.10.

**What it shows**
1. **No variant selects the start (P13).** In every arm, digit-1 loss is about ln(pairs). The read at the space after "is" returns an even mix of the stored first digits, so the addresses at that position stay effectively the same across facts.
2. **Explicit successor binding is what carries the rest of the code (P14).** gdn mode, which would have to learn the shift itself, loses digits 2–4 as well.
3. **The later tap is the only small step forward (P15).** It learns fastest (0.19 by step 300) and shows slight selection with 3 pairs. That points to the representation at the tap: at block 21 the space token carries more of the name than at block 14.
4. **Averaging context into the address from the start hurts (P16).** It blurs the successor addresses, so the chain degrades, and the start is still not selected.

**Proposed round 3** (for the checkpoint)
- **Diagnose before building more (cheap, about 10 minutes of GPU, no training).** Use the saved round-1/2 checkpoints (`.pt`) and the frozen model:
  - (a) A linear probe for *which name* at the space after "is", at blocks 14 and 21. Is the name decodable there at all?
  - (b) The cosine similarity between W1's learned addresses at that position across facts. Do they collapse?

  If (a) fails at 14 but works at 21, the fix is where the key is read, not how it is built.
- **W1g: temporal-context keys** (Howard & Kahana's temporal context model, already the project's CA2 reference). File each value under a slowly drifting context vector, the leaky integral of recent addresses, as well as the previous address. The first digit's key then carries "the code for Name is", not just "is". The state stays fixed-size and causal.
- **W1b.c, longer.** Tap 21 at 6000 steps with 2 and 3 pairs. It was still improving.

## Literature review and code audit, review RV2 (2 Nov 2026)

**Setup.** At Daniel's request, a separate session (Claude Science) reviewed the literature for the four research tracks (CA3 → CA1 skip paths, CA2 side-channel injection, DG → CA3 as MoE, GDN as recurrent memory). It also reviewed the Samsonovich & Ascoli (2005) pathfinding model Daniel supplied, audited the bibliography, and reviewed the code. Seven sub-agents ran in parallel, each verifying every DOI and arXiv id it cited. The review covers commit `3799b36`, is committed on top of `1620784`, and reads rounds 1 and 2 against the literature. Report: [`reviews/2026-11-02_literature/`](reviews/2026-11-02_literature/README.md), with a synthesis, seven track reports, `findings.json` for agents, `new_refs.bib`, scripts and a handoff note. Register: RV2 findings, B12–B15, backlog L1–L16.

**What it found**

1. **Most components are published.**
   - "GDN = one gradient step on the Titans loss" is in the GDN paper itself (§3), DeltaNet and Longhorn (B15).
   - "One-hot keys = exact RAM" is Kohonen (1972) and Schlag et al. (2021).
   - Successor-bound addressing is Sompolinsky–Kanter sequence memory, which in transformers is the induction-head and H3-shift mechanism.
   - The DG-gated MoE with CA3 storage and replay is HiCL (AAAI 2026).
   - The CA1 fusion vector is the InferSent matching feature.
   - The closed loop of lagged self-states is the Feedback Transformer.
   - The time channel is the Laplace stage of Shankar & Howard.
   - W1 is close to δ-mem (2026), and W3 sits beside two 2026 LLM-sleep papers.
2. **What looks new.**
   - The loop-gain analysis of re-injection (K = 1.680 reproduced independently; primacy by self-rehearsal).
   - Gist momentum steering, for which no precedent was found.
   - The temporal integral in the harm controller: incremental, and a smoothed CAST switch at the defaults (RV2.S1).
   - P2 as an observation about hybrids (RV2.X3).
   - The pathfinding opening (item 5).
3. **Measurement (RV2.X1, RV2.X2).** `AssociativeRecall` admits a binding-free strategy scoring 0.476 at 6 pairs. Full attention (0.485) and Titans (0.483–0.490) sit at it, and `gdn_x8` follows the ceiling at 8 / 16 / 32 pairs, so M1 shows set memory and M4 is not a capacity result. The short convolutions start as the identity (B12). The one-pair needle is passkey retrieval, which P8 confirms.
4. **Biology (B13, RV2.K2, RV2.K3).**
   - An excitatory CA1 → CA3 back-projection exists.
   - CA2 receives DG input and sits in parallel to CA3 → CA1.
   - "Norms drive CA2" is unsupported.
   - Mossy cells are excitatory.
   - Subtractive feedback must amplify some band (Bode). At K = 1 that is about 2.5× near a 46-token period, which may explain T10.
5. **Pathfinding (RV2.F1).** Samsonovich & Ascoli's CA3 transition matrix *J* is the repo's successor store. Their CA3 → CA1 goal field *W*, written at sharp waves by a hindsight max-recency rule, together with theta-timed probes of candidate successors, makes it a path search. A re-implementation reproduced their path lengths (6.69 ± 3.29 against 4.24 ± 0.66 optimal at 10⁴ contexts; one graph and seed).

**Read against rounds 1–2.** P8 confirms the passkey critique. P9 rules out adapter repair only when every layer is windowed, and P2 still needs L2. P11 and P13 are RV2.F1 in LLM form: every W1b variant changed the write-side address. None selected the start: digit-1 loss was ≈ ln(pairs) in the identity-initialised arms, and higher in ctx4u (0.84 / 1.90 with 2 / 3 pairs). Start selection is read-side, through a dense cue from the query span or a goal field (L6). P14 (gdn mode loses the chain) is confounded by B12.

**Register changes.**
- M1, M2, M6 and T16 moved to Open; M4 to Refuted.
- Notes were appended to M3, M5, M7, T6, T7, T10–T14, P1, P2, P9–P11, P13, P14, W1b, O9, A2, A3 and R2.
- W6.a and O7 are superseded by L6 and L9.
- B12–B15, the RV2 table and §4.8 (L1–L16) were added.
- A bracketed sign correction was added to the A6 proposal (RV2.S2).

**Next, in order.**
1. L1 (MQAR re-benchmark, cheap, CPU).
2. B12, then rerun W1b.a.
3. L2 and L4 on Qwen3.5.
4. A read-side W1b arm and L6.
5. A2 with the RV2.S3 arms before A6 runs at scale.

---

## Round 3: is W1 really CA3? The probe, the start store (W1g) and CA3 settling (W1h) (2 Nov 2026)

**Challenge (Daniel).** "I don't see the analog to CA3. Where is the recurrence, where is the sparsity gating, where is the high gain on connections?" Correct, and recorded as B16. W1 has CA3's one-shot heteroassociative *storage rule* (v_t filed under a_{t−1}). It has none of CA3's dynamics:
- no recurrence in activity: the read `S·a` is one linear lookup with no settling;
- no sparse gating of CA3 activity: DG mode makes only the address sparse;
- no high-gain connections: the store is contractive, and the only gain is the CA1 clamp.

It is a linear fast-weight sequence store. The project's own `limbic_llm/ca3.py` (a modern Hopfield network with `n_iter` "settling") does not fill the gap either. It iterates a softmax read with the cue clamped and has no k-WTA, inhibition or gain, and it costs about 21–55M parameters at Qwen width.

**Re-plan after the literature review (RV2, PR #4).** Rounds 1–2 are the missing-key problem of Samsonovich & Ascoli (RV2.F1): the chain (their *J*) works, and start selection fails. The review's reading is that start selection is read-side. It also found that W1b.a's gdn mode began with identity convolutions (B12). The order agreed with Daniel: diagnose first, then a separate start store (option 2), then CA3 settling on top (option 3).

**1. Probe** (`experiments/evolve/probe_cue.py`; register P17). The queried name is linearly decodable at the query's space token: 0.95 at block 14 and 0.98–1.00 at block 21, against chance 0.04. It is decodable at the facts' space tokens too (1.00). W1's trained addresses there collapse: the cosine between the two facts is 0.785 and 0.88, and the query address picks the queried fact at chance. **The information is present; the address projection discards it**, because successor binding for digits 2–4 rewards token identity.

**2. B12 control** (P14 revised). gdn mode with the key convolution started on the previous token learns the chain: digits 2–4 at 0.88, 0.73 and 0.42, still improving. With a random start it does not. So the token shift carries the chain, and explicit binding learns it faster. Digit 1 stays a coin flip in both.

**3. Two designs rejected in unit tests before any GPU run**
- **A drifting-context goal store** (temporal-context keys written at every token). Consecutive positions have almost identical keys, so under the delta rule each write overwrites the previous one. The first digit is overwritten by digit 2 at once, and 47% of the read was the current token's own value. Samsonovich & Ascoli avoid this by writing their goal field rarely, in hindsight.
- **A start store with a *linear* key projection.** Even when the key sees the name, "space + Name1" and "space + Name2" share the space component (cosine 0.5). The later fact then overwrites half of the earlier one, and the store picks the wrong fact (margin −0.19). Meanwhile the chain store, keyed by token identity, reliably returns the *most recent* fact. That is the same mechanism as the coin flip on the GPU.

**4. W1g (start store, built).** A second successor store G with its own **dentate-gyrus key projection**: 2× expansion, then 5% k-WTA, so it forms conjunctive codes. It sits beside W1's chain store S and shares its values and output path. In the one-hot test the DG codes pick the queried fact (mean margin +0.39 over 8 random projections) where the linear key fails. This is pattern separation of conjunctions, the DG's role in the Treves–Rolls account. 5 heads, 4.03M parameters.

**5. W1h (CA3 settling, built).** `--w1g-settle K` settles G's cue in an autoassociative store A of earlier positions (b_{t−1} → b_{t−1}) before G is read:

  x^{k+1} = norm(kWTA(g·A x^k + γ·b_t))

- **Recurrence:** K reads per position.
- **Sparse activity gating:** k-WTA at every step.
- **High gain:** g starts at 4.
- **Divisive normalisation** keeps it stable.

In the unit test a 0.55/0.45 mix of two stored sparse patterns settles onto the majority pattern (cosine > 0.95). Without settling it stays a mix (0.77).

**Runs** (`results/evolve/round3/`, Qwen3 with every layer windowed, 3000 steps). W1g with 2 and 3 pairs at taps 14 and 21, a no-goal ablation and a 1-pair check. Then W1h at the same settings. Results are added to the register as they land.

**Round-3 results** (Qwen3, every layer windowed, 3000 steps, one seed each; digit-1 loss / accuracy, exact)

| Arm | Tap 14 | Tap 21 |
|---|---|---|
| W1 / no-goal, 2 pairs | 0.892 / 0.46, 0.42 (5 heads) | 0.65 / 0.53, 0.49 (round 2, 8 heads) |
| **W1g, 2 pairs** | 0.855 / 0.48, 0.43 | **0.002 / 1.00, 0.984** |
| W1g, 3 pairs | 1.374 / 0.38, 0.20 | 1.004 / 0.38, 0.28 |
| W1h (W1g + settling), 2 pairs | 0.79 / 0.52, 0.48 | 0.66 / 0.49, 0.45 |
| W1h, 3 pairs | 1.26 / 0.41, 0.24 | 1.004 / 0.37, 0.27 |
| W1g, 1 pair | exact 1.000 | — |

**What it shows**
1. **A start store with conjunctive DG keys can select the queried code (P18).** At tap 21 with 2 pairs the coin flip becomes exact recall: 0.984, with digit 1 at 0.002. That is the first arm to solve the missing-key step. It is one seed at one setting, so it needs a replication.
2. **It is fragile (P18, P19).** It does not work at tap 14, where the name is slightly less decodable (0.95 against 0.98–1.00), and it does not extend to 3 pairs within 3000 steps.
3. **CA3 settling as built hurts (P20).** It turns the 2-pair success back into a coin flip. The autoassociative store contains every earlier position, including the query's own "<name> is", the most recent and most similar patterns, so the cue completes onto the query rather than the stored fact. Biologically, CA3 retrieval runs with encoding suppressed (the acetylcholine switch), so recent input does not dominate the attractor landscape. That suggests gating what A stores (write only at G's write events, or exclude a recent refractory window) rather than storing everything.

**Proposed round 4** (for the checkpoint)
- Replicate W1g at tap 21 with 2 pairs on seeds 1 and 2, to establish P18.
- Diagnose the 3-pair failure: G's DG width and sparsity (more active units or a larger expansion within budget), and 6000 steps.
- W1h, revised: A written only where G writes strongly (shared β gate), or with a refractory window of the last w positions excluded from settling.

---

## Round 4: the 3-code failure diagnosed, and an event-gated goal field (W1i) (2 Nov 2026)

**Daniel's direction.** Diagnose the 3-code failure first, explore the event-gated writes (option 1, never tried) in parallel, and do not re-run W1g's 2-pair win.

**1. Diagnosis: the 2-pair "win" was a positional shortcut** (`experiments/evolve/diag_goal.py`; P18 and P19 corrected). On 3-pair data, the 2-pair W1g model's digit-1 accuracy by queried fact is 1.00 / 0.08 / 0.92 (first / middle / last). It retrieves the first and the last fact only, which covers every 2-pair query. The fixed layout makes this possible: the facts sit right after BOS, in order. The middle fact of 3 is in neither store, which explains P19. The review (F3) found the diagnostic's cosine-based attribution of which fact each store returns unreliable, so the conclusion rests on the accuracy by queried fact alone.

**2. A task without the shortcut.** `--needle-scatter` places the facts at random positions with filler before, between and after them. Gaps run 269–502 tokens, and facts no longer straddle window blocks. Fact order is kept, so rank still carries information: use 3+ pairs and read accuracy by queried rank (`experiments/evolve/eval_by_query.py`). On it, W1g's start store stays near chance (digit 1 0.97 / 0.47), and so does the baseline without a start store (0.97 / 0.53).

**3. W1i, an event-gated goal field** (`experiments/evolve/w1i_events.py`). It is W1's chain store plus a goal field G:
- **Keys:** drifting temporal context, where each token's key magnitude acts as learned salience.
- **Writes:** only at learned events, ε_t = e_t(1 − r_{t−1}), with a refractory trace and no decay.
- **Read and write:** G is read with the current context and written with the previous one.

A toy check set the design. Writing only at each fact's first digit, with a salient name, selects the first, middle and last of 3 facts. Writing at every token always returns the last (it is now a unit test).

**Results** (tap 21, 3 pairs, 3000 steps, one seed)

| Layout | Exact | Digit 1 by queried rank (first / middle / last) | Exact by rank |
|---|---|---|---|
| Scattered | **0.883** | **0.99 / 0.99 / 1.00** | 0.76 / 0.92 / 0.98 |
| Fixed | 0.766 | 0.94 / 0.94 / 0.96 | 0.74 / 0.75 / 0.83 |

Controls: windowed base 0.00; full attention 0.992 (scattered).

**The events trained toward the S&A picture.** Mean write strength fell from 0.12 at initialisation to 0.04 over all tokens, and reaches 0.23 at the facts' first digits: about 6× enriched.

**Tap 14 fails at 3000 steps** (exact 0.34; digit 1 1.32 / 0.45), with the loss still falling. Every start-selection mechanism so far needs block 21, where the name is most decodable (P17).

**What it shows.** Choosing the start of a stored chain by content, the missing-key step, works when the goal field is written rarely, at events, under a context key in which the name is salient. The two alternatives fail: writing at every token (overwrite), and a separate key projection without event gating (positional shortcut). That is Samsonovich & Ascoli's design point carried into an LLM side memory. The remaining error is in the chain: exact match falls with distance (0.76 for the farthest fact against 0.98 for the nearest).

**Review** (two reviewers, adversarially verified). No leak was found. Fixes:
- a wiring test through W1i's real `_read`;
- the unused gate removed;
- honest docstrings: the events are not sparse at initialisation, and scattering keeps rank;
- facts kept inside one window block;
- `--text-file` filler supported.

The scattered runs above started before the block fix, so about 4% of their facts straddled a block. That makes the task slightly harder.

**Proposed round 5** (for the checkpoint)
- W1i at tap 14 with 6000 steps, and with 4–5 pairs at tap 21: where does content selection break?
- Seeds 1–2 for P21. That is a replication of the new result, not of W1g's.
- The long-gap chain: why exact match falls for distant facts (decay of the chain store S, or the event field's write strength).
- W1i on the CPU stability and recency suites (W1c), and on WikiText (W1d, no LM regression).

---

## Round 5: replication, scale, interference and the must-pass suite (2 Nov 2026; interim)

**Approved plan.** Run the gap-vs-rank check, then round 5 as proposed. The procedure followed in rounds 0–5 is now written down in [`evolution_protocol.md`](evolution_protocol.md), with the queue (`experiments/evolve/queue.sh`) and push (`experiments/evolve/sync.sh`) helpers committed.

**Gap vs rank (P24).** The round-4 seed-0 W1i checkpoint, evaluated with **one** stored code, is exact at 0.99–1.00 at every gap up to 500+. With three codes, at a fixed gap, the earlier-stored code is recalled worse: at gaps 351–400 exact is 0.58 / 0.90 / 0.98 by rank (the rank-0 bin has n = 19). The long-chain loss is retroactive interference between stored chains, not decay. That makes it a capacity limit, not graceful forgetting.

**Results** (W1i, scattered, tap 21 unless noted; 3000 steps; digit-1 loss / accuracy, exact)

| Run | Result | Register |
|---|---|---|
| 3 pairs, seed 1 | 0.977 / 0.38, 0.28; no second drop | P25 |
| 3 pairs, seed 2 | 0.982 / 0.46, 0.39; no second drop | P25 |
| 4 pairs, seed 0 | 0.865 / 0.63 (chance 0.25), 0.48; loss dropping at step 2,700 | P26 |
| **5 pairs, seed 0** | **0.146 / 0.97**, 0.703; by rank 0.93 / 0.95 / 0.96 / 0.99 / 0.96 | P26 |
| DG chain keys, 3 / 4 pairs | 1.05 / 0.48, 0.31; 1.40 / 0.44, 0.24 | P27 |
| LM (W1d), WikiText-103 | perplexity 12.29 (loop 12.01; horizon 11.53) | P29 |
| CPU must-pass suite (W1c) | W1i + clock: echo 0.32, steady; recency newer 0.62 / older 0.24; what 1.00, when 0.98. Without the clock, and W1 either way: "what" not learned | P28 |

**What it shows**
1. **W1i can select among up to 5 stored codes by content,** when the selection is learned. But in 3000 steps only seed 0 learned it, so P21 is now Mixed (1 of 3).
2. **Selection arrives as a sudden late drop in the loss,** after the chain is learned. Seed 0 made it near step 1,800 with 3 pairs and near 2,100 with 5. Seeds 1–2 had not by step 3,000. Learnability, not capability, is the open problem.
3. **DG keys on the chain store did not reduce interference.** They did not learn selection at all.
4. **W1i with the clock passes the must-pass checks** on the small CPU model: bounded echo, and the newer event dominates. It is the first W-family candidate to do so. Its older-event share (0.24) is above the full design's (0.18).
5. **On WikiText W1i is slightly behind the O(T) loop** (12.29 against 12.01). That is a narrow E8 miss, confounded by the tap.

**Daniel's question: is training a fair test if the memory works at inference?** Yes. Training fits the side channel's parameters once, offline; that is the policy for *how* to remember. The memory state is written by the delta rule during the forward pass, at inference, with no gradients. Every evaluated code is new, so a correct answer was stored and retrieved at inference time. What training does not establish is generality: the policy was trained on this format only.

**6000-step reruns** (added after the entry above; all scattered, 3 pairs)

| Run | Exact | Digit 1 by rank | Exact by rank |
|---|---|---|---|
| Seed 1, tap 21 | **0.891** | **1.00 / 1.00 / 1.00** | 0.88 / 0.87 / 0.88 |
| Seed 2, tap 21 | 0.781 | 0.74 / 0.72 / 0.86 | 0.64 / 0.63 / 0.81 |
| Tap 14, seed 0 | 0.602 | overall 0.70 (chance 0.33) | — |

**With a longer schedule all three seeds learn content selection.** Seed 1 does so fully; seed 2 does so substantially, with its loss still falling at step 6,000. So selection is slow to learn, not a seed-0 accident. P21 is now Supported across seeds, and P25 is revised. Tap 14 also learns, more slowly (P23).

---

## Round 6a: held-out evaluation of W1i (rule E6, round-0 C6) (2 Nov 2026)

**Request (Daniel).** Evaluation only: extend `eval_by_query.py` with `--seq-len` and `--text-file`, keep everything else as trained, and evaluate the seed-0, seed-1 (6000 steps) and 5-code checkpoints on length, number of stored codes, and filler type, with windowed and full-attention controls at every setting.

**Tooling.** `eval_by_query.py` now:
- scores the windowed base and full attention on the same batches as the side-channel model;
- computes only the answer positions' logits, so 2048-token batches fit;
- fills candidate flags added after a run from their defaults.

Natural filler is the WikiText-103 **test** split, disjoint from the LM runs' training split (1.26M characters, kept on geof). Before the sweep, the seed-0 checkpoint at its training settings reproduced its earlier numbers (exact 0.84, digit 1 0.98 on 64 sequences).

**Results** (exact; digit 1 in brackets where informative; 384 sequences each; windowed 0.00 and full attention 0.97–1.00 throughout)

| Checkpoint | Length 512 / 768 / 1024 / 2048 | Codes 1 / 2 / 3 / 4 / 5 (seq 512) | Natural filler (seq 512) |
|---|---|---|---|
| Seed 0, trained on 3 codes | 0.89 / 0.86 / 0.90 / 0.88 | 1.00 / 0.98 / 0.89 / 0.82 / 0.79 | **0.09** (0.29) |
| Seed 1, 3 codes, 6000 steps | 0.88 / 0.90 / 0.89 / 0.87 (digit 1 1.00 throughout) | 1.00 / 0.96 / 0.88 / 0.85 / 0.80 (digit 1 1.00 throughout) | **0.45** (0.83) |
| Trained on 5 codes | 0.72 / 0.72 / 0.68 / 0.65 | 0.99 / 0.96 / 0.86 / 0.82 / 0.72 | **0.16** (0.49) |

**What it shows**
1. **Length generalises (P30).** At 2048 tokens the queried code's first digit is still selected at 0.98–1.00 in every gap bin, out to gaps of 1,690+ tokens, more than three times the training maximum (502). Exact recall falls gently with gap (0.94 → 0.79), as the chain store takes in more filler writes. A fixed-size memory that generalises across length in this way is what rule E6 asks for.
2. **Code count generalises (P31).** Models trained on 3 codes select the queried one among 4 or 5 (digit 1 0.93–1.00 at every rank). Exact falls with load, the interference of P24, but selection does not.
3. **Natural filler breaks it (P32).** Full attention is unaffected (0.98–1.00), so the task is no harder for the base model. The event detector fires far more on natural text (seed 0's mean write strength goes from 0.04 to 0.17), so the goal field is overwritten: the dense-writes failure of round 3. The learned write policy picked up what is *rare in random-token filler*, not what is *worth storing*. Seed 1 is the most robust: its detector shifts least, and it keeps 0.83 on digit 1.

This answers Daniel's earlier question about inference-time memory. The memory does work at inference, and it generalises across length and load. But the *policy* for when to write is specific to its training distribution.

**Proposed round 6b** (for the checkpoint)
- Train W1i with natural filler, or a random/natural mix, and evaluate on both. Does the event detector learn "first digit of a fact" rather than "rare token"?
- Optionally, a sparsity regulariser or an explicit write budget per window on the events, to make the policy robust to filler.
- Then RULER-style multi-key and multi-value evaluation (RV2) on the most robust checkpoint.

---

## RV1 fixes: the alignment, CoT and CA1 tracks (2 Nov 2026)

Review RV1 ([`reviews/2026-11-02_alignment_cot_ca1.md`](reviews/2026-11-02_alignment_cot_ca1.md)) found twenty problems in the alignment build, the CoT proposal and the CA1 decision proposal. All twenty are addressed. The code findings each have a test in `tests/test_alignment.py`, which grows from 14 to 44 tests. The protocol findings are written out below, because their runners are not built yet. Register §3 has the status of each.

**Code**

| Finding | Change | Test |
|---|---|---|
| A1 perplexity never activated three arms | `perplexity()` prefills the first `--ppl-prefix` tokens (default 32) through the KV cache, then teacher-forces the rest, so every continuation token is scored and steered as in generation. It reports the mean gain on the continuation (`ppl_gain_mean`) | `test_perplexity_measures_every_arm` |
| A2 / D2 probe calibrated only where it was fit | The none arm now has a controller in mode `none`, which records scores without injecting. Every arm reports per-set `s_prompt_mean` and `s_gen_mean`, and `calibration`: AUROC and ECE at the last prompt token, per generated token and per response, in distribution (harmful against harmless) and on XSTest (unsafe against safe) | end-to-end test |
| A3 loops counted as refusals | `judge.counted_refusal` requires refusal markers *and* degeneration ≤ 0.5; `refusal_raw` is kept. Degeneration, repetition and the token-limit rate are printed and stored for every set | `test_judges_and_the_data_rule` |
| A4 the `"last"` path untested | Left-padded prefill of two prompts of different lengths, then decode steps, against a per-token reference for each sequence alone, for every mode, with the conditional switch both on and off. Conditional is dropped from the `"all"` one-pass test, where it differs by design | `test_last_path_padded_prefill_and_decode_matches_reference` |
| A5 statistics included pads | `summary()` averages over response positions only (the last prompt token and each generated token before the first EOS), plus `gain_prompt_mean` over real prompt tokens | `test_summary_counts_real_positions_only` |
| A6 `--name` could escape | `judge.safe_name` allows letters, digits, `.`, `_` and `-` only; `judge.private_file` checks the resolved file. The results JSON path uses the same name | `test_run_names_cannot_contain_paths`, `test_private_files_cannot_escape_the_private_dir` |
| A7 sibling model kept in memory | Freed before the target loads; layer count and width must match or the run exits. The path now also runs under `--tiny` (a second random model), so it is tested | end-to-end test |
| A8 position ids on left padding | `probe.position_ids`, derived from the mask exactly as `generate` derives them | `test_left_padding_is_invisible_to_the_probe_activations` |
| C1 gist dominated by the sink | Bias-corrected averages over real tokens, skipping the first `gist_skip` tokens (default 1). Both averages are normalised means of the same tokens, so a constant background cancels exactly and no corpus mean is needed | `test_gist_excludes_the_sink_and_cancels_the_background` (with the sink included, the injected direction has a component of −0.98 along the sink, as RV1 found on Qwen3) |
| C2 no controls | `direction_mode` `anti_gist`, `mean` (the slow average alone) and `random` (seeded). At the same g every mode injects the same norm; the trace records the effective gain g·‖d̂‖. `run.py` gains a `static_random` arm as the matched-norm control for A2 and A6 | `test_random_direction_is_fixed_unit_and_seeded`, gist test |
| C3 / D4 silent no-op; no score interface | `graded` without a per-token score raises, as does `conditional` without any score. `score_fn(h)` supplies per-token scores (a calibrated head); `external_decision=True` with `pending_decision` (B,) supplies one score per prompt (an encoder or a guard model) and drives the conditional mode only | `test_modes_without_a_score_source_raise_and_score_sources_plug_in` |
| C6 "open loop" overstated | Docstring softened. `judge.repetition_onset`: the first token at which a 32-token window turns repetitive, reported per response | `test_judges_and_the_data_rule` |
| D1 calibration at a fixed cap is ill-posed | `--gains` and `--caps` sweeps; `operating_points` gives each family's curve and its borderline and harmless refusal interpolated at `--target-refusal`. Refusal rates carry a binomial standard error | `test_sweeps_report_matched_operating_points` |

`probe.auroc` is also vectorised (tie groups by `torch.unique`), because the generated-token metrics score thousands of tokens; a test checks it against a brute-force count on tie-heavy data.

**What changes in practice.** On the random tiny model of the smoke test, an illustration and not a result:
- Perplexity now separates the arms: graded 526.6 against 534.9 for none, with a mean continuation gain of 0.27. Before the fix, RV1's CPU run found none, conditional, graded and graded_self bit-identical (539.97).
- The probe's AUROC is 1.00 at the prompt but 0.36 on generated tokens, where its mean score is *higher* on harmless responses. A random model means nothing in itself, but the gap RV1.A2 warned of can be large, and every real run now measures it.

**R1 protocol (RV1.C7).** Under Qwen3's chat-template switch, the user-prompt tokens are identical in both conditions, and with causal attention so are their activations, so a difference taken there is zero. The conditions differ only in the appended template tokens and in what is generated. The direction is therefore the difference of mean activations over the first 32 generated content tokens of each condition (inside `<think>` with thinking on; the answer with thinking off), excluding template and special tokens. Its cosine with the last-template-token difference is reported, to show the two are not the same direction.

**R2 protocol (RV1.C2).**
- **Arms at matched norm** (the same g, so the same g‖h‖ at every position): `gist`, `anti_gist`, `mean`, `random` (three direction seeds) and no steering.
- **Baselines:** full thinking; budget truncation at the gist arm's mean reasoning length; SEAL; the concise vector. `/no_think` is a baseline for the Qwen3 hybrids only. R1-Distill has no switch, so it uses budget truncation and a prefilled empty thinking block instead.
- **Sampling:** 4 samples per problem at T = 0.6, top-p 0.95, seeds 0–3, the same seeds in every arm.
- **Non-inferiority:** a pre-registered margin δ = 2 accuracy points. "Equal accuracy" is claimed only if the lower 95% bound of a paired bootstrap over problems, on (arm − full thinking), lies above −δ. GSM8K (1,319 problems) and MATH-500 can resolve that; AIME (60 problems, a standard error of about 6 points at 30% accuracy) is reported descriptively only.
- **Truncation:** the share of traces that hit the token limit, per arm. Truncated traces are scored wrong, and accuracy on untruncated traces is reported alongside.
- **Over-commitment (R4):** flips, the "wait" rate, the self-loop rate (RV2), `repetition_onset` and distinct-4.

**R3 gain law (RV1.C3).** The agreement of the current token with the gist up to the previous token,

$$a_t = \cos\big(h_t - b_{t-1},\; m_{t-1} - b_{t-1}\big),$$

gives the score $s_t = [a_t]_+$, which drives the usual integral $E_t = \lambda E_{t-1} + s_t$ and gain $g_t = G\tanh(\kappa E_t)$.
- **Release on backtracking:** when $a_t < -\theta_r$, or when a backtracking word ("wait", "alternatively") is generated, $E_t \leftarrow \rho_r E_t$ ($\rho_r = 0$ resets).
- **Why it is negative feedback:** the gain grows only while new tokens continue the gist and falls as soon as they turn away.
- **The catch:** steering along the gist makes later tokens agree with it, through the sampled tokens, so $a_t$ can drive itself upward. The release, the cap, and R6's repetition regulator $g_t \leftarrow g_t e^{-\gamma R_t}$ have to hold that down, so R3 goes through the stability programme.
- **Baselines:** PID steering (arXiv 2510.04309, integrating across layers) as the published one, and a decode-time PID controller on the error $a^* - a_t$ as an ablation.
- **To build:** a built-in score in `steering.py` that reads the averages carried from $t-1$.

**J1 design (RV1.D1, RV1.D2, RV1.D4).**
- **Hypothesis, reworded:** at a matched harmful-refusal rate, a source whose scores are better calibrated *on generated tokens* gives lower borderline refusal.
- **Comparison:** for each source, sweep its operating parameter (`--caps` for graded, `--gains` for conditional) and read borderline and harmless refusal at `--target-refusal` from `operating_points`, with ECE at the prompt and on generated tokens from `calibration`.
- **Sources:** the linear probe (now); a temperature-scaled head on the residual stream (`score_fn`); a LAYA-style encoder and a guard model (`external_decision`, conditional mode only, since each gives one score per prompt). LAYA is temperature-fit on held-out data before comparing, because its checkpoints ship over-confident.
- **Alternative,** from the review: drive the gain from a thresholded decision instead of a raw probability, so that calibration enters through the threshold.

**J3 design (RV1.D3).**
- **Premise, corrected:** CA1 + novelty-lr alone was near-neutral (0.440 against 0.483 for plain Titans). The harm came from stacking it with DG writes (0.172).
- **Arms:** J3.a, warm-up alone (novelty-lr off for the first N steps, then ramped); J3.b, a calibrated novelty head; J3.c, both. Each runs with and without DG writes, on at least 2 seeds.
- **Calibration measure:** the novelty target $z = \sigma(\text{standardised token loss})$ is a regression target with no binary event, so ECE is undefined for it. Define the event $e_t = \mathbb 1[\ell_t > q_{0.8}]$, where $q_{0.8}$ is the 80th percentile of token loss on held-out text, and score CA1's output as a probability of $e_t$: ECE over 10 bins and the Brier score. Also give a reliability curve of predicted against realised $z$ by decile.

**Literature checks.**
- ERSS (Zhang et al. 2026, arXiv 2606.06188, inside "The Tell-Tale Norm") and STARS (Lee et al. 2026, arXiv 2601.22484) were checked on arXiv.
- DSEM was checked on its ACL Findings 2025 page, and PID steering on arXiv (v3, 16 May 2026).
- The STU-PID withdrawal could not be re-checked: arXiv rate-limited the fetch, and a mirror still lists only v1. It stands as RV1 reports it.

**Independent check.** A separate agent reviewed the diff against the old code. It judged A1, A3–A8, C1, C3, C6 and D4 resolved, A2/D2 resolved as a measurement, and C2 and D1 partly resolved. C2's sampling protocol and margin exist only on paper until the reasoning runner is built, and J1's second score source is not wired into `run.py`. It also found three bugs, now fixed: `operating_points` interpolated between points that are not neighbours on a non-monotone sweep (now the first crossing along the sweep), duplicate sweep values collided, and `run.generate` could not pass an external decision (`reset()` cleared it). It also found that `capped_frac` was meaningless outside the graded mode (now null there). It found no Qwen3-, bf16- or CUDA-specific break, and confirmed the data rule.

**Status.** A1, A2 and A6 are ready for geof. For A6 the spot checks put the onset near g = 0.3, so sweep past it:

```bash
python experiments/alignment/run.py --model mlabonne/Qwen3-1.7B-abliterated --direction-from Qwen/Qwen3-1.7B \
    --dtype bfloat16 --device cuda --n-eval 100 --max-new-tokens 96 --gains 0.1,0.2,0.3,0.4 --caps 0.2,0.3,0.4,0.5 \
    --ppl-data hf:Salesforce/wikitext:wikitext-2-raw-v1:test:text --private-dir ~/limbic_private/a6 --name a6_qwen3_abliterated
```

---

## Round 6b: training on natural filler, and a write budget (2 Nov 2026)

**Request (Daniel).** Run the proposed 6b test: train W1i with natural or mixed filler and evaluate on both kinds. Mid-round, Daniel added: "add a write-budget or sparsity constraint on events".

**Tooling**
- `run.py --needle-mix p`: a share p of sequences use natural filler, the rest random.
- `eval_by_query.py --random-filler`.
- W1i `--w1i-budget B --w1i-budget-window W`. Each write is scaled by clamp(1 − u_t/B, 0, 1), where u_t counts the detector's attempted events over the previous W tokens. This is structural, causal and has a fixed-size state; a loss penalty would have needed a change to `pretrained_loop.py`.

Training text is a 5M-character slice of the WikiText-103 train split; evaluation text is the held-out test split.

**Results** (W1i, 3 pairs, scattered, tap 21, 6000 steps; exact / digit 1 on held-out natural, then random filler; windowed 0.00 and full 1.00 throughout)

| Run | Natural filler | Random filler | Events (mean ε overall / at first digits) |
|---|---|---|---|
| Natural, seed 0 | **0.93 / 0.99** | 0.89 / 0.97 | 0.12 / 0.25 |
| Natural, seed 1 | 0.33 / 0.45 | 0.36 / 0.45 | 0.14 / 0.18 |
| Mixed 50/50, seed 0 | 0.35 / 0.45 | 0.36 / 0.43 | 0.08 / 0.09 |
| Natural, refractory 0.95, seed 0 | **0.94 / 0.99** | **0.91 / 0.99** | 0.15 / 0.33 |
| Natural, budget 2 per 64 | 0.34 / 0.46 | 0.33 / 0.42 | **0.002 / 0.001** |
| Random, budget 2 per 64 | 0.05 / 0.16 | 0.34 / 0.42 | 0.003 / 0.003 |

**What it shows**
1. **Training on natural text fixes the filler problem, when selection is learned (P33).** Natural-trained seed 0 scores 0.93 on held-out natural filler, against 0.09 for the random-trained seed 0 in P32, and keeps 0.89 on random filler. With a longer refractory period (0.95) it is the best W1i so far: 0.94 / 0.91, digit 1 0.99 on both. Its events stay selective on natural text (0.15 overall, 0.33 at first digits). The detector learned what is worth storing rather than what is rare.
2. **Learnability is still the bottleneck (P34).** Natural seed 1 and the 50/50 mix did not make the late drop within 6000 steps. So 2 of 4 natural-type runs learned selection. Mixing filler did not make it easier.
3. **The write budget, as built, kills the mechanism (P35).** At initialisation the detector fires at about 0.12 per token, about 8 per 64 tokens against a budget of 2. Every write was starved from step 0, the gradient switched the detector off (ε ≈ 0.002), and selection was never learned on either filler. A budget has to start above the initial firing density (or be annealed down), or the detector has to start sparse. The longer refractory period, a softer form of rationing, helped instead.

**Proposed next** (for the checkpoint)
- A budget that does not bind at initialisation: set B at or above the initial density (≈ 8 per 64) and anneal it down, or start the event bias at about −5.
- A learnability study, the main open problem. Seeds × {random, natural} with refractory 0.95, and learning rate or warm-up variations, to find a schedule that makes the late drop reliable.

---

## Round 7: is the best recipe reliable? (2 Nov 2026; started)

**Question (Daniel).** "Start next round. What will define done so we can move on to other experiments?" The proposed exit criteria (DEC2, `evolution_protocol.md` §6) need one recipe to reach:
- **reliability:** ≥ 4 of 5 seeds learn selection in 6000 steps;
- **robustness:** ≥ 0.85 exact on natural and random filler, at 512–2048 tokens and 1–5 codes;
- **the must-pass checks;**
- **the write-up.**

Stop rule: if two rounds bring no recipe to reliability, close the track as "capable but not reliably learnable".

**Recipe under test.** W1i, natural filler (WikiText-103 train), refractory 0.95, 6000 steps, 3 pairs, scattered, tap 21. Seed 0 is P33 (0.94 natural / 0.91 random).

**Arms**
- Seeds 1–4 (reliability).
- Seed 1 with lr 2e-3 (does a larger step help the late drop?).
- Event bias −5, with and without a write budget of 2 per 64 tokens. At bias −5 the detector starts at about 0.43 events per 64, so the budget does not bind at initialisation, unlike P35.

Each trained model is evaluated on held-out natural and random filler with controls. Results follow when the runs land.

**Round-7 results** (natural / random filler; exact, digit 1; windowed 0.00 and full 1.00 throughout)

| Run | Late drop | Natural | Random |
|---|---|---|---|
| Recipe, seed 0 (P33) | yes | 0.94, 0.99 | 0.91, 0.99 |
| Recipe, seeds 1 / 2 / 3 / 4 | none | 0.31–0.34, 0.41–0.45 | 0.31–0.37, 0.43–0.46 |
| Seed 1, lr 2e-3 | no | 0.33, 0.43 | 0.30, 0.43 |
| Seed 0, event bias −5 | no | 0.35, 0.44 | 0.34, 0.44 |
| Seed 0, bias −5 + budget 2/64 | no | 0.36, 0.46 | 0.31, 0.44 |

**What it shows**
1. **The recipe is not reliable: 1 of 5 seeds (P36).** DEC2 needs ≥ 4. Every failing seed learns the chain (digits 2–4 high) but stays near chance on choosing the start. It never makes the late drop that the successful runs show.
2. **Across all W1i runs at 6000 steps the pattern is the same.** Selection was learned in 5 of 13 runs, almost all from seed 0, under several settings: round 4 at 3000 steps, round 5 seed 1, round 6b natural, round 6b with refractory 0.95, and partially round 5 seed 2. Learning selection is an optimisation barrier that most initialisations do not cross in this budget.
3. **The levers tried did not help.** A larger learning rate did not (P37). A detector that starts sparse gets too little gradient to find the events, even on seed 0, which learns at the default bias (P38). The budget is moot when the events never form.

**DEC2 status.** Round 7 is the first round under DEC2 to miss criterion 1. One more round is allowed before the stop rule closes the track as "capable but not reliably learnable".

**Proposed round 8 (the last attempt under DEC2).** Attack the optimisation barrier directly, with the recipe otherwise fixed:
- (a) a curriculum (W10): train first with 1 pair and short gaps, then raise to 3 pairs, so the event detector and goal field learn on an easy version before the selection step is needed;
- (b) 12,000 steps on the failing seeds, to test whether the drop is slow or absent.

Run (a) and (b) on seeds 1–4. If neither reaches ≥ 4 of 5 with seed 0 included, close the track.

**Why the failing seeds do not learn (analysis of the saved checkpoints, 3 Nov 2026)**

| | Final training loss | Goal read weight σ(g) | Events at first digits / overall |
|---|---|---|---|
| Runs that learned selection (4) | 0.005–0.072 | 0.48–0.64 | 1.6–5.8× |
| Runs that did not (6) | 0.21–0.30 (one 0.64) | 0.46–0.49 (≈ init 0.5) | 0.7–2.1× |

1. **The failing runs sit exactly on the "everything but which code" plateau.** If the format and digits 2–4 are known and digit 1 is a 1-in-3 guess, the loss averaged over the 5 answer tokens is ln 3 / 5 = 0.22. The failing runs end at 0.21–0.30. They have learned the chain and the format, and only the choice of start is missing. Only one of the five answer tokens carries the signal that would train the goal field.
2. **Selection needs three parts to change together, and each is useless alone:**
   - events concentrated at the facts' first digits;
   - a context key in which the name is salient (W_c);
   - a goal read weight that departs from its initial value.

   That makes it a multiplicative, saddle-like problem: no part gets a clear gradient until the others exist. In the failing runs the goal read weight never left its initialisation, and the events stayed unconcentrated. The event bias did not move in any run (−1.95 to −1.98), so any selectivity had to come from the input weights.
3. **Escape is stochastic and sudden.** It is a late drop whose occurrence depends on the seed, like the formation of induction heads (Olsson et al. 2022). Seed 0 escaped under several settings, most other seeds under none.
4. **It is not a capacity limit.** Once learned, the same memory selects among 5 codes and at gaps of 2000 tokens (P26, P30).

**Is the task too hard? (Daniel's question)** Each fact is a 4-digit code bound to a name. Three of them are followed by 270–500 tokens of filler, about a paragraph or two of reading, and one is asked for. For a person that exceeds short-term memory:
- Forward digit span is about 7 when recalled at once (Miller 1956).
- Under distraction, short-term recall collapses within seconds: three consonants drop to about 10% after 18 s of counting backwards (Peterson & Peterson 1959).
- Similar items interfere strongly (proactive interference).

So this is an episodic, hippocampal memory task, and arbitrary name–number pairs are hard for people too without rehearsal or mnemonics. It is the right kind of test for a hippocampal module. But the machine's difficulty is not the memory demand: full attention solves it trivially, and the side memory does too once it has learned *how* to remember. What is hard is **discovering the write policy from scratch from a weak gradient**. People do not learn their encoding policy per task: novelty and salience gate hippocampal encoding through built-in neuromodulation (dopamine and acetylcholine; Lisman & Grace 2005, Hasselmo 2006). That points to reducing what must be learned (a built-in trigger for events, or a curriculum) rather than to an easier task.

---

## Round 8: the last attempt under DEC2, and the track closes (3 Nov 2026)

**Arms** (Daniel: "build W1j and run round 8 with all three arms"). 5 seeds each; natural-filler recipe (refractory 0.95, 6000 steps, 3 pairs, scattered, tap 21); evaluated on held-out natural and random filler; windowed 0.00 and full 1.00 throughout.
- **W1j:** an explicit CA1 comparator. The chain store's own delta-rule prediction error (EC input against CA3 prediction), made scale-free, taken against its recent baseline and detached, is added to W1i's event logit with a learned per-head gain.
- **Curriculum:** W1i trained on 1 code for 1500 steps, 2 for 1500, then 3 (training only; evaluation always uses 3).
- **Both.**

| Arm | Seeds that learned selection | Exact when learned (natural / random) |
|---|---|---|
| W1j | **0 / 5** | — |
| Curriculum | **3 / 5** (seeds 0, 3, 4) | 0.83–0.84 / 0.83–0.85 |
| W1j + curriculum | 1 / 5 (seed 4) | 0.83 / 0.83 |

**What it shows**
1. **The curriculum is the only lever that moved reliability:** 3 of 5 seeds, against 1 of 5 for the same recipe without it (P36). Learning the event and goal parts on a 1-code version first, before selection is needed, helps them co-adapt. That fits the round-7 diagnosis of the plateau.
2. **CA1 novelty as built does not help, and hurts with the curriculum** (P39, P41). Even seed 0, which learns without it, failed. The chain store's prediction error is high on random filler and on unpredictable natural text alike, so it does not single out the facts' first digits. Added at a gain of 4 from the start, it dominates the detector's logit and pushes writes toward whatever is unpredictable. A learned or annealed gain starting near 0 would be the next variant, but it is outside the stop rule's budget. [Correction, 3 Nov 2026: the diagnosis below (P42) shows a different mechanism. The novelty is not merely unselective: at initialisation it is *lower* at the facts' first digits than in the filler, so κ = 4 suppresses the start writes. A gain starting near 0 would not help either, because κ never moved and the signal only becomes informative after selection is learned.]
3. **The process note is recorded in the evolution protocol:** one extra queue worker on a GPU already 95–97% busy gave no speed-up (7.1 against 7.5 steps/s). It also made the evaluation queue start before the extra worker's last run finished, so two evaluations had to be rerun.

**DEC2 outcome.** Round 7 (best 1 of 5) and round 8 (best 3 of 5) both missed criterion 1 (≥ 4 of 5). By the approved stop rule, **the side-memory track is closed as "capable but not reliably learnable"**. The conclusion is written up in RESULTS.md §4.9:
- what the memory can do once learned: content-addressed selection among up to 5 stored codes, at gaps of about 2000 tokens, robust to natural filler when trained on it;
- what blocks it: a seed-dependent optimisation barrier at the start-selection step, which only a curriculum has partly lowered.

---

## Why W1j failed: a diagnosis (3 Nov 2026)

Daniel asked why the explicit CA1 comparator failed. I wrote `experiments/evolve/diag_novelty.py`, which trains nothing: it rebuilds a checkpoint, or a fresh initialisation, and on new 3-pair scattered batches reports the comparator's signals by token class (the facts' first digits, the other digits, the fact head " The code for NAME is", filler, the query). Evidence: `results/evolve/round8/diag_novelty.json` (P42).

**1. At initialisation the novelty points the wrong way.** On natural filler, seeds 0 and 1:

| Token class | Relative error ‖v − S a_{t−1}‖² / ‖v‖² | Novelty n |
|---|---|---|
| First digit (where the start must be written) | 0.37–0.39 | −0.15 to −0.16 |
| Other digits | 0.27–0.32 | −0.20 to −0.22 |
| Fact head | 0.54–0.57 | +0.03 |
| Filler | 0.51–0.53 | +0.01 |

- The AUC of n for first digit against filler is 0.24, and 0.15–0.35 in every head. On random filler it is 0.34–0.40.
- Two things cause it:
  - To an untrained store, digits are among the *most* predictable tokens. An untrained delta store recalls a blend of recent values, so its error measures how typical a token's representation is, not whether a fact is new.
  - The baseline subtraction then compares the first digit with the fact head just before it, which has a high error.
- So "built-in structure" was built in backwards. At κ = 4:
  - ε at first digits falls from 0.14 to 0.09 (seed 0), and from 0.12 to 0.06 (seed 1);
  - ε on filler rises from 0.13 to 0.16;
  - the gate's AUC falls from 0.58 to 0.31 (seed 0), and from 0.37 to 0.19 (seed 1).
- The detector has to overcome this prior before it can begin co-adapting with the goal key and read weight (the round-7 plateau analysis).

**2. κ is inert.** After 6000 steps κ is 3.72–3.99 in all 10 W1j runs (initially 4). Novelty is baseline-subtracted, so it has mean about 0. Its contributions to κ's gradient cancel across tokens, and the model neither uses the term nor switches it off. This is why a gain that starts near 0 would not rescue it: the gain would stay near 0.

**3. After training, the term is unbounded and specific to each head.** Training the chain store changes its error profile:
- The relative error reaches 8.2 on filler (seed 0, head 0). There κn is −14 at first digits, so that head can never write a start.
- In seed 4, head 0 gets κn = +21 at first digits. That makes a selective head, but it still writes on 19% of filler tokens (against 0.5% in passing W1i heads), enough junk to defeat selection.

**4. Novelty becomes informative only after selection is learned.** In the passing W1i curriculum run (seed 0, where novelty is computed but never used), one head's novelty separates first digits from filler at AUC 0.995 (relative error 5.7 at first digits). The trained chain fails exactly at the start digit, because " is" does not predict which code follows. In the failing W1i seeds the same AUC is 0.53–0.60. So this mismatch signal is a *consequence* of a trained CA3, not a trigger that could train one. That matches the biology better than the design did: CA1 mismatch signals novelty relative to CA3's *learned* predictions, and with a naive CA3 there is nothing to mismatch against.

**5. What successful runs have in common.** Each of the 4 runs that learned selection has one head that writes almost only at first digits:
- curriculum seed 0, head 0: ε 0.99 at first digits, 0.005 on filler;
- curriculum seed 3, head 0: 0.78 / 0.06;
- curriculum seed 4, head 3: 0.98 / 0.006;
- W1j + curriculum seed 4, head 3: 0.93 / 0.03.

The failing W1i seeds have a head that is selective but weak (seed 1: 0.33 / 0.0005), or none (seed 2: at best 0.34 / 0.24). One failing run has the signature anyway (W1j seed 3, head 1: 0.74 / 0.07). So a sharp event head is necessary but not sufficient; the goal key or the read weight can still be missing.

**Consequence for rounds 9–10.** I won't run more variants of this comparator. A signal that fixes the bootstrap would have to be selective at initialisation, and the delta store's own error is not. Daniel extended the track by two rounds; they test the one lever that worked, the curriculum.

---

## Round 9: the track reopens for two rounds (DEC3), curriculum schedules (3 Nov 2026)

**Decision.** Daniel: "I say go 2 more rounds before giving up if you think we are close". I think we are: the curriculum took reliability from 1 of 5 (P36) to 3 of 5 (P40), one seed short of DEC2's bar. Passing seeds reach digit 1 ≥ 0.96 at every rank, with exact 0.83–0.85 against 0.85. Recorded as DEC3; the criteria are unchanged, including the 6000-step limit.

**What the round-8 curriculum curves show** (training loss, single batches of 8; stages 1 code → 2 codes at step 1500 → 3 at step 3000):
- The 1-code stage is solved by step 1200 in every seed (loss ≤ 0.005). It is passkey retrieval and needs no selection.
- At each switch to 2 codes the loss jumps (0.21–0.69 at step 1800).
- The two seeds that solve the 2-code stage within it (0 and 4: 0.016 and 0.009 at step 3000) pass. Seed 3 does not solve it, but drops after the switch to 3 codes (0.075 at step 4200) and passes.
- Seeds 1 and 2 never leave the 2-code plateau (0.17–0.34; ln 2 / 5 = 0.14) and fail.
- So the 2-code stage is where selection is learned, and 1500 steps there is too short for some seeds. P42's signature (one head that writes only at first digits) is what forms.

**Arms** (W12; W1i natural recipe, refractory 0.95, 6000 steps, 3 pairs at evaluation, tap 21; seeds 0–4 each):

| Arm | Curriculum | Steps at 1 / 2 / 3 codes | Question |
|---|---|---|---|
| `long2` | `1:1500,2:3000` | 1500 / 3000 / 1500 | Does more time on 2 codes let more seeds solve it? |
| `no1` | `2:3000` | 0 / 3000 / 3000 | Is the 1-code stage needed, or does its passkey solution have to be unlearned? |
| `short1` | `1:1000,2:2000` | 1000 / 2000 / 3000 | A shorter 1-code stage, more 2-code time, the full 3-code time |

The baseline is P40 (`1:1500,2:1500`, 3 of 5). No CA1-novelty arm (P42). Each run is evaluated on held-out natural and random filler with windowed and full controls. Pass: digit 1 ≥ 0.95 at every rank on both fillers. DEC2 criterion 1 needs 4 of 5 for one arm.

---

## Is W1j fixable? Novelty is the wrong trigger; the declarative statement is the right one (3 Nov 2026)

**Daniel asked whether W1j is fixable, then:** "Can the simple declarative statement be used as a trigger? The LLM in general knows the meaning of 'The code for NAME is'; that in itself is a trigger that the human brain would fire on." And: "using the declarative statement is akin to taking the output of the LLM and looping it back, again mimicking the biology."

**1. No novelty measure marks the facts (P43).** At initialisation, first digit against filler (`diag_surprise.py`):

| Signal | Natural filler | Random filler |
|---|---|---|
| Frozen LM's own surprisal (an oracle: needs token ids) | 2.55 against 3.64 nats, AUC 0.48 | 2.70 against 14.2 nats, AUC 0.00 |
| Change in h (1 − cos with the previous token) | AUC 0.25 | AUC 0.26 |
| Untrained delta store on the raw states | AUC 0.29 | AUC 0.39 |

A random code digit is about ln 10 nats to the LM: average for natural text, and far below random tokens. The part of a fact that is surprising is its frame (" The code for NAME is", 6.5 nats). So the problem is not how W1j's comparator was built: in this task the facts are not where the input is novel. W1j is not fixable as a novelty gate.

**2. The meaning of the statement is a near-perfect trigger (P44).** `diag_declarative.py` fits one fixed direction at block 21: first value token against all other tokens, in generic declarative bindings embedded in WikiText train text:
- e.g. "Marisol's phone number is 5550193.", "The password to the vault is falcon.";
- 12 templates, none containing "code for", names disjoint from the task's, values mixing digits, words and alphanumerics.

It generalises across templates: AUC 0.985 on 4 held-out ones. On the task's own template, with WikiText test or random filler, first digits against:

| Comparison | AUC |
|---|---|
| All filler | 1.00 on both fillers |
| Digits that occur in natural filler (4,121: years, counts) | 0.999 |
| Filler tokens right after is/was/are/were (1,039) | 1.00 |
| The other code digits | 0.98–0.99 |

So it is not a digit detector or a position-after-a-copula detector. It marks the onset of an asserted value. At a threshold that fires on 1% of filler it catches every first digit; it also catches 85–99% of the other code digits, which score lower (mean projection 11.2 at first digits, 5.8 at the others, −5.5 on natural filler).

**What this means.**
- **The event part is not missing from h; it is not found.** The event detector is linear in the same normalised h (σ(w_e·x + b)), so it *can* represent this trigger exactly. The seed-dependent barrier (round-7 analysis, P42) is an optimisation problem, not a representational one. A run has to discover w_e from one weak answer-token gradient while the key and the read weight co-adapt.
- **It also fits the biology better than W1j did.** The hippocampus does not judge relevance from its own prediction error alone. Its input from entorhinal cortex is already the cortex's interpretation of the scene. Encoding is boosted by what that interpretation says matters: semantic "levels of processing" (Craik & Lockhart 1972), and prefrontal and ventral-striatal relevance signals that gate encoding. Here the frozen LLM plays the cortex, and reading its "a value is being asserted" state as the trigger is Daniel's loop from cortex back to hippocampus. It is not the LM's output tokens but its mid-stack interpretation, which is what EC carries. W1j used the wrong comparator: the hippocampus's own untrained store, when the information is in the cortical input.
- **The rules.** The direction is a constant computed beforehand from the frozen model on other text (2048 floats, within E3's buffer limit). The module still sees only h (E4). It was fit without the task's template, names or filler split (E6). It would still be prior knowledge chosen with this kind of task in mind. Fair use needs an evaluation on fact templates the direction was not fit on, and a statement of where the direction came from.

---

## Design discussion: time, variable-length meaning, retained windows and reinstatement (3 Nov 2026)

Daniel's points, in order, with the GPU session's answers. The panel `meaning-trigger-design` (3 designers and a critic) was relaunched on the meaning framing after the first, novelty-framed panel was stopped. The first panel's two finished designers had both concluded "not fixable as a bootstrap" (P43).

**1. "Filler is always novel by definition, it is unique, but not necessarily meaningful. What is meaningful is the sequence of tokens, the window around the phrase."** This is why P44 works. Within its 64-token window, the frozen LLM's attention folds " The code for NAME is" into the value's first token. So the meaningful window arrives at block 21 as one vector.

**2. Depth and time ("tokens at a later layer are also temporally separated?"; tap the GDN layers; transformers as flattened time series).**
- **Depth is not delay in one forward pass.** Block 21 at position t has seen exactly the tokens block 1 has seen. With block windows, deeper layers do not even reach further back. The only true delay is the generation loop: the top layer's output at t is the bottom layer's input at t+1.
- **The brain has a processing lag a transformer lacks.** Higher cortical areas still hold the meaning of word t while sensory areas hold word t+2. Cortex also has a hierarchy of temporal receptive windows (Hasson et al. 2008; Lerner et al. 2011).
- **Options for CA3 to see delayed tokens:**
  - a pipeline lag across taps: the value from a low or mid layer at t, the key from a late layer at t − k;
  - tapped delay lines whose initialisation comes from a meaning direction;
  - writes at the boundary of a meaningful unit;
  - the recurrent states of a hybrid's GDN layers. Those are pretrained, but they are not windowed, so the base already remembers. Reading them needs a decision on E4;
  - phase and time-cell codes;
  - compressed replay.

**3. Variable-length statements and a cumulative write score ("summed over time, once a threshold is reached a write is triggered and the score is reset").**
- P45 tested the first part: modifiers make the trigger stronger, and only an interrupting clause weakens it.
- The proposal is an integrate-and-fire gate: u_t = λ u_{t−1}(1 − s_{t−1}) + salience_t, a write s_t when u_t ≥ θ, then reset. Unlike the soft, never-resetting gates of LSTMs and GDNs, it gives one discrete event per statement of any length. That is the precedent of the HM-LSTM boundary detectors (Chung et al. 2017).
- What it adds here:
  - it writes once, at the value's onset, although the meaning signal also fires on digits 2–4;
  - it carries evidence across window-block boundaries, which the frozen LLM cannot;
  - it is causal with a fixed state.

**4. "Can we somehow retain the important windows, thus giving the model a better sense of time?"**
- Every store so far superimposes facts and folds time away. The chain keeps order, but not timing.
- **Episodic slots** fix that: K fixed slots, each written at a firing event with the window that built up the evidence (projected), a timestamp and positions within the window. The query matches slot keys, and the chosen slot replays its window in order.
- Slots keep a fixed-size state (E2) and remove interference between facts (P24, the likely cause of the 0.83–0.85 exact ceiling). Their timestamps give an explicit sense of when. The price is a hard capacity with a replacement policy.
- **Biology:** separated codes, plus sequence structure within an episode (time cells, MacDonald et al. 2011), plus replay that keeps order and compresses duration.
- **Related work:** EM-LLM (Fountas et al. 2024; events cut by surprise and retrieved into attention) and InfLLM (blocks of context as memory units). P43 says that for facts the boundary must come from meaning, not surprise.

**5. "Our side channel injection could be the CA2 model (additive attention); inject entire recalled phrases, boosting the attention score. Or is this more the CA1 → CA3 loop?"** See the answer of the same date: putting a recalled phrase back where cortex can use it is *reinstatement*, the hippocampal output path (CA1 → subiculum → deep entorhinal cortex → neocortex). Its re-entry into the hippocampus is the big loop (deep EC → superficial EC → DG/CA3). CA2's best role is the context-dependent gain on what is reinstated.

**Decision.** Daniel: "We can go past 10 since we are delving into new territory." Recorded as DEC4.

---

## Design panel on the meaning trigger, and the round-10 proposal (3 Nov 2026)

The panel `meaning-trigger-design` had 3 independent designers (integration, legitimacy, biology and generality) and an adversarial critic. They read the code and records and ran CPU-only checks with the frozen model; nothing was trained. Their scripts and outputs are in the GPU session's scratchpad, not the repo. **The numbers below are unreproduced panel checks**: each becomes a register row only when reproduced as a pre-GPU test in the repo.

**Findings that change the picture**
- **The goal key, not the trigger, is the remaining barrier.**
  - With perfect (oracle) first-digit events at W1i's initialisation, selection is still at chance (0.31–0.34): the context keys of different facts are nearly identical.
  - In trained runs, the fact keys of passing heads have cosine 0.03–0.07 with each other, and retrieval with oracle events is 0.88–1.00. In failing heads the cosine is 0.84–0.99, and retrieval returns only the most recent fact.
  - Passing heads weight the name 1.6–4.1× the filler, and the shared frame tokens 0.35–0.6×.
- **The goal read weight is not a barrier:** σ(goal_logit) ends at 0.44–0.50 in all five round-8 curriculum seeds (0.50 at init), passing or failing. This revises the round-7 three-part analysis to two parts: events and key.
- **Every head that learned selection had a learned τ ≈ 2** (4 of 4 runs; about 0.03 by chance). Head index and τ are confounded, because the default cycle 2, 4, 8 puts τ 2 on heads 0 and 3.
- **The current refractory is too weak.** W1i's leaky trace puts 0.05 on one event, so after a sharp first-digit event it still passes 0.36–0.57 at digits 2–4. Those writes overwrite G under near-identical contexts. A max-trace refractory passes 0.02–0.05, and its parallel form matches the recurrence to 1e-5 and is prefix-bitwise causal. This is Daniel's "fire once, then reset", in effect.
- **Passing runs did not learn the P44 direction:** their detectors have cosine 0.14–0.23 with it. They found other selective features.
- **The trigger is about meaning in the ordered window (Daniel's point), with known limits:**
  - held-out paraphrases score 12.0–12.1 and a question frame 10.0;
  - the *same frame tokens scrambled* score 5.9, the bare-number level;
  - a non-binding count scores 1.9;
  - a credential-free refit (no "code", "PIN", "password", ...) keeps AUC 0.999 / 1.000, with cosine 0.89 to P44;
  - controls fail as they should: a shuffled-label direction 0.48 / 0.79, a random direction 0.23 / 0.095, both with 0% recall;
  - held-out fact templates are graded: PIN-like bindings about 12, "extension" or "was given the number" about 6;
  - it needs 5–7 frame tokens inside the visible block, fails when the value comes first ("4821 is the code for NAME", 4.9), and fires on negated statements;
  - false writes in WikiText fall at identifier onsets ("NY 93", "imported 6"): 0.18 per 1000 tokens above a projection of 8 and 2.1 above 6. That is 15–300× sparser than the filler writes of passing learned heads.
- **Label-free discovery failed in its crude form.** A "needed later" direction, fit from what the full-attention model reaches back for, is orthogonal to P44 (cos 0.001) and points the wrong way on the task.
- **A key start fit on generic name contrasts failed:** fact-key cosine stayed at 0.56–0.99. So a meaning-based key needs more than a fixed subspace.
- **A reporting bug:** B17.

**The critic's round-10 plan** (to run only after Daniel rules on DEC5):
- **A, W1k:** the frozen meaning prior OR'd into W1i's learned events with an onset gate, P40 curriculum, seeds 0–4.
  - The prior's threshold is set by rules on the contrast set only: θ is the 99.9th percentile of its negatives, about 8.2.
  - It is not replaced, because a surrogate suggests W1i's dense background writes help random-filler generalisation.
- **B:** the same with a matched random direction, at the same false-write rate on generic text, the content control.
- **C:** A with all five heads at τ = 2.
- **Pre-registered readings:**
  - A ≥ 4/5 with B < A: criterion 1 met under DEC5;
  - A and B both ≥ 4/5: not attributable to meaning;
  - A ≤ 3/5 but C ≥ 4/5: C is the recipe;
  - A and C ≤ 3/5: the key is the bottleneck.
- **Pre-GPU gates:**
  - bitwise identity with the prior off (so P40 is the baseline);
  - E1 with the prior firing;
  - E3 and provenance (sha256; maximum |cos| with any base weight 0.26);
  - the init ε profile through the real code path;
  - the B17 fix;
  - the meaning battery at the operating threshold, plus the bf16 shift;
  - a held-out-template harness (NEAR: PIN, passcode, "code:"; FAR: extension, "given the number", locker, jersey), with full attention ≥ 0.9 per family;
  - optionally, the gradient signal-to-noise of W_c at W1i init, W1k init and with oracle events.
- **DEC5 (proposed; needs Daniel):**
  - admissibility conditions for a pre-fit prior;
  - the attribution rule (B must learn fewer seeds);
  - held-out-template reporting;
  - the wording "reliably learnable given a cortical relevance prior";
  - oracles never count;
  - Daniel's ruling on whether a label-fit frozen direction is admissible at all.

**Note.** The panel's key finding fits Daniel's framing from the design discussion. A meaningful window must reach the *key*, not only the trigger. The drifting context with τ 2–8 that W1i uses mostly keeps the shared frame, so the facts' keys collide.

**[3 Nov 2026, later] DEC5 rejected.** Daniel: "frozen direction doesn't help, and isn't a good model. A different language, dialect, or something like domain specific phrasing (governmental, medical, police, etc) would break it." So arms A and B of the panel's plan are dropped. P44–P45 remain diagnosis: the gate's target is linearly present in h, so the seed barrier is an optimisation problem. What survives from the panel is general mechanism, not task knowledge: the key collision, short-context (τ ≈ 2) heads, and the max-trace refractory (one write per burst).

**[3 Nov 2026, later] Daniel: a write *window* with hysteresis.** "A window around the trigger point is probably more optimal as well. If the integration had both positive and negative values to sum then the trigger point can be viewed as a clock signal: once a threshold is reached writes are turned on, when the signal degrades and falls below a threshold, then the write trigger turns off, thus recording a window and fixing the problem of only triggering on the first number in a sequence."

Design notes (GPU session):
- **The gate.** A signed leaky integrator, u_t = λ u_{t−1} + (w·x_t + b), with evidence that can be negative, drives a gate with hysteresis (a Schmitt trigger). The gate turns on when u_t ≥ θ_on and off when u_t < θ_off < θ_on. A soft, trainable form is a bistable unit, g_t = σ(a(u_t − θ) + c·g_{t−1}), whose self-excitation c makes it latch. It is causal, with a state of two scalars per head (E1, E2).
- **Biology.** The closest match is graded persistent firing in entorhinal layer V (Egorov et al. 2002): input switches a neuron into sustained activity that holds until it is turned off. Hasselmo's encoding and retrieval phases are a second match.
- **What the window needs downstream.** In W1i's superposed goal field, the writes inside one window are under nearly the same drifting context. They overwrite each other, which is the digits-2–4 problem the panel measured. A window therefore needs a time base: while the gate is on, a counter of the time since onset is part of the key (time cells within an episode), or each window gets its own slot. The window then becomes an ordered episode that recall can replay, and the order is explicit rather than left to the successor chain.
- **A window "around" the trigger, causally.** A short ring buffer of the last k states (k is a fixed state) lets the window include the k tokens before onset, as a camera's pre-record does. That captures the name and frame as well as the value.
- **Why it may also ease learning.** A window that roughly covers the value is enough, so the gate has a wide target, instead of a single token it must hit. The answer loss still has to teach when to open (no priors, DEC5), but its landscape should be smoother.

---

## Round 9 checkpoint: curriculum schedules (3 Nov 2026)

**Trimmed at Daniel's request.** After 10 of 15 runs, no arm could reach 4 of 5: each already had two seeds stuck on the plateau, judged from training loss. Daniel: "trim". The two seed-4 runs in flight were stopped (rc 143; their partial logs are not committed), and the third never started. The 12 finished runs were evaluated on held-out natural and random filler; windowed 0.00 and full 1.00 in every evaluation.

| Seed | `long2` (1:1500, 2:3000) | `no1` (2:3000) | `short1` (1:1000, 2:2000) | Round 8 `1:1500,2:1500` (P40) |
|---|---|---|---|---|
| 0 | ✓ exact 0.83 / 0.84 | ✗ | ✓ 0.86 / 0.88 | ✓ |
| 1 | ✗ | ✗ | ✗ | ✗ |
| 2 | ✗ | ✓ 0.85 / 0.88 | ✗ | ✗ |
| 3 | partial: digit 1 0.95–0.97 natural, 0.90–0.92 random | ✓ 0.86 / 0.84 | ✗ | ✓ |
| 4 | not run | not run | not run | ✓ |

(Pass: digit 1 ≥ 0.95 at every queried rank on both fillers. Exact is natural / random.)

**What it shows**
- **No schedule reaches DEC2's 4 of 5** (P46–P48). The best, `no1`, learned 2 of 4. Per seed, the four schedules give 6 passes in 16 runs on seeds 0–3.
- **Schedules reshuffle which seeds learn rather than how many.**
  - Seed 0 learned under every schedule except `no1`.
  - Seed 2 learned only under `no1`.
  - Seed 1 has never learned under any recipe since round 7.
  - Outcome depends on the initialisation interacting with the data order. That fits the panel's diagnosis: the barrier is the key collision at initialisation, which the schedule does not address.
- **Exact recall when learned is 0.83–0.88.** Three passing runs reach ≥ 0.85 on both fillers at 512 tokens with 3 codes.
- **This is the first round under DEC3 to miss criterion 1.** Under DEC4, round 10 moves to new mechanisms. Merit tests are running for multi-tap keys, the hysteresis window, time-indexed episodes, reinstatement, time, additive replay and replay-driven key separation.

---

## Round-10 merit tests: which new ideas have merit (3 Nov 2026)

Daniel: "begin planning round 10 around new ideas like multi-tap architectures, saving entire windows around on/off events triggered by a leaky integrator style switch, incorporating time. We need to decide which of these ideas have merit." Then: "what about the ideas on side channel replays?"

Seven agents each ran a decisive test with no training of the side module: frozen-Qwen forward passes, oracles, CPU toys. Their scripts and outputs are in the GPU session's scratchpad (`r10_merit/`). **These are unreproduced panel checks; each becomes a register row only when reproduced in the repo.** The hysteresis toy was stopped after 3 hours of single-threaded CPU runs. Its interim numbers are below, and its automated judge did not run; the synthesis is the GPU session's.

**Ranking**

1. **Episodes that keep the window, indexed by offset (slots): strong.** Exact 4-digit recall with oracle windows:
   - A W1i-like drifting-context store fails *within a single fact*: 0.00–0.03 exact even with only the queried fact written, because consecutive digit contexts overwrite each other.
   - Indexing each value by its offset since onset fixes that: 1.00 with an oracle key.
   - With a fit-free key, slots recall 0.91–0.98 exact with 3–5 facts, natural or random filler. The key is the mean of the *layer-3* states over the window, centred at read time.
   - The same key in a *superposed* time-indexed store gives 0.17–0.34: near-collinear keys let the last write win. So the fix is separate slots, not time codes in superposition.
   - Risk: eviction. With FIFO and 16 false windows the fact is evicted in 35–54% of sequences, so a better replacement policy is needed.
2. **Reinstatement into attention: strong as an oracle, but not at our tap.**
   - With perfect selection, reinstating the statement window works at layers ≥ 0 (1.00) and ≥ 4 (0.96–0.99), but falls to 0.31–0.50 at ≥ 14 and 0.00 at ≥ 21. The copy circuit sits in layers about 4–20, below block 21.
   - If every position sees the slots from layer 0, the frozen LLM selects for itself (0.99–1.00), and false recalls barely matter (8 false windows cost 0.00–0.03).
   - A pre-record of 4–8 tokens is needed so the window holds the name.
   - It is a new interface (writing into most layers' attention), so it needs a decision.
3. **Additive replay at tap 21 with a small learned map: the readout is solved.**
   - Raw replay of stored states fails: they encode the current digit (probe 1.00) and not the next (probe 0.10).
   - A rank-10 map of 41k parameters, trained through the frozen upper blocks in about 150 steps, gives held-out exact 1.00 on both fillers with oracle recall. Its output follows the replayed fact, and the LM loss changes by +0.01 to 0.04. One seed.
   - The additive delta (full − windowed) at block 21 gives 1.00, so tap 21 has the capacity.
   - For a tap-21 memory, this beats reinstatement.
4. **Multi-tap keys: moderate, reframed.**
   - Name identity lives at the name token's own state and is not carried forward to " is" or the space. At the space, every layer and pooling choice tops out at 0.63. That is why W1i's keys, taken there, collide (cosine 0.87–0.94 at init; retrieval 0.32–0.52).
   - The name token's state works as a key (1.00) at layers 7–21. But a fixed lag is a template prior and falls to 0.51 under phrasing jitter.
   - Combining layers adds nothing beyond the right position.
   - What carries over is a *lower-layer* (layer 3–7, lexical) key pooled over a stored window, as in the slot test.
5. **Hysteresis write window: unproven as a learning aid; useful as a segmenter.**
   - CPU toy, 10 seeds per arm:
     - short toy: window 7–8/10, point trigger 4–7/10, max-trace 5/10;
     - harder toy: window 0/10, point 1–2/10;
     - oracle 10/10, dense writes 10/10 (so the toy lacks the real barrier);
     - window with time-since-onset keys 0/10;
     - integrate-and-fire with reset 0/10.
   - The reinstatement and slot tests show what a window trigger has to deliver: high recall (false windows are cheap) and a window that starts 4–8 tokens before the value.
6. **Replay for pattern separation: weak.** With no answer signal it takes fact keys from cosine 0.88–0.93 to about 0.0 and retrieval from 0.39 to 0.81–0.86. It never reaches 0.95 alone, adds nothing to a clean answer signal, and gives +0.16 to +0.30 only when labels are scarce. Replaying filler teaches nothing.
7. **Time stamps: weak.**
   - The trained W1i checkpoints already return the *updated* code when a fact is restated (digit 1 NEW 0.70–0.93), better than full attention (0.37–0.57, which often returns the old code). That comes from delta-rule overwrite.
   - Time-stamped recency is never better, and costs 0.09–0.23 on other facts.
   - The TimeChannel is a saturated absolute-position code.
   - Order questions have no full-attention ceiling.
   - The useful form of time is the offset index within an episode (item 1).

[Correction, 3 Nov 2026: an exploration of the scratch result files found this entry overstated several numbers. (1) **Slots:** "0.91–0.98 exact" is retrieval by cosine with an *oracle onset*, scored by selection (the read returns the stored true value), not behavioural accuracy. It holds with 0 false windows; with K = 8 and 4 false windows, 5 codes drop to 0.74–0.79. A causal read without the onset gives 0.78–0.89. The 16-false-window figures (0.44–0.65) mix eviction with selection errors. k-WTA sparse keys in a superposed store reach 0.35–0.64, which was not mentioned. (2) **Reinstatement:** the frozen LLM selects by itself (0.99) only with kv-mode reinstatement from layer 0 and every row seeing the slots. From layer 4 (kv or h mode) self-selection gives 0.24–0.60, so a selector is needed. The layer-14 range is 0.29–0.50. (3) **Hysteresis:** with gate centring (a causal running-mean subtraction), *every* gate type learns in 9–10 of 10 seeds on the short toy, so the window's apparent advantage was the lack of centring. The harder-toy "window 0/10" comes from an unfinished run. (4) **Replay separation:** +0.16 to +0.30 is the reconstruction loss (Rr); the contrastive loss (Rc) gives +0.08 to +0.17. (5) **Time:** for checkpoint s0, time-stamped recency at g = 0.5 gives 0.79–0.87 on the current code, above the delta rule (0.63–0.70), against "never better". It still costs 0.21–0.34 on other facts. Recorded as B18.]

**The overall picture.** The barrier is superposition in two places. Within a fact, digits overwrite each other. Between facts, keys taken at the wrong token, after the name has faded, collide. Slots holding ordered windows remove both. A low-layer pooled window key selects without fitting. A small learned map reads the recalled window out at tap 21, or the frozen LLM itself reads it if reinstated low enough.

---

## Round 10 plan: W1s, one configurable episodic-slot memory (3 Nov 2026)

Daniel approved multiple taps ("a second tap is fine, more if needed"), lexical keys and several key types ("keys are cheap"). He also approved reinstatement as an arm ("worth a test at the very least"), and asked whether a combination beats every idea in isolation. The plan is in the GPU session's plan file; here is a summary.

- **One module, `experiments/evolve/w1s_slots.py`, in which every arm is an ablation of the same code path:**
  - a gate (point / hysteresis / Schmitt / oracle) with causal centring;
  - windows around the gate's onsets with a pre-record, opened by non-maximum suppression;
  - K slots that keep ordered windows, evicting the lowest priority (gate strength plus recency), or a superposed store as the ablation;
  - keys, lexical (layer-3 pooled, no fitted parameters), learned (layers 3 and 21), or both;
  - a read: softmax over the slots plus a null slot, with offset alignment inside the chosen slot;
  - output either additive replay at block 21 through a small learned map in CA1 form, or reinstatement into attention at layers ≥ 4 with a learned bias.
- **Hooks:** `experiments/evolve/taps.py` (`TapLM`) adds read-only caching hooks for extra taps and the reinstatement mask.
- **Evaluation:** `eval_by_query.py` gains phrasing jitter, the update task, held-out names and the W1s controls, and fixes B17.
- **Stages:**
  - Stage 0: smoke tests on the real model (oracle gate) that must reproduce the merit numbers.
  - Stage A: the combination C, 9 leave-one-out arms, 3 isolation arms, a W1i-like everything-off arm, reinstatement and its oracle ceiling, all on seeds 1 and 2 (the seeds no recipe has solved).
  - Stage B: 5 seeds on the pruned winner, the full DEC2 grid, the LM run and the CPU suite.
- **Pre-registered rules:**
  - A component is necessary if removing it loses a seed, or ≥ 0.10 mean exact.
  - The combination beats isolation if C learns 2 of 2 and every isolation arm learns ≤ 1, or, with equal counts, C leads by ≥ 0.10 exact at 2048 tokens with 5 codes. Stage B must confirm it on 5 seeds.
  - Reinstatement is adopted only if it matches or beats additive output everywhere.

---

## Round 10 Stage A: slots are the answer; the combination is not (4 Nov 2026)

32 runs (16 arms × seeds 1–2, the two seeds no recipe has solved since round 7), natural recipe with the `1:1500,2:1500` curriculum, 6000 steps, `--needle-names train`. 128 evaluations. Windowed 0.00 and full 0.99 in every one.

Exact recall: natural 512 (3 codes) / random 512 / natural 2048 (5 codes) / random 2048.

| Arm | Seed 1 | Seed 2 |
|---|---|---|
| **I1** slots + offset read only (point gate, no centring, no pre-record, learned key at tap 21) | **0.95 / 0.95 / 0.93 / 0.86** | **0.98 / 0.95 / 0.96 / 0.86** |
| A2 = C without gate centring | 0.92 / 0.99 / 0.88 / 0.99 | 0.85 / 0.90 / 0.82 / 0.78 |
| C (full combination) | 0.93 / 0.65 / 0.78 / 0.63 | 0.71 / 0.20 / 0.51 / 0.09 |
| A9 (C, 8 slots) | 0.89 / 0.67 / 0.82 / 0.69 | 0.90 / 0.72 / 0.75 / 0.65 |
| A8 (C, no offset alignment) | 0.72 / 0.55 / 0.72 / 0.47 | 0.67 / 0.51 / 0.68 / 0.42 |
| Ro (reinstatement, oracle gate) | 0.93 / 0.94 / 0.85 / 0.89 | 0.92 / 0.94 / 0.84 / 0.90 |
| R (reinstatement, learned gate) | 0.57 / 0.66 / 0.20 / 0.35 | 0.54 / 0.61 / 0.22 / 0.30 |
| A1, A3–A7, I2, I3 | 0.04–0.40 | 0.02–0.46 |
| B0 (everything off, superposed) | 0.00 | 0.00 |

**Diagnostics.**
- C's hysteresis gate with causal centring and running-std normalisation adapts to the filler. On random filler the queried fact is retained at the query in only 0–22% of sequences, and the read falls to the null slot (weight 0.35–0.58).
- Without centring (A2), retention is 97–100% on random filler.
- I1 retains the fact in 95–100% of sequences everywhere, with null weight 0. Its argmax slot hit is 0.89–0.99 at 512 and 0.65–0.70 at 2048 with 5 codes, yet exact stays at 0.86–0.96: the soft read and the offset alignment recover.
- R's learned gate places windows badly (slot hit 0.18–0.38). With the oracle gate (Ro) the hit is 0.98–1.00.

**What it means**
1. **The slot store with an offset-aligned read is the mechanism.**
   - Every arm without slots fails (A4, I2, I3, B0); without alignment, digits 2–4 fail (A8).
   - The simplest slot arm, I1, is nearly at DEC2 on the two hardest seeds: exact 0.86–0.98 in every cell.
   - Its first digit is ≥ 0.95 at every rank on seed 2, and 0.94 at the weakest rank on seed 1.
   - This answers the barrier diagnosed in rounds 7–9: removing superposition removes the seed lottery.
2. **The combination does not beat isolation (pre-registered rule).** C learned 0 of 2; I1 learned 1 of 2, with the other 0.01 short.
   - The extras (lexical keys, tap 3, pre-record, hysteresis) did not add anything on top of slots in any arm that worked.
   - Causal centring of the gate is harmful: it makes retention depend on the filler's statistics.
   - The leave-one-out reading is confounded, because every A arm kept centring. The planned "necessary component" test is therefore valid only for slots, alignment and capacity.
3. **Reinstatement is a gate problem, not a mechanism problem.** With the oracle gate it reaches 0.84–0.94. Its learned gate, driven by tap-0 states, misplaces windows.

**Stage B (pending Daniel's checkpoint):** 5 seeds (0–4) and the full DEC2 grid, with jitter, held-out names, the update task, the LM run and the CPU suite, for I1 (the winner) and A2 (combination minus centring). Possibly also I1 + lexical/tap-3 keys, to test whether keys help once centring is gone.

---

## Round 11 planning: compression and time (4 Nov 2026)

**Daniel's questions.** Did we test temporal injection? Are the saved windows ordered? Has anyone tried compressing a snippet to its high-value (high-attention) tokens? He added: "there is also biological precedence as the hippocampus also uses compression." He then approved attention weights as inputs (DEC8).

**Planning workflow.** Three planners (hippocampal biology, ML context compression, repo experiment design) and a synthesis. The proposal is `research/plans/round11_plan.md`. The planners ran eval-time diagnostics on the trained I1 checkpoints (seeds 1–2, n = 24–96, CPU, the real frozen upper blocks). These numbers are **unreproduced** diagnostics, not register rows:

1. **Length fails because of the linear recency penalty.** Learned at 512 tokens, ω·age costs about 1.5 logits at an age of about 1800, against a cosine margin of 0.3–0.45. 90–98% of slot misses at 2048 tokens lose to a younger filler window.
   - Setting ω = 0 at eval raises random 2048 × 5 from 0.71 to 0.84–0.90, and 4096 × 8 from 0.38–0.46 to 0.83–0.92.
   - But the stale code then wins 46–50% of updates.
   - A content-gated "superseded" recency fixes both on the cached checkpoints: an old slot is penalised only when a newer slot matches the same query. Updates rise from 0.55 to 0.88 (512, seed 2) and from 0.77–0.83 to 0.92–0.94 (2048).
2. **Compression costs nothing.** The windows' onsets sit at a fixed offset before the first digit (−6 for seed 1, −3 for seed 2), and the read uses about 4–5 of the 24 stored entries. The 2048-d alignment code is 79–92% of the 0.89M-float state.
   - Cutting the window to 12 and passing the code through a fixed random 64-d projection shrinks the slot buffer 39× (801k to 20.7k floats) at unchanged exact.
3. **More slots hurt under linear recency** (K = 64: 0.73). The bottleneck is distractors, not capacity.
4. **Received attention does not mark the code.** Inside the 64-token block, the attention a token receives puts the 4 digits at mean rank 17–24 of 24, at layers 4–20 on both fillers. Nothing near a fact attends to its code; the code is used only at the distant query.
   - So keeping high-attention tokens at write time would drop the very content to be recalled.
   - The biological reading is synaptic tagging and capture: a tag set at encoding must be confirmed by a later signal. Here, tags must be learned from the answer loss, or chosen by position within the episode.
5. **The TimeChannel clock is unused.** Zeroing it changes exact by ≤ 0.03.

**Proposed round 11** (single-factor arms on I1, seeds 1–2):
- **Time:**
  - T-scale: ordinal or log recency;
  - T-supr: content-gated supersession in the read;
  - T-supw: a supersession tag set at write;
  - T0: no TimeChannel.
- **Compression:**
  - C-code: a learned 64-d alignment code;
  - C-keep: a learned scorer keeps 8 of 24 tokens (Daniel's idea, in a form that works);
  - C-n12: fixed truncation, the control for C-keep.
- **COMB:** the best time arm + C-code + n = 12 + K = 64.
- **Optional:** ordinal stamps with an order task, if one has a full-attention ceiling.

Stage 0 has no-training gates on Stage B's checkpoints first. The decision rules are fixed in the plan.

**[4 Nov 2026, later] Round 11 revision 1: time as data, not as a penalty.** Daniel: "time as a penalty might not be the most appropriate use of time … LLMs went away from distance-based position encoding to RoPE … time as a data point, ordinal encoding, position relative to other tokens I think still matters."
- The log/ordinal penalty arm is replaced by two time-as-data arms:
  - rotary slot time (partial rotary on the key and query, log-spaced periods);
  - a query-chosen ordinal rank feature.
- Supersession stays, since it is ordinal and content-gated.
- A CPU-toy gate (G0.8) checks first that this does not repeat the merit failure of time codes concatenated into keys.
- The plan and runbook are updated (`research/plans/round11_plan.md`, `round11_runbook.md`).

---

## Round 10 Stage B: I1 nearly meets DEC2; two new weaknesses (4 Nov 2026)

Seeds 0–4 (I1 and A2 reuse Stage A's seeds 1–2), 420 evaluations: the 18-cell DEC2 grid, plus jitter, held-out names and the update task. Windowed 0.00 in every evaluation.

| Arm | Learned (digit 1 ≥ 0.95 every rank) | Grid mean exact by seed (0–4) | Held-out names | Jitter |
|---|---|---|---|---|
| **I1** | **3 of 5** (seed 1 0.94, seed 4 0.87) | 0.93 / 0.93 / 0.95 / 0.96 / 0.83 | 0.58–0.74 | 0.88–0.99 |
| A2 (combination minus centring) | 0 of 5 | 0.85 / 0.96 / 0.90 / 0.66 / 0.86 | 0.46–0.63 | 0.57–0.84 |
| I1K (I1 + lexical keys from tap 3) | 1 of 5 | 0.22 / 0.12 / 0.26 / **0.98** / 0.89 | 0.20–0.88 | 0.16–0.87 |
| RI (reinstatement with I1's gate) | 0 of 5 | 0.63 / 0.20 / 0.23 / 0.16 / 0.21 | 0.21–0.67 | 0.14–0.41 |

**What it shows**
1. **I1 is the recipe, one seed short of DEC2** (P54).
   - Criterion 2 (≥ 0.85 exact across 512–2048 tokens, 1–5 codes, both fillers) holds for every learned seed: 17–18 of 18 cells.
   - Criterion 1 needs 4 of 5. Seed 1 is at 0.94, and seed 4 fails at long range: random 2048 × 5 gives 0.35 / 0.46 / 0.57 for the three oldest ranks.
2. **Held-out names fail** (P55). The fact is retained but the learned key picks the wrong slot: it has memorised the 20 training names. The one lexical-key run that learned (I1K seed 3) generalises (0.88), but lexical keys are unstable in 3 of 5 seeds.
3. **The update task is ill-posed** (P56). Full attention itself prefers the old code, so "new" has no ceiling in this form.
4. **The planners' recency explanation does not predict which seeds fail** (P58). Alignment sharpness does: the failing seeds have a lower log β_align.
5. **Reinstatement with a learned gate is not viable** (RI 0 of 5). A2 confirms the extras do not help (P57).

**Round 11 revised accordingly** (revision 2 in `research/plans/round11_plan.md` and the runbook):
- held-out names become a deciding cell, with name-generalisation arms;
- the update task needs a form with a full-attention ceiling;
- gate G0.1 is reframed around alignment;
- reinstatement stays deferred.


---

## Round 11 Stage 0: the age penalty, not alignment, limits long range; no task has a full-attention ceiling (4 Nov 2026)

**Setup.**
- Runbook revision 3, §3–§4, run on geof by the round-11 GPU session.
- Code: round-11 flags in `w1s_slots.py`, `eval_by_query.py` and `run.py`, plus the diagnostic scripts `fit_code.py`, `toy_time.py` and `toy_keep.py` (`8a0319a`).
- Code review: adversarial, three lenses with two refuters per finding. 16 findings, 14 confirmed and fixed, none a causality leak (`research/reviews/2026-11-04_round11_code.md`).
- Defaults are bitwise identical to round 10.
- Round-10 I1 checkpoints re-evaluated on four committed Stage-B cells reproduce them sequence for sequence (`compat_*`).
- Every Stage-0 evaluation carries the windowed, empty-store and shuffle controls in the same pass (`--ctrl-conds`). Of 247 jobs (257 evaluations), none is void: windowed and empty 0.00 everywhere; shuffle 0.00, which is chance for a 4-digit code.
- These are evaluation-time diagnostics on trained checkpoints (rule E6), not trained results: P59–P67.

| Gate | Pre-registered rule | Outcome |
|---|---|---|
| G0.1 (P59) | Seed 4's ranks 0–2 at random 2048 × 5 ≥ 0.75 under an alignment or recency override | **Neither passes as worded.** Removing the age penalty: seed 4 0.24 → 0.78 / 0.67 / 0.74; seeds 0–3 oldest rank 0.64–0.72 → 0.95–0.99. Sharper alignment (+0.3, +0.6): no change. I1K's collapse is untouched |
| G0.2 (P60) | supr or supw: 4096 × 8 ≥ 0.80 on both seeds with stale ≤ 0.10 | **Not met** (seed 4: 0.58 supr, 0.62 supw). none and log fix length (seed 1 at 4096 × 8: 0.64 → 0.88–0.89) but lose updates. ord wins updates (new 0.79–0.81, stale 0.07–0.10) but loses length |
| G0.3 (P62) | supw false matches ≤ 2% | **Fails** (held-out names 3.1–4.1%): A3 dropped |
| G0.4 (P63) | Value-norm keep within 0.03 of the ceiling | Value-norm top 8 matches the full window (−0.04 at worst) and exceeds the 4-transition "ceiling", which is not a ceiling for this readout. A6 starts from value norm. Random and attention keep: ≈ 0 |
| G0.5 (P64) | Learned 64-d code ≥ 2048-d − 0.02 | On average yes (+0.000), with 19× less state. One cell misses by 0.001 (2 sequences of 96) |
| G0.6 (P65) | K = 16 coverage ≤ 0.8 and K = 64 ≥ 0.95 | **Met**: K = 64 stays in A8. Exact beyond K is then limited by slot hit |
| G0.7 (P61) | Full ≥ 0.80 and windowed ≤ 0.10 | **No task admitted.** Full attention 0.07–0.44 on every update and order phrasing; it returns the OLD code 83% of the time after "has changed from OLD to NEW" |
| G0.8 (P66) | rope or ordinal learn ≥ 8/10, beat linear on order; drop if plain < 0.9 | Literal reading (training length): rope, ordinal and the concatenated reference all pass. Strict (every length): **ordinal only** (0.99 at 2× and 4×; rope 0.63 / 0.42) |
| Keep toy (P67) | Learned scorer finds the digits within 300 steps | 7/10 seeds with a strong shared digit direction, 2/10 with a weaker one |

**What it shows**
1. **Long-range failure is the age penalty, at least at evaluation time** (P59, P60).
   - Without it, every I1 seed's oldest facts at 2048 tokens rise to 0.95–0.99, and seed 4's to 0.78. That restores the planners' original diagnosis (round-11 planning entry, point 1).
   - Sharpening the alignment does nothing, so P58's "alignment is the weak link" is not supported. That reading was a correlation across seeds of a trained parameter; an eval-time β shift is not training with a sharper start, so A12 is not ruled out, only unsupported.
2. **Every penalty form trades length against updates.** No fixed form wins both. ordinal time in the CPU toy does win both at every length (P66). That makes A1b (query-chosen ordinal rank) the time arm with the best case.
3. **Full attention is not a ceiling for updates or order on this model** (P61), as RV4.E7 predicted from proactive interference. Read as written, the runbook's update rules cannot apply. RV4.P4 proposes scoring against ground truth with full attention as a reference.
4. **Compression is free at evaluation time** for the 13-bit code. Value-norm top-8 halves the state; the 64-d code cuts it 19×. RV4.E11 cautions that this is measured on a near-zero-entropy value.
5. **Two caveats on the diagnostics themselves.**
   - Keeping only the read's 4 transitions is worse than keeping all 24: the readout was trained with distractor mass, while the alignment is identical (checked: 44/48 sequences align space→d1→d2→d3 in both).
   - The G0.8 gate, read at the training length, cannot separate arms. Every arm, including concatenated time, learns at 1×.

**Decisions for Daniel before Stage A**
1. **G0.8's reading.** Literal: rope, ordinal and concat pass. Strict: ordinal only. Adopting the strict reading is a rule change (DEC).
2. **G0.1 did not pass as worded.** Recency is the lever, alignment is not. Should A12 still run (RV4.P3 promotes it to core, from P58)?
3. **Updates.** Should update rules be scored against ground truth (RV4.P4) now that no form has a ceiling (P61)? The runbook line "the better of log/ord on updates becomes A1" predates revision 1; on updates ord wins clearly (P60).
4. **G0.5** per cell or on average: dc = 64 is recommended.
5. **RV4 §6** asks to close Stage-B gaps first: E8 and the CPU suite for I1; I1-superposed and I1-no-align, 2 seeds each; eval-only paraphrase and repeated-digit cells. It also asks to reorder the arms (generalisation and training signal first, new A13–A15) and for a 10-seed reliability bar.

**Stage A as the rules currently stand:**
- runs: A1b, A2 (supr), A4, A5 (dc 64), A6 (value-norm start), A7, A8 (supr or the best time arm, dc 64, n = 12, K = 64), A10 and A11;
- needs the readings above: A1 and A12;
- dropped: A3 (G0.3) and A9 (G0.7).

Nothing in Stage A starts without Daniel's go-ahead.

**[4 Nov 2026, later] Daniel's readings (DEC10).** G0.8 is read literally, so A1 (rope) runs beside A1b. A12 runs despite G0.1, starting at log β_align = 3.68, the sharpest start below the β ≤ 40 clamp. Updates, dc, RV4 §6 and the go-ahead for Stage A are still open.

**[4 Nov 2026, later] DEC11: relative comparison instead of fixed margins.** Daniel: "Any improvement over base models is an improvement … What matters is the relative difference." From Stage A on, arms are reported against I1, full attention and windowed, with 95% intervals. An arm is better or worse only where its interval separates from I1's. Trade-offs go to Daniel, not to a veto, and updates are scored against ground truth. Deciding cells use n = 384.

**[4 Nov 2026, later] DEC12 (RV4 §6 adopted) and an alignment fix before Stage A.**
- **DEC12:** RV4 §6 adopted. Finalists run 10 seeds, judged at ≥ 8 of 10. Stage-B gaps first, then generalisation arms, then compression, then time. Backlog changes as RV4.P7.
- **The bug:** adding I1 to the CPU suite (`cpu_suite.py --models w1s_i1`) turned up a numerical bug inherited from round 10. When a slot's alignment exp(β(cos − 1)) underflowed, the backward pass of its normaliser (clamped at 1e-30) overflowed to NaN. The run diverged at the learning-rate peak.
- **The fix:** the read now shifts E by each query row's best held match (cancelled by the per-slot normalisation) and floors the normaliser at 1e-12, in both the parallel form and `stream()`. A regression test covers it.
- **Effect on round-10 checkpoints:** re-evaluated, they change by at most 0.005. The 512-token cells are identical; 3–4 of 384 sequences flip at 2048. That is within the runbook's 0.01 compatibility bar, so the Stage-0 diagnostics (old read) stand.
- **Why it matters now:** sharper alignment (A12) and a learned code (A5) make underflow more likely.

---

## Round 11 Stage A: one change at a time on I1, and what survives the seed-4 rescue (5 Nov 2026)

**Setup.**
- 16 arms, each a single change to I1, on seeds 1 and 4: the round-11 runbook arms, the RV4 §6 arms (DEC12), and the Stage-B gap runs.
- Every run is evaluated on 37 cells with windowed, empty and shuffle controls. No evaluation is void.
- Reading (DEC11, `analyse_round.py`): paired differences in exact against I1 with 95% intervals.
  - Seed 1 is compared with I1 seed 1.
  - Seed 4 is compared with I1's learned seeds 0, 2 and 3, because I1 seed 4 failed and almost any arm that trains cleanly "rescues" it (+0.10 to +0.14 on the grid).
- Register P68–P76.

| Arm | Seed 1: grid / names / query / updates | Seed 4 against the learned I1 seeds: grid / names |
|---|---|---|
| **A15** teacher attention target | **+0.024 / +0.047= / +0.076 / +0.004=** | **+0.023 / +0.026=** |
| A11 gated lexical keys | +0.046 / −0.021= / −0.056 / −0.052 | +0.023 / −0.073 |
| A1b ordinal time, no age penalty | +0.020 / −0.010= / +0.045 / **−0.240** | +0.022 / +0.042= |
| A12 sharper alignment start | +0.013 / +0.003= / +0.077 / −0.059 | +0.005 / +0.021= |
| A1 rotary time | +0.021 / −0.091 / 0.000= / −0.104 | −0.018 / −0.193 |
| A7 12-token windows | −0.018 / −0.052= / −0.004= / −0.012= | −0.012 / −0.068 |
| A4 no TimeChannel | +0.006= / +0.023= / +0.095 / −0.014= | −0.218 / −0.125 |
| A6, A5, A8 (compression) | −0.03 / −0.13 / −0.18 on the grid | −0.40 / −0.69 / −0.08 |
| A10 name pool, A13 native keys, A14 distillation | −0.33 / −0.74 / −0.44 | −0.32 / −0.48 / −0.06 |
| A2 supr; I1-superposed; I1-no-align | −0.75; −0.53; −0.82 | −0.70; −0.55; −0.80 |

(= marks a tie; the other values clear 0.)

**What it shows**
1. **A15 is the only arm better or tied on every group on both seeds.** Full attention's focus as the gate's training target improves the grid and paraphrases without costing updates or names. The teacher check shows full attention marks facts against filler (AUC 0.95), which suits an admission signal, but hardly separates the queried fact from the others (0.65).
2. **Most of the large seed-4 gains are rescue.** Against I1's learned seeds the best arms gain +0.02 on the grid. The first reading ("+0.3 on seed 4") overstated them.
3. **Every way of dropping the age penalty buys length and costs updates** (A1b −0.24, A1 −0.10). A11 and A12 also lose some update accuracy.
4. **Held-out names remain unsolved:** the best arms reach 0.70–0.73.
5. **The Stage-B gaps are closed.**
   - Slots and offset alignment are each necessary within I1 (RV4.E1).
   - E8: I1 recovers 41% of the windowed perplexity gap (14.62 → 12.34; full 9.01).
   - I1 passes the CPU must-pass checks.
6. **Compression that was free at evaluation fails when learned from scratch.** Fixed truncation (A7) is near neutral at 53% of the state.
7. **An inherited quirk** (A2 code review, Q1): I1's point gate fires a full-strength onset at position 0 in about 50–60% of sequences, because the running-SD floor is 1e-3, and holds that slot to the end. Its read weight is small (≤ 0.07), so it costs capacity, not accuracy.

**Corrections to the session's interim readings** (from the plan review and the fact-check):
- A1b and A1 lose updates heavily. The interim "no cell worse" was read before the update references existed.
- A11 has a worse cell (paraphrase 4 on seed 1).
- The seed-4 gains are mostly rescue.

**Next: Stage A2 (DEC14).** Phase 0 evaluation-time ablations, the gate diagnostic, Phase 1 I1 retrains (noise floor, current code), Phase 2 decompositions (incl. the hysteresis and Schmitt gates under both calibrations, pointer replay, compression rescues), then the Phase 3 factorial. The tap sweep follows.

---

## Proposal: windowed intent memory as the alignment controller's gain source (4 Nov 2026)

**Motivation.** The A6 abliterated-model sweep (`results/alignment/a6_qwen3_abliterated.json`)
diagnosed why `conditional` and `graded` underperform plain `static`. On the harmful set,
`conditional@0.5` and `graded@0.6` apply near-full gain through essentially the whole response
(`gain_mean` 0.50 / 0.59) yet barely move refusal (harmful 0.09), because by design they leave the
prompt's own encoding alone (`gain_prompt_mean` 0.02 / 0.01) to avoid `static`'s instability.
`static` is the only arm that injects into the prompt encoding, and it is the only one that moves
refusal meaningfully (0.20 at gain 0.2) — at the cost of over-refusing borderline/harmless prompts
(0.13 / 0.16 at that same gain) and degenerating at higher gain (0.37–0.77 by gain 0.4–0.5). The
harm signal driving all of this is also a single last-prompt-token linear probe: a snapshot, not a
trajectory, which cannot by construction catch intent that is built up gradually or spread across
many tokens — the way jailbreaks often "talk around" a harmful topic rather than stating it
plainly.

**Idea (Daniel).** Replace the single-token probe with a windowed, memory-based intent signal,
reusing the side-channel memory modules already in this repo (`HippocampalLoop`/CA3 in
`limbic_llm/temporal_loop.py`, the successor-bound addressing in `gated_delta.py`) rather than
inventing a new mechanism. Window the prompt (and eventually multi-turn history), score each window
for harmful intent with a classifier, and let the side channel's own windowed representation — not
a single last-token snapshot — drive the controller's gain. Two expected benefits over the current
design:
- it targets distributed/indirect intent that a last-token probe cannot see by construction;
- because a side channel can be hooked into the residual stream from the start of the prompt (as
  `pretrained_loop.py`'s side channels already do), gain could ramp into the prompt's own encoding
  as evidence accumulates, rather than switching on only at generation time — a principled route to
  recovering `static`'s effectiveness without its blast radius.

**Dependency.** This explicitly depends on the round-11 (or later) side-memory track landing a
selection mechanism that generalizes. As of round 10 Stage B (P54, P55), I1 is one seed short of
the reliability bar (3 of 5) and fails to generalize to held-out names (0.58–0.74 against
0.85–0.98 on training names) — round 11 is addressing exactly that. This proposal should not start
until that track lands something stable enough to repurpose.

**Risks and open questions.**
- New-mechanism work (a new candidate classifier and a new decision source for the controller) —
  out of the A1/A6 scope this round; needs its own go-ahead and its own build.
- Per `evolution_protocol.md`'s discipline, diagnose before building: a cheap synthetic check
  (does a windowed classifier separate indirect/distributed-intent synthetic prompts better than
  the current last-token probe, on the same prompts) should run before any GPU time.
- The natural home for the "windowed, multi-turn" half of this is **A3** (escalation and recovery
  under multi-turn pressure, H4), which isn't built yet — this may be the mechanism A3's runner
  needs, more than a tweak to A6.
- Data rule applies as everywhere else in this track: stored windowed representations are
  activations and scores only, never raw text.

Recorded as **A9** (REGISTER §4.4).

---

## Alignment track: A1 and A6 on Qwen3-1.7B (5 Nov 2026)

Run on a local RTX 4070 SUPER (12 GB), not geof — a separate session's machine, with no CUDA
GPU of its own in the task as originally written; adapted to run locally with Daniel's approval.
Single seed throughout (seed 0); per protocol, treat as provisional until a deciding comparison is
rerun at `--seed 1`.

**Layer choice.** A1's by-layer sweep (`--layer-sweep`) saturates to prompt-token AUROC ≈1.0 by
layer 2–4 on all three models and stays there through layer 26, so there is no AUROC-based reason
to prefer a later layer. Kept the default, `n_layers // 2` = **block 14** of 28, matching N3's
earlier spot-check choice.

**A1 results** (held-out probe AUROC/ECE at block 14; `results/alignment/a1_*.json`):

| Model | prompt AUROC | prompt ECE | gen-token AUROC (in-dist) | gen-token AUROC (XSTest) |
|---|---|---|---|---|
| `mlabonne/Qwen3-1.7B-abliterated` | 1.000 | 0.0004 | 0.851 | 0.680 |
| `Qwen/Qwen3-1.7B` (aligned) | 1.000 | ~0 | 0.829 | 0.710 |
| `Qwen/Qwen3-1.7B-Base` | 0.999 | 0.025 | 0.611 | 0.511 (chance) |

Harmfulness stays linearly decodable at the prompt token after abliteration, indistinguishable
from the aligned model (**A6.H1 supported**). The base model is the outlier on generated-token
calibration, barely above chance on XSTest — consistent with it never having had a reason to keep
"this is harmful" legible through its own (frequently degenerate) continuations.

**References** (`ref_qwen3_aligned.json`, `ref_abliterated_none.json`): the aligned model refuses
0.77 harmful / 0.54 contrast / 0.05 borderline / 0.00 harmless (ppl 19.92) — the target behaviour.
The abliterated floor is 0.02 / 0.01 / 0.00 / 0.00 (ppl 20.38) — nearly identical perplexity to the
aligned model, so abliteration does not touch the capability floor, only the refusal behaviour.

**A6 results** (full sweep: `static`, `static_random`, `conditional`, `graded`, `graded_self`;
gains 0.1–0.5, caps 0.2–0.6; n-eval 100, 96 new tokens; `a6_qwen3_abliterated.json`,
`a6_qwen3_base.json`). None of the 50 swept configs (25 per model) reach `--target-refusal` 0.8, so
RV1.D1's matched-rate comparison has no crossing point to report on either model; peaks instead:

| Model | Peak harmful refusal | Arm | Degeneration there | Direction cos to self |
|---|---|---|---|---|
| Abliterated | 0.20 | `static@0.2` | 0.00 | 0.603 |
| Base | 0.19 | `static@0.2` | 0.60–0.87 | 0.059 |

**What it shows.**
1. **Nothing comes close to the aligned reference** (0.20–0.19 of 0.77) at any swept gain/cap, on
   either model.
2. **The mechanism behind why the controllers underperform `static`.** On the harmful set,
   `conditional@0.5` and `graded@0.6` apply near-full gain through essentially the whole response
   (`gain_mean` 0.50 / 0.59, abliterated) yet barely move refusal (0.09), because they deliberately
   leave the prompt's own encoding alone (`gain_prompt_mean` 0.02 / 0.01) to avoid `static`'s
   instability. `static` is the only arm that injects into the prompt encoding
   (`gain_prompt_mean` ≈ its nominal gain), and the only one that moves refusal meaningfully — at
   the cost of over-refusing borderline/harmless prompts (0.13 / 0.16 at gain 0.2) and degenerating
   at higher gain (0.37–0.77 by gain 0.4–0.5). **A6.H2 is not supported as built**: the proposed
   graded controller is safer than static addition, as hoped, but weaker, not stronger.
3. **Direction-specificity is real but the destructive mode is not.** `static` beats its
   matched-norm control `static_random` at low–mid gain on the abliterated model (0.20 against 0.02
   at gain 0.2), but at high gain both collapse into degeneration at similar rates (static_random
   harder: 0.86–1.00 against 0.37–0.77) — large enough perturbation breaks the model regardless of
   direction.
4. **The base model's near-identical peak number (0.19) is not comparable to the abliterated
   model's (0.20).** It rides on 0.60–0.87 degeneration (against 0.00–0.01), i.e. it is mostly noise
   over collapsed generations. The causal reason is geometric: the aligned sibling's refusal
   direction has cosine similarity 0.603 with the abliterated model's own harm direction but only
   0.059 (orthogonal) with the base model's — a model that never learned to refuse has no
   refusal-adjacent direction for the borrowed vector to align with. **A6.H3 supported.**
5. **A6.H4 stays Open.** A6 is single-turn only; the multi-turn escalation/recovery claim needs
   A3's runner, which isn't built.
6. **Sanity checks all passed**: `none` matches `ref_abliterated_none` exactly on both runs;
   `conditional`'s `ppl_gain_mean = 0.000` on the base model's WikiText pass was investigated and is
   not a bug — the controller stats on the harmful set show it firing normally there (`gain_mean`
   0.10–0.49 across the sweep); the probe correctly never judges benign WikiText as harmful enough
   to trigger.

**Bug found and fixed along the way**: `layer_sweep` (`run.py`) was wrapped in a redundant and
broken `@torch.no_grad()` that disabled autograd for the probe's own training loop, so
`--layer-sweep` crashed every time (untested by the 44-test suite). Fixed, with a regression test
added (suite is now 45 tests). Also added `accelerate` to the `pretrained` extra
(`pyproject.toml`) — newer `transformers`' `device_map="auto"` path needs it; CI is unaffected (it
never installs the `pretrained` extra).

**Follow-on**: opened **A9** (above) — a windowed, memory-based intent signal (reusing this
repo's own side-channel modules) as a candidate fix for finding 2, blocked on the round-11
side-memory track landing a generalizing mechanism.


---

## Round 11 Stage A2, phases 0–2, and the tap sweep: no single change beats I1; the point gate wins; tap 7 and a fitted code are the useful finds (6 Nov 2026)

**Setup** (DEC13, DEC14):
- **Phase 0:** evaluation-time ablations, and a gate diagnostic on real states.
- **Phase 1:** I1 retrained under the current code: seeds 1–4, plus seed 4 with another initialisation.
- **Phase 2:** 13 decomposition and gate arms, and the six top Stage-A arms on seed 2.
- **Tap sweep:** odd blocks 1–27, needle on seeds 1 and 4, and E8 for I1 and the O(T) loop.
- **Reading:** `analyse_round.py`, at the recipe level: three seeds, with intervals that include run-to-run noise (s_seed from the five I1 runs).
- **Void:** 2 evaluations (P89). Register P81–P89.

| Finding | Numbers | Register |
|---|---|---|
| **Which seed fails is luck.** Retrained under the current code, I1 seed 4 learns and seed 3 fails | grid 0.967 / 0.932 / **0.830** / 0.956 / 0.947 | P81 |
| **No single change beats current-code I1** on grid, names or paraphrase once run-to-run noise is counted | all top arms tie, \|Δ grid\| ≤ 0.04 | P85 |
| **The point gate is the best gate.** Hysteresis and Schmitt lose even with calibration | grid −0.25 to −0.85 | P83, P84 |
| **The age penalty is a trade-off**, not a bug: removing it buys more facts and length, costs updates | updates −0.14 / −0.17; robustness +0.25 | P85 |
| A11's lexical key is nearly unused; A1b's rank adds nothing | ablations ≈ 0 | P82 |
| **A fitted 64-d alignment code (C5r)** ties I1 at 19× less memory, and helps updates | updates +0.085; state 47k against 907k | P86 |
| **Tap:** 7, 17, 21, 23 and 25 work on both seeds; tap 7 also has much better perplexity | E8 11.55 against 12.35 at tap 21 | P87, P88 |
| **E8 met:** I1 is no worse than the O(T) loop at the same tap (3–25) | tap 21: 12.35 against 12.70 | P88 |

**What it shows**
1. **Run-to-run luck, not the recipe, now dominates.** Stage A's large gains were partly rescue and partly a code-version difference in the references. Against current-code I1, the best single changes are within noise (about ±0.03).
2. **Reliability is the open problem.** About 1 run in 5 fails (P81), and no single change visibly fixes that. Phase 3's factorial pools 48 runs and is the right instrument for small effects. Stage B (10 seeds) measures reliability directly.
3. **"Point gates seem fragile"** (Daniel): the point gate fails about 1 run in 5, but the latch gates fail more, on every seed. The gate fragility is a training-trajectory problem, not the gate's form.
4. **Two useful levers came out of this phase:**
   - the fitted-code start, which shrinks the state 19× with no loss;
   - tap 7, with recall as at tap 21 and better language modelling.

**Next.** Phase 3 (the factorial, DEC14). Phase 0–2 results trim the factor list, and add two candidates (the fitted code; tap 7 against 21) that need Daniel's approval: see the checkpoint report.

## Round 12: a training-free CA2 side channel shows no benefit under full attention; read-back on a windowed model is the one positive (9–11 Nov 2026)

**Setup** (DEC15, re-planned by DEC16). New branch `feat/round12-ca2-gdn`. Code in `experiments/ca2/`; results in `results/evolve/round12/stage0/` and `stage1/`; raw per-token traces in `~/limbic-data/ca2_traces/` (not in the repo). Models: Qwen3-1.7B(-Base), the hybrid Qwen3.5-2B(-Base), RecurrentGemma-2b(-it). Nothing was trained except the P5 replica (P99). Everything is one run unless stated. Register P90–P101.

- **Stage 0** tested Activation Addition made dynamic: a controller adds g·‖h‖·d at one block, with d the "gist" (a fast running average of the stream minus a slow one) and matched controls, plus branch gains on the hybrid, lagged cross-layer injection and multi-site injection.
- **After Stage 0** Daniel re-planned (DEC16): cancel what cannot show a benefit, cast windowing correctly, allow training if reportable, measure tasks. Stage 1 then ran thinking baselines and controllers, the P5 adjudication, read-back on RecurrentGemma and the CA2 "finish thinking" injection.

| Finding | Numbers | Register |
|---|---|---|
| **Under full attention nothing helps**, on any tracked metric, across about 700 arms | best effects are 0.1% of the loss and not distinguishable from a random direction | P91, P92, P95, P96 |
| **The gist hurts everywhere** and drives repetition; its negation is the better direction under a window | hybrid, gain 0.3: +0.069 full, +0.198 windowed; anti-gist −0.021 to −0.030 with everything windowed | P91–P93 |
| **The hybrid's GDN state carries context, not items** | zeroing it under a 64-token window: +0.32 nats; recurrent-only digit accuracy 0.09 (chance 0.10) | P90 |
| **Rebalancing branches repairs the windowed hybrid** | attention × 0.75: −0.26 nats (48% of the gap), top-1 +3.6 points; hurts with GDN reset or full attention | P94 |
| Lagged injection helps only under a window, deep-to-early best | block 19 → 3, out-of-window mean: −0.038 | P95 |
| Splitting the dose over sites does not reduce repetition at matched effect | — | P96 |
| **The hybrid's thinking rarely ends**; vendor decoding does not fix it | GSM8K: 0.18 thinking, 0.78 without; presence penalty 1.5: 0.31 | P97 |
| No stop rule beats a fixed thinking budget; an oracle shows 23 points of headroom | cut at 256 tokens: 0.835 at 883 tokens (hybrid) | P98 |
| **The P5 loop was adaptation, not memory** | a matched memory-free adapter beats it: −1.9 and −7.3 closure points | P99 |
| **Read-back through attention works with no training when an oracle selects**, with ordinal positions | RecurrentGemma, beyond the window: 0.08 → 0.94; true positions 0.26; InfLLM-style selection 0.06 | P100 |
| A "finish thinking" injection is a real lever but loses to the fixed cut | hybrid 0.61–0.65 against 0.835; random 0.205, negated 0.00 | P101 |

**What it shows**
1. **Re-adding a model's own state to itself gives it no new information.** With the context in view there is nothing to gain; the benefits appear only where information is missing, and the large one (P94) is partly repair of a condition we imposed.
2. **Content needs a reader.** Attention read-back delivers content (P100); residual addition does not. Ordinal position works where true rotary distance does not, as P66 suggested.
3. **Low-pass state and discrete entries do different jobs.** Running averages and the GDN state carry context (P90, P93); only stored entries carried items (P100, P51).
4. **Trained side modules need two controls** from now on: a parameter-matched adapter with no memory, and another document's memory (P99).
5. **Selection is the open problem.** The oracle supplies it in P100; the training-free retriever does not.

**Problems met.** GPU 0 has about 23 GB usable (the desktop holds the rest): several runs failed on memory there and were rerun at smaller batches. `reason.py` rejects `trunc@B` arms (a bug in `resolve()`), so the baseline files hold only the `none` and `nothink` arms. The hybrid multi-site run was stopped at 160 of 209 runs. The free-generation collapse measure has a noise floor of about 0.1.

**Next.** Round 13 (DEC17): the gated memory circuit, `plans/round13_gated_memory_plan.md`, a draft under adversarial review.

[Correction, 11 Nov 2026: the adversarial review of the round-13 plan (register RV5) narrows P100 and point 5 above. The oracle reads back every statement and never uses the question, and the InfLLM-style arm is at chance through faults in our implementation, so "selection is the open problem" is not established: a plain selector has not been given a fair test. The 0.08 → 0.94 figure is the needle cell on the base model; pooled over needle and update it is 0.10 → 0.70 (base) and 0.14 → 0.72 (-it). The update cell on the base model (0.46) missed its gate, and the inside-window check failed (native 0.65 on update items with the facts in view). The review's verdict on draft 1 of the plan was "do not run as written"; draft 2 answers its findings and is not approved or built.]

## Round 13 step 0b: the circuit's selectors lose to a plain retriever; the comparator and the time question survive (12 Nov 2026)

**Setup** (DEC17–DEC19; plan `plans/round13_gated_memory_plan.md`, draft 2). Code: `experiments/ca2/circuit.py`, `items13.py`, `selectors13.py`, `rank13.py`. Ranking only, one forward pass per item, no read-back and no generation. RecurrentGemma-2b and -2b-it, 8,192-token inputs, 24 same-form statements per item with one queried, item types plain, decoy, jitter, paraphrase and update, 400 per type, half dev and half held out. Register P102.

**What happened on the way.**
- An 8-item smoke showed that 512 columns hold all 24 statements, so recall at that budget measures form detection. The primary measures became top-1 and MRR before any real run (DEC18).
- A code review (RV6) found that the comparator received an unnormalised state, and that the dentate salience is constant by construction (layer norm, then a Gaussian projection). The first runs were stopped unread and step 0b reran the same items with the fix, five other saliences, query-key variants that read at the answer position, per-layer and per-head families and a position reference (DEC19).

**Results** (held out, pooled decoy + paraphrase, top-1; base / -it).
- `bge-small` 0.89 / 0.89; Qwen3-Embedding-4B 0.97 / 0.97.
- The model's best single attention head 0.52 / 0.43; the circuit's dev-chosen scorer (early-tap cosine) 0.32 / 0.47; dentate key + context tag 0.07 / 0.09; the repaired InfLLM-style arm 0.01 / 0.03.
- Every model-internal key is lexical: 0.70–0.97 on same-wording items, 0.05–0.20 on paraphrase.
- Update items: retrievers rank the superseded statement above the current one in 75–87% of items.
- Parts: comparator AUC 0.84 / 0.74; context tag 0.77 / 0.75 (distance-matched); novelty salience 0.95 / 0.93 but confounded by insertion; dentate strength at chance.

**Reading.**
1. The dentate sparse key is not a usable address. With an exact store, a random sparse code can only add noise to the similarity it is derived from.
2. "What" matching from a verbal cue is the home ground of a trained text encoder. Losing it says little about the rest of the circuit.
3. The comparator works as a signal of whether a recall is right, even over a poor selector.
4. Time is open: content matching cannot tell the current value from the superseded one.

**Process note.** The 40-item smoke (90 seconds) already showed the ordering that 2,000 items per model confirmed; about five GPU-hours went on confirming a gap of 0.4–0.6. Daniel's comment, 12 Nov: "It seems as if you are spending an inordinate amount of time on obviously failing experiments." From here: cache activations once, screen each new part on about 50 items with a kill rule written first, and keep full-size runs for survivors.

**Literature review** (three Fable research agents, 12 Nov; `reviews/2026-11-12_hippocampus_models/`). The dentate code is for separating memories at write time, not for matching cues; no hippocampal model is expected to beat a trained encoder at single-fact cued recall (HippoRAG 2 reaches parity there); the expected wins are time and order, updating, near-duplicate events, novelty-gated storage and rejecting wrong recalls. Published retrievers are roughly indifferent between current and superseded facts, so our update result (stale statement first in 75–87% of items) is probably an item artefact: the updating statement alone contains "now" and is one token longer (RV6.3).

**Next (proposed, DEC20).** Round 14, `plans/round14_plan.md`: tests where content matching is at chance by construction (update, context binding, rejecting recalls, write gate on natural text), run as a screen-first loop on cached activations. Round 13's steps 1–3 are not run as planned. Awaiting Daniel.

