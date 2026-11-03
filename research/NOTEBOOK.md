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
