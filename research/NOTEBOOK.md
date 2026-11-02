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
- Prefill followed by token-by-token decoding matches a single pass, for every mode and both directions.
- The graded gain is capped, escalates under sustained score and decays when it stops.
- The conditional arm decides on the prompt.
- Left padding does not change real positions, and bf16 input comes back as bf16.
- The gist direction follows the recent average.
- The probe and judges behave on synthetic data.
- An end-to-end run writes labels only.

**Not yet run on real models.** Next on the GPU machine: A1 (a layer sweep on the abliterated model), then A6 against the aligned and base references. Commands are in the track README. The S1 saturation knee should set `--cap`.

[Review, 2 Nov 2026: the review RV1, [`reviews/2026-11-02_alignment_cot_ca1.md`](reviews/2026-11-02_alignment_cot_ca1.md), found problems in this build. RV1.A1 (High): the perplexity check never measures the conditional, graded or graded_self arms, so the capability metric is vacuous for them. RV1.A2: the probe is calibrated on the last prompt token but integrated over generated tokens. RV1.A3: degenerate loops count as refusals. RV1.A6: `--name` can bypass the data rule. A2 and A6 are blocked until these are fixed; A1 can run with the RV1.A2 caveat. The review's file:line anchors refer to commits 73259fc and 241a13c.]

---

## Proposal: CA2 beyond alignment, steering chain-of-thought (2 Nov 2026)

**Idea (Daniel).** Chain-of-thought steers a model towards a correct answer by giving it extra context it generates itself, and it improves answers substantially. Additive activations could be a shortcut, "almost like momentum in training": selectively boosting thoughts along the current thought process so the model converges on a better answer sooner. If so, CA2 could cut chain-of-thought token counts at similar accuracy, lowering inference cost and latency. Daniel also asked that CA2 work on a windowed view of the stream (exponentially averaged, autoregressive) so that a general idea is boosted.

**How CA2 is modelled today.**
- The temporal loop's injection $\Delta_t = F_t - m_t\mu_t$ is a damped, relevance-weighted average of *older* content. It excludes the most recent τ tokens and subtracts the long-run mean, so it boosts what is distinctive about the past, not the gist of the current thought.
- The alignment controller averages the harm *score* but steers along a fixed direction.

**Added.** `direction="gist"` in `SteeringController`, a momentum direction from two exponential averages of the stream before injection:

$$m_t = \beta m_{t-1} + (1-\beta)h_t,\qquad b_t = \beta_s b_{t-1} + (1-\beta_s)h_t,\qquad \hat d_t = \frac{m_t - b_t}{\lVert m_t - b_t\rVert}$$

The averages read the stream before injection (open loop) and the gain is capped. Reinforcing what is already in the stream is positive feedback, so the gist arm belongs in stability experiment S1.

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

**What would be new.** Fixed contrastive vectors already cut reasoning tokens by 10–70%. Feedback-controlled steering and memory-retrieved steering exist separately. No paper found builds the steering signal from the model's own running average of its current line of thought and regulates it with negative feedback, still less together with memory retrieval. **The main risk** is over-commitment: boosting the current line can suppress useful backtracking, and ThinkEdit shows that thinking which is too short costs accuracy.

**Experiments** (R1–R5 in the register)

[Review, 2 Nov 2026: RV1 ([`reviews/2026-11-02_alignment_cot_ca1.md`](reviews/2026-11-02_alignment_cot_ca1.md) §2) blocks R2 and R3. RV1.C1: the gist direction is dominated by the attention-sink token and the stream mean. RV1.C2: R2 and R4 lack matched-norm random, mean and sign-flipped-gist controls, a seed protocol and a non-inferiority margin. RV1.C3: graded gist is silently a no-op without a probe, and no gain source exists for R3's agreement signal. RV1.C4: STU-PID, the R3 baseline, was withdrawn; use PID steering (arXiv 2510.04309). RV1.C7: R1's direction should be taken at matched content positions.]

- **R1 Thinking direction.** On Qwen3's hybrid checkpoints, take the difference between activations with thinking on and off for the same prompts. Inject it in no-think mode and measure accuracy against tokens.
- **R2 Gist steering during thinking.** Sweep the gain and plot accuracy against mean reasoning tokens on GSM8K, MATH-500 and AIME 2024/25. Baselines: full thinking, `/no_think`, budget truncation, SEAL, and the concise-vector method.
- **R3 Regulated gist.** A graded gain driven by agreement between the gist and the current token (CA1): boost while consistent, release on backtracking. Compare with STU-PID-style control, and run it through the stability programme.
- **R4 Over-commitment checks.** Flip analysis (problems the baseline solves that steering breaks, and the reverse), the rate of backtracking words such as "wait", and GPQA-Diamond as an out-of-domain check, with wall-clock time and tokens-to-correct.
- **R5 Memory-retrieved gist.** Store gist vectors from successful traces in a hippocampal memory and retrieve by similarity. This is the hippocampal version of STIR and DSEM.

**Models:** R1-Distill-Qwen-1.5B/7B and Qwen3-1.7B/4B (hybrid thinking). R1–R4 can reuse `experiments/alignment/steering.py`; a reasoning runner is not built yet.

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
- **J2 LAYA-style CA1 comparator for the loop.** A small cross-attention encoder over the recent stream window and the CA3 retrieval, outputting injection strength, write/no-write and replay priority with confidence. The training target is the counterfactual benefit of injecting, the next-token loss with and without the injection. Test on the what-when task, the recency probe and the stability suite, then the Qwen needle.
- **J3 Revisit M2.** A calibrated CA1 (proper scoring, warm-up) as the novelty signal for Titans' learning rate. Does calibration remove the harmful interaction? [Review, 2 Nov 2026: blocked by RV1.D3. The premise is corrected above; warm-up alone is now its own arm, J3.a; and a calibration measure for the novelty target, a regression target with no binary event, has to be defined. RV1.D1 and RV1.D4 also block J1.]
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
