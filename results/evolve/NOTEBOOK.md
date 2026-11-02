# Evolution lab notebook

This is an AlphaEvolve-style search over the side channels: propose changes, score them on the GPUs, keep the best.
A human checkpoint follows every round. Each entry records what was tried, the scores, the winning diff, any surprises, and the proposed next step.
Round 0 is the current code (`limbic_llm/temporal_loop.py`, `experiments/pretrained_loop.py`) as it stood at commit `b39abab`.
The project's history is in [`docs/genesis.md`](../../docs/genesis.md).

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
