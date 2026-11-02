# Results (CPU, single seed, small models)

All runs: $d_{\text{model}} = 64$, 4 heads, batch 32, AdamW with one-cycle LR at a peak of $3\times10^{-3}$, seed 0. Raw logs and JSON curves are in `results/`. These are small, single-seed sanity experiments, not benchmarks; differences of a few points are within noise.

## 1. One-shot associative recall across a gap

Each sequence is: study 6 random key→value pairs, then 40 filler tokens, then query all 6 keys in shuffled order. Only recalled values are scored, so chance is $1/16 = 0.0625$. Local models (windowed attention, Strategy I segments, Titans chunks and windows) see 16 tokens at a time, so the study pairs are out of their direct view at query time. Training ran for 3,000 steps.

| Model | Params | Recall (final) | Notes |
|---|---:|---:|---|
| `full_attn` (reference, sees everything) | 221 k | **0.485** | Upper reference |
| `window_attn` (16-token window) | 221 k | 0.059 | Chance: no memory |
| `hippocampal` (Strategy I) | 329 k | **0.414** | DG → sequential CA3 store → CA2 → CA1 → Sub |
| `titans_mac` (plain Titans) | 238 k | **0.483** | After the loss-scale and meta-gradient fixes |
| `titans_seq` (+ sequential writes) | 238 k | **0.490** | |
| `titans_seq_dg` (+ DG sparse writes) | 346 k | 0.478 | Ablation |
| `titans_seq_ca1` (+ CA1 comparator, novelty-lr) | 322 k | 0.440 | Ablation |
| `titans_hippo` (all of the above) | 429 k | 0.172 | Still rising at 3k steps (0.11 → 0.17) |

**What this shows**

* **Both memory systems work.** Strategy I and Titans recall bindings from well outside their 16-token attention, while windowed attention with the same reach stays at chance. Titans matches full attention.
* **Strategy I is close** (0.41 against 0.49). Its CA3 is a discrete, novelty-gated episodic store written once per segment. That makes it inspectable, since you can see which events were stored, and it uses no test-time gradient computation.
* **The hippocampal add-ons are neutral alone and harmful together in this regime.** DG writes alone and CA1 + novelty-lr alone are each within a few points of plain Titans. Combining them slows optimization badly. A plausible cause is that the novelty-scaled learning rate is noisy early, when the CA1 signal is still untrained, and DG sparsification shrinks the effective key dimension at the same time. Warming up `novelty_lr`, or stopping gradients through novelty, are the next things to try.
* **Not a task ceiling.** Full-attention accuracy by query position runs 0.26, 0.27, 0.31, 0.37, 0.54, 0.82, so the ~0.49 plateau shared across models reflects small models and short training.

## 2. Continual learning with replay consolidation

Three tasks, each with its own fixed key→value facts, are learned one after another for 600 steps each, using Strategy I. Only the first occurrence of a key in a sequence is scored, so the facts must live in the weights.

| Condition | Acc after T0 (T0, T1, T2) | After T1 | After T2 | Avg final | BWT |
|---|---|---|---|---:|---:|
| Naive sequential | 1.00, 0.13, 0.10 | 0.35, 1.00, 0.14 | 0.21, 0.33, 1.00 | 0.513 | −0.731 |
| Novelty-prioritized replay + sleep | 1.00, 0.13, 0.10 | 1.00, 1.00, 0.14 | 1.00, 1.00, 1.00 | **1.000** | **+0.000** |

Naive fine-tuning forgets catastrophically (BWT −0.73). Replay of novelty-prioritized traces, with logit distillation and an offline sleep phase on the slow parameters, removes the forgetting entirely on this task. The task is easy, and this run doesn't separate the contribution of the sleep phase from that of interleaved awake replay. That would be the next ablation.

## Bugs the experiments caught

The first benchmark run had every memory model at chance. Fixing that turned up three issues (see README, "Design decisions the experiments forced"):

1. The Titans meta-gradient flowed through the read path, so the training-mode update didn't match the inference-mode update.
2. The memory loss was averaged over features, so writes were scaled down by $1/d$.
3. The CA3 store was purely autoassociative, so a cue returned itself instead of its successor.

## 3. Gated DeltaNet capacity and Titans with a delta-rule memory

Associative recall with 64 possible keys and a 40-token gap, 1,500 steps for the capacity sweep and 3,000 for the Titans runs. Recall is the fraction of queried values predicted correctly, so chance is $1/16 = 0.0625$.

**Capacity sweep (pure Gated DeltaNet, no attention)**

| Mode | Key dim per head | 8 pairs | 16 pairs | 32 pairs |
|---|---:|---:|---:|---:|
| `gdn` (standard) | 16 | 0.426 | 0.134 | 0.137 |
| `gdn_x8` (standard) | 128 | 0.428 | 0.320 | 0.222 |
| `tied_x8` (dense address, successor binding) | 128 | **1.000** | **1.000** | **1.000** |
| `dg_x8` (DG k-WTA address, successor binding) | 128 | **1.000** | **1.000** | **1.000** |

**Titans MAC with a Gated DeltaNet memory** (same recall task as section 1, 6 pairs)

| Model | Recall at step 500 | Final recall |
|---|---:|---:|
| `titans_mac` (deep Titans memory, section 1) | 0.484 | 0.483 |
| `titans_gdn` (linear delta-rule memory) | 0.680 | **1.000** |
| `titans_gdn_dg` (+ DG-sparse addresses) | 0.905 | 0.989 |

**What this shows**

* **Standard Gated DeltaNet runs out of capacity as predicted.** With 16-dimensional keys, recall falls from 0.43 at 8 pairs to about 0.13 at 16 and 32 pairs. Widening keys to 128 dimensions helps (0.32 and 0.22) but doesn't fix it, because the model must also learn to bind each key to the next token through its short convolution.
* **Successor-bound addresses solve the task outright.** When the address of token $t-1$ is the key for value $t$, the memory behaves as RAM and recall is perfect up to 32 pairs. A counterfactual check confirms this is genuine retrieval: after swapping the values shown at study time, predictions follow the new values 100 % of the time and the old values 0 %.
* **DG versus dense addressing is not separated by this task.** Both hit the ceiling. With 128-dimensional addresses and distinct keys, dense addresses barely interfere. Testing pattern separation needs similar keys (shared prefixes or noise) or loads above $d_k$.
* **An exact delta-rule memory beats the deep Titans memory here.** Swapping Titans' chunk-start gradient update for the exact per-token delta rule takes recall from 0.48 to 1.00. DG addresses reach high recall faster (0.905 vs 0.680 at step 500) but finish slightly lower (0.989 vs 1.000), so their advantage here is sample efficiency, not final accuracy.

## 4. Temporal loop: what, when, stability and feedback

`TemporalLoopLM`: token embedding, a windowed attention layer (16 tokens), the time channel and the hippocampal loop, a second windowed layer, and the output head; $d_{\text{model}} = 64$, about 0.11 M parameters, no position embedding. Loop period $\tau = 16$, attenuation $\rho = 0.7$. The *what-when* task places one event (8 types) 20–140 tokens before a query; the model must name the event (*what*, chance 0.125) and the elapsed-time bin (*when*; the four bins cover 15, 25, 40 and 40 of the 120 gaps, so the prior alone scores 0.333). Two windowed layers reach $2(W-1) = 30$ tokens, so about 8 % of training gaps are within attention's reach; windowed attention alone stays at chance, so it does not exploit this. Long rollouts run on 520-token sequences with gaps up to about 500. "Echo" is $\lVert\text{injection}\rVert / \lVert h\rVert$. Seed 0 unless noted; a rerun of the full design reproduced its seed-0 numbers exactly, so training is deterministic. Design: [`docs/design/temporal-loop.md`](docs/design/temporal-loop.md).

### 4.1 Which channel carries which answer (1,500 steps)

| Model | What | When |
|---|---:|---:|
| Windowed attention only | 0.109 | 0.338 |
| Memory loop (open, clamped sinusoid, CA1 control) | **0.996** | 0.852 |
| Time channel only | 0.102 | **0.883** |
| Memory loop + time channel | 0.102 | 0.885 |
| Memory loop + time channel, 3,000 steps | **1.000** | **0.922** |

The loop carries content and the clock carries elapsed time; each alone leaves the other answer at baseline. The combined model learned *what* only with the longer schedule (by step 500), so its 1,500-step failure was the timing of a late, sudden jump, not a missing capability. Recall falls with distance as damping intends: for the open memory loop, 1.00 within the trained gaps, 0.61 at 140–300 and 0.17 at 300–500.

### 4.2 Closed-loop stability

Closed loops write their own injected stream back into memory.

| Closed loop | Steps | What | When | Echo 16–128 | Echo 128–320 | Echo 320–519 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Undamped (`flat`) | 3,000 | 1.000 | 0.982 | 4.25 | 9.50 | 14.0 | 63.4 |
| Damped (`exp`) | 3,000 | 1.000 | 0.988 | 1.00 | 0.22 | 0.05 | 10.6 |
| Clamped sinusoid (`osc`), no control | 1,500 | 0.100 | 0.963 | 3.76 | 2.72 | 3.04 | 12.9 |
| **Clamped sinusoid + CA1 control**, seed 0 | 3,000 | **1.000** | 0.977 | **0.33** | **0.35** | **0.35** | **0.53** |
| **Clamped sinusoid + CA1 control**, seed 1 | 3,000 | **1.000** | **1.000** | **0.29** | **0.31** | **0.31** | **0.44** |

Only the full design keeps the echo below the new input everywhere, and it is stationary across the run. Undamped feedback grows with sequence length; damping alone settles late but overshoots in the first circuits; the clamped sinusoid without control resonates at 3–4×. None diverged numerically, because the next layer's LayerNorm rescales the stream: runaway shows up as echo outweighing input, not as overflow. The trade-off is forgetting. At gaps of 300–500, recall is 0.40 undamped, 0.00 damped and 0.04–0.10 for the full design.

### 4.3 CA1 feedback as an op-amp

`--ca1-control` bundles a hard clamp that bounds the injection by construction, so the feedback was tested on its own (`experiments/loop_systems.py`): an untrained closed loop with the clamp, projection and saturation removed, swept over forward gain $A$.

**Subtractive feedback obeys the op-amp law on persistent echo** (one repeated token, damped kernel, no contrast):

| $K = Ag\beta$ | 0.27 | 0.54 | 1.08 | 1.61 |
|---|---:|---:|---:|---:|
| Subtractive / no feedback | 0.787 | 0.647 | 0.475 | 0.373 |
| $1/(1+K)$ | 0.788 | 0.650 | 0.482 | 0.383 |

On a varied stream it barely changes the echo, because a lagged, smoothed copy can only cancel what persists: it is a high-pass filter. Because the copy arrives a circuit late, the loop is stable only while $z^{\tau} - \rho_E z^{\tau-1} + K(1-\rho_E)$ keeps its roots inside the unit circle, which for $\tau = 16$, $\rho_E = 0.9$ means $K < 1.68$ (ringing period 44 tokens). The sweep confirms it: stable at $K = 1.08$, ringing and growing at $K = 2.15$ and above, with both kernels. Divisive feedback on energy only compresses: with the undamped kernel at $A = 32$, the echo is 0.69 against 2.39 without feedback, growing sublinearly in $A$.

**Trained, without the clamp** (3,000 steps):

| Feedback | What | Echo 320–519 | Max | Probe: newer / older |
|---|---:|---:|---:|---:|
| None | 0.133 | 2.16 | 7.2 | 0.11 / 0.09 |
| Divisive | 0.152 | 2.99 | 189 | 0.11 / 0.13 |
| Subtractive (inverted copy) | **0.990** | 2.60 | 10.3 | 0.05 / 0.54 |
| Full design (clamp + divisive + projection + saturation) | **1.000** | **0.35** | **0.53** | 0.25 / 0.18 |

Subtractive feedback is the only unclamped loop that learned content, and it keeps recall at long gaps (0.45 at 300–500). It does not bound the echo: at a stable loop gain ($K \approx 0.33$ here) it removes only about a quarter of sustained echo. A guaranteed bound still needs the clamp.

### 4.4 Do new tokens dominate?

A trained recency task (an older, different event precedes the target) was learned by none of the three closed loops in 3,000 steps (*what* 0.14–0.15), so it cannot rank them. Instead, each model trained on single events was probed zero-shot with an older event 20–60 tokens before the target.

| Model (3,000 steps) | Names newer | Names older |
|---|---:|---:|
| Undamped closed loop | 0.25 | 0.61 |
| Damped closed loop | 0.06 | **0.84** |
| Subtractive feedback, no clamp | 0.05 | 0.54 |
| Full design without contrast | 0.32 | 0.63 |
| Full design | 0.25 | 0.18 |
| **Full design, DG write keys, direct read** | **0.68** | 0.31 |
| **Full design, DG write keys, DG read** | **0.68** | 0.29 |

**Closed loops favour the first event.** A closed loop re-stores its own echo every circuit, and each re-stored copy starts its decay again, so damping each trace does not damp content the loop keeps rehearsing: the older event, rehearsed longer, wins. This is primacy by self-rehearsal, the mechanism psychology proposes for the primacy effect. The episode-minus-prior contrast reduces it (0.63 → 0.18 older), plausibly because rehearsed content becomes part of the prior $\mu_t$ and is subtracted. Sparse dentate-gyrus write keys give the only clear recency win, presumably because the two events no longer collide in memory. The subtractive inverted copy did not break primacy, contrary to the high-pass prediction, for the gain reason in 4.3.

### 4.5 Read path and cue noise

Dentate-gyrus write keys, with the retrieval cue read either directly from the stream (entorhinal layer II → CA3) or through the same DG code; relative Gaussian noise on every read cue at evaluation.

| Model (3,000 steps) | σ = 0 | 0.5 | 1.0 | 2.0 |
|---|---:|---:|---:|---:|
| Dense keys, no contrast | 1.000 | 1.000 | 0.996 | 0.969 |
| DG keys, direct read | 1.000 | 1.000 | 0.957 | 0.840 |
| DG keys, DG read | 1.000 | 1.000 | 1.000 | 0.941 |

The Treves–Rolls prediction, that a direct dense cue completes degraded patterns better than a pattern-separated one, is not supported here (one seed). The k-WTA read acts as a threshold that removes noise, and with 4× expansion and 10 % sparsity this DG separates patterns only mildly.

### 4.6 Learning speed

*What* is learned in a sudden late jump. Steps to cross 0.5: damped 500, undamped 750–1,000, full design 1,000 (both seeds), full design without contrast 500, DG keys 500, subtractive 2,000, open memory loop 1,250. Unregulated clamped sinusoids (no feedback, divisive only) did not cross in 3,000 steps. With a 1,500-step one-cycle schedule the full design never crossed, because the learning rate had decayed before the jump; 3,000 steps were needed.

### 4.7 Pretrained models

Side channels on frozen Hugging Face models, needle task with a gap of about 500 tokens, block windows of 64 tokens, run on two RTX 5090s by a separate session (details and caveats in [`results/pretrained_HANDOFF.md`](results/pretrained_HANDOFF.md)). Exact match of the 4-digit value:

| Model | Windowed | Windowed + side channels | Full attention |
|---|---:|---:|---:|
| Qwen3-1.7B-Base, all layers windowed | 0.00 | 0.00 | 1.00 |
| Qwen3.5-2B-Base, attention windowed, GDN intact, tap at block 12 | 0.00 | **1.00** | 1.00 |
| Qwen3.5-2B-Base, same, tap at block 22 | 0.00 | 0.00 | 1.00 |

* **Qwen3.5's Gated DeltaNet layers alone do not recall the needle** (0.00). This suggests native GDN memory keeps a trace too weak for exact recall at this distance, and that the hybrid relies on its attention layers for exact recall.
* **With the side channels tapping block 12, recall is exact.** Moving the tap to block 22 removes it, so the success depends on the GDN layers after the tap.
* **The loop could not see the needle itself.** With $\rho = 0.7$ and $\tau = 16$ its span is $\tau/\ln(1/\rho) \approx 45$ tokens and its weight at a 500-token lag is about $10^{-19}$ (exactly 0 at lags 501–503, the clamped half of the cosine). So the Qwen3 failure is a setting, not evidence against the loop, and on Qwen3.5 the needle must reach the tap through the GDN layers before it. The working hypothesis is a chain: GDN layers before the tap carry a weak trace, the loop re-presents it, and GDN layers after the tap integrate it. Reruns with a horizon that covers the gap are queued.
* The first GPU runs had the side channels in bf16, where Adam updates near 1 round away; the table uses the float32 reruns.

### What this shows

* **The two side channels do separate jobs.** The loop carries content beyond the attention window; the clock carries elapsed time, which positional encodings do not provide once an event leaves the window.
* **A clamped, damped loop with CA1 control stays bounded and still learns**, but its bound comes from the clamp. Of the feedback forms, only the subtractive inverted copy follows the op-amp law, and only within a stability limit set by the one-circuit delay.
* **Per-circuit damping is not enough in a closed loop.** Rehearsal re-stores old content, so the first event captures the loop. The contrast against the prior and sparse dentate-gyrus write keys counter it; damping does not.
* **Damping trades memory for recency.** The loop's horizon, about $\tau/\ln(1/\rho)$ tokens, has to match the task; several bands with different $\rho$ may be needed to have both.

**Caveats.** Mostly single seeds and 0.1 M-parameter models on synthetic tasks; a difference of a few points is noise. The trained recency task was not learned by any model. The pretrained results are one seed each.
