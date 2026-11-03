# Results (single seed; small CPU models, plus frozen 2B models on GPU)

All runs: $d_{\text{model}} = 64$, 4 heads, batch 32, AdamW with one-cycle LR at a peak of $3\times10^{-3}$, seed 0. These are small, single-seed sanity experiments, not benchmarks; differences of a few points are within noise. Sections 4.7 and 4.8 are the exception: frozen 2B models on GPU, with the settings given there.

Raw logs and JSON curves are in `results/`, one folder per track: `results/synthetic/` (the CPU runs of §1–4.6), `results/pretrained/` (§4.7), `results/evolve/round1/` (§4.8), `results/evolve/round2/` (round 2, in progress) and `results/alignment/` (the alignment track, written by `experiments/alignment/run.py`; labels and scores only). [`results/README.md`](results/README.md) describes the layout, and [`research/REGISTER.md`](research/REGISTER.md) lists every result with its status and evidence file.

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
| `titans_hippo` (all of the above) | 429 k | 0.172 | Flat at 0.11–0.12 to step 1,500, then rose to a peak of 0.183 at step 2,500 |

**What this shows**

* **Both memory systems work.** Strategy I and Titans recall bindings from well outside their 16-token attention, while windowed attention with the same reach stays at chance. Titans matches full attention.
* **Strategy I is close** (0.41 against 0.49). Its CA3 is a discrete, novelty-gated episodic store written once per segment. That makes it inspectable, since you can see which events were stored, and it uses no test-time gradient computation.
* **The hippocampal add-ons are neutral alone and harmful together in this regime.** DG writes alone and CA1 + novelty-lr alone are each within a few points of plain Titans. Combining them slows optimization badly. A plausible cause is that the novelty-scaled learning rate is noisy early, when the CA1 signal is still untrained, and DG sparsification shrinks the effective key dimension at the same time. Warming up `novelty_lr`, or stopping gradients through novelty, are the next things to try.
* **Probably not a task ceiling.** In the first session, full-attention accuracy by query position ran 0.26, 0.27, 0.31, 0.37, 0.54, 0.82, which would mean the ~0.49 plateau shared across models reflects small models and short training. No stored result file records accuracy by query position (`results/synthetic/assoc_recall.json` keeps only the overall curve), so this is unverified until a rerun logs it.
* **Review RV2 (2 Nov 2026): the plateau is not binding.** The rising per-position profile is the signature of elimination. Values are drawn with replacement from 16 and queries are teacher-forced, so remembering the studied values and striking each as it is revealed scores 0.476, with first-query accuracy 0.288 (`research/reviews/2026-11-02_literature/scripts/binding_free_baseline.py`). Full attention's first query (0.26) is below that, so the ~0.49 plateau shared across models is this heuristic, not binding. Only the successor-tied GDN memories (§3) bind. Register RV2.X1, L1.

## 2. Continual learning with replay consolidation

Three tasks, each with its own fixed key→value facts, are learned one after another for 600 steps each, using Strategy I. Only the first occurrence of a key in a sequence is scored, so the facts must live in the weights.

| Condition | Acc after T0 (T0, T1, T2) | After T1 | After T2 | Avg final | BWT |
|---|---|---|---|---:|---:|
| Naive sequential | 1.00, 0.13, 0.10 | 0.35, 1.00, 0.14 | 0.21, 0.33, 1.00 | 0.513 | −0.731 |
| Novelty-prioritized replay + sleep | 1.00, 0.13, 0.10 | 1.00, 1.00, 0.14 | 1.00, 1.00, 1.00 | **1.000** | **+0.000** |

Naive fine-tuning forgets catastrophically (BWT −0.73). Replay of novelty-prioritized traces, with logit distillation and an offline sleep phase on the slow parameters, removes the forgetting entirely on this task. The task is easy, and this run doesn't separate the contribution of the sleep phase from that of interleaved awake replay. That would be the next ablation.

## Bugs the experiments caught

The first benchmark run had every memory model at chance. Fixing that turned up three issues, register B9–B11 (see README, "Lessons from the experiments"):

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
| Subtractive (inverted copy), seed 0 | **0.990** | 2.60 | 10.3 | 0.05 / 0.54 |
| Subtractive (inverted copy), seed 1 | **0.998** | 1.71 | 6.9 | 0.40 / 0.38 |
| Full design (clamp + divisive + projection + saturation) | **1.000** | **0.35** | **0.53** | 0.25 / 0.18 |

Subtractive feedback is the only unclamped loop that learned content, on both seeds. It does not bound the echo (1.7–2.6× the input): at a stable loop gain ($K \approx 0.31$–$0.33$ here) it removes only about a quarter of sustained echo. A guaranteed bound still needs the clamp. Its effect on recency differs between seeds (0.54 and 0.38 older).

### 4.4 Do new tokens dominate?

A trained recency task (an older, different event precedes the target) was learned by none of the three closed loops in 3,000 steps (*what* 0.14–0.15), so it cannot rank them. Instead, each model trained on single events was probed zero-shot with an older event 20–60 tokens before the target.

| Model (3,000 steps) | Names newer | Names older |
|---|---:|---:|
| Undamped closed loop | 0.25 | 0.61 |
| Damped closed loop | 0.06 | **0.84** |
| Subtractive feedback, no clamp, seed 0 | 0.05 | 0.54 |
| Subtractive feedback, no clamp, seed 1 | 0.40 | 0.38 |
| Full design without contrast | 0.32 | 0.63 |
| Full design | 0.25 | 0.18 |
| **Full design, DG write keys, direct read** | **0.68** | 0.31 |
| **Full design, DG write keys, DG read** | **0.68** | 0.29 |
| Open loop (stores the pre-injection stream), seed 1 | 0.45 | 0.38 |

**Closed loops favour the first event.** A closed loop re-stores its own echo every circuit, and each re-stored copy starts its decay again, so damping each trace does not damp content the loop keeps rehearsing: the older event, rehearsed longer, wins. This is primacy by self-rehearsal, the mechanism psychology proposes for the primacy effect. The episode-minus-prior contrast reduces it (0.63 → 0.18 older), plausibly because rehearsed content becomes part of the prior $\mu_t$ and is subtracted. Sparse dentate-gyrus write keys give the only clear recency win, presumably because the two events no longer collide in memory. The open loop, which stores the stream from before injection as the acetylcholine switch does, does not re-store its echo and shows little primacy (0.38 older against 0.54–0.84 for closed loops without the contrast; different seed). The subtractive inverted copy did not reliably break primacy, contrary to the high-pass prediction: one seed favoured the older event and the other was even, consistent with the small usable loop gain in 4.3.

### 4.5 Read path and cue noise

Dentate-gyrus write keys, with the retrieval cue read either directly from the stream (entorhinal layer II → CA3) or through the same DG code; relative Gaussian noise on every read cue at evaluation.

| Model (3,000 steps) | σ = 0 | 0.5 | 1.0 | 2.0 |
|---|---:|---:|---:|---:|
| Dense keys, no contrast | 1.000 | 1.000 | 0.996 | 0.969 |
| DG keys, direct read | 1.000 | 1.000 | 0.957 | 0.840 |
| DG keys, DG read | 1.000 | 1.000 | 1.000 | 0.941 |

The Treves–Rolls prediction, that a direct dense cue completes degraded patterns better than a pattern-separated one, is not supported here (one seed). The k-WTA read acts as a threshold that removes noise, and with 4× expansion and 10 % sparsity this DG separates patterns only mildly.

### 4.6 Learning speed

*What* is learned in a sudden late jump. Steps to cross 0.5: damped 500, undamped 750–1,000, full design 1,000 (both seeds), full design without contrast 500, DG keys 500, subtractive 2,000 and 1,000 (two seeds), open memory loop 1,250 at 1,500 steps and 1,000 at 3,000. Unregulated clamped sinusoids (no feedback, divisive only) did not cross in 3,000 steps. With a 1,500-step one-cycle schedule the full design never crossed, because the learning rate had decayed before the jump; 3,000 steps were needed.

### 4.7 Pretrained models

Side channels on frozen Hugging Face models, needle task with a gap of about 500 tokens, block windows of 64 tokens, run on two RTX 5090s by a separate session (provenance and caveats in [`results/README.md`](results/README.md)). Exact match of the 4-digit value:

| Model | Windowed | Windowed + side channels | Full attention |
|---|---:|---:|---:|
| Qwen3-1.7B-Base, all layers windowed | 0.00 | 0.00 | 1.00 |
| Qwen3.5-2B-Base, attention windowed, GDN intact, tap at block 12 | 0.00 | **1.00** | 1.00 |
| Qwen3.5-2B-Base, same, tap at block 22 | 0.00 | 0.00 | 1.00 |

With a loop horizon that covers the gap (`--kernel exp --rho 0.95 --tau 32`, weight 0.47 at a 500-token lag; 3,000 steps), every layer windowed, so the loop is the only long-range path. Digit loss against the answer-format floor $0.8\ln 10 = 1.842$:

| Model, all layers windowed | Windowed | Side, default kernel | Side, horizon kernel | Full attention |
|---|---:|---:|---:|---:|
| Qwen3-1.7B-Base, loss / token accuracy / exact | 3.07 / 0.11 / 0.00 | 1.84 / 0.29 / 0.00 | **1.38 / 0.46 / 0.00** | 0.01 / 1.00 / 1.00 |
| Qwen3.5-2B-Base, loss / token accuracy / exact | 3.04 / 0.13 / 0.00 | 1.84 / 0.29 / 0.00 | **1.24 / 0.55 / 0.02** | 0.03 / 1.00 / 1.00 |

The default-kernel entries are the float32 reruns, both at the format floor. An earlier bf16 figure of 1.69 for Qwen3.5 came from an anti-causal leak and is superseded.

**Superseded:** the Qwen3.5 horizon-kernel entry (`results/pretrained/pretrained_qwen35_needle_allwin_horizon.json`, 1.24 / 0.55 / 0.02) was undertrained. The same run with 6,000 steps reaches exact 1.00 at loss 0.001 (`g1_qwen35_allwin_horizon_6k`, §4.8, register P9). The Qwen3 horizon entry stands: round 1 reproduced it exactly (P7).

**How much of the value arrives.** With five scored tokens (the leading space, then four digits) and $k$ digits known perfectly, the rest at chance, mean digit loss is $\frac{4-k}{5}\ln 10$ and token accuracy is $\frac{1 + k + 0.1(4-k)}{5}$. Qwen3's 1.381 / 0.461 matches $k = 1$ exactly (1.382 / 0.46). Qwen3.5's 1.239 / 0.55 lies between $k = 1$ and $k = 2$. The loop stores each key and its value at the same position, so nothing moves the lookup on to the next digit. This is the gap successor-bound addressing (§3) fills. Round 1 confirmed both points: the digit the loop recovers is the first (P7), and a successor-bound side memory recalls all four (P10; §4.8).

* **Qwen3.5's Gated DeltaNet layers alone do not recall the needle** (0.00). This suggests native GDN memory keeps a trace too weak for exact recall at this distance, and that the hybrid relies on its attention layers for exact recall.
* **With the side channels tapping block 12, recall is exact.** Moving the tap to block 22 removes it, so the success depends on the GDN layers after the tap.
* **The loop could not see the needle itself.** With $\rho = 0.7$ and $\tau = 16$ its span is $\tau/\ln(1/\rho) \approx 45$ tokens and its weight at a 500-token lag is about $10^{-19}$ (exactly 0 at lags 501–503, the clamped half of the cosine). So the Qwen3 failure is a setting, not evidence against the loop, and on Qwen3.5 the needle must reach the tap through the GDN layers before it. The working hypothesis is a chain: GDN layers before the tap carry a weak trace, the loop re-presents it, and GDN layers after the tap integrate it. With a horizon that covers the gap, the loop alone carries needle information 500 tokens back in both models: one of the four digits on Qwen3, and between one and two on Qwen3.5. These 3,000-step runs did not reach exact recall of all four digits; the Qwen3.5 run left the floor only near step 2,700, as the learning rate reached zero. Round 1 (§4.8) settled both: the Qwen3.5 run was undertrained, and with 6,000 steps the loop alone recalls all four digits exactly (P9); on Qwen3 the loop stays at one digit, and the successor-bound side memory W1 recalls all four exactly (P10).
* The first GPU runs had the side channels in bf16, where Adam updates near 1 round away; the tables use the float32 reruns.
* **Language modelling** (WikiText-103, window 256, 1,000 steps): the side channels cut windowed perplexity from 14.6 to 12.0 on Qwen3 (full attention 9.0; 41% of the loss gap) and from 16.3 to 11.2 on Qwen3.5 with attention windowed (full 9.8; 74%). Only the windowed + side condition trains on WikiText, so part of each gain may be domain adaptation rather than memory. With the horizon kernel, Qwen3 reaches 11.5 (§4.8, P12).

### 4.8 Round 1 of the evolution search

Round 1 (2 Nov 2026) followed up the horizon-loop results of §4.7 and tested the first fixed-size candidate, W1. Same needle task: seq 512, block windows of 64 tokens on every layer (so the side channel is the only route past 64 tokens), batch 8, 3,000 steps unless noted, 128 evaluation sequences, one seed, float32 side channels on a bf16 base. The needle gap is 501–502 tokens with one stored pair and 490–502 with two. Runs go through `experiments/evolve/run.py`, which adds per-digit loss and saves the trained side channels. Results are in `results/evolve/round1/` (the files keep their `g1_*` names), register rows P7–P12; the reasoning is in [`research/NOTEBOOK.md`](research/NOTEBOOK.md) ("Round 1").

Two side channels:

* **Horizon loop.** The O(T) loop of §4.7 with `--kernel exp --rho 0.95 --tau 32`, 4.38 M trainable parameters. Key and value come from the same position.
* **W1** (`experiments/evolve/w1_gdn.py`). A Gated DeltaNet side memory that files each value under the previous position's address (successor binding, §3), plus the clock, injected through the same CA1 control. Its state is fixed-size (8 × 64 × 64 with dense keys, 4 × 64 × 256 with DG keys), as decision DEC1 requires for the pretrained track. Dense keys: 8 heads, 4.28 M parameters. DG k-WTA keys: 4 heads, key expansion 4, 5 % sparsity, 3.74 M parameters (12.6 % fewer).

**Needle recall.** Five tokens are scored: the leading space, then the four value digits. Chance loss per digit is $\ln 10 = 2.30$.

| Run | Model | Side channel | Pairs | Exact | Loss | Digit loss 1 / 2 / 3 / 4 | Digit accuracy 1 / 2 / 3 / 4 |
|---|---|---|---:|---:|---:|---|---|
| `g1_qwen3_horizon_perpos` | Qwen3-1.7B | Horizon loop | 1 | 0.00 | 1.381 | **0.001** / 2.30 / 2.30 / 2.30 | 1.00 / 0.13 / 0.08 / 0.09 |
| `g1_qwen3_horizon_pairs2` | Qwen3-1.7B | Horizon loop | 2 | 0.00 | 1.809 | 2.17 / 2.29 / 2.28 / 2.30 | 0.23 / 0.10 / 0.14 / 0.09 |
| `g1_qwen35_allwin_horizon_6k` | Qwen3.5-2B | Horizon loop, 6,000 steps | 1 | **1.00** | 0.001 | 0.002 / 0.001 / 0.001 / 0.001 | 1.00 / 1.00 / 1.00 / 1.00 |
| `g1_w1_tied` | Qwen3-1.7B | W1, dense keys | 1 | **1.00** | **0.0001** | 0.000 / 0.000 / 0.000 / 0.000 | 1.00 / 1.00 / 1.00 / 1.00 |
| `g1_w1_dg` | Qwen3-1.7B | W1, DG keys | 1 | **1.00** | **0.0001** | 0.000 / 0.000 / 0.000 / 0.000 | 1.00 / 1.00 / 1.00 / 1.00 |
| `g1_w1_tied_pairs2` | Qwen3-1.7B | W1, dense keys | 2 | 0.55 | 0.186 | **0.74** / 0.05 / 0.11 / 0.03 | 0.57 / 0.99 / 0.97 / 0.99 |
| `g1_w1_dg_pairs2` | Qwen3-1.7B | W1, DG keys | 2 | 0.43 | 0.231 | **0.76** / 0.11 / 0.15 / 0.14 | 0.51 / 0.96 / 0.94 / 0.94 |

Without side channels every run scores exact 0.00 (loss 3.07 on Qwen3 with one pair, 3.11 with two, 3.04 on Qwen3.5). Full attention scores exact 1.00, at loss 0.011 (Qwen3, one pair), 0.025 (two pairs) and 0.034 (Qwen3.5). The injection ratio settled at 0.22–0.24 for the loop and 0.31–0.33 for W1.

**Learning speed** (training-batch loss, logged every 300 steps, every 600 for the 6,000-step run). W1 with DG keys: 0.73 at step 300, 0.008 at 600. W1 with dense keys: still at the floor (1.84) at 600, 0.18 at 900, 0.0004 at 1,200. Horizon loop on Qwen3: 1.68 at 600, 1.37 at 900, then flat at 1.38 to step 3,000. Horizon loop on Qwen3.5: at the floor to step 1,800, 1.39 at 2,400, 0.23 at 3,000, 0.006 at 3,600.

**Language modelling** (Qwen3-1.7B, WikiText-103, window 256, every layer windowed, 1,000 steps; `g1_qwen3_lm_horizon` against `results/pretrained/pretrained_qwen3_lm.json`):

| | Windowed | Side, default kernel | Side, horizon kernel | Full attention |
|---|---:|---:|---:|---:|
| Perplexity | 14.62 | 12.01 | **11.53** | 9.01 |
| Share of the loss gap closed | — | 41 % | **49 %** | — |

**What this shows**

* **The horizon loop knows the first digit and nothing after it (P7).** Digit 1 is at loss 0.001; digits 2–4 sit at chance. The run reproduces `results/pretrained/pretrained_qwen3_needle_horizon.json` exactly (1.381 / 0.461) and confirms the "one digit known" reading of §4.7.
* **That digit does not come from the query (P8).** With two stored pairs the loop falls back to near chance even on digit 1 (loss 2.17, accuracy 0.23). The one-pair recall used a path that does not depend on which key is asked for.
* **With enough training the loop alone gives exact recall on Qwen3.5 (P9).** The 6,000-step run left the floor at about step 2,400 and recalls all four digits with every layer, GDN included, windowed. It supersedes the 3,000-step entry in §4.7. It used one pair only, so P8's caveat may apply; the two-pair run on Qwen3.5 is planned (W1f).
* **Successor binding supplies the missing shift (P10).** W1 recalls all four digits exactly on Qwen3 with every layer windowed, with a fixed-size state and fewer parameters than the loop. Its loss (0.0001) is below full attention's (0.011). DG keys learn about twice as fast (loss 0.008 at step 600, against 900–1,200 steps for dense keys).
* **W1 retrieves the chain but does not choose where it starts (P11).** With two pairs, digits 2–4 are 94–99 % correct, but digit 1 is a coin flip between the two stored codes: accuracy 0.57 / 0.51, loss 0.74 / 0.76 ≈ $\ln 2 = 0.69$. The first digit is keyed by the address of the space after "is", the same token in both facts, and that address carries little of the name before it. DG keys do not help: pattern separation cannot add context the address lacks. Context-bearing addresses are the next test (W1b).
* **A horizon kernel helps language modelling slightly (P12).** Perplexity 12.0 → 11.5, from 41 % to 49 % of the loss gap. Only the side condition trains on WikiText, so the domain-adaptation confound (O4) remains.

**Caveats.** One seed per run. W1's stability and recency, the two must-pass fitness criteria, are not yet measured: the CPU suite cannot host W1 until an adapter exists (W1c). Held-out gaps, random needle offsets and natural filler are not yet tested (C6).

### What section 4 shows

* **The two side channels do separate jobs.** The loop carries content beyond the attention window; the clock carries elapsed time, which positional encodings do not provide once an event leaves the window.
* **A clamped, damped loop with CA1 control stays bounded and still learns**, but its bound comes from the clamp. Of the feedback forms, only the subtractive inverted copy follows the op-amp law, and only within a stability limit set by the one-circuit delay.
* **Per-circuit damping is not enough in a closed loop.** Rehearsal re-stores old content, so the first event captures the loop. The contrast against the prior and sparse dentate-gyrus write keys counter it; damping does not.
* **Damping trades memory for recency.** The loop's horizon, about $\tau/\ln(1/\rho)$ tokens, has to match the task; several bands with different $\rho$ may be needed to have both.
* **On frozen 2B models, successor-bound addressing turns one recalled digit into exact recall** (§4.8). Choosing among several stored pairs still fails: the addresses need to carry context.

**Caveats.** Mostly single seeds and 0.1 M-parameter models on synthetic tasks; a difference of a few points is noise. The trained recency task was not learned by any model. The pretrained results are one seed each.


### 4.9 Rounds 2–8 of the evolution search: content-addressed side memory, and why the track closed

All on frozen Qwen3-1.7B-Base with every layer windowed (block window 64), side channels at block 21, needle task with 3 stored name → 4-digit codes unless noted. One-pair results are passkey retrieval, so selection is measured with 2–5 pairs on a *scattered* layout (facts at random positions, which removes position shortcuts) and read by queried rank. Register P13–P41; procedure in `research/evolution_protocol.md`.

| Step | Finding | Register |
|---|---|---|
| Context in the write address (W1b) | Does not select the queried code: digit-1 loss ≈ ln(pairs) | P13–P16 |
| Is the cue present? | The name is linearly decodable at the query's space token (0.95–1.00), but W1's addresses discard it | P17 |
| DG-keyed start store (W1g) | Its 2-pair "win" was a first/last positional shortcut | P18–P19 |
| CA3 settling (W1h) | Hurt as built: the cue settles onto the query's own recent tokens | P20 |
| **Event-gated goal field (W1i, after Samsonovich & Ascoli 2005)** | **Content-addressed selection:** digit 1 0.99 / 0.99 / 1.00 by queried rank (3 codes), 0.93–0.99 with 5 codes | P21, P26 |
| Generalisation (evaluation only) | Holds to seq 2048 (gaps about 2,000, four times the training maximum) and to 1–5 codes. Fails on natural filler when trained on random filler, because the event detector fires about 4× more | P30–P32 |
| Training on natural filler | Robust to both: 0.94 natural / 0.91 random (refractory 0.95), when selection is learned | P33 |
| **Reliability** | Selection is learned in a sudden, seed-dependent late drop. Natural recipe: 1 of 5 seeds; with a pair curriculum: 3 of 5. An explicit CA1 novelty gate: 0 of 5. Three other curriculum schedules: 1, 2 and 1 of 4 (round 9); they change which seeds learn, not how many | P25, P36–P41, P46–P48 |

**Conclusion (DEC2 stop rule, 3 Nov 2026).** A fixed-size, inference-time side memory on a frozen LLM *can* store several facts while reading and retrieve the queried one by content, far beyond its attention window: the missing-key step that Samsonovich & Ascoli's goal field addresses. It does so with event-timed, sparse writes under a context key in which the name is salient. Training does not reliably *find* that solution. The failing runs sit on the plateau where everything but the choice of start is learned (loss ≈ ln 3 / 5). Three parts must co-adapt (events at first digits, a name-salient key, the goal read weight), and only a curriculum has partly lowered the barrier (3 of 5 seeds). The track is closed as "capable but not reliably learnable". Open threads, should it be reopened: curricula with more stages, novelty gates whose gain starts near 0, longer schedules. [Reopened, 3 Nov 2026: Daniel extended the track by two rounds (DEC3) to test curriculum schedules. A diagnosis of W1j (P42) rules out the novelty-gate thread: the chain store's prediction error is *lower* at the facts' first digits than in filler at initialisation, the gain never moved, and the signal becomes selective only after selection is learned.]
