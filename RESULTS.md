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
| `titans_gdn_dg` (+ DG-sparse addresses) | 0.905 | see `results/assoc_recall_gdn.log` |

**What this shows**

* **Standard Gated DeltaNet runs out of capacity as predicted.** With 16-dimensional keys, recall falls from 0.43 at 8 pairs to about 0.13 at 16 and 32 pairs. Widening keys to 128 dimensions helps (0.32 and 0.22) but doesn't fix it, because the model must also learn to bind each key to the next token through its short convolution.
* **Successor-bound addresses solve the task outright.** When the address of token $t-1$ is the key for value $t$, the memory behaves as RAM and recall is perfect up to 32 pairs. A counterfactual check confirms this is genuine retrieval: after swapping the values shown at study time, predictions follow the new values 100 % of the time and the old values 0 %.
* **DG versus dense addressing is not separated by this task.** Both hit the ceiling. With 128-dimensional addresses and distinct keys, dense addresses barely interfere. Testing pattern separation needs similar keys (shared prefixes or noise) or loads above $d_k$.
* **An exact delta-rule memory beats the deep Titans memory here.** Swapping Titans' chunk-start gradient update for the exact per-token delta rule takes recall from 0.48 to 1.00. DG addresses reach high recall faster (0.905 vs 0.680 at step 500), a sample-efficiency effect that the final scores hide.
