# Round 0 plan: why the side channels fail the Qwen3 needle task

## 1. Diagnosis

Ranked surviving explanations:

| # | Status | Explanation | Code |
|---|---|---|---|
| D1 | Holds (fact) | Qwen3 windowed+side scores exactly the answer-format floor. Eval loss is 1.8421, against 0.8·ln10 = 1.8421. tok_acc is 0.287, so digit accuracy is 56/512 = chance. The side channels learned the leading space and the digit distribution, and nothing about the value. | scoring `experiments/pretrained_loop.py:433-461` (answer = `enc(f" {v}")`, mask covers the last n_ans), `:572-578` |
| D2 | Holds, primary suspected cause, not yet tested by intervention | **Kernel learnability barrier.** With osc/0.7/16, the content-addressed read gives the needle a weight of at most about 2e-5, against a row mass of 14.68. On some digit diagonals the weight is exactly 0 because of the cos₊ gate. The gradient on the needle's key is about 5e-6 with contrast on or off. This is not a hard capacity limit, because the injection is renormalised to a fixed norm. | kernel `limbic_llm/temporal_loop.py:100-103`; read `:263-268`; renorm `:230-234`; defaults `pretrained_loop.py:659-661` |
| D3 | Open; the key unknown | **Some block sits downstream of the read.** The contrast prior is an unweighted mean over available positions. It already sends a needle share of about 3-4.5%, and a needle-to-recent gradient ratio of about 0.075, to `wv`, whatever the gap. With pairs=1 no addressing is needed. Yet zero bits arrive. Candidate causes: (a) tap-14 states at block-0 positions do not linearly encode the value; (b) the frozen windowed upper layers ignore a ~4% component inside a 0.2‖h‖ injection; (c) a uniform prior shift washes out. If (a) holds, no kernel fix helps. | prior `temporal_loop.py:269-273`; `--pairs` default 1 at `pretrained_loop.py:677` |
| D4 | Holds | **Qwen3.5 attention-only exact 1.0 is carried by the intact GDN layers, not by loop retrieval.** The all-windowed fp32 rerun uses the same tap and the same side channels. It stays at 1.83-1.85 train loss through step 3000 (eval line not yet logged). The route is unresolved. Either the loop re-injects locally and the 8 GDN layers above the tap carry the value forward, or the readout happens at the query. The finished tap-22 run (GDN below the tap only) ends at eval 1.734 / tok_acc 0.345 / exact 0, which leans toward "layers above the tap must carry or decode". | GDN left unwindowed `pretrained_loop.py:234-245`; `reach()` `:335-338` |
| D5 | Holds; removes evidence | **The pre-fix all-windowed sub-floor result (eval 1.692) was an anti-causal leak, not long-range transfer.** bf16 `arange` in `decay_kernel` and `loop_kernel` merges positions. I checked on CPU: `arange(512, bf16)` has only 385 distinct values, and 505..511 become [504, 506, 508, 508, 508, 510, 512], so the `d>=0` mask lets earlier positions see later ones. That gives the d1/d2 predictions their own targets. Side channels now run in fp32. | `temporal_loop.py:94, 109-111`; `.float()` at `pretrained_loop.py:291, 294` |

True but not the binding constraint yet (they matter only after D2/D3 are fixed):
- The osc phase hole (taskeval H2).
- The CA1 tanh cap on a selective read (`temporal_loop.py:231-232`; capacity H2).
- The need for an address shift so each answer position can find the next digit (capacity H4). Keys and values come from the same position (`:262-263`).

Refuted:
- **Read sharpness (gradflow H3).** With contrast on, the direction already has about 0.99 cosine with the needle at sharpness 6.
- **Contrast as a cause (gradflow H4).** Contrast does not change the key gradient. The ~1e4 imbalance comes from the kernel.
- **CA1 gate as the current bottleneck (capacity H2).** Saturate on and off are nearly the same in the observed regime.
- **"Relays Δ16-64 GDN-enriched neighbours" (capacity H3, taskeval H3).** For answer rows 507-511 the latest position the loop can return is 491-495, which is before the query (starts at 502).
- **"TimeChannel cannot help addressing" (capacity H4).** Its features enter both q and k (`pretrained_loop.py:202-205`, `temporal_loop.py:145`).
- **TimeChannel or contrast prior as the source of the pre-fix drift (contrast H4, taskeval H4).** See D5.
- **"Loop structurally cannot reach the needle" (strong form of the H1 variants).** In a synthetic CPU check, the default osc/0.7/16 kernel reached 97% accuracy when the value is spread over block 0.

## 2. Round-0 diagnostic runs (existing flags only)

**Settings for all runs**
- Common prefix, abbreviated `$BASE` below:
  ```
  python experiments/pretrained_loop.py --model Qwen/Qwen3-1.7B-Base --dtype bfloat16 --device cuda --task needle --window 64 --window-mode block --seq-len 512 --batch 8 --steps 1000 --log-every 50 --seed 0
  ```
- **Steps.** I used 1000 rather than 500. D2 is a learnability claim: under the flat kernel the needle's share at init is about 1/493. The post-fix Qwen3 run took 631 s for 3000 steps, about 0.21 s/step, so each arm costs about 3.5 min. Every arm uses the same schedule (`pretrained_loop.py:608`).
- **Pass criterion.** Eval windowed+side loss ≤ 1.82 **and** tok_acc ≥ 0.315. That puts digit accuracy at least 3.3 standard errors above chance on 512 digits. The floor is 1.842 / 0.28.

| Name | Command | What it discriminates |
|---|---|---|
| `g0_ctrl` | `$BASE --out results/evolve/gen00/g0_ctrl.json` | Control with a matched schedule. Expected to stay at 1.842 / 0.28. |
| `g1_flat` | `$BASE --kernel flat --out results/evolve/gen00/g1_flat.json` | **D2 against D3.** A pass means the kernel was the barrier. Staying at the floor means the block is downstream (D3a/b/c), and kernel mutations alone are low value. |
| `g2_exp098` | `$BASE --kernel exp --rho 0.98 --out results/evolve/gen00/g2_exp098.json` | Decay with no phase gate, w(500) ≈ 0.54. If it passes like g1, a slow learnable decay is enough and the flat kernel's unbounded memory is not required. If g1 passes and g2 fails, residual decay or distractor mass still binds. |
| `g3_flat_notime` | `$BASE --kernel flat --no-time --out results/evolve/gen00/g3_flat_notime.json` | Only meaningful if g1 passes. If g3 ≈ g1, the loop is the route. If g3 drops to the floor, TimeChannel features (salience or addressing) are needed. |
| `g4_flat_nocontrast` | `$BASE --kernel flat --no-contrast --out results/evolve/gen00/g4_flat_nocontrast.json` | Does the value arrive through the content-addressed read or through the uniform prior? If g4 ≈ g1, it is content-addressed. If only g1 passes, the prior is doing the work, and that will not scale to pairs > 1. |
| `g5_flat_pairs2` | `$BASE --kernel flat --pairs 2 --out results/evolve/gen00/g5_flat_pairs2.json` | Forces key addressing, which the query-independent prior cannot do. A large gap between g5 and g1 means only the prior path works. A fixed-size-state candidate must pass this setting. |
| `g6_flat_l6` | `$BASE --kernel flat --layer 6 --grad-checkpointing --out results/evolve/gen00/g6_flat_l6.json` | **D3a/b via tap depth** (Qwen3 has 28 blocks; the default tap is 14). If g1 fails but g6 passes, the representation or readout depends on depth: more upper layers are needed to decode the injection. |
| `g7_flat_l22` | `$BASE --kernel flat --layer 22 --out results/evolve/gen00/g7_flat_l22.json` | The other side of the depth test: the value is more decodable at the tap, but fewer layers remain to read it. g6 passing and g7 failing points to D3b (readout). The reverse points to D3a (encoding). |

**Caveats**
- The flat and exp kernels are diagnostics, not candidates. Like the default loop, they store an O(T) memory (`temporal_loop.py:262`) and so fail rule E2 below.
- I did not include the taskeval-H2 phase test (seq-len 508). `main` requires `seq-len % window == 0` for block mode (`pretrained_loop.py` in `main`, the `--seq-len must be a multiple` check), so that run cannot be made.
- The post-fix Qwen3.5 all-windowed rerun now running is the existing control for "no long-range leak after the fix". Its eval line should be read when it finishes. No new arm is needed for it.

**Diagnostics that need a code change, or a GPU scratch script under /tmp**

- **C1.** Log per-position cross-entropy and accuracy (reshape `ce*m` from `scores()` at `:572-578` to [B, 5]), and save the side-channel `state_dict`. Nothing is saved today (`:757-765`), so no eval-time ablation is currently possible.
- **C2.** Linear probe for the value digits on frozen Qwen3 hidden states at layers 6, 14 and 22. Probe block-0 positions 11-63 and the "." position. This settles D3a. It can be a /tmp script that imports the module without editing the repo, but it needs a GPU.
- **C3.** Instrument the loop. At the answer rows, log the share of M on block 0 (M is formed at `temporal_loop.py:267`) and cos(delta, wv(x_needle) − prior).
- **C4.** Eval-time ablations on a trained Qwen3.5 attention-only checkpoint, which needs C1:
  - Zero the injection at positions < 128, and separately at positions ≥ 400. This separates "local write, GDN carries" from "readout at the query".
  - Replace the injection with its batch mean, and separately swap injections between sequences.
- **C5.** Run `check_window` on every windowed layer; today it checks only `lm.windowed[0]` (`:349`). Add an end-to-end test with the side channels off: perturbing the needle digits must leave the answer logits bitwise unchanged.
- **C6.** Held-out evaluation:
  - an eval-only `--eval-seq-len`, giving gaps such as 373, 757 and 1013;
  - a random needle offset (today the facts always sit right after the sink, `:452`);
  - natural-text filler at eval only (`--text-file` currently applies to both train and eval).

## 3. Evaluator constraints for code-evolution rounds

**How HookedLM feeds the side channel**
- The candidate is called from `_tap` (`pretrained_loop.py:281-298`). It receives `h[:, s:]` from the tap block, the whole body (B, L, D) in fp32 in a single call. It must return (h′, stats) of the same shape, and `_tap` then writes h′ back into the stream.
- In block mode `_seg` is `None` (it is set only in segment mode, `:316-326`), so the candidate sees every position, future ones included.
- In segment mode the `_seg` path (`:288-292`) reassembles the (B, n·W, D) segments into (B, L, D) before calling the candidate, so it also sees the full sequence.
- Causality is therefore entirely the candidate's responsibility. The window mask (`:145-157`) and the GDN fold (`:160-179`) make the side channel the only path across a block. The answer tokens are in the input under teacher forcing (`:456-461`). So any leak trivially solves the task, as D5 shows.

| Rule | Requirement | Automatic check |
|---|---|---|
| E1 Strict causality | out[:, t] depends only on h[:, ≤t] | Perturb h[:, t0:] for t0 ∈ {1, 63, 64, 255, 256, 257, 500, 505..511} at T ∈ {512, 1024}, and assert out[:, :t0] is **bitwise** equal. Run it twice: on fp32 input, and on input cast to bf16 then `.float()`, as `_tap` does. Add an autograd check that ∂out_t/∂h_{t′>t} = 0 on random rows. This catches the bf16-`arange` class of bug. |
| E2 Fixed-size state (this is the operational form of "no attention over past positions") | The candidate exposes `init_state(B)` and `step(h_chunk, state) -> (out, state)`. State size does not depend on T. | The harness produces outputs **only** by streaming chunks of 64 (also 1 and 17) and compares them with `forward(h)` to within 1e-4. State `numel` at T = 512 and T = 4096 must be identical and ≤ S_max (for example 2²⁰ floats per sequence). Wall time at 2T over T must be ≤ 2.3. Attention inside a chunk is allowed only within the chunk. The current `HippocampalLoop` fails this rule because it stores `x_pre[:, :start]` (`temporal_loop.py:262`). |
| E3 Parameter budget | Trainable parameters ≤ 4.4M (current 4,380,675, counted as at `:717`). Buffers ≤ 1M floats. No copies of base weights. | Count parameters. Hash the base `state_dict` before and after training. Assert no parameter or buffer shares storage with, or is numerically equal to, a base tensor. |
| E4 Input isolation | The only input is `h`. No token ids, tokenizer, task object, HookedLM or global mutable state. | AST lint banning those names and imports. Run in a separate module namespace. |
| E5 Precision | Position and index arithmetic stays in fp32 or integers. | Covered by the bf16 variant of E1. Also lint `arange(..., dtype=h.dtype)` patterns. |
| E6 Held-out generalisation (needs C6) | Training uses seq-len 512 and the fixed layout. Fitness is measured on held-out gaps (seq-len 384/768/1024), a random needle offset, natural-text filler, pairs ∈ {2, 3}, and ≥ 2 training seeds. Eval sequences are already drawn from a disjoint rng, `seed+999` (`:724`). | Fitness = min over seeds and conditions of (ln10 − mean digit CE). Reject any candidate whose training loss is below the floor while held-out stays at the floor (it memorised the layout). |
| E7 Window integrity | The base must not leak by itself. | C5 checks: every windowed layer, plus the end-to-end side-off perturbation test. |
| E8 No LM regression | Side channels must not hurt language modelling. | Qwen3 WikiText-103 windowed+side perplexity ≤ 11.9 (current result). |
| E9 Metric | Per-digit CE and accuracy relative to the 0.8·ln10 format floor. Exact match is a secondary metric only. | Uses C1 logging. |

## 4. Round-1 mutation directions, ranked by expected value

The ranking depends on round 0. If g1, g2 and g6/g7 all stay at the floor and the C2 probe is at chance, put direction 4 (read and write depth) first.

1. **Gated delta-rule associative memory at the tap.** Reuse `limbic_llm/gated_delta.py` (`gated_delta_scan` `:51`, `GatedDeltaMemory` `:294`).
   - The write strength β is gated by salience (TimeChannel `u`, `temporal_loop.py:131`), and the decay α is initialised near 1.
   - Injection goes through the existing CA1 clamp (`:230-234`).
   - Why first: GDN is the demonstrated long-range carrier in Qwen3.5 (D4), its state is fixed-size so it satisfies E2, and it has no distance attenuation.
2. **Fixed-state multi-timescale loop.** Replace exp(s·(q·k−1)) with a positive feature map φ. Keep the delay of τ and use per-head learnable decay, with S updated as ρS + φ(k)vᵀ and at least one head initialised with ρ ≥ 0.995. This keeps the loop and theta framing and is the state-space version of g2. High value if g2 passes.
3. **Write selectivity in place of the unweighted prior.** Gate writes on novelty or salience so filler is not stored. Replace the `avail`-mask mean (`:270-273`) with a decayed running mean, which fits a fixed state. This targets D3c (the uniform shift that washes the needle out).
4. **Read and write depth.** Allow writing at a lower layer and injecting at the tap, or injecting at two blocks, with the parameter budget covering both. Conditional on g6/g7 or C2 showing that the readout depends on depth.
5. **Address-shift keys.** Include x_pre shifted back by one token, and the TimeChannel lag features, in k and q, so each answer position finds the next digit (capacity H4). Only useful once g1 or g5 show retrieval working. Test with pairs ≥ 2.
6. **Gap curriculum on the training side** (short gaps first, then long), keeping the E6 held-out gaps fixed. This is a task-code change.

Lowest value, not recommended: changing sharpness, tuning the phase or τ, toggling contrast, or adjusting gain init or the tanh λ under the current O(T) loop. Each is refuted or fails E2.