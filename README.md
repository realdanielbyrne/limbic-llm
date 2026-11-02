# limbic-llm

Hippocampally inspired memory modules for sequence models in PyTorch. Each part of the hippocampal trisynaptic circuit is built as a neural-network component, and the package offers two ways to put them together:

* **Strategy I (direct circuit).** EC → DG → CA3 → CA2 → CA1 → Subiculum as one residual block, with a one-shot episodic store in CA3.
* **Strategy II (Titans hybrid).** The Titans long-term memory (Behrouz, Zhong & Mirrokni, 2024, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)) extended with a dentate-gyrus write path, a CA1 comparator, and novelty-scaled memory learning rates.
* **Gated DeltaNet memory.** The linear, momentum-free special case of the Titans memory (Yang, Kautz & Hatamizadeh, 2025, [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)), usable as a token mixer, as the memory inside Titans layers, or with dentate-gyrus addressing.

Both strategies can feed a replay-consolidation loop for continual learning, in the spirit of complementary learning systems.

The package was first called *hippoformer*; it was renamed because an ICLR 2026 paper already uses that name (Li et al., 2026). The bibliography is in [REFERENCES.md](REFERENCES.md), with BibTeX in [`docs/references.bib`](docs/references.bib).

```
                    ┌──────────── temporoammonic skip (EC III → CA1) ─────────────┐
                    │                                                             ▼
 tokens ─► EC ─► DG (expand ×r, k-WTA) ─► CA3 (Hopfield + episodic store) ─► CA2 (FiLM) ─► CA1 comparator ─► Subiculum ─► cortex ─► logits
                                              ▲                                              │ novelty n_t
                                              └──────── novelty-gated one-shot write ◄──────┘
                                                                                             └─► replay priority ─► sleep consolidation
```

[`docs/architecture.html`](docs/architecture.html) diagrams how the package fits together; open it in a browser. The `docs/` folder also holds the background write-ups: a hippocampus report, the ML-architecture parallels document, and an interactive circuit diagram. `docs/notebooks/` holds the Activation Addition GPT-2-XL notebook (Turner et al.), a candidate mechanism for a CA2 side channel.

## Install

```bash
pip install -e .            # requires torch >= 2.1
pytest -q                   # 51 tests: shapes, causality, sparsity, memory, replay, Gated DeltaNet, regressions
```

## Component map

| Region | Module | Computation |
|---|---|---|
| EC | `TransformerLayer` (in `HippocampalLM.ec`) | Contextual embedding $x_t$ |
| DG | `DentateGyrus` | $y = g \odot \mathbb{1}[h \in \text{top-}k(h)] \odot \text{ReLU}(h)$, with $h = W_{\text{exp}}\,\text{LN}(x)$ and $k = \lceil s \cdot d r \rceil$ |
| DG neurogenesis | `DentateGyrus.grow / tick` | Reserve units are switched on with plasticity $1 + \beta e^{-\text{age}/\tau}$ |
| CA3 | `CA3Autoassociative` | $\xi \leftarrow V^\top \text{softmax}(\beta K \xi)$ over $[\text{store}; h_{\le t}]$, iterated `n_iter` times |
| CA3 store | `EpisodicStore`, `CA3Autoassociative.write` | Writes the top-$m$ most novel transitions $(h_t \mapsto h_{t+1})$ into the lowest-salience slots; salience decays by $\gamma$ (LTD) and grows with retrieval (reconsolidation) |
| CA2 | `CA2Modulator` | $y = x \odot (1+\gamma(c)) + \beta(c)$; $c$ is an external context or a temporal context $c_t = (1-\rho)\sum_s \rho^{t-s} x_s$ |
| CA1 | `CA1Comparator` | $\phi = [a; r; a \odot r; \lvert a-r \rvert]$, $\text{fused} = W\phi$, $n_t = \sigma(\text{MLP}(\phi))$ |
| Subiculum | `Subiculum` | SwiGLU output scaled by a novelty-dependent gain $\text{softplus}(a\,n_t + b)$ |
| Titans LMM | `NeuralMemory` | $\ell = \lVert M(k_t) - v_t \rVert^2$, $S_t = \eta S_{t-1} - \theta_t \nabla\ell$, $M_t = (1-\alpha)M_{t-1} + S_t$; with `sequential=True`, $v_t = W_V x_{t+1}$ |
| Gated DeltaNet | `gated_delta_scan`, `gated_delta_chunked` | $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$, $o_t = S_t q_t$ (exact; chunked form solves $(I+L)U = \operatorname{diag}(\beta)(V - \operatorname{diag}(\Gamma)KS_0^\top)$ per chunk) |
| GDN addressing | `GatedDeltaNet(addressing=…)` | `"gdn"` standard q/k/v + short conv; `"tied"` address $a_t$ is the query at $t$ and the key of the write $a_{t-1} \mapsto v_t$; `"dg"` the same with $a_t = \text{norm}(\text{k-WTA}(W_a x_t))$ |
| Replay | `PrioritizedReplayBuffer`, `Consolidator` | $P(i) \propto p_i^{a}$; replay loss is CE $+\ \tau^2\,\text{KL}(\text{stored}\,\Vert\,\text{current})$; `sleep()` updates only the slow parameters |

## Usage

**Strategy I as a standalone LM, or as context augmentation in front of cortex layers**

```python
from limbic_llm import HippocampalLM, lm_loss, novelty_alignment_loss

model = HippocampalLM(vocab=32000, d_model=256, n_heads=8,
                      n_ec=2, n_hpc=1, n_neocortex=4,    # n_neocortex=0 gives standalone mode
                      segment=128, ca3_capacity=512, ca3_write=32,
                      dg_expansion=8, dg_sparsity=0.04)
out = model(tokens)                                   # out.logits, out.novelty, out.state
tl = lm_loss(out.logits[:, :-1], tokens[:, 1:], reduction="none")
loss = tl.mean() + 0.1 * novelty_alignment_loss(out.novelty[:, :-1], tl)

out2 = model(next_tokens, state=out.state)            # carry the episodic store across a stream
```

**Strategy II (Titans + hippocampus)**

```python
from limbic_llm import TitansHippocampalLM

model = TitansHippocampalLM(vocab=32000, d_model=256, n_heads=8, n_layers=4,
                            variant="mac",             # "mac" | "mag" | "mal"
                            chunk=64, window=64, n_persistent=16,
                            use_dg=True, use_ca1=True, novelty_lr=True, sequential=True)
# all four flags False gives plain Titans, so ablations are one flag each
```

**Gated DeltaNet (linear CA3-style memory)**

```python
from limbic_llm import GatedDeltaLM, TitansHippocampalLM

# token mixer stack; attn_every=4 gives the 3:1 DeltaNet:attention mix of Qwen3-Next
lm = GatedDeltaLM(vocab=32000, d_model=256, n_heads=8, n_layers=8, attn_every=4,
                  addressing="dg", key_expansion=8, dg_sparsity=0.05)
out = lm(tokens); out2 = lm(next_tokens, state=out.state)    # constant-size state, streamable

# or as the memory inside Titans layers (exact per-token delta rule instead of chunk-start gradients)
titans = TitansHippocampalLM(vocab=32000, d_model=256, n_layers=4, memory_type="gated_delta",
                             mem_heads=8, use_dg=True, sequential=True)
```

Gated DeltaNet is exactly one SGD step on the Titans loss $\tfrac12\lVert Sk_t - v_t\rVert^2$, taken at the decayed state $\alpha_t S_{t-1}$ with step size $\beta_t$. So $\alpha_t$ is the forgetting (weight-decay / LTD) gate, $\beta_t$ the encoding strength, and the bracket $v_t - \alpha_t S_{t-1} k_t$ is the CA1-style prediction error; `test_equals_one_sgd_step_on_titans_loss_at_decayed_state` checks this. With one-hot keys and $\alpha = \beta = 1$ the state is exact RAM, as in [Qwenputer](https://divagr.com/blog/qwenputer), which is the limit that dentate-gyrus pattern separation approaches.

**Drop the block into your own transformer**

```python
from limbic_llm import HippocampalBlock
blk = HippocampalBlock(d_model=512, n_heads=8)
out = blk(x, store=prev_store, context=speaker_embedding)   # x: (B, T, 512)
x, prev_store = out.hidden, out.store
```

**Replay consolidation**

```python
from limbic_llm import PrioritizedReplayBuffer, Consolidator, sequence_priority
buf = PrioritizedReplayBuffer(capacity=10_000)
cons = Consolidator(model, buf, slow_params=model.neocortex.parameters(), lr=1e-4)

buf.add(seq, sequence_priority(out.novelty, tl), out.logits)   # while awake
loss = task_loss + cons.replay_loss(batch_size=16)             # interleaved replay
cons.sleep(steps=200)                                          # offline consolidation
```

`replay.orthogonal_projection(params, bases)` is a gradient-projection hook for sleep updates. It protects subspaces from earlier tasks, which is the place to plug in ICA-identified functional networks.

## Experiments

```bash
python experiments/assoc_recall.py --steps 3000     # one-shot recall across a gap longer than attention
python experiments/continual.py --steps 600         # sequential fact tasks: naive vs. replay
python experiments/gdn_capacity.py --pairs 8,16,32  # does DG addressing raise linear-memory capacity?
```

**Associative recall.** The model studies 6 random key→value pairs, sees 40 filler tokens, then is queried on every pair in shuffled order. Only the recalled values are scored, and chance is 1/16. Local models use a 16-token window, segment, or chunk, so the bindings sit outside their attention by the time they're queried.

**Gated DeltaNet capacity.** Pure Gated DeltaNet models (no attention) are trained on associative recall at increasing load (8, 16 and 32 pairs, with 64 possible keys). Four variants are compared: standard keys at $d_k = 16$; standard, tied dense, and tied DG-sparse keys at $d_k = 128$. Tied dense against DG at the same state size isolates the effect of sparsity.

**Continual learning.** Three tasks, each with its own fixed key→value facts, are learned one after another. Only the first occurrence of a key in a sequence is scored, so the model must hold the facts in its weights. The script reports the accuracy matrix, average final accuracy, and backward transfer (BWT).

Results from a CPU run are in `results/` and summarized in [`RESULTS.md`](RESULTS.md). In short: Strategy I reaches 0.41 recall and Titans 0.48, against 0.06 for windowed attention and 0.49 for full attention. Replay removes catastrophic forgetting (BWT −0.73 → 0.00). The combined Titans+hippocampus hybrid underperforms, and the reasons are discussed there.

## Design decisions the experiments forced

These three changes came out of debugging. Each is covered by a test.

1. **CA3 stores sequences, not just patterns.** A purely autoassociative store returns the cue itself, so a key can't recall its value. The store now holds transitions $h_t \mapsto h_{t+1}$, following the asymmetric-STDP account of CA3 sequence memory (Levy, 1996; Jensen & Lisman, 1996). The same idea gives the `sequential` flag for Titans writes. Without it, neither memory system got above chance on recall.
2. **The Titans meta-gradient is a partial derivative.** In MAC and MAL, the chunk being written was computed from an earlier read of the same memory. Differentiating $\ell$ with respect to $M$ directly then also flowed through that read path, so training applied a different update than inference (low training loss, chance-level evaluation). The gradient is now taken with respect to a fresh alias of $M$; see `test_titans_train_and_eval_apply_same_memory_update`.
3. **Memory loss scale and a trust region.** Averaging $\lVert\cdot\rVert^2$ over features shrinks writes by $1/d$, and the memory barely learns. Summing as the paper does can diverge when correlated keys accumulate under momentum. The code sums, and clips each sample's fused chunk update to `max_update_norm`. Sparse DG keys were stable even without the clip, which is pattern separation earning its keep.

## Implementation notes and simplifications

* **Titans parallelization.** Inside each chunk, every gradient is taken against the memory state at the start of the chunk, as in the paper's mini-batch scheme. $\theta_t$ is per token, while $\eta$ and $\alpha$ are averaged per chunk, so each chunk applies one fused update.
* **Meta-gradients.** With `meta_grad=True`, training differentiates through the memory update (second order). Attention is written as explicit softmax because PyTorch's fused CPU kernels have no double-backward. `meta_grad=False` gives a cheaper first-order version.
* **Causality.** Memory is read before a chunk or segment and written after it. Tests check that perturbing a token never changes logits at earlier positions.
* **CA3 writes.** Slot choice is not differentiable, but stored patterns keep their autograd graph within a call, so later recall losses train the encoding path. The store is detached between calls (truncated BPTT).
* **Gated DeltaNet kernels.** `gated_delta_scan` is the reference per-token loop. `gated_delta_chunked` gives the same result to within $10^{-4}$ (tested) and is used by default. Decay ratios are computed as $\exp(g_C - g_i)$, so strong decay can't produce 0/0. There is no fused GPU kernel; for long sequences on GPU, use a dedicated implementation such as flash-linear-attention.
* **Novelty supervision.** `novelty_alignment_loss` trains CA1's mismatch signal to predict the z-scored next-token loss. Without it, novelty is only shaped indirectly.

## References

See [REFERENCES.md](REFERENCES.md) for the full bibliography, grouped by circuit component, and [`docs/references.bib`](docs/references.bib) for BibTeX.
