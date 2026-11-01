# hippoformer

Hippocampally inspired memory modules for sequence models in PyTorch. Each part of the hippocampal trisynaptic circuit is built as a neural-network component, and the package offers two ways to put them together:

* **Strategy I (direct circuit).** EC → DG → CA3 → CA2 → CA1 → Subiculum as one residual block, with a one-shot episodic store in CA3.
* **Strategy II (Titans hybrid).** The Titans long-term memory (Behrouz, Zhong & Mirrokni, 2024, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)) extended with a dentate-gyrus write path, a CA1 comparator, and novelty-scaled memory learning rates.

Both strategies can feed a replay-consolidation loop for continual learning, in the spirit of complementary learning systems.

```
                    ┌──────────── temporoammonic skip (EC III → CA1) ─────────────┐
                    │                                                             ▼
 tokens ─► EC ─► DG (expand ×r, k-WTA) ─► CA3 (Hopfield + episodic store) ─► CA2 (FiLM) ─► CA1 comparator ─► Subiculum ─► cortex ─► logits
                                              ▲                                              │ novelty n_t
                                              └──────── novelty-gated one-shot write ◄──────┘
                                                                                             └─► replay priority ─► sleep consolidation
```

## Install

```bash
pip install -e .            # requires torch >= 2.1
pytest -q                   # 29 tests: shapes, causality, sparsity, memory, replay, regressions
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
| Replay | `PrioritizedReplayBuffer`, `Consolidator` | $P(i) \propto p_i^{a}$; replay loss is CE $+\ \tau^2\,\text{KL}(\text{stored}\,\Vert\,\text{current})$; `sleep()` updates only the slow parameters |

## Usage

**Strategy I as a standalone LM, or as context augmentation in front of cortex layers**

```python
from hippoformer import HippocampalLM, lm_loss, novelty_alignment_loss

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
from hippoformer import TitansHippocampalLM

model = TitansHippocampalLM(vocab=32000, d_model=256, n_heads=8, n_layers=4,
                            variant="mac",             # "mac" | "mag" | "mal"
                            chunk=64, window=64, n_persistent=16,
                            use_dg=True, use_ca1=True, novelty_lr=True, sequential=True)
# all four flags False gives plain Titans, so ablations are one flag each
```

**Drop the block into your own transformer**

```python
from hippoformer import HippocampalBlock
blk = HippocampalBlock(d_model=512, n_heads=8)
out = blk(x, store=prev_store, context=speaker_embedding)   # x: (B, T, 512)
x, prev_store = out.hidden, out.store
```

**Replay consolidation**

```python
from hippoformer import PrioritizedReplayBuffer, Consolidator, sequence_priority
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
```

**Associative recall.** The model studies 6 random key→value pairs, sees 40 filler tokens, then is queried on every pair in shuffled order. Only the recalled values are scored, and chance is 1/16. Local models use a 16-token window, segment, or chunk, so the bindings sit outside their attention by the time they're queried.

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
* **Novelty supervision.** `novelty_alignment_loss` trains CA1's mismatch signal to predict the z-scored next-token loss. Without it, novelty is only shaped indirectly.

## References

Behrouz, Zhong & Mirrokni (2024), Titans: Learning to Memorize at Test Time. Ramsauer et al. (2020), Hopfield Networks is All You Need. McClelland, McNaughton & O'Reilly (1995), complementary learning systems. Wilson & McNaughton (1994), replay during sleep. Lisman & Grace (2005), the hippocampal–VTA loop and CA1 novelty detection. Perez et al. (2018), FiLM. Howard & Kahana (2002), temporal context model. Buzzega et al. (2020), Dark Experience Replay.
