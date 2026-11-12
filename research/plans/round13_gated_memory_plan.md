# Round 13, gated memory circuit: design and first experiment (draft 2, 11 Nov 2026)

Status: **draft 2, revised after an adversarial review (Opus, 11 Nov 2026). Approved by Daniel on 12 Nov 2026 ("Start round 13"); step 0 is being built.** Written by the round-12 GPU session from the design discussion with Daniel on 11 Nov 2026. The review's verdict on draft 1 was "do not run as written"; §6 lists its findings and where each is answered.

## 1. Why this, and what it must show

What round 12 measured (register P90–P101; files in `results/evolve/round12/stage1/`):

- Residual re-injection of the model's own recent state gives no benefit under full attention, on any metric.
- The trained hippocampal loop of P5 loses to a parameter-matched adapter with no memory on both models (`m1a_*.json`): its gain was adaptation.
- On RecurrentGemma (trained with a 2,048-token window), reading stored positions back through attention with an oracle selector and ordinal positions raises exact match beyond the window (`m2_readback_*.json`, 50 items per cell):

| Model | Cell | native | oracle, ordinal |
|---|---|---|---|
| `recurrentgemma-2b` | needle | 0.08 | 0.94 |
| | update | 0.12 | 0.46 |
| | pooled | 0.10 | 0.70 |
| `recurrentgemma-2b-it` | needle | 0.10 | 0.80 |
| | update | 0.18 | 0.64 |
| | pooled | 0.14 | 0.72 |

What those numbers do **not** show (corrections to draft 1):

- **0.94 and 0.80 are the needle cell only.** The pooled gain is 0.10 → 0.70 and 0.14 → 0.72. The update cell on the base model (0.46) missed the pre-registered gate of 0.5.
- **The inside-window check failed.** With the facts inside the window the native model scored 0.95 on needle and 0.65 on update (20 items); the requirement was 0.9. Read-back also cost 10 points on inside-window needle items on the base model (0.95 → 0.85). The update task is partly beyond the model even with everything in view.
- **The oracle does not use the question.** It reads back every statement (4–5 per item), so it shows that the read path works, not that selection is hard. A detector of "sentences that state a code" could match it.
- **The InfLLM-style baseline was broken, so "selection is the missing piece" is not established.** Its 0.06 is at chance (about 0.08). Causes found in `readback.py`: block representatives are chosen by a causal softmax inside each block, which favours early filler tokens; only 32 query rows drive selection and most are filler; the score favours large-norm keys; the hit is measured after the answer has started.
- **The answer prefix ("The code for NAME is") makes the task a copy.** The result says little about facts that are asked for in other words.

The questions for round 13, in order:

1. With a repaired baseline and a task that needs the question, how far is a plain selector (the model's own query-key scores, or an off-the-shelf retriever) from an oracle that reads only the queried fact?
2. Does the circuit (dentate key, context tag from the filter bank, CA1 read gate) beat those plain selectors at equal read budget?
3. Only if a gap remains: does a small trained selector close it, against controls that can tell memory from adaptation?

## 2. Design

A side circuit beside a frozen model. It reads hidden states at two layers and returns content only through attention read-back (`experiments/ca2/readback.py`). Nothing is added to the residual stream.

### 2.1 Taps and buffer

- **Early tap** (word-level state): output of an early block. RecurrentGemma: block 2 or 5 (attention layers are 2, 5, 8, 11, 14, 17, 20, 23 of 26).
- **Deep tap** (contextual state): output of block 20.
- **Ring buffer:** the last N = 4,096 states at each tap, RMS-normalised. Sink positions and massive activations (norm above 10× the running median) are excluded from the filters.
- The read-back already keeps an archive of the model's keys and values per attention layer; store entries point into it.

### 2.2 Filter banks (fixed, not learned)

Tap filters over the deep-tap buffer, as mixtures of fixed exponential kernels. Lag 0 is excluded, so every band holds strictly past context.

```
num_k[t] = a_k · num_k[t−1] + g[t−1] · x̃[t−1]
den_k[t] = a_k · den_k[t−1] + g[t−1]
content_k[t]  = num_k[t] / den_k[t]
strength_k[t] = den_k[t]          a_k = exp(−1/τ_k),  τ ∈ {4, 16, 64, 256, 1024}
```

- **Input is centred:** x̃ = x minus a running document mean (τ = 4,096). Without this the bands and the context tag are close to one constant vector per document.
- **Fast bank:** τ = 4, 16, 64, g ≡ 1. Its strength is a constant (about τ), so it is not used as a signal.
- **Slow bank:** τ = 256, 1024, g = 1 on gated events only. Its strength does vary: it is a discounted count of events, and is used.
- **Gate for round one:** a fixed threshold on dentate salience, set once on dev items (not a running top 5%: a running quantile fires at a constant rate whatever the content, which is how P50 failed).
- **Outputs:** differences of neighbouring bands plus the slowest band.

### 2.3 Dentate gate and span builder

- code = top-k(ReLU(R · LN(e_t))²), with e_t the early-tap state, R a fixed random expansion (4× width), k = 5% of units. Key: the sparse code. Salience: its total strength.
- **Span builder (shared by every selector):** candidate spans are sentences, cut at sentence punctuation and capped at 48 tokens. A selector scores positions; a span's score is the mean of its top three positions; whole spans are read back in score order until the budget is full. Every arm, including the baselines, uses this builder, so arms differ only in the score.
- **Budget:** 512 columns hold about 16–24 spans. With 24 distractor statements plus filler sentences, selection is forced.

### 2.4 Episodic store (CA3)

- An entry is a span. It holds the sparse key (pooled over the span), the context tag (band outputs at the span's last token), the **deep-tap state pooled over the span**, an ordinal index and a pointer into the K/V archive.
- **Recall:** score(entry) = overlap(key_t, key_entry) + λ · cos(context_t, tag_entry).

### 2.5 CA1 comparator

Draft 1 summed three deep-tap vectors with identity projections. The review showed that this "agreement" is near constant (the average pairwise cosine of deep states), and that ReLU breaks the cross-term reading. Revised:

- Three pathways, each centred and unit-normalised: direct d_t (current deep state), recalled r_t (deep state of the top-scoring entry), context c_t (slowest band difference).
- **Agreement:** a_t = cos(d_t, r_t) − mean over the other entries in the same store of cos(d_t, r_j). It measures whether the top entry fits the current state better than the store's other entries do, so it is a relevance signal inside one document.
- **Constructive interference:** out = normalise(ReLU(d + r + c)²) is kept as the comparator's output for later use, but the read gate uses a_t, which has a defined null.
- **Read gate:** open when a_t exceeds a threshold set on dev items.
- **Mismatch:** d_t − r_t, recorded for the learned write gate of step 3.

### 2.6 Read-back

Recalled spans are offered to every attention layer as extra key/value columns at ordinal positions just beyond the window, within 512 columns, with an optional recency bias −ω · rank on their attention logits. Note: these ordinal positions lie inside the range the model was trained on; that is the point of the ordinal code, and it is why true positions did worse.

## 3. Task and baselines (changed from round 12)

Model: `google/recurrentgemma-2b` and `-2b-it`; 8,192-token inputs; every arm holds 2,048 columns per attention layer.

**Items.** New flags on `readback_screen.py`:

- **Decoys:** 24 statements of the same form per item, one queried. Chance for a form detector is 1/24.
- **Jitter:** statement positions drawn at random, none in the sink.
- **Paraphrase:** the question names the entity in other words than the statement did ("Dr Ruiz's access code" stated, "the code the doctor uses" asked), and half the items use other fact forms (a date, a city, a colour) to break the code template.
- **No answer prefix** on half the items.
- **Cells:** beyond (fact more than 2,048 tokens back), **near** (1,600–2,040 back: tests the cost of giving up 512 recent columns; at least 100 items), inside (fact within the last 1,024).
- **Natural cell:** WikiText-103 documents (cached): the next-token loss on the second mention of a rare name whose first mention is beyond the window. No template, no question.

**Reference arms.**

| Arm | What it reads back | What it tells us |
|---|---|---|
| native | nothing | floor |
| oracle-queried | the queried statement only | ceiling for selection |
| oracle-all | all 24 statements, if they fit; else a random 512-column subset | what a form detector gets |
| random, foreign | random spans; another document's spans | controls |

**Baselines the circuit must beat** (same span builder, same budget):

| Arm | Score |
|---|---|
| qk top-k | the model's own pre-softmax query-key scores at its attention layers, from the question's tokens |
| retriever | `bge-small-en-v1.5` (cached) on span text against the question text; `Qwen3-Embedding-4B` (cached) as an upper reference |
| infllm (repaired) | representatives by attention received from the whole later context, question rows only, norm-corrected score |

## 4. Experiment

Items: 400 per beyond-window cell, half **dev** (every threshold, λ and choice of variant), half **held out**. At n = 200 a 95% interval on a paired difference is about ±0.07. Gates are stated on interval bounds. One winner per family is chosen on dev and pre-registered before the held-out run; comparisons across families are Holm-corrected.

### Step 0: ranking without generation (about 1 GPU-hour)

One forward pass per item. For each scorer, the rank of the queried statement among the item's candidate spans. No read-back, no decoding.

- Scorers: qk, retriever, repaired infllm, cos-early, dentate overlap, dentate + context tag, each alone and multiplied by write-time salience (dentate strength; the recurrent layers' own gate values, taken by hooks; attention received).
- Measures: recall@budget (is the queried span among those that fit in 512 columns), mean reciprocal rank, on plain, decoy, jitter and paraphrase items.
- [Change, 12 Nov 2026, register DEC18, made before the real runs: statements are 10–14 tokens, so 512 columns hold all 24 and recall@budget measures form detection. The primary measures are top-1 and MRR of the queried span, with recall@k spans (k = 2, 4, 8) beside recall@budget; the stop rule below is read on top-1 ≥ 0.9.]
- **Checks of the circuit's parts, same pass:**
  - Salience: AUC of dentate salience for "this sentence states a fact" against filler.
  - Context tag: AUC for "same passage" against "same document, other passage".
  - Comparator: AUC of a_t for "the top entry is the queried one".
- **Stop rule:** a part with AUC below 0.6 is dropped or redesigned before step 1. If qk or the retriever already reaches recall@budget ≥ 0.9 on decoy + paraphrase items, selection is not the missing piece and the round turns to the read path (update items, near-window cost).

### Step 1: read-back with the selectors (about 4 GPU-hours)

Arms: the reference arms, the three baselines, and the dev-chosen circuit selector.

- **Primary comparison:** circuit selector minus the better of qk and retriever, exact match beyond the window on held-out decoy + paraphrase items, paired.
- **Secondary:** near and inside cells (must not lose: lower bound above −0.05); stale rate on update items against the model's own stale rate with the facts in view; the natural cell's loss on re-mentions.
- **Gate to step 2:** the lower bound of (circuit − best baseline) is above 0, or the circuit matches the baseline within ±0.05 at lower cost.

### Step 2: filter bank and comparator (about 3 GPU-hours)

- **2a context tag:** score with and without the tag (λ from dev).
- **2b read gate:** comparator gate on and off, on beyond, near, inside and foreign-store cells. Primary: exact match with a foreign store and on inside items, gate on minus gate off; no loss beyond the window.
- **2c time code:** recency bias ω ∈ {0, low, high} on update items: new-value rate and stale rate.
- **2d ablations:** centred vs raw input; lag 0 excluded vs included; slow bank gated vs ungated; band differences vs raw bands.
- **Gate to step 3:** the lower bound of (oracle-queried − best training-free configuration) is above 0.15.

### Step 3: trained selector (about 6 GPU-hours, only if the gate passes)

Draft 1 had no gradient path: selection is by hard indices and top-k. Revised:

- **Targets, computed offline:** for each question, the drop in answer loss when each candidate span alone is read back. This needs no labels and no template.
- **Model:** a small scorer on (current deep state, entry key, entry tag, mismatch), trained with a listwise loss against those targets. The frozen model is not in the training graph.
- **Training data:** generic long text (FineWeb, cached), with re-mentions as the queries. Synthetic items are held out entirely.
- **Controls:** the same scorer with entries from another document (foreign) and with entry features shuffled within the store; the untrained circuit; qk and the retriever. A memory-free adapter is not a control for a selector and is dropped here.
- 5 seeds. **Primary comparison:** trained scorer minus the best training-free selector on held-out synthetic items and on the natural cell; the gain must vanish with a shuffled store.

Total: about 14 GPU-hours if every gate passes; step 0 alone decides whether steps 1–3 are worth running.

## 5. Fixed for round one

Memory budget 512 columns; ordinal positions; τ grid as above; no feedback from the bank into itself; no residual injection; no learned time constants; no reasoning tasks.

## 6. Review findings and where they are answered

| # | Finding | Answer |
|---|---|---|
| F1 | The oracle never uses the question; a form detector could match it | §3 decoys, oracle-queried vs oracle-all, jitter, paraphrase |
| F2 | The InfLLM baseline is at chance through implementation faults | §3 repaired infllm; qk and retriever as the baselines to beat |
| F3 | The answer prefix makes it a copy task | §3 no-prefix half, other fact forms, natural cell |
| F4 | Draft 1 misreported round 12 | §1 table and corrections |
| F5 | The inside-window null cannot see the cost of giving up recent columns | §3 near cell |
| F6 | Agreement with identity projections is near constant; r_t undefined | §2.5 centred pathways, contrast against the store's other entries; step 0 AUC |
| F7 | Strength is constant for ungated bands; context tag near constant per document; diagnostic 2e true by construction | §2.2 centred input, strength used for the slow bank only; step 0 AUC; 2e removed |
| F8 | Salience unvalidated; span undefined; running top 5% repeats P50; budget holds few spans | §2.3 span builder, fixed threshold, step 0 AUC |
| F9 | Step 1 did not test the design | Step 1 primary comparison is circuit against qk and retriever |
| F10 | Intervals too wide; gates on point estimates; forking paths; compute understated | §4 n = 200 held out, interval-bound gates, one dev-chosen winner, 14 GPU-hours |
| F11 | Step 3 had no gradient path; adapter is the wrong control | Step 3 offline targets, listwise scorer, foreign and shuffled stores, 5 seeds |
| F12 | Recurrent gate values need hooks; read-back sits inside the trained position range; facts in the sink | Step 0 hooks; §2.6 note; §3 jitter excludes the sink |

## 7. Known risks that remain

- The update task is partly beyond the model with the facts in view (0.65). Its ceiling is the in-window score, not 1.0.
- If qk or a small retriever already selects well, the circuit's contribution reduces to the read gate and the time code. Step 0 will say so before any generation is run.
- The paraphrase items are still templated. The natural cell is the check on that, and it measures loss, not exact match.
- RecurrentGemma cannot continue a prefill; the screen prefills once and steps the question one token at a time.

Figure: [`docs/design/gated-memory-circuit.svg`](../../docs/design/gated-memory-circuit.svg) (page source: `docs/design/gated-memory-circuit.html`). The figure's labels follow draft 2.
