# The episodic-slot side memory and its five strongest variants

*A description for readers who have not seen the project. It is written from the code
(`experiments/evolve/w1s_slots.py` and its neighbours) and from the interim results of round 11's first
screening stage, as of 5 November 2026. File paths and record IDs are listed in the last section.*

## 1. The problem

The base model is **Qwen3-1.7B-Base**: 28 transformer blocks with a hidden width of 2048. It is
**frozen**, so none of its weights change. Every attention layer is limited by a mask to
**non-overlapping 64-token blocks**. A token can attend only to the tokens in its own block and to
one fixed beginning-of-sequence token, the "sink", which every query can see; without it, models
that have had their window cut degrade badly. Nothing crosses a block boundary inside the model.
Anything further back than the current block is invisible to it.

The test task is **key–value recall**. A sequence of 512–2048 tokens (up to 8192 in the stress cells)
contains several statements of the form

    The code for Emma is 4821.

They are scattered at random positions in filler text, which is either natural Wikipedia prose or
random tokens. At the end comes a query, `The code for Emma is`, and the model must produce
`4821`. The last fact is at least 256 tokens before the query, so it always sits in an earlier block.
A run is scored by **exact match**: all five answer tokens right (the lone space before the digits
and the four digits), each with the earlier ones given (teacher forcing).

- With the window and nothing else, the model scores **0.00** exact, as it must: it cannot see the fact.
- With full attention it scores about **0.98–1.00** on the recall grid. It is a reference rather
  than a ceiling: on reworded facts (jitter, 0.57) and on two of the paraphrased queries (paraphrase 3,
  0.44; paraphrase 2, 0.86; defined in section 3) it scores below the memory, I1 or its variants. I1
  beats it on jitter (0.96 on seed 1). Nor is it a fair rival. Full
  attention keeps a key and a value for every past token in every layer, about **115 KB per token**
  in bf16 (28 layers × 8 KV heads × 128 dims × 2 × 2 bytes), so its memory grows with the length.

The object of study is a small trained module, the **side memory**. It reads the hidden state that
leaves **block 21** (0-indexed, so six frozen blocks follow it) and adds a vector back into the
residual stream at the same point. It is the only path by which information from earlier 64-token windows can
reach the query. Its state has a **fixed size**: about 0.91 million floats, or **3.6 MB** in
float32, whatever the length of the sequence. That is the KV cache of roughly 31 tokens. Its
trainable parameters number about 0.45 million. It is trained on the recall task only: cross-entropy
on the code's tokens, 6,000 steps, batch 8, 512-token sequences with natural filler, codes for 20
training names, and a curriculum of 1, then 2, then 3 facts per sequence.

## 2. The base design (I1), step by step

The base recipe is called **I1**. Its memory works like a small diary. It notices moments that seem
worth remembering, copies the next 24 tokens around each one into a slot of its own, keeps the 16
strongest moments, and at query time finds the right slot and **replays** it from the point that
matches the present, one token at a time.

### Data flow

```
 tokens ─► frozen blocks 0–20 (64-token block attention) ─► h_t (block 21 output, 2048-d)
                                                              │
                         clock: h_t ← h_t + C·(leaky timers)  │   (zero-initialised, learned)
                                                              │
        ┌─────────────────────────────────────────────────────┤
        │ per-token features                                  │
        │   x_t = LN(h_t)            2048-d   alignment code   │
        │   y_t = W_k x_t            128-d    learned key code │
        │   v_t = W_a LN'(h_t)       32-d     value            │
        │   p_t = σ(e^a · w_g·x_t / sd_t + b)  gate strength   │
        └───────────────┬─────────────────────────────────────┘
                        ▼
   GATE      onset s  ⇔  p_s is a strict maximum of p within ±8 tokens
                        ▼
   WINDOW    entries s … s+23; transitions (x_j, v_{j+1}) for j = s … s+22;
             slot key K_s = mean(y_s … y_{s+7})
             the window becomes readable at t = s + 23
                        ▼
   STORE     the 16 readable windows with the largest p_s, one slot each
                        ▼
   SELECT    a_{t,k} = softmax over slots + a null slot of
             log p_k + β_L·cos(q_t, K_k) − ω·(t − s_k),   q_t = mean(y_{t−7} … y_t)
                        ▼
   ALIGN     inside slot k: α_{t,k,j} ∝ exp(β_a · ĉos(x_t, x_j))
                        ▼
   REPLAY    r_t = Σ_k a_{t,k} Σ_j α_{t,k,j} v_{j+1}    (32-d);   o_t = W_b r_t   (2048-d)
                        ▼
   INJECT    h_t ← h_t + (part of o_t orthogonal to h_t, scaled to ≤ 0.9·σ(g)·‖h_t‖) ─► frozen blocks 22–27
```

### The steps

**0. Inputs and the clock.** The block-21 output `h_t` (float32) first passes through a small
"clock". This adds a learned linear map of 21 leaky-integrator and damped-oscillator traces (7
timescales, 4–256 tokens; one integrator and two oscillator components each) of a learned per-token
salience. The map starts at zero, so the clock
starts switched off. An earlier measurement found that zeroing it after training changes exact
recall by at most 0.03. It is part of the recipe but carries little. Everything below reads the
clocked `h_t`. `LN` is LayerNorm without learned scale or shift, and `LN'` is a LayerNorm with them.

**1. Gate (deciding where an event starts).** A learned vector `w_g` scores each token: `w_g·x_t`.
That score is divided by its own causal running standard deviation `sd_t`, computed over positions
≤ t, so the gate means the same thing at any input scale. It then becomes a strength
`p_t = σ(e^a · w_g·x_t / sd_t + b)`, with `a` and `b` learned. At the start, `b` is set for an average
density of `p_t ≈ 0.12` under a centred drive. I1 turns centring off (the running mean of `w_g·x` is
not subtracted), so its actual starting density can differ. An **onset** is a position `s` where `p_s` is *strictly greater than every
other `p` within ±8 tokens* (non-maximum suppression with radius 8). Onsets are therefore at least
9 tokens apart.

There is **no on/off threshold and no hysteresis** in this recipe. Any local peak is a candidate.
Its strength `p_s` matters only for which candidates are kept (step 3) and as a prior in the read
(step 4). The code also contains a hysteresis gate (`hyst`) and a hard two-threshold Schmitt gate
(`schmitt`). They also open a fixed window of `n` entries at the moment a smoothed score switches on
(with two thresholds, `θ_on` and `θ_off`, for `schmitt`). `hyst` additionally weights each entry by
its on-state; switching off never shortens a window. So in no gate does falling below a threshold
stop a write. **None of the five recipes below uses these gates**; all use the `point` gate.

**2. Window (a phrase, kept whole).** Onset `s` opens a window of `n = 24` entries, `s … s+23`. No
entries are recorded before the onset. For each entry `j` except the last, the slot keeps a
**transition**: the alignment code `x_j` (2048 floats) and the value of the *next* token, `v_{j+1}`
(32 floats). That makes 23 transitions per window. The slot's **key** is the mean of the learned
key code over the window's first 8 entries: `K_s = mean(y_s, …, y_{s+7})`, with
`y = W_k x` a learned 2048 → 128 map. A window becomes **readable** ("closes") at `t = s + 23`,
once all of its entries exist and the ±8 suppression around `s` has been settled. Nothing is ever
read before it exists, so the module is strictly causal.

**3. Store (16 slots).** At time `t`, the memory holds the `K = 16` readable windows with the
largest `p_s`. In streaming form: when a new window closes, it takes a slot, and if there are now 17,
the one with the smallest `p` is evicted (the older one on ties). Retention is by strength, not by
age. Each window has a slot of its own, so facts are never superposed in one matrix. Windows may
overlap, since onsets can be as close as 9 tokens, and each is stored separately.

**4. Select (which slot?).** At every position `t`, the query is
`q_t = normalise(mean(y_{t−7}, …, y_t))`, the learned key code averaged over the last 8 tokens. Each
held slot `k` gets the logit

    ℓ_{t,k} = log p_k  +  β_L · cos(q_t, K_k)  −  ω · (t − s_k)

and a **null slot**, meaning "nothing relevant", has a learned logit that starts at 3.0. The read
weights are `a_t = softmax(ℓ_{t,1..16}, ℓ_null)`. `β_L` (learned, starting at 10; about 12 after
training) sets how sharp the key match is. `ω` is a learned **linear age penalty**, starting at
0.002 per token and about 0.0012–0.0015 after training. At an age of 1,500 tokens it costs about
2 logits, which is as much as a cosine margin of about 0.15. That is why the oldest facts in long
sequences lose to younger filler windows, and why two of the variants below remove it.

**5. Align (where in the slot are we?).** Inside each slot, the current token's code `x_t` is
compared with every stored code `x_j`:

    α_{t,k,j} = exp(β_a · ĉos(x_t, x_j)) / Σ_{j' ∈ slot k} exp(β_a · ĉos(x_t, x_j'))

`ĉos` is a **centred cosine**: both vectors have the mean of the held slots' "lexical" keys
subtracted first. Those are the mean of `x` over each slot's first 8 entries. This removes the
direction that all states share. `β_a` is a learned sharpness that starts at 30 and is capped at 40.
Trained I1 runs drift down to about 11–16.

**6. Replay (the offset read).** The memory returns the stored value of the token *after* the
matched one:

    r_t = Σ_k a_{t,k} Σ_j α_{t,k,j} v_{j+1}          (32-d)
    o_t = W_b r_t                                   (2048-d)

This is the key trick. The read at `t` is injected at `t`, which is where the model predicts token
`t+1`. Matching "where am I" and returning "what came next" turns a stored window into a sequence
that can be played back.

**7. Inject.** As in the project's earlier CA1-style output stage, only the part of `o_t` orthogonal
to `h_t` is added. Its norm is tied to `‖h_t‖`:

    e_t   = o_t − (o_t·ĥ_t) ĥ_t
    inj_t = 0.9 · σ(g) · tanh(‖o_t‖/λ) · ‖h_t‖ · e_t/‖e_t‖ ,      inj_0 = 0
    h_t  ← h_t + inj_t

`g` and `λ` are learned. After training, `σ(g) ≈ 0.38–0.40`, so the injection is at most about 35%
of the hidden state's norm. Blocks 22–27 of the frozen model then turn the nudged state into
next-token logits.

### Sizes

| | Value |
|---|---|
| Slots `K` | 16 |
| Window `n` | 24 entries, no pre-onset entries |
| Suppression radius | 8 |
| Pool for keys and query | 8 tokens |
| Learned key width | 128 |
| Value width | 32 |
| Alignment code | 2048 (the LayerNormed block-21 state) |
| State at full occupancy | 906,727 floats ≈ 3.63 MB: slot codes 786,432 (87%); ring buffer of not-yet-closed entries 70,720; slot keys 34,848; values 12,288; other 2,439 |
| Trainable parameters | 450,569: learned key map 262,144; value maps 131,072; clock 51,201; LayerNorm 4,096; gate 2,050; scalars |

### Worked example: replaying "The code for Emma is 4821"

In Qwen's tokenizer the fact is 11 tokens: `␣The ␣code ␣for ␣Emma ␣is ␣ 4 8 2 1 .`. Each digit is
its own token, and there is a lone space token before the digits. Suppose these sit at positions
100–110, so the space is 105 and the `4` is 106.

1. **Writing.** On a trained run, the gate peaks at a fixed offset before the first digit. On the
   round-10 checkpoints it peaked 6 tokens before it on one seed (here `␣The`, position 100) and 3
   tokens before it on another (`␣Emma`). Take `s = 100`. The window covers 100–123: the whole fact
   plus 13 filler tokens. It closes at `t = 123`. Its key is the mean learned code of
   `␣The ␣code ␣for ␣Emma ␣is ␣ 4 8`. It keeps 23 transitions, among them
   `x_105 → v_106` (space → "4"), `x_106 → v_107` ("4" → "8"), `x_107 → v_108` and `x_108 → v_109`.
2. **Surviving.** Over the next ~1,500 tokens, other windows open on filler and on the other facts.
   The Emma window stays in memory as long as fewer than 16 readable windows, earlier or later, have
   a larger `p`.
3. **Selecting.** At the query `… ␣The ␣code ␣for ␣Emma ␣is ␣`, the query vector is the mean code
   of the last 8 tokens, mostly `The code for Emma is`. Its cosine with the Emma slot's key is
   higher than with the slots of other names or with filler, so `a` concentrates on the Emma slot.
4. **Aligning and replaying, token by token.**
   - At the query's lone space, the current code `x_t` is closest to the stored space `x_105`
     (the same token after the same `is`). The read returns `v_106`, the stored state of `4`, and
     the injection pushes the next-token prediction toward `4`.
   - At the next position the input is `4`. It matches stored `x_106`, the read returns `v_107`,
     and the prediction moves toward `8`.
   - The same happens for `2` and then `1`.

   The slot is chosen afresh at every token, and the alignment carries the position within it.

Two consequences follow. A query phrased differently from the fact has to match both the key (to find
the slot) and a stored token (to know where to start), which is why paraphrased queries are a
separate test. Repeated digits (`4848`) make the alignment ambiguous, so the evaluation also scores
them separately.

Two ablations show that both halves are necessary. Replacing the separate slots with one
superposed delta-rule matrix, or removing the alignment (averaging a slot's values instead),
collapses I1. On the same seeds, exact recall falls by 0.58–0.82 on average over the four referenced
grid cells (0.45–0.88 cell by cell).

## 3. The five strongest variants

Each variant is **I1 plus one change**. Each was trained on seeds 1 and 4 and evaluated on the same
sequences as I1. The quantity reported is the **paired difference in exact recall** against I1 on the
same seed, with a 95% interval: "tie" means the interval includes zero. Unless stated otherwise,
the headline cell is random filler, 2,048 tokens, 5 codes. There I1 scores 0.86 on seed 1 and
0.56 on seed 4. **Seed 4 is I1's failing seed**, so part of any gain on it may be training luck
rather than the change. The results below are interim. They come from a 2-seed screen and quote a
subset of an 18-cell grid (lengths 512, 1024 and 2048 × 1, 3 or 5 codes × natural or random filler)
plus robustness cells: held-out names, jittered phrasings, paraphrased queries. A 10-seed second
stage decides.

The paraphrase cells use the plain facts with four queries: (1) `Q: What is the code for Emma? A:`,
(2) `Emma's code is`, (3) `Remember the code for Emma. It is` and (4) `The number assigned to Emma
is`. Only the first does not end in `is`, the token the alignment normally starts from. All four are
reported below as "paraphrase 1–4".

### 3.1 Gated lexical keys from an early block (A11)

**Change.** `--w1s-key both --w1s-taps 3 --w1s-lex-init -2`. The memory now also reads the
output of **block 3**, where a token's state still looks much like its embedding. This has three
effects in the code, not one:

1. **Two-block gate and learned key.** The gate score and the learned key code read the LayerNormed
   states of both block 3 and block 21, through one linear map per stream, summed (each map scaled
   by 1/√2 at the start).
2. **A lexical key.** Each slot also gets `K^lex_s = mean of LN(block-3 state)` over its first 8
   entries. The query is the same mean over the last 8 tokens. Both are centred on the mean of the
   held slots' lexical keys. The selection logit gains a term:

       ℓ_{t,k} += β_lex · ĉos(q^lex_t, K^lex_k),    β_lex = e^{−2} ≈ 0.14 at the start, learned

   It starts near off, so it cannot outvote the learned key early in training.
3. **The alignment code moves to block 3.** The code takes the lowest tap as its "lexical" stream,
   and the alignment of step 5 uses the same stream. So `x_j` is now the LayerNormed block-3 state,
   not the block-21 state.

**Why it might help.** A learned key trained on 20 names may memorise those names. A block-3 state
is closer to the surface token, so it should match a new name, or a reworded query, more literally.
A block-3 alignment code matches "the same token" more literally too, which suits a replay that
works token by token.

**What training learned.** `β_lex` grew only to `e^{−1.55} ≈ 0.21` (seed 1) and `e^{−1.36} ≈ 0.26`
(seed 4), against `β_L ≈ 12` for the learned key. A cosine difference of 0.5 in the lexical key is
worth about 0.1 logits. **The lexical key itself is therefore nearly unused.** The gain more
probably comes from the block-3 alignment code or the two-block gate and key. The alignment
sharpness ended where I1's did (log β_a 2.73 and 2.35, against 2.77 and 2.39). A planned arm runs A11
without the lexical key (taps 3 and 21, learned key only) to separate these effects.

**Results.**
- Seed 1: +0.10 (2048 × 5, random filler); paraphrase 1 +0.04, paraphrase 2 +0.03, paraphrase 3 tie.
- **Seed 1, paraphrase 4: −0.24 [−0.30, −0.19]** (0.49 against I1's 0.73), the largest loss of any of
  the five arms. On seed 4 the same cell is +0.19.
- Seed 4: +0.36; paraphrases 1–4 +0.13, +0.15, +0.14 and +0.19; jitter +0.11.
- Held-out names: tied on both seeds (0.66 and 0.62 absolute).

**Cost.** The state is unchanged at 906,727 floats: the alignment code has the same width, it only
comes from a different block. Parameters rise by 264,193 (a second 2048 → 128 key map, a second gate
vector and `β_lex`), to 714,762. Compute adds one cached block-3 activation per token.

### 3.2 Ordinal time instead of an age penalty (A1b)

**Change.** `--w1s-recency-form none --w1s-time-feat ordinal`. The age penalty `−ω·(t − s_k)` is
removed. In its place:

    ℓ_{t,k} += u_t · E[r_k],     u_t = W_u q_t  (128 → 16),   E ∈ ℝ^{16×16}  (zero-initialised)

Here `r_k` is the slot's **rank among the held slots** (0 = newest … 15 = oldest; the code counts
the held slots newer than `k`). The query decides, through `u_t`, how much each rank should matter.

**Why it might help.** A penalty that grows with token age has a range that depends on the length.
Trained at 512 tokens, it is too harsh at 2,048, where old facts lose to young filler. A rank is
bounded by `K`, so the same table applies at any length. Because `u_t` depends on the query, the
table can also learn "prefer the newest matching slot" for some queries without imposing it on
others.

**What training learned.** The norms of the rows of `E` show how much the table can separate
ranks. The logit shift for rank `r` is at most `‖W_u‖₂ · ‖E_r‖` (the query is normalised), with
`‖W_u‖₂` = 2.6 (seed 1) and 2.2 (seed 4), so the row norms understate it by that factor.
- Seed 4: 0.46 for the newest, then 0.28 and 0.26, then 0.10–0.18 for the rest.
- Seed 1: 0.61, 0.51, 0.52 and 0.49 for the four newest, 0.22 and 0.19, then 0.02–0.10, and 0.27 for
  the oldest.

Both runs separate the few newest ranks most. A norm has no sign, so whether that is a bonus for
recent slots or a penalty on older ones depends on `u_t`, and it has not been measured.

**Results.**
- Seed 1: +0.08 (2048 × 5); paraphrase 1 +0.03, paraphrase 4 +0.10; paraphrases 2 and 3 tie.
- Seed 4: +0.40; held-out names +0.16 (to 0.73, the best held-out score so far); jitter +0.10;
  paraphrases 1–4 +0.12, +0.11, +0.10 and +0.23.
- No cell is worse.

**Cost.** Parameters +2,304 (`E`: 256, `W_u`: 2,048). The state is unchanged, because a slot's rank
follows from the slot order the store already keeps.

### 3.3 Sharper alignment at the start (A12)

**Change.** `--w1s-align-init 3.68`. The alignment sharpness starts at `log β_a = 3.68`
(`β_a ≈ 39.6`, just under the cap of 40), not at `log 30 = 3.40`. Nothing else changes.

**Why it might help.** On I1's failing seed, `β_a` drifted down further than on the good seeds. A
blunt alignment spreads the replay over neighbouring tokens, which mixes digits. Starting sharper
may keep training in the regime where the replay picks one token. An earlier test found that
sharpening a *trained* I1 at evaluation time did not help. So any effect here acts through training,
not through the value of `β_a` alone.

**What training learned.** I1 drifted to `log β_a` 2.77 / 2.39 (`β_a ≈ 16 / 11`, seeds 1 / 4). A12
stayed at 3.37 / 3.15 (`β_a ≈ 29 / 23`). The higher start persists through training.

**Results.**
- Seed 1: tie on 2048 × 5; paraphrase 1 +0.09, paraphrase 4 +0.13; paraphrases 2 and 3 tie.
- Seed 4: +0.31; held-out names +0.14; paraphrases 1–4 +0.12, +0.13, +0.10 and +0.25.
- No cell is worse.

**Cost.** None: one starting value changes.

### 3.4 A teacher signal for the gate from full attention (A15)

**Change.** `--admit-target teacher --admit-weight 1.0`. This affects **training only**. On each
training batch, an extra forward pass of the frozen model runs *with full attention*, without
gradients and with explicit ("eager") attention so that the weights can be read. It records how
much attention the query tokens (`␣The ␣code ␣for ␣Emma ␣is` and the answer's space) pay to every
earlier position, averaged over heads, all layers and those query tokens.

For each candidate onset `s` whose window `[s, s+23]` closed before the query, the **target** is the
attention mass inside that window, normalised over the candidates. The **student** is the gate's
strength `p_s` over the same candidates, normalised. The loss adds

    λ · KL(target ‖ student),    λ = 1

to the recall loss. Its gradient reaches the gate (`w_g`, `a`, `b`) and, through the gate's input,
the clock; nothing else. (The docstring of `teacher.py` says "only the gate", which misses the
clock.) At inference the module is identical to I1,
in parameters, state and computation.

**Why it might help.** In I1, the gate learns only indirectly, through the `log p_k` prior in the
read softmax. Full attention already "knows" which earlier tokens the query needs. This turns that
knowledge into a direct target for where the gate should fire and how strongly. In effect, the
gate is distilled from the model's own attention, which the windowed model cannot see at test time.

**What training learned.** On seed 4 the gate moved more than in I1: the log gain `a` fell to −0.35
and the bias `b` to −2.56, against 0.17 and −2.21 for I1. Its strengths are lower and less contrasted.
Because `p` is monotone in the gate score, a change in `a` and `b` alone moves no onset and evicts no
different window; it only makes the `log p` prior in the read lower and flatter. Changes in `w_g` can
move onsets and retention, but whether they moved the peaks onto facts has not been measured.

**Results.**
- Seed 1: tie on 2048 × 5; paraphrase 1 +0.07, paraphrase 4 +0.17; paraphrases 2 and 3 tie.
- Seed 4: +0.36; held-out names +0.14; jitter +0.10; paraphrase 2 +0.10, paraphrase 4 +0.14.
- On seed 4, paraphrases 1 and 3 have point estimates below I1's (0.81 against 0.85, and 0.82 against
  0.84), but both intervals include zero (−0.04 [−0.09, +0.01]; −0.02 [−0.07, +0.02]): ties.

**Cost.** Nothing at inference. Training needs one extra full-attention forward per needle batch,
with attention matrices that grow with the square of the length. At 512 tokens it about doubles the
training time (about 23 instead of 13 minutes).

### 3.5 Rotary time instead of an age penalty (A1)

**Change.** `--w1s-recency-form none --w1s-time-feat rope`. The age penalty is removed. The first 32
of the 128 learned-key dimensions are rotated in 16 pairs, as in RoPE: the normalised slot key by
`ω_f · s_k` (the slot's onset) and the normalised query by `ω_f · t`. The periods `2π/ω_f` are
log-spaced from 64 to 65,536 tokens. The selection term becomes

    β_L · ⟨R(t) q_t , R(s_k) K_k⟩ ,

which depends jointly on content and on the age `t − s_k`. The other 96 dimensions are unrotated.

**Why it might help.** Time becomes data the key can use, as in RoPE attention, rather than a fixed
penalty. The model can learn which ages matter for which content.

**What training learned.** The rotation has no parameters. On seed 4, `β_L` ended lower than in
the other arms (`e^{2.16} ≈ 8.6`, against ≈ 11–12), so the key match is less sharp.

**Results.**
- Seed 1: +0.08 (2048 × 5); paraphrases 1–4 all tie.
- Seed 4: +0.32; paraphrases 1–4 +0.07, +0.09, +0.07 and +0.10.
- **But held-out names are worse:** −0.09 on seed 1 (0.59 absolute) and −0.08 on seed 4 (0.50
  absolute). A1 and A11 each have a worse cell (A1: held-out names on both seeds; A11: paraphrase 4
  on seed 1).

One hypothesis, untested: giving a quarter of the key to time leaves less room for content, and
the rotation's phase at ages never seen in training (training is at 512 tokens) adds noise. For
both reasons it ranks below A1b.

**Cost.** No parameters, and no state beyond the onset `s_k` each slot already stores.

## 4. How they fit together

The five changes act on different stages of the pipeline:

| Stage | A11 | A1b | A12 | A15 | A1 |
|---|---|---|---|---|---|
| Gate input | blocks 3 + 21 | | | | |
| Gate training signal | | | | full-attention target | |
| Slot key / query | + learned on 3 + 21, + lexical on 3 | + ordinal term | | | rotary on 32 dims |
| Age penalty | kept | **removed** | kept | kept | **removed** |
| Alignment code | **block 3** | | | | |
| Alignment sharpness | | | higher start | | |
| Inference cost | +264k params | +2.3k params | none | none | none |

**Why they can combine.** Apart from A1b and A1, which are alternative time terms, each arm changes
a different term or input. Their flags do not conflict in the code. The natural stack is A11 + A1b,
then + A12, then + A15. A draft plan exists to train it in that order, alongside A11 without its
lexical key.

**What is untested.**
- **No combination has been trained.** In particular, nothing has been run with multi-tap streams,
  lexical keys and ordinal time together. The five results are single changes to I1.
- **Interactions are plausible.**
  - A12's starting sharpness was chosen for block-21 codes. Under A11 the alignment code comes from
    block 3, whose cosine distribution differs.
  - Under A11, A1b's `u_t` reads a two-stream query.
  - A15 trains a gate that, under A11, reads two blocks.
- **A11's working part is unknown:** the lexical key, the two-block gate and key, or the block-3
  alignment code.
- A1 and A1b cannot be stacked: the single flag `--w1s-time-feat` takes one value. The evidence
  favours A1b.

## 5. What is not shown yet

- **Seeds.** Two seeds per arm, one of them I1's failing seed. A 10-seed stage decides; the bar is
  learning on at least 8 of 10 seeds, with a binomial interval.
- **Names never seen in training.** The best held-out score is about **0.72–0.73**, against
  0.85–0.98 on training names. The fact is still retained (95–100% of the time); the key picks the
  wrong slot. None of the five fixes this, and A1 makes it worse.
- **The value is small.** A 4-digit code is about 13 bits from a fixed format. Recall of longer,
  open-ended values (a sentence, a multi-token entity) has not been tested.
- **Equal-memory baselines.** The memory holds about as many bytes as 31 tokens of KV cache. It has
  not yet been compared with the obvious rivals at equal bytes: a KV cache of 31 well-chosen tokens,
  or another fixed-state memory such as a delta-rule matrix of the same size.
- **Scale and position.** Only Qwen3-1.7B, a 64-token window and injection at block 21 have been
  tried. A sweep over the injection block is queued, and a 128-token window is planned.
- **Updates and capacity.** When a name is restated with a new code, I1 picks between the old and the
  new code unreliably. The retention rule (keep the 16 strongest) bounds capacity. The grid has cells
  with 8–32 codes at 4,096–8,192 tokens, but they are not part of this interim summary.
- **Effect on ordinary text.** Language-modelling quality with the memory attached is being measured
  for I1 only.
- **Evaluation is teacher-forced.** Free generation of the code has not been scored.

## 6. Sources

**Code (the ground truth for everything above).**
- `experiments/evolve/w1s_slots.py`:
  - `SlotLoop._gate`, `_store`, `_select`, `_alignment`, `_read` and `stream` (the streaming
    reference);
  - `W1sSide`;
  - `analytic_state`, for the state sizes.
- `experiments/evolve/w1_gdn.py`: `GDNLoop.forward`, the injection stage.
- `experiments/evolve/taps.py`: `TapLM`, the block-3 tap.
- `experiments/evolve/teacher.py`: `capture_attention`, `window_targets` and `admit_kl` (A15).
- `experiments/evolve/run.py`: the needle layout, the paraphrase templates and the training names.
- `experiments/pretrained_loop.py`: block windows and the sink.
- `limbic_llm/temporal_loop.py`: `TimeChannel`, the clock.

**Recipes.** The round-11 Stage-A job list: the I1 base flags and the `ARMS` list (`A11`, `A1b`,
`A12`, `A15`, `A1`, `I1sup`, `I1noal`). The combination draft is in
`research/plans/round11_combination_plan.md`. Round-11 plan and runbook:
`research/plans/round11_plan.md`, `research/plans/round11_runbook.md`.

**Results (interim, not yet in the register).**
- Training: `results/evolve/round11/r11_<ARM>_s{1,4}.json` with their `.log` files.
- Evaluations: `results/evolve/round11/eval/e11_r11_<ARM>_s{1,4}_<cell>.json`.
- I1 reference evaluations: `results/evolve/round11/eval/e11ref_r10_I1_s{1,4}_<cell>.json`.
- Checkpoint values quoted here (β's, ordinal table norms, gate parameters) were read from the
  uncommitted `.pt` files next to the JSONs and from `results/evolve/round10/r10_I1_s{1,4}.pt`.

**Register (`research/REGISTER.md`).**
- I1 and its round-10 record: P51, P54.
- Held-out names: P55.
- Lexical keys from block 3 in round 10 (I1K): P57.
- Age penalty and alignment sharpness: P58, P59.
- The slots-and-alignment ablations requested by review: RV4.E1.
- The 13-bit value caveat: RV4.E11.
- Decision rules for round 11: DEC11 (relative differences), DEC12 (10 seeds, arm order, A13–A15),
  DEC13 (tap sweep).
- Backlog rows: W13 (W1s), W14 (round 11), W17 (tap sweep).
- Onset offsets, the use of 4–5 of 24 entries, and the unused clock: NOTEBOOK, round-10 summary
  (3–4 Nov 2026).
