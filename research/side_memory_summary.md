# Side-memory track: where it stands, in plain language

*Written 4 Nov 2026, after round 10. The evidence is in [`REGISTER.md`](REGISTER.md) (P-rows cited below); the dated detail is in [`NOTEBOOK.md`](NOTEBOOK.md). Update this summary when a round changes the picture.*

## The setup

- A frozen language model (Qwen3-1.7B-Base) is made near-sighted: every layer sees only the last 64 tokens.
- It is given a few facts ("The code for Emma is 4821") scattered through hundreds or thousands of words of other text, then asked for one of the codes.
- Blinded, the model always fails (0%). With its full view it always succeeds.
- The question: can a small add-on memory, trained while the model stays frozen, recover what the model can no longer see?

## What works best: a memory that keeps each fact in its own slot (W1s, arm I1)

It does four simple things:
1. **Notice something worth keeping.** A small learned detector fires as the text is read.
2. **Save that stretch of text in its own slot.** It keeps the 24 tokens around that point, in order, in one of 16 slots.
3. **When asked, find the matching slot.** The question's wording is compared with each slot's label (a learned key), and the best match is chosen.
4. **Replay it in order.** Inside the chosen slot it finds where the question lines up and reads out what came next, digit by digit. A small learned translator passes those digits back into the model so it can say them.

**How well it does** (P51, P54):
- On names seen in training it gets the whole 4-digit code right about 93–96% of the time.
- That holds with 1 to 5 facts, at lengths of 512 to 2048 tokens, with real text or random tokens around the facts.
- It works in 3 of 5 training runs, a fourth falls just short (94% against a 95% bar), and one run is weaker.
- The success bar (DEC2) asks for 4 of 5, so it is one run short.

**Why it works when nine earlier rounds didn't.** Every earlier memory stored all facts on top of each other in one shared matrix. That caused two kinds of overwriting:
- the digits of one code overwrote each other (the 8 smudged the 4);
- facts that look alike ("The code for Emma…" and "The code for Jack…") collided, and the newest one usually won.

Training could sometimes find a way around this, but only by luck, in 1–3 runs out of 5 (P36, P40, P46–P48). Separate slots remove the overwriting by design, and the luck stopped mattering.

**Simpler turned out to be better** (P50, P52, P57). A combination of ideas was expected to win: a smarter trigger, labels from two layers, pre-recording the words before the fact, and a normalising trigger. Combined, they did worse. The normalising trigger adjusted itself to the kind of surrounding text and then failed to save facts when the text changed. The other extras added nothing once facts had their own slots.

## What doesn't work yet

- **New names** (P55). With names it never saw in training, accuracy drops to 58–74%. The facts are still stored, but the learned labels memorised the 20 training names. A variant using the raw word tokens as labels did generalise (88%), but it is unstable: it collapsed in 3 of 5 runs.
- **Long documents with many facts** (P54, P58). The weaker run loses the oldest facts at 2048 tokens. Its replay alignment is blunter than in the runs that succeed.
- **Corrected facts** ("Emma's code is now 5678") (P56). We can't score this yet: even the model with its full view returns the old code more often than the new one, so the test needs clearer wording.
- **Putting the memory back into the model's attention**, rather than adding it to the model's internal signal (P53, P57). It works with a perfect trigger (84–94%) but fails with a learned one.

## Why a 64-token window, and was it swept?

It was not swept. The 64-token window came from the first frozen-model experiments. It creates a clean "out of reach" regime cheaply:
- at 512 tokens each fact sits 4–8 windows before the question, and at 2048 tokens about 30;
- the blinded model scores 0%;
- each run trains in about 13 minutes, which made 100+ runs feasible.

It is a scaled-down stand-in for "the context the model can see is much shorter than the document". Production local-attention windows are 1,000–8,000 tokens (Mistral 7B v0.1: 4,096; Gemma 3's local layers: 1,024). Hybrid models such as Qwen3-Next replace most attention layers with fixed-size recurrent layers. How the result changes at real window sizes is untested: backlog W15, round 11 Stage C.

## How it could help a production model

The real-world version of the setup is any model whose memory is cheaper than its full context: a sliding-window or local-attention model, a hybrid with compressing recurrent layers, or a deployment that caps the KV cache. Such models lose exact details once text scrolls out of range. W1s adds an exact episodic memory of fixed size:
- **Constant memory.** About 0.9M numbers (3.6 MB) whatever the document length, and round-11 diagnostics suggest about 40× compression is possible. This model's KV cache is about 235 MB at 2,048 tokens and about 3.7 GB at 32k.
- **Exact recall of selected moments.** It keeps specific passages verbatim and in order, and replays the one that matches the question. Compressed memories (delta rule, GDN states) are poor at that: the difference between "roughly what was said" and "the exact code".
- **No change to the base model.** It attaches through hooks, has under 1M trained parameters, and when switched off the model's output is bit-for-bit unchanged.
- **Where it would matter:** long chats and agent sessions where an early detail (an ID, a value, a constraint) must be recalled exactly; streaming logs and transcripts; on-device models with tight memory.

## What is not shown yet

- **One synthetic task:** 4-digit codes bound to names. Open-ended content (sentences, multi-token values), questions needing several facts, and real benchmarks are untested.
- **Generalisation:** it fails on unseen names, and it was trained for this one task. A production version would need its write and lookup policies trained on diverse data.
- **Scale:** one 1.7B model, a 64-token window, documents up to 2,048 tokens (diagnostics to 4,096).
- **No baselines at equal memory:** retrieval-augmented generation, KV-cache retrieval (InfLLM, Memorizing Transformers), or simply a longer context. That is backlog W16.
- **Reliability:** 3 of 5 training runs meet the bar; DEC2 needs 4.

**In one sentence:** a memory that stores each important moment separately and replays it in order gives a frozen, near-sighted model near-perfect recall of facts thousands of words back, at constant memory cost. Making it work every time, for unseen names and content, at real window sizes and against real baselines, is what remains.

## Where things are

- **Code:** `experiments/evolve/w1s_slots.py` (the memory), `experiments/evolve/taps.py` (extra taps and reinstatement), `experiments/evolve/eval_by_query.py` (evaluation).
- **Results:** `results/evolve/round10/`.
- **Next:** round 11 ([`plans/round11_plan.md`](plans/round11_plan.md), runbook [`plans/round11_runbook.md`](plans/round11_runbook.md)): time as data, compression, name generalisation, alignment, and a window-size sweep.
