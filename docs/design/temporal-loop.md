# Closing the loop: hippocampal feedback, time, and a full Limbic LLM

This note records how the temporal-loop design was reached, step by step, and what it became. Code: [`limbic_llm/temporal_loop.py`](../../limbic_llm/temporal_loop.py). Experiment: [`experiments/temporal_loop.py`](../../experiments/temporal_loop.py). Results: [RESULTS.md §4](../../RESULTS.md).

## 1. How the idea developed

**Static steering as CA2.** Activation Addition (Turner et al., 2023) adds a fixed vector $c\,(h^l_+ - h^l_-)$ to a middle layer of a frozen model. That is a side channel beside the main pathway with an amplitude knob, which is what CA2 is: a mid-circuit input that conditions CA3 → CA1 traffic. But the vector is fixed and chosen by a user, while CA2's signal is dynamic and internal.

**Making it dynamic.** Tap an inner layer, write a time-lagged window of its activations into a Gated DeltaNet (CA3) memory, retrieve related context, contrast it against a negative, and re-inject the difference. Windows whose positive and negative retrievals differ strongly get a stronger signal. This unifies three regions: the memory stores and predicts (CA3), the contrast measures novelty (CA1), and the re-injection is the side channel (CA2).

**First catch: what is "negative"?** A linear memory cannot supply it by negating the query, since $S(-q) = -Sq$ and the difference is just twice the retrieval. The negative must come from a separate source. The choice here is *episode against prior*: $\Delta_t = F_t - m_t\mu_t$, what this episode's memory returns minus what the model would expect from everything it has seen. $\Delta_t$ is then CA1's mismatch in vector form, and "strong difference, strong signal" means "amplify in proportion to how specific the memory is".

**Second catch: the loop can run away.** Injections change later activations, which are written back, retrieved, and injected again. That is positive feedback.

**How biology prevents it.** The hippocampus report notes that the circuit "operates near a critical threshold between excitation and inhibition", held by basket, chandelier and O-LM interneurons; seizures start there when that fails. Its other example, the hippocampus as the negative-feedback sensor for the stress axis, shows what an unchecked loop does: Sapolsky's self-reinforcing glucocorticoid cascade. The full set of safeguards:

| Mechanism | Effect | Design element |
|---|---|---|
| Feedback inhibition (basket cells) | Inhibition grows with activity | Divide gain by recent injection energy |
| Acetylcholine switch (Hasselmo) | During encoding, recurrent CA3 synapses are suppressed more than incoming ones | Store the stream from *before* injection (open loop) |
| Theta-phase separation (Hasselmo, Bodelón & Wyble, 2002) | Encoding and retrieval at opposite phases of a 4–8 Hz cycle | Loop period τ; nothing returns within one circuit |
| Attenuated re-entry (CA1 → subiculum → EC) | The big loop is slow and weak | Attenuation ρ < 1 per circuit |
| Self-terminating bursts | CA3 bursts end within ~100 ms | Saturating gain |
| Sparse DG input | Bounded activity entering CA3 | k-WTA on written keys |

The acetylcholine switch also refines the gain rule: biology suppresses recall when *input* is novel. Input-versus-memory mismatch (encode) and memory-versus-prior contrast (recall) are different signals.

**Skip paths.** There is no DG → CA1 connection; mossy fibers end in CA3 and the hilus. The skip into CA1 comes from entorhinal layer III, which `HippocampalBlock` has. But the *direct entorhinal layer II → CA3 path* had been neglected: Treves and Rolls (1992) argued the dentate gyrus drives storage while this direct path supplies the retrieval cue. A pattern-separated cue maps to a different code than the stored one, which defeats completion. The loop therefore writes with (optionally sparse) DG keys and reads with a dense query taken straight from the stream.

**CA1 → CA3 negative feedback, as in an op-amp.** CA1 pyramidal cells do not project back to CA3, but CA1 has GABAergic back-projection interneurons whose axons return to CA3 and the hilus (Sik, Ylinen, Penttonen & Buzsáki, 1994). CA3's recurrence is a high-gain amplifier $A$; the inhibitory back-projection is the feedback network $\beta$:

$$G = \frac{A}{1 + A\beta} \approx \frac{1}{\beta} \qquad (A\beta \gg 1)$$

The output gain is set by the stable feedback path, not by the unstable recurrent gain. In the design, CA1 injects only what the stream lacks and divides the gain by the injection energy from the previous circuit.

**Divisive or subtractive?** That first version is *divisive*: it divides the gain by the energy of recent injections, $y = A\Delta/(1+\kappa E)$ with $E \propto y^2$. At high gain this only compresses, $y \propto A^{1/3}$; it is not an op-amp. An op-amp subtracts a fraction of its own output, so the closer analogue is a *subtractive* back-projection that carries an inverted copy of what was injected:

$$y_t = A\big(\Delta_t - \beta\,\bar y_t\big),\qquad \bar y_t = \sum_{s \le t-\tau} (1-\rho_E)\,\rho_E^{\,t-s-\tau}\,y_s \quad\Longrightarrow\quad y = \frac{A\,\Delta}{1 + A\beta} \xrightarrow{A\beta \gg 1} \frac{\Delta}{\beta}$$

Two properties follow. First, a lagged, smoothed copy can only cancel what persists from circuit to circuit, so subtractive feedback is a *high-pass filter* on the injection: sustained echo is attenuated by $1/(1+A\beta)$ while new content passes. That is "new tokens dominate" stated as a filter. Second, the copy arrives a circuit late, and a delayed negative-feedback loop rings when its gain is too high. The error obeys $e_t = -K\sum_{d\ge\tau}(1-\rho_E)\rho_E^{\,d-\tau}e_{t-d}$ with $K = A\beta$, whose characteristic polynomial is

$$z^{\tau} - \rho_E\,z^{\tau-1} + K(1-\rho_E) = 0.$$

For $\tau = 16$ and $\rho_E = 0.9$ a root leaves the unit circle at $K \approx 1.68$, with a ringing period of about 44 tokens. This is the reason op-amps need frequency compensation. The op-amp limit $\Delta/\beta$ is therefore out of reach with a one-circuit delay, and persistent echo can be cut to about $1/(1+1.68) \approx 0.37$ before the loop oscillates. Biology has the same constraint: slow GABA$_\text{B}$ inhibition and theta timing are candidates for the compensation.

**Is the inverted copy the negative prompt of activation addition?** Both subtract, but different things. Activation Addition's negative is an independent reference, $c\,(h_+ - h_-)$: subtracting "Hate" from "Love" removes what the prompts share and leaves the concept axis. In the loop that role is played by the prior $\mu_t$ in $\Delta_t = F_t - m_t\mu_t$. The CA1 inverted copy subtracts the loop's *own* recent output, a self-negative: "steer away from what I just injected". The subtractive mode is exactly that, so the side channel injects only what it has not already injected.

**Clamped, damped sinusoids.** Feedback should return as clamped sinusoids that decay on every circuit around the loop, so new tokens dominate and fed-back content is attenuated:

$$w(\Delta) = A\,\rho^{\Delta/\tau - 1}\,\big[\cos(2\pi\Delta/\tau)\big]_+,\qquad \Delta = t - s \ge \tau,\quad A \le 1,\ \rho < 1$$

Content returns once per circuit in a phase window, as theta sequences re-express recent experience each cycle. Each pass is attenuated by ρ, the clamp keeps it excitatory and bounded by $A$, and new input enters at weight 1.

**A sense of time passing.** The loop's decay and phase already say how long ago something was stored, and that can be made explicit. Transformers know token order inside their context through positional encodings, but nothing tells them how much time has passed since an event once it has left the attention window, or through a memory. The hippocampus has time cells that fire in sequence across a delay (MacDonald et al., 2011), and Shankar and Howard (2012) showed that a bank of leaky integrators with log-spaced time constants gives a scale-invariant code for elapsed time. The time channel implements that bank, plus damped oscillators for phase, and adds it to the stream.

## 2. The design

**Hippocampal loop** (`HippocampalLoop`), at a tap layer with stream $h_t$:

$$a(t,s) = e^{\,b(\hat q_t\cdot\hat k_s - 1)},\qquad F_t = \sum_s w(t-s)\,a(t,s)\,v_s,\qquad \Delta_t = F_t - m_t\,\mu_t$$

$$e_t = \Delta_t - \operatorname{proj}_{h_t}\Delta_t,\qquad g_t = \frac{G\,\tanh(\lVert\Delta_t\rVert/\lambda)}{1 + \kappa E_t},\qquad h_t \leftarrow h_t + g_t\,\lVert h_t\rVert\,\hat e_t$$

$\hat q$ comes from the stream directly (EC → CA3), $\hat k$ optionally through DG k-WTA, $\mu_t$ is the mean of returnable values and $m_t$ their total weight, and $E_t$ is the injection energy from at least one circuit earlier. With CA1 control, $\lVert\text{injection}\rVert \le G\lVert h_t\rVert$ with $G < 1$, so new input always dominates. Flags select the kernel (`osc`, `exp`, `flat`), ρ, $A$, τ, open or closed loop, contrast, CA1 control and DG keys.

`--ca1-control` bundles four parts, which can be switched separately:

| Part | Flag | Effect |
|---|---|---|
| Back-projection | `--feedback divisive\|subtractive\|none` | Divide the gain by $1+\kappa E_t$, or subtract the inverted copy $\beta\bar y_t$ |
| Hard bound | `--clamp` | Rescale the injection to $g\lVert h\rVert$ with $g \le G$; the bound holds by construction |
| Error-only injection | `--project` | Remove the component along $h_t$ |
| Saturation | `--saturate` | $\tanh(\lVert\Delta\rVert/\lambda)$ amplitude |

`--forward-gain` sets $A$ and `--fb-beta` sets β. `--read direct|dg` chooses whether the retrieval cue comes straight from the stream or through the same dentate-gyrus code as the keys, and the long rollout measures recall with relative Gaussian noise on the cue.

**Time channel** (`TimeChannel`): a salience $u_t\in(0,1)$ drives

$$L_j(t) = (1-\rho_j)\sum_{s\le t}\rho_j^{\,t-s}u_s,\qquad C_j + iS_j = (1-\rho_j)\sum_{s\le t}\rho_j^{\,t-s}e^{i\omega_j(t-s)}u_s,\qquad \rho_j = e^{-1/\tau_j}$$

for $\tau_j \in \{4, 8, \dots, 256\}$, through a zero-initialised projection into the stream. After a single event, each $L_j$ decodes elapsed time exactly (tested).

**Implementation.** Both are computed in parallel form. The closed loop is exact in chunks of τ, because nothing returns within one circuit; a test checks it against a token-by-token reference. Because the kernel is a sum of damped complex exponentials, the same feedback has an O(1)-state recurrent form, as in RetNet and LRU, for streaming.

## 3. Plug-in or new architecture?

| Piece | Closest existing mechanism | Plugs into |
|---|---|---|
| Associative memory of the stream | Gated DeltaNet, linear attention, Titans | Hybrid linear-attention layers (e.g. 3:1 DeltaNet:attention) |
| Side-channel injection | Activation Addition, CAA, CAST | Residual-stream hooks on a frozen model |
| Memory-retrieved steering | DSEM (Do et al., 2025) | Same hooks |
| Damped oscillatory kernel | RetNet, LRU, S4D decays | Any linear recurrence |
| Clock of elapsed time | Time-cell and Laplace models; not standard in LLMs | A new side channel |
| Looped depth | Ouro, Universal Transformers | Weight-shared blocks |

Each piece has a precedent and can be attached to an existing model on its own. **The combination is a new architecture**: a closed loop that writes the model's own lagged activations into a fast associative memory, re-injects contrastive, time-tagged feedback that is damped per circuit, and regulates it with negative feedback, beside a clock of elapsed time. The open loop is a plug-in; the closed, regulated, timed loop is a new organisation.

## 4. A full Limbic LLM

**Neocortex and white matter.** In the brain, consolidated long-term memory lives in cortical synapses (gray matter); white matter is the long-range wiring between areas. In a transformer, the weights of the layer stack are the consolidated cortical memory, and the depth is the cortical hierarchy. The residual stream is the white-matter bus that every layer reads and writes, and it is where the hippocampal side channels connect.

**Blueprint**, around a pretrained decoder (for example Llama or Qwen), frozen at first:

1. **Entorhinal tap** at about one third of the depth, where features are abstract but not yet task-specific.
2. **Hippocampal loop** with sparse DG write keys, a Gated DeltaNet store with exact updates and multi-band damped decays, a direct EC → CA3 read query, CA2 contrast (episode minus prior, plus an identity library), CA1 control (calibrated novelty, error-only injection, energy-normalised gain $G < 1$), and an acetylcholine gate that suppresses injection while input is novel.
3. **Injection** at about half the depth, so later layers integrate the memory.
4. **Time channel** injected at the same point.
5. **Optional big loop**: re-run the upper block, Ouro-style, with CA1 novelty deciding when to stop.
6. **Consolidation**: replay high-novelty episodes in an offline phase into LoRA adapters on the frozen stack, moving episodic memory into the "cortex".

**Training plan.** (1) Train only the side-loop parameters (a few million) on long documents with the context window deliberately truncated, so memory is the only route to distant information. (2) Lightly fine-tune jointly. (3) Add consolidation for continual learning.

## 5. The experiment

**What–when recall.** An event token (8 types) is followed by 20–140 filler tokens and a query. The model must answer *what* happened (chance 0.125) and *when*, in 4 log-spaced elapsed-time bins. The bins cover 15, 25, 40 and 40 of the 120 possible gaps, so a model that only learns the prior scores 0.333 on *when*. Attention sees 16 tokens and there is no absolute position embedding, so content can only arrive through the memory loop and elapsed time only through the side channels.

**Loop stability.** Closed loops (feedback written back into memory) with the undamped `flat` kernel, a damped `exp` kernel, the clamped damped `osc` kernel, and `osc` with CA1 control, trained on 160-token sequences and then run on 520-token sequences. The ratio $\lVert\text{injection}\rVert / \lVert h\rVert$ measures whether feedback grows and whether new input stays dominant.

```bash
python experiments/temporal_loop.py --suite what_when
python experiments/temporal_loop.py --suite stability
python experiments/temporal_loop.py --name mine --memory --time --kernel osc --rho 0.6 --loop closed --no-ca1-control
```

**The op-amp test** (`experiments/loop_systems.py`). An untrained closed loop with the clamp, projection and saturation removed, so feedback is the only regulator, swept over forward gain $A$ and feedback fraction β.

**Feedback and ablation suites.** Trained closed loops with no feedback, divisive feedback and subtractive feedback (clamp removed), and the full design without the contrast and with DG write keys read either directly or through the dentate gyrus. Every single-event run also gets a zero-shot recency probe (an older, different event precedes the target) and a cue-noise sweep.

```bash
python experiments/loop_systems.py
python experiments/temporal_loop.py --suite feedback --steps 3000
python experiments/temporal_loop.py --suite ablations --steps 3000
```

Results are in [RESULTS.md §4](../../RESULTS.md).

## 6. Pretrained models

The synthetic experiments use 0.1-million-parameter models. The next step is a pretrained decoder, frozen, with the loop and clock trained as side channels (`experiments/pretrained_loop.py`). Two questions need different models:

| Question | Model | Why |
|---|---|---|
| Does the loop work as a memory when nothing else can? | Qwen3 (pure attention), attention windowed | Windowing leaves the loop as the only route to distant tokens |
| Is Gated DeltaNet the CA3, and does the loop add anything on top? | Qwen3.5 (3:1 Gated DeltaNet to attention), only attention layers windowed | Its GDN layers already carry the past, so this tests native recall and whether damped, regulated, time-tagged feedback adds recency or elapsed time |

A Qwen3 and Qwen3.5 pair of similar size differs mainly in Gated DeltaNet versus attention, which makes it a cleaner comparison than GPT-2 versus Qwen3.5. GPT-2 remains the offline smoke test. A 2B-class model in bf16 fits on one 32 GB GPU with a frozen base, so both arms can run side by side.

## References

Hasselmo, Bodelón & Wyble (2002); Hasselmo (2006); Sik, Ylinen, Penttonen & Buzsáki (1994); Treves & Rolls (1992); Shankar & Howard (2012); MacDonald, Lepage, Eden & Eichenbaum (2011); Buzsáki (2002); Sapolsky, Krey & McEwen (1986); Turner et al. (2023); Rimsky et al. (2024); Lee et al. (2024); Do et al. (2025); Sun et al. (2023); Orvieto et al. (2023); Zhu et al. (2025); Yang, Kautz & Hatamizadeh (2025); Qwen Team (2025, 2026). Full entries in [REFERENCES.md](../../REFERENCES.md).
