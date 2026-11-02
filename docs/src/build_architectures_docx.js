// Builds docs/hippocampal_ml_architectures.docx
//   python docs/src/bib_to_json.py > docs/src/references.json
//   node docs/src/build_architectures_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Header, Footer, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageNumber, PageBreak, Table, TableRow, TableCell, LevelFormat,
} = require("docx");

const OUT = path.join(__dirname, "..", "hippocampal_ml_architectures.docx");
const REFS = JSON.parse(fs.readFileSync(path.join(__dirname, "references.json"), "utf8"));

// ─── Palette (unchanged from v1) ───
const NAVY = "0F2440", DARK_BLUE = "1B3A5C", MED_BLUE = "2E6B9E", ACCENT_TEAL = "1A8A7D";
const ACCENT_AMBER = "C87A20", BODY = "2D2D2D", LIGHT_BG = "F4F7FA", CODE_BG = "EDF2F7";

// ─── Text helpers ───
function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true,
    children: [new TextRun({ text, font: "Georgia", size: 32, bold: true, color: NAVY })],
    spacing: { before: 480, after: 200 }, border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: MED_BLUE } } });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true,
    children: [new TextRun({ text, font: "Georgia", size: 26, bold: true, color: DARK_BLUE })], spacing: { before: 360, after: 160 } });
}
function h3(text) {
  return new Paragraph({ keepNext: true, children: [new TextRun({ text, font: "Georgia", size: 23, bold: true, color: ACCENT_TEAL })], spacing: { before: 260, after: 120 } });
}
function p(text) {
  return new Paragraph({ children: [new TextRun({ text, font: "Calibri", size: 22, color: BODY })],
    spacing: { before: 80, after: 120, line: 312 }, alignment: AlignmentType.JUSTIFIED });
}
function runs(arr) {
  return new Paragraph({ children: arr.map(r => new TextRun({ font: "Calibri", size: 22, color: BODY, ...r })),
    spacing: { before: 80, after: 120, line: 312 }, alignment: AlignmentType.JUSTIFIED });
}
function lead(label, text) { return runs([{ text: label + " ", bold: true, color: DARK_BLUE }, { text }]); }
function codeBlock(lines) {
  return lines.map(line => new Paragraph({ children: [new TextRun({ text: line, font: "Consolas", size: 18, color: "1A1A1A" })],
    spacing: { before: 0, after: 0, line: 260 }, indent: { left: 480 }, shading: { type: ShadingType.CLEAR, fill: CODE_BG } }));
}
function spacer(h = 100) { return new Paragraph({ spacing: { before: h, after: 0 }, children: [] }); }
function note(text, color = ACCENT_TEAL) {
  return new Paragraph({ children: [new TextRun({ text, font: "Calibri", size: 21, italics: true, color })],
    spacing: { before: 80, after: 120, line: 300 }, indent: { left: 360 }, border: { left: { style: BorderStyle.SINGLE, size: 4, color } } });
}
function bullet(text, boldPrefix = "") {
  const children = [];
  if (boldPrefix) children.push(new TextRun({ text: boldPrefix, font: "Calibri", size: 22, bold: true, color: DARK_BLUE }));
  children.push(new TextRun({ text, font: "Calibri", size: 22, color: BODY }));
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { before: 40, after: 60, line: 300 }, children });
}
function pageBreak() { return new Paragraph({ children: [new PageBreak()] }); }

// Display equation with simple markup: x_{sub}, x^{sup}
function eq(src) {
  const children = [];
  const re = /([_^])\{([^}]*)\}/g;
  let last = 0, m;
  const push = (t, o = {}) => t && children.push(new TextRun({ text: t, font: "Cambria Math", size: 23, color: NAVY, ...o }));
  while ((m = re.exec(src))) {
    push(src.slice(last, m.index));
    push(m[2], m[1] === "_" ? { subScript: true } : { superScript: true });
    last = re.lastIndex;
  }
  push(src.slice(last));
  return new Paragraph({ children, alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 } });
}

// ─── Tables ───
const bdr = { style: BorderStyle.SINGLE, size: 1, color: "C0C8D0" };
const borders = { top: bdr, bottom: bdr, left: bdr, right: bdr };
function table(header, rows, widths, size = 17) {
  const total = widths.reduce((a, b) => a + b, 0);
  const keep = rows.length <= 12;
  const cell = (text, i, opts) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: { fill: opts.fill, type: ShadingType.CLEAR }, margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({ keepNext: keep && !opts.last, children: [new TextRun({ text, font: "Calibri", size: opts.size, bold: opts.bold, color: opts.color })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, { fill: NAVY, size: 19, bold: true, color: "FFFFFF" })) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, { fill: ri % 2 === 0 ? LIGHT_BG : "FFFFFF", size, bold: i === 0, color: BODY, last: ri === rows.length - 1 })) })),
    ],
  });
}

const numbering = { config: [
  { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
] };

// ═══════════════════════════════════════ CONTENT
const C = [];

// ─── Title page ───
C.push(spacer(1800));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "HIPPOCAMPAL ARCHITECTURES", font: "Georgia", size: 52, bold: true, color: NAVY })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "FOR MACHINE LEARNING", font: "Georgia", size: 52, bold: true, color: NAVY })] }));
C.push(spacer(120));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
  border: { top: { style: BorderStyle.SINGLE, size: 2, color: MED_BLUE } },
  children: [new TextRun({ text: "Structural Parallels Between Biological Memory Circuits and Neural Network Architectures,", font: "Calibri", size: 23, italics: true, color: MED_BLUE })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: MED_BLUE } },
  children: [new TextRun({ text: "with Implementation Strategies and Results from the limbic-llm Package", font: "Calibri", size: 23, italics: true, color: MED_BLUE })] }));
C.push(spacer(500));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Daniel Byrne", font: "Calibri", size: 24, color: BODY })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "Version 3 · November 2026", font: "Calibri", size: 22, color: "666666" })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "Code: github.com/realdanielbyrne/limbic-llm", font: "Consolas", size: 19, color: MED_BLUE })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Prepared with assistance from Claude (Anthropic)", font: "Calibri", size: 18, color: "999999" })] }));
C.push(pageBreak());

// ─── Table of contents ───
C.push(new Paragraph({ spacing: { after: 300 }, children: [new TextRun({ text: "TABLE OF CONTENTS", font: "Georgia", size: 28, bold: true, color: NAVY })] }));
const toc = [
  [true, "1.", "Introduction and Motivation"],
  [true, "2.", "Hippocampal Circuit Architecture: A Computational Summary"],
  [true, "3.", "Mapping Hippocampal Subregions to ML Components"],
  [true, "4.", "The Titans Architecture: A Convergent Design"],
  [false, "", "4.7  Gated DeltaNet: The Linear Special Case of Titans"],
  [true, "5.", "Extended Mappings from Recent Work"],
  [false, "", "5.1  CA2 as Activation Steering"],
  [false, "", "5.2  CA1 as a Calibrated Decision Readout (JEV, LAYA)"],
  [false, "", "5.3  CA1 as Predictive Comparison (LLM-JEPA, LeJEPA)"],
  [false, "", "5.4  Looped Transformers as Big-Loop Recurrence"],
  [false, "", "5.5  Revised Component Mapping"],
  [true, "6.", "Closing the Loop: Feedback, Time and a Full Architecture"],
  [false, "", "6.1  From Static to Dynamic Activation Addition"],
  [false, "", "6.2  How the Biological Loop Stays Stable"],
  [false, "", "6.3  Skip Paths and CA1 Negative Feedback"],
  [false, "", "6.4  Clamped, Damped Re-entry and a Sense of Time"],
  [false, "", "6.5  Plug-in or New Architecture?"],
  [false, "", "6.6  A Full Limbic LLM"],
  [false, "", "6.7  Experiment: What, When, and Loop Stability"],
  [false, "", "6.8  The Op-amp Test: Divisive or Subtractive Feedback"],
  [false, "", "6.9  Pretrained Models"],
  [true, "7.", "Implementation Strategy I: Direct Hippocampal Architecture"],
  [true, "8.", "Implementation Strategy II: Titans-Hippocampal Hybrid"],
  [true, "9.", "limbic-llm: Implementation and Results"],
  [true, "10.", "Prior Art and Positioning"],
  [true, "11.", "Connections to Continual Learning and Catastrophic Forgetting"],
  [true, "12.", "Open Questions and Future Directions"],
  [true, "13.", "References"],
];
for (const [major, num, title] of toc) {
  C.push(new Paragraph({ spacing: { before: major ? 100 : 30, after: 30 }, indent: { left: major ? 0 : 360 }, children: [
    new TextRun({ text: num ? `${num}  ` : "", font: "Calibri", size: 22, bold: true, color: NAVY }),
    new TextRun({ text: title, font: "Calibri", size: major ? 22 : 21, bold: major, color: major ? NAVY : BODY })] }));
}
C.push(spacer(200));
C.push(note("New in version 3: Section 6, closing the loop \u2014 dynamic activation addition, how the biological loop stays stable, the direct entorhinal path to CA3, CA1 negative feedback as an op-amp, clamped damped re-entry, a clock of elapsed time, the plug-in versus new-architecture question, a full Limbic LLM blueprint, the what-when, loop-stability and recency experiments, the op-amp test of CA1 feedback (6.8), and first results on pretrained Qwen3 and Qwen3.5 (6.9). Version 2 added Gated DeltaNet (4.7), recent inspirations (Section 5), limbic-llm results (Section 9), prior art (Section 10) and the bibliography (Section 13).", ACCENT_AMBER));

// ═══ 1. INTRODUCTION
C.push(h1("1. Introduction and Motivation"));
C.push(p("The hippocampal formation is the mammalian brain’s solution to a problem that modern machine learning is still struggling to solve: how to rapidly encode novel experiences into durable memory traces without destroying previously stored knowledge, while maintaining the ability to retrieve either on demand through partial cues. This document explores the structural and computational parallels between hippocampal circuitry and neural network architectures, with the goal of identifying concrete, implementable design principles that can improve memory, generalization, and continual learning in artificial systems."));
C.push(p("We examine two complementary implementation strategies. The first is a direct hippocampal architecture that translates the trisynaptic circuit’s subregional specializations into discrete neural network modules: sparse gating networks for the dentate gyrus, autoassociative Hopfield/attention layers for CA3, comparators for CA1, and offline replay mechanisms for systems consolidation. The second strategy enhances the Titans architecture (Behrouz, Zhong, and Mirrokni, 2024; arXiv:2501.00663) with hippocampally inspired modifications, demonstrating that Titans already implements several hippocampal principles and identifying specific gaps where biological design insights could improve its performance."));
C.push(p("Both strategies are now implemented and tested in the open-source PyTorch package limbic-llm. Version 2 of this document adds three things. First, mappings drawn from recent work: Gated DeltaNet as the linear special case of the Titans memory, activation steering as a model of CA2, calibrated decision models (JEV and LAYA) and joint-embedding predictive architectures as models of CA1, and looped transformers as hippocampal big-loop recurrence. Second, measured results from limbic-llm, including three design changes that the experiments forced. Third, a survey of prior art and a full bibliography."));
C.push(p("The convergence between these approaches is not accidental. The hippocampus evolved to solve the same fundamental tension that confronts modern sequence models: accurate modeling of current context versus utilization of long past history. Transformers solve the first problem well but scale quadratically. Recurrent models and state-space models solve the second but sacrifice local precision. The hippocampal formation solves both simultaneously through architectural specialization: distinct subregions with different computational properties, connected in a specific topology that routes information through complementary processing stages."));

// ═══ 2. CIRCUIT SUMMARY
C.push(h1("2. Hippocampal Circuit Architecture: A Computational Summary"));
C.push(h2("2.1  The Trisynaptic Loop as a Processing Pipeline"));
C.push(p("The hippocampal formation processes information through a largely unidirectional circuit called the trisynaptic loop, first formalized by Per Andersen and colleagues in the 1970s. Information flows through three sequential excitatory synaptic relays:"));
C.push(lead("Synapse 1 — Perforant Path (EC → DG):", "Entorhinal cortex layer II neurons project to dentate gyrus granule cells. The EC aggregates highly processed multimodal cortical input. The medial EC carries spatial and grid-cell information; the lateral EC carries object and item information."));
C.push(lead("Synapse 2 — Mossy Fibers (DG → CA3):", "Dentate granule cell axons project to CA3 pyramidal neurons through giant synaptic boutons containing hundreds of release sites. Despite extraordinary individual synapse strength, this projection is extremely sparse: each CA3 cell receives input from only about 50 granule cells. This sparse, high-gain connectivity is central to pattern separation."));
C.push(lead("Synapse 3 — Schaffer Collaterals (CA3 → CA1):", "CA3 pyramidal cell axons project to CA1 apical dendrites. This synapse is the most studied in neuroscience for long-term potentiation (LTP). CA1 also receives a direct bypass from EC layer III (the temporoammonic path), enabling it to compare retrieved memory patterns against current sensory reality."));
C.push(p("The circuit closes with CA1 projecting to the subiculum, which projects back to the entorhinal cortex deep layers, completing a re-entrant loop: EC → DG → CA3 → CA1 → Subiculum → EC. Section 5.4 returns to this re-entry, which supports inference across stored episodes."));
C.push(h2("2.2  Key Computational Properties by Subregion"));
C.push(table(["Subregion", "Principal Cells", "Core Computation", "Key Property"], [
  ["EC", "Layer II/III projection neurons", "Multimodal sensory convergence and contextualized embedding", "Two parallel streams: spatial (medial) and object (lateral)"],
  ["DG", "Granule cells (~1M in humans)", "Pattern separation via sparse activation (2–4% firing rate)", "Adult neurogenesis; massive expansion then sparsification"],
  ["CA3", "Large pyramidal neurons", "Autoassociative pattern completion and sequence prediction", "Single-trial learning; each cell contacts ~4% of all CA3 cells"],
  ["CA2", "Pyramidal (no mossy fiber input)", "Conditional modulation; social memory processing", "Receives unique vasopressin / supramammillary input"],
  ["CA1", "Smaller, dense pyramidal neurons", "Comparison of retrieved memory vs. current sensory input", "Dual dendritic input: Schaffer (CA3) + temporoammonic (EC)"],
  ["Subiculum", "Burst-firing pyramidal neurons", "Output amplification and distribution to targets", "Projects to EC, mammillary bodies, PFC, NAc, amygdala"],
], [1600, 2200, 2800, 2760], 18));

// ═══ 3. MAPPING
C.push(h1("3. Mapping Hippocampal Subregions to ML Components"));
C.push(p("This section establishes the primary structural analogy: each hippocampal subregion maps to a distinct, well-characterized neural network component. The value of this mapping lies not in superficial resemblance but in the fact that the hippocampal subregions solve specific computational problems (pattern separation, associative storage, novelty detection, output amplification) that have direct analogs in ML system design."));
C.push(h2("3.1  Entorhinal Cortex → Embedding / Encoder Stack"));
C.push(p("The entorhinal cortex aggregates information that has already been processed through multiple cortical stages, producing a contextualized, multimodal representation that serves as the input to the hippocampal circuit. This maps to the output of a transformer encoder stack, not raw token embeddings. The dual-stream architecture of the EC (medial spatial stream and lateral object stream) suggests that the input embedding should preserve distinct information channels rather than mixing all modalities into a single vector prematurely. The Tolman-Eichenbaum Machine (Whittington et al., 2020) and the ICLR 2026 Hippoformer model (Li et al., 2026) both implement this split between structural and sensory codes, and Whittington, Warren and Behrens (2022) showed that transformer position encodings can play the role of grid cells."));
C.push(note("Implementation implication: use separate embedding pathways for distinct input modalities, combined before the hippocampal module. limbic-llm currently uses a single TransformerLayer for EC; the medial/lateral split is not yet implemented."));
C.push(h2("3.2  Dentate Gyrus → Sparse MoE Gating Network"));
C.push(p("The dentate gyrus takes a dense input vector from EC and produces an extremely sparse activation pattern: only 2–4% of its ~1 million granule cells fire for any given input. It achieves this through a massive expansion in dimensionality followed by lateral inhibition. This is functionally identical to top-k sparse gating in a mixture of experts: expand into a large expert space, then select very few active experts. The purpose of this sparse coding is pattern separation, transforming similar inputs into nearly orthogonal codes so they can be stored in CA3 without interference."));
C.push(p("Adult neurogenesis adds another dimension. The dentate gyrus continuously generates new granule cells (~700 per day in adult humans), and these neurons are hyper-plastic for 4–6 weeks. This corresponds to progressively expanding the routing capacity of the gate, with new routes initially more plastic. HiCL (Kapoor et al., 2025) has since built a DG-gated mixture of experts for continual learning, routing whole tasks by cosine similarity between sparse DG codes and expert prototypes. Routing individual tokens by DG code inside a language model remains untested."));
C.push(note("Implementation implication: top-k sparse gate with a high expansion ratio (8–16×) and very low k (2–4%); higher learning rate on newly added dimensions. Implemented as limbic_llm.DentateGyrus, including neurogenesis (grow, tick)."));
C.push(h2("3.3  CA3 → Self-Attention / Hopfield Associative Memory"));
C.push(p("CA3 is the most important subregion for understanding why the hippocampal architecture differs from a standard feedforward network. Its pyramidal neurons form an extensive recurrent collateral system, each cell contacting about 4% of all other CA3 cells, creating an autoassociative network that performs pattern completion: a partial cue settles into the nearest stored attractor. Ramsauer et al. (2021), building on Krotov and Hopfield (2016), showed that modern continuous Hopfield networks are mathematically equivalent to attention, so the QKᵀ computation implements the same content-addressable retrieval that CA3 recurrent collaterals perform. CA3 also supports single-trial learning, driven by the sparse but powerful mossy fiber input."));
C.push(p("The limbic-llm experiments added a crucial refinement. A purely autoassociative store returns the cue itself, so a key can never recall its value; the first benchmark run had every memory model at chance for exactly this reason. The fix follows the asymmetric-plasticity account of CA3 sequence memory (Levy, 1996; Jensen and Lisman, 1996): the store binds each pattern to its successor, hₜ ↦ hₜ₊₁, so a cue retrieves what followed it. This heteroassociative form is what makes key → value recall possible."));
C.push(note("Implementation implication: self-attention or modern Hopfield retrieval with an explicit, one-shot, sequence-binding store. Implemented as limbic_llm.CA3Autoassociative with a novelty-gated, salience-decaying EpisodicStore."));
C.push(h2("3.4  CA2 → Conditional Modulation (FiLM / Side Channel)"));
C.push(p("CA2 is a narrow transitional zone between CA3 and CA1 that receives inputs no other hippocampal subfield gets: vasopressinergic projections and input from the supramammillary nucleus of the hypothalamus. It is required for social memory, the ability to recognize and remember individuals (Hitti and Siegelbaum, 2014). In neural network terms CA2 is a conditional modulation pathway: a side channel that injects contextual signals (identity, temporal state, arousal) that modulate the CA3→CA1 flow without passing through the main pipeline. This maps to Feature-wise Linear Modulation (Perez et al., 2018). Section 5.1 develops a sharper analogue, activation steering."));
C.push(note("Implementation implication: a parallel branch that scales and shifts the main path. Implemented as limbic_llm.CA2Modulator, y = x⊙(1+γ(c)) + β(c), initialized to the identity. It is implemented but not yet exercised by any experiment."));
C.push(h2("3.5  CA1 → Comparator and Decision Stage"));
C.push(p("CA1 is the primary output region of the hippocampus proper, but its most important computational property is that it serves as a comparator. It receives two independent inputs on different dendritic compartments: CA3 Schaffer collateral input (the retrieved memory) on proximal apical dendrites in stratum radiatum, and direct EC temporoammonic input (current sensory reality) on distal dendrites in stratum lacunosum-moleculare. CA1 computes a match/mismatch signal, and mismatch pushes the hippocampus toward encoding (Bein, Duncan and Davachi, 2020) and, through the hippocampal–VTA loop, toward learning and action (Lisman and Grace, 2005)."));
C.push(p("Version 1 described CA1 as a cross-attention or bilinear comparison layer. Version 2 treats it as a decision stage whose output must be calibrated, because the novelty signal decides what is written to memory, what is replayed and, potentially, how much computation to spend. Sections 5.2 and 5.3 develop this using calibrated decision models and joint-embedding predictive architectures."));
C.push(note("Implementation implication: a dual-input comparator that outputs a fused representation and a novelty probability. Implemented as limbic_llm.CA1Comparator using the features [a; r; a⊙r; |a−r|]."));
C.push(h2("3.6  Subiculum → Output Projection with Burst Amplification"));
C.push(p("The subiculum fires intrinsic high-frequency bursts that amplify hippocampal output for transmission to distant targets. In ML terms this is an output projection with a gain that depends on the upstream comparator, sharpening output when CA1 signals a confident match or a salient mismatch. Because it distributes CA1’s verdict to many targets (EC, PFC, nucleus accumbens, mammillary bodies), it is also a natural router of CA1 decisions."));
C.push(h2("3.7  Skip Connections: The Temporoammonic Bypass"));
C.push(p("The direct temporoammonic path from EC layer III to CA1 bypasses the entire DG→CA3 chain. This is a textbook residual connection: it lets the raw input reach the comparator directly so it can be compared against the processed version. The CA3→CA1 Schaffer collaterals are the main pathway, not the skip connection."));
C.push(h3("Summary: Subregion → ML Component Mapping"));
C.push(table(["Subregion", "ML Component", "Key Mechanism", "limbic-llm module"], [
  ["EC", "Contextualized embedding", "Dual-stream multimodal aggregation", "TransformerLayer (single stream)"],
  ["DG", "Top-k sparse gate", "Expand 8–16×, keep 2–4%", "DentateGyrus"],
  ["CA3", "Modern Hopfield / attention with a sequence store", "One-shot hₜ ↦ hₜ₊₁ binding, partial-cue retrieval", "CA3Autoassociative, EpisodicStore"],
  ["CA2", "FiLM / activation steering", "Context-driven scale and shift", "CA2Modulator (untested)"],
  ["CA1", "Comparator and calibrated decision", "Match vs. mismatch → novelty probability", "CA1Comparator"],
  ["Subiculum", "Gain-modulated output projection", "Novelty-scaled output", "Subiculum"],
  ["EC→CA1", "Residual connection", "Raw input reaches the comparator", "HippocampalBlock skip path"],
], [1400, 2600, 2900, 2460]));

// ═══ 4. TITANS
C.push(h1("4. The Titans Architecture: A Convergent Design"));
C.push(h2("4.1  Titans Overview: Three Memory Systems"));
C.push(p("Titans (Behrouz, Zhong, and Mirrokni, 2024) is built around three components. The Core uses sliding-window attention for accurate local dependency modeling (short-term memory). The Long-term Memory Module (LMM) is a deep network that memorizes history by updating its own weights at test time with gradient-based surprise (long-term memory). Persistent Memory is a set of learnable, data-independent parameters that encode task knowledge (semantic memory). Several of these choices converge on principles that evolution discovered in the hippocampal formation."));
C.push(h2("4.2  LMM as the DG→CA3 Complex"));
C.push(p("The LMM’s associative memory loss checks whether the current memory can reconstruct a value from its key. When reconstruction error is high the memory is updated strongly; when it is low, little changes. This mirrors the observation that LTP induction in CA3 is strongest for novel associations and weakest for already-stored patterns."));
C.push(eq("ℓ(M_{t−1}; x_{t}) = ‖ M_{t−1}(k_{t}) − v_{t} ‖^{2}"));
C.push(h2("4.3  The Surprise Metric and Biological Novelty Detection"));
C.push(p("The gradient of that loss is a computable novelty signal, playing the role the dentate gyrus plays biologically: large gradient means high novelty and a strong encoding signal."));
C.push(eq("S_{t} = η_{t} S_{t−1} − θ_{t} ∇ℓ(M_{t−1}; x_{t}),     M_{t} = (1 − α_{t}) M_{t−1} + S_{t}"));
C.push(h2("4.4  Momentum as Theta-Sequence Temporal Attention"));
C.push(p("Behrouz et al. decompose surprise into past surprise (the momentum term η) and momentary surprise (the current gradient). This keeps the memory paying attention for several steps after a surprising event, mirroring the temporally extended encoding window seen in hippocampal theta sequences."));
C.push(h2("4.5  Weight Decay as Synaptic Depotentiation"));
C.push(p("The forgetting term α corresponds to synaptic depotentiation and heterosynaptic long-term depression. Without it the autoassociative store saturates. The authors show it generalizes the forgetting gates in Mamba and gated linear attention."));
C.push(h2("4.6  Architectural Variants and Hippocampal Analogs"));
C.push(lead("MAC (Memory as Context) → hippocampus augmenting neocortex.", "LMM retrievals are prepended to the attention window, so the Core attends to current input and memory together."));
C.push(lead("MAG (Gated Memory) → CA2-like modulation.", "Memory and Core outputs are mixed by a learned gate, y = g⊙Core(x) + (1−g)⊙LMM(x)."));
C.push(lead("MAL (Memory as Layer) → the trisynaptic pipeline.", "Memory processing runs before attention, mirroring EC → DG → CA3 → CA1."));
C.push(h2("4.7  Gated DeltaNet: The Linear Special Case of Titans"));
C.push(p("Gated DeltaNet (Yang, Kautz and Hatamizadeh, 2025) keeps a fixed-size state matrix updated by the gated delta rule, with normalized keys:"));
C.push(eq("S_{t} = α_{t} S_{t−1} (I − β_{t} k_{t} k_{t}^{⊤}) + β_{t} v_{t} k_{t}^{⊤},     o_{t} = S_{t} q_{t}"));
C.push(p("Take the Titans memory, make it linear (M(k) = Sk), drop momentum (η = 0), and use the loss ½‖Skₜ − vₜ‖². One gradient step of size βₜ, taken after decaying the state by αₜ, gives exactly the rule above:"));
C.push(eq("S_{t} = α_{t} S_{t−1} − β_{t} (α_{t} S_{t−1} k_{t} − v_{t}) k_{t}^{⊤}"));
C.push(p("The only difference from Titans is the order of decay and update; the two coincide as α → 1. Each gate therefore has a hippocampal counterpart: α is forgetting (synaptic decay), β is encoding strength, and the bracket vₜ − αₜSₜ₋₁kₜ is a prediction error, the quantity CA1 computes. In effect Gated DeltaNet folds the CA3 store and the CA1 comparator into a single equation. limbic-llm tests this equivalence directly (test_equals_one_sgd_step_on_titans_loss_at_decayed_state)."));
C.push(p("The Qwenputer project shows what this rule can do when addressing is exact. Its compiler gives every memory address a one-hot key, so any two keys are orthogonal; with α = β = 1 each write replaces exactly one slot and the state behaves as RAM. Trained models use dense learned keys, where the delta rule reduces but does not remove interference between similar keys. Making overlapping inputs nearly orthogonal is precisely the dentate gyrus’s job: its 2–4% sparse codes are a soft version of one-hot addressing, and Qwenputer is the limit that pattern separation approaches. The cost is the same in both systems: a linear memory holds at most about dₖ clean bindings, so more address space means a larger state."));
C.push(spacer(80));
C.push(h3("Titans and Gated DeltaNet ↔ Hippocampus Correspondence"));
C.push(table(["Component", "Hippocampal Analog", "Shared Principle"], [
  ["Core (windowed attention)", "CA1 + local cortical processing", "Accurate local dependency modeling"],
  ["LMM surprise metric", "DG novelty / pattern separation signal", "Strong response to novel inputs"],
  ["LMM associative loss", "CA3 pattern completion", "Retrieval error as the learning signal"],
  ["Momentum η", "Theta-sequence encoding window", "Extended encoding after surprise"],
  ["Forgetting α (Titans and GDN)", "Synaptic depotentiation / LTD", "Prevents saturation"],
  ["Write strength β (GDN)", "Encoding strength, mossy-fiber drive", "Novelty-gated storage"],
  ["Delta error v − αSk (GDN)", "CA1 comparator", "Prediction vs. arrival"],
  ["One-hot keys (Qwenputer)", "Ideal DG pattern separation", "Orthogonal addresses, no interference"],
  ["Persistent Memory", "Neocortical knowledge", "Stable task knowledge"],
], [2800, 2900, 3660]));

// ═══ 5. EXTENDED MAPPINGS
C.push(h1("5. Extended Mappings from Recent Work"));
C.push(p("Four lines of recent work sharpen the mapping for the components that version 1 left least specified: CA2, CA1, and the circuit’s re-entrant loop."));

C.push(h2("5.1  CA2 as Activation Steering"));
C.push(p("Activation Addition (Turner et al., 2023) steers a frozen language model without training. It takes the activation difference between two contrasting prompts at layer l and adds a scaled copy back into the forward pass:"));
C.push(eq("h^{l} ← h^{l} + c (h^{l}_{+} − h^{l}_{−})"));
C.push(p("On GPT-2-XL a “weddings” vector moves completions onto that topic more than 90% of the time, against about 2% without it, and injecting at a middle layer (layer 6) works best. On LLaMA-3-8B it reduces toxicity (0.108 against 0.114 unsteered) with almost no fluency cost, and it is state of the art at steering sentiment from negative to positive (0.669 success against 0.417), though at a fluency cost. Side effects on factual recall are negligible and the inference overhead is under 0.4%. The accompanying notebook, kept in limbic-llm at docs/notebooks/, turns all four “I hate you because…” completions affectionate with a Love − Hate vector at layer 6 and c = 5."));
C.push(lead("Why it fits CA2.", "ActAdd injects a context-defined vector into an intermediate layer, beside the main pathway rather than through it, and scales it with a gain c. CA2 likewise sits mid-circuit between CA3 and CA1 and conditions their traffic. Biologically, stimulating CA2’s vasopressin inputs strengthens social memory (Smith et al., 2016), an amplification that matches the coefficient c. Formally, ActAdd is the FiLM special case γ = 0, β = cΔh of CA2Modulator. It also shows that residual streams are linear enough for this conditioning to work after the fact on a pretrained model."));
C.push(lead("Where it differs.", "ActAdd’s vector is fixed for a generation and chosen by the user. CA2’s signal is dynamic and internal, driven by identity and arousal, and CA2 encodes identity in a geometry that supports both discrimination and generalization (Boyle et al., 2024). The closer analogue is a library of identity vectors β(id) = c(h_id − h_neutral), selected or blended per token by context."));
C.push(note("Proposed test: multi-speaker recall, where the same key binds a different value for each speaker. Adding each speaker’s vector at the CA2 point should separate the bindings; without CA2 they interfere."));

C.push(h2("5.2  CA1 as a Calibrated Decision Readout (JEV, LAYA)"));
C.push(p("JEV (TypeSafe AI) and LAYA (NandhaKishorM) belong to a new class of decision models. Both take a state and a typed question, encode them together, and read a calibrated answer off the final layer in a single forward pass, with no text generation. JEV returns structured fields (a route, a score) with a confidence value. LAYA, an open-source encoder (ModernBERT and mmBERT backbones, Apache-2.0), answers choice, ordinal-score and yes/no questions, the last as a probability that a statement holds. Both are trained with RLCD, reinforcement learning for calibrated decisions, so that their confidences are honest probabilities."));
C.push(lead("The match.", "For CA1, the statement is CA3’s retrieved memory and the state is the current entorhinal input, so CA1 outputs P(match) and novelty is nₜ = 1 − P(match). Reading the answer from the final layer also fits, since CA1 is the last stage before the subiculum."));
C.push(lead("Joint encoding.", "LAYA encodes both inputs together, so they interact across every layer. CA1 is closer to this than to comparing two separately computed embeddings, because both inputs reach the same neuron and only coincident activity triggers its response. The current CA1Comparator sits in between, using the shallow feature mix [a; r; a⊙r; |a−r|] of classic sentence-matching models; a LAYA-style CA1 would be a small attention encoder over the current input and the CA3 retrieval together."));
C.push(lead("Calibration.", "CA1 novelty decides what CA3 writes and what gets replayed, so a miscalibrated CA1 misallocates memory. LAYA’s documentation shows the failure mode: an English checkpoint scored 0% on Khmer while reporting 0.952 confidence. limbic-llm currently trains novelty by regressing it onto a squashed token loss, which is not a proper scoring rule. The fix is to define an observed outcome and train against it with a strictly proper score:"));
C.push(eq("y_{t} = 1[memory’s prediction of x_{t+1} was wrong],     ℒ_{CA1} = −y_{t} log n_{t} − (1 − y_{t}) log(1 − n_{t})"));
C.push(lead("Decisions and routing.", "JEV’s typed outputs suggest the shape of CA1’s downstream role: write or not, a replay priority, loop again or exit. The subiculum then distributes that verdict, which makes it a router of CA1 decisions."));

C.push(h2("5.3  CA1 as Predictive Comparison (LLM-JEPA, LeJEPA)"));
C.push(p("Joint-embedding predictive architectures address the representation side of the same comparison. LLM-JEPA (Huang, LeCun and Balestriero, 2026) adds to the next-token loss a term that compares the final-layer, last-token embeddings of two views of the same content, such as a description and its code:"));
C.push(eq("ℒ = ℒ_{NTP} + λ (1 − cos(Pred(Enc(view_{1})), Enc(view_{2})))"));
C.push(p("It improves fine-tuning across Llama, Gemma, OpenELM and OLMo models and resists overfitting. For CA1 the informative comparison is temporal: CA3’s prediction of the next input against the input that actually arrives, nₜ = 1 − cos(P(rₜ), E(xₜ₊₁)), which fits the sequential CA3 store. If both sides are learned, the cheapest solution is for every embedding to collapse to a constant, making novelty zero everywhere. LeJEPA (Balestriero and LeCun, 2025) prevents collapse with SIGReg, which pushes embeddings toward an isotropic Gaussian using one hyperparameter and no stop-gradient or teacher-student machinery. Gorgan Mohammadi, Halvagal and Zenke (2026) map JEPA-style predictive learning onto cortical circuits; to our knowledge no one has mapped it onto CA1."));
C.push(note("JEPA-style models shape what is compared; JEV and LAYA show how the comparison becomes a calibrated decision. A complete CA1 needs both."));

C.push(h2("5.4  Looped Transformers as Big-Loop Recurrence"));
C.push(p("Ouro (Zhu et al., 2025) applies a stack of weight-shared layers up to T times with a learned exit gate, trained on 7.7 trillion tokens. The 1.4B model at T = 4 matches Qwen3-4B (BBH 71.0 vs. 71.0, GSM8K 78.9 vs. 72.9, MATH500 82.4 vs. 59.6), and the 2.6B model beats Qwen3-8B on MMLU-Pro, BBH and MATH500. Scores rise steeply up to the trained depth and degrade only slightly beyond it. Mechanistically, looping does not increase knowledge storage (about 2 bits per parameter either way) but improves the manipulation and composition of stored knowledge, including multi-hop question answering learned from fewer samples. Harmfulness falls as loops increase, even past the trained depth. A simple exit rule, stop when ‖hₜ − hₜ₋₁‖ falls below a threshold, comes within 1–2 points of the trained gate."));
C.push(lead("The hippocampal counterpart.", "Ouro loops the entire stack, which is recurrence through the whole circuit rather than a side channel, so it maps best onto big-loop recurrence: output re-entering the entorhinal cortex and passing through the circuit again. In REMERGE (Kumaran and McClelland, 2012) and the fMRI work of Koster et al. (2018), this recurrence lets stored episodes be combined into inferences that were never stored directly, such as transitive and multi-hop inference. That matches Ouro’s central finding. Iterating until the hidden state stops changing is also CA3 attractor settling, which CA3Autoassociative exposes through its n_iter setting."));
C.push(lead("Connecting CA1.", "The CA1 novelty signal is a natural driver for the exit gate: keep looping while retrieved memory and input disagree, stop when they match. Ouro’s hidden-state-difference rule is itself a comparator."));

C.push(h2("5.5  Revised Component Mapping"));
C.push(table(["Component", "Version 1 mapping", "Current mapping", "Key inspirations"], [
  ["EC", "Embedding + transformer", "Contextual encoder with structural and sensory streams", "TEM, TEM-t, Hippoformer (ICLR 2026)"],
  ["DG", "MoE gating network", "k-WTA pattern separation; soft one-hot addressing; candidate token router", "HiCL, Qwenputer"],
  ["CA3", "Sparse MLP → Hopfield", "Sequence-binding associative store (episodic, Titans or delta rule)", "Hopfield = attention, Titans, Gated DeltaNet"],
  ["CA2", "Autoencoder latent", "Context-conditioned steering vector (FiLM)", "Activation Addition"],
  ["CA1", "Output logits → comparator", "Calibrated decision readout of memory vs. input", "JEV, LAYA, LLM-JEPA, LeJEPA"],
  ["Subiculum", "Output projection", "Gain-modulated router of CA1 decisions", "Burst amplification"],
  ["EC→…→EC loop", "Not modeled", "Looped, weight-shared depth with a novelty-driven exit", "Ouro, REMERGE"],
  ["Theta rhythm and time cells", "Not modeled", "Loop period τ, clamped damped re-entry, clock of elapsed time", "Hasselmo 2002, Shankar & Howard"],
], [1300, 2000, 3260, 2800], 16));

// ═══ 6. CLOSING THE LOOP
C.push(h1("6. Closing the Loop: Feedback, Time and a Full Architecture"));
C.push(p("This section records how the temporal-loop design was reached, in the order the reasoning developed, and the experiment that tests it. The code is limbic_llm/temporal_loop.py; a fuller write-up is docs/design/temporal-loop.md."));
C.push(h2("6.1  From Static to Dynamic Activation Addition"));
C.push(p("Activation Addition injects one fixed vector, chosen by a user, into a middle layer. CA2’s signal is dynamic and internal. The dynamic version taps an inner layer, writes a time-lagged window of its activations into an associative (CA3) memory, retrieves related context, contrasts it against a negative, and re-injects the difference, with a stronger signal where the contrast is strong. One mechanism then spans three regions: the memory stores and predicts (CA3), the contrast measures novelty (CA1), and the re-injection is the side channel (CA2)."));
C.push(p("Two problems follow. First, a linear memory cannot supply the negative by negating the query, since S(−q) = −Sq. The negative is taken instead as the model’s prior: the episode’s retrieval minus what the model would expect from everything it has seen, so the contrast is what this episode adds. Second, injections change later activations, which are written back and injected again: positive feedback that can run away."));
C.push(eq("Δ_{t} = F_{t} − m_{t} μ_{t},     F_{t} = Σ_{s} w(t−s) a(t,s) v_{s}"));
C.push(h2("6.2  How the Biological Loop Stays Stable"));
C.push(p("The circuit operates near a critical threshold between excitation and inhibition, held by basket, chandelier and O-LM interneurons, and seizures begin there when the balance fails. The hippocampus’s role as the negative-feedback sensor of the stress axis shows the opposite case: when that feedback is damaged, the result is the self-reinforcing glucocorticoid cascade (Sapolsky, Krey and McEwen, 1986). Each safeguard maps onto a design element:"));
C.push(table(["Biological mechanism", "Effect", "Design element"], [
  ["Feedback inhibition (basket cells)", "Inhibition grows with activity", "Gain divided by recent injection energy"],
  ["Acetylcholine encoding/retrieval switch", "Recurrent CA3 synapses suppressed during encoding", "Store the stream from before injection"],
  ["Theta-phase separation", "Encoding and retrieval at opposite phases", "Loop period τ; nothing returns within a circuit"],
  ["Attenuated big-loop re-entry", "Slow, weak return path", "Attenuation ρ < 1 per circuit"],
  ["Self-terminating bursts", "CA3 bursts end within ~100 ms", "Saturating gain"],
  ["Sparse dentate-gyrus input", "Bounded activity entering CA3", "k-WTA on written keys"],
], [3000, 3160, 3200]));
C.push(spacer(80));
C.push(p("The acetylcholine switch also refines the gain rule. Biology suppresses recall when the input is novel, so two signals must be kept apart: input-versus-memory mismatch, which calls for encoding, and memory-versus-prior contrast, which says how much the memory has to contribute (Hasselmo, Bodelón and Wyble, 2002)."));
C.push(h2("6.3  Skip Paths and CA1 Negative Feedback"));
C.push(p("There is no projection from the dentate gyrus to CA1; mossy fibers end in CA3 and the hilus. The skip into CA1 comes from entorhinal layer III, which the HippocampalBlock already has. The neglected path is the direct one from entorhinal layer II to CA3. Treves and Rolls (1992) argued that the dentate gyrus drives storage while the direct path supplies the retrieval cue, because a pattern-separated cue maps to a different code than the stored one. The loop therefore writes with dentate-gyrus keys and reads with a dense query taken directly from the stream."));
C.push(p("CA1 pyramidal cells do not project back to CA3, but GABAergic back-projection interneurons in CA1 do (Sik, Ylinen, Penttonen and Buzsáki, 1994). The arrangement is a negative-feedback amplifier: CA3’s recurrence is a high-gain stage A, the inhibitory back-projection is the feedback network β, and the closed-loop gain is set by the stable feedback path."));
C.push(eq("G = A / (1 + Aβ) ≈ 1/β     (Aβ ≫ 1)"));
C.push(p("In the design, CA1 injects only the part of the contrast that the stream lacks, and divides the gain by the injection energy from the previous circuit, so the injection never exceeds a fixed fraction G < 1 of the incoming signal:"));
C.push(eq("e_{t} = Δ_{t} − proj_{h}(Δ_{t}),     g_{t} = G tanh(‖Δ_{t}‖/λ) / (1 + κE_{t})"));
C.push(h2("6.4  Clamped, Damped Re-entry and a Sense of Time"));
C.push(p("Feedback should return as a clamped sinusoid that decays on every circuit around the loop, so new tokens dominate and fed-back content is attenuated. Content written at time s returns once per circuit of period τ, in a phase window, attenuated by ρ on each pass and bounded by an amplitude A ≤ 1, while new input enters at weight 1:"));
C.push(eq("w(Δ) = A ρ^{Δ/τ − 1} [cos(2πΔ/τ)]_{+},     Δ = t − s ≥ τ"));
C.push(p("The same loop carries time. Transformers know token order within their context through positional encodings, but nothing tells them how long ago an event happened once it has left the attention window. The hippocampus has time cells that fire in sequence across a delay (MacDonald et al., 2011), and a bank of leaky integrators with log-spaced time constants gives a scale-invariant code for elapsed time (Shankar and Howard, 2012). A time channel implements that bank, with damped oscillators for phase, driven by a learned salience and added to the stream:"));
C.push(eq("L_{j}(t) = (1 − ρ_{j}) Σ_{s≤t} ρ_{j}^{t−s} u_{s},     ρ_{j} = exp(−1/τ_{j}),   τ_{j} ∈ {4, 8, …, 256}"));
C.push(h2("6.5  Plug-in or New Architecture?"));
C.push(table(["Piece", "Closest existing mechanism", "Plugs into"], [
  ["Associative memory of the stream", "Gated DeltaNet, linear attention, Titans", "Hybrid linear-attention layers"],
  ["Side-channel injection", "Activation Addition, CAA, CAST", "Residual-stream hooks on a frozen model"],
  ["Memory-retrieved steering", "Dynamic Steering with Episodic Memory", "The same hooks"],
  ["Damped oscillatory kernel", "RetNet, LRU decays", "Any linear recurrence"],
  ["Clock of elapsed time", "Time-cell and Laplace models", "A new side channel"],
  ["Looped depth", "Ouro, Universal Transformers", "Weight-shared blocks"],
], [3000, 3360, 3000]));
C.push(spacer(80));
C.push(p("Each piece has a precedent and can be attached to an existing model on its own. The combination is a new architecture: a closed loop that writes the model’s own lagged activations into a fast associative memory, re-injects contrastive, time-tagged feedback that is damped on every circuit, and regulates it with negative feedback, beside a clock of elapsed time. The open loop is a plug-in; the closed, regulated, timed loop is a new organisation."));
C.push(h2("6.6  A Full Limbic LLM"));
C.push(p("In the brain, consolidated long-term memory lives in cortical synapses, and white matter is the long-range wiring between areas. In a transformer, the weights of the layer stack are the consolidated cortical memory and the depth is the cortical hierarchy, while the residual stream is the white-matter bus that every layer reads and writes. The hippocampal side channels connect to that bus. Around a pretrained decoder, frozen at first:"));
C.push(bullet(" at about one third of the depth, where features are abstract but not task-specific.", "An entorhinal tap"));
C.push(bullet(" with sparse dentate-gyrus write keys, an exact delta-rule store with damped multi-band decays, a direct entorhinal read query, CA2 contrast and identity steering, CA1 negative-feedback control, and an acetylcholine gate that suppresses injection while input is novel.", "A hippocampal loop"));
C.push(bullet(" at about half the depth, together with the time channel, so later layers integrate memory and elapsed time.", "Injection"));
C.push(bullet(" re-running the upper block, with CA1 novelty deciding when to stop.", "An optional big loop"));
C.push(bullet(" replaying high-novelty episodes offline into adapters on the frozen stack, moving episodic memory into the cortex.", "Consolidation"));
C.push(p("Training would start with the side-loop parameters alone, on long documents with the context window deliberately truncated so that memory is the only route to distant information, then a light joint fine-tune, then consolidation for continual learning."));
C.push(h2("6.7  Experiment: What, When, and Loop Stability"));
C.push(p("In the what-when task an event token (8 types) is followed by 20–140 filler tokens and a query. The model must say what happened (chance 0.125) and when, in four log-spaced elapsed-time bins. The bins cover 15, 25, 40 and 40 of the 120 possible gaps, so a model that only learns the prior scores 0.333 on when. Attention sees only 16 tokens and there is no absolute position embedding, so content can only arrive through the memory loop and elapsed time only through the side channels. The stability suite trains closed loops, which write their own feedback back into memory, on 160-token sequences and then runs them on 520-token sequences, measuring the injection as a fraction of the incoming signal."));
const RES = name => { try { return JSON.parse(fs.readFileSync(path.join(__dirname, "..", "..", "results", name), "utf8")); } catch (e) { return null; } };
const f2 = x => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(2);
const f3 = x => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(3);
const R = n => { const d = RES(n); return d ? d.results : {}; };
const ALL = R("temporal_loop_all.json"), LONG0 = R("temporal_loop_long.json"), LONG1 = R("temporal_loop_long_seed1.json");
const PROBE = R("temporal_loop_probe.json"), FB = R("temporal_loop_feedback.json"), AB = R("temporal_loop_ablations.json"), OPEN = R("temporal_loop_open.json");
if (ALL.window) {
  C.push(h3("Which channel carries which answer"));
  const ww = [["Windowed attention only", ALL.window], ["Memory loop", ALL.memory], ["Time channel only", ALL.time],
    ["Memory loop + time channel", ALL.memory_time], ["Memory loop + time channel, 3,000 steps", LONG0.memory_time]].filter(r => r[1]);
  C.push(table(["Model (1,500 steps unless noted)", "What (chance 0.125)", "When (prior 0.333)"], ww.map(([l, r]) => [l, f3(r.what), f3(r.when)]), [5160, 2100, 2100]));
  C.push(spacer(80));
  C.push(p("The loop carries content and the clock carries elapsed time; each alone leaves the other answer at baseline. The combined model learned what only with the longer schedule, so its 1,500-step failure was the timing of a sudden late jump in accuracy, not a missing capability."));
  C.push(h3("Closed-loop stability"));
  const st = [["Undamped (flat), 3,000 steps", PROBE.closed_flat], ["Damped (exp), 3,000 steps", PROBE.closed_exp],
    ["Clamped sinusoid, no control, 3,000 steps", FB.closed_osc], ["Full design, seed 0, 3,000 steps", LONG0.closed_osc_ca1],
    ["Full design, seed 1, 3,000 steps", LONG1.closed_osc_ca1]].filter(r => r[1]);
  C.push(table(["Closed loop", "What", "Echo 16–128", "Echo 128–320", "Echo 320–519", "Max"],
    st.map(([l, r]) => [l, f3(r.what), f2(r.long.inj_ratio_16_128), f2(r.long.inj_ratio_128_320), f2(r.long.inj_ratio_320_519), f2(r.long.inj_ratio_max)]),
    [3560, 960, 1210, 1210, 1210, 1210], 16));
  C.push(spacer(80));
  C.push(p("Echo is the injection as a fraction of the incoming signal on 520-token sequences. Only the full design keeps it below the new input everywhere, and it stays stationary across the run. Undamped feedback grows with length; damping alone overshoots in the first circuits; the clamped sinusoid without control resonates. None diverged numerically, because the next layer’s LayerNorm rescales the stream. The cost is forgetting: at gaps of 300–500 tokens recall is 0.40 undamped, 0.00 damped and 0.04–0.10 for the full design."));
  C.push(h3("Do new tokens dominate?"));
  const pr = [["Undamped closed loop", PROBE.closed_flat], ["Damped closed loop", PROBE.closed_exp], ["Subtractive feedback, no clamp, seed 0", FB.closed_osc_sub],
    ["Subtractive feedback, no clamp, seed 1", OPEN.closed_osc_sub],
    ["Full design without contrast", AB.closed_osc_ca1_nocontrast], ["Full design", PROBE.closed_osc_ca1],
    ["Full design, DG keys, direct read", AB.closed_osc_ca1_dg], ["Full design, DG keys, DG read", AB.closed_osc_ca1_dgread],
    ["Open loop (stores the pre-injection stream), seed 1", OPEN.memory]].filter(r => r[1] && r[1].long && r[1].long.probe_newer_named !== undefined);
  C.push(table(["Model, probed zero-shot with an older event first", "Names newer", "Names older"],
    pr.map(([l, r]) => [l, f2(r.long.probe_newer_named), f2(r.long.probe_older_named)]), [5560, 1900, 1900]));
  C.push(spacer(80));
  C.push(p("A trained recency task was learned by no model in 3,000 steps, so each model trained on single events was probed instead. Closed loops favour the first event: they re-store their own echo every circuit, and each re-stored copy starts its decay again, so damping each trace does not damp content the loop keeps rehearsing. This is primacy by self-rehearsal. Subtracting the prior reduces it, plausibly because rehearsed content becomes part of the prior; sparse dentate-gyrus write keys give the only clear recency win."));
}

C.push(h2("6.8  The Op-amp Test: Divisive or Subtractive Feedback"));
C.push(p("The CA1 control used so far is divisive: it divides the gain by the energy of recent injections, which at high gain only compresses. An op-amp subtracts a fraction of its own output, so the closer analogue is a subtractive back-projection carrying an inverted copy of earlier injections, a self-negative in the language of activation addition:"));
C.push(eq("y_{t} = A (Δ_{t} − β ȳ_{t}),     y = AΔ / (1 + Aβ) → Δ/β"));
C.push(p("A lagged, smoothed copy can only cancel what persists from circuit to circuit, so subtractive feedback is a high-pass filter on the injection: sustained echo is attenuated by 1/(1 + K), with K = Agβ, while new content passes. Because the copy arrives a circuit late, the loop is stable only while the characteristic polynomial below keeps its roots inside the unit circle; for τ = 16 and an inhibition decay of 0.9 per token that requires K < 1.68, with a ringing period of about 44 tokens, which is why op-amps need compensation."));
C.push(eq("z^{τ} − ρ_{E} z^{τ−1} + K(1 − ρ_{E}) = 0"));
const SYS = RES("loop_systems.json");
if (SYS) {
  const pers = SYS.rows.filter(r => r.vocab === 1);
  const pairs = [];
  for (let i = 0; i + 1 < pers.length; i += 2) pairs.push([pers[i], pers[i + 1]]);
  C.push(table(["K = Agβ", ...pairs.map(([n, s]) => f2(s.K))],
    [["Subtractive / no feedback", ...pairs.map(([n, s]) => f3(s.inj_late / n.inj_late))], ["1/(1 + K)", ...pairs.map(([n, s]) => f3(1 / (1 + s.K)))]],
    [3360, 1500, 1500, 1500, 1500]));
  C.push(spacer(80));
}
C.push(p("On an untrained closed loop with the clamp removed and a persistent input, the measured attenuation matches the op-amp law within 1–3 %, and the loop rings and grows at K = 2.15 as predicted. Trained without the clamp, subtractive feedback was the only unregulated loop that learned content (what 0.99, against 0.13 with no feedback and 0.15 with divisive feedback). It learned on both seeds but did not bound the echo (1.7\u20132.6\u00D7 the input), and its effect on primacy differed between seeds, because at a stable loop gain it removes only about a quarter of sustained echo. A guaranteed bound still needs the clamp."));

C.push(h2("6.9  Pretrained Models"));
C.push(p("The side channels were attached to frozen Hugging Face models (experiments/pretrained_loop.py) and trained alone on a needle task with a gap of about 500 tokens and block attention windows of 64 tokens, on two RTX 5090s. Qwen3 is pure attention; Qwen3.5 is a 3:1 Gated DeltaNet hybrid whose GDN layers already carry the past."));
const PQ = n => { const d = RES(n); return d ? d.results : null; };
const prow = [["Qwen3-1.7B-Base, all layers windowed", PQ("pretrained_qwen3_needle.json")],
  ["Qwen3.5-2B-Base, attention windowed, tap block 12", PQ("pretrained_qwen35_needle.json")],
  ["Qwen3.5-2B-Base, attention windowed, tap block 22", PQ("pretrained_qwen35_needle_layer22.json")],
  ["Qwen3-1.7B-Base, all windowed, horizon kernel", PQ("pretrained_qwen3_needle_horizon.json")],
  ["Qwen3.5-2B-Base, all windowed, horizon kernel", PQ("pretrained_qwen35_needle_allwin_horizon.json")]].filter(r => r[1]);
if (prow.length) {
  C.push(table(["Model (exact match)", "Windowed", "Windowed + side", "Full attention"],
    prow.map(([l, r]) => [l, f2(r.windowed.exact), f2(r["windowed+side"].exact), f2(r.full.exact)]), [4560, 1600, 1600, 1600]));
  C.push(spacer(80));
}
C.push(p("Qwen3.5’s GDN layers alone did not recall the needle, which suggests native GDN memory keeps too weak a trace for exact recall at this distance. With the side channels at block 12 recall was exact; moving the tap to block 22 removed it, so the success depends on GDN layers after the tap. The loop could not see the needle itself: with ρ = 0.7 and τ = 16 its span is τ/ln(1/ρ) ≈ 45 tokens, so the Qwen3 failure reflected a setting rather than evidence against the loop. With a horizon that covers the gap (exp kernel, \u03C1 = 0.95, \u03C4 = 32) and every layer windowed, the loop alone carried needle information 500 tokens back in both models: digit loss fell from the answer-format floor of 1.84 to 1.38 (Qwen3) and 1.24 (Qwen3.5), with token accuracy 0.46\u20130.55, though exact recall of all four digits stayed near zero. The working hypothesis is a chain in which GDN layers before the tap carry a weak trace, the loop re-presents it, and GDN layers after the tap integrate it. Damping trades memory for recency, so the loop’s horizon has to match the task, and several bands with different ρ may be needed."));

// ═══ 7. STRATEGY I
C.push(h1("7. Implementation Strategy I: Direct Hippocampal Architecture"));
C.push(note("Implemented in limbic-llm as HippocampalBlock and HippocampalLM. The code below is the design sketch; the package versions add the sequential CA3 store, CA2 modulation, causal masking and streaming state."));
C.push(h2("7.1  Architecture Overview"));
C.push(p("This strategy translates the trisynaptic circuit into a module usable either as a standalone sequence model or as a memory block inside a larger transformer. It has five stages, EC embedding, DG sparse gating, CA3 associative memory, CA1 comparison and subicular output, plus a residual skip that routes the EC embedding directly to CA1."));
C.push(h2("7.2  DG Sparse Gating Module"));
C.push(...codeBlock([
  "class DentateGyrus(nn.Module):",
  "    def __init__(self, d_model, expansion=8, sparsity=0.04):",
  "        super().__init__()",
  "        self.d_total = d_model * expansion",
  "        self.k = max(1, int(self.d_total * sparsity))",
  "        self.norm = nn.LayerNorm(d_model)",
  "        self.expand = nn.Linear(d_model, self.d_total)",
  "",
  "    def forward(self, x):                        # (B, T, d_model)",
  "        h = self.expand(self.norm(x))            # expand",
  "        top = h.topk(self.k, dim=-1)             # lateral inhibition (k-WTA)",
  "        mask = torch.zeros_like(h).scatter(-1, top.indices, 1.0)",
  "        return F.relu(h) * mask                  # ~4% of units active",
]));
C.push(h2("7.3  CA3 Associative Memory with a Sequence Store"));
C.push(...codeBlock([
  "class CA3(nn.Module):",
  "    def forward(self, dg_code, store=None):",
  "        h = self.norm(self.mossy(dg_code))       # mossy-fiber terminal field",
  "        keys = [store.cues, h] if store else [h]  # stored cues + current segment",
  "        vals = [store.successors, h] if store else [h]",
  "        completed = h + hopfield_retrieve(h, keys, vals, causal=True)",
  "        return completed, h",
  "",
  "    def write(self, store, h, novelty):",
  "        idx = novelty[:, 1:].topk(m).indices    # most novel transitions",
  "        store.cues[weakest] = h[idx]             # cue h_t ...",
  "        store.successors[weakest] = h[idx + 1]   # ... bound to h_t+1",
]));
C.push(h2("7.4  CA1 Comparator Layer"));
C.push(...codeBlock([
  "class CA1Comparator(nn.Module):",
  "    def forward(self, retrieved, current):",
  "        a, r = self.distal(current), self.proximal(retrieved)",
  "        phi = torch.cat([a, r, a * r, (a - r).abs()], dim=-1)",
  "        novelty = torch.sigmoid(self.novelty_head(phi))  # P(mismatch)",
  "        return self.fuse(phi), novelty",
]));
C.push(h2("7.5  Offline Replay and Consolidation"));
C.push(p("High-novelty sequences are stored in a prioritized replay buffer together with the logits the model produced at encoding time. During an offline sleep phase, replayed traces train only the slow (neocortical) parameters at a low learning rate, with a distillation term that anchors the function as well as the labels."));
C.push(...codeBlock([
  "buffer.add(seq, priority=novelty.mean(1), logits=out.logits)   # awake",
  "loss = task_loss + consolidator.replay_loss(batch_size=16)     # interleaved replay",
  "consolidator.sleep(steps=200)                                  # slow params only",
]));
C.push(h2("7.6  Forward Pass"));
C.push(...codeBlock([
  "def forward(self, x, store=None, context=None):  # x: EC embedding",
  "    code = self.dg(x)                             # pattern separation",
  "    completed, h, mass = self.ca3(code, store)    # pattern completion",
  "    completed = self.ca2(completed, context)      # conditional modulation",
  "    fused, novelty = self.ca1(completed, x)       # EC -> CA1 skip comparison",
  "    out = x + self.subiculum(fused, novelty)      # gain-modulated output",
  "    store = self.ca3.write(store, h, novelty)     # novelty-gated one-shot write",
  "    return out, novelty, store",
]));

// ═══ 8. STRATEGY II
C.push(h1("8. Implementation Strategy II: Titans-Hippocampal Hybrid"));
C.push(note("Implemented in limbic-llm as TitansHippocampalLayer and TitansHippocampalLM, with each hippocampal addition behind its own flag, and with memory_type=\"gated_delta\" to swap the deep memory for an exact delta-rule memory."));
C.push(h2("8.1  Enhancing Titans with Hippocampal Principles"));
C.push(p("Titans already implements surprise-based encoding, momentum and adaptive forgetting. This strategy adds four features it lacks: DG sparse gating on memory writes, a CA1 comparator in the MAC variant, novelty-scaled memory learning rates, and replay-based consolidation. The experiments added a fifth, sequential (key → next value) writes."));
C.push(h2("8.2  DG-Gated, Novelty-Scaled Memory Writes"));
C.push(...codeBlock([
  "def write(self, x, state, novelty=None):",
  "    feat = self.norm(x)",
  "    if self.dg: feat = self.dg(feat)           # pattern separation",
  "    k, v = normalize(self.wk(feat)), self.wv(feat)",
  "    if self.sequential:                        # bind cue -> successor",
  "        k, v = k[:, :-1], v[:, 1:]",
  "    theta = self.lr(x) * (1 + gain * (novelty - 0.5))   # CA1-gated",
  "    err = (M(k) - v).pow(2).sum(-1)            # summed, as in the paper",
  "    g = grad((theta * err).sum(), alias(M))    # partial derivative only",
  "    S = eta * S - clip(g)                      # momentum + trust region",
  "    M = (1 - alpha) * M + S                    # forgetting",
]));
C.push(h2("8.3  CA1 Comparator in MAC"));
C.push(...codeBlock([
  "for chunk in chunks(x):",
  "    retrieved = memory.read(chunk, state)              # state before the chunk",
  "    fused, novelty = ca1(retrieved, chunk)             # compare memory vs. input",
  "    y = attention(chunk, kv=[persistent, fused, chunk])",
  "    state = memory.write(y, state, novelty)            # written after the chunk",
]));
C.push(h2("8.4  Replay-Based Consolidation"));
C.push(p("The Titans memory learns online but never transfers what it learns into the persistent parameters. Replaying high-surprise sequences into Persistent Memory and other slow parameters during an offline phase creates a complementary learning system: a fast hippocampal learner and a slow cortical consolidator."));

// ═══ 9. RESULTS
C.push(h1("9. limbic-llm: Implementation and Results"));
C.push(p("limbic-llm is a PyTorch research library, not a pretrained LLM. It provides the memory modules and four small decoder-only language models that share an embedding, an LM head and a next-token loss, so they can be compared on equal terms. Models in the experiments have 0.2–0.4 million parameters with d_model = 64 and run on a CPU. The package has 51 tests covering causality, memory rules, streaming state and regressions. Results are from single seeds and are sanity checks rather than benchmarks."));
C.push(h2("9.1  One-Shot Associative Recall"));
C.push(p("Each sequence presents 6 random key→value pairs, then 40 filler tokens, then queries every key in shuffled order. Only recalled values are scored, so chance is 1/16 = 0.0625. Local models see 16 tokens at a time, so the study pairs are out of their direct view at query time. Training ran for 3,000 steps."));
C.push(table(["Model", "Params", "Recall"], [
  ["Full attention (reference, sees everything)", "221 k", "0.485"],
  ["Windowed attention (16 tokens, no memory)", "221 k", "0.059"],
  ["Strategy I (HippocampalLM)", "329 k", "0.414"],
  ["Titans MAC (deep memory)", "238 k", "0.483"],
  ["Titans + sequential writes", "238 k", "0.490"],
  ["Titans + sequential + DG writes", "346 k", "0.478"],
  ["Titans + sequential + CA1 and novelty lr", "322 k", "0.440"],
  ["Titans + all hippocampal additions", "429 k", "0.172"],
  ["Titans with Gated DeltaNet memory", "198 k", "1.000"],
  ["Titans with Gated DeltaNet + DG addresses", "257 k", "0.989"],
], [5560, 1600, 2200]));
C.push(spacer(80));
C.push(p("Both memory systems recall bindings from well outside their attention window, which windowed attention cannot. Replacing the deep Titans memory with an exact delta-rule memory raises recall from 0.48 to 1.00, and DG addresses reach high recall faster (0.905 vs. 0.680 at step 500) while finishing slightly lower. The hippocampal additions are neutral individually but slow optimization badly when combined, probably because the novelty-scaled learning rate is noisy before CA1 has learned anything."));
C.push(h2("9.2  Gated DeltaNet Capacity"));
C.push(p("Pure Gated DeltaNet models without attention were trained on recall with 64 possible keys at increasing load."));
C.push(table(["Mode", "Key dim / head", "8 pairs", "16 pairs", "32 pairs"], [
  ["Standard GDN", "16", "0.426", "0.134", "0.137"],
  ["Standard GDN", "128", "0.428", "0.320", "0.222"],
  ["Tied dense address, successor binding", "128", "1.000", "1.000", "1.000"],
  ["DG k-WTA address, successor binding", "128", "1.000", "1.000", "1.000"],
], [3560, 1500, 1400, 1450, 1450]));
C.push(spacer(80));
C.push(p("Standard Gated DeltaNet runs out of capacity as load passes its key dimension. Successor-bound addresses turn the state into RAM: a counterfactual check, swapping the values shown at study time, confirmed that predictions follow the new values 100% of the time. Dense and DG addresses tie at the ceiling, because with 128-dimensional addresses and distinct keys even dense codes barely interfere; testing pattern separation needs similar keys."));
C.push(h2("9.3  Continual Learning with Replay"));
C.push(p("Three tasks with disjoint, fixed key→value facts were learned in sequence for 600 steps each with Strategy I. Naive sequential training forgot catastrophically (average final accuracy 0.513, backward transfer −0.731). Novelty-prioritized replay with logit distillation and an offline sleep phase removed the forgetting entirely (average 1.000, backward transfer 0.000). The task is easy, and the run does not separate the sleep phase from interleaved replay."));
C.push(h2("9.4  Design Changes the Experiments Forced"));
C.push(bullet(" A purely autoassociative CA3 returns the cue, not its value. Binding hₜ ↦ hₜ₊₁ was needed before any memory model rose above chance.", "CA3 stores sequences."));
C.push(bullet(" In MAC and MAL the chunk being written was computed from an earlier read of the same memory, so differentiating through it made training apply a different update than inference. The gradient is now taken with respect to a fresh alias of the memory.", "The Titans meta-gradient is a partial derivative."));
C.push(bullet(" Averaging the squared error over features shrank writes by 1/d; summing, as in the paper, needs a per-sample trust region to stay stable. Sparse DG keys were stable even without it.", "Memory loss scale and a trust region."));

// ═══ 10. PRIOR ART
C.push(h1("10. Prior Art and Positioning"));
C.push(p("Most components of this mapping have prior work; two remain largely open."));
C.push(table(["Work", "What it models", "Relation to this work"], [
  ["Hippoformer (Li et al., ICLR 2026)", "EC path integration + Titans-style relational memory beside a transformer", "Closest prior work; no DG, CA2 or CA1. Its name prompted renaming this package"],
  ["HiCL (Kapoor et al., 2025)", "DG-gated mixture of experts with prototype routing", "Task-level routing for vision continual learning"],
  ["TEM and TEM-t (Whittington et al.)", "EC structural vs. sensory codes; position encodings as grid cells", "EC mapping"],
  ["Larimar (Das et al., 2024)", "Episodic memory controller for LLMs", "One-shot fact editing"],
  ["HippoRAG (Gutiérrez et al., 2024)", "Hippocampal indexing theory for retrieval", "Graph-based pattern completion"],
  ["EM-LLM (Fountas et al., 2025)", "Surprise-based event segmentation", "Episodic memory over long contexts"],
], [2700, 3360, 3300], 16));
C.push(spacer(100));
C.push(p("Three directions appear to be open:"));
C.push(bullet(" HiCL routes whole tasks; routing individual tokens by sparse DG codes and prototypes inside a language model is untested.", "DG as a token-level router."));
C.push(bullet(" No ML architecture we found models CA2; identity-conditioned memory with activation-steering vectors would be new.", "CA2 as identity-conditioned memory."));
C.push(bullet(" Combining a JEPA-style predictive comparison with a calibrated, proper-scoring-rule decision readout, and letting it drive memory writes, replay and loop depth.", "CA1 as a calibrated decision stage."));

// ═══ 11. CONTINUAL LEARNING
C.push(h1("11. Connections to Continual Learning and Catastrophic Forgetting"));
C.push(p("The hippocampus is evolution’s solution to catastrophic forgetting. The complementary learning systems framework (McClelland, McNaughton and O’Reilly, 1995) holds that the brain uses two learning systems because no single system can both encode new information rapidly and maintain old knowledge stably. The hippocampus learns fast, with sparse representations and single-trial plasticity; the neocortex learns slowly, extracting regularities through interleaved replay (Wilson and McNaughton, 1994)."));
C.push(p("Both strategies embody this principle. The fast learner (CA3 store, Titans memory or delta-rule memory) absorbs new information at test time, and the slow learner (neocortical layers or Persistent Memory) consolidates replayed patterns at a low learning rate. The replay buffer, prioritized by CA1 novelty, focuses consolidation on the most informative experiences, and the continual-learning result in Section 9.3 shows the mechanism working on a small scale."));
C.push(p("This connects directly to representation-level approaches. Independent Component Analysis seeks maximally independent latent components, and the dentate gyrus performs an analogous operation: its sparse codes make individual granule-cell activations approximately independent, minimizing interference between stored patterns. limbic-llm includes an ICA-style decorrelation penalty on DG codes, and its replay consolidator exposes an orthogonal gradient-projection hook where ICA-identified functional networks can protect previously consolidated knowledge during sleep updates."));

// ═══ 12. OPEN QUESTIONS
C.push(h1("12. Open Questions and Future Directions"));
C.push(lead("Pattern separation under similar inputs.", "The capacity sweep could not separate DG from dense addressing because its keys were distinct tokens. A similar-keys task (shared prefixes or added noise) and a sweep of DG sparsity from one-hot (about 1%) to 25% would show where pattern separation pays off."));
C.push(lead("Combining hippocampal additions.", "DG writes and CA1 novelty each work alone but slow training together. Warming up the novelty-scaled learning rate, or stopping gradients through novelty, are the next things to try."));
C.push(lead("A calibrated CA1.", "Train novelty with a proper scoring rule against an observed outcome (Section 5.2), add a JEPA-style predictive target regularized by SIGReg (Section 5.3), and measure calibration directly with reliability diagrams."));
C.push(lead("CA2 identity steering.", "Multi-speaker recall with per-speaker steering vectors at the CA2 point (Section 5.1); also the post-hoc version, adding identity vectors to a pretrained model."));
C.push(lead("Big-loop recurrence.", "A looped HippocampalBlock whose exit gate is driven by CA1 novelty, tested on multi-hop recall where stored pairs must be chained."));
C.push(lead("Deep versus linear memory.", "The exact delta-rule memory beat the deep Titans memory on recall. Whether depth helps on tasks that need nonlinear associations, and whether chunk-start gradient updates are the real cost, is unresolved."));
C.push(lead("Neurogenesis and consolidation scheduling.", "Progressive DG expansion during continual learning, and whether replay should follow a fixed, surprise-triggered or adaptive schedule, remain untested."));
C.push(lead("Scale.", "All results so far use tiny models. The natural next step is to insert HippocampalBlock or GatedDeltaMemory into an existing pretrained model rather than scaling the toy models."));

// ═══ 13. REFERENCES
C.push(h1("13. References"));
C.push(p("The same bibliography is maintained in limbic-llm as REFERENCES.md and docs/references.bib."));
for (const g of REFS) {
  C.push(h3(g.group));
  for (const e of g.entries) {
    C.push(new Paragraph({ spacing: { before: 40, after: 60, line: 276 }, indent: { left: 480, hanging: 480 },
      children: [new TextRun({ text: e.text, font: "Calibri", size: 19, color: BODY })] }));
  }
}

// ═══ BUILD
const doc = new Document({
  creator: "Daniel Byrne",
  title: "Hippocampal Architectures for Machine Learning",
  description: "Version 3, with the temporal loop and results from limbic-llm",
  numbering,
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Georgia", color: NAVY }, paragraph: { spacing: { before: 480, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Georgia", color: DARK_BLUE }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "Hippocampal Architectures for Machine Learning · v3", font: "Calibri", size: 16, italics: true, color: "999999" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      border: { top: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" } }, spacing: { before: 80 },
      children: [new TextRun({ text: "Page ", font: "Calibri", size: 18, color: "999999" }),
                 new TextRun({ children: [PageNumber.CURRENT], font: "Calibri", size: 18, color: "999999" })] })] }) },
    children: C,
  }],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log(`wrote ${OUT} (${(buf.length / 1024).toFixed(0)} KB)`); });
