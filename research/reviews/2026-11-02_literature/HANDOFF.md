# Handoff: literature review and code audit (2 Nov 2026)

For any agent working on limbic-llm: the GPU session on `feat/round1-side-memory`, the coordinating session, or a new one. This note says what the review concluded, what it changed, and what to do with it. It does **not** replace [`research/HANDOFF.md`](../../HANDOFF.md), which belongs to the GPU session; that file was not edited.

## Provenance

- Written by a Claude Science session at Daniel's request; prompts quoted in [`docs/genesis.md`](../../../docs/genesis.md) §9.
- Review ID **RV2** in the register. Reviewed commit `3799b36` (main after PR #2). Committed on branch `docs/literature-review` on top of `feat/round1-side-memory` at `1620784`, so it sits on top of rounds 1–2 and the 2 Nov records reorganisation (B-, DEC- and RV-numbering).
- Method: seven parallel reviewers (CA3–CA1, CA2, DG–MoE, GDN, the Samsonovich & Ascoli paper, bibliography audit, code review). Each verified every DOI and arXiv id it cited against Crossref, arXiv or OpenAlex.
- What was re-computed: the binding-free ceiling (twice, independently), the critical gain K = 1.680 / period 44.1 tokens (twice), and the S&A 2005 path lengths. The test suite was run: 101 passed, 3 skipped on CPU. **No GPU experiment was re-run.** All other numbers are read from the result files.

## How to consume this folder

| File | Use |
|---|---|
| [`README.md`](README.md) | Synthesis: positioning, the four tracks, the paper, §3 measurement, §3a round-1 reading, biology corrections, novelty table, phased programme |
| [`findings.json`](findings.json) | Machine-readable: per track `summary`, `closest_prior_work` (citation, verified id, relation), `novelty` (verdicts), `proposed_experiments`, `corrections`, `declared_limits` |
| `ca3ca1.md`, `ca2.md`, `dg_moe.md`, `gdn.md`, `pathfinding.md` | Track reviews with full arguments and citations |
| `code_review.md` | Implementation fidelity, experimental validity, minimal publishable experiment set |
| `bib_audit.md` | Entry-by-entry audit of the old bibliography |
| `new_refs.bib` | 260 verified sources not yet in `docs/references.bib` |
| `scripts/binding_free_baseline.py` | Reproduces the 0.476 ceiling for `AssociativeRecall` |
| `scripts/sa2005_replicate.py`, `sa2005_replication.json` | Re-implementation of the S&A 2005 pathfinding simulation (one graph, one seed per condition; provisional) |

The track reports are snapshots at `3799b36` and use the old paths (`results/evolve/…`).

## Conclusions that change how results should be read

1. **RV2.X1: the CPU recall task does not measure binding.** It admits a binding-free strategy scoring 0.476 at 6 pairs, and 0.423 / 0.319 / 0.243 at 8 / 16 / 32. Full attention (0.485), Titans (0.483–0.490) and `gdn_x8` sit on it. Report every recall number against this ceiling and against first-query accuracy.
2. **B12: in `gated_delta.py` the short convolutions are initialised to the identity.** Standard `gdn` addressing therefore starts with no token history. **W1b.a (P14) is confounded by this**: it lost the chain as well as the start. Rerun it with random or shift-biased initialisation.
3. **One-pair needle = passkey.** P8 shows one-pair recall ignores the query. P9 and P10 show a digit string carried 500 tokens, not binding. Use 2–3 pairs (as W1b does), then RULER multi-key and multi-value variants.
4. **P2 is still two hypotheses**: re-injection through later GDN layers, or adapter repair of the windowing shift. Run L2 (memory-free adapter, LoRA, state-offset tuning) and L4 (GDN state probes) before writing it up. P9 rules out adapter repair only for the all-windowed case.
5. **P11 and P13 are the missing-key problem from Samsonovich & Ascoli (2005) (RV2.F1).** The chain (their *J*) works. Round 2 changed only the write-side address, and every variant stayed at digit-1 loss ≈ ln(pairs). Selecting the start is a read-side operation: a dense cue computed from the query span, or a goal field (their CA3 → CA1 *W*). That is L6, the review's highest-value new experiment.
6. **Anticipated work to cite and benchmark against.** W1 ≈ δ-mem (arXiv:2605.12357). W3 sits beside Lee et al. 2026 (arXiv:2605.26099) and Behrouz et al. 2026 (arXiv:2606.03979). The graded controller sits beside AdaSteer and PID steering. The DG MoE is HiCL. The skip track needs value-residual, DenseFormer and hyper-connection baselines.
7. **Feedback.** Subtractive feedback is not a clean high-pass. Bode's integral forces amplification in some band (about 2.5× near a 46-token period at K = 1), which may explain T10 (L7).
8. **CA2 controller at defaults.** With λ = 0.9 and κ = 0.5, the gain reaches 0.875 of the cap after 3 tokens, with a time constant of about 10 tokens, and it resets per sequence (analytic, not run). Multi-turn escalation (A3) needs a turn-level integrator (L14).

## What this commit changed in shared files

| File | Change |
|---|---|
| `research/REGISTER.md` | Status: M1, M2, M6, T16 → Open; M4 → Refuted. Review notes appended to M3, M5, M7, T6, T7, T10–T14, P1, P2, P9–P11, P13, P14, W1b, O9, A2, A3, R2; W6.a and O7 superseded. Added **B12–B15**, the **RV2** findings table (§3) and backlog **L1–L16** (§4.8, with L3/L13/L14/L15 marked duplicates); ID-scheme rows |
| `research/NOTEBOOK.md` | New dated entry, "Literature review and code audit (2 Nov 2026)" |
| `docs/genesis.md` | §9 added; notes in §3 and §5.6; five rows added to the §7 corrections table |
| `docs/references.bib` | Replaced with the audited version (82 keys kept, DOIs added) plus 6 additions and 72 promoted prior-art entries: 160 entries |
| `REFERENCES.md` | 24 rows corrected; 6 audit additions placed in existing sections; five new "Prior art identified by the literature review" sections |
| `docs/src/bib_to_json.py`, `docs/src/references.json` | Editor fallback for edited volumes; JSON regenerated (18 groups, 160 entries). **`docs/hippocampal_ml_architectures.docx` was not rebuilt** (needs `docs/src/build_architectures_docx.js`) |
| `README.md`, `RESULTS.md`, `docs/design/temporal-loop.md` | Caveats on the recall and capacity claims; attribution of the GDN identity; corrected back-projection anatomy, the high-pass claim, the kernel's form, and the time-channel description |
| `limbic_llm/ca2.py`, `dentate_gyrus.py`, `temporal_loop.py`, `gated_delta.py` | Docstrings only (B13); no code changes |
| `research/README.md`, `research/HANDOFF.md` | L and RV2 added to the ID table; a pointer section in HANDOFF (any session may edit handoff documents) |

## Suggested next actions by owner

**GPU session (`feat/round1-side-memory`).**
1. W1b follow-ups: fix B12 and rerun W1b.a. Add a read-side arm, a first-lookup query computed from the query span (Treves–Rolls), and score 2 and 3 pairs.
2. L2 and L4 on Qwen3.5 before any write-up of P2.
3. L6: a missing-key k-hop needle (k = 1, 2, 3) with chained successor reads, then a goal-field matrix written at replay events.
4. Run W1 against δ-mem at equal state size.

**Coordinating session.**
1. Review and merge the PR (`docs/literature-review` → `feat/round1-side-memory`). It edits files this session owns (REGISTER, NOTEBOOK, RESULTS, docs, `limbic_llm/` docstrings) at Daniel's request; adjust as needed.
2. Rebuild the design docx.
3. Decide whether to replace `AssociativeRecall` with MQAR (L1) before more CPU work.

**Alignment and CoT work.** Run L13 (matched-norm controller comparison with a token-level probe) and the earlier review's A1 and C1 fixes before A6 and R2 at scale.

## Conventions for later agents

- To cite a source from `new_refs.bib`, move its entry into `docs/references.bib` **and** add a row to REFERENCES.md. `docs/src/bib_to_json.py` exits with an error if the two key sets differ.
- Any record from late 2026 (after the review date) must be verified again; do not reuse ids from memory.
- Register statuses changed here cite this folder. If a new run overturns one, update the register row and note it in NOTEBOOK.md, as usual.

## Limits of this review

- Venues of some arXiv-first ML papers were not re-checked, so `new_refs.bib` lists them as preprints.
- Several neuroscience abstracts were checked through publisher pages because PubMed was not reachable.
- Two candidates were dropped as unverifiable: HippoTune as an arXiv record, and DAMO.
- The Qwen3.5 MoE expert counts come from secondary sources.
- The S&A replication's M = 2 and noise sensitivities did not match the paper.
- Novelty verdicts mean "no precedent found in a verified search", not proof of absence.
