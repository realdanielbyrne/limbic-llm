"""Round 12: a training-free, dynamic-gain CA2 side channel (experiments/ca2).

    h_t ← h_t + g_t ‖h_t‖ d̂_t,   g_t = g_min + (g_max − g_min) r_t,   r_t ∈ [0, 1] from a relevance source,

with d̂ the stream's own gist (fast minus slow running average) or a matched-norm control direction.
Nothing is trained. Modules:

  bank.py        GistBank: bias-corrected running averages (default two bands = steering.py), sink skip,
                 previous-step averages, massive-token flags over the relevance window, massive_dims()
  regulators.py  leaky_scan, release events, the R6 repetition regulator (ngram / hidden), error-only
                 projection, the accumulated-dose cap
  sources.py     StepContext, the Source protocol, agree, and the controls constant, yoked, position_mean
  controller.py  GainPipeline, CA2Controller (run.generate / run.perplexity contract, per-token trace),
                 Tap (extension interface), NormTap, CA2Attachment(tap, inject_before, relevance_layer)
  common.py      loaders (reused), SteerSide for pretrained_loop.HookedLM, paired_boot, gain_stats,
                 TraceWriter (per-token fp16 arrays + per-item records outside the repo, hashed manifest)

Taps added by other modules (attention.py, gdn.py) plug in through ``controller.Tap``; see the
controller's module docstring, "Extension points". Import as ``from ca2.controller import ...`` with
``experiments/`` on ``sys.path`` (each module puts it there).
"""
