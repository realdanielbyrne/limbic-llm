"""Multi-tap hooks for round-10 candidates: ``TapLM``, a ``pretrained_loop.HookedLM`` that also reads early blocks.

``HookedLM`` hands the side channels one hidden state, the output h of block ``--layer``. W1s
(``w1s_slots.py``) also reads lower blocks: its gate and lexical keys work on block 3, where a
token still looks like itself, while the stored values come from the tap at block 21. A side that
declares ``taps`` (block indices below the main tap) gets them here:

  taps          A read-only forward hook on each tap block caches ``h_i[:, sink:]`` in float32 while
                the side is on. ``_tap`` sets ``side.extra = {i: h_i}`` (same T as h) before calling
                the side, and ``__call__`` clears the caches in a ``finally``. With the side off
                nothing is cached and the masks are the base ones, so the windowed control is
                byte-identical to ``HookedLM``.
  reinstatement A side with ``out_mode == 'reinstate'`` does not inject into h. It returns
                ``reinstate_bias(extra)`` (B, T, T): rows are queries t, columns positions c, both in
                side coordinates, and the entry is log Σ_k a_{t,k} 1[c ∈ window_k], or −1e4 where
                nothing is reinstated. Once the highest tap has fired, the bias is made causal
                (c ≤ t), padded to the sink (−1e4), and every attention layer at index ≥
                ``side.reinstate_L`` uses max(window mask, bias) as its mask: the query rows see the
                reinstated columns' keys and values, which were computed in their own window below L.
                Layers below L keep the window mask. The bias must come from taps < L, since it is
                needed before layer L runs. It applies only while the window is active (the full
                condition needs none). Its gradient reaches the side only through the attention
                mask, which eager attention differentiates; check before relying on sdpa.

Segment mode folds the batch, so the tap caches would not line up with h; it is rejected.
``--grad-checkpointing`` is reimplemented so that the recompute in backward sees the same bias as
the forward pass (it is passed to the checkpointed block as an input).

``run.py`` swaps ``pl.HookedLM`` for ``TapLM`` when the candidate module sets ``TAPS_REQUIRED``;
``eval_by_query.py`` builds it when the rebuilt side has ``.taps``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402

NEG = -1e4                                                        # "not reinstated": exp underflows to 0


class TapLM(pl.HookedLM):
    """``HookedLM`` plus read-only taps on lower blocks and optional attention reinstatement.

    Same constructor as ``HookedLM``. A side without ``taps`` (or ``side=None``) gets exactly
    ``HookedLM``.
    """

    def __init__(self, model, side, layer=None, window=64, mode="sliding", window_layers="all", sink=1,
                 pad_id=0, grad_checkpointing=False) -> None:
        super().__init__(model, side, layer, window, mode, window_layers, sink, pad_id, grad_checkpointing=False)
        self.taps: tuple[int, ...] = tuple(sorted({int(i) for i in getattr(side, "taps", ())}))
        self._cache: dict[int, torch.Tensor] = {}
        self._rbias: torch.Tensor | None = None                   # (B, 1, T, T) float32, sink coordinates
        if self.taps:
            if mode == "segment":
                raise ValueError("taps need the unfolded stream; segment window mode is not supported")
            if not all(0 <= i < self.layer for i in self.taps):
                raise ValueError(f"taps {self.taps} must be block indices below the main tap {self.layer}")
            if getattr(side, "out_mode", None) == "reinstate" and self.taps[-1] >= side.reinstate_L:
                raise ValueError(f"reinstatement at layers >= {side.reinstate_L} needs taps below it, "
                                 f"got {self.taps}")
            for i in self.taps:
                self.handles.append(self.blocks[i].register_forward_hook(self._cache_hook(i)))
            for i in self.windowed:                               # after the base mask hooks: they run first
                if self.kinds[i] == "attention":
                    m = pl._child(self.blocks[i], pl.ATTN_MIXERS)
                    self.handles.append(m.register_forward_pre_hook(self._reinstate_hook(i), with_kwargs=True))
        if grad_checkpointing:                                    # frozen upper half: recompute in backward
            for b in self.blocks[self.layer + 1:]:
                b.forward = self._checkpointed(b.forward)

    # ---------------------------------------------------------------- hooks
    def _reinstating(self) -> bool:
        return self.side_on and self._active() and getattr(self.side, "out_mode", None) == "reinstate"

    def _cache_hook(self, i: int):
        def hook(mod, args, out):
            if not self.side_on:
                return None
            h = out[0] if isinstance(out, tuple) else out
            self._cache[i] = h[:, self.sink:].float()
            if i == self.taps[-1] and self._reinstating():        # the last early tap: the bias is computable
                self._rbias = self._pad_bias(self.side.reinstate_bias(self._cache))
            return None
        return hook

    def _pad_bias(self, rb: torch.Tensor) -> torch.Tensor:
        """(B, T', T') side coordinates → (B, 1, T, T): future columns and the sink row/column at NEG."""
        L = rb.shape[-1]
        future = torch.ones(L, L, dtype=torch.bool, device=rb.device).triu(1)
        rb = rb.float().masked_fill(future, NEG)
        s = self.sink
        return F.pad(rb, (s, 0, s, 0), value=NEG)[:, None]

    def _reinstate_hook(self, i: int):
        def hook(mod, args, kwargs):
            rb = self._rbias
            if rb is None or i < self.side.reinstate_L or not self._active():
                return None
            m = kwargs["attention_mask"]                          # the window mask set by _mask_hook
            kwargs["attention_mask"] = torch.maximum(m, rb.to(m.device, m.dtype))
            return args, kwargs
        return hook

    def _tap(self, mod, args, out):
        if self.taps and self.side_on:
            self.side.extra = self._cache
        return super()._tap(mod, args, out)

    def _checkpointed(self, fn):
        """Checkpoint a block with the reinstatement bias as an input, so the recompute sees it."""
        def inner(rb, *args, **kwargs):
            old, self._rbias = self._rbias, rb
            try:
                return fn(*args, **kwargs)
            finally:
                self._rbias = old

        def run(*args, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(inner, self._rbias, *args, use_reentrant=False, **kwargs)
            return fn(*args, **kwargs)
        return run

    # -------------------------------------------------------------- forward
    def __call__(self, ids: torch.Tensor, keep: int | None = None) -> torch.Tensor:
        try:
            return super().__call__(ids, keep)
        finally:
            self._cache, self._rbias = {}, None
            if self.taps:
                self.side.extra = None
