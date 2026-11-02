"""Language-model wrappers for both strategies plus a transformer baseline.

All models map tokens (B,T) → logits (B,T,V) and return an ``LMOutput`` with
the diagnostic signals (novelty, surprise) and the recurrent memory state so
long streams can be processed call-by-call.

* ``BaselineLM``          — causal transformer, optional sliding window.
* ``HippocampalLM``       — Strategy I. EC encoder → HippocampalBlock(s) →
                            neocortical layers, processed in segments; the
                            CA3 episodic store carries information across
                            segments. ``n_neocortex=0`` is standalone mode;
                            ``n_neocortex>0`` is context-augmentation mode.
* ``TitansHippocampalLM`` — Strategy II. A stack of TitansHippocampalLayer
                            (MAC / MAG / MAL with DG, CA1 and novelty-lr).
* ``GatedDeltaLM``        — Gated DeltaNet stack (linear CA3-style memory),
                            optionally hybrid with attention (Qwen3-Next 3:1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FeedForward, TransformerLayer
from .ca3 import EpisodicStore
from .gated_delta import GatedDeltaNet
from .hippocampal_block import HippocampalBlock
from .neural_memory import MemoryState
from .titans_hybrid import TitansHippocampalLayer


@dataclass
class LMOutput:
    logits: torch.Tensor
    novelty: torch.Tensor | None = None
    surprise: torch.Tensor | None = None
    state: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


class _Embed(nn.Module):
    def __init__(self, vocab: int, d_model: int, max_len: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, offset: int = 0) -> torch.Tensor:
        T = tokens.shape[1]
        pos = torch.arange(offset, offset + T, device=tokens.device) % self.pos.num_embeddings
        return self.tok(tokens) + self.pos(pos)


class _Head(nn.Module):
    def __init__(self, d_model: int, vocab: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


# --------------------------------------------------------------------- baseline
class BaselineLM(nn.Module):
    def __init__(self, vocab: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 window: int | None = None, max_len: int = 1024, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed = _Embed(vocab, d_model, max_len)
        self.layers = nn.ModuleList(TransformerLayer(d_model, n_heads, window, dropout) for _ in range(n_layers))
        self.head = _Head(d_model, vocab)

    def forward(self, tokens: torch.Tensor, state=None) -> LMOutput:
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        return LMOutput(self.head(x))


# ---------------------------------------------------------------- gated deltanet
class _GDNBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, **gdn_kwargs) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = GatedDeltaNet(d_model, n_heads, **gdn_kwargs)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, state=None):
        y, state, err = self.mixer(self.norm(x), state)
        x = x + y
        return x + self.ff(x), state, err


class GatedDeltaLM(nn.Module):
    """Stack of Gated DeltaNet blocks, optionally hybrid with attention.

    ``attn_every=N`` makes every N-th layer a (windowed) attention layer, e.g.
    ``attn_every=4`` gives the 3:1 DeltaNet:attention mix of Qwen3-Next.
    ``addressing`` is "gdn" | "tied" | "dg" (see ``gated_delta.py``).
    The recurrent state is returned and can be passed back to stream.
    """

    def __init__(self, vocab: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 attn_every: int | None = None, window: int | None = None, max_len: int = 1024,
                 dropout: float = 0.0, **gdn_kwargs) -> None:
        super().__init__()
        self.embed = _Embed(vocab, d_model, max_len)
        self.layers = nn.ModuleList(
            TransformerLayer(d_model, n_heads, window, dropout)
            if (attn_every and (i + 1) % attn_every == 0)
            else _GDNBlock(d_model, n_heads, dropout, **gdn_kwargs)
            for i in range(n_layers)
        )
        self.head = _Head(d_model, vocab)

    def forward(self, tokens: torch.Tensor, state: list | None = None, offset: int = 0) -> LMOutput:
        x = self.embed(tokens, offset)
        states = list(state) if state else [None] * len(self.layers)
        errs = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, _GDNBlock):
                x, states[i], err = layer(x, states[i])
                errs.append(err)
            else:
                x = layer(x)
        surprise = torch.stack(errs).mean(0) if errs else None
        return LMOutput(self.head(x), None, surprise,
                        [s.detach() if s is not None else None for s in states])


# ------------------------------------------------------------------- strategy I
class HippocampalLM(nn.Module):
    def __init__(
        self,
        vocab: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_ec: int = 1,
        n_hpc: int = 1,
        n_neocortex: int = 1,
        segment: int = 32,
        max_len: int = 1024,
        dropout: float = 0.0,
        **block_kwargs,
    ) -> None:
        super().__init__()
        self.segment = segment
        self.embed = _Embed(vocab, d_model, max_len)
        self.ec = nn.ModuleList(TransformerLayer(d_model, n_heads, dropout=dropout) for _ in range(n_ec))
        self.hpc = nn.ModuleList(HippocampalBlock(d_model, n_heads, dropout=dropout, **block_kwargs)
                                 for _ in range(n_hpc))
        self.neocortex = nn.ModuleList(TransformerLayer(d_model, n_heads, dropout=dropout)
                                       for _ in range(n_neocortex))
        self.head = _Head(d_model, vocab)

    @property
    def dentate_gyri(self):
        return [b.dg for b in self.hpc]

    def forward(self, tokens: torch.Tensor, state: list[EpisodicStore] | None = None,
                context: torch.Tensor | None = None, offset: int = 0) -> LMOutput:
        stores = list(state) if state else [None] * len(self.hpc)
        logits, novs, codes = [], [], []
        for s in range(0, tokens.shape[1], self.segment):
            x = self.embed(tokens[:, s:s + self.segment], offset + s)
            for layer in self.ec:                      # EC: contextualised embedding
                x = layer(x)
            seg_nov = []
            for i, block in enumerate(self.hpc):       # trisynaptic circuit
                out = block(x, stores[i], context)
                x, stores[i] = out.hidden, out.store
                seg_nov.append(out.novelty)
                codes.append(out.dg_code)
            for layer in self.neocortex:               # downstream cortex
                x = layer(x)
            logits.append(self.head(x))
            novs.append(torch.stack(seg_nov).mean(0))
        stores = [st.detach() for st in stores]        # truncate BPTT between calls
        return LMOutput(torch.cat(logits, 1), torch.cat(novs, 1), None, stores,
                        {"dg_codes": codes})


# ------------------------------------------------------------------ strategy II
class TitansHippocampalLM(nn.Module):
    def __init__(self, vocab: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 max_len: int = 1024, **layer_kwargs) -> None:
        super().__init__()
        self.embed = _Embed(vocab, d_model, max_len)
        self.layers = nn.ModuleList(TitansHippocampalLayer(d_model, n_heads, **layer_kwargs)
                                    for _ in range(n_layers))
        self.head = _Head(d_model, vocab)

    @property
    def dentate_gyri(self):
        return [l.memory.dg for l in self.layers if getattr(l.memory, "dg", None) is not None]

    def forward(self, tokens: torch.Tensor, state: list[MemoryState] | None = None,
                offset: int = 0) -> LMOutput:
        x = self.embed(tokens, offset)
        states = list(state) if state else [None] * len(self.layers)
        novs, surps = [], []
        for i, layer in enumerate(self.layers):
            out = layer(x, states[i])
            x, states[i] = out.hidden, out.state
            surps.append(out.surprise)
            if out.novelty is not None:
                novs.append(out.novelty)
        novelty = torch.stack(novs).mean(0) if novs else None
        return LMOutput(self.head(x), novelty, torch.stack(surps).mean(0),
                        [s.detach() for s in states])


# ----------------------------------------------------------------------- losses
def lm_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None,
            reduction: str = "mean") -> torch.Tensor:
    """Cross-entropy; ``mask`` (B,T) selects positions that count."""
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(targets)
    if reduction == "none":
        return loss
    if mask is None:
        return loss.mean()
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def novelty_alignment_loss(novelty: torch.Tensor, token_loss: torch.Tensor) -> torch.Tensor:
    """Teach the CA1 comparator to predict surprise.

    Target n*_t = σ( (ℓ_t − μ) / σ_ℓ ), the z-scored (detached) per-token
    prediction error. Novelty then ranks tokens by how badly the model
    predicts them, which is what the CA3 write policy and replay priority use.

    ``novelty[t]`` is aligned to the input at position t, whose surprise is
    the loss of predicting token t from position t−1.
    """
    tl = token_loss.detach()
    z = (tl - tl.mean()) / (tl.std() + 1e-6)
    target = torch.sigmoid(z)
    # novelty at input t ↔ loss of predicting input t (made at t−1)
    return F.mse_loss(novelty[:, 1:], target[:, :-1])
