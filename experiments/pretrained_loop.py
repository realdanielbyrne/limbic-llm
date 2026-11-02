"""Pretrained loop: hippocampal side channels on a frozen Hugging Face causal LM.

The pretrained analogue of Activation Addition (docs/notebooks/activation_additions_gpt2xl.ipynb).
That notebook adds a fixed steering vector to the residual stream of a middle GPT-2-XL block. Here a
forward hook on block ``--layer`` (default n_layers // 2) replaces that block's output h with

    h ← HippocampalLoop(TimeChannel(h))

so the added vector is computed on the fly: a contrastive, damped, gain-controlled read-out of an
associative memory of the model's own earlier activations (dynamic activation addition). The base
model is frozen and only the side-channel parameters are trained (docs/design/temporal-loop.md §4,
training plan step 1: truncate the context so memory is the only route to distant information).

Windowed attention
------------------
``--window W`` restricts every windowed token mixer to W tokens; ``--window 0`` is full attention.
Mechanism: a forward pre-hook on the attention submodule of each windowed block (``attn`` in GPT-2,
``self_attn`` in Llama/Qwen) overwrites its ``attention_mask`` keyword with an additive 4D mask
(1, 1, T, T), 0 where allowed and finfo.min elsewhere. This bypasses the model-level mask creation,
whose handling of custom 4D masks differs across transformers versions (4.x GPT-2 flattened them;
5.x passes them through), and lets a subset of layers be windowed. It works with eager and sdpa
attention; flash-attention kernels ignore dense masks, so models load with ``--attn-impl sdpa`` and
``check_window`` verifies at start-up, on the first windowed mixer, that the mask is honoured.

  sliding  position t attends to [t − W + 1, t]. Information can hop W − 1 tokens per
           layer, so the strict reach of the windowed base is L·(W − 1) for L windowed layers
           (756 tokens for GPT-2 small at W = 64), not W. The script prints it and warns when a
           needle gap is shorter.
  block    (default) non-overlapping blocks of W tokens. Nothing crosses a block boundary except through the
           side channels, whatever the depth; the strict version of "memory is the only route".
  segment  block semantics by folding the batch: each W-token segment runs as its own sequence
           (positions restart, each segment gets its own sink token), and the tap hook unfolds the
           stream so the side channels see the whole sequence. Fallback for attention backends
           that ignore 4D masks.

A fixed BOS token at position 0 stays visible to every query (``--sink``, default on): windowed
pretrained models degrade badly once their attention sink leaves the window (StreamingLLM). It only
attends to itself, so it carries no sequence content, and the side channels skip it.

Hybrid models (Qwen3-Next / Qwen3.5: 3 Gated DeltaNet layers per full-attention layer)
--------------------------------------------------------------------------------------
Layer types come from ``config.layer_types`` (via ``get_text_config()``): entries containing
"linear" are linear-attention layers, others ("full_attention", "sliding_attention") attention
layers. Without that field, a block with a ``linear_attn`` / ``mamba`` / ``mixer`` child counts as
linear. ``--window-layers attention`` windows only the attention layers and leaves GDN layers
untouched (they carry their own unbounded memory). ``--window-layers all`` also runs each GDN layer
on W-token blocks (its state resets every block, by folding), since a recurrence has no sliding
window. For GPT-2, Llama and Qwen3 both settings are the same. Decoder blocks are found as the
shallowest ``layers`` / ``h`` / ``blocks`` ModuleList outside vision/audio towers, which covers
``transformer.h`` (GPT-2), ``model.layers`` (Llama/Qwen3) and ``model.language_model.layers``
(multimodal Qwen3.5). Qwen3.5 checkpoints load text-only through AutoModelForCausalLM.

Tasks
-----
  needle   "The code for <name> is <4 digits>." (× --pairs), filler, then "The code for <name> is"
           and the digits are scored (exact match of all value tokens, teacher forced). Filler is
           a slice of --text-file, or random digit-free tokens from the vocabulary. The sequence
           is --seq-len tokens, so the gap is about seq-len − 20.
  lm       next-token loss on --text-file (last 10% held out) or a HF dataset (--dataset
           wikitext-103, needs `datasets`); perplexity of predictions made at positions ≥ W.
Both train and score only those positions. Three conditions are evaluated on the same data:
windowed base, windowed base + side channels, full-attention base (upper bound).

Usage
  OMP_NUM_THREADS=1 python experiments/pretrained_loop.py --tiny --task needle --steps 20 \\
      --seq-len 128 --window 16 --batch 4 --eval-n 16
  python experiments/pretrained_loop.py --model gpt2 --task needle --window 64 --window-mode block \\
      --seq-len 512 --steps 1000 --batch 8
  python experiments/pretrained_loop.py --model Qwen/Qwen3.5-2B --dtype bfloat16 --device auto \\
      --task lm --dataset wikitext-103 --window 256 --window-layers attention --grad-checkpointing
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config  # noqa: E402

from limbic_llm.temporal_loop import HippocampalLoop, LoopStats, TimeChannel  # noqa: E402

ATTN_MIXERS = ("self_attn", "attn", "attention", "self_attention")
LINEAR_MIXERS = ("linear_attn", "mamba", "mixer", "linear_attention")
BLOCK_LISTS = ("layers", "h", "blocks")
NOT_DECODER = ("vision", "visual", "audio", "encoder", "mtp", "vit")
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
DATASETS = {"wikitext-103": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
            "wikitext-2": ("Salesforce/wikitext", "wikitext-2-raw-v1")}
NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack", "Karen",
         "Liam", "Maria", "Noah", "Olivia", "Peter", "Quinn", "Rosa", "Sam", "Tara", "Victor", "Wendy",
         "Xavier", "Yara", "Zoe"]


# --------------------------------------------------------------- model discovery
def _child(block: nn.Module, names: tuple[str, ...]) -> nn.Module | None:
    for n in names:
        m = getattr(block, n, None)
        if isinstance(m, nn.Module):
            return m
    return None


def find_blocks(model: nn.Module) -> nn.ModuleList:
    """The decoder blocks: transformer.h (GPT-2), model.layers (Llama/Qwen3), model.language_model.layers."""
    lists = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.ModuleList) and len(m) and not any(s in n.lower() for s in NOT_DECODER)]
    named = [(n, m) for n, m in lists if n.split(".")[-1] in BLOCK_LISTS]
    if named:                                                     # shallowest, then longest (not MoE experts)
        return min(named, key=lambda nm: (nm[0].count("."), -len(nm[1])))[1]
    if lists:
        return max(lists, key=lambda nm: len(nm[1]))[1]
    raise ValueError("no ModuleList of decoder blocks found")


def text_config(model: nn.Module):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def layer_kinds(model: nn.Module, blocks: nn.ModuleList) -> list[str]:
    """'attention' or 'linear' per block, from config.layer_types, else from the block's children."""
    types = getattr(text_config(model), "layer_types", None)
    if types is not None and len(types) == len(blocks):
        return ["linear" if ("linear" in t or t in ("mamba", "ssm", "recurrent")) else "attention" for t in types]
    return ["linear" if _child(b, LINEAR_MIXERS) is not None else "attention" for b in blocks]


# ------------------------------------------------------------------ window tools
def window_mask(T: int, W: int, mode: str, sink: int, device, dtype) -> torch.Tensor:
    """Additive causal mask (1, 1, T, T): sliding window or W-blocks aligned to the first real token."""
    i = torch.arange(T, device=device)
    q, k = i[:, None], i[None, :]
    ok = k <= q
    if mode == "sliding":
        ok = ok & (q - k < W)
    else:
        ok = ok & ((q - sink).div(W, rounding_mode="floor") == (k - sink).div(W, rounding_mode="floor"))
    if sink:
        ok[:, :sink] = True                                       # attention sink stays visible
    mask = torch.zeros(T, T, device=device, dtype=dtype).masked_fill(~ok, torch.finfo(dtype).min)
    return mask[None, None]


def fold(x: torch.Tensor, W: int, sink: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """(B, T, ...) → (B·(n + [sink]), W, ...): W-token blocks after the sink; the sink is its own block."""
    B, T = x.shape[:2]
    L = T - sink
    n = -(-L // W)
    body = x[:, sink:]
    if n * W > L:
        body = torch.cat([body, body.new_zeros((B, n * W - L) + body.shape[2:])], 1)
    parts = body.reshape((B, n, W) + x.shape[2:])
    if sink:
        head = torch.cat([x[:, :sink], x.new_zeros((B, W - sink) + x.shape[2:])], 1)[:, None]
        parts = torch.cat([head, parts], 1)
    return parts.flatten(0, 1), (B, T)


def unfold(y: torch.Tensor, W: int, sink: int, shape: tuple[int, int]) -> torch.Tensor:
    B, T = shape
    y = y.reshape((B, -1, W) + y.shape[2:])
    body = y[:, int(sink > 0):].flatten(1, 2)[:, : T - sink]
    return torch.cat([y[:, 0, :sink], body], 1) if sink else body


def _checkpointed(fn):
    def run(*args, **kwargs):
        if torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False, **kwargs)
        return fn(*args, **kwargs)
    return run


# ----------------------------------------------------------------- side channels
class SideChannels(nn.Module):
    """TimeChannel then HippocampalLoop, either optional; identity when both are off."""

    def __init__(self, d_model: int, memory: bool = True, time: bool = True,
                 time_kwargs: dict | None = None, **loop_kwargs) -> None:
        super().__init__()
        self.time = TimeChannel(d_model, **(time_kwargs or {})) if time else None
        self.loop = HippocampalLoop(d_model, **loop_kwargs) if memory else None

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, LoopStats | None]:
        stats = None
        if self.time is not None:
            h, _ = self.time(h)
        if self.loop is not None:
            h, stats = self.loop(h)
        return h, stats


class HookedLM:
    """Frozen causal LM with side channels at one block and windowed token mixers, all via hooks.

    ``window`` and ``side_on`` are read at call time, so one model serves every condition
    (see ``condition``).
    """

    def __init__(self, model: nn.Module, side: SideChannels, layer: int | None = None, window: int = 64,
                 mode: str = "sliding", window_layers: str = "all", sink: int = 1, pad_id: int = 0,
                 grad_checkpointing: bool = False) -> None:
        if mode not in {"sliding", "block", "segment"} or window_layers not in {"all", "attention"}:
            raise ValueError((mode, window_layers))
        self.model, self.side = model, side
        self.window, self.mode, self.window_layers, self.sink, self.pad_id = window, mode, window_layers, sink, pad_id
        self.side_on = True
        self.blocks = find_blocks(model)
        self.kinds = layer_kinds(model, self.blocks)
        self.layer = (len(self.blocks) // 2 if layer is None else layer) % len(self.blocks)
        self.last_stats: LoopStats | None = None
        self._seg: tuple[int, int, int] | None = None
        self._masks: dict = {}
        self._keep_ok = "logits_to_keep" in inspect.signature(model.forward).parameters
        self.input_device = model.get_input_embeddings().weight.device
        self.handles = [self.blocks[self.layer].register_forward_hook(self._tap)]
        self.windowed: list[int] = []
        for i, (b, kind) in enumerate(zip(self.blocks, self.kinds)):
            if kind == "attention":
                m = _child(b, ATTN_MIXERS)
                if m is None:
                    raise ValueError(f"block {i} has no attention submodule among {ATTN_MIXERS}")
                self.handles.append(m.register_forward_pre_hook(self._mask_hook, with_kwargs=True))
                self.windowed.append(i)
            elif window_layers == "all":
                m = _child(b, LINEAR_MIXERS)
                self.handles.append(m.register_forward_pre_hook(self._fold_pre, with_kwargs=True))
                self.handles.append(m.register_forward_hook(self._fold_post))
                self.windowed.append(i)
        if grad_checkpointing:                                    # frozen upper half: recompute in backward
            for b in self.blocks[self.layer + 1:]:
                b.forward = _checkpointed(b.forward)

    # ---------------------------------------------------------------- hooks
    def _active(self) -> bool:
        return bool(self.window) and self.mode != "segment"

    def _mask_hook(self, mod, args, kwargs):
        if not self._active():
            return None
        h = args[0] if args else kwargs["hidden_states"]
        key = (h.shape[1], h.device, h.dtype, self.window, self.mode, self.sink)
        if key not in self._masks:
            self._masks[key] = window_mask(h.shape[1], self.window, self.mode, self.sink, h.device, h.dtype)
        kwargs["attention_mask"] = self._masks[key]
        return args, kwargs

    def _fold_pre(self, mod, args, kwargs):
        mod._limbic_fold = None
        if not self._active():
            return None
        h = args[0] if args else kwargs.pop("hidden_states")
        x, mod._limbic_fold = fold(h, self.window, self.sink)
        kwargs["attention_mask"] = None                           # padding mask of the unfolded batch; none here
        return (x,) + tuple(args[1:]), kwargs

    def _fold_post(self, mod, args, out):
        shape = getattr(mod, "_limbic_fold", None)
        if shape is None:
            return None
        if isinstance(out, tuple):
            return (unfold(out[0], self.window, self.sink, shape),) + tuple(out[1:])
        return unfold(out, self.window, self.sink, shape)

    def _tap(self, mod, args, out):
        if not self.side_on:
            return None
        h = out[0] if isinstance(out, tuple) else out
        s, dt = self.sink, h.dtype
        # side channels keep float32 weights and run in float32: with bf16 weights, Adam steps
        # of ~lr are below bf16's spacing near 1 (2^-7), so updates such as the loop gain round away
        if self._seg is not None:                                 # (B·n, s + W, D) → (B, L, D) and back
            B, n, L = self._seg
            W, D = self.window, h.shape[-1]
            body, stats = self.side(h[:, s:].reshape(B, n * W, D)[:, :L].float())
            body = F.pad(body.to(dt), (0, 0, 0, n * W - L)).reshape(B * n, W, D)
        else:
            body, stats = self.side(h[:, s:].float())
            body = body.to(dt)
        self.last_stats = stats
        h = torch.cat([h[:, :s], body], 1)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    # -------------------------------------------------------------- forward
    @contextmanager
    def condition(self, window: int, side: bool):
        old = self.window, self.side_on
        self.window, self.side_on = window, side
        try:
            yield
        finally:
            self.window, self.side_on = old

    def __call__(self, ids: torch.Tensor, keep: int | None = None) -> torch.Tensor:
        """Logits (B, T, V), or for the last ``keep`` positions only."""
        ids = ids.to(self.input_device)
        self.last_stats = None
        if self.window and self.mode == "segment":
            B, T = ids.shape
            s, W = self.sink, self.window
            L = T - s
            n = -(-L // W)
            segs = F.pad(ids[:, s:], (0, n * W - L), value=self.pad_id).reshape(B * n, W)
            if s:
                segs = torch.cat([ids[:, :s].repeat_interleave(n, 0), segs], 1)
            self._seg = (B, n, L)
            try:
                out = self.model(input_ids=segs, use_cache=False).logits
            finally:
                self._seg = None
            lg = out[:, s:].reshape(B, n * W, -1)[:, :L]
            if s:
                lg = torch.cat([out.view(B, n, W + s, -1)[:, 0, :s], lg], 1)
            return lg[:, -keep:] if keep else lg
        kw = {"logits_to_keep": keep} if keep and self._keep_ok else {}
        lg = self.model(input_ids=ids, use_cache=False, **kw).logits
        return lg[:, -keep:] if keep else lg

    def reach(self) -> float:
        """Strict look-back of the windowed base without side channels (inf: unbounded)."""
        if not self.window or len(self.windowed) < len(self.blocks):
            return math.inf
        W = self.window
        return len(self.windowed) * (W - 1) if self.mode == "sliding" else W - 1


@torch.no_grad()
def check_window(lm: HookedLM) -> str:
    """Feed a controlled input to the first windowed mixer and check that a perturbation of the first
    real position does not reach positions ≥ W later (catches backends that ignore the mask)."""
    if not lm._active() or not lm.windowed:
        return "skipped"
    i = lm.windowed[0]
    mixer = _child(lm.blocks[i], ATTN_MIXERS if lm.kinds[i] == "attention" else LINEAR_MIXERS)
    s, W = lm.sink, lm.window
    T = s + 2 * W
    p = next(mixer.parameters())
    X = torch.randn(1, T, text_config(lm.model).hidden_size, device=p.device, dtype=p.dtype)
    X2 = X.clone()
    X2[:, s] += 1.0
    cur, outs = [X], []

    def pre(mod, args, kwargs):
        if args:
            return (cur[0],) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = cur[0]
        return args, kwargs

    def post(mod, args, out):
        outs.append((out[0] if isinstance(out, tuple) else out).float())

    hs = [mixer.register_forward_pre_hook(pre, with_kwargs=True, prepend=True), mixer.register_forward_hook(post)]
    try:
        with lm.condition(lm.window, False):
            for x in (X, X2):
                cur[0] = x
                lm(torch.full((1, T), lm.pad_id, dtype=torch.long), keep=1)
    finally:
        for h in hs:
            h.remove()
    d = (outs[0] - outs[1]).abs()[0].amax(-1)
    near, far = d[s:s + W].max().item(), d[s + W:].max().item()
    if not (near > 0 and far <= 1e-3 * near):
        raise RuntimeError(f"window mask not honoured at block {i} (near {near:.2e}, far {far:.2e}); "
                           "use --attn-impl sdpa/eager or --window-mode segment")
    return f"ok (block {i}: far {far:.1e} vs near {near:.1e})"


# -------------------------------------------------------------------- tokenizers
class ByteTokenizer:
    """Stand-in for --tiny (no downloads): UTF-8 bytes are tokens 0–255, 256 is BOS/EOS."""

    vocab_size = 257
    bos_token_id = eos_token_id = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids if i < 256).decode("utf-8", "replace")

    def filler_pool(self) -> list[int]:
        return [i for i in range(32, 127) if not chr(i).isdigit()]


class HFTokenizer:
    def __init__(self, tok) -> None:
        self.tok = tok
        tok.model_max_length = 10 ** 12                           # we cut sequences ourselves
        self.bos_token_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        self.eos_token_id = tok.eos_token_id if tok.eos_token_id is not None else self.bos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids))

    def filler_pool(self) -> list[int]:
        special = set(self.tok.all_special_ids)
        pieces = self.tok.convert_ids_to_tokens(list(range(self.tok.vocab_size)))
        return [i for i, p in enumerate(pieces) if i not in special and p and not p.startswith("<|")
                and not any(c.isdigit() for c in p)]


# ------------------------------------------------------------------------- tasks
class NeedleTask:
    """Key → value recall across filler; the value tokens of the query's answer are scored."""

    def __init__(self, tok, seq_len: int, pairs: int = 1, sink: int = 1, corpus: list[int] | None = None) -> None:
        if corpus:                                                # repeat a short corpus to cover one sequence
            corpus = corpus * (seq_len // len(corpus) + 2)
        self.tok, self.seq_len, self.pairs, self.sink, self.corpus = tok, seq_len, pairs, sink, corpus
        self.pool = None if corpus else tok.filler_pool()
        self.bos = tok.bos_token_id

    def sample(self, rng: random.Random) -> tuple[list[int], int, int]:
        enc = self.tok.encode
        names = rng.sample(NAMES, self.pairs)
        values = ["".join(rng.choice("0123456789") for _ in range(4)) for _ in names]
        facts, value_at = [], []
        for name, v in zip(names, values):
            head = enc(f" The code for {name} is")
            value_at.append(len(facts) + len(head))
            facts += head + enc(f" {v}") + enc(".")
        j = rng.randrange(self.pairs)
        query, answer = enc(f" The code for {names[j]} is"), enc(f" {values[j]}")
        n_fill = self.seq_len - len(facts) - len(query) - len(answer)
        if n_fill < 1:
            raise ValueError(f"--seq-len {self.seq_len} too short for {self.pairs} pairs")
        if self.corpus:
            o = rng.randrange(len(self.corpus) - n_fill)
            filler = self.corpus[o:o + n_fill]
        else:
            filler = [self.pool[rng.randrange(len(self.pool))] for _ in range(n_fill)]
        ids = [self.bos] * self.sink + facts + filler + query + answer
        gap = (len(ids) - len(answer)) - (self.sink + value_at[j])   # stored value → its recall
        return ids, len(answer), gap

    def batch(self, B: int, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        rows = [self.sample(rng) for _ in range(B)]
        ids = torch.tensor([r[0] for r in rows])
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for b, (_, n_ans, _) in enumerate(rows):
            mask[b, -n_ans:] = True
        return ids, mask, [r[2] for r in rows]


class TextTask:
    """Next-token prediction; targets predicted from positions ≥ W are scored."""

    def __init__(self, train_ids: torch.Tensor, eval_ids: torch.Tensor, seq_len: int, sink: int, bos: int,
                 window: int) -> None:
        self.train_ids, self.eval_ids, self.seq_len, self.sink, self.bos = train_ids, eval_ids, seq_len, sink, bos
        self.first = sink + window + 1 if window else sink + 1
        for name, t in (("train", train_ids), ("eval", eval_ids)):
            if len(t) <= seq_len:
                raise ValueError(f"{name} text has {len(t)} tokens; need more than --seq-len {seq_len}")

    def _wrap(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sink:
            rows = torch.cat([torch.full((len(rows), self.sink), self.bos), rows], 1)
        mask = torch.zeros_like(rows, dtype=torch.bool)
        mask[:, self.first:] = True
        return rows, mask

    def batch(self, B: int, rng: random.Random):
        o = [rng.randrange(len(self.train_ids) - self.seq_len) for _ in range(B)]
        return (*self._wrap(torch.stack([self.train_ids[i:i + self.seq_len] for i in o])), None)

    def eval_batches(self, n: int, B: int) -> list:
        starts = list(range(0, len(self.eval_ids) - self.seq_len, self.seq_len))[:n]
        rows = [self.eval_ids[i:i + self.seq_len] for i in starts]
        return [(*self._wrap(torch.stack(rows[i:i + B])), None) for i in range(0, len(rows), B)]


def load_text(args, tok) -> tuple[list[int], list[int]]:
    if args.text_file:
        ids = tok.encode(Path(args.text_file).read_text(encoding="utf-8", errors="replace"))
        cut = int(len(ids) * 0.9)
        return ids[:cut], ids[cut:]
    if args.dataset:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise SystemExit("--dataset needs `pip install datasets`") from e
        name, cfg = DATASETS.get(args.dataset) or (args.dataset.split(":", 1) + [None])[:2]
        out = []
        for split, budget in (("train", args.max_chars), ("validation", args.max_chars // 10)):
            parts, n = [], 0
            for row in load_dataset(name, cfg, split=split, streaming=True):
                parts.append(row["text"])
                n += len(row["text"])
                if n >= budget:
                    break
            out.append(tok.encode("".join(parts)))
        return out[0], out[1]
    raise SystemExit("--task lm needs --text-file or --dataset")


# ------------------------------------------------------------------------ models
def build_tiny(arch: str = "gpt2", seed: int = 0, attn_impl: str = "sdpa"):
    """Random small model + byte tokenizer, for offline smoke tests."""
    torch.manual_seed(seed)
    if arch == "gpt2":
        cfg = GPT2Config(n_layer=4, n_embd=64, n_head=4, vocab_size=512, n_positions=1024,
                         bos_token_id=256, eos_token_id=256)
    elif arch == "qwen35":
        if not hasattr(transformers, "Qwen3_5TextConfig"):
            raise SystemExit(f"transformers {transformers.__version__} has no Qwen3.5 classes")
        cfg = transformers.Qwen3_5TextConfig(
            vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=16, linear_value_head_dim=16, max_position_embeddings=1024,
            layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
            bos_token_id=256, eos_token_id=256, pad_token_id=256)
    else:
        raise ValueError(arch)
    return AutoModelForCausalLM.from_config(cfg, attn_implementation=attn_impl), ByteTokenizer()


def load_model(args):
    dtype = DTYPES[args.dtype]
    cuda = torch.cuda.is_available()
    if args.tiny:
        model, tok = build_tiny(args.tiny_arch, args.seed, args.attn_impl)
        dev = "cuda" if args.device in ("cuda", "auto") and cuda else "cpu"
        model = model.to(dev, dtype)
    else:
        kw = dict(dtype=dtype, attn_implementation=args.attn_impl)
        if args.device == "auto" and cuda:
            kw["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
        if "device_map" not in kw:
            model = model.to("cuda" if args.device == "cuda" else "cpu")
        tok = HFTokenizer(AutoTokenizer.from_pretrained(args.model))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def loop_kwargs(args) -> dict:
    kw = dict(d_key=args.d_key, tau=args.tau, kernel=args.kernel, rho=args.rho, amp=args.amp, loop=args.loop,
              contrast=args.contrast, ca1_control=args.ca1_control, dg_keys=args.dg_keys)
    for item in args.loop_arg:                                    # pass-through for newer HippocampalLoop kwargs
        k, v = item.split("=", 1)
        try:
            kw[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            kw[k] = v
    return kw


# ------------------------------------------------------------- train / evaluate
def scores(lm: HookedLM, ids: torch.Tensor, mask: torch.Tensor):
    """Per-target cross-entropy, correctness and mask, from the last positions only."""
    first = int(mask.float().argmax(1).min())
    lg = lm(ids, keep=ids.shape[1] - first + 1)[:, :-1].float()
    tgt, m = ids[:, first:].to(lg.device), mask[:, first:].to(lg.device)
    ce = F.cross_entropy(lg.flatten(0, 1), tgt.flatten(), reduction="none").view_as(tgt)
    return ce, lg.argmax(-1) == tgt, m


@torch.no_grad()
def evaluate(lm: HookedLM, batches: list, window: int, side: bool, task: str) -> dict:
    ce_sum = n_tok = n_ok = exact = n_seq = 0.0
    inj = []
    with lm.condition(window, side):
        for ids, mask, _ in batches:
            ce, ok, m = scores(lm, ids, mask)
            ce_sum += (ce * m).sum().item()
            n_tok += m.sum().item()
            n_ok += (ok & m).sum().item()
            exact += (ok | ~m).all(1).sum().item()
            n_seq += len(ids)
            if lm.last_stats is not None:
                inj.append(lm.last_stats.inj_ratio[:, lm.side.loop.tau:].mean().item())
    res = {"loss": ce_sum / n_tok, "token_acc": n_ok / n_tok}
    if task == "needle":
        res["exact"] = exact / n_seq
    else:
        res["ppl"] = math.exp(res["loss"])
    if inj:
        res["inj_ratio"] = sum(inj) / len(inj)
    return res


def train(lm: HookedLM, task, args) -> list:
    params = [p for p in lm.side.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    warm = max(1, args.steps // 10)                               # linear warm-up, cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * i / args.steps)))
    rng, curve, t0 = random.Random(args.seed), [], time.time()
    log_every = args.log_every or max(1, args.steps // 10)
    for step in range(1, args.steps + 1):
        ids, mask, _ = task.batch(args.batch, rng)
        with lm.condition(args.window, True):
            ce, ok, m = scores(lm, ids, mask)
        loss = (ce * m).sum() / m.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == args.steps:
            acc = ((ok & m).sum() / m.sum()).item()
            inj = lm.last_stats.inj_ratio[:, args.tau:].mean().item() if lm.last_stats is not None else float("nan")
            curve.append({"step": step, "loss": loss.item(), "token_acc": acc, "inj_ratio": inj})
            print(f"step {step:5d} loss {loss.item():.3f} token_acc {acc:.3f} inj {inj:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return curve


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_argument_group("model")
    m.add_argument("--model", default="gpt2", help="HF model name or local path")
    m.add_argument("--tiny", action="store_true", help="random small model + byte tokenizer (offline)")
    m.add_argument("--tiny-arch", choices=["gpt2", "qwen35"], default="gpt2",
                   help="qwen35: 2 Gated DeltaNet + 2 full-attention layers (pure-torch GDN path)")
    m.add_argument("--dtype", choices=list(DTYPES), default="float32")
    m.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                   help="auto: device_map='auto' across GPUs (CPU if none)")
    m.add_argument("--attn-impl", choices=["sdpa", "eager"], default="sdpa",
                   help="must honour 4D masks; flash-attention kernels do not")
    m.add_argument("--grad-checkpointing", action="store_true", help="recompute the frozen upper blocks in backward")
    m.add_argument("--layer", type=int, default=None, help="tap/inject block (default n_layers // 2)")
    w = ap.add_argument_group("attention window")
    w.add_argument("--window", type=int, default=64, help="0: full attention")
    w.add_argument("--window-mode", choices=["sliding", "block", "segment"], default="block",
                   help="block (strict: nothing crosses a block except via the side channels), "
                        "sliding (leaks L·(W−1) tokens through depth), segment (fallback)")
    w.add_argument("--window-layers", choices=["all", "attention"], default="all",
                   help="attention: leave linear-attention (GDN) layers unwindowed")
    w.add_argument("--sink", action=argparse.BooleanOptionalAction, default=True,
                   help="prepend BOS and keep it visible as an attention sink")
    s = ap.add_argument_group("side channels (passed to HippocampalLoop / TimeChannel)")
    s.add_argument("--memory", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--time", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--kernel", choices=["osc", "exp", "flat"], default="osc")
    s.add_argument("--rho", type=float, default=0.7)
    s.add_argument("--tau", type=int, default=16)
    s.add_argument("--amp", type=float, default=1.0)
    s.add_argument("--loop", choices=["open", "closed"], default="open")
    s.add_argument("--ca1-control", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--contrast", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--dg-keys", action=argparse.BooleanOptionalAction, default=False)
    s.add_argument("--d-key", type=int, default=32)
    s.add_argument("--loop-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="extra HippocampalLoop kwarg, e.g. --loop-arg max_gain=0.5 (repeatable)")
    s.add_argument("--init-gain", type=float, default=None,
                   help="initial sigmoid(gain_logit) of the loop (module default 0.27)")
    t = ap.add_argument_group("task and training")
    t.add_argument("--task", choices=["needle", "lm"], default="needle")
    t.add_argument("--text-file", default=None, help="needle filler, or lm corpus")
    t.add_argument("--dataset", default=None, help="lm: wikitext-103, wikitext-2 or NAME:CONFIG")
    t.add_argument("--max-chars", type=int, default=5_000_000, help="lm: characters read from --dataset train")
    t.add_argument("--pairs", type=int, default=1, help="needle: key-value pairs before the filler")
    t.add_argument("--seq-len", type=int, default=512, help="tokens per sequence, excluding the sink")
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--eval-n", type=int, default=128, help="evaluation sequences")
    t.add_argument("--log-every", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default=None, help="default results/pretrained_loop_<task>.json")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    t0 = time.time()
    model, tok = load_model(args)
    tc = text_config(model)
    sink = int(args.sink)
    if args.window_mode in ("block", "segment") and args.window and args.seq_len % args.window:
        raise SystemExit("--seq-len must be a multiple of --window for block/segment windows")
    max_pos = getattr(tc, "max_position_embeddings", None) or getattr(tc, "n_positions", None)
    if max_pos and args.seq_len + sink > max_pos:
        raise SystemExit(f"--seq-len {args.seq_len} (+ sink) exceeds the model's {max_pos} positions")
    if not (args.memory or args.time):
        raise SystemExit("nothing to train: enable --memory and/or --time")

    side = SideChannels(tc.hidden_size, memory=args.memory, time=args.time, **loop_kwargs(args))
    if args.init_gain is not None and side.loop is not None:
        side.loop.gain_logit.data.fill_(math.log(args.init_gain / (1 - args.init_gain)))
    lm = HookedLM(model, side, args.layer, args.window, args.window_mode, args.window_layers, sink,
                  pad_id=tok.eos_token_id, grad_checkpointing=args.grad_checkpointing)
    p = next(lm.blocks[lm.layer].parameters())
    side.to(p.device)                                             # float32 master weights (see _tap)
    n_lin = lm.kinds.count("linear")
    print(f"{args.model if not args.tiny else 'tiny-' + args.tiny_arch}: {len(lm.blocks)} blocks "
          f"({n_lin} linear-attention), d_model {tc.hidden_size}, tap block {lm.layer}, "
          f"{len(lm.windowed)} windowed, {args.window_mode} window {args.window}, reach {lm.reach()}")
    check = check_window(lm)
    print(f"window check: {check}")
    n_train = sum(q.numel() for q in side.parameters() if q.requires_grad)
    n_base = sum(q.numel() for q in model.parameters())
    print(f"trainable side-channel parameters {n_train:,} (base {n_base:,}, frozen)")

    if args.task == "needle":
        corpus = tok.encode(Path(args.text_file).read_text(encoding="utf-8", errors="replace")) if args.text_file else None
        task = NeedleTask(tok, args.seq_len, args.pairs, sink, corpus)
        rng = random.Random(args.seed + 999)
        evals = [task.batch(min(args.batch, args.eval_n - i), rng) for i in range(0, args.eval_n, args.batch)]
        gaps = [g for b in evals for g in b[2]]
        print(f"needle gap {min(gaps)}–{max(gaps)} tokens")
        if args.window and not math.isfinite(lm.reach()):
            print("note: linear-attention layers are not windowed, so the windowed base keeps their memory")
        elif args.window and min(gaps) <= lm.reach():
            print(f"warning: gap ≤ reach {lm.reach()}: the windowed base can still see the value through "
                  "stacked windows; use --window-mode block or a longer --seq-len for a strict test")
    else:
        tr, ev = load_text(args, tok)
        task = TextTask(torch.tensor(tr), torch.tensor(ev), args.seq_len, sink, tok.bos_token_id, args.window)
        evals = task.eval_batches(args.eval_n, args.batch)
        gaps = None
        print(f"lm: {len(tr):,} train / {len(ev):,} eval tokens, {sum(len(b[0]) for b in evals)} eval sequences")

    curve = train(lm, task, args) if args.steps else []
    conds = {"windowed": (args.window, False), "windowed+side": (args.window, True), "full": (0, False)}
    results = {name: evaluate(lm, evals, w, s_on, args.task) for name, (w, s_on) in conds.items()}
    metric = "exact" if args.task == "needle" else "ppl"
    print(f"\n{'condition':15s} {'window':>6s} {'side':>4s} {metric:>8s} {'tok_acc':>8s} {'loss':>7s} {'inj':>6s}")
    for name, (w, s_on) in conds.items():
        r = results[name]
        inj = f"{r['inj_ratio']:.3f}" if "inj_ratio" in r else "-"
        print(f"{name:15s} {w:6d} {'on' if s_on else 'off':>4s} {r[metric]:8.3f} {r['token_acc']:8.3f} "
              f"{r['loss']:7.3f} {inj:>6s}")
    if side.loop is not None:
        results["gain"] = torch.sigmoid(side.loop.gain_logit.float()).item()

    out = Path(args.out or f"results/pretrained_loop_{args.task}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    info = {"blocks": len(lm.blocks), "linear_layers": n_lin, "tap_layer": lm.layer, "windowed_layers": len(lm.windowed),
            "reach": lm.reach() if math.isfinite(lm.reach()) else None, "window_check": check,
            "trainable_params": n_train, "base_params": n_base, "transformers": transformers.__version__}
    if gaps:
        info["gap_range"] = [min(gaps), max(gaps)]
    out.write_text(json.dumps({"args": vars(args), "model": info, "curve": curve, "results": results,
                               "seconds": time.time() - t0}, indent=2))
    print(f"saved {out}")
    return results


if __name__ == "__main__":
    main()
