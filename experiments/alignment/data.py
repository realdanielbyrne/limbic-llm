"""Prompt sets, chat formatting and left-padded batches for the alignment track.

A prompt set is named by a spec:

    hf:REPO:CONFIG:SPLIT:FIELD[:KEY=VALUE]    a Hugging Face dataset (CONFIG may be empty)
    path/to/file.{txt,jsonl,csv}[:FIELD]      one prompt per line, or a field of each record
    synthetic:harmful | synthetic:harmless    neutral stand-ins for offline smoke tests

Defaults (checked on the Hub, 2 Nov 2026):

    harmful    hf:mlabonne/harmful_behaviors::{train,test}:text     416 / 104 prompts
    harmless   hf:mlabonne/harmless_alpaca::{train,test}:text       25,058 / 6,265
    borderline hf:Paul/XSTest::train:prompt:label=safe              250 safe prompts that look unsafe
    contrast   hf:Paul/XSTest::train:prompt:label=unsafe            their 200 unsafe counterparts

The synthetic sets contain no harmful text: the two classes differ only in a
marker word, which is enough to test the pipeline.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import torch

DEFAULTS = {
    "harmful_train": "hf:mlabonne/harmful_behaviors::train:text",
    "harmful_test": "hf:mlabonne/harmful_behaviors::test:text",
    "harmless_train": "hf:mlabonne/harmless_alpaca::train:text",
    "harmless_test": "hf:mlabonne/harmless_alpaca::test:text",
    "borderline": "hf:Paul/XSTest::train:prompt:label=safe",
    "contrast": "hf:Paul/XSTest::train:prompt:label=unsafe",
}


def load_prompts(spec: str, n: int | None = None, seed: int = 0) -> list[str]:
    """Load up to ``n`` prompts (a seeded random subset when the set is larger)."""
    if spec.startswith("synthetic:"):
        marker = {"harmful": "crimson", "harmless": "azure"}[spec.split(":", 1)[1]]
        rows = [f"Please describe the {marker} object number {i}." for i in range(max(n or 64, 1))]
    elif spec.startswith("hf:"):
        parts = spec[3:].split(":")
        repo, config, split_, field = parts[0], parts[1] or None, parts[2], parts[3]
        flt = parts[4].split("=", 1) if len(parts) > 4 else None
        from datasets import load_dataset             # only needed for Hub sets
        ds = load_dataset(repo, config, split=split_)
        rows = [r[field] for r in ds if flt is None or str(r[flt[0]]) == flt[1]]
    else:
        path, _, field = spec.partition(":") if not Path(spec).exists() else (spec, "", "")
        p = Path(path)
        if p.suffix == ".jsonl":
            rows = [json.loads(l)[field or "prompt"] for l in p.read_text().splitlines() if l.strip()]
        elif p.suffix == ".csv":
            with p.open() as f:
                rows = [r[field or "prompt"] for r in csv.DictReader(f)]
        else:
            rows = [l for l in p.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if isinstance(r, str) and r.strip()]
    if n is not None and len(rows) > n:
        rows = random.Random(seed).sample(rows, n)
    return rows


def format_prompt(tok, text: str, chat: bool = True) -> list[int]:
    """Token ids of one user turn plus the start of the assistant turn.

    Uses the tokenizer's chat template when there is one (thinking disabled for
    Qwen3-style templates), otherwise a plain "User / Assistant" frame.
    """
    hf = getattr(tok, "tok", None)
    if chat and hf is not None and getattr(hf, "chat_template", None):
        rendered = hf.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
        return tok.encode(rendered)
    return tok.encode(f"User: {text}\nAssistant:")


def pad_left(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, T - len(s):] = 1
    return ids, mask


def batches(token_lists: list[list[int]], pad_id: int, batch: int):
    for i in range(0, len(token_lists), batch):
        yield pad_left(token_lists[i:i + batch], pad_id)
