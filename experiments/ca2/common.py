"""Shared runner pieces for round 12: loaders (reused), the HookedLM wrapper, paired bootstrap, trace storage.

Reused, not rewritten: ``pretrained_loop`` (``load_model``, ``find_blocks``, ``layer_kinds``, ``HookedLM``,
``TextTask``, ``scores``) and ``experiments/alignment`` (``run.generate``, ``run.perplexity``, ``pad_id``,
``eos_ids``; ``judge.degeneration``, ``repetition_onset``, ``assert_private``, ``private_file``; ``probe``),
the latter through ``alignment()`` so that their bare-name imports (``data``, ``judge``, ...) resolve.

``SteerSide`` lets ``pretrained_loop.HookedLM`` host a ``CA2Controller`` for windowed, teacher-forced,
single-pass runs (``HookedLM`` cannot decode with a KV cache). ``HookedLM._tap`` strips the sink
column(s) and calls ``side(h)`` expecting ``(h', stats)``; the side returns ``(ctrl(h), None)``, so
``condition(window, side)`` switches it as for P5. Each ``HookedLM`` call is a whole sequence, so with
``model`` given the controller is reset when the call starts (input-embedding hook, which also passes the
ids with the sink stripped) and the call is closed by a model forward hook (tap signals collected); the
per-call trace is then kept in ``last_trace``. Because the sink is already stripped, the controller must
use ``sink_skip=0`` (asserted), and ``steer_from`` should be an int (``last_prompt`` would steer one token).
Taps attached through ``SteerSide`` see model coordinates; ``ctrl.col_offset = sink`` makes the trace and
``ctx.col_offset`` line up. Segment window mode folds the batch and is not supported.

``paired_boot`` gives percentile intervals for a paired difference, resampling units (sequences or
problems) or clusters of them, with samples averaged within each cluster first.

``TraceWriter`` follows the plan's measurement policy: per-token arrays (fp16 for floats unless a value
exceeds the fp16 range, int32 for integers) and per-item JSON records, written only outside the
repository (default ``~/limbic-data/ca2_traces/<run>/``, checked with ``judge.assert_private``), in shards
per arm, with ``manifest.json`` listing every file's SHA-256, keys, dtypes and shapes. ``close()`` returns
the pointer (directory, manifest hash, counts) that goes into the results JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
sys.path.insert(0, str(EXP))
import pretrained_loop as pl  # noqa: E402
from ca2.controller import CA2Controller, Tap, _ids_from  # noqa: E402

TRACE_ROOT = Path("~/limbic-data/ca2_traces")
_ALIGN: SimpleNamespace | None = None


# ------------------------------------------------------------------------------------------ loading
def alignment() -> SimpleNamespace:
    """The alignment modules (run, judge, probe, data), loaded once; run.py is imported as ``alignment_run``."""
    global _ALIGN
    if _ALIGN is None:
        adir = EXP / "alignment"
        for p in (str(adir), str(EXP), str(ROOT)):
            if p not in sys.path:
                sys.path.append(p)
        import data as adata  # noqa: E402
        import judge as ajudge  # noqa: E402
        import probe as aprobe  # noqa: E402
        spec = importlib.util.spec_from_file_location("alignment_run", adir / "run.py")
        arun = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arun)
        _ALIGN = SimpleNamespace(run=arun, judge=ajudge, probe=aprobe, data=adata)
    return _ALIGN


def load(model: str | None, dtype: str = "bfloat16", device: str = "cuda", tiny: bool = False,
         tiny_arch: str = "gpt2", seed: int = 0, attn_impl: str = "sdpa"):
    """``pl.load_model`` with explicit arguments; returns (model, tok, blocks, kinds). The model is frozen."""
    ns = argparse.Namespace(model=model, tiny=tiny, tiny_arch=tiny_arch, seed=seed, dtype=dtype, device=device,
                            attn_impl=attn_impl)
    m, tok = pl.load_model(ns)
    blocks = pl.find_blocks(m)
    return m, tok, blocks, pl.layer_kinds(m, blocks)


def load_test_text(path: str | Path, tok) -> torch.Tensor:
    """The whole file as token ids (``pl.load_text`` keeps only a 90/10 split; the test file is all evaluation)."""
    return torch.tensor(tok.encode(Path(path).expanduser().read_text(encoding="utf-8", errors="replace")))


def text_task(ids: torch.Tensor, seq_len: int, sink: int, bos: int, window: int) -> "pl.TextTask":
    """A ``pl.TextTask`` whose evaluation set is ``ids`` (train = eval; nothing is trained)."""
    return pl.TextTask(ids, ids, seq_len, sink, bos, window)


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


# ---------------------------------------------------------------------------------------- HookedLM side
class SteerSide(nn.Module):
    """``HookedLM`` side wrapping a ``CA2Controller``: ``forward(h) -> (ctrl(h), None)``.

    With ``model`` (the HF model ``HookedLM`` wraps) the controller is reset and a call opened at every
    model call (ids with the first ``sink`` columns stripped), closed by a model forward hook, and taps
    are attached; the call's trace dict is kept in ``last_trace`` while ``ctrl.record``. Without ``model``
    the controller is reset on every side call (no ids, no taps).
    """

    def __init__(self, ctrl: CA2Controller, model: nn.Module | None = None, sink: int = 1,
                 taps: Sequence[Tap] = (), tap_layer: int | None = None) -> None:
        super().__init__()
        if sink and ctrl.sink_skip != 0:
            raise ValueError("HookedLM strips the sink before the side; build the controller with sink_skip=0")
        self.ctrl, self.sink = ctrl, int(sink)
        self.__dict__["model"] = model                     # not a submodule: the side owns no model parameters
        self.last_trace: dict | None = None
        self.handles: list = []
        self.taps = list(taps)
        ctrl.col_offset = self.sink if model is not None else 0
        if model is not None:
            self.__dict__["blocks"] = pl.find_blocks(model)
            self.kinds = pl.layer_kinds(model, self.blocks)
            n = len(self.blocks)
            self.tap = n // 2 if tap_layer is None else tap_layer % n
            self.inject_before, self.relevance_layer = self.tap + 1, self.tap
            self.handles = [model.get_input_embeddings().register_forward_hook(self._ids_hook, with_kwargs=True),
                            model.register_forward_hook(self._end_hook)]
            for t in self.taps:
                ctrl.add_tap(t)
                t.attach(self)
        elif self.taps:
            raise ValueError("taps need the model")

    def _ids_hook(self, mod, args, kwargs, out):
        ids = _ids_from(args, kwargs)
        self.ctrl.reset()
        self.ctrl.begin_call(None if ids is None else ids[:, self.sink:])

    def _end_hook(self, mod, args, out):
        self.ctrl.end_call()
        if self.ctrl.record and self.ctrl.trace:
            self.last_trace = self.ctrl.trace[-1]

    def forward(self, h: torch.Tensor):
        if self.model is None:
            self.ctrl.reset()
        out = self.ctrl(h)
        if self.model is None and self.ctrl.record and self.ctrl.trace:
            self.last_trace = self.ctrl.trace[-1]
        return out, None

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        for t in self.taps:
            t.remove()
            if self.ctrl.taps.get(t.name) is t:
                del self.ctrl.taps[t.name]


# -------------------------------------------------------------------------------------------- statistics
def paired_boot(a, b=None, clusters=None, n: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired bootstrap of mean(a − b) (or of mean(a) when b is None) with percentile intervals.

    ``clusters`` (same length) groups units (e.g. samples of one problem): differences are averaged within
    each cluster first and clusters are resampled. Returns diff, lo, hi, se, n_units, p_le_0 (share of
    bootstrap means ≤ 0) and p_ge_0.
    """
    a = np.asarray(a, dtype=np.float64)
    d = a - (0.0 if b is None else np.asarray(b, dtype=np.float64))
    ok = np.isfinite(d)
    if clusters is not None:
        c = np.asarray(clusters)[ok]
        d = d[ok]
        keys, inv = np.unique(c, return_inverse=True)
        d = np.bincount(inv, weights=d) / np.bincount(inv)
    else:
        d = d[ok]
    if d.size == 0:
        return {"diff": None, "lo": None, "hi": None, "se": None, "n_units": 0, "p_le_0": None, "p_ge_0": None}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(n, d.size))].mean(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "se": float(boots.std(ddof=1)),
            "n_units": int(d.size), "p_le_0": float((boots <= 0).mean()), "p_ge_0": float((boots >= 0).mean())}


def gain_stats(g: torch.Tensor, steer: torch.Tensor, g_min: float, g_max: float) -> dict:
    """Distribution of the raw gain over steered positions: mean, SD, share at g_min / g_max, quantiles."""
    v = g[steer.bool()].double()
    if v.numel() == 0:
        return {"n": 0}
    tol = 1e-6 * max(1.0, g_max)
    q = torch.quantile(v, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)).tolist()
    return {"n": int(v.numel()), "mean": float(v.mean()), "sd": float(v.std()) if v.numel() > 1 else 0.0,
            "frac_at_gmin": float((v <= g_min + tol).double().mean()),
            "frac_at_gmax": float((v >= g_max - tol).double().mean()),
            "q05": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q95": q[4],
            "varied": bool(float(v.std()) >= 0.1 * (g_max - g_min)) if v.numel() > 1 and g_max > g_min else False}


# -------------------------------------------------------------------------------------------- traces
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _np(v) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
        if v.dtype == torch.bfloat16:
            v = v.float()
        v = v.numpy()
    v = np.asarray(v)
    if v.dtype == np.bool_:
        return v.astype(np.uint8)
    if np.issubdtype(v.dtype, np.integer):
        return v.astype(np.int32) if v.size == 0 or np.abs(v).max() < 2 ** 31 else v.astype(np.int64)
    v = v.astype(np.float32)
    fin = v[np.isfinite(v)]
    return v.astype(np.float16) if fin.size == 0 or np.abs(fin).max() <= 6.0e4 else v


class TraceWriter:
    """Per-token arrays and per-item records of one run, outside the repository, with a hashed manifest.

    ``add(arm, item, tokens)``: ``item`` a JSON-able dict (one problem / sequence / sample); ``tokens`` a
    dict of per-token arrays (T,) or (T, K) for that item (all with the same T). Tokens are concatenated
    per arm and key; the item record gets ``tok_shard``, ``tok_offset`` and ``n_tok`` so that its slice can
    be recovered. Shards are flushed every ``shard_tokens`` tokens per arm.
    """

    def __init__(self, run: str, root: str | Path = TRACE_ROOT, meta: dict | None = None,
                 shard_tokens: int = 2_000_000) -> None:
        aj = alignment().judge
        self.run = aj.safe_name(run)
        self.dir = aj.assert_private(Path(root).expanduser() / self.run, ROOT)
        self.meta = {"run": self.run, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state(),
                     **(meta or {})}
        self.shard_tokens = int(shard_tokens)
        self._buf: dict[str, dict[str, list]] = {}
        self._n: dict[str, int] = {}
        self._shard: dict[str, int] = {}
        self._items: dict[str, int] = {}
        self.files: list[dict] = []
        self.closed = False

    def add(self, arm: str, item: dict, tokens: dict | None = None) -> None:
        arm = alignment().judge.safe_name(arm)
        (self.dir / arm).mkdir(exist_ok=True)
        rec = dict(item)
        if tokens:
            arrs = {k: _np(v) for k, v in tokens.items()}
            lens = {a.shape[0] for a in arrs.values()}
            if len(lens) != 1:
                raise ValueError(f"per-token arrays of one item must share T, got {sorted(lens)}")
            T = lens.pop()
            buf = self._buf.setdefault(arm, {})
            if buf and set(buf) != set(arrs):
                missing = set(buf) ^ set(arrs)
                raise ValueError(f"arm {arm!r}: per-token keys differ from earlier items: {sorted(missing)}")
            for k, a in arrs.items():
                buf.setdefault(k, []).append(a)
            off = self._n.get(arm, 0)
            rec.update(tok_shard=self._shard.get(arm, 0), tok_offset=off, n_tok=int(T))
            self._n[arm] = off + T
        with (self.dir / arm / "items.jsonl").open("a") as f:
            f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        self._items[arm] = self._items.get(arm, 0) + 1
        if self._n.get(arm, 0) >= self.shard_tokens:
            self._flush(arm)

    def _flush(self, arm: str) -> None:
        buf = self._buf.get(arm)
        if not buf:
            return
        k = self._shard.get(arm, 0)
        path = self.dir / arm / f"tokens_{k:04d}.npz"
        arrays = {}
        for key, parts in buf.items():
            dts = {p.dtype for p in parts}
            if len(dts) > 1:                                # some item overflowed fp16: keep everything fp32
                parts = [p.astype(np.float32) if p.dtype == np.float16 else p for p in parts]
            arrays[key.replace("/", "__")] = np.concatenate(parts, 0)
        np.savez(path, **arrays)
        self.files.append({"path": str(path.relative_to(self.dir)), "arm": arm, "sha256": _sha256(path),
                           "n_tok": int(self._n[arm]),
                           "keys": {k: {"dtype": str(a.dtype), "shape": list(a.shape)} for k, a in arrays.items()}})
        self._buf[arm], self._n[arm], self._shard[arm] = {}, 0, k + 1

    def close(self) -> dict:
        """Flush, write manifest.json; returns the pointer for the results JSON."""
        if not self.closed:
            for arm in list(self._buf):
                self._flush(arm)
            for arm in self._items:
                p = self.dir / arm / "items.jsonl"
                self.files.append({"path": str(p.relative_to(self.dir)), "arm": arm, "sha256": _sha256(p),
                                   "n_items": self._items[arm]})
            man = {**self.meta, "files": self.files, "key_sep": "__ stands for / in token keys"}
            (self.dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
            self.closed = True
        mp = self.dir / "manifest.json"
        return {"dir": str(self.dir), "manifest": str(mp), "manifest_sha256": _sha256(mp),
                "n_items": dict(self._items), "n_files": len(self.files)}


def row_tokens(tt: dict[str, torch.Tensor], b: int, sel: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Row ``b`` of ``CA2Controller.trace_tensors()`` output, optionally restricted to columns ``sel`` (bool, T)."""
    out = {}
    for k, v in tt.items():
        r = v[b]
        out[k] = r[sel] if sel is not None else r
    return out
