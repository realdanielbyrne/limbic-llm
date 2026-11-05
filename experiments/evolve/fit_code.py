"""Gate G0.5 (round 11): is a learned low-dimensional alignment code enough?

A trained W1s with lexical alignment (the I1 recipe) aligns the query with a stored token by the
centred cosine of the full-width lexical stream x = LN(TimeChannel(h²¹)) (2048 floats per ring-buffer
entry). Round 11's ``--w1s-code learned`` replaces it with z = W_z x (dc = 64 or 128 floats):

    teacher  E_T[t, j] = exp(β_T (ĉos(x_t, x_j) − 1)),  ĉos centred on the retained keys (the checkpoint's
             own ``SlotLoop._alignment``, its trained log β_align), read weights a[t, k] over the retained
             slots (``SlotLoop._select``), and per slot α_T[t, k, j] = E_T[t, j] mem[k, j] / Σ_j E_T[t, j] mem[k, j]
    student  E_S[t, j] = exp(β_S (cos(W_z x_t, W_z x_j) − 1)), α_S normalised the same way
    loss     Σ_{t,k} a[t, k] KL(α_T[t, k] ‖ α_S[t, k]) / #{t : a slot is retained at t}  (teacher detached)

This script captures what the side sees on needle-task sequences built exactly like training
(``ScatterNeedleTask``, the run's pairs, scatter and names, natural filler from ``--text-file``), with
the frozen model in the windowed condition and the side on: a forward pre-hook on ``side.loop``
records h after the sink drop and the side's own TimeChannel, the teacher's quantities are computed
from it by the loop's own code, and the forward is cut there (nothing above the tap is needed). It
then fits W_z and a scalar log β_S jointly with Adam on the cached features (initialised from the
principal directions of x with its mean direction projected out, and from the teacher's β), and saves
the state-dict fragment

    {"loop.wz.weight": (dc, d), "loop.log_beta_align": ()}

to ``--out`` (a .pt, the scratchpad by default: never commit it), loaded at evaluation with
``eval_by_query.py --w1s-set w1s_code=learned --w1s-set w1s_dc=64 --w1s-load-extra <pt>``. A JSON next
to it records the fit loss, the held-out KL (sequences with ``--heldout-text-file`` filler), and a
cheap proxy of the real score: how often the teacher and the student pick the same aligned position,
in the teacher's most-read slot, at the answer positions. The real score is the evaluation run.

    python experiments/evolve/fit_code.py results/evolve/round10/r10_I1_s1.json --dc 64 \
        --n-seq 512 --steps 3000 --out $SCRATCH/r11_g05/I1_s1_dc64.pt --device cuda
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402
from evolve.run import TRAIN_NAMES, ScatterNeedleTask  # noqa: E402

DATA = Path.home() / "limbic-data"
MASKED = -1e9                                                     # logit of a position outside the slot
MAX_BETA = 40.0                                                   # as w1s_slots.MAX_BETA (the eval clamps β too)


class _Stop(Exception):
    """Raised by the capture hook: nothing above the side's tap is needed."""


# ----------------------------------------------------------------- model and side
def build(rec: dict, device: str, tiny: bool):
    """The frozen model, its tokenizer, the hooked LM and the checkpoint's side, as eval_by_query.py
    rebuilds them (flags the run predates take their defaults; ``strict=False``)."""
    ra, cand = rec["args"], rec.get("candidate")
    if not cand or not cand["class"].startswith("evolve.w1s_slots:"):
        raise SystemExit("fit_code.py needs a W1s result (candidate evolve.w1s_slots:W1sSide)")
    if tiny and not ra.get("tiny"):
        raise SystemExit("--tiny: the result is not a --tiny run")
    if ra.get("tiny"):
        pa = pl.parse_args(["--tiny", "--tiny-arch", ra["tiny_arch"], "--seed", str(ra["seed"]),
                            "--dtype", ra["dtype"], "--device", device, "--attn-impl", ra["attn_impl"]])
    else:
        pa = pl.parse_args(["--model", ra["model"], "--dtype", "bfloat16", "--device", device,
                            "--attn-impl", ra.get("attn_impl", "sdpa")])
    model, tok = pl.load_model(pa)
    d = pl.text_config(model).hidden_size
    mod_name, cls = cand["class"].split(":")
    mod = importlib.import_module(mod_name)
    flags = {**vars(mod.split_args([])[0]), **cand["args"]}
    side = getattr(mod, cls)(d, argparse.Namespace(**ra), argparse.Namespace(**flags))
    state = torch.load(Path(rec["_path"]).with_suffix(".pt"), map_location="cpu")
    missing, unexpected = side.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys in the checkpoint: {unexpected}")
    side = side.to(device).eval()
    loop = side.loop
    if side.out_mode != "additive" or loop.store != "slots" or loop.align != "lex":
        raise SystemExit("the teacher must be additive slots with lexical alignment (--w1s-align lex)")
    if getattr(loop, "code", "lex") != "lex" or hasattr(loop, "wz"):
        raise SystemExit("the teacher already has a learned code (--w1s-code learned)")
    if loop.gate == "oracle" or loop.ctrl != "none":
        raise SystemExit("the teacher must run its own gate, without controls")
    LM = pl.HookedLM
    if hasattr(side, "taps"):
        from evolve.taps import TapLM
        LM = TapLM
    lm = LM(model, side, ra["layer"], ra["window"], ra["window_mode"], ra["window_layers"], int(ra.get("sink", True)),
            pad_id=tok.eos_token_id)
    return model, tok, lm, side, list(missing)


def make_task(rec: dict, tok, text_file: str | None) -> ScatterNeedleTask:
    """The run's training task (pairs, scatter, names, length), with natural filler from ``text_file``."""
    ra = rec["args"]
    scatter = rec.get("needle_scatter", 0)
    if not scatter:
        raise SystemExit("fit_code.py needs a scattered-layout run (--needle-scatter)")
    corpus = tok.encode(Path(text_file).read_text(encoding="utf-8", errors="replace")) if text_file else None
    nn = rec.get("needle_names", "all")                           # as run.py trained it (review F13)
    if nn == "pool":
        from evolve.names import syllable_pool
        names = syllable_pool(rec["needle_name_pool"], 0)
    else:
        names = TRAIN_NAMES if nn == "train" else None
    return ScatterNeedleTask(tok, ra["seq_len"], ra["pairs"], int(ra.get("sink", True)), corpus, min_gap=scatter,
                             block=ra["window"], names=names, tail=rec.get("needle_tail") or 256,
                             mix=rec.get("needle_mix", 1.0))


# ------------------------------------------------------------------------ capture
def _teacher(loop, h: torch.Tensor, extra) -> dict:
    """What the checkpoint's own read uses, from the loop's input h (B, T, D) (after the TimeChannel):
    the lexical stream x, the read weights a (B, T, N), the alignment E (B, T, T) and the slots'
    stored positions mem (B, N, T), through ``SlotLoop._run_store`` and ``_read`` (the forward's path)."""
    f, st = loop._run_store(loop.norm(h), h, extra, None)
    loop._read(f, st)
    return dict(x=f["x"], a=st["a"], E=st["E"], mem=st["mem"], ret=st["ret"])


def _rows_of(t: dict, b: int, n1: int, ans: list[int], cache_dtype) -> dict:
    """One sequence's cache: x (T, D), log E_T (T, T), and the slots ever retained (Nr): a (T, Nr),
    J (Nr, n − 1) their stored positions (a slot's are contiguous), ok (Nr, n − 1)."""
    x, a, E, mem, ret = t["x"][b], t["a"][b], t["E"][b], t["mem"][b], t["ret"][b]
    T = x.shape[0]
    cols = ret.any(0).nonzero().squeeze(-1)
    mem = mem[cols]
    first = mem.int().argmax(-1)                                # the first stored position of each slot
    J = first[:, None] + torch.arange(n1, device=x.device)
    ok = (J < T) & mem.gather(1, J.clamp_max(T - 1))
    if not torch.equal(ok.sum(-1), mem.sum(-1)):
        raise AssertionError("a slot's stored positions are not contiguous")
    return dict(x=x.to(cache_dtype).cpu(), logE=torch.log(E.clamp_min(1e-38)).float().cpu(),
                a=a[:, cols].float().cpu(), J=torch.where(ok, J, torch.zeros_like(J)).cpu(), ok=ok.cpu(),
                valid=ret.any(-1).cpu(), ans=torch.tensor(ans))


@torch.no_grad()
def capture(lm, side, task, n: int, batch: int, rng: random.Random, cache_dtype=torch.float16) -> dict:
    """Run n task sequences through the frozen model (windowed, side on) and cache the teacher."""
    loop, rows, n1, got = side.loop, [], side.loop.n - 1, {}

    def hook(mod, args):
        h, extra = args[0], (args[1] if len(args) > 1 else None)
        got["t"] = _teacher(mod, h, extra)
        raise _Stop

    handle = loop.register_forward_pre_hook(hook)
    try:
        while len(rows) < n:
            seqs, ans = [], []
            for _ in range(min(batch, n - len(rows))):
                ids, n_ans, _ = task.sample(rng)
                L = len(ids) - lm.sink
                seqs.append(ids)                                  # the positions predicting the digits
                ans.append([L - n_ans + k - 1 for k in range(1, n_ans)])   # (side coordinates)
            got.clear()
            try:
                with lm.condition(lm.window, True):
                    lm(torch.tensor(seqs), keep=1)
            except _Stop:
                pass
            if "t" not in got:
                raise RuntimeError("the side's loop was not called (is the side on?)")
            rows += [_rows_of(got["t"], b, n1, ans[b], cache_dtype) for b in range(len(seqs))]
    finally:
        handle.remove()
    return collate(rows)


def collate(rows: list[dict]) -> dict:
    """Stack per-sequence caches, padding the slot axis (padded slots: a = 0, ok = False)."""
    N = max(r["a"].shape[1] for r in rows)
    pad = lambda v, dim, val=0: F.pad(v, [0, 0] * (v.dim() - 1 - dim) + [0, N - v.shape[dim]], value=val)
    return dict(x=torch.stack([r["x"] for r in rows]), logE=torch.stack([r["logE"] for r in rows]),
                a=torch.stack([pad(r["a"], 1) for r in rows]), J=torch.stack([pad(r["J"], 0) for r in rows]),
                ok=torch.stack([pad(r["ok"], 0, False) for r in rows]), valid=torch.stack([r["valid"] for r in rows]),
                ans=torch.stack([r["ans"] for r in rows]))


# ---------------------------------------------------------------------- the fit
def _slot_logits(c: dict, wz: torch.Tensor, log_beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher and student log α over each slot's stored positions, (B, T, N, n − 1) each."""
    x, J, ok = c["x"].float(), c["J"], c["ok"]
    (B, T, _), (N, n1) = x.shape, J.shape[1:]
    z = F.normalize(x @ wz.T, dim=-1)
    ls = log_beta.exp().clamp(max=MAX_BETA) * (z @ z.transpose(1, 2))   # the − 1 cancels in each slot's softmax
    idx = J.reshape(B, 1, N * n1).expand(B, T, N * n1)
    m = ok[:, None]
    lS = ls.gather(2, idx).view(B, T, N, n1).masked_fill(~m, MASKED).log_softmax(-1)
    lT = c["logE"].gather(2, idx).view(B, T, N, n1).masked_fill(~m, MASKED).log_softmax(-1)
    return lT, lS


def kl_loss(c: dict, wz: torch.Tensor, log_beta: torch.Tensor) -> torch.Tensor:
    """Σ_{t,k} a[t, k] KL(α_T ‖ α_S) / #{t with a retained slot}, for one batch of cached sequences."""
    lT, lS = _slot_logits(c, wz, log_beta)
    kl = torch.where(c["ok"][:, None], lT.exp() * (lT - lS), torch.zeros_like(lT)).sum(-1)   # (B, T, N)
    return (c["a"] * kl).sum() / c["valid"].sum().clamp_min(1)


def _batches(cache: dict, idx: list[int], device):
    return {k: v[idx].to(device) for k, v in cache.items()}


@torch.no_grad()
def evaluate(cache: dict, wz, log_beta, device, chunk: int = 8) -> dict:
    """The loss over a whole cache, and the argmax agreement at the answer positions (all digits, and
    digit 1): in the teacher's most-read retained slot, the same stored position under both codes."""
    S = cache["x"].shape[0]
    num = den = 0.0
    agree, agree1 = [], []
    for i in range(0, S, chunk):
        c = _batches(cache, list(range(i, min(S, i + chunk))), device)
        lT, lS = _slot_logits(c, wz, log_beta)
        kl = torch.where(c["ok"][:, None], lT.exp() * (lT - lS), torch.zeros_like(lT)).sum(-1)
        num += (c["a"] * kl).sum().item()
        den += c["valid"].sum().item()
        b = torch.arange(len(c["ans"]), device=device)[:, None]
        t = c["ans"]                                              # (B, n_ans − 1)
        k = c["a"][b, t].argmax(-1)                               # the teacher's most-read slot
        eT, eS = lT[b, t, k].argmax(-1), lS[b, t, k].argmax(-1)
        has = c["valid"][b, t]
        same = (eT == eS).float()
        agree += same[has].tolist()
        agree1 += same[:, 0][has[:, 0]].tolist()
    mean = lambda v: sum(v) / len(v) if v else None
    return {"kl": num / max(den, 1), "argmax_agree_answer": mean(agree), "argmax_agree_digit1": mean(agree1),
            "n_answer_positions": len(agree)}


def init_wz(x: torch.Tensor, dc: int, how: str, gen: torch.Generator) -> torch.Tensor:
    """(dc, D): the top principal directions of x, its mean direction projected out (the student's
    cosine is uncentred, the teacher's centred), or random; random rows fill in when dc > D."""
    D = x.shape[-1]
    w = torch.randn(dc, D, generator=gen) / D ** 0.5
    if how == "pca":
        x = x.reshape(-1, D).float()
        x = x[torch.randperm(len(x), generator=gen)[:20000]]
        mu = x.mean(0)
        xc = x - mu
        evals, evecs = torch.linalg.eigh(xc.T @ xc / len(xc))
        top = evecs[:, evals.argsort(descending=True)[:min(dc, D)]].T
        w[:len(top)] = top
        u = F.normalize(mu, dim=0)
        w = w - (w @ u)[:, None] * u
    return w


def fit(train: dict, held: dict, dc: int, log_beta0: float, a) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Adam on (W_z, log β_S) over random batches of cached sequences."""
    gen = torch.Generator().manual_seed(a.seed)
    wz = init_wz(train["x"], dc, a.init, gen).to(a.device).requires_grad_(True)
    lb = torch.tensor(float(log_beta0), device=a.device, requires_grad=True)
    rep = {"init": {"train": evaluate(train, wz, lb, a.device), "heldout": evaluate(held, wz, lb, a.device)}}
    opt = torch.optim.Adam([wz, lb], lr=a.lr)
    rng, S, curve = random.Random(a.seed), train["x"].shape[0], []
    for step in range(a.steps):
        loss = kl_loss(_batches(train, rng.sample(range(S), min(a.fit_batch, S)), a.device), wz, lb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % a.log_every == 0 or step == a.steps - 1:
            curve.append((step, loss.item(), lb.item()))
            print(f"step {step}: kl {loss.item():.4f}, log beta {lb.item():.3f}", flush=True)
    rep["final"] = {"train": evaluate(train, wz, lb, a.device), "heldout": evaluate(held, wz, lb, a.device)}
    rep["curve"] = curve
    return wz.detach().cpu(), lb.detach().cpu(), rep


# --------------------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("result", help="a W1s result JSON (lexical alignment) with its .pt checkpoint next to it")
    ap.add_argument("--dc", type=int, nargs="+", default=[64],
                    help="code width(s); with several, one fit each on the same caches (--out gets .dc<N>)")
    ap.add_argument("--n-seq", type=int, default=512, help="training sequences cached")
    ap.add_argument("--n-heldout", type=int, default=64, help="held-out sequences cached")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--fit-batch", type=int, default=8, help="cached sequences per Adam step")
    ap.add_argument("--batch", type=int, default=8, help="sequences per model forward while caching")
    ap.add_argument("--init", choices=["pca", "random"], default="pca")
    ap.add_argument("--seed", type=int, default=0, help="task sampling (held out: seed + 1) and the fit")
    ap.add_argument("--text-file", default=str(DATA / "wikitext103_train.txt"),
                    help="training filler ('none': random tokens)")
    ap.add_argument("--heldout-text-file", default=str(DATA / "wikitext103_test.txt"),
                    help="held-out filler ('none': random tokens)")
    ap.add_argument("--cache-dtype", choices=["float16", "bfloat16", "float32"], default="float16",
                    help="dtype of the cached lexical stream x (the teacher is computed in float32 first)")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--tiny", action="store_true", help="the result must be a --tiny run (tests; no download)")
    ap.add_argument("--out", required=True, help="the fragment .pt (a .json goes next to it)")
    a = ap.parse_args(argv)
    a.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for k in ("text_file", "heldout_text_file"):
        if getattr(a, k) in ("", "none"):
            setattr(a, k, None)
    return a


def main(argv=None) -> dict:
    a = parse_args(argv)
    t0 = time.time()
    rec = json.loads(Path(a.result).read_text())
    rec["_path"] = a.result
    torch.manual_seed(a.seed)
    _, tok, lm, side, missing = build(rec, a.device, a.tiny)
    loop = side.loop
    dt = getattr(torch, a.cache_dtype)
    train = capture(lm, side, make_task(rec, tok, a.text_file), a.n_seq, a.batch, random.Random(a.seed), dt)
    held = capture(lm, side, make_task(rec, tok, a.heldout_text_file), a.n_heldout, a.batch,
                   random.Random(a.seed + 1), dt)
    log_beta_t = loop.log_beta_align.item()
    D = train["x"].shape[-1]
    base = {"result": a.result, "d_model": D, "n_seq": a.n_seq, "n_heldout": a.n_heldout, "steps": a.steps,
            "lr": a.lr, "fit_batch": a.fit_batch, "init": a.init, "seed": a.seed, "text_file": a.text_file,
            "heldout_text_file": a.heldout_text_file, "cache_dtype": a.cache_dtype,
            "teacher_log_beta_align": log_beta_t, "missing_keys": missing,
            "query_positions_train": int(train["valid"].sum()), "slots_max": int(train["a"].shape[-1])}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    reports = {}
    for dc in a.dc:
        wz, lb, rep = fit(train, held, dc, log_beta_t, a)
        path = out if len(a.dc) == 1 else out.with_name(f"{out.stem}.dc{dc}{out.suffix}")
        torch.save({"loop.wz.weight": wz, "loop.log_beta_align": lb}, path)
        res = {**base, "dc": dc, "fragment": str(path), "student_log_beta_align": lb.item(),
               "fit_loss_init": rep["init"]["train"]["kl"], "fit_loss": rep["final"]["train"]["kl"],
               "heldout_kl_init": rep["init"]["heldout"]["kl"], "heldout_kl": rep["final"]["heldout"]["kl"],
               "init_eval": rep["init"], "final_eval": rep["final"], "curve": rep["curve"],
               "seconds": time.time() - t0}
        path.with_suffix(".json").write_text(json.dumps(res, indent=2))
        print(json.dumps({k: res[k] for k in ("dc", "fit_loss_init", "fit_loss", "heldout_kl_init", "heldout_kl")}
                         | {"heldout_agree_answer": rep["final"]["heldout"]["argmax_agree_answer"],
                            "heldout_agree_digit1": rep["final"]["heldout"]["argmax_agree_digit1"]}))
        print(f"saved {path}")
        reports[dc] = res
    return reports


if __name__ == "__main__":
    main()
