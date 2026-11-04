"""Are the facts novel at all? Candidate encoding triggers, by token class, with no training.

W1j's comparator (the chain store's own error) is anti-selective at initialisation (P42). Before
building another trigger, this asks whether *any* novelty measure picks out the facts' first digits
in the filler. On fresh scattered needle batches (natural and random filler), by token class (as
``diag_novelty.py``: d1, d234, space, head, filler, query), it reports mean and AUC (d1 against
filler) for:

  surprisal   the frozen LM's own −log p(token_t | context), in the windowed condition the side
              memory works in. An oracle: it needs the token ids and the final layer, which a side
              memory at the tap may not see (rule E4), so it bounds what any surprise signal can do
  change      1 − cos(h_t, h_{t−1}) at the tap, parameter-free
  raw_store   the relative error of a parameter-free delta store on the tap's states
              (keys: fixed random projection of h_{t−1}, normalised; values: of h_t; α = 1, β = 0.5)

    python experiments/evolve/diag_surprise.py --text-file <wikitext test> \
        --out results/evolve/round8/diag_surprise.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl  # noqa: E402
from evolve.diag_novelty import CLASSES, auc, labels  # noqa: E402
from evolve.run import ScatterNeedleTask  # noqa: E402
from limbic_llm.gated_delta import _run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    pa = pl.parse_args(["--model", args.model, "--dtype", "bfloat16", "--device", "cuda"])
    model, tok = pl.load_model(pa)
    d = pl.text_config(model).hidden_size
    g = torch.Generator().manual_seed(0)
    P = (torch.randn(d, 4 * 64, generator=g) / d ** 0.5).cuda()       # 4 heads × 64 dims
    lm = pl.HookedLM(model, None, args.layer, args.window, "block", "all", 1, pad_id=tok.eos_token_id)
    corpus = tok.encode(Path(args.text_file).read_text(encoding="utf-8", errors="replace"))
    report = {}
    for filler, corp in (("natural", corpus), ("random", None)):
        task = ScatterNeedleTask(tok, args.seq_len, args.pairs, 1, corp, min_gap=8, block=args.window)
        rng = random.Random(args.seed)
        acc = {k: {c: [] for c in CLASSES} for k in ("surprisal", "change", "raw_store")}
        for _ in range(args.n // args.batch):
            rows, labs = [], []
            for _ in range(args.batch):
                ids, n_ans, _ = task.sample(rng)
                rows.append(ids)
                labs.append(labels(len(ids), task.sink, task.last_value_at, task.last_heads,
                                   task.last_query_len + n_ans))
            ids = torch.tensor(rows)
            cap = {}
            hook = lm.blocks[args.layer].register_forward_hook(
                lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).float()))
            with torch.no_grad(), lm.condition(args.window, False):
                lg = lm(ids).float()
            hook.remove()
            idc = ids.cuda()
            nll = F.cross_entropy(lg[:, :-1].transpose(1, 2), idc[:, 1:], reduction="none")
            surpr = F.pad(nll, (1, 0))                                     # token t's surprisal at t
            h = cap["h"]
            change = F.pad(1 - F.cosine_similarity(h[:, 1:], h[:, :-1], dim=-1), (1, 0))
            z = (F.layer_norm(h, (d,)) @ P).view(h.shape[0], h.shape[1], 4, 64).transpose(1, 2)
            k = F.normalize(torch.cat([torch.zeros_like(z[:, :, :1]), z[:, :, :-1]], 2), dim=-1)
            v = F.normalize(z, dim=-1)
            B, H, T, _ = v.shape
            _, _, err = _run(v, k, v, torch.ones(B, H, T, device=h.device), torch.full((B, H, T), 0.5, device=h.device),
                             None, 64)
            sig = {"surprisal": surpr, "change": change, "raw_store": err.mean(1)}
            lab = torch.stack(labs).cuda()
            for name, val in sig.items():
                for ci, c in enumerate(CLASSES):
                    acc[name][c].append(val[lab == ci].cpu())
        out = {}
        for name, per in acc.items():
            cat = {c: torch.cat(v) for c, v in per.items()}
            out[name] = {c: round(t.mean().item(), 4) for c, t in cat.items()}
            neg = cat["filler"][torch.randperm(len(cat["filler"]))[:20000]]
            out[name]["auc_d1_vs_filler"] = round(auc(cat["d1"], neg), 3)
            out[name]["auc_d1_vs_d234"] = round(auc(cat["d1"], cat["d234"]), 3)
        report[filler] = out
        print(filler, json.dumps(out), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
