"""Round-11 pre-GPU check: does W1s's learned keep-scorer (``--w1s-keep M --w1s-keep-by learned``) find the
code digits? (runbook §3: "a CPU toy in which the learned keep-scorer finds the code digits within 300 steps").

The world is the one-hot toy of ``tests/test_evolve_w1s.py`` (three facts [name, is, space, d, d, d, d] in
filler, 12 false windows, oracle onsets at each fact's first digit, then a query), but tokens are dense
random embeddings (D = 128), as LM states are, not one-hots: the LayerNorm of a one-hot puts −1/V on every
other token, which couples every score to every update and, under Adam, turns that harmless uniform shift
into noise. Digit tokens share a component ``--share`` × u (u a random direction), standing in for the
"digit-ness" a real hidden state carries (in the I1 checkpoints the value norm alone puts the needed
entries in the top 8 of 24 every time; round-11 plan, NOTEBOOK "Round 11 planning").

Only the scorer trains (``keep_w``: a score w·x_{j+1} + b of each transition's successor); the read is the
module's own (``SlotLoop._features → _store → _read``), with lexical keys and lexical alignment, K = 16, and
M = 4 of the 23 transitions per window kept. The loss is the cross-entropy of the recalled value against
each stored token's value at the query's four answer positions. The scorer learns only through σ(score)
on the values it already keeps (selection is hard), so it must discover digits by eliminating what it
keeps wrongly.

Reported per seed: the held-out tag hit, i.e. the share of kept transitions in the facts' windows whose
successor is one of the fact's digits (1.0 = it keeps exactly space→d1, d1→d2, d2→d3, d3→d4), and the
held-out digit accuracy of the read. Chance tag hit is about 4/23 ≈ 0.17.

    python experiments/evolve/toy_keep.py --seeds 10 --steps 300 --out results/evolve/round11/stage0/toy_keep.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evolve.w1s_slots import W1sSide, split_args  # noqa: E402

V = 64
NAMES, IS, SP, DIGITS, FILLER = [0, 1, 2, 3, 4, 5], 6, 7, list(range(8, 18)), list(range(20, 64))
FLAGS = ["--w1s-gate=oracle", "--w1s-pre=3", "--w1s-pool=4", "--w1s-key=lex", "--w1s-align=lex"]


def world(q: int, seed: int):
    """Token ids (T,), oracle onsets (1, T), per fact its digit positions, and the query position."""
    g = torch.Generator().manual_seed(seed)
    fill = lambda k: [FILLER[i] for i in torch.randint(len(FILLER), (k,), generator=g)]
    digs = [[8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 8, 13]]
    toks, onsets, facts = [], [], []
    for i, name in enumerate((0, 1, 2)):
        onsets += [len(toks) + 10, len(toks) + 22, len(toks) + 34]   # false windows in the filler
        toks += fill(60)
        facts.append([len(toks) + 3 + k for k in range(4)])
        onsets.append(len(toks) + 3)
        toks += [name, IS, SP] + digs[i]
    onsets += [len(toks) + 10, len(toks) + 22, len(toks) + 34]
    toks += fill(60)
    qpos = len(toks)
    toks += [q, IS, SP] + digs[q]
    o = torch.zeros(1, len(toks), dtype=torch.bool)
    o[0, onsets] = True
    return torch.tensor(toks), o, facts, qpos


def run(seed: int, steps: int, lr: float, share: float, batch: int, d: int = 128) -> dict:
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(V, d, generator=g)
    emb[DIGITS] += share * torch.randn(d, generator=g)
    torch.manual_seed(seed)
    side = W1sSide(d, argparse.Namespace(time=False), split_args([*FLAGS, "--w1s-keep=4", "--w1s-keep-by=learned"])[0])
    loop = side.loop
    for p in side.parameters():
        p.requires_grad_(False)
    for p in loop.keep_w.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(loop.keep_w.parameters(), lr=lr)
    vtok = F.normalize(loop.wa(loop.norm(emb)), dim=-1).detach()

    def batch_eval(step: int, base: int):
        loss, hit, tot, acc = 0.0, 0, 0, 0
        for i in range(batch):
            q = (step * batch + i) % 3
            toks, o, facts, qpos = world(q, base + step * batch + i)
            x = emb[toks][None]
            f = loop._features(loop.norm(x), x, None, o)
            st = loop._store(f)
            st.update(x=f["x"], y=f["y"], v=f["v"], p=f["p"], g=f["g"])
            r = loop._read(f, st)[0, qpos + 2:qpos + 6]
            tgt = toks[torch.tensor(facts[q])]
            logits = 20 * F.normalize(r, dim=-1) @ vtok.T
            loss = loss + F.cross_entropy(logits, tgt)
            acc += int((logits.argmax(-1) == tgt).sum())
            for ds in facts:
                k = int((st["s"][0] == ds[0]).nonzero()[0, 0])
                kept = st["mem"][0, k].nonzero().flatten().tolist()
                tot += len(kept)
                hit += sum(1 for j in kept if j + 1 in ds)
        return loss / batch, hit / max(1, tot), acc / (4 * batch)

    curve = []
    for step in range(steps):
        loss, hit, _ = batch_eval(step, 1000)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            curve.append({"step": step, "loss": round(loss.item(), 4), "tag_hit": round(hit, 4)})
    with torch.no_grad():
        _, hit, acc = batch_eval(0, 900_000)
    return {"seed": seed, "heldout_tag_hit": hit, "heldout_digit_acc": acc, "curve": curve}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--shares", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    out = {"args": vars(args), "chance_tag_hit": 4 / 23, "by_share": {}}
    for sh in args.shares:
        runs = [run(s, args.steps, args.lr, sh, args.batch) for s in range(args.seeds)]
        hits = [r["heldout_tag_hit"] for r in runs]
        out["by_share"][str(sh)] = {"runs": runs, "tag_hit_mean": sum(hits) / len(hits),
                                    "seeds_ge_0.95": sum(h >= 0.95 for h in hits), "tag_hits": hits}
        print(f"share {sh}: tag hit {[round(h, 2) for h in hits]}; ≥ 0.95 in "
              f"{out['by_share'][str(sh)]['seeds_ge_0.95']}/{args.seeds}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
