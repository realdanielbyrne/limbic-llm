"""Round-4 diagnostic: why does W1g's start store select the queried code with 2 pairs but not 3?

Loads trained W1g checkpoints and, on frozen windowed Qwen3 needle batches with P pairs, measures
the start store G at the positions that matter (no training):

  keys      cosine between the facts' G keys at their space tokens (are the codes separated?),
            and how often the query's G key is closest to the queried fact's key
  gates     G's write strength β and decay α at the facts' first digits, where the start is written
  retrieval which fact's first-digit value G's read returns at the query's space token
            (cosine of the read with each fact's digit-1 value), and how often it is the queried one
  overlap   the k-WTA support overlap between facts' keys (shared active units)
  chain     which fact's first-digit value the chain store S returns at the query
  accuracy  the full model's digit-1 accuracy with the side channels on, split by which fact
            (first, middle, last) was queried

Finding that prompted the last two (2 Nov): the query's G key is closest to the *first* fact's
key whatever fact is queried, so the 2-pair solution may be "G holds the first fact, S the last".

    python experiments/evolve/diag_goal.py --layer 21 \
        --ckpt results/evolve/round3/r3_w1g_t21_pairs2.pt --ckpt results/evolve/round3/r3_w1g_t21_pairs3.pt \
        --pairs 2 3 --out results/evolve/round4/diag_goal.json
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
from evolve.probe_cue import sample  # noqa: E402
from evolve.w1g_goal import StartLoop, W1gSide, split_args  # noqa: E402
from limbic_llm.gated_delta import _gates, _run  # noqa: E402


@torch.no_grad()
def analyse(side: W1gSide, h: torch.Tensor, rows, P: int) -> dict:
    loop: StartLoop = side.loop
    m = loop.mem
    x = loop.norm(side.time(h)[0] if side.time is not None else h)
    b = loop.goal_keys(x)                                          # (B, H, T, gk)
    v = m._heads(m.wv(x), m.dv)                                    # (B, H, T, dv)
    ga, gb = _gates(loop.g_alpha, loop.g_beta, x)                  # (B, H, T)
    og, _, _ = _run(b, StartLoop._shift(b), v, ga, gb, None, m.chunk_size)
    B = len(rows)
    ar = torch.arange(B, device=h.device)
    fs = torch.tensor([r[3] for r in rows], device=h.device)       # facts' space tokens (B, P)
    qs = torch.tensor([r[4] for r in rows], device=h.device)       # query space token (B,)
    tgt = torch.tensor([r[2] for r in rows], device=h.device)
    bf = b[ar[:, None], :, fs]                                     # (B, P, H, gk)
    bq = b[ar, :, qs]                                              # (B, H, gk)
    iu = torch.triu_indices(P, P, 1)
    cos_ff = (bf[:, iu[0]] * bf[:, iu[1]]).sum(-1).mean((1, 2))    # mean over fact pairs and heads
    act = bf > 0
    overlap = (act[:, iu[0]] & act[:, iu[1]]).float().sum(-1).mean((1, 2)) / act.float().sum(-1).mean((1, 2))
    sim_q = (bf * bq[:, None]).sum(-1).mean(-1)                    # (B, P)
    key_hit = (sim_q.argmax(-1) == tgt).float()
    d1 = fs + 1                                                    # first digit of each fact
    vd1 = v[ar[:, None], :, d1]                                    # (B, P, H, dv)
    read = og[ar, :, qs]                                           # (B, H, dv), read at the query space
    sim_r = F.cosine_similarity(read[:, None], vd1, dim=-1).mean(-1)   # (B, P)
    read_hit = (sim_r.argmax(-1) == tgt).float()
    later = sim_r.argmax(-1) == (P - 1)                            # the most recently stored fact
    a = m.address(x)                                               # chain store S
    alpha, beta = _gates(m.w_alpha, m.w_beta, x)
    os_, _, _ = _run(a, StartLoop._shift(a), v, alpha, beta, None, m.chunk_size)
    sim_s = F.cosine_similarity(os_[ar, :, qs][:, None], vd1, dim=-1).mean(-1)
    return {
        "g_read_fact_index_hist": torch.bincount(sim_r.argmax(-1), minlength=P).tolist(),
        "s_read_fact_index_hist": torch.bincount(sim_s.argmax(-1), minlength=P).tolist(),
        "fact_key_cos": cos_ff.mean().item(),
        "fact_key_active_overlap": overlap.mean().item(),
        "query_key_closest_to_queried_fact": key_hit.mean().item(),
        "read_closest_to_queried_digit1": read_hit.mean().item(),
        "read_closest_to_last_fact_digit1": later.float().mean().item(),
        "beta_at_digit1": gb[ar[:, None], :, d1].mean().item(),
        "beta_mean_all": gb.mean().item(),
        "alpha_at_digit1": ga[ar[:, None], :, d1].mean().item(),
        "goal_weight": torch.sigmoid(loop.goal_logit).item(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--ckpt", action="append", required=True, help="W1g checkpoint trained at --layer")
    ap.add_argument("--pairs", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results/evolve/round4/diag_goal.json")
    args = ap.parse_args(argv)
    torch.manual_seed(0)
    pa = pl.parse_args(["--model", args.model, "--dtype", "bfloat16", "--device", "cuda"])
    model, tok = pl.load_model(pa)
    d = pl.text_config(model).hidden_size
    lm = pl.HookedLM(model, pl.SideChannels(d, memory=False, time=False), args.layer, 64, "block", "all", 1,
                     pad_id=tok.eos_token_id)
    grab = {}
    lm.blocks[args.layer].register_forward_hook(
        lambda mod, i, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o).float()))
    sides = {}
    for path in args.ckpt:
        side = W1gSide(d, type("A", (), {"time": True})(), split_args([])[0]).cuda().eval()
        side.load_state_dict(torch.load(path, map_location="cuda"))
        sides[path] = side
    out = {"args": vars(args), "results": {}}
    for P in args.pairs:
        task = pl.NeedleTask(tok, 512, P, 1)
        rng = random.Random(4321 + P)
        acc = {path: [] for path in sides}
        by_q = {path: torch.zeros(P, 2) for path in sides}         # [queried index] -> (correct, n)
        for _ in range(args.n // args.batch):
            rows = [sample(task, rng) for _ in range(args.batch)]
            ids = torch.tensor([r[0] for r in rows])
            with torch.no_grad(), lm.condition(64, False):
                lm(ids)
            h = grab["h"]
            for path, side in sides.items():
                acc[path].append(analyse(side, h, rows, P))
                lm.side = side                                     # full model with this side channel
                with torch.no_grad(), lm.condition(64, True):
                    lg = lm(ids)
                lm.side = pl.SideChannels(d, memory=False, time=False)
                for i, r in enumerate(rows):
                    by_q[path][r[2], 0] += float(lg[i, r[4]].argmax().item() == r[0][r[4] + 1])
                    by_q[path][r[2], 1] += 1
        for path, rs in acc.items():
            mean = {k: (sum(r[k] for r in rs) / len(rs) if not isinstance(rs[0][k], list)
                        else [sum(x) for x in zip(*(r[k] for r in rs))]) for k in rs[0]}
            mean["digit1_acc_by_queried_fact"] = (by_q[path][:, 0] / by_q[path][:, 1]).tolist()
            mean["chance_queried"] = 1 / P
            out["results"][f"pairs{P} | {Path(path).name}"] = mean
            print(f"pairs {P} | {Path(path).name}: " + ", ".join(
                f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in mean.items()), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
