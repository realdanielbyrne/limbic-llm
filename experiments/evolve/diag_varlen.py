"""Does the declarative trigger (P44) hold for statements of variable length?

Pads the task's statement with adjectives, prepositional phrases and an interrupting clause
(" The new four-digit secret code for NAME's old blue storage unit at the harbor, as of last
Tuesday afternoon, is 4821.") inside WikiText test filler, and scores each token with the fixed
direction saved by diag_declarative.py at block 21 (windowed condition). Reports the mean
projection at the first digit, the other digits, the statement head and the filler, and the AUC
of the first digit against filler and head, for 6 lengths of head (5 to 38 tokens).

    python experiments/evolve/diag_varlen.py --text-file <wikitext test> --out results/evolve/round8/diag_varlen.json
"""
import argparse, json, random, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrained_loop as pl
from evolve.diag_declarative import tap_states
from evolve.diag_novelty import auc
ap = argparse.ArgumentParser()
ap.add_argument("--text-file", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
pa = pl.parse_args(["--model", "Qwen/Qwen3-1.7B-Base", "--dtype", "bfloat16", "--device", "cuda"])
model, tok = pl.load_model(pa)
lm = pl.HookedLM(model, None, 21, 64, "block", "all", 1, pad_id=tok.eos_token_id)
w = torch.load("results/evolve/round8/diag_declarative.direction.pt").cuda()
corpus = tok.encode(Path(args.text_file).read_text(encoding="utf-8", errors="replace"))
MODS = {  # (before "code", after NAME, before "is")
    0: ("", "", ""),
    1: ("secret ", "", ""),
    2: ("new secret ", "'s old storage unit", ""),
    3: ("new four-digit secret ", "'s old blue storage unit at the harbor", ""),
    4: ("new four-digit secret ", "'s old blue storage unit at the harbor", ", as of last Tuesday afternoon,"),
    5: ("new four-digit secret ", "'s old blue storage unit at the harbor near the station", ", as far as anyone in the office can remember, and as of last Tuesday afternoon,"),
}
rng = random.Random(7)
out = {}
for lvl, (a, b, c) in MODS.items():
    rows, pos, extra = [], [], []
    for _ in range(64):
        name = rng.choice(pl.NAMES); code = "".join(rng.choice("0123456789") for _ in range(4))
        head = tok.encode(f" The {a}code for {name}{b}{c} is"); val = tok.encode(f" {code}")
        o = rng.randrange(len(corpus) - 200)
        ids = [tok.bos_token_id] + corpus[o:o + 40] + head + val + tok.encode(".") + corpus[o + 40:o + 100]
        rows.append(ids); pos.append(1 + 40 + len(head) + 1); extra.append(len(head))
    h = tap_states(lm, rows, 21, 64)
    s = h @ w
    d1 = torch.stack([s[i, p] for i, p in enumerate(pos)]).cpu()
    d234 = torch.cat([s[i, p + 1:p + 4] for i, p in enumerate(pos)]).cpu()
    hd = torch.cat([s[i, p - 1 - L:p - 1] for i, (p, L) in enumerate(zip(pos, extra))]).cpu()
    fill = torch.cat([torch.cat([s[i, 1:41], s[i, p + 5:len(rows[i])]]) for i, p in enumerate(pos)]).cpu()
    out[lvl] = {"head_tokens": round(sum(extra) / len(extra), 1), "d1": round(d1.mean().item(), 2),
                "d234": round(d234.mean().item(), 2), "head": round(hd.mean().item(), 2), "filler": round(fill.mean().item(), 2),
                "auc_d1_vs_filler": round(auc(d1, fill), 3), "auc_d1_vs_head": round(auc(d1, hd), 3)}
    print(lvl, json.dumps(out[lvl]), flush=True)
Path(args.out).write_text(json.dumps(out, indent=2))
print(f"saved {args.out}")
