"""Is "a value is being asserted for a key" a usable encoding trigger? A no-training test.

No novelty measure picks out the facts' first digits (diag_novelty.py, diag_surprise.py: the codes
are less surprising than the filler). People encode "The code for NAME is 4821" because of what the
sentence *means*: a declarative statement binding a value to a key. The frozen LLM represents that
meaning. This asks whether one fixed direction in the tap's hidden state, found **without the
task's template**, marks the onset of an asserted value:

  contrast set  generic declarative bindings ("Marisol's phone number is 5550193.", "The password to
                the vault is falcon.", ...) embedded in WikiText *train* text. The templates never
                contain "code for", the names are disjoint from the task's NAMES, and the values mix
                digits, words and alphanumerics. Positives: each value's first content token;
                negatives: every other token (the rest of the statement and the surrounding text)
  direction     the difference of class means of layer-normed h at the tap (windowed condition),
                fit on 12 templates; checked on the 4 held-out templates
  task test     on the scattered needle (held-out template, NAMES, WikiText *test* or random
                filler): the AUC of the projection for each fact's first digit against filler,
                against the filler's own digit tokens (years, counts), against the other code
                digits, and the fact head

A direction like this could initialise one head of W1i's event detector (w_e), or enter its logit
as a fixed prior, which would supply the event part of the three parts that must co-adapt.

    python experiments/evolve/diag_declarative.py --train-text <wikitext train> --text-file <wikitext test> \
        --out results/evolve/round8/diag_declarative.json
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

PEOPLE = ["Marisol", "Kenji", "Olu", "Priya", "Lars", "Aisha", "Tomas", "Mei", "Rafael", "Ingrid", "Kwame",
          "Sofia", "Dmitri", "Leila", "Mateo", "Hana", "Arjun", "Zofia", "Emeka", "Yusuf"]
THINGS = ["vault", "server", "garage", "office safe", "storage unit", "laptop", "shed", "router", "gym locker"]
TEMPLATES = [                                                          # {p} person, {t} thing, {v} value
    "{p}'s phone number is {v}.", "The password to the {t} is {v}.", "{p}'s locker combination is {v}.",
    "The PIN for {p}'s card is {v}.", "{p}'s flight leaves from gate {v}.", "{p}'s badge ID is {v}.",
    "The meeting with {p} is in room {v}.", "The serial number of the {t} is {v}.",
    "{p}'s favourite colour is {v}.", "The key to the {t} is hidden under the {v}.",
    "{p}'s account number is {v}.", "The Wi-Fi network for the {t} is called {v}.",
    # held out for checking the direction
    "{p}'s apartment number is {v}.", "The combination of the {t} is {v}.", "{p}'s ticket number is {v}.",
    "The secret word for the {t} is {v}.",
]
WORDS = ["falcon", "blue", "marble", "doormat", "orchid", "lantern", "copper", "willow", "nebula", "harbor"]


def value(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.5:
        return "".join(rng.choice("0123456789") for _ in range(rng.randint(2, 7)))
    if r < 0.8:
        return rng.choice(WORDS)
    return rng.choice("ABCDEFGHKMNPRSTX") + "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 3)))


def contrast(tok, corpus, rng, templates, n, ctx=96):
    """Sequences of WikiText with one declarative binding inside; the value's first content token is marked."""
    rows, pos = [], []
    for _ in range(n):
        tpl = rng.choice(templates)
        pre, post = tpl.split("{v}")
        pre = pre.format(p=rng.choice(PEOPLE), t=rng.choice(THINGS))
        head, val = tok.encode(" " + pre.rstrip()), tok.encode(" " + value(rng))
        first = 1 if tok.tok.decode(val[:1]).strip() == "" else 0     # " 4821" → " ", "4", ...
        o = rng.randrange(len(corpus) - 2 * ctx)
        ids = [tok.bos_token_id] + corpus[o:o + ctx] + head + val + tok.encode(post) + corpus[o + ctx:o + 2 * ctx]
        rows.append(ids[:2 * ctx + 32])
        pos.append(1 + ctx + len(head) + first)
    return rows, pos


@torch.no_grad()
def tap_states(lm, rows, layer, window):
    L = max(map(len, rows))
    ids = torch.tensor([r + [lm.pad_id] * (L - len(r)) for r in rows])
    cap = {}
    hook = lm.blocks[layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).float()))
    with lm.condition(window, False):
        lm(ids, keep=1)
    hook.remove()
    return F.layer_norm(cap["h"], (cap["h"].shape[-1],))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--train-text", required=True, help="filler for the contrast set (WikiText train)")
    ap.add_argument("--text-file", required=True, help="filler for the task test (WikiText test)")
    ap.add_argument("--n-contrast", type=int, default=768)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    assert not set(PEOPLE) & set(pl.NAMES), "contrast names must be disjoint from the task's"
    pa = pl.parse_args(["--model", args.model, "--dtype", "bfloat16", "--device", "cuda"])
    model, tok = pl.load_model(pa)
    lm = pl.HookedLM(model, None, args.layer, args.window, "block", "all", 1, pad_id=tok.eos_token_id)
    rng = random.Random(args.seed)
    train_corpus = tok.encode(Path(args.train_text).read_text(encoding="utf-8", errors="replace")[:3_000_000])

    def collect(templates, n):
        pos_h, neg_h = [], []
        for _ in range(n // args.batch):
            rows, pos = contrast(tok, train_corpus, rng, templates, args.batch)
            h = tap_states(lm, rows, args.layer, args.window)
            for i, (r, p) in enumerate(zip(rows, pos)):
                pos_h.append(h[i, p])
                keep = [t for t in range(1, len(r)) if t != p]
                neg_h.append(h[i, keep])
        return torch.stack(pos_h), torch.cat(neg_h)

    P_fit, N_fit = collect(TEMPLATES[:12], args.n_contrast)
    w = F.normalize(P_fit.mean(0) - N_fit.mean(0), dim=0)
    P_ho, N_ho = collect(TEMPLATES[12:], args.n_contrast // 3)
    report = {"layer": args.layer, "contrast_fit_auc": round(auc((P_fit @ w).cpu(), (N_fit @ w).cpu()[:20000]), 3),
              "contrast_heldout_auc": round(auc((P_ho @ w).cpu(), (N_ho @ w).cpu()[:20000]), 3)}
    print(json.dumps(report), flush=True)

    corpus = tok.encode(Path(args.text_file).read_text(encoding="utf-8", errors="replace"))
    for filler, corp in (("natural", corpus), ("random", None)):
        task = ScatterNeedleTask(tok, 512, 3, 1, corp, min_gap=8, block=args.window)
        trng = random.Random(args.seed + 1)
        score = {c: [] for c in CLASSES}
        filler_digits, after_copula = [], []
        for _ in range(args.n // args.batch):
            rows, labs = [], []
            for _ in range(args.batch):
                ids, n_ans, _ = task.sample(trng)
                rows.append(ids)
                labs.append(labels(len(ids), task.sink, task.last_value_at, task.last_heads,
                                   task.last_query_len + n_ans))
            s = (tap_states(lm, rows, args.layer, args.window) @ w).cpu()
            lab = torch.stack(labs)
            for ci, c in enumerate(CLASSES):
                score[c].append(s[lab == ci])
            dig = torch.tensor([[tok.tok.decode([t]).strip().isdigit() for t in r] for r in rows])
            filler_digits.append(s[(lab == CLASSES.index("filler")) & dig])
            cop = torch.tensor([[False] + [tok.tok.decode([t]).strip() in ("is", "was", "are", "were") for t in r[:-1]]
                                for r in rows])                          # token right after a copula
            after_copula.append(s[(lab == CLASSES.index("filler")) & cop])
        cat = {c: torch.cat(v) for c, v in score.items()}
        fd = torch.cat(filler_digits)
        fill = cat["filler"][torch.randperm(len(cat["filler"]))[:20000]]
        thr = torch.quantile(fill, 0.99)                                 # a detector firing on 1% of filler
        out = {"mean": {c: round(t.mean().item(), 3) for c, t in cat.items()},
               "auc_d1_vs_filler": round(auc(cat["d1"], fill), 3),
               "auc_d1_vs_filler_digits": round(auc(cat["d1"], fd), 3) if len(fd) else None,
               "n_filler_digits": len(fd),
               "auc_d1_vs_filler_after_copula": round(auc(cat["d1"], torch.cat(after_copula)), 3)
               if len(torch.cat(after_copula)) else None,
               "n_filler_after_copula": len(torch.cat(after_copula)),
               "auc_d1_vs_d234": round(auc(cat["d1"], cat["d234"]), 3),
               "auc_d1_vs_head": round(auc(cat["d1"], cat["head"]), 3),
               "d1_recall_at_1pct_filler": round((cat["d1"] > thr).float().mean().item(), 3),
               "d234_rate_at_that_threshold": round((cat["d234"] > thr).float().mean().item(), 3)}
        report[filler] = out
        print(filler, json.dumps(out), flush=True)
    torch.save(w.cpu(), Path(args.out).with_suffix(".direction.pt"))
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
