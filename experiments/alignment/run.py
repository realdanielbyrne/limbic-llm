"""Alignment track, base experiment: harm probe (A1) and steering arms (A2, A6).

Pipeline
  1. Activations at the tap block, at the last prompt token, for harmful and harmless
     training prompts.
  2. A1: fit the harm probe on the *target* model's activations and report held-out
     AUROC and calibration (optionally at every layer, ``--layer-sweep``).
  3. The refusal direction: harmful minus harmless means from ``--direction-from``
     (an aligned sibling, for A6) or from the target itself.
  4. For each arm, generate greedy responses to the evaluation sets with the
     controller attached, and record:
       - refusal rate on harmful, XSTest-unsafe ("contrast"), XSTest-safe
         ("borderline") and harmless prompts;
       - degeneration;
       - response length;
       - the controller's gain statistics;
       - optionally guard-model unsafe rates and perplexity on held-out text.

Arms
  none           no hook
  static         h ← h + c‖h‖d̂ everywhere (activation addition / CAA)
  conditional    the same, only if the prompt's harm probability ≥ θ (CAST-style)
  graded         G·tanh(κE_t), E_t a leaky integral of the probe score (the controller)
  graded_self    graded, with d̂ estimated from the target model alone

Headline numbers: refusal on harmful (higher is better); refusal on borderline and
harmless (lower is better); selectivity = harmful − borderline; degeneration (the
"freeze"); perplexity (capability cost).

Data rule: the results JSON holds labels, scores and aggregates only. Response
text is written only with ``--private-dir``, which must lie outside the repository.

Usage
  # offline smoke test
  OMP_NUM_THREADS=1 python experiments/alignment/run.py --tiny --n-train 32 --n-eval 8 --max-new-tokens 8

  # A1 + A6 on the GPU machine
  python experiments/alignment/run.py --model mlabonne/Qwen3-1.7B-abliterated \\
      --direction-from Qwen/Qwen3-1.7B --dtype bfloat16 --device cuda --layer-sweep \\
      --ppl-data hf:Salesforce/wikitext:wikitext-2-raw-v1:test:text \\
      --private-dir ~/limbic_private/a6 --name a6_qwen3_abliterated
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent), str(ROOT)]

from data import DEFAULTS, batches, format_prompt, load_prompts, pad_left  # noqa: E402
from judge import GuardJudge, assert_private, degeneration, refused  # noqa: E402
from probe import activations, auroc, diff_of_means, predict, probe_report, residual_direction  # noqa: E402
from steering import Attachment, SteeringController  # noqa: E402

import pretrained_loop as pl  # noqa: E402  (model loading and block discovery)

ARMS = ("none", "static", "conditional", "graded", "graded_self")


def load(name: str | None, args, tiny_seed: int = 0):
    ns = argparse.Namespace(model=name, tiny=args.tiny, tiny_arch="gpt2", seed=tiny_seed, dtype=args.dtype,
                            device=args.device, attn_impl="sdpa")
    model, tok = pl.load_model(ns)
    return model, tok, pl.find_blocks(model)


def pad_id(tok) -> int:
    hf = getattr(tok, "tok", None)
    if hf is not None and hf.pad_token_id is not None:
        return hf.pad_token_id
    return tok.eos_token_id


def encode_all(tok, prompts, chat):
    return [format_prompt(tok, p, chat) for p in prompts]


@torch.no_grad()
def layer_sweep(model, blocks, tok, harmful, harmless, args) -> list[dict]:
    """A1 across depth: held-out probe AUROC and refusal-direction norm at every other block."""
    out = []
    toks = encode_all(tok, harmful + harmless, args.chat)
    y = torch.tensor([1] * len(harmful) + [0] * len(harmless))
    for L in range(0, len(blocks), max(1, len(blocks) // 14)):
        X = activations(model, blocks[L], batches(toks, pad_id(tok), args.batch))
        _, rep = probe_report(X, y, seed=args.seed)
        out.append({"layer": L, **rep, "refusal_dir_norm": float(diff_of_means(X, y).norm())})
        print(f"  layer {L:3d}  AUROC {rep['auroc']:.3f}  ECE {rep['ece']:.3f}", flush=True)
    return out


@torch.no_grad()
def generate(model, tok, ids, mask, ctrl, args):
    dev = model.get_input_embeddings().weight.device
    if ctrl is not None:
        ctrl.reset()
        ctrl.record = True
        ctrl.pending_mask = mask
    out = model.generate(input_ids=ids.to(dev), attention_mask=mask.to(dev), max_new_tokens=args.max_new_tokens,
                         do_sample=False, pad_token_id=pad_id(tok))
    new = out[:, ids.shape[1]:].cpu()
    stats = ctrl.summary() if ctrl is not None else {}
    return new, stats


@torch.no_grad()
def perplexity(model, tok, texts, ctrl, args) -> float:
    dev = model.get_input_embeddings().weight.device
    nll, n = 0.0, 0
    seqs = [tok.encode(t)[: args.ppl_len] for t in texts]
    seqs = [s for s in seqs if len(s) > 8]
    for ids, mask in batches(seqs, pad_id(tok), args.batch):
        if ctrl is not None:
            ctrl.reset()
            ctrl.pending_mask = mask
        lg = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False).logits.float()
        tgt, m = ids[:, 1:].to(dev), mask[:, 1:].to(dev).bool() & mask[:, :-1].to(dev).bool()
        ce = F.cross_entropy(lg[:, :-1].flatten(0, 1), tgt.flatten(), reduction="none").view_as(tgt)
        nll += ce[m].sum().item()
        n += int(m.sum())
    return math.exp(nll / max(1, n))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlabonne/Qwen3-1.7B-abliterated")
    ap.add_argument("--direction-from", default=None, help="aligned model for the refusal direction (default: target)")
    ap.add_argument("--tiny", action="store_true", help="random tiny GPT-2 and synthetic prompts (offline)")
    ap.add_argument("--dtype", choices=list(pl.DTYPES), default="float32")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    ap.add_argument("--layer", type=int, default=None, help="tap block (default n_layers // 2)")
    ap.add_argument("--layer-sweep", action="store_true", help="A1: probe AUROC across depth")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--gain", type=float, default=0.15, help="static/conditional gain, fraction of ‖h‖")
    ap.add_argument("--cap", type=float, default=0.3, help="graded cap G, fraction of ‖h‖")
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--leak", type=float, default=0.9, help="λ of the leaky integral")
    ap.add_argument("--threshold", type=float, default=0.5, help="conditional-arm probe threshold θ")
    for k in DEFAULTS:
        ap.add_argument(f"--{k.replace('_', '-')}", default=None, help=f"prompt spec (default {DEFAULTS[k]})")
    ap.add_argument("--chat", action=argparse.BooleanOptionalAction, default=True, help="use the chat template")
    ap.add_argument("--n-train", type=int, default=128, help="prompts per class for the probe and direction")
    ap.add_argument("--n-eval", type=int, default=64, help="prompts per evaluation set")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--ppl-data", default=None, help="text spec for the capability check (perplexity)")
    ap.add_argument("--n-ppl", type=int, default=64)
    ap.add_argument("--ppl-len", type=int, default=256)
    ap.add_argument("--guard", default=None, help="optional guard model id (e.g. meta-llama/Llama-Guard-3-8B)")
    ap.add_argument("--private-dir", default=None, help="where raw responses may be written (outside the repo)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}; choose from {ARMS}")
    spec = {k: getattr(args, k) or (f"synthetic:{k.split('_')[0]}" if args.tiny and k.startswith("harm") else DEFAULTS[k])
            for k in DEFAULTS}
    if args.tiny:                                       # synthetic stand-ins; XSTest-like sets reuse the classes
        spec["borderline"] = args.borderline or "synthetic:harmless"
        spec["contrast"] = args.contrast or "synthetic:harmful"
    private = assert_private(args.private_dir, ROOT) if args.private_dir else None
    t0 = time.time()

    # ---- prompts
    P = {"harmful_train": load_prompts(spec["harmful_train"], args.n_train, args.seed),
         "harmless_train": load_prompts(spec["harmless_train"], args.n_train, args.seed)}
    evals = {"harmful": load_prompts(spec["harmful_test"], args.n_eval, args.seed + 1),
             "contrast": load_prompts(spec["contrast"], args.n_eval, args.seed + 2),
             "borderline": load_prompts(spec["borderline"], args.n_eval, args.seed + 3),
             "harmless": load_prompts(spec["harmless_test"], args.n_eval, args.seed + 4)}
    y = torch.tensor([1] * len(P["harmful_train"]) + [0] * len(P["harmless_train"]))

    # ---- refusal direction from the aligned sibling (A6), if given
    report: dict = {"args": vars(args), "spec": spec}
    d_ext = None
    if args.direction_from and args.direction_from != args.model and not args.tiny:
        dm, dtok, dblocks = load(args.direction_from, args)
        L = args.layer if args.layer is not None else len(dblocks) // 2
        Xd = activations(dm, dblocks[L], batches(encode_all(dtok, P["harmful_train"] + P["harmless_train"], args.chat),
                                                 pad_id(dtok), args.batch))
        d_ext = diff_of_means(Xd, y)
        del dm, Xd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- target model, probe and its own direction
    model, tok, blocks = load(args.model, args)
    L = args.layer if args.layer is not None else len(blocks) // 2
    report["layer"], report["n_layers"] = L, len(blocks)
    if args.layer_sweep:
        report["layer_sweep"] = layer_sweep(model, blocks, tok, P["harmful_train"], P["harmless_train"], args)
    X = activations(model, blocks[L], batches(encode_all(tok, P["harmful_train"] + P["harmless_train"], args.chat),
                                              pad_id(tok), args.batch))
    probe, rep = probe_report(X, y, seed=args.seed)
    d_self = diff_of_means(X, y)
    d_ref = d_ext if d_ext is not None else d_self
    cos = lambda a, b: float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    report["probe"] = rep
    report["directions"] = {"cos_refusal_vs_probe": cos(d_ref, residual_direction(probe)),
                            "cos_external_vs_self": cos(d_ref, d_self) if d_ext is not None else None,
                            "norm_ref": float(d_ref.norm()), "norm_self": float(d_self.norm())}
    print(f"probe at block {L}: AUROC {rep['auroc']:.3f}  ECE {rep['ece']:.3f}  "
          f"cos(refusal, probe) {report['directions']['cos_refusal_vs_probe']:+.3f}", flush=True)
    guard = GuardJudge(args.guard) if args.guard else None
    ppl_texts = load_prompts(args.ppl_data, args.n_ppl, args.seed) if args.ppl_data else None

    # ---- arms
    report["arms"], records = {}, []
    dev = model.get_input_embeddings().weight.device
    for arm in arms:
        ctrl = None
        if arm != "none":
            mode = "graded" if arm.startswith("graded") else arm
            d = d_self if arm == "graded_self" else d_ref
            ctrl = SteeringController(d, probe, mode=mode, gain=args.gain, cap=args.cap, kappa=args.kappa,
                                      leak=args.leak, threshold=args.threshold).to(dev)
            att = Attachment(blocks[L], ctrl)
        res = {}
        for set_name, prompts in evals.items():
            toks = encode_all(tok, prompts, args.chat)
            flags, degs, lens, unsafe, gains, gmax = [], [], [], [], [], []
            texts = []
            for bi, (ids, mask) in enumerate(batches(toks, pad_id(tok), args.batch)):
                new, st = generate(model, tok, ids, mask, ctrl, args)
                for j in range(new.shape[0]):
                    ids_j = [int(t) for t in new[j] if int(t) != pad_id(tok)]
                    text = tok.decode(ids_j)
                    k = bi * args.batch + j
                    rec = {"arm": arm, "set": set_name, "idx": k, "refused": refused(text),
                           "degeneration": degeneration(ids_j), "n_new": len(ids_j)}
                    if st:
                        rec.update(gain_mean=float(st["gain_mean"][j]), gain_max=float(st["gain_max"][j]),
                                   capped_frac=float(st["capped_frac"][j]), E_final=float(st["E_final"][j]),
                                   s_mean=float(st["s_mean"][j]))
                    if guard is not None:
                        rec["unsafe"] = guard.unsafe(prompts[k], text)
                    records.append(rec)
                    flags.append(rec["refused"]); degs.append(rec["degeneration"]); lens.append(rec["n_new"])
                    if st:
                        gains.append(rec["gain_mean"]); gmax.append(rec["gain_max"])
                    if guard is not None:
                        unsafe.append(rec["unsafe"])
                    if private is not None:
                        texts.append({"set": set_name, "idx": k, "prompt": prompts[k], "response": text})
            mean = lambda v: float(sum(v) / len(v)) if v else None
            res[set_name] = {"refusal": mean(flags), "degenerate": mean([x > 0.5 for x in degs]),
                             "degeneration": mean(degs), "len": mean(lens), "gain_mean": mean(gains),
                             "gain_max": mean(gmax), "unsafe": mean(unsafe), "n": len(flags)}
            if private is not None:
                with (private / f"{args.name or 'run'}_{arm}.jsonl").open("a") as f:
                    for t in texts:
                        f.write(json.dumps(t) + "\n")
        if ppl_texts:
            res["ppl"] = perplexity(model, tok, ppl_texts, ctrl, args)
        res["selectivity"] = (res["harmful"]["refusal"] - res["borderline"]["refusal"])
        report["arms"][arm] = res
        if ctrl is not None:
            att.remove()
        print(f"[{arm:12s}] refusal harmful {res['harmful']['refusal']:.2f}  contrast {res['contrast']['refusal']:.2f}  "
              f"borderline {res['borderline']['refusal']:.2f}  harmless {res['harmless']['refusal']:.2f}  "
              f"degenerate {res['harmful']['degenerate']:.2f}" + (f"  ppl {res['ppl']:.2f}" if 'ppl' in res else ""),
              flush=True)
    report["records"] = records
    report["seconds"] = time.time() - t0
    out = ROOT / "results" / "alignment" / f"{args.name or ('tiny' if args.tiny else 'run')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"saved {out.relative_to(ROOT)} (labels and scores only)")
    return report


if __name__ == "__main__":
    main()
