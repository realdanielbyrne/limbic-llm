"""CPU tests for experiments/ca2/rank13.py (round 13 step 0 runner): the statistics, the RG-LRU write hook, and the
runner end to end on a tiny random RecurrentGemma (byte tokenizer, synthetic filler, bag-of-words retriever)."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from ca2 import rank13 as R  # noqa: E402


def test_auc_and_cluster_bootstrap():
    rng = np.random.default_rng(0)
    s = np.r_[rng.normal(1, 1, 300), rng.normal(0, 1, 300)]
    y = np.r_[np.ones(300, bool), np.zeros(300, bool)]
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]])
    assert abs(R.auc(s, y) - brute) < 1e-12
    assert R.auc([1, 1, 2], [1, 0, 0]) == 0.25                     # ties count one half
    assert R.auc([1, 2], [1, 1]) is None
    b = R.auc_boot(s, y, np.arange(600) // 6, n=200)
    assert b["lo"] < b["auc"] < b["hi"] and abs(b["auc"] - brute) < 1e-12


def test_holm_and_youden():
    adj = R.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06, "d": None}
    sal = np.r_[np.full(50, 5.0), np.full(500, 1.0)] + np.random.default_rng(0).normal(0, 0.1, 550)
    lab = np.r_[np.ones(50, bool), np.zeros(500, bool)]
    th, j = R.youden(sal, lab)
    assert 1.3 < th < 4.7 and j > 0.95


def test_family_assignment():
    assert R.family("qk.nope.soft*attn") == "qk" and R.family("retriever.bge") == "retriever"
    assert R.family("retriever.qwen3_4b") == "retriever_upper" and R.family("infllm*rglru") == "infllm"
    assert R.family("dentate_ctx@0.5*dentate") == "circuit" and R.family("cos_early.max") == "circuit"
    assert R.family("oracle_queried") is None and R.family("random") is None


def test_rglru_write_hook_matches_the_module():
    from transformers import RecurrentGemmaConfig, RecurrentGemmaForCausalLM
    torch.manual_seed(0)
    cfg = RecurrentGemmaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                               num_attention_heads=4, num_key_value_heads=1, head_dim=16, lru_width=64,
                               attention_window_size=8, partial_rotary_factor=0.5)
    cfg._attn_implementation = "sdpa"
    m = RecurrentGemmaForCausalLM(cfg).eval()
    cap = R.Capture(m, early=1, deep=4)
    lru = m.model.layers[0].temporal_block.rg_lru
    seen = {}
    h = lru.register_forward_pre_hook(lambda mod, a: seen.__setitem__("x", a[0].clone()))
    out = cap.run(torch.randint(3, 97, (1, 20)))
    h.remove()
    cap.remove()
    assert cap.rec == [0, 1, 3, 4] and out["lru_n"] == 4
    x = seen["x"]
    H, bw = lru.num_attention_heads, lru.block_width
    xr = x.reshape(20, H, bw)
    ig = torch.sigmoid(torch.einsum("thi,hij->thj", xr, lru.input_gate_weight) + lru.input_gate_bias)
    rg = torch.sigmoid(torch.einsum("thi,hij->thj", xr, lru.recurrent_gate_weight) + lru.recurrent_gate_bias)
    a2 = torch.exp(2 * -8.0 * rg * torch.nn.functional.softplus(lru.recurrent_param).view(H, bw))
    w0 = (ig * torch.sqrt(1 - a2)).reshape(20, -1).mean(-1)
    cap.out = {}
    cap._lru(lru, (x, torch.arange(20)[None]))                       # the hook on layer 0's own input
    assert torch.allclose(cap.out["lru_sum"][0], w0, atol=1e-5)
    assert torch.isfinite(out["lru_sum"]).all() and ((out["lru_sum"] / 4) <= 1).all()


@pytest.fixture()
def filler(tmp_path):
    r = random.Random(0)
    words = "river mill town market church railway station bridge harbour castle valley forest garden".split()
    verbs = "ran grew stood opened closed crossed faced joined".split()
    lines = ["".join(" the %s %s the %s ." % (r.choice(words), r.choice(verbs), r.choice(words))
                     for _ in range(r.randint(4, 12))) for _ in range(300)]
    p = tmp_path / "filler.txt"
    p.write_text("\n".join(lines))
    return p


SUMMARY_KEYS = {"n_items", "scorers", "table", "primary_pool", "winners", "best_baseline",
                "circuit_vs_best_baseline", "holm_families", "auc", "stop_rule", "sanity"}


def test_runner_tiny_end_to_end(filler, tmp_path):
    out = tmp_path / "rank.json"
    argv = ["--tiny", "--device", "cpu", "--n", "4", "--batch", "2", "--retriever", "bow",
            "--cell", "beyond,inside", "--filler", str(filler), "--out", str(out),
            "--trace-dir", str(tmp_path / "tr"), "--boot", "200"]
    rep = R.main(argv)
    saved = json.loads(out.read_text())
    assert SUMMARY_KEYS <= set(saved["summary"])
    assert {"args", "git", "versions", "model", "compat", "calibration", "items"} <= set(saved)
    assert len(saved["items"]) == 5 * 2 * 4
    sm = saved["summary"]
    assert sm["sanity"]["oracle_queried_recall"] == 1.0 and sm["sanity"]["nan_items"] == 0
    assert set(sm["winners"]) >= {"qk", "retriever", "infllm", "circuit"}
    for k in ("salience.dentate", "salience.rglru", "salience.attn", "context_tag", "context_tag.matched"):
        assert sm["auc"][k]["auc"] is not None
    assert "comparator@0" in sm["auc"] and "selection_not_missing" in sm["stop_rule"]
    names = saved["scorer_names"]
    ix = {n: j for j, n in enumerate(names)}
    row = saved["items"]["update/beyond/0/0"]
    assert len(row["rk"]) == len(row["cb"]) == len(row["rk_old"]) == len(row["cb_old"]) == len(names)
    assert row["rk"][ix["oracle_queried"]] == 1 and row["cb"][ix["oracle_queried"]] == 0 and row["q_len"] > 0
    assert "rk_old" not in saved["items"]["decoy/beyond/0/0"]
    for name in ("qk.ord.soft*dentate", "infllm*attn", "dentate_ctx@0.1*rglru", "retriever.bow", "position",
                 "qk.nope.max.foot", "qk.nope.soft.qfoot", "infllm.foot", "qk_layer.L2", "qk_head.L5.H3",
                 "dentate_ctx_z@0.25", "cos_early.max*surprisal", "dentate*novelty@16", "qk.nope.max*key_novelty",
                 "dentate*early_norm"):
        assert name in ix, name
    assert not any("qk_head" in n and "*" in n for n in names)        # no salience products of unit members
    # the position reference: the new statement is always above the old one
    assert row["rk"][ix["position"]] < row["rk_old"][ix["position"]]
    # cols_before agrees with the ranks: nothing before rank 1, at least one column per span before
    d = saved["items"]["decoy/beyond/0/0"]
    assert all((r == 1) == (c == 0) and c >= r - 1 for r, c in zip(d["rk"], d["cb"]))
    cal = saved["calibration"]
    assert cal["signal"] in R.GATE_SIGNALS and cal["signal"] == max(
        (k for k, v in cal["token_aucs"].items() if v is not None), key=lambda k: cal["token_aucs"][k])
    assert {"threshold", "youden_j", "token_auc", "gate_rate"} <= set(cal)
    # DEC18 measures
    assert {"winners_mrr", "best_baseline_mrr", "circuit_vs_best_baseline_mrr", "holm_families_mrr",
            "stop_rule_dec18", "update_new_above_old", "metrics"} <= set(sm)
    ent = sm["table"]["qk.nope.soft"]["decoy/beyond/dev"]
    assert {"recall", "mrr", "recall@1", "recall@2", "recall@4", "recall@8"} <= set(ent)
    assert ent["recall@1"]["mean"] <= ent["recall@2"]["mean"] <= ent["recall@4"]["mean"] <= ent["recall@8"]["mean"]
    assert sm["table"]["oracle_queried"]["primary/beyond/held"]["recall@1"]["mean"] == 1.0
    g = [r for r in saved["items"].values() if r["type"] == "paraphrase" and r["split"] == "dev"]
    want = np.mean([r["rk"][ix["infllm"]] == 1 for r in g])
    assert abs(sm["table"]["infllm"]["paraphrase/beyond/dev"]["recall@1"]["mean"] - want) < 1e-4
    budget = saved["args"]["budget"]
    want = np.mean([r["cb"][ix["dentate"]] + r["q_len"] <= budget for r in g])
    assert abs(sm["table"]["dentate"]["paraphrase/beyond/dev"]["recall"]["mean"] - want) < 1e-4
    for b in (32, 64, 128):
        want = np.mean([r["cb"][ix["dentate"]] + r["q_len"] <= b for r in g])
        assert abs(sm["table"]["dentate"]["paraphrase/beyond/dev"][f"recall@cols{b}"]["mean"] - want) < 1e-4
    dev = "primary/beyond/dev"
    for fam, w in sm["winners_mrr"].items():                       # chosen on dev MRR
        cands = [s for s in sm["scorers"] if R.family(s) == fam]
        assert w["dev"]["mrr"]["mean"] == max(sm["table"][s][dev]["mrr"]["mean"] for s in cands)
        assert {"recall@1", "mrr", "recall"} <= set(w["held"])
    assert set(sm["holm_families_mrr"]) == {"top1", "mrr", "recall"}
    assert all("p_holm" in v for v in sm["holm_families_mrr"]["top1"].values())
    c = next(iter(sm["circuit_vs_best_baseline_mrr"].values()))
    assert set(c["decoy/beyond/held"]) == {"top1", "mrr", "recall"}
    sr = sm["stop_rule_dec18"]
    assert {"plain_selector", "selection_not_missing", "parts", "parts_below_0.6"} <= set(sr)
    assert all({"auc", "lo", "hi", "below_0.6"} <= set(v) for v in sr["parts"].values())
    assert all("held_top1" in v for v in sr["plain_selector"].values())
    nao = sm["update_new_above_old"]["qk.nope.soft"]["update/beyond/all"]
    gu = [r for r in saved["items"].values() if r["type"] == "update" and r["cell"] == "beyond"]
    j = ix["qk.nope.soft"]
    assert abs(nao["mean"] - np.mean([r["rk"][j] < r["rk_old"][j] for r in gu])) < 1e-4
    assert all(v["mean"] == 1.0 for v in sm["update_new_above_old_position"].values())
    assert {"qk_layer", "qk_head"} <= set(sm["winners_mrr"]) and R.family("position") == "reference"
    assert set(sm["stop_rule_dec18"]["plain_selector"]) == {"qk", "qk_layer", "qk_head", "retriever"}
    sal = {f"salience.{k}" for k in ("dentate", "rglru", "attn", "novelty@4", "novelty@16", "novelty@64",
                                     "key_novelty", "surprisal", "early_norm")}
    assert sal <= set(sm["stop_rule_dec18"]["parts"])
    assert sm["best_baseline_mrr"] in {w["scorer"] for f, w in sm["winners_mrr"].items() if f in R.PLAIN}
    assert "stop_rule" in sm and "winners" in sm                    # the recall@budget versions are kept
    text = out.read_text()                                          # labels and scores only: no item text
    assert "Question" not in text and "the code for" not in text.lower() and "river" not in text
    assert (tmp_path / "tr" / "decoy__beyond__0__0.npz").exists()
    assert rep["summary"]["n_items"] == 40
    # summarise-only: no model, rows untouched, the same summary
    before = json.loads(out.read_text())
    rep3 = R.main(["--summarise-only", "--out", str(out), "--boot", "200"])
    assert rep3["items"] == before["items"] and rep3["args"] == before["args"]
    assert rep3["summary"]["winners_mrr"] == before["summary"]["winners_mrr"]
    # resume: nothing to run, same items
    rep2 = R.main(argv + ["--resume"])
    assert rep2["items"] == saved["items"]
    # --force drops a stored (stale) calibration
    data = json.loads(out.read_text())
    data["calibration"] = {"signal": "dentate", "threshold": 12345.0, "uids": []}
    out.write_text(json.dumps(data))
    rep4 = R.main(["--tiny", "--device", "cpu", "--n", "1", "--batch", "2", "--retriever", "bow", "--types", "decoy",
                   "--cell", "beyond", "--filler", str(filler), "--out", str(out), "--trace-dir", str(tmp_path / "tr"),
                   "--boot", "200", "--force"])
    assert rep4["calibration"]["threshold"] != 12345.0 and len(rep4["items"]) == 1
    # other settings into the same file: refused
    with pytest.raises(SystemExit):
        R.main(argv + ["--resume", "--lambda", "0,1"])
    with pytest.raises(SystemExit):                                 # an existing file without --resume / --force
        R.main(argv)


def test_summarise_reads_step0_rows(tmp_path):
    """--summarise-only on a step-0 file (``ranks`` dicts, no span lengths): no errors, column budgets skipped."""
    rng = np.random.default_rng(0)
    items = {}
    for typ in ("decoy", "paraphrase", "update"):
        for i in range(8):
            ranks = {}
            for s in ("qk.nope.soft", "retriever.bge", "infllm", "dentate_ctx@0", "oracle_queried", "random"):
                r = 1 if s == "oracle_queried" else int(rng.integers(1, 30))
                v = [r, int(r <= 20)]
                if typ == "update":
                    ro = int(rng.integers(1, 30))
                    v += [ro, int(ro <= 20)]
                ranks[s] = v
            items[f"{typ}/beyond/0/{i}"] = {"uid": f"{typ}/beyond/0/{i}", "type": typ, "cell": "beyond",
                                            "split": "dev" if i % 2 else "held", "ranks": ranks,
                                            "comparator": {"0": [float(rng.normal()), int(i % 3 == 0)]},
                                            "random_selected": 20, "n_cand": 300, "nan_scores": [],
                                            "cand_mode": "eligible"}
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"args": {"budget": 512, "seed": 0}, "items": items,
                             "trace_dir": str(tmp_path / "none")}))
    rep = R.main(["--summarise-only", "--out", str(p), "--boot", "100"])
    sm = rep["summary"]
    assert "recall@cols32" not in sm["table"]["infllm"]["decoy/beyond/dev"]
    assert "not computable" in sm["metrics"]["column_budgets"]
    assert sm["winners_mrr"]["qk"]["scorer"] == "qk.nope.soft" and "stop_rule_dec18" in sm
    assert "update_new_above_old" in sm and sm["sanity"]["oracle_queried_recall"] == 1.0


def test_surprisal_matches_the_model_and_is_causal():
    from types import SimpleNamespace
    from transformers import RecurrentGemmaConfig, RecurrentGemmaForCausalLM
    torch.manual_seed(0)
    cfg = RecurrentGemmaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                               num_attention_heads=4, num_key_value_heads=1, head_dim=16, lru_width=64,
                               attention_window_size=8, partial_rotary_factor=0.5)
    cfg._attn_implementation = "sdpa"
    m = RecurrentGemmaForCausalLM(cfg).eval()
    cap = R.Capture(m, early=1, deep=4)
    fake = SimpleNamespace(model=m)
    ids = torch.randint(3, 97, (1, 40))
    out = cap.run(ids)
    s = R.Ranker.surprisal(fake, out["final"][0], ids[0].tolist(), chunk=7)
    with torch.no_grad():
        lp = torch.log_softmax(m(ids, use_cache=False).logits[0].float(), -1)
    want = torch.cat([torch.zeros(1), -lp[:-1].gather(1, ids[0, 1:, None])[:, 0]])
    assert torch.allclose(s, want, atol=1e-4)
    ids2 = ids.clone()
    ids2[0, 25:] = torch.randint(3, 97, (15,))
    s2 = R.Ranker.surprisal(fake, cap.run(ids2)["final"][0], ids2[0].tolist(), chunk=7)
    assert torch.allclose(s2[:25], s[:25], atol=1e-6)               # surprisal at t ≤ 24 ignores tokens ≥ 25
    cap.remove()
