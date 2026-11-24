"""
final_checks.py — Statistical checks on the real-LLM evaluation
================================================================
  1. Paired bootstrap (2000 resamples) of the held-out test set:
     95 % CIs for AUROC / F1 of each H-score, and for the improvement of the
     logistic H-score over the original fixed 0.5/0.5 formula.
  2. Manual audit of the automatic correctness judge (60 random answers,
     15 per benchmark, labelled by hand) → judge agreement rate.

Writes results/real_llm/final_checks.json.
Usage:  python scripts/final_checks.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "real_llm"
sys.stdout.reconfigure(encoding="utf-8")

recs = json.loads((OUT / "results_real_llm.json").read_text(encoding="utf-8"))["results"]
summ = json.loads((OUT / "metrics_summary.json").read_text(encoding="utf-8"))
test = [r for r in recs if r["split"] == "test"]
y = np.array([0 if r["correct"] else 1 for r in test])
scores = {
    "Fixed 0.5/0.5": (np.array([r["h_fixed"] for r in test]), 0.50),
    "LogReg (core)": (np.array([r["h_logreg_core"] for r in test]),
                      summ["hscore_models"]["core"]["threshold"]),
    "LogReg (full)": (np.array([r["h_logreg_full"] for r in test]),
                      summ["hscore_models"]["full"]["threshold"]),
}


def f1(yy, s, t):
    p = s >= t
    tp = np.sum(p & (yy == 1))
    return 0.0 if tp == 0 else 2 * tp / (p.sum() + (yy == 1).sum())


rng = np.random.default_rng(42)
B = 2000
boot = {n: {"auroc": [], "f1": []} for n in scores}
for _ in range(B):
    i = rng.integers(0, len(y), len(y))
    for n, (s, t) in scores.items():
        boot[n]["auroc"].append(roc_auc_score(y[i], s[i]))
        boot[n]["f1"].append(f1(y[i], s[i], t))


def ci(v):
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


bootstrap = {n: {"auroc_ci95": ci(b["auroc"]), "f1_ci95": ci(b["f1"])} for n, b in boot.items()}
for n in ("LogReg (core)", "LogReg (full)"):
    for m in ("auroc", "f1"):
        d = np.array(boot[n][m]) - np.array(boot["Fixed 0.5/0.5"][m])
        bootstrap[n][f"{m}_gain_vs_fixed"] = {
            "mean": round(float(d.mean()), 4), "ci95": ci(d),
            "p_gain_le_0": round(float(np.mean(d <= 0)), 4)}

# ── Manual judge audit (labels assigned by hand after reading each answer) ──
# Same sample as printed by: random.Random(7), 15 per benchmark, in this order.
# Disagreements (all are judge-says-WRONG but answer is actually correct):
DISAGREE = {
    "HaluEval-QA": [0, 5],   # definition of "trade credit"; "Mysore" (Chamundi Hills are in Mysore)
    "SciQ": [13],            # "gravity between Earth and Moon is stronger" ≈ gold
    "SQuAD": [10, 12],       # paraphrased list of professionals; paraphrased IPCC comparison
}
rr = random.Random(7)
audit = {"n": 0, "agree": 0, "false_hallucination": 0, "missed_hallucination": 0,
         "per_benchmark": {}, "items": []}
for b in ["TruthfulQA", "HaluEval-QA", "SciQ", "SQuAD"]:
    sample = rr.sample([r for r in recs if r["benchmark"] == b], 15)
    bad = DISAGREE.get(b, [])
    for k, r in enumerate(sample):
        human_correct = (not r["correct"]) if k in bad else r["correct"]
        audit["items"].append({"id": r["id"], "judge_correct": r["correct"],
                               "human_correct": human_correct})
    audit["per_benchmark"][b] = {"n": 15, "agree": 15 - len(bad)}
    audit["n"] += 15
    audit["agree"] += 15 - len(bad)
audit["false_hallucination"] = sum(i["human_correct"] and not i["judge_correct"] for i in audit["items"])
audit["missed_hallucination"] = sum(i["judge_correct"] and not i["human_correct"] for i in audit["items"])
audit["agreement"] = round(audit["agree"] / audit["n"], 4)

out = {"bootstrap_test": {"n_test": int(len(y)), "resamples": B, "results": bootstrap},
       "judge_audit": audit}
(OUT / "final_checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

print(f"Paired bootstrap on test (n={len(y)}, B={B})")
for n, r in bootstrap.items():
    line = f"  {n:<15} AUROC CI {r['auroc_ci95']}  F1 CI {r['f1_ci95']}"
    if "auroc_gain_vs_fixed" in r:
        g = r["auroc_gain_vs_fixed"]
        h = r["f1_gain_vs_fixed"]
        line += (f"\n      ΔAUROC {g['mean']:+.3f} CI {g['ci95']} (P≤0: {g['p_gain_le_0']})"
                 f"   ΔF1 {h['mean']:+.3f} CI {h['ci95']} (P≤0: {h['p_gain_le_0']})")
    print(line)
print(f"Judge audit: {audit['agree']}/{audit['n']} agree ({audit['agreement']:.1%}); "
      f"false hallucination labels {audit['false_hallucination']}, missed {audit['missed_hallucination']}")
