"""Show results summary from the real TruthfulQA run."""
import json
from pathlib import Path

base = Path(__file__).parent

with open(base / "results_truthfulqa.json", encoding="utf-8") as f:
    data = json.load(f)

meta = data["metadata"]
results = data["results"]
ablation = data.get("ablation", [])

print("=" * 60)
print("  TRUTHFULQA DATASET RESULTS")
print("=" * 60)
print(f"  Dataset     : {meta['dataset']}")
print(f"  Total samples: {meta['total_samples']}")
print()

correct = sum(1 for r in results if r["correct"])
flagged = sum(1 for r in results if r["flagged"])
total = len(results)

print(f"  Accuracy    : {correct}/{total} = {correct/total:.1%}")
print(f"  Flagged     : {flagged}/{total} = {flagged/total:.1%}")

tp = sum(1 for r in results if r["flagged"] and not r["correct"])
fp = sum(1 for r in results if r["flagged"] and r["correct"])
fn = sum(1 for r in results if not r["flagged"] and not r["correct"])
tn = sum(1 for r in results if not r["flagged"] and r["correct"])
prec = tp / flagged if flagged else 0
rec = tp / (total - correct) if (total - correct) else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"  Precision   : {prec:.1%}")
print(f"  Recall      : {rec:.1%}")
print(f"  F1 Score    : {f1:.1%}")

avg_unc = sum(r["uncertainty"] for r in results) / total
avg_nli = sum(r["nli_score"] for r in results) / total
avg_h = sum(r["hallucination_score"] for r in results) / total
print(f"  Avg Uncertainty  : {avg_unc:.4f}")
print(f"  Avg NLI Score    : {avg_nli:.4f}")
print(f"  Avg H-Score      : {avg_h:.4f}")

if ablation:
    print()
    print("=" * 60)
    print("  ABLATION STUDY")
    print("=" * 60)
    print(f"  {'Configuration':<36} {'Acc':>7} {'F1':>7} {'Flagged':>8}")
    print(f"  {'-'*36} {'-'*7} {'-'*7} {'-'*8}")
    for a in ablation:
        print(f"  {a['config']:<36} {a['accuracy']:>6.1%} {a['f1']:>6.1%} {a['flagged_rate']:>7.1%}")
    print("=" * 60)

# Top 5 wrong answers
wrong = [r for r in results if not r["correct"]]
wrong.sort(key=lambda x: x["hallucination_score"], reverse=True)
print()
print("=" * 60)
print("  TOP 10 WRONG ANSWERS (Highest H-Score)")
print("=" * 60)
for i, r in enumerate(wrong[:10], 1):
    flag = "FLAGGED" if r["flagged"] else "MISSED"
    print(f"  {i:2d}. H={r['hallucination_score']:.3f} [{flag:>7}]")
    print(f"      Q: {r['question'][:65]}")
    print(f"      A: {r['answer'][:40]} | Truth: {r['ground_truth'][:40]}")
