"""
evaluate.py — Enhanced Analysis & Visualisation of Pipeline Results
====================================================================
Run after pipeline.py to produce charts, a summary table, and a detailed report.

Usage:  python evaluate.py [results.json]
"""

import json
import sys
from pathlib import Path

# ── Try to import plotting libraries ─────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for Windows compatibility
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


SCRIPT_DIR = Path(__file__).parent


def load_results(path=None):
    if path is None:
        path = SCRIPT_DIR / "results.json"
    with open(path) as f:
        return json.load(f)


def compute_metrics(results):
    """Compute all evaluation metrics from results."""
    total = len(results)
    correct = sum(r["correct"] for r in results)
    flagged = sum(r["flagged"] for r in results)
    wrong = total - correct

    tp = sum(r["flagged"] and not r["correct"] for r in results)
    fp = sum(r["flagged"] and r["correct"] for r in results)
    fn = sum(not r["flagged"] and not r["correct"] for r in results)
    tn = sum(not r["flagged"] and r["correct"] for r in results)

    precision = tp / flagged if flagged > 0 else 0.0
    recall = tp / wrong if wrong > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_uncertainty = sum(r["uncertainty"] for r in results) / total
    avg_nli = sum(r["nli_score"] for r in results) / total
    avg_h_score = sum(r["hallucination_score"] for r in results) / total

    return {
        "total": total, "correct": correct, "wrong": wrong,
        "accuracy": correct / total, "flagged": flagged,
        "flagged_rate": flagged / total,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "avg_uncertainty": avg_uncertainty,
        "avg_nli": avg_nli,
        "avg_h_score": avg_h_score,
    }


def confusion_bar(results):
    """Visualise TP/FP/FN/TN flag breakdown."""
    tp = sum(r["flagged"] and not r["correct"] for r in results)
    fp = sum(r["flagged"] and r["correct"] for r in results)
    fn = sum(not r["flagged"] and not r["correct"] for r in results)
    tn = sum(not r["flagged"] and r["correct"] for r in results)

    labels = ["True Pos\n(flagged & wrong)", "False Pos\n(flagged & correct)",
              "False Neg\n(missed & wrong)", "True Neg\n(passed & correct)"]
    values = [tp, fp, fn, tn]
    colors = ["#e05252", "#f0a04b", "#f0d04b", "#4caf7d"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                str(val), ha="center", va="bottom", fontsize=13, fontweight="bold", color="#333")
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Hallucination Flag Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, max(values) + 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_path = SCRIPT_DIR / "confusion.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close()


def score_distribution(results):
    """Plot hallucination score vs. correctness."""
    correct_scores = [r["hallucination_score"] for r in results if r["correct"]]
    wrong_scores   = [r["hallucination_score"] for r in results if not r["correct"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 12)
    ax.hist(correct_scores, bins=bins, alpha=0.75, label="Correct answers", color="#4caf7d", edgecolor="white")
    ax.hist(wrong_scores,   bins=bins, alpha=0.75, label="Wrong answers",   color="#e05252", edgecolor="white")
    ax.axvline(0.50, color="#333", linestyle="--", linewidth=1.5, label="Flag threshold (0.50)")
    ax.set_xlabel("Hallucination Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Hallucination Score Distribution by Correctness", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_path = SCRIPT_DIR / "score_dist.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close()


def score_breakdown_chart(results):
    """Bar chart showing uncertainty, NLI, and H-Score per question."""
    questions = [r["question"][:30] + "..." if len(r["question"]) > 30 else r["question"]
                 for r in results]
    uncertainties = [r["uncertainty"] for r in results]
    nli_gaps = [1 - r["nli_score"] for r in results]
    h_scores = [r["hallucination_score"] for r in results]

    x = np.arange(len(questions))
    width = 0.28

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, uncertainties, width, label="Uncertainty (Entropy)", color="#fbbf24", edgecolor="white")
    ax.bar(x, nli_gaps, width, label="1 - NLI (Ungrounded)", color="#c084fc", edgecolor="white")
    ax.bar(x + width, h_scores, width, label="H-Score (Composite)", color="#f87171", edgecolor="white")
    ax.axhline(0.50, color="#333", linestyle="--", linewidth=1, alpha=0.6, label="Threshold")

    ax.set_xticks(x)
    ax.set_xticklabels(questions, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Question Score Breakdown", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_path = SCRIPT_DIR / "score_breakdown.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close()


def radar_chart(metrics):
    """Radar chart of pipeline performance metrics."""
    categories = ["Accuracy", "Precision", "Recall", "F1", "Avg NLI"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"],
              metrics["f1"], metrics["avg_nli"]]
    values += values[:1]  # close the polygon

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="#667eea", alpha=0.25)
    ax.plot(angles, values, color="#667eea", linewidth=2, marker="o", markersize=6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Pipeline Performance Radar", fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = SCRIPT_DIR / "performance_radar.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close()


def print_table(results):
    """Text-mode summary table."""
    header = f"{'Question':<45} {'Answer':<25} {'H-Score':>7} {'Flag':>6} {'OK':>5}"
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        q  = r["question"][:43] + ".." if len(r["question"]) > 43 else r["question"]
        a  = r["answer"][:23] + ".." if len(r["answer"]) > 23 else r["answer"]
        hs = f"{r['hallucination_score']:.3f}"
        fl = "!!" if r["flagged"] else "OK"
        ok = "Y" if r["correct"] else "N"
        print(f"{q:<45} {a:<25} {hs:>7} {fl:>6} {ok:>5}")
    print(sep)


def print_detailed_report(metrics):
    """Print a comprehensive metrics report."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION REPORT — Hallucination Mitigation Pipeline")
    print(f"{'='*60}")
    print(f"  Total questions      : {metrics['total']}")
    print(f"  Correct answers      : {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.1%})")
    print(f"  Wrong answers        : {metrics['wrong']}/{metrics['total']}")
    print(f"{'─'*60}")
    print(f"  DETECTION PERFORMANCE")
    print(f"{'─'*60}")
    print(f"  Flagged as risky     : {metrics['flagged']}/{metrics['total']} ({metrics['flagged_rate']:.1%})")
    print(f"  True Positives  (TP) : {metrics['tp']}  (correctly flagged hallucinations)")
    print(f"  False Positives (FP) : {metrics['fp']}  (correct answers wrongly flagged)")
    print(f"  False Negatives (FN) : {metrics['fn']}  (missed hallucinations)")
    print(f"  True Negatives  (TN) : {metrics['tn']}  (correct answers passed)")
    print(f"{'─'*60}")
    print(f"  Flag Precision       : {metrics['precision']:.1%}")
    print(f"  Flag Recall          : {metrics['recall']:.1%}")
    print(f"  Flag F1 Score        : {metrics['f1']:.1%}")
    print(f"{'─'*60}")
    print(f"  SCORE AVERAGES")
    print(f"{'─'*60}")
    print(f"  Avg Uncertainty      : {metrics['avg_uncertainty']:.4f}")
    print(f"  Avg NLI Score        : {metrics['avg_nli']:.4f}")
    print(f"  Avg H-Score          : {metrics['avg_h_score']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    resolved = Path(path) if path else (SCRIPT_DIR / "results.json")

    if not resolved.exists():
        print(f"results.json not found at {resolved}. Run pipeline.py first.")
        sys.exit(1)

    results = load_results(resolved)
    metrics = compute_metrics(results)

    print_table(results)
    print_detailed_report(metrics)

    if HAS_PLOT:
        print("\nGenerating charts...")
        confusion_bar(results)
        score_distribution(results)
        score_breakdown_chart(results)
        radar_chart(metrics)
        print("All charts saved successfully!")
    else:
        print("\nmatplotlib not available — skipping charts.")
        print("Install with: pip install matplotlib numpy")
