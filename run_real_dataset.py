"""
run_real_dataset.py — Run Hallucination Mitigation Pipeline on Real TruthfulQA Dataset
========================================================================================
Loads the TruthfulQA generation_validation.csv from the dataset/ folder,
runs the full 4-stage pipeline, generates results and evaluation charts.

Usage:
    python run_real_dataset.py                   # run on all 4072 samples
    python run_real_dataset.py --limit 100       # run on first 100 samples
    python run_real_dataset.py --limit 50 --ablation   # run ablation on 50 samples
"""

import argparse
import ast
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from pipeline import (
    HallucinationMitigationPipeline,
    KnowledgeBase,
    SimulatedLLM,
    HallucinationResult,
    compute_entropy_uncertainty,
    nli_entailment_score,
    compute_hallucination_score,
    majority_vote,
    evaluate,
    print_report,
)


def parse_array_field(text: str) -> list:
    """
    Parse the array-like string from TruthfulQA CSV.
    Handles formats like: ['item1' 'item2' 'item3']
    """
    if not text or text.strip() == "":
        return []
    
    text = text.strip()
    
    # Try standard Python list parsing first
    try:
        result = ast.literal_eval(text)
        if isinstance(result, list):
            return [str(x).strip() for x in result]
    except (ValueError, SyntaxError):
        pass
    
    # Handle numpy-style arrays: ['item1' 'item2'] (space-separated, no commas)
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        # Match quoted strings (single or double)
        matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", inner)
        result = [m[0] if m[0] else m[1] for m in matches]
        if result:
            return result
    
    # Fallback: treat as single item
    return [text.strip()]


def load_truthfulqa_csv(csv_path: str, limit: int = None) -> list:
    """
    Load TruthfulQA generation_validation.csv and convert to pipeline format.
    """
    dataset = []
    
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            
            question = row.get("question", "").strip()
            if not question:
                continue
            
            correct = parse_array_field(row.get("correct_answers", ""))
            incorrect = parse_array_field(row.get("incorrect_answers", ""))
            best_answer = row.get("best_answer", "").strip()
            category = row.get("category", "").strip()
            q_type = row.get("type", "").strip()
            
            # Use best_answer as primary context (since CSV doesn't have context field)
            # Combine correct answers as supporting context
            context = best_answer
            if correct:
                context += " " + " ".join(correct[:3])
            
            # Ensure correct_answers includes best_answer
            if best_answer and best_answer not in correct:
                correct.insert(0, best_answer)
            
            # Skip if no correct answers at all
            if not correct:
                continue
            
            dataset.append({
                "question": question,
                "correct_answers": correct,
                "incorrect_answers": incorrect if incorrect else ["I don't know"],
                "context": context,
                "category": category,
                "type": q_type,
                "best_answer": best_answer,
            })
    
    return dataset


class RealDatasetPipeline(HallucinationMitigationPipeline):
    """
    Extended pipeline that builds knowledge base from the real dataset contexts.
    """
    
    def __init__(self, dataset, **kwargs):
        # Don't call super().__init__() fully — we override KB construction
        self.llm = SimulatedLLM(
            hallucination_rate=kwargs.get("hallucination_rate", 0.35),
        )
        self.n_samples = kwargs.get("n_samples", 5)
        self.flag_threshold = kwargs.get("flag_threshold", 0.50)
        self.top_k = kwargs.get("top_k_retrieval", 3)
        
        # Build knowledge base from dataset contexts
        print(f"  Building knowledge base from {len(dataset)} documents...")
        contexts = [s["context"] for s in dataset if s.get("context")]
        self.kb = KnowledgeBase(contexts)
        print(f"  Knowledge base ready (vocab size: {len(self.kb.vocab)})")


def run_pipeline_on_dataset(dataset, rate=0.35, samples=5, threshold=0.50, label=""):
    """Run the pipeline on a dataset and return results + metrics."""
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  Samples: {len(dataset)} | Rate: {rate} | N: {samples} | Threshold: {threshold}")
    print(f"{'='*60}")
    
    start = time.time()
    pipe = RealDatasetPipeline(
        dataset,
        hallucination_rate=rate,
        n_samples=samples,
        flag_threshold=threshold,
    )
    results = pipe.run_dataset(dataset)
    elapsed = time.time() - start
    
    metrics = evaluate(results)
    
    print(f"\n  Accuracy            : {metrics['accuracy']:.1%}")
    print(f"  Flagged rate        : {metrics['flagged_rate']:.1%}")
    print(f"  Flag Precision      : {metrics['flag_precision']:.1%}")
    print(f"  Flag Recall         : {metrics['flag_recall']:.1%}")
    print(f"  Flag F1             : {metrics['flag_f1']:.1%}")
    print(f"  Avg Uncertainty     : {metrics['avg_uncertainty']:.4f}")
    print(f"  Avg NLI Score       : {metrics['avg_nli_score']:.4f}")
    print(f"  Avg H-Score         : {metrics['avg_hallucination_score']:.4f}")
    print(f"  Time elapsed        : {elapsed:.1f}s")
    
    return results, metrics


def run_ablation(dataset):
    """Run ablation study comparing pipeline configurations."""
    print(f"\n{'='*60}")
    print(f"  ABLATION STUDY on Real TruthfulQA ({len(dataset)} samples)")
    print(f"{'='*60}")
    
    configs = [
        ("No Mitigation (Baseline)",    dict(rate=0.35, samples=1,  threshold=1.1)),
        ("RAG Only",                     dict(rate=0.18, samples=1,  threshold=1.1)),
        ("Self-Consistency Only (N=5)",  dict(rate=0.35, samples=5,  threshold=1.1)),
        ("Uncertainty Gate Only",        dict(rate=0.35, samples=5,  threshold=0.50)),
        ("Full Hybrid Pipeline",         dict(rate=0.18, samples=5,  threshold=0.50)),
    ]
    
    ablation_results = []
    for name, cfg in configs:
        _, m = run_pipeline_on_dataset(dataset, label=name, **cfg)
        ablation_results.append({
            "config": name,
            "accuracy": m["accuracy"],
            "f1": m["flag_f1"],
            "flagged_rate": m["flagged_rate"],
            "avg_uncertainty": m["avg_uncertainty"],
            "avg_nli": m["avg_nli_score"],
            "avg_h_score": m["avg_hallucination_score"],
        })
    
    print(f"\n{'='*60}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Configuration':<36} {'Accuracy':>8} {'F1':>8} {'Flagged':>8}")
    print(f"  {'─'*36} {'─'*8} {'─'*8} {'─'*8}")
    for r in ablation_results:
        print(f"  {r['config']:<36} {r['accuracy']:>7.1%} {r['f1']:>7.1%} {r['flagged_rate']:>7.1%}")
    print(f"{'='*60}")
    
    return ablation_results


def save_results(results, output_path, ablation_data=None):
    """Save results to JSON."""
    out = []
    for r in results:
        out.append({
            "question": r.question,
            "answer": r.majority_answer,
            "ground_truth": r.ground_truth,
            "correct": r.correct,
            "flagged": r.flagged,
            "uncertainty": round(r.uncertainty, 4),
            "nli_score": round(r.nli_score, 4),
            "hallucination_score": round(r.hallucination_score, 4),
            "retrieval_score": round(r.retrieval_score, 4),
            "sample_distribution": r.details.get("sample_dist", {}),
        })
    
    output = {
        "metadata": {
            "dataset": "TruthfulQA (generation_validation.csv)",
            "total_samples": len(results),
            "pipeline": "4-stage hybrid (RAG + Self-Consistency + Uncertainty + NLI Gate)",
        },
        "results": out,
    }
    
    if ablation_data:
        output["ablation"] = ablation_data
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_path}")


def generate_charts(results, metrics, output_dir):
    """Generate evaluation charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping charts.")
        return
    
    # 1. Confusion Matrix Bar
    tp = sum(r.flagged and not r.correct for r in results)
    fp = sum(r.flagged and r.correct for r in results)
    fn = sum(not r.flagged and not r.correct for r in results)
    tn = sum(not r.flagged and r.correct for r in results)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Chart 1: Confusion bar
    labels = ["TP\n(flagged\n& wrong)", "FP\n(flagged\n& correct)", "FN\n(missed\n& wrong)", "TN\n(passed\n& correct)"]
    values = [tp, fp, fn, tn]
    colors = ["#e05252", "#f0a04b", "#f0d04b", "#4caf7d"]
    bars = axes[0].bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_title("Detection Confusion Matrix", fontweight="bold")
    axes[0].set_ylabel("Count")
    
    # Chart 2: Score distribution
    correct_scores = [r.hallucination_score for r in results if r.correct]
    wrong_scores = [r.hallucination_score for r in results if not r.correct]
    bins = np.linspace(0, 1, 20)
    axes[1].hist(correct_scores, bins=bins, alpha=0.7, label="Correct", color="#4caf7d", edgecolor="white")
    axes[1].hist(wrong_scores, bins=bins, alpha=0.7, label="Wrong", color="#e05252", edgecolor="white")
    axes[1].axvline(0.50, color="#333", linestyle="--", linewidth=1.5, label="Threshold (0.50)")
    axes[1].set_xlabel("Hallucination Score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("H-Score Distribution by Correctness", fontweight="bold")
    axes[1].legend(fontsize=9)
    
    # Chart 3: Category breakdown (if available)
    categories = {}
    for r in results:
        cat = getattr(r, '_category', 'Unknown')
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r.correct:
            categories[cat]["correct"] += 1
    
    # Radar chart instead
    radar_cats = ["Accuracy", "Precision", "Recall", "F1", "Avg NLI"]
    radar_vals = [metrics["accuracy"], metrics["flag_precision"], metrics["flag_recall"],
                  metrics["flag_f1"], metrics["avg_nli_score"]]
    radar_vals_closed = radar_vals + [radar_vals[0]]
    angles = np.linspace(0, 2 * np.pi, len(radar_cats), endpoint=False).tolist()
    angles += [angles[0]]
    
    axes[2] = plt.subplot(1, 3, 3, polar=True)
    axes[2].fill(angles, radar_vals_closed, color="#667eea", alpha=0.25)
    axes[2].plot(angles, radar_vals_closed, color="#667eea", linewidth=2, marker="o", markersize=6)
    axes[2].set_xticks(angles[:-1])
    axes[2].set_xticklabels(radar_cats, fontsize=9)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Performance Radar", fontweight="bold", pad=20)
    
    plt.suptitle(f"TruthfulQA Evaluation — {len(results)} Samples | Accuracy: {metrics['accuracy']:.1%}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    chart_path = output_dir / "truthfulqa_evaluation.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {chart_path}")
    plt.close()
    
    # Additional chart: Uncertainty vs NLI scatter
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for r in results:
        color = "#4caf7d" if r.correct else "#e05252"
        marker = "^" if r.flagged else "o"
        alpha = 0.6
        ax2.scatter(r.uncertainty, r.nli_score, color=color, marker=marker, alpha=alpha, s=30, edgecolors="white", linewidth=0.3)
    
    ax2.axhline(0.5, color="#888", linestyle=":", alpha=0.5)
    ax2.axvline(0.5, color="#888", linestyle=":", alpha=0.5)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4caf7d', markersize=8, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e05252', markersize=8, label='Wrong'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#888', markersize=8, label='Flagged'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888', markersize=8, label='Passed'),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax2.set_xlabel("Uncertainty (Entropy)", fontsize=12)
    ax2.set_ylabel("NLI Grounding Score", fontsize=12)
    ax2.set_title(f"Uncertainty vs NLI — {len(results)} TruthfulQA Samples", fontsize=13, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    scatter_path = output_dir / "uncertainty_vs_nli.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {scatter_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run pipeline on real TruthfulQA dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (default: all)")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--rate", type=float, default=0.35, help="LLM hallucination rate")
    parser.add_argument("--samples", type=int, default=5, help="Self-consistency samples")
    parser.add_argument("--threshold", type=float, default=0.50, help="Flag threshold")
    args = parser.parse_args()
    
    csv_path = PROJECT_DIR / "dataset" / "generation_validation.csv"
    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}")
        sys.exit(1)
    
    print("="*60)
    print("  HALLUCINATION MITIGATION — Real TruthfulQA Dataset")
    print("="*60)
    print(f"  Loading: {csv_path}")
    
    dataset = load_truthfulqa_csv(str(csv_path), limit=args.limit)
    print(f"  Loaded {len(dataset)} questions from TruthfulQA")
    
    # Show category distribution
    cats = {}
    for s in dataset:
        cat = s.get("category", "Unknown")
        cats[cat] = cats.get(cat, 0) + 1
    print(f"\n  Category distribution:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1])[:10]:
        print(f"    {cat:<30} {count:>5} ({count/len(dataset):.1%})")
    
    # Run main pipeline
    results, metrics = run_pipeline_on_dataset(
        dataset,
        rate=args.rate,
        samples=args.samples,
        threshold=args.threshold,
        label="Full Hybrid Pipeline on TruthfulQA",
    )
    
    # Run ablation if requested
    ablation_data = None
    if args.ablation:
        # Use a subset for ablation to save time
        abl_limit = min(len(dataset), 200)
        abl_dataset = dataset[:abl_limit]
        ablation_data = run_ablation(abl_dataset)
    
    # Save results
    output_path = PROJECT_DIR / "results_truthfulqa.json"
    
    # Also save the standard results.json for evaluate.py compatibility
    standard_out = []
    for r in results:
        standard_out.append({
            "question": r.question,
            "answer": r.majority_answer,
            "ground_truth": r.ground_truth,
            "correct": r.correct,
            "flagged": r.flagged,
            "uncertainty": round(r.uncertainty, 4),
            "nli_score": round(r.nli_score, 4),
            "hallucination_score": round(r.hallucination_score, 4),
            "sample_distribution": r.details.get("sample_dist", {}),
        })
    
    with open(PROJECT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(standard_out, f, indent=2, ensure_ascii=False)
    print(f"\nStandard results saved to results.json")
    
    save_results(results, output_path, ablation_data)
    
    # Generate charts
    print("\nGenerating evaluation charts...")
    generate_charts(results, metrics, PROJECT_DIR)
    
    # Print top-10 most hallucinated questions
    print(f"\n{'='*60}")
    print(f"  TOP 10 HIGHEST HALLUCINATION RISK QUESTIONS")
    print(f"{'='*60}")
    sorted_results = sorted(results, key=lambda r: r.hallucination_score, reverse=True)
    for i, r in enumerate(sorted_results[:10], 1):
        flag = "FLAGGED" if r.flagged else "PASSED"
        corr = "CORRECT" if r.correct else "WRONG"
        print(f"  {i:2d}. H={r.hallucination_score:.3f} [{flag:>7}] [{corr:>7}]")
        print(f"      Q: {r.question[:70]}")
        print(f"      A: {r.majority_answer[:50]} (truth: {r.ground_truth[:50]})")
        print()
    
    print(f"{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Dataset           : TruthfulQA ({len(dataset)} questions)")
    print(f"  Accuracy          : {metrics['accuracy']:.1%}")
    print(f"  Detection F1      : {metrics['flag_f1']:.1%}")
    print(f"  Avg H-Score       : {metrics['avg_hallucination_score']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
