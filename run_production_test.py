"""
run_production_test.py — Production Model Evaluation on TruthfulQA
===================================================================
Upgrades the demo pipeline with real ML models:
  • RAG:  sentence-transformers/all-MiniLM-L6-v2  +  FAISS index
  • NLI:  cross-encoder/nli-deberta-v3-base
  • LLM:  SimulatedLLM (kept for controlled evaluation)

Evaluates on the full TruthfulQA dataset (817 questions).

Usage:
    python run_production_test.py                   # full 817 samples
    python run_production_test.py --limit 100       # first 100
    python run_production_test.py --ablation        # include ablation study
"""

import argparse
import ast
import csv
import json
import math
import os
import random
import re
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Load TruthfulQA CSV
# ─────────────────────────────────────────────────────────────────────────────

def parse_array_field(text: str) -> list:
    """Parse array-like string from TruthfulQA CSV."""
    if not text or text.strip() == "":
        return []
    text = text.strip()
    try:
        result = ast.literal_eval(text)
        if isinstance(result, list):
            return [str(x).strip() for x in result]
    except (ValueError, SyntaxError):
        pass
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", inner)
        result = [m[0] if m[0] else m[1] for m in matches]
        if result:
            return result
    return [text.strip()]


def load_truthfulqa_csv(csv_path: str, limit: int = None) -> list:
    """Load TruthfulQA CSV into pipeline-ready format."""
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

            context = best_answer
            if correct:
                context += " " + " ".join(correct[:3])
            if best_answer and best_answer not in correct:
                correct.insert(0, best_answer)
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


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Production RAG: Sentence-Transformers + FAISS
# ─────────────────────────────────────────────────────────────────────────────

class ProductionKnowledgeBase:
    """
    Semantic RAG using sentence-transformers embeddings + FAISS index.
    Replaces TF-IDF cosine similarity with dense vector retrieval.
    """

    def __init__(self, documents: List[str], model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.documents = documents
        print(f"    [RAG] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"    [RAG] Encoding {len(documents)} documents...")
        t0 = time.time()
        self.doc_embeddings = self.model.encode(
            documents, show_progress_bar=False, batch_size=64, normalize_embeddings=True
        )
        encode_time = time.time() - t0
        print(f"    [RAG] Encoded in {encode_time:.1f}s  (dim={self.embedding_dim})")

        # Build FAISS index (Inner Product = cosine when normalized)
        print(f"    [RAG] Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.doc_embeddings.astype(np.float32))
        print(f"    [RAG] FAISS index ready ({self.index.ntotal} vectors)")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve top-k most relevant documents for a query."""
        q_embedding = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_embedding, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Production NLI: DeBERTa Cross-Encoder
# ─────────────────────────────────────────────────────────────────────────────

class ProductionNLI:
    """
    NLI-based grounding verification using cross-encoder/nli-deberta-v3-base.
    Returns entailment probability as the grounding score.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        print(f"    [NLI] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"    [NLI] Model ready on {self.device}")

        # DeBERTa NLI label mapping: 0=contradiction, 1=entailment, 2=neutral
        self.entailment_idx = 1

    def score(self, premise: str, hypothesis: str) -> float:
        """
        Compute entailment score: P(premise entails hypothesis).
        premise  = retrieved context (ground truth knowledge)
        hypothesis = LLM's answer
        Returns float ∈ [0, 1] where 1.0 = fully entailed/grounded.
        """
        import torch

        if not premise.strip() or not hypothesis.strip():
            return 0.5

        # Truncate to avoid exceeding model's max length
        premise = premise[:512]
        hypothesis = hypothesis[:128]

        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        entailment_prob = probs[0, self.entailment_idx].item()
        return entailment_prob

    def score_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[float]:
        """Batch scoring for efficiency."""
        import torch

        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            premises = [p[:512] for p, h in batch]
            hypotheses = [h[:128] for p, h in batch]

            inputs = self.tokenizer(
                premises, hypotheses,
                return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

            batch_scores = probs[:, self.entailment_idx].cpu().tolist()
            all_scores.extend(batch_scores)

        return all_scores


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Simulated LLM (kept for controlled evaluation)
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedLLM:
    """
    Simulates an LLM that occasionally hallucinates.
    Grounding reduces hallucination rate by 50%.
    """

    def __init__(self, hallucination_rate: float = 0.35, seed: int = 42):
        self.hallucination_rate = hallucination_rate
        self.rng = random.Random(seed)

    def generate(self, question, context, correct_answers, incorrect_answers,
                 grounded=False):
        effective_rate = self.hallucination_rate * (0.5 if grounded else 1.0)
        if self.rng.random() < effective_rate:
            return self.rng.choice(incorrect_answers) if incorrect_answers else "I don't know."
        return self.rng.choice(correct_answers)

    def generate_n(self, question, context, correct_answers, incorrect_answers,
                   n=5, grounded=False):
        return [self.generate(question, context, correct_answers, incorrect_answers, grounded)
                for _ in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Uncertainty Estimation (Paper §II-G)
# ─────────────────────────────────────────────────────────────────────────────

def compute_entropy_uncertainty(answers: List[str]) -> float:
    """Normalized Shannon entropy over sampled answers."""
    if not answers:
        return 1.0
    counts = Counter(a.strip().lower() for a in answers)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log2(p + 1e-10) for p in probs)
    max_entropy = math.log2(len(answers))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def majority_vote(answers: List[str]) -> str:
    """Self-consistency: pick most frequent answer."""
    if not answers:
        return ""
    counts = Counter(a.strip().lower() for a in answers)
    return counts.most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Hallucination Scoring & Gate (Paper §II-G, Table I)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HallucinationResult:
    question: str
    retrieved_context: str
    retrieval_score: float
    raw_answers: List[str]
    majority_answer: str
    uncertainty: float
    nli_score: float
    hallucination_score: float
    flagged: bool
    ground_truth: str
    correct: bool
    category: str = ""
    details: dict = field(default_factory=dict)


def compute_hallucination_score(uncertainty: float, nli_score: float) -> float:
    """H_score = 0.5 × Entropy + 0.5 × (1 - NLI_entailment)"""
    return 0.5 * uncertainty + 0.5 * (1.0 - nli_score)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Production Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ProductionPipeline:
    """
    Full 4-stage production pipeline with real ML models:
      Stage 1: Sentence-Transformers + FAISS retrieval
      Stage 2: Self-consistency sampling (SimulatedLLM)
      Stage 3: DeBERTa NLI + Entropy uncertainty
      Stage 4: Hallucination gate
    """

    def __init__(
        self,
        dataset: list,
        hallucination_rate: float = 0.35,
        n_samples: int = 5,
        flag_threshold: float = 0.50,
        top_k_retrieval: int = 3,
        use_production_rag: bool = True,
        use_production_nli: bool = True,
    ):
        self.n_samples = n_samples
        self.flag_threshold = flag_threshold
        self.top_k = top_k_retrieval
        self.use_production_rag = use_production_rag
        self.use_production_nli = use_production_nli

        # LLM
        self.llm = SimulatedLLM(hallucination_rate=hallucination_rate)

        # RAG
        contexts = [s["context"] for s in dataset if s.get("context")]
        if use_production_rag:
            self.kb = ProductionKnowledgeBase(contexts)
        else:
            # Fallback to TF-IDF
            from pipeline import KnowledgeBase
            self.kb = KnowledgeBase(contexts)

        # NLI
        if use_production_nli:
            self.nli_model = ProductionNLI()
        else:
            self.nli_model = None

    def _nli_score(self, answer: str, context: str) -> float:
        """Get NLI entailment score (production or fallback)."""
        if self.nli_model:
            return self.nli_model.score(context, answer)
        else:
            # Fallback: lexical overlap
            answer_tokens = set(re.findall(r"[a-z]+", answer.lower()))
            context_tokens = set(re.findall(r"[a-z]+", context.lower()))
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "it", "in",
                         "of", "to", "and", "or", "that", "this", "not", "no", "yes"}
            answer_tokens -= stopwords
            if not answer_tokens:
                return 0.5
            overlap = answer_tokens & context_tokens
            return len(overlap) / len(answer_tokens)

    def run_single(self, sample: dict) -> HallucinationResult:
        question = sample["question"]
        correct = sample["correct_answers"]
        incorrect = sample["incorrect_answers"]
        ground_truth = correct[0]
        category = sample.get("category", "")

        # Stage 1: RAG retrieval
        retrieved = self.kb.retrieve(question, top_k=self.top_k)
        context = " ".join(doc for doc, _ in retrieved)
        retrieval_score = retrieved[0][1] if retrieved else 0.0

        # Stage 2: Multi-sample generation (self-consistency)
        answers = self.llm.generate_n(
            question, context, correct, incorrect,
            n=self.n_samples, grounded=True,
        )
        majority = majority_vote(answers)

        # Stage 3: Uncertainty + NLI
        uncertainty = compute_entropy_uncertainty(answers)
        nli_score = self._nli_score(majority, context)

        # Stage 4: Hallucination gate
        h_score = compute_hallucination_score(uncertainty, nli_score)
        flagged = h_score >= self.flag_threshold

        # Evaluate correctness
        is_correct = any(
            ans.lower() in majority.lower() or majority.lower() in ans.lower()
            for ans in correct
        )

        return HallucinationResult(
            question=question,
            retrieved_context=context[:200] + "...",
            retrieval_score=retrieval_score,
            raw_answers=answers,
            majority_answer=majority,
            uncertainty=uncertainty,
            nli_score=nli_score,
            hallucination_score=h_score,
            flagged=flagged,
            ground_truth=ground_truth,
            correct=is_correct,
            category=category,
            details={"n_samples": self.n_samples, "sample_dist": dict(Counter(answers))},
        )

    def run_dataset(self, dataset: list) -> List[HallucinationResult]:
        results = []
        total = len(dataset)
        t0 = time.time()

        for i, sample in enumerate(dataset):
            result = self.run_single(sample)
            results.append(result)

            # Progress bar
            if (i + 1) % 50 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                correct_so_far = sum(r.correct for r in results)
                acc = correct_so_far / len(results)
                print(f"\r    [{i+1:>4}/{total}]  {acc:.1%} acc  |  "
                      f"{elapsed:.0f}s elapsed  |  {eta:.0f}s ETA  |  "
                      f"{rate:.1f} q/s", end="", flush=True)

        print()  # newline after progress
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(results: List[HallucinationResult]) -> dict:
    total = len(results)
    correct = sum(r.correct for r in results)
    flagged = sum(r.flagged for r in results)
    wrong = total - correct

    tp = sum(r.flagged and not r.correct for r in results)
    fp = sum(r.flagged and r.correct for r in results)
    fn = sum(not r.flagged and not r.correct for r in results)
    tn = sum(not r.flagged and r.correct for r in results)

    precision = tp / flagged if flagged > 0 else 0.0
    recall = tp / wrong if wrong > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_uncertainty = float(np.mean([r.uncertainty for r in results]))
    avg_nli = float(np.mean([r.nli_score for r in results]))
    avg_h_score = float(np.mean([r.hallucination_score for r in results]))

    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy": correct / total,
        "flagged": flagged,
        "flagged_rate": flagged / total,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_uncertainty": avg_uncertainty,
        "avg_nli": avg_nli,
        "avg_h_score": avg_h_score,
    }


def evaluate_by_category(results: List[HallucinationResult]) -> dict:
    """Per-category accuracy breakdown."""
    cats = {}
    for r in results:
        cat = r.category or "Unknown"
        if cat not in cats:
            cats[cat] = {"total": 0, "correct": 0, "flagged": 0, "wrong_flagged": 0}
        cats[cat]["total"] += 1
        if r.correct:
            cats[cat]["correct"] += 1
        if r.flagged:
            cats[cat]["flagged"] += 1
        if r.flagged and not r.correct:
            cats[cat]["wrong_flagged"] += 1

    for cat in cats:
        cats[cat]["accuracy"] = cats[cat]["correct"] / cats[cat]["total"]
        cats[cat]["flag_rate"] = cats[cat]["flagged"] / cats[cat]["total"]

    return cats


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Chart Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_charts(results, metrics, cat_metrics, ablation_data, output_dir):
    """Generate comprehensive evaluation charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available — skipping charts.")
        return

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })

    # ── Chart 1: Confusion Matrix Heatmap ──
    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
    fig, ax = plt.subplots(figsize=(7, 6))
    matrix = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Flagged", "Flagged"])
    ax.set_yticklabels(["Correct", "Wrong"])
    ax.set_xlabel("Predicted (Gate Decision)")
    ax.set_ylabel("Actual (Correctness)")
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.5 else "black"
            ax.text(j, i, f"{val}\n({val/metrics['total']:.1%})",
                    ha="center", va="center", fontsize=14, fontweight="bold", color=color)
    ax.set_title(f"Confusion Matrix — {metrics['total']} Samples", fontweight="bold", pad=15)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "prod_confusion_matrix.png", dpi=150, bbox_inches="tight")
    print(f"    Saved prod_confusion_matrix.png")
    plt.close()

    # ── Chart 2: Score Distribution ──
    fig, ax = plt.subplots(figsize=(10, 5))
    correct_scores = [r.hallucination_score for r in results if r.correct]
    wrong_scores = [r.hallucination_score for r in results if not r.correct]
    bins = np.linspace(0, 1, 25)
    ax.hist(correct_scores, bins=bins, alpha=0.75, label=f"Correct (n={len(correct_scores)})",
            color="#4caf7d", edgecolor="white")
    ax.hist(wrong_scores, bins=bins, alpha=0.75, label=f"Wrong (n={len(wrong_scores)})",
            color="#e05252", edgecolor="white")
    ax.axvline(0.50, color="#333", linestyle="--", linewidth=2, label="Threshold (0.50)")
    ax.set_xlabel("Hallucination Score (H-Score)")
    ax.set_ylabel("Frequency")
    ax.set_title("H-Score Distribution by Correctness (Production Models)", fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "prod_score_distribution.png", dpi=150, bbox_inches="tight")
    print(f"    Saved prod_score_distribution.png")
    plt.close()

    # ── Chart 3: Uncertainty vs NLI Scatter ──
    fig, ax = plt.subplots(figsize=(10, 7))
    for r in results:
        color = "#4caf7d" if r.correct else "#e05252"
        marker = "^" if r.flagged else "o"
        ax.scatter(r.uncertainty, r.nli_score, color=color, marker=marker,
                   alpha=0.5, s=25, edgecolors="white", linewidth=0.3)
    ax.axhline(0.5, color="#888", linestyle=":", alpha=0.4)
    ax.axvline(0.5, color="#888", linestyle=":", alpha=0.4)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4caf7d", markersize=8, label="Correct"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e05252", markersize=8, label="Wrong"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#888", markersize=8, label="Flagged"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markersize=8, label="Passed"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax.set_xlabel("Uncertainty (Entropy)", fontsize=12)
    ax.set_ylabel("NLI Entailment Score (DeBERTa)", fontsize=12)
    ax.set_title(f"Uncertainty vs NLI — Production Models ({metrics['total']} Samples)",
                 fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "prod_uncertainty_vs_nli.png", dpi=150, bbox_inches="tight")
    print(f"    Saved prod_uncertainty_vs_nli.png")
    plt.close()

    # ── Chart 4: Performance Radar ──
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    categories_r = ["Accuracy", "Precision", "Recall", "F1", "Avg NLI"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"],
              metrics["f1"], metrics["avg_nli"]]
    values_closed = values + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(categories_r), endpoint=False).tolist()
    angles += [angles[0]]
    ax.fill(angles, values_closed, color="#667eea", alpha=0.25)
    ax.plot(angles, values_closed, color="#667eea", linewidth=2, marker="o", markersize=6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories_r, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Production Pipeline — Performance Radar", fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    for angle, val, cat in zip(angles[:-1], values, categories_r):
        ax.text(angle, val + 0.05, f"{val:.2f}", ha="center", fontsize=9, fontweight="bold",
                color="#667eea")
    plt.tight_layout()
    plt.savefig(output_dir / "prod_performance_radar.png", dpi=150, bbox_inches="tight")
    print(f"    Saved prod_performance_radar.png")
    plt.close()

    # ── Chart 5: Category Breakdown ──
    if cat_metrics:
        sorted_cats = sorted(cat_metrics.items(), key=lambda x: -x[1]["total"])[:15]
        fig, ax = plt.subplots(figsize=(14, 6))
        cat_names = [c[:25] for c, _ in sorted_cats]
        cat_accs = [d["accuracy"] for _, d in sorted_cats]
        cat_totals = [d["total"] for _, d in sorted_cats]
        colors = ["#4caf7d" if a >= 0.8 else "#fbbf24" if a >= 0.6 else "#e05252" for a in cat_accs]
        bars = ax.bar(range(len(cat_names)), cat_accs, color=colors, edgecolor="white", linewidth=0.8)
        for bar, total, acc in zip(bars, cat_totals, cat_accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{acc:.0%}\n(n={total})", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(cat_names)))
        ax.set_xticklabels(cat_names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.15)
        ax.axhline(0.8, color="#4caf7d", linestyle="--", alpha=0.3, label="80% target")
        ax.set_title("Accuracy by TruthfulQA Category (Production)", fontweight="bold", pad=15)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_dir / "prod_category_breakdown.png", dpi=150, bbox_inches="tight")
        print(f"    Saved prod_category_breakdown.png")
        plt.close()

    # ── Chart 6: Ablation Comparison ──
    if ablation_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        configs = [d["config"] for d in ablation_data]
        accs = [d["accuracy"] for d in ablation_data]
        f1s = [d["f1"] for d in ablation_data]
        x = np.arange(len(configs))
        w = 0.35
        bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#667eea", edgecolor="white")
        bars2 = ax.bar(x + w/2, f1s, w, label="Flag F1", color="#c084fc", edgecolor="white")
        for bar, val in zip(bars1, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", fontsize=9, fontweight="bold")
        for bar, val in zip(bars2, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.15)
        ax.set_title("Ablation Study — Production Models", fontweight="bold", pad=15)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_dir / "prod_ablation_comparison.png", dpi=150, bbox_inches="tight")
        print(f"    Saved prod_ablation_comparison.png")
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Production Model Evaluation on TruthfulQA")
    parser.add_argument("--limit", type=int, default=None, help="Limit sample count")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--rate", type=float, default=0.35, help="LLM hallucination rate")
    parser.add_argument("--samples", type=int, default=5, help="Self-consistency samples")
    parser.add_argument("--threshold", type=float, default=0.50, help="Flag threshold")
    args = parser.parse_args()

    csv_path = PROJECT_DIR / "dataset" / "generation_validation.csv"
    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}")
        sys.exit(1)

    print("=" * 70)
    print("  PRODUCTION MODEL EVALUATION — HalluciGuard Pipeline")
    print("  Models: sentence-transformers + FAISS | DeBERTa NLI")
    print("=" * 70)

    # Load dataset
    dataset = load_truthfulqa_csv(str(csv_path), limit=args.limit)
    print(f"\n  📊 Loaded {len(dataset)} questions from TruthfulQA")

    # Category distribution
    cats = {}
    for s in dataset:
        cat = s.get("category", "Unknown")
        cats[cat] = cats.get(cat, 0) + 1
    print(f"  📂 Categories: {len(cats)}")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1])[:8]:
        print(f"      {cat:<30} {count:>5} ({count/len(dataset):.1%})")

    # ── Run Full Hybrid Pipeline ──
    print(f"\n{'─' * 70}")
    print(f"  Running FULL HYBRID PIPELINE (production models)")
    print(f"  Rate={args.rate} | N={args.samples} | Threshold={args.threshold}")
    print(f"{'─' * 70}")

    t_start = time.time()
    pipeline = ProductionPipeline(
        dataset,
        hallucination_rate=args.rate,
        n_samples=args.samples,
        flag_threshold=args.threshold,
        use_production_rag=True,
        use_production_nli=True,
    )
    build_time = time.time() - t_start
    print(f"\n  ⏱️  Pipeline built in {build_time:.1f}s\n")

    results = pipeline.run_dataset(dataset)
    total_time = time.time() - t_start
    metrics = evaluate(results)
    cat_metrics = evaluate_by_category(results)

    # ── Print Results ──
    print(f"\n{'=' * 70}")
    print(f"  PRODUCTION EVALUATION RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total samples        : {metrics['total']}")
    print(f"  Accuracy             : {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Wrong answers        : {metrics['wrong']}")
    print(f"{'─' * 70}")
    print(f"  DETECTION PERFORMANCE")
    print(f"{'─' * 70}")
    print(f"  Flagged              : {metrics['flagged']} ({metrics['flagged_rate']:.1%})")
    print(f"  True Positives  (TP) : {metrics['tp']}  (correctly flagged hallucinations)")
    print(f"  False Positives (FP) : {metrics['fp']}  (correct answers wrongly flagged)")
    print(f"  False Negatives (FN) : {metrics['fn']}  (missed hallucinations)")
    print(f"  True Negatives  (TN) : {metrics['tn']}  (correct answers passed)")
    print(f"{'─' * 70}")
    print(f"  Flag Precision       : {metrics['precision']:.1%}")
    print(f"  Flag Recall          : {metrics['recall']:.1%}")
    print(f"  Flag F1 Score        : {metrics['f1']:.1%}")
    print(f"{'─' * 70}")
    print(f"  SCORE AVERAGES")
    print(f"{'─' * 70}")
    print(f"  Avg Uncertainty      : {metrics['avg_uncertainty']:.4f}")
    print(f"  Avg NLI (DeBERTa)    : {metrics['avg_nli']:.4f}")
    print(f"  Avg H-Score          : {metrics['avg_h_score']:.4f}")
    print(f"  Total time           : {total_time:.1f}s")
    print(f"{'=' * 70}")

    # ── Category Breakdown ──
    print(f"\n  ACCURACY BY CATEGORY")
    print(f"  {'Category':<30} {'Total':>6} {'Correct':>8} {'Accuracy':>9} {'Flagged':>8}")
    print(f"  {'─' * 30} {'─' * 6} {'─' * 8} {'─' * 9} {'─' * 8}")
    for cat, d in sorted(cat_metrics.items(), key=lambda x: -x[1]["total"]):
        print(f"  {cat:<30} {d['total']:>6} {d['correct']:>8} {d['accuracy']:>8.1%} {d['flagged']:>8}")

    # ── Top 15 highest risk ──
    print(f"\n{'=' * 70}")
    print(f"  TOP 15 HIGHEST HALLUCINATION RISK QUESTIONS")
    print(f"{'=' * 70}")
    sorted_results = sorted(results, key=lambda r: r.hallucination_score, reverse=True)
    for i, r in enumerate(sorted_results[:15], 1):
        flag = "⚠ FLAGGED" if r.flagged else "✓ PASSED"
        corr = "✓ CORRECT" if r.correct else "✗ WRONG"
        print(f"  {i:2d}. H={r.hallucination_score:.3f} [{flag}] [{corr}]")
        print(f"      Q: {r.question[:70]}")
        print(f"      A: {r.majority_answer[:50]}")
        print(f"      Truth: {r.ground_truth[:50]}")
        print()

    # ── Ablation Study ──
    ablation_data = None
    if args.ablation:
        print(f"\n{'=' * 70}")
        print(f"  ABLATION STUDY (Production Models)")
        print(f"{'=' * 70}")

        abl_limit = min(len(dataset), 200)
        abl_dataset = dataset[:abl_limit]

        configs = [
            ("No Mitigation",           dict(hallucination_rate=0.35, n_samples=1,  flag_threshold=1.1,
                                             use_production_rag=False, use_production_nli=False)),
            ("RAG Only (TF-IDF)",       dict(hallucination_rate=0.18, n_samples=1,  flag_threshold=1.1,
                                             use_production_rag=False, use_production_nli=False)),
            ("RAG Only (FAISS+ST)",     dict(hallucination_rate=0.18, n_samples=1,  flag_threshold=1.1,
                                             use_production_rag=True,  use_production_nli=False)),
            ("Self-Consistency (N=5)",  dict(hallucination_rate=0.35, n_samples=5,  flag_threshold=1.1,
                                             use_production_rag=False, use_production_nli=False)),
            ("Hybrid (TF-IDF + Lex)",   dict(hallucination_rate=0.18, n_samples=5,  flag_threshold=0.50,
                                             use_production_rag=False, use_production_nli=False)),
            ("Hybrid (FAISS + DeBERTa)",dict(hallucination_rate=0.18, n_samples=5,  flag_threshold=0.50,
                                             use_production_rag=True,  use_production_nli=True)),
        ]

        ablation_data = []
        for name, cfg in configs:
            print(f"\n  ▶ {name}")
            pipe = ProductionPipeline(abl_dataset, **cfg)
            res = pipe.run_dataset(abl_dataset)
            m = evaluate(res)
            ablation_data.append({
                "config": name,
                "accuracy": m["accuracy"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "flagged_rate": m["flagged_rate"],
                "avg_nli": m["avg_nli"],
                "avg_h_score": m["avg_h_score"],
            })
            print(f"    Accuracy={m['accuracy']:.1%}  F1={m['f1']:.1%}  "
                  f"Flagged={m['flagged_rate']:.1%}")

        print(f"\n{'=' * 70}")
        print(f"  ABLATION SUMMARY ({abl_limit} samples)")
        print(f"{'=' * 70}")
        print(f"  {'Configuration':<30} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Flag%':>7}")
        print(f"  {'─'*30} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
        for d in ablation_data:
            print(f"  {d['config']:<30} {d['accuracy']:>6.1%} {d['f1']:>6.1%} "
                  f"{d['precision']:>6.1%} {d['recall']:>6.1%} {d['flagged_rate']:>6.1%}")
        print(f"{'=' * 70}")

    # ── Save Results ──
    output = {
        "metadata": {
            "pipeline": "Production (sentence-transformers + FAISS + DeBERTa NLI)",
            "rag_model": "all-MiniLM-L6-v2",
            "nli_model": "cross-encoder/nli-deberta-v3-base",
            "dataset": f"TruthfulQA ({len(dataset)} questions)",
            "hallucination_rate": args.rate,
            "n_samples": args.samples,
            "threshold": args.threshold,
            "total_time_seconds": round(total_time, 1),
        },
        "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        "category_metrics": {
            cat: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
            for cat, d in cat_metrics.items()
        },
        "results": [
            {
                "question": r.question,
                "answer": r.majority_answer,
                "ground_truth": r.ground_truth,
                "correct": r.correct,
                "flagged": r.flagged,
                "uncertainty": round(r.uncertainty, 4),
                "nli_score": round(r.nli_score, 4),
                "hallucination_score": round(r.hallucination_score, 4),
                "retrieval_score": round(r.retrieval_score, 4),
                "category": r.category,
                "sample_distribution": r.details.get("sample_dist", {}),
            }
            for r in results
        ],
    }

    if ablation_data:
        output["ablation"] = ablation_data

    results_dir = PROJECT_DIR / "production_results"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "results_production.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    # Also keep a copy in project root for backward compatibility
    with open(PROJECT_DIR / "results_production.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  📄 Results saved to {output_path}")

    # Also save standard format for evaluate.py
    standard_out = [
        {
            "question": r.question,
            "answer": r.majority_answer,
            "ground_truth": r.ground_truth,
            "correct": r.correct,
            "flagged": r.flagged,
            "uncertainty": round(r.uncertainty, 4),
            "nli_score": round(r.nli_score, 4),
            "hallucination_score": round(r.hallucination_score, 4),
            "sample_distribution": r.details.get("sample_dist", {}),
        }
        for r in results
    ]
    with open(PROJECT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(standard_out, f, indent=2, ensure_ascii=False)
    print(f"  📄 Standard results saved to results.json")

    # ── Generate Charts ──
    print(f"\n  📊 Generating evaluation charts in {results_dir}...")
    generate_all_charts(results, metrics, cat_metrics, ablation_data, results_dir)
    # Also generate in project root
    generate_all_charts(results, metrics, cat_metrics, ablation_data, PROJECT_DIR)

    print(f"\n{'=' * 70}")
    print(f"  ✅ PRODUCTION EVALUATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Dataset           : TruthfulQA ({len(dataset)} questions)")
    print(f"  RAG Model         : all-MiniLM-L6-v2 + FAISS")
    print(f"  NLI Model         : cross-encoder/nli-deberta-v3-base")
    print(f"  Accuracy          : {metrics['accuracy']:.1%}")
    print(f"  Detection F1      : {metrics['f1']:.1%}")
    print(f"  Avg H-Score       : {metrics['avg_h_score']:.4f}")
    print(f"  Total Time        : {total_time:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
