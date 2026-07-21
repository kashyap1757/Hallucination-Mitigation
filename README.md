# 🛡️ HalluciGuard — LLM Hallucination Mitigation Pipeline

> **Mini Project** — 2nd Semester, MIT-ADT University  
> **Paper:** "Demystifying Machine Imagination: A Comprehensive Review of Hallucination Control in Large Language Models"  
> **Author:** Kashyap Barad et al.

---

## 📌 Overview

A **4-stage hybrid pipeline** for detecting and mitigating hallucinations in Large Language Models, implementing techniques surveyed in the IEEE conference paper.

```
Query → [RAG Retrieval] → [Multi-Sample LLM] → [Uncertainty + NLI] → [Gate] → ✅ VERIFIED / ⚠️ FLAGGED
```

## 🏗️ Architecture

| Stage | Paper Section | Technique | Implementation |
|---|---|---|---|
| **Stage 1** | §II-A | RAG (Retrieval-Augmented Generation) | TF-IDF cosine retrieval (FAISS-upgradeable) |
| **Stage 2** | §II-C, §II-F | Self-Consistency Sampling | N=5 independent samples + majority vote |
| **Stage 3a** | §II-G | Uncertainty Estimation | Semantic entropy across N samples |
| **Stage 3b** | §II-F | Post-hoc Verification | NLI entailment scoring |
| **Stage 4** | §II-F, Table I | Hallucination Gate | Composite H-score thresholding |

### Key Equation

```
H_score = 0.5 × Entropy(S₁...Sₙ) + 0.5 × (1 - NLI_entailment)
If H_score ≥ threshold → FLAGGED
```

## 📁 Project Structure

```
Hallucination Mitigation/
├── pipeline.py          # Core 4-stage pipeline (9 modules)
├── evaluate.py          # Charts & evaluation metrics
├── dashboard.py         # Interactive Streamlit dashboard
├── results.json         # Pipeline output (per-question scores)
├── requirements.txt     # Python dependencies
├── hallucination_doc.md # Full project documentation
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install numpy scikit-learn matplotlib pandas streamlit plotly
```

### 2. Run the Pipeline
```bash
python pipeline.py
```

### 3. Evaluate Results
```bash
python evaluate.py
```

### 4. Launch Interactive Dashboard
```bash
streamlit run dashboard.py
```

## ⚙️ Command-Line Options

```bash
# Run with synthetic dataset (default)
python pipeline.py

# Run on real TruthfulQA dataset (requires HuggingFace access)
python pipeline.py --real

# Single custom query
python pipeline.py --query "Who was the first president?"

# Run ablation study
python pipeline.py --ablation

# Custom parameters
python pipeline.py --rate 0.40 --samples 7 --threshold 0.55
```

## 📊 Results

Evaluated on a **15-sample synthetic TruthfulQA-style dataset**:

| Configuration | Accuracy | Notes |
|---|---|---|
| No mitigation (baseline) | ~65% | Raw LLM output |
| RAG only | **93.3%** | Best single technique |
| Self-consistency only | ~73% | Majority voting helps |
| Full hybrid pipeline | **86.7%** | Most robust overall |

### Key Findings
- **RAG is the strongest single technique** — confirms the paper's claims
- Uncertainty estimation adds robustness when retrieval context is weak
- NLI entailment reduces false negatives
- **Composite gate outperforms any single technique on edge cases**

## 🔧 Production Upgrades (Drop-in Swaps)

| Component | Current (Demo) | Production Replacement |
|---|---|---|
| LLM | `SimulatedLLM` | `Mistral-7B` via HuggingFace |
| Retrieval | TF-IDF cosine | `sentence-transformers` + FAISS |
| NLI | Lexical overlap | `cross-encoder/nli-deberta-v3-base` |
| Dataset | Synthetic (15 Qs) | TruthfulQA (817 Qs) |

## 📚 References

- Lin et al. (2022) — TruthfulQA: Measuring How Models Mimic Human Falsehoods
- Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- HaluEval — A Large-Scale Hallucination Evaluation Benchmark
- Mistral AI — Mistral-7B / Mixtral
- Microsoft Research — DeBERTa-v3

---

**MIT-ADT University** | 2nd Semester Mini Project
