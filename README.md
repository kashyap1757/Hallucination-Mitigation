# 🛡️ HalluciGuard — LLM Hallucination Mitigation Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Dataset](https://img.shields.io/badge/Benchmark-TruthfulQA%20(817%20Qs)-6f42c1)

> **Mini Project** — 2nd Semester, MIT-ADT University
> **Paper:** *"Demystifying Machine Imagination: A Comprehensive Review of Hallucination Control in Large Language Models"*
> **Author:** Kashyap Barad et al.
> **Project page:** [kashyap1757.github.io/Hallucination-Mitigation](https://kashyap1757.github.io/Hallucination-Mitigation/)

---

## 📌 Overview

HalluciGuard is a **4-stage hybrid pipeline** that detects and mitigates hallucinations in Large Language Model answers. It combines the techniques surveyed in the accompanying IEEE conference paper — retrieval grounding, self-consistency sampling, uncertainty estimation and NLI verification — into a single gate that marks each answer as **VERIFIED** or **FLAGGED**.

```
Query ─► [1. RAG Retrieval] ─► [2. Multi-Sample LLM] ─► [3. Uncertainty + NLI] ─► [4. Gate] ─► ✅ VERIFIED / ⚠️ FLAGGED
```

## 🏗️ Architecture

| Stage | Paper Section | Technique | Prototype (`pipeline.py`) | Production (`run_production_test.py`) |
|---|---|---|---|---|
| **1** | §II-A | Retrieval-Augmented Generation | TF-IDF cosine retrieval | `all-MiniLM-L6-v2` embeddings + FAISS `IndexFlatIP` |
| **2** | §II-C, §II-F | Self-Consistency Sampling | N = 5 samples + majority vote | N = 5 samples + majority vote |
| **3a** | §II-G | Uncertainty Estimation | Shannon entropy over samples | Shannon entropy over samples |
| **3b** | §II-F | Post-hoc Verification | Lexical-overlap NLI | `cross-encoder/nli-deberta-v3-base` |
| **4** | §II-F, Table I | Hallucination Gate | Composite H-score threshold | Composite H-score threshold |

**Hallucination score**

```
H_score = 0.5 × Entropy(S₁ … Sₙ) + 0.5 × (1 − NLI_entailment)
FLAGGED  if  H_score ≥ τ   (default τ = 0.50)
```

> **Note:** answer generation uses a `SimulatedLLM` with a configurable hallucination rate (default 35%), so experiments are reproducible without GPU or API access. Retrieval and NLI use real models in the production evaluation.

## 📊 Results

### Production evaluation — full TruthfulQA (817 questions)

| Metric | Value |
|---|---|
| Answer accuracy | **90.45%** (739 / 817) |
| Hallucination detection recall | **75.64%** (caught 59 of 78 wrong answers) |
| Detection precision | 22.61% |
| Detection F1 | 34.81% |
| Gate flag rate | 31.95% |
| Mean uncertainty / NLI grounding / H-score | 0.466 / 0.659 / 0.403 |
| Runtime (CPU) | 371 s |

### Ablation study (first 200 questions)

| Configuration | Accuracy | Recall | Flag rate |
|---|:---:|:---:|:---:|
| No mitigation (raw LLM) | 83.0% | — | 0% |
| RAG only (TF-IDF) | 91.5% | — | 0% |
| RAG only (FAISS + Sentence-Transformers) | 91.5% | — | 0% |
| Self-consistency only (N = 5) | 91.5% | — | 0% |
| Hybrid (TF-IDF + lexical overlap) | 98.5% | 66.7% | 13.5% |
| **Hybrid (FAISS + DeBERTa NLI)** | **98.5%** | **100%** | 25.5% |

**Key findings**
- Retrieval grounding gives the largest single jump in accuracy (83% → 91.5%).
- Only the hybrid configurations can *detect* hallucinations; the full production stack caught every wrong answer in the ablation subset.
- High recall comes at the cost of precision — many correct answers are flagged conservatively. Tuning τ trades one for the other.

<p align="center">
  <img src="figures/production/prod_confusion_matrix.png" width="45%" alt="Confusion matrix" />
  <img src="figures/production/prod_ablation_comparison.png" width="45%" alt="Ablation comparison" />
</p>

Full write-up: [`docs/PRODUCTION_MIGRATION_AND_RESULTS.md`](docs/PRODUCTION_MIGRATION_AND_RESULTS.md) · Raw metrics: [`results/production/`](results/production/)

## 📁 Repository Structure

```
Hallucination-Mitigation/
├── src/                          # Source code
│   ├── pipeline.py               # Core 4-stage pipeline (prototype, synthetic + TruthfulQA)
│   ├── run_real_dataset.py       # Runs the prototype pipeline on the TruthfulQA CSV
│   ├── run_production_test.py    # Production evaluation (FAISS + DeBERTa NLI)
│   ├── evaluate.py               # Metrics report + static charts
│   ├── dashboard.py              # Interactive Streamlit dashboard
│   └── api.py                    # FastAPI REST service
├── scripts/
│   ├── inspect_dataset.py        # Quick look at the dataset CSVs
│   └── show_results.py           # Print a summary of TruthfulQA results
├── tests/
│   └── test_api.py               # Smoke tests against a running API
├── data/                         # TruthfulQA validation split (generation + multiple choice)
├── results/
│   ├── results.json              # Latest per-question output (read by dashboard / API / evaluate)
│   ├── results_truthfulqa.json   # Prototype run on TruthfulQA
│   └── production/               # Production run: full results + metrics summary
├── figures/
│   ├── production/               # Charts from run_production_test.py
│   ├── truthfulqa/               # Charts from run_real_dataset.py
│   └── evaluation/               # Charts from evaluate.py
├── docs/
│   ├── SYSTEM_DOCUMENTATION.md   # Full system documentation
│   ├── PRODUCTION_MIGRATION_AND_RESULTS.md
│   ├── hallucination_doc.md      # Original design notes
│   └── reports/                  # IEEE paper, project overview, results & seminar reports (.docx)
├── index.html                    # GitHub Pages landing page
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/kashyap1757/Hallucination-Mitigation.git
cd Hallucination-Mitigation

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

All commands below are run from the repository root.

### Run the pipeline

```bash
python src/pipeline.py                                   # synthetic demo dataset
python src/pipeline.py --query "Who was the first president?"
python src/pipeline.py --ablation                        # ablation study
python src/pipeline.py --rate 0.40 --samples 7 --threshold 0.55
```

### Evaluate on TruthfulQA

```bash
python src/run_real_dataset.py --limit 100               # prototype pipeline
python src/run_production_test.py --ablation             # production models, all 817 Qs
python src/run_production_test.py --limit 50             # quick check
python src/evaluate.py                                   # metrics + charts from results/results.json
```

> The first production run downloads `all-MiniLM-L6-v2` and `nli-deberta-v3-base` from Hugging Face.

### Interactive dashboard

```bash
streamlit run src/dashboard.py
```

### REST API

```bash
uvicorn api:app --app-dir src --reload --port 8000
# Swagger UI: http://localhost:8000/docs

python tests/test_api.py      # in another terminal, with the API running
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status |
| `POST` | `/check` | Run the pipeline on one question |
| `POST` | `/check/batch` | Run the pipeline on several questions |
| `GET` | `/results`, `/results/{index}` | Stored per-question results |
| `GET` | `/metrics` | Aggregate evaluation metrics |
| `GET` | `/ablation` | Run an ablation study |
| `GET` | `/dataset/info`, `/dataset/categories`, `/dataset/questions` | Dataset exploration |

## 📄 Documentation & Reports

- [System documentation](docs/SYSTEM_DOCUMENTATION.md)
- [Production migration & results](docs/PRODUCTION_MIGRATION_AND_RESULTS.md)
- [IEEE conference paper](docs/reports/IEEE_Final_Conference_Paper_Updated.docx)
- [Project overview](docs/reports/Project_Overview.docx) · [Results & evaluation](docs/reports/Results_and_Evaluation.docx) · [Technical seminar report](docs/reports/Technical%20Seminar%20Report%20Format.docx)

## 📚 References

- Lin et al. (2022) — *TruthfulQA: Measuring How Models Mimic Human Falsehoods*
- Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Wang et al. (2023) — *Self-Consistency Improves Chain of Thought Reasoning in Language Models*
- He et al. (2021) — *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*
- Li et al. (2023) — *HaluEval: A Large-Scale Hallucination Evaluation Benchmark*

---

**MIT-ADT University** · 2nd Semester Mini Project · [Kashyap Barad](https://github.com/kashyap1757)
