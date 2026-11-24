# Real LLM, 3000-Sample Multi-Benchmark Evaluation & Logistic H-Score

This document describes three upgrades to HalluciGuard and their results:

1. **More benchmark data** — 817 → **3000** questions from **4 benchmarks**.
2. **Logistic-regression H-score** — the fixed `0.5·U + 0.5·(1−E)` formula is replaced by a learned probability of hallucination.
3. **A real LLM** — `SimulatedLLM` (which picked from the reference answers) is replaced by **Llama-3.2-1B-Instruct**, run locally on the GPU.

Everything is produced by `src/run_benchmark_eval.py`. The raw numbers live in [`results/real_llm/`](../results/real_llm/).

---

## 1. Data — 3000 samples, 4 benchmarks (`src/benchmarks.py`)

| Benchmark | Samples | What it tests | Evidence used as the RAG knowledge base |
|---|---:|---|---|
| TruthfulQA | 817 (all) | Imitative falsehoods and misconceptions | `best_answer` |
| HaluEval-QA | 728 | Multi-hop QA (HotpotQA) with hallucinated distractors | `knowledge` passage |
| SciQ | 728 | Science exam questions with 3 distractors | `support` paragraph |
| SQuAD v1.1 | 727 | Wikipedia reading comprehension | Wikipedia paragraph |
| **Total** | **3000** | | 2928 unique passages in one FAISS index |

* Sampling is seeded (`seed=42`), and SQuAD uses one question per paragraph. The combined set is versioned at `data/benchmark_3000.jsonl`.
* Raw files are fetched over plain HTTP into `data/raw/` (git-ignored). `datasets`/pyarrow is not needed.
* Retrieval searches **all 2928 passages**, not just the matching one. The gold passage is in the top 3 for **89.1%** of questions, so retrieval errors are part of the evaluation.
* **Bug fix:** TruthfulQA stores answer lists numpy-style, `['a' 'b']`. `ast.literal_eval` silently concatenated these into a single string, so earlier runs compared against one glued-together reference. `parse_array_field` now tokenizes each string literal separately; the fix is in both `run_production_test.py` and `run_real_dataset.py`.

## 2. Real LLM (`src/real_llm.py`)

* **Model:** `meta-llama/Llama-3.2-1B-Instruct`, bf16, on an RTX 5050 (8 GB). Any HF chat model can be used via `--llm`.
* **Prompt:** a system instruction to answer briefly, plus the top-3 retrieved passages (≤ 800 characters each) and the question. The model **never sees the reference answers**.
* **Sampling:** N = 5 samples (temperature 0.7, top-p 0.9) per question for self-consistency. Greedy decoding is used for the single-answer ablations.
* **Confidence:** the mean token log-probability ℓ of each answer is computed under the raw model distribution, in a separate memory-bounded forward pass.
* **Cost:** 12,000 generation calls (4 configurations × 3000 questions) in about 75 min. Results are cached in `results/real_llm/gen_*.jsonl`, so re-runs take about 5 min.

### Semantic entropy replaces string entropy

Free-text answers almost never match exactly as strings. The old entropy counted distinct strings, so it would score nearly every question as maximally uncertain. Answers are therefore clustered by meaning instead: two answers share a cluster when DeBERTa-NLI finds that **each entails the other**, given the question (Kuhn et al., 2023). Then:

* **U** = normalized entropy over the cluster sizes.
* **Self-consistency answer** = the most confident member of the largest cluster.

### Correctness judge (label y = 1 means hallucination)

* **TruthfulQA:** the similarity-difference rule from the TruthfulQA paper. The answer counts as correct when its maximum cosine similarity (MiniLM) to the true references exceeds its maximum similarity to the false ones.
* **HaluEval, SciQ, SQuAD:** the normalized gold answer is contained in the response, or token-F1 ≥ 0.5 (SQuAD-style). A SciQ answer that names a distractor counts as wrong.
* **Manual audit** (`scripts/final_checks.py`): 60 random answers, 15 per benchmark, were checked by hand. The judge agreed on **55/60 (91.7%)**. All 5 disagreements were correct paraphrases labelled as hallucinations. The judge is therefore conservative: hallucination rates are slightly overestimated, never hidden. Both H-scores are evaluated against the same labels, so the comparison between them is unaffected.

## 3. Logistic-regression H-score (`src/hscore.py`)

**Before (hand-weighted):**

```
H = 0.5 · U + 0.5 · (1 − E)                 flag if H ≥ 0.50
```

**After (learned):**

```
H = σ(z) = 1 / (1 + e^(−z))  = P(answer is a hallucination)

core:  z = −1.385 + 2.073·U + 0.871·(1−E)                                   flag if H ≥ 0.521
full:  z = −2.074 + 1.373·U + 0.481·(1−E) + 0.469·C + 1.152·(−ℓ) + 2.008·(1−R)   flag if H ≥ 0.470
```

| Symbol | Feature |
|---|---|
| U | semantic entropy of the 5 samples |
| E | DeBERTa NLI entailment P(retrieved context ⊨ question + answer) |
| C | DeBERTa NLI contradiction probability |
| ℓ | mean token log-probability of the chosen answer (LLM confidence) |
| R | FAISS cosine similarity of the top retrieved passage |

**Training protocol:**

* Stratified 70/30 split (by benchmark × label): 2100 train / 900 test.
* Features are standardized; L2 logistic regression with balanced class weights; C selected by 5-fold CV log-loss.
* The threshold τ maximizes F1 on out-of-fold *training* predictions, so the test set is never used for tuning.
* Coefficients are converted back to the raw feature scale, so the formulas above can be applied directly. They are saved in `results/real_llm/hscore_logreg_{core,full}.json`.

The **core** model uses exactly the two inputs of the original formula, so it serves as a drop-in replacement. `src/pipeline.py` (and therefore the API, dashboard and older scripts) now computes `compute_hallucination_score` with this model. The original formula is kept as `compute_hallucination_score_fixed`.

**Standardized coefficients (full model).** These show which signals matter most: semantic entropy +0.43, negative log-prob +0.40, retrieval gap +0.30, NLI gap +0.17, NLI contradiction +0.17. All are positive, as expected.

## 4. Results (Llama-3.2-1B-Instruct, 3000 questions)

### Ablation

| Configuration | Accuracy |
|---|---:|
| No mitigation (closed-book, greedy) | 28.3% |
| Self-consistency only (closed-book, N=5) | 28.7% |
| RAG only (greedy) | 63.2% |
| RAG + self-consistency (N=5) | 64.2% |
| + Fixed 0.5/0.5 gate *(answers passed, test)* | 78.5% (51.8% answered) |
| + Logistic core gate *(answers passed, test)* | 79.3% (51.4% answered) |
| **+ Logistic full gate** *(answers passed, test)* | **80.1% (53.7% answered)** |

Accuracy by benchmark (RAG + SC): TruthfulQA 63.5%, HaluEval-QA 62.2%, SciQ 71.6%, SQuAD 59.6%.

### Hallucination detection on the held-out test split (n = 900, 35.8% hallucinated)

| H-score | τ | AUROC | AUPRC | Precision | Recall | F1 | Flag rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed 0.5/0.5 | 0.50 | 0.702 | 0.555 | 51.2% | 68.9% | 58.7% | 48.2% |
| Fixed 0.5/0.5 (τ tuned on train) | 0.62 | 0.702 | 0.555 | 51.5% | 67.4% | 58.4% | 46.8% |
| Logistic — core (U, 1−E) | 0.52 | 0.714 | 0.560 | 51.7% | 70.2% | 59.6% | 48.6% |
| **Logistic — full (5 features)** | 0.47 | **0.752** | **0.616** | **54.2%** | **70.2%** | **61.2%** | 46.3% |

* **5-fold CV AUROC (all 3000):** fixed 0.699 ± 0.020, core 0.703 ± 0.015, full **0.739 ± 0.018**.
* **Paired bootstrap** (2000 resamples of the test set):

  | Model vs. fixed | ΔAUROC | 95% CI | ΔF1 | 95% CI |
  |---|---:|---|---:|---|
  | Logistic full | **+0.050** | [+0.028, +0.072] | +0.017 | [−0.012, +0.045] |
  | Logistic core | +0.011 | [−0.001, +0.024] | +0.001 | |

  The full model's AUROC gain is significant. Its F1 gain, and the core model's gains, are within noise.
* **Per benchmark (test, fixed → full):**

  | Benchmark | AUROC | F1 |
  |---|---|---|
  | TruthfulQA | 0.695 → 0.706 | 57.9 → 61.4% |
  | HaluEval-QA | 0.691 → 0.713 | 56.0 → 55.9% |
  | SciQ | 0.714 → 0.797 | 55.7 → 54.3% |
  | SQuAD | 0.715 → 0.813 | 63.6 → 69.4% |

### Takeaways

1. **Retrieval does the heavy lifting:** a 1B model goes from 28% to 63% accuracy. Self-consistency adds only about one point with a real LLM.
2. **The gate is the mitigation:** withholding the riskiest ~46% of answers raises the accuracy of delivered answers from 64% to 80%.
3. **Learning the weights alone helps little** (core vs. fixed: +0.011 AUROC). The real gain comes from letting the regression **combine more signals**, especially LLM log-probability and retrieval quality, which the fixed formula could not weight (+0.050 AUROC, significant).
4. The logistic score is **monotone in the true risk** (`figures/real_llm/calibration.png`): higher H always means more hallucinations. It overstates the absolute rate by about 10 points, an expected effect of the balanced class weights. The threshold is tuned for this, but H should be read as a risk ranking, not an exact percentage.

### Limitations

* The LLM is small (1B). Larger models will be more accurate, and the coefficients should be refit for them. Re-running `run_benchmark_eval.py --llm <model>` does this.
* The labels come from an automatic judge (91.7% agreement with a manual audit, biased toward strictness).
* The prototype pipeline (`pipeline.py`, API, dashboard) still uses `SimulatedLLM` with string entropy and lexical NLI. It uses the logistic *formula*, but its features are only approximations of those the model was trained on.

## 5. Reproduce

```bash
python src/benchmarks.py              # build data/benchmark_3000.jsonl
python src/run_benchmark_eval.py      # ~75 min first time on an 8 GB GPU; ~5 min from cache
python scripts/final_checks.py        # bootstrap CIs + judge audit
```

Figures are written to `figures/real_llm/`: `roc_curves.png`, `hscore_distribution.png`, `lr_coefficients.png`, `calibration.png`, `ablation_accuracy.png`, `per_benchmark_f1.png`.

**Environment notes:** `src/env_compat.py` hides a torchvision build that doesn't match the installed torch. MiniLM embeddings are computed with plain `transformers`, because `sentence-transformers` 6.x requires `datasets`/pyarrow, which Windows Application Control blocks on this machine.
