# Real-LLM evaluation outputs

Produced by `python src/run_benchmark_eval.py` (Llama-3.2-1B-Instruct, 3000 questions) and `python scripts/final_checks.py`.

| File | Contents |
|---|---|
| `metrics_summary.json` | Accuracies, held-out test metrics for every H-score, 5-fold CV AUROC, per-benchmark metrics, ablation, fitted models |
| `results_real_llm.json` | Per-question records: answer, 5 samples, clusters, features, all H-scores, correctness, train/test split |
| `hscore_logreg_core.json` / `hscore_logreg_full.json` | Fitted logistic H-score models (raw-scale coefficients + threshold); `pipeline.py` loads the core one |
| `final_checks.json` | Paired-bootstrap 95% CIs (test set) and the manual judge audit (60 items) |
| `gen_hybrid.jsonl` | Cached generations: RAG + 5 samples |
| `gen_rag_greedy.jsonl`, `gen_cb_greedy.jsonl`, `gen_cb_sc.jsonl` | Ablation generations (RAG greedy, closed-book greedy, closed-book 5 samples) |
| `run_log.txt`, `run_log_generation.txt` | Console logs (final run from cache; original generation run) |

Write-up: [`docs/REAL_LLM_AND_LOGISTIC_HSCORE.md`](../../docs/REAL_LLM_AND_LOGISTIC_HSCORE.md)
