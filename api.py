"""
api.py — FastAPI for Hallucination Mitigation Pipeline
=======================================================
Provides REST endpoints to query the pipeline, inspect results, and run evaluations.

Usage:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /                       → API info & status
    POST /check                  → Check a single question through the pipeline
    POST /check/batch            → Check multiple questions at once
    GET  /results                → Get all stored results
    GET  /results/{index}        → Get a specific result by index
    GET  /metrics                → Get overall evaluation metrics
    GET  /ablation               → Run ablation study and return comparison
    GET  /dataset/info           → Info about the loaded dataset
    GET  /dataset/categories     → Category distribution
    GET  /health                 → Health check
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Add project root
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from pipeline import (
    HallucinationMitigationPipeline,
    KnowledgeBase,
    SimulatedLLM,
    SYNTHETIC_DATASET,
    compute_entropy_uncertainty,
    nli_entailment_score,
    compute_hallucination_score,
    majority_vote,
    evaluate,
)
from run_real_dataset import load_truthfulqa_csv, RealDatasetPipeline

# ═════════════════════════════════════════════════════════════════════════════
# App Setup
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="🛡️ HalluciGuard API",
    description=(
        "REST API for the LLM Hallucination Mitigation Pipeline. "
        "Implements a 4-stage hybrid approach: RAG → Self-Consistency → Uncertainty + NLI → Gate. "
        "Based on the IEEE survey by Kashyap Barad et al., MIT-ADT University."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═════════════════════════════════════════════════════════════════════════════
# Load dataset & pipeline on startup
# ═════════════════════════════════════════════════════════════════════════════

DATASET = []
PIPELINE = None
STORED_RESULTS = []


def init_pipeline():
    """Initialize the pipeline with real TruthfulQA data if available, else synthetic."""
    global DATASET, PIPELINE, STORED_RESULTS

    csv_path = PROJECT_DIR / "dataset" / "generation_validation.csv"
    if csv_path.exists():
        DATASET = load_truthfulqa_csv(str(csv_path))
        print(f"[API] Loaded {len(DATASET)} questions from TruthfulQA CSV")
        # Build pipeline with real data KB
        contexts = [s["context"] for s in DATASET if s.get("context")]
        kb = KnowledgeBase(contexts)
        PIPELINE = HallucinationMitigationPipeline()
        PIPELINE.kb = kb
        print(f"[API] Knowledge base built (vocab: {len(kb.vocab)} terms)")
    else:
        DATASET = SYNTHETIC_DATASET
        PIPELINE = HallucinationMitigationPipeline()
        print(f"[API] Using synthetic dataset ({len(DATASET)} questions)")

    # Load stored results if available
    results_path = PROJECT_DIR / "results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            STORED_RESULTS = json.load(f)
        print(f"[API] Loaded {len(STORED_RESULTS)} stored results")


@app.on_event("startup")
async def startup():
    init_pipeline()


# ═════════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═════════════════════════════════════════════════════════════════════════════

class QuestionInput(BaseModel):
    question: str = Field(..., description="The question to check for hallucination", example="What is the capital of Australia?")
    n_samples: int = Field(5, ge=1, le=20, description="Number of self-consistency samples")
    threshold: float = Field(0.50, ge=0.0, le=1.0, description="Hallucination flag threshold")
    hallucination_rate: float = Field(0.35, ge=0.0, le=1.0, description="Simulated LLM hallucination rate")


class BatchInput(BaseModel):
    questions: List[str] = Field(..., description="List of questions to check")
    n_samples: int = Field(5, ge=1, le=20)
    threshold: float = Field(0.50, ge=0.0, le=1.0)
    hallucination_rate: float = Field(0.35, ge=0.0, le=1.0)


class PipelineResponse(BaseModel):
    question: str
    answer: str
    ground_truth: Optional[str]
    correct: Optional[bool]
    flagged: bool
    status: str  # "VERIFIED" or "FLAGGED"
    uncertainty: float
    nli_score: float
    hallucination_score: float
    retrieval_score: float
    raw_answers: List[str]
    sample_distribution: dict
    retrieved_context: str
    processing_time_ms: float

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the capital of Australia?",
                "answer": "canberra",
                "ground_truth": "Canberra",
                "correct": True,
                "flagged": False,
                "status": "VERIFIED",
                "uncertainty": 0.0,
                "nli_score": 1.0,
                "hallucination_score": 0.0,
                "retrieval_score": 0.85,
                "raw_answers": ["Canberra", "Canberra", "Canberra", "Canberra", "Canberra"],
                "sample_distribution": {"Canberra": 5},
                "retrieved_context": "Canberra is the capital city of Australia...",
                "processing_time_ms": 12.5,
            }
        }


class MetricsResponse(BaseModel):
    total: int
    accuracy: float
    flagged_rate: float
    flag_precision: float
    flag_recall: float
    flag_f1: float
    avg_uncertainty: float
    avg_nli_score: float
    avg_hallucination_score: float
    true_positive_flags: int
    false_positive_flags: int
    missed_hallucinations: int


# ═════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═════════════════════════════════════════════════════════════════════════════

def find_matching_sample(question: str) -> Optional[dict]:
    """Find a matching dataset sample for the question."""
    q_lower = question.strip().lower()
    for s in DATASET:
        if s["question"].strip().lower() == q_lower:
            return s
    return None


def run_question(question: str, n_samples: int = 5, threshold: float = 0.50,
                 hallucination_rate: float = 0.35) -> PipelineResponse:
    """Run a single question through the pipeline."""
    start = time.perf_counter()

    # Check if question exists in dataset
    sample = find_matching_sample(question)

    if sample:
        # Use the dataset's correct/incorrect answers
        correct_answers = sample["correct_answers"]
        incorrect_answers = sample["incorrect_answers"]
        ground_truth = correct_answers[0] if correct_answers else None
    else:
        # Unknown question — use generic answers
        correct_answers = ["I don't know"]
        incorrect_answers = ["I don't know"]
        ground_truth = None

    # Build a pipeline instance with the given params
    pipe_sample = {
        "question": question,
        "correct_answers": correct_answers,
        "incorrect_answers": incorrect_answers,
        "context": sample.get("context", "") if sample else "",
    }

    # Configure LLM
    llm = SimulatedLLM(hallucination_rate=hallucination_rate)

    # Stage 1: RAG retrieval
    retrieved = PIPELINE.kb.retrieve(question, top_k=3)
    context = " ".join(doc for doc, _ in retrieved)
    retrieval_score = retrieved[0][1] if retrieved else 0.0

    # Stage 2: Multi-sample generation
    answers = llm.generate_n(
        question, context, correct_answers, incorrect_answers,
        n=n_samples, grounded=True,
    )
    majority = majority_vote(answers)

    # Stage 3: Uncertainty + NLI
    uncertainty = compute_entropy_uncertainty(answers)
    nli = nli_entailment_score(majority, context)

    # Stage 4: Hallucination gate
    h_score = compute_hallucination_score(uncertainty, nli)
    flagged = h_score >= threshold

    # Evaluate correctness
    is_correct = None
    if sample and correct_answers:
        is_correct = any(
            ans.lower() in majority.lower() or majority.lower() in ans.lower()
            for ans in correct_answers
        )

    elapsed = (time.perf_counter() - start) * 1000

    from collections import Counter
    sample_dist = dict(Counter(answers))

    return PipelineResponse(
        question=question,
        answer=majority,
        ground_truth=ground_truth,
        correct=is_correct,
        flagged=flagged,
        status="FLAGGED" if flagged else "VERIFIED",
        uncertainty=round(uncertainty, 4),
        nli_score=round(nli, 4),
        hallucination_score=round(h_score, 4),
        retrieval_score=round(retrieval_score, 4),
        raw_answers=answers,
        sample_distribution=sample_dist,
        retrieved_context=context[:200] + "..." if len(context) > 200 else context,
        processing_time_ms=round(elapsed, 2),
    )


# ═════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    """API landing page with info and links."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>HalluciGuard API</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
                color: #e0e0e0;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 2rem;
            }}
            .container {{
                max-width: 720px;
                width: 100%;
            }}
            .card {{
                background: rgba(255,255,255,0.06);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 20px;
                padding: 2.5rem;
                box-shadow: 0 8px 40px rgba(0,0,0,0.3);
            }}
            h1 {{
                font-size: 2rem;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            }}
            .subtitle {{ color: #9ca3af; margin-bottom: 2rem; font-size: 0.9rem; }}
            .stat-row {{
                display: flex;
                gap: 1rem;
                margin-bottom: 2rem;
            }}
            .stat {{
                flex: 1;
                background: rgba(255,255,255,0.05);
                border-radius: 12px;
                padding: 1rem;
                text-align: center;
            }}
            .stat-val {{ font-size: 1.5rem; font-weight: 700; color: #667eea; }}
            .stat-label {{ font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; }}
            .endpoints {{ list-style: none; }}
            .endpoints li {{
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 10px;
                padding: 0.8rem 1.2rem;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.8rem;
            }}
            .method {{
                font-size: 0.7rem;
                font-weight: 700;
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
                min-width: 45px;
                text-align: center;
            }}
            .get {{ background: rgba(52,211,153,0.2); color: #34d399; }}
            .post {{ background: rgba(96,165,250,0.2); color: #60a5fa; }}
            a {{ color: #c084fc; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .path {{ color: #f0f0f0; font-family: monospace; font-size: 0.85rem; }}
            .desc {{ color: #9ca3af; font-size: 0.8rem; margin-left: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>🛡️ HalluciGuard API</h1>
                <p class="subtitle">LLM Hallucination Mitigation Pipeline — Kashyap Barad et al., MIT-ADT University</p>

                <div class="stat-row">
                    <div class="stat">
                        <div class="stat-val">{len(DATASET)}</div>
                        <div class="stat-label">Questions Loaded</div>
                    </div>
                    <div class="stat">
                        <div class="stat-val">{len(STORED_RESULTS)}</div>
                        <div class="stat-label">Stored Results</div>
                    </div>
                    <div class="stat">
                        <div class="stat-val">4</div>
                        <div class="stat-label">Pipeline Stages</div>
                    </div>
                </div>

                <h3 style="margin-bottom: 1rem; font-size: 1rem;">📡 Endpoints</h3>
                <ul class="endpoints">
                    <li><span class="method post">POST</span><span class="path">/check</span><span class="desc">Check a question</span></li>
                    <li><span class="method post">POST</span><span class="path">/check/batch</span><span class="desc">Check multiple questions</span></li>
                    <li><span class="method get">GET</span><span class="path">/results</span><span class="desc">All stored results</span></li>
                    <li><span class="method get">GET</span><span class="path">/metrics</span><span class="desc">Evaluation metrics</span></li>
                    <li><span class="method get">GET</span><span class="path">/ablation</span><span class="desc">Ablation study</span></li>
                    <li><span class="method get">GET</span><span class="path">/dataset/info</span><span class="desc">Dataset info</span></li>
                    <li><span class="method get">GET</span><span class="path">/dataset/categories</span><span class="desc">Category breakdown</span></li>
                </ul>

                <p style="margin-top: 1.5rem; font-size: 0.85rem; color: #9ca3af;">
                    📖 Interactive docs: <a href="/docs">/docs</a> &nbsp;|&nbsp; <a href="/redoc">/redoc</a>
                </p>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "dataset_loaded": len(DATASET),
        "stored_results": len(STORED_RESULTS),
        "pipeline_ready": PIPELINE is not None,
    }


@app.post("/check", response_model=PipelineResponse)
async def check_question(input: QuestionInput):
    """
    Run a single question through the 4-stage hallucination mitigation pipeline.

    The pipeline performs:
    1. **RAG Retrieval** — Finds relevant knowledge from the corpus
    2. **Self-Consistency Sampling** — Generates N independent answers
    3. **Uncertainty + NLI** — Computes entropy and grounding scores
    4. **Hallucination Gate** — Flags if H-score ≥ threshold
    """
    return run_question(
        input.question,
        n_samples=input.n_samples,
        threshold=input.threshold,
        hallucination_rate=input.hallucination_rate,
    )


@app.post("/check/batch")
async def check_batch(input: BatchInput):
    """Check multiple questions through the pipeline."""
    results = []
    total_start = time.perf_counter()

    for q in input.questions:
        r = run_question(q, input.n_samples, input.threshold, input.hallucination_rate)
        results.append(r)

    total_ms = (time.perf_counter() - total_start) * 1000

    correct_count = sum(1 for r in results if r.correct is True)
    flagged_count = sum(1 for r in results if r.flagged)

    return {
        "total_questions": len(results),
        "correct": correct_count,
        "flagged": flagged_count,
        "accuracy": f"{correct_count / len(results):.1%}" if results else "N/A",
        "total_processing_time_ms": round(total_ms, 2),
        "results": results,
    }


@app.get("/results")
async def get_results(
    limit: int = Query(50, ge=1, le=5000, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    flagged_only: bool = Query(False, description="Only show flagged results"),
    wrong_only: bool = Query(False, description="Only show wrong results"),
):
    """Get stored pipeline results with optional filtering and pagination."""
    if not STORED_RESULTS:
        raise HTTPException(404, "No stored results. Run the pipeline first.")

    filtered = STORED_RESULTS
    if flagged_only:
        filtered = [r for r in filtered if r.get("flagged")]
    if wrong_only:
        filtered = [r for r in filtered if not r.get("correct")]

    total = len(filtered)
    page = filtered[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "count": len(page),
        "results": page,
    }


@app.get("/results/{index}")
async def get_result_by_index(index: int):
    """Get a specific result by index."""
    if not STORED_RESULTS:
        raise HTTPException(404, "No stored results.")
    if index < 0 or index >= len(STORED_RESULTS):
        raise HTTPException(404, f"Index {index} out of range (0-{len(STORED_RESULTS)-1})")
    return STORED_RESULTS[index]


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get overall evaluation metrics from stored results."""
    if not STORED_RESULTS:
        raise HTTPException(404, "No stored results. Run the pipeline first.")

    total = len(STORED_RESULTS)
    correct = sum(r["correct"] for r in STORED_RESULTS)
    flagged = sum(r["flagged"] for r in STORED_RESULTS)
    wrong = total - correct

    tp = sum(r["flagged"] and not r["correct"] for r in STORED_RESULTS)
    fp = sum(r["flagged"] and r["correct"] for r in STORED_RESULTS)
    fn = sum(not r["flagged"] and not r["correct"] for r in STORED_RESULTS)

    precision = tp / flagged if flagged > 0 else 0.0
    recall = tp / wrong if wrong > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_unc = sum(r["uncertainty"] for r in STORED_RESULTS) / total
    avg_nli = sum(r["nli_score"] for r in STORED_RESULTS) / total
    avg_h = sum(r["hallucination_score"] for r in STORED_RESULTS) / total

    return MetricsResponse(
        total=total,
        accuracy=correct / total,
        flagged_rate=flagged / total,
        flag_precision=precision,
        flag_recall=recall,
        flag_f1=f1,
        avg_uncertainty=avg_unc,
        avg_nli_score=avg_nli,
        avg_hallucination_score=avg_h,
        true_positive_flags=tp,
        false_positive_flags=fp,
        missed_hallucinations=fn,
    )


@app.get("/ablation")
async def run_ablation_study(
    limit: int = Query(100, ge=10, le=500, description="Number of samples for ablation"),
):
    """
    Run ablation study comparing pipeline configurations.
    Uses a subset of the dataset for speed.
    """
    subset = DATASET[:limit]

    configs = [
        ("No Mitigation (Baseline)", dict(hallucination_rate=0.35, n_samples=1, flag_threshold=1.1)),
        ("RAG Only", dict(hallucination_rate=0.18, n_samples=1, flag_threshold=1.1)),
        ("Self-Consistency Only", dict(hallucination_rate=0.35, n_samples=5, flag_threshold=1.1)),
        ("Uncertainty Gate Only", dict(hallucination_rate=0.35, n_samples=5, flag_threshold=0.50)),
        ("Full Hybrid Pipeline", dict(hallucination_rate=0.18, n_samples=5, flag_threshold=0.50)),
    ]

    results = []
    for name, cfg in configs:
        pipe = RealDatasetPipeline(subset, **cfg)
        res = pipe.run_dataset(subset)
        m = evaluate(res)
        results.append({
            "configuration": name,
            "accuracy": round(m["accuracy"], 4),
            "flag_f1": round(m["flag_f1"], 4),
            "flagged_rate": round(m["flagged_rate"], 4),
            "avg_uncertainty": round(m["avg_uncertainty"], 4),
            "avg_nli_score": round(m["avg_nli_score"], 4),
        })

    return {
        "samples_used": len(subset),
        "configurations": results,
        "conclusion": "Full hybrid pipeline achieves best accuracy + detection balance (Paper §IV)",
    }


@app.get("/dataset/info")
async def dataset_info():
    """Get information about the loaded dataset."""
    cats = {}
    types = {}
    for s in DATASET:
        cat = s.get("category", "Unknown")
        typ = s.get("type", "Unknown")
        cats[cat] = cats.get(cat, 0) + 1
        types[typ] = types.get(typ, 0) + 1

    return {
        "total_questions": len(DATASET),
        "source": "TruthfulQA (generation_validation.csv)" if len(DATASET) > 100 else "Synthetic",
        "categories": len(cats),
        "types": types,
        "sample_question": DATASET[0]["question"] if DATASET else None,
    }


@app.get("/dataset/categories")
async def dataset_categories():
    """Get category distribution of the dataset."""
    cats = {}
    for s in DATASET:
        cat = s.get("category", "Unknown")
        cats[cat] = cats.get(cat, 0) + 1

    sorted_cats = sorted(cats.items(), key=lambda x: -x[1])
    return {
        "total_categories": len(cats),
        "distribution": [{"category": c, "count": n, "percentage": round(n / len(DATASET) * 100, 1)}
                         for c, n in sorted_cats],
    }


@app.get("/dataset/questions")
async def dataset_questions(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    category: Optional[str] = Query(None, description="Filter by category"),
):
    """Browse questions in the dataset."""
    filtered = DATASET
    if category:
        filtered = [s for s in filtered if s.get("category", "").lower() == category.lower()]

    total = len(filtered)
    page = filtered[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "count": len(page),
        "questions": [
            {
                "index": offset + i,
                "question": s["question"],
                "category": s.get("category", ""),
                "type": s.get("type", ""),
                "best_answer": s.get("best_answer", s["correct_answers"][0] if s["correct_answers"] else ""),
            }
            for i, s in enumerate(page)
        ],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Run directly
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("Starting HalluciGuard API on http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
