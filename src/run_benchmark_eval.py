"""
run_benchmark_eval.py — Real-LLM evaluation on 3000 multi-benchmark samples
============================================================================
Full HalluciGuard pipeline with a REAL language model and a LEARNED
(logistic-regression) hallucination score.

  Data   : data/benchmark_3000.jsonl  — TruthfulQA 817 + HaluEval-QA 728
           + SciQ 728 + SQuAD 727   (built by src/benchmarks.py)
  Stage 1: RAG — all-MiniLM-L6-v2 embeddings + FAISS (inner product)
  Stage 2: Real LLM (default Llama-3.2-1B-Instruct), N=5 sampled answers;
           self-consistency = largest semantic cluster
  Stage 3: Semantic entropy (bidirectional-entailment clustering, DeBERTa NLI)
           + NLI grounding of the chosen answer against the retrieved context
  Stage 4: Gate — H-score via logistic regression (vs. the original fixed
           0.5/0.5 formula), fitted on a 70 % train split, reported on 30 % test

Correctness labels (y = 1 → hallucination) come from an automatic judge:
  • TruthfulQA : embedding similarity-difference (max sim to true refs >
                 max sim to false refs), as in the TruthfulQA paper's
                 BLEURT/ROUGE metrics, with MiniLM embeddings
  • HaluEval / SciQ / SQuAD : normalised gold-answer containment or
                 token-F1 ≥ 0.5 (SQuAD-style), rejected if a SciQ distractor
                 is named instead

Generations are cached in results/real_llm/gen_*.jsonl, so re-runs (e.g. to
re-fit the regression) skip the LLM entirely.

Usage:
    python src/run_benchmark_eval.py                    # full 3000 + ablation
    python src/run_benchmark_eval.py --limit 200        # quick check
    python src/run_benchmark_eval.py --llm Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import json
import math
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import env_compat  # noqa: F401  (must precede transformers)
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmarks import load_benchmark                       # noqa: E402
from hscore import (FEATURE_SETS, LogisticHScore, fixed_hscore,  # noqa: E402
                    best_f1_threshold)
from real_llm import DEFAULT_LLM                              # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
BENCHMARKS = ["TruthfulQA", "HaluEval-QA", "SciQ", "SQuAD"]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Embeddings + FAISS retrieval  (Paper §II-A)
# ─────────────────────────────────────────────────────────────────────────────

class Embedder:
    """all-MiniLM-L6-v2 = BERT encoder + mean pooling + L2 norm (same vectors as
    sentence-transformers, loaded with plain transformers)."""

    def __init__(self, name: str = EMBED_MODEL):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()

    def encode(self, texts, batch_size: int = 128) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                           max_length=256, return_tensors="pt").to(self.device)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.vstack(out).astype(np.float32)


class KnowledgeBase:
    def __init__(self, docs, embedder: Embedder):
        import faiss
        self.docs = docs
        self.emb = embedder
        vecs = embedder.encode(docs)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def retrieve_batch(self, queries, top_k: int = 3):
        scores, idx = self.index.search(self.emb.encode(queries), top_k)
        return [[(self.docs[j], float(s)) for j, s in zip(ii, ss) if j >= 0]
                for ii, ss in zip(idx, scores)]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DeBERTa NLI  (Paper §II-F)
# ─────────────────────────────────────────────────────────────────────────────

class NLI:
    def __init__(self, name: str = NLI_MODEL):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.model.to(self.device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.ent, self.con = labels["entailment"], labels["contradiction"]

    def probs(self, pairs, batch_size: int = 64) -> np.ndarray:
        """pairs = [(premise, hypothesis)] → array [n, 3] of class probabilities."""
        torch = self.torch
        out = []
        for i in range(0, len(pairs), batch_size):
            b = pairs[i:i + batch_size]
            enc = self.tok([p for p, _ in b], [h for _, h in b], padding=True,
                           truncation="only_first", max_length=512,
                           return_tensors="pt").to(self.device)
            with torch.no_grad():
                out.append(torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 3))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Semantic entropy + self-consistency  (Paper §II-G, §II-F)
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def semantic_clusters(items, nli: NLI):
    """
    items: list of (question, [ {text, logprob}, ... ]).
    Two answers are in the same meaning-cluster iff each entails the other
    (Kuhn et al., 2023). Returns per item: entropy ∈ [0,1], chosen sample,
    cluster sizes.
    """
    pairs, owners = [], []
    uniq_per_item = []
    for k, (q, samples) in enumerate(items):
        uniq = list(dict.fromkeys(_norm(s["text"]) or s["text"] for s in samples))
        uniq_per_item.append(uniq)
        for i, a in enumerate(uniq):
            for j, b in enumerate(uniq):
                if i != j:
                    pairs.append((f"{q} {a}", f"{q} {b}"))
                    owners.append((k, i, j))
    probs = nli.probs(pairs)
    entails = {}
    for (k, i, j), p in zip(owners, probs):
        entails[(k, i, j)] = int(np.argmax(p)) == nli.ent

    results = []
    for k, (q, samples) in enumerate(items):
        uniq = uniq_per_item[k]
        reps, assign = [], {}
        for i in range(len(uniq)):
            for c, r in enumerate(reps):
                if entails[(k, i, r)] and entails[(k, r, i)]:
                    assign[i] = c
                    break
            else:
                assign[i] = len(reps)
                reps.append(i)
        key = {u: assign[i] for i, u in enumerate(uniq)}
        labels = [key[_norm(s["text"]) or s["text"]] for s in samples]
        counts = Counter(labels)
        n = len(samples)
        if n > 1:
            ent = -sum((c / n) * math.log(c / n) for c in counts.values()) / math.log(n)
        else:
            ent = 0.0
        # largest cluster; ties → higher mean log-prob
        def cluster_conf(c):
            return max(s["logprob"] for s, l in zip(samples, labels) if l == c)
        best = max(counts, key=lambda c: (counts[c], cluster_conf(c)))
        chosen = max((s for s, l in zip(samples, labels) if l == best),
                     key=lambda s: s["logprob"])
        results.append({"entropy": float(ent), "chosen": chosen,
                        "cluster_sizes": sorted(counts.values(), reverse=True),
                        "agreement": counts[best] / n})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Correctness judge
# ─────────────────────────────────────────────────────────────────────────────

def _token_f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def _contains(pred: str, gold: str) -> bool:
    g = _norm(gold)
    return bool(g) and re.search(rf"\b{re.escape(g)}\b", _norm(pred)) is not None


def judge(rows, answers, embedder: Embedder):
    """Return list of (correct: bool, method: str)."""
    out = [None] * len(rows)
    tqa = [k for k, r in enumerate(rows) if r["benchmark"] == "TruthfulQA"]
    if tqa:
        texts, spans = [], []
        for k in tqa:
            r = rows[k]
            refs_t, refs_f = r["correct_answers"], r["incorrect_answers"] or ["I don't know"]
            s = len(texts)
            texts += [answers[k]] + refs_t + refs_f
            spans.append((s, len(refs_t), len(refs_f)))
        vec = embedder.encode(texts)
        for k, (s, nt, nf) in zip(tqa, spans):
            a = vec[s]
            sim_t = float(np.max(vec[s + 1:s + 1 + nt] @ a))
            sim_f = float(np.max(vec[s + 1 + nt:s + 1 + nt + nf] @ a))
            out[k] = (sim_t > sim_f, f"sim-diff {sim_t - sim_f:+.3f}")
    for k, r in enumerate(rows):
        if out[k] is not None:
            continue
        a = answers[k]
        gold_hit = any(_contains(a, g) or _token_f1(a, g) >= 0.5 for g in r["correct_answers"])
        wrong_hit = r["benchmark"] == "SciQ" and any(
            _contains(a, d) for d in r["incorrect_answers"])
        out[k] = (gold_hit and not wrong_hit,
                  "gold-match" if gold_hit and not wrong_hit else
                  "distractor" if wrong_hit else "no-match")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Generation with on-disk cache
# ─────────────────────────────────────────────────────────────────────────────

def cached_generate(name, rows, contexts, n, llm_name, out_dir, get_llm, chunk=256):
    """Generate answers for every row (resumable); returns {id: [samples]}."""
    path = out_dir / f"gen_{name}.jsonl"
    cache = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            if d.get("llm") == llm_name and len(d["samples"]) == n:
                cache[d["id"]] = d["samples"]
    todo = [k for k, r in enumerate(rows) if r["id"] not in cache]
    print(f"  ▶ {name:<14} n={n}  cached={len(rows) - len(todo)}  to-generate={len(todo)}")
    if todo:
        llm = get_llm()
        llm.batch_size = 8 if n > 1 else 32
        with open(path, "a", encoding="utf-8") as f:
            for s in range(0, len(todo), chunk):
                ks = todo[s:s + chunk]
                prompts = [llm.build_prompt(rows[k]["question"], contexts[k]) for k in ks]
                gens = llm.generate(prompts, n=n)
                for k, g in zip(ks, gens):
                    cache[rows[k]["id"]] = g
                    f.write(json.dumps({"id": rows[k]["id"], "llm": llm_name,
                                        "samples": g}, ensure_ascii=False) + "\n")
                f.flush()
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def gate_metrics(y_halluc, scores, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y_halluc, dtype=int)
    s = np.asarray(scores, dtype=float)
    flag = s >= threshold
    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1)))
    tn = int(np.sum(~flag & (y == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    both = len(set(y.tolist())) > 1
    return {
        "n": int(len(y)), "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "flag_rate": float(flag.mean()),
        "auroc": float(roc_auc_score(y, s)) if both else float("nan"),
        "auprc": float(average_precision_score(y, s)) if both else float("nan"),
        "accuracy_all": float(1 - y.mean()),
        # accuracy of the answers the gate lets through (the mitigation effect)
        "accuracy_passed": float(1 - y[~flag].mean()) if (~flag).any() else float("nan"),
        "coverage": float((~flag).mean()),
    }


def _r(d, nd=4):
    if isinstance(d, dict):
        return {k: _r(v, nd) for k, v in d.items()}
    if isinstance(d, list):
        return [_r(v, nd) for v in d]
    if isinstance(d, float):
        return round(d, nd) if math.isfinite(d) else None
    return d


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Charts
# ─────────────────────────────────────────────────────────────────────────────

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def make_charts(fig_dir, test, scorers, lr_full, ablation, per_bench):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_curve

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "legend.frameon": False,
    })
    y = test["y"]
    colors = {"Fixed 0.5/0.5": AQUA, "LogReg (core)": BLUE, "LogReg (full)": ORANGE}

    # 1. ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls="--")
    for name, s in scorers.items():
        fpr, tpr, _ = roc_curve(y, s)
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, color=colors[name], lw=2, label=f"{name}  AUROC={roc_auc_score(y, s):.3f}")
    ax.set_xlabel("False positive rate (correct answers flagged)")
    ax.set_ylabel("True positive rate (hallucinations caught)")
    ax.set_title(f"Hallucination detection ROC — held-out test (n={len(y)})")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(fig_dir / "roc_curves.png", dpi=150); plt.close(fig)

    # 2. H-score distribution (LogReg full)
    s = scorers["LogReg (full)"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 31)
    ax.hist(s[y == 0], bins=bins, color=BLUE, alpha=0.75, edgecolor=SURFACE, lw=1,
            label=f"Correct (n={(y == 0).sum()})")
    ax.hist(s[y == 1], bins=bins, color=ORANGE, alpha=0.75, edgecolor=SURFACE, lw=1,
            label=f"Hallucinated (n={(y == 1).sum()})")
    ax.axvline(lr_full.threshold, color=INK, ls="--", lw=1.5)
    ax.text(lr_full.threshold, ax.get_ylim()[1] * 0.95, f"  τ = {lr_full.threshold:.2f}",
            color=INK, va="top")
    ax.set_xlabel("Logistic H-score  P(hallucination)")
    ax.set_ylabel("Answers")
    ax.set_title("Logistic H-score by correctness — held-out test")
    ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "hscore_distribution.png", dpi=150); plt.close(fig)

    # 3. Standardised coefficients
    std = lr_full.info["standardized_coef"]
    labels = {"semantic_entropy": "Semantic entropy  U", "nli_gap": "NLI gap  1−E",
              "nli_contradiction": "NLI contradiction  C",
              "neg_logprob": "Neg. log-prob  −ℓ", "retrieval_gap": "Retrieval gap  1−R"}
    items = sorted(std.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([labels[k] for k, _ in items], [v for _, v in items], color=BLUE, height=0.55)
    for i, (_, v) in enumerate(items):
        ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", color=INK2, fontsize=10)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("Standardised coefficient (log-odds per 1 SD)")
    ax.set_title("What drives the logistic H-score")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(fig_dir / "lr_coefficients.png", dpi=150); plt.close(fig)

    # 4. Calibration
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls="--")
    for name in ("LogReg (core)", "LogReg (full)"):
        pt, pp = calibration_curve(y, scorers[name], n_bins=10, strategy="quantile")
        ax.plot(pp, pt, color=colors[name], lw=2, marker="o", ms=8, label=name)
    ax.set_xlabel("Predicted P(hallucination)")
    ax.set_ylabel("Observed hallucination rate")
    ax.set_title("Calibration — held-out test")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(fig_dir / "calibration.png", dpi=150); plt.close(fig)

    # 5. Ablation accuracy
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [a["config"] for a in ablation]
    vals = [a["accuracy"] for a in ablation]
    ax.barh(names[::-1], vals[::-1], color=BLUE, height=0.55)
    for i, a in enumerate(ablation[::-1]):
        label = f"{a['accuracy']:.1%}"
        if "coverage" in a:
            label += f"   ({a['coverage']:.0%} answered, {1 - a['coverage']:.0%} flagged)"
        ax.text(a["accuracy"] + 0.005, i, label, va="center", color=INK2, fontsize=10)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Answer accuracy  (gated rows: accuracy of answers that pass the gate, test split)")
    ax.set_title("Ablation — real LLM")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(fig_dir / "ablation_accuracy.png", dpi=150); plt.close(fig)

    # 6. Per-benchmark F1: fixed vs logistic
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(per_bench))
    w = 0.38
    f_fixed = [per_bench[b]["Fixed 0.5/0.5"]["f1"] for b in per_bench]
    f_lr = [per_bench[b]["LogReg (full)"]["f1"] for b in per_bench]
    b1 = ax.bar(x - w / 2 - 0.01, f_fixed, w, color=AQUA, label="Fixed 0.5/0.5")
    b2 = ax.bar(x + w / 2 + 0.01, f_lr, w, color=ORANGE, label="LogReg (full)")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", color=INK2, fontsize=9)
    ax.set_xticks(x, [f"{b}\n(n={per_bench[b]['LogReg (full)']['n']})" for b in per_bench])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Detection F1")
    ax.set_title("Hallucination-detection F1 per benchmark — held-out test")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(fig_dir / "per_benchmark_f1.png", dpi=150); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Real-LLM 3000-sample evaluation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--llm", default=DEFAULT_LLM)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--out", default="results/real_llm")
    args = ap.parse_args()

    from sklearn.model_selection import train_test_split

    out_dir = ROOT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT_DIR / "figures" / "real_llm"
    fig_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows = load_benchmark(limit=args.limit)
    print("=" * 72)
    print("  HalluciGuard — REAL LLM + LOGISTIC H-SCORE EVALUATION")
    print("=" * 72)
    print(f"  Samples : {len(rows)}  " +
          "  ".join(f"{b}={sum(r['benchmark'] == b for r in rows)}" for b in BENCHMARKS))
    print(f"  LLM     : {args.llm}   N={args.samples}")

    # Stage 1 — retrieval over the evidence of ALL samples
    print("\n[1] RAG: MiniLM + FAISS")
    embedder = Embedder()
    docs = list(dict.fromkeys(r["context"] for r in load_benchmark() if r["context"]))
    kb = KnowledgeBase(docs, embedder)
    hits = kb.retrieve_batch([r["question"] for r in rows], args.top_k)
    contexts = ["\n\n".join(d[:800] for d, _ in h) for h in hits]
    retrieval = [h[0][1] if h else 0.0 for h in hits]
    gold_hit = np.mean([r["context"] in [d for d, _ in h] for r, h in zip(rows, hits)])
    print(f"    {len(docs)} passages indexed; gold passage in top-{args.top_k}: {gold_hit:.1%}")

    # Stage 2 — real LLM generation
    print(f"\n[2] Real LLM generation (cached in {args.out}/)")
    holder = {}

    def get_llm():
        if "llm" not in holder:
            from real_llm import RealLLM
            holder["llm"] = RealLLM(args.llm)
        return holder["llm"]

    configs = {"hybrid": (contexts, args.samples)}
    if not args.no_ablation:
        none = [None] * len(rows)
        configs.update({"cb_greedy": (none, 1), "rag_greedy": (contexts, 1),
                        "cb_sc": (none, args.samples)})
    gens = {name: cached_generate(name, rows, ctx, n, args.llm, out_dir, get_llm)
            for name, (ctx, n) in configs.items()}
    if "llm" in holder:
        import gc
        import torch
        holder.clear()
        gc.collect()
        torch.cuda.empty_cache()

    # Stage 3 — semantic entropy, self-consistency, NLI grounding
    print("\n[3] Semantic entropy + NLI grounding (DeBERTa)")
    nli = NLI()
    chosen = {}
    sem = {}
    for name, cache in gens.items():
        samples = [cache[r["id"]] for r in rows]
        if len(samples[0]) > 1:
            sem[name] = semantic_clusters(
                [(r["question"], s) for r, s in zip(rows, samples)], nli)
            chosen[name] = [x["chosen"] for x in sem[name]]
        else:
            chosen[name] = [s[0] for s in samples]
    hyb = sem["hybrid"]
    answers = [c["text"] for c in chosen["hybrid"]]
    p = nli.probs([(c, f"{r['question']} {a}") for c, r, a in zip(contexts, rows, answers)],
                  batch_size=16)  # long (≤512-token) premises
    entail, contra = p[:, nli.ent], p[:, nli.con]

    # Correctness labels
    print("\n[4] Judging correctness")
    verdicts = {name: judge(rows, [c["text"] for c in ch], embedder)
                for name, ch in chosen.items()}
    correct = np.array([v[0] for v in verdicts["hybrid"]])
    y = (~correct).astype(int)
    for name in chosen:
        acc = np.mean([v[0] for v in verdicts[name]])
        print(f"    {name:<12} accuracy = {acc:.1%}")

    # Features
    feats = {
        "semantic_entropy": np.array([h["entropy"] for h in hyb]),
        "nli_gap": 1 - entail,
        "nli_contradiction": contra,
        "neg_logprob": -np.array([c["logprob"] for c in chosen["hybrid"]]),
        "retrieval_gap": 1 - np.array(retrieval),
    }
    X = {k: np.column_stack([feats[f] for f in fs]) for k, fs in FEATURE_SETS.items()}
    bench = np.array([r["benchmark"] for r in rows])

    # Stage 4 — logistic H-score: train / test split
    print("\n[5] Fitting logistic-regression H-score")
    idx = np.arange(len(rows))
    tr, te = train_test_split(idx, test_size=args.test_size, random_state=args.seed,
                              stratify=[f"{b}|{v}" for b, v in zip(bench, y)])
    models = {}
    for k in FEATURE_SETS:
        models[k] = LogisticHScore.fit(X[k][tr], y[tr], FEATURE_SETS[k], seed=args.seed)
        models[k].save(out_dir / f"hscore_logreg_{k}.json")
        print(f"    [{k}] {models[k].formula()}")

    h_fixed = np.array([fixed_hscore(u, e) for u, e in zip(feats["semantic_entropy"], entail)])
    h_core = models["core"].predict_proba(X["core"])
    h_full = models["full"].predict_proba(X["full"])
    fixed_tuned_tau = best_f1_threshold(y[tr], h_fixed[tr])
    scorers = {  # name → (scores, threshold)
        "Fixed 0.5/0.5": (h_fixed, 0.50),
        "Fixed 0.5/0.5 (tuned τ)": (h_fixed, fixed_tuned_tau),
        "LogReg (core)": (h_core, models["core"].threshold),
        "LogReg (full)": (h_full, models["full"].threshold),
    }
    test_metrics = {n: gate_metrics(y[te], s[te], t) for n, (s, t) in scorers.items()}
    per_bench = {
        b: {n: gate_metrics(y[te][bench[te] == b], s[te][bench[te] == b], t)
            for n, (s, t) in scorers.items()}
        for b in BENCHMARKS if (bench[te] == b).any()
    }

    # 5-fold CV AUROC over the whole set (robustness of the comparison)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    cv_auc = {"Fixed 0.5/0.5": [], "LogReg (core)": [], "LogReg (full)": []}
    for ftr, fte in StratifiedKFold(5, shuffle=True, random_state=args.seed).split(idx, y):
        cv_auc["Fixed 0.5/0.5"].append(roc_auc_score(y[fte], h_fixed[fte]))
        for k, nm in (("core", "LogReg (core)"), ("full", "LogReg (full)")):
            m = LogisticHScore.fit(X[k][ftr], y[ftr], FEATURE_SETS[k], seed=args.seed)
            cv_auc[nm].append(roc_auc_score(y[fte], m.predict_proba(X[k][fte])))
    cv_auc = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in cv_auc.items()}

    # Ablation
    ablation = []
    if not args.no_ablation:
        def acc_of(name):
            return float(np.mean([v[0] for v in verdicts[name]]))
        ablation = [
            {"config": "No mitigation (closed-book, greedy)", "accuracy": acc_of("cb_greedy")},
            {"config": "RAG only (greedy)", "accuracy": acc_of("rag_greedy")},
            {"config": f"Self-consistency only (N={args.samples})", "accuracy": acc_of("cb_sc")},
            {"config": f"RAG + self-consistency (N={args.samples})", "accuracy": acc_of("hybrid")},
        ]
        for n in ("Fixed 0.5/0.5", "LogReg (core)", "LogReg (full)"):
            m = test_metrics[n]
            ablation.append({"config": f"Hybrid + {n} gate", "accuracy": m["accuracy_passed"],
                             "coverage": m["coverage"], "f1": m["f1"], "recall": m["recall"],
                             "precision": m["precision"], "evaluated_on": "test split"})

    # ── Report ──
    print("\n" + "=" * 72)
    print(f"  HELD-OUT TEST RESULTS  (n={len(te)}, hallucination rate {y[te].mean():.1%})")
    print("=" * 72)
    print(f"  {'Scorer':<26}{'τ':>6}{'AUROC':>8}{'AUPRC':>8}{'Prec':>8}{'Rec':>8}"
          f"{'F1':>8}{'Flag%':>8}{'AccPass':>9}")
    for n, m in test_metrics.items():
        print(f"  {n:<26}{m['threshold']:>6.2f}{m['auroc']:>8.3f}{m['auprc']:>8.3f}"
              f"{m['precision']:>8.1%}{m['recall']:>8.1%}{m['f1']:>8.1%}"
              f"{m['flag_rate']:>8.1%}{m['accuracy_passed']:>9.1%}")
    print("\n  5-fold CV AUROC (all samples): " +
          "  ".join(f"{k}={v['mean']:.3f}±{v['std']:.3f}" for k, v in cv_auc.items()))
    print("\n  Per benchmark (test, LogReg full vs fixed):")
    for b, d in per_bench.items():
        lf, fx = d["LogReg (full)"], d["Fixed 0.5/0.5"]
        print(f"    {b:<12} n={lf['n']:<4} acc={lf['accuracy_all']:.1%}  "
              f"AUROC {fx['auroc']:.3f}→{lf['auroc']:.3f}  F1 {fx['f1']:.1%}→{lf['f1']:.1%}")
    if ablation:
        print("\n  Ablation:")
        for a in ablation:
            extra = f"  (coverage {a['coverage']:.1%})" if "coverage" in a else ""
            print(f"    {a['config']:<42} acc={a['accuracy']:.1%}{extra}")

    # ── Save ──
    split = np.array(["train"] * len(rows), dtype=object)
    split[te] = "test"
    records = []
    for k, r in enumerate(rows):
        records.append({
            "id": r["id"], "benchmark": r["benchmark"], "category": r["category"],
            "split": split[k], "question": r["question"], "answer": answers[k],
            "ground_truth": r["correct_answers"][0], "correct": bool(correct[k]),
            "judge": verdicts["hybrid"][k][1],
            "samples": [s["text"] for s in gens["hybrid"][r["id"]]],
            "cluster_sizes": hyb[k]["cluster_sizes"],
            "semantic_entropy": feats["semantic_entropy"][k], "nli_entailment": float(entail[k]),
            "nli_contradiction": float(contra[k]), "answer_logprob": -feats["neg_logprob"][k],
            "retrieval_score": retrieval[k],
            "h_fixed": float(h_fixed[k]), "h_logreg_core": float(h_core[k]),
            "h_logreg_full": float(h_full[k]),
            "flagged": bool(h_full[k] >= models["full"].threshold),
        })
    total_time = time.time() - t0
    summary = {
        "metadata": {
            "llm": args.llm, "n_samples": args.samples, "embedding_model": EMBED_MODEL,
            "nli_model": NLI_MODEL, "top_k": args.top_k, "n_total": len(rows),
            "benchmarks": dict(Counter(bench.tolist())), "n_train": int(len(tr)),
            "n_test": int(len(te)), "seed": args.seed,
            "gold_passage_retrieved_topk": float(gold_hit),
            "total_time_seconds": round(total_time, 1),
        },
        "accuracy": {name: float(np.mean([v[0] for v in verdicts[name]])) for name in verdicts},
        "accuracy_by_benchmark": {
            b: float(correct[bench == b].mean()) for b in BENCHMARKS if (bench == b).any()},
        "hscore_models": {k: m.to_dict() for k, m in models.items()},
        "test_metrics": test_metrics, "cv_auroc": cv_auc,
        "per_benchmark_test": per_bench, "ablation": ablation,
    }
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(_r(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "results_real_llm.json").write_text(
        json.dumps(_r({"metadata": summary["metadata"], "results": records}), indent=1,
                   ensure_ascii=False), encoding="utf-8")
    # Standard format read by evaluate.py / dashboard / API
    std = [{"question": rec["question"], "answer": rec["answer"],
            "ground_truth": rec["ground_truth"], "correct": rec["correct"],
            "flagged": rec["flagged"], "uncertainty": round(rec["semantic_entropy"], 4),
            "nli_score": round(rec["nli_entailment"], 4),
            "hallucination_score": round(rec["h_logreg_full"], 4),
            "sample_distribution": dict(Counter(rec["samples"]))} for rec in records]
    (ROOT_DIR / "results" / "results.json").write_text(
        json.dumps(std, indent=2, ensure_ascii=False), encoding="utf-8")

    make_charts(fig_dir, {"y": y[te]},
                {"Fixed 0.5/0.5": h_fixed[te], "LogReg (core)": h_core[te],
                 "LogReg (full)": h_full[te]},
                models["full"], ablation or [{"config": "RAG + SC", "accuracy": correct.mean()}],
                per_bench)
    print(f"\n  Saved metrics → {args.out}/metrics_summary.json, charts → figures/real_llm/")
    print(f"  Total time {total_time:.0f}s")


if __name__ == "__main__":
    main()
