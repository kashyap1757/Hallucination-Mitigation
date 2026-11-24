"""
benchmarks.py — Multi-benchmark loader (3000 QA samples)
=========================================================
Builds a single, unified evaluation set from four public QA / hallucination
benchmarks so the pipeline is not evaluated on TruthfulQA alone:

  Benchmark      Source                                   Samples   Evidence used for RAG
  ───────────    ──────────────────────────────────────   ───────   ─────────────────────
  TruthfulQA     data/generation_validation.csv (local)       817   best_answer
  HaluEval-QA    RUCAIBox/HaluEval  data/qa_data.json          728   `knowledge` passage
  SciQ           allenai SciQ (valid + test, with support)     728   `support` paragraph
  SQuAD v1.1     rajpurkar SQuAD dev-v1.1.json                 727   Wikipedia paragraph
                                                             ─────
                                                              3000

Every record has the same schema:
  id, benchmark, question, correct_answers, incorrect_answers, context, category

The raw files are downloaded once (plain HTTP — no `datasets`/pyarrow needed)
into data/raw/, sampled with a fixed seed, and written to
data/benchmark_3000.jsonl, which is the file every evaluation script reads.

Usage:
    python src/benchmarks.py                 # build data/benchmark_3000.jsonl
    python src/benchmarks.py --total 3000 --seed 42
"""

import argparse
import json
import random
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_PATH = DATA_DIR / "benchmark_3000.jsonl"

SOURCES = {
    "halueval": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
    "sciq": "https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip",
    "squad": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}


def _download(name: str) -> Path:
    url = SOURCES[name]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / Path(url).name
    if not dest.exists():
        print(f"  ↓ downloading {name}: {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Individual loaders → unified schema
# ─────────────────────────────────────────────────────────────────────────────

def load_truthfulqa() -> list:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_production_test import load_truthfulqa_csv

    rows = load_truthfulqa_csv(str(DATA_DIR / "generation_validation.csv"))
    out = []
    for i, r in enumerate(rows):
        out.append({
            "id": f"truthfulqa-{i}",
            "benchmark": "TruthfulQA",
            "question": r["question"],
            "correct_answers": r["correct_answers"],
            "incorrect_answers": [a for a in r["incorrect_answers"] if a != "I don't know"],
            # Reference fact only (not every accepted variant) — same role as the
            # evidence passages of the other benchmarks.
            "context": r["best_answer"],
            "category": r["category"],
        })
    return out


def load_halueval() -> list:
    out = []
    with open(_download("halueval"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            out.append({
                "id": f"halueval-{i}",
                "benchmark": "HaluEval-QA",
                "question": r["question"].strip(),
                "correct_answers": [r["right_answer"].strip()],
                "incorrect_answers": [r["hallucinated_answer"].strip()],
                "context": r["knowledge"].strip(),
                "category": "Multi-hop (HotpotQA)",
            })
    return out


def load_sciq() -> list:
    z = zipfile.ZipFile(_download("sciq"))
    out = []
    for split in ("valid", "test"):
        name = next(n for n in z.namelist()
                    if n.endswith(f"/{split}.json") and "__MACOSX" not in n)
        for i, r in enumerate(json.loads(z.read(name))):
            if not r.get("support", "").strip():
                continue  # need an evidence passage for RAG
            out.append({
                "id": f"sciq-{split}-{i}",
                "benchmark": "SciQ",
                "question": r["question"].strip(),
                "correct_answers": [r["correct_answer"].strip()],
                "incorrect_answers": [r[f"distractor{k}"].strip() for k in (1, 2, 3)],
                "context": r["support"].strip(),
                "category": "Science",
            })
    return out


def load_squad() -> list:
    with open(_download("squad"), encoding="utf-8") as f:
        data = json.load(f)["data"]
    out = []
    for article in data:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                golds = list(dict.fromkeys(a["text"].strip() for a in qa["answers"]))
                out.append({
                    "id": f"squad-{qa['id']}",
                    "benchmark": "SQuAD",
                    "question": qa["question"].strip(),
                    "correct_answers": golds,
                    "incorrect_answers": [],
                    "context": p["context"].strip(),
                    "category": article["title"].replace("_", " "),
                })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Build the combined set
# ─────────────────────────────────────────────────────────────────────────────

def build_benchmark(total: int = 3000, seed: int = 42) -> list:
    rng = random.Random(seed)
    tqa = load_truthfulqa()  # always use all of TruthfulQA
    others = {"HaluEval-QA": load_halueval(), "SciQ": load_sciq(), "SQuAD": load_squad()}

    remaining = total - len(tqa)
    names = list(others)
    quota = {n: remaining // len(names) for n in names}
    for n in names[: remaining % len(names)]:
        quota[n] += 1

    combined = list(tqa)
    for n in names:
        pool = others[n]
        if n == "SQuAD":
            # one question per paragraph → more diverse contexts
            by_ctx = {}
            for r in pool:
                by_ctx.setdefault(r["context"], []).append(r)
            pool = [rng.choice(v) for v in by_ctx.values()]
        combined.extend(rng.sample(pool, quota[n]))
    return combined


def load_benchmark(path: Path = OUT_PATH, limit: int = None) -> list:
    """Load the combined benchmark (build it first if missing)."""
    if not path.exists():
        save(build_benchmark(), path)
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return rows[:limit] if limit else rows


def save(rows: list, path: Path = OUT_PATH):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Build the 3000-sample multi-benchmark set")
    ap.add_argument("--total", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = build_benchmark(args.total, args.seed)
    save(rows)
    print(f"  Saved {len(rows)} samples → {OUT_PATH.relative_to(ROOT_DIR)}")
    for name, n in Counter(r["benchmark"] for r in rows).items():
        print(f"    {name:<12} {n:>5}")


if __name__ == "__main__":
    main()
