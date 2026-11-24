"""
hscore.py — Hallucination score: fixed-weight formula vs. logistic regression
==============================================================================

Original (hand-weighted) score:
    H_fixed = 0.5 · U  +  0.5 · (1 − E)                       flag if H ≥ 0.50

Logistic-regression score (learned from labelled real-LLM outputs):
    H_LR = σ(z) = 1 / (1 + e^(−z))
    z    = β₀ + β₁·U + β₂·(1 − E)                                ("core",  2 features)
    z    = β₀ + β₁·U + β₂·(1 − E) + β₃·C + β₄·(−ℓ) + β₅·(1 − R)  ("full",  5 features)
    flag if H_LR ≥ τ   (τ chosen on the training split to maximise F1)

  U  = semantic entropy of the N sampled answers           (Paper §II-G)
  E  = DeBERTa NLI entailment P(context ⊨ answer)          (Paper §II-F)
  C  = DeBERTa NLI contradiction P(context ⊥ answer)
  ℓ  = mean token log-probability of the chosen answer    (LLM confidence)
  R  = FAISS cosine similarity of the top retrieved passage (retrieval quality)

H_LR is a probability-scaled risk that the answer is a hallucination
(target y = 1 when the answer is judged incorrect). Because of the balanced
class weights it overstates the absolute rate (~+0.1 on the test split) but
ranks answers monotonically; τ is tuned for this. The coefficients are fitted
on standardised features (L2, C chosen by 5-fold CV log-loss, balanced class weights) and
then converted back to raw-feature scale so the final formula can be applied
directly — see LogisticHScore.formula().
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

FEATURE_SETS = {
    "core": ["semantic_entropy", "nli_gap"],
    "full": ["semantic_entropy", "nli_gap", "nli_contradiction",
             "neg_logprob", "retrieval_gap"],
}

FEATURE_SYMBOLS = {
    "semantic_entropy": "U",
    "nli_gap": "(1-E)",
    "nli_contradiction": "C",
    "neg_logprob": "(-l)",
    "retrieval_gap": "(1-R)",
}


def fixed_hscore(uncertainty: float, nli_entail: float) -> float:
    """Original hand-weighted score: 0.5·U + 0.5·(1 − E)."""
    return 0.5 * uncertainty + 0.5 * (1.0 - nli_entail)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def best_f1_threshold(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Threshold that maximises F1 for flagging y=1 (hallucination)."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.round(s, 4)):
        pred = s >= t
        tp = np.sum(pred & (y == 1))
        if tp == 0:
            continue
        p = tp / pred.sum()
        r = tp / (y == 1).sum()
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


class LogisticHScore:
    """H = σ(β₀ + Σ βᵢ·xᵢ) with coefficients stored on the raw feature scale."""

    def __init__(self, features: List[str], intercept: float = 0.0,
                 coef: Dict[str, float] = None, threshold: float = 0.5, info: dict = None):
        self.features = list(features)
        self.intercept = intercept
        self.coef = coef or {f: 0.0 for f in features}
        self.threshold = threshold
        self.info = info or {}

    # ── training ─────────────────────────────────────────────────────────────
    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, features: List[str], seed: int = 42):
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        def make():
            return make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(Cs=np.logspace(-3, 2, 12), cv=cv, scoring="neg_log_loss",
                                     class_weight="balanced", max_iter=5000),
            )

        # Out-of-fold probabilities on the training set → unbiased threshold choice
        oof = cross_val_predict(make(), X, y, cv=cv, method="predict_proba")[:, 1]
        threshold = best_f1_threshold(y, oof)

        model = make().fit(X, y)
        scaler, lr = model[0], model[1]
        beta_std = lr.coef_[0]
        beta_raw = beta_std / scaler.scale_
        b0_raw = float(lr.intercept_[0] - np.sum(beta_std * scaler.mean_ / scaler.scale_))

        info = {
            "C": float(lr.C_[0]),
            "standardized_coef": {f: float(b) for f, b in zip(features, beta_std)},
            "feature_mean": {f: float(m) for f, m in zip(features, scaler.mean_)},
            "feature_std": {f: float(s) for f, s in zip(features, scaler.scale_)},
            "n_train": int(len(y)),
            "train_positive_rate": float(np.mean(y)),
            "threshold_rule": "max F1 on 5-fold out-of-fold training predictions",
        }
        return cls(features, b0_raw, {f: float(b) for f, b in zip(features, beta_raw)},
                   threshold, info)

    # ── inference ────────────────────────────────────────────────────────────
    def logit(self, X: np.ndarray) -> np.ndarray:
        w = np.array([self.coef[f] for f in self.features])
        return self.intercept + np.asarray(X, dtype=float) @ w

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.logit(X))

    def score(self, **feats) -> float:
        """Score a single answer: model.score(semantic_entropy=.., nli_gap=..)."""
        z = self.intercept + sum(self.coef[f] * feats[f] for f in self.features)
        return 1.0 / (1.0 + math.exp(-z))

    def formula(self) -> str:
        terms = " ".join(f"{'+' if self.coef[f] >= 0 else '-'} {abs(self.coef[f]):.3f}·"
                         f"{FEATURE_SYMBOLS.get(f, f)}" for f in self.features)
        return f"H = σ({self.intercept:.3f} {terms}),  flag if H ≥ {self.threshold:.3f}"

    # ── persistence ──────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {"features": self.features, "intercept": self.intercept, "coef": self.coef,
                "threshold": self.threshold, "formula": self.formula(), "info": self.info}

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d["features"], d["intercept"], d["coef"], d["threshold"], d.get("info"))

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
