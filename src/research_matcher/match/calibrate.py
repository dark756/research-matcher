"""Weight calibration on hand-labelled pairs (plan §8.2): LogisticRegression on signal vectors, keep defaults if AUC doesn't improve."""
from __future__ import annotations

import csv
import json

import numpy as np

from ..config import load_settings
from ..similarity.scorer import combine, default_weights
from ..similarity.signals import SIGNALS, Corpus


def load_pairs() -> list[dict]:
    p = load_settings().path("data_dir") / "gold" / "pairs.csv"
    return list(csv.DictReader(open(p, encoding="utf-8")))


def signal_matrix(corpus: Corpus, rows: list[dict]) -> np.ndarray:
    return np.vstack([corpus.signals(corpus.pos[r["a"]], [corpus.pos[r["b"]]]) for r in rows])


def best_f1_threshold(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best = (0.0, 0.0)
    for t in np.unique(score):
        pred = score >= t
        tp = (pred & (y == 1)).sum()
        f1 = 2 * tp / max(pred.sum() + (y == 1).sum(), 1)
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best


def run(corpus: Corpus, positive: str = "close", seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    rows = load_pairs()
    X = signal_matrix(corpus, rows)
    y = np.array([r["label"] == positive for r in rows], dtype=int)
    mean = np.nanmean(X, axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)

    def prep(M):  # impute with column means; the NaN pattern stays out of the model
        return np.where(np.isnan(M), mean, M)

    def to_weights(coef):
        c = np.clip(coef, 0, None)
        return c / c.sum() if c.sum() > 0 else np.array([default_weights()[s] for s in SIGNALS])

    cv_default, cv_fit = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(prep(X[tr]), y[tr])
        w = dict(zip(SIGNALS, to_weights(lr.coef_[0])))
        cv_fit[te] = combine(X[te], w)
        cv_default[te] = combine(X[te], default_weights())
    auc_def, auc_fit = roc_auc_score(y, cv_default), roc_auc_score(y, cv_fit)
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(prep(X), y)
    fitted = dict(zip(SIGNALS, (float(x) for x in to_weights(lr.coef_[0]))))
    use_fit = auc_fit > auc_def + 0.01
    weights = fitted if use_fit else default_weights()
    score = combine(X, weights)
    tau, f1 = best_f1_threshold(score, y)
    top = np.array([r["kind"] == "top20" for r in rows])
    kept = top & (score >= tau)
    return {"n": len(y), "positives": int(y.sum()), "auc_default_cv": round(auc_def, 3), "auc_fitted_cv": round(auc_fit, 3),
            "used": "fitted" if use_fit else "defaults", "weights": weights, "fitted_weights": fitted,
            "tau_close": round(tau, 3), "f1_at_tau": round(f1, 3),
            "top20_slots": int(top.sum()), "top20_close_rate": round(float(y[top].mean()), 3),
            "top20_above_tau": int(kept.sum()), "precision_above_tau": round(float(y[kept].mean()), 3) if kept.any() else None}
