"""Combined score S(A,B) = sum(w_i s_i) / sum(w_i) over signals available for the pair (plan §8.2)."""
from __future__ import annotations

import numpy as np

from ..config import load_settings
from .signals import SIGNALS


def default_weights() -> dict[str, float]:
    return dict(load_settings().similarity["weights"])


def combine(S: np.ndarray, weights: dict[str, float] | None = None) -> np.ndarray:
    """S: (n, 14) with NaN for unavailable. Weights renormalised per row over available signals."""
    w = np.array([(weights or default_weights()).get(s, 0.0) for s in SIGNALS])
    avail = ~np.isnan(S)
    num = np.where(avail, np.nan_to_num(S) * w, 0).sum(axis=1)
    den = np.where(avail, w, 0).sum(axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)


def benchmark_eligible(shared: dict[str, list[str]]) -> bool:
    """Shares >=1 dataset, or >=1 baseline, or >=2 metrics (plan §8.2)."""
    return bool(shared.get("datasets") or shared.get("baselines") or len(shared.get("metrics", [])) >= 2)


def why(S_row: np.ndarray, weights: dict[str, float] | None = None, top: int = 3) -> list[tuple[str, float]]:
    """Top contributing signals (weight x value), for the 'why matched' line."""
    w = weights or default_weights()
    contrib = [(s, w.get(s, 0.0) * v) for s, v in zip(SIGNALS, S_row) if v == v]
    return [(s, round(float(S_row[SIGNALS.index(s)]), 2)) for s, _ in sorted(contrib, key=lambda x: -x[1])[:top]]
