from __future__ import annotations
import numpy as np
import pandas as pd

def _safe_quantile_bins(x: np.ndarray, bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        # если совсем одинаковые значения, делаем линейные бины
        mn, mx = float(np.min(x)), float(np.max(x))
        if mn == mx:
            mn -= 1e-6
            mx += 1e-6
        edges = np.linspace(mn, mx, bins + 1)
    return edges

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = _safe_quantile_bins(expected, bins=bins)

    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)

    e_perc = e_counts / max(e_counts.sum(), 1)
    a_perc = a_counts / max(a_counts.sum(), 1)

    e_perc = np.clip(e_perc, eps, 1.0)
    a_perc = np.clip(a_perc, eps, 1.0)

    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))

def drift_report(df_ref: pd.DataFrame, df_cur: pd.DataFrame, features: list[str], bins: int = 10) -> dict:
    per_feature = {}
    for f in features:
        per_feature[f] = psi(df_ref[f].values, df_cur[f].values, bins=bins)

    max_psi = max(per_feature.values()) if per_feature else 0.0
    mean_psi = float(np.mean(list(per_feature.values()))) if per_feature else 0.0
    return {
        "psi_per_feature": per_feature,
        "psi_max": float(max_psi),
        "psi_mean": float(mean_psi),
    }
