"""Point, probabilistic and calibration metrics from posterior predictive draws."""
from __future__ import annotations

import numpy as np


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def r2(y, yhat):
    y = np.asarray(y)
    return float(1 - np.sum((y - np.asarray(yhat)) ** 2) / np.sum((y - y.mean()) ** 2))


def crps_from_samples(samples: np.ndarray, y: np.ndarray) -> float:
    """Sample-based CRPS. samples: (n_samples, n_points)."""
    s = np.asarray(samples, dtype=float)
    n = s.shape[0]
    t1 = np.mean(np.abs(s - y[None, :]), axis=0)
    t2 = 0.0
    for i in range(n):
        t2 += np.mean(np.abs(s[i][None, :] - s), axis=0)
    t2 = t2 / (n * n)
    return float(np.mean(t1 - t2))


def crps_from_samples_fast(samples: np.ndarray, y: np.ndarray) -> float:
    """CRPS via the sorted-sample identity. samples: (n_samples, n_points), ascending sorted internally."""
    s = np.sort(np.asarray(samples, dtype=float), axis=0)
    n = s.shape[0]
    idx = np.arange(1, n + 1)[:, None]
    term = (2 * idx - n - 1) * s
    crps = np.mean(np.abs(s - y[None, :]), axis=0) - (1.0 / (n * n)) * term.sum(axis=0)
    return float(np.mean(crps))


def log_score_from_samples(samples: np.ndarray, y: np.ndarray) -> float:
    """Mean log posterior predictive density via Gaussian KDE per point."""
    from scipy.stats import gaussian_kde

    s = np.asarray(samples, dtype=float)
    out = []
    for j in range(s.shape[1]):
        try:
            kde = gaussian_kde(s[:, j])
            out.append(float(np.log(max(kde(y[j])[0], 1e-12))))
        except Exception:
            out.append(np.nan)
    return float(np.nanmean(out))


def interval_metrics(samples: np.ndarray, y: np.ndarray, level: float) -> dict:
    lo = np.quantile(samples, (1 - level) / 2, axis=0)
    hi = np.quantile(samples, 1 - (1 - level) / 2, axis=0)
    within = (y >= lo) & (y <= hi)
    return {
        f"coverage_{int(level*100)}": float(np.mean(within)),
        f"width_{int(level*100)}": float(np.mean(hi - lo)),
    }


def point_metrics(y, yhat):
    return {"rmse": rmse(y, yhat), "mae": mae(y, yhat), "r2": r2(y, yhat)}


def full_metrics(samples: np.ndarray, y: np.ndarray, median_pred: np.ndarray | None = None) -> dict:
    y = np.asarray(y, dtype=float)
    med = median_pred if median_pred is not None else np.median(samples, axis=0)
    m = point_metrics(y, med)
    m["crps"] = crps_from_samples_fast(samples, y)
    m["log_score"] = log_score_from_samples(samples, y)
    for lvl in (0.5, 0.8, 0.95):
        m.update(interval_metrics(samples, y, lvl))
    return m
