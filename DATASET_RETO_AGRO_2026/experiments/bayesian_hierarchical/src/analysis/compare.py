"""Paired comparison between model OOF predictions on identical folds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as cfg


def _unit_frame(oof: pd.DataFrame, pred_col="y_pred") -> pd.DataFrame:
    agg = oof.groupby(["repeat", cfg.ID_COL]).agg(
        y_true=("y_true", "first"), y_pred=(pred_col, "mean"), municipio=("municipio", "first")
    ).reset_index()
    agg["err2"] = (agg["y_true"] - agg["y_pred"]) ** 2
    agg["abserr"] = (agg["y_true"] - agg["y_pred"]).abs()
    return agg


def paired_against(oof_a: pd.DataFrame, oof_b: pd.DataFrame, a_name="A", b_name="B") -> dict:
    ua = _unit_frame(oof_a).rename(columns={"err2": "err2_a", "abserr": "abserr_a"})
    ub = _unit_frame(oof_b).rename(columns={"err2": "err2_b", "abserr": "abserr_b"})
    m = ua.merge(ub[["repeat", cfg.ID_COL, "err2_b", "abserr_b"]], on=["repeat", cfg.ID_COL], how="inner")
    d_sq = m["err2_a"] - m["err2_b"]
    d_abs = m["abserr_a"] - m["abserr_b"]
    return {
        "a": a_name, "b": b_name, "n_pairs": int(len(m)),
        "rmse_a": float(np.sqrt(m["err2_a"].mean())),
        "rmse_b": float(np.sqrt(m["err2_b"].mean())),
        "delta_rmse": float(np.sqrt(m["err2_a"].mean()) - np.sqrt(m["err2_b"].mean())),
        "mean_err2_diff": float(m["err2_a"].mean() - m["err2_b"].mean()),
        "mean_abserr_diff": float(m["abserr_a"].mean() - m["abserr_b"].mean()),
        "median_abserr_diff": float((ua["abserr_a"].values - ub["abserr_b"].values).mean()) if False else float(np.median(m["abserr_a"] - m["abserr_b"])),
        "bootstrap_ci_err2_diff": _bootstrap_ci((m["err2_a"] - m["err2_b"]).to_numpy()),
    }


def _bootstrap_ci(x: np.ndarray, n=5000, seed=cfg.SEED) -> list:
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def wilcoxon_test(oof_a: pd.DataFrame, oof_b: pd.DataFrame) -> dict:
    from scipy.stats import wilcoxon

    ua = _unit_frame(oof_a).rename(columns={"err2": "err2_a"})
    ub = _unit_frame(oof_b).rename(columns={"err2": "err2_b"})
    m = ua.merge(ub[["repeat", cfg.ID_COL, "err2_b"]], on=["repeat", cfg.ID_COL])
    d = (m["err2_a"] - m["err2_b"]).to_numpy()
    if np.allclose(d, 0):
        return {"stat": 0.0, "pvalue": 1.0}
    stat, p = wilcoxon(d)
    return {"stat": float(stat), "pvalue": float(p)}
