"""Municipality-level structure of the training data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as cfg


def municipality_table(df: pd.DataFrame) -> pd.DataFrame:
    t = (
        df.groupby([cfg.MUNICIPIO_COL, cfg.PARTITION_COL])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[cfg.TRAIN_PARTITION, cfg.PRED_PARTITION], fill_value=0)
    )
    t["total"] = t.sum(axis=1)
    t = t.sort_values("total", ascending=False)
    t["small_train"] = t[cfg.TRAIN_PARTITION] < 10
    return t.reset_index()


def variance_decomposition(train: pd.DataFrame) -> dict:
    """Between vs within municipality variance of the target."""
    y = train[cfg.TARGET]
    grp = train[cfg.MUNICIPIO_COL]
    grand = y.mean()
    groups = y.groupby(grp)
    ss_between = float((groups.count() * (groups.mean() - grand) ** 2).sum())
    ss_within = float(((y - groups.transform("mean")) ** 2).sum())
    ss_total = ss_between + ss_within
    return {
        "grand_mean": float(grand),
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,
        "frac_between": ss_between / ss_total if ss_total else np.nan,
        "frac_within": ss_within / ss_total if ss_total else np.nan,
        "icc_approx": ss_between / ss_total if ss_total else np.nan,
    }


def per_municipality_yield(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby(cfg.MUNICIPIO_COL)[cfg.TARGET]
    out = pd.DataFrame(
        {
            "n": g.count(),
            "mean": g.mean(),
            "std": g.std(),
            "min": g.min(),
            "max": g.max(),
        }
    ).sort_values("n", ascending=False)
    out["sem"] = out["std"] / np.sqrt(out["n"])
    return out.reset_index()
