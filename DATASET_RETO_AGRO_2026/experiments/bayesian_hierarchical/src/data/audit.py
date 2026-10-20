"""Data quality audit: missingness, constants, redundancy, target distribution."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config as cfg
from src.data.municipality import municipality_table, variance_decomposition


def column_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in feature_cols:
        s = df[c]
        nun = s.nunique(dropna=True)
        rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "n_missing": int(s.isna().sum()),
                "pct_missing": round(100 * s.isna().mean(), 3),
                "n_unique": int(nun),
                "is_constant": bool(nun <= 1),
                "min": float(np.nanmin(s)) if pd.api.types.is_numeric_dtype(s) and s.notna().any() else np.nan,
                "max": float(np.nanmax(s)) if pd.api.types.is_numeric_dtype(s) and s.notna().any() else np.nan,
                "mean": float(s.mean()) if pd.api.types.is_numeric_dtype(s) else np.nan,
                "std": float(s.std()) if pd.api.types.is_numeric_dtype(s) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)


def high_correlations(df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.95) -> pd.DataFrame:
    num = df[feature_cols].select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return pd.DataFrame(columns=["a", "b", "r"])
    corr = num.corr().abs()
    up = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = up.stack().reset_index()
    pairs.columns = ["a", "b", "r"]
    return pairs[pairs["r"] >= threshold].sort_values("r", ascending=False).reset_index(drop=True)


def run_audit(df: pd.DataFrame) -> dict:
    train = df[df[cfg.PARTITION_COL] == cfg.TRAIN_PARTITION]
    y = train[cfg.TARGET]
    highcorr = high_correlations(df, [c for c in df.columns if c not in (cfg.ID_COL, cfg.PARTITION_COL, cfg.MUNICIPIO_COL, cfg.TARGET)])
    yield_summary = {
        "n": int(y.count()),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "q25": float(y.quantile(0.25)),
        "median": float(y.median()),
        "q75": float(y.quantile(0.75)),
        "max": float(y.max()),
        "frac_multiple_0p5": float(np.isclose((y * 2) % 2, 0).mean()),
        "n_outliers_iqr": int(((y < y.quantile(0.25) - 1.5 * (y.quantile(0.75) - y.quantile(0.25))) | (y > y.quantile(0.75) + 1.5 * (y.quantile(0.75) - y.quantile(0.25)))).sum()),
    }
    numeric = df.select_dtypes(include=[np.number])
    return {
        "n_rows": int(len(df)),
        "n_features_available": int(numeric.shape[1]),
        "n_missing_rows_target": int(df[cfg.TARGET].isna().sum()),
        "n_constant_features": int((column_summary(df, [c for c in df.columns if c not in (cfg.ID_COL, cfg.PARTITION_COL, cfg.MUNICIPIO_COL, cfg.TARGET)])["is_constant"]).sum()),
        "yield_summary": yield_summary,
        "variance_decomposition": variance_decomposition(train),
        "municipality_counts": municipality_table(df).to_dict(orient="records"),
        "n_municipios_train_lt10": int((municipality_table(df)[cfg.TRAIN_PARTITION] < 10).sum()),
        "n_high_corr_pairs_095": int(len(high_correlations(df, [c for c in df.columns if c not in (cfg.ID_COL, cfg.PARTITION_COL, cfg.MUNICIPIO_COL, cfg.TARGET)]))),
    }


def run_and_save(df: pd.DataFrame) -> dict:
    cfg.ensure_dirs()
    out = cfg.RESULTS_DIR / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    feature_cols = [c for c in df.columns if c not in (cfg.ID_COL, cfg.PARTITION_COL, cfg.MUNICIPIO_COL, cfg.TARGET)]
    column_summary(df, feature_cols).to_csv(out / "column_summary.csv", index=False)
    high_correlations(df, feature_cols).to_csv(out / "high_correlations.csv", index=False)
    municipality_table(df).to_csv(out / "municipality_counts.csv", index=False)
    train = df[df[cfg.PARTITION_COL] == cfg.TRAIN_PARTITION]
    summary = run_audit(df)
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary
