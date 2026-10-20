"""Data loading for the Bayesian hierarchical experiments."""
from __future__ import annotations

import pandas as pd

from src import config as cfg


def load_master() -> pd.DataFrame:
    """Load the per-parcel feature table joined with the 5 official XAI concepts.

    Columns used: ID_POLIGONO, CONJUNTO, RENDIMIENTO_T_HA, loc_municipio plus all
    engineered features and the exact concept columns from perfiles_conceptuales.csv.
    The *_scaled concept columns are dropped to avoid any cross-parcel scaling leakage.
    """
    feats = pd.read_csv(cfg.FEATURES_CSV)
    perfiles = pd.read_csv(cfg.PERFILES_CSV)
    scale_cols = [c for c in perfiles.columns if c.endswith("_scaled")]
    perfiles = perfiles.drop(columns=scale_cols)
    missing = [c for c in cfg.XAI_CONCEPTS if c not in perfiles.columns]
    if missing:
        raise ValueError(f"Missing XAI concepts in perfiles: {missing}")

    df = feats.merge(perfiles, on=cfg.ID_COL, how="left", validate="one_to_one")
    if perfiles.shape[0] != feats.shape[0]:
        raise ValueError("Concept file row count does not match feature table")
    return df


def train_frame(master: pd.DataFrame) -> pd.DataFrame:
    return master[master[cfg.PARTITION_COL] == cfg.TRAIN_PARTITION].copy()


def prediction_frame(master: pd.DataFrame) -> pd.DataFrame:
    return master[master[cfg.PARTITION_COL] == cfg.PRED_PARTITION].copy()
