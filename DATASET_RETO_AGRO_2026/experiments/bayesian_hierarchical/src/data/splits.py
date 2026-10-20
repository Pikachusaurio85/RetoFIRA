"""Reconstruct and persist official cross-validation partitions (RCV5x10)."""
from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd

from src import config as cfg


def load_splits(scheme: str = cfg.CV_SCHEME) -> pd.DataFrame:
    s = pd.read_csv(cfg.SPLITS_CSV)
    s = s[s["scheme"] == scheme].copy()
    if s.empty:
        raise ValueError(f"No rows for scheme={scheme}")
    return s


def build_folds(scheme: str = cfg.CV_SCHEME) -> dict:
    """Return {fold_id: {repeat, fold, train, test}} with parcel IDs."""
    s = load_splits(scheme)
    folds: dict = {}
    for (rep, fld), g in s.groupby(["repeat", "fold"]):
        train = sorted(g.loc[g["role"] == "train", cfg.ID_COL].tolist())
        test = sorted(g.loc[g["role"] == "test", cfg.ID_COL].tolist())
        folds[f"{scheme}_rep{rep}_fold{fld}"] = {
            "repeat": int(rep),
            "fold": int(fld),
            "train": train,
            "test": test,
        }
    return folds


def validate_folds(master: pd.DataFrame, folds: dict, scheme: str = cfg.CV_SCHEME) -> dict:
    """Check that every train/test parcel is an ENTRENAMIENTO parcel and no overlap."""
    train_ids = set(master.loc[master[cfg.PARTITION_COL] == cfg.TRAIN_PARTITION, cfg.ID_COL])
    pred_ids = set(master.loc[master[cfg.PARTITION_COL] == cfg.PRED_PARTITION, cfg.ID_COL])
    report = {"scheme": scheme, "n_train_pool": len(train_ids), "n_pred": len(pred_ids), "errors": []}
    for fid, f in folds.items():
        tr, te = set(f["train"]), set(f["test"])
        if tr & te:
            report["errors"].append(f"{fid}: train/test overlap")
        if not tr <= train_ids:
            report["errors"].append(f"{fid}: train contains non-ENTRENAMIENTO ids")
        if not te <= train_ids:
            report["errors"].append(f"{fid}: test contains non-ENTRENAMIENTO ids")
        if (tr | te) != train_ids:
            report["errors"].append(f"{fid}: train+test != 138 pool")
        if tr & pred_ids or te & pred_ids:
            report["errors"].append(f"{fid}: PREDICCION parcel leaked")
    report["n_folds"] = len(folds)
    report["ok"] = not report["errors"]
    return report


def save_folds(folds: dict, path=None) -> None:
    import json as _json
    path = path or (cfg.CONFIGS_DIR / f"splits_{cfg.CV_SCHEME}.json")
    path.write_text(_json.dumps(folds, indent=2))
