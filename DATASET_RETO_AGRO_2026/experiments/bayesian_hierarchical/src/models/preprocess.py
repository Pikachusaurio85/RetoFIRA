"""Fold-safe preprocessing: imputation, scaling and VIF pruning fitted on TRAIN only."""
from __future__ import annotations

import numpy as np


def _vif(X: np.ndarray) -> np.ndarray:
    """VIF_j = [R^-1]_jj for the correlation matrix R (exact, no external deps)."""
    Xc = X - X.mean(axis=0)
    sd = Xc.std(axis=0)
    sd[sd == 0] = 1.0
    Z = Xc / sd
    R = np.corrcoef(Z, rowvar=False)
    R = np.atleast_2d(R)
    Rinv = np.linalg.pinv(R)
    return np.clip(np.diag(Rinv), 1.0, np.inf)


def select_vif(columns: list[str], X_train: np.ndarray, threshold: float = 10.0) -> list[int]:
    """Greedy VIF pruning on TRAIN only. Returns kept column indices."""
    keep = list(range(X_train.shape[1]))
    while len(keep) > 1:
        vifs = _vif(X_train[:, keep])
        worst = float(np.nanmax(vifs))
        if not np.isfinite(worst) or worst <= threshold:
            break
        keep.pop(int(np.nanargmax(vifs)))
    return keep


def prepare_fold(
    df,
    feature_cols: list[str],
    train_ids: list[str],
    test_ids: list[str],
    target_col: str,
    id_col: str,
    municipio_col: str,
    vif_threshold: float | None = None,
):
    """Return X_train, X_test, y_train, groups_train, groups_test, kept_names, scaler."""
    idx = df.set_index(id_col)
    tr = idx.loc[train_ids]
    te = idx.loc[test_ids]
    Xtr = tr[feature_cols].to_numpy(dtype=float)
    Xte = te[feature_cols].to_numpy(dtype=float)
    ytr = tr[target_col].to_numpy(dtype=float)

    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)

    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    kept = list(range(len(feature_cols)))
    if vif_threshold is not None and len(feature_cols) > 1:
        kept = select_vif(feature_cols, Xtr, vif_threshold)
        Xtr = Xtr[:, kept]
        Xte = Xte[:, kept]

    kept_names = [feature_cols[i] for i in kept]
    gtr = tr[municipio_col].to_numpy()
    gte = te[municipio_col].to_numpy()
    return (
        Xtr,
        Xte,
        ytr,
        gtr,
        gte,
        kept_names,
        {"mean": mu.tolist(), "sd": sd.tolist(), "median": med.tolist()},
    )
