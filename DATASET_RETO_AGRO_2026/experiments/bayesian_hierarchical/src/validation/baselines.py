"""Municipal-prior baseline (official predictor) recomputed and loaded per fold."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as cfg

K_GRID = [0.05 * (10 ** (2 * i / 24)) for i in range(25)]
K_OFFICIAL = 0.07775395769865924


def municipal_prior_predict(train: pd.DataFrame, test: pd.DataFrame, k: float, target=cfg.TARGET, mun=cfg.MUNICIPIO_COL):
    S = train.groupby(mun)[target].sum()
    n = train.groupby(mun)[target].count()
    G = float(train[target].mean())
    preds = []
    for m in test[mun]:
        if m in S.index and n.get(m, 0) > 0:
            preds.append((S[m] + k * G) / (n[m] + k))
        else:
            preds.append(G)
    return np.array(preds)


def tune_k_loo(train: pd.DataFrame, target=cfg.TARGET, mun=cfg.MUNICIPIO_COL, grid=None):
    grid = grid or K_GRID
    S = train.groupby(mun)[target].sum()
    n = train.groupby(mun)[target].count()
    G = float(train[target].mean())
    y = train[target].to_numpy()
    m_arr = train[mun].to_numpy()
    best_k, best_sse = grid[0], np.inf
    for k in grid:
        pred = np.empty_like(y)
        for i, (m, yi) in enumerate(zip(m_arr, y)):
            s, cnt = S[m] - yi, n[m] - 1
            gi = (G * (n.sum()) - yi) / (n.sum() - 1) if n.sum() > 1 else G
            pred[i] = (s + k * gi) / (cnt + k) if cnt > 0 else gi
        sse = float(np.sum((y - pred) ** 2))
        if sse < best_sse:
            best_sse, best_k = sse, k
    return best_k


def municipal_prior_oof(master: pd.DataFrame, folds: dict, target=cfg.TARGET, mun=cfg.MUNICIPIO_COL, id_col=cfg.ID_COL):
    rows = []
    idx = master.set_index(id_col)
    for fid, f in folds.items():
        tr = idx.loc[f["train"]].reset_index()
        te = idx.loc[f["test"]].reset_index()
        k = tune_k_loo(tr)
        pred = municipal_prior_predict(tr, te, k)
        for pid, m, yt, yp in zip(te[id_col], te[mun], te[target], pred):
            rows.append({"repeat": f["repeat"], "fold": f["fold"], id_col: pid, "municipio": m,
                         "y_true": yt, "y_pred": yp, "k": k})
    return pd.DataFrame(rows)


def load_official_prior_oof(scheme: str = cfg.CV_SCHEME) -> pd.DataFrame:
    oof = pd.read_csv(cfg.PREDICTOR_DIR / "predicciones_oof.csv")
    oof = oof[(oof["model_id"] == "PRIOR") & (oof["scheme"] == scheme)].copy()
    return oof.rename(columns={"y_pred_t_ha": "y_pred", "y_true_t_ha": "y_true"})[
        ["repeat", "fold", cfg.ID_COL, "municipio", "y_true", "y_pred"]
    ]
