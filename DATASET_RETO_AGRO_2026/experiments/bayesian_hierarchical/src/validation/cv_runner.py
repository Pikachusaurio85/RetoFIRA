"""Cross-validation runner for the hierarchical Bayesian model."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg
from src.data import loader, splits
from src.models import hierarchical as hmod
from src.models.preprocess import prepare_fold
from src.models.priors import beta_prior_sd, load_feature_sets, load_mcmc
from src.validation import metrics


def _diagnostics(idata) -> dict:
    import arviz as az

    rhat = az.rhat(idata)
    ess = az.ess(idata)
    rhat_max = float(np.nanmax(np.asarray(rhat.to_array().values)))
    ess_min = float(np.nanmin(np.asarray(ess.to_array().values)))
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else 0
    return {"max_rhat": rhat_max, "min_ess": ess_min, "divergences": div}


def run_cv(
    config: str,
    mcmc_key: str = "screen",
    repeats: list[int] | None = None,
    vif_threshold: float | None = None,
    out_dir: Path | None = None,
    n_samples: int = 2000,
    tag: str = "",
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg.ensure_dirs()
    mcmc = load_mcmc()[mcmc_key]
    fs = load_feature_sets()
    feature_cols = fs[config]
    master = loader.load_master()
    folds = json.loads((cfg.CONFIGS_DIR / f"splits_{cfg.CV_SCHEME}.json").read_text())
    if repeats is None:
        repeats = sorted({v["repeat"] for v in folds.values()})

    out_dir = out_dir or (cfg.RESULTS_DIR / "phase3")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cv_{config}_{mcmc_key}{('_' + tag) if tag else ''}"
    folds_csv = out_dir / f"{stem}_folds.csv"
    oof_csv = out_dir / f"{stem}_oof.csv"

    done = set()
    if resume and folds_csv.exists():
        try:
            prev = pd.read_csv(folds_csv)
            done = set(zip(prev["repeat"], prev["fold"]))
        except Exception:
            done = set()

    fold_cols = ["config", "mcmc", "repeat", "fold", "n_train", "n_test", "n_features_used",
                 "features_used", "elapsed_s", "rmse", "mae", "r2", "crps", "log_score",
                 "coverage_50", "width_50", "coverage_80", "width_80", "coverage_95", "width_95",
                 "max_rhat", "min_ess", "divergences"]
    oof_cols = ["config", "mcmc", "repeat", "fold", cfg.ID_COL, "municipio", "y_true", "y_pred", "q025", "q50", "q975"]
    if not folds_csv.exists():
        pd.DataFrame(columns=fold_cols).to_csv(folds_csv, index=False)
    if not oof_csv.exists():
        pd.DataFrame(columns=oof_cols).to_csv(oof_csv, index=False)

    for fid, f in sorted(folds.items()):
        if f["repeat"] not in repeats:
            continue
        if (f["repeat"], f["fold"]) in done:
            continue
        t0 = time.time()
        Xtr, Xte, ytr, gtr, gte, kept, scaler = prepare_fold(
            master, feature_cols, f["train"], f["test"], cfg.TARGET, cfg.ID_COL, cfg.MUNICIPIO_COL,
            vif_threshold=vif_threshold,
        )
        idata, gidx = hmod.fit_model(
            Xtr, ytr, gtr, config,
            draws=mcmc["draws"], tune=mcmc["tune"], chains=mcmc["chains"],
            target_accept=mcmc.get("target_accept", 0.9), cores=mcmc.get("cores", 2),
            seed=cfg.SEED + f["repeat"] * 100 + f["fold"],
        )
        pp = hmod.posterior_predict(idata, Xte, gte, gidx, seed=cfg.SEED, n_samples=n_samples)
        med = np.median(pp, axis=0)
        yte = master.set_index(cfg.ID_COL).loc[f["test"], cfg.TARGET].to_numpy(dtype=float)
        m = metrics.full_metrics(pp, yte, med)
        diag = _diagnostics(idata)
        elapsed = time.time() - t0

        row = {
            "config": config, "mcmc": mcmc_key, "repeat": f["repeat"], "fold": f["fold"],
            "n_train": len(f["train"]), "n_test": len(f["test"]),
            "n_features_used": len(kept), "features_used": "|".join(kept),
            "elapsed_s": round(elapsed, 2), **m, **diag,
        }
        pd.DataFrame([row]).to_csv(folds_csv, mode="a", header=False, index=False)

        q = np.quantile(pp, [0.025, 0.5, 0.975], axis=0)
        oof_rows = pd.DataFrame({
            "config": config, "mcmc": mcmc_key, "repeat": f["repeat"], "fold": f["fold"],
            cfg.ID_COL: f["test"], "municipio": gte, "y_true": yte,
            "y_pred": med, "q025": q[0], "q50": q[1], "q975": q[2],
        })
        oof_rows.to_csv(oof_csv, mode="a", header=False, index=False)
        print(f"[{config}/{mcmc_key}] rep{f['repeat']} fold{f['fold']} "
              f"rmse={m['rmse']:.4f} mae={m['mae']:.4f} rhat={diag['max_rhat']:.3f} "
              f"ess={diag['min_ess']:.0f} div={diag['divergences']} {elapsed:.1f}s", flush=True)

    folds_df = pd.read_csv(folds_csv)
    oof_df = pd.read_csv(oof_csv)
    return folds_df[~folds_df.duplicated(["repeat", "fold"])], oof_df.drop_duplicates(["config", "repeat", "fold", cfg.ID_COL])
