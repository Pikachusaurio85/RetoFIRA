"""Phase 3.1: exploratory screening of feature sets on selection repeats."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as cfg
from src.models.priors import load_feature_sets, load_mcmc
from src.validation import baselines
from src.validation.cv_runner import run_cv


def main():
    cfg.ensure_dirs()
    mcmc = load_mcmc()
    repeats = mcmc["selection_repeats"]
    configs = list(load_feature_sets().keys())
    out = cfg.RESULTS_DIR / "phase3"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for config in configs:
        vif = 10.0 if config == "B20" else None
        print(f"\n===== SCREEN {config} (vif={vif}) =====", flush=True)
        run_cv(config, mcmc_key="screen", repeats=repeats, vif_threshold=vif, out_dir=out,
               n_samples=1000, tag="selection", resume=True)

    print(f"\nScreening done in {time.time()-t0:.1f}s. Building summary...")

    frames = []
    for f in sorted(out.glob("cv_*_selection_folds.csv")):
        d = pd.read_csv(f)
        if len(d):
            frames.append(d)
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(out / "screen_folds_all.csv", index=False)

    summ = allf.groupby("config").agg(
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"), r2_mean=("r2", "mean"),
        crps_mean=("crps", "mean"), cov80_mean=("coverage_80", "mean"),
        cov95_mean=("coverage_95", "mean"),
        max_rhat=("max_rhat", "max"), min_ess=("min_ess", "min"),
        div_total=("divergences", "sum"), time_mean=("elapsed_s", "mean"),
    ).reset_index()
    summ.to_csv(out / "screen_summary.csv", index=False)
    print(summ.to_string(index=False))

    folds = baselines.__dict__  # noqa
    import json
    from src.data import loader
    master = loader.load_master()
    fdict = json.loads((cfg.CONFIGS_DIR / f"splits_{cfg.CV_SCHEME}.json").read_text())
    prior_oof = baselines.municipal_prior_oof(master, fdict)
    prior_oof = prior_oof[prior_oof["repeat"].isin(repeats)]
    prior_oof.to_csv(out / "municipal_prior_oof_selection.csv", index=False)
    pm = prior_oof.eval("y_true - y_pred")
    print(f"\nMunicipal prior (selection folds): RMSE={((pm**2).mean())**0.5:.4f} MAE={pm.abs().mean():.4f}")


if __name__ == "__main__":
    main()
