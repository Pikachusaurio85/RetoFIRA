"""Phase 5: final model fit, artifact export and decision summary."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as cfg
from src.data import loader
from src.models.hierarchical import fit_model
from src.models.priors import load_feature_sets, load_mcmc, load_priors
from src.models.preprocess import prepare_fold

OUT = cfg.RESULTS_DIR / "phase5"


def main():
    cfg.ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    master = loader.load_master()
    fsets = load_feature_sets()
    mcmc = load_mcmc()["final_fit"]
    priors = load_priors()

    ids = master.loc[master[cfg.PARTITION_COL] == cfg.TRAIN_PARTITION, cfg.ID_COL].to_numpy()
    artifacts = {}

    for config in ["B0", "B5"]:
        feats = fsets[config]
        Xtr, _, ytr, gtr, _, kept, scaler = prepare_fold(
            master, feats, ids, ids, cfg.TARGET, cfg.ID_COL, cfg.MUNICIPIO_COL)
        idata, idx = fit_model(Xtr, ytr, gtr, config, draws=mcmc["draws"], tune=mcmc["tune"],
                               chains=mcmc["chains"], cores=mcmc["cores"], seed=cfg.SEED)
        p = idata.posterior
        alpha = p["alpha"].values.reshape(-1)
        sigma = p["sigma"].values.reshape(-1)
        tau = p["tau"].values.reshape(-1)
        u = p["u"].values.reshape(-1, p["u"].shape[-1])
        labels = sorted(set(gtr))
        mun = {g: {"u_mean": float(u[:, i].mean()),
                   "u_lo": float(np.quantile(u[:, i], .025)),
                   "u_hi": float(np.quantile(u[:, i], .975)),
                   "n_train": int((gtr == g).sum())} for i, g in enumerate(labels)}
        art = {
            "config": config,
            "formula": "y ~ Normal(alpha + u_municipio + X @ beta, sigma); u = tau*z, z~N(0,1)",
            "features": kept,
            "scaling": "z-score ajustado sobre las 138 parcelas de ENTRENAMIENTO",
            "priors": {"alpha": [priors["alpha_mu"], priors["alpha_sigma"]],
                       "beta_sd": priors["beta_prior_sd"][config],
                       "sigma_scale": priors["sigma_scale"], "tau_scale": priors["tau_scale"]},
            "posterior": {
                "alpha_mean": float(alpha.mean()),
                "sigma_mean": float(sigma.mean()),
                "tau_mean": float(tau.mean()),
                "icc_mean": float((tau**2 / (tau**2 + sigma**2)).mean()),
                "icc_ci": [float(np.quantile(tau**2 / (tau**2 + sigma**2), .025)),
                           float(np.quantile(tau**2 / (tau**2 + sigma**2), .975))],
            },
            "municipio_effects": {("Nanacamilpa de Mariano Arista" if k.startswith("Nanacamilpa") else k): v
                                  for k, v in mun.items()},
            "scaler": scaler,
            "kept_features": kept,
            "n_train": int(len(ytr)),
        }
        if "beta" in p:
            b = p["beta"].values.reshape(-1, p["beta"].shape[-1])
            artifacts_beta = [{"feature": kept[i], "beta_mean": float(b[:, i].mean()),
                               "beta_lo": float(np.quantile(b[:, i], .025)),
                               "beta_hi": float(np.quantile(b[:, i], .975)),
                               "p_pos": float((b[:, i] > 0).mean())} for i in range(b.shape[1])]
            art["beta"] = artifacts_beta
        artifacts[config] = art
        (OUT / f"final_model_{config}.json").write_text(json.dumps(art, indent=2, default=float))
        try:
            idata.to_netcdf(OUT / f"final_model_{config}_posterior.nc")
        except Exception as e:  # netcdf backend may be unavailable
            print(f"[warn] could not save netcdf for {config}: {e}")
        print(f"[ok] {config} final model saved")

    (OUT / "final_model.json").write_text(json.dumps(artifacts, indent=2, default=float))
    print("Phase 5 artifacts written to", OUT)


if __name__ == "__main__":
    main()
