"""Phase 4: interpretation of the hierarchical model posteriors and diagnostics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src.data import loader
from src.models.hierarchical import fit_model
from src.models.priors import load_feature_sets, load_mcmc
from src.models.preprocess import prepare_fold

FIGDIR = cfg.REPORTS_DIR / "figures"
OUT = cfg.RESULTS_DIR / "phase4"


def full_fit(master, feature_cols, config, mcmc):
    ids = master.loc[master[cfg.PARTITION_COL] == "ENTRENAMIENTO", cfg.ID_COL].to_numpy()
    Xtr, _, ytr, gtr, _, kept, _ = prepare_fold(master, feature_cols, ids, ids, cfg.TARGET, cfg.ID_COL, cfg.MUNICIPIO_COL)
    idata, idx = fit_model(Xtr, ytr, gtr, config, draws=mcmc["draws"], tune=mcmc["tune"],
                           chains=mcmc["chains"], cores=mcmc["cores"], seed=cfg.SEED)
    return idata, idx, Xtr, ytr, gtr, kept


def summarize_posterior(idata, group_labels):
    p = idata.posterior
    def ms(v):
        a = p[v].values.reshape(-1) if p[v].ndim == 3 and p[v].shape[2] == 1 else p[v].values
        return a
    alpha = p["alpha"].values.reshape(-1)
    sigma = p["sigma"].values.reshape(-1)
    tau = p["tau"].values.reshape(-1)
    icc = tau**2 / (tau**2 + sigma**2)
    u = p["u"].values.reshape(-1, p["u"].shape[-1])
    out = {
        "alpha_mean": float(alpha.mean()), "alpha_ci": [float(np.quantile(alpha, .025)), float(np.quantile(alpha, .975))],
        "sigma_mean": float(sigma.mean()), "tau_mean": float(tau.mean()),
        "icc_mean": float(icc.mean()), "icc_ci": [float(np.quantile(icc, .025)), float(np.quantile(icc, .975))],
    }
    mun_rows = []
    for i, g in enumerate(group_labels):
        mun_rows.append({"municipio": g, "u_mean": float(u[:, i].mean()),
                         "u_lo": float(np.quantile(u[:, i], .025)), "u_hi": float(np.quantile(u[:, i], .975))})
    beta_rows = []
    if "beta" in p:
        b = p["beta"].values.reshape(-1, p["beta"].shape[-1])
        for i in range(b.shape[1]):
            beta_rows.append({"coef_idx": i, "beta_mean": float(b[:, i].mean()), "beta_sd": float(b[:, i].std()),
                              "beta_lo": float(np.quantile(b[:, i], .025)), "beta_hi": float(np.quantile(b[:, i], .975)),
                              "p_pos": float((b[:, i] > 0).mean())})
    return out, pd.DataFrame(mun_rows), pd.DataFrame(beta_rows), sigma, tau


def forest_municipios(mun_df, path, title):
    d = mun_df.sort_values("u_mean")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(d))
    ax.errorbar(d["u_mean"], y, xerr=[d["u_mean"] - d["u_lo"], d["u_hi"] - d["u_mean"]], fmt="o", color="C0", label="Efecto municipal (posterior)")
    ax.scatter(d["raw_dev"], y, marker="x", color="C1", label="Desviación media observada")
    ax.set_yticks(y); ax.set_yticklabels(d["municipio"])
    ax.axvline(0, color="grey", lw=.8)
    ax.set_xlabel("Desviación respecto a la media global (t/ha)")
    ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def forest_beta(beta_df, kept, path, title):
    d = beta_df.copy(); d["feature"] = kept
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(d))
    ax.errorbar(d["beta_mean"], y, xerr=[d["beta_mean"] - d["beta_lo"], d["beta_hi"] - d["beta_mean"]], fmt="o", color="C2")
    ax.set_yticks(y); ax.set_yticklabels(d["feature"])
    ax.axvline(0, color="grey", lw=.8)
    ax.set_xlabel("Coeficiente estandarizado (t/ha por DE)")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def within_municipio_test(oof_b0, oof_b5, y_true_by_mun):
    a = oof_b0.rename(columns={"y_pred": "pred_b0"})[["repeat", cfg.ID_COL, "municipio", "y_true", "pred_b0"]]
    b = oof_b5.rename(columns={"y_pred": "pred_b5"})[["repeat", cfg.ID_COL, "pred_b5"]]
    m = a.merge(b, on=["repeat", cfg.ID_COL])
    m["delta"] = m["pred_b5"] - m["pred_b0"]
    m["resid"] = m["y_true"] - m["municipio"].map(y_true_by_mun)
    rows = []
    for mun, g in m.groupby("municipio"):
        if g["delta"].std() < 1e-12 or g["resid"].std() < 1e-12:
            rows.append({"municipio": mun, "n": len(g), "corr_delta_resid": np.nan})
            continue
        rows.append({"municipio": mun, "n": len(g), "corr_delta_resid": float(np.corrcoef(g["delta"], g["resid"])[0, 1])})
    per_mun = pd.DataFrame(rows)
    d = m.copy()
    d["delta_c"] = d["delta"] - d.groupby("municipio")["delta"].transform("mean")
    d["resid_c"] = d["resid"] - d.groupby("municipio")["resid"].transform("mean")
    pooled = float(np.corrcoef(d["delta_c"], d["resid_c"])[0, 1]) if d["delta_c"].std() > 0 else np.nan
    return per_mun, pooled, d


def scatter_delta(d, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for mun, g in d.groupby("municipio"):
        ax.scatter(g["delta_c"], g["resid_c"], s=12, alpha=.6, label=mun)
    ax.axhline(0, color="grey", lw=.8); ax.axvline(0, color="grey", lw=.8)
    ax.set_xlabel("Ajuste ambiental de B5 vs B0 (t/ha, centrado)")
    ax.set_ylabel("Residual de rendimiento intrapredio (t/ha)")
    ax.set_title("¿El ajuste ambiental acierta la dirección intrapredio?")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    cfg.ensure_dirs(); FIGDIR.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    master = loader.load_master()
    fsets = load_feature_sets()["sets"] if "sets" in load_feature_sets() else load_feature_sets()
    mcmc = load_mcmc()["final_fit"]
    y_true_by_mun = master[master[cfg.PARTITION_COL] == "ENTRENAMIENTO"].groupby(cfg.MUNICIPIO_COL)[cfg.TARGET].mean().to_dict()

    summary = {}
    for config, feats in [("B0", fsets["B0"]), ("B3", fsets["B3"]), ("B5", fsets["B5"])]:
        print(f"\n=== Full-data fit {config} ===", flush=True)
        idata, idx, Xtr, ytr, gtr, kept = full_fit(master, feats, config, mcmc)
        labels = sorted(set(gtr))
        s, mun_df, beta_df, sigma, tau = summarize_posterior(idata, labels)
        s["config"] = config; s["n_obs"] = len(ytr); s["n_feat"] = len(kept)
        summary[config] = s
        raw = master[master[cfg.PARTITION_COL] == "ENTRENAMIENTO"].groupby(cfg.MUNICIPIO_COL)[cfg.TARGET].agg(["mean", "count"]).reset_index()
        raw = raw.rename(columns={"mean": "raw_mean", cfg.MUNICIPIO_COL: "municipio"})
        raw["raw_dev"] = raw["raw_mean"] - ytr.mean()
        mun_df = mun_df.merge(raw, on="municipio", how="left")
        mun_df["shrinkage_lambda"] = (tau.mean()**2) / (tau.mean()**2 + sigma.mean()**2 / mun_df["count"])
        mun_df.to_csv(OUT / f"{config}_municipio_effects.csv", index=False)
        if len(beta_df):
            beta_df["feature"] = kept
            beta_df.to_csv(OUT / f"{config}_betas.csv", index=False)
            forest_beta(beta_df, kept, FIGDIR / f"{config}_betas.png", f"Coeficientes ambientales — {config}")
        forest_municipios(mun_df, FIGDIR / f"{config}_municipios.png", f"Efectos municipales — {config}")
        print(f"  ICC={s['icc_mean']:.3f} [{s['icc_ci'][0]:.3f},{s['icc_ci'][1]:.3f}]  tau={s['tau_mean']:.3f} sigma={s['sigma_mean']:.3f}")
        pd.Series(s).to_json(OUT / f"{config}_posterior_summary.json", indent=2)
        print(mun_df[["municipio", "count", "raw_dev", "u_mean", "shrinkage_lambda"]].to_string(index=False))

    # within-municipio discrimination from OOF
    oof_b0 = pd.read_csv(cfg.RESULTS_DIR / "phase3" / "cv_B0_finalists_finalists_oof.csv")
    oof_b5 = pd.read_csv(cfg.RESULTS_DIR / "phase3" / "cv_B5_finalists_finalists_oof.csv")
    per_mun, pooled, d = within_municipio_test(oof_b0, oof_b5, y_true_by_mun)
    per_mun.to_csv(OUT / "within_municipio_corr.csv", index=False)
    scatter_delta(d, FIGDIR / "within_municipio_delta.png")
    print(f"\nWithin-municipio pooled corr(delta_B5, resid) = {pooled:.4f}")
    print(per_mun.to_string(index=False))
    (OUT / "phase4_summary.json").write_text(json.dumps({"posterior": summary, "within_muni_pooled_corr": pooled}, indent=2, default=float))
    print("\nPhase 4 done.")


if __name__ == "__main__":
    main()
