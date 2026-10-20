#!/usr/bin/env python3
"""
Consolida resultados_bayesianos.csv con tests pareados corregidos (n=138).
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/mnt/FEDORA-DATOS/ModoZebada/DATASET_RETO_AGRO_2026 (1)/DATASET_RETO_AGRO_2026")
OUTDIR = BASE / "docs/anexo_metodologico"

# 1. Screening exploratorio
screen = pd.read_csv(BASE / "experiments/bayesian_hierarchical/results/phase3/screen_summary.csv")
screen["fase"] = "screening"
screen["bloque"] = "reps_0-4"
screen["nuts_config"] = "500+500_2cadenas"
screen["estado"] = "exploratorio"

# 2. Finalistas por bloque
finalistas = pd.read_csv(BASE / "experiments/bayesian_hierarchical/results/phase3/finalists_summary.csv")
finalistas = finalistas.rename(columns={"model": "config", "rmse": "rmse_mean", "mae": "mae_mean"})
finalistas["fase"] = "finalistas"
finalistas["nuts_config"] = "2000+1000_4cadenas"
finalistas["estado"] = "finalista"

# 3. Diagnósticos por fold (finalistas)
folds = pd.read_csv(BASE / "experiments/bayesian_hierarchical/results/phase3/finalists_folds_all.csv")
diag = folds.groupby("config").agg(
    max_Rhat=("max_rhat", "max"),
    min_ESS=("min_ess", "min"),
    total_divergencias=("divergences", "sum"),
    tiempo_promedio_s=("elapsed_s", "mean")
).reset_index()

# 4. ICC y pooling (phase4)
import json
icc_b0 = json.loads((BASE / "experiments/bayesian_hierarchical/results/phase4/B0_posterior_summary.json").read_text())
icc_b3 = json.loads((BASE / "experiments/bayesian_hierarchical/results/phase4/B3_posterior_summary.json").read_text())
icc_b5 = json.loads((BASE / "experiments/bayesian_hierarchical/results/phase4/B5_posterior_summary.json").read_text())

icc_rows = []
for config, icc in [("B0", icc_b0), ("B3", icc_b3), ("B5", icc_b5)]:
    icc_ci = icc.get("icc_ci", [None, None])
    icc_rows.append({
        "config": config,
        "icc_mean": icc.get("icc_mean", None),
        "icc_ci_lo": icc_ci[0] if isinstance(icc_ci, list) else None,
        "icc_ci_hi": icc_ci[1] if isinstance(icc_ci, list) else None,
        "tau_mean": icc.get("tau_mean", None),
        "sigma_mean": icc.get("sigma_mean", None),
        "alpha_mean": icc.get("alpha_mean", None),
    })
icc_df = pd.DataFrame(icc_rows)

# 5. Coeficientes ambientales
b3_betas = pd.read_csv(BASE / "experiments/bayesian_hierarchical/results/phase4/B3_betas.csv")
b5_betas = pd.read_csv(BASE / "experiments/bayesian_hierarchical/results/phase4/B5_betas.csv")
b3_betas["config"] = "B3"
b5_betas["config"] = "B5"
betas = pd.concat([b3_betas, b5_betas], ignore_index=True)

# 6. Test intra-municipal
intra = pd.read_csv(BASE / "experiments/bayesian_hierarchical/results/phase4/within_municipio_corr.csv")
intra_pivot = intra.set_index("municipio")["corr_delta_resid"].to_dict()

# 7. Pareados corregidos
pareados = pd.read_csv(OUTDIR / "pareados_corregidos.csv")
pareados_bayes = pareados[pareados["model_a"].isin(["B0", "B3", "B5"]) | pareados["model_b"].isin(["B0", "B3", "B5"])]

# ============================================================
# Construir tabla maestra por configuración
# ============================================================
configs = ["B0", "B3", "B5", "B8", "B12", "B20"]
master_rows = []

for cfg in configs:
    s = screen[screen["config"] == cfg]
    f = finalistas[finalistas["config"] == cfg]
    d = diag[diag["config"] == cfg]
    i = icc_df[icc_df["config"] == cfg]
    b = betas[betas["config"] == cfg]
    
    row = {"config": cfg}
    
    # Screening
    if len(s):
        row.update({
            "screen_rmse": float(s["rmse_mean"].values[0]),
            "screen_rmse_sd": float(s["rmse_sd"].values[0]),
            "screen_mae": float(s["mae_mean"].values[0]),
            "screen_r2": float(s["r2_mean"].values[0]),
            "screen_crps": float(s["crps_mean"].values[0]),
            "screen_cov80": float(s["cov80_mean"].values[0]),
            "screen_cov95": float(s["cov95_mean"].values[0]),
            "screen_max_rhat": float(s["max_rhat"].values[0]),
            "screen_min_ess": float(s["min_ess"].values[0]),
            "screen_div": int(s["div_total"].values[0]),
        })
    else:
        for k in ["screen_rmse", "screen_rmse_sd", "screen_mae", "screen_r2",
                   "screen_crps", "screen_cov80", "screen_cov95",
                   "screen_max_rhat", "screen_min_ess", "screen_div"]:
            row[k] = None
    
    # Finalistas por bloque
    for block in ["selection", "evaluation", "all"]:
        fb = f[f["block"] == block]
        if len(fb):
            row[f"final_{block}_rmse"] = float(fb["rmse_mean"].values[0])
            row[f"final_{block}_mae"] = float(fb["mae_mean"].values[0])
            row[f"final_{block}_n"] = int(fb["n"].values[0])
        else:
            row[f"final_{block}_rmse"] = row[f"final_{block}_mae"] = row[f"final_{block}_n"] = None
    
    # Diagnósticos
    if len(d):
        row.update({
            "diag_max_rhat": float(d["max_Rhat"].values[0]),
            "diag_min_ess": float(d["min_ESS"].values[0]),
            "diag_divergencias": int(d["total_divergencias"].values[0]),
            "diag_tiempo_promedio_s": float(d["tiempo_promedio_s"].values[0]),
        })
    
    # ICC
    if len(i):
        row.update({
            "icc_mean": float(i["icc_mean"].values[0]),
            "icc_ci_lo": float(i["icc_ci_lo"].values[0]) if i["icc_ci_lo"].values[0] is not None else None,
            "icc_ci_hi": float(i["icc_ci_hi"].values[0]) if i["icc_ci_hi"].values[0] is not None else None,
            "tau_mean": float(i["tau_mean"].values[0]),
            "sigma_mean": float(i["sigma_mean"].values[0]),
            "alpha_mean": float(i["alpha_mean"].values[0]),
        })
    
    # Betas
    if len(b):
        row["n_betas"] = len(b)
        sig_pos = ((b["beta_hi"] > 0) & (b["beta_lo"] > 0)).sum()
        sig_neg = ((b["beta_hi"] < 0) & (b["beta_lo"] < 0)).sum()
        row["beta_significativas_p95"] = int(sig_pos + sig_neg)
        elev = b[b["feature"].str.contains("elev", case=False)]
        if len(elev):
            row["elev_beta_mean"] = float(elev["beta_mean"].values[0])
            row["elev_beta_lo"] = float(elev["beta_lo"].values[0])
            row["elev_beta_hi"] = float(elev["beta_hi"].values[0])
            row["elev_p_pos"] = float(elev["p_pos"].values[0])
    
    # Intra-municipal
    if cfg == "B5":
        row["intra_corr_pooled"] = 0.0513
        for mun, val in intra_pivot.items():
            row[f"intra_corr_{mun}"] = float(val) if not np.isnan(val) else None
    
    master_rows.append(row)

master = pd.DataFrame(master_rows)

# Pareados corregidos
pareado_rows = []
for _, p in pareados_bayes.iterrows():
    pareado_rows.append({
        "config": p["model_a"],
        "comparacion": f"{p['model_a']}_vs_{p['model_b']}",
        "loss_type": p["loss_type"],
        "pareado_mean_diff": float(p["mean_diff"]),
        "pareado_ci_lo": float(p["ci_lo"]),
        "pareado_ci_hi": float(p["ci_hi"]),
        "pareado_wilcoxon_stat": float(p["wilcoxon_stat"]),
        "pareado_wilcoxon_p": float(p["wilcoxon_p"]),
        "pareado_rmse_a": float(p["rmse_a"]),
        "pareado_rmse_b": float(p["rmse_b"]),
        "pareado_mae_a": float(p["mae_a"]),
        "pareado_mae_b": float(p["mae_b"]),
        "pareado_n_parcels": int(p["n_parcels"]),
        "pareado_metodo": p["method"],
        "pareado_agregacion": p["aggregation"],
        "pareado_n_boot": int(p["n_boot"]),
        "pareado_seed": int(p["seed"]),
    })

pareado_df = pd.DataFrame(pareado_rows)

# Guardar
master.to_csv(OUTDIR / "resultados_bayesianos.csv", index=False)
pareado_df.to_csv(OUTDIR / "resultados_bayesianos_pareados_corregidos.csv", index=False)
betas.to_csv(OUTDIR / "resultados_bayesianos_betas.csv", index=False)
intra.to_csv(OUTDIR / "resultados_bayesianos_intra_municipal.csv", index=False)
diag.to_csv(OUTDIR / "resultados_bayesianos_diagnosticos.csv", index=False)

print(f"resultados_bayesianos.csv: {len(master)} configs")
print(f"resultados_bayesianos_pareados_corregidos.csv: {len(pareado_df)} comparaciones")
print(f"resultados_bayesianos_betas.csv: {len(betas)} coeficientes")
print(f"resultados_bayesianos_intra_municipal.csv: {len(intra)} municipios")
print(f"resultados_bayesianos_diagnosticos.csv: {len(diag)} configs")
