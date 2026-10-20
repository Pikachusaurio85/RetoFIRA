#!/usr/bin/env python3
"""
Construye modelos_comparativa.csv con todos los modelos ejecutados.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/mnt/FEDORA-DATOS/ModoZebada/DATASET_RETO_AGRO_2026 (1)/DATASET_RETO_AGRO_2026")
OUTDIR = BASE / "docs/anexo_metodologico"

rows = []

# ============================================================
# 1. OFICIALES - RCV5x10 (comparables entre sí y con bayesianos)
# ============================================================
print("Cargando oficiales RCV5x10...")
metricas_rep = pd.read_csv(BASE / "outputs/predictor/metricas_por_repeticion.csv")
rcv = metricas_rep[metricas_rep["scheme"] == "RCV5x10"]

for modelo in rcv["modelo"].unique():
    sub = rcv[rcv["modelo"] == modelo]
    for _, r in sub.iterrows():
        rows.append({
            "modelo": r["modelo"],
            "familia": "oficial",
            "conjunto_variables": "prior_municipal" if "PRIOR" in r["modelo"] else 
                                 ("media_global" if r["modelo"] == "MEDIA_GLOBAL" else 
                                  ("ridge_148" if r["modelo"] == "RIDGE148" else "hibrido_topo")),
            "n_variables": 0 if "PRIOR" in r["modelo"] or r["modelo"] == "MEDIA_GLOBAL" else 
                           (148 if r["modelo"] == "RIDGE148" else None),
            "tipo_entrenamiento": "prior_municipal_encogido" if "PRIOR" in r["modelo"] else
                                  ("media_global" if r["modelo"] == "MEDIA_GLOBAL" else
                                   ("ridge_regression" if r["modelo"] == "RIDGE148" else "prior_plus_ridge_topo")),
            "validacion": "RCV5x10",
            "n_folds": 5,
            "n_reps": 10,
            "semilla": 20260219,
            "repeticion": int(r["repeat"]),
            "RMSE": r["RMSE"],
            "MAE": r["MAE"],
            "R2": r["R2"],
            "archivo_fuente_metricas": "outputs/predictor/metricas_por_repeticion.csv",
            "archivo_fuente_predicciones": "outputs/predictor/predicciones_oof.csv",
            "tiempo_s": None,
            "estado": "oficial",
            "limitaciones": ""
        })
    # Fila agregada (media ± sd)
    rmse_mean, rmse_std = sub["RMSE"].mean(), sub["RMSE"].std()
    mae_mean, mae_std = sub["MAE"].mean(), sub["MAE"].std()
    r2_mean, r2_std = sub["R2"].mean(), sub["R2"].std()
    rows.append({
        "modelo": f"{modelo}_AGREGADO",
        "familia": "oficial",
        "conjunto_variables": "prior_municipal" if "PRIOR" in modelo else 
                             ("media_global" if modelo == "MEDIA_GLOBAL" else 
                              ("ridge_148" if modelo == "RIDGE148" else "hibrido_topo")),
        "n_variables": 0 if "PRIOR" in modelo or modelo == "MEDIA_GLOBAL" else 
                       (148 if modelo == "RIDGE148" else None),
        "tipo_entrenamiento": "prior_municipal_encogido" if "PRIOR" in modelo else
                              ("media_global" if modelo == "MEDIA_GLOBAL" else
                               ("ridge_regression" if modelo == "RIDGE148" else "prior_plus_ridge_topo")),
        "validacion": "RCV5x10",
        "n_folds": 5,
        "n_reps": 10,
        "semilla": 20260219,
        "repeticion": "AGREGADO_10_reps",
        "RMSE": rmse_mean, "MAE": mae_mean, "R2": r2_mean,
        "RMSE_std": rmse_std, "MAE_std": mae_std, "R2_std": r2_std,
        "archivo_fuente_metricas": "outputs/predictor/metricas_por_repeticion.csv",
        "archivo_fuente_predicciones": "outputs/predictor/predicciones_oof.csv",
        "tiempo_s": None,
        "estado": "oficial",
        "limitaciones": "Agregado sobre 10 repeticiones RCV5x10"
    })

# ============================================================
# 2. OFICIALES - OTROS ESQUEMAS (KM6, LMO, RCV5_S2026)
# ============================================================
for scheme in ["KM6", "LMO", "RCV5_S2026"]:
    sub = metricas_rep[metricas_rep["scheme"] == scheme]
    if len(sub) == 0:
        continue
    for modelo in sub["modelo"].unique():
        msub = sub[sub["modelo"] == modelo]
        for _, r in msub.iterrows():
            rows.append({
                "modelo": r["modelo"],
                "familia": "oficial",
                "conjunto_variables": "prior_municipal" if "PRIOR" in r["modelo"] else 
                                     ("media_global" if r["modelo"] == "MEDIA_GLOBAL" else 
                                      ("ridge_148" if r["modelo"] == "RIDGE148" else "hibrido_topo")),
                "n_variables": 0 if "PRIOR" in r["modelo"] or r["modelo"] == "MEDIA_GLOBAL" else 
                               (148 if r["modelo"] == "RIDGE148" else None),
                "tipo_entrenamiento": "prior_municipal_encogido" if "PRIOR" in r["modelo"] else
                                      ("media_global" if r["modelo"] == "MEDIA_GLOBAL" else
                                       ("ridge_regression" if r["modelo"] == "RIDGE148" else "prior_plus_ridge_topo")),
                "validacion": scheme,
                "n_folds": 6 if scheme == "KM6" else (9 if scheme == "LMO" else 5),
                "n_reps": 1 if scheme in ["KM6", "LMO"] else 1,
                "semilla": 20260219,
                "repeticion": int(r["repeat"]),
                "RMSE": r["RMSE"], "MAE": r["MAE"], "R2": r["R2"],
                "archivo_fuente_metricas": "outputs/predictor/metricas_por_repeticion.csv",
                "archivo_fuente_predicciones": "outputs/predictor/predicciones_oof.csv",
                "tiempo_s": None,
                "estado": "oficial",
                "limitaciones": f"Esquema {scheme} no comparable con RCV5x10"
            })

# ============================================================
# 3. BAYESIANOS - SCREENING (exploratorio, reps 0-4, NUTS 500x2)
# ============================================================
print("Cargando bayesianos screening...")
screen = pd.read_csv(BASE / "experiments/bayesian_hierarchical/results/phase3/screen_summary.csv")
for _, r in screen.iterrows():
    rows.append({
        "modelo": r["config"],
        "familia": "bayesiano",
        "conjunto_variables": r["config"],
        "n_variables": {"B0": 0, "B3": 3, "B5": 5, "B8": 8, "B12": 12, "B20": 20}.get(r["config"], None),
        "tipo_entrenamiento": "NUTS_MCMC_hierarchico",
        "validacion": "RCV5x10_reps_0-4",
        "n_folds": 5,
        "n_reps": 5,
        "semilla": 20260219,
        "repeticion": "screening",
        "RMSE": r["rmse_mean"], "MAE": r["mae_mean"], "R2": r["r2_mean"],
        "CRPS": r["crps_mean"], "cobertura_80": r["cov80_mean"], "cobertura_95": r["cov95_mean"],
        "max_Rhat": r["max_rhat"], "min_ESS": r["min_ess"], "divergencias": r["div_total"],
        "tiempo_promedio_s": r["time_mean"],
        "archivo_fuente_metricas": "experiments/bayesian_hierarchical/results/phase3/screen_summary.csv",
        "archivo_fuente_predicciones": f"experiments/bayesian_hierarchical/results/phase3/cv_{r['config']}_screen_selection_oof.csv",
        "tiempo_s": r["time_mean"] * 25,
        "estado": "exploratorio",
        "limitaciones": "NUTS 500+500 draws, 2 cadenas; etiquetado EXPLORATORIO"
    })

# ============================================================
# 4. BAYESIANOS - FINALISTAS (50 folds, NUTS 2000x4)
# ============================================================
print("Cargando bayesianos finalistas...")
finalistas = pd.read_csv(BASE / "experiments/bayesian_hierarchical/results/phase3/finalists_summary.csv")
for _, r in finalistas.iterrows():
    rows.append({
        "modelo": r["model"],
        "familia": "bayesiano",
        "conjunto_variables": r["model"],
        "n_variables": {"B0": 0, "B3": 3, "B5": 5}.get(r["model"], None),
        "tipo_entrenamiento": "NUTS_MCMC_hierarchico",
        "validacion": f"RCV5x10_{r['block']}",
        "n_folds": 5,
        "n_reps": 5 if r["block"] in ["selection", "evaluation"] else 10,
        "semilla": 20260219,
        "repeticion": r["block"],
        "RMSE": r["rmse"], "MAE": r["mae"], "R2": None,
        "archivo_fuente_metricas": "experiments/bayesian_hierarchical/results/phase3/finalists_summary.csv",
        "archivo_fuente_predicciones": f"experiments/bayesian_hierarchical/results/phase3/cv_{r['model']}_finalists_finalists_oof.csv",
        "tiempo_s": None,
        "estado": "finalista",
        "limitaciones": "NUTS 2000+1000 draws, 4 cadenas; selection=reps 0-4, evaluation=reps 5-9"
    })

# ============================================================
# 5. EXPERIMENTALES DESARROLLO
# ============================================================
print("Cargando modelos experimentales desarrollo...")
dev_base = BASE / "desarrollo/cebada_datos_baseline_1/modelos_experimentales"

modelos_dev = [
    ("ExtraTrees_model/resultados_extra_trees_agro/metricas_extra_trees.csv", "ExtraTrees"),
    ("RandomForest_model/resultados_random_forest_agro/metricas_random_forest.csv", "RandomForest"),
    ("Ridge_model/resultados_ridge_agrocebada/metricas_ridge.csv", "Ridge"),
    ("LinearRegresion_model/resultados_regresion_lineal_agro/metricas_regresion_lineal.csv", "Lineal"),
]

for fpath, fam in modelos_dev:
    full = dev_base / fpath
    if not full.exists():
        print(f"  ADVERTENCIA: no existe {full}")
        continue
    df = pd.read_csv(full)
    for _, r in df.iterrows():
        n_vars = int(r.get("n_variables", 0))
        scenario = str(r.get("escenario", ""))
        val_type = "aleatoria_5_folds" if "aleatoria" in scenario.lower() else "espacial_6_bloques"
        rows.append({
            "modelo": r.get("modelo", f"{fam}_{n_vars}"),
            "familia": f"experimental_{fam.lower()}",
            "conjunto_variables": f"{n_vars}_vars_{scenario}",
            "n_variables": n_vars,
            "tipo_entrenamiento": f"{fam}_{scenario}",
            "validacion": val_type,
            "n_folds": 5 if "aleatoria" in val_type else 6,
            "n_reps": 1,
            "semilla": None,
            "repeticion": 0,
            "RMSE": r.get("RMSE_t_ha"), "MAE": r.get("MAE_t_ha"), "R2": r.get("R2"),
            "archivo_fuente_metricas": str(full.relative_to(BASE)),
            "archivo_fuente_predicciones": str(full.relative_to(BASE)).replace("metricas", "predicciones_validacion"),
            "tiempo_s": None,
            "estado": "exploratorio",
            "limitaciones": f"Validación {val_type}; no comparable con RCV5x10 oficial"
        })

# ============================================================
# Guardar
# ============================================================
df_out = pd.DataFrame(rows)
cols_order = [
    "modelo", "familia", "conjunto_variables", "n_variables", "tipo_entrenamiento",
    "validacion", "n_folds", "n_reps", "semilla", "repeticion",
    "RMSE", "MAE", "R2", "CRPS", "cobertura_80", "cobertura_95",
    "max_Rhat", "min_ESS", "divergencias", "tiempo_promedio_s",
    "RMSE_std", "MAE_std", "R2_std",
    "archivo_fuente_metricas", "archivo_fuente_predicciones",
    "tiempo_s", "estado", "limitaciones"
]
for c in cols_order:
    if c not in df_out.columns:
        df_out[c] = None
df_out = df_out[cols_order]

out_path = OUTDIR / "modelos_comparativa.csv"
df_out.to_csv(out_path, index=False)
print(f"\nGuardado: {out_path} ({len(df_out)} filas)")

# Copiar archivos de detalle por repetición
import shutil
detail_dir = OUTDIR / "modelos_comparativa_by_repetition"
detail_dir.mkdir(exist_ok=True)

metricas_rep.to_csv(detail_dir / "oficiales_metricas_por_repeticion.csv", index=False)

screen_folds = pd.read_csv(BASE / "experiments/bayesian_hierarchical/results/phase3/screen_folds_all.csv")
screen_folds.to_csv(detail_dir / "bayesianos_screening_folds.csv", index=False)

final_folds = pd.read_csv(BASE / "experiments/bayesian_hierarchical/results/phase3/finalists_folds_all.csv")
final_folds.to_csv(detail_dir / "bayesianos_finalistas_folds.csv", index=False)

for fpath, fam in modelos_dev:
    base = dev_base / fpath
    fold_file = base.parent / f"metricas_por_fold_{fam.lower()}.csv"
    if fold_file.exists():
        pd.read_csv(fold_file).to_csv(detail_dir / f"desarrollo_{fam}_por_fold.csv", index=False)

print(f"Archivos de detalle en: {detail_dir}")
