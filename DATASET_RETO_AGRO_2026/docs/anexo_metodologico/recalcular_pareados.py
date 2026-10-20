#!/usr/bin/env python3
"""
Recalculación de contrastes pareados usando 138 parcelas únicas como unidad estadística.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

BASE = Path("/mnt/FEDORA-DATOS/ModoZebada/DATASET_RETO_AGRO_2026 (1)/DATASET_RETO_AGRO_2026")
OUTDIR = BASE / "docs/anexo_metodologico"

SEED = 20260219
N_BOOT = 5000
RNG = np.random.default_rng(SEED)

COMPARISONS = [
    ("MEDIA_GLOBAL", "PRIOR"),
    ("PRIOR_K3", "PRIOR"),
    ("RIDGE148", "PRIOR"),
    ("HIBRIDO_TOPO", "PRIOR"),
    ("B0", "PRIOR"),
    ("B3", "PRIOR"),
    ("B5", "PRIOR"),
    ("B3", "B0"),
    ("B5", "B0"),
]

def load_official_oof():
    df = pd.read_csv(BASE / "outputs/predictor/predicciones_oof.csv")
    df = df[df["scheme"] == "RCV5x10"].copy()
    df = df.rename(columns={"y_true_t_ha": "y_true", "y_pred_t_ha": "y_pred", "model_id": "model"})
    return df[["model", "repeat", "fold", "ID_POLIGONO", "y_true", "y_pred"]]

def load_bayesian_oof(model_name):
    f = BASE / f"experiments/bayesian_hierarchical/results/phase3/cv_{model_name}_finalists_finalists_oof.csv"
    df = pd.read_csv(f)
    df = df.rename(columns={"config": "model"})
    df["model"] = model_name
    return df[["model", "repeat", "fold", "ID_POLIGONO", "y_true", "y_pred"]]

def aggregate_parcel_losses(df):
    df = df.copy()
    df["abs_loss"] = (df["y_true"] - df["y_pred"]).abs()
    df["sq_loss"] = (df["y_true"] - df["y_pred"]) ** 2
    agg = df.groupby(["ID_POLIGONO", "model"]).agg(
        mean_abs_loss=("abs_loss", "mean"),
        mean_sq_loss=("sq_loss", "mean"),
        y_true=("y_true", "first")
    ).reset_index()
    return agg

def bootstrap_ci(data, n_boot=N_BOOT, alpha=0.05):
    means = np.array([RNG.choice(data, size=len(data), replace=True).mean() for _ in range(n_boot)])
    lo = float(np.quantile(means, alpha/2))
    hi = float(np.quantile(means, 1 - alpha/2))
    return lo, hi

def run_comparison(model_a, model_b, agg_a, agg_b):
    merged = agg_a[["ID_POLIGONO", "mean_abs_loss", "mean_sq_loss"]].merge(
        agg_b[["ID_POLIGONO", "mean_abs_loss", "mean_sq_loss"]],
        on="ID_POLIGONO", suffixes=("_a", "_b")
    )
    n_parcels = len(merged)
    diff_abs = merged["mean_abs_loss_a"] - merged["mean_abs_loss_b"]
    diff_sq = merged["mean_sq_loss_a"] - merged["mean_sq_loss_b"]
    mean_diff_abs = float(diff_abs.mean())
    mean_diff_sq = float(diff_sq.mean())
    rmse_a = float(np.sqrt(agg_a["mean_sq_loss"].mean()))
    rmse_b = float(np.sqrt(agg_b["mean_sq_loss"].mean()))
    mae_a = float(agg_a["mean_abs_loss"].mean())
    mae_b = float(agg_b["mean_abs_loss"].mean())
    ci_abs = bootstrap_ci(diff_abs.values)
    ci_sq = bootstrap_ci(diff_sq.values)
    try:
        stat_abs, p_abs = wilcoxon(diff_abs.values, alternative="two-sided")
    except ValueError:
        stat_abs, p_abs = 0.0, 1.0
    try:
        stat_sq, p_sq = wilcoxon(diff_sq.values, alternative="two-sided")
    except ValueError:
        stat_sq, p_sq = 0.0, 1.0
    return {
        "model_a": model_a, "model_b": model_b, "n_parcels": n_parcels,
        "loss_type": "absolute", "mean_diff": mean_diff_abs,
        "ci_lo": ci_abs[0], "ci_hi": ci_abs[1],
        "wilcoxon_stat": float(stat_abs), "wilcoxon_p": float(p_abs),
        "rmse_a": rmse_a, "rmse_b": rmse_b, "mae_a": mae_a, "mae_b": mae_b,
        "method": "paired_parcel_bootstrap_wilcoxon",
        "aggregation": "mean_over_10_reps_per_parcel",
        "n_boot": N_BOOT, "seed": SEED,
    }, {
        "model_a": model_a, "model_b": model_b, "n_parcels": n_parcels,
        "loss_type": "squared", "mean_diff": mean_diff_sq,
        "ci_lo": ci_sq[0], "ci_hi": ci_sq[1],
        "wilcoxon_stat": float(stat_sq), "wilcoxon_p": float(p_sq),
        "rmse_a": rmse_a, "rmse_b": rmse_b, "mae_a": mae_a, "mae_b": mae_b,
        "method": "paired_parcel_bootstrap_wilcoxon",
        "aggregation": "mean_over_10_reps_per_parcel",
        "n_boot": N_BOOT, "seed": SEED,
    }

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Cargando OOF oficiales...")
    official = load_official_oof()
    print(f"  Oficial RCV5x10: {len(official)} filas, modelos: {official['model'].unique()}")
    print("Cargando OOF bayesianos...")
    bayes = {}
    for m in ["B0", "B3", "B5"]:
        bayes[m] = load_bayesian_oof(m)
        print(f"  {m}: {len(bayes[m])} filas")
    all_oof = pd.concat([official] + list(bayes.values()), ignore_index=True)
    print(f"Total OOF: {len(all_oof)} filas")
    print("Agregando pérdidas por parcela (10 reps -> 1)...")
    agg = aggregate_parcel_losses(all_oof)
    print(f"Parcelas únicas: {agg['ID_POLIGONO'].nunique()}")
    print(f"Modelos: {agg['model'].unique()}")
    agg_by_model = {m: agg[agg["model"] == m].set_index("ID_POLIGONO") for m in agg["model"].unique()}
    results_abs = []
    results_sq = []
    for model_a, model_b in COMPARISONS:
        if model_a not in agg_by_model or model_b not in agg_by_model:
            print(f"  ADVERTENCIA: {model_a} o {model_b} no disponibles, saltando")
            continue
        print(f"Contraste: {model_a} vs {model_b}")
        r_abs, r_sq = run_comparison(model_a, model_b, agg_by_model[model_a].reset_index(), agg_by_model[model_b].reset_index())
        results_abs.append(r_abs)
        results_sq.append(r_sq)
        print(f"  |abs| diff={r_abs['mean_diff']:.6f} CI=[{r_abs['ci_lo']:.6f},{r_abs['ci_hi']:.6f}] p={r_abs['wilcoxon_p']:.4e}")
        print(f"  |sq| diff={r_sq['mean_diff']:.6f} CI=[{r_sq['ci_lo']:.6f},{r_sq['ci_hi']:.6f}] p={r_sq['wilcoxon_p']:.4e}")
    all_results = pd.DataFrame(results_abs + results_sq)
    csv_path = OUTDIR / "pareados_corregidos.csv"
    all_results.to_csv(csv_path, index=False)
    print(f"\nCSV guardado: {csv_path}")
    json_data = {
        "metadata": {
            "n_parcelas_unicas": 138, "n_repeticiones_por_parcela": 10,
            "esquema_validacion": "RCV5x10", "n_bootstrap": N_BOOT,
            "semilla_bootstrap": SEED, "perdidas_evaluadas": ["absolute", "squared"],
            "metodo_agregacion": "media_de_10_evaluaciones_por_parcela",
            "contraste": "wilcoxon_signed_rank_dos_colas",
            "fecha_calculo": pd.Timestamp.now().isoformat(),
        },
        "resultados": results_abs + results_sq
    }
    json_path = OUTDIR / "pareados_corregidos.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON guardado: {json_path}")
    agg.to_csv(OUTDIR / "perdidas_por_parcela_agregadas.csv", index=False)
    print(f"Pérdidas por parcela guardadas: {OUTDIR / 'perdidas_por_parcela_agregadas.csv'}")

if __name__ == "__main__":
    main()
