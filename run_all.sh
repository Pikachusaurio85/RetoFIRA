#!/usr/bin/env bash
# Reproduce los experimentos de AgroCebada de principio a fin (NO toca el pipeline original).
# Uso:  bash run_all.sh /ruta/a/DATASET_RETO_AGRO_2026.zip      (por defecto /mnt/user-data/uploads/DATASET_RETO_AGRO_2026.zip)
# Tiempo aprox. en 2 nucleos: ~70 min (malla principal) + ~40 min (BLOO) + ~60 min (06) + ~15 min (resto).
# Requisitos: python>=3.11, numpy, scipy, pandas, scikit-learn, catboost, shap, statsmodels, matplotlib, joblib.
set -euo pipefail
ZIP="${1:-/mnt/user-data/uploads/DATASET_RETO_AGRO_2026.zip}"
EXP="$(cd "$(dirname "$0")" && pwd)"            # scripts/common.py asume EXP=/home/claude/experiments; ajuste la constante EXP si se mueve
RAW="/home/claude/data_raw"
mkdir -p "$RAW" "$EXP/data" "$EXP/results" "$EXP/logs"
# 0. extraer el ZIP en un directorio aparte y guardar hashes ANTES de cualquier analisis
sha256sum "$ZIP" > "$EXP/results/00_zip_sha256.txt"
unzip -q -o "$ZIP" -d "$RAW"
( cd "$RAW" && find . -type f -print0 | sort -z | xargs -0 sha256sum ) > "$EXP/results/00_original_tree_sha256_BEFORE.txt"
# copiar SOLO los insumos (solo lectura)
cp "$RAW/DATASET_RETO_AGRO_2026/desarrollo/cebada_datos_baseline_1/data_processed/features_parcela.csv" "$EXP/data/"
cp "$RAW/DATASET_RETO_AGRO_2026/desarrollo/cebada_datos_baseline_1/data_processed/diccionario_features.csv" "$EXP/data/" 2>/dev/null || true
cp "$RAW/DATASET_RETO_AGRO_2026/desarrollo/datos/entradas/ID_area_rendimiento_70_30_Reto_AgroCebada.csv" "$EXP/data/asignacion_oficial.csv"
chmod a-w "$EXP"/data/*
cd "$EXP"
python3 scripts/00_audit.py                  2>&1 | tee logs/00_audit.log
python3 scripts/01_make_splits.py            2>&1 | tee logs/01_splits.log
python3 scripts/02_variance_spatial.py       2>&1 | tee logs/02_variance.log
python3 scripts/03_run_cv_grid.py --group main --n_jobs 2 2>&1 | tee logs/03_main.log
python3 scripts/03_run_cv_grid.py --group bloo --n_jobs 1 2>&1 | tee logs/03_bloo.log
python3 scripts/02b_residual_autocorr.py     2>&1 | tee logs/02b_resid.log
python3 scripts/04_ablation_analysis.py      2>&1 | tee logs/04_ablation.log
python3 scripts/04b_bloo_analysis.py         2>&1 | tee logs/04b_bloo.log
python3 scripts/05_stability.py              2>&1 | tee logs/05_stability.log
python3 scripts/05b_cur24_control.py         2>&1 | tee logs/05b_cur24.log
python3 scripts/06_within_municipality.py    2>&1 | tee logs/06_within.log
python3 scripts/06b_prior_shrink_control.py  2>&1 | tee logs/06b_prior_shrink.log
python3 scripts/07_shap_groups.py            2>&1 | tee logs/07_shap.log
python3 scripts/08_sparse_interpretable.py   2>&1 | tee logs/08_sparse.log
python3 scripts/09_temporal_persistence.py   2>&1 | tee logs/09_temporal.log
python3 scripts/10_figures.py fig01 fig02 fig03 fig04 fig05 fig06 fig07 fig09 fig10 fig11 fig12
python3 scripts/99_verify_integrity.py "$RAW" 2>&1 | tee logs/99_integrity.log
