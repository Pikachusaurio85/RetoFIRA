# AgroCebada FIRA 2026 — paquete de experimentos de validación (separado del pipeline original)

Este paquete contiene **todo lo necesario para reproducir y auditar** el informe `report/INFORME_AgroCebada_validacion.md`:
scripts, configuración, semillas, particiones, predicciones fuera de muestra (OOF), errores por parcela/bloque/municipio,
tablas de resultados, figuras y verificaciones de integridad.

**No se modificó el pipeline original.** El ZIP `DATASET_RETO_AGRO_2026.zip` se extrajo en un directorio aparte, su árbol
(274 archivos) se hasheó antes del análisis (`results/00_original_tree_sha256_BEFORE.txt`) y `scripts/99_verify_integrity.py`
confirma que sigue idéntico. **No se usaron las etiquetas de las 59 parcelas ocultas ni predicciones para ellas** como
evaluación; esas 59 solo aparecen *sin etiqueta* como features en la validación adversaria (`00_audit.py`) y en la
persistencia temporal (`09_temporal_persistence.py`).

## Estructura

| Carpeta / archivo | Contenido |
|---|---|
| `scripts/` | Código. `common.py` (datos, familias de variables, prior municipal, ajuste/predicción), `grid.py` (lista de 136 + 19 configuraciones), `oofio.py` (inferencia pareada: bootstrap por parcela y por clúster, Holm, etiquetas de decisión), `spatial.py` (Moran, variogramas, vecindades) y los scripts numerados `00`…`99` |
| `config/` | `splits_config.json` (semillas y parámetros de particiones), `regimen_captura_por_municipio.csv` |
| `splits/` | `splits_long.csv` (scheme, rep, fold, role, id); huella SHA-256 `56c85a25a3cd31a8…`; todas las configuraciones usan estas mismas particiones |
| `oof/main/` | Una tabla por configuración: `cfg, scheme, rep, fold, id, y, pred, n_train` (OOF por repetición, solo parcelas etiquetadas) |
| `oof/bloo/` | Igual, para LOO con buffer (radio 0–8 km) |
| `oof/shap/` | SHAP agrupado por parcela, **solo sobre parcelas fuera de entrenamiento** (`id, municipio, LOC, S2, LS, PL, CLIM, TOPO, GEOM, MUN_prior`) |
| `oof/oof_promedio_reps_todas_configs.csv.gz` | `cfg, scheme, id, municipio, y, pred_oof_mean, abs_err, sq_err_mean_reps` (promedio sobre repeticiones) |
| `results/` | Todas las tablas del informe (`00_…` auditoría, `01_…` esquemas, `02_…` varianza/espacial, `04_…` ablaciones y contrastes, `05_…` estabilidad, `06_…` dentro de municipio, `07_…` SHAP/permutación, `08_…` variables interpretables, `09_…` persistencia temporal, `99_…` integridad) |
| `figures/` | `fig01`–`fig07` y `fig09`–`fig12` (la numeración salta `fig08`) |
| `report/` | Informe ejecutivo en Markdown |
| `logs/` | Salidas de consola de las corridas |
| `run_all.sh` | Orquestador (ver advertencia) |
| `modelos/` | Cuatro experimentos independientes: Ridge, regresión lineal, Random Forest y Extra Trees, con scripts, requisitos y resultados existentes |

## Experimentos de modelos

Los modelos independientes que antes estaban en `MODELOS/` ahora forman parte de `experiments/modelos/`. Se conservaron sus scripts, documentación, archivos de requisitos y resultados generados. Para reproducirlos, ejecuta los comandos de cada `LEEME_*.md` desde esta carpeta; los scripts buscan por defecto `DATASET_RETO_AGRO_2026.zip` aquí y guardan nuevas salidas en la carpeta de resultados de su propio modelo. No es necesario volver a ejecutarlos para consultar los resultados incluidos.

Estos cuatro experimentos son distintos de la validación principal descrita en este paquete: emplean sus propios esquemas aleatorios/espaciales y reportan las métricas documentadas en sus respectivos README.

## Diccionario de las tablas de errores

- **`oof/oof_promedio_reps_todas_configs.csv.gz` y `results/04_errores_por_parcela_configs_clave.csv.gz`**
  - `cfg`: `<enfoque>|<modelo>|<conjunto>`; enfoque `direct` (predice el rendimiento) o `resid_muni` (prior municipal + modelo sobre el residuo); modelo `Ridge`/`ET`/`CB`; referencias como `ref_global` y `ref_muni`
  - `scheme`: esquema de validación (`RCV5x10`, `SPA_GRID3B05`, `SPA_GRID8B1`, `SPA_BAND6`, `SPA_KM4/6/8`, `LMO`).
  - `y`: rendimiento 2025 (t/ha). `pred_oof_mean`: predicción fuera de muestra promediada sobre repeticiones. `abs_err`: |y − pred_oof_mean|. `sq_err_mean_reps`: media sobre repeticiones del error cuadrático (es la pérdida por parcela usada en los contrastes pareados).
- **`results/04_metricas_por_bloque.csv`**: `esquema, bloque, cfg, n, municipios, y_media, pred_media, sesgo, RMSE, MAE` (bloques de los esquemas KM/BAND/GRID/LMO).
- **`results/04_metricas_por_municipio.csv`**: `cfg, esquema, municipio, n, RMSE, sesgo`.
- **`results/04_contrastes_pareados.csv`**: un renglón por contraste A vs B en un esquema. `dRMSE` (negativo = A mejor), IC95 por bootstrap de parcelas (`ci_*_par`) y por 25 micro-clústeres espaciales (`ci_*_clu`; es el que decide), `p_wilcoxon_se`, `p_clu_holm`/`p_par_holm` (Holm por familia de contrastes `grupo`), `decision` ∈ {A MEJOR, A PEOR, EQUIVALENTE, INDETERMINADO} con ROPE de ±0.02 t/ha.

## Reproducir

```
bash run_all.sh /ruta/a/DATASET_RETO_AGRO_2026.zip
```

Orden de scripts: `00_audit` → `01_make_splits` → `02_variance_spatial` → `03_run_cv_grid --group main` y `--group bloo` →
`02b_residual_autocorr` → `04_ablation_analysis` → `04b_bloo_analysis` → `05_stability` → `05b_cur24_control` →
`06_within_municipality` → `06b_prior_shrink_control` → `07_shap_groups` → `08_sparse_interpretable` →
`09_temporal_persistence` → `10_figures` → `99_verify_integrity`.

Semillas: global `20261013`; RCV 1000–1009; mallas 2000+rep+lado; k-means `n_init=10, random_state=0`; modelos con la misma semilla
por (rep, fold) para que los errores sean pareables. Tiempos con 2 núcleos: malla principal ≈ 70 min, BLOO ≈ 40 min,
permutaciones intra-municipio (06) ≈ 1 h.

## Advertencias

1. `scripts/common.py` fija `EXP=/home/claude/experiments`; si mueves el directorio, cambia esa constante.
2. Cada script se ejecutó individualmente en esta sesión; **`run_all.sh` no se corrió completo de una sola vez en un entorno limpio.**
3. La carpeta `data/` (copias de solo lectura de los insumos) no se incluye en este paquete; `run_all.sh` la recrea desde el ZIP original (hashes de las copias en `results/00_input_copies_sha256.txt`).
4. **Transferencia temporal: no ejecutable** (un solo ciclo con rendimiento; las etiquetas no traen año).
5. Las cifras del informe salen de `results/*.csv`; el control del prior (`06b`) solo cubre Ridge.

Entorno: Python 3.13.16, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, scikit-learn 1.9.1, CatBoost 1.2.10, shap 0.53.0, statsmodels 0.15.0, matplotlib 3.11.2.
