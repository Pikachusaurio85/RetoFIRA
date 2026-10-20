# Fase 3 — Resultados de la experimentación progresiva

> Todos los números provienen de `RCV5x10` (50 folds). Etiquetas: **EXPLORATORIO** = cribado
> (NUTS corto), **FINAL** = pool de 50 folds con NUTS 2000×4. Selección = repeticiones 0–4,
> evaluación limpia = repeticiones 5–9.

## 1. Cribado exploratorio (rep 0–4, NUTS 500×2, 25 folds) — EXPLORATORIO

| Set | p | RMSE | MAE | R² | CRPS | cov80 | cov95 | max R-hat | min ESS | div | t/fold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0** | 0 | **0.5051** | **0.3422** | **0.6132** | **0.2692** | 0.877 | 0.939 | 1.029 | 154 | 0 | 4.4 s |
| B3 | 3 | 0.5052 | 0.3599 | 0.6119 | 0.2720 | 0.862 | 0.958 | 1.028 | 177 | 1 | 5.2 s |
| B5 | 5 | 0.5135 | 0.3639 | 0.5984 | 0.2761 | 0.856 | 0.958 | 1.047 | 66 | 2 | 5.4 s |
| B8 | 8 | 0.5149 | 0.3649 | 0.5984 | 0.2763 | 0.861 | 0.958 | 1.025 | 151 | 1 | 5.8 s |
| B12 | 12 | 0.5201 | 0.3707 | 0.5892 | 0.2799 | 0.863 | 0.961 | 1.027 | 266 | 0 | 6.8 s |
| B20 | 20 | 0.5307 | 0.3738 | 0.5770 | 0.2836 | 0.851 | 0.958 | 1.018 | 233 | 1 | 6.2 s |

Prior municipal (recomputado, mismos folds): **RMSE = 0.5136, MAE = 0.3378**.

**Lectura:** el baseline municipal B0 es el mejor en RMSE/MAE/CRPS; añadir variables ambientales
degrada monótonamente la predicción. La ESS/baja de B5 en este cribado refleja el NUTS corto
(de 2 cadenas), no un problema del modelo final.

## 2. Finalistas (FINAL, NUTS 2000×4, 50 folds)

Finalistas elegidos tras el cribado: **B0** (baseline), **B3** (mejor set ambiental), **B5**
(conceptos XAI oficiales). B8/B12/B20 se descartan por dominancia (peores y más costosos).

| Modelo | Bloque | RMSE | MAE | n |
|---|---|---|---|---|
| **B0** | selección (0–4) | 0.5114 | 0.3411 | 690 |
| **B0** | evaluación (5–9) | **0.5104** | 0.3393 | 690 |
| **B0** | 50 folds | **0.5109** | 0.3402 | 1380 |
| B3 | selección | 0.5122 | 0.3598 | 690 |
| B3 | evaluación | 0.5128 | 0.3596 | 690 |
| B3 | 50 folds | 0.5125 | 0.3597 | 1380 |
| B5 | selección | 0.5212 | 0.3633 | 690 |
| B5 | evaluación | 0.5156 | 0.3596 | 690 |
| B5 | 50 folds | 0.5184 | 0.3614 | 1380 |
| PRIOR oficial | selección | 0.5136 | 0.3377 | 690 |
| PRIOR oficial | evaluación | 0.5110 | 0.3351 | 690 |
| PRIOR oficial | 50 folds | 0.5123 | 0.3364 | 1380 |

El patrón se mantiene idéntico en selección y en evaluación limpia (no hay sorpresa de
sobreajuste a las particiones de selección).

## 3. Comparaciones emparejadas (diferencias por parcela, bootstrap 95%)

| Comparación | RMSE A | RMSE B | Δ(err²) IC95% | Wilcoxon p |
|---|---|---|---|---|
| B0 vs PRIOR | 0.5109 | 0.5123 | **[−0.0044, +0.0014]** (cruza 0) | 8.5e-4 |
| B3 vs PRIOR | 0.5125 | 0.5123 | [−0.0065, +0.0071] (cruza 0) | 2.1e-6 |
| B5 vs PRIOR | 0.5184 | 0.5123 | [−0.0009, +0.0134] | 8.3e-7 |
| B3 vs B0 | 0.5125 | 0.5109 | [−0.0038, +0.0072] (cruza 0) | 4.6e-8 |
| B5 vs B0 | 0.5184 | 0.5109 | **[+0.0019, +0.0138] (todo > 0)** | 2.5e-6 |

**Lecturas clave:**
- **B0 ≈ PRIOR oficial**: la mejora de RMSE de B0 no es robusta (IC de la diferencia de error
  cuadrático cruza 0); en MAE el prior oficial es algo mejor (0.3364 vs 0.3402). Empate técnico.
- **Ningún set ambiental mejora a B0** de forma robusta. B3 es indistinguible de B0.
- **B5 (conceptos XAI) es significativamente peor que B0** (IC de Δ(err²) enteramente > 0).

## 4. Diagnósticos y calibración (FINAL, 50 folds)

| Modelo | RMSE | MAE | CRPS | cov50 | cov80 | cov95 | ancho80 | max R-hat | min ESS | divergencias |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 0.5028 | 0.3406 | 0.2696 | 0.607 | 0.878 | 0.940 | 1.34 | 1.006 | 1329 | 31 |
| B3 | 0.5054 | 0.3600 | 0.2731 | 0.606 | 0.867 | 0.958 | 1.35 | 1.005 | 1634 | 51 |
| B5 | 0.5105 | 0.3617 | 0.2756 | 0.585 | 0.860 | 0.956 | 1.36 | 1.006 | 1894 | 60 |

(El RMSE por fold, 0.5028, es ligeramente menor que el agregado OOF 0.5109, como es esperable.)

- **Convergencia:** R-hat máximo 1.006 (< 1.01) con 4 cadenas; ESS mínima >1300. Excelente.
- **Divergencias:** ~0.6–1.2 por ajuste (31–60 en 150 ajustes × 4 cadenas × 1000 tune),
  ≈0.03 % de las iteraciones. Marginal; probablemente embudo de municipios con n=1.
- **Calibración:** cobertura 80 % ≈ 0.86–0.88 (ligeramente sobre-cubierta), 95 % ≈ 0.94–0.96.
  Los intervalos son algo anchos (ancho80 ≈ 1.34 t/ha) pero honestos.

## 5. Conclusión de la Fase 3

La información ambiental **no mejora** la predicción frente al baseline municipal. El límite del
desempeño está en el número/calidad de etiquetas por municipio, no en la disponibilidad de
covariables ambientales. Detalle mecanístico en `04_interpretability.md`.

Artefactos: `results/phase3/screen_summary.csv`, `screen_folds_all.csv`,
`finalists_summary.csv`, `finalists_folds_all.csv`, `finalists_paired_vs_prior.csv`,
`finalists_paired_env_vs_B0.csv`, `official_prior_oof.csv`,
`cv_*_finalists_finalists_{folds,oof}.csv`.
