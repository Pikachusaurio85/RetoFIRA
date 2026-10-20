# Experimento bayesiano jerárquico — Reto AgroCebada 2026

Estudio **independiente** que responde si la información ambiental mejora de forma robusta el prior
municipal de rendimiento, o si el límite está en la cantidad/calidad de etiquetas.

**Conclusión: Escenario C** — las covariables ambientales no mejoran de forma robusta al baseline
municipal; el jerárquico solo-municipio (B0) iguala al prior oficial. Ver
[`reports/05_final_report.md`](reports/05_final_report.md).

> Este experimento **no modifica** ningún módulo oficial (`outputs/`, `desarrollo/.../integration`).
> Solo lee las fuentes oficiales para comparar y validar.

## Estructura

```
experiments/bayesian_hierarchical/
├── configs/                 feature_sets.json, priors.json, mcmc_config.json, splits_RCV5x10.json
├── src/
│   ├── config.py            rutas y constantes (SEED, RCV5x10)
│   ├── data/               loader, splits, audit, municipality
│   ├── models/             hierarchical, preprocess, priors
│   ├── validation/         metrics, cv_runner, baselines
│   └── analysis/           compare
├── tests/                  test_hierarchical.py, test_metrics.py
├── run_phase1.py ... run_phase5.py   entrada por fase
├── results/phase1..5/      auditoría, folds, OOF, resúmenes, modelo final
├── reports/                informes 01–05 (+ figures/)
└── progress.md
```

## Entorno

```bash
conda env: agrocebada-xai   (Python 3.10.22, PyMC 5.25.1, ArviZ 0.23.4, numpy/pandas/scipy/matplotlib)
# se instaló: pip install pymc arviz   (aviso inofensivo de xport/pandas)
```

Ejecutar desde el **root del proyecto**.

## Reproducción

```bash
cd experiments/bayesian_hierarchical

# Fase 1: auditoría + folds + validación de conjuntos
conda run -n agrocebada-xai python run_phase1.py
# Fase 2: tests + smoke test
conda run -n agrocebada-xai python -m pytest tests -q
conda run -n agrocebada-xai python run_phase2.py
# Fase 3: cribado exploratorio + finalistas (50 folds NUTS 2000×4)
conda run -n agrocebada-xai python run_phase3_screen.py
conda run -n agrocebada-xai python run_phase3_finalists.py
# Fase 4: interpretación (ICC, pooling, betas, test intra-municipio)
conda run -n agrocebada-xai python run_phase4.py
# Fase 5: modelo final + exportación
conda run -n agrocebada-xai python run_phase5.py
```

Todos los runners son **reanudables** (saltan folds ya presentes en los CSV). `conda run` no
soporta heredoc; usar los scripts `.py`.

## Esquema experimental

- **Validación:** `RCV5x10` oficial (50 folds, train 110–111 / test 27–28), reconstruido y validado
  contra `outputs/predictor/splits.csv`. Comparación **emparejada** contra el OOF oficial del
  `PRIOR` (`predicciones_oof.csv`).
- **Sin fuga:** imputación, escalado y VIF se ajustan **solo con el train** de cada fold.
- **Selección vs evaluación:** repeticiones 0–4 = selección; 5–9 = evaluación limpia.
- **Conjuntos:** B0 (solo municipio), B3, B5 (5 conceptos XAI exactos), B8, B12, B20.
- **Métricas:** RMSE, MAE, R², CRPS, log-score, cobertura/ancho 50-80-95 %, ESS, R-hat, divergencias.

## Datos

- `data_processed/features_parcela.csv` (197×220) + `outputs/xai/perfiles_conceptuales.csv`
  (197×29) → master 197×236. Objetivo = `RENDIMIENTO_T_HA` (solo 138 ENTRENAMIENTO).
- Las 59 parcelas PREDICCION nunca se usan como etiqueta; sus rendimientos no se tocan.
