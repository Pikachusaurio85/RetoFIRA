# Progreso — Bayesian Hierarchical AgroCebada

## Fase 1 — Auditoría de datos y diseño: COMPLETADA
- Cargador, auditoría, análisis municipal y reconstrucción de folds RCV5x10.
- 138 entrenamiento / 59 predicción; 5 municipios con <10 etiquetas.
- Folds RCV5x10 validados (50 folds, train+test = 138, sin fugas).
- Conjuntos B0/B3/B5/B8/B12/B20 definidos y validados (todas las columnas existen).
- Informes: `reports/01_data_audit.md`, `reports/01_experiment_design.md`.
- Resultados: `results/phase1/*`.
- ICC≈0.686 (69% varianza entre municipios).

## Fase 2 — Modelo y tests: COMPLETADA
- `src/models/hierarchical.py`, `preprocess.py`, priors.
- 8 tests pasan (recuperación, municipio no visto, no-fuga, CRPS).
- Smoke test OK: B0 rmse 0.542, B5 rmse 0.558 (fold rep0/0), sin divergencias.
- Informe: `reports/02_model_implementation.md`.

## Fase 3 — Experimentación progresiva: COMPLETADA
- Cribado (rep 0–4, NUTS corto): B0 mejor (RMSE 0.5051); añadir variables degrada monótonamente.
- Finalistas B0/B3/B5 a 50 folds NUTS 2000×4: B0 0.5109 < PRIOR 0.5123 < B3 0.5125 < B5 0.5184.
- Paired: B0≈PRIOR (IC cruza 0); B5 peor que B0 (IC todo>0). Diagnósticos excelentes.
- Informe: `reports/03_experiment_results.md`.

## Fase 4 — Interpretación: COMPLETADA
- ICC 0.695 [0.464, 0.878]; pooling parcial contrae n=1 (λ≈0.72).
- Único efecto creíble: elevación (~0.1 t/ha); resto incluyen 0.
- Test intra-municipio corr(delta_B5, resid) = +0.05 → sin señal ambiental intra-municipio.
- Informe: `reports/04_interpretability.md`; figuras en `reports/figures/`.

## Fase 5 — Modelo final e informe: COMPLETADA
- Modelos finales B0/B5 exportados a `results/phase5/` (JSON + netCDF).
- Informe final `reports/05_final_report.md` + `README.md`.
- **Decisión: Escenario C** — sin mejora robusta ambiental; límite en las etiquetas.
  Mantener el prior municipal oficial.

## ESTADO GLOBAL: TERMINADO (Fases 1–5). 8/8 tests OK. Ningún módulo oficial modificado.

