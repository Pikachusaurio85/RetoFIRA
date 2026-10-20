# Fase 2 — Implementación y tests del modelo

## Modelo

```
y_i ~ Normal(mu_i, sigma)
mu_i = alpha + u_{m(i)} + x_i . beta
u_j = tau * z_j ,  z_j ~ Normal(0, 1)      # no centrada
```

Implementado en `src/models/hierarchical.py` (PyMC 5.25). Las covariables se estandarizan z-score
dentro de cada fold de entrenamiento (`src/models/preprocess.py`), por lo que `beta` está en t/ha por
1 DE. Para municipios no vistos en el fold, la predicción posterior-predictiva usa
`u_nuevo = tau * z_nuevo` con `z_nuevo ~ Normal(0,1)`, propagando la incertidumbre entre municipios
(no hay fallback determinista).

Priors (`configs/priors.json`): alpha~N(4,1.5), sigma~HalfNormal(1), tau~HalfNormal(0.7),
beta~N(0, s) con s=0.5 (B3,B5), 0.3 (B8,B12), 0.2 (B20).

## Tests (`tests/`)

`pytest tests/test_hierarchical.py tests/test_metrics.py`:

| Test | Resultado |
|---|---|
| Recuperación de parámetros (sintético, alpha/beta/sigma/tau) | PASS |
| Predicción de municipio no visto con varianza inter-municipal propagada | PASS |
| Modelo sin covariables (B0) y predicción con 0 features | PASS |
| Escalador ajustado solo con train (sin fuga) | PASS |
| CRPS: distribución degenerada = MAE | PASS |
| CRPS: predicción perfecta = 0 | PASS |
| CRPS: coincide con fuerza bruta | PASS |
| Cobertura monótona 50% ≤ 95% | PASS |

## Smoke test (`results/phase2/smoke_test.json`)

Fold `RCV5x10_rep0_fold0` (28 test), NUTS 500/500, 2 cadenas, ~5 s por config.

| Config | n_feat | RMSE | MAE | CRPS | cob80 | cob95 | max R-hat | min ESS | div |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0 | 0.5424 | 0.3835 | 0.2913 | 0.786 | 0.893 | 1.013 | 282 | 0 |
| B5 | 5 | 0.5584 | 0.3859 | 0.2941 | 0.821 | 0.893 | 1.007 | 193 | 0 |

Observaciones:
- Pipeline funcional; sin divergencias.
- R-hat ~1.01 con solo 2 cadenas/500 draws (esperado); los finalistas usan 4 cadenas.
- Cobertura al 95% ≈ 0.89 (ligeramente estrecha), consistente con la subcobertura documentada por el
  predictor oficial (0.865 al 80%).
- En este único fold, B5 **no** mejora a B0 (0.558 vs 0.542). Señal temprana a confirmar en el cribado.

**Checkpoint 2 superado:** tests verdes, smoke test válido → se procede a la Fase 3.
