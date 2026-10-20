# Diseño experimental — Modelo jerárquico bayesiano

## 1. Pregunta

¿La información ambiental permite mejorar de forma robusta el prior municipal, o el límite está
fijado por la cantidad/calidad de etiquetas?

## 2. Esquema de validación

- **Primario:** `RCV5x10` oficial (10 repeticiones × 5 folds = 50 folds), reconstruido desde
  `outputs/predictor/splits.csv` y validado (`configs/splits_RCV5x10.json`).
  Cada fold: train 110–111 parcelas ENTRENAMIENTO, test 27–28. Ninguna parcela PREDICCION entra
  en train/test.
- **Comparación emparejada:** se comparan predicciones por la misma (repeat, fold, parcela) contra
  el prior municipal oficial. Las predicciones OOF oficiales del modelo `PRIOR` están disponibles en
  `outputs/predictor/predicciones_oof.csv` (1380 filas RCV5x10 = 138 × 10). Se usan directamente y se
  recomputa su RMSE para verificar consistencia (~0.5123).
- **Sin selección sobre evaluación:** para evitar elegir hiperparámetros/conjuntos sobre las mismas
  particiones que se reportan como evaluación independiente:
  - **Repeticiones 0–4 (25 folds):** selección de modelo/variables (cribado exploratorio).
  - **Repeticiones 5–9 (25 folds):** evaluación final clean (no se revisa el modelo tras verlas).
  - Se reportan ambos además del agregado de 50 folds, etiquetado explícitamente.

## 3. Prevención de fuga

Dentro de CADA fold, y solo con el train de ese fold:
1. Imputación (mediana) — aunque las features elegidas no tienen faltantes.
2. Estandarización z-score (media/desv. del train).
3. Filtro de multicolinealidad VIF < 10 (solo B20; los conjuntos ≤12 son lo bastante pequeños).
4. Cualquier estadístico municipal para predicción.

Nada se calcula con el test del fold. Los parámetros del escalador por fold se registran.

## 4. Modelo

```
y_i ~ Normal(mu_i, sigma)
mu_i = alpha + u_{m(i)} + x_i . beta
u_j = tau * z_j ,  z_j ~ Normal(0,1)        (parametrización no centrada)
```

- `alpha`: intercepto global.
- `u_j`: efecto municipal con pooling parcial (contrae municipios pequeños hacia la media global).
- `beta`: efectos de covariables estandarizadas.
- `sigma`: desviación residual.
- `tau`: desviación entre municipios.

### Priors (escala estandarizada de covariables)

| Parámetro | Prior | Justificación |
|---|---|---|
| alpha | Normal(4.0, 1.5) | rendimiento ~4 t/ha, débilmente informativo |
| beta (B3,B5) | Normal(0, 0.5) | 1 SD de feature ~ hasta 0.5 t/ha |
| beta (B8,B12) | Normal(0, 0.3) | más contracción con más variables |
| beta (B20) | Normal(0, 0.2) | contracción fuerte |
| sigma | HalfNormal(1.0) | escala residual plausible (<SD objetivo 0.85) |
| tau | HalfNormal(0.7) | SD entre municipios observada ~0.7 |

Los coeficientes se interpretan en t/ha por 1 desviación estándar de cada covariable (porque las
covariables se estandarizan dentro del fold). Esto hace comparables los priors entre conjuntos.

### Municipios no vistos en un fold

Se propaga la incertidumbre: `u_nuevo = tau * z_nuevo` con `z_nuevo ~ Normal(0,1)`, por lo que la
predicción posterior predictiva para una parcela de municipio desconocido es
`alpha + x.beta + tau*z`, integrando tau. NO se usa un fallback determinista al intercepto global
para construir intervalos. El punto (mediana) coincide con `alpha + x.beta`.

## 5. Métricas

- **Punto:** RMSE, MAE, R² (OOF agregado y por fold; media y SD entre repeticiones).
- **Emparejadas:** distribución de diferencias de error por parcela (Bayes − PRIOR) con intervalo y
  contraste (Wilcoxon firmado + diferencia de medias bootstrap).
- **Calibración:** cobertura empírica de intervalos posterior-predictivos al 50/80/95%, ancho medio
  de intervalo, CRPS y log-predictive density.
- **Por municipio:** RMSE/MAE/cobertura por municipio, con atención a n<10.

## 6. Conjuntos de variables (`configs/feature_sets.json`)

| Set | p | Contenido |
|---|---|---|
| B0 | 0 | solo efecto municipal (baseline jerárquico) |
| B3 | 3 | clim_prec_temp, topo_elev_media, s2_ndvi_amp |
| B5 | 5 | los 5 conceptos XAI oficiales exactos |
| B8 | 8 | B5 + clim_tmean_temp, clim_prec_cv, topo_pend_media |
| B12 | 12 | B8 + s2_evi_amp, s2_lai_amp, clim_gdd0_temp, clim_prec_anom_pct |
| B20 | 20 | B12 + 8 (dosel, humedad, fenología, helada, geometría) |

## 7. Presupuesto y parada

- **Smoke test:** B0, 1 fold, NUTS 500+500, 2 cadenas → validar pipeline.
- **Cribado:** B0/B3/B5/B8/B12/B20 en 5 folds (rep selección 0–4) con inferencia económica
  (NUTS corto 500 draws, 2 cadenas) — etiquetado EXPLORATORIO.
- **Finalistas:** NUTS 2000 draws, 4 cadenas, 50 folds, sujeto a convergencia.
- **Parada:** >5% divergencias o R-hat>1.05 sostenido o >10 min/fold → se marca inviable y se documenta
  (sin borrar resultados negativos).
- Logs por fold (tiempos reales, ESS, R-hat, divergencias). Checkpoints reanudables.

## 8. Deliverables

Ver README. Todos en `experiments/bayesian_hierarchical/`. No se modifica ningún módulo oficial.
