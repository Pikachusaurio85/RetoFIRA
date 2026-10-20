# Limitaciones y Hallazgos Fase 1

## 1. Matriz de correlación entre componentes base

|                       |   C1a_ndvi_s2_mean |   C1b_evi_s2_mean |   C2a_amplitud_ndvi |   C2b_auc_exceso_ndvi |   C2c_doy_pico_ndvi |   C3a_ndwi_s2_mean |   C3b_msi_s2_mean |   C3c_nddi_s2_mean |   C4a_prec_acum_abr_oct |   C4b_tmean_abr_oct |   C5a_elev_media |   C5b_pend_media |
|:----------------------|-------------------:|------------------:|--------------------:|----------------------:|--------------------:|-------------------:|------------------:|-------------------:|------------------------:|--------------------:|-----------------:|-----------------:|
| C1a_ndvi_s2_mean      |              1     |             0.935 |               0.424 |                 0.714 |              -0.364 |              0.637 |            -0.486 |             -0.098 |                   0.096 |              -0.059 |            0.177 |            0.151 |
| C1b_evi_s2_mean       |              0.935 |             1     |               0.523 |                 0.758 |              -0.327 |              0.592 |            -0.427 |             -0.042 |                   0.031 |              -0.158 |            0.08  |            0.047 |
| C2a_amplitud_ndvi     |              0.424 |             0.523 |               1     |                 0.674 |              -0.084 |              0.41  |            -0.264 |             -0.047 |                   0.263 |              -0.316 |            0.107 |           -0.165 |
| C2b_auc_exceso_ndvi   |              0.714 |             0.758 |               0.674 |                 1     |              -0.284 |              0.438 |            -0.277 |              0.058 |                  -0.04  |              -0.038 |            0.121 |           -0.196 |
| C2c_doy_pico_ndvi     |             -0.364 |            -0.327 |              -0.084 |                -0.284 |               1     |             -0.213 |             0.169 |              0.074 |                  -0.182 |               0.068 |            0.175 |            0.032 |
| C3a_ndwi_s2_mean      |              0.637 |             0.592 |               0.41  |                 0.438 |              -0.213 |              1     |            -0.965 |             -0.776 |                   0.378 |              -0.14  |            0.475 |            0.306 |
| C3b_msi_s2_mean       |             -0.486 |            -0.427 |              -0.264 |                -0.277 |               0.169 |             -0.965 |             1     |              0.877 |                  -0.386 |               0.117 |           -0.504 |           -0.38  |
| C3c_nddi_s2_mean      |             -0.098 |            -0.042 |              -0.047 |                 0.058 |               0.074 |             -0.776 |             0.877 |              1     |                  -0.395 |               0.031 |           -0.448 |           -0.364 |
| C4a_prec_acum_abr_oct |              0.096 |             0.031 |               0.263 |                -0.04  |              -0.182 |              0.378 |            -0.386 |             -0.395 |                   1     |              -0.174 |            0.016 |           -0.017 |
| C4b_tmean_abr_oct     |             -0.059 |            -0.158 |              -0.316 |                -0.038 |               0.068 |             -0.14  |             0.117 |              0.031 |                  -0.174 |               1     |            0.025 |           -0.038 |
| C5a_elev_media        |              0.177 |             0.08  |               0.107 |                 0.121 |               0.175 |              0.475 |            -0.504 |             -0.448 |                   0.016 |               0.025 |            1     |            0.452 |
| C5b_pend_media        |              0.151 |             0.047 |              -0.165 |                -0.196 |               0.032 |              0.306 |            -0.38  |             -0.364 |                  -0.017 |              -0.038 |            0.452 |            1     |

## 2. Cobertura temporal S2

- Parcelas con 7 meses válidos: 197/197
- Observaciones por parcela: media=31.1, std=2.5, min=25, max=38

## 3. Comparación con baseline (features_parcela.csv)

Parcelas comunes: 197/197

- C1a_ndvi_s2_mean vs s2_ndvi_base (NDVI baseline pre-ciclo vs media abr-oct (distintas definiciones)): r=0.3754, n=197
- C2a_amplitud_ndvi vs s2_ndvi_amp (Amplitud pico-baseline): r=0.9541, n=197
- C2b_auc_exceso_ndvi vs s2_ndvi_auc (AUC exceso sobre baseline): r=0.9461, n=197
- C2c_doy_pico_ndvi vs s2_ndvi_peak_doy (DOY pico NDVI): r=0.8013, n=197

## 4. Sensibilidad bandwidth ProtoDash (teórica)

- ProtoDash usa kernel RBF: k(x,y) = exp(-||x-y||² / (2σ²))
- σ = 1.0 usado en smoke test (heurística)
- Alternativas: σ = mediana(distancias pareadas), 0.5, 2.0
- Cambios en σ alteran selección de prototipos y pesos
- Recomendación: probar rango y documentar en Fase 2

## 5. Limitaciones estructurales

1. **Ningún componente mide estado fisiológico** — son descriptores espectrales/temporales/climáticos/cartográficos
2. **Ventanas calendáricas, no fenológicas** — DOY 91–304 y ≤151 son cortes fijos
3. **Sensores no armonizados** — S2 y Planet tienen escalas/sesgos distintos; solo S2 en espacio conceptual
4. **Clima no parcelario** — 25 celdas 0.05° para 197 parcelas (no independientes)
5. **Topografía 120m** — Parcelas pequeñas = pocos píxeles DEM (mediana 6, mínimo 1)
6. **Etiquetas sin año/ciclo** — No se puede alinear observaciones 2025 con etiqueta de ciclo
7. **Ponderación igual por defecto** — Familias con más componentes (F2, F3 = 3) dominan distancia euclidiana
8. **Baseline pre-ciclo** — Mediana DOY≤151 puede ser ruidosa (pocas obs en algunas parcelas)
9. **NDVI/EVI redundantes** — Correlación ~0.97 → doble peso efectivo en F1
10. **Índices hídricos correlacionados** — NDWI, MSI, NDDI no ortogonales

## 6. Decisiones de diseño documentadas

- ProtoDash: optimizer="osqp" (cvxpy falla por compatibilidad scipy/cvxpy)
- ProtoDash: kernelType="Gaussian", sigma=1.0 (heurística, probar sensibilidad Fase 2)
- RobustScaler(quantile_range=(25,75)) para todos los componentes
- Ventana temporal fija: DOY 91–304 (abr-oct), baseline DOY≤151
- Baseline = mediana pre-ciclo (no media, más robusto a outliers)
- Pico = máximo observado crudo (sin suavizado 3-meses)
- AUC exceso = trapecio sobre NDVI - baseline, clipeado a >=0
- Todas las 197 parcelas conservadas (no exclusión por cobertura)
- Flags de calidad por parcela para transparencia
