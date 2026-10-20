# Fase 1 — Auditoría de datos

**Proyecto:** Modelos jerárquicos bayesianos para AgroCebada 2026
**Directorio:** `experiments/bayesian_hierarchical/`
**Fecha de ejecución:** generado por `run_phase1.py`
**Fuente maestra:** `desarrollo/cebada_datos_baseline_1/data_processed/features_parcela.csv` (197 × 220) unida con
`outputs/xai/perfiles_conceptuales.csv` (197 × 29, solo columnas crudas, se descartan las `*_scaled`).

Ningún módulo oficial fue modificado. Este directorio es de solo lectura respecto a los artefactos oficiales.

---

## 1. Estructura del conjunto

| Elemento | Valor |
|---|---|
| Parcelas totales | 197 |
| ENTRENAMIENTO (con rendimiento observado) | 138 |
| PREDICCION (sin objetivo) | 59 |
| Columnas de features numéricas disponibles | 231 |
| Valor objetivo media (entrenamiento) | 3.9554 t/ha (coincide con `media_global_t_ha` oficial) |
| Desv. estándar objetivo | 0.8512 t/ha |
| Rango objetivo | 2.0 – 5.59 t/ha |
| Valores objetivo múltiplos de 0.5 | 14.5% (redondeo no dominante) |
| Outliers por IQR (1.5·IQR) | 0 |

Las 59 parcelas de PREDICCION **no** tienen rendimiento observado (0 valores objetivo). Se respeta la
regla: solo los 138 rendimientos de ENTRENAMIENTO se usan como etiquetas.

## 2. Municipios

| Municipio | Entrenamiento | Predicción | Total | <10 etiquetas |
|---|---:|---:|---:|:--:|
| Chignahuapan | 71 | 29 | 100 | no |
| Calpulalpan | 30 | 14 | 44 | no |
| Singuilucan | 16 | 7 | 23 | no |
| Almoloya | 10 | 4 | 14 | no |
| Cuautepec de Hinojosa | 5 | 4 | 9 | **sí** |
| Emiliano Zapata | 3 | 0 | 3 | sí |
| Nanacamilpa de Mariano Arista | 1 | 1 | 2 | sí |
| Apan | 1 | 0 | 1 | sí |
| Tepeapulco | 1 | 0 | 1 | sí |

**Cinco** municipios tienen menos de 10 etiquetas de entrenamiento (verificado desde los datos).
Todos los municipios con parcelas de PREDICCION están presentes en entrenamiento (Chignahuapan,
Calpulalpan, Singuilucan, Almoloya, Cuautepec, Nanacamilpa). Aun así, en las particiones RCV5x10
algunos municipios pequeños pueden quedar con 0 observaciones en el entrenamiento de un fold;
el modelo debe propagar la incertidumbre de esos efectos (ver Fase 2).

### Medias de rendimiento por municipio (entrenamiento)

| Municipio | n | media | sd | icc-contexto |
|---|---:|---:|---:|---|
| Chignahuapan | 71 | 4.477 | 0.584 | municipio grande |
| Calpulalpan | 30 | 3.000 | 0.371 | |
| Singuilucan | 16 | 3.081 | 0.434 | |
| Almoloya | 10 | 4.521 | 0.188 | |
| Cuautepec de Hinojosa | 5 | 4.598 | 0.151 | etiquetas escasas |
| Emiliano Zapata | 3 | 3.500 | 0.000 | varianza nula |
| Nanacamilpa | 1 | 2.500 | – | 1 etiqueta |
| Apan | 1 | 3.0 | — | 1 etiqueta |
| Tepeapulco | 1 | 4.45 | – | 1 etiqueta |

**Emiliano Zapata** presenta desviación 0 (las tres parcelas comparten valor 3.5). Los municipios de
1 etiqueta no permiten estimar varianza intra-municipal.

## 3. Descomposición de varianza (entrenamiento)

| Componente | SS | Fracción |
|---|---:|---:|
| Entre municipios | 68.11 | 0.686 |
| Dentro de municipios | 31.14 | 0.314 |
| Total | 99.25 | 1.000 |

**ICC aproximado = 0.686**: ~69% de la varianza del rendimiento se explica por el municipio.
Esto valida el modelado jerárquico con efecto municipal y deja ~31% de varianza intra-municipio
como margen teórico para variables ambientales.

## 4. Calidad de features

- **Faltantes en las features del diseño B3/B5/B8/B12/B20: 0%** en todas las columnas seleccionadas.
- **3 features constantes** en todo el dataset: `flag_n_meses_s2_abr_oct`, `flag_meses_clima_2025`,
  `flag_topo_completa` (todas banderas de cobertura = 1). Excluidas de los conjuntos.
- **207 pares de features con |r| ≥ 0.95**: alta multicolinealidad global. Justifica cribado VIF
  dentro de cada fold de entrenamiento y el uso de priors que contraen los coeficientes.

## 5. Mapeo de conceptos XAI (verificación)

Los 5 conceptos oficiales ya están calculados en `outputs/xai/perfiles_conceptuales.csv`. Se usan
sus columnas **exactas** (sin re-escalar) en B5. Se comparó su reproducibilidad desde las features
de ingeniería:

| Concepto | Mejor feature de ingeniería | r de Pearson |
|---|---|---:|
| C1a_ndvi_s2_mean | media(4 ventanas NDVI) | 0.775 |
| C2a_amplitud_ndvi | s2_ndvi_amp | 0.954 |
| C3a_ndwi_s2_mean | media(4 ventanas NDWI) | 0.812 |
| C4a_prec_acum_abr_oct | clim_prec_temp | **1.000** |
| C5a_elev_media | topo_elev_media | 1.000 |

**Conclusión metodológica:** C4a y C5a son idénticos a features de ingeniería. C1a y C3a son medias
de temporada (abr–oct) que no se reproducen exactamente a partir de las medias por ventana; C2a difiere
ligeramente (r=0.95). Por indicación metodológica se usa la **columna conceptual exacta** para B5, y se
documenta que su definición (media de la curva estacional) difiere de las features de ventana.

## 6. Riesgos de fuga (leakage)

- El objetivo y la partición se leen de `features_parcela.csv` (sin duplicar etiquetas en features).
- Las features derivan de satélite 2025, clima 2025, topografía y geometría — ninguna deriva del
  rendimiento. Revisión manual de nombres no detectó columnas derivadas del objetivo.
- Las columnas `*_scaled` de `perfiles_conceptuales.csv` se descartan para evitar escalado global
  (potencial fuga leve). El escalado se recalcula dentro de cada fold de entrenamiento.
- Se excluyen deliberadamente `loc_lon`, `loc_lat`, `loc_x_utm`, `loc_y_utm` para que el efecto
  municipal (variable categórica) capture la estructura espacial y no sea absorbida por coordenadas.

## 6b. Archivos producidos

- `results/phase1/audit_summary.json`
- `results/phase1/column_summary.csv`
- `results/phase1/municipality_counts.csv`
- `results/phase1/per_municipality_yield.csv`
- `results/phase1/high_correlations.csv`
- `results/phase1/concept_mapping.csv`
- `results/phase1/feature_sets_validation.json`
- `configs/splits_RCV5x10.json`
