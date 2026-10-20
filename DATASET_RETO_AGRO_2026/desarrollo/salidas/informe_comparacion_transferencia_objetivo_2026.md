# Comparación unificada de baselines — AgroCebada

## Objetivo y autorización

Horizonte de transferencia: estimación **antes de la siembra 2026**. No hay observaciones de clima, satélite ni rendimiento de 2026. Los predictores fechados llegan hasta 2025. La fuente canónica de partición y etiquetas es `ID_area_rendimiento_70_30_Reto_AgroCebada.csv`: sus 138 parcelas `ENTRENAMIENTO` tienen `RENDIMIENTO_T_HA`; las 59 parcelas `PREDICCION` no tienen etiqueta. El archivo no especifica el ciclo al que corresponde cada rendimiento etiquetado.

Este informe compara señales y referencias con evaluación espacial común para revisar el comportamiento con las etiquetas disponibles; no es una evaluación temporal ni demuestra rendimiento futuro. **No autoriza estimaciones respaldadas para 2026**. Las 59 parcelas sin etiqueta no intervienen en la selección ni en las métricas de error; después de congelar el modelo solo se inspecciona en ellas cobertura/dominio y municipios ausentes, sin calcular predicciones de rendimiento.

## Procedencia y huellas

| Fuente | bytes | SHA-256 | Estado |
|---|---:|---|---|
| `cebada_datos_baseline_1\data_processed\features_parcela.csv` | 736389 | `07d444f56da566130b64033795b762311017eb8b74544746566c8f9ea95b1af2` | archivo_derivado |
| `cebada_datos_baseline_1\data_processed\diccionario_features.csv` | 26035 | `24f341bf7e246bf7a9b95752b313442516658c008a4eace795c27185ed2d9a61` | archivo_derivado |
| `cebada_datos_baseline_1\data_processed\series_dekadales_2025.csv` | 11639659 | `ae913d26bc4ba866c2eec72c4c4ee358a91afbf04f8200ea3ff75280c1c8696b` | archivo_derivado |
| `salidas\tabla_modelado_parcelaria.csv` | 575894 | `b1840eade25b022d7cc134fd57149a9f8b37986691f24fdf32cf6385478b7c8f` | archivo_derivado |
| `salidas\trazabilidad_archivos.csv` | 1936 | `d8af98c0f0d72e29ac66411abf1ff2718f799ee2349ed2b28145824798fd3b72` | archivo_derivado |
| `datos\entradas\Conjunto_datos_BASICO_AgroCebada2026.csv` | 33668409 | `96d68040d954d0903d761fd8ceed0dcd4a08e3842915a3ca44517d60e8bee4ef` | coincide_con_trazabilidad |
| `datos\entradas\Conjunto_datos_PRO_AgroCebada.csv` | 5000244 | `91697f8f8d4ad5e59f766acbce2ef76f3c405eb02b41c1de92d62a07e8321888` | coincide_con_trazabilidad |
| `datos\entradas\ID_area_rendimiento_70_30_Reto_AgroCebada.csv` | 6241 | `ec07b7ede0ae692109d91b3e941b1a353fe7f89410507db9bfbe6d7c112590ef` | coincide_con_trazabilidad |
| `salidas\ml_evolucion_parcelas_2022_2025.csv` | 153573392 | `4f5f001258cdd1136b8d1e267c018478c29d1f358723cfe2c438f832dfc5da8c` | coincide_con_trazabilidad |
| `datos\geoespaciales\Parcelas_Reto_AGC_CONJUNTO_70_30\Parcelas_Reto_AGC_CONJUNTO.shp` | 168628 | `8f8001a4bc4da00c2fcc0797536e94183d1abc88e4a797a7c63bcd2b65f29f29` | coincide_con_trazabilidad |
| `datos\geoespaciales\Parcelas_Reto_AGC_CONJUNTO_70_30\Parcelas_Reto_AGC_CONJUNTO.shx` | 1676 | `b03e99bc1c619fd9351693091558f549e0b75220f41e1c3b6343dbc41cb7d1f7` | coincide_con_trazabilidad |
| `datos\geoespaciales\Parcelas_Reto_AGC_CONJUNTO_70_30\Parcelas_Reto_AGC_CONJUNTO.dbf` | 83951 | `9b7d9e06beb70c688c59d4c2e4227ef851596e3382af7d5566b9c669dc35debb` | coincide_con_trazabilidad |
| `datos\geoespaciales\Parcelas_Reto_AGC_CONJUNTO_70_30\Parcelas_Reto_AGC_CONJUNTO.prj` | 145 | `a02a27b1d1982c8516d83398e85a3c8b1aef1713c13ef4d84d7bde17430c07c4` | coincide_con_trazabilidad |
| `datos\geoespaciales\Reto_AgroCebada_Topografia_INEGI_CEM4\Elevacion_INEGI_CEM4_120m.tif` | 4488499 | `2f9b0a3deeb28089ef3253303b0fed1b67b3cac6e452a2d3d75c913d87fcc408` | coincide_con_trazabilidad |
| `datos\geoespaciales\Reto_AgroCebada_Topografia_INEGI_CEM4\Pendiente_INEGI_CEM4_120m_grados.tif` | 13405501 | `f0fa8b07e06873d40d3741a8e5f2bba8f652be810fea21fa0f6fe13f8dc48d45` | coincide_con_trazabilidad |

- Los archivos originales disponibles coinciden con sus hashes registrados por desarrollo.

## Conteos y controles de calidad

- Tabla maestra: 197 parcelas, clave única `ID_POLIGONO`.
- Etiquetas de entrenamiento disponibles en la tabla canónica: 138; parcelas `PREDICCION` sin rendimiento: 59.
- Features descritas en el diccionario versionado: 363.
- Se validó coincidencia exacta de IDs, particiones y etiquetas entre los dos desarrollos antes de concatenar.
- Las filas longitudinales permanecen como producto separado; su promedio no se mezcla con capturas individuales en la tabla maestra.
- No se generan reportes de calidad; el diccionario de features documenta período, unidad, agregación y disponibilidad temporal.

## Features, definiciones y calidad

| Candidato | n features | Definición |
|---|---:|---|
| `ref_media_global` | 0 | Media global de entrenamiento |
| `ref_media_municipal_encogimiento` | 0 | Media municipal con encogimiento n/(n+3) hacia la media global |
| `ref_knn_espacial_k5` | 0 | Media inversamente ponderada por distancia de los 5 vecinos |
| `ridge_satelite_solo` | 177 | Sentinel-2, Landsat y Planet; sin clima ni topografía |
| `ridge_satelite_topografia_geometria_baseline1` | 184 | Satélite + topografía zonal + geometría del baseline_1 |
| `ridge_satelite_topografia_geometria_clima_centroide` | 210 | Baseline anterior + clima 2025 por celda que contiene el centroide |
| `ridge_satelite_topografia_desarrollo_geometria` | 181 | Satélite + topografía zonal del desarrollo + geometría |
| `ridge_satelite_topografia_geometria_clima_zonal_historico` | 220 | Mismo set satélite/topografía/geometría + clima histórico zonal de desarrollo |
| `ridge_fenologia_curada_24` | 24 | Set curado de 24 variables fenológicas, topográficas y geométricas |
| `ridge_historico_desarrollo` | 148 | Promedios históricos de capturas BASIC/PRO/clima 2022-2025 + topografía parcelaria |
| `ridge_historico_desarrollo_sin_clima` | 112 | Features históricas de desarrollo sin variables de clima |
| `ridge_ubicacion_sola` | 4 | Longitud/latitud y coordenadas proyectadas del centroide |
| `ridge_ubicacion_mas_features_curadas` | 28 | Coordenadas del centroide + features fenológicas curadas |
| `ridge_prior_municipal_residuo_curado` | 24 | Prior municipal encogido + Ridge sobre residuo con features curadas |
| `ridge_prior_knn_residuo_curado` | 24 | Prior k-NN espacial + Ridge sobre residuo con features curadas |

`features_parcela.csv` del baseline_1 resume curvas de 2025: observaciones válidas agrupadas por dekada y sensor con mediana; los umbrales de nubosidad son Sentinel-2/Landsat <=5% y Planet <=20%. Si quedan al menos 3 dekadas válidas, se interpola linealmente y los extremos se extrapolan con el valor más cercano; con menos de 3, la serie queda faltante. `series_dekadales_2025.csv` conserva la serie longitudinal.

Desarrollo conserva observaciones longitudinales 2022–2025. Para modelar, primero calcula la media de cada feature entre capturas de cada parcela-año y luego promedia las medias anuales disponibles (no pondera años por número de capturas). Sus variables `HISTORICO_PROMEDIO__CLIMA__...` son distintas de `clim_*` del baseline_1: estas últimas son clima 2025 de la celda que contiene el centroide; las de desarrollo resumen capturas/valores climáticos por parcela. La topografía zonal de desarrollo se etiqueta por separado de `topo_*` del baseline_1; no se equiparan columnas por nombre.

## Protocolo espacial y selección

- Universo común: las mismas 138 parcelas con etiqueta, objetivo y partición reconciliados por `ID_POLIGONO`.
- Bloques: bandas espaciales contiguas y balanceadas sobre el eje principal de centroides UTM, configuraciones 4, 6, 8 bloques.
- En cada configuración, se reserva externamente el bloque más alejado del centro medio de los centroides. Los otros bloques alimentan validación interna GroupKFold; se elige el menor RMSE agrupado y, entre candidatos a no más de 0.02 t/ha tanto en RMSE como MAE, el de menos features. La prueba externa no interviene en la selección.
- En cada fold todos los candidatos comparten exactamente índices train/test. Los tamaños por bloque, predicciones externas, MAE, RMSE, R² y sesgo por fold se guardan en `comparacion_folds.csv` y `comparacion_predicciones.csv`.
- Ridge usa `StandardScaler`, imputación por mediana del entrenamiento (cero para columnas totalmente vacías) y alpha fijo 10.0; no se ajustan hiperparámetros con los bloques externos.
- La selección interna minimiza RMSE; entre alternativas con RMSE y MAE a no más de 0.02 t/ha del mejor, prefiere la de menor número de features. La regla se fija sin consultar el holdout.
- La media municipal usa encogimiento n/(n+3) hacia la media global; municipios ausentes del entrenamiento recurren a la media global. k-NN usa k=5, peso 1/max(distancia en m,1). En los residuos, los priors de entrenamiento se calculan leave-one-out.
- `comparacion_por_municipio.csv` desglosa el holdout por municipio; `comparacion_diagnostico_holdout.csv` informa rangos de predicción y cobertura; `comparacion_municipios_sin_entrenar.csv` identifica municipios de la reserva sin referencia municipal de entrenamiento, que recurren a la media global.
- Los RMSE informados en las columnas `interno_*` se calculan sobre predicciones OOF agrupadas. `externo_*` corresponde solo al holdout bloque reservado de esa configuración; no se elige el ganador a partir de ese resultado.

## Resultados por modelo y configuración

| Bloques | Candidato | RMSE interno | MAE interno | R² interno | RMSE externo | MAE externo | R² externo | Seleccionado interno | n externo |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| pca_bands_4 | ref_media_global | 0.780 | 0.585 | -0.159 | 1.298 | 1.242 | -10.855 | no | 35 |
| pca_bands_4 | ref_media_municipal_encogimiento | 0.588 | 0.404 | 0.339 | 1.298 | 1.242 | -10.855 | no | 35 |
| pca_bands_4 | ref_knn_espacial_k5 | 0.582 | 0.404 | 0.355 | 1.437 | 1.386 | -13.528 | sí | 35 |
| pca_bands_4 | ridge_satelite_solo | 0.745 | 0.546 | -0.058 | 1.188 | 1.054 | -8.939 | no | 35 |
| pca_bands_4 | ridge_satelite_topografia_geometria_baseline1 | 0.851 | 0.668 | -0.382 | 1.155 | 1.022 | -8.385 | no | 35 |
| pca_bands_4 | ridge_satelite_topografia_geometria_clima_centroide | 0.774 | 0.601 | -0.144 | 1.458 | 1.336 | -13.961 | no | 35 |
| pca_bands_4 | ridge_satelite_topografia_desarrollo_geometria | 0.834 | 0.648 | -0.328 | 1.179 | 1.044 | -8.793 | no | 35 |
| pca_bands_4 | ridge_satelite_topografia_geometria_clima_zonal_historico | 0.743 | 0.571 | -0.053 | 1.045 | 0.927 | -6.685 | no | 35 |
| pca_bands_4 | ridge_fenologia_curada_24 | 0.893 | 0.678 | -0.520 | 1.188 | 1.077 | -8.941 | no | 35 |
| pca_bands_4 | ridge_historico_desarrollo | 0.909 | 0.660 | -0.578 | 0.727 | 0.570 | -2.726 | no | 35 |
| pca_bands_4 | ridge_historico_desarrollo_sin_clima | 0.721 | 0.556 | 0.008 | 0.982 | 0.859 | -5.787 | no | 35 |
| pca_bands_4 | ridge_ubicacion_sola | 0.666 | 0.485 | 0.153 | 0.760 | 0.655 | -3.065 | no | 35 |
| pca_bands_4 | ridge_ubicacion_mas_features_curadas | 0.795 | 0.569 | -0.205 | 0.803 | 0.688 | -3.540 | no | 35 |
| pca_bands_4 | ridge_prior_municipal_residuo_curado | 0.693 | 0.487 | 0.084 | 1.222 | 1.156 | -9.516 | no | 35 |
| pca_bands_4 | ridge_prior_knn_residuo_curado | 0.647 | 0.461 | 0.200 | 1.449 | 1.392 | -13.784 | no | 35 |
| pca_bands_6 | ref_media_global | 0.843 | 0.696 | -0.145 | 1.191 | 1.120 | -7.721 | no | 23 |
| pca_bands_6 | ref_media_municipal_encogimiento | 0.679 | 0.488 | 0.257 | 0.545 | 0.433 | -0.825 | no | 23 |
| pca_bands_6 | ref_knn_espacial_k5 | 0.744 | 0.532 | 0.109 | 0.456 | 0.403 | -0.279 | no | 23 |
| pca_bands_6 | ridge_satelite_solo | 0.764 | 0.578 | 0.060 | 0.817 | 0.626 | -3.108 | no | 23 |
| pca_bands_6 | ridge_satelite_topografia_geometria_baseline1 | 0.751 | 0.579 | 0.092 | 0.841 | 0.655 | -3.348 | no | 23 |
| pca_bands_6 | ridge_satelite_topografia_geometria_clima_centroide | 0.793 | 0.581 | -0.013 | 0.687 | 0.570 | -1.899 | no | 23 |
| pca_bands_6 | ridge_satelite_topografia_desarrollo_geometria | 0.743 | 0.568 | 0.112 | 0.861 | 0.681 | -3.558 | no | 23 |
| pca_bands_6 | ridge_satelite_topografia_geometria_clima_zonal_historico | 0.735 | 0.554 | 0.130 | 0.908 | 0.726 | -4.070 | no | 23 |
| pca_bands_6 | ridge_fenologia_curada_24 | 0.797 | 0.618 | -0.022 | 0.802 | 0.642 | -2.957 | no | 23 |
| pca_bands_6 | ridge_historico_desarrollo | 0.582 | 0.425 | 0.456 | 1.115 | 0.904 | -6.644 | sí | 23 |
| pca_bands_6 | ridge_historico_desarrollo_sin_clima | 0.682 | 0.504 | 0.251 | 0.536 | 0.426 | -0.768 | no | 23 |
| pca_bands_6 | ridge_ubicacion_sola | 0.645 | 0.473 | 0.331 | 0.424 | 0.337 | -0.107 | no | 23 |
| pca_bands_6 | ridge_ubicacion_mas_features_curadas | 0.740 | 0.543 | 0.119 | 0.472 | 0.374 | -0.368 | no | 23 |
| pca_bands_6 | ridge_prior_municipal_residuo_curado | 0.667 | 0.484 | 0.283 | 0.527 | 0.421 | -0.707 | no | 23 |
| pca_bands_6 | ridge_prior_knn_residuo_curado | 0.779 | 0.560 | 0.023 | 0.441 | 0.384 | -0.198 | no | 23 |
| pca_bands_8 | ref_media_global | 0.856 | 0.723 | -0.150 | 1.196 | 1.131 | -8.409 | no | 18 |
| pca_bands_8 | ref_media_municipal_encogimiento | 0.696 | 0.495 | 0.241 | 0.567 | 0.447 | -1.118 | no | 18 |
| pca_bands_8 | ref_knn_espacial_k5 | 0.783 | 0.575 | 0.039 | 0.444 | 0.389 | -0.297 | no | 18 |
| pca_bands_8 | ridge_satelite_solo | 0.767 | 0.590 | 0.078 | 0.784 | 0.613 | -3.043 | no | 18 |
| pca_bands_8 | ridge_satelite_topografia_geometria_baseline1 | 0.771 | 0.601 | 0.068 | 0.818 | 0.662 | -3.397 | no | 18 |
| pca_bands_8 | ridge_satelite_topografia_geometria_clima_centroide | 0.856 | 0.625 | -0.150 | 0.620 | 0.509 | -1.530 | no | 18 |
| pca_bands_8 | ridge_satelite_topografia_desarrollo_geometria | 0.770 | 0.600 | 0.069 | 0.840 | 0.686 | -3.642 | no | 18 |
| pca_bands_8 | ridge_satelite_topografia_geometria_clima_zonal_historico | 0.789 | 0.579 | 0.025 | 0.845 | 0.644 | -3.695 | no | 18 |
| pca_bands_8 | ridge_fenologia_curada_24 | 0.825 | 0.638 | -0.068 | 0.784 | 0.618 | -3.047 | no | 18 |
| pca_bands_8 | ridge_historico_desarrollo | 0.612 | 0.453 | 0.413 | 1.018 | 0.805 | -5.815 | no | 18 |
| pca_bands_8 | ridge_historico_desarrollo_sin_clima | 0.713 | 0.519 | 0.203 | 0.541 | 0.418 | -0.927 | no | 18 |
| pca_bands_8 | ridge_ubicacion_sola | 0.587 | 0.429 | 0.460 | 0.433 | 0.347 | -0.232 | sí | 18 |
| pca_bands_8 | ridge_ubicacion_mas_features_curadas | 0.700 | 0.521 | 0.231 | 0.456 | 0.367 | -0.369 | no | 18 |
| pca_bands_8 | ridge_prior_municipal_residuo_curado | 0.712 | 0.523 | 0.204 | 0.534 | 0.413 | -0.873 | no | 18 |
| pca_bands_8 | ridge_prior_knn_residuo_curado | 0.834 | 0.621 | -0.092 | 0.462 | 0.406 | -0.402 | no | 18 |

No se interpretan las 59 parcelas sin etiqueta para declarar un ganador. La comparación tiene muestras pequeñas y el holdout cambia entre configuraciones; reportar la dispersión entre bloques y configuraciones es parte del resultado, no una promesa de rendimiento futuro.

## Criterios previos para autorizar transferencia

La variante escogida internamente solo podrá considerarse respaldada si: (1) MAE y RMSE mejoran frente a la mejor referencia geográfica tanto en CV interna como en el holdout externo; (2) R² >= 0 en ambas; (3) los IC95 del sesgo observado-menos-predicho incluyen cero en ambas; y (4) para las 59 parcelas sin etiqueta, las fracciones faltantes y fuera de rango no superan las del holdout etiquetado. Son los criterios conservadores existentes de desarrollo, adaptados para comparar contra referencias geográficas; se fijan en código antes de ejecutar y quedan evaluados en `comparacion_resumen_transferencia_objetivo_2026.csv`. Aun superándolos, no sustituyen la validación temporal con etiquetas del ciclo objetivo. No se generan predicciones de rendimiento para las 59 parcelas.

## Límites

- El año/ciclo de las etiquetas de entrenamiento no está indicado en la tabla canónica; no hay observaciones ni verdad-terreno 2026, por lo que no existe validación temporal.
- Las bandas espaciales contiguas y las 138 etiquetas producen estimaciones inciertas; una mejora agregada puede depender de una zona. Revisar `comparacion_folds.csv`, tamaños por bloque y errores parcela a parcela.
- Clima 2025 es antecedente legítimo antes de siembra 2026, pero no se debe trasladar a una predicción de 2025 hecha antes de observar el ciclo.
- Los resultados recientes (RMSE 0.653/0.661/0.729 t/ha) son hipótesis de trabajo bajo protocolos previos diferentes; no se mezclan como si fueran comparaciones de folds comunes ni garantías.
- Las imágenes originales del baseline_1 no están incluidas en `data_processed`; el runner portátil puede reconstruirlo si se proporciona el árbol de fuentes descrito en su README. Este informe regenera la unión y evaluación a partir de las tablas parcelarias procesadas disponibles.
