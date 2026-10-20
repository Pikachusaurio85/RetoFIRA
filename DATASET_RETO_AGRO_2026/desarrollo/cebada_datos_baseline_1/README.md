# Reto AgroCebada 2026 – Datos y baseline

Pipeline de datos hasta el baseline (equipo DATASET CEBADA). Todo sale de los archivos del zip `Cebada.zip`.

## Qué hay aquí

| Carpeta / archivo | Contenido |
| --- | --- |
| `data_processed/features_parcela.csv` | **Tabla principal**: 197 parcelas × (CONJUNTO, RENDIMIENTO_T_HA, 214 features numéricas, 3 columnas de ubicación en texto). Sin faltantes. |
| `data_processed/diccionario_features.csv` | Qué significa cada columna (grupo y descripción). |
| `data_processed/series_dekadales_2025.csv` | Series 2025 por parcela, sensor y variable, en dekadas de 10 días (mediana de observaciones válidas + interpolación lineal). Útil para conceptos/prototipos sobre curvas. |
| `data_processed/parcelas_base.csv` | Atributos del shapefile + geometría (área, perímetro, compacidad, centroide) + topografía (media, std, mín, máx, valor en centroide) + celda de clima. |
| `data_processed/clima_mensual_parcela.csv` | Precipitación, Tmin y Tmax mensuales 2022–2025 por parcela (celda de 0.05° del centroide). |
| `reports/baseline_resultados_cv.csv` | Resultados de todos los baselines (RMSE, MAE, R², por bloques espaciales, dentro de Chignahuapan). |
| `reports/experimentos_residuo_curado.csv` | Experimentos "prior geográfico + modelo sobre el residuo" y conjunto curado de 24 features. |
| `reports/baseline_oof_predicciones.csv` | Predicciones fuera de muestra (promedio de 5 repeticiones) de las 138 parcelas de entrenamiento. |
| `reports/baseline_predicciones_59_parcelas.csv` | Predicciones de las 59 parcelas de PREDICCION con tres modelos (referencia por municipio, Baseline 1, Baseline 2). |
| `reports/baseline1_importancias_permutacion.csv` | Importancia por permutación (en validación cruzada) del Baseline 1. |
| `figures/` | Gráficas usadas en el documento. |
| `scripts/` | Código numerado en orden de ejecución. |

## Resultados del baseline (validación cruzada 5 particiones × 5 repeticiones, 138 parcelas)

| Modelo | Información que usa | RMSE (t/ha) | R² |
| --- | --- | --- | --- |
| Media global | — | 0.85 | −0.01 |
| Media por municipio | municipio | 0.52 | 0.62 |
| **Baseline 1**: Extra Trees | Sentinel-2, Landsat, Planet, topografía y geometría (sin ubicación ni clima) | 0.64 | 0.43 |
| Extra Trees | Baseline 1 + clima | 0.54 | 0.60 |
| **Baseline 2** (híbrido) | media por municipio + Ridge sobre el residuo con 24 features curadas | 0.51 | 0.64 |

El documento completo (hallazgos, tablas, gráficas, riesgos y preguntas abiertas) está en el archivo de Docs que acompaña esta entrega.

## Cómo reproducir

Python 3.13 con: pandas, numpy, scikit-learn, lightgbm, rasterio, pyshp, pyproj, shapely, scipy, matplotlib.

Las fuentes de este proyecto se conservan en `../datos/entradas/` y `../datos/geoespaciales/`: CSV y DOCX en `entradas`, y capas/rásteres ya extraídos en `geoespaciales`. Esas son las rutas predeterminadas. Si las fuentes están en otro sitio, define `CEBADA_SOURCE_ROOT` con la ruta a una carpeta que contenga las subcarpetas `entradas` y `geoespaciales`; no hace falta editar scripts.

Desde `desarrollo`:

```powershell
$env:CEBADA_SOURCE_ROOT = "D:\datos\cebada"
python .\cebada_datos_baseline_1\scripts\run_pipeline.py
```

El lanzador ejecuta los scripts en orden, verifica las fuentes principales y detiene el proceso ante errores. `data_processed/`, `reports/` y `figures/` se ubican junto a este README. El paso 04 es exploratorio y no forma parte de la cadena.

El pipeline vuelve a crear las features parcelarias y longitudinales, además de los resultados experimentales, usando las fuentes originales. Las tablas existentes en `data_processed/` son derivados/cachés, no sustituyen las fuentes.

## Decisiones clave

- Las features satelitales resumen observaciones de 2025; se identifican como antecedentes, no como observaciones de 2026. La fuente de las etiquetas y particiones es `ID_area_rendimiento_70_30_Reto_AgroCebada.csv`: la tabla incluye rendimiento para `ENTRENAMIENTO`, no para `PREDICCION`, y no especifica el ciclo del rendimiento. Los sensores no se mezclan (`s2_`, `ls_`, `pl_`).
- Nubes: Sentinel-2 y Landsat ≤ 5 %, Planet ≤ 20 % (con nubes parciales el NDVI de Sentinel-2 queda sesgado a la baja). Nubosidad = 100 % significa índices vacíos.
- Clima de celdas de 5 km: las 197 parcelas caen en 25 celdas. Sirve como descriptor de zona.
- Validación: 5 particiones × 5 repeticiones + 6 bloques espaciales. Lo que usa el rendimiento de otras parcelas se calcula solo con el entrenamiento de cada partición.

## Advertencias

1. Casi toda la señal es geográfica: la media por municipio da RMSE 0.52 t/ha, el mejor modelo solo satelital (Baseline 1) 0.64, y ninguno con satélite mejora de forma medible el prior por municipio.
2. 45 de 138 rendimientos son múltiplos de 0.5 (Calpulalpan, Singuilucan, otros); parece otra fuente de captura. Hipótesis sin confirmar.
3. Con n = 138, diferencias de 0.01–0.02 en RMSE son ruido.
4. `baseline_predicciones_59_parcelas.csv` es un punto de partida para probar la interfaz de evaluación, no una entrega final.

Las métricas de este README provienen de experimentos previos. No deben compararse directamente con nuevas métricas hasta compartir etiquetas, parcelas y particiones espaciales.
