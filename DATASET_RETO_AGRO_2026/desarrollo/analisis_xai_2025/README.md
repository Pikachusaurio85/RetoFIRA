# AgroCebada 2025: auditoría de conceptos XAI

Análisis paralelo y reproducible de cobertura, calidad, curvas espectrales, prototipos y características para XAI. No modifica el pipeline original ni usa rendimiento para crear curvas o clusters.

## Ejecución

Desde `desarrollo`, con las fuentes y productos parcelarios ya disponibles:

```powershell
..\.venv\Scripts\python.exe .\analisis_xai_2025\analizar.py
```

Si se ejecuta desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe .\desarrollo\analisis_xai_2025\analizar.py
```

El análisis usa las fuentes originales BASIC, PRO y de asignación/etiqueta, además de `parcelas_base.csv` y `clima_mensual_parcela.csv` existentes en `cebada_datos_baseline_1\data_processed`. No regenera, escribe ni modifica esos datos o el pipeline original. Las salidas regenerables están en `resultados\`; `manifest.json` guarda versiones y hashes SHA-256.

La ejecución necesita las dependencias ya usadas por el proyecto: pandas, numpy, scipy, scikit-learn y matplotlib. Los filtros de nubosidad son Sentinel-2/Landsat <=5 % y Planet <=20 %; las ventanas fenológicas son cortes calendáricos exploratorios, no etiquetas de etapa.

## Productos

- [INFORME.md](./INFORME.md): metodología, resultados, incertidumbre y decisiones XAI.
- [analizar.py](./analizar.py): código de extracción y experimentos.
- `resultados/observaciones_2025_trazables.csv`: observaciones 2025 con archivo fuente, fila original, fecha, parcela, sensor, señal y calidad.
- `resultados/auditoria_etiquetas_por_parcela.csv`, `clima_mensual_trazable_2025.csv` y `topografia_por_parcela.csv`: trazabilidad mensual/parcelaria de las otras fuentes.
- `resultados/calidad_*.csv`, `faltantes_por_variable_sensor.csv`, `integridad_fuentes.csv`: cobertura, faltantes y controles por fuente, parcela, municipio, sensor, mes y etapa de calendario.
- `resultados/indices_resumen_etapa_2025.csv`: índices agrupados por ventana y sensor.
- `resultados/comparacion_sensores_pares_temporales.csv` y `pares_sensor_ndvi_5dias.csv`: contraste pareado S2-Planet.
- `resultados/caracteristicas_temporales_parcela_2025.csv` y `estadisticas_agregadas_parcela_2025.csv`: características sin etiquetas para reutilización.
- `resultados/clusters_parcela_s2_ndvi.csv`, `prototipos_clusters_s2_ndvi.csv`, `estabilidad_clusters.csv` y `sensibilidad_numero_clusters.csv`: asignaciones y estabilidad descriptivas, sin rendimiento.
- `resultados/comparacion_temporal_vs_agregado_folds.csv` y `..._predicciones.csv`: métricas y predicciones, con folds espaciales compartidos.
- `resultados/catalogo_conceptos.csv`: catálogo estructurado de conceptos, evidencia y validación requerida.
- `resultados/graficas/`: curvas, índices, cobertura y prototipos.

Las variables de rendimiento solo intervienen en la comparación supervisada de características con validación espacial. No se usan para definir conceptos, curvas, clusters o prototipos.
