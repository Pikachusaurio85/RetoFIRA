# Informe ejecutivo — AgroCebada

## Resumen

- Se analizaron 155,470 filas de la base ML existente y 138 parcelas con rendimiento etiquetado.
- El identificador `ID_POLIGONO` se trató como clave de parcela: sus capturas repetidas son longitudinales, no duplicados que deban eliminarse.
- El análisis 2022–2025 describe variables satelitales BASIC, precipitación, temperatura y topografía; no se calcularon métricas históricas de rendimiento porque no hay etiquetas de rendimiento por año.
- No hay observaciones de 2026. La fuente canónica de partición y etiquetas es `ID_area_rendimiento_70_30_Reto_AgroCebada.csv`: las parcelas `ENTRENAMIENTO` tienen rendimiento etiquetado y las de `PREDICCION` no. Esa tabla no identifica el ciclo del rendimiento.
- El modelo seleccionado por validación fue **CatBoost**. Evaluación espacial: entrenamiento 97 parcelas, validación 20, prueba 21; reserva externa 59.
- Estado de transferencia a 2026: **TRANSFERENCIA_NO_RESPALDADA: R2_validacion_negativo; R2_test_espacial_negativo; sesgo_test_espacial_distinto_de_cero**.
- MAE/RMSE/R² en validación: 0.625 / 0.754 t/ha / -0.323; línea base MAE 0.922.
- MAE/RMSE/R² en test espacial: 0.494 / 0.624 t/ha / -1.200; línea base MAE 0.878.
- Sesgo medio test (observado menos estimado): 0.471 t/ha; IC95 aproximado por t de Student 0.281 a 0.662 t/ha.

## Objetivo y limitaciones

Las etiquetas disponibles (n=138) varían de 2.00 a 5.59 t/ha, con media 3.96 y mediana 4.34. El año/ciclo de esos rendimientos no está indicado en la tabla de asignación. La evaluación espacial permite revisar el comportamiento en parcelas etiquetadas retenidas; no es una validación temporal ni mide el error de una transferencia a 2026.

La base procesada contiene múltiples observaciones por parcela. Para regresión se consolidaron características históricas a una fila por parcela mediante promedio de las medias anuales disponibles hasta 2025, y se agregaron elevación/pendiente como medias zonales. `ID_POLIGONO` no es predictor. Se evaluó una separación espacial contigua por bandas en el eje principal de centros de bounding box; con una muestra pequeña, esta evaluación es exigente pero sigue teniendo incertidumbre.

## Criterio de robustez

La validación espacial y/o el diagnóstico de cobertura no respaldaron la transferencia. El archivo de reserva conserva las referencias de línea base, pero deja vacías las estimaciones de modelo para no comunicar valores no respaldados como predicciones fiables. Criterios que no se cumplieron: R2_validacion_negativo, R2_test_espacial_negativo, sesgo_test_espacial_distinto_de_cero.

La partición final de 59 parcelas nunca participó en ajuste ni evaluación: no tiene etiquetas. La prueba espacial compara el modelo seleccionado en validación con una línea base de media. Además, se compara el porcentaje de predictores fuera del rango de ajuste y la fracción faltante entre test y reserva. Estas comprobaciones no prueban causalidad ni garantizan éxito fuera del dominio estudiado.

## Análisis descriptivo histórico

Las gráficas de índices BASIC, clima mensual y topografía presentan resúmenes descriptivos por parcela/año. Temperatura y precipitación tienen cobertura mensual 2022–2025. Elevación y pendiente son atributos estáticos; no son tendencias anuales. Las asociaciones entre topografía y rendimiento etiquetado son descriptivas y no causales.

## Archivos principales

- `datos_agrocebada_organizados.xlsx`: datos originales procesados reorganizados, diccionario y descripción de clases.
- `tabla_modelado_parcelaria.csv`: una fila por parcela, predictores con observaciones hasta 2025 y rendimiento etiquetado según la tabla de asignación.
- `datos_reorganizados_todas_observaciones.csv`: todas las observaciones reordenadas por clase, sin columnas separadoras para lectura automatizada.
- `importancia_variables_evaluacion_espacial.csv`: importancia por permutación en el bloque espacial de prueba; es un diagnóstico del modelo, no causalidad ni prueba de transferencia fiable.
- `metricas_evaluacion_transferencia_objetivo_2026.csv`, `predicciones_prueba_espacial_etiquetadas.csv` y `diagnostico_transferencia.csv`.
- `predicciones_rendimiento_transferencia_objetivo_2026.csv`: resultados para el horizonte objetivo 2026, separados de las observaciones; estará vacío el campo estimado si la transferencia no superó el criterio declarado.
- `graficas/`: gráficas descriptivas y de evaluación/modelado.
- `originales/` y `trazabilidad_archivos.csv`: copias y huellas SHA-256 de las fuentes tabulares principales.

## Reproducibilidad

Desde `desarrollo`, instalar `requirements.txt` y ejecutar `python main.py`. La semilla de los modelos es 2026. La extracción de topografía usa geometrías del shapefile, reproyección al CRS de cada ráster, píxeles que intersectan el polígono y media de celdas válidas. Los datos fuente no se modifican.
