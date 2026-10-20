#Redacción del informe ejecutivo del análisis

from __future__ import annotations
from typing import Any
import numpy as np
from analisis.config import OUTPUT, SEED

__all__ = ["_write_report"]

def _write_report(
    audit: dict[str, Any],
    model: dict[str, Any],
    target_values: dict[str, float],
) -> None:
    yields = np.asarray(list(target_values.values()), dtype=float)
    status_text = (
        "La validación interna espacial y el diagnóstico de cobertura respaldaron la transferencia "
        "según los criterios predefinidos; las estimaciones siguen sin validación con verdad-terreno 2026."
        if model["robust"]
        else "La validación espacial y/o el diagnóstico de cobertura no respaldaron la transferencia. "
        "El archivo de reserva conserva las referencias de línea base, pero deja vacías las estimaciones "
        "de modelo para no comunicar valores no respaldados como predicciones fiables. "
        f"Criterios que no se cumplieron: {', '.join(model['failed_criteria'])}."
    )
    report = f"""# Informe ejecutivo — AgroCebada

## Resumen

- Se analizaron {audit['row_count']:,} filas de la base ML existente y {len(target_values)} parcelas con rendimiento etiquetado.
- El identificador `ID_POLIGONO` se trató como clave de parcela: sus capturas repetidas son longitudinales, no duplicados que deban eliminarse.
- El análisis 2022–2025 describe variables satelitales BASIC, precipitación, temperatura y topografía; no se calcularon métricas históricas de rendimiento porque no hay etiquetas de rendimiento por año.
- No hay observaciones de 2026. La fuente canónica de partición y etiquetas es `ID_area_rendimiento_70_30_Reto_AgroCebada.csv`: las parcelas `ENTRENAMIENTO` tienen rendimiento etiquetado y las de `PREDICCION` no. Esa tabla no identifica el ciclo del rendimiento.
- El modelo seleccionado por validación fue **{model['selected_model']}**. Evaluación espacial: entrenamiento {model['train_count']} parcelas, validación {model['validation_count']}, prueba {model['test_count']}; reserva externa {model['prediction_count']}.
- Estado de transferencia a 2026: **{model['status']}**.
- MAE/RMSE/R² en validación: {model['validation_metrics']['MAE_t_ha']:.3f} / {model['validation_metrics']['RMSE_t_ha']:.3f} t/ha / {model['validation_metrics']['R2']:.3f}; línea base MAE {model['baseline_validation_metrics']['MAE_t_ha']:.3f}.
- MAE/RMSE/R² en test espacial: {model['test_metrics']['MAE_t_ha']:.3f} / {model['test_metrics']['RMSE_t_ha']:.3f} t/ha / {model['test_metrics']['R2']:.3f}; línea base MAE {model['baseline_test_metrics']['MAE_t_ha']:.3f}.
- Sesgo medio test (observado menos estimado): {model['test_bias']:.3f} t/ha; IC95 aproximado por t de Student {model['test_bias_interval'][0]:.3f} a {model['test_bias_interval'][1]:.3f} t/ha.

## Objetivo y limitaciones

Las etiquetas disponibles (n={len(yields)}) varían de {yields.min():.2f} a {yields.max():.2f} t/ha, con media {yields.mean():.2f} y mediana {np.median(yields):.2f}. El año/ciclo de esos rendimientos no está indicado en la tabla de asignación. La evaluación espacial permite revisar el comportamiento en parcelas etiquetadas retenidas; no es una validación temporal ni mide el error de una transferencia a 2026.

La base procesada contiene múltiples observaciones por parcela. Para regresión se consolidaron características históricas a una fila por parcela mediante promedio de las medias anuales disponibles hasta 2025, y se agregaron elevación/pendiente como medias zonales. `ID_POLIGONO` no es predictor. Se evaluó una separación espacial contigua por bandas en el eje principal de centros de bounding box; con una muestra pequeña, esta evaluación es exigente pero sigue teniendo incertidumbre.

## Criterio de robustez

{status_text}

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

Desde `desarrollo`, instalar `requirements.txt` y ejecutar `python main.py`. La semilla de los modelos es {SEED}. La extracción de topografía usa geometrías del shapefile, reproyección al CRS de cada ráster, píxeles que intersectan el polígono y media de celdas válidas. Los datos fuente no se modifican.
"""
    (OUTPUT / "informe_ejecutivo_transferencia_objetivo_2026.md").write_text(
        report, encoding="utf-8"
    )