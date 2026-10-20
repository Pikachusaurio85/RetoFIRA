#Gráficas descriptivas e interpretaciones de resultados

from __future__ import annotations
import csv
from typing import Any
import numpy as np
from matplotlib import pyplot as plt
from analisis.config import OUTPUT, YEARS

__all__ = [
    "_write_chart_interpretations",
    "_write_descriptive_outputs",
    "_write_test_residual_plot",
]

def _write_descriptive_outputs(
    annual_features: dict[tuple[str, int], dict[str, float]],
    topography: dict[str, dict[str, float]],
    target_values: dict[str, float],
) -> None:
    annual_rows: list[dict[str, Any]] = []
    for year in YEARS:
        values_for_year = [
            values for (_, obs_year), values in annual_features.items()
            if obs_year == year
        ]
        feature_names = sorted(
            {name for values in values_for_year for name in values}
        )
        for feature in feature_names:
            observed = [
                values[feature]
                for values in values_for_year
                if feature in values
            ]
            if not observed:
                continue
            annual_rows.append(
                {
                    "anio": year,
                    "variable": feature,
                    "n_parcelas": len(observed),
                    "media_por_parcela": float(np.mean(observed)),
                    "mediana_por_parcela": float(np.median(observed)),
                    "desviacion_estandar": float(np.std(observed, ddof=1))
                    if len(observed) > 1
                    else "",
                    "p05": float(np.quantile(observed, 0.05)),
                    "p95": float(np.quantile(observed, 0.95)),
                }
            )
    if annual_rows:
        with (OUTPUT / "resumen_descriptivo_anual.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as output:
            writer = csv.DictWriter(output, fieldnames=list(annual_rows[0]))
            writer.writeheader()
            writer.writerows(annual_rows)

    plots_folder = OUTPUT / "graficas"
    plots_folder.mkdir(parents=True, exist_ok=True)
    index_names = (
        "BASICO__ndvi_promedio",
        "BASICO__evi_promedio",
        "BASICO__lai_promedio",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for feature in index_names:
        means = []
        for year in YEARS:
            matching = [
                values[feature]
                for (_, obs_year), values in annual_features.items()
                if obs_year == year and feature in values
            ]
            means.append(float(np.mean(matching)) if matching else np.nan)
        if np.isfinite(means).any():
            ax.plot(YEARS, means, marker="o", linewidth=2, label=feature.split("__")[1].upper())
            plotted = True
    if plotted:
        ax.set(
            title="Índices satelitales BASIC por año",
            xlabel="Año de captura",
            ylabel="Media del índice (sin unidad física)",
        )
        ax.legend(title="Variable")
        ax.grid(alpha=0.25)
        fig.text(
            0.01,
            0.01,
            "Fuente: Dataset Básico 2022–2025. Media de los promedios por parcela-año; "
            "descriptivo, no representa rendimiento ni causalidad.",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(plots_folder / "indices_basicos_2022_2025.png", dpi=160)
    plt.close(fig)

    climate_specs = [
        ("CLIMA__precipitacion_mm", "Precipitación acumulada (mm/mes)"),
        ("CLIMA__tmin_c", "Temperatura mínima promedio (°C)"),
        ("CLIMA__tmax_c", "Temperatura máxima promedio (°C)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for ax, (prefix, label) in zip(axes, climate_specs):
        for year in YEARS:
            series = []
            for month in range(1, 13):
                feature = f"{prefix}_{month:02d}"
                matching = [
                    values[feature]
                    for (_, obs_year), values in annual_features.items()
                    if obs_year == year and feature in values
                ]
                series.append(float(np.mean(matching)) if matching else np.nan)
            if np.isfinite(series).any():
                ax.plot(range(1, 13), series, marker=".", label=str(year))
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(title="Año", ncol=4)
    axes[0].set_title("Clima mensual medio por parcela, 2022–2025")
    axes[-1].set_xlabel("Mes")
    fig.text(
        0.01,
        0.01,
        "Fuentes: CHIRPS (precipitación) y CHIRTS-ERA5 (temperaturas). "
        "Media espacial descriptiva; no constituye pronóstico.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(plots_folder / "clima_mensual_2022_2025.png", dpi=160)
    plt.close(fig)

    labels = sorted(target_values)
    yields = [target_values[polygon_id] for polygon_id in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(yields, bins="auto", color="#4C78A8", edgecolor="white")
    ax.set(
        title="Distribución del rendimiento etiquetado",
        xlabel="Rendimiento etiquetado observado (t/ha)",
        ylabel="Número de parcelas",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.text(
        0.01,
        0.01,
        "n=138 parcelas ENTRENAMIENTO según la tabla de asignaciones. El archivo no indica "
        "el ciclo al que corresponde el rendimiento; no hay datos observados de 2026.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(plots_folder / "distribucion_rendimiento_etiquetado.png", dpi=160)
    plt.close(fig)

    elevation_name = "TOPOGRAFIA__elevacion_media_m_snm"
    slope_name = "TOPOGRAFIA__pendiente_media_grados"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    elevation = [
        item[elevation_name] for item in topography.values() if elevation_name in item
    ]
    slope = [item[slope_name] for item in topography.values() if slope_name in item]
    axes[0].hist(elevation, bins="auto", color="#59A14F", edgecolor="white")
    axes[0].set(xlabel="Elevación media (m s.n.m.)", ylabel="Parcelas", title="Elevación")
    axes[1].hist(slope, bins="auto", color="#F28E2B", edgecolor="white")
    axes[1].set(xlabel="Pendiente media (grados)", ylabel="Parcelas", title="Pendiente")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Distribución topográfica por parcela (atributos estáticos)")
    fig.text(
        0.01,
        0.01,
        "Fuente: rásteres INEGI CEM4 de 120 m; media zonal por polígono. "
        "Descriptivo, no implica causalidad.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(plots_folder / "distribucion_topografia.png", dpi=160)
    plt.close(fig)

    if elevation and slope:
        fig, ax = plt.subplots(figsize=(8, 5))
        points = ax.scatter(
            [topography[key][elevation_name] for key in labels if key in topography],
            [target_values[key] for key in labels if key in topography],
            c=[topography[key][slope_name] for key in labels if key in topography],
            cmap="viridis",
            alpha=0.8,
        )
        ax.set(
            title="Rendimiento etiquetado y elevación por parcela",
            xlabel="Elevación media de parcela (m s.n.m.)",
            ylabel="Rendimiento etiquetado (t/ha)",
        )
        fig.colorbar(points, ax=ax, label="Pendiente media (grados)")
        ax.grid(alpha=0.25)
        fig.text(
            0.01,
            0.01,
            "Asociación descriptiva en parcelas etiquetadas; no implica causalidad "
            "ni validación de predicción 2026.",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(plots_folder / "rendimiento_y_topografia.png", dpi=160)
        plt.close(fig)

def _write_chart_interpretations(
    annual_features: dict[tuple[str, int], dict[str, float]],
    topography: dict[str, dict[str, float]],
    target_values: dict[str, float],
    model: dict[str, Any],
) -> None:
    rows: list[dict[str, str]] = []

    def add(
        chart: str,
        observation: str,
        result: str,
        confidence: str,
        limitation: str,
        relevance: str,
    ) -> None:
        rows.append(
            {
                "grafica": chart,
                "que_observamos": observation,
                "resultado_principal": result,
                "confiabilidad": confidence,
                "limitaciones": limitation,
                "relevancia_proyecto": relevance,
            }
        )

    def yearly_mean(variable: str, year: int) -> float | None:
        found = [
            values[variable]
            for (_, obs_year), values in annual_features.items()
            if obs_year == year and variable in values
        ]
        return float(np.mean(found)) if found else None

    ndvi_2022 = yearly_mean("BASICO__ndvi_promedio", 2022)
    ndvi_2025 = yearly_mean("BASICO__ndvi_promedio", 2025)
    if ndvi_2022 is not None and ndvi_2025 is not None:
        ndvi_change = ndvi_2025 - ndvi_2022
        ndvi_direction = "aumentó" if ndvi_change > 0 else "disminuyó"
        ndvi_result = (
            f"El promedio parcelario NDVI {ndvi_direction} {abs(ndvi_change):.3f} "
            "entre 2022 y 2025."
        )
    else:
        ndvi_result = "No hubo cobertura suficiente para calcular NDVI en ambos años extremos."
    add(
        "indices_basicos_2022_2025.png",
        "Medias por parcela-año de NDVI, EVI y LAI del Dataset Básico.",
        ndvi_result,
        "Descriptiva; cada parcela-año contribuye una media.",
        "La cantidad de capturas/sensores puede variar; no hay etiquetas históricas "
        "de rendimiento para interpretar la serie como desempeño.",
        "Describe la señal satelital disponible antes de construir predictores de 2026.",
    )

    precipitation_year_means = []
    for year in YEARS:
        monthly = [
            yearly_mean(f"CLIMA__precipitacion_mm_{month:02d}", year)
            for month in range(1, 13)
        ]
        present = [value for value in monthly if value is not None]
        if present:
            precipitation_year_means.append(
                (year, float(np.mean(present)))
            )
    if precipitation_year_means:
        driest = min(precipitation_year_means, key=lambda item: item[1])
        wettest = max(precipitation_year_means, key=lambda item: item[1])
        climate_result = (
            f"El promedio espacial de la media mensual varió entre "
            f"{driest[1]:.1f} mm/mes ({driest[0]}) y "
            f"{wettest[1]:.1f} mm/mes ({wettest[0]})."
        )
    else:
        climate_result = "No se encontró cobertura climática suficiente para comparar años."
    add(
        "clima_mensual_2022_2025.png",
        "Precipitación acumulada y temperaturas mínimas/máximas mensuales, resumidas por parcela.",
        climate_result,
        "Descriptiva; unidades y productos se contrastaron con inventarios.",
        "Media de las parcelas disponibles; no equivale a pronóstico y las estaciones no "
        "se derivan sin calendario agrícola documentado.",
        "Permite reconocer variabilidad climática histórica que puede incorporarse como contexto.",
    )

    elevation = [
        values["TOPOGRAFIA__elevacion_media_m_snm"]
        for values in topography.values()
        if "TOPOGRAFIA__elevacion_media_m_snm" in values
    ]
    slope = [
        values["TOPOGRAFIA__pendiente_media_grados"]
        for values in topography.values()
        if "TOPOGRAFIA__pendiente_media_grados" in values
    ]
    topo_result = (
        f"Se resumieron {len(elevation)} parcelas; elevación media "
        f"{np.mean(elevation):.0f} m s.n.m. y pendiente media {np.mean(slope):.1f}°."
        if elevation and slope
        else "No hubo cobertura topográfica completa."
    )
    add(
        "distribucion_topografia.png",
        "Distribución de la elevación y pendiente medias extraídas de los rásteres INEGI.",
        topo_result,
        "Extracción zonal reproducible usando píxeles válidos que intersectan el polígono.",
        "Atributos estáticos de rásteres de 120 m; una media zonal pierde heterogeneidad interna.",
        "Proporciona contexto espacial estático para describir y modelar parcelas.",
    )

    y_values = list(target_values.values())
    add(
        "distribucion_rendimiento_etiquetado.png",
        "Frecuencia del rendimiento de las parcelas con etiqueta disponible.",
        (
            f"n={len(y_values)}, mediana {np.median(y_values):.2f} t/ha, "
            f"rango {min(y_values):.2f}–{max(y_values):.2f} t/ha."
            if y_values
            else "No hay etiquetas numéricas."
        ),
        "Resumen directo de las etiquetas disponibles.",
        "Las etiquetas proceden de la tabla de asignación; esa fuente no indica el ciclo "
        "del rendimiento. No hay observaciones de 2026.",
        "Revela el rango de respuesta etiquetada disponible para la evaluación espacial.",
    )

    add(
        "mae_modelo_vs_baseline.png",
        "MAE del modelo elegido en validación y test espacial frente a la media como línea base.",
        (
            f"MAE validación: {model['validation_metrics']['MAE_t_ha']:.3f} vs "
            f"{model['baseline_validation_metrics']['MAE_t_ha']:.3f} t/ha; "
            f"MAE test: {model['test_metrics']['MAE_t_ha']:.3f} vs "
            f"{model['baseline_test_metrics']['MAE_t_ha']:.3f} t/ha. "
            "La transferencia superó todos los criterios."
            if model["robust"]
            else (
                f"MAE validación: {model['validation_metrics']['MAE_t_ha']:.3f} vs "
                f"{model['baseline_validation_metrics']['MAE_t_ha']:.3f} t/ha; "
                f"MAE test: {model['test_metrics']['MAE_t_ha']:.3f} vs "
                f"{model['baseline_test_metrics']['MAE_t_ha']:.3f} t/ha; "
                f"R² test={model['test_metrics']['R2']:.3f}. "
                f"No respaldan transferencia: {', '.join(model['failed_criteria'])}."
            )
        ),
        (
            "Test espacial separado de ajuste y selección; tamaño de test limitado. "
            f"IC95 del sesgo medio test: {model['test_bias_interval'][0]:.3f} a "
            f"{model['test_bias_interval'][1]:.3f} t/ha."
        ),
        "Es una evaluación de generalización espacial usando las etiquetas disponibles, "
        "no un backtest temporal ni validación del rendimiento objetivo 2026.",
        "Determina si existe evidencia suficiente para extrapolar a las parcelas reservadas.",
    )
    add(
        "observado_vs_estimado_test.png",
        "Rendimientos etiquetados frente a predicciones en parcelas espaciales retenidas.",
        "La distancia a la diagonal representa el error por parcela; consulte las métricas MAE/RMSE.",
        "Evaluación interna espacial; la prueba no fue usada para ajustar los hiperparámetros.",
        "El tamaño de muestra es reducido y el ciclo de las etiquetas no está indicado en la fuente.",
        "Permite ver sesgos de sobre/subestimación que un promedio de error puede ocultar.",
    )
    residual_path = OUTPUT / "graficas" / "error_prediccion_test.png"
    add(
        residual_path.name,
        "Distribución de residuos observado menos estimado en el test espacial.",
        "Residuos centrados alrededor de cero indican menor sesgo medio; la dispersión refleja incertidumbre.",
        "Calculada en parcelas no usadas en ajuste.",
        "El histograma de una prueba pequeña no establece distribución de errores futura.",
        "Ayuda a identificar errores sistemáticos antes de cualquier uso operativo.",
    )
    if model["feature_importances"]:
        add(
            "importancia_variables_test.png",
            "Aumento de MAE al permutar cada predictor en test.",
            "Las variables con mayor aumento alteran más el error predictivo del modelo.",
            "Permutación evaluada en test espacial.",
            "Diagnóstico de un único bloque espacial pequeño; no implica causalidad ni prueba transferencia estable. Importancias pueden repartirse entre predictores correlacionados.",
            "Ayuda a inspeccionar qué señales usa el modelo y a orientar la recopilación de datos.",
        )
    if model["robust"]:
        add(
            "estimaciones_transferencia_objetivo_2026.png",
            "Ubicación de las estimaciones en las 59 parcelas reservadas.",
            "Cada punto es una parcela; el color muestra rendimiento estimado (t/ha).",
            "Emitida solo al superar los criterios de transferencia definidos.",
            "No hay etiquetas reales 2026 en el conjunto para calcular su error.",
            "Comunica la distribución espacial de las estimaciones no observadas.",
        )

    with (OUTPUT / "interpretacion_graficas.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def _write_test_residual_plot(actual: np.ndarray, predicted: np.ndarray) -> None:
    residuals = actual - predicted
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins="auto", color="#E15759", edgecolor="white")
    ax.axvline(0, color="#333333", linestyle="--", label="Error cero")
    ax.set(
        title="Errores de predicción en test espacial",
        xlabel="Residuo: observado − estimado (t/ha; objetivo atribuido a 2025)",
        ylabel="Número de parcelas",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.text(
        0.01,
        0.01,
        "Test espacial interno; residuos positivos indican subestimación y negativos sobreestimación.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUTPUT / "graficas" / "error_prediccion_test.png", dpi=160)
    plt.close(fig)