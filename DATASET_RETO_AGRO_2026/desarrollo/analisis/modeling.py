#Evaluación espacial, selección y estimación del modelo

from __future__ import annotations
import csv
import json
import warnings
from typing import Any
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from scipy.stats import t as student_t
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
from analisis.config import OUTPUT, SEED, TARGET
from analisis.data import _number

__all__ = ["_train_evaluate_predict"]

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

def _spatial_split(
    polygon_ids: list[str],
    centroids: dict[str, tuple[float, float]],
) -> tuple[list[str], list[str], list[str], dict[str, float]]:
    #Separa parcelas por posición espacial para reducir fuga entre vecinas
    missing = set(polygon_ids) - centroids.keys()
    if missing:
        raise ValueError(
            "No hay coordenadas para parcelas etiquetadas: " + ", ".join(sorted(missing))
        )
    coordinates = np.asarray([centroids[polygon_id] for polygon_id in polygon_ids])
    centered = coordinates - coordinates.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    projections = centered @ axis
    ordered = sorted(zip(polygon_ids, projections), key=lambda item: item[1])
    n = len(ordered)
    train_end = int(round(n * 0.70))
    validation_end = int(round(n * 0.85))
    if train_end < 2 or validation_end <= train_end or validation_end >= n:
        raise ValueError(
            f"No hay parcelas etiquetadas suficientes para split espacial 70/15/15: n={n}."
        )
    train = [item[0] for item in ordered[:train_end]]
    validation = [item[0] for item in ordered[train_end:validation_end]]
    test = [item[0] for item in ordered[validation_end:]]
    positions = {polygon_id: float(value) for polygon_id, value in ordered}
    return train, validation, test, positions

def _fit_preprocessor(
    x_train: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    #Ajusta selección e imputación usando solo filas de entrenamiento.
    missing_rates = np.mean(~np.isfinite(x_train), axis=0)
    keep = missing_rates <= 0.50
    if not keep.any():
        raise ValueError("Todas las variables exceden 50% de faltantes en entrenamiento.")
    kept_names = [
        name for name, keep_feature in zip(feature_names, keep) if keep_feature
    ]
    train_kept = x_train[:, keep]
    medians = np.nanmedian(
        np.where(np.isfinite(train_kept), train_kept, np.nan), axis=0
    )
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(train_kept), train_kept, medians)
    variable = np.nanstd(filled, axis=0) > 1e-12
    if not variable.any():
        raise ValueError("No hay predictores variables después de filtrar faltantes.")
    kept_names = [name for name, keep_feature in zip(kept_names, variable) if keep_feature]
    return filled[:, variable], kept_names, keep, variable, medians

def _apply_preprocessor(
    matrix: np.ndarray,
    keep: np.ndarray,
    variable: np.ndarray,
    medians: np.ndarray,
) -> np.ndarray:
    selected = matrix[:, keep]
    filled = np.where(np.isfinite(selected), selected, medians)
    return filled[:, variable]

def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | str]:
    return {
        "MAE_t_ha": float(mean_absolute_error(actual, predicted)),
        "RMSE_t_ha": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else "",
    }

def _mean_residual_interval(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float, float]:
    residuals = actual - predicted
    if len(residuals) < 2:
        return float(np.mean(residuals)), float("-inf"), float("inf")
    standard_error = float(np.std(residuals, ddof=1) / np.sqrt(len(residuals)))
    critical_value = float(student_t.ppf(0.975, df=len(residuals) - 1))
    mean_error = float(np.mean(residuals))
    margin = critical_value * standard_error
    return mean_error, mean_error - margin, mean_error + margin

def _estimators() -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "CatBoost",
            {
                "iterations": 200,
                "depth": depth,
                "learning_rate": rate,
                "l2_leaf_reg": 5.0,
                "loss_function": "RMSE",
                "verbose": False,
                "random_seed": SEED,
                "thread_count": 1,
            },
        )
        for depth in (3, 4)
        for rate in (0.03, 0.05)
    ] + [
        (
            "XGBoost",
            {
                "n_estimators": trees,
                "max_depth": depth,
                "learning_rate": 0.04,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 5.0,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "n_jobs": 1,
                "random_state": SEED,
                "verbosity": 0,
            },
        )
        for depth in (2, 3)
        for trees in (100, 200)
    ] + [
        (
            "LightGBM",
            {
                "n_estimators": trees,
                "num_leaves": leaves,
                "max_depth": 3,
                "learning_rate": 0.04,
                "reg_lambda": 5.0,
                "verbosity": -1,
                "n_jobs": 1,
                "random_state": SEED,
            },
        )
        for leaves in (5, 8)
        for trees in (100, 200)
    ]

def _create_model(name: str, parameters: dict[str, Any]) -> Any:
    if name == "CatBoost":
        return CatBoostRegressor(**parameters)
    if name == "XGBoost":
        return XGBRegressor(**parameters)
    if name == "LightGBM":
        return LGBMRegressor(**parameters)
    raise ValueError(f"Modelo desconocido: {name}")

def _coverage_diagnostic(
    reference: np.ndarray,
    comparison: np.ndarray,
) -> tuple[float, float]:
    present = np.isfinite(comparison)
    denominator = int(present.sum())
    if denominator == 0:
        return 1.0, 0.0
    selected_ref = np.where(np.isfinite(reference), reference, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        min_values = np.nanmin(selected_ref, axis=0)
        max_values = np.nanmax(selected_ref, axis=0)
        medians = np.nanmedian(selected_ref, axis=0)
    min_values = np.where(np.isfinite(min_values), min_values, medians)
    max_values = np.where(np.isfinite(max_values), max_values, medians)
    out_of_range = present & (
        (comparison < min_values) | (comparison > max_values)
    )
    return float(out_of_range.sum() / denominator), float((~present).mean())

def _train_evaluate_predict(
    feature_names: list[str],
    parcel_data: dict[str, dict[str, Any]],
    topography: dict[str, dict[str, float]],
    centroids: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Compara línea base y modelos con validación y prueba espaciales."""
    for polygon_id, values in parcel_data.items():
        values.update(topography.get(polygon_id, {}))
    feature_names = [
        name
        for name in feature_names
        if name != "AREA_HA"
    ] + [
        "AREA_HA",
        "TOPOGRAFIA__elevacion_media_m_snm",
        "TOPOGRAFIA__pendiente_media_grados",
    ]
    feature_names = list(dict.fromkeys(feature_names))

    training_ids = sorted(
        polygon_id
        for polygon_id, values in parcel_data.items()
        if values.get("PARTICION") == "ENTRENAMIENTO"
        and _number(str(values.get(TARGET, ""))) is not None
    )
    prediction_ids = sorted(
        polygon_id
        for polygon_id, values in parcel_data.items()
        if values.get("PARTICION") == "PREDICCION"
    )
    if len(training_ids) < 30:
        raise ValueError(
            f"Solo hay {len(training_ids)} parcelas etiquetadas; "
            "no es suficiente para una evaluación espacial 70/15/15."
        )
    x_all = np.asarray(
        [
            [
                _number(str(parcel_data[polygon_id].get(name, "")))
                if _number(str(parcel_data[polygon_id].get(name, ""))) is not None
                else np.nan
                for name in feature_names
            ]
            for polygon_id in training_ids
        ],
        dtype=float,
    )
    y_all = np.asarray(
        [_number(str(parcel_data[polygon_id][TARGET])) for polygon_id in training_ids],
        dtype=float,
    )
    train_ids, validation_ids, test_ids, axis_positions = _spatial_split(
        training_ids, centroids
    )
    index = {polygon_id: position for position, polygon_id in enumerate(training_ids)}
    train_indices = np.asarray([index[item] for item in train_ids])
    validation_indices = np.asarray([index[item] for item in validation_ids])
    test_indices = np.asarray([index[item] for item in test_ids])

    x_train_raw = x_all[train_indices]
    x_train, model_feature_names, keep, variable, medians = _fit_preprocessor(
        x_train_raw, feature_names
    )
    x_validation = _apply_preprocessor(
        x_all[validation_indices], keep, variable, medians
    )
    y_train = y_all[train_indices]
    y_validation = y_all[validation_indices]
    baseline_validation = np.full(len(validation_indices), float(np.mean(y_train)))
    baseline_val_metrics = _metrics(y_validation, baseline_validation)
    metric_rows: list[dict[str, Any]] = [
        {
            "modelo": "Línea base (media)",
            "conjunto": "validacion",
            "n_parcelas": len(validation_indices),
            **baseline_val_metrics,
            "parametros": "media de parcelas de entrenamiento",
        }
    ]
    candidates: list[tuple[str, dict[str, Any], Any, float]] = []
    for model_name, parameters in _estimators():
        estimator = _create_model(model_name, parameters)
        estimator.fit(x_train, y_train)
        validation_prediction = estimator.predict(x_validation)
        scores = _metrics(y_validation, validation_prediction)
        metric_rows.append(
            {
                "modelo": model_name,
                "conjunto": "validacion",
                "n_parcelas": len(validation_indices),
                **scores,
                "parametros": json.dumps(parameters, sort_keys=True),
            }
        )
        candidates.append(
            (model_name, parameters, estimator, float(scores["MAE_t_ha"]))
        )

    selected_name, selected_parameters, _, selected_val_mae = min(
        candidates, key=lambda item: item[3]
    )
    selected_validation_estimator = min(candidates, key=lambda item: item[3])[2]
    selected_validation_prediction = selected_validation_estimator.predict(
        x_validation
    )
    selected_validation_metrics = _metrics(
        y_validation, selected_validation_prediction
    )
    validation_bias, validation_bias_low, validation_bias_high = (
        _mean_residual_interval(y_validation, selected_validation_prediction)
    )
    # Test parcels are spatially held out and used once, after validation selected a model.
    development_indices = np.concatenate((train_indices, validation_indices))
    y_development = y_all[development_indices]
    selected_indices = np.flatnonzero(keep)[variable]
    selected_mask = np.zeros(len(feature_names), dtype=bool)
    selected_mask[selected_indices] = True
    development_raw = x_all[development_indices][:, selected_mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        development_medians = np.nanmedian(
            np.where(np.isfinite(development_raw), development_raw, np.nan),
            axis=0,
        )
    development_medians = np.where(
        np.isfinite(development_medians), development_medians, 0.0
    )
    x_development = np.where(
        np.isfinite(development_raw), development_raw, development_medians
    )
    test_raw = x_all[test_indices][:, selected_mask]
    x_test = np.where(np.isfinite(test_raw), test_raw, development_medians)
    selected_for_test = _create_model(selected_name, selected_parameters)
    selected_for_test.fit(x_development, y_development)
    test_prediction = selected_for_test.predict(x_test)
    baseline_test_prediction = np.full(len(test_indices), float(np.mean(y_development)))
    baseline_test_scores = _metrics(y_all[test_indices], baseline_test_prediction)
    test_scores = _metrics(y_all[test_indices], test_prediction)
    test_bias, test_bias_low, test_bias_high = _mean_residual_interval(
        y_all[test_indices], test_prediction
    )
    metric_rows.extend(
        [
            {
                "modelo": "Línea base (media)",
                "conjunto": "prueba_espacial",
                "n_parcelas": len(test_indices),
                **baseline_test_scores,
                "parametros": "media de parcelas de entrenamiento+validacion",
            },
            {
                "modelo": selected_name,
                "conjunto": "prueba_espacial",
                "n_parcelas": len(test_indices),
                **test_scores,
                "parametros": json.dumps(selected_parameters, sort_keys=True),
            },
        ]
    )

    x_prediction_raw = np.asarray(
        [
            [
                _number(str(parcel_data[polygon_id].get(name, "")))
                if _number(str(parcel_data[polygon_id].get(name, ""))) is not None
                else np.nan
                for name in feature_names
            ]
            for polygon_id in prediction_ids
        ],
        dtype=float,
    )
    oob_test, missing_test = _coverage_diagnostic(
        development_raw, test_raw
    )
    prediction_raw = x_prediction_raw[:, selected_mask]
    oob_prediction, missing_prediction = _coverage_diagnostic(
        development_raw, prediction_raw
    )
    validation_mae_improves = selected_val_mae < float(
        baseline_val_metrics["MAE_t_ha"]
    )
    validation_rmse_improves = selected_validation_metrics["RMSE_t_ha"] < float(
        baseline_val_metrics["RMSE_t_ha"]
    )
    test_mae_improves = float(test_scores["MAE_t_ha"]) < float(
        baseline_test_scores["MAE_t_ha"]
    )
    test_rmse_improves = float(test_scores["RMSE_t_ha"]) < float(
        baseline_test_scores["RMSE_t_ha"]
    )
    validation_improves = validation_mae_improves and validation_rmse_improves
    test_improves = test_mae_improves and test_rmse_improves
    prediction_domain_supported = (
        oob_prediction <= oob_test and missing_prediction <= missing_test
    )
    validation_r2_supported = float(selected_validation_metrics["R2"]) >= 0
    test_r2_supported = float(test_scores["R2"]) >= 0
    validation_bias_supported = validation_bias_low <= 0 <= validation_bias_high
    test_bias_supported = test_bias_low <= 0 <= test_bias_high
    failed_criteria = []
    for passed, reason in (
        (validation_mae_improves, "no_mejora_MAE_validacion"),
        (validation_rmse_improves, "no_mejora_RMSE_validacion"),
        (test_mae_improves, "no_mejora_MAE_test"),
        (test_rmse_improves, "no_mejora_RMSE_test"),
        (validation_r2_supported, "R2_validacion_negativo"),
        (test_r2_supported, "R2_test_espacial_negativo"),
        (validation_bias_supported, "sesgo_validacion_distinto_de_cero"),
        (test_bias_supported, "sesgo_test_espacial_distinto_de_cero"),
        (prediction_domain_supported, "cambio_cobertura_fuera_de_soporte"),
    ):
        if not passed:
            failed_criteria.append(reason)
    robust = not failed_criteria
    status = (
        "TRANSFERENCIA_RESPALDADA_POR_VALIDACION_INTERNA"
        if robust
        else "TRANSFERENCIA_NO_RESPALDADA: " + "; ".join(failed_criteria)
    )

    prediction_values: np.ndarray | None = None
    feature_importances: list[dict[str, Any]] = []
    if robust:
        final_raw = x_all[:, selected_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            final_medians = np.nanmedian(
                np.where(np.isfinite(final_raw), final_raw, np.nan), axis=0
            )
        final_medians = np.where(np.isfinite(final_medians), final_medians, 0.0)
        x_final = np.where(np.isfinite(final_raw), final_raw, final_medians)
        x_prediction = np.where(
            np.isfinite(prediction_raw), prediction_raw, final_medians
        )
        final_model = _create_model(selected_name, selected_parameters)
        final_model.fit(x_final, y_all)
        prediction_values = final_model.predict(x_prediction)

    permutation = permutation_importance(
        selected_for_test,
        x_test,
        y_all[test_indices],
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=SEED,
        n_jobs=1,
    )
    feature_importances = [
        {
            "variable": name,
            "importancia_permutacion": float(value),
            "n_parcelas_prueba": len(test_indices),
            "nota": (
                "Aumento en MAE al permutar en test espacial; diagnóstico predictivo, "
                "no causal ni evidencia suficiente de transferencia."
            ),
        }
        for name, value in zip(model_feature_names, permutation.importances_mean)
    ]
    feature_importances.sort(
        key=lambda item: item["importancia_permutacion"], reverse=True
    )

    predictions_file = OUTPUT / "predicciones_rendimiento_transferencia_objetivo_2026.csv"
    with predictions_file.open("w", encoding="utf-8-sig", newline="") as output:
        headers = (
            "ID_POLIGONO",
            "fuente_etiquetas_entrenamiento",
            "horizonte_estimacion",
            "RENDIMIENTO_ESTIMADO_T_HA",
            "referencia_media_etiquetas_t_ha",
            "modelo",
            "estado_transferencia",
        )
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        mean_target = float(np.mean(y_all))
        for position, polygon_id in enumerate(prediction_ids):
            writer.writerow(
                {
                    "ID_POLIGONO": polygon_id,
                    "fuente_etiquetas_entrenamiento": (
                        "ID_area_rendimiento_70_30_Reto_AgroCebada.csv"
                    ),
                    "horizonte_estimacion": "objetivo 2026; sin datos observados de 2026",
                    "RENDIMIENTO_ESTIMADO_T_HA": (
                        float(prediction_values[position])
                        if prediction_values is not None
                        else ""
                    ),
                    "referencia_media_etiquetas_t_ha": mean_target,
                    "modelo": selected_name if robust else "sin modelo robusto validado",
                    "estado_transferencia": status,
                }
            )

    test_file = OUTPUT / "predicciones_prueba_espacial_etiquetadas.csv"
    with test_file.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "ID_POLIGONO",
                "rendimiento_etiquetado_t_ha",
                "prediccion_test_espacial_t_ha",
                "modelo",
                "observacion",
            ),
        )
        writer.writeheader()
        for polygon_id, actual, predicted in zip(
            (training_ids[index] for index in test_indices),
            y_all[test_indices],
            test_prediction,
        ):
            writer.writerow(
                {
                    "ID_POLIGONO": polygon_id,
                    "rendimiento_etiquetado_t_ha": float(actual),
                    "prediccion_test_espacial_t_ha": float(predicted),
                    "modelo": selected_name,
                    "observacion": "Test espacial interno sobre parcelas con etiqueta disponible.",
                }
            )

    with (OUTPUT / "metricas_evaluacion_transferencia_objetivo_2026.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        headers = (
            "modelo",
            "conjunto",
            "n_parcelas",
            "MAE_t_ha",
            "RMSE_t_ha",
            "R2",
            "parametros",
        )
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(metric_rows)

    diagnostic_rows = [
        ("MAE_validacion_modelo", selected_val_mae, "t/ha"),
        ("MAE_validacion_linea_base", baseline_val_metrics["MAE_t_ha"], "t/ha"),
        ("MAE_test_espacial_modelo", test_scores["MAE_t_ha"], "t/ha"),
        ("MAE_test_espacial_linea_base", baseline_test_scores["MAE_t_ha"], "t/ha"),
        ("RMSE_validacion_modelo", selected_validation_metrics["RMSE_t_ha"], "t/ha"),
        ("RMSE_validacion_linea_base", baseline_val_metrics["RMSE_t_ha"], "t/ha"),
        ("RMSE_test_espacial_modelo", test_scores["RMSE_t_ha"], "t/ha"),
        ("RMSE_test_espacial_linea_base", baseline_test_scores["RMSE_t_ha"], "t/ha"),
        ("R2_validacion_modelo", selected_validation_metrics["R2"], "adimensional"),
        ("R2_test_espacial_modelo", test_scores["R2"], "adimensional"),
        ("error_medio_validacion_observado_menos_estimado", validation_bias, "t/ha"),
        ("IC95_inferior_error_medio_validacion", validation_bias_low, "t/ha"),
        ("IC95_superior_error_medio_validacion", validation_bias_high, "t/ha"),
        ("error_medio_test_observado_menos_estimado", test_bias, "t/ha"),
        ("IC95_inferior_error_medio_test", test_bias_low, "t/ha"),
        ("IC95_superior_error_medio_test", test_bias_high, "t/ha"),
        ("fraccion_valores_test_fuera_rango_ajuste", oob_test, "fracción"),
        ("fraccion_valores_reserva_fuera_rango_ajuste", oob_prediction, "fracción"),
        ("fraccion_faltantes_test", missing_test, "fracción"),
        ("fraccion_faltantes_reserva", missing_prediction, "fracción"),
        ("criterio_MAE_mejora_validacion", validation_mae_improves, "booleano"),
        ("criterio_RMSE_mejora_validacion", validation_rmse_improves, "booleano"),
        ("criterio_MAE_mejora_test", test_mae_improves, "booleano"),
        ("criterio_RMSE_mejora_test", test_rmse_improves, "booleano"),
        ("criterio_modelo_mejora_validacion", validation_improves, "booleano"),
        ("criterio_modelo_mejora_test", test_improves, "booleano"),
        ("criterio_R2_no_negativo_validacion", validation_r2_supported, "booleano"),
        ("criterio_R2_no_negativo_test", test_r2_supported, "booleano"),
        ("criterio_IC95_sesgo_validacion_incluye_cero", validation_bias_supported, "booleano"),
        ("criterio_IC95_sesgo_test_incluye_cero", test_bias_supported, "booleano"),
        ("criterio_reserva_no_mas_desplazada_que_test", prediction_domain_supported, "booleano"),
        ("transferencia_respaldada", robust, "booleano"),
        ("modelo_seleccionado", selected_name, "texto"),
        ("estado", status, "texto"),
        ("n_entrenamiento", len(train_indices), "parcelas"),
        ("n_validacion", len(validation_indices), "parcelas"),
        ("n_test_espacial", len(test_indices), "parcelas"),
        ("n_reserva", len(prediction_ids), "parcelas"),
        ("metodo_bloques", "bandas contiguas por eje principal de centroides de bounding box", "texto"),
    ]
    with (OUTPUT / "diagnostico_transferencia.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        writer = csv.DictWriter(
            output, fieldnames=("diagnostico", "valor", "unidad_o_tipo")
        )
        writer.writeheader()
        writer.writerows(
            [
                {"diagnostico": name, "valor": value, "unidad_o_tipo": unit}
                for name, value, unit in diagnostic_rows
            ]
        )

    if feature_importances:
        with (OUTPUT / "importancia_variables_evaluacion_espacial.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as output:
            writer = csv.DictWriter(
                output,
                fieldnames=(
                    "variable",
                    "importancia_permutacion",
                    "n_parcelas_prueba",
                    "nota",
                ),
            )
            writer.writeheader()
            writer.writerows(feature_importances)

    plots_folder = OUTPUT / "graficas"
    plots_folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Validación", "Prueba espacial"]
    baseline_values = [
        float(baseline_val_metrics["MAE_t_ha"]),
        float(baseline_test_scores["MAE_t_ha"]),
    ]
    selected_values = [
        selected_val_mae,
        float(test_scores["MAE_t_ha"]),
    ]
    positions = np.arange(2)
    ax.bar(positions - 0.18, baseline_values, width=0.36, label="Línea base (media)", color="#9C9C9C")
    ax.bar(positions + 0.18, selected_values, width=0.36, label=selected_name, color="#4C78A8")
    ax.set_xticks(positions, categories)
    ax.set_ylabel("MAE (t/ha; rendimiento etiquetado)")
    ax.set_title("Evaluación espacial interna de transferencia")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.text(
        0.01,
        0.01,
        "Validación elige el modelo; el bloque de prueba espacial se evalúa una sola vez. "
        "No equivale a validar el ciclo 2026.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(plots_folder / "mae_modelo_vs_baseline.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_all[test_indices], test_prediction, color="#4C78A8", alpha=0.8)
    lower = min(float(np.min(y_all[test_indices])), float(np.min(test_prediction)))
    upper = max(float(np.max(y_all[test_indices])), float(np.max(test_prediction)))
    ax.plot([lower, upper], [lower, upper], "--", color="#E15759", label="Coincidencia ideal")
    ax.set(
        title="Rendimiento observado vs. estimado en test espacial",
        xlabel="Rendimiento etiquetado observado (t/ha)",
        ylabel="Estimado por el modelo (t/ha)",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.text(
        0.01,
        0.01,
        f"n={len(test_indices)} parcelas. Evaluación interna espacial; las etiquetas del test "
        "no se usaron para ajustar.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(plots_folder / "observado_vs_estimado_test.png", dpi=160)
    plt.close(fig)

    if feature_importances:
        top = feature_importances[:15][::-1]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            [item["variable"] for item in top],
            [item["importancia_permutacion"] for item in top],
            color="#59A14F",
        )
        ax.set(
            title="Variables influyentes en test (importancia por permutación)",
            xlabel="Aumento del MAE al permutar (t/ha)",
        )
        ax.grid(axis="x", alpha=0.25)
        importance_note = (
            "Diagnóstico predictivo, no causal; calculado en el bloque de prueba espacial."
        )
        if not robust:
            importance_note += " Transferencia no respaldada; no implica utilidad fiable en 2026."
        fig.text(
            0.01,
            0.01,
            importance_note,
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(plots_folder / "importancia_variables_test.png", dpi=160)
        plt.close(fig)

    if prediction_values is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            [centroids[item][0] for item in prediction_ids],
            [centroids[item][1] for item in prediction_ids],
            c=prediction_values,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.3,
        )
        ax.set(
            title="Estimaciones de rendimiento 2026 en parcelas reservadas",
            xlabel="Coordenada X del centro del bounding box",
            ylabel="Coordenada Y del centro del bounding box",
        )
        fig.colorbar(ax.collections[0], ax=ax, label="Estimación (t/ha)")
        fig.text(
            0.01,
            0.01,
            "Solo se dibujan estimaciones si la evaluación espacial y el diagnóstico de cobertura "
            "respaldan la transferencia.",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(
            plots_folder / "estimaciones_transferencia_objetivo_2026.png", dpi=160
        )
        plt.close(fig)
    else:
        stale_outputs = (
            plots_folder / "estimaciones_transferencia_objetivo_2026.png",
            plots_folder / "estimaciones_2026_parcelas.png",
        )
        for stale_output in stale_outputs:
            if stale_output.is_file():
                stale_output.unlink()

    return {
        "selected_model": selected_name,
        "robust": robust,
        "status": status,
        "failed_criteria": failed_criteria,
        "validation_metrics": selected_validation_metrics,
        "baseline_validation_metrics": baseline_val_metrics,
        "test_metrics": test_scores,
        "baseline_test_metrics": baseline_test_scores,
        "validation_bias": validation_bias,
        "validation_bias_interval": (validation_bias_low, validation_bias_high),
        "test_bias": test_bias,
        "test_bias_interval": (test_bias_low, test_bias_high),
        "metrics": metric_rows,
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "test_count": len(test_indices),
        "prediction_count": len(prediction_ids),
        "prediction_values": prediction_values,
        "feature_importances": feature_importances,
        "prediction_ids": prediction_ids,
        "axis_positions": axis_positions,
        "test_actual": y_all[test_indices],
        "test_prediction": test_prediction,
    }