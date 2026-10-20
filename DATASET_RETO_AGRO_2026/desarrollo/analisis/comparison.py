"""Compara baselines y fuentes de features bajo particiones espaciales comunes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import t as student_t
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from analisis.config import OUTPUT, TARGET
from config import ASSIGNMENT_PATH, PROJECT_ROOT

BASELINE_ROOT = PROJECT_ROOT / "cebada_datos_baseline_1"
BASELINE_DATA = BASELINE_ROOT / "data_processed"
PARCEL_PATH = OUTPUT / "tabla_modelado_parcelaria.csv"
BASELINE_FEATURE_PATH = BASELINE_DATA / "features_parcela.csv"
BASELINE_DICTIONARY_PATH = BASELINE_DATA / "diccionario_features.csv"
SERIES_PATH = BASELINE_DATA / "series_dekadales_2025.csv"
TRACE_PATH = OUTPUT / "trazabilidad_archivos.csv"
VERSION = "pre_sowing_2026_comparison_v4"
RIDGE_ALPHA = 10.0
BLOCK_COUNTS = (4, 6, 8)
SHRINKAGE = 3.0
K_NEIGHBORS = 5
SIMILARITY_TOLERANCE_T_HA = 0.02

CURATED_FEATURES = [
    "s2_ndvi_peak",
    "s2_ndvi_auc",
    "s2_ndvi_los",
    "s2_ndvi_sos_doy",
    "s2_ndvi_base",
    "s2_ndvi_pico",
    "s2_evi_base",
    "s2_evi_pico",
    "s2_lai_pico",
    "s2_msi_pico",
    "s2_ndwi_pico",
    "s2_ndti_pre",
    "s2_crc_pre",
    "s2_nddi_pico",
    "pl_ndvi_peak",
    "pl_ndvi_auc",
    "pl_ndvi_los",
    "pl_evi_pico",
    "pl_lai_pico",
    "ls_vi6t_pico",
    "topo_elev_media",
    "topo_pend_media",
    "geo_area_ha",
    "geo_compacidad",
]

GATES = {
    "MAE/RMSE": "ambas deben mejorar respecto a la mejor referencia geográfica en validación interna y holdout externo",
    "R2": "debe ser >= 0 en validación interna y holdout externo",
    "sesgo": "el IC del 95% del error medio debe incluir cero en validación interna y holdout externo",
    "dominio": "en las 59 parcelas sin etiqueta, faltantes y valores fuera de rango no superan los del holdout etiquetado",
}


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Falta una entrada requerida para la comparación: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    for path in (
        BASELINE_FEATURE_PATH,
        BASELINE_DICTIONARY_PATH,
        SERIES_PATH,
        PARCEL_PATH,
        TRACE_PATH,
        ASSIGNMENT_PATH,
    ):
        _require_file(path)

    baseline = pd.read_csv(BASELINE_FEATURE_PATH, dtype={"ID_POLIGONO": str})
    development = pd.read_csv(PARCEL_PATH, dtype={"ID_POLIGONO": str})
    assignments = pd.read_csv(ASSIGNMENT_PATH, dtype={"ID_POLIGONO": str})
    for name, table in (
        ("baseline_1", baseline),
        ("desarrollo", development),
        ("asignación oficial", assignments),
    ):
        if "ID_POLIGONO" not in table:
            raise ValueError(f"La tabla {name} no contiene ID_POLIGONO.")
        if table["ID_POLIGONO"].isna().any() or table["ID_POLIGONO"].duplicated().any():
            raise ValueError(f"La clave ID_POLIGONO no es única y completa en {name}.")

    if "CONJUNTO" not in baseline or "PARTICION" not in development:
        raise ValueError("Falta la partición oficial en una de las tablas parcelarias.")
    if TARGET not in baseline or TARGET not in development:
        raise ValueError("Falta la columna de rendimiento etiquetado.")
    if not {"CONJUNTO", TARGET}.issubset(assignments.columns):
        raise ValueError(
            "La tabla de asignación debe incluir CONJUNTO y RENDIMIENTO_T_HA."
        )
    baseline = baseline.set_index("ID_POLIGONO").sort_index()
    development = development.set_index("ID_POLIGONO").sort_index()
    assignments = assignments.set_index("ID_POLIGONO").sort_index()
    baseline_ids, development_ids = set(baseline.index), set(development.index)
    assignment_ids = set(assignments.index)
    if (
        baseline_ids != development_ids
        or baseline_ids != assignment_ids
    ):
        raise ValueError(
            "Las fuentes no cubren las mismas parcelas; "
            f"solo baseline={len(baseline_ids-development_ids)}, "
            f"solo desarrollo={len(development_ids-baseline_ids)}, "
            f"solo asignación={len(baseline_ids-assignment_ids)}."
        )

    baseline_partition = baseline["CONJUNTO"].fillna("").str.strip().str.upper()
    development_partition = development["PARTICION"].fillna("").str.strip().str.upper()
    official_partition = assignments["CONJUNTO"].fillna("").str.strip().str.upper()
    if (
        baseline_partition.ne(development_partition).any()
        or baseline_partition.ne(official_partition).any()
    ):
        mismatched = baseline_partition[
            baseline_partition.ne(development_partition)
            | baseline_partition.ne(official_partition)
        ].index
        raise ValueError(
            "Las particiones difieren de la tabla de asignación oficial para: "
            + ", ".join(mismatched[:10])
        )
    baseline_target = pd.to_numeric(baseline[TARGET], errors="coerce")
    development_target = pd.to_numeric(development[TARGET], errors="coerce")
    official_target = pd.to_numeric(assignments[TARGET], errors="coerce")
    targets = (baseline_target, development_target, official_target)
    both_missing = pd.concat([series.isna() for series in targets], axis=1).all(axis=1)
    both_present = pd.concat([series.notna() for series in targets], axis=1).all(axis=1)
    equal_values = np.isclose(
        baseline_target[both_present], development_target[both_present], rtol=0, atol=1e-9
    )
    equal_official_values = np.isclose(
        baseline_target[both_present], official_target[both_present], rtol=0, atol=1e-9
    )
    if (
        not (both_missing | both_present).all()
        or not equal_values.all()
        or not equal_official_values.all()
    ):
        raise ValueError(
            "Las etiquetas difieren de la fuente oficial ID_area_rendimiento_70_30."
        )
    if (
        official_partition.eq("ENTRENAMIENTO") != official_target.notna()
    ).any():
        raise ValueError(
            "La fuente oficial debe tener rendimiento en ENTRENAMIENTO y estar vacía "
            "en PREDICCION."
        )
    if not official_partition.isin(("ENTRENAMIENTO", "PREDICCION")).all():
        raise ValueError("Se encontró una partición desconocida o vacía.")

    dev_features = [
        column
        for column in development.columns
        if column.startswith("HISTORICO_PROMEDIO__")
        or column.startswith("TOPOGRAFIA__")
    ]
    if not dev_features:
        raise ValueError("No se encontraron features históricas/topográficas de desarrollo.")

    master = baseline.drop(columns=["CONJUNTO", TARGET]).copy()
    master.insert(0, "PARTICION", official_partition)
    master.insert(1, TARGET, official_target)
    master["ID_POLIGONO"] = master.index
    master = pd.concat(
        [
            master,
            development[dev_features].add_prefix("DESARROLLO__"),
        ],
        axis=1,
    )
    master = master.reset_index(drop=True)
    labelled_count = int(master[TARGET].notna().sum())
    if labelled_count < 30:
        raise ValueError(f"Solo hay {labelled_count} parcelas etiquetadas.")

    trace_rows: list[dict[str, Any]] = []
    source_specs = [
        ("features_baseline_1", BASELINE_FEATURE_PATH),
        ("diccionario_baseline_1", BASELINE_DICTIONARY_PATH),
        ("series_dekadales_2025", SERIES_PATH),
        ("tabla_parcelaria_desarrollo", PARCEL_PATH),
        ("trazabilidad_desarrollo", TRACE_PATH),
    ]
    for source_name, path in source_specs:
        trace_rows.append(
            {
                "fuente": source_name,
                "ruta": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "estado_hash_fuente_original": "archivo_derivado",
            }
        )
    with TRACE_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        trace_reader = csv.DictReader(source)
        for row in trace_reader:
            relative_path = row.get("archivo_fuente", "")
            if not relative_path:
                continue
            source_path = PROJECT_ROOT / Path(relative_path)
            if not source_path.is_file():
                trace_rows.append(
                    {
                        "fuente": "fuente_original_desarrollo",
                        "ruta": relative_path,
                        "bytes": row.get("bytes", ""),
                        "sha256": row.get("sha256", ""),
                        "estado_hash_fuente_original": "fuente_no_disponible",
                    }
                )
                continue
            current_hash = _sha256(source_path)
            expected_hash = row.get("sha256", "")
            trace_rows.append(
                {
                    "fuente": "fuente_original_desarrollo",
                    "ruta": relative_path,
                    "bytes": source_path.stat().st_size,
                    "sha256": current_hash,
                    "estado_hash_fuente_original": (
                        "coincide_con_trazabilidad"
                        if current_hash == expected_hash
                        else "hash_difiere_de_trazabilidad"
                    ),
                }
            )
    return master, development, trace_rows


def _feature_sets(master: pd.DataFrame) -> dict[str, dict[str, Any]]:
    columns = list(master.columns)
    satellite = [
        name for name in columns if name.startswith(("s2_", "pl_", "ls_"))
    ]
    baseline_topography_geometry = [
        name for name in columns if name.startswith(("topo_", "geo_"))
    ]
    baseline_climate = [name for name in columns if name.startswith("clim_")]
    historical = [
        name
        for name in columns
        if name.startswith("DESARROLLO__HISTORICO_PROMEDIO__")
    ]
    dev_topography = [
        name for name in columns if name.startswith("DESARROLLO__TOPOGRAFIA__")
    ]
    dev_climate = [
        name
        for name in historical
        if "__CLIMA__" in name.upper()
    ]
    location = [
        name for name in ("loc_lon", "loc_lat", "loc_x_utm", "loc_y_utm")
        if name in columns
    ]
    missing_curated = [name for name in CURATED_FEATURES if name not in columns]
    if missing_curated:
        raise ValueError(
            "Faltan columnas del set fenológico curado: "
            + ", ".join(missing_curated)
        )

    definitions = {
        "ref_media_global": ([], "global_mean", "Media global de entrenamiento"),
        "ref_media_municipal_encogimiento": (
            [],
            "municipal",
            "Media municipal con encogimiento n/(n+3) hacia la media global",
        ),
        "ref_knn_espacial_k5": (
            [],
            "knn",
            "Media inversamente ponderada por distancia de los 5 vecinos",
        ),
        "ridge_satelite_solo": (
            satellite,
            "ridge",
            "Sentinel-2, Landsat y Planet; sin clima ni topografía",
        ),
        "ridge_satelite_topografia_geometria_baseline1": (
            satellite + baseline_topography_geometry,
            "ridge",
            "Satélite + topografía zonal + geometría del baseline_1",
        ),
        "ridge_satelite_topografia_geometria_clima_centroide": (
            satellite + baseline_topography_geometry + baseline_climate,
            "ridge",
            "Baseline anterior + clima 2025 por celda que contiene el centroide",
        ),
        "ridge_satelite_topografia_desarrollo_geometria": (
            satellite + dev_topography + ["geo_area_ha", "geo_compacidad"],
            "ridge",
            "Satélite + topografía zonal del desarrollo + geometría",
        ),
        "ridge_satelite_topografia_geometria_clima_zonal_historico": (
            satellite
            + baseline_topography_geometry
            + dev_climate,
            "ridge",
            "Mismo set satélite/topografía/geometría + clima histórico zonal de desarrollo",
        ),
        "ridge_fenologia_curada_24": (
            CURATED_FEATURES,
            "ridge",
            "Set curado de 24 variables fenológicas, topográficas y geométricas",
        ),
        "ridge_historico_desarrollo": (
            historical + dev_topography,
            "ridge",
            "Promedios históricos de capturas BASIC/PRO/clima 2022-2025 + topografía parcelaria",
        ),
        "ridge_historico_desarrollo_sin_clima": (
            [name for name in historical if name not in dev_climate] + dev_topography,
            "ridge",
            "Features históricas de desarrollo sin variables de clima",
        ),
        "ridge_ubicacion_sola": (
            location,
            "ridge",
            "Longitud/latitud y coordenadas proyectadas del centroide",
        ),
        "ridge_ubicacion_mas_features_curadas": (
            location + CURATED_FEATURES,
            "ridge",
            "Coordenadas del centroide + features fenológicas curadas",
        ),
        "ridge_prior_municipal_residuo_curado": (
            CURATED_FEATURES,
            "municipal_residual",
            "Prior municipal encogido + Ridge sobre residuo con features curadas",
        ),
        "ridge_prior_knn_residuo_curado": (
            CURATED_FEATURES,
            "knn_residual",
            "Prior k-NN espacial + Ridge sobre residuo con features curadas",
        ),
    }
    for name, (features, kind, description) in definitions.items():
        if kind == "ridge" and not features:
            raise ValueError(f"El modelo {name} no tiene predictores.")
        definitions[name] = {
            "features": list(dict.fromkeys(features)),
            "kind": kind,
            "description": description,
            "parameters": _model_parameters(kind),
        }
    return definitions


def _model_parameters(kind: str) -> str:
    if kind in ("ridge", "municipal_residual", "knn_residual"):
        return json.dumps(
            {
                "model": "StandardScaler + Ridge",
                "alpha": RIDGE_ALPHA,
                "imputation": "mediana de entrenamiento; 0 si la columna está totalmente vacía",
            },
            sort_keys=True,
        )
    if kind == "municipal":
        return json.dumps({"shrinkage_prior": SHRINKAGE}, sort_keys=True)
    if kind == "knn":
        return json.dumps({"k": K_NEIGHBORS, "weight": "1/max(distancia_m,1)"}, sort_keys=True)
    return json.dumps({"model": kind}, sort_keys=True)


def _impute_fit_apply(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    train = np.where(np.isfinite(train), train, np.nan)
    test = np.where(np.isfinite(test), test, np.nan)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return (
        np.where(np.isfinite(train), train, medians),
        np.where(np.isfinite(test), test, medians),
    )


def _municipal_prior(
    y_train: np.ndarray,
    train_municipalities: np.ndarray,
    test_municipalities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(y_train) < 2:
        raise ValueError("El prior leave-one-out requiere al menos dos etiquetas.")
    global_mean = float(np.mean(y_train))
    train_prediction = np.empty(len(y_train), dtype=float)
    test_prediction = np.empty(len(test_municipalities), dtype=float)
    train_positions = np.arange(len(y_train))
    for index, municipality in enumerate(train_municipalities):
        selected = (train_municipalities == municipality) & (train_positions != index)
        local_y = y_train[selected]
        global_without_self = float(
            (y_train.sum() - y_train[index]) / (len(y_train) - 1)
        )
        train_prediction[index] = (
            float(local_y.sum()) + SHRINKAGE * global_without_self
        ) / (len(local_y) + SHRINKAGE)
    for index, municipality in enumerate(test_municipalities):
        selected_y = y_train[train_municipalities == municipality]
        test_prediction[index] = (
            float(selected_y.sum()) + SHRINKAGE * global_mean
        ) / (len(selected_y) + SHRINKAGE)
    return train_prediction, test_prediction


def _knn_prior(
    y_train: np.ndarray,
    train_coordinates: np.ndarray,
    test_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = cdist(train_coordinates, train_coordinates)
    np.fill_diagonal(distances, np.inf)
    train_indices = np.argsort(distances, axis=1)[:, : min(K_NEIGHBORS, len(y_train) - 1)]
    train_distances = np.take_along_axis(distances, train_indices, axis=1)
    weights = 1.0 / np.maximum(train_distances, 1.0)
    train_prediction = np.sum(y_train[train_indices] * weights, axis=1) / weights.sum(axis=1)

    test_distances = cdist(test_coordinates, train_coordinates)
    test_indices = np.argsort(test_distances, axis=1)[:, : min(K_NEIGHBORS, len(y_train))]
    selected_distances = np.take_along_axis(test_distances, test_indices, axis=1)
    weights = 1.0 / np.maximum(selected_distances, 1.0)
    test_prediction = np.sum(y_train[test_indices] * weights, axis=1) / weights.sum(axis=1)
    return train_prediction, test_prediction


def _ridge_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    x_train_filled, x_test_filled = _impute_fit_apply(x_train, x_test)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_filled)
    x_test_scaled = scaler.transform(x_test_filled)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(x_train_scaled, y_train)
    return model.predict(x_test_scaled)


def _predict(
    spec: dict[str, Any],
    labelled: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    target: np.ndarray,
    coordinates: np.ndarray,
    municipalities: np.ndarray,
) -> np.ndarray:
    y_train = target[train_indices]
    kind = spec["kind"]
    if kind == "global_mean":
        return np.full(len(test_indices), float(np.mean(y_train)))
    if kind == "municipal":
        return _municipal_prior(
            y_train,
            municipalities[train_indices],
            municipalities[test_indices],
        )[1]
    if kind == "knn":
        return _knn_prior(
            y_train,
            coordinates[train_indices],
            coordinates[test_indices],
        )[1]

    features = spec["features"]
    x_train = labelled.iloc[train_indices][features].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    x_test = labelled.iloc[test_indices][features].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if kind == "ridge":
        return _ridge_predict(x_train, y_train, x_test)

    if kind == "municipal_residual":
        train_prior, test_prior = _municipal_prior(
            y_train,
            municipalities[train_indices],
            municipalities[test_indices],
        )
    elif kind == "knn_residual":
        train_prior, test_prior = _knn_prior(
            y_train,
            coordinates[train_indices],
            coordinates[test_indices],
        )
    else:
        raise ValueError(f"Tipo de modelo no reconocido: {kind}")
    residual_prediction = _ridge_predict(x_train, y_train - train_prior, x_test)
    return test_prior + residual_prediction


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | str]:
    return {
        "MAE_t_ha": float(mean_absolute_error(actual, predicted)),
        "RMSE_t_ha": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else "",
        "sesgo_observado_menos_predicho_t_ha": float(np.mean(actual - predicted)),
    }


def _mean_residual_interval(
    actual: np.ndarray, predicted: np.ndarray
) -> tuple[float, float]:
    residuals = actual - predicted
    if len(residuals) < 2:
        return float("-inf"), float("inf")
    standard_error = float(np.std(residuals, ddof=1) / np.sqrt(len(residuals)))
    critical_value = float(student_t.ppf(0.975, df=len(residuals) - 1))
    mean_error = float(np.mean(residuals))
    margin = critical_value * standard_error
    return mean_error - margin, mean_error + margin


def _coverage_diagnostic(
    reference: np.ndarray, comparison: np.ndarray
) -> tuple[float, float]:
    present = np.isfinite(comparison)
    denominator = int(present.sum())
    if denominator == 0:
        return 1.0, 0.0
    selected_reference = np.where(np.isfinite(reference), reference, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        minimum = np.nanmin(selected_reference, axis=0)
        maximum = np.nanmax(selected_reference, axis=0)
        medians = np.nanmedian(selected_reference, axis=0)
    minimum = np.where(np.isfinite(minimum), minimum, medians)
    maximum = np.where(np.isfinite(maximum), maximum, medians)
    out_of_range = present & (
        (comparison < minimum) | (comparison > maximum)
    )
    return float(out_of_range.sum() / denominator), float((~present).mean())


def _evaluate_configuration(
    labelled: pd.DataFrame,
    unlabelled: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    block_count: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    coordinates = labelled[["loc_x_utm", "loc_y_utm"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Hay coordenadas UTM faltantes en parcelas etiquetadas.")
    target = labelled[TARGET].to_numpy(dtype=float)
    municipalities = labelled["loc_municipio"].fillna("").astype(str).to_numpy()
    if not np.isfinite(target).all():
        raise ValueError("Hay etiquetas inválidas en parcelas de entrenamiento.")
    centered_coordinates = coordinates - coordinates.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.cov(centered_coordinates, rowvar=False)
    )
    main_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    spatial_order = np.argsort(centered_coordinates @ main_axis, kind="stable")
    groups = np.empty(len(labelled), dtype=int)
    for label, indices in enumerate(np.array_split(spatial_order, block_count)):
        groups[indices] = label
    group_sizes = pd.Series(groups).value_counts().sort_index().to_dict()
    centers = np.vstack(
        [coordinates[groups == label].mean(axis=0) for label in range(block_count)]
    )
    outer_label = int(
        np.argmax(np.linalg.norm(centers - centers.mean(axis=0), axis=1))
    )
    outer_test = np.flatnonzero(groups == outer_label)
    outer_train = np.flatnonzero(groups != outer_label)
    unseen_municipality_rows: list[dict[str, Any]] = []
    known_municipalities = set(municipalities[outer_train])
    for parcel in unlabelled.itertuples(index=False):
        municipality = str(getattr(parcel, "loc_municipio", "") or "")
        unseen_municipality_rows.append(
            {
                "version": VERSION,
                "configuracion_bloques": f"pca_bands_{block_count}",
                "ID_POLIGONO": getattr(parcel, "ID_POLIGONO"),
                "municipio": municipality,
                "visto_en_entrenamiento_externo": municipality in known_municipalities,
                "n_entrenamiento_municipio": int(
                    np.sum(municipalities[outer_train] == municipality)
                ),
                "regla_prior_municipal_si_ausente": "media_global",
            }
        )
    inner_groups = groups[outer_train]
    splitter = GroupKFold(n_splits=len(np.unique(inner_groups)))

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    actual_outer = target[outer_test]
    inner_scores: dict[str, dict[str, float | str]] = {}
    inner_predictions_by_model: dict[str, np.ndarray] = {}
    outer_scores_by_model: dict[str, dict[str, float | str]] = {}
    for model_name, spec in specs.items():
        predictions = np.full(len(outer_train), np.nan, dtype=float)
        for inner_train, inner_test in splitter.split(
            outer_train, groups=inner_groups
        ):
            train_indices = outer_train[inner_train]
            test_indices = outer_train[inner_test]
            prediction = _predict(
                spec,
                labelled,
                train_indices,
                test_indices,
                target,
                coordinates,
                municipalities,
            )
            predictions[inner_test] = prediction
            scores = _metrics(target[test_indices], prediction)
            fold_rows.append(
                {
                    "version": VERSION,
                    "configuracion_bloques": f"pca_bands_{block_count}",
                    "evaluacion": "interna_seleccion",
                    "fold": f"bloque_{int(groups[test_indices[0]])}",
                    "modelo": model_name,
                    "n_entrenamiento": len(train_indices),
                    "n_evaluacion": len(test_indices),
                    "features": len(spec["features"]),
                    "parametros": spec["parameters"],
                    **scores,
                }
            )
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"No se generaron predicciones OOF para {model_name}.")
        inner_scores[model_name] = _metrics(target[outer_train], predictions)
        inner_predictions_by_model[model_name] = predictions
    best_inner_model = min(
        inner_scores,
        key=lambda name: (float(inner_scores[name]["RMSE_t_ha"]), name),
    )
    best_inner_rmse = float(inner_scores[best_inner_model]["RMSE_t_ha"])
    best_inner_mae = float(inner_scores[best_inner_model]["MAE_t_ha"])
    similar_models = [
        name
        for name, scores in inner_scores.items()
        if float(scores["RMSE_t_ha"])
        <= best_inner_rmse + SIMILARITY_TOLERANCE_T_HA
        and float(scores["MAE_t_ha"])
        <= best_inner_mae + SIMILARITY_TOLERANCE_T_HA
    ]
    selected_inner = min(
        similar_models,
        key=lambda name: (len(specs[name]["features"]), name),
    )

    for model_name, spec in specs.items():
        prediction = _predict(
            spec,
            labelled,
            outer_train,
            outer_test,
            target,
            coordinates,
            municipalities,
        )
        outer_scores = _metrics(actual_outer, prediction)
        outer_scores_by_model[model_name] = outer_scores
        fold_rows.append(
            {
                "version": VERSION,
                "configuracion_bloques": f"pca_bands_{block_count}",
                "evaluacion": "externa_holdout",
                "fold": f"bloque_{outer_label}",
                "modelo": model_name,
                "n_entrenamiento": len(outer_train),
                "n_evaluacion": len(outer_test),
                "features": len(spec["features"]),
                "parametros": spec["parameters"],
                **outer_scores,
            }
        )
        summary_rows.append(
            {
                "version": VERSION,
                "configuracion_bloques": f"pca_bands_{block_count}",
                "modelo": model_name,
                "n_features": len(spec["features"]),
                "descripcion": spec["description"],
                **{f"interno_{key}": value for key, value in inner_scores[model_name].items()},
                **{f"externo_{key}": value for key, value in outer_scores.items()},
                "seleccionado_solo_por_cv_interna": model_name == selected_inner,
                "parametros": spec["parameters"],
                "tamano_holdout_bloque": len(outer_test),
                "tamano_bloques": json.dumps(group_sizes, sort_keys=True),
            }
        )
        for position, predicted in zip(outer_test, prediction):
            prediction_rows.append(
                {
                    "version": VERSION,
                    "configuracion_bloques": f"pca_bands_{block_count}",
                    "bloque_holdout": outer_label,
                    "ID_POLIGONO": labelled.iloc[position]["ID_POLIGONO"],
                    "municipio": municipalities[position],
                    "municipio_visto_en_entrenamiento": bool(
                        municipalities[position] in municipalities[outer_train]
                    ),
                    "modelo": model_name,
                    "rendimiento_etiquetado_t_ha": target[position],
                    "prediccion_holdout_t_ha": float(predicted),
                }
            )

    geographic_baselines = (
        "ref_media_municipal_encogimiento",
        "ref_knn_espacial_k5",
    )
    best_geographic_mae = min(
        float(outer_scores_by_model[name]["MAE_t_ha"])
        for name in geographic_baselines
    )
    best_geographic_rmse = min(
        float(outer_scores_by_model[name]["RMSE_t_ha"])
        for name in geographic_baselines
    )
    selected_spec = specs[selected_inner]
    if selected_spec["features"]:
        train_features = labelled.iloc[outer_train][selected_spec["features"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        holdout_features = labelled.iloc[outer_test][selected_spec["features"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        holdout_coverage_by_parcel = np.isfinite(holdout_features).mean(axis=1)
        minimum_holdout_coverage = float(holdout_coverage_by_parcel.min())
        future_features = unlabelled[selected_spec["features"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
    else:
        train_features = np.empty((len(outer_train), 0))
        holdout_features = np.empty((len(outer_test), 0))
        future_features = np.empty((len(unlabelled), 0))
        minimum_holdout_coverage = 1.0
    selected_outer_scores = outer_scores_by_model[selected_inner]
    selected_inner_scores = inner_scores[selected_inner]
    best_geographic_inner_mae = min(
        float(inner_scores[name]["MAE_t_ha"]) for name in geographic_baselines
    )
    best_geographic_inner_rmse = min(
        float(inner_scores[name]["RMSE_t_ha"]) for name in geographic_baselines
    )
    inner_bias_interval = _mean_residual_interval(
        target[outer_train], inner_predictions_by_model[selected_inner]
    )
    outer_bias_interval = _mean_residual_interval(
        actual_outer, np.asarray(
            [
                row["prediccion_holdout_t_ha"]
                for row in prediction_rows
                if row["configuracion_bloques"] == f"pca_bands_{block_count}"
                and row["modelo"] == selected_inner
            ],
            dtype=float,
        )
    )
    holdout_domain = _coverage_diagnostic(train_features, holdout_features)
    future_domain = _coverage_diagnostic(train_features, future_features)
    gates = {
        "MAE_mejora_vs_mejor_referencia_geografica_en_validacion_interna": (
            float(selected_inner_scores["MAE_t_ha"]) < best_geographic_inner_mae
        ),
        "RMSE_mejora_vs_mejor_referencia_geografica_en_validacion_interna": (
            float(selected_inner_scores["RMSE_t_ha"]) < best_geographic_inner_rmse
        ),
        "MAE_mejora_vs_mejor_referencia_geografica_en_holdout_externo": (
            float(selected_outer_scores["MAE_t_ha"]) < best_geographic_mae
        ),
        "RMSE_mejora_vs_mejor_referencia_geografica_en_holdout_externo": (
            float(selected_outer_scores["RMSE_t_ha"]) < best_geographic_rmse
        ),
        "R2_no_negativo_interno_y_externo": (
            float(selected_inner_scores["R2"]) >= 0
            and float(selected_outer_scores["R2"]) >= 0
        ),
        "IC95_sesgo_interno_y_externo_incluye_cero": (
            inner_bias_interval[0] <= 0 <= inner_bias_interval[1]
            and outer_bias_interval[0] <= 0 <= outer_bias_interval[1]
        ),
        "dominio_59_sin_etiqueta_no_peor_que_holdout": (
            future_domain[0] <= holdout_domain[0]
            and future_domain[1] <= holdout_domain[1]
        ),
    }
    gates_passed = all(gates.values())
    for row in summary_rows:
        selected = row["modelo"] == selected_inner
        row["modelo_seleccionado_interno"] = selected_inner
        row["holdout_reservado_para_evaluacion"] = True
        row["seleccion_usa_holdout"] = False
        row["gates_transferencia_predefinidos"] = json.dumps(GATES, ensure_ascii=False)
        row["cobertura_minima_holdout_modelo_seleccionado"] = (
            minimum_holdout_coverage if selected else ""
        )
        row["fraccion_fuera_rango_holdout_seleccionado"] = (
            holdout_domain[0] if selected else ""
        )
        row["fraccion_faltantes_holdout_seleccionado"] = (
            holdout_domain[1] if selected else ""
        )
        row["fraccion_fuera_rango_59_sin_etiqueta"] = (
            future_domain[0] if selected else ""
        )
        row["fraccion_faltantes_59_sin_etiqueta"] = (
            future_domain[1] if selected else ""
        )
        row["ic95_sesgo_interno"] = (
            json.dumps(inner_bias_interval) if selected else ""
        )
        row["ic95_sesgo_holdout"] = (
            json.dumps(outer_bias_interval) if selected else ""
        )
        row["gates_holdout_modelo_seleccionado"] = (
            json.dumps(gates, sort_keys=True) if selected else ""
        )
        row["gates_holdout_superados"] = gates_passed if selected else ""
        row["estado_autorizacion_2026"] = "pendiente_etiquetas_temporales_2026"
    for label, size in group_sizes.items():
        fold_rows.append(
            {
                "version": VERSION,
                "configuracion_bloques": f"pca_bands_{block_count}",
                "evaluacion": "tamano_bloque",
                "fold": f"bloque_{label}",
                "modelo": "",
                "n_entrenamiento": len(labelled) - size,
                "n_evaluacion": size,
                "features": "",
                "parametros": "",
                "MAE_t_ha": "",
                "RMSE_t_ha": "",
                "R2": "",
                "sesgo_observado_menos_predicho_t_ha": "",
            }
        )
    return fold_rows, summary_rows, prediction_rows, unseen_municipality_rows


def _write_holdout_diagnostics(
    labelled: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    predictions: pd.DataFrame,
) -> None:
    parcel_index = labelled.set_index("ID_POLIGONO")
    municipality_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for (configuration, model_name), group in predictions.groupby(
        ["configuracion_bloques", "modelo"], sort=True
    ):
        actual = group["rendimiento_etiquetado_t_ha"].to_numpy(dtype=float)
        predicted = group["prediccion_holdout_t_ha"].to_numpy(dtype=float)
        scores = _metrics(actual, predicted)
        selected = parcel_index.loc[group["ID_POLIGONO"].tolist()]
        features = specs[model_name]["features"]
        if features:
            feature_values = selected[features].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            coverage = np.isfinite(feature_values).mean(axis=1)
            mean_coverage = float(coverage.mean())
            minimum_coverage = float(coverage.min())
        else:
            mean_coverage = minimum_coverage = 1.0
        diagnostic_rows.append(
            {
                "version": VERSION,
                "configuracion_bloques": configuration,
                "modelo": model_name,
                "n_holdout": len(group),
                "observado_min_t_ha": float(actual.min()),
                "observado_max_t_ha": float(actual.max()),
                "predicho_min_t_ha": float(predicted.min()),
                "predicho_max_t_ha": float(predicted.max()),
                "predicho_media_t_ha": float(predicted.mean()),
                "cobertura_media_features": mean_coverage,
                "cobertura_minima_parcela": minimum_coverage,
                "municipios_no_vistos_n": int(
                    (~group["municipio_visto_en_entrenamiento"]).sum()
                ),
                **scores,
            }
        )
        for municipality, municipality_group in group.groupby(
            "municipio", dropna=False, sort=True
        ):
            local_actual = municipality_group[
                "rendimiento_etiquetado_t_ha"
            ].to_numpy(dtype=float)
            local_predicted = municipality_group[
                "prediccion_holdout_t_ha"
            ].to_numpy(dtype=float)
            municipality_rows.append(
                {
                    "version": VERSION,
                    "configuracion_bloques": configuration,
                    "modelo": model_name,
                    "municipio": municipality,
                    "n_holdout": len(municipality_group),
                    "municipio_no_visto_n": int(
                        (~municipality_group["municipio_visto_en_entrenamiento"]).sum()
                    ),
                    **_metrics(local_actual, local_predicted),
                }
            )
    pd.DataFrame(municipality_rows).to_csv(
        OUTPUT / "comparacion_por_municipio.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        OUTPUT / "comparacion_diagnostico_holdout.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _write_dictionary(master: pd.DataFrame, specs: dict[str, dict[str, Any]]) -> int:
    baseline_dictionary = pd.read_csv(BASELINE_DICTIONARY_PATH).set_index("columna")
    numeric_columns = [
        name
        for name in master.columns
        if name not in ("ID_POLIGONO", "PARTICION", TARGET)
        and pd.to_numeric(master[name], errors="coerce").notna().any()
    ]
    dictionary_columns = numeric_columns + [
        name for name in ("loc_municipio",) if name in master.columns
    ]
    used = {name for spec in specs.values() for name in spec["features"]}
    used.add("loc_municipio")
    rows = []
    for name in dictionary_columns:
        is_categorical = name == "loc_municipio"
        series = (
            master[name]
            if is_categorical
            else pd.to_numeric(master[name], errors="coerce")
        )
        if name.startswith("DESARROLLO__HISTORICO_PROMEDIO__"):
            raw_name = name.removeprefix("DESARROLLO__")
            group = "desarrollo_historico_clima" if "__CLIMA__" in name.upper() else "desarrollo_historico_satelite"
            description = (
                "Promedio de medias anuales por parcela; dentro de cada año se promedian las capturas disponibles."
            )
            source = "desarrollo; capturas BASIC/PRO y variables CLIMA"
            period = "2022-2025"
            aggregation = "media de medias anuales"
        elif name.startswith("DESARROLLO__TOPOGRAFIA__"):
            raw_name = name.removeprefix("DESARROLLO__")
            group = "topografia_desarrollo_zonal"
            description = "Media zonal del ráster topográfico dentro de la parcela."
            source = "desarrollo; ráster INEGI extraído por polígono"
            period = "estático"
            aggregation = "media zonal del polígono"
        else:
            raw_name = name
            record = baseline_dictionary.loc[name] if name in baseline_dictionary.index else None
            group = str(record["grupo"]) if record is not None else "ubicacion_geometria"
            description = str(record["descripcion"]) if record is not None else name
            source = "cebada_datos_baseline_1"
            if name.startswith(("s2_", "pl_", "ls_")):
                period = "2025"
                aggregation = "mediana dekadal, ventana o métrica fenológica"
            elif name.startswith("clim_"):
                period = "2025 observado; antecedente para el ciclo 2026"
                aggregation = "celda climática que contiene el centroide"
            elif name.startswith("topo_"):
                period = "estático"
                aggregation = "resumen zonal por parcela"
            elif name.startswith("geo_"):
                period = "estático"
                aggregation = "geometría del polígono"
            elif name in ("loc_lon", "loc_lat", "loc_x_utm", "loc_y_utm"):
                period = "estático"
                aggregation = "coordenadas del centroide"
            else:
                period = "estático"
                aggregation = "categoría municipal de la parcela"
        if "area" in name.lower():
            unit = "ha"
        elif name in ("loc_lon", "loc_lat"):
            unit = "grados"
        elif name in ("loc_x_utm", "loc_y_utm"):
            unit = "m"
        elif "prec_cv" in name.lower():
            unit = "adimensional"
        elif "anom_pct" in name.lower():
            unit = "%"
        elif "pend" in name.lower():
            unit = "grados"
        elif "compacidad" in name.lower():
            unit = "adimensional"
        elif "elev" in name.lower():
            unit = "m s.n.m."
        elif "prec" in name.lower():
            unit = "mm"
        elif any(token in name.lower() for token in ("tmin", "tmax", "tmean", "dtr")):
            unit = "°C"
        elif "gdd" in name.lower():
            unit = "°C·día"
        elif "slope" in name.lower():
            unit = "unidades del índice/día"
        elif "nvalid" in name.lower():
            unit = "dekadas"
        elif name.endswith(("_doy", "_los")) or "doy" in name or "los" in name:
            unit = "días"
        elif is_categorical:
            unit = "categoría"
        else:
            unit = "según variable de origen"
        rows.append(
            {
                "feature": name,
                "nombre_origen": raw_name,
                "fuente": source,
                "grupo": group,
                "definicion": description,
                "unidad": unit,
                "periodo_cubierto": period,
                "regla_agregacion": aggregation,
                "disponible_antes_siembra_2026": True,
                "fraccion_faltante_197_parcelas": float(series.isna().mean()),
                "incluida_en_algun_candidato": name in used,
                "version": VERSION,
            }
        )
    pd.DataFrame(rows).to_csv(
        OUTPUT / "diccionario_features_comparacion.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return len(rows)


def _write_source_hashes(trace_rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(trace_rows).to_csv(
        OUTPUT / "hashes_fuentes_comparacion.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _write_report(
    master: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    summary: pd.DataFrame,
    dictionary_count: int,
) -> None:
    labelled_count = int(master[TARGET].notna().sum())
    prediction_count = len(master) - labelled_count
    source_lines = "\n".join(
        f"| `{record['ruta']}` | {record['bytes']} | `{record['sha256']}` | {record['estado_hash_fuente_original']} |"
        for record in trace_rows
    )
    feature_lines = "\n".join(
        f"| `{name}` | {len(spec['features'])} | {spec['description']} |"
        for name, spec in specs.items()
    )
    result_lines = []
    if not summary.empty:
        for row in summary.itertuples(index=False):
            result_lines.append(
                f"| {row.configuracion_bloques} | {row.modelo} | "
                f"{row.interno_RMSE_t_ha:.3f} | {row.interno_MAE_t_ha:.3f} | "
                f"{row.interno_R2:.3f} | {row.externo_RMSE_t_ha:.3f} | "
                f"{row.externo_MAE_t_ha:.3f} | {row.externo_R2:.3f} | "
                f"{'sí' if row.seleccionado_solo_por_cv_interna else 'no'} | "
                f"{row.tamano_holdout_bloque} |"
            )
    results = "\n".join(result_lines)
    hash_mismatches = [
        record["ruta"]
        for record in trace_rows
        if record["estado_hash_fuente_original"] == "hash_difiere_de_trazabilidad"
    ]
    mismatch_text = (
        "- **Alerta:** fuentes cuyo hash difiere del registro de desarrollo: "
        + ", ".join(f"`{path}`" for path in hash_mismatches)
        if hash_mismatches
        else "- Los archivos originales disponibles coinciden con sus hashes registrados por desarrollo."
    )
    report = f"""# Comparación unificada de baselines — AgroCebada

## Objetivo y autorización

Horizonte de transferencia: estimación **antes de la siembra 2026**. No hay observaciones de clima, satélite ni rendimiento de 2026. Los predictores fechados llegan hasta 2025. La fuente canónica de partición y etiquetas es `ID_area_rendimiento_70_30_Reto_AgroCebada.csv`: sus 138 parcelas `ENTRENAMIENTO` tienen `RENDIMIENTO_T_HA`; las 59 parcelas `PREDICCION` no tienen etiqueta. El archivo no especifica el ciclo al que corresponde cada rendimiento etiquetado.

Este informe compara señales y referencias con evaluación espacial común para revisar el comportamiento con las etiquetas disponibles; no es una evaluación temporal ni demuestra rendimiento futuro. **No autoriza estimaciones respaldadas para 2026**. Las 59 parcelas sin etiqueta no intervienen en la selección ni en las métricas de error; después de congelar el modelo solo se inspecciona en ellas cobertura/dominio y municipios ausentes, sin calcular predicciones de rendimiento.

## Procedencia y huellas

| Fuente | bytes | SHA-256 | Estado |
|---|---:|---|---|
{source_lines}

{mismatch_text}

## Conteos y controles de calidad

- Tabla maestra: {len(master)} parcelas, clave única `ID_POLIGONO`.
- Etiquetas de entrenamiento disponibles en la tabla canónica: {labelled_count}; parcelas `PREDICCION` sin rendimiento: {prediction_count}.
- Features descritas en el diccionario versionado: {dictionary_count}.
- Se validó coincidencia exacta de IDs, particiones y etiquetas entre los dos desarrollos antes de concatenar.
- Las filas longitudinales permanecen como producto separado; su promedio no se mezcla con capturas individuales en la tabla maestra.
- No se generan reportes de calidad; el diccionario de features documenta período, unidad, agregación y disponibilidad temporal.

## Features, definiciones y calidad

| Candidato | n features | Definición |
|---|---:|---|
{feature_lines}

`features_parcela.csv` del baseline_1 resume curvas de 2025: observaciones válidas agrupadas por dekada y sensor con mediana; los umbrales de nubosidad son Sentinel-2/Landsat <=5% y Planet <=20%. Si quedan al menos 3 dekadas válidas, se interpola linealmente y los extremos se extrapolan con el valor más cercano; con menos de 3, la serie queda faltante. `series_dekadales_2025.csv` conserva la serie longitudinal.

Desarrollo conserva observaciones longitudinales 2022–2025. Para modelar, primero calcula la media de cada feature entre capturas de cada parcela-año y luego promedia las medias anuales disponibles (no pondera años por número de capturas). Sus variables `HISTORICO_PROMEDIO__CLIMA__...` son distintas de `clim_*` del baseline_1: estas últimas son clima 2025 de la celda que contiene el centroide; las de desarrollo resumen capturas/valores climáticos por parcela. La topografía zonal de desarrollo se etiqueta por separado de `topo_*` del baseline_1; no se equiparan columnas por nombre.

## Protocolo espacial y selección

- Universo común: las mismas {labelled_count} parcelas con etiqueta, objetivo y partición reconciliados por `ID_POLIGONO`.
- Bloques: bandas espaciales contiguas y balanceadas sobre el eje principal de centroides UTM, configuraciones {", ".join(map(str, BLOCK_COUNTS))} bloques.
- En cada configuración, se reserva externamente el bloque más alejado del centro medio de los centroides. Los otros bloques alimentan validación interna GroupKFold; se elige el menor RMSE agrupado y, entre candidatos a no más de {SIMILARITY_TOLERANCE_T_HA:.2f} t/ha tanto en RMSE como MAE, el de menos features. La prueba externa no interviene en la selección.
- En cada fold todos los candidatos comparten exactamente índices train/test. Los tamaños por bloque, predicciones externas, MAE, RMSE, R² y sesgo por fold se guardan en `comparacion_folds.csv` y `comparacion_predicciones.csv`.
- Ridge usa `StandardScaler`, imputación por mediana del entrenamiento (cero para columnas totalmente vacías) y alpha fijo {RIDGE_ALPHA}; no se ajustan hiperparámetros con los bloques externos.
- La selección interna minimiza RMSE; entre alternativas con RMSE y MAE a no más de {SIMILARITY_TOLERANCE_T_HA:.2f} t/ha del mejor, prefiere la de menor número de features. La regla se fija sin consultar el holdout.
- La media municipal usa encogimiento n/(n+3) hacia la media global; municipios ausentes del entrenamiento recurren a la media global. k-NN usa k={K_NEIGHBORS}, peso 1/max(distancia en m,1). En los residuos, los priors de entrenamiento se calculan leave-one-out.
- `comparacion_por_municipio.csv` desglosa el holdout por municipio; `comparacion_diagnostico_holdout.csv` informa rangos de predicción y cobertura; `comparacion_municipios_sin_entrenar.csv` identifica municipios de la reserva sin referencia municipal de entrenamiento, que recurren a la media global.
- Los RMSE informados en las columnas `interno_*` se calculan sobre predicciones OOF agrupadas. `externo_*` corresponde solo al holdout bloque reservado de esa configuración; no se elige el ganador a partir de ese resultado.

## Resultados por modelo y configuración

| Bloques | Candidato | RMSE interno | MAE interno | R² interno | RMSE externo | MAE externo | R² externo | Seleccionado interno | n externo |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
{results}

No se interpretan las 59 parcelas sin etiqueta para declarar un ganador. La comparación tiene muestras pequeñas y el holdout cambia entre configuraciones; reportar la dispersión entre bloques y configuraciones es parte del resultado, no una promesa de rendimiento futuro.

## Criterios previos para autorizar transferencia

La variante escogida internamente solo podrá considerarse respaldada si: (1) MAE y RMSE mejoran frente a la mejor referencia geográfica tanto en CV interna como en el holdout externo; (2) R² >= 0 en ambas; (3) los IC95 del sesgo observado-menos-predicho incluyen cero en ambas; y (4) para las 59 parcelas sin etiqueta, las fracciones faltantes y fuera de rango no superan las del holdout etiquetado. Son los criterios conservadores existentes de desarrollo, adaptados para comparar contra referencias geográficas; se fijan en código antes de ejecutar y quedan evaluados en `comparacion_resumen_transferencia_objetivo_2026.csv`. Aun superándolos, no sustituyen la validación temporal con etiquetas del ciclo objetivo. No se generan predicciones de rendimiento para las 59 parcelas.

## Límites

- El año/ciclo de las etiquetas de entrenamiento no está indicado en la tabla canónica; no hay observaciones ni verdad-terreno 2026, por lo que no existe validación temporal.
- Las bandas espaciales contiguas y las 138 etiquetas producen estimaciones inciertas; una mejora agregada puede depender de una zona. Revisar `comparacion_folds.csv`, tamaños por bloque y errores parcela a parcela.
- Clima 2025 es antecedente legítimo antes de siembra 2026, pero no se debe trasladar a una predicción de 2025 hecha antes de observar el ciclo.
- Los resultados recientes (RMSE 0.653/0.661/0.729 t/ha) son hipótesis de trabajo bajo protocolos previos diferentes; no se mezclan como si fueran comparaciones de folds comunes ni garantías.
- Las imágenes originales del baseline_1 no están incluidas en `data_processed`; el runner portátil puede reconstruirlo si se proporciona el árbol de fuentes descrito en su README. Este informe regenera la unión y evaluación a partir de las tablas parcelarias procesadas disponibles.
"""
    (OUTPUT / "informe_comparacion_transferencia_objetivo_2026.md").write_text(
        report, encoding="utf-8"
    )


def run_comparison() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    master, development, trace_rows = _read_inputs()
    master.to_csv(
        OUTPUT / "tabla_maestra_comparacion_parcelaria.csv",
        index=False,
        encoding="utf-8-sig",
    )
    specs = _feature_sets(master)
    dictionary_count = _write_dictionary(master, specs)
    labelled = master.loc[master[TARGET].notna()].sort_values("ID_POLIGONO").reset_index(drop=True)
    unlabelled = (
        master.loc[master["PARTICION"].eq("PREDICCION")]
        .sort_values("ID_POLIGONO")
        .reset_index(drop=True)
    )
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    unseen_municipality_rows: list[dict[str, Any]] = []
    for block_count in BLOCK_COUNTS:
        fold_part, summary_part, prediction_part, unseen_municipality_part = _evaluate_configuration(
            labelled, unlabelled, specs, block_count
        )
        fold_rows.extend(fold_part)
        summary_rows.extend(summary_part)
        prediction_rows.extend(prediction_part)
        unseen_municipality_rows.extend(unseen_municipality_part)
    folds = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows)
    folds.to_csv(
        OUTPUT / "comparacion_folds.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        OUTPUT / "comparacion_resumen_transferencia_objetivo_2026.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(
        OUTPUT / "comparacion_predicciones.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_holdout_diagnostics(labelled, specs, predictions)
    pd.DataFrame(unseen_municipality_rows).to_csv(
        OUTPUT / "comparacion_municipios_sin_entrenar.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_source_hashes(trace_rows)
    _write_report(master, specs, trace_rows, summary, dictionary_count)
    print(f"Tabla maestra: {OUTPUT / 'tabla_maestra_comparacion_parcelaria.csv'}")
    print(
        "Comparación espacial: "
        f"{OUTPUT / 'comparacion_resumen_transferencia_objetivo_2026.csv'}"
    )
    print(f"Informe: {OUTPUT / 'informe_comparacion_transferencia_objetivo_2026.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Une baseline_1 y desarrollo y evalúa candidatos con folds espaciales comunes."
    )
    parser.parse_args()
    run_comparison()


if __name__ == "__main__":
    main()
