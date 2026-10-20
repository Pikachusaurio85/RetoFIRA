import csv
from datetime import date
from pathlib import Path
import re
from config import ASSIGNMENT_PATH, OUTPUT_FOLDER
from procesamiento.climate_features import climate_column_names, extract_climate_features
from procesamiento.data_normalization import (
    QualityReport,
    clean_text,
    is_missing,
    normalize_number,
    normalize_text,
)
from lecturas.dataset_readers import DATASETS, DATASET_YEARS, read_year_observations

TARGET_COLUMN = "RENDIMIENTO_T_HA"
METADATA_COLUMNS = (
    "ID_POLIGONO",
    "PARTICION",
    "FUENTE",
    "FECHA_CAPTURA",
    TARGET_COLUMN,
)

def _find_header(headers: list[str], expected: str) -> str:
    matches = [
        header
        for header in headers
        if normalize_text(header).replace(" ", "_") == expected.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Se esperaba exactamente una columna {expected}; "
            f"se encontraron {len(matches)}."
        )
    return matches[0]

def _category_column(prefix: str, value: str) -> str:
    normalized = normalize_text(value)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if not slug:
        raise ValueError(f"No se puede convertir la categoría a columna: {value!r}")
    return f"{prefix}_{slug.upper()}"

def _load_assignments(
    path: Path,
    quality: QualityReport,
) -> dict[str, dict[str, str]]:
    assignments: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        headers = next(reader, None)
        if headers is None:
            raise ValueError(f"El archivo de asignación está vacío: {path}")
        id_header = _find_header(headers, "ID_POLIGONO")
        split_header = _find_header(headers, "CONJUNTO")
        area_header = _find_header(headers, "AREA_HA")
        yield_header = _find_header(headers, TARGET_COLUMN)
        for row_number, values in enumerate(reader, start=2):
            if not values or all(is_missing(value) for value in values):
                quality.add("FILA_COMPLETAMENTE_VACIA", "fila_vacia", row_number)
                continue
            if len(values) > len(headers):
                quality.add(
                    "FILA",
                    "cantidad_de_campos_invalida",
                    row_number,
                    f"{len(values)} valores; se esperaban {len(headers)}",
                )
                values = values[: len(headers)]
            values.extend([""] * (len(headers) - len(values)))
            row = dict(zip(headers, values))

            polygon_id = clean_text(row[id_header])
            if not polygon_id:
                quality.add(id_header, "faltante", row_number)
                continue
            if polygon_id in assignments:
                raise ValueError(
                    f"ID_POLIGONO duplicado en la tabla de asignación: {polygon_id}."
                )

            split = normalize_text(row[split_header]).upper()
            if split not in {"ENTRENAMIENTO", "PREDICCION"}:
                quality.add(split_header, "categoria_invalida", row_number, row[split_header])
                continue

            area = ""
            try:
                area = normalize_number(row[area_header]) or ""
            except ValueError:
                quality.add(area_header, "formato_invalido", row_number, row[area_header])

            yield_value = ""
            if split == "ENTRENAMIENTO":
                try:
                    yield_value = normalize_number(row[yield_header]) or ""
                except ValueError:
                    raise ValueError(
                        f"Rendimiento inválido en {yield_header}, fila {row_number}: "
                        f"{row[yield_header]!r}."
                    )
                if not yield_value:
                    raise ValueError(
                        f"Falta rendimiento etiquetado en parcela de entrenamiento "
                        f"{polygon_id} (fila {row_number})."
                    )
            elif not is_missing(row[yield_header]):
                raise ValueError(
                    f"La parcela de predicción {polygon_id} tiene rendimiento en la "
                    f"fuente canónica (fila {row_number}); revise la partición."
                )

            assignments[polygon_id] = {
                "PARTICION": split,
                "AREA_HA": area,
                TARGET_COLUMN: yield_value if split == "ENTRENAMIENTO" else "",
            }

    return assignments

def _source_feature_headers() -> dict[str, list[str]]:
    source_features: dict[str, list[str]] = {}
    for source_name, source_path in DATASETS.items():
        with source_path.open("r", encoding="utf-8-sig", newline="") as source:
            headers = next(csv.reader(source), None)
        if headers is None:
            raise ValueError(f"El archivo no contiene encabezados: {source_path}")
        id_header = _find_header(headers, "ID_POLIGONO")
        date_header = _find_header(headers, "fecha_captura")
        sensor_header = _find_header(headers, "sensor")
        source_features[source_name] = [
            header
            for header in headers
            if header not in {id_header, date_header, sensor_header}
        ]
    return source_features

def build_machine_learning_rows(
    assignment_path: Path | None = None,
) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
    #Une capturas, asignaciones y clima; separa filas etiquetadas y objetivo
    quality = QualityReport("ml")
    assignments = _load_assignments(assignment_path or ASSIGNMENT_PATH, quality)
    valid_polygon_ids = set(assignments)
    source_features = _source_feature_headers()

    capture_rows: list[dict[str, object]] = []
    sensors: set[str] = set()
    climate_cache: dict[int, dict[str, dict[str, float]]] = {}
    quality_rows: list[dict[str, object]] = []

    for source_name, dataset_path in DATASETS.items():
        id_header = _find_header(
            list(_source_feature_headers_for_path(dataset_path)), "ID_POLIGONO"
        )
        for year in DATASET_YEARS[source_name]:
            headers, observations, source_quality = read_year_observations(
                dataset_path, valid_polygon_ids, year
            )
            quality_rows.extend(source_quality.records())
            sensor_header = _find_header(headers, "sensor")
            date_header = _find_header(headers, "fecha_captura")

            observation_ids = {
                clean_text(row[id_header])
                for row in observations
                if clean_text(row[id_header]) in assignments
            }
            if observation_ids and year not in climate_cache:
                climate_cache[year] = extract_climate_features(year, valid_polygon_ids)

            for row in observations:
                polygon_id = clean_text(row[id_header])
                assignment = assignments[polygon_id]
                capture_date = date.fromisoformat(row[date_header])
                sensor = clean_text(row[sensor_header])
                if sensor:
                    sensors.add(sensor)

                features: dict[str, object] = {
                    f"{source_name.upper()}__{field}": row.get(field, "")
                    for field in source_features[source_name]
                    if field != TARGET_COLUMN
                }
                features.update(
                    {
                        "AREA_HA": assignment["AREA_HA"],
                        "ANIO_CAPTURA": capture_date.year,
                        "MES_CAPTURA": capture_date.month,
                        "DIA_DEL_ANIO": capture_date.timetuple().tm_yday,
                    }
                )
                climate = climate_cache.get(year, {}).get(polygon_id, {})
                features.update(
                    {
                        f"CLIMA__{column}": value
                        for column, value in climate.items()
                    }
                )
                capture_rows.append(
                    {
                        "ID_POLIGONO": polygon_id,
                        "PARTICION": assignment["PARTICION"],
                        "FUENTE": source_name.upper(),
                        "FECHA_CAPTURA": capture_date.isoformat(),
                        TARGET_COLUMN: assignment[TARGET_COLUMN],
                        "_SENSOR": sensor,
                        "_FEATURES": features,
                    }
                )

    feature_headers = [
        "AREA_HA",
        "ANIO_CAPTURA",
        "MES_CAPTURA",
        "DIA_DEL_ANIO",
    ]
    feature_headers.extend(
        f"{source_name.upper()}__{field}"
        for source_name, fields in source_features.items()
        for field in fields
        if field != TARGET_COLUMN
    )
    feature_headers.extend(
        f"CLIMA__{column}" for column in climate_column_names()
    )
    source_columns = [_category_column("FUENTE", name) for name in DATASETS]
    sensor_columns = sorted(_category_column("SENSOR", sensor) for sensor in sensors)
    feature_headers.extend(source_columns + sensor_columns)
    output_headers = list(METADATA_COLUMNS) + feature_headers
    if len(output_headers) != len(set(output_headers)):
        raise ValueError("Se generaron nombres de columna duplicados para ML.")

    normalized_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for capture in capture_rows:
        row: dict[str, object] = {
            column: capture[column] for column in METADATA_COLUMNS
        }
        features = capture["_FEATURES"]
        row.update({header: features.get(header, "") for header in feature_headers})
        row[_category_column("FUENTE", str(capture["FUENTE"]).casefold())] = 1
        sensor = str(capture["_SENSOR"])
        if sensor:
            row[_category_column("SENSOR", sensor)] = 1
        normalized_rows.append(row)
        if capture["PARTICION"] == "ENTRENAMIENTO" and capture[TARGET_COLUMN] != "":
            training_rows.append(row)
        elif capture["PARTICION"] == "PREDICCION":
            prediction_rows.append(row)

    return output_headers, normalized_rows, quality.records() + quality_rows

def _write_rows(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=headers, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

def run_machine_learning_export() -> tuple[Path, Path, Path]:
    #Escribe las tablas longitudinal, entrenamiento, predicción y calidad
    headers, rows, quality_rows = build_machine_learning_rows()
    evolution_path = OUTPUT_FOLDER / "ml_evolucion_parcelas_2022_2025.csv"
    training_path = OUTPUT_FOLDER / "ml_entrenamiento_etiquetado.csv"
    prediction_path = OUTPUT_FOLDER / "ml_prediccion_sin_etiqueta.csv"
    _write_rows(evolution_path, headers, rows)
    _write_rows(
        training_path,
        headers,
        [
            row
            for row in rows
            if row["PARTICION"] == "ENTRENAMIENTO"
            and row[TARGET_COLUMN] != ""
        ],
    )
    _write_rows(
        prediction_path,
        headers,
        [row for row in rows if row["PARTICION"] == "PREDICCION"],
    )
    print(f"Capturas para evolución y análisis: {len(rows)}")
    print(f"Capturas etiquetadas para entrenamiento: {training_path}")
    print(f"Capturas para predecir rendimiento: {prediction_path}")
    print(f"Datos temporales por parcela: {evolution_path}")
    print(f"Incidencias de validación detectadas: {len(quality_rows)}")
    return training_path, prediction_path, evolution_path

def _source_feature_headers_for_path(dataset_path: Path) -> list[str]:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as source:
        headers = next(csv.reader(source), None)
    if headers is None:
        raise ValueError(f"El archivo no contiene encabezados: {dataset_path}")
    return headers