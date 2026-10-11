#Carga, validación y preparación de datos AgroCebada

from __future__ import annotations
import csv
import hashlib
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import rasterio
import shapefile
from rasterio.features import geometry_mask, geometry_window
from rasterio.warp import transform_geom
from analisis.config import (
    ASSIGNMENT_PATH,
    EVOLUTION_PATH,
    OUTPUT,
    TARGET,
    TOPOGRAPHY_ROOT,
    YEARS,
)
from config import PROJECT_ROOT, SHAPEFILE_PATH
from procesamiento.data_normalization import normalize_text

__all__ = [
    "_copy_and_hash_sources",
    "_load_assignments",
    "_make_annual_parcel_features",
    "_number",
    "_read_spatial_target_data",
    "_write_iqr_outlier_review",
    "_write_parcel_features",
    "_write_traceability",
]

def _number(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None

def _feature_columns(headers: list[str]) -> list[str]:
    excluded = {
        "ID_POLIGONO",
        "PARTICION",
        "FUENTE",
        "FECHA_CAPTURA",
        TARGET,
        "ANIO_CAPTURA",
        "MES_CAPTURA",
        "DIA_DEL_ANIO",
    }
    return [
        header
        for header in headers
        if header
        and header not in excluded
        and not header.startswith(("FUENTE_", "SENSOR_"))
        and header.startswith(("BASICO__", "PRO__", "CLIMA__"))
    ]

def _make_annual_parcel_features() -> tuple[
    list[str],
    dict[str, dict[str, Any]],
    dict[tuple[str, int], dict[str, float]],
    dict[str, tuple[float, float]],
    dict[str, dict[str, float]],
    dict[str, Any],
]:
    """Audita la tabla longitudinal y resume predictores por parcela y año."""
    if not EVOLUTION_PATH.is_file():
        raise FileNotFoundError(
            f"No existe la base procesada esperada: {EVOLUTION_PATH}. "
            "Ejecute primero el lector ML existente."
        )

    annual_sums: dict[tuple[str, int], np.ndarray] = {}
    annual_counts: dict[tuple[str, int], np.ndarray] = {}
    targets: dict[str, set[float]] = defaultdict(set)
    partitions: dict[str, set[str]] = defaultdict(set)
    audit_nonmissing: Counter[str] = Counter()
    audit_invalid_numeric: Counter[str] = Counter()
    seen_rows: set[bytes] = set()
    seen_observation_keys: set[tuple[str, str, str, str]] = set()
    duplicate_rows = 0
    duplicate_observation_keys = 0
    row_count = 0
    with EVOLUTION_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        headers = reader.fieldnames
        if not headers:
            raise ValueError(f"La base procesada no tiene encabezados: {EVOLUTION_PATH}")
        feature_names = _feature_columns(headers)
        if not feature_names:
            raise ValueError("No se encontraron columnas numéricas para analizar.")
        feature_index = {name: index for index, name in enumerate(feature_names)}
        numeric_columns = set(feature_names) | {
            "AREA_HA",
            TARGET,
            "ANIO_CAPTURA",
            "MES_CAPTURA",
            "DIA_DEL_ANIO",
        }

        for row in reader:
            row_count += 1
            fingerprint = hashlib.blake2b(
                "\x1f".join(row.get(name, "") or "" for name in headers).encode(
                    "utf-8"
                ),
                digest_size=16,
            ).digest()
            if fingerprint in seen_rows:
                duplicate_rows += 1
            else:
                seen_rows.add(fingerprint)

            for name in headers:
                value = row.get(name, "")
                if value is None or not value.strip():
                    continue
                audit_nonmissing[name] += 1
                if name in numeric_columns and _number(value) is None:
                    audit_invalid_numeric[name] += 1

            polygon_id = (row.get("ID_POLIGONO") or "").strip()
            if not polygon_id:
                continue
            sensor = "|".join(
                name
                for name in headers
                if name.startswith("SENSOR_") and _number(row.get(name)) == 1
            )
            observation_key = (
                polygon_id,
                row.get("FECHA_CAPTURA") or "",
                row.get("FUENTE") or "",
                sensor,
            )
            if observation_key in seen_observation_keys:
                duplicate_observation_keys += 1
            else:
                seen_observation_keys.add(observation_key)
            partition = (row.get("PARTICION") or "").strip().upper()
            if partition:
                partitions[polygon_id].add(partition)
            target_value = _number(row.get(TARGET))
            if target_value is not None:
                targets[polygon_id].add(target_value)
            year_value = _number(row.get("ANIO_CAPTURA"))
            if year_value is None or int(year_value) not in YEARS:
                continue
            year = int(year_value)
            key = (polygon_id, year)
            if key not in annual_sums:
                annual_sums[key] = np.zeros(len(feature_names), dtype=float)
                annual_counts[key] = np.zeros(len(feature_names), dtype=np.int32)
            sums = annual_sums[key]
            counts = annual_counts[key]
            for name, index in feature_index.items():
                numeric_value = _number(row.get(name))
                if numeric_value is not None:
                    sums[index] += numeric_value
                    counts[index] += 1

    annual_features: dict[tuple[str, int], dict[str, float]] = {}
    parcel_year_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, sums in annual_sums.items():
        counts = annual_counts[key]
        values: dict[str, float] = {}
        for index, name in enumerate(feature_names):
            if counts[index]:
                value = float(sums[index] / counts[index])
                values[name] = value
                parcel_year_values[key[0]][name].append(value)
        annual_features[key] = values

    parcel_features: dict[str, dict[str, Any]] = {}
    for polygon_id, by_feature in parcel_year_values.items():
        summarized: dict[str, Any] = {}
        for source_name, values in by_feature.items():
            summarized[f"HISTORICO_PROMEDIO__{source_name}"] = float(
                np.mean(values)
            )
        parcel_features[polygon_id] = summarized

    for polygon_id, values in targets.items():
        if len(values) > 1:
            raise ValueError(
                f"El rendimiento cambia entre capturas de la parcela {polygon_id}: "
                f"{sorted(values)}. Revise la etiqueta objetivo antes de modelar."
            )
        parcel_features.setdefault(polygon_id, {})
        parcel_features[polygon_id][TARGET] = next(iter(values), "")

    for polygon_id, values in partitions.items():
        if len(values) != 1:
            raise ValueError(
                f"Partición inconsistente para ID_POLIGONO={polygon_id}: "
                f"{sorted(values)}."
            )
        parcel_features.setdefault(polygon_id, {})
        parcel_features[polygon_id]["PARTICION"] = next(iter(values))

    for polygon_id, values in parcel_features.items():
        values.setdefault("PARTICION", "")
        values.setdefault(TARGET, "")
        values.setdefault("AREA_HA", "")

    columns = [
        f"HISTORICO_PROMEDIO__{name}"
        for name in feature_names
        if any(
            f"HISTORICO_PROMEDIO__{name}" in row
            for row in parcel_features.values()
        )
    ]
    audit = {
        "headers": headers,
        "source_feature_columns": feature_names,
        "row_count": row_count,
        "duplicate_rows": duplicate_rows,
        "duplicate_observation_keys": duplicate_observation_keys,
        "nonmissing": audit_nonmissing,
        "invalid_numeric": audit_invalid_numeric,
    }
    return (
        columns,
        parcel_features,
        annual_features,
        _load_polygon_centroids(),
        _extract_topography(set(parcel_features)),
        audit,
    )

def _load_polygon_centroids() -> dict[str, tuple[float, float]]:
    reader = shapefile.Reader(str(SHAPEFILE_PATH))
    try:
        field_names = [field[0] for field in reader.fields[1:]]
        matches = [
            index
            for index, name in enumerate(field_names)
            if normalize_text(name).replace(" ", "_") in {"id_poligono", "id_poligon"}
        ]
        if len(matches) != 1:
            raise ValueError(
                "No se identificó una columna única ID_POLIGONO en el shapefile."
            )
        centroids: dict[str, tuple[float, float]] = {}
        for item in reader.iterShapeRecords():
            polygon_id = str(item.record[matches[0]]).strip()
            bounds = item.shape.bbox
            if len(bounds) != 4:
                raise ValueError(f"Geometría sin límites válidos para {polygon_id}.")
            if polygon_id in centroids:
                raise ValueError(f"ID_POLIGONO duplicado en el shapefile: {polygon_id}.")
            centroids[polygon_id] = (
                (float(bounds[0]) + float(bounds[2])) / 2,
                (float(bounds[1]) + float(bounds[3])) / 2,
            )
        return centroids
    finally:
        reader.close()

def _extract_topography(
    polygon_ids: set[str],
) -> dict[str, dict[str, float]]:
    """Calcula elevación y pendiente medias por parcela desde rásteres fuente."""
    prj_path = SHAPEFILE_PATH.with_suffix(".prj")
    if not prj_path.is_file():
        raise FileNotFoundError(f"No se encontró CRS del shapefile: {prj_path}")
    source_crs = prj_path.read_text(encoding="utf-8-sig").strip()
    if not source_crs:
        raise ValueError(f"El CRS del shapefile está vacío: {prj_path}")

    geometries: dict[str, dict[str, Any]] = {}
    reader = shapefile.Reader(str(SHAPEFILE_PATH))
    try:
        field_names = [field[0] for field in reader.fields[1:]]
        matches = [
            index
            for index, name in enumerate(field_names)
            if normalize_text(name).replace(" ", "_") in {"id_poligono", "id_poligon"}
        ]
        if len(matches) != 1:
            raise ValueError("No se pudo resolver ID_POLIGONO en la capa espacial.")
        id_index = matches[0]
        for item in reader.iterShapeRecords():
            polygon_id = str(item.record[id_index]).strip()
            if polygon_id not in polygon_ids:
                continue
            if item.shape.shapeType == shapefile.NULL:
                raise ValueError(f"La parcela {polygon_id} no tiene geometría.")
            geometries[polygon_id] = item.shape.__geo_interface__
    finally:
        reader.close()

    missing_ids = polygon_ids - geometries.keys()
    if missing_ids:
        raise ValueError(
            "Faltan geometrías para parcelas: " + ", ".join(sorted(missing_ids))
        )

    rasters = (
        (
            "TOPOGRAFIA__elevacion_media_m_snm",
            TOPOGRAPHY_ROOT / "Elevacion_INEGI_CEM4_120m.tif",
        ),
        (
            "TOPOGRAFIA__pendiente_media_grados",
            TOPOGRAPHY_ROOT / "Pendiente_INEGI_CEM4_120m_grados.tif",
        ),
    )
    result = {polygon_id: {} for polygon_id in polygon_ids}
    for output_name, raster_path in rasters:
        if not raster_path.is_file():
            raise FileNotFoundError(f"No se encontró el ráster topográfico: {raster_path}")
        with rasterio.open(raster_path) as raster:
            if raster.crs is None:
                raise ValueError(f"El ráster no declara CRS: {raster_path}")
            if raster.nodata is None:
                raise ValueError(f"El ráster no declara NoData: {raster_path}")
            for polygon_id, geometry in geometries.items():
                projected_geometry = transform_geom(
                    source_crs, raster.crs, geometry
                )
                window = geometry_window(raster, [projected_geometry])
                data = raster.read(1, window=window, masked=True)
                inside = geometry_mask(
                    [projected_geometry],
                    out_shape=data.shape,
                    transform=raster.window_transform(window),
                    all_touched=True,
                    invert=True,
                )
                valid = inside & ~np.ma.getmaskarray(data) & np.isfinite(data.data)
                values = data.data[valid]
                if values.size == 0:
                    raise ValueError(
                        f"No hay píxeles topográficos válidos en {polygon_id}, "
                        f"ráster {raster_path.name}."
                    )
                result[polygon_id][output_name] = float(values.mean())
    return result

def _write_iqr_outlier_review(
    parcel_data: dict[str, dict[str, Any]],
    feature_names: list[str],
    topography: dict[str, dict[str, float]],
) -> None:
    candidates = [
        ("RENDIMIENTO_T_2025_T_HA", TARGET, "t/ha"),
        ("TOPOGRAFIA__elevacion_media_m_snm", "TOPOGRAFIA__elevacion_media_m_snm", "m s.n.m."),
        ("TOPOGRAFIA__pendiente_media_grados", "TOPOGRAFIA__pendiente_media_grados", "grados"),
    ]
    for name in feature_names:
        if not name.startswith(
            (
                "HISTORICO_PROMEDIO__BASICO__",
                "HISTORICO_PROMEDIO__PRO__",
                "HISTORICO_PROMEDIO__CLIMA__",
            )
        ):
            continue
        feature_name = name.lower()
        if "precipitacion_mm" in feature_name:
            unit = "mm/mes"
        elif "tmin_c" in feature_name or "tmax_c" in feature_name:
            unit = "°C"
        else:
            unit = "no indicada en los metadatos de la variable"
        candidates.append((name, name, unit))
    with (OUTPUT / "revision_valores_atipicos_iqr.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        headers = (
            "variable",
            "unidad",
            "n_observaciones",
            "q1",
            "q3",
            "limite_inferior_1_5_iqr",
            "limite_superior_1_5_iqr",
            "n_candidatos_atipicos",
            "ids_candidatos_atipicos",
            "accion",
        )
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        for label, source_name, unit in candidates:
            values_by_id: dict[str, float] = {}
            for polygon_id, record in parcel_data.items():
                value = _number(str(record.get(source_name, "")))
                if value is None and source_name in topography.get(polygon_id, {}):
                    value = topography[polygon_id][source_name]
                if value is not None:
                    values_by_id[polygon_id] = value
            if len(values_by_id) < 4:
                continue
            data = np.asarray(list(values_by_id.values()), dtype=float)
            q1, q3 = np.quantile(data, [0.25, 0.75])
            iqr = q3 - q1
            lower = float(q1 - 1.5 * iqr)
            upper = float(q3 + 1.5 * iqr)
            flagged = [
                polygon_id
                for polygon_id, value in values_by_id.items()
                if value < lower or value > upper
            ]
            writer.writerow(
                {
                    "variable": label,
                    "unidad": unit,
                    "n_observaciones": len(data),
                    "q1": float(q1),
                    "q3": float(q3),
                    "limite_inferior_1_5_iqr": lower,
                    "limite_superior_1_5_iqr": upper,
                    "n_candidatos_atipicos": len(flagged),
                    "ids_candidatos_atipicos": ";".join(sorted(flagged)),
                    "accion": "solo revisar; no eliminar ni corregir automáticamente",
                }
            )

def _write_parcel_features(
    path: Path,
    feature_names: list[str],
    parcel_data: dict[str, dict[str, Any]],
    topography: dict[str, dict[str, float]],
    assignments: dict[str, dict[str, str]],
) -> list[str]:
    topography_names = [
        "TOPOGRAFIA__elevacion_media_m_snm",
        "TOPOGRAFIA__pendiente_media_grados",
    ]
    output_headers = [
        "ID_POLIGONO",
        "PARTICION",
        "RENDIMIENTO_T_2025_T_HA",
        "AREA_HA",
        *feature_names,
        *topography_names,
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=output_headers)
        writer.writeheader()
        for polygon_id in sorted(parcel_data):
            source = parcel_data[polygon_id]
            assignment = assignments.get(polygon_id, {})
            row: dict[str, Any] = {
                "ID_POLIGONO": polygon_id,
                "PARTICION": source.get(
                    "PARTICION", assignment.get("CONJUNTO", "")
                ),
                "RENDIMIENTO_T_2025_T_HA": source.get(TARGET, ""),
                "AREA_HA": assignment.get("AREA_HA", source.get("AREA_HA", "")),
            }
            row.update(
                {
                    name: source.get(name, "")
                    for name in feature_names
                }
            )
            row.update(topography.get(polygon_id, {}))
            writer.writerow(row)
    return output_headers

def _load_assignments() -> dict[str, dict[str, str]]:
    assignments: dict[str, dict[str, str]] = {}
    with ASSIGNMENT_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"Tabla de parcelas sin encabezados: {ASSIGNMENT_PATH}")
        for row in reader:
            polygon_id = (row.get("ID_POLIGONO") or "").strip()
            if not polygon_id:
                continue
            if polygon_id in assignments:
                raise ValueError(f"ID_POLIGONO duplicado en asignaciones: {polygon_id}.")
            assignments[polygon_id] = row
    return assignments

def _copy_and_hash_sources() -> list[dict[str, Any]]:
    backup_folder = OUTPUT / "originales"
    backup_folder.mkdir(parents=True, exist_ok=True)
    paths = (
        PROJECT_ROOT / "Conjunto_datos_BASICO_AgroCebada2026.csv",
        PROJECT_ROOT / "Conjunto_datos_PRO_AgroCebada.csv",
        ASSIGNMENT_PATH,
    )
    records = []
    for source_path in paths:
        if not source_path.is_file():
            raise FileNotFoundError(f"No se encontró fuente original: {source_path}")
        digest = hashlib.sha256()
        with source_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        destination = backup_folder / source_path.name
        if destination.exists():
            existing_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if existing_digest != digest.hexdigest():
                raise ValueError(
                    f"La copia intacta existente difiere de la fuente: {destination}"
                )
        else:
            shutil.copy2(source_path, destination)
        records.append(
            {
                "archivo_fuente": str(source_path.relative_to(PROJECT_ROOT)),
                "copia_preservada": str(destination.relative_to(OUTPUT)),
                "bytes": source_path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return records

def _write_traceability(records: list[dict[str, Any]]) -> None:
    supplemental = (
        EVOLUTION_PATH,
        SHAPEFILE_PATH,
        SHAPEFILE_PATH.with_suffix(".shx"),
        SHAPEFILE_PATH.with_suffix(".dbf"),
        SHAPEFILE_PATH.with_suffix(".prj"),
        TOPOGRAPHY_ROOT / "Elevacion_INEGI_CEM4_120m.tif",
        TOPOGRAPHY_ROOT / "Pendiente_INEGI_CEM4_120m_grados.tif",
    )
    known = {item["archivo_fuente"] for item in records}
    for source_path in supplemental:
        if not source_path.is_file():
            raise FileNotFoundError(f"No se encontró fuente para trazabilidad: {source_path}")
        relative = str(source_path.relative_to(PROJECT_ROOT))
        if relative in known:
            continue
        digest = hashlib.sha256()
        with source_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        records.append(
            {
                "archivo_fuente": relative,
                "copia_preservada": "fuente original, sin copia adicional",
                "bytes": source_path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    headers = ("archivo_fuente", "copia_preservada", "bytes", "sha256")
    with (OUTPUT / "trazabilidad_archivos.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)

def _read_spatial_target_data() -> dict[str, float]:
    result: dict[str, float] = {}
    with ASSIGNMENT_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            polygon_id = (row.get("ID_POLIGONO") or "").strip()
            target_value = _number(row.get(TARGET))
            if polygon_id and target_value is not None:
                result[polygon_id] = target_value
    return result