from collections.abc import Iterable
from pathlib import Path
import numpy as np
import rasterio
import shapefile
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from config import PROJECT_ROOT, SHAPEFILE_PATH
from procesamiento.data_normalization import normalize_text

CLIMATE_SOURCES = (
    (
        "precipitacion_mm",
        PROJECT_ROOT / "Reto_AgroCebada_CHIRPS_Precipitacion_2022_2025",
        "Precipitacion_mensual",
        "PREC",
    ),
    (
        "tmin_c",
        PROJECT_ROOT / "Reto_AgroCebada_Temperatura_2022_2025",
        "Tmin",
        "Tmin",
    ),
    (
        "tmax_c",
        PROJECT_ROOT / "Reto_AgroCebada_Temperatura_2022_2025",
        "Tmax",
        "Tmax",
    ),
)

def climate_column_names() -> list[str]:
    return [
        f"{variable}_{month:02d}"
        for variable, _, _, _ in CLIMATE_SOURCES
        for month in range(1, 13)
    ]

def load_polygon_geometries(
    polygon_ids: Iterable[str],
    shapefile_path: Path = SHAPEFILE_PATH,
) -> tuple[dict[str, dict[str, object]], str]:
    #Recupera geometrías de parcelas y su CRS tras validar sus identificadores
    required_ids = set(polygon_ids)
    if not required_ids:
        return {}, ""

    projection_path = shapefile_path.with_suffix(".prj")
    if not projection_path.is_file():
        raise FileNotFoundError(f"No se encontró la proyección de la capa: {projection_path}")
    source_crs = projection_path.read_text(encoding="utf-8-sig").strip()
    if not source_crs:
        raise ValueError(f"El archivo de proyección está vacío: {projection_path}")

    reader = shapefile.Reader(str(shapefile_path))
    try:
        fields = [field[0] for field in reader.fields[1:]]
        id_indexes = [
            index
            for index, field in enumerate(fields)
            if normalize_text(field) in {"id_poligono", "id_poligon"}
        ]
        if len(id_indexes) != 1:
            raise ValueError(
                "La capa debe tener una sola columna ID_POLIGONO "
                "(en DBF puede aparecer truncada como ID_POLIGON)."
            )

        id_index = id_indexes[0]
        geometries: dict[str, dict[str, object]] = {}
        for shape_record in reader.iterShapeRecords():
            polygon_id = str(shape_record.record[id_index]).strip()
            if polygon_id not in required_ids:
                continue
            if shape_record.shape.shapeType == shapefile.NULL:
                raise ValueError(f"El polígono {polygon_id} no tiene geometría.")
            if polygon_id in geometries:
                raise ValueError(f"ID_POLIGONO duplicado en la capa espacial: {polygon_id}")
            geometries[polygon_id] = shape_record.shape.__geo_interface__
    finally:
        reader.close()

    missing_ids = required_ids - geometries.keys()
    if missing_ids:
        raise ValueError(
            "No se encontraron geometrías para estos ID_POLIGONO: "
            + ", ".join(sorted(missing_ids))
        )
    return geometries, source_crs

def zonal_mean(
    raster: rasterio.io.DatasetReader,
    geometry: dict[str, object],
    geometry_crs: str,
    polygon_id: str,
) -> float:
    #Calcula la media de celdas ráster válidas que intersectan un polígono
    if raster.crs is None:
        raise ValueError(f"El ráster no declara CRS: {raster.name}")
    if raster.nodata is None:
        raise ValueError(f"El ráster no declara NoData: {raster.name}")

    raster_geometry = transform_geom(geometry_crs, raster.crs, geometry)
    inside = geometry_mask(
        [raster_geometry],
        out_shape=(raster.height, raster.width),
        transform=raster.transform,
        all_touched=True,
        invert=True,
    )
    if not inside.any():
        raise ValueError(
            f"El polígono {polygon_id} no intersecta el ráster {raster.name}."
        )

    data = np.ma.masked_invalid(raster.read(1, masked=True))
    values = data[inside].compressed()
    if values.size == 0:
        raise ValueError(
            f"El polígono {polygon_id} solo intersecta celdas NoData en {raster.name}."
        )
    return float(values.mean())

def extract_climate_features(
    year: int,
    polygon_ids: Iterable[str],
    shapefile_path: Path = SHAPEFILE_PATH,
) -> dict[str, dict[str, float]]:
    #Extrae por parcela los 12 meses de precipitación, Tmin y Tmax del año
    geometries, geometry_crs = load_polygon_geometries(polygon_ids, shapefile_path)
    features = {polygon_id: {} for polygon_id in geometries}

    for variable, source_root, subdirectory, prefix in CLIMATE_SOURCES:
        for month in range(1, 13):
            raster_path = (
                source_root / subdirectory / str(year) / f"{prefix}_{year}_{month:02d}.tif"
            )
            if not raster_path.is_file():
                raise FileNotFoundError(f"No se encontró el ráster climático: {raster_path}")

            column_name = f"{variable}_{month:02d}"
            with rasterio.open(raster_path) as raster:
                for polygon_id, geometry in geometries.items():
                    features[polygon_id][column_name] = zonal_mean(
                        raster, geometry, geometry_crs, polygon_id
                    )
    return features