import csv
from pathlib import Path
from config import OUTPUT_FOLDER, PROJECT_ROOT
from lecturas.csv_reader import (
    find_input_file,
    read_prediction_records,
    write_csv,
)
from procesamiento.climate_features import (
    climate_column_names,
    extract_climate_features,
)
from procesamiento.data_normalization import (
    QualityReport,
    clean_text,
    is_missing,
    normalize_date,
    normalize_number,
    normalize_text,
)

DATASETS = {
    "basico": PROJECT_ROOT / "Conjunto_datos_BASICO_AgroCebada2026.csv",
    "pro": PROJECT_ROOT / "Conjunto_datos_PRO_AgroCebada.csv",
}
DATASET_YEARS = {
    "basico": range(2022, 2026),
    "pro": (2025,),
}
TEXT_COLUMNS = {"id_poligono", "fecha_captura", "sensor"}

def _required_column(headers: list[str], expected_name: str) -> str:
    matches = [
        header
        for header in headers
        if normalize_text(header).replace(" ", "_") == expected_name.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Se esperaba exactamente una columna {expected_name}; "
            f"se encontraron {len(matches)}."
        )
    return matches[0]

def read_year_observations(
    dataset_path: Path,
    prediction_ids: set[str],
    year: int,
) -> tuple[list[str], list[dict[str, str]], QualityReport]:
    #Lee solo capturas del año y parcelas indicados, registrando problemas
    observations: list[dict[str, str]] = []
    quality = QualityReport(dataset_path.stem, year)

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.reader(source_file)
        headers = next(reader, None)
        if headers is None:
            raise ValueError(f"El archivo no contiene encabezados: {dataset_path}")
        id_column = _required_column(headers, "ID_POLIGONO")
        date_column = _required_column(headers, "fecha_captura")

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

            row = dict(zip(headers, values))
            for header in headers[len(values):]:
                row[header] = ""

            if is_missing(row.get(id_column)):
                quality.add(id_column, "faltante", row_number)
                continue
            polygon_id = clean_text(row[id_column])
            if polygon_id not in prediction_ids:
                continue
            if is_missing(row.get(date_column)):
                quality.add(date_column, "faltante", row_number)
                continue

            try:
                normalized_date = normalize_date(
                    row[date_column], date_column, row_number
                )
            except ValueError:
                quality.add(
                    date_column, "formato_invalido", row_number, row[date_column]
                )
                continue
            if normalized_date[:4] != str(year):
                continue

            normalized_row: dict[str, str] = {}
            for header in headers:
                value = row.get(header, "")
                if is_missing(value):
                    quality.add(header, "faltante", row_number)
                    normalized_row[header] = ""
                elif normalize_text(header).replace(" ", "_") == "fecha_captura":
                    normalized_row[header] = normalized_date
                elif normalize_text(header).replace(" ", "_") in TEXT_COLUMNS:
                    normalized_row[header] = clean_text(value)
                else:
                    try:
                        normalized_row[header] = normalize_number(value) or ""
                    except ValueError:
                        quality.add(header, "formato_invalido", row_number, value)
                        normalized_row[header] = ""
            observations.append(normalized_row)

    return headers, observations, quality

def run_dataset_reader(dataset_name: str, year: int) -> Path:
    """Enriquece capturas BASIC/PRO con clima y escribe la tabla anual."""
    dataset_name = dataset_name.casefold()
    if dataset_name not in DATASETS:
        raise ValueError(f"Conjunto desconocido: {dataset_name}. Use 'basico' o 'pro'.")
    if year not in DATASET_YEARS[dataset_name]:
        allowed_years = ", ".join(map(str, DATASET_YEARS[dataset_name]))
        raise ValueError(
            f"Año no disponible para {dataset_name}: {year}. "
            f"Años disponibles: {allowed_years}."
        )

    zone_headers, predictions = read_prediction_records(find_input_file())
    dataset_path = DATASETS[dataset_name]
    if not dataset_path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {dataset_path}")

    dataset_headers, observations, quality = read_year_observations(
        dataset_path, set(predictions), year
    )
    id_column = _required_column(dataset_headers, "ID_POLIGONO")
    climate_columns = climate_column_names()

    climate_features: dict[str, dict[str, float]] = {}
    observed_ids = {(row.get(id_column) or "").strip() for row in observations}
    if observed_ids:
        climate_features = extract_climate_features(year, observed_ids)

    output_headers = (
        zone_headers
        + [header for header in dataset_headers if header != id_column]
        + ["ANIO"]
        + climate_columns
    )
    if len(output_headers) != len(set(output_headers)):
        raise ValueError(
            "Hay nombres de columna duplicados entre la zona, el dataset y las variables climáticas."
        )

    output_rows: list[dict[str, object]] = []
    for observation in observations:
        polygon_id = (observation.get(id_column) or "").strip()
        output_row: dict[str, object] = {
            **predictions[polygon_id],
            **{
                header: observation.get(header, "")
                for header in dataset_headers
                if header != id_column
            },
            "ANIO": year,
            **climate_features[polygon_id],
        }
        output_rows.append(output_row)

    output_path = OUTPUT_FOLDER / f"lectura_conjunto_de_datos_{dataset_name}{year}.csv"
    write_csv(output_path, output_headers, output_rows)
    quality_path = OUTPUT_FOLDER / f"calidad_{dataset_name}{year}.csv"
    quality.write(quality_path)

    print(f"Conjunto: {dataset_name.upper()} | año: {year}")
    print(f"Observaciones de predicción: {len(output_rows)}")
    print(f"Polígonos distintos: {len(observed_ids)}")
    if not output_rows:
        print("No hay capturas para este conjunto y año; se generó un CSV con encabezados.")
    print(f"CSV generado: {output_path}")
    print(f"Reporte de calidad: {quality_path}")
    return output_path