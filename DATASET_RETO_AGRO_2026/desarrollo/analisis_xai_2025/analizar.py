"""Auditoria reproducible de calidad, curvas y conceptos candidatos AgroCebada 2025.

Consume las fuentes originales y las tablas parcelarias existentes; no modifica el
pipeline original ni usa rendimiento para construir curvas o grupos descriptivos.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


HERE = Path(__file__).resolve().parent
DEV = HERE.parent
RAW = DEV / "datos" / "entradas"
PROCESSED = DEV / "cebada_datos_baseline_1" / "data_processed"
OUT = HERE / "resultados"
FIG = OUT / "graficas"
SEED = 2026
THRESHOLDS = {"Sentinel-2": 5.0, "Landsat": 5.0, "Planet": 20.0}
STAGES = {
    "pre_ciclo_calendario": (91, 151),
    "vegetativo_calendario": (152, 212),
    "pico_calendario": (213, 273),
    "senescencia_calendario": (274, 304),
}
BLOCK_COUNTS = (4, 6, 8)
RIDGE_ALPHA = 10.0


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_observations() -> pd.DataFrame:
    basics_path = RAW / "Conjunto_datos_BASICO_AgroCebada2026.csv"
    planet_path = RAW / "Conjunto_datos_PRO_AgroCebada.csv"
    frames = []
    for path, expected_sensor in (
        (basics_path, None),
        (planet_path, "Planet"),
    ):
        data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        data["source_file"] = path.name
        data["source_row"] = np.arange(2, len(data) + 2)
        data["date_raw"] = data["fecha_captura"].astype("string")
        data["date"] = pd.to_datetime(
            data["fecha_captura"], dayfirst=True, errors="coerce"
        )
        if expected_sensor:
            recorded = data["sensor"].dropna().astype(str).str.strip()
            unexpected = sorted(set(recorded) - {expected_sensor})
            if unexpected:
                raise ValueError(
                    f"{path.name} reporta sensores inesperados: {unexpected}"
                )
        data["cloud_pct"] = pd.to_numeric(
            data["porcentaje_nubosidad"], errors="coerce"
        )
        for col in data.columns:
            if col.endswith("_promedio") or col.endswith("_std"):
                data[col] = pd.to_numeric(data[col], errors="coerce")
        frames.append(data)
    observations = pd.concat(frames, ignore_index=True, sort=False).copy()
    observations["doy"] = observations["date"].dt.dayofyear
    observations["year"] = observations["date"].dt.year
    observations["month"] = observations["date"].dt.month
    observations["cloud_limit_pct"] = observations["sensor"].map(THRESHOLDS)
    observations["cloud_pass"] = (
        observations["cloud_pct"].notna()
        & (observations["cloud_pct"] <= observations["cloud_limit_pct"])
    )
    observations["signal_variable"] = observations["sensor"].map(
        {
            "Sentinel-2": "ndvi_promedio",
            "Planet": "ndvi_promedio",
            "Landsat": "vi6t_promedio",
        }
    )
    observations["ndvi_valid"] = observations.get(
        "ndvi_promedio", pd.Series(index=observations.index, dtype=float)
    ).notna()
    observations["signal_valid"] = False
    for variable in observations["signal_variable"].dropna().unique():
        observations.loc[
            observations["signal_variable"].eq(variable), "signal_valid"
        ] = observations.loc[
            observations["signal_variable"].eq(variable), variable
        ].notna()
    observations["valid_core"] = observations["signal_valid"] & observations["cloud_pass"]
    observations["stage"] = "fuera_ventanas"
    for stage, (start, end) in STAGES.items():
        observations.loc[observations["doy"].between(start, end), "stage"] = stage
    return observations


def summarize_quality(
    data: pd.DataFrame, group_cols: list[str], denominator: pd.Series | None = None
) -> pd.DataFrame:
    grouped = data.groupby(group_cols, dropna=False, observed=True)
    result = grouped.agg(
        filas=("ID_POLIGONO", "size"),
        parcelas=("ID_POLIGONO", "nunique"),
        ndvi_valido=("ndvi_valid", "sum"),
        ndvi_valido_pct=("ndvi_valid", "mean"),
        senal_valida=("signal_valid", "sum"),
        senal_valida_pct=("signal_valid", "mean"),
        obs_nucleo_valida=("valid_core", "sum"),
        obs_nucleo_valida_pct=("valid_core", "mean"),
        nubosidad_mediana_pct=("cloud_pct", "median"),
        nubosidad_media_pct=("cloud_pct", "mean"),
        nubosidad_faltante_pct=("cloud_pct", lambda s: s.isna().mean()),
        cloud_pass_pct=("cloud_pass", "mean"),
    ).reset_index()
    valid_parcels = (
        data.loc[data["valid_core"]]
        .groupby(group_cols, dropna=False, observed=True)["ID_POLIGONO"]
        .nunique()
        .rename("parcelas_con_senal_nube_admisible")
        .reset_index()
    )
    result = result.merge(valid_parcels, on=group_cols, how="left")
    result["parcelas_con_senal_nube_admisible"] = (
        result["parcelas_con_senal_nube_admisible"].fillna(0).astype(int)
    )
    if denominator is not None:
        key = group_cols[0]
        totals = denominator.rename("parcelas_denominador")
        result = result.merge(totals, left_on=key, right_index=True, how="left")
        result["cobertura_parcelas_pct"] = (
            result["parcelas"] / result["parcelas_denominador"]
        )
        result["cobertura_senal_nube_admisible_pct"] = (
            result["parcelas_con_senal_nube_admisible"]
            / result["parcelas_denominador"]
        )
    return result


def quality_products(obs: pd.DataFrame, parcels: pd.DataFrame) -> None:
    year = obs.loc[obs["year"].eq(2025)].copy()
    municipalities = parcels.set_index("ID_POLIGONO")["Municipio"]
    year["Municipio"] = year["ID_POLIGONO"].map(municipalities).fillna("SIN_MUNICIPIO")
    denominator = parcels.groupby("Municipio")["ID_POLIGONO"].nunique()
    write_csv(summarize_quality(year, ["sensor"]), "calidad_por_sensor_2025.csv")
    write_csv(
        summarize_quality(year, ["Municipio", "sensor"], denominator),
        "calidad_por_municipio_sensor_2025.csv",
    )
    write_csv(
        summarize_quality(year, ["stage", "sensor"]),
        "calidad_por_etapa_calendario_sensor_2025.csv",
    )
    monthly_rows = []
    for (sensor, month), group in year.groupby(["sensor", "month"], observed=True):
        monthly_rows.append(
            {
                "sensor": sensor,
                "month": month,
                "filas": len(group),
                "parcelas": group.ID_POLIGONO.nunique(),
                "parcelas_con_ndvi": group.loc[group.ndvi_valid, "ID_POLIGONO"].nunique(),
                "ndvi_valido_pct": group.ndvi_valid.mean(),
                "parcelas_con_senal": group.loc[
                    group.signal_valid, "ID_POLIGONO"
                ].nunique(),
                "parcelas_con_senal_nube_admisible": group.loc[
                    group.valid_core, "ID_POLIGONO"
                ].nunique(),
                "nubosidad_mediana_pct": group.cloud_pct.median(),
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    monthly["cobertura_parcelas_con_captura_pct"] = monthly["parcelas"] / len(parcels)
    monthly["cobertura_senal_nube_admisible_pct"] = (
        monthly["parcelas_con_senal_nube_admisible"] / len(parcels)
    )
    write_csv(monthly, "cobertura_mensual_sensor_2025.csv")

    all_monthly_rows = []
    all_dates = obs.loc[obs["date"].notna()]
    for (year_value, sensor, month), group in all_dates.groupby(
        ["year", "sensor", "month"], observed=True
    ):
        all_monthly_rows.append(
            {
                "anio": year_value,
                "sensor": sensor,
                "mes": month,
                "filas": len(group),
                "parcelas_con_captura": group.ID_POLIGONO.nunique(),
                "parcelas_con_senal_valida": group.loc[
                    group.signal_valid, "ID_POLIGONO"
                ].nunique(),
                "parcelas_con_senal_y_nube_admisible": group.loc[
                    group.valid_core, "ID_POLIGONO"
                ].nunique(),
                "nubosidad_mediana_pct": group.cloud_pct.median(),
            }
        )
    write_csv(pd.DataFrame(all_monthly_rows), "cobertura_mensual_sensor_2022_2025.csv")

    capture = (
        year.groupby(["ID_POLIGONO", "sensor", "stage"], observed=True)
        .agg(
            filas=("ID_POLIGONO", "size"),
            observaciones_ndvi_validas=("ndvi_valid", "sum"),
            observaciones_senal_valida=("signal_valid", "sum"),
            observaciones_nucleo_validas=("valid_core", "sum"),
            nubosidad_mediana_pct=("cloud_pct", "median"),
            n_fechas_distintas=("date", "nunique"),
        )
        .reset_index()
        .rename(columns={"stage": "etapa_calendario"})
    )
    all_keys = pd.MultiIndex.from_product(
        [
            parcels.ID_POLIGONO,
            sorted(obs.sensor.dropna().unique()),
            sorted(year.stage.unique()),
        ],
        names=["ID_POLIGONO", "sensor", "etapa_calendario"],
    ).to_frame(index=False)
    capture = all_keys.merge(
        capture, on=["ID_POLIGONO", "sensor", "etapa_calendario"], how="left"
    )
    capture["Municipio"] = capture.ID_POLIGONO.map(municipalities).fillna("SIN_MUNICIPIO")
    count_cols = [
        "filas",
        "observaciones_ndvi_validas",
        "observaciones_senal_valida",
        "observaciones_nucleo_validas",
        "n_fechas_distintas",
    ]
    capture[count_cols] = capture[count_cols].fillna(0).astype(int)
    valid_gaps = []
    for (pid, sensor, stage), group in year.loc[year.valid_core].groupby(
        ["ID_POLIGONO", "sensor", "stage"], observed=True
    ):
        ordered = group.date.sort_values()
        gaps = ordered.diff().dt.days.dropna()
        valid_gaps.append(
            {
                "ID_POLIGONO": pid,
                "sensor": sensor,
                "etapa_calendario": stage,
                "max_gap_dias": gaps.max() if not gaps.empty else np.nan,
            }
        )
    capture = capture.merge(
        pd.DataFrame(valid_gaps),
        on=["ID_POLIGONO", "sensor", "etapa_calendario"],
        how="left",
    )
    write_csv(capture, "calidad_parcela_sensor_etapa_2025.csv")
    parcel_sensor = (
        year.groupby(["ID_POLIGONO", "sensor"], observed=True)
        .agg(
            capturas=("ID_POLIGONO", "size"),
            fechas=("date", "nunique"),
            observaciones_con_senal=("signal_valid", "sum"),
            observaciones_senal_nube_admisible=("valid_core", "sum"),
            nubosidad_mediana_pct=("cloud_pct", "median"),
            meses_con_senal=("month", lambda s: year.loc[s.index, "signal_valid"].groupby(s).any().sum()),
            meses_con_senal_nube_admisible=(
                "month",
                lambda s: year.loc[s.index, "valid_core"].groupby(s).any().sum(),
            ),
        )
        .reset_index()
    )
    parcel_sensor_keys = pd.MultiIndex.from_product(
        [parcels.ID_POLIGONO, sorted(obs.sensor.dropna().unique())],
        names=["ID_POLIGONO", "sensor"],
    ).to_frame(index=False)
    parcel_sensor = parcel_sensor_keys.merge(
        parcel_sensor, on=["ID_POLIGONO", "sensor"], how="left"
    )
    count_cols = [
        "capturas",
        "fechas",
        "observaciones_con_senal",
        "observaciones_senal_nube_admisible",
        "meses_con_senal",
        "meses_con_senal_nube_admisible",
    ]
    parcel_sensor[count_cols] = parcel_sensor[count_cols].fillna(0).astype(int)
    gap_rows = []
    for (pid, sensor), group in year.loc[year.valid_core].groupby(
        ["ID_POLIGONO", "sensor"], observed=True
    ):
        dates = group.date.dropna().sort_values().drop_duplicates()
        gaps = dates.diff().dt.days.dropna()
        gap_rows.append(
            {
                "ID_POLIGONO": pid,
                "sensor": sensor,
                "fecha_primera_senal_admisible": dates.min(),
                "fecha_ultima_senal_admisible": dates.max(),
                "brecha_mediana_dias": gaps.median() if not gaps.empty else np.nan,
                "brecha_maxima_dias": gaps.max() if not gaps.empty else np.nan,
            }
        )
    parcel_sensor = parcel_sensor.merge(
        pd.DataFrame(gap_rows), on=["ID_POLIGONO", "sensor"], how="left"
    )
    parcel_sensor["Municipio"] = parcel_sensor.ID_POLIGONO.map(municipalities)
    parcel_sensor.to_csv(
        OUT / "calidad_parcela_sensor_2025.csv", index=False, encoding="utf-8-sig"
    )

    by_sensor = []
    for sensor, g in obs.groupby("sensor", dropna=False):
        source_file = g["source_file"].iloc[0]
        raw = pd.read_csv(RAW / source_file, encoding="utf-8-sig", low_memory=False)
        numeric_cols = [
            c
            for c in raw.columns
            if c not in ("ID_POLIGONO", "fecha_captura", "sensor")
        ]
        for col in numeric_cols:
            values = pd.to_numeric(raw[col], errors="coerce")
            by_sensor.append(
                {
                    "sensor": sensor,
                    "variable": col,
                    "filas_fuente": len(raw),
                    "faltantes_pct_fuente": values.isna().mean(),
                    "faltantes_pct_2025": g.loc[g["year"].eq(2025), col].isna().mean()
                    if col in g
                    else 1.0,
                }
            )
    write_csv(pd.DataFrame(by_sensor), "faltantes_por_variable_sensor.csv")

    duplicate_key = obs.duplicated(["ID_POLIGONO", "date", "sensor"], keep=False)
    integrity = [
        ("filas_observadas_totales", len(obs), "Capturas concatenadas; no eliminar"),
        ("fechas_invalidas", int(obs["date"].isna().sum()), "Fecha no parseable"),
        ("ids_parcela_faltantes", int(obs["ID_POLIGONO"].isna().sum()), "Clave"),
        ("ids_fuera_capa_parcelas", int((~obs["ID_POLIGONO"].isin(parcels.ID_POLIGONO)).sum()), "Cruce espacial"),
        ("duplicados_id_fecha_sensor", int(duplicate_key.sum()), "Filas que pertenecen a claves repetidas"),
        ("filas_duplicadas_completas", int(obs.duplicated().sum()), "Todas las columnas iguales"),
        ("parcelas_geometria", parcels.ID_POLIGONO.nunique(), "Unicas en tabla espacial"),
        ("parcelas_con_observacion_2025", year.ID_POLIGONO.nunique(), "Alguna fuente / alguna fecha"),
    ]
    write_csv(
        pd.DataFrame(integrity, columns=["control", "valor", "interpretacion"]),
        "integridad_fuentes.csv",
    )
    label_rows = []
    for partition, group in parcels.groupby("CONJUNTO", dropna=False):
        label_rows.append(
            {
                "CONJUNTO": partition,
                "parcelas": group.ID_POLIGONO.nunique(),
                "rendimiento_disponible": int(group.RENDIMIENTO_T_HA.notna().sum()),
                "rendimiento_faltante": int(group.RENDIMIENTO_T_HA.isna().sum()),
                "min_t_ha": group.RENDIMIENTO_T_HA.min(),
                "mediana_t_ha": group.RENDIMIENTO_T_HA.median(),
                "max_t_ha": group.RENDIMIENTO_T_HA.max(),
                "fecha_ciclo_en_archivo": False,
            }
        )
    write_csv(pd.DataFrame(label_rows), "auditoria_etiquetas.csv")
    label_detail = parcels[
        [
            "ID_POLIGONO",
            "fila_fuente_asignacion",
            "AREA_HA",
            "CONJUNTO",
            "RENDIMIENTO_T_HA",
            "Municipio",
        ]
    ].copy()
    label_detail["fuente"] = "ID_area_rendimiento_70_30_Reto_AgroCebada.csv"
    label_detail["anio_ciclo_etiqueta_disponible"] = False
    label_detail.to_csv(
        OUT / "auditoria_etiquetas_por_parcela.csv",
        index=False,
        encoding="utf-8-sig",
    )
    year.to_csv(OUT / "observaciones_2025_trazables.csv", index=False, encoding="utf-8-sig")


def environment_products(parcels: pd.DataFrame) -> None:
    municipality = parcels.set_index("ID_POLIGONO")["Municipio"]
    climate = pd.read_csv(PROCESSED / "clima_mensual_parcela.csv")
    climate_2025 = climate.loc[climate.anio.eq(2025)].copy()
    climate_2025["Municipio"] = climate_2025.ID_POLIGONO.map(municipality)
    climate_2025["celda_clima"] = climate_2025.ID_POLIGONO.map(
        parcels.set_index("ID_POLIGONO")["celda_clima"]
    )
    climate_2025["fuente_archivo"] = "clima_mensual_parcela.csv"
    climate_2025["fuente_precipitacion"] = "CHIRPS"
    climate_2025["fuente_temperatura"] = "CHIRTS-ERA5"
    climate_2025.to_csv(
        OUT / "clima_mensual_trazable_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )
    monthly_count = climate_2025.groupby("ID_POLIGONO").mes.nunique()
    climate_2025["tmean"] = (
        pd.to_numeric(climate_2025.tmin, errors="coerce")
        + pd.to_numeric(climate_2025.tmax, errors="coerce")
    ) / 2
    april_oct = climate_2025.loc[climate_2025.mes.between(4, 10)]
    weather = april_oct.groupby("ID_POLIGONO").agg(
        prec_abr_oct_mm=("prec_mm", "sum"),
        tmin_abr_oct_c=("tmin", "mean"),
        tmax_abr_oct_c=("tmax", "mean"),
        tmean_abr_oct_c=("tmean", "mean"),
        meses_clima_validos=("prec_mm", "count"),
    )
    climate_per_parcel = pd.DataFrame({"meses_2025_disponibles": monthly_count}).join(weather)
    climate_per_parcel["Municipio"] = climate_per_parcel.index.map(municipality)
    climate_per_parcel.index.name = "ID_POLIGONO"
    climate_per_parcel.reset_index().to_csv(
        OUT / "clima_por_parcela_2025.csv", index=False, encoding="utf-8-sig"
    )
    climate_summary = climate_per_parcel.groupby("Municipio").agg(
        parcelas=("meses_2025_disponibles", "size"),
        meses_completos=("meses_2025_disponibles", lambda s: int(s.eq(12).sum())),
        cobertura_media_meses=("meses_2025_disponibles", "mean"),
        prec_abr_oct_mediana_mm=("prec_abr_oct_mm", "median"),
        tmean_abr_oct_mediana_c=("tmean_abr_oct_c", "median"),
    )
    climate_summary.reset_index().to_csv(
        OUT / "clima_resumen_municipio_2025.csv", index=False, encoding="utf-8-sig"
    )
    static = parcels.copy()
    static["celda_clima"] = static["celda_clima"].astype("string")
    topography_columns = [
        "ID_POLIGONO",
        "Municipio",
        "AREA_HA",
        "celda_clima",
        "elev_media",
        "elev_std",
        "elev_min",
        "elev_max",
        "elev_npix",
        "pend_media",
        "pend_std",
        "pend_min",
        "pend_max",
        "pend_npix",
    ]
    topography = static[topography_columns].copy()
    topography["fuente_archivo"] = "parcelas_base.csv"
    topography["fuente_topografia"] = "INEGI CEM4; elevacion/pendiente, 120 m"
    topography.to_csv(
        OUT / "topografia_por_parcela.csv", index=False, encoding="utf-8-sig"
    )
    static_rows = []
    for scope, group in [("TOTAL", static), *list(static.groupby("Municipio"))]:
        static_rows.append(
            {
                "ambito": scope,
                "parcelas": group.ID_POLIGONO.nunique(),
                "celdas_clima_05deg": group.celda_clima.nunique(),
                "elev_media_mediana_m": group.elev_media.median(),
                "elev_media_min_m": group.elev_media.min(),
                "elev_media_max_m": group.elev_media.max(),
                "pend_media_mediana_grados": group.pend_media.median(),
                "pend_media_min_grados": group.pend_media.min(),
                "pend_media_max_grados": group.pend_media.max(),
                "area_ha_mediana": group.AREA_HA.median(),
                "n_pix_elevacion_mediana": group.elev_npix.median(),
                "n_pix_pendiente_mediana": group.pend_npix.median(),
                "faltantes_topografia": int(
                    group[["elev_media", "pend_media"]].isna().any(axis=1).sum()
                ),
            }
        )
    write_csv(pd.DataFrame(static_rows), "topografia_y_escala_climatica.csv")


def index_stage_products(obs: pd.DataFrame) -> None:
    selected = {
        "Sentinel-2": (
            "ndvi_promedio", "evi_promedio", "lai_promedio", "fapar_promedio",
            "savi_promedio", "ndwi_promedio", "endwi_promedio", "msi_promedio",
            "emsi_promedio", "nddi_promedio", "enddi_promedio", "dswi2_promedio",
            "dswi3_promedio", "dswi4_promedio", "dswi5_promedio", "crc_promedio",
            "mcrc_promedio", "ndsvi_promedio", "srndi_promedio", "sti_promedio",
            "ndti_promedio", "mra_promedio",
        ),
        "Planet": ("ndvi_promedio", "evi_promedio", "lai_promedio", "msavi_promedio"),
        "Landsat": ("vi6t_promedio",),
    }
    year = obs.loc[obs.year.eq(2025)]
    rows = []
    for sensor, variables in selected.items():
        sensor_data = year.loc[year.sensor.eq(sensor)]
        for variable in variables:
            if variable not in sensor_data:
                continue
            for stage, group in sensor_data.groupby("stage", observed=True):
                values = group.loc[group.cloud_pass, variable].dropna()
                rows.append(
                    {
                        "sensor": sensor,
                        "variable": variable,
                        "etapa_calendario": stage,
                        "filas_fuente": len(group),
                        "filas_con_nube_admisible": int(group.cloud_pass.sum()),
                        "observaciones_validas": len(values),
                        "parcelas_con_observacion_valida": int(
                            group.loc[group.cloud_pass & group[variable].notna(), "ID_POLIGONO"].nunique()
                        ),
                        "faltantes_entre_filas_admisibles_pct": (
                            group.loc[group.cloud_pass, variable].isna().mean()
                            if group.cloud_pass.any()
                            else np.nan
                        ),
                        "mediana": values.median(),
                        "p10": values.quantile(0.10),
                        "p90": values.quantile(0.90),
                    }
                )
    write_csv(pd.DataFrame(rows), "indices_resumen_etapa_2025.csv")

    d = year.loc[
        year.sensor.eq("Sentinel-2")
        & year.cloud_pass
    ].copy()
    variables = (
        "ndvi_promedio", "evi_promedio", "lai_promedio",
        "ndwi_promedio", "msi_promedio", "nddi_promedio",
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, variable in zip(axes.flat, variables):
        monthly = d.loc[d[variable].notna()].groupby("month")[variable].median()
        ax.plot(monthly.index, monthly.values, marker="o")
        ax.set(title=variable.replace("_promedio", "").upper(), xlabel="Mes", ylabel="Valor fuente")
        ax.set_xticks(range(1, 13))
    fig.suptitle("Sentinel-2: medianas mensuales con nubosidad <=5%; escalas no equiparadas")
    fig.tight_layout()
    fig.savefig(FIG / "indices_relevantes_s2_2025.png", dpi=150)
    plt.close(fig)


def write_catalog() -> None:
    catalog = [
        {
            "concepto_candidato": "Vigor espectral",
            "variables": "NDVI, EVI; sensor y fecha siempre separados",
            "definicion_operativa": "Respuesta espectral del dosel/superficie agregada dentro del poligono; no es vigor fisiologico medido.",
            "evidencia": "Curvas 2025 completas por mes; S2-Planet NDVI correlacion 0.983 pero sesgo medio +0.081 Planet-S2.",
            "nivel_evidencia": "Moderado para señal espectral; bajo para interpretación fisiológica",
            "confianza": "Media",
            "limitaciones": "Índices y reflectancias dependen de sensor, procesado, mezcla suelo/cultivo y nubosidad.",
            "validacion_requerida": "Parcelas/fechas con cobertura visual, biomasa, clorofila o LAI de campo.",
        },
        {
            "concepto_candidato": "Cobertura vegetal",
            "variables": "NDVI, EVI, SAVI, MSAVI y heterogeneidad espacial disponible",
            "definicion_operativa": "Proxies espectrales de respuesta verde y suelo expuesto, no porcentaje de cobertura observado.",
            "evidencia": "Ascenso mensual de señales entre junio y agosto-septiembre y descenso posterior en curvas 2025.",
            "nivel_evidencia": "Bajo-moderado como proxy espectral",
            "confianza": "Media-baja",
            "limitaciones": "No hay fotos/transectos de cobertura ni definición comparable de los productos entre fuentes.",
            "validacion_requerida": "Cobertura fraccional por fotografía, cuadrantes o medición de campo sincronizada.",
        },
        {
            "concepto_candidato": "Dinámica fenológica",
            "variables": "Pico/DOY, amplitud, SOS/EOS a media amplitud, duración, AUC, pendientes y medias por ventana",
            "definicion_operativa": "Descriptores de forma temporal calculados de observaciones filtradas; las ventanas son calendáricas.",
            "evidencia": "Picos S2 en julio-agosto y Planet en agosto-septiembre; suavizado 3-meses desplaza el pico una mediana de un mes.",
            "nivel_evidencia": "Moderado para descriptores de curva; bajo para asignar una fase biológica",
            "confianza": "Media-baja",
            "limitaciones": "No hay fechas de siembra/emergencia/espigamiento/madurez; observaciones no uniformes y suavizado altera picos.",
            "validacion_requerida": "Calendario fenológico por parcela/cultivar/ciclo con visitas y criterio de etapa explícito.",
        },
        {
            "concepto_candidato": "Condición hídrica espectral",
            "variables": "NDWI, ENDWI, MSI, EMSI, NDDI, ENDDI, DSWI2-5",
            "definicion_operativa": "Indicadores espectrales asociados a agua/sequedad; ninguno mide agua del suelo ni estrés fisiológico.",
            "evidencia": "Cobertura de capturas y perfiles por etapa en indices_resumen_etapa_2025.csv.",
            "nivel_evidencia": "Exploratorio",
            "confianza": "Baja",
            "limitaciones": "Fórmulas/metadatos de índices no completamente verificables; confusión por suelo, estructura, atmósfera y sensor.",
            "validacion_requerida": "Humedad de suelo, potencial hídrico foliar, riego y síntomas observados en fechas de imagen.",
        },
        {
            "concepto_candidato": "Exposición climática",
            "variables": "CHIRPS precipitación mensual; CHIRTS-ERA5 Tmin/Tmax mensual",
            "definicion_operativa": "Contexto meteorológico de celda de 0.05 grados, no microclima ni estrés del cultivo.",
            "evidencia": "12/12 meses por parcela en 2025; 197 parcelas en 25 celdas climáticas.",
            "nivel_evidencia": "Alto para cobertura de la serie suministrada; moderado para exposición local",
            "confianza": "Media",
            "limitaciones": "Resolución aproximada de 5 km y clima parcelario repetido entre parcelas/celdas.",
            "validacion_requerida": "Comparar con estaciones locales; validar eventos/heladas y registros de manejo/irrigación.",
        },
        {
            "concepto_candidato": "Características topográficas",
            "variables": "INEGI CEM4 elevación y pendiente por polígono",
            "definicion_operativa": "Estadísticos de terreno del DEM/pendiente de 120 m; descriptivos, no prueba de efecto causal.",
            "evidencia": "197/197 parcelas con elevación y pendiente; 25 celdas de clima independientes de la resolución topográfica.",
            "nivel_evidencia": "Moderado-alto para el descriptor cartográfico",
            "confianza": "Media-alta para terreno; baja para efecto sobre cultivo",
            "limitaciones": "Parcelas pequeñas cubren pocos píxeles; clasificación de Tepeapulco tiene 1 parcela.",
            "validacion_requerida": "Contraste DEM/topografía con levantamiento y medición de posición/pendiente en parcela.",
        },
        {
            "concepto_candidato": "Prototipos/grupos temporales",
            "variables": "Medianas mensuales S2 NDVI abril-octubre; S2 solamente",
            "definicion_operativa": "KMeans euclidiano de 3 grupos para visualización, sin usar rendimiento.",
            "evidencia": "197 parcelas elegibles; silhouette 0.245; ARI bootstrap observacional medio 0.580 (100 remuestreos); ARI por correlación 0.668.",
            "nivel_evidencia": "Insuficiente para taxonomía estable",
            "confianza": "Baja",
            "limitaciones": "Asignación sensible a remuestreo/distancia y k; mezcla de manejo/localidad no observada.",
            "validacion_requerida": "Repetir en ciclos independientes y contrastar etiquetas de manejo/cultivar/fenología de campo.",
        },
        {
            "concepto_candidato": "Equivalencia intersensor",
            "variables": "NDVI, EVI, LAI pareados Sentinel-2 y Planet",
            "definicion_operativa": "Diferencias sobre captura de la misma parcela con tolerancia temporal de cinco días.",
            "evidencia": "11,284 pares / 197 parcelas; sesgo Planet-S2 NDVI +0.081, EVI -0.0006, LAI +1.057.",
            "nivel_evidencia": "Alto para no intercambiar valores sin calibración",
            "confianza": "Alta en esta colección y filtros",
            "limitaciones": "Pares repetidos por parcela/fecha y fuente, no una calibración independiente de campo.",
            "validacion_requerida": "Armonizar reflectancia, bandas/resolución/máscaras y probar calibración en escenas independientes.",
        },
    ]
    write_csv(pd.DataFrame(catalog), "catalogo_conceptos.csv")


def sensor_comparisons(obs: pd.DataFrame) -> pd.DataFrame:
    use = obs.loc[
        obs["year"].eq(2025)
        & obs["sensor"].isin(["Sentinel-2", "Planet"])
        & obs["cloud_pass"]
    ].copy()
    outputs = []
    for variable in ("ndvi_promedio", "evi_promedio", "lai_promedio"):
        if variable not in use:
            continue
        left = use.loc[
            use.sensor.eq("Sentinel-2") & use[variable].notna(),
            ["ID_POLIGONO", "date", variable, "source_row"],
        ].sort_values("date")
        right = use.loc[
            use.sensor.eq("Planet") & use[variable].notna(),
            ["ID_POLIGONO", "date", variable, "source_row"],
        ].sort_values("date")
        left = left.rename(columns={variable: "s2", "date": "date_s2", "source_row": "row_s2"})
        right = right.rename(
            columns={variable: "planet", "date": "date_planet", "source_row": "row_planet"}
        )
        for tolerance in (3, 5, 10):
            joined_parts = []
            for pid, lparcel in left.groupby("ID_POLIGONO"):
                rparcel = right.loc[right.ID_POLIGONO.eq(pid)]
                if rparcel.empty:
                    continue
                pair = pd.merge_asof(
                    lparcel.sort_values("date_s2"),
                    rparcel.drop(columns=["ID_POLIGONO"]).sort_values("date_planet"),
                    left_on="date_s2",
                    right_on="date_planet",
                    direction="nearest",
                    tolerance=pd.Timedelta(days=tolerance),
                )
                pair = pair.loc[pair["planet"].notna()].copy()
                pair["delta_days"] = (pair.date_planet - pair.date_s2).dt.days.abs()
                joined_parts.append(pair)
            joined = pd.concat(joined_parts, ignore_index=True) if joined_parts else pd.DataFrame()
            if joined.empty:
                outputs.append(
                    {"variable": variable, "tolerancia_dias": tolerance, "pares": 0}
                )
                continue
            delta = joined["planet"].to_numpy() - joined["s2"].to_numpy()
            corr = (
                float(pd.Series(joined["s2"]).corr(joined["planet"]))
                if len(joined) > 1
                else np.nan
            )
            outputs.append(
                {
                    "variable": variable,
                    "tolerancia_dias": tolerance,
                    "pares": len(joined),
                    "parcelas_pareadas": joined.ID_POLIGONO.nunique(),
                    "desfase_mediano_dias": joined.delta_days.median(),
                    "diferencia_mediana_Planet_menos_S2": np.median(delta),
                    "sesgo_medio_Planet_menos_S2": np.mean(delta),
                    "MAE_diferencia": np.mean(np.abs(delta)),
                    "RMSE_diferencia": np.sqrt(np.mean(delta**2)),
                    "correlacion_Pearson": corr,
                    "rho_Spearman": spearmanr(joined["s2"], joined["planet"]).statistic
                    if len(joined) > 2
                    else np.nan,
                }
            )
            if tolerance == 5 and variable == "ndvi_promedio":
                joined.to_csv(
                    OUT / "pares_sensor_ndvi_5dias.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
    result = pd.DataFrame(outputs)
    write_csv(result, "comparacion_sensores_pares_temporales.csv")
    return result


def valid_observations(obs: pd.DataFrame) -> pd.DataFrame:
    return obs.loc[
        obs["year"].eq(2025)
        & obs["cloud_pass"]
        & obs["date"].notna()
    ].copy()


def build_temporal_features(valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = {
        "Sentinel-2": ("ndvi_promedio", "evi_promedio", "lai_promedio"),
        "Planet": ("ndvi_promedio", "evi_promedio", "lai_promedio", "msavi_promedio"),
        "Landsat": ("vi6t_promedio",),
    }
    temporal_rows, aggregate_rows = [], []
    doy_start, doy_end = 91, 304
    for (sensor, pid), parcel in valid.groupby(["sensor", "ID_POLIGONO"], observed=True):
        trow: dict[str, object] = {"ID_POLIGONO": pid}
        arow: dict[str, object] = {"ID_POLIGONO": pid}
        for variable in variables.get(sensor, ()):
            if variable not in parcel:
                continue
            d = parcel.loc[parcel[variable].notna(), ["date", "doy", variable]].sort_values("date")
            if d.empty:
                continue
            prefix = {"Sentinel-2": "s2", "Planet": "planet", "Landsat": "ls"}[sensor]
            name = variable.replace("_promedio", "")
            col_prefix = f"{prefix}_{name}"
            season = d.loc[d["doy"].between(doy_start, doy_end)]
            if season.empty:
                continue
            values = season[variable].to_numpy(dtype=float)
            arow[f"{col_prefix}_mean"] = float(np.mean(values))
            arow[f"{col_prefix}_median"] = float(np.median(values))
            arow[f"{col_prefix}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            arow[f"{col_prefix}_p10"] = float(np.quantile(values, 0.10))
            arow[f"{col_prefix}_p90"] = float(np.quantile(values, 0.90))
            if len(season) < 3:
                continue
            x = season["doy"].to_numpy(dtype=float)
            y = season[variable].to_numpy(dtype=float)
            peak_index = int(np.argmax(y))
            peak = float(y[peak_index])
            baseline = d.loc[d["doy"].le(151), variable]
            base = float(baseline.median()) if len(baseline) else np.nan
            amp = peak - base if np.isfinite(base) else np.nan
            gaps = np.diff(x)
            slopes = np.diff(y) / gaps
            slopes[gaps > 45] = np.nan
            above = np.clip(y - base, 0, None) if np.isfinite(base) else np.full_like(y, np.nan)
            integral = (
                float(np.trapezoid(above, x))
                if np.isfinite(above).all()
                else np.nan
            )
            threshold = base + 0.5 * amp if np.isfinite(amp) else np.nan
            sos = eos = np.nan
            if np.isfinite(threshold):
                for idx in range(1, peak_index + 1):
                    if x[idx] - x[idx - 1] <= 45 and y[idx - 1] < threshold <= y[idx]:
                        sos = x[idx - 1] + (threshold - y[idx - 1]) * (
                            x[idx] - x[idx - 1]
                        ) / (y[idx] - y[idx - 1])
                        break
                for idx in range(peak_index + 1, len(x)):
                    if x[idx] - x[idx - 1] <= 45 and y[idx - 1] >= threshold > y[idx]:
                        eos = x[idx - 1] + (y[idx - 1] - threshold) * (
                            x[idx] - x[idx - 1]
                        ) / (y[idx - 1] - y[idx])
                        break
            trow.update(
                {
                    f"{col_prefix}_season_mean": float(np.mean(y)),
                    f"{col_prefix}_season_peak": peak,
                    f"{col_prefix}_peak_doy": float(x[peak_index]),
                    f"{col_prefix}_baseline_pre151": base,
                    f"{col_prefix}_amplitude": amp,
                    f"{col_prefix}_auc_excess_pre151": integral,
                    f"{col_prefix}_sos_halfamp_doy": float(sos),
                    f"{col_prefix}_eos_halfamp_doy": float(eos),
                    f"{col_prefix}_duration_halfamp_days": float(eos - sos)
                    if np.isfinite(sos) and np.isfinite(eos)
                    else np.nan,
                    f"{col_prefix}_max_slope_per_day": float(np.nanmax(slopes))
                    if np.isfinite(slopes).any()
                    else np.nan,
                    f"{col_prefix}_min_slope_per_day": float(np.nanmin(slopes))
                    if np.isfinite(slopes).any()
                    else np.nan,
                }
            )
            for stage, (start, end) in STAGES.items():
                in_stage = season.loc[season["doy"].between(start, end), variable]
                trow[f"{col_prefix}_{stage}_mean"] = (
                    float(in_stage.mean()) if len(in_stage) else np.nan
                )
        if len(trow) > 1:
            temporal_rows.append(trow)
        if len(arow) > 1:
            aggregate_rows.append(arow)
    temporal = pd.DataFrame(temporal_rows).groupby("ID_POLIGONO").first()
    aggregate = pd.DataFrame(aggregate_rows).groupby("ID_POLIGONO").first()
    smooth_rows = []
    for sensor in ("Sentinel-2", "Planet"):
        signal = valid.loc[
            valid.sensor.eq(sensor)
            & valid.month.between(4, 10)
            & valid.ndvi_promedio.notna()
        ]
        monthly = signal.pivot_table(
            index="ID_POLIGONO",
            columns="month",
            values="ndvi_promedio",
            aggfunc="median",
        ).reindex(columns=range(4, 11))
        for pid, row in monthly.iterrows():
            observed = row.dropna()
            if len(observed) < 4:
                continue
            smoothed = row.rolling(window=3, center=True, min_periods=2).mean()
            raw_peak_month = int(row.idxmax())
            smooth_peak_month = int(smoothed.idxmax())
            raw_peak = float(row.max())
            smooth_peak = float(smoothed.max())
            smooth_rows.append(
                {
                    "ID_POLIGONO": pid,
                    "sensor": sensor,
                    "meses_observados": len(observed),
                    "pico_sin_suavizar_mes": raw_peak_month,
                    "pico_media_movil_3meses_mes": smooth_peak_month,
                    "cambio_fecha_pico_meses": smooth_peak_month - raw_peak_month,
                    "NDVI_pico_sin_suavizar": raw_peak,
                    "NDVI_pico_suavizado": smooth_peak,
                    "cambio_NDVI_pico": smooth_peak - raw_peak,
                }
            )
    smooth_frame = pd.DataFrame(smooth_rows)
    write_csv(smooth_frame, "sensibilidad_suavizado_pico_ndvi.csv")
    return temporal, aggregate


def fit_curve_clusters(obs: pd.DataFrame, parcels: pd.DataFrame) -> pd.DataFrame:
    d_all = obs.loc[
        obs["year"].eq(2025)
        & obs["sensor"].eq("Sentinel-2")
        & obs["cloud_pct"].notna()
        & (obs["cloud_pct"] <= 10)
        & obs["ndvi_promedio"].notna()
    ].copy()
    d_all = d_all.loc[d_all["month"].between(4, 10)]
    d = d_all.loc[d_all["cloud_pct"].le(5)]
    monthly = d.pivot_table(
        index="ID_POLIGONO", columns="month", values="ndvi_promedio", aggfunc="median"
    ).reindex(index=parcels.ID_POLIGONO, columns=range(4, 11))
    eligible = monthly.notna().sum(axis=1).ge(4)
    monthly = monthly.loc[eligible]
    ids = monthly.index
    x = monthly.to_numpy(dtype=float)
    global_medians = np.nanmedian(x, axis=0)
    x_filled = np.where(np.isfinite(x), x, global_medians)
    kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=50).fit(x_filled)
    labels = kmeans.labels_
    canonical = pd.DataFrame(
        {
            "ID_POLIGONO": ids,
            "cluster_euclidiano_k3": labels + 1,
            "meses_observados_abr_oct": monthly.notna().sum(axis=1).to_numpy(),
        }
    )
    sil = silhouette_score(x_filled, labels) if len(np.unique(labels)) > 1 else np.nan

    prototypes = []
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        distances = np.linalg.norm(x_filled[members] - kmeans.cluster_centers_[cluster], axis=1)
        medoid_position = members[int(np.argmin(distances))]
        row = {
            "cluster": int(cluster + 1),
            "parcelas": int(len(members)),
            "medoid_ID_POLIGONO": str(ids[medoid_position]),
            "silhouette_global": float(sil),
        }
        for month_idx, month in enumerate(range(4, 11)):
            row[f"ndvi_m{month:02d}_media_prototipo"] = float(
                np.nanmedian(x[members, month_idx])
            )
        prototypes.append(row)
    prototypes_df = pd.DataFrame(prototypes)
    write_csv(canonical, "clusters_parcela_s2_ndvi.csv")
    write_csv(prototypes_df, "prototipos_clusters_s2_ndvi.csv")

    stability = []
    rng = np.random.default_rng(SEED)
    for replicate in range(100):
        boot_matrix = np.full_like(x, np.nan)
        for row_idx, pid in enumerate(ids):
            parcel_records = d.loc[d.ID_POLIGONO.eq(pid)].copy()
            sampled_idx = rng.integers(0, len(parcel_records), size=len(parcel_records))
            boot = parcel_records.iloc[sampled_idx]
            for month in range(4, 11):
                monthly_values = boot.loc[boot.month.eq(month), "ndvi_promedio"]
                if len(monthly_values):
                    boot_matrix[row_idx, month - 4] = monthly_values.median()
        boot_global = np.nanmedian(boot_matrix, axis=0)
        boot_filled = np.where(np.isfinite(boot_matrix), boot_matrix, boot_global)
        if np.isnan(boot_filled).any():
            boot_filled = np.where(np.isfinite(boot_filled), boot_filled, global_medians)
        boot_labels = KMeans(n_clusters=3, random_state=SEED, n_init=20).fit_predict(boot_filled)
        stability.append(
            {"prueba": "bootstrap_observaciones", "replica": replicate + 1, "ARI_vs_base": adjusted_rand_score(labels, boot_labels)}
        )

    corr = np.corrcoef(x_filled)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    corr_distance = np.clip(1.0 - corr, 0.0, 2.0)
    corr_labels = AgglomerativeClustering(
        n_clusters=3, metric="precomputed", linkage="average"
    ).fit_predict(corr_distance)
    stability.append(
        {
            "prueba": "distancia_correlacion_average_linkage",
            "replica": 0,
            "ARI_vs_base": adjusted_rand_score(labels, corr_labels),
        }
    )
    for cloud_limit in (1, 5, 10):
        dd = d_all.loc[d_all["cloud_pct"].le(cloud_limit)]
        alternate = dd.pivot_table(
            index="ID_POLIGONO",
            columns="month",
            values="ndvi_promedio",
            aggfunc="median",
        ).reindex(index=ids, columns=range(4, 11))
        alt_raw = alternate.to_numpy(dtype=float)
        alt_fill = np.where(np.isfinite(alt_raw), alt_raw, np.nanmedian(alt_raw, axis=0))
        alt_labels = KMeans(n_clusters=3, random_state=SEED, n_init=30).fit_predict(alt_fill)
        stability.append(
            {
                "prueba": f"umbral_nubes_{cloud_limit}_pct",
                "replica": 0,
                "ARI_vs_base": adjusted_rand_score(labels, alt_labels),
            }
        )
    stability_df = pd.DataFrame(stability)
    write_csv(stability_df, "estabilidad_clusters.csv")
    k_sensitivity = []
    for cluster_count in (2, 3, 4, 5):
        estimator = KMeans(
            n_clusters=cluster_count, random_state=SEED, n_init=50
        ).fit(x_filled)
        candidate = estimator.labels_
        k_sensitivity.append(
            {
                "k": cluster_count,
                "silhouette_euclidiano": silhouette_score(x_filled, candidate),
                "inercia": estimator.inertia_,
                "tamanos": ";".join(
                    map(str, np.bincount(candidate, minlength=cluster_count).tolist())
                ),
                "ARI_vs_k3": adjusted_rand_score(labels, candidate)
                if cluster_count != 3
                else 1.0,
            }
        )
    write_csv(
        pd.DataFrame(k_sensitivity), "sensibilidad_numero_clusters.csv"
    )
    cluster_municipality = canonical.merge(
        parcels[["ID_POLIGONO", "Municipio"]], on="ID_POLIGONO", how="left"
    )
    contingency = (
        cluster_municipality.groupby(["Municipio", "cluster_euclidiano_k3"])
        .size()
        .rename("parcelas")
        .reset_index()
    )
    write_csv(contingency, "clusters_por_municipio.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    for row_idx, (pid, row) in enumerate(monthly.iterrows()):
        cluster = labels[row_idx] + 1
        ax.plot(range(4, 11), row.values, color=f"C{cluster-1}", alpha=0.15, linewidth=0.8)
    for row in prototypes:
        cluster = row["cluster"]
        values = [row[f"ndvi_m{month:02d}_media_prototipo"] for month in range(4, 11)]
        ax.plot(range(4, 11), values, color=f"C{cluster-1}", linewidth=3, marker="o", label=f"Prototipo cluster {cluster} (n={row['parcelas']})")
    ax.set(title="S2 NDVI 2025: curvas observadas y prototipos descriptivos", xlabel="Mes", ylabel="NDVI mediano", xticks=range(4, 11))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "prototipos_curvas_s2_ndvi.png", dpi=150)
    plt.close(fig)
    return canonical


def spatial_groups(coordinates: np.ndarray, n_blocks: int) -> tuple[np.ndarray, int]:
    centered = coordinates - coordinates.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    main_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    order = np.argsort(centered @ main_axis, kind="stable")
    groups = np.empty(len(coordinates), dtype=int)
    for label, indices in enumerate(np.array_split(order, n_blocks)):
        groups[indices] = label
    centers = np.vstack([coordinates[groups == i].mean(axis=0) for i in range(n_blocks)])
    outer = int(np.argmax(np.linalg.norm(centers - centers.mean(axis=0), axis=1)))
    return groups, outer


def scores(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "RMSE_t_ha": float(np.sqrt(mean_squared_error(y, prediction))),
        "MAE_t_ha": float(mean_absolute_error(y, prediction)),
        "R2": float(r2_score(y, prediction)) if len(y) > 1 else np.nan,
        "sesgo_observado_menos_predicho_t_ha": float(np.mean(y - prediction)),
    }


def municipal_prior(
    y: np.ndarray, train: np.ndarray, test: np.ndarray, municipalities: np.ndarray
) -> np.ndarray:
    global_mean = float(np.mean(y[train]))
    output = []
    for idx in test:
        mask = train[municipalities[train] == municipalities[idx]]
        output.append((np.sum(y[mask]) + 3 * global_mean) / (len(mask) + 3))
    return np.asarray(output)


def knn_prior(
    y: np.ndarray, train: np.ndarray, test: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=min(5, len(train))).fit(coordinates[train])
    distances, neighbors = model.kneighbors(coordinates[test])
    weights = 1.0 / np.maximum(distances, 1.0)
    return np.sum(y[train][neighbors] * weights, axis=1) / np.sum(weights, axis=1)


def compare_feature_groups(
    temporal: pd.DataFrame, aggregate: pd.DataFrame, parcels: pd.DataFrame
) -> pd.DataFrame:
    base = parcels[
        ["ID_POLIGONO", "RENDIMIENTO_T_HA", "CONJUNTO", "x_utm", "y_utm", "Municipio"]
    ].merge(temporal, on="ID_POLIGONO", how="left").merge(
        aggregate, on="ID_POLIGONO", how="left", suffixes=("", "_agregado")
    )
    labelled = base.loc[
        base.CONJUNTO.eq("ENTRENAMIENTO") & base.RENDIMIENTO_T_HA.notna()
    ].sort_values("ID_POLIGONO").reset_index(drop=True)
    y = labelled.RENDIMIENTO_T_HA.to_numpy(dtype=float)
    coords = labelled[["x_utm", "y_utm"]].to_numpy(dtype=float)
    municipalities = labelled.Municipio.fillna("").astype(str).to_numpy()
    temporal_cols = [
        c for c in temporal.columns if c != "ID_POLIGONO" and c in labelled.columns
    ]
    aggregate_cols = [
        c for c in aggregate.columns if c != "ID_POLIGONO" and c in labelled.columns
    ]
    fold_metrics, prediction_rows = [], []
    feature_sets = {"temporal": temporal_cols, "agregado_anual": aggregate_cols}

    for block_count in BLOCK_COUNTS:
        groups, outer_label = spatial_groups(coords, block_count)
        outer_test = np.flatnonzero(groups == outer_label)
        outer_train = np.flatnonzero(groups != outer_label)
        inner_groups = groups[outer_train]
        for model_name in ("prior_municipio", "prior_knn_k5", "temporal", "agregado_anual"):
            if model_name in ("temporal", "agregado_anual"):
                columns = feature_sets[model_name]
                X = labelled[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                outer_prediction = np.empty(len(outer_test))
                inner_oof = np.full(len(outer_train), np.nan)
                for inner_label in np.unique(inner_groups):
                    local_test = np.flatnonzero(inner_groups == inner_label)
                    local_train = np.flatnonzero(inner_groups != inner_label)
                    train_idx, test_idx = outer_train[local_train], outer_train[local_test]
                    pipeline = make_pipeline(
                        SimpleImputer(strategy="median", keep_empty_features=True),
                        StandardScaler(),
                        Ridge(alpha=RIDGE_ALPHA),
                    )
                    pipeline.fit(X[train_idx], y[train_idx])
                    inner_oof[local_test] = pipeline.predict(X[test_idx])
                    for row_idx, pred in zip(test_idx, inner_oof[local_test]):
                        prediction_rows.append(
                            {
                                "bloques": block_count,
                                "evaluacion": "interna_oof",
                                "modelo": model_name,
                                "ID_POLIGONO": labelled.iloc[row_idx].ID_POLIGONO,
                                "y_observado_t_ha": y[row_idx],
                                "y_predicho_t_ha": pred,
                            }
                        )
                pipeline = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    StandardScaler(),
                    Ridge(alpha=RIDGE_ALPHA),
                )
                pipeline.fit(X[outer_train], y[outer_train])
                outer_prediction = pipeline.predict(X[outer_test])
                for row_idx, pred in zip(outer_test, outer_prediction):
                    prediction_rows.append(
                        {
                            "bloques": block_count,
                            "evaluacion": "externa_holdout",
                            "modelo": model_name,
                            "ID_POLIGONO": labelled.iloc[row_idx].ID_POLIGONO,
                            "y_observado_t_ha": y[row_idx],
                            "y_predicho_t_ha": pred,
                        }
                    )
                for evaluation, actual, predicted in (
                    ("interna_oof", y[outer_train], inner_oof),
                    ("externa_holdout", y[outer_test], outer_prediction),
                ):
                    fold_metrics.append(
                        {
                            "bloques": block_count,
                            "n_holdout": len(outer_test),
                            "n_features": len(columns),
                            "modelo": model_name,
                            "evaluacion": evaluation,
                            **scores(actual, predicted),
                        }
                    )
            else:
                if model_name == "prior_municipio":
                    inner_oof = np.full(len(outer_train), np.nan)
                    for inner_label in np.unique(inner_groups):
                        local_test = np.flatnonzero(inner_groups == inner_label)
                        local_train = np.flatnonzero(inner_groups != inner_label)
                        train_idx, test_idx = outer_train[local_train], outer_train[local_test]
                        pred = municipal_prior(y, train_idx, test_idx, municipalities)
                        inner_oof[local_test] = pred
                    outer_prediction = municipal_prior(y, outer_train, outer_test, municipalities)
                else:
                    inner_oof = np.full(len(outer_train), np.nan)
                    for inner_label in np.unique(inner_groups):
                        local_test = np.flatnonzero(inner_groups == inner_label)
                        local_train = np.flatnonzero(inner_groups != inner_label)
                        train_idx, test_idx = outer_train[local_train], outer_train[local_test]
                        inner_oof[local_test] = knn_prior(y, train_idx, test_idx, coords)
                    outer_prediction = knn_prior(y, outer_train, outer_test, coords)
                for evaluation, actual, predicted in (
                    ("interna_oof", y[outer_train], inner_oof),
                    ("externa_holdout", y[outer_test], outer_prediction),
                ):
                    fold_metrics.append(
                        {
                            "bloques": block_count,
                            "n_holdout": len(outer_test),
                            "n_features": 1 if model_name == "prior_municipio" else 5,
                            "modelo": model_name,
                            "evaluacion": evaluation,
                            **scores(actual, predicted),
                        }
                    )
    metrics = pd.DataFrame(fold_metrics)
    write_csv(metrics, "comparacion_temporal_vs_agregado_folds.csv")
    write_csv(pd.DataFrame(prediction_rows), "comparacion_temporal_vs_agregado_predicciones.csv")
    feature_count = pd.DataFrame(
        [
            {"grupo": "temporal", "n_features": len(temporal_cols), "features": ";".join(temporal_cols)},
            {"grupo": "agregado_anual", "n_features": len(aggregate_cols), "features": ";".join(aggregate_cols)},
        ]
    )
    write_csv(feature_count, "comparacion_temporal_vs_agregado_features.csv")
    return metrics


def draw_curves(obs: pd.DataFrame) -> None:
    valid = valid_observations(obs)
    sensors = ["Sentinel-2", "Planet", "Landsat"]
    variables = {"Sentinel-2": "ndvi_promedio", "Planet": "ndvi_promedio", "Landsat": "vi6t_promedio"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, sensor in zip(axes, sensors):
        variable = variables[sensor]
        d = valid.loc[valid.sensor.eq(sensor) & valid[variable].notna()].copy()
        d["month_anchor"] = d.date.dt.to_period("M").dt.to_timestamp()
        stats = d.groupby("month_anchor")[variable].agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            n="count",
        )
        if not stats.empty:
            ax.plot(stats.index, stats["median"], marker="o", label="Mediana")
            ax.fill_between(stats.index, stats["q25"], stats["q75"], alpha=0.2, label="IQR")
        ax.set_title(f"{sensor}: {variable.replace('_promedio', '').upper()}")
        ax.set_xlabel("Fecha 2025")
        ax.set_ylabel("Valor fuente")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
    fig.suptitle("Curvas satelitales por sensor (observaciones con umbral de nubosidad)")
    fig.tight_layout()
    fig.savefig(FIG / "curvas_mediana_por_sensor.png", dpi=150)
    plt.close(fig)

    coverage = (
        valid.loc[valid.sensor.isin(["Sentinel-2", "Planet"])]
        .groupby(["sensor", "month"])
        .ID_POLIGONO.nunique()
        .unstack("sensor")
        .reindex(index=range(1, 13))
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    coverage.plot(kind="bar", ax=ax)
    ax.set(title="Parcelas con al menos una observación admisible por mes", xlabel="Mes", ylabel="Parcelas")
    fig.tight_layout()
    fig.savefig(FIG / "cobertura_mensual.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    obs = load_observations()
    parcels = pd.read_csv(PROCESSED / "parcelas_base.csv")
    assignment = pd.read_csv(
        RAW / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv", encoding="utf-8-sig"
    )
    assignment = assignment.loc[:, ~assignment.columns.str.startswith("Unnamed")]
    assignment["fila_fuente_asignacion"] = np.arange(2, len(assignment) + 2)
    if parcels.ID_POLIGONO.duplicated().any():
        raise ValueError("La tabla parcelaria tiene ID_POLIGONO repetidos.")
    if assignment.ID_POLIGONO.duplicated().any():
        raise ValueError("La asignación canónica tiene ID_POLIGONO repetidos.")
    parcel_ids, assignment_ids = set(parcels.ID_POLIGONO), set(assignment.ID_POLIGONO)
    if parcel_ids != assignment_ids:
        raise ValueError(
            f"IDs no reconciliados: faltan asignaciones={len(parcel_ids-assignment_ids)}, "
            f"faltan geometrías={len(assignment_ids-parcel_ids)}"
        )
    parcels = parcels.drop(
        columns=["CONJUNTO", "RENDIMIENTO_T_HA", "AREA_HA"], errors="ignore"
    ).merge(assignment, on="ID_POLIGONO", how="left", validate="one_to_one")
    quality_products(obs, parcels)
    environment_products(parcels)
    index_stage_products(obs)
    write_catalog()
    paired = sensor_comparisons(obs)
    valid = valid_observations(obs)
    temporal, aggregate = build_temporal_features(valid)
    temporal.reset_index().to_csv(
        OUT / "caracteristicas_temporales_parcela_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )
    aggregate.reset_index().to_csv(
        OUT / "estadisticas_agregadas_parcela_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cluster_labels = fit_curve_clusters(obs, parcels)
    draw_curves(obs)
    comparison = compare_feature_groups(temporal, aggregate, parcels)

    source_paths = [
        RAW / "Conjunto_datos_BASICO_AgroCebada2026.csv",
        RAW / "Conjunto_datos_PRO_AgroCebada.csv",
        RAW / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
        PROCESSED / "parcelas_base.csv",
        PROCESSED / "clima_mensual_parcela.csv",
    ]
    manifest = {
        "script": str(Path(__file__).relative_to(DEV)),
        "seed": SEED,
        "threshold_cloud_pct": THRESHOLDS,
        "calendar_windows_doy_hypotheses_not_ground_truth": STAGES,
        "spatial_protocol": {
            "source": "desarrollo/analisis/comparison.py",
            "block_counts": BLOCK_COUNTS,
            "outer_holdout": "balanced contiguous bands along principal UTM centroid axis; outermost center",
            "inner": "GroupKFold on remaining bands",
            "ridge_alpha": RIDGE_ALPHA,
            "imputation_scaling": "fold-local training median and StandardScaler",
        },
        "no_yield_for_descriptive_curves_or_clustering": True,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "sources": [
            {"path": str(path.relative_to(DEV)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "observations": len(obs),
                "observations_2025": int(obs.year.eq(2025).sum()),
                "parcels": int(parcels.ID_POLIGONO.nunique()),
                "eligible_clusters": len(cluster_labels),
                "comparison": comparison.to_dict(orient="records"),
                "sensor_pairs": paired.to_dict(orient="records"),
                "outputs": str(OUT),
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
