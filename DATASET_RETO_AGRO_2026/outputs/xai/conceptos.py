#!/usr/bin/env python
"""Construcción de perfiles conceptuales observacionales AgroCebada 2025.

No usa rendimiento ni etiquetas. Conserva 197 parcelas con flags de calidad.
"""
from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

SEED = 2026
np.random.seed(SEED)

# Rutas (solo lectura)
RAW_DIR = Path("desarrollo/datos/entradas")
PROC_DIR = Path("desarrollo/cebada_datos_baseline_1/data_processed")
OUT_DIR = Path("outputs/xai")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columnas exactas esperadas (validar al cargar)
BASIC_COLS = [
    "ID_POLIGONO", "fecha_captura", "sensor", "ndvi_promedio", "evi_promedio",
    "ndwi_promedio", "msi_promedio", "nddi_promedio", "porcentaje_nubosidad"
]
PRO_COLS = [
    "ID_POLIGONO", "fecha_captura", "sensor", "ndvi_promedio", "evi_promedio",
    "lai_promedio", "msavi_promedio", "porcentaje_nubosidad"
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga y valida columnas. No modifica originales."""
    basic = pd.read_csv(RAW_DIR / "Conjunto_datos_BASICO_AgroCebada2026.csv", encoding="utf-8-sig")
    pro = pd.read_csv(RAW_DIR / "Conjunto_datos_PRO_AgroCebada.csv", encoding="utf-8-sig")
    parcels = pd.read_csv(PROC_DIR / "parcelas_base.csv", encoding="utf-8-sig")
    climate = pd.read_csv(PROC_DIR / "clima_mensual_parcela.csv", encoding="utf-8-sig")

    # Validar columnas mínimas
    for df, cols, name in [(basic, BASIC_COLS, "BASIC"), (pro, PRO_COLS, "PRO")]:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name} faltan columnas: {missing}")

    return basic, pro, parcels, climate


def filter_s2_2025_valid(basic: pd.DataFrame) -> pd.DataFrame:
    """Sentinel-2 2025, nube <=5%, DOY 91-304 (abr-oct)."""
    df = basic.copy()
    df["fecha"] = pd.to_datetime(df["fecha_captura"], dayfirst=True, errors="coerce")
    df["year"] = df["fecha"].dt.year
    df["doy"] = df["fecha"].dt.dayofyear
    df["month"] = df["fecha"].dt.month

    mask = (
        df["sensor"].eq("Sentinel-2") &
        df["year"].eq(2025) &
        df["porcentaje_nubosidad"].notna() &
        (df["porcentaje_nubosidad"] <= 5.0) &
        df["doy"].between(91, 304) &
        df["ndvi_promedio"].notna()
    )
    return df.loc[mask].copy()


def compute_temporal_components(s2_valid: pd.DataFrame) -> pd.DataFrame:
    """12 componentes base por parcela desde S2 válido abr-oct 2025."""
    results = []

    for pid, parcel in s2_valid.groupby("ID_POLIGONO", observed=True):
        # Ordenar por fecha
        parcel = parcel.sort_values("fecha")

        # Componentes C1: NDVI y EVI medios temporales
        ndvi_mean = parcel["ndvi_promedio"].mean()
        evi_mean = parcel["evi_promedio"].mean()

        # Baseline pre-ciclo (DOY <= 151)
        pre_cycle = parcel.loc[parcel["doy"] <= 151, "ndvi_promedio"]
        baseline = pre_cycle.median() if len(pre_cycle) else np.nan

        # Pico en ciclo principal (DOY 91-304)
        cycle = parcel.loc[parcel["doy"].between(91, 304), "ndvi_promedio"]
        if len(cycle) == 0:
            peak = np.nan
            peak_doy = np.nan
        else:
            peak = cycle.max()
            # DOY del pico (primera ocurrencia)
            peak_idx = cycle.idxmax()
            peak_doy = parcel.loc[peak_idx, "doy"]

        # Amplitud
        amp = peak - baseline if np.isfinite(baseline) and np.isfinite(peak) else np.nan

        # AUC exceso sobre baseline (trapezoidal, DOY ordenados)
        if np.isfinite(baseline) and len(parcel) >= 2:
            doy_sorted = parcel.sort_values("doy")["doy"].values
            ndvi_sorted = parcel.sort_values("doy")["ndvi_promedio"].values
            excess = np.clip(ndvi_sorted - baseline, 0, None)
            auc_excess = np.trapz(excess, doy_sorted) if np.all(np.isfinite(excess)) else np.nan
        else:
            auc_excess = np.nan

        # C3: Índices hídricos medios
        ndwi_mean = parcel["ndwi_promedio"].mean()
        msi_mean = parcel["msi_promedio"].mean()
        nddi_mean = parcel["nddi_promedio"].mean()

        # Flags calidad
        n_meses_valid = parcel["month"].nunique()
        n_obs_valid = len(parcel)

        results.append({
            "ID_POLIGONO": pid,
            # C1 familia
            "C1a_ndvi_s2_mean": ndvi_mean,
            "C1b_evi_s2_mean": evi_mean,
            # C2 familia
            "C2a_amplitud_ndvi": amp,
            "C2b_auc_exceso_ndvi": auc_excess,
            "C2c_doy_pico_ndvi": peak_doy,
            # C3 familia
            "C3a_ndwi_s2_mean": ndwi_mean,
            "C3b_msi_s2_mean": msi_mean,
            "C3c_nddi_s2_mean": nddi_mean,
            # Flags
            "flag_n_meses_s2_abr_oct": n_meses_valid,
            "flag_n_obs_s2_abr_oct": n_obs_valid,
        })

    return pd.DataFrame(results)


def compute_climate_components(climate: pd.DataFrame, parcels: pd.DataFrame) -> pd.DataFrame:
    """C4: precipitación acumulada y temperatura media abr-oct 2025."""
    climate_2025 = climate.loc[climate["anio"].eq(2025)].copy()
    # Abril (4) a Octubre (10)
    cycle = climate_2025.loc[climate_2025["mes"].between(4, 10)].copy()

    # tmean por fila
    cycle["tmean"] = (
        pd.to_numeric(cycle["tmin"], errors="coerce") +
        pd.to_numeric(cycle["tmax"], errors="coerce")
    ) / 2

    agg = cycle.groupby("ID_POLIGONO").agg(
        C4a_prec_acum_abr_oct=("prec_mm", "sum"),
        C4b_tmean_abr_oct=("tmean", "mean"),
        flag_meses_clima_2025=("mes", "nunique"),
    ).reset_index()

    return agg


def compute_topo_components(parcels: pd.DataFrame) -> pd.DataFrame:
    """C5: elevación y pendiente medias parcelarias (INEGI CEM4 120m)."""
    topo = parcels[["ID_POLIGONO", "elev_media", "pend_media"]].copy()
    topo.columns = ["ID_POLIGONO", "C5a_elev_media", "C5b_pend_media"]
    topo["flag_topo_completa"] = topo[["C5a_elev_media", "C5b_pend_media"]].notna().all(axis=1).astype(int)
    return topo


def merge_all(components_list: list[pd.DataFrame], parcels: pd.DataFrame) -> pd.DataFrame:
    """Merge por ID_POLIGONO, conservar todas las 197 parcelas."""
    base = parcels[["ID_POLIGONO"]].copy()
    for df in components_list:
        base = base.merge(df, on="ID_POLIGONO", how="left")
    return base


def robust_scale_components(df: pd.DataFrame, component_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """RobustScaler por componente, guardar parámetros."""
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    scaled = df[component_cols].copy()
    scaled_values = scaler.fit_transform(scaled)
    scaled_df = pd.DataFrame(scaled_values, columns=[f"{c}_scaled" for c in component_cols], index=df.index)

    params = {
        c: {"median": float(scaler.center_[i]), "iqr": float(scaler.scale_[i])}
        for i, c in enumerate(component_cols)
    }
    return scaled_df, params


def compute_concordancia_s2_planet(basic: pd.DataFrame, pro: pd.DataFrame) -> pd.DataFrame:
    """Reutilizar lógica analizar.py: pareo ±5 días, NDVI/EVI/LAI."""
    # Filtrar 2025, nube admisible
    s2 = basic.loc[
        basic["sensor"].eq("Sentinel-2") &
        (pd.to_datetime(basic["fecha_captura"], dayfirst=True).dt.year == 2025) &
        basic["porcentaje_nubosidad"].notna() &
        (basic["porcentaje_nubosidad"] <= 5.0) &
        basic["ndvi_promedio"].notna()
    ].copy()
    s2["fecha"] = pd.to_datetime(s2["fecha_captura"], dayfirst=True)

    pl = pro.loc[
        pro["sensor"].eq("Planet") &
        (pd.to_datetime(pro["fecha_captura"], dayfirst=True).dt.year == 2025) &
        pro["porcentaje_nubosidad"].notna() &
        (pro["porcentaje_nubosidad"] <= 20.0) &
        pro["ndvi_promedio"].notna()
    ].copy()
    pl["fecha"] = pd.to_datetime(pl["fecha_captura"], dayfirst=True)

    results = []
    for var_s2, var_pl in [("ndvi_promedio", "ndvi_promedio"), ("evi_promedio", "evi_promedio")]:
        for pid in s2["ID_POLIGONO"].unique():
            s2_p = s2.loc[s2["ID_POLIGONO"].eq(pid), ["fecha", var_s2]].sort_values("fecha")
            pl_p = pl.loc[pl["ID_POLIGONO"].eq(pid), ["fecha", var_pl]].sort_values("fecha")
            if s2_p.empty or pl_p.empty:
                continue
            # merge_asof nearest ±5 días
            merged = pd.merge_asof(s2_p, pl_p, on="fecha", direction="nearest",
                                   tolerance=pd.Timedelta(days=5), suffixes=("_s2", "_pl"))
            merged = merged.dropna()
            if len(merged) < 2:
                continue
            delta = merged[f"{var_pl}_pl"] - merged[f"{var_s2}_s2"]
            results.append({
                "ID_POLIGONO": pid,
                "variable": var_s2,
                "n_pairs": len(merged),
                "median_delta_pl_minus_s2": float(delta.median()),
                "mean_delta_pl_minus_s2": float(delta.mean()),
                "mae_delta": float(np.abs(delta).mean()),
                "pearson_r": float(merged[f"{var_s2}_s2"].corr(merged[f"{var_pl}_pl"])) if len(merged) > 1 else np.nan,
            })
    return pd.DataFrame(results)


def build_manifest(basic_hash, pro_hash, parcels_hash, climate_hash, scaler_params, component_cols) -> dict:
    return {
        "version": "1.0",
        "seed": SEED,
        "python": platform.python_version(),
        "source_hashes": {
            "Conjunto_datos_BASICO_AgroCebada2026.csv": basic_hash,
            "Conjunto_datos_PRO_AgroCebada.csv": pro_hash,
            "parcelas_base.csv": parcels_hash,
            "clima_mensual_parcela.csv": climate_hash,
        },
        "component_columns": component_cols,
        "scaler": "RobustScaler(quantile_range=(25,75))",
        "scaler_params": scaler_params,
        "temporal_window": {"start_doy": 91, "end_doy": 304, "baseline_doy_max": 151},
        "cloud_thresholds": {"Sentinel-2": 5.0, "Planet": 20.0, "Landsat": 5.0},
        "n_parcelas_total": 197,
    }


def main():
    print("=== FASE 1: Construcción de perfiles conceptuales ===")

    # Cargar y validar
    basic, pro, parcels, climate = load_and_validate()

    # Hashes de fuentes
    basic_hash = sha256_file(RAW_DIR / "Conjunto_datos_BASICO_AgroCebada2026.csv")
    pro_hash = sha256_file(RAW_DIR / "Conjunto_datos_PRO_AgroCebada.csv")
    parcels_hash = sha256_file(PROC_DIR / "parcelas_base.csv")
    climate_hash = sha256_file(PROC_DIR / "clima_mensual_parcela.csv")

    # Componentes temporales S2
    s2_valid = filter_s2_2025_valid(basic)
    temporal = compute_temporal_components(s2_valid)

    # Clima
    climate_comp = compute_climate_components(climate, parcels)

    # Topografía
    topo_comp = compute_topo_components(parcels)

    # Merge all (197 parcelas)
    merged = merge_all([temporal, climate_comp, topo_comp], parcels)

    # Componentes base (12) para escalado
    component_cols = [
        "C1a_ndvi_s2_mean", "C1b_evi_s2_mean",
        "C2a_amplitud_ndvi", "C2b_auc_exceso_ndvi", "C2c_doy_pico_ndvi",
        "C3a_ndwi_s2_mean", "C3b_msi_s2_mean", "C3c_nddi_s2_mean",
        "C4a_prec_acum_abr_oct", "C4b_tmean_abr_oct",
        "C5a_elev_media", "C5b_pend_media",
    ]

    # Verificar columnas presentes
    missing = [c for c in component_cols if c not in merged.columns]
    if missing:
        raise ValueError(f"Faltan componentes: {missing}")

    # RobustScaler
    scaled_df, scaler_params = robust_scale_components(merged, component_cols)
    merged = pd.concat([merged, scaled_df], axis=1)

    # Guardar perfiles conceptuales (ordenado por ID_POLIGONO para invariancia al orden)
    out_path = OUT_DIR / "perfiles_conceptuales.csv"
    merged = merged.sort_values("ID_POLIGONO").reset_index(drop=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Perfiles guardados: {out_path} ({len(merged)} filas x {len(merged.columns)} cols)")

    # Concordancia sensores
    concord = compute_concordancia_s2_planet(basic, pro)
    concord_path = OUT_DIR / "concordancia_sensores.csv"
    concord.to_csv(concord_path, index=False, encoding="utf-8-sig")
    print(f"Concordancia guardada: {concord_path}")

    # Manifest
    manifest = build_manifest(basic_hash, pro_hash, parcels_hash, climate_hash, scaler_params, component_cols)
    manifest_path = OUT_DIR / "manifest_xai.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest guardado: {manifest_path}")

    print("=== FASE 1 COMPLETADA ===")


if __name__ == "__main__":
    main()