"""Regresión lineal múltiple (OLS) — Reto AgroCebada 2026.

Lee directamente DATASET_RETO_AGRO_2026.zip, sin extraerlo.
Compara OLS con 8, 24 y 148 variables mediante las MISMAS particiones
usadas en ridge_regresion_agro.py: KFold(5, shuffle=True, random_state=2026)
y GroupKFold(6) con 6 bloques KMeans sobre UTM (random_state=2026).

USO:
    pip install numpy pandas scikit-learn matplotlib joblib
    python regresion_lineal_agro.py
    python regresion_lineal_agro.py --zip /ruta/DATASET_RETO_AGRO_2026.zip

ADVERTENCIA: las 59 parcelas de predicción no contienen etiqueta.
El ciclo/año del rendimiento observado no está documentado. NO se valida
el pronóstico de la cosecha 2026. OLS-148 es un experimento diagnóstico,
NO una recomendación para uso predictivo.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 2026
OBJETIVO = "RENDIMIENTO_T_HA"
FEATURES_24 = [
    "s2_ndvi_peak", "s2_ndvi_auc", "s2_ndvi_los", "s2_ndvi_sos_doy",
    "s2_ndvi_base", "s2_ndvi_pico", "s2_evi_base", "s2_evi_pico",
    "s2_lai_pico", "s2_msi_pico", "s2_ndwi_pico", "s2_ndti_pre",
    "s2_crc_pre", "s2_nddi_pico", "pl_ndvi_peak", "pl_ndvi_auc",
    "pl_ndvi_los", "pl_evi_pico", "pl_lai_pico", "ls_vi6t_pico",
    "topo_elev_media", "topo_pend_media", "geo_area_ha", "geo_compacidad",
]
# Selección agronómica FIJA, anterior a la validación: no depende de la etiqueta.
FEATURES_8 = [
    "s2_ndvi_peak", "s2_evi_pico", "s2_lai_pico", "pl_ndvi_auc",
    "s2_msi_pico", "topo_elev_media", "topo_pend_media", "geo_area_ha",
]


def csv_en_zip(archivo_zip: Path, terminacion: str) -> pd.DataFrame:
    with zipfile.ZipFile(archivo_zip) as z:
        coincidencias = [n for n in z.namelist() if n.endswith(terminacion)]
        if len(coincidencias) != 1:
            raise ValueError(f"Se esperaba exactamente un CSV terminado en {terminacion}: {coincidencias}")
        return pd.read_csv(io.BytesIO(z.read(coincidencias[0])))


def estimador_ols() -> Pipeline:
    # Para evitar fugas de información, ajustar estos pasos DENTRO de cada fold.
    return Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler()),
        ("regresion_lineal", LinearRegression(fit_intercept=True)),
    ])


def metricas(y: np.ndarray, y_estimado: np.ndarray) -> dict:
    return {
        "MAE_t_ha": float(mean_absolute_error(y, y_estimado)),
        "RMSE_t_ha": float(np.sqrt(mean_squared_error(y, y_estimado))),
        "R2": float(r2_score(y, y_estimado)),
    }


def ejecutar(archivo_zip: Path, salida: Path) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    base = csv_en_zip(archivo_zip, "/cebada_datos_baseline_1/data_processed/features_parcela.csv")
    historico = csv_en_zip(archivo_zip, "/salidas/tabla_modelado_parcelaria.csv")

    if not base.ID_POLIGONO.is_unique or not historico.ID_POLIGONO.is_unique:
        raise ValueError("Se encontraron parcelas duplicadas; revisar las tablas")
    if set(base.ID_POLIGONO) != set(historico.ID_POLIGONO):
        raise ValueError("Los ID de ambas tablas no coinciden")
    historico = historico.set_index("ID_POLIGONO").loc[base.ID_POLIGONO].reset_index()
    if not np.allclose(base[OBJETIVO], historico[OBJETIVO], equal_nan=True):
        raise ValueError("Rendimiento no coincide entre ambas tablas")

    etiquetadas = base.CONJUNTO.eq("ENTRENAMIENTO") & base[OBJETIVO].notna()
    no_etiquetadas = base.CONJUNTO.eq("PREDICCION") & base[OBJETIVO].isna()
    if int(etiquetadas.sum()) != 138 or int(no_etiquetadas.sum()) != 59:
        raise ValueError("Se esperaba 138 parcelas etiquetadas y 59 sin etiqueta")

    columnas_148 = [c for c in historico if c not in (
        "ID_POLIGONO", "PARTICION", OBJETIVO, "AREA_HA")]
    if len(columnas_148) != 148:
        raise ValueError(f"Columnas históricas inesperadas: {len(columnas_148)}")

    modelos = {
        "Lineal_8_agronomicas": (base, FEATURES_8),
        "Lineal_24_curadas": (base, FEATURES_24),
        "Lineal_148_historicas": (historico, columnas_148),
    }
    y = base.loc[etiquetadas, OBJETIVO].to_numpy(dtype=float)
    ids = base.loc[etiquetadas, "ID_POLIGONO"].to_numpy()
    coords = base.loc[etiquetadas, ["loc_x_utm", "loc_y_utm"]].to_numpy(dtype=float)
    grupos = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit_predict(coords)

    particiones = {
        "aleatoria_5_folds": list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(y)),
        "espacial_6_bloques": list(GroupKFold(n_splits=6).split(y, groups=grupos)),
    }
    filas_metricas, filas_preds, filas_folds = [], [], []

    for nombre, (tabla, columnas) in modelos.items():
        X = tabla.loc[etiquetadas, columnas].astype(float).to_numpy()
        for escenario, folds in particiones.items():
            preds = np.full(len(y), np.nan)
            for fold, (indices_train, indices_test) in enumerate(folds, 1):
                pipeline = estimador_ols()
                pipeline.fit(X[indices_train], y[indices_train])
                pred_fold = pipeline.predict(X[indices_test])
                preds[indices_test] = pred_fold
                lm = pipeline.named_steps["regresion_lineal"]
                filas_folds.append({
                    "escenario": escenario, "modelo": nombre, "fold": fold,
                    "n_entrenamiento": len(indices_train), "n_validacion": len(indices_test),
                    "n_variables": len(columnas), "rango_matriz": int(lm.rank_),
                    **metricas(y[indices_test], pred_fold),
                })
            if not np.isfinite(preds).all():
                raise ValueError(f"Predicciones no finitas: {nombre} / {escenario}")
            filas_metricas.append({
                "escenario": escenario, "modelo": nombre,
                "n_parcelas": len(y), "n_variables": len(columnas),
                **metricas(y, preds),
                "pred_min_t_ha": float(preds.min()),
                "pred_max_t_ha": float(preds.max()),
                "n_pred_negativas": int((preds < 0).sum()),
            })
            filas_preds.extend({
                "escenario": escenario, "modelo": nombre, "ID_POLIGONO": p,
                "observado_t_ha": float(v), "estimado_t_ha": float(pred),
                "residual_t_ha": float(v - pred), "bloque_espacial": int(g),
            } for p, v, pred, g in zip(ids, y, preds, grupos))

    df_metricas = pd.DataFrame(filas_metricas)
    df_pred = pd.DataFrame(filas_preds)
    df_folds = pd.DataFrame(filas_folds)
    df_metricas.to_csv(salida / "metricas_regresion_lineal.csv", index=False)
    df_pred.to_csv(salida / "predicciones_validacion_lineal.csv", index=False)
    df_folds.to_csv(salida / "metricas_por_fold_lineal.csv", index=False)

    diagnosticos = []
    for nombre, (tabla, columnas) in modelos.items():
        X = tabla.loc[etiquetadas, columnas].astype(float).to_numpy()
        pipeline = estimador_ols().fit(X, y)
        regresion = pipeline.named_steps["regresion_lineal"]
        pred_train = pipeline.predict(X)
        diagnosticos.append({
            "modelo": nombre, "n_variables": len(columnas),
            "n_entrenamiento": len(y), "rango_matriz": int(regresion.rank_),
            **{f"entrenamiento_{clave}": valor for clave, valor in metricas(y, pred_train).items()},
            "norma_coeficientes": float(np.linalg.norm(regresion.coef_)),
            "coeficiente_abs_max": float(np.max(np.abs(regresion.coef_))),
        })
        joblib.dump({
            "pipeline": pipeline, "columnas": columnas, "nombre": nombre,
            "objetivo": OBJETIVO,
            "advertencia": "Exploratorio; etiquetas sin ciclo fechado; no validado para 2026. NO usar OLS-148 en producción.",
        }, salida / f"{nombre}.joblib")
        pd.DataFrame({
            "variable": columnas,
            "coeficiente_por_1_desviacion_estandar": regresion.coef_,
            "magnitud_absoluta": np.abs(regresion.coef_),
        }).sort_values("magnitud_absoluta", ascending=False).to_csv(
            salida / f"coeficientes_{nombre}.csv", index=False)

    pd.DataFrame(diagnosticos).to_csv(salida / "diagnostico_sobreajuste_lineal.csv", index=False)

    import matplotlib.pyplot as plt
    # Visualización de los dos OLS con dimensiones razonables.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, nombre in zip(axes, ["Lineal_8_agronomicas", "Lineal_24_curadas"]):
        d = df_pred[(df_pred.escenario == "espacial_6_bloques") & (df_pred.modelo == nombre)]
        ax.scatter(d.observado_t_ha, d.estimado_t_ha, alpha=0.65, s=26)
        ax.plot([1.5, 6], [1.5, 6], "k--", linewidth=1)
        m = df_metricas[(df_metricas.escenario == "espacial_6_bloques") & (df_metricas.modelo == nombre)].iloc[0]
        ax.set_title(f"{nombre}\nRMSE={m.RMSE_t_ha:.3f} t/ha · R²={m.R2:.3f}")
        ax.set_xlabel("Rendimiento observado (t/ha)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Rendimiento estimado (t/ha)")
    fig.suptitle("Regresión lineal — validación por 6 bloques espaciales")
    fig.tight_layout()
    fig.savefig(salida / "lineal_validacion_espacial.png", dpi=175)
    plt.close(fig)

    # La escala log permite apreciar también la inestabilidad del OLS-148.
    m_esp = df_metricas[df_metricas.escenario.eq("espacial_6_bloques")]
    orden = ["Lineal_8_agronomicas", "Lineal_24_curadas", "Lineal_148_historicas"]
    ordenado = m_esp.set_index("modelo").loc[orden]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    barras = ax.bar(["OLS 8", "OLS 24", "OLS 148"], ordenado.RMSE_t_ha.to_numpy())
    ax.set_yscale("log")
    ax.set_ylabel("RMSE espacial (t/ha) · escala logarítmica")
    ax.set_title("Error espacial OLS y sobreajuste con 148 variables")
    for barra, error in zip(barras, ordenado.RMSE_t_ha):
        ax.text(barra.get_x() + barra.get_width()/2, error*1.07, f"{error:.3f}", ha="center")
    ax.set_ylim(0.4, 18)
    fig.tight_layout()
    fig.savefig(salida / "comparacion_rmse_lineal.png", dpi=175)
    plt.close(fig)

    resumen = {
        "parcelas_etiquetadas": int(etiquetadas.sum()),
        "parcelas_sin_etiqueta": int(no_etiquetadas.sum()),
        "objetivo": OBJETIVO,
        "escenarios": {k: len(v) for k, v in particiones.items()},
        "tamanos_bloques": {str(k): int(v) for k, v in enumerate(np.bincount(grupos))},
        "metricas": filas_metricas,
        "diagnosticos_entrenamiento": diagnosticos,
        "advertencia": "No se documenta ciclo/año de los rendimientos; no se pueden validar predicciones 2026. OLS-148 sobreajusta.",
    }
    (salida / "resumen_experimento_lineal.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== MÉTRICAS REGRESIÓN LINEAL ===")
    print(df_metricas.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== DIAGNÓSTICO SOBREAJUSTE ===")
    print(pd.DataFrame(diagnosticos).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResultados guardados en: {salida.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Modelo OLS AgroCebada y validación espacial/aleatoria")
    parser.add_argument("--zip", type=Path, default=Path(__file__).resolve().parent / "DATASET_RETO_AGRO_2026.zip")
    parser.add_argument("--salida", type=Path, default=Path(__file__).resolve().parent / "resultados_regresion_lineal_agro")
    args = parser.parse_args()
    if not args.zip.is_file():
        parser.error(f"No se encuentra el dataset ZIP: {args.zip}")
    ejecutar(args.zip, args.salida)


if __name__ == "__main__":
    main()
