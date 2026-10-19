"""Ridge Regression para AgroCebada: evaluación aleatoria y espacial.

Requisitos: pip install numpy pandas scikit-learn matplotlib joblib
Uso: python modelos/Ridge_model/ridge_regresion_agro.py --zip DATASET_RETO_AGRO_2026.zip

La etiqueta no indica el año/ciclo productivo: los resultados NO validan
predicciones de la cosecha 2026. La reserva de 59 parcelas no tiene etiqueta.
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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_CURADAS = [
    "s2_ndvi_peak", "s2_ndvi_auc", "s2_ndvi_los", "s2_ndvi_sos_doy",
    "s2_ndvi_base", "s2_ndvi_pico", "s2_evi_base", "s2_evi_pico",
    "s2_lai_pico", "s2_msi_pico", "s2_ndwi_pico", "s2_ndti_pre",
    "s2_crc_pre", "s2_nddi_pico", "pl_ndvi_peak", "pl_ndvi_auc",
    "pl_ndvi_los", "pl_evi_pico", "pl_lai_pico", "ls_vi6t_pico",
    "topo_elev_media", "topo_pend_media", "geo_area_ha", "geo_compacidad",
]
ALPHAS = np.logspace(-2, 4, 13)
SEED = 2026


def leer_csv(zip_path: Path, sufijo: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as arc:
        archivos = [p for p in arc.namelist() if p.endswith(sufijo)]
        if len(archivos) != 1:
            raise ValueError(f"Se esperaba un archivo terminado en {sufijo!r}, encontrados: {archivos}")
        return pd.read_csv(io.BytesIO(arc.read(archivos[0])))


def modelo() -> Pipeline:
    # Imputación, normalización y selección de alfa SIEMPRE dentro del fold.
    return Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler()),
        ("ridge", RidgeCV(alphas=ALPHAS, cv=5, scoring="neg_mean_squared_error")),
    ])


def evaluar(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE_t_ha": float(mean_absolute_error(y_real, y_pred)),
        "RMSE_t_ha": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "R2": float(r2_score(y_real, y_pred)),
    }


def media_municipal(municipios_train: np.ndarray, y_train: np.ndarray,
                    municipios_test: np.ndarray, k: float = 3.0) -> np.ndarray:
    # Contraste geográfico; usa exclusivamente etiquetas de entrenamiento del fold.
    general = float(np.mean(y_train))
    resultado = []
    for municipio in municipios_test:
        valores = y_train[municipios_train == municipio]
        resultado.append((valores.sum() + k * general) / (len(valores) + k))
    return np.asarray(resultado)


def ejecutar(zip_path: Path, salida: Path, predecir_reserva: bool) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    base = leer_csv(zip_path, "/cebada_datos_baseline_1/data_processed/features_parcela.csv")
    historico = leer_csv(zip_path, "/salidas/tabla_modelado_parcelaria.csv")

    assert base["ID_POLIGONO"].is_unique and historico["ID_POLIGONO"].is_unique
    assert set(base["ID_POLIGONO"]) == set(historico["ID_POLIGONO"])
    historico = historico.set_index("ID_POLIGONO").loc[base["ID_POLIGONO"]].reset_index()
    if not np.allclose(base["RENDIMIENTO_T_HA"], historico["RENDIMIENTO_T_HA"], equal_nan=True):
        raise ValueError("Rendimientos no coinciden entre los dos CSV")

    etiqueta = "RENDIMIENTO_T_HA"
    entr = base["CONJUNTO"].eq("ENTRENAMIENTO") & base[etiqueta].notna()
    reserva = base["CONJUNTO"].eq("PREDICCION") & base[etiqueta].isna()
    if int(entr.sum()) != 138 or int(reserva.sum()) != 59:
        raise ValueError("Conteos de etiquetas inesperados; revisar origen de los datos")

    columnas_historico = [c for c in historico.columns
                          if c not in ("ID_POLIGONO", "PARTICION", etiqueta, "AREA_HA")]
    assert len(columnas_historico) == 148, len(columnas_historico)
    conjuntos = {
        "Ridge_24_curadas": (base, FEATURES_CURADAS),
        "Ridge_148_historicas": (historico, columnas_historico),
    }

    y = base.loc[entr, etiqueta].to_numpy(dtype=float)
    ids = base.loc[entr, "ID_POLIGONO"].to_numpy()
    municipio = base.loc[entr, "loc_municipio"].astype(str).to_numpy()
    coords = base.loc[entr, ["loc_x_utm", "loc_y_utm"]].to_numpy(dtype=float)
    grupos_espaciales = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit_predict(coords)

    metricas = []
    predicciones = []
    alfas = []
    escenarios = {
        "aleatoria_5_folds": list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(y)),
        "espacial_6_bloques": list(GroupKFold(n_splits=6).split(y, groups=grupos_espaciales)),
    }

    for escenario, folds in escenarios.items():
        preds = {nombre: np.full(len(y), np.nan) for nombre in [
            "Media_global", "Media_municipal_suavizada", *conjuntos.keys()]}
        for fold_num, (idx_train, idx_test) in enumerate(folds, start=1):
            preds["Media_global"][idx_test] = np.mean(y[idx_train])
            preds["Media_municipal_suavizada"][idx_test] = media_municipal(
                municipio[idx_train], y[idx_train], municipio[idx_test]
            )
            for nombre, (tabla, columnas) in conjuntos.items():
                X = tabla.loc[entr, columnas].astype(float).to_numpy()
                m = modelo()
                m.fit(X[idx_train], y[idx_train])
                preds[nombre][idx_test] = m.predict(X[idx_test])
                alfas.append({"escenario": escenario, "fold": fold_num,
                              "modelo": nombre, "alpha_optimo": float(m.named_steps["ridge"].alpha_)})
        for nombre, valores in preds.items():
            if not np.all(np.isfinite(valores)):
                raise ValueError(f"Predicciones no válidas para {nombre}")
            metricas.append({"escenario": escenario, "modelo": nombre,
                             "n_parcelas": len(y), **evaluar(y, valores)})
            predicciones.extend({"escenario": escenario, "modelo": nombre,
                                 "ID_POLIGONO": p, "real_t_ha": float(v),
                                 "predicho_t_ha": float(pv), "error_t_ha": float(v - pv),
                                 "bloque_espacial": int(g)}
                                for p, v, pv, g in zip(ids, y, valores, grupos_espaciales))

    df_metricas = pd.DataFrame(metricas)
    df_pred = pd.DataFrame(predicciones)
    df_metricas.to_csv(salida / "metricas_ridge.csv", index=False)
    df_pred.to_csv(salida / "predicciones_validacion_ridge.csv", index=False)
    pd.DataFrame(alfas).to_csv(salida / "alfas_elegidos_ridge.csv", index=False)

    for nombre, (tabla, columnas) in conjuntos.items():
        X_total = tabla.loc[entr, columnas].astype(float).to_numpy()
        m = modelo().fit(X_total, y)
        # Guardar bundle: modelo entrenado, columnas y metadatos de uso.
        bundle = {"modelo": m, "columnas": columnas, "nombre": nombre,
                  "objetivo": etiqueta, "advertencia":
                  "Modelo exploratorio; el año de las etiquetas no está documentado; no validado para 2026"}
        joblib.dump(bundle, salida / f"{nombre}.joblib")
        # Los predictores están estandarizados: comparar magnitudes es posible,
        # pero con variables correlacionadas NO implica importancia causal.
        betas = m.named_steps["ridge"].coef_
        pd.DataFrame({"variable": columnas, "coeficiente_estandarizado": betas,
                      "magnitud_absoluta": np.abs(betas)}).sort_values(
            "magnitud_absoluta", ascending=False
        ).to_csv(salida / f"coeficientes_{nombre}.csv", index=False)
        if predecir_reserva:
            X_reserva = tabla.loc[reserva, columnas].astype(float).to_numpy()
            pd.DataFrame({"ID_POLIGONO": tabla.loc[reserva, "ID_POLIGONO"].to_numpy(),
                          "estimacion_exploratoria_t_ha": m.predict(X_reserva),
                          "modelo": nombre}).to_csv(salida / f"reserva_{nombre}_NO_VALIDADA.csv", index=False)

    # Grafica: solo validación espacial; no confundir con entrenamiento o extrapolación a 2026.
    try:
        import matplotlib.pyplot as plt
        filtrado = df_pred[df_pred.escenario.eq("espacial_6_bloques") &
                            df_pred.modelo.isin(conjuntos)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
        for ax, (nombre, grp) in zip(axes, filtrado.groupby("modelo", sort=False)):
            ax.scatter(grp.real_t_ha, grp.predicho_t_ha, s=25, alpha=0.65)
            ax.plot([2, 5.6], [2, 5.6], linestyle="--", color="black", linewidth=1)
            met = df_metricas[(df_metricas.escenario == "espacial_6_bloques") & (df_metricas.modelo == nombre)].iloc[0]
            ax.set_title(f"{nombre}\nRMSE {met.RMSE_t_ha:.3f}, R² {met.R2:.3f}")
            ax.set_xlabel("Observado (t/ha)")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Predicho (t/ha)")
        fig.suptitle("Ridge — predicciones fuera de muestra por bloques espaciales")
        fig.tight_layout()
        fig.savefig(salida / "ridge_validacion_espacial.png", dpi=170)
        plt.close(fig)
    except ImportError:
        print("matplotlib no instalado: se omite la imagen")

    resumen = {
        "etiquetadas": int(entr.sum()), "reserva": int(reserva.sum()),
        "features_curadas": len(FEATURES_CURADAS),
        "features_historicas": len(columnas_historico),
        "n_bloques": 6,
        "tamano_bloques": pd.Series(grupos_espaciales).value_counts().sort_index().to_dict(),
        "metricas": metricas,
        "advertencia": "El año de las etiquetas es desconocido. No se garantiza transferibilidad a 2026."
    }
    (salida / "resumen_experimento.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=int), encoding="utf-8")
    print("\n=== MÉTRICAS RIDGE (validación fuera de muestra) ===")
    print(df_metricas.round(3).to_string(index=False))
    print("\nGuardado en:", salida.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar y validar Ridge AgroCebada")
    model_dir = Path(__file__).resolve().parent
    root = model_dir.parents[1]
    parser.add_argument("--zip", dest="zip_path", type=Path,
                        default=root / "DATASET_RETO_AGRO_2026.zip")
    parser.add_argument("--salida", type=Path,
                        default=model_dir / "resultados_ridge_agrocebada")
    parser.add_argument("--predecir-reserva", action="store_true",
                        help="Genera estimaciones exploratorias NO validadas para 59 parcelas")
    args = parser.parse_args()
    ejecutar(args.zip_path, args.salida, args.predecir_reserva)
