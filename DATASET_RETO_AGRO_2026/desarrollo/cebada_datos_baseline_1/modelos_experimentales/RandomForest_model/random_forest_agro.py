"""Random Forest Regressor para AgroCebada 2026.

Evalua RF-8, RF-24 y RF-148 frente a los experimentos previos.
Usa las mismas particiones externas que ridge_regresion_agro.py y
regresion_lineal_agro.py (seed=2026). No hay ajuste de hiperparametros
sobre los folds externos: configuracion conservadora FIJA de antemano.

Uso:
    pip install pandas numpy scikit-learn matplotlib joblib
    python random_forest_agro.py
    python random_forest_agro.py --zip DATASET_RETO_AGRO_2026.zip

IMPORTANTE: El ciclo agricola de las etiquetas no esta documentado.
Las parcelas sin etiqueta no se usan para validar; NO se valida 2026.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline

SEED = 2026
TARGET = 'RENDIMIENTO_T_HA'
FEATURES_8 = [
    's2_ndvi_peak', 's2_evi_pico', 's2_lai_pico', 'pl_ndvi_auc',
    's2_msi_pico', 'topo_elev_media', 'topo_pend_media', 'geo_area_ha',
]
FEATURES_24 = [
    's2_ndvi_peak', 's2_ndvi_auc', 's2_ndvi_los', 's2_ndvi_sos_doy',
    's2_ndvi_base', 's2_ndvi_pico', 's2_evi_base', 's2_evi_pico',
    's2_lai_pico', 's2_msi_pico', 's2_ndwi_pico', 's2_ndti_pre',
    's2_crc_pre', 's2_nddi_pico', 'pl_ndvi_peak', 'pl_ndvi_auc',
    'pl_ndvi_los', 'pl_evi_pico', 'pl_lai_pico', 'ls_vi6t_pico',
    'topo_elev_media', 'topo_pend_media', 'geo_area_ha', 'geo_compacidad',
]
RF_CONFIG = dict(n_estimators=350, max_features=0.7,
                 min_samples_leaf=3, max_depth=None,
                 bootstrap=True, random_state=SEED, n_jobs=-1)


def csv_zip(zip_path: Path, suffix: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        paths = [n for n in z.namelist() if n.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError(f'Se esperaba 1 CSV con terminacion {suffix}: {paths}')
        return pd.read_csv(io.BytesIO(z.read(paths[0])))


def metrics(real, pred) -> dict:
    return dict(MAE_t_ha=float(mean_absolute_error(real, pred)),
                RMSE_t_ha=float(np.sqrt(mean_squared_error(real, pred))),
                R2=float(r2_score(real, pred)))


def make_pipeline() -> Pipeline:
    return Pipeline([
        ('imputacion', SimpleImputer(strategy='median')),
        ('random_forest', RandomForestRegressor(**RF_CONFIG))
    ])


def ejecutar(zip_path: Path, salida: Path):
    salida.mkdir(parents=True, exist_ok=True)
    base = csv_zip(zip_path, '/cebada_datos_baseline_1/data_processed/features_parcela.csv')
    hist = csv_zip(zip_path, '/salidas/tabla_modelado_parcelaria.csv')
    if not base.ID_POLIGONO.is_unique or not hist.ID_POLIGONO.is_unique:
        raise ValueError('ID_POLIGONO debe ser unico en ambas tablas')
    if set(base.ID_POLIGONO) != set(hist.ID_POLIGONO):
        raise ValueError('Las dos tablas contienen distintos IDs de parcela')
    hist = hist.set_index('ID_POLIGONO').loc[base.ID_POLIGONO].reset_index()
    if not np.allclose(base[TARGET], hist[TARGET], equal_nan=True):
        raise ValueError('Las etiquetas no coinciden entre tablas')

    mask = base.CONJUNTO.eq('ENTRENAMIENTO') & base[TARGET].notna()
    reserva = base.CONJUNTO.eq('PREDICCION') & base[TARGET].isna()
    if mask.sum() != 138 or reserva.sum() != 59:
        raise ValueError(f'Etiquetadas={mask.sum()}, reserva={reserva.sum()}; verificar dataset')

    cols_148 = [c for c in hist if c not in ['ID_POLIGONO','PARTICION',TARGET,'AREA_HA']]
    if len(cols_148) != 148:
        raise ValueError(f'Se esperaban 148 caracteristicas historicas: {len(cols_148)}')
    datasets = {
        'RF_8_agronomicas': (base, FEATURES_8),
        'RF_24_curadas': (base, FEATURES_24),
        'RF_148_historicas': (hist, cols_148),
    }

    y = base.loc[mask, TARGET].to_numpy(float)
    ids = base.loc[mask, 'ID_POLIGONO'].to_numpy()
    municipios = base.loc[mask, 'loc_municipio'].to_numpy()
    coords = base.loc[mask, ['loc_x_utm','loc_y_utm']].to_numpy(float)
    bloques = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit_predict(coords)
    particiones = {
        'aleatoria_5_folds': list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(y)),
        'espacial_6_bloques': list(GroupKFold(n_splits=6).split(y, groups=bloques)),
    }
    filas_metricas, filas_predicciones, filas_fold, filas_importancia = [], [], [], []
    Xs = {nombre: tabla.loc[mask, cols].astype(float).to_numpy()
          for nombre, (tabla, cols) in datasets.items()}

    for nombre, (tabla, cols) in datasets.items():
        X = Xs[nombre]
        for escenario, folds in particiones.items():
            pred = np.full(len(y), np.nan)
            for nfold, (tr, te) in enumerate(folds, 1):
                pipeline = make_pipeline().fit(X[tr], y[tr])
                pred[te] = pipeline.predict(X[te])
                pred_train = pipeline.predict(X[tr])
                filas_fold.append(dict(modelo=nombre, escenario=escenario, fold=nfold,
                                       n_train=len(tr), n_test=len(te),
                                       train_RMSE=float(np.sqrt(mean_squared_error(y[tr], pred_train))),
                                       **metrics(y[te], pred[te])))
                forest = pipeline.named_steps['random_forest']
                for col, imp in zip(cols, forest.feature_importances_):
                    filas_importancia.append(dict(modelo=nombre, escenario=escenario,
                                                 fold=nfold, variable=col,
                                                 importancia_impureza=float(imp)))
            if not np.all(np.isfinite(pred)):
                raise ValueError(f'No se calcularon todas las predicciones: {nombre}/{escenario}')
            filas_metricas.append(dict(modelo=nombre, escenario=escenario, n_parcelas=len(y),
                                       n_variables=len(cols), **metrics(y, pred),
                                       pred_min=float(pred.min()), pred_max=float(pred.max())))
            filas_predicciones.extend(dict(modelo=nombre, escenario=escenario,
                                          ID_POLIGONO=str(i), municipio=str(m),
                                          bloque_espacial=int(g), real_t_ha=float(r),
                                          predicho_t_ha=float(p), error_t_ha=float(r-p))
                                     for i, m, g, r, p in zip(ids, municipios, bloques, y, pred))
        # Modelo final entrenado con todas las 138 etiquetas, separado de la evaluacion CV.
        pipe_final = make_pipeline().fit(X, y)
        joblib.dump(dict(pipeline=pipe_final, columnas=list(cols), nombre=nombre,
                         objetivo=TARGET, config=RF_CONFIG,
                         advertencia='Uso exploratorio. No validado para rendimientos del ciclo 2026.'),
                    salida / f'{nombre}.joblib')
        pd.DataFrame({'variable': cols,
                      'importancia_impureza': pipe_final.named_steps['random_forest'].feature_importances_})\
            .sort_values('importancia_impureza', ascending=False)\
            .to_csv(salida / f'importancias_{nombre}.csv', index=False)

    df_met = pd.DataFrame(filas_metricas)
    df_pred = pd.DataFrame(filas_predicciones)
    df_fold = pd.DataFrame(filas_fold)
    df_imp = pd.DataFrame(filas_importancia)
    df_met.to_csv(salida / 'metricas_random_forest.csv', index=False)
    df_pred.to_csv(salida / 'predicciones_validacion_random_forest.csv', index=False)
    df_fold.to_csv(salida / 'metricas_por_fold_random_forest.csv', index=False)
    df_imp.to_csv(salida / 'importancias_por_fold_random_forest.csv', index=False)

    # Comparacion contra resultados reales de experimentos previos, si estan presentes.
    anterior_ridge = zip_path.parent / 'resultados_ridge_agrocebada' / 'metricas_ridge.csv'
    anterior_ols = zip_path.parent / 'resultados_regresion_lineal_agro' / 'metricas_regresion_lineal.csv'
    comparacion = df_met[['escenario','modelo','MAE_t_ha','RMSE_t_ha','R2']].copy()
    for ruta, nombres in [(anterior_ridge, ['Media_global', 'Media_municipal_suavizada', 'Ridge_24_curadas','Ridge_148_historicas']),
                          (anterior_ols, ['Lineal_24_curadas'])]:
        if ruta.exists():
            p = pd.read_csv(ruta)
            p = p.loc[p.modelo.isin(nombres), ['escenario','modelo','MAE_t_ha','RMSE_t_ha','R2']]
            comparacion = pd.concat([comparacion, p], ignore_index=True)
    comparacion.sort_values(['escenario','RMSE_t_ha'], inplace=True)
    comparacion.to_csv(salida / 'comparacion_modelos.csv', index=False)

    import matplotlib.pyplot as plt
    escenario = 'espacial_6_bloques'
    candidatos = ['RF_24_curadas', 'RF_148_historicas']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    low, high = float(y.min()) - .25, float(y.max()) + .25
    for ax, nombre in zip(axes, candidatos):
        d = df_pred[(df_pred.escenario==escenario) & (df_pred.modelo==nombre)]
        ax.scatter(d.real_t_ha, d.predicho_t_ha, s=27, alpha=.7)
        ax.plot([low,high],[low,high], 'k--', linewidth=1)
        rm = df_met[(df_met.escenario==escenario)&(df_met.modelo==nombre)].iloc[0]
        ax.set_title(f'{nombre}\nRMSE {rm.RMSE_t_ha:.3f} | R2 {rm.R2:.3f}')
        ax.set_xlabel('Observado (t/ha)')
        ax.set_xlim(low, high); ax.set_ylim(low, high)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Predicho (t/ha)')
    fig.suptitle('Random Forest - validacion espacial (predicciones fuera de muestra)')
    fig.tight_layout()
    fig.savefig(salida / 'random_forest_validacion_espacial.png', dpi=170)
    plt.close(fig)

    esp = comparacion[comparacion.escenario.eq(escenario)]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    sub = esp.sort_values('RMSE_t_ha')
    ax.barh(sub.modelo, sub.RMSE_t_ha)
    ax.invert_yaxis(); ax.set_xlabel('RMSE espacial (t/ha; menor es mejor)')
    ax.set_title('AgroCebada: Random Forest vs. Ridge, OLS y referencias')
    for i, v in enumerate(sub.RMSE_t_ha):
        ax.text(v + .008, i, f'{v:.3f}', va='center', fontsize=9)
    ax.set_xlim(0, max(sub.RMSE_t_ha)*1.15)
    fig.tight_layout()
    fig.savefig(salida / 'comparacion_rmse_espacial.png', dpi=170)
    plt.close(fig)

    # Resumen por municipio (interpretar con precaucion: municipios con 1-5 etiquetas)
    elegido = df_met[df_met.escenario.eq(escenario)].sort_values('RMSE_t_ha').iloc[0].modelo
    por_mun = df_pred[(df_pred.escenario==escenario) & (df_pred.modelo==elegido)].copy()
    datos_municipio = por_mun.groupby('municipio', as_index=False).agg(
        n_parcelas=('real_t_ha', 'count'),
        MAE_t_ha=('error_t_ha', lambda v: float(np.abs(v).mean())),
        RMSE_t_ha=('error_t_ha', lambda v: float(np.sqrt(np.square(v).mean()))),
        rendimiento_medio=('real_t_ha', 'mean'), estimacion_media=('predicho_t_ha', 'mean'))
    datos_municipio.to_csv(salida / 'errores_por_municipio.csv', index=False)
    summary = dict(etiquetadas=int(mask.sum()), reserva=int(reserva.sum()),
                   config=RF_CONFIG, modelos={n:len(c) for n,(d,c) in datasets.items()},
                   bloques={str(k):int(v) for k,v in enumerate(np.bincount(bloques))},
                   metricas=filas_metricas, mejor_rf_espacial=elegido,
                   advertencia='Resultados de validacion espacial no demuestran generalizacion a un ciclo 2026.')
    (salida / 'resumen_experimento_random_forest.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=int), encoding='utf-8')
    print('\nMETRICAS RANDOM FOREST')
    print(df_met.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    print('\nCOMPARACION ESPACIAL')
    print(esp.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    print('\nResultados:', salida.resolve())


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument('--zip', type=Path, default=root/'DATASET_RETO_AGRO_2026.zip')
    parser.add_argument('--salida', type=Path, default=root/'resultados_random_forest_agro')
    args = parser.parse_args()
    if not args.zip.is_file():
        parser.error(f'No existe {args.zip}')
    ejecutar(args.zip, args.salida)

if __name__ == '__main__':
    main()
