"""Extra Trees Regressor para AgroCebada 2026.

Repite exactamente las particiones y columnas de los experimentos previos
(Ridge, OLS, Random Forest), con 8, 24 y 148 características.

Ejecutar:
  python -m pip install numpy pandas scikit-learn matplotlib joblib
  python extra_trees_agro.py
  python extra_trees_agro.py --zip ruta/datos.zip --salida ruta/resultados

Limitaciones: 138 parcelas con rendimiento, algunas etiquetas regionalmente
repetidas, sin año agrícola comprobado. No es pronóstico validado de 2026.
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
from sklearn.ensemble import ExtraTreesRegressor
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
# Fijado antes de validar, no seleccionamos hiperparámetros con los folds de prueba.
ET_CONFIG = dict(n_estimators=350, min_samples_leaf=3, max_features=0.7,
                 bootstrap=False, max_depth=None, random_state=SEED, n_jobs=-1)


def csv_zip(zip_path: Path, suffix: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        files = [n for n in z.namelist() if n.endswith(suffix)]
        if len(files) != 1:
            raise ValueError(f'Se esperaba un CSV que terminara en {suffix}; encontrados: {files}')
        return pd.read_csv(io.BytesIO(z.read(files[0])))


def metricas(y, pred) -> dict:
    return dict(MAE_t_ha=float(mean_absolute_error(y, pred)),
                RMSE_t_ha=float(np.sqrt(mean_squared_error(y, pred))),
                R2=float(r2_score(y, pred)))


def pipeline() -> Pipeline:
    return Pipeline([
        ('imputacion', SimpleImputer(strategy='median')),
        ('extra_trees', ExtraTreesRegressor(**ET_CONFIG)),
    ])


def ejecutar(zip_path: Path, salida: Path) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    base = csv_zip(zip_path, '/cebada_datos_baseline_1/data_processed/features_parcela.csv')
    historicas = csv_zip(zip_path, '/salidas/tabla_modelado_parcelaria.csv')
    if not base.ID_POLIGONO.is_unique or not historicas.ID_POLIGONO.is_unique:
        raise ValueError('Hay identificadores de parcela repetidos.')
    if set(base.ID_POLIGONO) != set(historicas.ID_POLIGONO):
        raise ValueError('Identificadores inconsistentes entre tablas.')
    historicas = historicas.set_index('ID_POLIGONO').loc[base.ID_POLIGONO].reset_index()
    if not np.allclose(base[TARGET], historicas[TARGET], equal_nan=True):
        raise ValueError('Rendimientos diferentes entre los dos CSV.')

    mask = base.CONJUNTO.eq('ENTRENAMIENTO') & base[TARGET].notna()
    reserva = base.CONJUNTO.eq('PREDICCION') & base[TARGET].isna()
    if int(mask.sum()) != 138 or int(reserva.sum()) != 59:
        raise ValueError(f'Conjunto inesperado: entrenamiento={mask.sum()}, reserva={reserva.sum()}')
    cols_148 = [c for c in historicas if c not in ['ID_POLIGONO', 'PARTICION', TARGET, 'AREA_HA']]
    if len(cols_148) != 148:
        raise ValueError(f'Las variables historicas son {len(cols_148)} y se esperaban 148.')

    datasets = {
        'ET_8_agronomicas': (base, FEATURES_8),
        'ET_24_curadas': (base, FEATURES_24),
        'ET_148_historicas': (historicas, cols_148),
    }
    Xs = {name: tab.loc[mask, cols].astype(float).to_numpy()
          for name, (tab, cols) in datasets.items()}
    y = base.loc[mask, TARGET].to_numpy(float)
    ids = base.loc[mask, 'ID_POLIGONO'].to_numpy()
    municipios = base.loc[mask, 'loc_municipio'].to_numpy()
    coords = base.loc[mask, ['loc_x_utm', 'loc_y_utm']].to_numpy(float)
    bloques = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit_predict(coords)
    cv = {
        'aleatoria_5_folds': list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(y)),
        'espacial_6_bloques': list(GroupKFold(n_splits=6).split(y, groups=bloques)),
    }
    filas_metricas, filas_predicciones, filas_fold, filas_importancia, filas_reserva = [], [], [], [], []
    for nombre, (tabla, columnas) in datasets.items():
        X = Xs[nombre]
        for escenario, particiones in cv.items():
            pred_oof = np.full(len(y), np.nan)
            for nfold, (tr, te) in enumerate(particiones, start=1):
                mod = pipeline().fit(X[tr], y[tr])
                pred_oof[te] = mod.predict(X[te])
                pred_entreno = mod.predict(X[tr])
                filas_fold.append({
                    'modelo':nombre, 'escenario':escenario, 'fold':nfold,
                    'n_train':len(tr), 'n_test':len(te),
                    'train_RMSE':float(np.sqrt(mean_squared_error(y[tr], pred_entreno))),
                    **metricas(y[te], pred_oof[te]),
                })
                for c, v in zip(columnas, mod.named_steps['extra_trees'].feature_importances_):
                    filas_importancia.append({'modelo':nombre, 'escenario':escenario,
                                              'fold':nfold, 'variable':c,
                                              'importancia_impureza':float(v)})
            if not np.isfinite(pred_oof).all():
                raise RuntimeError(f'Predicciones incompletas para {nombre}, {escenario}')
            filas_metricas.append({
                'modelo':nombre, 'escenario':escenario,
                'n_parcelas':len(y), 'n_variables':len(columnas),
                **metricas(y, pred_oof),
                'pred_min':float(np.min(pred_oof)), 'pred_max':float(np.max(pred_oof)),
            })
            filas_predicciones.extend({
                'modelo':nombre, 'escenario':escenario, 'ID_POLIGONO':str(i),
                'municipio':str(m), 'bloque_espacial':int(b),
                'real_t_ha':float(obs), 'predicho_t_ha':float(pr),
                'error_t_ha':float(obs-pr),
            } for i, m, b, obs, pr in zip(ids, municipios, bloques, y, pred_oof))
        # Artefacto final entrenado con TODAS las etiquetas: no se usa para medir CV.
        fitted = pipeline().fit(X, y)
        joblib.dump({'pipeline':fitted, 'columnas':list(columnas), 'nombre':nombre,
                     'objetivo':TARGET, 'config':ET_CONFIG,
                     'advertencia':'Exploratorio, sin validacion del rendimiento 2026.'},
                    salida / f'{nombre}.joblib')
        # Estimaciones exploratorias para las 59 parcelas SIN etiqueta.
        # Son predicciones, no observaciones ni evaluación de la cosecha 2026.
        X_reserva = tabla.loc[reserva, columnas].astype(float).to_numpy()
        estimaciones = fitted.predict(X_reserva)
        for poligono, mun, valor in zip(base.loc[reserva, 'ID_POLIGONO'],
                                         base.loc[reserva, 'loc_municipio'], estimaciones):
            filas_reserva.append({'modelo':nombre, 'ID_POLIGONO':str(poligono),
                                  'municipio':str(mun), 'prediccion_t_ha':float(valor),
                                  'estado':'ESTIMACION_EXPLORATORIA_SIN_VERIFICACION_2026'})
        pd.DataFrame({'variable':columnas,
                      'importancia_impureza':fitted.named_steps['extra_trees'].feature_importances_})\
            .sort_values('importancia_impureza', ascending=False)\
            .to_csv(salida / f'importancias_{nombre}.csv', index=False)

    df_met = pd.DataFrame(filas_metricas)
    df_pred = pd.DataFrame(filas_predicciones)
    df_folds = pd.DataFrame(filas_fold)
    df_imp = pd.DataFrame(filas_importancia)
    df_met.to_csv(salida/'metricas_extra_trees.csv', index=False)
    df_pred.to_csv(salida/'predicciones_validacion_extra_trees.csv', index=False)
    df_folds.to_csv(salida/'metricas_por_fold_extra_trees.csv', index=False)
    df_imp.to_csv(salida/'importancias_por_fold_extra_trees.csv', index=False)
    pd.DataFrame(filas_reserva).to_csv(salida/'predicciones_reserva_extra_trees_NO_VALIDADAS.csv', index=False)

    # Resultados comparables de los experimentos previos (mismos folds y semillas).
    comparacion = df_met[['escenario','modelo','MAE_t_ha','RMSE_t_ha','R2']].copy()
    previos = [
        ('resultados_ridge_agrocebada/metricas_ridge.csv',
         ['Media_global','Media_municipal_suavizada', 'Ridge_24_curadas','Ridge_148_historicas']),
        ('resultados_regresion_lineal_agro/metricas_regresion_lineal.csv', ['Lineal_24_curadas']),
        ('resultados_random_forest_agro/metricas_random_forest.csv',
         ['RF_8_agronomicas','RF_24_curadas','RF_148_historicas']),
    ]
    for filename, models in previos:
        ruta = zip_path.parent/filename
        if ruta.exists():
            datos = pd.read_csv(ruta)
            comparacion = pd.concat([
                comparacion, datos.loc[datos.modelo.isin(models),
                                      ['escenario','modelo','MAE_t_ha','RMSE_t_ha','R2']],
            ], ignore_index=True)
    comparacion.sort_values(['escenario','RMSE_t_ha'], inplace=True)
    comparacion.to_csv(salida/'comparacion_modelos.csv', index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    escenario='espacial_6_bloques'
    # una figura con dos versiones para examinar dispersion espacial
    fig, axes=plt.subplots(1,2,figsize=(11,4.8),sharex=True,sharey=True)
    low, high=float(y.min())-.25,float(y.max())+.25
    for ax, modelo in zip(axes,['ET_24_curadas','ET_148_historicas']):
        datos=df_pred[(df_pred.escenario==escenario)&(df_pred.modelo==modelo)]
        ax.scatter(datos.real_t_ha, datos.predicho_t_ha,s=24,alpha=.7)
        ax.plot([low,high],[low,high],'k--',lw=1)
        fila=df_met[(df_met.escenario==escenario)&(df_met.modelo==modelo)].iloc[0]
        ax.set_title(f'{modelo}\nRMSE {fila.RMSE_t_ha:.3f} | R² {fila.R2:.3f}')
        ax.set_xlim(low,high);ax.set_ylim(low,high)
        ax.set_xlabel('Observado (t/ha)'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Predicho (t/ha)')
    fig.suptitle('Extra Trees: validacion espacial (fuera de muestra)')
    fig.tight_layout()
    fig.savefig(salida/'extra_trees_validacion_espacial.png',dpi=170)
    plt.close(fig)

    esp=comparacion[comparacion.escenario==escenario].sort_values('RMSE_t_ha')
    fig, ax=plt.subplots(figsize=(10,6))
    ax.barh(esp.modelo,esp.RMSE_t_ha)
    ax.invert_yaxis(); ax.set_xlabel('RMSE espacial (t/ha): menor es mejor')
    ax.set_title('Comparacion de modelos: AgroCebada')
    for i, v in enumerate(esp.RMSE_t_ha):
        ax.text(float(v)+.012,i,f'{v:.3f}',va='center',fontsize=9)
    ax.set_xlim(0,float(esp.RMSE_t_ha.max())*1.15)
    fig.tight_layout()
    fig.savefig(salida/'comparacion_rmse_espacial.png',dpi=170)
    plt.close(fig)

    imp=pd.read_csv(salida/'importancias_ET_148_historicas.csv').head(12)
    fig,ax=plt.subplots(figsize=(10,6.2))
    ax.barh(imp.variable.iloc[::-1], imp.importancia_impureza.iloc[::-1])
    ax.set_xlabel('Importancia por impureza (no causal)')
    ax.set_title('Extra Trees 148: 12 variables mas utilizadas')
    ax.tick_params(axis='y',labelsize=8)
    fig.tight_layout()
    fig.savefig(salida/'importancia_variables_et148.png',dpi=175)
    plt.close(fig)

    elegido=df_met[df_met.escenario==escenario].sort_values('RMSE_t_ha').iloc[0].modelo
    pmun=df_pred[(df_pred.escenario==escenario)&(df_pred.modelo==elegido)].copy()
    pmun.groupby('municipio',as_index=False).agg(
        n_parcelas=('real_t_ha','count'),
        MAE_t_ha=('error_t_ha',lambda v:float(np.abs(v).mean())),
        RMSE_t_ha=('error_t_ha',lambda v:float(np.sqrt(np.square(v).mean()))),
        rendimiento_medio=('real_t_ha','mean'), estimacion_media=('predicho_t_ha','mean')
    ).to_csv(salida/'errores_por_municipio.csv',index=False)
    resumen={'etiquetadas':int(mask.sum()),'sin_etiqueta':int(reserva.sum()),
             'config':ET_CONFIG,'bloques':np.bincount(bloques).tolist(),
             'modelos':{k:len(v[1]) for k,v in datasets.items()},
             'metricas':filas_metricas,'mejor_ET_espacial':elegido,
             'advertencia':'El ciclo agricola de las etiquetas no esta verificado. Sin validacion 2026.'}
    (salida/'resumen_experimento_extra_trees.json').write_text(
        json.dumps(resumen,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nRESULTADOS EXTRA TREES')
    print(df_met.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nCOMPARACION ESPACIAL')
    print(esp.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nResultados guardados en:',salida.resolve())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent
    parser.add_argument('--zip',type=Path,default=root/'DATASET_RETO_AGRO_2026.zip')
    parser.add_argument('--salida',type=Path,default=root/'resultados_extra_trees_agro')
    args=parser.parse_args()
    if not args.zip.is_file():
        parser.error(f'No se encontro el ZIP: {args.zip}')
    ejecutar(args.zip.resolve(), args.salida)

if __name__=='__main__':
    main()
