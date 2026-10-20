"""Baselines de rendimiento (t/ha): CV repetida (aleatoria) + CV por bloques espaciales. Referencias triviales + modelos de caja negra por conjunto de features."""
import pandas as pd, numpy as np, warnings, time, json; warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial import cKDTree
import lightgbm as lgb
from paths import DATA_PROCESSED, REPORTS
W=str(DATA_PROCESSED); Rp=str(REPORTS)
d = pd.read_csv(f"{W}/features_parcela.csv", index_col=0)
tr = d[d.CONJUNTO=='ENTRENAMIENTO'].copy(); y = tr.RENDIMIENTO_T_HA.values; n = len(tr)
cols = lambda *pre: [c for c in d.columns if c.startswith(pre) and d[c].dtype != object]
SETS = {
 'A_S2':            cols('s2_'),
 'B_Planet':        cols('pl_'),
 'C_Sat(S2+LS+PL)': cols('s2_','ls_','pl_'),
 'D_Sat+Topo+Geom': cols('s2_','ls_','pl_','topo_','geo_'),
 'E_D+Clima':       cols('s2_','ls_','pl_','topo_','geo_','clim_'),
 'F_E+Ubic+kNN':    cols('s2_','ls_','pl_','topo_','geo_','clim_','loc_lon','loc_lat','loc_x','loc_y') + ['knn_y'],
 'G_Solo Ubic+kNN': cols('loc_lon','loc_lat','loc_x','loc_y') + ['knn_y'],
}
XY = tr[['loc_x_utm','loc_y_utm']].values
def knn_lag(tr_idx, te_idx, k=5):
    """promedio ponderado (1/dist) del rendimiento de los k vecinos mas cercanos del conjunto de ENTRENAMIENTO del fold (leave-self-out para filas de train)"""
    t = cKDTree(XY[tr_idx]); out = np.zeros(len(tr_idx)+len(te_idx))
    dtr, itr = t.query(XY[tr_idx], k=k+1); dtr, itr = dtr[:,1:], itr[:,1:]
    dte, ite = t.query(XY[te_idx], k=k)
    f = lambda D, I: (y[tr_idx][I]/np.maximum(D,1.0)).sum(1)/(1/np.maximum(D,1.0)).sum(1)
    return f(dtr, itr), f(dte, ite)
def make_X(cols_, tr_idx, te_idx):
    base = [c for c in cols_ if c != 'knn_y']
    Xtr, Xte = tr[base].values[tr_idx], tr[base].values[te_idx]
    if 'knn_y' in cols_:
        a, b = knn_lag(tr_idx, te_idx); Xtr = np.c_[Xtr, a]; Xte = np.c_[Xte, b]
    return Xtr, Xte
MODELS = {
 'Ridge': lambda s: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))),
 'RF':    lambda s: RandomForestRegressor(300, min_samples_leaf=2, max_features=0.33, n_jobs=-1, random_state=s),
 'ET':    lambda s: ExtraTreesRegressor(300, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=s),
 'LGBM':  lambda s: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, random_state=s, verbose=-1, n_jobs=1),
}
mun = tr.loc_municipio.values
def trivial(kind, tr_idx, te_idx):
    if kind == 'media_global': return np.full(len(te_idx), y[tr_idx].mean())
    if kind == 'media_municipio':   # con encogimiento hacia la media global (peso n/(n+3))
        gm = y[tr_idx].mean(); out = []
        for m in mun[te_idx]:
            v = y[tr_idx][mun[tr_idx]==m]; out.append((v.sum() + 3*gm)/(len(v)+3))
        return np.array(out)
    if kind == 'knn_espacial': return knn_lag(tr_idx, te_idx)[1]
def metrics(yt, yp): return dict(RMSE=mean_squared_error(yt, yp)**.5, MAE=mean_absolute_error(yt, yp), R2=r2_score(yt, yp))
NREP = 5; res = []; oof_store = {}
def run(name, predict_fn, tag):
    ms, oofs = [], []
    for r in range(NREP):
        oof = np.zeros(n)
        for tri, tei in KFold(5, shuffle=True, random_state=r).split(tr):
            oof[tei] = predict_fn(tri, tei, r)
        ms.append(metrics(y, oof)); oofs.append(oof)
    m = pd.DataFrame(ms); o = np.mean(oofs, axis=0); oof_store[name] = o
    wm = tr.loc_municipio.values=='Chignahuapan'
    # CV por bloques espaciales (6 bloques k-means sobre coordenadas): sin vecinos cercanos de entrenamiento
    blocks = KMeans(6, n_init=10, random_state=0).fit_predict(XY); ob = np.zeros(n)
    for tri, tei in GroupKFold(6).split(tr, groups=blocks): ob[tei] = predict_fn(tri, tei, 0)
    mb = metrics(y, ob)
    res.append(dict(modelo=name, **tag, RMSE=m.RMSE.mean(), RMSE_sd=m.RMSE.std(), MAE=m.MAE.mean(), R2=m.R2.mean(), R2_sd=m.R2.std(),
                    R2_dentro_Chignahuapan=r2_score(y[wm], o[wm]), RMSE_bloques=mb['RMSE'], R2_bloques=mb['R2']))
    print(f"{name:42s} RMSE={m.RMSE.mean():.3f}±{m.RMSE.std():.3f} MAE={m.MAE.mean():.3f} R2={m.R2.mean():.3f} | R2 dentro Chignahuapan={r2_score(y[wm], o[wm]):.3f} | bloques RMSE={mb['RMSE']:.3f} R2={mb['R2']:.3f}", flush=True)
t0 = time.time()
for k in ['media_global','media_municipio','knn_espacial']:
    run(f"[ref] {k}", lambda tri, tei, s, k=k: trivial(k, tri, tei), dict(tipo='referencia', set=k, algoritmo=k))
for sname, sc in SETS.items():
    for mname, mk in MODELS.items():
        def fn(tri, tei, s, sc=sc, mk=mk):
            Xa, Xb = make_X(sc, tri, tei); mdl = mk(s); mdl.fit(Xa, y[tri]); return mdl.predict(Xb)
        run(f"{sname} | {mname}", fn, dict(tipo='modelo', set=sname, algoritmo=mname))
print("tiempo", round(time.time()-t0), "s")
R = pd.DataFrame(res); R.to_csv(f"{Rp}/baseline_resultados_cv.csv", index=False)
pd.DataFrame(oof_store, index=tr.index).assign(y=y, municipio=mun).to_csv(f"{Rp}/baseline_oof_predicciones.csv")
