"""Experimentos para medir el aporte real de los satelites sobre un 'prior' geografico (municipio / vecinos) y con conjuntos curados de features."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial import cKDTree
from paths import DATA_PROCESSED, REPORTS
W=str(DATA_PROCESSED); Rp=str(REPORTS)
d = pd.read_csv(f"{W}/features_parcela.csv", index_col=0); tr = d[d.CONJUNTO=='ENTRENAMIENTO'].copy()
y = tr.RENDIMIENTO_T_HA.values; n = len(tr); mun = tr.loc_municipio.values; XY = tr[['loc_x_utm','loc_y_utm']].values
cols = lambda *pre: [c for c in d.columns if c.startswith(pre) and d[c].dtype != object]
SAT = cols('s2_','ls_','pl_'); SATT = cols('s2_','ls_','pl_','topo_','geo_')
CUR = ['s2_ndvi_peak','s2_ndvi_auc','s2_ndvi_los','s2_ndvi_sos_doy','s2_ndvi_base','s2_ndvi_pico','s2_evi_base','s2_evi_pico','s2_lai_pico','s2_msi_pico','s2_ndwi_pico','s2_ndti_pre','s2_crc_pre','s2_nddi_pico',
       'pl_ndvi_peak','pl_ndvi_auc','pl_ndvi_los','pl_evi_pico','pl_lai_pico','ls_vi6t_pico','topo_elev_media','topo_pend_media','geo_area_ha','geo_compacidad']
def prior_muni(tri, tei, shrink=3):
    gm = y[tri].mean()
    def f(m, excl=None):
        v = y[tri][mun[tri]==m]
        if excl is not None: v = np.delete(v, np.where(tri[mun[tri]==m]==excl)[0])
        return (v.sum()+shrink*gm)/(len(v)+shrink)
    return np.array([f(mun[i], i) for i in tri]), np.array([f(mun[i]) for i in tei])
def prior_knn(tri, tei, k=5):
    t = cKDTree(XY[tri]); dtr, itr = t.query(XY[tri], k=k+1); dtr, itr = dtr[:,1:], itr[:,1:]; dte, ite = t.query(XY[tei], k=k)
    f = lambda D, I: (y[tri][I]/np.maximum(D,1)).sum(1)/(1/np.maximum(D,1)).sum(1); return f(dtr, itr), f(dte, ite)
MK = {'Ridge': lambda s: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 30))),
      'RF5':   lambda s: RandomForestRegressor(300, min_samples_leaf=5, max_features=0.33, n_jobs=-1, random_state=s),
      'ET5':   lambda s: ExtraTreesRegressor(300, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=s),
      'PLS2':  lambda s: make_pipeline(StandardScaler(), PLSRegression(2))}
def evalcfg(name, fn, NREP=5):
    ms, oofs = [], []
    for r in range(NREP):
        oof = np.zeros(n)
        for tri, tei in KFold(5, shuffle=True, random_state=r).split(tr): oof[tei] = fn(tri, tei, r)
        ms.append((mean_squared_error(y, oof)**.5, mean_absolute_error(y, oof), r2_score(y, oof))); oofs.append(oof)
    o = np.mean(oofs, 0); wm = mun=='Chignahuapan'; ms = np.array(ms)
    print(f"{name:46s} RMSE={ms[:,0].mean():.3f}±{ms[:,0].std():.3f} MAE={ms[:,1].mean():.3f} R2={ms[:,2].mean():.3f} | R2 Chignahuapan={r2_score(y[wm], o[wm]):.3f}", flush=True)
    return dict(config=name, RMSE=ms[:,0].mean(), RMSE_sd=ms[:,0].std(), MAE=ms[:,1].mean(), R2=ms[:,2].mean(), R2_Chignahuapan=r2_score(y[wm], o[wm]))
out = []
for pname, pf in [('prior=municipio', prior_muni), ('prior=kNN5', prior_knn)]:
    for fsname, fs in [('Sat+Topo+Geom', SATT), ('Curado', CUR)]:
        for mname, mk in MK.items():
            def fn(tri, tei, s, pf=pf, fs=fs, mk=mk):
                ptr, pte = pf(tri, tei); m = mk(s); m.fit(tr[fs].values[tri], y[tri]-ptr); return pte + np.ravel(m.predict(tr[fs].values[tei]))
            out.append(evalcfg(f"{pname} + residuo[{fsname}|{mname}]", fn))
# modelos directos con set curado (sin ubicacion)
for mname, mk in MK.items():
    def fn(tri, tei, s, mk=mk):
        m = mk(s); m.fit(tr[CUR].values[tri], y[tri]); return np.ravel(m.predict(tr[CUR].values[tei]))
    out.append(evalcfg(f"directo[Curado|{mname}]", fn))
pd.DataFrame(out).to_csv(f"{Rp}/experimentos_residuo_curado.csv", index=False)
