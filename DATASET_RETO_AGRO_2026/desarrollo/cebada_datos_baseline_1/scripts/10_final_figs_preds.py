"""Ajuste final de baselines, predicciones para las 59 parcelas de PREDICCION, importancias y figuras."""
import pandas as pd, numpy as np, warnings, matplotlib; warnings.filterwarnings('ignore'); matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
from paths import DATA_PROCESSED, FIGURES, REPORTS
W=str(DATA_PROCESSED); Rp=str(REPORTS); F=str(FIGURES)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
d = pd.read_csv(f"{W}/features_parcela.csv", index_col=0); tr = d[d.CONJUNTO=='ENTRENAMIENTO']; te = d[d.CONJUNTO=='PREDICCION']
y = tr.RENDIMIENTO_T_HA.values
cols = lambda *pre: [c for c in d.columns if c.startswith(pre) and d[c].dtype != object]
SATT = cols('s2_','ls_','pl_','topo_','geo_')
CUR = ['s2_ndvi_peak','s2_ndvi_auc','s2_ndvi_los','s2_ndvi_sos_doy','s2_ndvi_base','s2_ndvi_pico','s2_evi_base','s2_evi_pico','s2_lai_pico','s2_msi_pico','s2_ndwi_pico','s2_ndti_pre','s2_crc_pre','s2_nddi_pico',
       'pl_ndvi_peak','pl_ndvi_auc','pl_ndvi_los','pl_evi_pico','pl_lai_pico','ls_vi6t_pico','topo_elev_media','topo_pend_media','geo_area_ha','geo_compacidad']
# ---- predicciones finales
gm = y.mean(); mun_tr = tr.loc_municipio.values
def prior(m):
    v = y[mun_tr==m]; return (v.sum()+3*gm)/(len(v)+3)
p_tr = np.array([((y[mun_tr==m].sum() - y[j]) + 3*gm)/((mun_tr==m).sum() - 1 + 3) for j, m in enumerate(mun_tr)])   # leave-self-out
p_te = np.array([prior(m) for m in te.loc_municipio])
et = ExtraTreesRegressor(1000, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=42).fit(tr[SATT], y)
rg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 30))).fit(tr[CUR], y - p_tr)
pred = pd.DataFrame({'ID_POLIGONO': te.index, 'municipio': te.loc_municipio.values,
                     'pred_ref_media_municipio': p_te.round(3),
                     'pred_baseline1_satelital_ET': et.predict(te[SATT]).round(3),
                     'pred_baseline2_hibrido': (p_te + rg.predict(te[CUR])).round(3)})
pred.to_csv(f"{Rp}/baseline_predicciones_59_parcelas.csv", index=False)
print(pred.describe().round(2)); print("ridge alpha:", rg[-1].alpha_)
print("corr entre predicciones:\n", pred.iloc[:,2:].corr().round(2))
# ---- importancias por permutacion (CV 5-fold x 3) del baseline 1
imp = []
for r in range(3):
    for tri, tei in KFold(5, shuffle=True, random_state=r).split(tr):
        m = ExtraTreesRegressor(300, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=r).fit(tr[SATT].iloc[tri], y[tri])
        pi = permutation_importance(m, tr[SATT].iloc[tei], y[tei], n_repeats=5, random_state=r, scoring='neg_root_mean_squared_error', n_jobs=-1)
        imp.append(pi.importances_mean)
imp = pd.Series(np.mean(imp, axis=0), index=SATT).sort_values(ascending=False)
grp = lambda c: c.split('_')[0] + ('' if c.split('_')[0] in ('s2','ls','pl') else '')
imp.rename('importancia_perm_RMSE').to_csv(f"{Rp}/baseline1_importancias_permutacion.csv")
print(imp.head(15).round(4)); print("por fuente:", imp.groupby([grp(c) for c in imp.index]).sum().round(3).to_dict())
fig, ax = plt.subplots(figsize=(8, 5.5)); top = imp.head(20)[::-1]
ax.barh(top.index, top.values, color=['#2a7f62' if c.startswith('s2') else '#c05621' if c.startswith('pl') else '#6b46c1' if c.startswith('ls') else '#2b6cb0' for c in top.index])
ax.set_xlabel('Aumento de RMSE al permutar (t/ha), CV'); ax.set_title('Baseline 1 (ET, satélite+topo+geom): top-20 features'); plt.tight_layout(); plt.savefig(f"{F}/importancias_baseline1.png", dpi=120); plt.close()
# ---- comparacion de modelos
R = pd.read_csv(f"{Rp}/baseline_resultados_cv.csv")
best = R[R.tipo=='modelo'].sort_values('RMSE').groupby('set').head(1).copy(); refs = R[R.tipo=='referencia']
exp = pd.read_csv(f"{Rp}/experimentos_residuo_curado.csv")
rows = [(r.modelo, r.RMSE, r.RMSE_sd) for r in refs.itertuples()] + [(r.modelo, r.RMSE, r.RMSE_sd) for r in best.itertuples()]
rows.append(('[híbrido] municipio + Ridge(residuo, curado)', exp[exp.config.str.startswith('prior=municipio + residuo[Curado|Ridge')].RMSE.iloc[0], exp[exp.config.str.startswith('prior=municipio + residuo[Curado|Ridge')].RMSE_sd.iloc[0]))
C = pd.DataFrame(rows, columns=['m','rmse','sd']).sort_values('rmse', ascending=False)
fig, ax = plt.subplots(figsize=(9, 5.2)); colors = ['#9aa5b1' if m.startswith('[ref]') else '#2a7f62' if m.startswith('[híb') else '#2b6cb0' for m in C.m]
ax.barh(C.m, C.rmse, xerr=C.sd, color=colors); ax.set_xlabel('RMSE (t/ha) – CV 5-fold ×5 repeticiones (menor es mejor)')
for i, v in enumerate(C.rmse): ax.text(v+0.02, i, f"{v:.2f}", va='center', fontsize=9)
ax.set_title('Baselines: ¿cuánto aporta cada bloque de información?'); plt.tight_layout(); plt.savefig(f"{F}/comparacion_baselines.png", dpi=120); plt.close()
# ---- obs vs pred (OOF promedio)
O = pd.read_csv(f"{Rp}/baseline_oof_predicciones.csv", index_col=0)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
pal = {'Chignahuapan':'#2b6cb0','Calpulalpan':'#c05621','Singuilucan':'#2a7f62','Almoloya':'#6b46c1','Cuautepec de Hinojosa':'#b7791f'}
for a, key, ttl in [(ax[0], 'D_Sat+Topo+Geom | ET', 'Baseline 1: ET satélite+topo+geom (sin ubicación)'), (ax[1], '[ref] media_municipio', 'Referencia: media por municipio')]:
    for mname, grp_ in O.groupby('municipio'):
        a.scatter(grp_.y, grp_[key], s=22, alpha=.8, label=mname, color=pal.get(mname, '#718096'))
    a.plot([2, 5.8], [2, 5.8], 'k--', lw=.8); r2 = r2_score(O.y, O[key]); rm = mean_squared_error(O.y, O[key])**.5
    a.set_title(f"{ttl}\nR²={r2:.2f}  RMSE={rm:.2f}", fontsize=10); a.set_xlabel('Rendimiento observado (t/ha)')
ax[0].set_ylabel('Predicho fuera de muestra (t/ha)'); ax[1].legend(fontsize=7, frameon=False); plt.tight_layout(); plt.savefig(f"{F}/obs_vs_pred.png", dpi=120); plt.close()
# ---- mapa
fig, ax = plt.subplots(figsize=(7.5, 6)); 
sc = ax.scatter(tr.loc_lon, tr.loc_lat, c=y, s=np.clip(tr.geo_area_ha*6, 14, 150), cmap='viridis', edgecolor='k', lw=.3)
ax.scatter(te.loc_lon, te.loc_lat, marker='x', color='crimson', s=22, label='Predicción (rend. oculto)'); plt.colorbar(sc, label='Rendimiento (t/ha)')
for m, g_ in d.groupby('loc_municipio'):
    if len(g_) >= 9: ax.annotate(m, (g_.loc_lon.mean(), g_.loc_lat.mean()), fontsize=8, weight='bold', color='#222')
ax.set_xlabel('Longitud'); ax.set_ylabel('Latitud'); ax.set_title('Parcelas: rendimiento (train) y ubicación de las 59 de predicción'); ax.legend(frameon=False, loc='lower right', fontsize=8); plt.tight_layout(); plt.savefig(f"{F}/mapa_parcelas.png", dpi=120); plt.close()
print("figuras ok")
