import pandas as pd, numpy as np
from paths import DATA_PROCESSED
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 100); pd.set_option('display.max_rows', 200)
W=str(DATA_PROCESSED)
g = pd.read_pickle(f"{W}/parcelas_base.pkl")
tr = g[g.CONJUNTO=='ENTRENAMIENTO'].copy(); te = g[g.CONJUNTO=='PREDICCION']
y = 'RENDIMIENTO_T_HA'
print(tr.groupby('Municipio')[y].describe().round(2))
print("\nTest por municipio:\n", te.Municipio.value_counts())
# "redondez" del valor reportado
tr['redondo'] = (tr[y]*2 % 1 == 0)  # multiplos de 0.5
print("\n% multiplos de 0.5 por municipio:"); print(tr.groupby('Municipio').redondo.agg(['mean','sum','count']).round(2))
print(pd.crosstab(tr.redondo, pd.cut(tr[y],[0,2.9,3.2,3.6,4.2,6])))
print("\nRend por redondo:"); print(tr.groupby('redondo')[y].describe().round(2))
print("\nTopo NaN:"); print(g[g[['elev_media','pend_media']].isna().any(axis=1)][['ID_POLIGONO','lon','lat','elev_media','pend_media','pend_npix','elev_npix','area_ha_shp','CONJUNTO']])
# correlaciones simples (spearman)
cols = ['area_ha_shp','compacidad','lon','lat','elev_media','pend_media','elev_std','pend_std']
print("\nSpearman con rendimiento (train):"); print(tr[cols+[y]].corr(method='spearman')[y].round(3))
# dentro de redondos / no redondos
for k, d in tr.groupby('redondo'):
    print(f"\nredondo={k} n={len(d)}"); print(d[cols+[y]].corr(method='spearman')[y].round(3).to_string())
# autocorrelacion espacial: vecino mas cercano en train
from scipy.spatial import cKDTree
xy = tr[['x_utm','y_utm']].values; t = cKDTree(xy); d, i = t.query(xy, k=2)
nn_y = tr[y].values[i[:,1]]; print("\nCorr rend vs rend vecino mas cercano (train):", np.corrcoef(tr[y].values, nn_y)[0,1].round(3), "| dist mediana vecino (m):", np.median(d[:,1]).round(0))
# distancia de cada test al train mas cercano vs distancia train-train
dt, _ = t.query(te[['x_utm','y_utm']].values, k=1); print("Dist test->train mas cercano (m) mediana:", np.median(dt).round(0), "| train->train:", np.median(d[:,1]).round(0))
# Celdas climaticas: rend por celda
print("\nRend por celda clima (n>=5):"); cc = tr.groupby('celda_clima')[y].agg(['count','mean','std']); print(cc[cc['count']>=5].round(2))
# area vs rend dentro de municipio
for mpo in ['Chignahuapan','Calpulalpan','Singuilucan','Almoloya']:
    d = tr[tr.Municipio==mpo]; print(mpo, len(d), "spearman area:", d[['area_ha_shp',y]].corr('spearman').iloc[0,1].round(2), "elev:", d[['elev_media',y]].corr('spearman').iloc[0,1].round(2))
