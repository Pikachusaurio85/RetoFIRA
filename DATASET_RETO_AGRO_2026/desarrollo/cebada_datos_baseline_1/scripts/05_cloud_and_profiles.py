import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from paths import DATA_PROCESSED, FIGURES
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 100)
W=str(DATA_PROCESSED); F=str(FIGURES)
b = pd.read_pickle(f"{W}/_basico_raw.pkl"); p = pd.read_pickle(f"{W}/_pro_raw.pkl"); g = pd.read_pickle(f"{W}/parcelas_base.pkl")
s2 = b[(b.sensor=='Sentinel-2')&b.ndvi_promedio.notna()&(b.fecha.dt.year==2025)].merge(g[['ID_POLIGONO','CONJUNTO','RENDIMIENTO_T_HA','Municipio']], on='ID_POLIGONO')
pl = p[p.ndvi_promedio.notna()].merge(g[['ID_POLIGONO','CONJUNTO','RENDIMIENTO_T_HA','Municipio']], on='ID_POLIGONO')
# efecto de nubosidad parcial sobre NDVI (Jul-Sep, donde el cultivo esta verde): NDVI relativo a la mediana de la parcela en ese mes con nubosidad 0
for name, d in [('S2', s2), ('Planet', pl)]:
    d = d[d.fecha.dt.month.between(7,9)].copy()
    ref = d[d.porcentaje_nubosidad==0].groupby(['ID_POLIGONO', d.fecha.dt.month]).ndvi_promedio.median().rename('ref')
    d = d.join(ref, on=['ID_POLIGONO', d.fecha.dt.month]); d['dev'] = d.ndvi_promedio - d.ref
    d['bin'] = pd.cut(d.porcentaje_nubosidad, [-1,0,5,10,20,30,50,80,100])
    print(name, "desviacion NDVI vs mediana mensual de parcela (obs sin nubes), por bin de nubosidad"); print(d.groupby('bin', observed=True).dev.agg(['count','mean','std']).round(3))
# perfiles temporales medios 2025 por grupo de municipio y por tercil de rendimiento
tr = s2[s2.CONJUNTO=='ENTRENAMIENTO'].copy(); tr = tr[tr.porcentaje_nubosidad<=10]
tr['q'] = pd.qcut(tr.RENDIMIENTO_T_HA, 3, labels=['bajo','medio','alto'])
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
for k, grp in tr.groupby('q', observed=True):
    s = grp.groupby(grp.fecha.dt.to_period('SM-1' if False else 'M').dt.to_timestamp() + pd.Timedelta(days=14)).ndvi_promedio.median(); ax[0].plot(s.index, s.values, marker='o', label=f'rend {k}')
ax[0].set_title('S2 NDVI mediano mensual 2025 (nub<=10%) por tercil rendimiento'); ax[0].legend()
for k, grp in tr.groupby('Municipio'):
    if grp.ID_POLIGONO.nunique() >= 10:
        s = grp.groupby(grp.fecha.dt.to_period('M').dt.to_timestamp() + pd.Timedelta(days=14)).ndvi_promedio.median(); ax[1].plot(s.index, s.values, marker='o', label=k)
ax[1].set_title('S2 NDVI por municipio'); ax[1].legend(fontsize=8)
trp = pl[(pl.CONJUNTO=='ENTRENAMIENTO')&(pl.porcentaje_nubosidad<=10)].copy(); trp['q'] = pd.qcut(trp.RENDIMIENTO_T_HA, 3, labels=['bajo','medio','alto'])
for k, grp in trp.groupby('q', observed=True):
    s = grp.groupby(grp.fecha.dt.to_period('M').dt.to_timestamp() + pd.Timedelta(days=14)).ndvi_promedio.median(); ax[2].plot(s.index, s.values, marker='o', label=f'rend {k}')
ax[2].set_title('Planet NDVI mediano mensual 2025 por tercil rendimiento'); ax[2].legend()
for a in ax: a.tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.savefig(f"{F}/perfiles_ndvi_2025.png", dpi=110)
# hist rendimiento
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); t = g[g.CONJUNTO=='ENTRENAMIENTO']
ax[0].hist(t.RENDIMIENTO_T_HA, bins=30); ax[0].set_title('Rendimiento (t/ha), train n=138')
for k, d in t.groupby('Municipio'):
    if len(d)>=10: ax[1].hist(d.RENDIMIENTO_T_HA, bins=15, alpha=.6, label=k)
ax[1].legend(); ax[1].set_title('por municipio (n>=10)'); plt.tight_layout(); plt.savefig(f"{F}/hist_rendimiento.png", dpi=110)
print("ok")
