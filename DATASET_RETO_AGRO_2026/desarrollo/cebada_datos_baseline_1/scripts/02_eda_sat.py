import pandas as pd, numpy as np
from paths import DATA_PROCESSED
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 100)
W=str(DATA_PROCESSED)
b = pd.read_pickle(f"{W}/_basico_raw.pkl"); p = pd.read_pickle(f"{W}/_pro_raw.pkl")
s2 = b[b.sensor=='Sentinel-2'].copy(); ls = b[b.sensor=='Landsat'].copy()
# relacion nubosidad vs NaN
s2['null'] = s2.ndvi_promedio.isna()
print("S2: nubosidad cuando NDVI nulo vs no nulo"); print(s2.groupby('null').porcentaje_nubosidad.describe())
print(pd.crosstab(pd.cut(s2.porcentaje_nubosidad,[-1,0,10,30,50,80,99.99,100]), s2.null))
ls['null'] = ls.vi6t_promedio.isna()
print("LS: nubosidad (NaN count)", ls.porcentaje_nubosidad.isna().sum())
print(pd.crosstab(pd.cut(ls.porcentaje_nubosidad.fillna(-5),[-6,-1,0,10,30,50,80,99.99,100]), ls.null))
p['null'] = p.ndvi_promedio.isna()
print("PRO"); print(pd.crosstab(pd.cut(p.porcentaje_nubosidad,[-1,0,10,30,50,80,99.99,100]), p.null))
# valid obs per parcel in growing season 2025 Apr-Oct
gs = (s2.fecha.dt.year==2025)&(s2.fecha.dt.month.between(4,10))
v = s2[gs & ~s2.null]
c = v.groupby('ID_POLIGONO').size()
print("S2 validas Abr-Oct 2025 por parcela:"); print(c.describe()); print("parcelas sin obs:", 197 - c.shape[0])
gsp = (p.fecha.dt.month.between(4,10)) & ~p.null
cp = p[gsp].groupby('ID_POLIGONO').size(); print("PRO validas Abr-Oct 2025:"); print(cp.describe())
# frecuencia temporal S2 por parcela: fechas unicas por parcela en 2025
print("S2 fechas/parcela 2025:", s2[s2.fecha.dt.year==2025].groupby('ID_POLIGONO').fecha.nunique().describe())
# rangos plausibles
cols = [c for c in s2.columns if c.endswith('_promedio')]
print(s2[cols].describe(percentiles=[.01,.5,.99]).T)
print(ls[['vi6t_promedio','vi6t_std','vi6t_max','vi6t_min']].describe().T)
# valores extremos / inf
num = b.select_dtypes('number'); print("inf:", np.isinf(num).sum().sum())
# mes-a-mes S2 NDVI mediano valido en 2025
print(v.groupby(v.fecha.dt.month)[['ndvi_promedio','evi_promedio','lai_promedio','msi_promedio']].agg(['count','median']))
