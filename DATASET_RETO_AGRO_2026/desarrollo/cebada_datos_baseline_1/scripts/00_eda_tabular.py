import sys, pandas as pd, numpy as np
from paths import DATA_PROCESSED, RAW_DATASET
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 200); pd.set_option('display.max_rows', 200)
D = sys.argv[1] if len(sys.argv) > 1 else str(RAW_DATASET)
idr = pd.read_csv(f"{D}/ID_area_rendimiento_70_30_Reto_AgroCebada.csv", encoding='utf-8-sig')
print("=== ID/area/rend"); print(idr.shape); print(idr.dtypes); print(idr.head())
idr = idr.loc[:, ~idr.columns.str.startswith('Unnamed')]
print(idr.CONJUNTO.value_counts())
print(idr.groupby('CONJUNTO')[['AREA_HA','RENDIMIENTO_T_HA']].describe().T)
print("IDs unicos:", idr.ID_POLIGONO.nunique())
tr = idr[idr.CONJUNTO=='ENTRENAMIENTO']
print("Rend nulos en train:", tr.RENDIMIENTO_T_HA.isna().sum())
print(tr.RENDIMIENTO_T_HA.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))
print("Rend ordenados (bottom 10 / top 10):"); print(tr.nsmallest(10,'RENDIMIENTO_T_HA')); print(tr.nlargest(10,'RENDIMIENTO_T_HA'))
print("Rend value counts top:"); print(tr.RENDIMIENTO_T_HA.round(2).value_counts().head(15))
print("corr area-rend", tr[['AREA_HA','RENDIMIENTO_T_HA']].corr(method='spearman'))

b = pd.read_csv(f"{D}/Conjunto_datos_BASICO_AgroCebada2026.csv", encoding='utf-8-sig')
print("\n=== BASICO", b.shape)
print(b.sensor.value_counts())
b['fecha'] = pd.to_datetime(b.fecha_captura, format='%d/%m/%Y', errors='coerce')
print("fechas NaT:", b.fecha.isna().sum(), b.fecha.min(), b.fecha.max())
print("parcelas:", b.ID_POLIGONO.nunique())
print(b.groupby(['sensor', b.fecha.dt.year]).size().unstack())
print("nubosidad:"); print(b.groupby('sensor').porcentaje_nubosidad.describe())
print("duplicados (ID,fecha,sensor):", b.duplicated(['ID_POLIGONO','fecha_captura','sensor']).sum())
print("duplicados filas completas:", b.duplicated().sum())
# non-null per sensor
nn = b.groupby('sensor').apply(lambda g: g.notna().mean()).T
print(nn.round(2).to_string())
b.to_pickle(DATA_PROCESSED / "_basico_raw.pkl")

p = pd.read_csv(f"{D}/Conjunto_datos_PRO_AgroCebada.csv", encoding='utf-8-sig')
print("\n=== PRO", p.shape); print(p.sensor.value_counts())
p['fecha'] = pd.to_datetime(p.fecha_captura, format='%d/%m/%Y', errors='coerce')
print("fechas NaT:", p.fecha.isna().sum(), p.fecha.min(), p.fecha.max(), "parcelas:", p.ID_POLIGONO.nunique())
print(p.groupby(p.fecha.dt.to_period('M')).size())
print(p.describe().T)
print("dup:", p.duplicated(['ID_POLIGONO','fecha_captura']).sum(), p.duplicated().sum())
print("nulos:", p.isna().mean().round(3).to_dict())
p.to_pickle(DATA_PROCESSED / "_pro_raw.pkl")

# cobertura por parcela
ids = set(idr.ID_POLIGONO)
print("\nParcelas en basico no en ID:", set(b.ID_POLIGONO)-ids, " | en ID no en basico:", ids-set(b.ID_POLIGONO))
print("Parcelas en PRO no en ID:", set(p.ID_POLIGONO)-ids, " | en ID no en PRO:", len(ids-set(p.ID_POLIGONO)))
