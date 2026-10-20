import pandas as pd, numpy as np
from paths import DATA_PROCESSED
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
W=str(DATA_PROCESSED)
d = pd.read_csv(f"{W}/features_parcela.csv", index_col=0)
tr = d[d.CONJUNTO=='ENTRENAMIENTO']; y='RENDIMIENTO_T_HA'
num = [c for c in d.columns if c not in ('CONJUNTO',y,'loc_municipio','loc_estado','loc_celda_clima')]
print("constantes:", [c for c in num if d[c].nunique()<=1])
print(d[['s2_ndvi_peak','s2_ndvi_peak_doy','s2_ndvi_sos_doy','s2_ndvi_eos_doy','s2_ndvi_los','pl_ndvi_peak','pl_ndvi_peak_doy','pl_ndvi_sos_doy','pl_ndvi_eos_doy','pl_ndvi_los']].describe().T.round(2))
# train vs test: distribucion similar? (KS por feature)
from scipy.stats import ks_2samp
te = d[d.CONJUNTO=='PREDICCION']
ks = pd.Series({c: ks_2samp(tr[c], te[c]).pvalue for c in num}).sort_values(); print("\nfeatures con p<0.01 en KS train vs test:", (ks<0.01).sum(), "de", len(ks)); print(ks.head(8).round(4))
cor = tr[num+[y]].corr(method='spearman')[y].drop(y)
print("\nTOP |spearman| con rendimiento (train, n=138):"); print(cor.reindex(cor.abs().sort_values(ascending=False).index).head(30).round(3))
# dentro de cada regimen de municipio
big = tr[tr.loc_municipio=='Chignahuapan']; print("\nDentro de Chignahuapan (n=%d) top:" % len(big)); c2 = big[num+[y]].corr(method='spearman')[y].drop(y); print(c2.reindex(c2.abs().sort_values(ascending=False).index).head(15).round(3))
low = tr[tr.loc_municipio.isin(['Calpulalpan','Singuilucan'])]; print("\nDentro de Calpulalpan+Singuilucan (n=%d) top:" % len(low)); c3 = low[num+[y]].corr(method='spearman')[y].drop(y); print(c3.reindex(c3.abs().sort_values(ascending=False).index).head(15).round(3))
# correlacion entre sensores NDVI pico
print("\nCorr s2_ndvi_peak vs pl_ndvi_peak:", d[['s2_ndvi_peak','pl_ndvi_peak']].corr().iloc[0,1].round(3))
