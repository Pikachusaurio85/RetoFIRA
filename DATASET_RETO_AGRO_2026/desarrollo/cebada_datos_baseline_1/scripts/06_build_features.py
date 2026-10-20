"""Features por parcela: satelite (S2, Landsat, Planet) 2025 + clima + topografia + geometria. Salida: data_processed/features_parcela.csv"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from paths import DATA_PROCESSED
W=str(DATA_PROCESSED)
b = pd.read_pickle(f"{W}/_basico_raw.pkl"); p = pd.read_pickle(f"{W}/_pro_raw.pkl")
g = pd.read_pickle(f"{W}/parcelas_base.pkl"); clim = pd.read_pickle(f"{W}/clima_mensual_parcela.pkl")
ids = g.ID_POLIGONO.sort_values().tolist()
CLOUD = {'s2': 5, 'pl': 20, 'ls': 5}     # umbrales de nubosidad (%) tras analisis de sesgo (ver informe)
NB = 36                                   # dekadas (10 dias) en el anio
def dekad(d): return np.minimum((d.dt.dayofyear.values - 1)//10, NB-1)
WIN = {'pre': (91,151), 'veg': (152,212), 'pico': (213,273), 'sen': (274,304)}   # DOY (abr-may / jun-jul / ago-sep / oct)
centers = np.array([10*k+5.5 for k in range(NB)])

COMP = {}
def composite(df, col, year=2025):
    """matriz parcela x dekada: mediana de observaciones validas, interpolada linealmente (extrapola con el valor mas cercano)"""
    d = df[(df.fecha.dt.year==year) & df[col].notna()]
    d = d.assign(k=dekad(d.fecha))
    M = d.groupby(['ID_POLIGONO','k'])[col].median().unstack().reindex(index=ids, columns=range(NB))
    nvalid = M.notna().sum(axis=1)
    out = M.copy()
    for pid in ids:
        v = M.loc[pid].values; ok = ~np.isnan(v)
        out.loc[pid] = np.interp(centers, centers[ok], v[ok]) if ok.sum() >= 3 else np.nan
    COMP[f"{df.sensor.iloc[0]}|{col}"] = out
    return out, nvalid, M

def window_feats(C, prefix):
    res = {}
    for w, (a, z) in WIN.items():
        m = (centers >= a) & (centers <= z); res[f"{prefix}_{w}"] = C.loc[:, m].mean(axis=1)
    return pd.DataFrame(res)

def phenology(C, prefix):
    """metricas fenologicas sobre curva (suavizada) 2025"""
    sm = C.T.rolling(3, center=True, min_periods=1).mean().T
    rows = {}
    for pid in ids:
        y = sm.loc[pid].values
        if np.isnan(y).any(): rows[pid] = {}; continue
        base = np.median(y[(centers>=1)&(centers<=151)])           # ene-may: suelo/rastrojo
        sea = (centers>=152)&(centers<=304)                         # jun-oct
        ip = np.argmax(np.where(sea, y, -np.inf)); peak = y[ip]; amp = peak - base
        thr = base + 0.5*amp
        # inicio de temporada: primer cruce ascendente del 50% antes del pico; fin: primer cruce descendente tras el pico
        sos = np.nan
        for k in range(1, ip+1):
            if y[k-1] < thr <= y[k]: sos = centers[k-1] + (thr-y[k-1])/(y[k]-y[k-1])*10; break
        eos = np.nan
        for k in range(ip+1, NB):
            if y[k-1] >= thr > y[k]: eos = centers[k-1] + (y[k-1]-thr)/(y[k-1]-y[k])*10; break
        dy = np.diff(y)/10
        auc = np.sum(np.clip(y[(centers>=152)&(centers<=334)] - base, 0, None))*10
        rows[pid] = {f"{prefix}_base": base, f"{prefix}_peak": peak, f"{prefix}_amp": amp, f"{prefix}_peak_doy": centers[ip],
                     f"{prefix}_sos_doy": sos, f"{prefix}_eos_doy": eos, f"{prefix}_los": eos-sos if not np.isnan(sos) and not np.isnan(eos) else np.nan,
                     f"{prefix}_auc": auc, f"{prefix}_slope_up": dy[:ip].max() if ip>0 else np.nan, f"{prefix}_slope_down": dy[ip:].min() if ip<len(dy) else np.nan}
    return pd.DataFrame.from_dict(rows, orient='index')

F = pd.DataFrame(index=ids)
# ---------------- Sentinel-2
s2 = b[(b.sensor=='Sentinel-2') & b.ndvi_promedio.notna() & (b.porcentaje_nubosidad<=CLOUD['s2'])]
idx = ['ndvi','evi','crc','mcrc','ndsvi','srndi','mra','lai','fapar','savi','ndti','sti','ndwi','endwi','msi','emsi','dswi2','dswi3','dswi4','dswi5','nddi','enddi']
for ix in idx:
    C, nv, _ = composite(s2, f"{ix}_promedio"); F = F.join(window_feats(C, f"s2_{ix}"))
    if ix in ('ndvi','evi','lai'): F = F.join(phenology(C, f"s2_{ix}"))
# heterogeneidad intra-parcela (std espacial) en pico y NDVI min/max en pico
for ix in ['ndvi','evi']:
    for st in ['std','min','max']:
        C, _, _ = composite(s2, f"{ix}_{st}"); F[f"s2_{ix}_{st}_pico"] = window_feats(C, 'x')['x_pico']
# ---------------- Landsat VI6T
ls = b[(b.sensor=='Landsat') & b.vi6t_promedio.notna() & (b.porcentaje_nubosidad<=CLOUD['ls'])]
C, _, _ = composite(ls, 'vi6t_promedio'); F = F.join(window_feats(C, 'ls_vi6t'))
C, _, _ = composite(ls, 'vi6t_std'); F['ls_vi6t_std_pico'] = window_feats(C, 'x')['x_pico']
# ---------------- Planet
pl = p[p.ndvi_promedio.notna() & (p.porcentaje_nubosidad<=CLOUD['pl'])]
for ix in ['ndvi','evi','lai','msavi']:
    C, nv, _ = composite(pl, f"{ix}_promedio"); F = F.join(window_feats(C, f"pl_{ix}"))
    if ix in ('ndvi','evi','lai'): F = F.join(phenology(C, f"pl_{ix}"))
for ix in ['ndvi','evi']:
    C, _, _ = composite(pl, f"{ix}_std"); F[f"pl_{ix}_std_pico"] = window_feats(C, 'x')['x_pico']
# ---------------- Clima (celda CHIRPS/CHIRTS 0.05 grados del centroide)
c25 = clim[clim.anio==2025].pivot(index='ID_POLIGONO', columns='mes', values=['prec_mm','tmin','tmax']).reindex(ids)
cb = clim[clim.anio<=2024].groupby(['ID_POLIGONO','mes'])[['prec_mm','tmin','tmax']].mean().unstack().reindex(ids)   # climatologia 2022-2024
mw = {'pre':[4,5], 'veg':[6,7], 'pico':[8,9], 'sen':[10]}
for w, ms in mw.items():
    F[f"clim_prec_{w}"] = c25['prec_mm'][ms].sum(axis=1); F[f"clim_tmin_{w}"] = c25['tmin'][ms].mean(axis=1); F[f"clim_tmax_{w}"] = c25['tmax'][ms].mean(axis=1)
    F[f"clim_dtr_{w}"] = F[f"clim_tmax_{w}"] - F[f"clim_tmin_{w}"]
S = list(range(4, 11))
F['clim_prec_temp'] = c25['prec_mm'][S].sum(axis=1); F['clim_prec_prev'] = c25['prec_mm'][[1,2,3]].sum(axis=1) + clim[(clim.anio==2024)&(clim.mes.isin([11,12]))].groupby('ID_POLIGONO').prec_mm.sum().reindex(ids)
F['clim_tmin_temp'] = c25['tmin'][S].mean(axis=1); F['clim_tmax_temp'] = c25['tmax'][S].mean(axis=1); F['clim_tmean_temp'] = (F.clim_tmin_temp + F.clim_tmax_temp)/2
dim = np.array([30,31,30,31,31,30,31])   # dias abr..oct
tmean = (c25['tmin'][S] + c25['tmax'][S])/2
F['clim_gdd0_temp'] = (tmean.clip(lower=0)*dim).sum(axis=1)       # grados-dia base 0 C aproximados con medias mensuales
F['clim_tmin_minmes'] = c25['tmin'][S].min(axis=1)                  # mes mas frio (riesgo de helada)
F['clim_prec_cv'] = c25['prec_mm'][S].std(axis=1)/c25['prec_mm'][S].mean(axis=1)   # irregularidad mensual
F['clim_prec_anom_pct'] = 100*(F.clim_prec_temp/cb['prec_mm'][S].sum(axis=1) - 1)  # vs. media 2022-2024 (3 anios: referencia debil)
F['clim_tmax_anom'] = F.clim_tmax_temp - cb['tmax'][S].mean(axis=1)
# ---------------- Topografia + geometria + ubicacion
g2 = g.set_index('ID_POLIGONO').reindex(ids)
for c in ['elev_media','elev_std','pend_media','pend_std']: F[f"topo_{c}"] = g2[c]
F['geo_area_ha'] = g2.AREA_HA; F['geo_compacidad'] = g2.compacidad; F['geo_perim_m'] = g2.perim_m
F['loc_lon'] = g2.lon; F['loc_lat'] = g2.lat; F['loc_x_utm'] = g2.x_utm; F['loc_y_utm'] = g2.y_utm
F['loc_municipio'] = g2.Municipio; F['loc_estado'] = g2.Estado; F['loc_celda_clima'] = g2.celda_clima
out = pd.concat([g2[['CONJUNTO','RENDIMIENTO_T_HA']], F], axis=1); out.index.name = 'ID_POLIGONO'
out.to_csv(f"{W}/features_parcela.csv")
# series dekadales limpias (mediana de obs validas por dekada + interpolacion lineal), formato largo
long = []
for k, M in COMP.items():
    sensor, var = k.split('|'); t = M.stack().rename('valor').reset_index(); t.columns = ['ID_POLIGONO','dekada','valor']
    t['sensor'] = sensor; t['variable'] = var; long.append(t)
L = pd.concat(long); L['doy_centro'] = L.dekada*10 + 5.5
L[['ID_POLIGONO','sensor','variable','dekada','doy_centro','valor']].round(5).to_csv(f"{W}/series_dekadales_2025.csv", index=False)
print("series dekadales:", L.shape)
print(out.shape, "NaN por columna (max):", out.drop(columns=['RENDIMIENTO_T_HA']).isna().sum().sort_values(ascending=False).head(12).to_dict())
print(Q.describe().T.round(1))
