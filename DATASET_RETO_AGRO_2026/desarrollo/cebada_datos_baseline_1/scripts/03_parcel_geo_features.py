"""Construye la tabla base de parcelas: atributos del shapefile + geometria + topografia (zonal stats) + clima mensual (celda CHIRPS/CHIRTS)."""
import shapefile, pandas as pd, numpy as np, rasterio, glob, os
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from pyproj import Transformer
from shapely.geometry import shape, Polygon
from paths import DATA_PROCESSED, RAW_DATASET, RAW_INNER
R = str(RAW_INNER); W = str(DATA_PROCESSED)
D = str(RAW_DATASET)
sf = shapefile.Reader(f"{R}/Parcelas_Reto_AGC_CONJUNTO_70_30/Parcelas_Reto_AGC_CONJUNTO")
names = [f[0] for f in sf.fields[1:]]
rows, geoms = [], []
for sr in sf.shapeRecords():
    rec = dict(zip(names, sr.record)); pts = [(x, y) for x, y in sr.shape.points]
    geoms.append(Polygon(pts)); rows.append(rec)
g = pd.DataFrame(rows).rename(columns={'ID_POLIGON':'ID_POLIGONO','área_ha':'area_ha_shp'})
g['Cultivo'] = g.Cultivo.str.upper().str.strip()
print("Cultivo:", g.Cultivo.value_counts().to_dict())
print("poligonos validos:", sum(p.is_valid for p in geoms), "/", len(geoms))
# geometria en UTM 14N (EPSG:32614) para area y perimetro en metros
to_utm = Transformer.from_crs(4326, 32614, always_xy=True)
from shapely.ops import transform
utm = [transform(to_utm.transform, p) for p in geoms]
g['area_ha_calc'] = [p.area/1e4 for p in utm]
g['perim_m'] = [p.length for p in utm]
g['compacidad'] = [4*np.pi*p.area/(p.length**2) for p in utm]       # 1 = circulo
g['lon'] = [p.centroid.x for p in geoms]; g['lat'] = [p.centroid.y for p in geoms]
g['x_utm'] = [p.centroid.x for p in utm]; g['y_utm'] = [p.centroid.y for p in utm]
print(g[['area_ha_shp','area_ha_calc']].describe().T)
print("max |shp-calc| ha:", (g.area_ha_shp-g.area_ha_calc).abs().max())
# --- unir con ID/rendimiento
idr = pd.read_csv(f"{D}/ID_area_rendimiento_70_30_Reto_AgroCebada.csv", encoding='utf-8-sig').loc[:, lambda d: ~d.columns.str.startswith('Unnamed')]
m = g.merge(idr, on='ID_POLIGONO', how='outer', suffixes=('_shp','_csv'), indicator=True)
print(m._merge.value_counts().to_dict())
print("CONJUNTO coincide:", (m.CONJUNTO_shp==m.CONJUNTO_csv).all(), "| |area csv - shp| max:", (m.AREA_HA-m.area_ha_shp).abs().max())
g = m.drop(columns=['_merge','CONJUNTO_shp']).rename(columns={'CONJUNTO_csv':'CONJUNTO'})
# --- topografia: zonal stats con mascara de poligono (reproyectado a EPSG:6372)
to_lcc = Transformer.from_crs(4326, 6372, always_xy=True)
lcc = [transform(to_lcc.transform, p) for p in geoms]
T = f"{R}/Reto_AgroCebada_Topografia_INEGI_CEM4"
def zonal(tif, name):
    out = []
    with rasterio.open(tif) as r:
        for poly in lcc:
            minx, miny, maxx, maxy = poly.bounds
            pad = 2*r.res[0]
            win = from_bounds(minx-pad, miny-pad, maxx+pad, maxy+pad, r.transform).round_offsets().round_lengths()
            arr = r.read(1, window=win, masked=True)
            tr = r.window_transform(win)
            mask = geometry_mask([poly], arr.shape, tr, invert=True, all_touched=False)
            vals = arr[mask & ~np.ma.getmaskarray(arr)]
            n_in = mask.sum()
            if vals.size == 0:   # parcela menor a un pixel: pixel del centroide
                mask = geometry_mask([poly], arr.shape, tr, invert=True, all_touched=True)
                vals = arr[mask & ~np.ma.getmaskarray(arr)]
            cx, cy = poly.centroid.x, poly.centroid.y
            cv = next(r.sample([(cx, cy)], masked=True))[0]
            out.append(dict(**{f"{name}_media": float(np.mean(vals)) if vals.size else np.nan,
                               f"{name}_std": float(np.std(vals)) if vals.size>1 else 0.0,
                               f"{name}_min": float(np.min(vals)) if vals.size else np.nan,
                               f"{name}_max": float(np.max(vals)) if vals.size else np.nan,
                               f"{name}_centroide": float(cv) if cv is not np.ma.masked else np.nan,
                               f"{name}_npix": int(vals.size)}))
    return pd.DataFrame(out)
topo = pd.concat([zonal(f"{T}/Elevacion_INEGI_CEM4_120m.tif", "elev"), zonal(f"{T}/Pendiente_INEGI_CEM4_120m_grados.tif", "pend")], axis=1)
# sf order == g order? g merged on ID so re-align by original shp order
base = pd.DataFrame(rows).rename(columns={'ID_POLIGON':'ID_POLIGONO'})[['ID_POLIGONO']]
topo['ID_POLIGONO'] = base.ID_POLIGONO.values
g = g.merge(topo, on='ID_POLIGONO', how='left')
print(g[['elev_media','elev_std','elev_npix','pend_media','pend_std','pend_npix']].describe().T)
print("NaN topo:", g[[c for c in g.columns if c.startswith(('elev','pend'))]].isna().sum().sum())
# --- clima: celda 0.05 grados que contiene el centroide
cells = {}
def sample_series(prefix, folder, fname_fmt, var):
    recs = []
    for y in range(2022, 2026):
        for mth in range(1, 13):
            tif = f"{folder}/{y}/{fname_fmt.format(y=y, m=mth)}"
            with rasterio.open(tif) as r:
                vals = [v[0] for v in r.sample(list(zip(g.lon, g.lat)))]
                nod = r.nodata
            recs.append(pd.DataFrame({'ID_POLIGONO': g.ID_POLIGONO, 'anio': y, 'mes': mth, var: [np.nan if (v==nod or np.isnan(v)) else float(v) for v in vals]}))
    return pd.concat(recs)
P = f"{R}/Reto_AgroCebada_CHIRPS_Precipitacion_2022_2025/Precipitacion_mensual"
Tn = f"{R}/Reto_AgroCebada_Temperatura_2022_2025"
clim = sample_series('p', P, "PREC_{y}_{m:02d}.tif", 'prec_mm')
for v in ['Tmin','Tmax']:
    clim = clim.merge(sample_series(v, f"{Tn}/{v}", v+"_{y}_{m:02d}.tif", v.lower()), on=['ID_POLIGONO','anio','mes'])
print("clima NaN:", clim.isna().sum().to_dict(), clim.shape)
# celda unica (fila,col) de cada parcela
with rasterio.open(f"{P}/2025/PREC_2025_06.tif") as r:
    rc = [r.index(x, y) for x, y in zip(g.lon, g.lat)]
g['celda_clima'] = [f"{a}_{b}" for a, b in rc]
print("parcelas:", len(g), "| celdas climaticas unicas:", g.celda_clima.nunique())
print(g.groupby('celda_clima').size().sort_values(ascending=False).head(10))
g.to_pickle(f"{W}/parcelas_base.pkl"); clim.to_pickle(f"{W}/clima_mensual_parcela.pkl")
g.to_csv(f"{W}/parcelas_base.csv", index=False)
print(g.groupby(['Estado','Municipio']).agg(n=('ID_POLIGONO','size'), ntr=('RENDIMIENTO_T_HA','count'), rend=('RENDIMIENTO_T_HA','mean'), elev=('elev_media','mean')).round(2))
