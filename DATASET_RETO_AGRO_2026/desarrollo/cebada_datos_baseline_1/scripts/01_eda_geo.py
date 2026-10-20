import sys, shapefile, pandas as pd, numpy as np, rasterio
from pyproj import CRS, Transformer
from paths import RAW_INNER
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 100)
R = str(RAW_INNER)
shp = f"{R}/Parcelas_Reto_AGC_CONJUNTO_70_30/Parcelas_Reto_AGC_CONJUNTO"
sf = shapefile.Reader(shp)
print("shape type:", sf.shapeType, "n:", len(sf))
print("fields:", sf.fields)
print(open(shp+".prj").read()[:600])
recs = [dict(zip([f[0] for f in sf.fields[1:]], r)) for r in sf.records()]
df = pd.DataFrame(recs); print(df.head()); print(df.describe(include='all').T)
print("bbox", sf.bbox)
# parts & points
s0 = sf.shape(0); print("n parts", len(s0.parts), "n pts", len(s0.points), "first pt", s0.points[0])
for tif in [f"{R}/Reto_AgroCebada_Topografia_INEGI_CEM4/Reto_AgroCebada_Topografia_INEGI_CEM4/Reto_AgroCebada_Topografia_INEGI_CEM4/Elevacion_INEGI_CEM4_120m.tif",
            f"{R}/Reto_AgroCebada_CHIRPS_Precipitacion_2022_2025/Reto_AgroCebada_CHIRPS_Precipitacion_2022_2025/Precipitacion_mensual/2025/PREC_2025_06.tif",
            f"{R}/Reto_AgroCebada_Temperatura_2022_2025_RECORTADA_CHIRPS/Reto_AgroCebada_Temperatura_2022_2025/Tmax/2025/Tmax_2025_06.tif"]:
    with rasterio.open(tif) as r:
        print(tif.split('/')[-1], r.crs, r.shape, r.res, r.bounds, r.nodata, r.dtypes, r.transform)
