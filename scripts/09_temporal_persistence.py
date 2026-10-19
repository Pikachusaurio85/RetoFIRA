"""
09_temporal_persistence.py - TRANSFERENCIA TEMPORAL: NO EJECUTABLE con rendimiento (solo hay etiquetas de un ciclo, sin anio/ciclo declarado).
Lo que SI se puede hacer sin etiquetas: ¿la posicion relativa de una parcela dentro de su municipio en variables satelitales (Sentinel-2, nubes<=5%)
persiste entre anios 2022-2025? Si persiste, la senal 2025 puede reflejar rasgos estables de la parcela (suelo/manejo/cultivo habitual);
si no persiste, es en buena parte especifica de la temporada. No usa ningun rendimiento.
"""
from scipy.stats import spearmanr, rankdata
from common import *

R = "/home/claude/data_raw/DATASET_RETO_AGRO_2026/desarrollo/salidas/ml_evolucion_parcelas_2022_2025.csv"
cols = ["ID_POLIGONO", "ANIO_CAPTURA", "MES_CAPTURA", "DIA_DEL_ANIO", "BASICO__ndvi_promedio", "BASICO__evi_promedio", "BASICO__lai_promedio", "BASICO__porcentaje_nubosidad", "SENSOR_SENTINEL_2"]
d = pd.read_csv(R, usecols=cols)
d = d[(d.SENSOR_SENTINEL_2 == 1) & (d.BASICO__porcentaje_nubosidad <= 5)]
allf = load_all_features(); mun = allf.loc_municipio
feat = {}
for var in ("ndvi", "evi", "lai"):
    v = d[d[f"BASICO__{var}_promedio"].notna()]
    for yr, g in v.groupby("ANIO_CAPTURA"):
        season = g[(g.DIA_DEL_ANIO >= 152) & (g.DIA_DEL_ANIO <= 304)].groupby(["ID_POLIGONO", "MES_CAPTURA"])[f"BASICO__{var}_promedio"].median().unstack()
        feat[(var, "pico_estacional", yr)] = season.max(axis=1)
        feat[(var, "media_estacional", yr)] = season.mean(axis=1)
        base = g[(g.DIA_DEL_ANIO >= 1) & (g.DIA_DEL_ANIO <= 151)].groupby("ID_POLIGONO")[f"BASICO__{var}_promedio"].median()
        feat[(var, "base_ene_may", yr)] = base
F = pd.DataFrame(feat).reindex(allf.index)
def wrank(s):
    r = pd.Series(np.nan, index=s.index)
    for m, k in s.groupby(mun).groups.items():
        x = s.loc[k].dropna()
        if len(x) >= 3:
            r.loc[x.index] = rankdata(x) / len(x) - 0.5
    return r
rows = []
for (var, kind) in sorted({(a, b) for a, b, _ in F.columns}):
    for y1, y2 in [(2022, 2023), (2022, 2024), (2022, 2025), (2023, 2024), (2023, 2025), (2024, 2025)]:
        a, b = F[(var, kind, y1)], F[(var, kind, y2)]; ok = a.notna() & b.notna()
        ra, rb = wrank(a), wrank(b); ok2 = ra.notna() & rb.notna()
        rows.append(dict(variable=f"s2_{var}_{kind}", par_anios=f"{y1}-{y2}", n=int(ok.sum()), rho_global=spearmanr(a[ok], b[ok])[0], n_intra=int(ok2.sum()), rho_intra_municipio=spearmanr(ra[ok2], rb[ok2])[0]))
T = pd.DataFrame(rows); T.round(3).to_csv(f"{PATH['res']}/09_persistencia_temporal_sentinel2.csv", index=False)
print(T.pivot_table(index="variable", columns="par_anios", values="rho_intra_municipio").round(2).to_string())
print("\nmediana rho intra-municipio por variable"); print(T.groupby("variable").rho_intra_municipio.median().round(2))
