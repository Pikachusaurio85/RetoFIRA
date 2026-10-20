import pandas as pd, numpy as np
base = "/mnt/FEDORA-DATOS/ModoZebada/DATASET_RETO_AGRO_2026 (1)/DATASET_RETO_AGRO_2026"
dp = base + "/desarrollo/cebada_datos_baseline_1/data_processed"
F = pd.read_csv(dp + "/features_parcela.csv")
print("features_parcela shape:", F.shape)
print("CONJUNTO counts:\n", F.CONJUNTO.value_counts(dropna=False))
print("\nRENDIMIENTO NaN by CONJUNTO:\n", F.groupby('CONJUNTO').RENDIMIENTO_T_HA.apply(lambda s: s.isna().sum()))
print("\nMunicipio counts (all):\n", F.loc_municipio.value_counts(dropna=False))
print("\nMunicipio x CONJUNTO:\n", pd.crosstab(F.loc_municipio, F.CONJUNTO))
print("\nYield describe by conjunto:\n", F.groupby('CONJUNTO').RENDIMIENTO_T_HA.describe())
y = F.RENDIMIENTO_T_HA.dropna()
print("\nYield multiples of 0.5 frac:", np.isclose((y*2)%2,0).mean())
print("unique yields sample:", sorted(y.unique())[:20])

PC = pd.read_csv(base + "/outputs/xai/perfiles_conceptuales.csv")
print("\nperfiles shape:", PC.shape, "ids match:", set(PC.ID_POLIGONO)==set(F.ID_POLIGONO))
concepts = ["C1a_ndvi_s2_mean","C2a_amplitud_ndvi","C3a_ndwi_s2_mean","C4a_prec_acum_abr_oct","C5a_elev_media"]
M = F.merge(PC, on="ID_POLIGONO", how="inner")
cand = {
 "C1a_ndvi_s2_mean": ["s2_ndvi_pre","s2_ndvi_veg","s2_ndvi_pico","s2_ndvi_sen"],
 "C2a_amplitud_ndvi": ["s2_ndvi_amp"],
 "C3a_ndwi_s2_mean": ["s2_ndwi_pre","s2_ndwi_veg","s2_ndwi_pico","s2_ndwi_sen"],
 "C4a_prec_acum_abr_oct": ["clim_prec_temp"],
 "C5a_elev_media": ["topo_elev_media"],
}
for c, cols in cand.items():
    print(f"\n{c}:")
    for col in cols:
        r = M[[c,col]].corr().iloc[0,1]
        print(f"   corr with {col}: {r:.4f}")
    # also test mean of window cols
    if len(cols)>1 and all(x in M for x in cols):
        mm = M[cols].mean(axis=1)
        print(f"   corr with mean({cols}): {M[c].corr(mm):.4f}")
