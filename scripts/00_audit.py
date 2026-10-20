"""
00_audit.py - Auditoria de datos e integridad (sin modelos de rendimiento).
  * etiquetas oficiales vs tabla de features; busqueda de etiquetas ocultas en CUALQUIER csv del zip
  * cuantizacion del rendimiento por municipio ("captura gruesa" multiplo de 0.5)
  * validacion adversaria 59 ocultas vs 138 etiquetadas (SOLO features, sin etiquetas)
  * redundancia de features (correlacion) y dimension efectiva por familia
"""
import glob
from scipy.stats import binomtest, chi2_contingency, ks_2samp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import *

RAW = "/home/claude/data_raw/DATASET_RETO_AGRO_2026/desarrollo"
d = load_all_features(); D = Data()
out = {}

# ---- 1. integridad etiquetas / asignacion
asg = pd.read_csv(f"{PATH['data']}/asignacion_oficial.csv", encoding="utf-8-sig"); asg = asg.loc[:, ~asg.columns.str.startswith("Unnamed")].set_index("ID_POLIGONO")
chk = dict(n_parcelas=len(d), n_entrenamiento=int((d.CONJUNTO == "ENTRENAMIENTO").sum()), n_prediccion=int((d.CONJUNTO == "PREDICCION").sum()),
           etiquetas_iguales_a_asignacion_oficial=bool(np.allclose(asg.RENDIMIENTO_T_HA.reindex(D.ids).values, D.y)),
           max_dif_area_ha=float((asg.AREA_HA.reindex(d.index) - d.geo_area_ha).abs().max()),
           etiquetas_en_PREDICCION_en_asignacion=int(asg[asg.CONJUNTO == "PREDICCION"].RENDIMIENTO_T_HA.notna().sum()),
           faltantes_en_features_numericas=int(d.drop(columns=META).isna().sum().sum()),
           features_constantes=int((d.drop(columns=META).nunique() <= 1).sum()),
           n_municipios=int(d.loc_municipio.nunique()), n_celdas_clima=int(d.loc_celda_clima.nunique()),
           n_celdas_clima_en_entrenamiento=int(D.tr.loc_celda_clima.nunique()))
pd.Series(chk).to_csv(f"{PATH['res']}/00_integridad.csv", header=["valor"])
print(pd.Series(chk))

# ---- 2. busqueda de etiquetas ocultas en todos los csv del zip
hidden = set(asg[asg.CONJUNTO == "PREDICCION"].index); recs = []
for f in sorted(glob.glob(f"{RAW}/**/*.csv", recursive=True)):
    try:
        h = pd.read_csv(f, nrows=2, encoding="utf-8-sig")
    except Exception:
        continue
    cols = [c for c in h.columns if "REND" in str(c).upper() or "YIELD" in str(c).upper()]
    if not cols:
        continue
    df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
    idc = "ID_POLIGONO" if "ID_POLIGONO" in df.columns else None
    for c in cols:
        nn = int(df.loc[df[idc].isin(hidden), c].notna().sum()) if idc else -1
        recs.append(dict(archivo=os.path.relpath(f, RAW), columna=c, tiene_ID_POLIGONO=bool(idc), valores_no_nulos_en_las_59_ocultas=nn))
pd.DataFrame(recs).to_csv(f"{PATH['res']}/00_escaneo_etiquetas_ocultas.csv", index=False)
print("archivos con columna de rendimiento:", len(recs), "| max no-nulos en ocultas:", max(r["valores_no_nulos_en_las_59_ocultas"] for r in recs))

# ---- 3. cuantizacion por municipio
tr = D.tr; y = tr.RENDIMIENTO_T_HA
is05 = np.isclose(y * 2, np.round(y * 2))
q = (tr.assign(mult05=is05).groupby("loc_municipio")
     .agg(n=("mult05", "size"), media=("RENDIMIENTO_T_HA", "mean"), sd=("RENDIMIENTO_T_HA", "std"), min=("RENDIMIENTO_T_HA", "min"), max=("RENDIMIENTO_T_HA", "max"),
          n_mult_0_5=("mult05", "sum"), n_valores_distintos=("RENDIMIENTO_T_HA", "nunique"), estado=("loc_estado", "first")))
q["frac_mult_0_5"] = q.n_mult_0_5 / q.n
q["p_binom_vs_2pct"] = [binomtest(int(k), int(n_), 0.02, alternative="greater").pvalue for k, n_ in zip(q.n_mult_0_5, q.n)]
q["regimen_captura"] = np.where(q.frac_mult_0_5 >= 0.5, "gruesa(0.5 t/ha)", "fina(continua)")
q.round(4).sort_values("n", ascending=False).to_csv(f"{PATH['res']}/00_cuantizacion_por_municipio.csv")
print(q.round(3).sort_values("n", ascending=False))
tot = binomtest(int(is05.sum()), 138, 0.02, alternative="greater")
print(f"multiplos de 0.5 global: {int(is05.sum())}/138, p(binom vs 2%) = {tot.pvalue:.2e}")
print("Calpulalpan valores:", tr[tr.loc_municipio == "Calpulalpan"].RENDIMIENTO_T_HA.value_counts().sort_index().to_dict())
print("Singuilucan valores:", tr[tr.loc_municipio == "Singuilucan"].RENDIMIENTO_T_HA.value_counts().sort_index().to_dict())
regime_map = q.regimen_captura.to_dict()
pd.Series(regime_map).to_csv(f"{PATH['cfg']}/regimen_captura_por_municipio.csv", header=["regimen"])

# ---- 4. el split 70/30 oficial: ¿las 59 ocultas son una submuestra aleatoria estratificada por municipio?
ct = pd.crosstab(d.loc_municipio, d.CONJUNTO); ct["frac_oculta"] = ct.PREDICCION / (ct.ENTRENAMIENTO + ct.PREDICCION)
print(ct.round(3))
big = ct[(ct.ENTRENAMIENTO + ct.PREDICCION) >= 9]
chi = chi2_contingency(big[["ENTRENAMIENTO", "PREDICCION"]].values)
print(f"chi2 homogeneidad de fraccion oculta entre municipios (n>=9): p={chi[1]:.3f}")
ct.to_csv(f"{PATH['res']}/00_split_oficial_por_municipio.csv")

# ---- 5. validacion adversaria (SOLO features; sin etiquetas)
lab = (d.CONJUNTO == "PREDICCION").astype(int).values
sets = {"todas_features_sin_ubicacion": D.F["ALLNOLOC"], "solo_ubicacion_xy": D.F["LOC"], "satelite": D.F["SAT"], "clima": D.F["CLIM"], "topo+geom": D.F["TOPO"] + D.F["GEOM"]}
rows = []
for name, cols in sets.items():
    X = d[cols].to_numpy(float)
    aucs = []
    for rep in range(5):
        skf = StratifiedKFold(5, shuffle=True, random_state=rep)
        p = cross_val_predict(ExtraTreesClassifier(300, min_samples_leaf=3, max_features=0.3, random_state=rep, n_jobs=1), X, lab, cv=skf, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(lab, p))
    # nula: etiquetas permutadas (misma tasa) -> distribucion del AUC bajo "ocultas = muestra aleatoria"
    rng = np.random.default_rng(0); null = []
    for b in range(30):
        yp = rng.permutation(lab); skf = StratifiedKFold(5, shuffle=True, random_state=b)
        p = cross_val_predict(ExtraTreesClassifier(100, min_samples_leaf=3, max_features=0.3, random_state=b, n_jobs=1), X, yp, cv=skf, method="predict_proba")[:, 1]
        null.append(roc_auc_score(yp, p))
    rows.append(dict(conjunto=name, auc_media=np.mean(aucs), auc_sd=np.std(aucs), auc_nula_media=np.mean(null), auc_nula_p95=np.percentile(null, 95)))
adv = pd.DataFrame(rows); adv.round(3).to_csv(f"{PATH['res']}/00_validacion_adversaria_ocultas.csv", index=False)
print(adv.round(3))

# ---- 6. redundancia / dimension efectiva por familia (sobre las 197, sin etiquetas)
from sklearn.decomposition import PCA
recs = []
for fam in ["S2", "LS", "PL", "SAT", "CLIM", "TOPO", "GEOM", "T_PRE", "T_VEG", "T_PICO", "T_SEN", "T_PHTIM", "T_PHLEV", "T_HET"]:
    X = StandardScaler().fit_transform(d[D.F[fam]].to_numpy(float)); p = X.shape[1]
    ev = PCA().fit(X).explained_variance_ratio_ if p > 1 else np.array([1.0])
    C = np.corrcoef(X.T) if p > 1 else np.array([[1.0]])
    iu = np.triu_indices(p, 1)
    recs.append(dict(familia=fam, p=p, pc_para_90pct=int(np.searchsorted(np.cumsum(ev), 0.90) + 1), pc_para_99pct=int(np.searchsorted(np.cumsum(ev), 0.99) + 1),
                     frac_pares_absr_gt_0_9=float((np.abs(C[iu]) > 0.9).mean()) if p > 1 else np.nan, abs_r_mediana=float(np.median(np.abs(C[iu]))) if p > 1 else np.nan))
pd.DataFrame(recs).round(3).to_csv(f"{PATH['res']}/00_redundancia_por_familia.csv", index=False)
print(pd.DataFrame(recs).round(3).to_string(index=False))
print("OK 00_audit")
