"""
02_variance_spatial.py - E1: descomposicion de varianza (entre/dentro de municipio) y autocorrelacion espacial del rendimiento.
Salidas (results/): 02_*.csv.  Las figuras se generan en 10_figures.py a partir de estos CSV.
Los residuos OOF de modelos se analizan en 02b_residual_autocorr.py (necesita la malla ya corrida).
"""
import warnings, itertools
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.spatial.distance import cdist
from common import *

warnings.filterwarnings("ignore")
D = Data(); tr = D.tr.copy(); y = D.y; mun = D.mun; n = D.n
regime = pd.read_csv(f"{PATH['cfg']}/regimen_captura_por_municipio.csv", index_col=0).regimen
tr["regimen"] = tr.loc_municipio.map(regime)
rng = np.random.default_rng(GLOBAL_SEED)
RES = PATH["res"]


# =============================== A. DESCOMPOSICION DE VARIANZA ===============================
def anova_stats(yv, g):
    g = pd.Series(g).reset_index(drop=True); yv = np.asarray(yv, float)
    codes, _ = pd.factorize(g); K = codes.max() + 1; N = len(yv)
    S = np.bincount(codes, weights=yv, minlength=K); C = np.bincount(codes, minlength=K)
    gm = yv.mean(); ssb = (C * (S / C - gm) ** 2).sum(); sst = ((yv - gm) ** 2).sum(); ssw = sst - ssb
    msb, msw = ssb / (K - 1), ssw / (N - K)
    n0 = (N - (C ** 2).sum() / N) / (K - 1)
    icc1 = (msb - msw) / (msb + (n0 - 1) * msw)
    omega2 = (ssb - (K - 1) * msw) / (sst + msw)
    return dict(K=K, N=N, SS_total=sst, SS_between=ssb, SS_within=ssw, eta2=ssb / sst, omega2=omega2, ICC1_anova=icc1,
                sd_total=np.sqrt(sst / (N - 1)), sd_within_pooled=np.sqrt(msw))


def reml_icc(yv, g, nboot=300):
    df = pd.DataFrame(dict(y=yv, g=pd.Series(g).astype(str).values))
    m = smf.mixedlm("y ~ 1", df, groups=df["g"]).fit(reml=True)
    va, ve, mu = float(m.cov_re.iloc[0, 0]), float(m.scale), float(m.fe_params.iloc[0])
    # bootstrap parametrico (mantiene diseño desbalanceado)
    codes, _ = pd.factorize(df.g); K = codes.max() + 1; iccs = []
    for b in range(nboot):
        a = rng.normal(0, np.sqrt(max(va, 1e-12)), K)
        ys = mu + a[codes] + rng.normal(0, np.sqrt(ve), len(df))
        try:
            mb = smf.mixedlm("y ~ 1", pd.DataFrame(dict(y=ys, g=df.g)), groups=df.g).fit(reml=True)
            vb = float(mb.cov_re.iloc[0, 0]); iccs.append(vb / (vb + float(mb.scale)))
        except Exception:
            pass
    return dict(var_between_REML=va, var_within_REML=ve, ICC_REML=va / (va + ve), ICC_REML_p2_5=np.percentile(iccs, 2.5), ICC_REML_p97_5=np.percentile(iccs, 97.5))


def strat_boot_eta2(yv, g, nb=2000):
    g = np.asarray(g); idx_by = [np.where(g == k)[0] for k in np.unique(g)]; out = []
    for _ in range(nb):
        ii = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by])
        out.append(anova_stats(np.asarray(yv)[ii], g[ii])["eta2"])
    return np.percentile(out, [2.5, 97.5])


rows = []
subsets = {
    "A. 9 municipios (n=138)": np.ones(n, bool),
    "B. municipios con n>=10 (Chignahuapan, Calpulalpan, Singuilucan, Almoloya)": pd.Series(mun).isin(["Chignahuapan", "Calpulalpan", "Singuilucan", "Almoloya"]).values,
    "C. solo captura FINA (Chignahuapan, Almoloya, Cuautepec, Tepeapulco)": (tr.regimen == "fina(continua)").values,
    "D. solo captura GRUESA (Calpulalpan, Singuilucan, E.Zapata, Apan, Nanacamilpa)": (tr.regimen == "gruesa(0.5 t/ha)").values,
    "E. solo Chignahuapan+Calpulalpan+Singuilucan (3 grandes)": pd.Series(mun).isin(["Chignahuapan", "Calpulalpan", "Singuilucan"]).values,
}
for name, msk in subsets.items():
    s = anova_stats(y[msk], mun[msk])
    try:
        s.update(reml_icc(y[msk], mun[msk], nboot=200))
    except Exception as e:
        s["reml_error"] = str(e)[:60]
    lo, hi = strat_boot_eta2(y[msk], mun[msk], 1000)
    s.update(eta2_boot_p2_5=lo, eta2_boot_p97_5=hi, subconjunto=name)
    rows.append(s)
# regimen de captura como factor (2 niveles) y municipio anidado en regimen
s = anova_stats(y, tr.regimen.values); lo, hi = strat_boot_eta2(y, tr.regimen.values, 1000)
s.update(subconjunto="F. REGIMEN de captura (fina/gruesa) como unico factor", eta2_boot_p2_5=lo, eta2_boot_p97_5=hi); rows.append(s)
V = pd.DataFrame(rows).set_index("subconjunto"); V.round(4).to_csv(f"{RES}/02_varianza_entre_dentro_municipio.csv")
print(V[["K", "N", "eta2", "eta2_boot_p2_5", "eta2_boot_p97_5", "omega2", "ICC1_anova", "ICC_REML", "ICC_REML_p2_5", "ICC_REML_p97_5", "sd_within_pooled"]].round(3).to_string())

# Municipio dentro de regimen: SS secuencial
tr["y"] = y
m0 = smf.ols("y ~ C(regimen)", tr).fit(); m1 = smf.ols("y ~ C(regimen) + C(loc_municipio)", tr).fit()
seq = anova_lm(m0, m1); sst = ((y - y.mean()) ** 2).sum()
nest = dict(R2_regimen=m0.rsquared, R2_regimen_mas_municipio=m1.rsquared, R2_incremental_municipio_dado_regimen=m1.rsquared - m0.rsquared,
            R2adj_regimen=m0.rsquared_adj, R2adj_regimen_mas_municipio=m1.rsquared_adj, F_p_municipio_dado_regimen=float(seq["Pr(>F)"].iloc[1]))
pd.Series(nest).round(4).to_csv(f"{RES}/02_regimen_vs_municipio_secuencial.csv", header=["valor"]); print(pd.Series(nest).round(4))

# Aporte de cada municipio a la SS dentro
w = tr.groupby("loc_municipio").y.agg(n="size", media="mean", sd="std")
w["SS_dentro"] = tr.groupby("loc_municipio").y.apply(lambda v: ((v - v.mean()) ** 2).sum())
w["pct_de_SS_dentro_total"] = 100 * w.SS_dentro / w.SS_dentro.sum()
w["pct_de_SS_total"] = 100 * w.SS_dentro / sst
w["var_cuantizacion_Sheppard_0.5^2/12"] = np.where(tr.groupby("loc_municipio").apply(lambda g: np.isclose(g.y * 2, np.round(g.y * 2)).mean()) >= 0.5, 0.5 ** 2 / 12, 0.0)
w.round(4).sort_values("n", ascending=False).to_csv(f"{RES}/02_dentro_por_municipio.csv"); print(w.round(3).sort_values("n", ascending=False))

# Celda de clima anidada en municipio (SS secuencial + permutacion intra-municipio del incremento)
tr["mc"] = tr.loc_municipio + "|" + tr.loc_celda_clima
mm = smf.ols("y ~ C(loc_municipio)", tr).fit(); mc = smf.ols("y ~ C(loc_municipio) + C(mc)", tr).fit()
inc = mc.rsquared - mm.rsquared; perm = []
for b in range(1000):
    t2 = tr.copy(); t2["mc"] = t2.groupby("loc_municipio")["mc"].transform(lambda s: rng.permutation(s.values))
    perm.append(smf.ols("y ~ C(loc_municipio) + C(mc)", t2).fit().rsquared - mm.rsquared)
cell = dict(n_combinaciones_municipio_celda=int(tr.mc.nunique()), R2_municipio=mm.rsquared, R2adj_municipio=mm.rsquared_adj,
            R2_municipio_mas_celda=mc.rsquared, R2adj_municipio_mas_celda=mc.rsquared_adj, R2_incremental_celda=inc,
            p_perm_incremental_celda_intra_municipio=(1 + (np.array(perm) >= inc).sum()) / (1 + len(perm)), R2_incremental_esperado_bajo_H0=float(np.mean(perm)))
pd.Series(cell).round(4).to_csv(f"{RES}/02_celda_clima_dentro_municipio.csv", header=["valor"]); print(pd.Series(cell).round(4))

# =============================== B. VARIANZA DE LAS FEATURES ===============================
fams = ["S2", "LS", "PL", "SAT", "CLIM", "TOPO", "GEOM", "LOC", "T_PRE", "T_VEG", "T_PICO", "T_SEN", "T_PHTIM", "T_PHLEV", "T_HET"]
fe = []
for fam in fams:
    for c in D.F[fam]:
        s = anova_stats(tr[c].values, mun); s2 = anova_stats(tr[c].values, tr.regimen.values)
        fe.append(dict(familia=fam, feature=c, eta2_municipio=s["eta2"], omega2_municipio=s["omega2"], omega2_regimen=s2["omega2"]))
FE = pd.DataFrame(fe); FE.to_csv(f"{RES}/02_eta2_por_feature.csv", index=False)
fs = FE.groupby("familia").agg(p=("feature", "size"), omega2_mun_mediana=("omega2_municipio", "median"), omega2_mun_p25=("omega2_municipio", lambda v: v.quantile(.25)),
                               omega2_mun_p75=("omega2_municipio", lambda v: v.quantile(.75)), frac_omega2_mun_gt_0_5=("omega2_municipio", lambda v: (v > .5).mean()),
                               omega2_regimen_mediana=("omega2_regimen", "median")).reindex(fams)
fs.round(3).to_csv(f"{RES}/02_varianza_features_por_familia.csv"); print(fs.round(3))

# ¿Cuanto identifica cada familia el municipio / el regimen? (CV estratificada 5-fold x3, ET, SOLO n>=10 municipios; sin usar rendimiento)
big = pd.Series(mun).isin(["Chignahuapan", "Calpulalpan", "Singuilucan", "Almoloya"]).values
lab = pd.factorize(tr.loc_municipio[big])[0]; reg = (tr.regimen[big] == "gruesa(0.5 t/ha)").astype(int).values
ident = []
for fam in ["S2", "LS", "PL", "SAT", "CLIM", "TOPO", "GEOM", "LOC", "T_PRE", "T_VEG", "T_PICO", "T_SEN", "T_PHTIM", "T_PHLEV", "T_HET", "CUR"]:
    X = tr.loc[big, D.F[fam]].to_numpy(float); acc = []; auc = []
    for r in range(3):
        skf = StratifiedKFold(5, shuffle=True, random_state=r)
        P = cross_val_predict(ExtraTreesClassifier(300, min_samples_leaf=2, max_features=0.5, random_state=r, n_jobs=1), X, lab, cv=skf, method="predict_proba")
        acc.append(accuracy_score(lab, P.argmax(1)))
        pr = cross_val_predict(ExtraTreesClassifier(300, min_samples_leaf=2, max_features=0.5, random_state=r, n_jobs=1), X, reg, cv=skf, method="predict_proba")[:, 1]
        auc.append(roc_auc_score(reg, pr))
    ident.append(dict(familia=fam, p=X.shape[1], acc_municipio_4clases=np.mean(acc), acc_mayoritaria=np.bincount(lab).max() / len(lab), auc_regimen_captura=np.mean(auc)))
pd.DataFrame(ident).round(3).to_csv(f"{RES}/02_identificabilidad_municipio_por_familia.csv", index=False); print(pd.DataFrame(ident).round(3).to_string(index=False))


# =============================== C. AUTOCORRELACION ESPACIAL ===============================
def W_knn(XY, k, mask=None):
    m = len(XY); W = np.zeros((m, m)); Dm = cdist(XY, XY); np.fill_diagonal(Dm, np.inf)
    if mask is not None:
        Dm = np.where(mask, Dm, np.inf)
    for i in range(m):
        o = np.argsort(Dm[i])[:k]; o = o[np.isfinite(Dm[i, o])]
        if len(o):
            W[i, o] = 1.0 / len(o)
    return W


def W_band(XY, lo, hi, mask=None):
    Dm = cdist(XY, XY); A = (Dm > lo) & (Dm <= hi) if lo > 0 else (Dm <= hi) & (Dm > 0)
    if mask is not None:
        A &= mask
    W = A.astype(float); rs = W.sum(1); W[rs > 0] /= rs[rs > 0][:, None]
    return W


def moran(z, W):
    z = z - z.mean(); S0 = W.sum()
    return (len(z) / S0) * (z @ W @ z) / (z @ z)


def moran_perm(z, W, groups=None, nperm=9999, seed=0):
    """I observada y p por permutacion. Si groups: z se centra por grupo y se permuta DENTRO de grupo (W debe ser intra-grupo)."""
    r = np.random.default_rng(seed); z = np.asarray(z, float).copy()
    if groups is not None:
        z = z - pd.Series(z).groupby(np.asarray(groups)).transform("mean").values
    else:
        z = z - z.mean()
    S0 = W.sum(); den = z @ z
    I = (len(z) / S0) * (z @ W @ z) / den
    Z = np.empty((nperm, len(z)))
    if groups is None:
        for b in range(nperm):
            Z[b] = z[r.permutation(len(z))]
    else:
        gi = [np.where(np.asarray(groups) == u)[0] for u in np.unique(groups)]
        for b in range(nperm):
            zz = z.copy()
            for ix in gi:
                zz[ix] = z[r.permutation(ix)]
            Z[b] = zz
    num = np.einsum("pi,ij,pj->p", Z, W, Z); Ip = (len(z) / S0) * num / den
    return dict(I=I, E_I_perm=Ip.mean(), sd_perm=Ip.std(), z=(I - Ip.mean()) / Ip.std(), p_one_sided=(1 + (Ip >= I).sum()) / (1 + nperm),
                p_two_sided=(1 + (np.abs(Ip - Ip.mean()) >= abs(I - Ip.mean())).sum()) / (1 + nperm))


def moran_table(label, z, XY, groups=None, nperm=9999):
    out = []
    mask = None if groups is None else (np.asarray(groups)[:, None] == np.asarray(groups)[None, :])
    specs = [("kNN5", lambda: W_knn(XY, 5, mask)), ("kNN8", lambda: W_knn(XY, 8, mask)), ("banda<=1km", lambda: W_band(XY, 0, 1000, mask)),
             ("banda<=2km", lambda: W_band(XY, 0, 2000, mask)), ("banda<=5km", lambda: W_band(XY, 0, 5000, mask))]
    for wn, mk in specs:
        W = mk(); has = W.sum(1) > 0
        if has.sum() < 8:
            continue
        # filas sin vecinos se quitan del estadistico
        Wr = W[np.ix_(has, has)]; Wr = Wr / np.where(Wr.sum(1, keepdims=True) == 0, 1, Wr.sum(1, keepdims=True))
        gg = None if groups is None else np.asarray(groups)[has]
        r_ = moran_perm(np.asarray(z)[has], Wr, gg, nperm=nperm, seed=1)
        out.append(dict(serie=label, pesos=wn, n_usadas=int(has.sum()), **r_))
    return out


XY = D.XY; ms = []
ms += moran_table("rendimiento (138, todos los municipios; incluye variacion ENTRE municipios)", y, XY)
ms += moran_table("rendimiento centrado por municipio (138; pesos y permutacion INTRA-municipio)", y, XY, groups=mun)
for m_ in ["Chignahuapan", "Calpulalpan", "Singuilucan"]:
    k = mun == m_
    ms += moran_table(f"rendimiento dentro de {m_} (n={k.sum()})", y[k], XY[k])
MT = pd.DataFrame(ms); MT.round(5).to_csv(f"{RES}/02_moran_rendimiento.csv", index=False)
print(MT[["serie", "pesos", "n_usadas", "I", "E_I_perm", "z", "p_one_sided", "p_two_sided"]].round(4).to_string(index=False))


# Correlograma de Moran dentro de Chignahuapan y dentro-municipio pooled (anillos de distancia)
def correlogram(z, XY, groups, edges, nperm=4999):
    mask = None if groups is None else (np.asarray(groups)[:, None] == np.asarray(groups)[None, :])
    rows_ = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        W = W_band(XY, lo, hi, mask); has = W.sum(1) > 0
        if has.sum() < 8:
            rows_.append(dict(d_lo_km=lo / 1000, d_hi_km=hi / 1000, n_usadas=int(has.sum()), I=np.nan, p_one_sided=np.nan)); continue
        Wr = W[np.ix_(has, has)]; Wr = Wr / Wr.sum(1, keepdims=True)
        gg = None if groups is None else np.asarray(groups)[has]
        r_ = moran_perm(np.asarray(z)[has], Wr, gg, nperm=nperm, seed=2)
        rows_.append(dict(d_lo_km=lo / 1000, d_hi_km=hi / 1000, n_usadas=int(has.sum()), I=r_["I"], z=r_["z"], p_one_sided=r_["p_one_sided"]))
    return pd.DataFrame(rows_)


edges = [0, 500, 1000, 2000, 4000, 8000]
cg = pd.concat([correlogram(y[mun == "Chignahuapan"], XY[mun == "Chignahuapan"], None, edges).assign(serie="Chignahuapan"),
                correlogram(y, XY, mun, edges).assign(serie="intra-municipio (pooled, centrado)")])
cg.round(4).to_csv(f"{RES}/02_correlograma_moran.csv", index=False); print(cg.round(3).to_string(index=False))


# Semivariograma con envolvente de permutacion (intra-municipio)
def variogram(z, XY, mask, edges):
    Dm = cdist(XY, XY); G = 0.5 * (z[:, None] - z[None, :]) ** 2; iu = np.triu_indices(len(z), 1)
    d_, g_, ok = Dm[iu], G[iu], mask[iu]; res = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = ok & (d_ > lo) & (d_ <= hi); res.append((s.sum(), g_[s].mean() if s.sum() else np.nan))
    return res


def variogram_env(label, z, XY, groups, edges, nperm=999):
    groups = np.asarray(groups); mask = groups[:, None] == groups[None, :]
    z0 = z - pd.Series(z).groupby(groups).transform("mean").values
    obs = variogram(z0, XY, mask, edges); gi = [np.where(groups == u)[0] for u in np.unique(groups)]; r = np.random.default_rng(3); sims = []
    for b in range(nperm):
        zz = z0.copy()
        for ix in gi:
            zz[ix] = z0[r.permutation(ix)]
        sims.append([v[1] for v in variogram(zz, XY, mask, edges)])
    sims = np.array(sims)
    out = pd.DataFrame(dict(d_lo_km=np.array(edges[:-1]) / 1000, d_hi_km=np.array(edges[1:]) / 1000, n_pares=[v[0] for v in obs], semivarianza=[v[1] for v in obs],
                            env_p2_5=np.nanpercentile(sims, 2.5, 0), env_p50=np.nanpercentile(sims, 50, 0), env_p97_5=np.nanpercentile(sims, 97.5, 0)))
    out["serie"] = label; return out


ve = [[0, 250, 500, 1000, 2000, 4000, 8000]]
vg = pd.concat([variogram_env("Chignahuapan (n=71)", y[mun == "Chignahuapan"], XY[mun == "Chignahuapan"], np.zeros((mun == "Chignahuapan").sum(), int), ve[0]),
                variogram_env("intra-municipio pooled (n=138)", y, XY, mun, ve[0])])
vg.round(4).to_csv(f"{RES}/02_semivariograma.csv", index=False); print(vg.round(3).to_string(index=False))

# Potencia de Moran en Chignahuapan (n=71): campo gaussiano exponencial + ruido, coordenadas reales
k = mun == "Chignahuapan"; xy = XY[k]; Dm = cdist(xy, xy); W = W_knn(xy, 5); pw = []
r = np.random.default_rng(11)
for share in (0.2, 0.4, 0.6, 0.8):
    for rng_km in (0.5, 1, 2, 4):
        cov = share * np.exp(-3 * Dm / (rng_km * 1000)) + (1 - share) * np.eye(len(xy))   # rango practico = rng_km (95% de la meseta)
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(xy))); hits = 0; nsim = 300
        for s_ in range(nsim):
            zs = L @ r.normal(size=len(xy))
            res_ = moran_perm(zs, W, None, nperm=199, seed=int(r.integers(1e9)))
            hits += res_["p_one_sided"] < 0.05
        pw.append(dict(proporcion_varianza_espacial=share, rango_practico_km=rng_km, potencia=hits / nsim))
PW = pd.DataFrame(pw); PW.round(3).to_csv(f"{RES}/02_potencia_moran_chignahuapan.csv", index=False); print(PW.pivot(index="proporcion_varianza_espacial", columns="rango_practico_km", values="potencia").round(2))
print("OK 02_variance_spatial")
