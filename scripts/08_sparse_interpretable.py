"""
08_sparse_interpretable.py - Pregunta de curiosidad: con 5-10 variables interpretables (sin redes complejas), ¿cuales y con que evidencia?
  A. evidencia univariada INTRA-municipio (correlacion parcial de rango con el residuo de municipio) con permutacion intra-municipio,
     FWER por estadistico maximo y FDR (BH); consistencia de signo entre municipios.
  B. seleccion progresiva (forward) ANIDADA: la seleccion ocurre solo con el entrenamiento de cada fold externo (CV interna 5-fold).
  C. conjunto pre-especificado a priori (sin mirar etiquetas) vs conjunto anidado vs prior municipal solo vs CUR24.
"""
import time
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import KFold
from oofio import *

D = Data(); RES = PATH["res"]; rng = np.random.default_rng(GLOBAL_SEED)
SPL = load_splits(D, ["RCV5x10", "SPA_GRID3B05", "SPA_KM6"])
ALPHA = 10.0   # Ridge fija (variables estandarizadas) para la seleccion; la evaluacion final tambien usa esta misma Ridge

CAND = ["s2_ndvi_peak", "s2_ndvi_auc", "s2_ndvi_los", "s2_ndvi_sos_doy", "s2_ndvi_eos_doy", "s2_ndvi_amp", "s2_ndvi_base", "s2_ndvi_pre", "s2_ndvi_veg", "s2_ndvi_pico", "s2_ndvi_sen",
        "s2_evi_pico", "s2_lai_pico", "s2_fapar_pico", "s2_msi_pico", "s2_ndwi_pico", "s2_crc_pre", "s2_ndti_pre", "s2_ndvi_std_pico",
        "pl_ndvi_peak", "pl_ndvi_auc", "pl_evi_pico", "pl_lai_pico", "ls_vi6t_pico",
        "clim_prec_temp", "clim_prec_prev", "clim_tmin_minmes", "clim_tmean_temp", "clim_gdd0_temp", "clim_prec_cv", "clim_tmax_anom", "clim_prec_anom_pct",
        "topo_elev_media", "topo_pend_media", "topo_elev_std", "geo_area_ha", "geo_compacidad"]
CAND = [c for c in CAND if c in D.col_index]
cidx = np.array([D.col_index[c] for c in CAND])
# conjunto PRE-ESPECIFICADO (criterio agronomico a priori; definido sin consultar el rendimiento)
APRIORI = ["s2_ndvi_peak", "s2_ndvi_auc", "s2_lai_pico", "s2_msi_pico", "s2_crc_pre", "clim_prec_temp", "clim_tmin_minmes", "topo_elev_media", "topo_pend_media", "geo_area_ha"]
APRIORI_IDX = np.array([D.col_index[c] for c in APRIORI])


# ---------- ridge cerrada rapida (variables estandarizadas, intercepto libre) ----------
def ridge_fit(Xtr, t, alpha=ALPHA):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-12] = 1.0
    Z = (Xtr - mu) / sd; tm = t.mean(); w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (t - tm))
    return mu, sd, tm, w


def ridge_pred(model, X):
    mu, sd, tm, w = model; return tm + ((X - mu) / sd) @ w


def forward_select(X, t, kmax, inner_seed):
    """Seleccion hacia adelante con CV interna 5-fold (RMSE). Devuelve lista ordenada de columnas y trayectoria de RMSE interno."""
    n_, p = X.shape; chosen, traj = [], []
    folds = list(KFold(5, shuffle=True, random_state=inner_seed).split(np.arange(n_)))
    for step in range(kmax):
        best, bj = np.inf, None
        for j in range(p):
            if j in chosen:
                continue
            cols = chosen + [j]; se = 0.0
            for a, b in folds:
                m = ridge_fit(X[a][:, cols], t[a]); se += ((t[b] - ridge_pred(m, X[b][:, cols])) ** 2).sum()
            if se < best:
                best, bj = se, j
        chosen.append(bj); traj.append(np.sqrt(best / n_))
    base = np.sqrt(np.mean([((t[b] - t[a].mean()) ** 2).mean() for a, b in folds]))
    return chosen, traj, base


# ===================== A. evidencia univariada intra-municipio =====================
big = pd.Series(D.mun).isin(["Chignahuapan", "Calpulalpan", "Singuilucan", "Almoloya"]).values
def within_rank(v, groups):
    r = np.empty(len(v))
    for g in np.unique(groups):
        k = groups == g; r[k] = rankdata(v[k]) - (k.sum() + 1) / 2
    return r

gm = D.mun[big]; yr = within_rank(D.y[big], gm)
def pr_corr(x, yr_):
    xr = within_rank(x, gm); return (xr * yr_).sum() / np.sqrt((xr ** 2).sum() * (yr_ ** 2).sum())

obs = np.array([pr_corr(D.X[big][:, j], yr) for j in cidx])
gi = [np.where(gm == u)[0] for u in np.unique(gm)]
NP = 5000; null = np.zeros((NP, len(cidx)))
Xr = np.column_stack([within_rank(D.X[big][:, j], gm) for j in cidx])      # rangos intra-municipio de X
for b in range(NP):
    idx = np.arange(len(gm))
    for ix in gi:
        idx[ix] = rng.permutation(ix)
    yp = yr[idx]
    null[b] = (Xr * yp[:, None]).sum(0) / np.sqrt((Xr ** 2).sum(0) * (yp ** 2).sum())
p_two = (1 + (np.abs(null) >= np.abs(obs)).sum(0)) / (1 + NP)
maxnull = np.abs(null).max(1); p_fwer = np.array([(1 + (maxnull >= abs(o)).sum()) / (1 + NP) for o in obs])
# BH
o = np.argsort(p_two); q = np.empty(len(p_two)); prev = 1.0
for r_, i in zip(range(len(o), 0, -1), o[::-1]):
    prev = min(prev, p_two[i] * len(o) / r_); q[i] = prev
# consistencia de signo por municipio (Spearman intra-municipio)
sg = {}
for m_ in ["Chignahuapan", "Calpulalpan", "Singuilucan", "Almoloya"]:
    k = D.mun == m_; sg[m_] = [spearmanr(D.X[k][:, j], D.y[k])[0] for j in cidx]
A = pd.DataFrame(dict(variable=CAND, rho_parcial_intra=obs, p_perm_dos_colas=p_two, p_FWER_max=p_fwer, q_FDR_BH=q,
                      **{f"rho_{m_}": v for m_, v in sg.items()}))
A["signos_concordantes_4_municipios"] = (np.sign(A[[f"rho_{m_}" for m_ in sg]].values) == np.sign(A.rho_parcial_intra.values)[:, None]).sum(1)
A["grupo"] = [("S2" if c.startswith("s2") else "PL" if c.startswith("pl") else "LS" if c.startswith("ls") else "CLIM" if c.startswith("clim") else "TOPO" if c.startswith("topo") else "GEOM") for c in CAND]
A = A.sort_values("p_perm_dos_colas"); A.round(4).to_csv(f"{RES}/08_evidencia_univariada_intra_municipio.csv", index=False)
print(A.head(12).round(3).to_string(index=False)); print("max|rho| nulo p95:", round(np.percentile(maxnull, 95), 3), "| n con q<0.10:", int((A.q_FDR_BH < 0.10).sum()), "| n con FWER<0.05:", int((A.p_FWER_max < 0.05).sum()))


# ===================== B/C. seleccion anidada + conjunto a priori =====================
def run_scheme(scheme, nrep, kmax=10, label=""):
    res = {k: [] for k in ("prior", "nested_k5", "nested_k10", "apriori10", "apriori5", "nested_kopt")}; sel_log = []
    for rep in range(nrep):
        oofs = {k: np.zeros(D.n) for k in res}
        for fold, tr, te in SPL[scheme][rep]:
            p_tr, p_te = prior_muni(D, tr, te); t = D.y[tr] - p_tr
            oofs["prior"][te] = p_te
            Xc_tr, Xc_te = D.X[tr][:, cidx], D.X[te][:, cidx]
            chosen, traj, base = forward_select(Xc_tr, t, kmax, inner_seed=500 + 10 * rep + fold)
            sel_log.append(dict(scheme=scheme, rep=rep, fold=fold, order=";".join(CAND[j] for j in chosen), traj=";".join(f"{v:.4f}" for v in traj), base_inner=base))
            for k_, key in ((5, "nested_k5"), (10, "nested_k10")):
                m = ridge_fit(Xc_tr[:, chosen[:k_]], t); oofs[key][te] = p_te + ridge_pred(m, Xc_te[:, chosen[:k_]])
            kopt = int(np.argmin(traj)) + 1 if min(traj) < base else 0           # k optimo por CV interna (0 => solo prior)
            if kopt == 0:
                oofs["nested_kopt"][te] = p_te
            else:
                m = ridge_fit(Xc_tr[:, chosen[:kopt]], t); oofs["nested_kopt"][te] = p_te + ridge_pred(m, Xc_te[:, chosen[:kopt]])
            for key, idx_ in (("apriori10", APRIORI_IDX), ("apriori5", APRIORI_IDX[:5])):
                m = ridge_fit(D.X[tr][:, idx_], t); oofs[key][te] = p_te + ridge_pred(m, D.X[te][:, idx_])
        for k in res:
            res[k].append(oofs[k])
    return ({k: np.mean([(D.y - o) ** 2 for o in v], 0) for k, v in res.items()},     # perdida por parcela = media de errores^2 sobre repeticiones (comparable con 04)
            {k: [rmse(D.y, o) for o in v] for k, v in res.items()}, pd.DataFrame(sel_log))


B = Boot(D, B=3000); rows = []; allsel = []
for scheme, nrep in (("RCV5x10", 10), ("SPA_GRID3B05", 3), ("SPA_KM6", 1)):
    t0 = time.time(); oofm, rms, sel = run_scheme(scheme, nrep); allsel.append(sel)
    for k in oofm:        # oofm[k] = perdida por parcela (media sobre repeticiones)
        pc = B.paired(oofm[k], oofm["prior"]) if k != "prior" else None
        rows.append(dict(esquema=scheme, modelo=k, RMSE_media_reps=np.mean(rms[k]), RMSE_sd_reps=np.std(rms[k]), RMSE_perdida_media_parcelas=float(np.sqrt(oofm[k].mean())),
                         **({} if pc is None else dict(dRMSE_vs_prior=pc["dRMSE"], ic_lo_cluster=pc["ci_lo_clu"], ic_hi_cluster=pc["ci_hi_clu"], p_cluster=pc["p_clu"]))))
    print(f"{time.strftime('%H:%M:%S')} {scheme} listo ({time.time()-t0:.0f}s)", flush=True)
R = pd.DataFrame(rows); R.round(4).to_csv(f"{RES}/08_seleccion_anidada_vs_apriori.csv", index=False); print(R.round(4).to_string(index=False))
SEL = pd.concat(allsel); SEL.to_csv(f"{RES}/08_trayectorias_seleccion_por_fold.csv", index=False)

# frecuencia de seleccion (primeros 5 y 10 de cada fold) bajo CV aleatoria
fq = []
for k_ in (5, 10):
    cnt = pd.Series([v for o in SEL[SEL.scheme == "RCV5x10"].order for v in o.split(";")[:k_]]).value_counts() / (SEL.scheme == "RCV5x10").sum()
    fq.append(cnt.rename(f"freq_top{k_}_RCV"))
cnt2 = pd.Series([v for o in SEL[SEL.scheme != "RCV5x10"].order for v in o.split(";")[:10]]).value_counts() / (SEL.scheme != "RCV5x10").sum()
FQ = pd.concat(fq + [cnt2.rename("freq_top10_espaciales")], axis=1).fillna(0).sort_values("freq_top10_RCV", ascending=False)
FQ.round(3).to_csv(f"{RES}/08_frecuencia_seleccion.csv"); print(FQ.head(15).round(3))
print("OK 08_sparse_interpretable")
