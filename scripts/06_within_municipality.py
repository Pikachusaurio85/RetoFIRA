"""
06_within_municipality.py - E4: ¿los modelos aprenden variacion DENTRO de municipio o solo diferencias geograficas?
  A. descomposicion entre/dentro de las predicciones OOF + correlacion intra-municipal + skill residual sobre prior municipal
  B. prueba de permutacion INTRA-municipio de las features con TODO el pipeline (prior + residuo) bajo la misma CV
  C. prueba limpia dentro de Chignahuapan (n=71, captura continua) y de Calpulalpan / Singuilucan (captura gruesa)
  D. estabilidad del aporte residual: por repeticion, por esquema y por municipio
  E. sensibilidad del encogimiento del prior municipal
"""
import time
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from oofio import *

D = Data(); RES = PATH["res"]; rng = np.random.default_rng(GLOBAL_SEED)
O = load_oof("main"); L, P = per_parcel(O, D); B = Boot(D, B=3000)
regime = pd.read_csv(f"{PATH['cfg']}/regimen_captura_por_municipio.csv", index_col=0).regimen
reg = D.tr.loc_municipio.map(regime).values
big = {m: (D.mun == m) for m in ["Chignahuapan", "Calpulalpan", "Singuilucan", "Almoloya"]}

# ---------------- A. entre/dentro en predicciones OOF ----------------
def centered(v, groups):
    return v - pd.Series(v).groupby(groups).transform("mean").values


keys = ["ref_global", "ref_muni", "ref_knn", "ref_muni_knnres", "direct|Ridge|LOC", "direct|ET|LOC", "direct|Ridge|CLIM", "direct|ET|CLIM", "direct|Ridge|SAT", "direct|ET|SAT", "direct|ET|B1",
        "direct|Ridge|ALLNOLOC", "direct|ET|ALLNOLOC", "direct|ET|ALL", "direct|CB|SAT", "direct|CB|ALL", "resid_muni|Ridge|SAT", "resid_muni|ET|SAT", "resid_muni|Ridge|CUR", "resid_muni|ET|CUR",
        "resid_muni|Ridge|ALLNOLOC", "resid_muni|ET|ALLNOLOC", "resid_muni|ET|ALL", "resid_muni|CB|ALL", "resid_muni|CB|SAT", "resid_muni|Ridge|CLIM", "resid_muni|ET|CLIM", "resid_muni|Ridge|TOPO"]
keys = [k for k in keys if ("RCV5x10" in {s for (c, s) in L if c == k})]
mask_big = np.isin(D.mun, list(big))
yc_big = centered(D.y, D.mun)
ra = []
for sch in ("RCV5x10", "SPA_GRID3B05", "SPA_GRID8B1", "SPA_KM6"):
    for c in keys:
        if (c, sch) not in P:
            continue
        p = P[(c, sch)]; pc = centered(p, D.mun)
        res_prior = D.y - P[("ref_muni", sch)]; pred_res = p - P[("ref_muni", sch)]
        row = dict(esquema=sch, cfg=c, RMSE=np.sqrt(L[(c, sch)].mean()),
                   eta2_predicciones_por_municipio=float(((pd.Series(p).groupby(D.mun).transform("mean") - p.mean()) ** 2).sum() / ((p - p.mean()) ** 2).sum()) if p.std() > 1e-9 else np.nan,
                   r_global=pearsonr(D.y, p)[0] if p.std() > 1e-9 else np.nan)
        # intra-municipal (municipios n>=10): correlacion y R2 de la parte centrada
        a, b = yc_big[mask_big], pc[mask_big]
        row.update(r_intra_pooled=pearsonr(a, b)[0] if b.std() > 1e-9 else np.nan, rho_intra_pooled=spearmanr(a, b)[0] if b.std() > 1e-9 else np.nan,
                   R2_intra_pooled=1 - ((a - b) ** 2).sum() / (a ** 2).sum(), pendiente_intra=np.polyfit(b, a, 1)[0] if b.std() > 1e-9 else np.nan)
        for m, k in big.items():
            bb = centered(p, D.mun)[k]
            row[f"r_intra_{m}"] = pearsonr(D.y[k], p[k])[0] if p[k].std() > 1e-9 else np.nan
        # skill residual sobre prior municipal OOF
        if c != "ref_muni" and pred_res.std() > 1e-9:
            row.update(r_residuo=pearsonr(res_prior, pred_res)[0], rho_residuo=spearmanr(res_prior, pred_res)[0])
        ra.append(row)
RA = pd.DataFrame(ra); RA.round(4).to_csv(f"{RES}/06_entre_vs_dentro_predicciones.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(RA[RA.esquema == "RCV5x10"][["cfg", "RMSE", "eta2_predicciones_por_municipio", "r_global", "r_intra_pooled", "R2_intra_pooled", "r_intra_Chignahuapan", "r_residuo"]].round(3).to_string(index=False))


# ---------------- B. permutacion intra-municipio de X con todo el pipeline ----------------
SPL = load_splits(D, ["RCV5x10", "SPA_GRID3B05"])


def perm_within(rng_):
    idx = np.arange(D.n)
    for m in range(D.K):
        ii = np.where(D.mun_code == m)[0]
        idx[ii] = rng_.permutation(ii)
    return idx


def cv_resid_rmse(X, model, sch, reps, cols_idx):
    out = []
    for rep in range(reps):
        oof = np.zeros(D.n)
        for fold, tr, te in SPL[sch][rep]:
            p_tr, p_te = prior_muni(D, tr, te)
            if model == "Ridge":
                m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 40)))
            else:
                m = ExtraTreesRegressor(100, min_samples_leaf=3, max_features=0.5, n_jobs=1, random_state=1000 * rep + fold)
            m.fit(X[tr][:, cols_idx], D.y[tr] - p_tr); oof[te] = p_te + np.ravel(m.predict(X[te][:, cols_idx]))
        out.append(rmse(D.y, oof))
    return np.mean(out)


prior_rm = {}
for sch, reps in (("RCV5x10", 3), ("SPA_GRID3B05", 3)):
    pr = []
    for rep in range(reps):
        oof = np.zeros(D.n)
        for fold, tr, te in SPL[sch][rep]:
            oof[te] = prior_muni(D, tr, te)[1]
        pr.append(rmse(D.y, oof))
    prior_rm[sch] = np.mean(pr)

pt = []
t0 = time.time()
plan = [("Ridge", s, 2, 200) for s in ["SAT", "S2", "PL", "CLIM", "TOPO", "GEOM", "CUR", "ALLNOLOC", "T_PRE", "T_VEG", "T_PICO", "T_SEN", "T_PHTIM", "T_PHLEV", "T_HET"]] + \
       [("ET", s, 2, 100) for s in ["SAT", "CLIM", "ALLNOLOC", "CUR"]]
for model, fs, reps, nperm in plan:
    ci = D.idx(fs)
    for sch in ("RCV5x10",):
        obs = cv_resid_rmse(D.X, model, sch, reps, ci) - prior_rm[sch]
        null = []
        for b in range(nperm):
            Xp = D.X[perm_within(rng)]
            null.append(cv_resid_rmse(Xp, model, sch, reps, ci) - prior_rm[sch])
        null = np.array(null)
        pt.append(dict(modelo=model, conjunto=fs, esquema=sch, reps_cv=reps, n_perm=nperm, dRMSE_observado_vs_prior=obs, null_media=null.mean(), null_p5=np.percentile(null, 5), null_p95=np.percentile(null, 95),
                       p_perm_une_cola=(1 + (null <= obs).sum()) / (1 + nperm), frac_null_mejor_que_prior=(null < 0).mean()))
        print(f"{time.strftime('%H:%M:%S')} {model} {fs:9s} d={obs:+.4f} null={null.mean():+.4f} [{np.percentile(null,5):+.4f},{np.percentile(null,95):+.4f}] p={pt[-1]['p_perm_une_cola']:.3f}", flush=True)
PT = pd.DataFrame(pt); PT["p_holm"] = holm(PT.p_perm_une_cola.values); PT.round(4).to_csv(f"{RES}/06_permutacion_intra_municipio.csv", index=False)


# ---------------- C. prueba limpia dentro de un municipio ----------------
def within_cv(idx, model, cols_idx, reps=10, seed0=3000, X=None, y=None, return_oof=False):
    X = D.X if X is None else X; y = D.y if y is None else y
    n_ = len(idx); losses = np.zeros(n_); base = np.zeros(n_); preds = np.zeros(n_)
    for rep in range(reps):
        oof = np.zeros(n_); ob = np.zeros(n_)
        for tr, te in KFold(5, shuffle=True, random_state=seed0 + rep).split(np.arange(n_)):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 40))) if model == "Ridge" else ExtraTreesRegressor(100 if reps < 10 else 200, min_samples_leaf=3, max_features=0.5, n_jobs=1, random_state=seed0 + rep)
            m.fit(X[idx][tr][:, cols_idx], y[idx][tr]); oof[te] = m.predict(X[idx][te][:, cols_idx]); ob[te] = y[idx][tr].mean()
        losses += (y[idx] - oof) ** 2 / reps; base += (y[idx] - ob) ** 2 / reps; preds += oof / reps
    skill = 1 - losses.sum() / base.sum()
    return (skill, losses, base, preds) if return_oof else skill


wc = []
for m_, k in big.items():
    if k.sum() < 15:
        continue
    idx = np.where(k)[0]
    nperm_r, nperm_e = (200, 100) if m_ == "Chignahuapan" else (200, 60)
    sets = ["SAT", "S2", "PL", "CLIM", "TOPO", "GEOM", "LOC", "CUR", "ALLNOLOC", "T_PRE", "T_VEG", "T_PICO", "T_SEN", "T_PHTIM", "T_PHLEV", "T_HET"]
    for model in ("Ridge", "ET"):
        for fs in sets:
            if model == "ET" and fs not in ("SAT", "ALLNOLOC", "CLIM", "LOC", "CUR", "TOPO"):
                continue
            ci = D.idx(fs); reps = 10 if model == "Ridge" else 3
            skill, lo, ba, pr_ = within_cv(idx, model, ci, reps, return_oof=True)
            nperm = nperm_r if model == "Ridge" else nperm_e
            null = []
            for b in range(nperm):
                yperm = D.y.copy(); yperm[idx] = rng.permutation(D.y[idx])
                null.append(within_cv(idx, model, ci, min(reps, 3), y=yperm))
            null = np.array(null)
            # IC bootstrap de la skill (por parcela)
            bsk = []
            for b in range(1000):
                ii = rng.integers(0, len(idx), len(idx)); bsk.append(1 - lo[ii].sum() / ba[ii].sum())
            wc.append(dict(municipio=m_, n=int(k.sum()), modelo=model, conjunto=fs, skill_cv_vs_media_municipal=skill, ic95_lo=np.percentile(bsk, 2.5), ic95_hi=np.percentile(bsk, 97.5),
                           null_p95=np.percentile(null, 95), p_perm=(1 + (null >= skill).sum()) / (1 + len(null)), r_pred_obs=pearsonr(D.y[idx], pr_)[0] if pr_.std() > 1e-9 else np.nan))
            print(f"{time.strftime('%H:%M:%S')} {m_[:12]:12s} {model:5s} {fs:9s} skill={skill:+.3f} [{np.percentile(bsk,2.5):+.3f},{np.percentile(bsk,97.5):+.3f}] null95={np.percentile(null,95):+.3f} p={wc[-1]['p_perm']:.3f}", flush=True)
WC = pd.DataFrame(wc)
WC["p_holm_por_municipio_modelo"] = WC.groupby(["municipio", "modelo"]).p_perm.transform(lambda s: holm(s.values))
WC.round(4).to_csv(f"{RES}/06_dentro_de_municipio_cv_y_permutacion.csv", index=False)


# ---------------- D. estabilidad del aporte residual ----------------
O2 = O[O.cfg.isin(["ref_muni", "resid_muni|Ridge|SAT", "resid_muni|ET|SAT", "resid_muni|ET|ALLNOLOC", "resid_muni|ET|ALL", "resid_muni|Ridge|CUR", "resid_muni|Ridge|CLIM", "resid_muni|ET|CLIM", "resid_muni|CB|ALL", "resid_muni|Ridge|TOPO"]) & (O.scheme == "RCV5x10")]
Rr = O2.groupby(["cfg", "rep"]).apply(lambda t: rmse(t.y, t.pred)).unstack()
dd = Rr.sub(Rr.loc["ref_muni"], axis=1).drop(index="ref_muni")
st = pd.DataFrame(dict(dRMSE_medio_reps=dd.mean(1), dRMSE_min=dd.min(1), dRMSE_max=dd.max(1), frac_reps_mejor_que_prior=(dd < 0).mean(1)))
st.round(4).to_csv(f"{RES}/06_estabilidad_por_repeticion.csv"); print(st.round(4))

mrow = []
for c in ["resid_muni|Ridge|SAT", "resid_muni|ET|SAT", "resid_muni|ET|ALLNOLOC", "resid_muni|ET|ALL", "resid_muni|Ridge|CUR", "resid_muni|Ridge|CLIM", "resid_muni|ET|CLIM", "resid_muni|CB|ALL"]:
    for sch in ("RCV5x10", "SPA_GRID3B05", "SPA_GRID8B1"):
        if (c, sch) not in L:
            continue
        for m_, k in big.items():
            la, lb = L[(c, sch)][k], L[("ref_muni", sch)][k]
            bs = []
            for b in range(2000):
                ii = rng.integers(0, k.sum(), k.sum()); bs.append(np.sqrt(la[ii].mean()) - np.sqrt(lb[ii].mean()))
            mrow.append(dict(cfg=c, esquema=sch, municipio=m_, n=int(k.sum()), RMSE_modelo=np.sqrt(la.mean()), RMSE_prior=np.sqrt(lb.mean()), dRMSE=np.sqrt(la.mean()) - np.sqrt(lb.mean()),
                             ic_lo=np.percentile(bs, 2.5), ic_hi=np.percentile(bs, 97.5)))
pd.DataFrame(mrow).round(4).to_csv(f"{RES}/06_aporte_residual_por_municipio.csv", index=False)

# ---------------- E. sensibilidad del encogimiento del prior ----------------
sh = []
for s_ in (0.0, 0.5, 1, 3, 10, 30, 1e6):
    v = []
    for rep in range(10):
        oof = np.zeros(D.n)
        for fold, tr, te in SPL["RCV5x10"][rep]:
            oof[te] = prior_muni(D, tr, te, shrink=s_)[1]
        v.append(rmse(D.y, oof))
    sh.append(dict(encogimiento=s_, RMSE_RCV=np.mean(v), sd_reps=np.std(v)))
pd.DataFrame(sh).round(4).to_csv(f"{RES}/06_sensibilidad_encogimiento_prior.csv", index=False); print(pd.DataFrame(sh).round(4))
print("OK 06_within_municipality")
