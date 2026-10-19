"""
05_stability.py - E3: estabilidad de coeficientes, seleccion de variables, hiperparametros/semillas y errores
bajo multiples esquemas de validacion. Incluye el experimento de FUGA POR SELECCION (seleccion dentro vs fuera de fold).
"""
import itertools, time
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from oofio import *

D = Data(); RES = PATH["res"]; rng = np.random.default_rng(GLOBAL_SEED)
SPL = load_splits(D, ["RCV5x10", "SPA_KM6", "SPA_BAND6", "SPA_GRID3B05", "SPA_GRID8B1", "LMO"])
USE_REPS = {"RCV5x10": 5, "SPA_KM6": 1, "SPA_BAND6": 1, "SPA_GRID3B05": 3, "SPA_GRID8B1": 3, "LMO": 1}


def iter_folds(scheme):
    for rep, folds in SPL[scheme].items():
        if int(rep) < USE_REPS[scheme]:
            for fold, tr, te in folds:
                yield rep, fold, tr, te


def jaccard(a, b):
    a, b = set(a), set(b); return len(a & b) / max(1, len(a | b))


# ============== A. estabilidad de coeficientes Ridge ==============
rows, topfreq = [], []
for cfg in [dict(mode="direct", model="Ridge", set="ALLNOLOC"), dict(mode="resid_muni", model="Ridge", set="SAT"),
            dict(mode="resid_muni", model="Ridge", set="ALLNOLOC"), dict(mode="resid_muni", model="Ridge", set="CUR")]:
    cols = D.F[cfg["set"]]
    for sch in USE_REPS:
        coefs, alphas = [], []
        for rep, fold, tr, te in iter_folds(sch):
            _, m = fit_predict(D, cfg, tr, te, 1000 * int(rep) + int(fold), return_model=True)
            r = m.named_steps["ridgecv"]; coefs.append(r.coef_.copy()); alphas.append(r.alpha_)
        C = np.array(coefs); K = len(C)
        sign_cons = np.abs(np.sign(C).mean(0))
        meanabs = np.abs(C).mean(0); top = np.argsort(-meanabs)[:10]
        pairs = list(itertools.combinations(range(K), 2)); pairs = [pairs[i] for i in rng.choice(len(pairs), min(300, len(pairs)), replace=False)]
        rho = np.mean([spearmanr(C[a], C[b])[0] for a, b in pairs])
        jac = np.mean([jaccard(np.argsort(-np.abs(C[a]))[:10], np.argsort(-np.abs(C[b]))[:10]) for a, b in pairs])
        rows.append(dict(cfg=f"{cfg['mode']}|{cfg['model']}|{cfg['set']}", esquema=sch, n_fits=K, p=C.shape[1], alpha_mediana=np.median(alphas), alpha_p10=np.percentile(alphas, 10), alpha_p90=np.percentile(alphas, 90),
                         spearman_medio_entre_vectores_coef=rho, jaccard_top10=jac, consist_signo_top10_media=sign_cons[top].mean(), frac_features_consist_signo_ge_0_9=(sign_cons >= 0.9).mean(),
                         coef_abs_media_top10=meanabs[top].mean()))
        for j in top:
            topfreq.append(dict(cfg=f"{cfg['mode']}|{cfg['model']}|{cfg['set']}", esquema=sch, feature=cols[j], coef_medio_estandarizado=C[:, j].mean(), coef_sd=C[:, j].std(),
                                consist_signo=sign_cons[j], frecuencia_en_top10=np.mean([j in np.argsort(-np.abs(C[k]))[:10] for k in range(K)])))
A = pd.DataFrame(rows); A.round(4).to_csv(f"{RES}/05_estabilidad_coeficientes_ridge.csv", index=False)
pd.DataFrame(topfreq).round(4).to_csv(f"{RES}/05_top10_coeficientes_ridge.csv", index=False)
print(A.round(3).to_string(index=False))


# ============== B. estabilidad de la seleccion de variables ==============
def sel_filter(Xtr, ttr, k):
    s = np.array([abs(spearmanr(Xtr[:, j], ttr)[0]) for j in range(Xtr.shape[1])]); s = np.nan_to_num(s)
    return np.argsort(-s)[:k]


def sel_lasso(Xtr, ttr, k=None):
    Z = StandardScaler().fit_transform(Xtr); m = LassoCV(cv=5, alphas=40, max_iter=5000, random_state=0).fit(Z, ttr - ttr.mean())
    return np.where(np.abs(m.coef_) > 1e-8)[0]


def sel_et(Xtr, ttr, k, seed=0):
    m = ExtraTreesRegressor(300, min_samples_leaf=3, max_features=0.5, random_state=seed, n_jobs=1).fit(Xtr, ttr)
    return np.argsort(-m.feature_importances_)[:k]


pool = D.idx("ALLNOLOC"); names = np.array(D.cols)[pool]
srows, sfreq = [], []
for target in ("resid_muni", "direct"):
    for method, fn in (("filtro_spearman_top10", lambda X, t: sel_filter(X, t, 10)), ("lasso_cv", lambda X, t: sel_lasso(X, t)), ("ET_importancia_top10", lambda X, t: sel_et(X, t, 10))):
        for sch in USE_REPS:
            sets_ = []
            for rep, fold, tr, te in iter_folds(sch):
                t = (D.y[tr] - prior_muni(D, tr, te)[0]) if target == "resid_muni" else D.y[tr]
                sets_.append(pool[fn(D.X[tr][:, pool], t)])
            K = len(sets_); pairs = list(itertools.combinations(range(K), 2)); pairs = [pairs[i] for i in rng.choice(len(pairs), min(300, len(pairs)), replace=False)]
            jac = np.mean([jaccard(sets_[a], sets_[b]) for a, b in pairs])
            cnt = pd.Series(np.concatenate(sets_)).value_counts() / K
            srows.append(dict(objetivo=target, metodo=method, esquema=sch, n_folds=K, tam_conjunto_medio=np.mean([len(s) for s in sets_]), jaccard_medio=jac,
                              freq_max=cnt.iloc[0], n_features_con_freq_ge_0_8=int((cnt >= 0.8).sum()), n_features_distintas=len(cnt)))
            for j, f in cnt.head(5).items():
                sfreq.append(dict(objetivo=target, metodo=method, esquema=sch, feature=D.cols[j], frecuencia=f))
SR = pd.DataFrame(srows); SR.round(4).to_csv(f"{RES}/05_estabilidad_seleccion.csv", index=False)
pd.DataFrame(sfreq).round(4).to_csv(f"{RES}/05_features_mas_frecuentes.csv", index=False)
print(SR.round(3).to_string(index=False))


# ============== C. FUGA POR SELECCION: seleccion dentro vs fuera del fold (Ridge, CV aleatoria 10x5) ==============
def ridge_fit_pred(Xtr, t, Xte):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 40))).fit(Xtr, t); return m.predict(Xte)


def cv_selection(target, k, leaky, reps=10, scheme="RCV5x10", feat_pool=pool):
    out = []
    sp = SPL[scheme] if scheme in SPL else load_splits(D, [scheme])[scheme]
    if leaky:
        t_all = (D.y - prior_muni(D, np.arange(D.n), np.arange(D.n))[0]) if target == "resid_muni" else D.y
        sel_all = feat_pool[sel_filter(D.X[:, feat_pool], t_all, k)]
    for rep in range(reps if scheme == "RCV5x10" else 1):
        oof = np.zeros(D.n)
        for fold, tr, te in sp[rep]:
            if target == "resid_muni":
                p_tr, p_te = prior_muni(D, tr, te); t = D.y[tr] - p_tr; off = p_te
            else:
                t, off = D.y[tr], 0.0
            sel = sel_all if leaky else feat_pool[sel_filter(D.X[tr][:, feat_pool], t, k)]
            oof[te] = off + ridge_fit_pred(D.X[tr][:, sel], t, D.X[te][:, sel])
        out.append(rmse(D.y, oof))
    return np.mean(out), np.std(out)


leak = []
for scheme in ("RCV5x10", "SPA_KM6"):
    if scheme not in SPL:
        continue
    for target in ("resid_muni", "direct"):
        for k in (5, 10, 24):
            h = cv_selection(target, k, leaky=False, scheme=scheme); l = cv_selection(target, k, leaky=True, scheme=scheme)
            leak.append(dict(esquema=scheme, objetivo=target, k=k, RMSE_seleccion_DENTRO_fold=h[0], sd_h=h[1], RMSE_seleccion_FUERA_fold_con_fuga=l[0], sd_l=l[1], sesgo_optimista=h[0] - l[0]))
LK = pd.DataFrame(leak); LK.round(4).to_csv(f"{RES}/05_fuga_por_seleccion.csv", index=False); print(LK.round(3).to_string(index=False))

# CUR24 (lista elegida por el equipo) vs 24 variables aleatorias vs seleccion honesta - Ridge sobre residuo municipal, RCV
def cv_fixed(cols_idx, target="resid_muni", reps=3, scheme="RCV5x10"):
    r_ = []
    for rep in range(reps):
        oof = np.zeros(D.n)
        for fold, tr, te in SPL[scheme][rep]:
            p_tr, p_te = prior_muni(D, tr, te); oof[te] = p_te + ridge_fit_pred(D.X[tr][:, cols_idx], D.y[tr] - p_tr, D.X[te][:, cols_idx])
        r_.append(rmse(D.y, oof))
    return np.mean(r_)


cur = D.idx("CUR"); r_cur = cv_fixed(cur)
rnd_all = [cv_fixed(rng.choice(pool, 24, replace=False)) for _ in range(150)]
sat_pool = D.idx("SAT"); rnd_sat = [cv_fixed(rng.choice(sat_pool, 24, replace=False)) for _ in range(150)]
rnd_cur_like = [cv_fixed(np.concatenate([rng.choice(sat_pool, 22, replace=False), rng.choice(D.idx("TOPO"), 2, replace=False)])) for _ in range(150)]
honest24 = cv_selection("resid_muni", 24, leaky=False)[0]
CMP = pd.DataFrame([dict(conjunto="CUR24 (lista curada del baseline)", RMSE=r_cur),
                    dict(conjunto="24 aleatorias de ALLNOLOC (media)", RMSE=np.mean(rnd_all)), dict(conjunto="  p5 / p95", RMSE=f"{np.percentile(rnd_all,5):.4f} / {np.percentile(rnd_all,95):.4f}"),
                    dict(conjunto="24 aleatorias de SAT (media)", RMSE=np.mean(rnd_sat)), dict(conjunto="  p5 / p95", RMSE=f"{np.percentile(rnd_sat,5):.4f} / {np.percentile(rnd_sat,95):.4f}"),
                    dict(conjunto="percentil de CUR24 entre aleatorias de SAT (menor RMSE = mejor)", RMSE=float((np.array(rnd_sat) < r_cur).mean())),
                    dict(conjunto="percentil de CUR24 entre aleatorias de ALLNOLOC", RMSE=float((np.array(rnd_all) < r_cur).mean())),
                    dict(conjunto="seleccion honesta top24 por Spearman dentro de fold", RMSE=honest24)])
CMP.to_csv(f"{RES}/05_cur24_vs_aleatorias.csv", index=False); print(CMP.to_string(index=False))

# ============== D. sensibilidad a semillas e hiperparametros (ET) ==============
hp = []
for cfg in (dict(mode="resid_muni", set="ALL"), dict(mode="direct", set="ALL"), dict(mode="resid_muni", set="SAT")):
    for msl, mf in itertools.product((1, 3, 5, 10), (0.2, 0.5, 1.0)):
        for sch, nrep in (("RCV5x10", 3), ("SPA_KM6", 1)):
            vals = []
            for rep in range(nrep):
                oof = np.zeros(D.n)
                for fold, tr, te in SPL[sch][rep]:
                    ci = D.idx(cfg["set"])
                    if cfg["mode"] == "resid_muni":
                        p_tr, p_te = prior_muni(D, tr, te); t, off = D.y[tr] - p_tr, p_te
                    else:
                        t, off = D.y[tr], 0.0
                    m = ExtraTreesRegressor(200, min_samples_leaf=msl, max_features=mf, n_jobs=1, random_state=1000 * rep + fold).fit(D.X[tr][:, ci], t)
                    oof[te] = off + m.predict(D.X[te][:, ci])
                vals.append(rmse(D.y, oof))
            hp.append(dict(modo=cfg["mode"], conjunto=cfg["set"], esquema=sch, min_samples_leaf=msl, max_features=mf, RMSE=np.mean(vals)))
HP = pd.DataFrame(hp); HP.round(4).to_csv(f"{RES}/05_sensibilidad_hiperparametros_ET.csv", index=False)
print(HP.pivot_table(index=["modo", "conjunto", "esquema"], values="RMSE", aggfunc=["min", "max"]).round(3))

# semillas (mismo split, 8 semillas del modelo)
sd_rows = []
for cfg in (dict(mode="resid_muni", model="ET", set="ALL"), dict(mode="direct", model="ET", set="ALL")):
    v = []
    for s in range(8):
        oof = np.zeros(D.n)
        for fold, tr, te in SPL["RCV5x10"][0]:
            oof[te] = fit_predict(D, cfg, tr, te, 777 + 10 * s + fold)
        v.append(rmse(D.y, oof))
    sd_rows.append(dict(cfg=cfg_id(cfg), RMSE_media=np.mean(v), RMSE_sd_por_semilla=np.std(v), min=np.min(v), max=np.max(v)))
pd.DataFrame(sd_rows).round(4).to_csv(f"{RES}/05_sensibilidad_semillas.csv", index=False); print(pd.DataFrame(sd_rows).round(4))
print("OK 05_stability (A-D)")
