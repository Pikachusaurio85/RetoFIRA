"""
07_shap_groups.py - E5: SHAP agrupado por familias (valores SHAP calculados SOLO sobre parcelas fuera de entrenamiento de cada fold),
importancia por permutacion de grupos (global / intra-municipio / intra-cluster) y diagnostico de colinealidad.
"""
import time, sys
import shap
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from oofio import *

D = Data(); RES = PATH["res"]; rng = np.random.default_rng(GLOBAL_SEED)
SPL = load_splits(D, ["RCV5x10", "SPA_KM6", "SPA_GRID3B05"])
CL = micro_clusters(D)
N_PFI = 10

FAM = ["LOC", "S2", "LS", "PL", "CLIM", "TOPO", "GEOM"]
TEMP = ["LOC", "CLIM", "TOPO", "GEOM", "T_PRE", "T_VEG", "T_PICO", "T_SEN", "T_PHTIM", "T_PHLEV", "T_HET"]


def groups_for(setname, kind):
    ci = list(D.idx(setname)); pos = {D.cols[j]: i for i, j in enumerate(ci)}
    if kind == "familia":
        names = [g for g in FAM if all(c in pos for c in D.F[g][:1])]
        return {g: [pos[c] for c in D.F[g] if c in pos] for g in names}
    if kind == "temporal":
        names = [g for g in TEMP if all(c in pos for c in D.F[g][:1])]
        return {g: [pos[c] for c in D.F[g] if c in pos] for g in names}
    if kind == "concepto":
        cg = sat_concept_groups([D.cols[j] for j in ci])
        return {k: [pos[c] for c in v] for k, v in cg.items()}


def shap_vals(m, model, Xtr, Xte):
    if model == "ET":
        e = shap.TreeExplainer(m); sv = e.shap_values(Xte, check_additivity=False); base = float(np.ravel(e.expected_value)[0])
    elif model == "CB":
        from catboost import Pool
        out = m.get_feature_importance(Pool(Xte), type="ShapValues"); sv, base = out[:, :-1], float(out[0, -1])
    else:   # Ridge en espacio estandarizado: contribucion = coef * z (exacto, intervencional)
        sc, rg = m.named_steps["standardscaler"], m.named_steps["ridgecv"]
        Z = sc.transform(Xte); sv = Z * rg.coef_[None, :]; base = float(rg.intercept_)
    return np.asarray(sv), base


def perm_idx(kind, te, rng_):
    n_ = len(te)
    if kind == "global":
        return rng_.permutation(n_)
    lab = D.mun_code[te] if kind == "intra_municipio" else CL[te]
    idx = np.arange(n_)
    for u in np.unique(lab):
        ii = np.where(lab == u)[0]; idx[ii] = rng_.permutation(ii)
    return idx


def explain(cfg, scheme, nrep, gkind, tag):
    gdef = groups_for(cfg["set"], gkind); ci = D.idx(cfg["set"])
    sh_par = {g: np.zeros(D.n) for g in list(gdef) + ["MUN_prior"]}; cnt = np.zeros(D.n)
    pf = {(g, v): [] for g in gdef for v in ("global", "intra_municipio", "intra_cluster")}   # lista de (rep, dMSE, n_te)
    mse0_rep = {}
    t0 = time.time()
    for rep, folds in SPL[scheme].items():
        if int(rep) >= nrep:
            continue
        fold_acc = {k: [] for k in pf}; se0 = []; nte = []
        for fold, tr, te in folds:
            seed = 1000 * int(rep) + int(fold)
            if cfg["mode"] == "resid_muni":
                p_tr, p_te = prior_muni(D, tr, te); target, off = D.y[tr] - p_tr, p_te
            else:
                target, off = D.y[tr], 0.0
            Xtr, Xte = D.X[tr][:, ci], D.X[te][:, ci]
            m = make_model(cfg["model"], seed); m.fit(Xtr, target)
            sv, base = shap_vals(m, cfg["model"], Xtr, Xte)
            for g, pos in gdef.items():
                sh_par[g][te] += sv[:, pos].sum(1)
            if cfg["mode"] == "resid_muni":
                sh_par["MUN_prior"][te] += p_te - D.y[tr].mean()
            cnt[te] += 1
            pred0 = off + np.ravel(m.predict(Xte)); e0 = (D.y[te] - pred0) ** 2; se0.append(e0.sum()); nte.append(len(te))
            r_ = np.random.default_rng(seed + 99)
            for g, pos in gdef.items():
                for v in ("global", "intra_municipio", "intra_cluster"):
                    d_ = []
                    for _ in range(N_PFI):
                        Xp = Xte.copy(); Xp[:, pos] = Xte[perm_idx(v, te, r_)][:, pos]
                        d_.append((((D.y[te] - (off + np.ravel(m.predict(Xp)))) ** 2).sum()))
                    fold_acc[(g, v)].append(np.mean(d_) - e0.sum())          # suma de dSE en el fold
        n_tot = sum(nte); mse0 = sum(se0) / n_tot; mse0_rep[rep] = mse0
        for k, v in fold_acc.items():
            pf[k].append((rep, sum(v) / n_tot, mse0))
    G = pd.DataFrame({g: v / np.maximum(cnt, 1) for g, v in sh_par.items()}); G.insert(0, "id", D.ids); G.insert(1, "municipio", D.mun)
    # tablas
    rows = []
    for g in [c for c in G.columns if c not in ("id", "municipio")]:
        v = G[g].to_numpy(); between = (pd.Series(v).groupby(D.mun).transform("mean") - v.mean()); within = v - pd.Series(v).groupby(D.mun).transform("mean").values
        rows.append(dict(cfg=cfg_id(cfg), esquema=scheme, agrupacion=gkind, grupo=g, n_features=len(gdef.get(g, [])), mean_abs_shap=np.mean(np.abs(v)),
                         frac_var_shap_entre_municipios=float((between ** 2).sum() / max(((v - v.mean()) ** 2).sum(), 1e-12)), sd_shap_intra_municipio=float(within.std()),
                         mean_abs_shap_intra_municipio=float(np.mean(np.abs(within)))))
    S = pd.DataFrame(rows); S["share_mean_abs"] = S.mean_abs_shap / S.mean_abs_shap.sum()
    prow = []
    for (g, v), lst in pf.items():
        d = np.array([x[1] for x in lst]); m0 = np.array([x[2] for x in lst]); drm = np.sqrt(m0 + d) - np.sqrt(m0)
        prow.append(dict(cfg=cfg_id(cfg), esquema=scheme, agrupacion=gkind, grupo=g, tipo_permutacion=v, dMSE_media=d.mean(), dRMSE_media=drm.mean(), dRMSE_sd_entre_reps=drm.std() if len(drm) > 1 else np.nan, n_reps=len(drm)))
    print(f"{time.strftime('%H:%M:%S')} {tag} {cfg_id(cfg)} {scheme} {gkind} ({time.time()-t0:.0f}s)", flush=True)
    return S, pd.DataFrame(prow), G


plan = [
    (dict(mode="direct", model="ET", set="ALL"), "RCV5x10", 3, "familia"),
    (dict(mode="direct", model="ET", set="ALL"), "RCV5x10", 3, "temporal"),
    (dict(mode="direct", model="ET", set="ALL"), "SPA_KM6", 1, "familia"),
    (dict(mode="direct", model="ET", set="ALL"), "SPA_GRID3B05", 2, "familia"),
    (dict(mode="resid_muni", model="ET", set="ALL"), "RCV5x10", 3, "familia"),
    (dict(mode="resid_muni", model="ET", set="ALL"), "RCV5x10", 3, "temporal"),
    (dict(mode="resid_muni", model="ET", set="ALL"), "SPA_KM6", 1, "familia"),
    (dict(mode="resid_muni", model="ET", set="ALL"), "SPA_GRID3B05", 2, "familia"),
    (dict(mode="direct", model="Ridge", set="ALL"), "RCV5x10", 3, "familia"),
    (dict(mode="direct", model="Ridge", set="ALL"), "SPA_KM6", 1, "familia"),
    (dict(mode="resid_muni", model="Ridge", set="ALL"), "RCV5x10", 3, "familia"),
    (dict(mode="direct", model="CB", set="ALL"), "RCV5x10", 2, "familia"),
    (dict(mode="resid_muni", model="CB", set="ALL"), "RCV5x10", 2, "familia"),
    (dict(mode="direct", model="ET", set="SAT"), "RCV5x10", 3, "concepto"),
    (dict(mode="direct", model="ET", set="ALLNOLOC"), "RCV5x10", 3, "familia"),
]
S_all, P_all, G_all = [], [], {}
for i, (cfg, sch, nrep, gk) in enumerate(plan):
    S, Pm, G = explain(cfg, sch, nrep, gk, f"[{i+1}/{len(plan)}]")
    S_all.append(S); P_all.append(Pm); G_all[(cfg_id(cfg), sch, gk)] = G
    pd.concat(S_all).round(5).to_csv(f"{RES}/07_shap_grupos.csv", index=False)
    pd.concat(P_all).round(5).to_csv(f"{RES}/07_permutacion_grupos.csv", index=False)
SA = pd.concat(S_all); PA = pd.concat(P_all)
os.makedirs(f"{PATH['oof']}/shap", exist_ok=True)
for (c, s, g), G in G_all.items():
    G.round(5).to_csv(f"{PATH['oof']}/shap/shap_por_parcela__{safe(c)}__{s}__{g}.csv.gz", index=False, compression="gzip")
pd.set_option("display.width", 250)
print(SA[(SA.agrupacion == "familia") & (SA.esquema == "RCV5x10")].pivot_table(index="grupo", columns="cfg", values="share_mean_abs").round(3))

# --------- acuerdo de rangos entre modelos y esquemas (grupos 'familia') ---------
fam_tab = SA[SA.agrupacion == "familia"].pivot_table(index="grupo", columns=["cfg", "esquema"], values="mean_abs_shap")
fam_tab = fam_tab.drop(index="MUN_prior", errors="ignore")
keys = list(fam_tab.columns); agree = pd.DataFrame(index=[f"{a}@{b}" for a, b in keys], columns=[f"{a}@{b}" for a, b in keys], dtype=float)
for a in keys:
    for b in keys:
        x, y_ = fam_tab[a].dropna(), fam_tab[b].dropna(); ix = x.index.intersection(y_.index)
        agree.loc[f"{a[0]}@{a[1]}", f"{b[0]}@{b[1]}"] = kendalltau(x[ix], y_[ix])[0] if len(ix) > 3 else np.nan
agree.round(3).to_csv(f"{RES}/07_acuerdo_rangos_shap_entre_modelos.csv");

# --------- redundancia entre grupos (sin etiquetas, 197 parcelas): ¿cuanto de cada familia se reconstruye con las demas? ---------
dall = load_all_features()
red = []
for g in ["S2", "LS", "PL", "CLIM", "TOPO", "GEOM", "T_PRE", "T_VEG", "T_PICO", "T_SEN", "T_PHTIM", "T_PHLEV", "T_HET"]:
    Xg = StandardScaler().fit_transform(dall[D.F[g]].to_numpy(float)); k = min(3, Xg.shape[1]); pcs = PCA(k).fit_transform(Xg)
    others = {"ubicacion(LOC)": D.F["LOC"], "todo_lo_demas_sin_LOC": [c for c in D.F["ALLNOLOC"] if c not in D.F[g]], "LOC+resto": [c for c in D.F["ALL"] if c not in D.F[g]]}
    if g.startswith("T_"):
        others["todo_lo_demas_sin_LOC"] = [c for c in D.F["ALLNOLOC"] if c not in D.F[g]]
    row = dict(grupo=g, p=Xg.shape[1])
    for oname, cols in others.items():
        Xo = StandardScaler().fit_transform(dall[cols].to_numpy(float)); r2s = []
        for j in range(k):
            pr = cross_val_predict(RidgeCV(alphas=np.logspace(-1, 4, 30)), Xo, pcs[:, j], cv=KFold(5, shuffle=True, random_state=0))
            r2s.append(1 - ((pcs[:, j] - pr) ** 2).sum() / ((pcs[:, j] - pcs[:, j].mean()) ** 2).sum())
        row[f"R2cv_PC1-{k}_desde_{oname}"] = float(np.mean(r2s)); row[f"R2cv_PC1_desde_{oname}"] = float(r2s[0])
    red.append(row)
pd.DataFrame(red).round(3).to_csv(f"{RES}/07_redundancia_entre_grupos.csv", index=False); print(pd.DataFrame(red).round(3).iloc[:, :6].to_string(index=False))

# --------- gemelos correlacionados entre las variables mas importantes (ET directo ALL, RCV) ---------
ci = D.idx("ALL"); tops = []
for rep in range(2):
    for fold, tr, te in SPL["RCV5x10"][rep]:
        m = make_model("ET", 1000 * rep + fold).fit(D.X[tr][:, ci], D.y[tr]); tops.append(m.feature_importances_)
imp = np.mean(tops, 0); order = np.argsort(-imp)[:20]
C = pd.DataFrame(D.X[:, ci]).corr(method="spearman").abs().values
tw = []
for j in order:
    cj = C[j].copy(); cj[j] = 0; partner = np.argsort(-cj)[:3]
    tw.append(dict(feature=D.cols[ci[j]], importancia_ET=imp[j], n_features_con_absrho_ge_0_8=int((cj >= 0.8).sum()), n_features_con_absrho_ge_0_9=int((cj >= 0.9).sum()),
                   max_absrho=cj.max(), mejores_gemelos=";".join(D.cols[ci[p]] + f"({cj[p]:.2f})" for p in partner)))
pd.DataFrame(tw).round(3).to_csv(f"{RES}/07_top20_features_gemelos_correlacionados.csv", index=False)
print("OK 07_shap_groups")
