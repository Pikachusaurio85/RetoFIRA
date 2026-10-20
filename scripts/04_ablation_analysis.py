"""
04_ablation_analysis.py - E2/E3: metricas por esquema, ablaciones incrementales pareadas, errores por parcela y por bloque.
Lee oof/main/*.csv.gz (predicciones OOF con particiones identicas para todas las configuraciones).
"""
from oofio import *

D = Data(); RES = PATH["res"]
O = load_oof("main"); print("OOF filas", len(O), "configs", O.cfg.nunique())
L, P = per_parcel(O, D)
M = rep_metrics(O)
cfgs = sorted(O.cfg.unique()); schemes = [s for s in MAIN_SCHEMES if s in O.scheme.unique()]

# --- 1. metricas por esquema (media sobre repeticiones) + skill vs prior municipal + RMSE por parcela-promedio
ref = M[M.cfg == "ref_muni"].set_index("scheme").RMSE
M["skill_vs_prior_municipal"] = [1 - (r.RMSE / ref[r.scheme]) ** 2 for r in M.itertuples()]
M["modo"] = M.cfg.str.split("|").str[0]; M["modelo"] = M.cfg.str.split("|").str[1]; M["conjunto"] = M.cfg.str.split("|").str[2]
M.round(4).to_csv(f"{RES}/04_metricas_por_esquema.csv", index=False)
W = M.pivot(index="cfg", columns="scheme", values="RMSE")[schemes]; W.round(3).to_csv(f"{RES}/04_rmse_ancho.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
print(W.round(3).sort_values("RCV5x10").to_string())

# --- 2. contrastes pareados
B = Boot(D, B=5000)
contr = []


def add(group, A, Bc, models, note=""):
    for sch in schemes:
        if (A, sch) in L and (Bc, sch) in L:
            r = B.paired(L[(A, sch)], L[(Bc, sch)]); r.update(grupo=group, A=A, B=Bc, esquema=sch, nota=note); contr.append(r)


for model in ("Ridge", "ET", "CB"):
    fam_add = ["LOC", "S2", "LS", "PL", "SAT", "CLIM", "TOPO", "GEOM", "T_PRE", "T_VEG", "T_PICO", "T_SEN", "T_PHTIM", "T_PHLEV", "T_HET", "CUR", "B1", "ALLNOLOC", "ALL"]
    # C1: valor sobre el prior municipal (residuo)
    for f in fam_add:
        add(f"C1 residuo[F] vs prior municipal | {model}", f"resid_muni|{model}|{f}", "ref_muni", model, f)
    # C2: valor sobre ubicacion sola (modelo directo)
    for f in ["S2", "LS", "PL", "SAT", "CLIM", "TOPO", "GEOM", "CUR", "B1", "ALLNOLOC", "ALL"]:
        add(f"C2 directo[F] vs directo[LOC] | {model}", f"direct|{model}|{f}", f"direct|{model}|LOC", model, f)
    # C3: drop-one desde ALL (directo)
    for x, full in (("ALL-SAT", "ALL"), ("ALL-CLIM", "ALL"), ("ALL-TOPO", "ALL"), ("ALL-GEOM", "ALL"), ("ALLNOLOC", "ALL")):
        add(f"C3 drop-one: ALL sin X vs ALL | {model}", f"direct|{model}|{x}", f"direct|{model}|{full}", model, "quitar " + x.replace("ALL-", "").replace("ALLNOLOC", "LOC"))
    # C4: secuencia aditiva (residuo sobre prior municipal)
    seq = [("LOC", "ref_muni"), ("LOC+CLIM", f"resid_muni|{model}|LOC"), ("LOC+CLIM+TOPO", f"resid_muni|{model}|LOC+CLIM"),
           ("LOC+CLIM+TOPO+GEOM", f"resid_muni|{model}|LOC+CLIM+TOPO"), ("ALL", f"resid_muni|{model}|LOC+CLIM+TOPO+GEOM")]
    for f, base in seq:
        add(f"C4 secuencia MUN>LOC>CLIM>TOPO>GEOM>SAT | {model}", f"resid_muni|{model}|{f}", base, model, f)
    # C5: complejidad: residuo vs directo con el mismo conjunto
    for f in ["SAT", "ALLNOLOC", "ALL", "CUR", "CLIM"]:
        add(f"C5 residuo vs directo | {model}", f"resid_muni|{model}|{f}", f"direct|{model}|{f}", model, f)
    # C6: sat dado todo lo demas, en modo residuo
    add(f"C6 SAT dado MUN+LOC+CLIM+TOPO+GEOM | {model}", f"resid_muni|{model}|ALL", f"resid_muni|{model}|ALL-SAT", model, "SAT")
# C7: referencias entre si
for a, b in (("ref_muni", "ref_global"), ("ref_knn", "ref_muni"), ("ref_muni_knnres", "ref_muni"), ("ref_muni_knnres", "ref_knn")):
    add("C7 referencias", a, b, "-", "")
# C8: prior espacial vs prior municipal con residuo satelital
for model in ("Ridge", "ET"):
    for f in ["SAT", "ALLNOLOC", "ALL", "CUR", "CLIM"]:
        add(f"C8 prior kNN vs prior municipal (residuo) | {model}", f"resid_knn|{model}|{f}", f"resid_muni|{model}|{f}", model, f)
        add(f"C8b residuo[F] sobre prior kNN vs prior kNN solo | {model}", f"resid_knn|{model}|{f}", "ref_knn", model, f)
C = pd.DataFrame(contr)
C["p_clu_holm"] = C.groupby(["grupo", "esquema"]).p_clu.transform(lambda s: holm(s.values))
C["p_par_holm"] = C.groupby(["grupo", "esquema"]).p_par.transform(lambda s: holm(s.values))
C["decision"] = C.apply(decide, axis=1)
C.round(5).to_csv(f"{RES}/04_contrastes_pareados.csv", index=False)
print(C[C.esquema == "RCV5x10"][["grupo", "nota", "RMSE_A", "RMSE_B", "dRMSE", "ci_lo_clu", "ci_hi_clu", "p_clu_holm", "decision"]].round(3).to_string(index=False))

# --- 3. errores por parcela (pareados): una fila por parcela x cfg x esquema, solo para configuraciones clave
KEY = ["ref_global", "ref_muni", "ref_knn", "ref_muni_knnres",
       "direct|Ridge|SAT", "direct|ET|SAT", "direct|ET|B1", "direct|ET|ALLNOLOC", "direct|ET|ALL", "direct|ET|LOC", "direct|ET|CLIM", "direct|CB|ALL", "direct|CB|SAT",
       "resid_muni|Ridge|CUR", "resid_muni|Ridge|SAT", "resid_muni|ET|SAT", "resid_muni|ET|ALLNOLOC", "resid_muni|ET|ALL", "resid_muni|ET|CUR", "resid_muni|CB|ALL", "resid_muni|CB|SAT", "resid_muni|ET|CLIM"]
KEY = [k for k in KEY if k in cfgs]
rows = []
for c in cfgs:
    for s in schemes:
        if (c, s) not in P:
            continue
        rows.append(pd.DataFrame(dict(cfg=c, scheme=s, id=D.ids, municipio=D.mun, y=D.y, pred_oof_mean=P[(c, s)], abs_err=np.abs(D.y - P[(c, s)]), sq_err_mean_reps=L[(c, s)])))
PE = pd.concat(rows, ignore_index=True)
PE[PE.cfg.isin(KEY)].round(5).to_csv(f"{PATH['res']}/04_errores_por_parcela_configs_clave.csv.gz", index=False, compression="gzip")
PE.round(5).to_csv(f"{PATH['oof']}/oof_promedio_reps_todas_configs.csv.gz", index=False, compression="gzip")

# --- 4. metricas por bloque / por municipio para configuraciones clave
blk = []
for s in ("SPA_KM4", "SPA_KM6", "SPA_KM8", "SPA_BAND6", "LMO"):
    sub = O[(O.scheme == s) & O.cfg.isin(KEY)]
    for (c, f), t in sub.groupby(["cfg", "fold"]):
        blk.append(dict(esquema=s, bloque=f, cfg=c, n=len(t), municipios=",".join(sorted(set(D.mun[[D.id_index[i] for i in t.id]]))), y_media=t.y.mean(), pred_media=t.pred.mean(),
                        sesgo=(t.pred - t.y).mean(), RMSE=rmse(t.y, t.pred), MAE=mae(t.y, t.pred)))
BK = pd.DataFrame(blk); BK.round(4).to_csv(f"{RES}/04_metricas_por_bloque.csv", index=False)
# por municipio bajo CV aleatoria (promedio de repeticiones) y bajo LMO
mrow = []
for c in KEY:
    for s in ("RCV5x10", "SPA_KM6", "LMO", "SPA_GRID3B05"):
        if (c, s) not in P:
            continue
        for m in D.mun_levels:
            k = D.mun == m
            if k.sum() >= 3:
                mrow.append(dict(cfg=c, esquema=s, municipio=m, n=int(k.sum()), RMSE=np.sqrt(L[(c, s)][k].mean()), sesgo=(P[(c, s)][k] - D.y[k]).mean()))
pd.DataFrame(mrow).round(4).to_csv(f"{RES}/04_metricas_por_municipio.csv", index=False)

# --- 5. estabilidad del ranking entre esquemas
from scipy.stats import spearmanr
rk = W.drop(index=[i for i in W.index if i.startswith("ref_global")], errors="ignore").rank(axis=0)
cors = pd.DataFrame({a: {b: spearmanr(W[a], W[b])[0] for b in schemes} for a in schemes})
cors.round(3).to_csv(f"{RES}/04_correlacion_ranking_entre_esquemas.csv"); print("Spearman de RMSE de configs entre esquemas\n", cors.round(2))
best = {s: W[s].drop(index=[i for i in W.index if i.startswith("ref_")]).idxmin() for s in schemes}
print("mejor config no-referencia por esquema:", best)
pd.Series(best).to_csv(f"{RES}/04_mejor_config_por_esquema.csv", header=["cfg"])
print("OK 04_ablation_analysis")
