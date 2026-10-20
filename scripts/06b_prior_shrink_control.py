"""
06b_prior_shrink_control.py - Control del baseline: el encogimiento del prior municipal (k=3, heredado del equipo) es un hiperparametro.
06_sensibilidad_encogimiento_prior.csv mostro que k=0.5-1 baja el RMSE de RCV de 0.5245 a ~0.513 (-0.012), del mismo tamano que las 'mejoras' de los modelos residuales.
Aqui se evalua con honestidad:
  (a) prior k=3 (referencia del equipo)
  (b) prior con k elegido DENTRO del fold (minimizando el error leave-one-out de las filas de entrenamiento; malla K_GRID)
  (c) Ridge sobre el residuo del prior k=3  vs  Ridge sobre el residuo del prior k* (k anidado)
Se compara todo con las mismas particiones RCV5x10 (10 reps), perdida por parcela = media de errores^2 sobre reps, bootstrap por clusteres espaciales.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from oofio import *
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = Data(); RES = PATH["res"]
SPL = load_splits(D, ["RCV5x10"]); B = Boot(D, B=3000)
K_GRID = [0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10]
SETS = ["TOPO", "CUR", "CLIM", "SAT", "LOC", "ALLNOLOC"]
reps = range(10)


def ridge_fp(Xtr, t, Xte):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 40))).fit(Xtr, t); return np.ravel(m.predict(Xte))


def best_k(tr):
    errs = []
    for k in K_GRID:
        p_tr, _ = prior_muni(D, tr, tr[:1], shrink=k); errs.append(np.mean((D.y[tr] - p_tr) ** 2))
    return K_GRID[int(np.argmin(errs))]


oof = {"prior_k3": [], "prior_kopt": []}; ks = []
for s in SETS:
    oof[f"res_k3|{s}"] = []; oof[f"res_kopt|{s}"] = []
for rep in reps:
    o = {k: np.zeros(D.n) for k in oof}
    for fold, tr, te in SPL["RCV5x10"][rep]:
        k = best_k(tr); ks.append(dict(rep=rep, fold=fold, k_opt=k))
        p3_tr, p3_te = prior_muni(D, tr, te, shrink=3.0); pk_tr, pk_te = prior_muni(D, tr, te, shrink=k)
        o["prior_k3"][te] = p3_te; o["prior_kopt"][te] = pk_te
        for s in SETS:
            ci = D.idx(s)
            o[f"res_k3|{s}"][te] = p3_te + ridge_fp(D.X[tr][:, ci], D.y[tr] - p3_tr, D.X[te][:, ci])
            o[f"res_kopt|{s}"][te] = pk_te + ridge_fp(D.X[tr][:, ci], D.y[tr] - pk_tr, D.X[te][:, ci])
    for k_ in oof:
        oof[k_].append(o[k_])
    print("rep", rep, flush=True)
L = {k: np.mean([(D.y - o) ** 2 for o in v], 0) for k, v in oof.items()}
Rm = {k: np.mean([rmse(D.y, o) for o in v]) for k, v in oof.items()}
Rsd = {k: np.std([rmse(D.y, o) for o in v]) for k, v in oof.items()}
pd.DataFrame(ks).to_csv(f"{RES}/06b_k_elegido_por_fold.csv", index=False)
print("k elegido (frecuencia):", pd.DataFrame(ks).k_opt.value_counts().sort_index().to_dict())
rows = []
def add(a, b, name):
    r = B.paired(L[a], L[b]); rows.append(dict(contraste=name, A=a, B=b, RMSE_A=Rm[a], RMSE_B=Rm[b], dRMSE=r["dRMSE"], ic_lo_cluster=r["ci_lo_clu"], ic_hi_cluster=r["ci_hi_clu"], p_cluster=r["p_clu"]))
add("prior_kopt", "prior_k3", "prior con k anidado vs prior k=3")
for s in SETS:
    add(f"res_k3|{s}", "prior_k3", f"residual Ridge[{s}] sobre prior k=3 vs prior k=3")
    add(f"res_k3|{s}", "prior_kopt", f"residual Ridge[{s}] sobre prior k=3 vs prior k anidado")
    add(f"res_kopt|{s}", "prior_kopt", f"residual Ridge[{s}] sobre prior k anidado vs prior k anidado")
R = pd.DataFrame(rows); R["p_holm_cluster"] = holm(R.p_cluster.values)
R.round(4).to_csv(f"{RES}/06b_prior_shrink_control.csv", index=False)
print(pd.DataFrame(dict(RMSE=Rm, sd_reps=Rsd)).round(4).to_string()); print(R.round(4).to_string(index=False))
