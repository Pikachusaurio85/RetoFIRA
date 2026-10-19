"""
05b_cur24_control.py - Control con composicion igualada para CUR24.
CUR24 = 20 variables de satelite (14 S2 + 5 PL + 1 LS) + 2 TOPO + 2 GEOM. En 05_stability.py los sorteos "aleatorios" se hacian solo de SAT/ALLNOLOC;
aqui se sortean 300 conjuntos con la MISMA composicion (20 SAT + 2 de 4 TOPO + 2 de 3 GEOM) y se evalua igual (Ridge sobre residuo del prior municipal, RCV5x10, 3 reps).
Tambien: el mismo CUR24 sin sus 4 variables TOPO/GEOM (20 SAT) vs 20 SAT aleatorias.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from oofio import *
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = Data(); RES = PATH["res"]; rng = np.random.default_rng(GLOBAL_SEED + 5)
SPL = load_splits(D, ["RCV5x10"])


def ridge_fp(Xtr, t, Xte):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 40))).fit(Xtr, t); return np.ravel(m.predict(Xte))


def cv_fixed(cols_idx, reps=3):
    r_ = []
    for rep in range(reps):
        oof = np.zeros(D.n)
        for fold, tr, te in SPL["RCV5x10"][rep]:
            p_tr, p_te = prior_muni(D, tr, te); oof[te] = p_te + ridge_fp(D.X[tr][:, cols_idx], D.y[tr] - p_tr, D.X[te][:, cols_idx])
        r_.append(rmse(D.y, oof))
    return float(np.mean(r_))


cur = D.idx("CUR"); sat = D.idx("SAT"); topo = D.idx("TOPO"); geom = D.idx("GEOM")
cur_sat = [c for c in cur if c in set(sat)]; cur_tg = [c for c in cur if c not in set(sat)]
print("CUR24: n SAT =", len(cur_sat), "| n TOPO/GEOM =", len(cur_tg))
r_cur = cv_fixed(cur); r_cursat = cv_fixed(np.array(cur_sat))
N = 300
matched = [cv_fixed(np.concatenate([rng.choice(sat, len(cur_sat), replace=False), rng.choice(topo, 2, replace=False), rng.choice(geom, 2, replace=False)])) for _ in range(N)]
sat_only = [cv_fixed(rng.choice(sat, len(cur_sat), replace=False)) for _ in range(N)]
rows = [dict(conjunto="CUR24", RMSE=r_cur), dict(conjunto="CUR24 sin TOPO/GEOM (solo sus %d de satelite)" % len(cur_sat), RMSE=r_cursat),
        dict(conjunto=f"{N} sorteos misma composicion ({len(cur_sat)} SAT + 2 TOPO + 2 GEOM): media", RMSE=np.mean(matched)),
        dict(conjunto="   p5 / p95", RMSE=f"{np.percentile(matched,5):.4f} / {np.percentile(matched,95):.4f}"),
        dict(conjunto="   fraccion de sorteos con RMSE menor que CUR24", RMSE=float((np.array(matched) < r_cur).mean())),
        dict(conjunto=f"{N} sorteos de {len(cur_sat)} SAT solas: media", RMSE=np.mean(sat_only)),
        dict(conjunto="   p5 / p95", RMSE=f"{np.percentile(sat_only,5):.4f} / {np.percentile(sat_only,95):.4f}"),
        dict(conjunto="   fraccion de sorteos con RMSE menor que CUR24-solo-satelite", RMSE=float((np.array(sat_only) < r_cursat).mean()))]
R = pd.DataFrame(rows); R.to_csv(f"{RES}/05b_cur24_control_composicion.csv", index=False); print(R.to_string(index=False))
