"""
01_make_splits.py - Genera UNA SOLA VEZ todas las particiones y las guarda en splits/splits_long.csv.
Todos los experimentos posteriores (ablaciones, estabilidad, SHAP, residual...) leen este archivo => mismas particiones.

Esquemas
  RCV5x10        CV aleatoria 5-fold x 10 repeticiones (semillas 1000..1009). Torre de marfil: vecinos a ~400 m en train.
  SPA_KM4/6/8    Bloques espaciales k-means sobre UTM (k=4,6,8), un bloque por fold (SPA_KM6 = esquema del baseline original).
  SPA_BAND6      6 bandas de igual conteo sobre el eje principal (PCA) de los centroides (esquema del comparador del equipo).
  SPA_GRID3B05   Bloques cuadrados 3 km con desplazamiento aleatorio, 6 folds, buffer de 0.5 km (10 repeticiones).
  SPA_GRID8B1    Bloques cuadrados 8 km, 6 folds, buffer de 1 km (10 repeticiones).
  LMO            Leave-one-municipality-out (9 folds): transferencia a un municipio nunca visto.
  BLOO_<r>       Leave-one-out con exclusion (buffer) de r km alrededor de cada parcela: r = 0, .5, 1, 2, 4, 8.
  TEMPORAL       NO EJECUTABLE: solo hay etiquetas de un ciclo (ver informe).
"""
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from common import *

D = Data()
n, XY, ids = D.n, D.XY, D.ids
rows = []   # (scheme, rep, fold, role, id)


def add(scheme, rep, fold, te, buf=()):
    for i in te:
        rows.append((scheme, rep, fold, "test", ids[i]))
    for i in buf:
        rows.append((scheme, rep, fold, "buffer", ids[i]))


# --- CV aleatoria
for rep in range(10):
    for f, (tri, tei) in enumerate(KFold(5, shuffle=True, random_state=1000 + rep).split(np.arange(n))):
        add("RCV5x10", rep, f, tei)

# --- bloques k-means
for k in (4, 6, 8):
    blk = KMeans(k, n_init=10, random_state=0).fit_predict(XY)
    for f in range(k):
        add(f"SPA_KM{k}", 0, f, np.where(blk == f)[0])

# --- bandas sobre eje principal
c = XY - XY.mean(0); u, s, vt = np.linalg.svd(c, full_matrices=False); pc1 = c @ vt[0]
order = np.argsort(pc1); bands = np.array_split(order, 6)
for f, b in enumerate(bands):
    add("SPA_BAND6", 0, f, b)

# --- grids con buffer
def grid_scheme(name, side, buffer, nfold=6, nrep=10):
    tree = cKDTree(XY)
    for rep in range(nrep):
        rng = np.random.default_rng(2000 + rep + int(side))
        ox, oy = rng.uniform(0, side, 2)
        bx = np.floor((XY[:, 0] - XY[:, 0].min() + ox) / side).astype(int)
        by = np.floor((XY[:, 1] - XY[:, 1].min() + oy) / side).astype(int)
        key = pd.Series(list(zip(bx, by)))
        codes, uniq = pd.factorize(key)
        nb = len(uniq)
        F = min(nfold, nb)
        perm = rng.permutation(nb)
        sizes = np.bincount(codes, minlength=nb)
        fold_of_block = np.zeros(nb, int); load = np.zeros(F)
        for b in perm[np.argsort(-sizes[perm], kind="stable")]:     # greedy: bloques grandes primero al fold mas vacio
            f = int(np.argmin(load)); fold_of_block[b] = f; load[f] += sizes[b]
        fold_of_par = fold_of_block[codes]
        for f in range(F):
            te = np.where(fold_of_par == f)[0]
            if len(te) == 0:
                continue
            near = set()
            for j in te:
                near.update(tree.query_ball_point(XY[j], buffer))
            buf = np.array(sorted(near - set(te.tolist())), dtype=int)
            add(name, rep, f, te, buf)

grid_scheme("SPA_GRID3B05", 3000.0, 500.0)
grid_scheme("SPA_GRID8B1", 8000.0, 1000.0)

# --- leave-one-municipality-out
for f, m in enumerate(D.mun_levels):
    add("LMO", 0, f, np.where(D.mun_code == f)[0])

# --- LOO con buffer
tree = cKDTree(XY)
for r_km in (0, 0.5, 1, 2, 4, 8):
    for i in range(n):
        near = tree.query_ball_point(XY[i], r_km * 1000.0) if r_km > 0 else []
        buf = [j for j in near if j != i]
        add(f"BLOO_{r_km:g}", 0, i, [i], buf)

S = pd.DataFrame(rows, columns=["scheme", "rep", "fold", "role", "id"])
S.to_csv(f"{PATH['splits']}/splits_long.csv", index=False)
print("filas", len(S), "| sha256", sha256_file(f"{PATH['splits']}/splits_long.csv")[:16])

# --- descriptores de cada esquema (que tan "dificil/independiente" es el test respecto al train)
splits = load_splits(D)
tree_all = None
desc = []
for sch, reps in splits.items():
    nn, ntr, nte, nf = [], [], [], 0
    for rep, folds in reps.items():
        for fold, tr, te in folds:
            nf += 1
            t = cKDTree(XY[tr]); d, _ = t.query(XY[te], k=1)
            nn.extend(d.tolist()); ntr.append(len(tr)); nte.append(len(te))
    nn = np.array(nn)
    desc.append(dict(scheme=sch, n_reps=len(reps), n_folds_total=nf, test_por_fold_med=np.median(nte), train_por_fold_med=np.median(ntr),
                     nn_test_a_train_m_mediana=np.median(nn), nn_p10=np.percentile(nn, 10), nn_p90=np.percentile(nn, 90),
                     frac_test_con_vecino_train_lt500m=np.mean(nn < 500)))
desc = pd.DataFrame(desc)
# referencia: distancia del conjunto oculto (59, SIN etiquetas) a las 138 etiquetadas
allf = load_all_features(); hid = allf[allf.CONJUNTO == "PREDICCION"][["loc_x_utm", "loc_y_utm"]].to_numpy()
dh, _ = cKDTree(XY).query(hid, k=1)
desc.loc[len(desc)] = dict(scheme="[REFERENCIA] 59 ocultas -> 138 etiquetadas", n_reps=1, n_folds_total=1, test_por_fold_med=59, train_por_fold_med=138,
                           nn_test_a_train_m_mediana=np.median(dh), nn_p10=np.percentile(dh, 10), nn_p90=np.percentile(dh, 90),
                           frac_test_con_vecino_train_lt500m=np.mean(dh < 500))
desc.round(3).to_csv(f"{PATH['res']}/01_descriptor_esquemas_validacion.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(desc.round(2).to_string(index=False))
json.dump(dict(GLOBAL_SEED=GLOBAL_SEED, RCV_seeds=list(range(1000, 1010)), grid_seeds="2000+rep+side", kmeans="n_init=10, random_state=0",
               shrink=SHRINK, knn_k=KNN_K), open(f"{PATH['cfg']}/splits_config.json", "w"), indent=1)
