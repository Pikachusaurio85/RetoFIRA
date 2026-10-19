"""
common.py - utilidades compartidas de los experimentos AgroCebada (NO modifica el pipeline original).

Principios:
  * Solo se leen las 138 parcelas ENTRENAMIENTO con etiqueta. Las 59 PREDICCION se usan unicamente
    (sin etiqueta) en 00_audit.py para validacion adversaria features-solo.
  * Todo lo que usa el rendimiento de otras parcelas (prior municipal, kNN espacial, seleccion de variables,
    escalado, imputacion, alfa de Ridge) se ajusta DENTRO de cada fold, con leave-self-out en entrenamiento.
  * Las particiones se generan una vez (01_make_splits.py), se guardan en splits/ y todas las configuraciones las reutilizan.
"""
import os, re, json, hashlib, warnings
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

EXP = "/home/claude/experiments"
PATH = dict(data=f"{EXP}/data", splits=f"{EXP}/splits", oof=f"{EXP}/oof", res=f"{EXP}/results",
            fig=f"{EXP}/figures", cfg=f"{EXP}/config", log=f"{EXP}/logs", rep=f"{EXP}/report")
for _p in PATH.values():
    os.makedirs(_p, exist_ok=True)

GLOBAL_SEED = 20261013
META = ["CONJUNTO", "RENDIMIENTO_T_HA", "loc_municipio", "loc_estado", "loc_celda_clima"]
SHRINK = 3.0      # encogimiento del prior municipal (igual que el baseline original)
KNN_K = 5         # vecinos del prior espacial (igual que el baseline original)

# ----------------------------------------------------------------------------------------------
# Datos y familias de variables
# ----------------------------------------------------------------------------------------------
def load_all_features():
    """Tabla completa (197 parcelas). SOLO para analisis sin etiqueta (validacion adversaria)."""
    return pd.read_csv(f"{PATH['data']}/features_parcela.csv", index_col=0)


def load_labeled():
    d = load_all_features()
    tr = d[d.CONJUNTO == "ENTRENAMIENTO"].copy()
    assert len(tr) == 138 and tr.RENDIMIENTO_T_HA.notna().all()
    return tr


CONCEPT = {  # familia conceptual del indice espectral (para SHAP por concepto)
    "vigor": ["ndvi", "evi", "lai", "fapar", "savi", "msavi"],
    "residuo_labranza": ["crc", "mcrc", "ndsvi", "srndi", "mra", "ndti", "sti"],
    "agua_estres": ["ndwi", "endwi", "msi", "emsi", "dswi2", "dswi3", "dswi4", "dswi5", "nddi", "enddi"],
    "termico": ["vi6t"],
}
IDX2CONCEPT = {i: k for k, v in CONCEPT.items() for i in v}

CURATED24 = ['s2_ndvi_peak', 's2_ndvi_auc', 's2_ndvi_los', 's2_ndvi_sos_doy', 's2_ndvi_base', 's2_ndvi_pico', 's2_evi_base',
             's2_evi_pico', 's2_lai_pico', 's2_msi_pico', 's2_ndwi_pico', 's2_ndti_pre', 's2_crc_pre', 's2_nddi_pico',
             'pl_ndvi_peak', 'pl_ndvi_auc', 'pl_ndvi_los', 'pl_evi_pico', 'pl_lai_pico', 'ls_vi6t_pico',
             'topo_elev_media', 'topo_pend_media', 'geo_area_ha', 'geo_compacidad']   # lista del baseline original (09_experiments_residual.py)


def feature_families(columns):
    cols = [c for c in columns if c not in META]
    F = {}
    F["S2"] = [c for c in cols if c.startswith("s2_")]
    F["LS"] = [c for c in cols if c.startswith("ls_")]
    F["PL"] = [c for c in cols if c.startswith("pl_")]
    F["SAT"] = F["S2"] + F["LS"] + F["PL"]
    F["CLIM"] = [c for c in cols if c.startswith("clim_")]
    F["TOPO"] = [c for c in cols if c.startswith("topo_")]
    F["GEOM"] = [c for c in cols if c.startswith("geo_")]
    F["LOC"] = ["loc_x_utm", "loc_y_utm"]                     # lon/lat son colineales con x/y UTM
    sat = F["SAT"]
    het = [c for c in sat if re.search(r"_(std|min|max)_pico$", c)]
    tim = [c for c in sat if re.search(r"_(peak_doy|sos_doy|eos_doy|los)$", c)]
    lev = [c for c in sat if re.search(r"_(base|peak|amp|auc|slope_up|slope_down)$", c)]
    win = {w: [c for c in sat if c.endswith("_" + w) and c not in het] for w in ["pre", "veg", "pico", "sen"]}
    F["T_PRE"], F["T_VEG"], F["T_PICO"], F["T_SEN"] = win["pre"], win["veg"], win["pico"], win["sen"]
    F["T_PHTIM"], F["T_PHLEV"], F["T_HET"] = tim, lev, het
    parts = sum([F[k] for k in ["T_PRE", "T_VEG", "T_PICO", "T_SEN", "T_PHTIM", "T_PHLEV", "T_HET"]], [])
    assert sorted(parts) == sorted(sat), "las familias temporales deben particionar SAT"
    F["CUR"] = CURATED24
    F["ALLNOLOC"] = F["SAT"] + F["CLIM"] + F["TOPO"] + F["GEOM"]
    F["B1"] = F["SAT"] + F["TOPO"] + F["GEOM"]                 # Baseline 1 del README: satelite+topo+geom (sin ubicacion ni clima)
    F["ALL"] = F["ALLNOLOC"] + F["LOC"]
    F["LOC+CLIM"] = F["LOC"] + F["CLIM"]
    F["LOC+CLIM+TOPO"] = F["LOC"] + F["CLIM"] + F["TOPO"]
    F["LOC+CLIM+TOPO+GEOM"] = F["LOC"] + F["CLIM"] + F["TOPO"] + F["GEOM"]
    F["CLIM+TOPO"] = F["CLIM"] + F["TOPO"]
    F["CLIM+TOPO+GEOM"] = F["CLIM"] + F["TOPO"] + F["GEOM"]
    F["ALL-LOC"] = F["ALLNOLOC"]
    F["ALL-SAT"] = F["LOC"] + F["CLIM"] + F["TOPO"] + F["GEOM"]
    F["ALL-CLIM"] = F["SAT"] + F["TOPO"] + F["GEOM"] + F["LOC"]
    F["ALL-TOPO"] = F["SAT"] + F["CLIM"] + F["GEOM"] + F["LOC"]
    F["ALL-GEOM"] = F["SAT"] + F["CLIM"] + F["TOPO"] + F["LOC"]
    F["SAT+TOPO+GEOM+CLIM"] = F["ALLNOLOC"]
    return F


def sat_concept_groups(columns):
    """Grupos por concepto del indice (vigor / residuo-labranza / agua-estres / termico) y sensor."""
    out = {}
    for c in columns:
        m = re.match(r"^(s2|ls|pl)_([a-z0-9]+)_", c)
        if not m:
            continue
        sensor, idx = m.groups()
        con = IDX2CONCEPT.get(idx)
        if con is None:
            continue
        out.setdefault(f"{sensor.upper()}:{con}", []).append(c)
    return out


class Data:
    """Contenedor de las 138 parcelas etiquetadas, matriz X y codigos auxiliares."""
    def __init__(self):
        self.tr = load_labeled()
        self.ids = self.tr.index.to_numpy()
        self.n = len(self.tr)
        self.y = self.tr.RENDIMIENTO_T_HA.to_numpy(float)
        self.mun = self.tr.loc_municipio.to_numpy()
        self.mun_code, self.mun_levels = pd.factorize(self.tr.loc_municipio)
        self.K = len(self.mun_levels)
        self.cell = self.tr.loc_celda_clima.to_numpy()
        self.XY = self.tr[["loc_x_utm", "loc_y_utm"]].to_numpy(float)
        self.F = feature_families(self.tr.columns)
        self.cols = [c for c in self.tr.columns if c not in META]
        self.X = self.tr[self.cols].to_numpy(float)
        assert np.isfinite(self.X).all(), "se esperaban 0 faltantes en las 138 etiquetadas"
        self.col_index = {c: i for i, c in enumerate(self.cols)}
        self.id_index = {i: k for k, i in enumerate(self.ids)}

    def idx(self, setname):
        return np.array([self.col_index[c] for c in self.F[setname]], dtype=int)


# ----------------------------------------------------------------------------------------------
# Priors (siempre ajustados con el entrenamiento del fold; leave-self-out para las filas de train)
# ----------------------------------------------------------------------------------------------
def prior_muni(D, tr_idx, te_idx, shrink=SHRINK):
    y, m = D.y[tr_idx], D.mun_code[tr_idx]
    N, T = len(y), y.sum()
    S = np.bincount(m, weights=y, minlength=D.K); C = np.bincount(m, minlength=D.K)
    gm = T / N
    p_te = (S[D.mun_code[te_idx]] + shrink * gm) / (C[D.mun_code[te_idx]] + shrink)
    gm_loo = (T - y) / (N - 1)
    p_tr = (S[m] - y + shrink * gm_loo) / (C[m] - 1 + shrink)
    return p_tr, p_te


def _idw(vals, dist):
    w = 1.0 / np.maximum(dist, 1.0)
    return (vals * w).sum(1) / w.sum(1)


def prior_knn(D, tr_idx, te_idx, k=KNN_K, values=None):
    """Promedio ponderado 1/d del rendimiento (o de `values` indexado por tr_idx) de los k vecinos del TRAIN."""
    v = D.y[tr_idx] if values is None else values
    t = cKDTree(D.XY[tr_idx])
    dtr, itr = t.query(D.XY[tr_idx], k=k + 1)
    dte, ite = t.query(D.XY[te_idx], k=k)
    return _idw(v[itr[:, 1:]], dtr[:, 1:]), _idw(v[ite], dte)


# ----------------------------------------------------------------------------------------------
# Modelos (hiperparametros FIJOS; no se ajustan con los folds externos)
# ----------------------------------------------------------------------------------------------
def make_model(name, seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import ExtraTreesRegressor
    if name == "Ridge":     # escalado + alfa por GCV/LOO interno: todo dentro del fold
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 40)))
    if name == "ET":
        return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5, n_jobs=1, random_state=seed)
    if name == "CB":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=5.0, rsm=0.5, random_seed=seed,
                                 verbose=0, thread_count=1, allow_writing_files=False)
    raise ValueError(name)


def fit_predict(D, cfg, tr_idx, te_idx, seed, return_model=False):
    """cfg = dict(mode, model, set). Devuelve prediccion para te_idx."""
    mode = cfg["mode"]
    if mode == "ref_global":
        return np.full(len(te_idx), D.y[tr_idx].mean())
    if mode == "ref_muni":
        return prior_muni(D, tr_idx, te_idx)[1]
    if mode == "ref_knn":
        return prior_knn(D, tr_idx, te_idx)[1]
    if mode == "ref_muni_knnres":      # prior municipal + suavizado espacial del residuo (sin satelite)
        p_tr, p_te = prior_muni(D, tr_idx, te_idx)
        r_tr = D.y[tr_idx] - p_tr
        return p_te + prior_knn(D, tr_idx, te_idx, values=r_tr)[1]
    ci = D.idx(cfg["set"])
    Xtr, Xte = D.X[tr_idx][:, ci], D.X[te_idx][:, ci]
    if mode == "direct":
        target, off = D.y[tr_idx], 0.0
    elif mode == "resid_muni":
        p_tr, off = prior_muni(D, tr_idx, te_idx); target = D.y[tr_idx] - p_tr
    elif mode == "resid_knn":
        p_tr, off = prior_knn(D, tr_idx, te_idx); target = D.y[tr_idx] - p_tr
    else:
        raise ValueError(mode)
    m = make_model(cfg["model"], seed)
    m.fit(Xtr, target)
    pred = off + np.ravel(m.predict(Xte))
    return (pred, m) if return_model else pred


def cfg_id(cfg):
    if cfg["mode"].startswith("ref_"):
        return cfg["mode"]
    return f"{cfg['mode']}|{cfg['model']}|{cfg['set']}"


# ----------------------------------------------------------------------------------------------
# Particiones
# ----------------------------------------------------------------------------------------------
def load_splits(D, schemes=None):
    """Devuelve {scheme: {rep: [(fold, tr_idx, te_idx), ...]}} a partir de splits/splits_long.csv.
    tr_idx se reconstruye como todo lo que no es test, menos el buffer (columna role == 'buffer')."""
    s = pd.read_csv(f"{PATH['splits']}/splits_long.csv")
    if schemes is not None:
        s = s[s.scheme.isin(schemes)]
    out = {}
    for (sch, rep, fold), g in s.groupby(["scheme", "rep", "fold"], sort=True):
        te = g.loc[g.role == "test", "id"].map(D.id_index).to_numpy()
        buf = g.loc[g.role == "buffer", "id"].map(D.id_index).to_numpy()
        tr = np.setdiff1d(np.arange(D.n), np.concatenate([te, buf]))
        out.setdefault(sch, {}).setdefault(rep, []).append((fold, tr, te))
    return out


# ----------------------------------------------------------------------------------------------
# Metricas
# ----------------------------------------------------------------------------------------------
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
