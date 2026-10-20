"""grid.py - definicion de la malla de configuraciones (cfg) y de los esquemas. Importado por runner y analisis."""
from common import *

MAIN_SCHEMES = ["RCV5x10", "SPA_KM4", "SPA_KM6", "SPA_KM8", "SPA_BAND6", "SPA_GRID3B05", "SPA_GRID8B1", "LMO"]
BLOO_SCHEMES = [f"BLOO_{r:g}" for r in (0, 0.5, 1, 2, 4, 8)]

SETS_DIRECT = ["LOC", "S2", "LS", "PL", "SAT", "CLIM", "TOPO", "GEOM",
               "T_PRE", "T_VEG", "T_PICO", "T_SEN", "T_PHTIM", "T_PHLEV", "T_HET",
               "CUR", "B1", "ALLNOLOC", "ALL", "ALL-SAT", "ALL-CLIM", "ALL-TOPO", "ALL-GEOM", "CLIM+TOPO+GEOM"]
SETS_RESID = SETS_DIRECT + ["LOC+CLIM", "LOC+CLIM+TOPO", "LOC+CLIM+TOPO+GEOM", "CLIM+TOPO"]
SETS_RESID_KNN = ["SAT", "ALLNOLOC", "ALL", "CUR", "CLIM"]
CB_DIRECT = ["LOC", "SAT", "CLIM", "TOPO", "GEOM", "CUR", "B1", "ALLNOLOC", "ALL", "ALL-SAT"]
CB_RESID = ["LOC", "SAT", "CLIM", "TOPO", "GEOM", "CUR", "ALLNOLOC", "ALL"]

BLOO_ET = [("direct", "LOC"), ("direct", "SAT"), ("direct", "ALL"), ("resid_muni", "SAT"), ("resid_muni", "ALLNOLOC"), ("resid_muni", "CUR")]
BLOO_RIDGE = BLOO_ET + [("direct", "CLIM"), ("resid_muni", "CLIM"), ("resid_muni", "ALL"), ("direct", "ALLNOLOC")]


def refs():
    return [dict(mode=m) for m in ("ref_global", "ref_muni", "ref_knn", "ref_muni_knnres")]


def main_configs():
    out = refs()
    for model in ("Ridge", "ET"):
        out += [dict(mode="direct", model=model, set=s) for s in SETS_DIRECT]
        out += [dict(mode="resid_muni", model=model, set=s) for s in SETS_RESID]
        out += [dict(mode="resid_knn", model=model, set=s) for s in SETS_RESID_KNN]
    out += [dict(mode="direct", model="CB", set=s) for s in CB_DIRECT]
    out += [dict(mode="resid_muni", model="CB", set=s) for s in CB_RESID]
    return out


def bloo_configs():
    out = [dict(mode=m) for m in ("ref_muni", "ref_knn", "ref_muni_knnres")]
    out += [dict(mode=m, model="ET", set=s) for m, s in BLOO_ET]
    out += [dict(mode=m, model="Ridge", set=s) for m, s in BLOO_RIDGE]
    return out


def safe(cid):
    return cid.replace("|", "__").replace("+", "p").replace("-", "m")
