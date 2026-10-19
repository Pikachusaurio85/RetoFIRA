"""
03_run_cv_grid.py - Ejecuta la malla de configuraciones sobre las particiones FIJAS (splits/splits_long.csv).
Guarda predicciones fuera de muestra por (esquema, repeticion, fold, parcela) en oof/<grupo>/<cfg>.csv.gz. Reanudable.
Uso: python 03_run_cv_grid.py --group main|bloo [--models Ridge ET CB]
"""
import argparse, time, sys
from joblib import Parallel, delayed
from grid import *

ap = argparse.ArgumentParser()
ap.add_argument("--group", default="main")
ap.add_argument("--models", nargs="*", default=None)
ap.add_argument("--n_jobs", type=int, default=2)
a = ap.parse_args()

D = Data()
schemes = MAIN_SCHEMES if a.group == "main" else BLOO_SCHEMES
cfgs = main_configs() if a.group == "main" else bloo_configs()
if a.models:
    cfgs = [c for c in cfgs if c.get("model", "ref") in a.models or (c["mode"].startswith("ref_") and "ref" in a.models)]
outdir = f"{PATH['oof']}/{a.group}"; os.makedirs(outdir, exist_ok=True)
SPL = load_splits(D, schemes)


def run_one(cfg):
    cid = cfg_id(cfg); fn = f"{outdir}/{safe(cid)}.csv.gz"
    if os.path.exists(fn):
        return cid, 0.0, "skip"
    t0 = time.time(); recs = []
    for sch in schemes:
        for rep, folds in SPL[sch].items():
            for fold, tr, te in folds:
                seed = 1000 * int(rep) + int(fold)             # misma semilla para todas las configuraciones => comparacion pareada
                p = fit_predict(D, cfg, tr, te, seed)
                recs.append(pd.DataFrame(dict(scheme=sch, rep=rep, fold=fold, id=D.ids[te], y=D.y[te], pred=p, n_train=len(tr))))
    out = pd.concat(recs, ignore_index=True); out.insert(0, "cfg", cid)
    out.to_csv(fn + ".tmp", index=False, compression="gzip"); os.replace(fn + ".tmp", fn)
    return cid, time.time() - t0, "ok"


order = {"ref": 0, "Ridge": 1, "ET": 2, "CB": 3}
cfgs = sorted(cfgs, key=lambda c: order[c.get("model", "ref")])
print(f"[{a.group}] {len(cfgs)} configuraciones x {len(schemes)} esquemas", flush=True)
t0 = time.time()
for cid, dt, st in Parallel(n_jobs=a.n_jobs, return_as="generator")(delayed(run_one)(c) for c in cfgs):
    print(f"{time.strftime('%H:%M:%S')} {st:4s} {dt:7.1f}s {cid}", flush=True)
print("TOTAL", round(time.time() - t0), "s", flush=True)
