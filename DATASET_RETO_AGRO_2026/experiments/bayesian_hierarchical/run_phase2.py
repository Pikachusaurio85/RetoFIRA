"""Phase 2: model tests + smoke test on one fold."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as cfg
from src.data import loader
from src.models import hierarchical as hmod
from src.models.preprocess import prepare_fold
from src.models.priors import load_feature_sets, load_mcmc
from src.validation import metrics
from src.validation.cv_runner import _diagnostics


def smoke(fold_key: str = "RCV5x10_rep0_fold0") -> dict:
    folds = json.loads((cfg.CONFIGS_DIR / f"splits_{cfg.CV_SCHEME}.json").read_text())
    f = folds[fold_key]
    master = loader.load_master()
    fs = load_feature_sets()
    mcmc = load_mcmc()["smoke"]
    out = {}
    for config in ["B0", "B5"]:
        Xtr, Xte, ytr, gtr, gte, kept, scaler = prepare_fold(
            master, fs[config], f["train"], f["test"], cfg.TARGET, cfg.ID_COL, cfg.MUNICIPIO_COL
        )
        t0 = time.time()
        idata, gidx = hmod.fit_model(
            Xtr, ytr, gtr, config, draws=mcmc["draws"], tune=mcmc["tune"],
            chains=mcmc["chains"], target_accept=mcmc.get("target_accept", 0.9), cores=2, seed=cfg.SEED,
        )
        el = time.time() - t0
        pp = hmod.posterior_predict(idata, Xte, gte, gidx, seed=cfg.SEED, n_samples=1000)
        yte = master.set_index(cfg.ID_COL).loc[f["test"], cfg.TARGET].to_numpy(dtype=float)
        m = metrics.full_metrics(pp, yte)
        diag = _diagnostics(idata)
        out[config] = {"n_test": len(f["test"]), "n_features": len(kept), "elapsed_s": round(el, 2), **m, **diag}
        print(f"[smoke {config}] {out[config]}", flush=True)
    (cfg.RESULTS_DIR / "phase2").mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "phase2" / "smoke_test.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    smoke()
