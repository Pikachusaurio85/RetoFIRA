"""Phase 1: data audit, splits reconstruction, feature sets, concept mapping."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as cfg
from src.data import audit, loader, splits
from src.data.municipality import per_municipality_yield, variance_decomposition

CANDIDATE_MAP = {
    "C1a_ndvi_s2_mean": ["s2_ndvi_pre", "s2_ndvi_veg", "s2_ndvi_pico", "s2_ndvi_sen"],
    "C2a_amplitud_ndvi": ["s2_ndvi_amp"],
    "C3a_ndwi_s2_mean": ["s2_ndwi_pre", "s2_ndwi_veg", "s2_ndwi_pico", "s2_ndwi_sen"],
    "C4a_prec_acum_abr_oct": ["clim_prec_temp"],
    "C5a_elev_media": ["topo_elev_media"],
}


def concept_mapping_table(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for concept, cols in CANDIDATE_MAP.items():
        best_col, best_r = None, -1.0
        for c in cols:
            if c in master.columns:
                r = master[[concept, c]].corr().iloc[0, 1]
                rows.append({"concept": concept, "engineered": c, "pearson_r": round(float(r), 4)})
                if r > best_r:
                    best_r, best_col = r, c
        if len(cols) > 1 and all(c in master.columns for c in cols):
            mm = master[cols].mean(axis=1)
            r = master[concept].corr(mm)
            rows.append({"concept": concept, "engineered": "mean(" + ",".join(cols) + ")", "pearson_r": round(float(r), 4)})
            if r > best_r:
                best_r, best_col = r, "mean"
        rows.append({"concept": concept, "engineered": f"BEST={best_col}", "pearson_r": round(float(best_r), 4)})
    return pd.DataFrame(rows)


def validate_feature_sets(master: pd.DataFrame) -> dict:
    fs = json.loads((cfg.CONFIGS_DIR / "feature_sets.json").read_text())
    report = {}
    for name, cols in fs["sets"].items():
        missing = [c for c in cols if c not in master.columns]
        report[name] = {"n": len(cols), "missing": missing, "ok": not missing}
    return report


def main() -> None:
    cfg.ensure_dirs()
    master = loader.load_master()
    print("Master shape:", master.shape)
    print("Partitions:", master[cfg.PARTITION_COL].value_counts().to_dict())

    summary = audit.run_and_save(master)
    per_municipality_yield(loader.train_frame(master)).to_csv(
        cfg.RESULTS_DIR / "phase1" / "per_municipality_yield.csv", index=False
    )

    folds = splits.build_folds(cfg.CV_SCHEME)
    freport = splits.validate_folds(master, folds, cfg.CV_SCHEME)
    print("Fold validation:", freport["ok"], "n_folds:", freport["n_folds"])
    if not freport["ok"]:
        for e in freport["errors"][:10]:
            print("  ERROR:", e)
        raise SystemExit("Fold validation failed")
    splits.save_folds(folds)

    concept_map = concept_mapping_table(master)
    concept_map.to_csv(cfg.RESULTS_DIR / "phase1" / "concept_mapping.csv", index=False)

    fs_report = validate_feature_sets(master)
    (cfg.RESULTS_DIR / "phase1" / "feature_sets_validation.json").write_text(
        json.dumps(fs_report, indent=2)
    )
    print("Feature set validation:", {k: v["ok"] for k, v in fs_report.items()})

    print("\nAudit summary:")
    print(json.dumps(summary, indent=2, default=float)[:2000])
    print("\nConcept mapping:")
    print(concept_map.to_string(index=False))


if __name__ == "__main__":
    main()
