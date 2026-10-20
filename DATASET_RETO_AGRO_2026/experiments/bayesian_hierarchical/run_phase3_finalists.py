"""Phase 3.3: rigorous RCV5x10 evaluation of finalists (4 chains x 2000 draws)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as cfg
from src.analysis import compare
from src.data import loader
from src.validation import baselines
from src.validation.cv_runner import run_cv
from src.validation.metrics import mae, rmse

FINALISTS = ["B0", "B3", "B5"]


def evaluate(selection_repeats, eval_repeats):
    mcmc = pd.read_json(cfg.CONFIGS_DIR / "mcmc_config.json", typ="series")
    out = cfg.RESULTS_DIR / "phase3"
    for config in FINALISTS:
        print(f"\n===== FINALIST {config} =====", flush=True)
        run_cv(config, mcmc_key="finalists", repeats=None, out_dir=out,
               n_samples=2000, tag="finalists", resume=True)

    frames, oofs = [], {}
    for config in FINALISTS:
        f = out / f"cv_{config}_finalists_finalists_folds.csv"
        o = out / f"cv_{config}_finalists_finalists_oof.csv"
        frames.append(pd.read_csv(f))
        oofs[config] = pd.read_csv(o)
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(out / "finalists_folds_all.csv", index=False)

    prior = baselines.load_official_prior_oof()
    prior.to_csv(out / "official_prior_oof.csv", index=False)

    def block_metrics(df):
        return {"rmse": rmse(df["y_true"], df["y_pred"]), "mae": mae(df["y_true"], df["y_pred"]), "n": len(df)}

    summary_rows = []
    for config, o in oofs.items():
        for block, reps in [("selection", selection_repeats), ("evaluation", eval_repeats), ("all", selection_repeats + eval_repeats)]:
            sub = o[o["repeat"].isin(reps)]
            summary_rows.append({"model": config, "block": block, **block_metrics(sub)})
    for block, reps in [("selection", selection_repeats), ("evaluation", eval_repeats), ("all", selection_repeats + eval_repeats)]:
        sub = prior[prior["repeat"].isin(reps)]
        summary_rows.append({"model": "PRIOR_OFICIAL", "block": block, **block_metrics(sub)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "finalists_summary.csv", index=False)
    print(summary.to_string(index=False))

    paired_rows = []
    for config in FINALISTS:
        cmp = compare.paired_against(oofs[config], prior)
        cmp["model"] = config
        cmp["vs"] = "PRIOR_OFICIAL"
        wt = compare.wilcoxon_test(oofs[config], prior)
        cmp.update(wt)
        paired_rows.append(cmp)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "finalists_paired_vs_prior.csv", index=False)
    print("\nPAIRED vs PRIOR (rmse_a=model, rmse_b=PRIOR):")
    print(paired[["model", "rmse_a", "rmse_b", "bootstrap_ci_err2_diff", "pvalue"]].to_string(index=False))

    b0 = oofs["B0"]
    env_rows = []
    for config in ["B3", "B5"]:
        cmp = compare.paired_against(oofs[config], b0)
        cmp["model"] = config
        cmp["vs"] = "B0"
        cmp.update(compare.wilcoxon_test(oofs[config], b0))
        env_rows.append(cmp)
    pd.DataFrame(env_rows).to_csv(out / "finalists_paired_env_vs_B0.csv", index=False)
    print("\nENV vs B0 (rmse_a=env, rmse_b=B0):")
    print(pd.DataFrame(env_rows)[["model", "rmse_a", "rmse_b", "bootstrap_ci_err2_diff", "pvalue"]].to_string(index=False))
    return summary


if __name__ == "__main__":
    mcmc = json.loads((cfg.CONFIGS_DIR / "mcmc_config.json").read_text())
    t0 = time.time()
    evaluate(mcmc["selection_repeats"], mcmc["evaluation_repeats"])
    print(f"\nTotal finalist time {time.time()-t0:.1f}s")
