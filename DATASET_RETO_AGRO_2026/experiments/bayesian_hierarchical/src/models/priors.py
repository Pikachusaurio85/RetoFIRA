"""Prior and feature-set configuration."""
from __future__ import annotations

import json

from src import config as cfg


def load_feature_sets() -> dict:
    return json.loads((cfg.CONFIGS_DIR / "feature_sets.json").read_text())["sets"]


def load_priors() -> dict:
    return json.loads((cfg.CONFIGS_DIR / "priors.json").read_text())


def beta_prior_sd(set_name: str) -> float:
    return float(load_priors()["beta_prior_sd"][set_name])


def load_mcmc() -> dict:
    return json.loads((cfg.CONFIGS_DIR / "mcmc_config.json").read_text())
