"""Bayesian hierarchical model: partial-pooling municipality means plus covariate effects."""
from __future__ import annotations

import numpy as np
import pymc as pm

from src.models.priors import load_priors


def _encode_groups(groups_train: np.ndarray, groups_test: np.ndarray):
    uniq = {g: i for i, g in enumerate(sorted(set(groups_train.tolist())))}
    idx_train = np.array([uniq[g] for g in groups_train])
    unknown = np.array([g not in uniq for g in groups_test])
    idx_test = np.array([uniq.get(g, -1) for g in groups_test])
    return uniq, groups_train.shape[0] and len(uniq), groups_train, groups_test


def standardize_fit(X_train: np.ndarray):
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(ddof=0, a=X_train, axis=0)
    sd = np.where((sd == 0) | np.isnan(sd), 1.0, sd)
    return mu, sd


def apply_scaler(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def _impute(train: np.ndarray, test: np.ndarray):
    med = np.nanmedian(train, axis=0)
    tr = np.where(np.isnan(train), med, train)
    te = np.where(np.isnan(test), med, test)
    return tr, te


def fit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    config: str,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float = 0.9,
    cores: int = 2,
    seed: int = 0,
):
    """Fit the hierarchical model. Returns (idata, group_index_map)."""
    priors = load_priors()
    labels = sorted(set(groups_train))
    idx = {g: i for i, g in enumerate(labels)}
    gidx = np.array([idx[g] for g in groups_train], dtype="int64")
    n_groups = len(labels)
    n_obs, n_feat = X_train.shape
    beta_sd = float(priors["beta_prior_sd"][config])

    coords = {"obs": np.arange(n_obs), "group": labels, "feature": np.arange(n_feat)}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=priors["alpha_mu"], sigma=priors["alpha_sigma"])
        sigma = pm.HalfNormal("sigma", sigma=priors["sigma_scale"])
        tau = pm.HalfNormal("tau", sigma=priors["tau_scale"])
        z = pm.Normal("z", 0.0, 1.0, dims="group")
        u = pm.Deterministic("u", tau * z, dims="group")
        if n_feat > 0:
            beta = pm.Normal("beta", 0.0, beta_sd, dims="feature")
            lin = alpha + u[gidx] + pm.math.dot(X_train, beta)
        else:
            beta = None
            lin = alpha + u[gidx]
        pm.Normal("y", mu=lin, sigma=sigma, observed=y_train, dims="obs")
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(cores, chains),
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
        )
    return idata, idx


def posterior_predict(
    idata,
    X_test: np.ndarray,
    groups_test: np.ndarray,
    group_index: dict,
    seed: int = 0,
    n_samples: int = 2000,
):
    """Posterior predictive draws for test parcels of shape (n_samples, n_test).

    Unseen municipalities get u_new = tau * z_new with z_new ~ Normal(0, 1) drawn per sample,
    propagating the between-municipality uncertainty.
    """
    post = idata.posterior
    alpha = post["alpha"].values.reshape(-1)
    sigma = post["sigma"].values.reshape(-1)
    tau = post["tau"].values.reshape(-1)
    u = post["u"].values.reshape(-1, post["u"].shape[-1])
    beta = post["beta"].values.reshape(-1, post["beta"].shape[-1]) if "beta" in post else None

    n_draws = alpha.shape[0]
    base = alpha[:, None] + (X_test @ beta.T).T if beta is not None else np.repeat(alpha[:, None], X_test.shape[0], axis=1)

    rng = np.random.default_rng(seed)
    mu = base.copy()
    for j, g in enumerate(groups_test):
        if g in group_index:
            mu[:, j] = base[:, j] + u[:, group_index[g]]
        else:
            mu[:, j] = base[:, j] + tau * rng.standard_normal(n_draws)

    if n_samples != n_draws:
        sel = rng.choice(n_draws, size=n_samples, replace=n_draws < n_samples)
        mu = mu[sel]
        sigma = sigma[sel]
    y = rng.normal(mu, sigma[:, None])
    return y
