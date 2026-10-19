"""spatial.py - funciones de Moran (copiadas tal cual de 02_variance_spatial.py para reutilizarlas en 02b)."""
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist

def W_knn(XY, k, mask=None):
    m = len(XY); W = np.zeros((m, m)); Dm = cdist(XY, XY); np.fill_diagonal(Dm, np.inf)
    if mask is not None:
        Dm = np.where(mask, Dm, np.inf)
    for i in range(m):
        o = np.argsort(Dm[i])[:k]; o = o[np.isfinite(Dm[i, o])]
        if len(o):
            W[i, o] = 1.0 / len(o)
    return W


def W_band(XY, lo, hi, mask=None):
    Dm = cdist(XY, XY); A = (Dm > lo) & (Dm <= hi) if lo > 0 else (Dm <= hi) & (Dm > 0)
    if mask is not None:
        A &= mask
    W = A.astype(float); rs = W.sum(1); W[rs > 0] /= rs[rs > 0][:, None]
    return W


def moran(z, W):
    z = z - z.mean(); S0 = W.sum()
    return (len(z) / S0) * (z @ W @ z) / (z @ z)


def moran_perm(z, W, groups=None, nperm=9999, seed=0):
    """I observada y p por permutacion. Si groups: z se centra por grupo y se permuta DENTRO de grupo (W debe ser intra-grupo)."""
    r = np.random.default_rng(seed); z = np.asarray(z, float).copy()
    if groups is not None:
        z = z - pd.Series(z).groupby(np.asarray(groups)).transform("mean").values
    else:
        z = z - z.mean()
    S0 = W.sum(); den = z @ z
    I = (len(z) / S0) * (z @ W @ z) / den
    Z = np.empty((nperm, len(z)))
    if groups is None:
        for b in range(nperm):
            Z[b] = z[r.permutation(len(z))]
    else:
        gi = [np.where(np.asarray(groups) == u)[0] for u in np.unique(groups)]
        for b in range(nperm):
            zz = z.copy()
            for ix in gi:
                zz[ix] = z[r.permutation(ix)]
            Z[b] = zz
    num = np.einsum("pi,ij,pj->p", Z, W, Z); Ip = (len(z) / S0) * num / den
    return dict(I=I, E_I_perm=Ip.mean(), sd_perm=Ip.std(), z=(I - Ip.mean()) / Ip.std(), p_one_sided=(1 + (Ip >= I).sum()) / (1 + nperm),
                p_two_sided=(1 + (np.abs(Ip - Ip.mean()) >= abs(I - Ip.mean())).sum()) / (1 + nperm))


