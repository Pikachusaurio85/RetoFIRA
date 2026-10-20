"""oofio.py - carga de predicciones OOF y utilidades de inferencia pareada (bootstrap por parcela y por cluster espacial)."""
import glob
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from common import *
from grid import *

ROPE = 0.02          # t/ha: diferencia de RMSE por debajo de la cual no se declara mejora (nota del README del equipo: 0.01-0.02 es ruido)
N_CLUSTERS = 25      # micro-bloques espaciales para el bootstrap por cluster (conservador ante autocorrelacion)


def load_oof(group="main"):
    fr = [pd.read_csv(f) for f in sorted(glob.glob(f"{PATH['oof']}/{group}/*.csv.gz"))]
    return pd.concat(fr, ignore_index=True)


def per_parcel(O, D):
    """Devuelve dicts {(cfg,scheme): (loss[138], pred[138])} promediando sobre repeticiones."""
    O = O.assign(se=(O.y - O.pred) ** 2)
    g = O.groupby(["cfg", "scheme", "id"]).agg(se=("se", "mean"), pred=("pred", "mean")).reset_index()
    L, P = {}, {}
    for (c, s), t in g.groupby(["cfg", "scheme"]):
        t = t.set_index("id").reindex(D.ids)
        assert t.se.notna().all(), (c, s)
        L[(c, s)] = t.se.to_numpy(); P[(c, s)] = t.pred.to_numpy()
    return L, P


def rep_metrics(O):
    rows = []
    for (c, s, r), t in O.groupby(["cfg", "scheme", "rep"]):
        rows.append(dict(cfg=c, scheme=s, rep=r, RMSE=rmse(t.y, t.pred), MAE=mae(t.y, t.pred), R2=r2(t.y, t.pred), n=len(t)))
    R = pd.DataFrame(rows)
    return (R.groupby(["cfg", "scheme"]).agg(RMSE=("RMSE", "mean"), RMSE_sd_reps=("RMSE", "std"), MAE=("MAE", "mean"), R2=("R2", "mean"), n_reps=("rep", "size")).reset_index())


def micro_clusters(D, k=N_CLUSTERS):
    return KMeans(k, n_init=10, random_state=0).fit_predict(D.XY)


class Boot:
    def __init__(self, D, B=5000, seed=GLOBAL_SEED):
        rng = np.random.default_rng(seed)
        self.cl = micro_clusters(D); self.B = B; n = D.n; C = self.cl.max() + 1
        cc = rng.multinomial(C, np.ones(C) / C, size=B)              # conteos de clusters remuestreados
        self.Wc = cc[:, self.cl].astype(float)                          # B x n pesos (cluster bootstrap)
        self.Wp = rng.multinomial(n, np.ones(n) / n, size=B).astype(float)   # pesos (parcel bootstrap)

    def delta(self, LA, LB, W):
        ok = W.sum(1) > 0
        ra = np.sqrt((W * LA).sum(1) / W.sum(1)); rb = np.sqrt((W * LB).sum(1) / W.sum(1))
        return (ra - rb)[ok]

    def paired(self, LA, LB):
        """Delta = RMSE(A) - RMSE(B) (negativo => A mejor). IC95 por parcel y por cluster; p bootstrap; Wilcoxon pareado sobre errores cuadraticos."""
        d0 = np.sqrt(LA.mean()) - np.sqrt(LB.mean())
        out = dict(RMSE_A=np.sqrt(LA.mean()), RMSE_B=np.sqrt(LB.mean()), dRMSE=d0, dMSE=LA.mean() - LB.mean())
        for tag, W in (("par", self.Wp), ("clu", self.Wc)):
            d = self.delta(LA, LB, W)
            out[f"ci_lo_{tag}"], out[f"ci_hi_{tag}"] = np.percentile(d, [2.5, 97.5])
            out[f"p_{tag}"] = min(1.0, 2 * min((1 + (d <= 0).sum()) / (1 + len(d)), (1 + (d >= 0).sum()) / (1 + len(d))))
        diff = LA - LB
        try:
            out["p_wilcoxon_se"] = wilcoxon(diff, zero_method="wilcox").pvalue if np.any(diff != 0) else 1.0
        except Exception:
            out["p_wilcoxon_se"] = np.nan
        out["frac_parcelas_A_mejor"] = float((LA < LB).mean())
        return out


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def decide(row, rope=ROPE):
    lo, hi, d, padj = row["ci_lo_clu"], row["ci_hi_clu"], row["dRMSE"], row["p_clu_holm"]
    if hi < 0 and abs(d) >= rope and padj < 0.05:
        return "A MEJOR (relevante)"
    if lo > 0 and abs(d) >= rope and padj < 0.05:
        return "A PEOR (relevante)"
    if lo > -rope and hi < rope:
        return "EQUIVALENTE (+-0.02)"
    return "INDETERMINADO"
