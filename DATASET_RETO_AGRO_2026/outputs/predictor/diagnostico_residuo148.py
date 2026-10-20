"""Diagnostico post hoc (NO candidato): PRIOR calibrado + Ridge(148) sobre el residuo leave-self-out.

Motivo: la permutacion intra-municipio de la Fase 2 indica que RIDGE148 usa variacion satelital dentro
del municipio. Este diagnostico pregunta si esa senal se traduce en mejora sobre el prior calibrado.
Se definio DESPUES de ver resultados; no es elegible para adopcion bajo la regla preregistrada.

Uso:  python diagnostico_residuo148.py [--base ...] [--out ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import comun as C
import fase1_comparacion as F


class PriorMasRidge148Residuo:
    nombre = "DIAG_PRIOR_MAS_RIDGE148_RESIDUO"

    def __init__(self, cols):
        self.cols = list(cols)

    def fit(self, d):
        self.prior_ = C.ModeloPrior().fit(d)
        mun = d[C.MUN].astype(str).to_numpy(); y = d[C.TARGET].to_numpy(float)
        self.pipe_ = C.pipeline_ridge().fit(d[self.cols].to_numpy(float), y - C.prior_loo(mun, y, self.prior_.k_))
        return self

    def predict(self, d):
        return self.prior_.predict(d) + self.pipe_.predict(d[self.cols].to_numpy(float))


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=here.parents[1])
    ap.add_argument("--out", type=Path, default=here)
    a = ap.parse_args(argv)
    out = a.out / "comparativo_ridge148"
    out.mkdir(parents=True, exist_ok=True)
    datos = C.cargar_datos(a.base)
    lab = datos["lab"].set_index(C.KEY, drop=False)
    splits = pd.read_csv(a.out / "splits.csv")
    base_oof = C.leer_oof(a.out / "predicciones_oof.csv")
    filas, alphas = [], []
    for (esq, rep, fold), g in splits[splits.scheme.isin(["RCV5x10", "KM6"])].groupby(["scheme", "repeat", "fold"]):
        tr = lab.loc[lab.index.isin(g.loc[g.role == "train", C.KEY])]
        te = lab.loc[lab.index.isin(g.loc[g.role == "test", C.KEY])]
        m = PriorMasRidge148Residuo(datos["cols148"]).fit(tr)
        alphas.append(float(m.pipe_.named_steps["ridge"].alpha_))
        filas.append(pd.DataFrame({"scheme": esq, "repeat": rep, "fold": fold, "modelo": m.nombre,
                                   C.KEY: te[C.KEY].to_numpy(), "municipio": te[C.MUN].to_numpy(),
                                   "y_obs": te[C.TARGET].to_numpy(), "y_pred": m.predict(te)}))
    d = pd.concat(filas, ignore_index=True); d["error"] = d.y_obs - d.y_pred
    oof = pd.concat([base_oof[base_oof.scheme.isin(["RCV5x10", "KM6"])], d], ignore_index=True)
    clus = F.microclusteres(datos["lab"])
    res = []
    for esq in ["RCV5x10", "KM6"]:
        se, _ = F.perdida_por_parcela(oof, esq)
        g = clus.loc[se.index].to_numpy()
        rep_rmse = (oof[oof.scheme == esq].assign(se=lambda x: x.error ** 2)
                    .groupby(["repeat", "modelo"])["se"].mean().pow(0.5).unstack("modelo"))
        for b in ["PRIOR", "RIDGE148"]:
            rng = np.random.default_rng(F.SEED_BOOT)
            la, lb = se["DIAG_PRIOR_MAS_RIDGE148_RESIDUO"].to_numpy(), se[b].to_numpy()
            bc = F.boot_delta(la, lb, g, rng)
            fila = {"scheme": esq, "modelo_a": "DIAG_PRIOR_MAS_RIDGE148_RESIDUO", "modelo_b": b,
                    "RMSE_a": np.sqrt(la.mean()), "RMSE_b": np.sqrt(lb.mean()), "dRMSE": np.sqrt(la.mean()) - np.sqrt(lb.mean()),
                    "IC95_cluster_lo": np.percentile(bc, 2.5), "IC95_cluster_hi": np.percentile(bc, 97.5),
                    "reps_a_mejor": int((rep_rmse["DIAG_PRIOR_MAS_RIDGE148_RESIDUO"] < rep_rmse[b]).sum()),
                    "n_reps": len(rep_rmse)}
            if esq == "RCV5x10":
                fila.update(F.nadeau_bengio(oof, "DIAG_PRIOR_MAS_RIDGE148_RESIDUO", b))
            res.append(fila)
        mun = oof[oof.scheme == esq].drop_duplicates(C.KEY).set_index(C.KEY)["municipio"].loc[se.index]
        for mu in C.MUNICIPIOS_N10:
            msk = (mun == mu).to_numpy()
            res.append({"scheme": esq, "modelo_a": "DIAG_PRIOR_MAS_RIDGE148_RESIDUO", "modelo_b": f"PRIOR [{mu}]",
                        "dRMSE": np.sqrt(se["DIAG_PRIOR_MAS_RIDGE148_RESIDUO"][msk].mean()) - np.sqrt(se["PRIOR"][msk].mean())})
    res = pd.DataFrame(res)
    res["nota"] = "Diagnostico post hoc, no candidato; alpha mas frecuente = %g" % pd.Series(alphas).mode()[0]
    res.to_csv(out / "diagnostico_prior_mas_ridge148_residuo.csv", index=False)
    print(res.round(4).to_string(index=False))
    print(pd.Series(alphas).value_counts())


if __name__ == "__main__":
    main()
