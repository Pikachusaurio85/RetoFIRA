"""Fase 1 - Comparacion pareada definitiva: PRIOR calibrado vs RIDGE148 vs HIBRIDO_TOPO.

Uso:  python fase1_comparacion.py [--base RUTA/cebada_datos_baseline_1] [--out RUTA/outputs/predictor]

Genera: splits.csv, descriptor_particiones.csv, predicciones_oof.csv, hiperparametros_por_fold.csv,
metricas_por_repeticion.csv, errores_pareados_parcela.csv, contrastes_pareados.csv,
comparacion_por_municipio.csv, comparacion_final.csv, decision_fase1.json.
Todo hiperparametro (k del prior, alpha de Ridge) se ajusta solo con el entrenamiento de cada fold.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

import comun as C

N_BOOT = 5000
SEED_BOOT = 20261018


# --------------------------------------------------------------------------- particiones
def generar_particiones(lab: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ids = lab[C.KEY].to_numpy()
    xy = lab[["loc_x_utm", "loc_y_utm"]].to_numpy(float)
    filas = []

    def agregar(esquema, rep, fold, tr, te):
        filas.extend((esquema, rep, fold, ids[i], "train") for i in tr)
        filas.extend((esquema, rep, fold, ids[i], "test") for i in te)

    for r, seed in enumerate(C.RCV_SEEDS):
        for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(ids)):
            agregar("RCV5x10", r, f, tr, te)
    for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=C.SEED_RIDGE_EXPERIMENTO).split(ids)):
        agregar("RCV5_S2026", 0, f, tr, te)
    km0 = KMeans(6, n_init=10, random_state=0).fit_predict(xy)
    km2026 = KMeans(6, n_init=10, random_state=C.SEED_RIDGE_EXPERIMENTO).fit_predict(xy)
    for f, (tr, te) in enumerate(GroupKFold(6).split(ids, groups=km0)):
        agregar("KM6", 0, f, tr, te)
    for f, (tr, te) in enumerate(LeaveOneGroupOut().split(ids, groups=lab[C.MUN].astype(str).to_numpy())):
        agregar("LMO", 0, f, tr, te)
    splits = pd.DataFrame(filas, columns=["scheme", "repeat", "fold", C.KEY, "role"])
    info = {"KM6_ARI_random_state_0_vs_2026": float(adjusted_rand_score(km0, km2026)),
            "KM6_tamanos": pd.Series(km0).value_counts().sort_index().tolist()}
    return splits, info


def describir_particiones(splits: pd.DataFrame, lab: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    xy = lab.set_index(C.KEY)[["loc_x_utm", "loc_y_utm"]]
    filas = []
    for (esq, rep, fold), g in splits.groupby(["scheme", "repeat", "fold"]):
        tr = g.loc[g.role == "train", C.KEY]; te = g.loc[g.role == "test", C.KEY]
        d, _ = cKDTree(xy.loc[tr].to_numpy()).query(xy.loc[te].to_numpy(), k=1)
        filas.append((esq, rep, fold, len(tr), len(te), np.median(d), np.mean(d < 500)))
    out = pd.DataFrame(filas, columns=["scheme", "repeat", "fold", "n_train", "n_test",
                                       "dist_mediana_test_a_train_m", "frac_test_vecino_menor_500m"])
    d, _ = cKDTree(xy.to_numpy()).query(pred[["loc_x_utm", "loc_y_utm"]].to_numpy(float), k=1)
    ref = pd.DataFrame([("REFERENCIA_59_PREDICCION_vs_138", -1, -1, 138, 59, np.median(d), np.mean(d < 500))],
                       columns=out.columns)
    return pd.concat([ref, out], ignore_index=True)


# --------------------------------------------------------------------------- ejecucion CV
def ejecutar_cv(datos: dict, splits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lab = datos["lab"].set_index(C.KEY, drop=False)
    fab = C.fabricas(datos["cols148"])
    oof, hip = [], []
    for (esq, rep, fold), g in splits.groupby(["scheme", "repeat", "fold"], sort=True):
        tr_ids = g.loc[g.role == "train", C.KEY].to_numpy()
        te_ids = g.loc[g.role == "test", C.KEY].to_numpy()
        # el orden de entrenamiento es el de la tabla (ID ascendente), como en el experimento original
        tr = lab.loc[lab.index.isin(tr_ids)]
        te = lab.loc[lab.index.isin(te_ids)]
        assert len(tr) == len(tr_ids) and len(te) == len(te_ids) and not set(tr_ids) & set(te_ids)
        for nombre, mk in fab.items():
            m = mk().fit(tr)
            p = m.predict(te)
            oof.append(pd.DataFrame({"scheme": esq, "repeat": rep, "fold": fold, "modelo": nombre,
                                     C.KEY: te[C.KEY].to_numpy(), "municipio": te[C.MUN].to_numpy(),
                                     "y_obs": te[C.TARGET].to_numpy(), "y_pred": p}))
            hip.append({"scheme": esq, "repeat": rep, "fold": fold, "modelo": nombre,
                        "n_train": len(tr), "n_test": len(te), **m.hiper()})
    oof = pd.concat(oof, ignore_index=True)
    oof["error"] = oof["y_obs"] - oof["y_pred"]
    return oof, pd.DataFrame(hip)


# --------------------------------------------------------------------------- inferencia
def microclusteres(lab: pd.DataFrame, k: int = 25) -> pd.Series:
    xy = lab[["loc_x_utm", "loc_y_utm"]].to_numpy(float)
    return pd.Series(KMeans(k, n_init=10, random_state=0).fit_predict(xy), index=lab[C.KEY].to_numpy())


def perdida_por_parcela(oof: pd.DataFrame, esquema: str) -> pd.DataFrame:
    """Media sobre repeticiones del error cuadratico (y absoluto) por parcela y modelo."""
    o = oof[oof.scheme == esquema].assign(se=lambda d: d.error ** 2, ae=lambda d: d.error.abs())
    se = o.pivot_table(index=C.KEY, columns="modelo", values="se", aggfunc="mean")
    ae = o.pivot_table(index=C.KEY, columns="modelo", values="ae", aggfunc="mean")
    return se, ae


def boot_delta(la: np.ndarray, lb: np.ndarray, grupos: np.ndarray | None, rng) -> np.ndarray:
    """Distribucion bootstrap de RMSE(a)-RMSE(b) remuestreando parcelas o grupos completos."""
    if grupos is None:
        n = len(la)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        return np.sqrt(la[idx].mean(1)) - np.sqrt(lb[idx].mean(1))
    ug = np.unique(grupos)
    sa = np.array([la[grupos == u].sum() for u in ug]); sb = np.array([lb[grupos == u].sum() for u in ug])
    nn = np.array([(grupos == u).sum() for u in ug])
    idx = rng.integers(0, len(ug), size=(N_BOOT, len(ug)))
    return np.sqrt(sa[idx].sum(1) / nn[idx].sum(1)) - np.sqrt(sb[idx].sum(1) / nn[idx].sum(1))


def boot_media(d: np.ndarray, grupos: np.ndarray, rng) -> np.ndarray:
    """Bootstrap por grupos completos de la media de una diferencia por parcela (p. ej. dMAE)."""
    ug = np.unique(grupos)
    s = np.array([d[grupos == u].sum() for u in ug]); nn = np.array([(grupos == u).sum() for u in ug])
    idx = rng.integers(0, len(ug), size=(N_BOOT, len(ug)))
    return s[idx].sum(1) / nn[idx].sum(1)


def nadeau_bengio(oof: pd.DataFrame, a: str, b: str, esquema: str = "RCV5x10") -> dict:
    o = oof[(oof.scheme == esquema) & oof.modelo.isin([a, b])].assign(se=lambda d: d.error ** 2)
    f = o.groupby(["repeat", "fold", "modelo"])["se"].mean().unstack("modelo")
    d = (f[a] - f[b]).to_numpy()
    n_te = oof[(oof.scheme == esquema) & (oof.modelo == a)].groupby(["repeat", "fold"]).size().mean()
    n_tr = 138 - n_te
    J = len(d)
    var = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1.0 / J + n_te / n_tr) * var)
    return {"NB_t": float(t), "NB_p": float(2 * stats.t.sf(abs(t), df=J - 1)), "NB_J": J,
            "dMSE_fold_media": float(d.mean())}


def etiqueta(lo: float, hi: float, p_holm: float | None) -> str:
    if -C.ROPE < lo and hi < C.ROPE:
        return "EQUIVALENTE (IC95 dentro de +-0.02)"
    if hi < -C.ROPE and (p_holm is None or p_holm < 0.05):
        return "MEJOR (relevante)"
    if lo > C.ROPE and (p_holm is None or p_holm < 0.05):
        return "PEOR (relevante)"
    if hi < 0:
        return "MEJOR (IC<0, magnitud dentro de ROPE)"
    if lo > 0:
        return "PEOR (IC>0, magnitud dentro de ROPE)"
    return "INDETERMINADO"


def contrastes(oof: pd.DataFrame, lab: pd.DataFrame, clus: pd.Series) -> pd.DataFrame:
    pares = [("RIDGE148", "PRIOR"), ("HIBRIDO_TOPO", "PRIOR"), ("RIDGE148", "HIBRIDO_TOPO"),
             ("PRIOR", "PRIOR_K3"), ("RIDGE148", "PRIOR_K3"), ("HIBRIDO_TOPO", "PRIOR_K3"),
             ("PRIOR", "MEDIA_GLOBAL")]
    filas = []
    for esq in oof.scheme.unique():
        se, ae = perdida_por_parcela(oof, esq)
        g = clus.loc[se.index].to_numpy()
        rep_rmse = (oof[oof.scheme == esq].assign(se=lambda d: d.error ** 2)
                    .groupby(["repeat", "modelo"])["se"].mean().pow(0.5).unstack("modelo"))
        for a, b in pares:
            rng = np.random.default_rng(SEED_BOOT)
            la, lb = se[a].to_numpy(), se[b].to_numpy()
            delta = np.sqrt(la.mean()) - np.sqrt(lb.mean())
            bp = boot_delta(la, lb, None, rng); bc = boot_delta(la, lb, g, rng)
            w = stats.wilcoxon(la - lb) if np.any(la != lb) else None
            bmae = boot_media(ae[a].to_numpy() - ae[b].to_numpy(), g, rng)
            fila = {"scheme": esq, "modelo_a": a, "modelo_b": b,
                    "RMSE_a": float(np.sqrt(la.mean())), "RMSE_b": float(np.sqrt(lb.mean())),
                    "dRMSE": float(delta), "dMAE": float(ae[a].mean() - ae[b].mean()),
                    "dMAE_IC95_cluster_lo": float(np.percentile(bmae, 2.5)),
                    "dMAE_IC95_cluster_hi": float(np.percentile(bmae, 97.5)),
                    "IC95_parcela_lo": float(np.percentile(bp, 2.5)), "IC95_parcela_hi": float(np.percentile(bp, 97.5)),
                    "IC95_cluster_lo": float(np.percentile(bc, 2.5)), "IC95_cluster_hi": float(np.percentile(bc, 97.5)),
                    "P_boot_cluster_dRMSE_menor_0": float(np.mean(bc < 0)),
                    "wilcoxon_p_parcela": float(w.pvalue) if w is not None else np.nan,
                    "reps_a_mejor": int((rep_rmse[a] < rep_rmse[b]).sum()), "n_reps": int(len(rep_rmse)),
                    "dRMSE_rep_min": float((rep_rmse[a] - rep_rmse[b]).min()),
                    "dRMSE_rep_max": float((rep_rmse[a] - rep_rmse[b]).max())}
            if esq == "RCV5x10":
                fila.update(nadeau_bengio(oof, a, b, esq))
            filas.append(fila)
    df = pd.DataFrame(filas)
    # Holm sobre los 2 candidatos frente a PRIOR en el esquema principal
    m = (df.scheme == "RCV5x10") & df.modelo_b.eq("PRIOR") & df.modelo_a.isin(["RIDGE148", "HIBRIDO_TOPO"])
    p = df.loc[m, "NB_p"].to_numpy(); orden = np.argsort(p); holm = np.empty_like(p)
    acc = 0.0
    for rank, i in enumerate(orden):
        acc = max(acc, min(1.0, (len(p) - rank) * p[i])); holm[i] = acc
    df["NB_p_holm"] = np.nan
    df.loc[m, "NB_p_holm"] = holm
    df["etiqueta"] = [etiqueta(r.IC95_cluster_lo, r.IC95_cluster_hi,
                               None if pd.isna(r.NB_p_holm) else r.NB_p_holm) for r in df.itertuples()]
    return df


def por_municipio(oof: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for esq in ["RCV5x10", "KM6", "LMO"]:
        se, ae = perdida_por_parcela(oof, esq)
        o = oof[oof.scheme == esq]
        mun = o.drop_duplicates(C.KEY).set_index(C.KEY)["municipio"].loc[se.index]
        grupo = mun.where(mun.isin(C.MUNICIPIOS_N10), "OTROS (n<10)")
        sesgo = o.groupby([C.KEY, "modelo"])["error"].mean().unstack("modelo").loc[se.index]
        for gname in C.MUNICIPIOS_N10 + ["OTROS (n<10)"]:
            msk = (grupo == gname).to_numpy()
            rng = np.random.default_rng(SEED_BOOT)
            for modelo in se.columns:
                fila = {"scheme": esq, "municipio": gname, "n": int(msk.sum()), "modelo": modelo,
                        "RMSE": float(np.sqrt(se[modelo][msk].mean())), "MAE": float(ae[modelo][msk].mean()),
                        "sesgo_medio_obs_menos_pred": float(sesgo[modelo][msk].mean())}
                if modelo != "PRIOR":
                    bp = boot_delta(se[modelo][msk].to_numpy(), se["PRIOR"][msk].to_numpy(), None, rng)
                    fila.update({"dRMSE_vs_PRIOR": float(np.sqrt(se[modelo][msk].mean()) - np.sqrt(se["PRIOR"][msk].mean())),
                                 "IC95_parcela_lo": float(np.percentile(bp, 2.5)),
                                 "IC95_parcela_hi": float(np.percentile(bp, 97.5))})
                filas.append(fila)
    return pd.DataFrame(filas)


def resumen(oof: pd.DataFrame, cont: pd.DataFrame) -> pd.DataFrame:
    rep = (oof.assign(se=lambda d: d.error ** 2, ae=lambda d: d.error.abs())
           .groupby(["scheme", "modelo", "repeat"])
           .apply(lambda g: pd.Series({"RMSE": np.sqrt(g.se.mean()), "MAE": g.ae.mean(), "R2": C.r2(g.y_obs, g.y_pred)}),
                  include_groups=False).reset_index())
    agg = rep.groupby(["scheme", "modelo"]).agg(n_reps=("repeat", "nunique"), RMSE_media_reps=("RMSE", "mean"),
                                                 RMSE_sd_reps=("RMSE", "std"), MAE_media_reps=("MAE", "mean"),
                                                 R2_media_reps=("R2", "mean")).reset_index()
    filas = []
    for esq in oof.scheme.unique():
        se, ae = perdida_por_parcela(oof, esq)
        for mdl in se.columns:
            filas.append({"scheme": esq, "modelo": mdl, "RMSE_perdida_parcela": float(np.sqrt(se[mdl].mean())),
                          "MAE_perdida_parcela": float(ae[mdl].mean())})
    agg = agg.merge(pd.DataFrame(filas), on=["scheme", "modelo"])
    vs = cont[cont.modelo_b.eq("PRIOR")].rename(columns={"modelo_a": "modelo"})[
        ["scheme", "modelo", "dRMSE", "dMAE", "dMAE_IC95_cluster_lo", "dMAE_IC95_cluster_hi",
         "IC95_cluster_lo", "IC95_cluster_hi", "IC95_parcela_lo",
         "IC95_parcela_hi", "reps_a_mejor", "NB_p", "NB_p_holm", "wilcoxon_p_parcela", "etiqueta"]]
    vs = vs.add_suffix("_vs_PRIOR").rename(columns={"scheme_vs_PRIOR": "scheme", "modelo_vs_PRIOR": "modelo"})
    out = agg.merge(vs, on=["scheme", "modelo"], how="left")
    orden_esq = {"RCV5x10": 0, "RCV5_S2026": 1, "KM6": 2, "LMO": 3}
    orden_mod = {"MEDIA_GLOBAL": 0, "PRIOR_K3": 1, "PRIOR": 2, "RIDGE148": 3, "HIBRIDO_TOPO": 4}
    out["rol"] = out.scheme.map(lambda s: "PRINCIPAL (CV aleatoria, ~ reto)" if s == "RCV5x10" else
                                ("REPRODUCCION experimento Ridge equipo" if s == "RCV5_S2026" else
                                 "DIAGNOSTICO espacial secundario"))
    return out.sort_values(["scheme", "modelo"], key=lambda s: s.map(orden_esq) if s.name == "scheme"
                           else s.map(orden_mod)).reset_index(drop=True), rep


def decidir(cont: pd.DataFrame, mun: pd.DataFrame) -> dict:
    res = {}
    for cand in ["RIDGE148", "HIBRIDO_TOPO"]:
        r = cont[(cont.scheme == "RCV5x10") & (cont.modelo_a == cand) & (cont.modelo_b == "PRIOR")].iloc[0]
        mm = mun[(mun.scheme == "RCV5x10") & (mun.modelo == cand) & mun.municipio.isin(C.MUNICIPIOS_N10)]
        km6 = cont[(cont.scheme == "KM6") & (cont.modelo_a == cand) & (cont.modelo_b == "PRIOR")].iloc[0]
        res[cand] = {
            "dRMSE": r.dRMSE, "IC95_cluster": [r.IC95_cluster_lo, r.IC95_cluster_hi],
            "NB_p": r.NB_p, "NB_p_holm": r.NB_p_holm, "reps_mejor": int(r.reps_a_mejor),
            "E1": bool(r.IC95_cluster_hi < 0), "E2": bool(r.NB_p_holm < 0.05), "E3": bool(r.reps_a_mejor >= 9),
            "S1": bool((mm.dRMSE_vs_PRIOR <= 0).sum() >= 3),
            "municipios_dRMSE": dict(zip(mm.municipio, mm.dRMSE_vs_PRIOR.round(4))),
            "diagnostico_KM6_dRMSE": km6.dRMSE, "diagnostico_KM6_IC95_cluster": [km6.IC95_cluster_lo, km6.IC95_cluster_hi],
        }
        res[cand]["cumple"] = all(res[cand][e] for e in ["E1", "E2", "E3", "S1"])
    rh = cont[(cont.scheme == "RCV5x10") & (cont.modelo_a == "RIDGE148") & (cont.modelo_b == "HIBRIDO_TOPO")].iloc[0]
    if res["HIBRIDO_TOPO"]["cumple"] and res["RIDGE148"]["cumple"]:
        decision = "ADOPTAR RIDGE" if (rh.IC95_cluster_hi < 0 and rh.NB_p < 0.05) else "ADOPTAR HIBRIDO"
    elif res["HIBRIDO_TOPO"]["cumple"]:
        decision = "ADOPTAR HIBRIDO"
    elif res["RIDGE148"]["cumple"]:
        decision = "ADOPTAR RIDGE"
    else:
        decision = "ADOPTAR PRIOR"
    return {"decision": decision, "criterios": res,
            "RIDGE148_vs_HIBRIDO": {"dRMSE": rh.dRMSE, "IC95_cluster": [rh.IC95_cluster_lo, rh.IC95_cluster_hi],
                                    "NB_p": rh.NB_p},
            "regla": "regla_decision_preregistrada.json"}


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=here.parents[1], help="carpeta cebada_datos_baseline_1")
    ap.add_argument("--out", type=Path, default=here)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    datos = C.cargar_datos(a.base)
    lab, pred = datos["lab"], datos["pred"]

    splits, info = generar_particiones(lab)
    assert not set(splits[C.KEY]) & set(pred[C.KEY]), "Una parcela PREDICCION entro en particiones"
    splits.to_csv(a.out / "splits.csv", index=False)
    describir_particiones(splits, lab, pred).to_csv(a.out / "descriptor_particiones.csv", index=False)

    oof, hip = ejecutar_cv(datos, splits)
    assert not set(oof[C.KEY]) & set(pred[C.KEY])
    assert oof.groupby(["scheme", "repeat", "modelo"]).size().eq(138).all(), "OOF incompleto"
    C.escribir_oof(oof, a.out / "predicciones_oof.csv")
    hip.to_csv(a.out / "hiperparametros_por_fold.csv", index=False)

    clus = microclusteres(lab)
    cont = contrastes(oof, lab, clus)
    cont.to_csv(a.out / "contrastes_pareados.csv", index=False)
    mun = por_municipio(oof)
    mun.to_csv(a.out / "comparacion_por_municipio.csv", index=False)
    comp, rep = resumen(oof, cont)
    comp.to_csv(a.out / "comparacion_final.csv", index=False)
    rep.to_csv(a.out / "metricas_por_repeticion.csv", index=False)

    se, ae = perdida_por_parcela(oof, "RCV5x10")
    o = oof[oof.scheme == "RCV5x10"]
    pm = o.pivot_table(index=C.KEY, columns="modelo", values="y_pred", aggfunc="mean").add_prefix("pred_oof_media_")
    ep = se.add_prefix("ecm_reps_")
    base_p = lab.set_index(C.KEY)[[C.MUN, C.TARGET]].rename(columns={C.MUN: "municipio", C.TARGET: "y_true_t_ha"})
    pares = base_p.join(pm).join(ep)
    for mdl in ["RIDGE148", "HIBRIDO_TOPO", "PRIOR_K3", "MEDIA_GLOBAL"]:
        pares[f"d_ecm_{mdl}_menos_PRIOR"] = se[mdl] - se["PRIOR"]
    pares["microcluster"] = clus.loc[pares.index]
    pares.reset_index().to_csv(a.out / "errores_pareados_parcela.csv", index=False)

    dec = decidir(cont, mun)
    dec.update({"particiones": info, "hashes_fuente": datos["hashes"], "segundos": round(time.time() - t0, 1),
                "k_prior_por_fold_RCV": hip[(hip.scheme == "RCV5x10") & (hip.modelo == "PRIOR")].k_prior.describe().to_dict(),
                "alpha_ridge148_por_fold_RCV": hip[(hip.scheme == "RCV5x10") & (hip.modelo == "RIDGE148")]
                .alpha_ridge.value_counts().to_dict()})
    C.escribir_json(a.out / "decision_fase1.json", dec)
    print(comp[["scheme", "modelo", "RMSE_media_reps", "RMSE_sd_reps", "MAE_media_reps", "R2_media_reps",
                "dRMSE_vs_PRIOR", "IC95_cluster_lo_vs_PRIOR", "IC95_cluster_hi_vs_PRIOR", "reps_a_mejor_vs_PRIOR",
                "NB_p_holm_vs_PRIOR"]].round(4).to_string(index=False))
    print("DECISION:", dec["decision"])
    print(f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
