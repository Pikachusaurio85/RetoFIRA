"""Fase 2 - SHAP lineal de RIDGE148 y contraste con permutacion intra-municipio.

ANALISIS COMPARATIVO: si la Fase 1 adopta el PRIOR, estos valores NO explican la prediccion final;
explican un modelo experimental no adoptado. Las salidas van a comparativo_ridge148/.

Uso:  python fase2_shap.py [--base RUTA/cebada_datos_baseline_1] [--out RUTA/outputs/predictor]
Requiere haber ejecutado fase1_comparacion.py (lee splits.csv, predicciones_oof.csv, decision_fase1.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import comun as C

N_PERM = 30
SEED_PERM = 20261019
# bloques que se permutan juntos para medir compensacion entre familias correlacionadas
BLOQUES_COMBINADOS = {
    "COMB_SAT_BASICO+PRO": ["SAT_BASICO_S2_LANDSAT", "SAT_PRO_PLANET"],
    "CONTRATO_clima": ["CLIMA_PRECIPITACION", "CLIMA_TMIN", "CLIMA_TMAX"],
    "CONTRATO_satelite": ["SAT_BASICO_S2_LANDSAT", "SAT_PRO_PLANET", "CALIDAD_NUBOSIDAD"],
}


def eta2(valores: np.ndarray, grupos: np.ndarray) -> float:
    """Fraccion de la varianza explicada por el grupo (entre-municipios / total)."""
    v = pd.Series(valores); g = pd.Series(grupos)
    tot = np.sum((v - v.mean()) ** 2)
    if tot == 0:
        return np.nan
    entre = np.sum((g.map(v.groupby(g).mean()) - v.mean()) ** 2)
    return float(entre / tot)


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=here.parents[1])
    ap.add_argument("--out", type=Path, default=here)
    a = ap.parse_args(argv)
    dec = json.loads((a.out / "decision_fase1.json").read_text(encoding="utf-8"))
    final_es_ridge = dec["decision"] == "ADOPTAR RIDGE"
    etiqueta_modelo = "RIDGE148" if final_es_ridge else "RIDGE148_COMPARATIVO_NO_ADOPTADO"
    out = a.out if final_es_ridge else a.out / "comparativo_ridge148"
    out.mkdir(parents=True, exist_ok=True)

    datos = C.cargar_datos(a.base)
    lab = datos["lab"].set_index(C.KEY, drop=False)
    cols = datos["cols148"]
    fam = pd.Series({c: C.familia_148(c) for c in cols})
    familias = list(C.FAMILIAS_DESC)
    splits = pd.read_csv(a.out / "splits.csv")
    oof1 = C.leer_oof(a.out / "predicciones_oof.csv")
    oof1 = oof1[(oof1.scheme == "RCV5x10") & (oof1.modelo == "RIDGE148")].set_index(["repeat", C.KEY])["y_pred"]

    try:
        import shap
        shap_version = shap.__version__
    except ImportError:
        shap = None
        shap_version = None

    sp = splits[splits.scheme == "RCV5x10"]
    reg_local, reg_fam, perm, checks = [], [], [], []
    rng = np.random.default_rng(SEED_PERM)
    for (rep, fold), g in sp.groupby(["repeat", "fold"]):
        tr = lab.loc[lab.index.isin(g.loc[g.role == "train", C.KEY])]
        te = lab.loc[lab.index.isin(g.loc[g.role == "test", C.KEY])]
        m = C.ModeloRidge148(cols).fit(tr)
        Xtr, Xte = tr[cols].to_numpy(float), te[cols].to_numpy(float)
        phi, base, pred = C.shap_lineal_pipeline(m.pipe_, Xtr, Xte)
        # 1) reconstruccion y coincidencia con la Fase 1
        recon = base + phi.sum(1)
        checks.append({"repeat": rep, "fold": fold,
                       "max_abs_recon": float(np.max(np.abs(recon - pred))),
                       "max_abs_vs_fase1": float(np.max(np.abs(pred - oof1.loc[rep].loc[te.index].to_numpy()))),
                       "base_menos_media_y_train": float(base - tr[C.TARGET].mean())})
        # 2) contraste con la libreria shap (LinearExplainer, masker Independent con todo el fondo)
        if shap is not None and rep == 0:
            esc, lin = m.pipe_.named_steps["escalado"], m.pipe_.named_steps["ridge"]
            Ztr = esc.transform(m.pipe_.named_steps["imputacion"].transform(Xtr))
            Zte = esc.transform(m.pipe_.named_steps["imputacion"].transform(Xte))
            ex = shap.LinearExplainer((lin.coef_, lin.intercept_), shap.maskers.Independent(Ztr, max_samples=len(Ztr)))
            sv = ex.shap_values(Zte)
            checks[-1]["max_abs_vs_libreria_shap"] = float(np.max(np.abs(sv - phi)))
            checks[-1]["abs_base_vs_libreria_shap"] = float(abs(float(np.ravel(ex.expected_value)[0]) - base))
        # 3) registros
        phif = pd.DataFrame(phi, columns=cols).T.groupby(fam).sum().T  # contribucion por familia (suma exacta)
        for i, pid in enumerate(te.index):
            reg_fam.append({"repeat": rep, "fold": fold, C.KEY: pid, "municipio": te.loc[pid, C.MUN],
                            "y_obs": te.loc[pid, C.TARGET], "base": base, "y_pred": pred[i],
                            **{f: phif.iloc[i].get(f, 0.0) for f in familias}})
            if rep == 0:
                reg_local.append(pd.DataFrame({"contexto": "OOF_RCV5x10_rep0", "modelo": etiqueta_modelo,
                                               "repeat": rep, "fold": fold, C.KEY: pid, "feature": cols,
                                               "familia": fam.values, "familia_contrato": fam.map(C.FAMILIA_CONTRATO).values, "valor_x": Xte[i], "media_fondo": Xtr.mean(0),
                                               "shap_t_ha": phi[i], "valor_base_t_ha": base,
                                               "prediccion_t_ha": pred[i]}))
        # 4) permutacion de familias (modelo fijo, filas de prueba del fold): global e intra-municipio
        mun_te = te[C.MUN].astype(str).to_numpy()
        for modo in ["global", "intra_municipio"]:
            for f in familias + list(BLOQUES_COMBINADOS):
                cf = fam.index[fam.isin(BLOQUES_COMBINADOS.get(f, [f]))].tolist()
                jj = [cols.index(c) for c in cf]
                for b in range(N_PERM):
                    if modo == "global":
                        orden = rng.permutation(len(te))
                    else:
                        orden = np.arange(len(te))
                        for mu in np.unique(mun_te):
                            ii = np.where(mun_te == mu)[0]
                            orden[ii] = ii[rng.permutation(len(ii))]
                    Xp = Xte.copy(); Xp[:, jj] = Xte[orden][:, jj]   # bloque completo: conserva correlacion interna
                    perm.append({"repeat": rep, "fold": fold, "modo": modo, "familia": f, "b": b,
                                 "sse_perm": float(np.sum((te[C.TARGET].to_numpy() - m.pipe_.predict(Xp)) ** 2)),
                                 "sse_orig": float(np.sum((te[C.TARGET].to_numpy() - pred) ** 2))})

    checks = pd.DataFrame(checks)
    fam_df = pd.DataFrame(reg_fam)
    perm = pd.DataFrame(perm)

    # ---------------- importancia global SHAP por familia (10 repeticiones OOF)
    for nb, miembros in BLOQUES_COMBINADOS.items():
        fam_df[nb] = fam_df[miembros].sum(axis=1)
    filas = []
    for f in familias + list(BLOQUES_COMBINADOS):
        por_rep = fam_df.groupby("repeat")[f].apply(lambda s: s.abs().mean())
        eta_rep = fam_df.groupby("repeat").apply(lambda d: eta2(d[f].to_numpy(), d["municipio"].to_numpy()),
                                                  include_groups=False)
        sd_intra = fam_df.groupby("repeat").apply(
            lambda d: np.sqrt(np.mean((d[f] - d.groupby("municipio")[f].transform("mean")) ** 2)), include_groups=False)
        miembros = BLOQUES_COMBINADOS.get(f, [f])
        filas.append({"familia": f, "n_variables": int(fam.isin(miembros).sum()),
                      "descripcion": C.FAMILIAS_DESC.get(f, "Bloque combinado (suma exacta de familias): " + " + ".join(miembros)),
                      "media_abs_shap_t_ha": por_rep.mean(), "sd_entre_reps": por_rep.std(),
                      "eta2_municipio_shap": eta_rep.mean(), "sd_intra_municipio_shap_t_ha": sd_intra.mean()})
    imp = pd.DataFrame(filas)
    simples = imp.familia.isin(familias)
    imp["participacion_pct"] = np.where(simples, 100 * imp.media_abs_shap_t_ha / imp.media_abs_shap_t_ha[simples].sum(), np.nan)

    # ---------------- permutacion: dRMSE a nivel repeticion (138 parcelas OOF por permutacion b)
    agg = perm.groupby(["modo", "familia", "repeat", "b"])[["sse_perm", "sse_orig"]].sum()
    agg["dRMSE"] = np.sqrt(agg.sse_perm / 138) - np.sqrt(agg.sse_orig / 138)
    pr = agg.groupby(["modo", "familia", "repeat"])["dRMSE"].mean().reset_index()
    pres = pr.groupby(["modo", "familia"])["dRMSE"].agg(
        dRMSE_media="mean", dRMSE_sd_reps="std", dRMSE_min_rep="min", dRMSE_max_rep="max",
        reps_positivas=lambda s: int((s > 0).sum())).reset_index()
    piv = pres.pivot(index="familia", columns="modo")
    piv.columns = [f"perm_{m}_{c}" for c, m in piv.columns]
    imp = imp.merge(piv.reset_index(), on="familia")
    for col, src in [("rango_shap", "media_abs_shap_t_ha"), ("rango_perm_global", "perm_global_dRMSE_media"),
                     ("rango_perm_intra", "perm_intra_municipio_dRMSE_media")]:
        imp[col] = imp[src].where(imp.familia.isin(familias)).rank(ascending=False)  # rango solo entre familias simples
    imp.insert(0, "modelo", etiqueta_modelo)
    imp.sort_values("rango_shap").to_csv(out / "importancia_familias_shap_vs_permutacion.csv", index=False)
    # resumen en las 5 familias del contrato comun (ubicacion y geometria no tienen variables en RIDGE148)
    contrato = []
    for fc in C.FAMILIAS_CONTRATO:
        miembros = [f for f in familias if C.FAMILIA_CONTRATO[f] == fc]
        if not miembros:
            contrato.append({"modelo": etiqueta_modelo, "familia_contrato": fc, "n_variables": 0,
                             "nota": "RIDGE148 no incluye variables de esta familia"})
            continue
        fila = imp.set_index("familia").loc["CONTRATO_" + fc if len(miembros) > 1 else miembros[0]]
        contrato.append({"modelo": etiqueta_modelo, "familia_contrato": fc, "n_variables": int(fila.n_variables),
                         "familias_fuente": " + ".join(miembros), "media_abs_shap_t_ha": fila.media_abs_shap_t_ha,
                         "eta2_municipio_shap": fila.eta2_municipio_shap,
                         "perm_global_dRMSE_media": fila.perm_global_dRMSE_media,
                         "perm_intra_municipio_dRMSE_media": fila.perm_intra_municipio_dRMSE_media,
                         "nota": "agrupacion descriptiva; no causal"})
    pd.DataFrame(contrato).to_csv(out / "importancia_familias_contrato.csv", index=False)

    # ---------------- compensacion entre familias dentro del municipio
    wc = fam_df.copy()
    for f in familias + ["y_pred"]:
        wc[f] = wc[f] - wc.groupby(["repeat", "municipio"])[f].transform("mean")
    corr_intra = wc[familias].corr()
    corr_intra.to_csv(out / "correlacion_intra_municipio_contribuciones_familia.csv")
    sd_intra_pred = float(np.sqrt(wc.groupby("repeat")["y_pred"].apply(lambda s: np.mean(s ** 2)).mean()))
    sd_intra_suma_fam = float(np.sqrt(np.sum(imp.sd_intra_municipio_shap_t_ha ** 2)))

    local = pd.concat(reg_local, ignore_index=True)

    # ---------------- parcelas representativas (rep 0, criterios fijos)
    f0 = fam_df[fam_df.repeat == 0].copy()
    f0["error"] = f0.y_obs - f0.y_pred
    f0 = f0.sort_values(C.KEY)
    elegidas = {}
    for caso, serie, asc in [("acierto (|error| minimo)", f0.error.abs(), True),
                             ("error alto (|error| maximo)", f0.error.abs(), False),
                             ("rendimiento bajo (y minimo)", f0.y_obs, True),
                             ("rendimiento alto (y maximo)", f0.y_obs, False)]:
        orden = f0.assign(s=serie).sort_values(["s", C.KEY], ascending=[asc, True])
        pid = next(p for p in orden[C.KEY] if p not in elegidas.values())
        elegidas[caso] = pid
    prior_oof = C.leer_oof(a.out / "predicciones_oof.csv")
    prior_oof = prior_oof[(prior_oof.scheme == "RCV5x10") & (prior_oof.repeat == 0) & (prior_oof.modelo == "PRIOR")]
    prior_oof = prior_oof.set_index(C.KEY)["y_pred"]
    rep_rows = []
    for caso, pid in elegidas.items():
        r = f0.set_index(C.KEY).loc[pid]
        lp = local[local[C.KEY] == pid].assign(a=lambda d: d.shap_t_ha.abs()).sort_values("a", ascending=False)
        rep_rows.append({"caso": caso, C.KEY: pid, "municipio": r.municipio, "y_true_t_ha": r.y_obs,
                         "pred_ridge148_oof": r.y_pred, "error_ridge148": r.error,
                         "pred_PRIOR_oof_mismo_fold": float(prior_oof.loc[pid]), "valor_base": r.base,
                         **{f"contrib_{f}": r[f] for f in familias},
                         "suma_contrib_mas_base": r.base + sum(r[f] for f in familias),
                         "top5_variables": "; ".join(f"{x.feature} ({x.shap_t_ha:+.3f})" for x in lp.head(5).itertuples())})
    reps = pd.DataFrame(rep_rows)
    reps.insert(0, "modelo", etiqueta_modelo)
    reps.to_csv(out / "explicaciones_locales_representativas.csv", index=False)

    # ---------------- modelo completo (138) explicando las 197 parcelas
    full = datos["df"].set_index(C.KEY, drop=False)
    mfull = C.ModeloRidge148(cols).fit(lab)
    Xl = lab[cols].to_numpy(float); Xall = full[cols].to_numpy(float)
    phi_all, base_all, pred_all = C.shap_lineal_pipeline(mfull.pipe_, Xl, Xall)
    checks_full = float(np.max(np.abs(base_all + phi_all.sum(1) - pred_all)))
    comp = []
    for i, pid in enumerate(full.index):
        comp.append(pd.DataFrame({"contexto": np.where(full.loc[pid, "CONJUNTO"] == "PREDICCION",
                                                       "MODELO_COMPLETO_parcela_PREDICCION",
                                                       "MODELO_COMPLETO_in_sample"),
                                  "modelo": etiqueta_modelo, "repeat": -1, "fold": -1, C.KEY: pid,
                                  "feature": cols, "familia": fam.values,
                                  "familia_contrato": fam.map(C.FAMILIA_CONTRATO).values, "valor_x": Xall[i],
                                  "media_fondo": Xl.mean(0), "shap_t_ha": phi_all[i],
                                  "valor_base_t_ha": base_all, "prediccion_t_ha": pred_all[i]}))
    expl = pd.concat([local] + comp, ignore_index=True)
    expl.to_csv(out / "explicaciones_shap.csv", index=False)

    # coeficientes del modelo completo vs archivo del equipo (reproducibilidad)
    coef = pd.DataFrame({"variable": cols, "familia": fam.values,
                         "coef_estandarizado": mfull.pipe_.named_steps["ridge"].coef_})
    ruta_eq = Path(a.base) / "modelos_experimentales/Ridge_model/resultados_ridge_agrocebada/coeficientes_Ridge_148_historicas.csv"
    coef_check = None
    if ruta_eq.exists():
        eq = pd.read_csv(ruta_eq).set_index("variable")["coeficiente_estandarizado"]
        coef["coef_equipo"] = coef.variable.map(eq)
        coef_check = float(np.max(np.abs(coef.coef_estandarizado - coef.coef_equipo)))
    coef.to_csv(out / "coeficientes_ridge148_modelo_completo.csv", index=False)

    # eta2 municipal de las predicciones OOF (que parte de lo que predice cada modelo es entre-municipios)
    o = C.leer_oof(a.out / "predicciones_oof.csv")
    o = o[o.scheme == "RCV5x10"]
    eta_pred = (o.groupby(["modelo", "repeat"]).apply(lambda d: eta2(d.y_pred.to_numpy(), d.municipio.to_numpy()),
                                                      include_groups=False)
                .groupby("modelo").mean().rename("eta2_municipio_predicciones_oof").reset_index())
    eta_pred.to_csv(out / "eta2_municipal_predicciones.csv", index=False)

    verif = {
        "modelo_explicado": etiqueta_modelo,
        "es_modelo_final": final_es_ridge,
        "explicador": "SHAP lineal intervencional exacto (phi_j = w_j/s_j * (x_j - media_fondo_j)); fondo = parcelas de entrenamiento del ajuste",
        "supuesto_dependencia": "Expectativa marginal: rompe la dependencia entre variables (fiel al modelo, no a los datos). "
                                "La version condicional (correlation_dependent) no se uso: con n_train ~110 < p = 148 la covarianza es singular.",
        "preprocesamiento": "SHAP calculado en unidades originales a traves de imputacion mediana y StandardScaler del mismo pipeline; "
                            "las 197 parcelas no tienen faltantes en las 148 variables, la imputacion es identidad.",
        "max_abs_reconstruccion_OOF": float(checks.max_abs_recon.max()),
        "max_abs_diferencia_vs_predicciones_fase1": float(checks.max_abs_vs_fase1.max()),
        "max_abs_base_menos_media_y_train": float(checks.base_menos_media_y_train.abs().max()),
        "max_abs_vs_libreria_shap": float(checks["max_abs_vs_libreria_shap"].max()) if "max_abs_vs_libreria_shap" in checks else None,
        "max_abs_base_vs_libreria_shap": float(checks["abs_base_vs_libreria_shap"].max()) if "abs_base_vs_libreria_shap" in checks else None,
        "shap_version": shap_version,
        "max_abs_reconstruccion_modelo_completo": checks_full,
        "max_abs_coef_vs_archivo_equipo": coef_check,
        "alpha_modelo_completo": float(mfull.pipe_.named_steps["ridge"].alpha_),
        "n_permutaciones_por_fold_y_familia": N_PERM,
        "sd_intra_municipio_prediccion_t_ha": sd_intra_pred,
        "sd_intra_municipio_si_familias_fueran_independientes_t_ha": sd_intra_suma_fam,
        "corr_intra_SAT_BASICO_vs_CLIMA_TMIN": float(corr_intra.loc["SAT_BASICO_S2_LANDSAT", "CLIMA_TMIN"]),
        "corr_intra_SAT_BASICO_vs_SAT_PRO": float(corr_intra.loc["SAT_BASICO_S2_LANDSAT", "SAT_PRO_PLANET"]),
        "parcelas_representativas": elegidas,
    }
    C.escribir_json(out / "verificacion_shap.json", verif)
    pd.set_option("display.width", 250)
    print(imp.sort_values("rango_shap")[["familia", "n_variables", "media_abs_shap_t_ha", "participacion_pct",
                                         "eta2_municipio_shap", "sd_intra_municipio_shap_t_ha",
                                         "perm_global_dRMSE_media", "perm_intra_municipio_dRMSE_media",
                                         "perm_intra_municipio_reps_positivas"]].round(4).to_string(index=False))
    print(reps.drop(columns=["top5_variables"]).round(3).T.to_string())
    print(eta_pred.round(3).to_string(index=False))
    print(corr_intra.round(2).to_string())
    print(json.dumps(verif, indent=1, default=str))


if __name__ == "__main__":
    main()
