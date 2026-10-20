"""Fase 3 - Modelo final (PRIOR calibrado): artefacto, validacion de intervalos, salidas JSON y manifiesto.

Uso:  python fase3_modelo_final.py [--base RUTA/cebada_datos_baseline_1] [--out RUTA/outputs/predictor]
Requiere fase1 (y fase2/diagnostico para listar sus archivos en el manifiesto).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import comun as C
import modelo_final as MF

CONTRATO_CLAVES = ["schema_version", "ID_POLIGONO", "model_id", "prediction_t_ha", "uncertainty",
                   "prediction_components", "warnings"]


def validar_en_cv(datos: dict, splits: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Reajusta modelo_final.ajustar en cada fold: (1) coincide con PRIOR de la Fase 1; (2) cobertura del intervalo."""
    lab = datos["lab"].set_index(C.KEY, drop=False)
    ref = oof[oof.modelo == "PRIOR"].set_index(["scheme", "repeat", C.KEY])["y_pred"]
    filas, max_diff = [], 0.0
    for (esq, rep, fold), g in splits[splits.scheme.isin(["RCV5x10", "KM6"])].groupby(["scheme", "repeat", "fold"]):
        tr = lab.loc[lab.index.isin(g.loc[g.role == "train", C.KEY])]
        te = lab.loc[lab.index.isin(g.loc[g.role == "test", C.KEY])]
        art = MF.ajustar(tr)
        for pid, r in te.iterrows():
            o = MF.predecir_uno(art, {"loc_municipio": r[C.MUN]})
            m = art["municipios"].get(r[C.MUN])
            p_sin_redondeo = m["prediccion_t_ha"] if m else art["media_global_t_ha"]
            max_diff = max(max_diff, abs(p_sin_redondeo - ref.loc[(esq, rep, pid)]))
            filas.append({"scheme": esq, "repeat": rep, "fold": fold, C.KEY: pid, "municipio": r[C.MUN],
                          "y_obs": r[C.TARGET], "pred": o["prediction_t_ha"],
                          "lower": o["uncertainty"]["lower_t_ha"], "upper": o["uncertainty"]["upper_t_ha"]})
    v = pd.DataFrame(filas)
    v[["lower", "upper"]] = v[["lower", "upper"]].astype(float)  # None (municipio no visto en el fold) -> NaN
    v["con_intervalo"] = v.lower.notna()
    v["cubre"] = np.where(v.con_intervalo, (v.y_obs >= v.lower) & (v.y_obs <= v.upper), np.nan)
    v["ancho"] = v.upper - v.lower
    v["grupo"] = v.municipio.where(v.municipio.isin(C.MUNICIPIOS_N10), "OTROS (n<10)")
    res = []
    for esq, d in v.groupby("scheme"):
        cov_rep = d.groupby("repeat").cubre.mean()
        res.append({"scheme": esq, "grupo": "TODAS", "n_parcelas": d[C.KEY].nunique(),
                    "frac_predicciones_sin_intervalo": 1 - d.con_intervalo.mean(), "cobertura": d.cubre.mean(),
                    "cobertura_min_rep": cov_rep.min(), "cobertura_max_rep": cov_rep.max(), "ancho_medio_t_ha": d.ancho.mean()})
        for gname, dg in d.groupby("grupo"):
            res.append({"scheme": esq, "grupo": gname, "n_parcelas": dg[C.KEY].nunique(),
                        "frac_predicciones_sin_intervalo": 1 - dg.con_intervalo.mean(), "cobertura": dg.cubre.mean(),
                        "ancho_medio_t_ha": dg.ancho.mean()})
    res = pd.DataFrame(res)
    res["nivel_nominal"] = MF.NIVEL_INTERVALO
    return res, float(max_diff)


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=here.parents[1])
    ap.add_argument("--out", type=Path, default=here)
    a = ap.parse_args(argv)
    dec = json.loads((a.out / "decision_fase1.json").read_text(encoding="utf-8"))
    if dec["decision"] != "ADOPTAR PRIOR":
        raise SystemExit(f"La Fase 1 decidio {dec['decision']}; este modelo_final implementa solo el PRIOR. Revisar.")
    datos = C.cargar_datos(a.base)
    splits = pd.read_csv(a.out / "splits.csv")
    oof = C.leer_oof(a.out / "predicciones_oof.csv")

    # 1) artefacto final (138 parcelas)
    art = MF.entrenar(a.base)
    MF.guardar(art, a.out / "modelo_final.json")

    # 2) validacion: coincidencia con Fase 1 y cobertura del intervalo
    cob, max_diff = validar_en_cv(datos, splits, oof)
    cob.to_csv(a.out / "validacion_intervalo_prior.csv", index=False)
    art["incertidumbre"]["cobertura_validada"] = {
        "esquema": "RCV5x10 (cuantiles recalculados solo con el entrenamiento de cada fold)",
        "cobertura_global": float(cob[(cob.scheme == "RCV5x10") & (cob.grupo == "TODAS")].cobertura.iloc[0]),
        "cobertura_por_grupo": {r.grupo: round(float(r.cobertura), 4) for r in cob[cob.scheme == "RCV5x10"].itertuples()},
        "cobertura_KM6_diagnostico": float(cob[(cob.scheme == "KM6") & (cob.grupo == "TODAS")].cobertura.iloc[0]),
    }
    MF.guardar(art, a.out / "modelo_final.json")
    art = MF.cargar(a.out / "modelo_final.json")

    # 3) predicciones para las 197 parcelas (JSON contrato 1.0) y tabla plana
    todos = datos["df"]
    salida = MF.predecir(art, [{"ID_POLIGONO": i} for i in todos[C.KEY]])
    (a.out / "predicciones_ui.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    plano = pd.DataFrame([{C.KEY: o["ID_POLIGONO"], "CONJUNTO": todos.set_index(C.KEY).loc[o["ID_POLIGONO"], "CONJUNTO"],
                           "municipio": o["prediction_components"][1]["municipio"],
                           "prediction_t_ha": o["prediction_t_ha"], "lower_t_ha": o["uncertainty"]["lower_t_ha"],
                           "upper_t_ha": o["uncertainty"]["upper_t_ha"],
                           "media_global_t_ha": o["prediction_components"][0]["value_t_ha"],
                           "ajuste_municipal_t_ha": o["prediction_components"][1]["value_t_ha"],
                           "n_parcelas_municipio": o["prediction_components"][1]["n_parcelas_municipio"],
                           "warnings": " | ".join(w.split(":")[0] for w in o["warnings"])} for o in salida])
    plano.to_csv(a.out / "predicciones_modelo_final.csv", index=False)

    # 4) ejemplo de integracion UI
    pred59 = datos["pred"]
    ej_ids = [pred59.loc[pred59[C.MUN] == "Chignahuapan", C.KEY].iloc[0],
              pred59.loc[pred59[C.MUN] == "Nanacamilpa de Mariano Arista", C.KEY].iloc[0]]
    entrada = {"schema_version": "1.0",
               "descripcion": "Ejemplo de solicitud de la UI. Basta ID_POLIGONO; loc_municipio es opcional "
                              "(se usa si el ID no esta en el universo del reto). El tercer registro es ficticio y "
                              "solo ilustra el caso de municipio no visto.",
               "parcelas": [{"ID_POLIGONO": ej_ids[0]}, {"ID_POLIGONO": ej_ids[1]},
                            {"ID_POLIGONO": "AGC_EJEMPLO_FICTICIO", "loc_municipio": "MUNICIPIO_EJEMPLO_NO_VISTO"}]}
    (a.out / "ejemplo_entrada.json").write_text(json.dumps(entrada, indent=2, ensure_ascii=False), encoding="utf-8")
    (a.out / "ejemplo_salida.json").write_text(json.dumps(MF.predecir(art, entrada["parcelas"]), indent=2,
                                                           ensure_ascii=False), encoding="utf-8")
    # entregable del README 01: un objeto del contrato para una parcela PREDICCION
    (a.out / "ejemplo_prediccion.json").write_text(json.dumps(MF.predecir_parcela({"ID_POLIGONO": ej_ids[0]}, art),
                                                              indent=2, ensure_ascii=False), encoding="utf-8")

    # explicacion principal del modelo final: la referencia municipal (no SHAP)
    est = art["incertidumbre"]["estratos"]
    ref = pd.DataFrame([{"municipio": m, "n_parcelas_etiquetadas": v["n"], "media_municipal_t_ha": v["media_t_ha"],
                         "media_global_t_ha": art["media_global_t_ha"], "k_encogimiento": art["k_encogimiento"],
                         "peso_municipal": v["peso_municipal"],
                         "ajuste_municipal_t_ha": v["prediccion_t_ha"] - art["media_global_t_ha"],
                         "prediccion_t_ha": v["prediccion_t_ha"], "estrato_incertidumbre": v["estrato_incertidumbre"],
                         "semiancho_intervalo80_t_ha": est[v["estrato_incertidumbre"]]["q_abs_t_ha"],
                         "n_parcelas_PREDICCION": int((datos["pred"][C.MUN] == m).sum())}
                        for m, v in art["municipios"].items()]).sort_values("n_parcelas_etiquetadas", ascending=False)
    ref.to_csv(a.out / "explicacion_referencia_municipal.csv", index=False)
    auditoria = json.loads((a.out / "auditoria_datos.json").read_text(encoding="utf-8")) \
        if (a.out / "auditoria_datos.json").exists() else {"estado_global": "NO_EJECUTADA"}

    # 5) manifiesto
    comp = pd.read_csv(a.out / "comparacion_final.csv")
    principal = comp[comp.scheme == "RCV5x10"].set_index("modelo")
    try:
        import sklearn, scipy
        versiones = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}
        try:
            import shap
            versiones["shap"] = shap.__version__
        except ImportError:
            versiones["shap"] = None
    except ImportError:
        versiones = {}
    regla = json.loads((here / "regla_decision_preregistrada.json").read_text(encoding="utf-8"))
    archivos = sorted(p for p in a.out.rglob("*") if p.is_file() and p.name != "manifest_modelo.json"
                      and "__pycache__" not in p.parts)
    manifest = {
        "schema_version": "1.0", "model_id": MF.MODEL_ID, "model_version": MF.MODEL_VERSION,
        "generado_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decision": dec["decision"],
        "contrato": {"documento": "README 01 - Claude: Predictor final y explicabilidad del modelo",
                     "verificacion_punto_por_punto": "CUMPLIMIENTO_README.md",
                     "predicciones_oof_columnas": C.OOF_COLUMNAS,
                     "familias_contrato": C.FAMILIAS_CONTRATO},
        "auditoria_datos": {"archivo": "auditoria_datos.json", "estado_global": auditoria["estado_global"]},
        "modelo_final": {
            "tipo": art["tipo"], "formula": art["formula"], "variables": ["loc_municipio"],
            "k_encogimiento": art["k_encogimiento"], "media_global_t_ha": art["media_global_t_ha"],
            "n_entrenamiento": art["n_entrenamiento"], "artefacto": "modelo_final.json",
            "funcion_predictiva": "modelo_final.predecir_parcela(datos_parcela) -> objeto JSON 1.0; "
                                  "modelo_final.predecir(artefacto, registros) para lotes",
            "explicacion_principal": "explicacion_referencia_municipal.csv y prediction_components "
                                     "(media global + ajuste municipal); SHAP de RIDGE148 solo como analisis auxiliar "
                                     "en comparativo_ridge148/",
            "metricas_oof_RCV5x10": {
                "RMSE_media_reps": principal.loc["PRIOR", "RMSE_media_reps"], "RMSE_sd_reps": principal.loc["PRIOR", "RMSE_sd_reps"],
                "MAE_media_reps": principal.loc["PRIOR", "MAE_media_reps"], "R2_media_reps": principal.loc["PRIOR", "R2_media_reps"]},
            "incertidumbre": art["incertidumbre"],
        },
        "candidatos_experimentales": {
            "RIDGE148": {"variables": datos["cols148"], "familias": {c: C.familia_148(c) for c in datos["cols148"]},
                         "fuente": "data_processed/tabla_modelado_parcelaria.csv",
                         "pipeline": "SimpleImputer(median) -> StandardScaler -> RidgeCV(alphas=logspace(-2,4,13), cv=5)",
                         "estado": "EXPERIMENTAL, no adoptado", "criterios": dec["criterios"]["RIDGE148"]},
            "HIBRIDO_TOPO": {"variables": ["loc_municipio"] + C.TOPO4, "fuente": "data_processed/features_parcela.csv",
                             "pipeline": "PRIOR calibrado + RidgeCV(TOPO4) sobre residuo leave-self-out",
                             "estado": "EXPERIMENTAL, no adoptado", "criterios": dec["criterios"]["HIBRIDO_TOPO"]},
        },
        "fuentes": {k: {"ruta_relativa": "data_processed/" + Path(v).name, "sha256": datos["hashes"][k]}
                    for k, v in datos["rutas"].items()},
        "particiones": {
            "archivo": "splits.csv", "sha256": C.sha256(a.out / "splits.csv"),
            "esquema_columnas": "scheme,repeat,fold,ID_POLIGONO,role",
            "esquemas": {
                "RCV5x10": "PRINCIPAL. KFold(5, shuffle=True, random_state=1000..1009) sobre las 138 en orden de ID_POLIGONO",
                "RCV5_S2026": "Reproduccion: KFold(5, shuffle=True, random_state=2026) del experimento Ridge del equipo",
                "KM6": "Diagnostico: GroupKFold(6) sobre KMeans(6, n_init=10, random_state=0) en UTM",
                "LMO": "Diagnostico: leave-municipality-out"},
            "procedencia": "Regeneradas en esta sesion con las semillas documentadas de la auditoria previa; el archivo "
                           "original de la auditoria no estaba en el ZIP. Las metricas de PRIOR_K3 (0.5245) y PRIOR (0.5123) "
                           "coinciden con la auditoria a 4 decimales, lo que es consistente con particiones identicas, pero no "
                           "se pudo verificar byte a byte.",
            "info": dec["particiones"]},
        "configuracion": {"k_malla": art["k_malla"], "alphas_ridge": [float(x) for x in C.ALPHAS],
                          "rope_t_ha": C.ROPE, "bootstrap": "5000 remuestreos; 25 microclusteres KMeans(25, random_state=0) en UTM",
                          "prueba_t": "Nadeau-Bengio corregida, 50 folds, Holm sobre 2 candidatos",
                          "regla_decision": regla},
        "advertencias": [
            "Comparacion EXPLORATORIA: las 138 etiquetas y particiones equivalentes ya se usaron en analisis previos; no hay prueba independiente.",
            "Las 59 etiquetas ocultas no estan disponibles y no se usaron.",
            "El efecto municipal esta confundido con el regimen de registro del rendimiento.",
            "Sin etiquetas de varios ciclos: no se demostro generalizacion temporal.",
            "El intervalo de incertidumbre es empirico (80% nominal) y se valido solo dentro de RCV5x10 con las mismas 138 parcelas.",
            "SHAP y coeficientes de RIDGE148 son analisis comparativo de un modelo no adoptado; no explican la prediccion final.",
            "Municipios con n<10 (Apan, Emiliano Zapata, Nanacamilpa, Tepeapulco, Cuautepec) tienen medias inestables.",
        ],
        "pruebas_ejecutadas": {
            "reproduccion_experimento_ridge_equipo_RCV5_S2026": {
                "RIDGE148_RMSE": float(comp[(comp.scheme == "RCV5_S2026") & (comp.modelo == "RIDGE148")].RMSE_media_reps.iloc[0]),
                "PRIOR_K3_RMSE": float(comp[(comp.scheme == "RCV5_S2026") & (comp.modelo == "PRIOR_K3")].RMSE_media_reps.iloc[0]),
                "referencia_equipo": {"RIDGE148": 0.4980, "Media_municipal_suavizada_k3": 0.5328}},
            "modelo_final_vs_PRIOR_fase1_max_abs_diff": max_diff,
        },
        "entorno": versiones,
        "archivos_salida_sha256": {str(p.relative_to(a.out)).replace("\\", "/"): C.sha256(p) for p in archivos},
    }
    C.escribir_json(a.out / "manifest_modelo.json", manifest)
    print(cob.round(3).to_string(index=False))
    print("max |modelo_final - PRIOR fase1| =", max_diff)
    print(json.dumps(MF.predecir(art, entrada["parcelas"])[:2], indent=1, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
