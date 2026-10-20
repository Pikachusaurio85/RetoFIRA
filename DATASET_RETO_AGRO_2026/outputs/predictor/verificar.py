"""Pruebas de integridad y consistencia del modulo predictor. Escribe verificacion.json y termina con
codigo 1 si alguna prueba falla.

Uso:  python verificar.py [--base RUTA/cebada_datos_baseline_1] [--out RUTA/outputs/predictor]
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

CONTRATO = ["schema_version", "ID_POLIGONO", "model_id", "prediction_t_ha", "uncertainty",
            "prediction_components", "warnings"]


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=here.parents[1])
    ap.add_argument("--out", type=Path, default=here)
    a = ap.parse_args(argv)
    res = {}

    def prueba(nombre, ok, detalle=None):
        res[nombre] = {"resultado": "OK" if ok else "FALLA", "detalle": detalle}

    datos = C.cargar_datos(a.base)
    lab, pred = datos["lab"], datos["pred"]
    ids138, ids59 = set(lab[C.KEY]), set(pred[C.KEY])
    man = json.loads((a.out / "manifest_modelo.json").read_text(encoding="utf-8"))

    # 1. hashes de tablas fuente
    prueba("hash_tablas_fuente", all(man["fuentes"][k]["sha256"] == v for k, v in datos["hashes"].items()),
           datos["hashes"])

    # 2. arbol original intacto (si se incluye la lista de referencia)
    ref = here / "integridad_fuentes_sha256.csv"
    if ref.exists():
        r = pd.read_csv(ref)
        malos = [p for p, h in zip(r.ruta_relativa, r.sha256)
                 if not (Path(a.base) / p).exists() or C.sha256(Path(a.base) / p) != h]
        prueba("arbol_cebada_datos_baseline_1_sin_cambios", not malos,
               {"archivos_revisados": len(r), "distintos_o_ausentes": malos[:20]})

    # 3. particiones
    sp = pd.read_csv(a.out / "splits.csv")
    ok = list(sp.columns) == ["scheme", "repeat", "fold", C.KEY, "role"] and set(sp.role) == {"train", "test"}
    ok &= set(sp[C.KEY]) == ids138 and not (set(sp[C.KEY]) & ids59)
    for _, g in sp.groupby(["scheme", "repeat", "fold"]):
        tr, te = set(g.loc[g.role == "train", C.KEY]), set(g.loc[g.role == "test", C.KEY])
        ok &= not (tr & te) and (tr | te) == ids138
    ok &= sp[sp.role == "test"].groupby(["scheme", "repeat"])[C.KEY].apply(lambda s: s.is_unique and set(s) == ids138).all()
    prueba("particiones_validas", bool(ok), sp.groupby("scheme")[["repeat", "fold"]].nunique().to_dict())

    # 0. auditoria de datos
    aud = a.out / "auditoria_datos.json"
    estado = json.loads(aud.read_text(encoding="utf-8"))["estado_global"] if aud.exists() else "NO_EJECUTADA"
    prueba("auditoria_datos_OK", estado == "OK", estado)

    # 4-5. OOF: columnas del contrato, completo, etiquetas correctas, coincide con splits
    crudo = pd.read_csv(a.out / "predicciones_oof.csv", nrows=5)
    minimas = ["schema_version", "model_id", "scheme", "repeat", "fold", C.KEY, "y_true_t_ha", "y_pred_t_ha"]
    prueba("oof_columnas_contrato", list(crudo.columns[:8]) == minimas and (crudo.schema_version.astype(str) == "1.0").all(),
           list(crudo.columns))
    oof = C.leer_oof(a.out / "predicciones_oof.csv")
    y = lab.set_index(C.KEY)[C.TARGET]
    ok = (not (set(oof[C.KEY]) & ids59)) and np.allclose(oof.y_obs, oof[C.KEY].map(y))
    ok &= oof.groupby(["scheme", "repeat", "modelo"])[C.KEY].apply(lambda s: s.is_unique and set(s) == ids138).all()
    tst = sp[sp.role == "test"].set_index(["scheme", "repeat", C.KEY])["fold"]
    ok &= (oof.set_index(["scheme", "repeat", C.KEY])["fold"] == tst.reindex(oof.set_index(["scheme", "repeat", C.KEY]).index)).all()
    prueba("oof_completo_y_coherente_con_splits", bool(ok), {"filas": len(oof)})

    # 6. decision reproducible desde los contrastes
    cont = pd.read_csv(a.out / "contrastes_pareados.csv")
    mun = pd.read_csv(a.out / "comparacion_por_municipio.csv")
    dec = json.loads((a.out / "decision_fase1.json").read_text(encoding="utf-8"))
    cumple = {}
    for cand in ["RIDGE148", "HIBRIDO_TOPO"]:
        r = cont[(cont.scheme == "RCV5x10") & (cont.modelo_a == cand) & (cont.modelo_b == "PRIOR")].iloc[0]
        mm = mun[(mun.scheme == "RCV5x10") & (mun.modelo == cand) & mun.municipio.isin(C.MUNICIPIOS_N10)]
        cumple[cand] = bool(r.IC95_cluster_hi < 0 and r.NB_p_holm < 0.05 and r.reps_a_mejor >= 9
                            and (mm.dRMSE_vs_PRIOR <= 0).sum() >= 3)
    esperado = ("ADOPTAR PRIOR" if not any(cumple.values()) else
                "ADOPTAR HIBRIDO" if cumple["HIBRIDO_TOPO"] and not cumple["RIDGE148"] else
                "ADOPTAR RIDGE" if cumple["RIDGE148"] and not cumple["HIBRIDO_TOPO"] else "AMBOS")
    prueba("decision_coincide_con_regla", esperado == dec["decision"] or esperado == "AMBOS",
           {"esperado": esperado, "registrado": dec["decision"], "cumple": cumple})

    # 7. SHAP: reconstruccion y coincidencia con OOF
    ruta_shap = a.out / ("explicaciones_shap.csv" if dec["decision"] == "ADOPTAR RIDGE"
                         else "comparativo_ridge148/explicaciones_shap.csv")
    if ruta_shap.exists():
        sh = pd.read_csv(ruta_shap)
        g = sh.groupby(["contexto", "fold", C.KEY]).agg(s=("shap_t_ha", "sum"), b=("valor_base_t_ha", "first"),
                                                        p=("prediccion_t_ha", "first"), n=("feature", "size"))
        err = float(np.max(np.abs(g.b + g.s - g.p)))
        o0 = oof[(oof.scheme == "RCV5x10") & (oof.repeat == 0) & (oof.modelo == "RIDGE148")].set_index(C.KEY).y_pred
        g0 = g.loc["OOF_RCV5x10_rep0"].reset_index().set_index(C.KEY).p
        err_oof = float(np.max(np.abs(g0 - o0.loc[g0.index])))
        prueba("shap_reconstruye_prediccion", err < 1e-9 and err_oof < 1e-9 and (g.n == 148).all(),
               {"max_abs_base_mas_shap_menos_pred": err, "max_abs_vs_oof_fase1": err_oof,
                "etiqueta_modelo": sh.modelo.unique().tolist()})
        prueba("shap_etiquetado_como_comparativo", dec["decision"] == "ADOPTAR RIDGE"
               or set(sh.modelo) == {"RIDGE148_COMPARATIVO_NO_ADOPTADO"}, sh.modelo.unique().tolist())

    # 8. contrato JSON
    fallos = []
    for nombre in ["predicciones_ui.json", "ejemplo_salida.json", "ejemplo_prediccion.json"]:
        objs = json.loads((a.out / nombre).read_text(encoding="utf-8"))
        for o in (objs if isinstance(objs, list) else [objs]):
            if list(o) != CONTRATO or o["schema_version"] != "1.0" or o["model_id"] != "modelo_final":
                fallos.append((nombre, o.get("ID_POLIGONO"), "claves"))
                continue
            if set(o["uncertainty"]) != {"method", "lower_t_ha", "upper_t_ha"}:
                fallos.append((nombre, o["ID_POLIGONO"], "uncertainty"))
            if o["prediction_t_ha"] is not None:
                s = sum(c["value_t_ha"] for c in o["prediction_components"])
                if abs(s - o["prediction_t_ha"]) > 1e-4:
                    fallos.append((nombre, o["ID_POLIGONO"], "componentes"))
                u = o["uncertainty"]
                if u["lower_t_ha"] is None or u["upper_t_ha"] is None:
                    if u["method"] != "pending_validation" or u["lower_t_ha"] is not None or u["upper_t_ha"] is not None:
                        fallos.append((nombre, o["ID_POLIGONO"], "intervalo_parcial"))
                elif not u["lower_t_ha"] <= o["prediction_t_ha"] <= u["upper_t_ha"]:
                    fallos.append((nombre, o["ID_POLIGONO"], "intervalo"))
    prueba("contrato_json_1_0", not fallos, fallos[:10])
    ej = json.loads((a.out / "ejemplo_salida.json").read_text(encoding="utf-8"))
    no_visto = [o for o in ej if any(w.startswith("MUNICIPIO_NO_VISTO") for w in o["warnings"])]
    prueba("sin_intervalo_no_validado", bool(no_visto) and all(o["uncertainty"]["method"] == "pending_validation"
                                                               for o in no_visto), [o["uncertainty"] for o in no_visto])

    # 9. prediccion final recalculada de forma independiente
    art = json.loads((a.out / "modelo_final.json").read_text(encoding="utf-8"))
    k, G = art["k_encogimiento"], lab[C.TARGET].mean()
    agg = lab.groupby(C.MUN)[C.TARGET].agg(["sum", "count"])
    esperado = pred[C.MUN].map((agg["sum"] + k * G) / (agg["count"] + k)).fillna(G)
    pf = pd.read_csv(a.out / "predicciones_modelo_final.csv").set_index(C.KEY)
    err = float(np.max(np.abs(pf.loc[pred[C.KEY], "prediction_t_ha"].to_numpy() - esperado.to_numpy())))
    prueba("prediccion_final_recalculada", err < 1e-4 and len(pf) == 197, {"max_abs": err})

    # 10. ninguna salida con y_obs contiene parcelas PREDICCION
    malos = []
    for p in a.out.rglob("*.csv"):
        cols = pd.read_csv(p, nrows=0).columns
        if ({"y_obs", "y_true_t_ha"} & set(cols)) and C.KEY in cols and set(pd.read_csv(p, usecols=[C.KEY])[C.KEY]) & ids59:
            malos.append(str(p.relative_to(a.out)))
    prueba("sin_rendimiento_para_parcelas_PREDICCION", not malos, malos)

    # 11. hashes del manifiesto
    distintos = [f for f, h in man["archivos_salida_sha256"].items()
                 if f not in ("verificacion.json",) and (not (a.out / f).exists() or C.sha256(a.out / f) != h)]
    prueba("manifiesto_hashes_salidas", not distintos, distintos[:20])

    C.escribir_json(a.out / "verificacion.json", res)
    for k_, v in res.items():
        print(f"{v['resultado']:5s} {k_}")
    sys.exit(0 if all(v["resultado"] == "OK" for v in res.values()) else 1)


if __name__ == "__main__":
    main()
