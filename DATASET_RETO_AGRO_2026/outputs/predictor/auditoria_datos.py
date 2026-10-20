"""Auditoria de datos (README 01 - Claude, Fase 1). Solo lectura: no modifica ninguna tabla.

Verifica identidad de parcelas, etiquetas, columnas y hashes; comprueba que tabla_modelado_parcelaria.csv
contiene las 148 caracteristicas que usaron los experimentos previos (Ridge, OLS, RF, ET) y compara los
datos centralizados en data_processed/ con los que consumian los scripts originales.

Uso:  python auditoria_datos.py [--base RUTA/cebada_datos_baseline_1] [--out RUTA/outputs/predictor]
Salidas: auditoria_datos.json (resumen de comprobaciones) y auditoria_datos_tablas.csv (inventario con hashes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import comun as C

TABLAS = ["features_parcela", "tabla_modelado_parcelaria", "parcelas_base", "clima_mensual_parcela",
          "series_dekadales_2025", "diccionario_features"]
CLAVES = {"features_parcela": [C.KEY], "tabla_modelado_parcelaria": [C.KEY], "parcelas_base": [C.KEY],
          "clima_mensual_parcela": [C.KEY, "anio", "mes"], "series_dekadales_2025": [C.KEY, "sensor", "variable", "dekada"],
          "diccionario_features": ["columna"]}
EXPERIMENTOS_148 = {
    "Ridge_148": "modelos_experimentales/Ridge_model/resultados_ridge_agrocebada/coeficientes_Ridge_148_historicas.csv",
    "Lineal_148": "modelos_experimentales/LinearRegresion_model/resultados_regresion_lineal_agro/coeficientes_Lineal_148_historicas.csv",
    "RF_148": "modelos_experimentales/RandomForest_model/resultados_random_forest_agro/importancias_RF_148_historicas.csv",
    "ET_148": "modelos_experimentales/ExtraTrees_model/resultados_extra_trees_agro/importancias_ET_148_historicas.csv",
}


def leer(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, encoding="utf-8-sig")


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=here.parents[1])
    ap.add_argument("--out", type=Path, default=here)
    a = ap.parse_args(argv)
    base = Path(a.base).resolve()
    desarrollo = base.parent
    dp = base / "data_processed"
    checks = {}

    def chk(nombre, estado, detalle):
        checks[nombre] = {"estado": estado, "detalle": detalle}

    # ---------------- A. inventario y hashes
    inv, t = [], {}
    for n in TABLAS:
        p = dp / f"{n}.csv"
        if not p.exists():
            inv.append({"tabla": n, "ruta_relativa": f"data_processed/{n}.csv", "existe": False})
            continue
        t[n] = leer(p)
        d = t[n]
        ids = d[C.KEY] if C.KEY in d.columns else None
        clave = [c for c in CLAVES[n] if c in d.columns]
        inv.append({"tabla": n, "ruta_relativa": f"data_processed/{n}.csv", "existe": True, "sha256": C.sha256(p),
                    "filas": len(d), "columnas": d.shape[1],
                    "parcelas_unicas": int(ids.nunique()) if ids is not None else None,
                    "clave": "+".join(clave), "filas_con_clave_duplicada": int(d.duplicated(subset=clave).sum()),
                    "celdas_faltantes": int(d.isna().sum().sum())})
    inv = pd.DataFrame(inv)
    inv.to_csv(a.out / "auditoria_datos_tablas.csv", index=False)
    chk("A_tablas_presentes_y_claves_unicas", "OK" if inv.existe.all() and (inv.filas_con_clave_duplicada.fillna(0) == 0).all()
        else "DIFERENCIA", {"ausentes": inv.loc[~inv.existe, "tabla"].tolist(),
                            "claves_duplicadas": inv.set_index("tabla")["filas_con_clave_duplicada"].to_dict()})

    # ---------------- B. identidad de parcelas entre tablas
    f, tm = t["features_parcela"], t["tabla_modelado_parcelaria"]
    universo = set(f[C.KEY])
    conjuntos = {n: set(t[n][C.KEY]) for n in t if C.KEY in t[n].columns}
    difs = {n: {"faltan": sorted(universo - s)[:10], "sobran": sorted(s - universo)[:10]}
            for n, s in conjuntos.items() if s != universo}
    anchas_dup = {n: int(t[n][C.KEY].duplicated().sum()) for n in ["features_parcela", "tabla_modelado_parcelaria", "parcelas_base"]}
    chk("B_identidad_parcelas", "OK" if not difs and len(universo) == 197 and not any(anchas_dup.values()) else "DIFERENCIA",
        {"n_universo": len(universo), "tablas_con_otro_universo": difs, "duplicados_en_tablas_anchas": anchas_dup})

    # ---------------- C. etiquetas y particion contra la fuente oficial
    fuente = desarrollo / "datos" / "entradas" / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv"
    if fuente.exists():
        of = leer(fuente)
        of = of.loc[:, ~of.columns.str.startswith("Unnamed")]
        m = f[[C.KEY, "CONJUNTO", C.TARGET]].merge(of, on=C.KEY, how="outer", suffixes=("", "_oficial"), indicator=True)
        y_ok = np.allclose(m[C.TARGET], m[C.TARGET + "_oficial"], equal_nan=True)
        conj_ok = (m["CONJUNTO"] == m["CONJUNTO_oficial"]).all()
        oficial_59_sin_y = of.loc[of.CONJUNTO == "PREDICCION", C.TARGET].isna().all()
        tm_ok = np.allclose(f.set_index(C.KEY)[C.TARGET], tm.set_index(C.KEY).loc[f[C.KEY], C.TARGET], equal_nan=True) \
            and (f.set_index(C.KEY)["CONJUNTO"] == tm.set_index(C.KEY).loc[f[C.KEY], "PARTICION"]).all()
        pb = t["parcelas_base"].set_index(C.KEY).loc[f[C.KEY]]
        pb_ok = np.allclose(pb[C.TARGET].to_numpy(), f[C.TARGET].to_numpy(), equal_nan=True) and (pb["CONJUNTO"].to_numpy() == f["CONJUNTO"].to_numpy()).all()
        area_tm = tm.set_index(C.KEY).loc[of[C.KEY], "AREA_HA"].to_numpy()
        ok = all([(m["_merge"] == "both").all(), y_ok, conj_ok, oficial_59_sin_y, tm_ok, pb_ok])
        chk("C_etiquetas_vs_fuente_oficial", "OK" if ok else "DIFERENCIA", {
            "fuente": "desarrollo/datos/entradas/ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
            "sha256_fuente": C.sha256(fuente),
            "mismas_197_parcelas": bool((m["_merge"] == "both").all()),
            "rendimiento_identico_features_parcela": bool(y_ok), "conjunto_identico": bool(conj_ok),
            "rendimiento_y_particion_identicos_en_tabla_modelado": bool(tm_ok),
            "rendimiento_y_conjunto_identicos_en_parcelas_base": bool(pb_ok),
            "n_entrenamiento": int((of.CONJUNTO == "ENTRENAMIENTO").sum()), "n_prediccion": int((of.CONJUNTO == "PREDICCION").sum()),
            "prediccion_sin_rendimiento_en_fuente": bool(oficial_59_sin_y),
            "AREA_HA_tabla_modelado_igual_a_fuente": bool(np.allclose(area_tm, of["AREA_HA"].to_numpy())),
        })
    else:
        chk("C_etiquetas_vs_fuente_oficial", "NO_DISPONIBLE", {"fuente_esperada": str(fuente)})

    # ---------------- D. las 148 caracteristicas
    cols148 = [c for c in tm.columns if c not in (C.KEY, "PARTICION", C.TARGET, "AREA_HA")]
    prohibidas = [c for c in cols148 if c in (C.KEY, "PARTICION", "CONJUNTO", C.TARGET) or "RENDIMIENTO" in c.upper()
                  or "PARTICION" in c.upper() or "ID_POLIGONO" in c.upper()]
    exp = {}
    for nombre, rel in EXPERIMENTOS_148.items():
        p = base / rel
        if p.exists():
            v = set(leer(p)["variable"])
            exp[nombre] = {"n": len(v), "igual_al_conjunto_148": v == set(cols148),
                           "faltan": sorted(set(cols148) - v)[:5], "sobran": sorted(v - set(cols148))[:5]}
        else:
            exp[nombre] = {"n": None, "igual_al_conjunto_148": None, "nota": "archivo no disponible"}
    fam = pd.Series([C.familia_148(c) for c in cols148]).value_counts().to_dict()
    sub = tm.loc[tm.PARTICION.isin(["ENTRENAMIENTO", "PREDICCION"]), cols148]
    constantes = [c for c in cols148 if sub[c].nunique(dropna=True) <= 1]
    ok = len(cols148) == 148 and not prohibidas and all(e["igual_al_conjunto_148"] in (True, None) for e in exp.values()) \
        and int(sub.isna().sum().sum()) == 0 and not constantes
    chk("D_148_caracteristicas", "OK" if ok else "DIFERENCIA", {
        "n_columnas": len(cols148), "excluidas_por_diseno": [C.KEY, "PARTICION", C.TARGET, "AREA_HA"],
        "columnas_prohibidas_detectadas": prohibidas, "coinciden_con_experimentos_previos": exp,
        "variables_por_familia": fam, "faltantes_197_parcelas": int(sub.isna().sum().sum()),
        "columnas_constantes": constantes, "todas_numericas": bool(all(pd.api.types.is_numeric_dtype(tm[c]) for c in cols148))})

    # ---------------- E. datos centralizados vs datos que consumian los scripts originales
    pares = [
        ("tabla_modelado_parcelaria (Ridge/OLS/RF/ET leian desarrollo/salidas/)",
         dp / "tabla_modelado_parcelaria.csv", desarrollo / "salidas" / "tabla_modelado_parcelaria.csv"),
        ("fuente de etiquetas (copia en salidas/originales)",
         desarrollo / "datos" / "entradas" / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
         desarrollo / "salidas" / "originales" / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv"),
    ]
    comp = []
    for desc, p1, p2 in pares:
        comp.append({"comparacion": desc, "centralizado": str(p1.relative_to(desarrollo)).replace("\\", "/"),
                     "consumido": str(p2.relative_to(desarrollo)).replace("\\", "/") if p2.exists() else "NO_DISPONIBLE",
                     "identicos_byte_a_byte": bool(p2.exists() and C.sha256(p1) == C.sha256(p2))})
    comp.append({"comparacion": "features_parcela (los scripts Ridge/OLS/RF/ET y 08-11 leian la misma ruta data_processed/)",
                 "centralizado": "cebada_datos_baseline_1/data_processed/features_parcela.csv",
                 "consumido": "cebada_datos_baseline_1/data_processed/features_parcela.csv", "identicos_byte_a_byte": True})
    rep = base / "modelos_experimentales/Ridge_model/resultados_ridge_agrocebada/metricas_ridge.csv"
    reproduccion = None
    if rep.exists():
        # re-ejecuta el protocolo del experimento Ridge del equipo (KFold 5, shuffle, random_state=2026) con las tablas centralizadas
        from sklearn.model_selection import KFold
        datos = C.cargar_datos(base)
        lab = datos["lab"]
        pr, pk = np.zeros(len(lab)), np.zeros(len(lab))
        for tr, te in KFold(5, shuffle=True, random_state=C.SEED_RIDGE_EXPERIMENTO).split(lab):
            pr[te] = C.ModeloRidge148(datos["cols148"]).fit(lab.iloc[tr]).predict(lab.iloc[te])
            pk[te] = C.ModeloPrior(k=3.0).fit(lab.iloc[tr]).predict(lab.iloc[te])
        eq = pd.read_csv(rep)
        eq = eq[eq.escenario == "aleatoria_5_folds"].set_index("modelo")["RMSE_t_ha"]
        y = lab[C.TARGET].to_numpy()
        reproduccion = {"Ridge_148_equipo": float(eq["Ridge_148_historicas"]), "Ridge_148_aqui": C.rmse(y, pr),
                        "media_municipal_k3_equipo": float(eq["Media_municipal_suavizada"]), "PRIOR_K3_aqui": C.rmse(y, pk)}
        reproduccion["reproduce_exacto"] = bool(abs(reproduccion["Ridge_148_equipo"] - reproduccion["Ridge_148_aqui"]) < 1e-9
                                                and abs(reproduccion["media_municipal_k3_equipo"] - reproduccion["PRIOR_K3_aqui"]) < 1e-9)
    notas = [
        "El README historico de cebada_datos_baseline_1 menciona data_processed/calidad_observaciones.csv y archivos en reports/; "
        "no estan en el ZIP (reports/ y figures/ vacios). No se usan en este modulo.",
        "Las 148 variables 'HISTORICO_PROMEDIO__' no estan descritas en data_processed/diccionario_features.csv "
        "(que documenta features_parcela); su definicion esta en desarrollo/salidas/diccionario_features_comparacion.csv y en analisis/data.py.",
        "AREA_HA (fuente oficial) y geo_area_ha (geometria) son distintas columnas; ninguna se usa en el predictor final.",
    ]
    dic_comp = desarrollo / "salidas" / "diccionario_features_comparacion.csv"
    if dic_comp.exists():
        dc = set(leer(dic_comp)["feature"])
        cubiertas = sum((c in dc) or (f"DESARROLLO__{c}" in dc) for c in cols148)
        notas.append(f"diccionario_features_comparacion.csv documenta {cubiertas} de las 148 variables historicas.")
    chk("E_centralizado_vs_consumido", "OK" if all(c["identicos_byte_a_byte"] for c in comp)
        and (reproduccion is None or reproduccion["reproduce_exacto"]) else "DIFERENCIA",
        {"comparaciones": comp, "reproduccion_metricas_equipo": reproduccion, "notas": notas})

    # ---------------- F. municipio y variables del predictor final
    pb = t["parcelas_base"].set_index(C.KEY).loc[f[C.KEY]]
    mun_ok = (pb["Municipio"].astype(str).str.strip().to_numpy() == f[C.MUN].astype(str).str.strip().to_numpy()).all()
    chk("F_municipio_consistente", "OK" if mun_ok and f[C.MUN].notna().all() else "DIFERENCIA", {
        "loc_municipio_igual_a_parcelas_base_Municipio": bool(mun_ok), "faltantes": int(f[C.MUN].isna().sum()),
        "municipios": int(f[C.MUN].nunique()),
        "municipios_PREDICCION_sin_etiquetadas": sorted(set(f.loc[f.CONJUNTO == "PREDICCION", C.MUN]) -
                                                       set(f.loc[f.CONJUNTO == "ENTRENAMIENTO", C.MUN]))})

    resumen = {"estado_global": "OK" if all(v["estado"] == "OK" for v in checks.values()) else "REVISAR",
               "transformaciones_aplicadas_a_tablas": "ninguna (solo lectura; cruces por ID_POLIGONO)",
               "comprobaciones": checks}
    C.escribir_json(a.out / "auditoria_datos.json", resumen)
    for k, v in checks.items():
        print(f"{v['estado']:13s} {k}")
    print("estado_global:", resumen["estado_global"])


if __name__ == "__main__":
    main()
