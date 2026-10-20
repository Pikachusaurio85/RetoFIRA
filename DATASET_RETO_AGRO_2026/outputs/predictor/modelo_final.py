"""AgroCebada 2026 - Predictor final: PRIOR MUNICIPAL CALIBRADO (decision Fase 1: ADOPTAR PRIOR).

    prediccion(m) = (S_m + k * G) / (n_m + k)     # S_m, n_m: suma y numero de rendimientos del municipio m
                  = G + w_m * (media_m - G),  w_m = n_m / (n_m + k)
    municipio sin parcelas etiquetadas -> G (media global)

La prediccion solo usa el municipio. No explica diferencias entre parcelas de un mismo municipio.

Uso como modulo (la prediccion no requiere numpy/pandas/sklearn):
    from modelo_final import predecir_parcela
    predecir_parcela({"ID_POLIGONO": "AGC_011"})                      # objeto del contrato JSON 1.0
    predecir_parcela({"loc_municipio": "Chignahuapan"})              # tambien acepta solo el municipio

Uso por linea de comandos:
    python modelo_final.py --entrenar [--base RUTA/cebada_datos_baseline_1]     # regenera modelo_final.json
    python modelo_final.py --ids AGC_003 AGC_010
    python modelo_final.py --entrada ejemplo_entrada.json --salida salida.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ARTEFACTO = AQUI / "modelo_final.json"
SCHEMA_VERSION = "1.0"
MODEL_ID = "modelo_final"
MODEL_VERSION = "prior_municipal_calibrado_v1"
NIVEL_INTERVALO = 0.80
N_MIN_ESTRATO = 10
METODO_INCERTIDUMBRE = "residuos_LOO_por_municipio_q80"

ADVERTENCIAS_GENERALES = [
    "PRIOR_MUNICIPAL: la estimacion es la media municipal encogida; no distingue parcelas dentro del mismo municipio.",
    "REGIMEN_CAPTURA: el nivel municipal puede reflejar como se registro el rendimiento (p. ej. valores en multiplos de 0.5 t/ha) ademas de diferencias agronomicas; no se puede separar con los datos actuales.",
    "CICLO_NO_VERIFICADO: las etiquetas no indican ano o ciclo agricola; la estimacion no esta validada para 2026.",
]


# --------------------------------------------------------------------------- entrenamiento
def _cuantil_conformal(valores: list[float], nivel: float) -> float:
    """Cuantil de muestra finita ceil((n+1)*nivel)/n (metodo 'higher')."""
    v = sorted(valores)
    n = len(v)
    pos = min(n, math.ceil((n + 1) * nivel))
    return float(v[pos - 1])


def ajustar(lab, base_meta: dict | None = None) -> dict:
    """Ajusta el prior con un DataFrame de parcelas etiquetadas (columnas ID_POLIGONO, loc_municipio, RENDIMIENTO_T_HA).

    k se elige con error leave-one-out solo sobre estas filas; los intervalos usan los residuos LOO.
    """
    sys.path.insert(0, str(AQUI))
    import numpy as np
    import comun as C

    mun = lab[C.MUN].astype(str).to_numpy()
    y = lab[C.TARGET].to_numpy(float)
    prior = C.ModeloPrior().fit(lab)
    k = prior.k_
    res_loo = np.abs(y - C.prior_loo(mun, y, k))
    municipios, estratos = {}, {}
    for m, r in prior.tabla_.iterrows():
        n = int(r["n"])
        estrato = m if n >= N_MIN_ESTRATO else "OTROS_n<10"
        municipios[m] = {"n": n, "media_t_ha": float(r["media"]), "peso_municipal": float(r["peso"]),
                         "prediccion_t_ha": float(r["prediccion"]), "estrato_incertidumbre": estrato}
    for e in sorted({v["estrato_incertidumbre"] for v in municipios.values()}):
        msk = np.array([municipios[m]["estrato_incertidumbre"] == e for m in mun])
        estratos[e] = {"n_residuos": int(msk.sum()), "q_abs_t_ha": _cuantil_conformal(res_loo[msk].tolist(), NIVEL_INTERVALO)}
    art = {
        "schema_version": SCHEMA_VERSION, "model_id": MODEL_ID, "model_version": MODEL_VERSION,
        "tipo": "prior_municipal_encogido",
        "formula": "pred = (S_m + k*G) / (n_m + k); municipio no visto -> G",
        "k_encogimiento": k, "k_malla": [float(x) for x in C.K_GRID],
        "k_criterio": "minimo error cuadratico leave-one-out en las parcelas etiquetadas",
        "media_global_t_ha": prior.media_global_, "n_entrenamiento": int(len(y)),
        "municipios": municipios,
        "incertidumbre": {"method": METODO_INCERTIDUMBRE, "nivel_nominal": NIVEL_INTERVALO,
                          "descripcion": "prediccion +- cuantil 80% (muestra finita) de |residuo leave-one-out| del prior; "
                                         "estrato = municipio si n>=10, si no OTROS_n<10. Es un intervalo de prediccion "
                                         "empirico, no un intervalo de confianza. Municipio no visto: sin intervalo "
                                         "(method=pending_validation, limites null).",
                          "estratos": estratos},
        "advertencias_generales": ADVERTENCIAS_GENERALES,
    }
    if base_meta:
        art.update(base_meta)
    return art


def entrenar(base: Path) -> dict:
    sys.path.insert(0, str(AQUI))
    import comun as C
    datos = C.cargar_datos(base)
    parcelas = {r[C.KEY]: {"municipio": str(r[C.MUN]), "conjunto": r["CONJUNTO"]}
                for _, r in datos["df"].iterrows()}
    meta = {"entrenado_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fuentes_sha256": datos["hashes"], "parcelas": parcelas,
            "nota_parcelas": "Mapa ID_POLIGONO -> municipio y particion oficial (sin rendimientos) para resolver solicitudes por ID."}
    return ajustar(datos["lab"], meta)


def guardar(art: dict, ruta: Path = ARTEFACTO) -> None:
    Path(ruta).write_text(json.dumps(art, indent=2, ensure_ascii=False), encoding="utf-8")


def cargar(ruta: Path = ARTEFACTO) -> dict:
    art = json.loads(Path(ruta).read_text(encoding="utf-8"))
    if art.get("model_id") != MODEL_ID or art.get("tipo") != "prior_municipal_encogido":
        raise ValueError("Artefacto no reconocido")
    return art


# --------------------------------------------------------------------------- prediccion
def predecir_uno(art: dict, registro: dict) -> dict:
    pid = registro.get("ID_POLIGONO")
    warnings = list(art["advertencias_generales"])
    info = art.get("parcelas", {}).get(pid, {}) if pid else {}
    municipio = registro.get("loc_municipio") or info.get("municipio")
    municipio = municipio.strip() if isinstance(municipio, str) else None
    if pid and not info:
        warnings.append("ID_DESCONOCIDO: el ID_POLIGONO no esta en el universo de 197 parcelas del reto.")
    if info and registro.get("loc_municipio") and registro["loc_municipio"].strip() != info["municipio"]:
        warnings.append(f"MUNICIPIO_DISCREPANTE: se recibio '{registro['loc_municipio']}' y el ID corresponde a "
                        f"'{info['municipio']}'; se usa el valor recibido.")
    G = art["media_global_t_ha"]
    est = art["incertidumbre"]["estratos"]
    if municipio is None:
        return {"schema_version": SCHEMA_VERSION, "ID_POLIGONO": pid, "model_id": MODEL_ID, "prediction_t_ha": None,
                "uncertainty": {"method": "pending_validation", "lower_t_ha": None, "upper_t_ha": None},
                "prediction_components": [],
                "warnings": warnings + ["SIN_MUNICIPIO: no se pudo determinar el municipio; no hay prediccion."]}
    m = art["municipios"].get(municipio)
    if m is None:
        pred, q = G, None
        comp_mun = {"component": "ajuste_municipal", "value_t_ha": 0.0, "municipio": municipio,
                    "n_parcelas_municipio": 0, "media_municipal_t_ha": None, "peso_municipal": 0.0,
                    "k_encogimiento": art["k_encogimiento"],
                    "description": "Municipio sin parcelas etiquetadas: sin ajuste."}
        warnings.append("MUNICIPIO_NO_VISTO: no hay parcelas etiquetadas en este municipio; se usa la media global. "
                        "Sin intervalo: no hay validacion para municipios nuevos (uncertainty = pending_validation).")
    else:
        pred, q = m["prediccion_t_ha"], est[m["estrato_incertidumbre"]]["q_abs_t_ha"]
        comp_mun = {"component": "ajuste_municipal", "value_t_ha": round(pred - G, 6), "municipio": municipio,
                    "n_parcelas_municipio": m["n"], "media_municipal_t_ha": round(m["media_t_ha"], 6),
                    "peso_municipal": round(m["peso_municipal"], 6), "k_encogimiento": art["k_encogimiento"],
                    "description": "peso_municipal * (media_municipal - media_global)"}
        if m["n"] < N_MIN_ESTRATO:
            warnings.append(f"MUNICIPIO_POCAS_PARCELAS: solo {m['n']} parcela(s) etiquetada(s) en {municipio}; "
                            "el intervalo usa el estrato agrupado OTROS_n<10.")
    if info.get("conjunto") == "ENTRENAMIENTO":
        warnings.append("PARCELA_ETIQUETADA: su propio rendimiento forma parte de la media municipal; "
                        "la estimacion es in-sample, no una evaluacion.")
    return {
        "schema_version": SCHEMA_VERSION, "ID_POLIGONO": pid, "model_id": MODEL_ID,
        "prediction_t_ha": round(pred, 4),
        "uncertainty": ({"method": art["incertidumbre"]["method"], "lower_t_ha": round(pred - q, 4),
                         "upper_t_ha": round(pred + q, 4)} if q is not None else
                        {"method": "pending_validation", "lower_t_ha": None, "upper_t_ha": None}),
        "prediction_components": [
            {"component": "media_global_entrenamiento", "value_t_ha": round(G, 6),
             "description": f"Media de las {art['n_entrenamiento']} parcelas etiquetadas"},
            comp_mun,
        ],
        "warnings": warnings,
    }


def predecir(art: dict, registros: list[dict]) -> list[dict]:
    return [predecir_uno(art, r) for r in registros]


_ART_CACHE: dict = {}


def predecir_parcela(datos_parcela: dict, artefacto: Path | str | dict | None = None) -> dict:
    """Funcion de integracion: recibe los datos de una parcela y devuelve el objeto del contrato JSON 1.0.

    datos_parcela: {"ID_POLIGONO": ...} y/o {"loc_municipio": ...}. El unico predictor usado es el municipio;
    si solo llega el ID, el municipio se resuelve con el mapa del artefacto (no se usa el ID como caracteristica).
    """
    if isinstance(artefacto, dict):
        art = artefacto
    else:
        ruta = str(artefacto or ARTEFACTO)
        if ruta not in _ART_CACHE:
            _ART_CACHE[ruta] = cargar(Path(ruta))
        art = _ART_CACHE[ruta]
    return predecir_uno(art, datos_parcela)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predictor final AgroCebada (prior municipal calibrado)")
    ap.add_argument("--entrenar", action="store_true", help="regenera modelo_final.json desde data_processed")
    ap.add_argument("--base", type=Path, default=AQUI.parents[1], help="carpeta cebada_datos_baseline_1")
    ap.add_argument("--artefacto", type=Path, default=ARTEFACTO)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--entrada", type=Path, help='JSON {"parcelas": [{"ID_POLIGONO": ...}, ...]}')
    ap.add_argument("--salida", type=Path)
    a = ap.parse_args(argv)
    if a.entrenar:
        guardar(entrenar(a.base), a.artefacto)
        print("Artefacto guardado en", a.artefacto)
        return
    art = cargar(a.artefacto)
    if a.entrada:
        registros = json.loads(a.entrada.read_text(encoding="utf-8"))["parcelas"]
    else:
        registros = [{"ID_POLIGONO": i} for i in (a.ids or [])]
    out = predecir(art, registros)
    txt = json.dumps(out, indent=2, ensure_ascii=False)
    if a.salida:
        a.salida.write_text(txt, encoding="utf-8")
    else:
        print(txt)


if __name__ == "__main__":
    main()
