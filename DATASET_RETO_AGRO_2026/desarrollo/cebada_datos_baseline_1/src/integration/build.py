"""Build: une predictor y XAI por ID_POLIGONO, valida y exporta los 3 JSON."""
from __future__ import annotations
import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    PARCELAS_JSON, PROTOTIPOS_JSON, MANIFESTO_INTEGRACION,
    MANIFESTO_PREDICTOR, MANIFESTO_FASE1, MANIFESTO_FASE2,
    PERFILES_CONCEPTUALES, PROTOTIPOS_CSV, SIMILITUDES_CSV,
    COMPONENT_UNITS, OFFICIAL_PROTOTYPE_IDS, XAI_FROZEN_CONFIG,
    PREDICCIONES_UI, MODELO_FINAL
)
from .predictor_adapter import PredictorAdapter, load_predictor
from .xai_adapter import XAIAdapter, load_xai


def build_prediccion(
    pred_adapter: PredictorAdapter,
    model_meta: dict[str, Any],
    id_poligono: str
) -> dict[str, Any]:
    pred_ui = pred_adapter.get_prediction(id_poligono)
    if not pred_ui:
        return {}
    
    municipio = pred_adapter.get_municipio(id_poligono)
    muni_comp = pred_adapter.get_municipal_component(municipio)
    
    comps = {c["component"]: c for c in pred_ui.get("prediction_components", [])}
    media_global_comp = comps.get("media_global_entrenamiento", {})
    ajuste_comp = comps.get("ajuste_municipal", {})
    
    explicacion = {
        "componentes": {
            "media_global": {
                "valor_t_ha": media_global_comp.get("value_t_ha"),
                "descripcion": media_global_comp.get("description", "")
            },
            "ajuste_municipal": {
                "valor_t_ha": ajuste_comp.get("value_t_ha"),
                "municipio": ajuste_comp.get("municipio", ""),
                "n_parcelas_municipio": ajuste_comp.get("n_parcelas_municipio"),
                "media_municipal_t_ha": ajuste_comp.get("media_municipal_t_ha"),
                "peso_municipal": ajuste_comp.get("peso_municipal"),
                "k_encogimiento": ajuste_comp.get("k_encogimiento"),
                "descripcion": ajuste_comp.get("description", "")
            }
        }
    }
    
    muni_comp_data = muni_comp or {}
    explicacion["contexto"] = {
        "media_global_t_ha": muni_comp_data.get("media_global_t_ha"),
        "media_municipal_t_ha": muni_comp_data.get("media_municipal_t_ha"),
        "n_parcelas_municipio": muni_comp_data.get("n_parcelas_etiquetadas"),
        "peso_municipal": muni_comp_data.get("peso_municipal"),
        "k_encogimiento": muni_comp_data.get("k_encogimiento"),
        "estrato_incertidumbre": muni_comp_data.get("estrato_incertidumbre"),
        "semiancho_intervalo80_t_ha": muni_comp_data.get("semiancho_intervalo80_t_ha"),
    }
    
    explicacion["prediction_components_raw"] = [
        {"component": c["component"], "value_t_ha": c.get("value_t_ha"), "description": c.get("description", "")}
        for c in pred_ui.get("prediction_components", [])
    ]
    
    return {
        "valor_t_ha": pred_ui.get("prediction_t_ha"),
        "incertidumbre": pred_ui.get("uncertainty", {}),
        "model_id": pred_ui.get("model_id"),
        "model_version": model_meta.get("model_version"),
        "explicacion": explicacion,
        "warnings": pred_ui.get("warnings", [])
    }


def build_perfil_conceptual(xai_adapter: XAIAdapter, id_poligono: str) -> dict[str, Any]:
    perfil = xai_adapter.get_perfil(id_poligono)
    if not perfil:
        return {"original": {}, "scaled": {}, "indicadores_calidad": {}}
    
    return {
        "original": perfil.get("original", {}),
        "scaled": perfil.get("scaled", {}),
        "indicadores_calidad": perfil.get("flags", {})
    }


def build_prototipos_cercanos(
    xai_adapter: XAIAdapter,
    id_poligono: str,
    perfiles_dict: dict[str, Any]
) -> list[dict[str, Any]]:
    similitudes = xai_adapter.get_similitudes(id_poligono)
    if not similitudes:
        return []
    
    protos = []
    for sim in similitudes:
        pid = sim["prototype_id"]
        rank = sim["local_similarity_rank"]
        
        # Contribuciones oficiales solo disponibles para rank 1 (proto mas similar).
        # Se asocian por nombre de componente; si el concepto no coincide, no se inventa valor.
        original_contribs = {}
        if rank == 1:
            for comp in sim.get("top_differentiating_components", []):
                original_contribs[comp["component"]] = comp.get("relative_contribution_to_distance")
        
        # Calcular diferencia escalada de los 5 componentes oficiales para esta pareja
        candidates = []
        for order_idx, comp_name in enumerate(XAI_FROZEN_CONFIG["components"]):
            parcel_orig = perfiles_dict.get(id_poligono, {}).get("original", {}).get(comp_name)
            parcel_scaled = perfiles_dict.get(id_poligono, {}).get("scaled", {}).get(comp_name)
            proto_orig = perfiles_dict.get(pid, {}).get("original", {}).get(comp_name)
            proto_scaled = perfiles_dict.get(pid, {}).get("scaled", {}).get(comp_name)
            
            abs_diff_orig = abs(parcel_orig - proto_orig) if parcel_orig is not None and proto_orig is not None else None
            abs_diff_scaled = abs(parcel_scaled - proto_scaled) if parcel_scaled is not None and proto_scaled is not None else None
            
            candidates.append((order_idx, comp_name, parcel_orig, parcel_scaled, proto_orig, proto_scaled, abs_diff_orig, abs_diff_scaled))
        
        # Top 3 por diff escalada descendente; desempate por orden en XAI_FROZEN_CONFIG["components"]
        candidates.sort(key=lambda c: (-c[7] if c[7] is not None else float("inf"), c[0]))
        top3 = candidates[:3]
        
        top_comps = []
        for (order_idx, comp_name, parcel_orig, parcel_scaled, proto_orig, proto_scaled, abs_diff_orig, abs_diff_scaled) in top3:
            # Contribución oficial del rank 1 solo si el componente coincide con el del CSV fuente
            rel_contrib = original_contribs.get(comp_name) if rank == 1 else None
            top_comps.append({
                "component": comp_name,
                "unit": COMPONENT_UNITS.get(comp_name, ""),
                "parcel_value_original": parcel_orig,
                "parcel_value_scaled": parcel_scaled,
                "prototype_value_original": proto_orig,
                "prototype_value_scaled": proto_scaled,
                "absolute_diff_original": abs_diff_orig,
                "absolute_diff_scaled": abs_diff_scaled,
                "relative_contribution_to_distance": rel_contrib
            })
        
        protos.append({
            "local_similarity_rank": sim["local_similarity_rank"],
            "global_selection_rank": sim.get("global_selection_rank"),
            "prototype_id": sim["prototype_id"],
            "prototype_municipio": sim["prototype_municipio"],
            "weight_global_representativeness_original": sim["weight_original"],
            "weight_global_representativeness_normalized": sim["weight_normalized"],
            "distance": sim["distance"],
            "similarity_rbf": sim["similarity_rbf"],
            "top_differentiating_components": top_comps
        })
    
    return protos


def _generate_data(order_ids: list[str] | None = None) -> tuple[dict, dict]:
    """Genera parcelas y prototipos en memoria. order_ids permite permutar el orden de iteración."""
    pred_adapter = load_predictor()
    xai_adapter = load_xai()
    
    model_meta = pred_adapter.get_model_meta()
    
    pred_ids = set(pred_adapter.get_all_ids())
    xai_ids = set(xai_adapter.get_all_ids())
    all_ids = sorted(pred_ids & xai_ids)
    
    # Si se proporciona un orden, usarlo (para test de invariancia)
    if order_ids is not None:
        # Verificar que contienen los mismos IDs
        assert set(order_ids) == set(all_ids), "order_ids debe contener los mismos IDs"
        all_ids = order_ids
    
    perfiles_dict = {}
    df_perfiles = pd.read_csv(PERFILES_CONCEPTUALES, encoding="utf-8-sig")
    for _, row in df_perfiles.iterrows():
        pid = row["ID_POLIGONO"]
        original = {}
        scaled = {}
        flags = {}
        for col in df_perfiles.columns:
            if col == "ID_POLIGONO":
                continue
            if col.endswith("_scaled"):
                base = col[:-7]
                scaled[base] = float(row[col]) if pd.notna(row[col]) else None
            elif col.startswith("flag_"):
                flags[col] = int(row[col]) if pd.notna(row[col]) else 0
            else:
                original[col] = float(row[col]) if pd.notna(row[col]) else None
        perfiles_dict[pid] = {"original": original, "scaled": scaled, "flags": flags}
    
    parcelas = {}
    for pid in all_ids:
        municipio = pred_adapter.get_municipio(pid)
        particion = pred_adapter.get_conjunto(pid)
        
        prediccion = build_prediccion(pred_adapter, model_meta, pid)
        perfil = build_perfil_conceptual(xai_adapter, pid)
        prototipos = build_prototipos_cercanos(xai_adapter, pid, perfiles_dict)
        warnings = pred_adapter.get_warnings(pid)
        
        parcelas[pid] = {
            "id": pid,
            "municipio": municipio,
            "particion": particion,
            "prediccion": prediccion,
            "perfil_conceptual": perfil,
            "prototipos_cercanos": prototipos,
            "warnings": warnings
        }
    
    prototipos_catalogo = {
        "version": "1.0",
        "configuracion": {
            "space": "compacto",
            "method": "ProtoDash",
            "m": 5,
            "sigma_factor": 1.0,
            "optimizer": "osqp",
            "components": [
                "C1a_ndvi_s2_mean",
                "C2a_amplitud_ndvi",
                "C3a_ndwi_s2_mean",
                "C4a_prec_acum_abr_oct",
                "C5a_elev_media"
            ]
        },
        "prototipos": []
    }
    
    for proto in xai_adapter.get_prototipos_oficiales():
        pid = proto["prototype_id"]
        prototipos_catalogo["prototipos"].append({
            "prototype_id": proto["prototype_id"],
            "global_selection_rank": proto["global_selection_rank"],
            "municipio": proto["municipio"],
            "particion": proto["particion"],
            "weight_original": proto["weight_original"],
            "weight_normalized": proto["weight_normalized"],
            "conceptual_original": {
                "C1a_ndvi_s2_mean": perfiles_dict.get(pid, {}).get("original", {}).get("C1a_ndvi_s2_mean"),
                "C2a_amplitud_ndvi": perfiles_dict.get(pid, {}).get("original", {}).get("C2a_amplitud_ndvi"),
                "C3a_ndwi_s2_mean": perfiles_dict.get(pid, {}).get("original", {}).get("C3a_ndwi_s2_mean"),
                "C4a_prec_acum_abr_oct": perfiles_dict.get(pid, {}).get("original", {}).get("C4a_prec_acum_abr_oct"),
                "C5a_elev_media": perfiles_dict.get(pid, {}).get("original", {}).get("C5a_elev_media"),
            },
            "conceptual_scaled": {
                "C1a_ndvi_s2_mean": perfiles_dict.get(pid, {}).get("scaled", {}).get("C1a_ndvi_s2_mean"),
                "C2a_amplitud_ndvi": perfiles_dict.get(pid, {}).get("scaled", {}).get("C2a_amplitud_ndvi"),
                "C3a_ndwi_s2_mean": perfiles_dict.get(pid, {}).get("scaled", {}).get("C3a_ndwi_s2_mean"),
                "C4a_prec_acum_abr_oct": perfiles_dict.get(pid, {}).get("scaled", {}).get("C4a_prec_acum_abr_oct"),
                "C5a_elev_media": perfiles_dict.get(pid, {}).get("scaled", {}).get("C5a_elev_media"),
            },
            "metadata": {
                "space": "compacto",
                "method": "ProtoDash",
                "m": 5,
                "sigma_factor": 1.0,
                "optimizer": "osqp",
                "global_selection_rank": proto["global_selection_rank"]
            }
        })
    
    return parcelas, prototipos_catalogo


def _build_manifest(model_meta: dict[str, Any]) -> dict[str, Any]:
    """Genera el manifest de integración."""
    def file_hash(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
        return h.hexdigest()
    
    return {
        "contract_version": "1.0",
        "generation_utc": datetime.utcnow().isoformat() + "Z",
        "predictor": {
            "model_id": "modelo_final",
            "model_version": model_meta.get("model_version"),
            "manifest_hash": file_hash(MANIFESTO_PREDICTOR),
            "predicciones_ui_hash": file_hash(PREDICCIONES_UI),
            "modelo_final_hash": file_hash(MODELO_FINAL),
        },
        "xai": {
            "config_frozen": {
                "space": "compacto",
                "method": "ProtoDash",
                "m": 5,
                "sigma_factor": 1.0,
                "optimizer": "osqp",
                "components": [
                    "C1a_ndvi_s2_mean",
                    "C2a_amplitud_ndvi",
                    "C3a_ndwi_s2_mean",
                    "C4a_prec_acum_abr_oct",
                    "C5a_elev_media"
                ]
            },
            "manifest_fase1_hash": file_hash(MANIFESTO_FASE1),
            "manifest_fase2_hash": file_hash(MANIFESTO_FASE2),
            "perfiles_conceptuales_hash": file_hash(PERFILES_CONCEPTUALES),
            "prototipos_hash": file_hash(PROTOTIPOS_CSV),
            "similitudes_hash": file_hash(SIMILITUDES_CSV),
        },
        "counts": {
            "n_parcelas": 197,
            "n_entrenamiento": 138,
            "n_prediccion": 59,
            "n_prototipos": 5
        },
        "sources": {
            "predictor": ["predicciones_ui.json", "modelo_final.json", "explicacion_referencia_municipal.csv", "manifest_modelo.json"],
            "xai": ["perfiles_conceptuales.csv", "fase2/prototipos.csv", "fase2/similitudes.csv", "manifest_xai.json", "fase2/manifest_fase2.json"]
        },
        "warnings": [
            "Predictor: prior municipal encogido; no distingue parcelas dentro del mismo municipio.",
            "Predictor: etiquetas sin año/ciclo; no validado para 2026.",
            "XAI: ProtoDash optimiza MMD (representatividad distributiva), no explicabilidad causal ni predictiva.",
            "XAI: Pesos ProtoDash miden representatividad distributiva, no importancia de variables ni contribucion a prediccion.",
            "XAI: Similitud RBF mide proximidad en espacio conceptual, no probabilidad de mismo rendimiento."
        ],
        "validation_status": "PASSED"
    }


def build():
    """Genera los 3 JSON, valida y publica solo si todas las validaciones pasan."""
    print("=== Integracion AgroCebada 2026 ===")
    
    pred_adapter = load_predictor()
    model_meta = pred_adapter.get_model_meta()
    
    print("Generando datos...")
    parcelas, prototipos_catalogo = _generate_data()
    manifest = _build_manifest(model_meta)
    
    print("Exportando JSONs a archivos temporales...")
    
    # Escribir a temporales primero
    tmp_parcelas = PARCELAS_JSON.with_suffix(".json.tmp")
    tmp_prototipos = PROTOTIPOS_JSON.with_suffix(".json.tmp")
    tmp_manifest = MANIFESTO_INTEGRACION.with_suffix(".json.tmp")
    
    with open(tmp_parcelas, "w", encoding="utf-8") as f:
        json.dump(parcelas, f, indent=2, ensure_ascii=False)
    
    with open(tmp_prototipos, "w", encoding="utf-8") as f:
        json.dump(prototipos_catalogo, f, indent=2, ensure_ascii=False)
    
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    # Validar antes de publicar (usando rutas temporales)
    from .validate import validate_all
    print("\nValidando antes de publicar...")
    errors = validate_all(
        parcelas_path=tmp_parcelas,
        prototipos_path=tmp_prototipos,
        manifest_path=tmp_manifest
    )
    
    if errors:
        print(f"\n❌ VALIDACIÓN FALLÓ ({len(errors)} errores). No se publican los archivos.")
        for e in errors:
            print(f"  - {e}")
        # Limpiar temporales
        for tmp in [tmp_parcelas, tmp_prototipos, tmp_manifest]:
            if tmp.exists():
                tmp.unlink()
        raise RuntimeError(f"Validación falló con {len(errors)} errores. Archivos no publicados.")
    
    # Si todo OK, mover temporales a destino final
    os.replace(tmp_parcelas, PARCELAS_JSON)
    os.replace(tmp_prototipos, PROTOTIPOS_JSON)
    os.replace(tmp_manifest, MANIFESTO_INTEGRACION)
    
    print(f"  {PARCELAS_JSON}")
    print(f"  {PROTOTIPOS_JSON}")
    print(f"  {MANIFESTO_INTEGRACION}")
    
    print("\n=== Integracion completada ===")


if __name__ == "__main__":
    build()
