#!/usr/bin/env python3
"""
Genera CSVs de ProtoDash: global (5 oficiales), local (ejemplos), estabilidad.
Fuente principal: outputs/integration/agrocebada_prototipos.json (congelado).
Fuente histórica: outputs/xai/fase2/ (CSV, ejemplo_xai.json, estabilidad.csv).
"""
import json
import pandas as pd
from pathlib import Path

BASE = Path("/mnt/FEDORA-DATOS/ModoZebada/DATASET_RETO_AGRO_2026 (1)/DATASET_RETO_AGRO_2026")
OUTDIR = BASE / "docs/anexo_metodologico"

# ============================================================
# 1. PROTOTIPOS GLOBALES OFICIALES (congelados)
# ============================================================
with open(BASE / "outputs/integration/agrocebada_prototipos.json") as f:
    proto_oficial = json.load(f)

config = proto_oficial["configuracion"]
global_rows = []
for p in proto_oficial["prototipos"]:
    global_rows.append({
        "config_space": config["space"],
        "config_method": config["method"],
        "config_m": config["m"],
        "config_sigma_factor": config["sigma_factor"],
        "config_optimizer": config["optimizer"],
        "global_selection_rank": p["global_selection_rank"],
        "prototype_id": p["prototype_id"],
        "municipio": p["municipio"],
        "particion": p["particion"],
        "weight_original": p["weight_original"],
        "weight_normalized": p["weight_normalized"],
        "conceptual_original_C1a_ndvi_s2_mean": p["conceptual_original"]["C1a_ndvi_s2_mean"],
        "conceptual_original_C2a_amplitud_ndvi": p["conceptual_original"]["C2a_amplitud_ndvi"],
        "conceptual_original_C3a_ndwi_s2_mean": p["conceptual_original"]["C3a_ndwi_s2_mean"],
        "conceptual_original_C4a_prec_acum_abr_oct": p["conceptual_original"]["C4a_prec_acum_abr_oct"],
        "conceptual_original_C5a_elev_media": p["conceptual_original"]["C5a_elev_media"],
    })

df_global = pd.DataFrame(global_rows)
df_global.to_csv(OUTDIR / "prototipos_resumen.csv", index=False)
print(f"prototipos_resumen.csv: {len(df_global)} prototipos oficiales")

# ============================================================
# 2. SIMILITUDES LOCALES (ejemplos del catálogo integrado)
# ============================================================
# El archivo agrocebada_parcelas.json contiene las similitudes locales para todas las parcelas
with open(BASE / "outputs/integration/agrocebada_parcelas.json") as f:
    parcelas = json.load(f)

# Extraer 3 ejemplos representativos (uno por municipio principal: Calpulalpan, Singuilucan, Chignahuapan)
ejemplos_ids = ["AGC_001", "AGC_003", "AGC_170"]  # Calpulalpan, Singuilucan, Calpulalpan (prototype itself)

local_rows = []
for pid in ejemplos_ids:
    if pid not in parcelas:
        print(f"ADVERTENCIA: {pid} no en catálogo integrado")
        continue
    parcela = parcelas[pid]
    # La estructura tiene "prototipos_cercanos" o similar
    if "prototipos_cercanos" in parcela:
        for i, proto in enumerate(parcela["prototipos_cercanos"]):
            local_rows.append({
                "parcela_id": pid,
                "parcela_municipio": parcela.get("municipio", ""),
                "parcela_conjunto": parcela.get("CONJUNTO", ""),
                "local_rank": i + 1,
                "prototype_id": proto.get("prototype_id", ""),
                "prototype_municipio": proto.get("prototype_municipio", ""),
                "global_selection_rank": proto.get("global_selection_rank", None),
                "weight_global_representativeness_normalized": proto.get("weight_global_representativeness_normalized", None),
                "distance": proto.get("distance", None),
                "similarity_rbf": proto.get("similarity_rbf", None),
                # Top 3 componentes diferenciadores
                "top_comp_1": proto.get("top_differentiating_components", [{}])[0].get("component", "") if proto.get("top_differentiating_components") else "",
                "top_comp_1_contrib": proto.get("top_differentiating_components", [{}])[0].get("relative_contribution_to_distance", 0) if proto.get("top_differentiating_components") else None,
                "top_comp_2": proto.get("top_differentiating_components", [{}])[1].get("component", "") if len(proto.get("top_differentiating_components", [])) > 1 else "",
                "top_comp_2_contrib": proto.get("top_differentiating_components", [{}])[1].get("relative_contribution_to_distance", 0) if len(proto.get("top_differentiating_components", [])) > 1 else None,
                "top_comp_3": proto.get("top_differentiating_components", [{}])[2].get("component", "") if len(proto.get("top_differentiating_components", [])) > 2 else "",
                "top_comp_3_contrib": proto.get("top_differentiating_components", [{}])[2].get("relative_contribution_to_distance", 0) if len(proto.get("top_differentiating_components", [])) > 2 else None,
            })

df_local = pd.DataFrame(local_rows)
df_local.to_csv(OUTDIR / "prototipos_similitudes_locales.csv", index=False)
print(f"prototipos_similitudes_locales.csv: {len(df_local)} filas ({len(ejemplos_ids)} parcelas × 5 prototipos)")

# ============================================================
# 3. ESTABILIDAD (desde fase2/estabilidad.csv)
# ============================================================
estab = pd.read_csv(BASE / "outputs/xai/fase2/estabilidad.csv")
# Añadir columna de configuración para claridad
estab["config_space"] = estab["space"]
estab["config_method"] = estab["method"]
estab.to_csv(OUTDIR / "prototipos_estabilidad.csv", index=False)
print(f"prototipos_estabilidad.csv: {len(estab)} filas")

# ============================================================
# Verificación: comparar prototipos.csv (histórico) vs oficial
# ============================================================
proto_hist = pd.read_csv(BASE / "outputs/xai/fase2/prototipos.csv")
# Filtrar configuración oficial: compacto, ProtoDash, m=5, sigma_factor=1.0
oficial_hist = proto_hist[
    (proto_hist["space"] == "compacto") & 
    (proto_hist["method"] == "ProtoDash") & 
    (proto_hist["m"] == 5) & 
    (proto_hist["sigma_factor"] == 1.0)
].sort_values("global_selection_rank")

print("\n--- Verificación: prototipos históricos vs oficiales ---")
print("Históricos (compacto, ProtoDash, m=5, sigma=1.0):")
for _, r in oficial_hist.iterrows():
    print(f"  Rank {r['global_selection_rank']}: {r['prototype_id']} ({r['municipio']}) w_norm={r['weight_normalized']:.6f}")

print("\nOficiales (agrocebada_prototipos.json):")
for p in proto_oficial["prototipos"]:
    print(f"  Rank {p['global_selection_rank']}: {p['prototype_id']} ({p['municipio']}) w_norm={p['weight_normalized']:.6f}")

# Comparar IDs
hist_ids = set(oficial_hist["prototype_id"].values)
ofic_ids = set(p["prototype_id"] for p in proto_oficial["prototipos"])
print(f"\nCoincidencia IDs: {hist_ids == ofic_ids}")
if hist_ids != ofic_ids:
    print(f"  En histórico no en oficial: {hist_ids - ofic_ids}")
    print(f"  En oficial no en histórico: {ofic_ids - hist_ids}")
