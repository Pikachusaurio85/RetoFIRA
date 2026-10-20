"""Adaptador XAI: consume perfiles conceptuales, prototipos oficiales, similitudes."""
from __future__ import annotations
import json
import csv
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    PERFILES_CONCEPTUALES, PROTOTIPOS_CSV, SIMILITUDES_CSV,
    MANIFESTO_FASE1, MANIFESTO_FASE2, EJEMPLO_XAI,
    XAI_FROZEN_CONFIG, OFFICIAL_PROTOTYPE_IDS, COMPONENT_UNITS
)


class XAIAdapter:
    """Adaptador desacoplado para XAI. No conoce predictor."""
    
    def __init__(self):
        self._perfiles: dict[str, dict[str, Any]] = {}
        self._prototipos_oficiales: list[dict[str, Any]] = []
        self._similitudes: dict[str, list[dict[str, Any]]] = {}
        self._manifest_fase1: dict[str, Any] = {}
        self._manifest_fase2: dict[str, Any] = {}
        self._loaded = False
    
    def load(self) -> None:
        """Carga todas las fuentes XAI."""
        if self._loaded:
            return
        
        # 1. Perfiles conceptuales
        df_perfiles = pd.read_csv(PERFILES_CONCEPTUALES, encoding="utf-8-sig")
        self._perfiles = {}
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
            
            self._perfiles[pid] = {
                "original": original,
                "scaled": scaled,
                "flags": flags
            }
        
        # 2. Prototipos - filtrar oficiales
        df_proto = pd.read_csv(PROTOTIPOS_CSV, encoding="utf-8-sig")
        mask = (
            (df_proto["space"] == "compacto") &
            (df_proto["method"] == "ProtoDash") &
            (df_proto["m"] == 5) &
            (abs(df_proto["sigma_factor"] - 1.0) < 0.001)
        )
        df_oficiales = df_proto[mask].copy()
        
        self._prototipos_oficiales = []
        for _, row in df_oficiales.iterrows():
            pid = row["prototype_id"]
            perfil = self._perfiles.get(pid, {})
            original = perfil.get("original", {})
            scaled = perfil.get("scaled", {})
            
            particion = "ENTRENAMIENTO"
            if pid == "AGC_076":
                particion = "PREDICCION"
            
            self._prototipos_oficiales.append({
                "prototype_id": pid,
                "global_selection_rank": int(row["global_selection_rank"]),
                "municipio": row["municipio"],
                "particion": particion,
                "weight_original": float(row["weight_original"]),
                "weight_normalized": float(row["weight_normalized"]),
                "conceptual_original": {k: original.get(k) for k in ["C1a_ndvi_s2_mean", "C2a_amplitud_ndvi", "C3a_ndwi_s2_mean", "C4a_prec_acum_abr_oct", "C5a_elev_media"]},
                "conceptual_scaled": {k: scaled.get(k) for k in ["C1a_ndvi_s2_mean", "C2a_amplitud_ndvi", "C3a_ndwi_s2_mean", "C4a_prec_acum_abr_oct", "C5a_elev_media"]},
            })
        
        self._prototipos_oficiales.sort(key=lambda x: x["global_selection_rank"])
        
        # 3. Similitudes - filtrar configuración oficial
        df_sim = pd.read_csv(SIMILITUDES_CSV, encoding="utf-8-sig")
        mask_sim = (
            (df_sim["space"] == "compacto") &
            (df_sim["method"] == "ProtoDash") &
            (df_sim["m"] == 5) &
            (abs(df_sim["sigma_factor"] - 1.0) < 0.001)
        )
        df_sim_oficial = df_sim[mask_sim].copy()
        
        self._similitudes = {}
        for pid, group in df_sim_oficial.groupby("ID_POLIGONO"):
            protos = []
            for _, row in group.iterrows():
                # Pesos desde catálogo oficial (fuente de verdad), no del CSV de similitudes
                w1_orig, w1_norm = self._get_prototype_weights(row["proto_1_id"])
                protos.append({
                    "local_similarity_rank": int(row["local_similarity_rank_1"]),
                    "global_selection_rank": self._get_global_selection_rank(row["proto_1_id"]),
                    "prototype_id": row["proto_1_id"],
                    "prototype_municipio": self._get_municipio_prototipo(row["proto_1_id"]),
                    "weight_original": w1_orig,
                    "weight_normalized": w1_norm,
                    "distance": float(row["dist_1"]),
                    "similarity_rbf": float(row["sim_1"]),
                    "top_differentiating_components": self._extract_diff_components(row, 1),
                })
                # Top 2 - pesos desde catálogo oficial
                if "proto_2_id" in row and pd.notna(row["proto_2_id"]):
                    p2_id = row["proto_2_id"]
                    w2_orig, w2_norm = self._get_prototype_weights(p2_id)
                    protos.append({
                        "local_similarity_rank": int(row.get("local_similarity_rank_2", 2)),
                        "global_selection_rank": self._get_global_selection_rank(p2_id),
                        "prototype_id": p2_id,
                        "prototype_municipio": self._get_municipio_prototipo(p2_id),
                        "weight_original": w2_orig,
                        "weight_normalized": w2_norm,
                        "distance": float(row["proto_2_dist"]),
                        "similarity_rbf": float(row["proto_2_sim"]),
                        "top_differentiating_components": self._extract_diff_components(row, 2),
                    })
                # Top 3
                if "proto_3_id" in row and pd.notna(row["proto_3_id"]):
                    p3_id = row["proto_3_id"]
                    w3_orig, w3_norm = self._get_prototype_weights(p3_id)
                    protos.append({
                        "local_similarity_rank": int(row.get("local_similarity_rank_3", 3)),
                        "global_selection_rank": self._get_global_selection_rank(p3_id),
                        "prototype_id": p3_id,
                        "prototype_municipio": self._get_municipio_prototipo(p3_id),
                        "weight_original": w3_orig,
                        "weight_normalized": w3_norm,
                        "distance": float(row["proto_3_dist"]),
                        "similarity_rbf": float(row["proto_3_sim"]),
                        "top_differentiating_components": self._extract_diff_components(row, 3),
                    })
            
            protos.sort(key=lambda x: x["local_similarity_rank"])
            self._similitudes[pid] = protos
        
        # Manifests
        with open(MANIFESTO_FASE1, "r", encoding="utf-8") as f:
            self._manifest_fase1 = json.load(f)
        with open(MANIFESTO_FASE2, "r", encoding="utf-8") as f:
            self._manifest_fase2 = json.load(f)
        
        self._loaded = True
    
    def _get_prototype_weights(self, prototype_id: str) -> tuple[float, float]:
        """Get original and normalized weights for a prototype from official list."""
        for p in self._prototipos_oficiales:
            if p["prototype_id"] == prototype_id:
                return float(p["weight_original"]), float(p["weight_normalized"])
        return 0.0, 0.0
    
    def _get_global_selection_rank(self, prototype_id: str) -> int | None:
        """Get global_selection_rank from official catalog (CSV similitudes no lo incluye)."""
        for p in self._prototipos_oficiales:
            if p["prototype_id"] == prototype_id:
                return int(p["global_selection_rank"])
        return None
    
    def _get_municipio_prototipo(self, prototype_id: str) -> str:
        for p in self._prototipos_oficiales:
            if p["prototype_id"] == prototype_id:
                return p["municipio"]
        return "DESCONOCIDO"
    
    def _extract_diff_components(self, row: pd.Series, rank: int) -> list[dict]:
        """Extract differentiating components from similarity row.
        
        El CSV similitudes.csv solo incluye contribuciones al ranking (top_comp_N_contrib)
        para el prototipo más similar (rank=1). Para rank 2 y 3, las contribuciones se
        establecen en None ya que no están disponibles en la fuente original.
        """
        comps = []
        for i in range(1, 4):
            comp_col = f"top_comp_{i}"
            contrib_col = f"top_comp_{i}_contrib"
            diff_col = f"top_comp_{i}_diff_scaled"
            
            if comp_col in row and pd.notna(row[comp_col]):
                comp_name = row[comp_col]
                # Contribuciones solo disponibles para rank 1 (proto más similar)
                if rank == 1:
                    rel_contrib = float(row[contrib_col]) if contrib_col in row and pd.notna(row[contrib_col]) else None
                else:
                    rel_contrib = None  # No disponible en CSV para rank > 1
                comps.append({
                    "component": comp_name,
                    "unit": "",
                    "parcel_value_original": None,
                    "parcel_value_scaled": None,
                    "prototype_value_original": None,
                    "prototype_value_scaled": None,
                    "absolute_diff_original": None,
                    "absolute_diff_scaled": float(row[diff_col]) if diff_col in row and pd.notna(row[diff_col]) else None,
                    "relative_contribution_to_distance": rel_contrib,
                })
        return comps
    
    # --- Getters ---
    
    def get_perfil(self, id_poligono: str) -> dict[str, Any] | None:
        return self._perfiles.get(id_poligono)
    
    def get_prototipos_oficiales(self) -> list[dict[str, Any]]:
        return self._prototipos_oficiales
    
    def get_similitudes(self, id_poligono: str) -> list[dict[str, Any]]:
        return self._similitudes.get(id_poligono, [])
    
    def get_manifest_fase1(self) -> dict[str, Any]:
        return self._manifest_fase1
    
    def get_manifest_fase2(self) -> dict[str, Any]:
        return self._manifest_fase2
    
    def get_all_ids(self) -> list[str]:
        return list(self._perfiles.keys())
    
    def get_hashes_sources(self) -> dict[str, str]:
        import hashlib
        hashes = {}
        m1 = self._manifest_fase1
        m2 = self._manifest_fase2
        
        if "source_hashes" in m1:
            hashes.update({f"fase1_{k}": v for k, v in m1["source_hashes"].items()})
        if "perfiles_conceptuales_hash" in m2:
            hashes["perfiles_conceptuales.csv"] = m2["perfiles_conceptuales_hash"]
        if "phase1_manifest_hash" in m2:
            hashes["manifest_fase1.json"] = m2["phase1_manifest_hash"]
        
        for path, name in [
            (PERFILES_CONCEPTUALES, "perfiles_conceptuales.csv"),
            (PROTOTIPOS_CSV, "prototipos.csv"),
            (SIMILITUDES_CSV, "similitudes.csv"),
            (MANIFESTO_FASE1, "manifest_xai.json"),
            (MANIFESTO_FASE2, "manifest_fase2.json"),
        ]:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024*1024), b""):
                    h.update(chunk)
            hashes[name] = h.hexdigest()
        
        return hashes


def load_xai() -> XAIAdapter:
    adapter = XAIAdapter()
    adapter.load()
    return adapter