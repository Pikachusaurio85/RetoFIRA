"""Adaptador del predictor: consume predicciones_ui.json, modelo_final.json, explicacion_referencia_municipal.csv."""
from __future__ import annotations
import json
import csv
from pathlib import Path
from typing import Any

from .config import (
    PREDICCIONES_UI, MODELO_FINAL, EXPLICACION_MUNICIPAL, MANIFESTO_PREDICTOR
)


class PredictorAdapter:
    """Adaptador desacoplado para el predictor. No conoce XAI."""
    
    def __init__(self):
        self._predicciones: dict[str, dict[str, Any]] = {}
        self._modelo_final: dict[str, Any] = {}
        self._componentes_municipales: dict[str, dict[str, Any]] = {}
        self._manifest: dict[str, Any] = {}
        self._parcelas_info: dict[str, dict[str, Any]] = {}
        self._loaded = False
    
    def load(self) -> None:
        """Carga todas las fuentes del predictor."""
        if self._loaded:
            return
        
        # 1. predicciones_ui.json
        with open(PREDICCIONES_UI, "r", encoding="utf-8") as f:
            predicciones = json.load(f)
        self._predicciones = {p["ID_POLIGONO"]: p for p in predicciones}
        
        # 2. modelo_final.json
        with open(MODELO_FINAL, "r", encoding="utf-8") as f:
            self._modelo_final = json.load(f)
        
        # Extraer parcelas (municipio + conjunto)
        self._parcelas_info = self._modelo_final.get("parcelas", {})
        
        # 3. explicacion_referencia_municipal.csv
        with open(EXPLICACION_MUNICIPAL, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._componentes_municipales[row["municipio"]] = {
                    "n_parcelas_etiquetadas": int(row["n_parcelas_etiquetadas"]),
                    "media_municipal_t_ha": float(row["media_municipal_t_ha"]),
                    "media_global_t_ha": float(row["media_global_t_ha"]),
                    "k_encogimiento": float(row["k_encogimiento"]),
                    "peso_municipal": float(row["peso_municipal"]),
                    "ajuste_municipal_t_ha": float(row["ajuste_municipal_t_ha"]),
                    "prediccion_t_ha": float(row["prediccion_t_ha"]),
                    "estrato_incertidumbre": row["estrato_incertidumbre"],
                    "semiancho_intervalo80_t_ha": float(row["semiancho_intervalo80_t_ha"]),
                    "n_parcelas_prediccion": int(row["n_parcelas_PREDICCION"]),
                }
        
        # 4. manifest_modelo.json
        with open(MANIFESTO_PREDICTOR, "r", encoding="utf-8") as f:
            self._manifest = json.load(f)
        
        self._loaded = True
    
    # --- Getters ---
    
    def get_prediction(self, id_poligono: str) -> dict[str, Any] | None:
        """Retorna la predicción completa para un ID."""
        return self._predicciones.get(id_poligono)
    
    def get_municipio(self, id_poligono: str) -> str | None:
        info = self._parcelas_info.get(id_poligono)
        return info.get("municipio") if info else None
    
    def get_conjunto(self, id_poligono: str) -> str | None:
        info = self._parcelas_info.get(id_poligono)
        return info.get("conjunto") if info else None
    
    def get_municipal_component(self, municipio: str) -> dict[str, Any] | None:
        """Componentes municipales para un municipio (desde explicacion_referencia_municipal.csv)."""
        return self._componentes_municipales.get(municipio)
    
    def get_prediction_components(self, id_poligono: str) -> list[dict[str, Any]]:
        """Componentes de explicación del predictor para una parcela."""
        pred = self._predicciones.get(id_poligono)
        if not pred:
            return []
        return pred.get("prediction_components", [])
    
    def get_warnings(self, id_poligono: str) -> list[str]:
        """Warnings del predictor para una parcela."""
        pred = self._predicciones.get(id_poligono)
        if not pred:
            return []
        return pred.get("warnings", [])
    
    def get_model_meta(self) -> dict[str, Any]:
        """Metadatos del modelo (desde modelo_final.json)."""
        return {
            "model_id": self._modelo_final.get("model_id"),
            "model_version": self._modelo_final.get("model_version"),
            "tipo": self._modelo_final.get("tipo"),
            "formula": self._modelo_final.get("formula"),
            "k_encogimiento": self._modelo_final.get("k_encogimiento"),
            "media_global_t_ha": self._modelo_final.get("media_global_t_ha"),
            "n_entrenamiento": self._modelo_final.get("n_entrenamiento"),
            "municipios": self._modelo_final.get("municipios", {}),
            "entrenado_utc": self._modelo_final.get("entrenado_utc"),
            "fuentes_sha256": self._modelo_final.get("fuentes_sha256", {}),
        }
    
    def get_manifest(self) -> dict[str, Any]:
        return self._manifest
    
    def get_all_ids(self) -> list[str]:
        return list(self._predicciones.keys())
    
    def get_hashes_sources(self) -> dict[str, str]:
        """Hashes de los archivos fuente del predictor."""
        manifest = self._manifest
        archivos = manifest.get("archivos_salida_sha256", {})
        return {
            "predicciones_ui.json": archivos.get("predicciones_ui.json", ""),
            "modelo_final.json": archivos.get("modelo_final.json", ""),
            "explicacion_referencia_municipal.csv": archivos.get("explicacion_referencia_municipal.csv", ""),
            "manifest_modelo.json": archivos.get("manifest_modelo.json", ""),
        }


def load_predictor() -> PredictorAdapter:
    """Factory function para cargar el adaptador."""
    adapter = PredictorAdapter()
    adapter.load()
    return adapter