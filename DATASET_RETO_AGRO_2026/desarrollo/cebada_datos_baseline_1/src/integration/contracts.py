"""Contratos tipados para la integración AgroCebada 2026."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path


@dataclass
class PredictorConfig:
    """Configuración del predictor (cargada desde modelo_final.json)."""
    model_id: str
    model_version: str
    tipo: str
    formula: str
    k_encogimiento: float
    media_global_t_ha: float
    n_entrenamiento: int
    municipios: dict[str, dict[str, Any]]


@dataclass
class MunicipalComponent:
    """Componentes municipales del predictor (desde explicacion_referencia_municipal.csv)."""
    municipio: str
    n_parcelas_etiquetadas: int
    media_municipal_t_ha: float
    media_global_t_ha: float
    k_encogimiento: float
    peso_municipal: float
    ajuste_municipal_t_ha: float
    prediccion_t_ha: float
    estrato_incertidumbre: str
    semiancho_intervalo80_t_ha: float
    n_parcelas_prediccion: int


@dataclass
class PredictionComponent:
    """Componente de explicación del predictor (desde prediction_components)."""
    component: str
    value_t_ha: float
    description: str
    # Campos opcionales específicos del prior municipal
    municipio: Optional[str] = None
    n_parcelas_municipio: Optional[int] = None
    media_municipal_t_ha: Optional[float] = None
    peso_municipal: Optional[float] = None
    k_encogimiento: Optional[float] = None


@dataclass
class PredictionUI:
    """Predicción completa para UI (desde predicciones_ui.json)."""
    schema_version: str
    id_poligono: str
    model_id: str
    prediction_t_ha: float
    uncertainty: dict[str, Any]
    prediction_components: list[PredictionComponent]
    warnings: list[str]

    @property
    def explicacion_componentes(self) -> dict[str, Any]:
        """Estructura explícita para frontend."""
        comps = {c.component: c.value_t_ha for c in self.prediction_components}
        media_global = comps.get("media_global_entrenamiento", 0.0)
        ajuste_municipal = comps.get("ajuste_municipal", 0.0)
        # Buscar metadatos municipales en el componente de ajuste
        muni_comp = next((c for c in self.prediction_components if c.component == "ajuste_municipal"), None)
        return {
            "media_global": media_global,
            "ajuste_municipal": ajuste_municipal,
            "contexto": {
                "media_municipal_t_ha": muni_comp.media_municipal_t_ha if muni_comp else None,
                "n_parcelas_municipio": muni_comp.n_parcelas_municipio if muni_comp else None,
                "peso_municipal": muni_comp.peso_municipal if muni_comp else None,
                "k_encogimiento": muni_comp.k_encogimiento if muni_comp else None,
            },
            "prediction_components_raw": [
                {"component": c.component, "value_t_ha": c.value_t_ha, "description": c.description}
                for c in self.prediction_components
            ]
        }

    def validate_consistency(self, tol: float = 1e-9) -> bool:
        """Valida que media_global + ajuste_municipal ≈ prediction_t_ha."""
        comps = {c.component: c.value_t_ha for c in self.prediction_components}
        return abs((comps.get("media_global_entrenamiento", 0.0) + 
                    comps.get("ajuste_municipal", 0.0)) - self.prediction_t_ha) <= tol


@dataclass
class ConceptualProfile:
    """Perfil conceptual de una parcela (desde perfiles_conceptuales.csv)."""
    id_poligono: str
    original: dict[str, float]
    scaled: dict[str, float]
    flags: dict[str, Any]


@dataclass
class PrototypeInfo:
    """Información de un prototipo (desde prototipos.csv - filtrado a oficiales)."""
    prototype_id: str
    global_selection_rank: int
    weight_original: float
    weight_normalized: float
    municipio: str
    particion: str
    conceptual_original: dict[str, float]
    conceptual_scaled: dict[str, float]


@dataclass
class DifferentiatingComponent:
    """Componente diferenciador entre parcela y prototipo."""
    component: str
    unit: str
    parcel_value_original: float
    parcel_value_scaled: float
    prototype_value_original: float
    prototype_value_scaled: float
    absolute_diff_original: float
    absolute_diff_scaled: float
    relative_contribution_to_distance: float


@dataclass
class NearbyPrototype:
    """Prototipo cercano para una parcela (desde similitudes.csv - config oficial)."""
    local_similarity_rank: int
    global_selection_rank: Optional[int]
    prototype_id: str
    prototype_municipio: str
    weight_original: float
    weight_normalized: float
    distance: float
    similarity_rbf: float
    top_differentiating_components: list[DifferentiatingComponent]


@dataclass
class ParcelIntegrated:
    """Parcelas integrada final para agrocebada_parcelas.json."""
    id: str
    municipio: str
    particion: str
    prediccion: dict[str, Any]
    perfil_conceptual: dict[str, Any]
    prototipos_cercanos: list[dict[str, Any]]
    warnings: list[str]


@dataclass
class PrototypeCatalog:
    """Catálogo de prototipos para agrocebada_prototipos.json."""
    version: str
    configuracion: dict[str, Any]
    prototipos: list[dict[str, Any]]


@dataclass
class IntegrationManifest:
    """Manifest integrado para agrocebada_manifest.json."""
    contract_version: str
    generation_utc: str
    predictor: dict[str, Any]
    xai: dict[str, Any]
    counts: dict[str, int]
    sources: dict[str, list[str]]
    warnings: list[str]
    validation_status: str


# Configuración XAI congelada (fuente única de verdad)
XAI_FROZEN_CONFIG = {
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
}

OFFICIAL_PROTOTYPE_IDS = ["AGC_076", "AGC_133", "AGC_162", "AGC_074", "AGC_170"]


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()