"""Configuración de rutas para el módulo de integración."""
from pathlib import Path

# Raíz del proyecto (repo root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

# Directorios base
PREDICTOR_DIR = BASE_DIR / "outputs" / "predictor"
XAI_DIR = BASE_DIR / "outputs" / "xai"
XAI_FASE2_DIR = XAI_DIR / "fase2"

# Archivos del predictor
PREDICCIONES_UI = PREDICTOR_DIR / "predicciones_ui.json"
MODELO_FINAL = PREDICTOR_DIR / "modelo_final.json"
EXPLICACION_MUNICIPAL = PREDICTOR_DIR / "explicacion_referencia_municipal.csv"
MANIFESTO_PREDICTOR = PREDICTOR_DIR / "manifest_modelo.json"

# Archivos XAI Fase 1
PERFILES_CONCEPTUALES = XAI_DIR / "perfiles_conceptuales.csv"
MANIFESTO_XAI = XAI_DIR / "manifest_xai.json"
CONCORDANCIA_SENSORES = XAI_DIR / "concordancia_sensores.csv"

# Archivos XAI Fase 2
PROTOTIPOS_CSV = XAI_FASE2_DIR / "prototipos.csv"
SIMILITUDES_CSV = XAI_FASE2_DIR / "similitudes.csv"
MANIFESTO_FASE2 = XAI_FASE2_DIR / "manifest_fase2.json"
EJEMPLO_XAI = XAI_FASE2_DIR / "ejemplo_xai.json"
MANIFESTO_FASE1 = XAI_DIR / "manifest_xai.json"

# Directorio de salida
INTEGRATION_OUTPUT_DIR = BASE_DIR / "outputs" / "integration"
INTEGRATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Archivos de salida
PARCELAS_JSON = INTEGRATION_OUTPUT_DIR / "agrocebada_parcelas.json"
PROTOTIPOS_JSON = INTEGRATION_OUTPUT_DIR / "agrocebada_prototipos.json"
MANIFESTO_INTEGRACION = INTEGRATION_OUTPUT_DIR / "agrocebada_manifest.json"

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

# Mapeo de unidades para componentes XAI
COMPONENT_UNITS = {
    "C1a_ndvi_s2_mean": "NDVI [0-1]",
    "C2a_amplitud_ndvi": "NDVI",
    "C3a_ndwi_s2_mean": "NDWI [-1,1]",
    "C4a_prec_acum_abr_oct": "mm",
    "C5a_elev_media": "m",
}