"""Shared paths and constants for the Bayesian hierarchical experiments."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = PROJECT_ROOT / "desarrollo" / "cebada_datos_baseline_1"
DATA_PROCESSED = BASELINE_DIR / "data_processed"
PREDICTOR_DIR = PROJECT_ROOT / "outputs" / "predictor"
XAI_DIR = PROJECT_ROOT / "outputs" / "xai"

EXP_DIR = PROJECT_ROOT / "experiments" / "bayesian_hierarchical"
CONFIGS_DIR = EXP_DIR / "configs"
RESULTS_DIR = EXP_DIR / "results"
REPORTS_DIR = EXP_DIR / "reports"
FIGURES_DIR = EXP_DIR / "figures"
LOGS_DIR = EXP_DIR / "logs"

FEATURES_CSV = DATA_PROCESSED / "features_parcela.csv"
PERFILES_CSV = XAI_DIR / "perfiles_conceptuales.csv"
SPLITS_CSV = PREDICTOR_DIR / "splits.csv"
SPLITS_DESCRIPTOR_CSV = PREDICTOR_DIR / "descriptor_particiones.csv"
MODELO_FINAL_JSON = PREDICTOR_DIR / "modelo_final.json"

TARGET = "RENDIMIENTO_T_HA"
PARTITION_COL = "CONJUNTO"
MUNICIPIO_COL = "loc_municipio"
ID_COL = "ID_POLIGONO"

TRAIN_PARTITION = "ENTRENAMIENTO"
PRED_PARTITION = "PREDICCION"

XAI_CONCEPTS = [
    "C1a_ndvi_s2_mean",
    "C2a_amplitud_ndvi",
    "C3a_ndwi_s2_mean",
    "C4a_prec_acum_abr_oct",
    "C5a_elev_media",
]
XAI_CONCEPT_UNITS = {
    "C1a_ndvi_s2_mean": "NDVI [0-1]",
    "C2a_amplitud_ndvi": "NDVI",
    "C3a_ndwi_s2_mean": "NDWI",
    "C4a_prec_acum_abr_oct": "mm",
    "C5a_elev_media": "m",
}

SEED = 20260219
CV_SCHEME = "RCV5x10"


def ensure_dirs() -> None:
    for d in (CONFIGS_DIR, RESULTS_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
