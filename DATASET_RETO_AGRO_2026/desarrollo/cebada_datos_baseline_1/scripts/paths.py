from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = PROJECT_ROOT / "cebada_datos_baseline_1"
DATA_PROCESSED = BASELINE_ROOT / "data_processed"
REPORTS = BASELINE_ROOT / "reports"
FIGURES = BASELINE_ROOT / "figures"
SOURCE_ROOT = Path(
    os.environ.get("CEBADA_SOURCE_ROOT", str(PROJECT_ROOT / "datos"))
).expanduser().resolve()
if (SOURCE_ROOT / "entradas").is_dir() and (SOURCE_ROOT / "geoespaciales").is_dir():
    RAW_INNER = SOURCE_ROOT / "geoespaciales"
    RAW_DATASET = SOURCE_ROOT / "entradas"
else:
    RAW_INNER = SOURCE_ROOT / "inner"
    RAW_DATASET = SOURCE_ROOT / "Cebada" / "DATASET_RETO_AGRO_2026"

for directory in (DATA_PROCESSED, REPORTS, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)
