from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = Path(__file__).resolve().parent / "salidas"
SHAPEFILE_PATH = (
    PROJECT_ROOT
    / "Parcelas_Reto_AGC_CONJUNTO_70_30"
    / "Parcelas_Reto_AGC_CONJUNTO.shp"
)