from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "datos"
INPUT_DATA_ROOT = DATA_ROOT / "entradas"
GEOSPATIAL_DATA_ROOT = DATA_ROOT / "geoespaciales"
OUTPUT_FOLDER = PROJECT_ROOT / "salidas"
ASSIGNMENT_PATH = INPUT_DATA_ROOT / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv"
SHAPEFILE_PATH = (
    GEOSPATIAL_DATA_ROOT
    / "Parcelas_Reto_AGC_CONJUNTO_70_30"
    / "Parcelas_Reto_AGC_CONJUNTO.shp"
)