"""Ejecuta el pipeline baseline en orden usando rutas relativas al proyecto."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from paths import BASELINE_ROOT, RAW_DATASET, RAW_INNER


def main() -> None:
    required = (
        RAW_DATASET / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv",
        RAW_DATASET / "Conjunto_datos_BASICO_AgroCebada2026.csv",
        RAW_DATASET / "Conjunto_datos_PRO_AgroCebada.csv",
        RAW_INNER
        / "Parcelas_Reto_AGC_CONJUNTO_70_30"
        / "Parcelas_Reto_AGC_CONJUNTO.shp",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "No se encontraron las fuentes extraídas requeridas. "
            "Defina CEBADA_SOURCE_ROOT para la carpeta que contiene "
            "'Cebada\\DATASET_RETO_AGRO_2026' e 'inner'.\n"
            + formatted
        )

    script_root = BASELINE_ROOT / "scripts"
    sequence = (
        ("00_eda_tabular.py", str(RAW_DATASET)),
        ("01_eda_geo.py",),
        ("02_eda_sat.py",),
        ("03_parcel_geo_features.py",),
        ("05_cloud_and_profiles.py",),
        ("06_build_features.py",),
        ("07_feature_check.py",),
        ("08_baselines.py",),
        ("09_experiments_residual.py",),
        ("10_final_figs_preds.py",),
        ("11_data_dictionary.py",),
    )
    for entry in sequence:
        command = [sys.executable, str(script_root / entry[0]), *entry[1:]]
        subprocess.run(command, cwd=BASELINE_ROOT, check=True)


if __name__ == "__main__":
    main()
