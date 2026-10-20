#!/usr/bin/env python
"""Pruebas automatizadas Fase 1 — Conceptos observacionales AgroCebada XAI."""
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PERFILES = Path("outputs/xai/perfiles_conceptuales.csv")
MANIFEST = Path("outputs/xai/manifest_xai.json")
CONCEPTOS_PY = Path("outputs/xai/conceptos.py")


@pytest.fixture(scope="session")
def perfiles():
    return pd.read_csv(PERFILES, encoding="utf-8-sig")


@pytest.fixture(scope="session")
def manifest():
    import json
    with open(MANIFEST) as f:
        return json.load(f)


def test_197_parcelas(perfiles):
    assert len(perfiles) == 197
    assert perfiles["ID_POLIGONO"].nunique() == 197


def test_componentes_presentes(perfiles, manifest):
    for c in manifest["component_columns"]:
        assert c in perfiles.columns, f"Falta componente base: {c}"
        assert c + "_scaled" in perfiles.columns, f"Falta componente escalado: {c}_scaled"


def test_no_fuga_rendimiento():
    code = CONCEPTOS_PY.read_text()
    assert "RENDIMIENTO_T_HA" not in code, "conceptos.py no debe leer RENDIMIENTO_T_HA"
    assert "CONJUNTO" not in code, "conceptos.py no debe leer CONJUNTO"


def test_reproducibilidad():
    # Ejecutar conceptos.py de nuevo y comparar hash de salida
    result = subprocess.run([sys.executable, str(CONCEPTOS_PY)],
                           capture_output=True, text=True, cwd=Path.cwd())
    assert result.returncode == 0, f"Segunda ejecución falló: {result.stderr}"

    with open(PERFILES, "rb") as f:
        hash1 = hashlib.sha256(f.read()).hexdigest()

    result2 = subprocess.run([sys.executable, str(CONCEPTOS_PY)],
                            capture_output=True, text=True, cwd=Path.cwd())
    assert result2.returncode == 0, f"Tercera ejecución falló: {result2.stderr}"

    with open(PERFILES, "rb") as f:
        hash2 = hashlib.sha256(f.read()).hexdigest()

    assert hash1 == hash2, "Salida no reproducible (hash cambió entre ejecuciones)"


def test_orden_invariante(perfiles):
    # Permutar y volver a ordenar → misma matriz
    perm = perfiles.sample(frac=1, random_state=2026).sort_values("ID_POLIGONO").reset_index(drop=True)
    orig = perfiles.sort_values("ID_POLIGONO").reset_index(drop=True)
    pd.testing.assert_frame_equal(perm, orig, check_dtype=False, check_exact=False, rtol=1e-10)


def test_rangos_plausibles(perfiles):
    # NDVI/EVI en rango físico aproximado
    assert perfiles["C1a_ndvi_s2_mean"].between(-0.5, 1.5).all(), "NDVI fuera de rango"
    assert perfiles["C1b_evi_s2_mean"].between(-0.5, 1.5).all(), "EVI fuera de rango"
    # DOY pico en ventana de búsqueda
    assert perfiles["C2c_doy_pico_ndvi"].dropna().between(91, 304).all(), "DOY pico fuera de ventana"
    # Precipitación no negativa
    assert (perfiles["C4a_prec_acum_abr_oct"].dropna() >= 0).all(), "Precipitación negativa"
    # Elevación en rango esperado región
    assert perfiles["C5a_elev_media"].dropna().between(2000, 3500).all(), "Elevación fuera de rango"
    # Pendiente no negativa
    assert (perfiles["C5b_pend_media"].dropna() >= 0).all(), "Pendiente negativa"


def test_flags_calidad(perfiles):
    for flag in ["flag_n_meses_s2_abr_oct", "flag_n_obs_s2_abr_oct",
                 "flag_meses_clima_2025", "flag_topo_completa"]:
        assert flag in perfiles.columns, f"Falta flag: {flag}"
        assert perfiles[flag].notna().all(), f"Flag {flag} tiene NaN"

    # Todas las parcelas tienen 7 meses S2 y 7 meses clima
    assert (perfiles["flag_n_meses_s2_abr_oct"] == 7).all()
    assert (perfiles["flag_meses_clima_2025"] == 7).all()
    assert (perfiles["flag_topo_completa"] == 1).all()


def test_manifest_completo(manifest):
    required = ["version", "seed", "source_hashes", "component_columns",
                "scaler_params", "temporal_window", "cloud_thresholds"]
    for k in required:
        assert k in manifest, f"Manifest falta clave: {k}"
    assert manifest["seed"] == 2026
    assert len(manifest["component_columns"]) == 12
    # Verificar hashes presentes
    for src in ["Conjunto_datos_BASICO_AgroCebada2026.csv",
                "Conjunto_datos_PRO_AgroCebada.csv",
                "parcelas_base.csv",
                "clima_mensual_parcela.csv"]:
        assert src in manifest["source_hashes"], f"Falta hash de {src}"
        assert len(manifest["source_hashes"][src]) == 64, f"Hash inválido para {src}"


def test_scaler_params_consistency(perfiles, manifest):
    # Verificar que scaled = (base - median) / iqr aproximadamente
    for comp in manifest["component_columns"]:
        base = perfiles[comp].dropna().values
        scaled = perfiles[comp + "_scaled"].dropna().values
        median = manifest["scaler_params"][comp]["median"]
        iqr = manifest["scaler_params"][comp]["iqr"]
        expected = (base - median) / iqr
        np.testing.assert_allclose(scaled, expected, rtol=1e-5, atol=1e-8,
                                   err_msg=f"Scaler inconsistente para {comp}")


def test_concordancia_sensores_existe():
    path = Path("outputs/xai/concordancia_sensores.csv")
    assert path.exists(), "Falta concordancia_sensores.csv"
    df = pd.read_csv(path)
    assert len(df) > 0, "Concordancia vacía"
    assert "ID_POLIGONO" in df.columns
    assert "variable" in df.columns
    assert "pearson_r" in df.columns
    # Debe tener NDVI y EVI
    vars_found = set(df["variable"].unique())
    assert "ndvi_promedio" in vars_found
    assert "evi_promedio" in vars_found


if __name__ == "__main__":
    pytest.main([__file__, "-v"])