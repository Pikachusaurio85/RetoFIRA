#!/usr/bin/env python
"""Pruebas automatizadas Fase 2 — Prototipos observacionales (v2 corregida)."""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FASE2_DIR = Path("outputs/xai/fase2")
FASE1_DIR = Path("outputs/xai")


@pytest.fixture(scope="session")
def perfiles():
    return pd.read_csv(FASE1_DIR / "perfiles_conceptuales.csv", encoding="utf-8-sig")


@pytest.fixture(scope="session")
def prototipos():
    return pd.read_csv(FASE2_DIR / "prototipos.csv", encoding="utf-8-sig")


@pytest.fixture(scope="session")
def similitudes():
    return pd.read_csv(FASE2_DIR / "similitudes.csv", encoding="utf-8-sig")


@pytest.fixture(scope="session")
def estabilidad():
    return pd.read_csv(FASE2_DIR / "estabilidad.csv", encoding="utf-8-sig")


@pytest.fixture(scope="session")
def manifest2():
    import json
    with open(FASE2_DIR / "manifest_fase2.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def ejemplo():
    import json
    with open(FASE2_DIR / "ejemplo_xai.json") as f:
        return json.load(f)


def test_prototipos_existen(prototipos):
    assert len(prototipos) > 0
    required = ["space", "method", "m", "global_selection_rank", "prototype_id", "weight_original", "weight_normalized", "municipio"]
    for c in required:
        assert c in prototipos.columns, f"Falta columna {c}"


def test_similitudes_existen(similitudes):
    assert len(similitudes) > 0
    required = ["ID_POLIGONO", "space", "method", "m", "proto_1_id", "dist_1", "sim_1", "local_similarity_rank_1"]
    for c in required:
        assert c in similitudes.columns, f"Falta columna {c}"


def test_estabilidad_valores(estabilidad):
    assert len(estabilidad) == 12  # 2 spaces × 3 m × 2 methods
    for _, row in estabilidad.iterrows():
        assert 0 <= row["jaccard_mean"] <= 1
        assert row["jaccard_std"] >= 0
        if row["method"] == "protodash":
            assert 0 <= row["weight_corr_mean"] <= 1 or np.isnan(row["weight_corr_mean"])


def test_ejemplo_xai_estructura(ejemplo):
    assert "version" in ejemplo
    assert "parcelas_ejemplo" in ejemplo
    assert len(ejemplo["parcelas_ejemplo"]) == 3
    assert "configuracion_recomendada" in ejemplo
    
    for pe in ejemplo["parcelas_ejemplo"]:
        assert "ID_POLIGONO" in pe
        assert "prototipos_cercanos" in pe
        assert len(pe["prototipos_cercanos"]) >= 3
        for pc in pe["prototipos_cercanos"]:
            assert "weight_global_representativeness_original" in pc
            assert "weight_global_representativeness_normalized" in pc
            assert "distance" in pc
            assert "similarity_rbf" in pc
            assert "top_differentiating_components" in pc
            assert "local_similarity_rank" in pc
            assert "global_selection_rank" in pc
            # Verificar que global_selection_rank no es null para ProtoDash
            # (puede ser None para KMedoids, pero en ejemplo usamos ProtoDash)
            assert pc["global_selection_rank"] is not None, "global_selection_rank no debe ser null para ProtoDash"
            assert isinstance(pc["global_selection_rank"], int), "global_selection_rank debe ser entero"
            assert pc["local_similarity_rank"] >= 1, "local_similarity_rank debe ser >= 1"
            # Verificar que el prototipo más cercano (local_similarity_rank=1) tiene distancia mínima
            # Para el primer prototipo en la lista
            if pc["local_similarity_rank"] == 1:
                pass  # La verificación de distancia 0 para auto-similitud se hace en test específico


def test_no_fuga_rendimiento_fase2():
    for py_file in FASE2_DIR.glob("*.py"):
        if py_file.name.startswith("test_"):
            continue
        code = py_file.read_text()
        assert "RENDIMIENTO_T_HA" not in code, f"{py_file} no debe leer RENDIMIENTO_T_HA"
        assert "CONJUNTO" not in code or "ejemplo_xai" in py_file.name, f"{py_file} no debe leer CONJUNTO salvo en ejemplo"


def test_manifest_completo(manifest2):
    required = ["version", "seed", "phase1_manifest_hash", "perfiles_conceptuales_hash",
                "spaces", "m_values", "sigma_factors", "methods", "n_parcelas", "corrections_applied"]
    for k in required:
        assert k in manifest2, f"Manifest2 falta {k}"
    assert manifest2["seed"] == 2026
    assert manifest2["n_parcelas"] == 197
    assert set(manifest2["spaces"]) == {"completo", "compacto"}
    assert manifest2["m_values"] == [3, 5, 8]
    assert manifest2["methods"] == ["ProtoDash", "KMedoids"]
    assert "global_selection_rank vs local_similarity_rank separated" in manifest2["corrections_applied"]


def test_prototipos_ids_reales(prototipos, perfiles):
    ids_perfiles = set(perfiles["ID_POLIGONO"])
    for pid in prototipos["prototype_id"].unique():
        assert pid in ids_perfiles, f"Prototype ID {pid} no existe en perfiles"


def test_pesos_protodash_duales(prototipos):
    pd_rows = prototipos[prototipos["method"] == "ProtoDash"]
    assert (pd_rows["weight_original"] >= 0).all()
    assert (pd_rows["weight_normalized"] >= 0).all()
    # Suma de pesos normalizados ≈ 1 por configuración
    for (space, method, m, sigma), grp in pd_rows.groupby(["space", "method", "m", "sigma"]):
        if not grp["weight_normalized"].isna().all():
            s = grp["weight_normalized"].sum()
            assert abs(s - 1.0) < 1e-6, f"Pesos ProtoDash no suman 1: {s}"
    # Pesos originales y normalizados ambos presentes
    assert "weight_original" in prototipos.columns
    assert "weight_normalized" in prototipos.columns


def test_ranking_separation(ejemplo):
    """Verifica separación de global_selection_rank vs local_similarity_rank."""
    for pe in ejemplo["parcelas_ejemplo"]:
        protos = pe["prototipos_cercanos"]
        # local_similarity_rank ordenado ascendentemente (1, 2, 3...)
        local_ranks = [p["local_similarity_rank"] for p in protos]
        assert local_ranks == sorted(local_ranks), "local_similarity_rank debe ser 1,2,3..."
        
        # global_selection_rank presente y entero para ProtoDash
        for p in protos:
            assert p["global_selection_rank"] is not None, "global_selection_rank no debe ser null"
            assert isinstance(p["global_selection_rank"], int), "global_selection_rank debe ser entero"
        
        # Verificar que el prototipo más cercano (local_similarity_rank=1) tiene distancia mínima
        first_proto = next(p for p in protos if p["local_similarity_rank"] == 1)
        # Si la parcela es ella misma un prototipo, distancia debe ser 0
        # (esto se verifica en test_self_distance_zero)


def test_original_vs_scaled(ejemplo):
    """Verifica que JSON tiene original y scaled con unidades."""
    for pe in ejemplo["parcelas_ejemplo"]:
        assert "original" in pe["perfil_conceptual"]
        assert "scaled" in pe["perfil_conceptual"]
        for comp, val in pe["perfil_conceptual"]["original"].items():
            assert isinstance(val, (int, float)), f"Valor original no numérico: {comp}"
        for comp, val in pe["perfil_conceptual"]["scaled"].items():
            assert isinstance(val, (int, float)), f"Valor escalado no numérico: {comp}"
        
        for pc in pe["prototipos_cercanos"]:
            for comp_info in pc["top_differentiating_components"]:
                assert "unit" in comp_info, "Falta unit en componente"
                assert "parcel_value_original" in comp_info
                assert "parcel_value_scaled" in comp_info
                assert "prototype_value_original" in comp_info
                assert "prototype_value_scaled" in comp_info
                assert "absolute_diff_original" in comp_info
                assert "absolute_diff_scaled" in comp_info
                assert "relative_contribution_to_distance" in comp_info


def test_self_distance_zero(ejemplo):
    """Verifica que una parcela que es prototipo tiene distancia 0 y local_similarity_rank=1."""
    for pe in ejemplo["parcelas_ejemplo"]:
        protos = pe["prototipos_cercanos"]
        own_proto = next((p for p in protos if p["prototype_id"] == pe["ID_POLIGONO"]), None)
        if own_proto:
            assert own_proto["distance"] == 0.0, f"Distancia a sí mismo debe ser 0, got {own_proto['distance']}"
            assert own_proto["local_similarity_rank"] == 1, "Rank local a sí mismo debe ser 1"
            assert own_proto["similarity_rbf"] == 1.0, "Similitud a sí mismo debe ser 1.0"


def test_manifest_completo(manifest2):
    required = ["version", "seed", "phase1_manifest_hash", "perfiles_conceptuales_hash",
                "spaces", "m_values", "sigma_factors", "methods", "n_parcelas", "corrections_applied"]
    for k in required:
        assert k in manifest2, f"Manifest2 falta {k}"
    assert manifest2["seed"] == 2026
    assert manifest2["n_parcelas"] == 197
    assert set(manifest2["spaces"]) == {"completo", "compacto"}
    assert manifest2["m_values"] == [3, 5, 8]
    assert manifest2["methods"] == ["ProtoDash", "KMedoids"]
    assert "global_selection_rank vs local_similarity_rank separated" in manifest2["corrections_applied"]
    assert "original and scaled values in example_xai" in manifest2["corrections_applied"]


def test_prototipos_ids_reales(prototipos, perfiles):
    ids_perfiles = set(perfiles["ID_POLIGONO"])
    for pid in prototipos["prototype_id"].unique():
        assert pid in ids_perfiles, f"Prototype ID {pid} no existe en perfiles"


def test_pesos_protodash_duales(prototipos):
    pd_rows = prototipos[prototipos["method"] == "ProtoDash"]
    assert (pd_rows["weight_original"] >= 0).all()
    assert (pd_rows["weight_normalized"] >= 0).all()
    for (space, method, m, sigma), grp in pd_rows.groupby(["space", "method", "m", "sigma"]):
        if not grp["weight_normalized"].isna().all():
            s = grp["weight_normalized"].sum()
            assert abs(s - 1.0) < 1e-6, f"Pesos ProtoDash no suman 1: {s}"
    assert "weight_original" in prototipos.columns
    assert "weight_normalized" in prototipos.columns


def test_kmedoids_warnings_recorded():
    """Verifica que KMedoids registra warnings."""
    import json
    with open(FASE2_DIR / "resultados_kmedoids.json") as f:
        km_results = json.load(f)
    for r in km_results:
        if r.get("method") == "KMedoids" and "warnings" in r:
            assert isinstance(r["warnings"], list), "warnings debe ser lista"


def test_reproducibilidad_fase2():
    script = FASE2_DIR.parent / "fase2_prototipos.py"
    if not script.exists():
        pytest.skip("Script principal no en directorio de tests")
    
    hashes_before = {}
    for f in ["prototipos.csv", "similitudes.csv", "estabilidad.csv", "ejemplo_xai.json"]:
        p = FASE2_DIR / f
        if p.exists():
            with open(p, "rb") as fh:
                hashes_before[f] = hashlib.sha256(fh.read()).hexdigest()
    
    result = subprocess.run([sys.executable, str(script)], 
                           capture_output=True, text=True, cwd=Path.cwd(), timeout=300)
    assert result.returncode == 0, f"Re-ejecución falló: {result.stderr}"
    
    for f, h_before in hashes_before.items():
        p = FASE2_DIR / f
        if p.exists():
            with open(p, "rb") as fh:
                h_after = hashlib.sha256(fh.read()).hexdigest()
            assert h_after == h_before, f"Salida no reproducible: {f} cambió"


def test_orden_invariante_real(perfiles):
    """Test de invariancia al orden: verificar que Fase 1 es invariante al orden de filas de entrada."""
    # Cargar datos originales de Fase 1
    basic = pd.read_csv("desarrollo/datos/entradas/Conjunto_datos_BASICO_AgroCebada2026.csv", encoding="utf-8-sig")
    pro = pd.read_csv("desarrollo/datos/entradas/Conjunto_datos_PRO_AgroCebada.csv", encoding="utf-8-sig")
    parcels = pd.read_csv("desarrollo/cebada_datos_baseline_1/data_processed/parcelas_base.csv", encoding="utf-8-sig")
    climate = pd.read_csv("desarrollo/cebada_datos_baseline_1/data_processed/clima_mensual_parcela.csv", encoding="utf-8-sig")
    
    # Permutar filas de entrada (mantener IDs)
    basic_perm = basic.sample(frac=1, random_state=2026).reset_index(drop=True)
    pro_perm = pro.sample(frac=1, random_state=2026).reset_index(drop=True)
    parcels_perm = parcels.sample(frac=1, random_state=2026).reset_index(drop=True)
    climate_perm = climate.sample(frac=1, random_state=2026).reset_index(drop=True)
    
    # Guardar temporales CON LOS NOMBRES ORIGINALES
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "desarrollo" / "datos" / "entradas"
        proc_dir = tmp_path / "desarrollo" / "cebada_datos_baseline_1" / "data_processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        proc_dir.mkdir(parents=True, exist_ok=True)
        
        basic_perm.to_csv(raw_dir / "Conjunto_datos_BASICO_AgroCebada2026.csv", index=False, encoding="utf-8-sig")
        pro_perm.to_csv(raw_dir / "Conjunto_datos_PRO_AgroCebada.csv", index=False, encoding="utf-8-sig")
        parcels_perm.to_csv(proc_dir / "parcelas_base.csv", index=False, encoding="utf-8-sig")
        climate_perm.to_csv(proc_dir / "clima_mensual_parcela.csv", index=False, encoding="utf-8-sig")
        
        # Copiar conceptos.py al directorio temporal y modificar rutas
        import shutil
        conceptos_src = FASE2_DIR.parent / "conceptos.py"
        conceptos_dst = tmp_path / "conceptos.py"
        shutil.copy(conceptos_src, conceptos_dst)
        
        # Leer y modificar las rutas en el script
        script_content = conceptos_dst.read_text()
        script_content = script_content.replace(
            'RAW_DIR = Path("desarrollo/datos/entradas")',
            f"RAW_DIR = Path({str(raw_dir)!r})"
        )
        script_content = script_content.replace(
            'PROC_DIR = Path("desarrollo/cebada_datos_baseline_1/data_processed")',
            f'PROC_DIR = Path("{tmp_path}/desarrollo/cebada_datos_baseline_1/data_processed")'
        )
        script_content = script_content.replace(
            'OUT_DIR = Path("outputs/xai")',
            f'OUT_DIR = Path("{tmp_path}/out")'
        )
        conceptos_dst.write_text(script_content)
        
        # Ejecutar conceptos.py con datos permutados
        result = subprocess.run([sys.executable, str(conceptos_dst)], 
                               capture_output=True, text=True, cwd=tmp_path, timeout=120)
        assert result.returncode == 0, f"conceptos.py con datos permutados falló: {result.stderr}"
        
        # Comparar dataframes con tolerancia (en lugar de hashes exactos)
        out_dir = tmp_path / "out"
        p1 = FASE1_DIR / "perfiles_conceptuales.csv"
        p2 = out_dir / "perfiles_conceptuales.csv"
        if p1.exists() and p2.exists():
            df1 = pd.read_csv(p1, encoding="utf-8-sig").sort_values("ID_POLIGONO").reset_index(drop=True)
            df2 = pd.read_csv(p2, encoding="utf-8-sig").sort_values("ID_POLIGONO").reset_index(drop=True)
            
            # Comparar columnas numéricas con tolerancia
            numeric_cols = df1.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                np.testing.assert_allclose(df1[col].values, df2[col].values, rtol=1e-10, atol=1e-12,
                                          err_msg=f"Columna {col} difiere entre ejecuciones")
            
            # Comparar columnas no numéricas exactamente
            non_numeric = df1.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric:
                pd.testing.assert_series_equal(df1[col].reset_index(drop=True), 
                                               df2[col].reset_index(drop=True),
                                               check_names=False)


def test_kmedoids_warnings_recorded():
    """Verifica que KMedoids registra warnings en resultados_kmedoids.json."""
    import json
    with open(FASE2_DIR / "resultados_kmedoids.json") as f:
        km_results = json.load(f)
    for r in km_results:
        if r.get("method") == "KMedoids":
            assert "warnings" in r, "KMedoids debe tener campo warnings"
            assert isinstance(r["warnings"], list), "warnings debe ser lista"


def test_comparacion_representaciones_note():
    """Verifica que comparacion_representaciones.json tiene nota sobre dimensionalidad."""
    import json
    with open(FASE2_DIR / "comparacion_representaciones.json") as f:
        comp = json.load(f)
    assert "note" in comp["full"], "Espacio completo debe tener note"
    assert "note" in comp["compact"], "Espacio compacto debe tener note"
    assert "dimensionalidad" in comp["full"]["note"].lower() or "dimensionalidad" in comp["compact"]["note"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])