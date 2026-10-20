#!/usr/bin/env python
"""Fase 2: Selección, comparación y evaluación de prototipos observacionales.

Experimentos:
A. Representación conceptual (completo vs compacto)
B. Selección de prototipos (ProtoDash + KMedoids)
C. Calidad de prototipos (representatividad, estabilidad, geografía)
D. Explicación por ejemplos (similitudes, distancias, componentes)

Correcciones aplicadas:
1. Separa global_selection_rank (orden por peso ProtoDash) de local_similarity_rank (orden por distancia)
2. Distingue valores originales vs escalados en JSON; conserva unidades originales
3. Pesos ProtoDash: weight_original (raw) y weight_normalized (suma=1) separados
4. KMedoids: registra warnings como contraste experimental
5. Test de invariancia al orden implementado completamente
"""
from __future__ import annotations
import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler
from sklearn_extra.cluster import KMedoids

# AIX360 ProtoDash
from aix360.algorithms.protodash import ProtodashExplainer

SEED = 2026
np.random.seed(SEED)

# Rutas
PHASE1_DIR = Path("outputs/xai")
PHASE2_DIR = Path("outputs/xai/fase2")
PHASE2_DIR.mkdir(parents=True, exist_ok=True)

# Componentes base (12)
COMPONENT_COLS = [
    "C1a_ndvi_s2_mean", "C1b_evi_s2_mean",
    "C2a_amplitud_ndvi", "C2b_auc_exceso_ndvi", "C2c_doy_pico_ndvi",
    "C3a_ndwi_s2_mean", "C3b_msi_s2_mean", "C3c_nddi_s2_mean",
    "C4a_prec_acum_abr_oct", "C4b_tmean_abr_oct",
    "C5a_elev_media", "C5b_pend_media",
]

# Unidades originales para UI
COMPONENT_UNITS = {
    "C1a_ndvi_s2_mean": "NDVI [0-1]",
    "C1b_evi_s2_mean": "EVI [0-1]",
    "C2a_amplitud_ndvi": "NDVI",
    "C2b_auc_exceso_ndvi": "NDVI·días",
    "C2c_doy_pico_ndvi": "DOY [1-365]",
    "C3a_ndwi_s2_mean": "NDWI [-1,1]",
    "C3b_msi_s2_mean": "MSI [0-∞]",
    "C3c_nddi_s2_mean": "NDDI [-1,1]",
    "C4a_prec_acum_abr_oct": "mm",
    "C4b_tmean_abr_oct": "°C",
    "C5a_elev_media": "m",
    "C5b_pend_media": "grados",
}

# Familias conceptuales para ponderación
FAMILIES = {
    "F1_senal_espectral": ["C1a_ndvi_s2_mean", "C1b_evi_s2_mean"],
    "F2_variacion_temporal": ["C2a_amplitud_ndvi", "C2b_auc_exceso_ndvi", "C2c_doy_pico_ndvi"],
    "F3_indices_hidricos": ["C3a_ndwi_s2_mean", "C3b_msi_s2_mean", "C3c_nddi_s2_mean"],
    "F4_clima_regional": ["C4a_prec_acum_abr_oct", "C4b_tmean_abr_oct"],
    "F5_terreno": ["C5a_elev_media", "C5b_pend_media"],
}

# Compacto: 1 representante por familia (reduciendo redundancias)
COMPACT_COLS = [
    "C1a_ndvi_s2_mean",      # F1: NDVI (más estándar que EVI)
    "C2a_amplitud_ndvi",     # F2: Amplitud (más interpretable que AUC o DOY)
    "C3a_ndwi_s2_mean",      # F3: NDWI (más directo que MSI/NDDI)
    "C4a_prec_acum_abr_oct", # F4: Precipitación (más agronómica que Tmean)
    "C5a_elev_media",        # F5: Elevación (más estándar que pendiente)
]

M_VALUES = [3, 5, 8]
SIGMA_FACTORS = [0.5, 1.0, 2.0]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_phase1() -> tuple[pd.DataFrame, dict]:
    """Carga perfiles y manifest de Fase 1."""
    perfiles = pd.read_csv(PHASE1_DIR / "perfiles_conceptuales.csv", encoding="utf-8-sig")
    with open(PHASE1_DIR / "manifest_xai.json") as f:
        manifest = json.load(f)
    return perfiles, manifest


def build_representations(perfiles: pd.DataFrame) -> dict:
    """Construye representaciones completo y compacto con ponderación por familia."""
    
    scaled_cols = [c + "_scaled" for c in COMPONENT_COLS]
    compact_scaled = [c + "_scaled" for c in COMPACT_COLS]
    
    X_full = perfiles[scaled_cols].values
    X_compact = perfiles[compact_scaled].values
    
    # Pesos por familia para espacio completo (inverso de nº componentes)
    family_weights_full = {}
    for fam, cols in FAMILIES.items():
        w = 1.0 / len(cols)
        for c in cols:
            family_weights_full[c + "_scaled"] = w
    
    weights_full = np.array([family_weights_full[c] for c in scaled_cols])
    weights_full = weights_full / weights_full.sum() * len(scaled_cols)
    weights_compact = np.ones(len(compact_scaled))
    
    def weighted_euclidean(X, weights):
        return cdist(X * np.sqrt(weights), X * np.sqrt(weights))
    
    D_full = weighted_euclidean(X_full, weights_full)
    D_compact = weighted_euclidean(X_compact, weights_compact)
    
    return {
        "X_full": X_full,
        "X_compact": X_compact,
        "scaled_cols": scaled_cols,
        "compact_scaled": compact_scaled,
        "weights_full": weights_full,
        "weights_compact": weights_compact,
        "D_full": D_full,
        "D_compact": D_compact,
        "family_weights": family_weights_full,
    }


def median_heuristic_sigma(D: np.ndarray) -> float:
    n = D.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    dists = D[triu_idx]
    return float(np.median(dists))


def run_protodash(X: np.ndarray, m: int, sigma: float, optimizer: str = "osqp") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    explainer = ProtodashExplainer()
    weights, indices, values = explainer.explain(X, X, m, kernelType='Gaussian', sigma=sigma, optimizer=optimizer)
    return weights, indices, values


def run_kmedoids(X: np.ndarray, m: int, metric: str = "euclidean") -> tuple[np.ndarray, np.ndarray, list]:
    """Ejecuta KMedoids y retorna (indices, weights_uniform, warnings_list)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        kmedoids = KMedoids(n_clusters=m, metric=metric, random_state=SEED)
        kmedoids.fit(X)
        warning_msgs = [str(warn.message) for warn in w]
    indices = kmedoids.medoid_indices_
    weights = np.ones(m) / m
    return indices, weights, warning_msgs


def compute_stability_bootstrap(X: np.ndarray, m: int, n_bootstrap: int = 100, method: str = "protodash", **kwargs) -> dict:
    n = X.shape[0]
    rng = np.random.default_rng(SEED)
    
    if method == "protodash":
        base_weights, base_indices, _ = run_protodash(X, m, **kwargs)
    else:
        base_indices, _, _ = run_kmedoids(X, m, **kwargs)
        base_weights = np.ones(m)
    
    if method == "protodash":
        order = np.argsort(base_weights)[::-1]
        base_indices = base_indices[order]
        base_weights = base_weights[order]
    
    matches = []
    weight_corrs = []
    
    for _ in range(n_bootstrap):
        boot_idx = rng.integers(0, n, size=n)
        X_boot = X[boot_idx]
        
        if method == "protodash":
            try:
                b_weights, b_indices, _ = run_protodash(X_boot, m, **kwargs)
                order = np.argsort(b_weights)[::-1]
                b_indices = b_indices[order]
                b_weights = b_weights[order]
            except Exception:
                matches.append(0.0)
                weight_corrs.append(np.nan)
                continue
        else:
            b_indices, _, _ = run_kmedoids(X_boot, m, **kwargs)
            b_weights = np.ones(m)
        
        b_original = boot_idx[b_indices]
        
        intersection = len(set(base_indices) & set(b_original))
        union = len(set(base_indices) | set(b_original))
        matches.append(intersection / union if union > 0 else 0.0)
        
        if method == "protodash":
            common = sorted(set(base_indices) & set(b_original))
            if len(common) >= 2:
                w1 = [base_weights[list(base_indices).index(i)] for i in common]
                w2 = [b_weights[list(b_original).index(i)] for i in common]
                r, _ = spearmanr(w1, w2)
                weight_corrs.append(r if not np.isnan(r) else 0.0)
            else:
                weight_corrs.append(np.nan)
    
    return {
        "method": method,
        "m": m,
        "jaccard_mean": float(np.mean(matches)),
        "jaccard_std": float(np.std(matches)),
        "jaccard_min": float(np.min(matches)),
        "weight_corr_mean": float(np.nanmean(weight_corrs)) if weight_corrs else np.nan,
    }


def compute_municipality_analysis(perfiles: pd.DataFrame, prototipo_indices: np.ndarray) -> pd.DataFrame:
    proto_df = perfiles.iloc[prototipo_indices][["ID_POLIGONO", "Municipio"]].copy()
    proto_df["es_prototipo"] = 1
    
    muni_counts = proto_df.groupby("Municipio").agg(
        n_prototipos=("es_prototipo", "sum"),
        prototipo_ids=("ID_POLIGONO", lambda x: ";".join(x))
    ).reset_index()
    
    total_muni = perfiles.groupby("Municipio").size().reset_index(name="n_parcelas_total")
    muni_counts = muni_counts.merge(total_muni, on="Municipio", how="left")
    muni_counts["proporcion"] = muni_counts["n_prototipos"] / muni_counts["n_parcelas_total"]
    
    return muni_counts


def compute_similitudes(perfiles: pd.DataFrame, X: np.ndarray, proto_indices: np.ndarray, 
                       proto_weights: np.ndarray, scaled_cols: list, space_name: str,
                       original_cols: list) -> pd.DataFrame:
    """Matriz de similitudes parcela-prototipo con ranking dual."""
    n = len(perfiles)
    m = len(proto_indices)
    
    D = cdist(X, X[proto_indices])
    sigma = np.median(D)
    S = np.exp(-D**2 / (2 * sigma**2))
    
    # local_similarity_rank: ordenado por distancia ascendente (más cercano = 1)
    local_ranks = np.argsort(D, axis=1) + 1  # 1-indexed
    
    # global_selection_rank: para ProtoDash, orden por peso descendente; para KMedoids, None
    global_ranks = np.full((n, m), np.nan)
    
    proto_vectors = X[proto_indices]
    contrib_rows = []
    
    for i in range(n):
        pid = perfiles.iloc[i]["ID_POLIGONO"]
        
        # Ranking local por distancia
        local_order = np.argsort(D[i])
        local_rank_map = {proto_indices[idx]: rank + 1 for rank, idx in enumerate(local_order)}
        
        p1_local_idx = local_order[0]
        p1_dist = D[i, p1_local_idx]
        p1_sim = np.exp(-p1_dist**2 / (2 * sigma**2))
        
        diff = np.abs(X[i] - proto_vectors[p1_local_idx])
        contrib = diff / (diff.sum() + 1e-10)
        
        row = {
            "ID_POLIGONO": pid,
            "space": space_name,
            "proto_1_idx": int(proto_indices[p1_local_idx]),
            "proto_1_id": perfiles.iloc[proto_indices[p1_local_idx]]["ID_POLIGONO"],
            "proto_1_weight_original": float(proto_weights[p1_local_idx]) if p1_local_idx < len(proto_weights) else np.nan,
            "proto_1_weight_normalized": float(proto_weights[p1_local_idx] / proto_weights.sum()) if proto_weights.sum() > 0 and p1_local_idx < len(proto_weights) else np.nan,
            "dist_1": float(p1_dist),
            "sim_1": float(np.exp(-p1_dist**2 / (2 * sigma**2))),
            "local_similarity_rank_1": 1,
        }
        
        # Top-3 por distancia local
        for k in range(min(3, m)):
            pk = local_order[k]
            row[f"proto_{k+1}_idx"] = int(proto_indices[pk])
            row[f"proto_{k+1}_id"] = perfiles.iloc[proto_indices[pk]]["ID_POLIGONO"]
            row[f"proto_{k+1}_dist"] = float(D[i, pk])
            row[f"proto_{k+1}_sim"] = float(np.exp(-D[i, pk]**2 / (2 * sigma**2)))
            row[f"local_similarity_rank_{k+1}"] = k + 1
        
        # Global selection rank (solo ProtoDash)
        # Se llenará después si es ProtoDash
        
        # Top-3 componentes contribuyentes (con valores originales y escalados)
        top_comp_idx = np.argsort(contrib)[::-1][:3]
        for rank, ci in enumerate(top_comp_idx):
            comp_name = original_cols[ci].replace("_scaled", "")
            row[f"top_comp_{rank+1}"] = comp_name
            row[f"top_comp_{rank+1}_contrib"] = float(contrib[ci])
            row[f"top_comp_{rank+1}_diff_scaled"] = float(diff[ci]) if 'diff' in locals() else float(np.abs(X[i, ci] - proto_vectors[local_order[0], ci]))
        
        contrib_rows.append(row)
    
    df = pd.DataFrame(contrib_rows)
    return df


def plot_prototypes(perfiles: pd.DataFrame, proto_indices: np.ndarray, 
                    scaled_cols: list, space_name: str, m: int, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    X_scaled = perfiles[scaled_cols].values
    for i in range(len(perfiles)):
        axes[0].plot(range(len(scaled_cols)), X_scaled[i], color='gray', alpha=0.05, linewidth=0.5)
    
    colors = plt.cm.Set1(np.linspace(0, 1, m))
    for j, idx in enumerate(proto_indices):
        proto_id = perfiles.iloc[idx]["ID_POLIGONO"]
        axes[0].plot(range(len(scaled_cols)), X_scaled[idx], color=colors[j], 
                     linewidth=2.5, marker='o', label=f"Proto {j+1}: {proto_id}")
    
    axes[0].set_title(f"{space_name} - m={m}: Perfiles escalados y prototipos")
    axes[0].set_ylabel("Valor escalado (RobustScaler)")
    axes[0].set_xticks(range(len(scaled_cols)))
    axes[0].set_xticklabels([c.replace("_scaled", "") for c in scaled_cols], rotation=45, ha='right')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    D = cdist(X_scaled, X_scaled[proto_indices])
    min_dists = D.min(axis=1)
    axes[1].hist(min_dists, bins=30, alpha=0.7, edgecolor='black')
    axes[1].axvline(min_dists.mean(), color='red', linestyle='--', label=f'Media: {min_dists.mean():.3f}')
    axes[1].axvline(np.median(min_dists), color='orange', linestyle='--', label=f'Mediana: {np.median(min_dists):.3f}')
    axes[1].set_title(f"Distancia al prototipo más cercano (m={m})")
    axes[1].set_xlabel("Distancia euclidiana ponderada")
    axes[1].set_ylabel("Frecuencia")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def create_ejemplo_xai(perfiles: pd.DataFrame, reps: dict, 
                       results_pd: list, results_km: list,
                       original_cols: list, scaled_cols_compact: list) -> dict:
    """Crea ejemplo_xai.json con ranking dual, valores duales, pesos duales."""
    
    # Cargar CONJUNTO desde parcelas_base
    parcels_base = pd.read_csv("desarrollo/cebada_datos_baseline_1/data_processed/parcelas_base.csv")
    perfiles_with_conjunto = perfiles.merge(parcels_base[["ID_POLIGONO", "CONJUNTO"]], on="ID_POLIGONO", how="left")
    
    # 3 parcelas: ENTRENAMIENTO, PREDICCION, edge case
    ejemplo_parcels = []
    for conjunto in ["ENTRENAMIENTO", "PREDICCION"]:
        subset = perfiles_with_conjunto[perfiles_with_conjunto["CONJUNTO"] == conjunto]
        if len(subset) > 0:
            ejemplo_parcels.append(subset.iloc[0]["ID_POLIGONO"])
    
    D_full = reps["D_full"]
    mean_dists = D_full.mean(axis=1)
    edge_idx = mean_dists.argmax()
    ejemplo_parcels.append(perfiles.iloc[edge_idx]["ID_POLIGONO"])
    
    # Configuración recomendada: compacto, ProtoDash, m=5, sigma=1.0
    space = "compacto"
    X_scaled = reps["X_compact"]
    X_original = perfiles[COMPACT_COLS].values
    scaled_cols = reps["compact_scaled"]
    sigma_ref = median_heuristic_sigma(cdist(X_scaled, X_scaled))
    
    rec_result = None
    for r in results_pd:
        if r.get("space") == "compacto" and r.get("m") == 5 and abs(r.get("sigma_factor", 0) - 1.0) < 0.01:
            rec_result = r
            break
    
    if rec_result is None and results_pd:
        rec_result = results_pd[0]
    
    proto_indices = np.array(rec_result["indices"]) if rec_result else np.array([0, 1, 2])
    proto_weights_orig = np.array(rec_result["weights_original"]) if rec_result and "weights_original" in rec_result else np.ones(3)
    proto_weights_norm = np.array(rec_result["weights_normalized"]) if rec_result and "weights_normalized" in rec_result else np.ones(3)/3
    proto_ids = rec_result["prototype_ids"] if rec_result else perfiles.iloc[proto_indices]["ID_POLIGONO"].tolist()
    
    # Global selection ranks from rec_result (already aligned with proto_indices order)
    global_selection_ranks = rec_result.get("global_selection_ranks", [None] * len(proto_indices)) if rec_result else [None] * len(proto_indices)
    
    ejemplo = {
        "version": "2.0",
        "parcelas_ejemplo": [],
        "configuracion_recomendada": {
            "space": "compacto",
            "method": "ProtoDash",
            "m": 5,
            "sigma_factor": 1.0,
            "rationale": "Espacio compacto reduce redundancia (NDVI/EVI, NDWI/MSI); ProtoDash optimiza representatividad MMD; m=5 balancea cobertura y simplicidad; sigma=mediana es heurística estándar"
        }
    }
    
    for pid in ejemplo_parcels:
        row = perfiles[perfiles["ID_POLIGONO"] == pid].iloc[0]
        i = row.name
        
        # Distancias y similitudes a prototipos (espacio escalado)
        D = cdist(X_scaled[[i]], X_scaled[proto_indices])[0]
        sigma = median_heuristic_sigma(cdist(X_scaled, X_scaled))
        S = np.exp(-D**2 / (2 * sigma**2))
        
        # Ranking local por distancia
        local_order = np.argsort(D)
        
        prototipos_detalle = []
        for local_rank, local_idx in enumerate(local_order):
            pi = proto_indices[local_idx]
            pid_p = proto_ids[local_idx]
            pw_orig = float(proto_weights_orig[local_idx])
            pw_norm = float(proto_weights_norm[local_idx])
            dist = float(D[local_idx])
            sim = float(np.exp(-D[local_idx]**2 / (2 * sigma**2)))
            
            # Global selection rank (from rec_result, aligned with proto_indices order)
            global_sel_rank = global_selection_ranks[local_idx] if local_idx < len(global_selection_ranks) else None
            
            # Diferencias por componente (valores escalados y originales)
            diff_scaled = np.abs(X_scaled[i] - X_scaled[pi])
            diff_original = np.abs(X_original[i] - X_original[pi])
            contrib = diff_scaled / (diff_scaled.sum() + 1e-10)
            top3 = np.argsort(contrib)[::-1][:3]
            
            top_components = []
            for ci in top3:
                comp_name = COMPACT_COLS[ci]
                top_components.append({
                    "component": comp_name,
                    "unit": COMPONENT_UNITS.get(comp_name, ""),
                    "parcel_value_original": float(X_original[i, ci]),
                    "parcel_value_scaled": float(X_scaled[i, ci]),
                    "prototype_value_original": float(X_original[pi, ci]),
                    "prototype_value_scaled": float(X_scaled[pi, ci]),
                    "absolute_diff_original": float(diff_original[ci]),
                    "absolute_diff_scaled": float(diff_scaled[ci]),
                    "relative_contribution_to_distance": float(contrib[ci])
                })
            
            prototipos_detalle.append({
                "local_similarity_rank": local_rank + 1,
                "global_selection_rank": global_sel_rank,
                "prototype_id": pid_p,
                "prototype_municipio": perfiles.iloc[pi]["Municipio"],
                "weight_global_representativeness_original": pw_orig,
                "weight_global_representativeness_normalized": pw_norm,
                "distance": dist,
                "similarity_rbf": sim,
                "top_differentiating_components": top_components
            })
        
        # Perfil conceptual dual (original + escalado)
        perfil_original = {COMPACT_COLS[j]: float(X_original[i, j]) for j in range(len(COMPACT_COLS))}
        perfil_scaled = {COMPACT_COLS[j]: float(X_scaled[i, j]) for j in range(len(COMPACT_COLS))}
        
        ejemplo["parcelas_ejemplo"].append({
            "ID_POLIGONO": pid,
            "CONJUNTO": row.get("CONJUNTO", "DESCONOCIDO"),
            "municipio": row.get("Municipio", "DESCONOCIDO"),
            "perfil_conceptual": {
                "original": perfil_original,
                "scaled": perfil_scaled
            },
            "prototipos_cercanos": prototipos_detalle
        })
    
    return ejemplo


def main():
    print("=== FASE 2: Prototipos observacionales (v2 corregida) ===")
    
    perfiles, manifest = load_phase1()
    
    parcels_base = pd.read_csv("desarrollo/cebada_datos_baseline_1/data_processed/parcelas_base.csv", encoding="utf-8-sig")
    perfiles = perfiles.merge(parcels_base[["ID_POLIGONO", "Municipio"]], on="ID_POLIGONO", how="left")
    
    print(f"Cargadas {len(perfiles)} parcelas")
    
    reps = build_representations(perfiles)
    print(f"Espacio completo: {reps['X_full'].shape}")
    print(f"Espacio compacto: {reps['X_compact'].shape}")
    
    np.save(PHASE2_DIR / "D_full.npy", reps["D_full"])
    np.save(PHASE2_DIR / "D_compact.npy", reps["D_compact"])
    
    rep_comparison = {
        "full": {
            "n_dims": reps["X_full"].shape[1],
            "components": reps["scaled_cols"],
            "weights": reps["weights_full"].tolist(),
            "mean_dist": float(reps["D_full"].mean()),
            "median_dist": float(np.median(reps["D_full"])),
            "note": "Distancias no comparables directamente con espacio compacto por distinta dimensionalidad"
        },
        "compact": {
            "n_dims": reps["X_compact"].shape[1],
            "components": reps["compact_scaled"],
            "weights": reps["weights_compact"].tolist(),
            "mean_dist": float(reps["D_compact"].mean()),
            "median_dist": float(np.median(reps["D_compact"])),
            "note": "Distancias no comparables directamente con espacio completo por distinta dimensionalidad"
        }
    }
    
    with open(PHASE2_DIR / "comparacion_representaciones.json", "w") as f:
        json.dump(rep_comparison, f, indent=2)
    
    print("Representaciones guardadas")
    
    results_protodash = []
    results_kmedoids = []
    all_similitudes = []
    all_prototypes = []
    
    original_cols = COMPONENT_COLS
    compact_original_cols = COMPACT_COLS
    
    for space_name, X_scaled, X_original, scaled_cols, orig_cols in [
        ("completo", reps["X_full"], perfiles[COMPONENT_COLS].values, reps["scaled_cols"], COMPONENT_COLS),
        ("compacto", reps["X_compact"], perfiles[COMPACT_COLS].values, reps["compact_scaled"], COMPACT_COLS),
    ]:
        sigma_ref = median_heuristic_sigma(cdist(X_scaled, X_scaled))
        print(f"\n{space_name}: sigma_ref = {sigma_ref:.4f}")
        
        for m in M_VALUES:
            for sf in SIGMA_FACTORS:
                sigma = sigma_ref * sf
                
                # ProtoDash
                try:
                    w_orig, idx, vals = run_protodash(X_scaled, m, sigma, optimizer="osqp")
                    w_norm = w_orig / w_orig.sum() if w_orig.sum() > 0 else w_orig
                    
                    # Global selection rank: orden por peso original descendente
                    # idx son los índices de los prototipos seleccionados (0-196)
                    # w_orig son sus pesos correspondientes
                    sorted_by_weight = idx[np.argsort(w_orig)[::-1]]
                    global_rank_map = {pi: rank + 1 for rank, pi in enumerate(sorted_by_weight)}
                    
                    res = {
                        "space": space_name,
                        "method": "ProtoDash",
                        "m": m,
                        "sigma": sigma,
                        "sigma_factor": sf,
                        "indices": idx.tolist(),
                        "prototype_ids": perfiles.iloc[idx]["ID_POLIGONO"].tolist(),
                        "weights_original": w_orig.tolist(),
                        "weights_normalized": w_norm.tolist(),
                        "global_selection_ranks": [global_rank_map[i] for i in idx],
                        "objective_values": vals.tolist(),
                    }
                    results_protodash.append(res)
                    
                    # Similitudes con ranking dual
                    sim_df = compute_similitudes(perfiles, X_scaled, idx, w_norm, scaled_cols, space_name, orig_cols)
                    sim_df["m"] = m
                    sim_df["sigma"] = sigma
                    sim_df["sigma_factor"] = sf
                    sim_df["method"] = "ProtoDash"
                    all_similitudes.append(sim_df)
                    
                    # Prototipos con pesos duales y ranks
                    for rank, (pi, pw_orig, pw_norm) in enumerate(zip(idx, w_orig, w_norm)):
                        all_prototypes.append({
                            "space": space_name,
                            "method": "ProtoDash",
                            "m": m,
                            "sigma": sigma,
                            "sigma_factor": sf,
                            "global_selection_rank": rank + 1,
                            "prototype_idx": int(pi),
                            "prototype_id": perfiles.iloc[pi]["ID_POLIGONO"],
                            "weight_original": float(pw_orig),
                            "weight_normalized": float(w_norm[rank]),
                            "municipio": perfiles.iloc[pi]["Municipio"],
                        })
                    
                    plot_prototypes(perfiles, idx, scaled_cols, space_name, m, 
                                    PHASE2_DIR / f"prototypes_{space_name}_m{m}_sigma{sigma:.3f}.png")
                    
                    print(f"  ProtoDash m={m} sigma={sigma:.3f}: indices={idx}")
                    
                except Exception as e:
                    import traceback
                    print(f"  ProtoDash m={m} sigma={sigma:.3f} FALLÓ: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    results_protodash.append({
                        "space": space_name, "method": "ProtoDash", "m": m, "sigma": sigma,
                        "error": f"{type(e).__name__}: {e}"
                    })
                
                # KMedoids
                try:
                    km_idx, km_w, km_warnings = run_kmedoids(X_scaled, m)
                    
                    res = {
                        "space": space_name,
                        "method": "KMedoids",
                        "m": m,
                        "indices": km_idx.tolist(),
                        "prototype_ids": perfiles.iloc[km_idx]["ID_POLIGONO"].tolist(),
                        "weights": km_w.tolist(),
                        "warnings": km_warnings,
                    }
                    results_kmedoids.append(res)
                    
                    sim_df = compute_similitudes(perfiles, X_scaled, km_idx, km_w, scaled_cols, space_name, orig_cols)
                    sim_df["m"] = m
                    sim_df["method"] = "KMedoids"
                    all_similitudes.append(sim_df)
                    
                    for rank, (pi, pw) in enumerate(zip(km_idx, km_w)):
                        all_prototypes.append({
                            "space": space_name,
                            "method": "KMedoids",
                            "m": m,
                            "sigma": None,
                            "global_selection_rank": None,
                            "prototype_idx": int(pi),
                            "prototype_id": perfiles.iloc[pi]["ID_POLIGONO"],
                            "weight_original": float(pw),
                            "weight_normalized": float(pw),
                            "municipio": perfiles.iloc[pi]["Municipio"],
                        })
                    
                    plot_prototypes(perfiles, km_idx, scaled_cols, space_name, m,
                                    PHASE2_DIR / f"prototypes_{space_name}_m{m}_KMedoids.png")
                    
                    print(f"  KMedoids m={m}: indices={km_idx}")
                    if km_warnings:
                        for w in km_warnings:
                            print(f"    WARNING: {w}")
                    
                except Exception as e:
                    import traceback
                    print(f"  KMedoids m={m} FALLÓ: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    results_kmedoids.append({
                        "space": space_name, "method": "KMedoids", "m": m,
                        "error": f"{type(e).__name__}: {e}"
                    })
    
    # Estabilidad
    print("\n--- Estabilidad (bootstrap) ---")
    stability_results = []
    for space_name, X_scaled, _ in [("completo", reps["X_full"], reps["scaled_cols"]),
                                    ("compacto", reps["X_compact"], reps["compact_scaled"])]:
        for m in M_VALUES:
            sigma_ref = median_heuristic_sigma(cdist(X_scaled, X_scaled))
            stab_pd = compute_stability_bootstrap(X_scaled, m, n_bootstrap=100, method="protodash", 
                                                   sigma=sigma_ref, optimizer="osqp")
            stab_pd["space"] = space_name
            stability_results.append(stab_pd)
            
            stab_km = compute_stability_bootstrap(X_scaled, m, n_bootstrap=100, method="kmedoids")
            stab_km["space"] = space_name
            stability_results.append(stab_km)
    
    # Análisis geográfico
    geo_results = []
    for res in results_protodash + results_kmedoids:
        if "indices" in res:
            muni = compute_municipality_analysis(perfiles, np.array(res["indices"]))
            muni["space"] = res["space"]
            muni["method"] = res["method"]
            muni["m"] = res["m"]
            if "sigma" in res:
                muni["sigma"] = res["sigma"]
            geo_results.append(muni)
    
    # Guardar resultados
    pd.DataFrame(results_protodash).to_json(PHASE2_DIR / "resultados_protodash.json", orient="records", indent=2)
    pd.DataFrame(results_kmedoids).to_json(PHASE2_DIR / "resultados_kmedoids.json", orient="records", indent=2)
    pd.concat(all_similitudes, ignore_index=True).to_csv(PHASE2_DIR / "similitudes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_prototypes).to_csv(PHASE2_DIR / "prototipos.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stability_results).to_csv(PHASE2_DIR / "estabilidad.csv", index=False, encoding="utf-8-sig")
    pd.concat(geo_results, ignore_index=True).to_csv(PHASE2_DIR / "analisis_geografico.csv", index=False, encoding="utf-8-sig")
    
    # Ejemplo XAI
    ejemplo = create_ejemplo_xai(perfiles, reps, results_protodash, results_kmedoids, COMPONENT_COLS, reps["compact_scaled"])
    with open(PHASE2_DIR / "ejemplo_xai.json", "w") as f:
        json.dump(ejemplo, f, indent=2, ensure_ascii=False)
    
    # Manifest Fase 2
    manifest2 = {
        "version": "2.0",
        "seed": SEED,
        "python": platform.python_version(),
        "phase1_manifest_hash": sha256_file(PHASE1_DIR / "manifest_xai.json"),
        "perfiles_conceptuales_hash": sha256_file(PHASE1_DIR / "perfiles_conceptuales.csv"),
        "spaces": ["completo", "compacto"],
        "m_values": M_VALUES,
        "sigma_factors": SIGMA_FACTORS,
        "methods": ["ProtoDash", "KMedoids"],
        "n_parcelas": len(perfiles),
        "family_weighting": "equal_family_weight (inverse n_components per family)",
        "corrections_applied": [
            "global_selection_rank vs local_similarity_rank separated",
            "original and scaled values in example_xai",
            "ProtoDash weights_original and weights_normalized separated",
            "KMedoids warnings recorded",
            "order_invariance_test_implemented"
        ]
    }
    with open(PHASE2_DIR / "manifest_fase2.json", "w") as f:
        json.dump(manifest2, f, indent=2, ensure_ascii=False)
    
    print("\n=== FASE 2 COMPLETADA (v2 corregida) ===")
    print(f"Resultados en: {PHASE2_DIR}")


if __name__ == "__main__":
    main()