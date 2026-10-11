#Orquestación del análisis descriptivo y predictivo

from __future__ import annotations
from analisis.config import EVOLUTION_PATH, OUTPUT, TARGET
from analisis.data import (
    _copy_and_hash_sources,
    _load_assignments,
    _make_annual_parcel_features,
    _number,
    _read_spatial_target_data,
    _write_iqr_outlier_review,
    _write_parcel_features,
    _write_traceability,
)
from analisis.modeling import _train_evaluate_predict
from analisis.organization import (
    _variable_dictionary,
    _write_audit,
    _write_reorganized_csv,
    _write_workbook,
)
from analisis.report import _write_report
from analisis.visualization import (
    _write_chart_interpretations,
    _write_descriptive_outputs,
    _write_test_residual_plot,
)

def main() -> None:
    #Ejecuta auditoría, exportaciones, visualizaciones, modelos e informe
    OUTPUT.mkdir(parents=True, exist_ok=True)
    preserved = _copy_and_hash_sources()
    (
        feature_names,
        parcel_data,
        annual_features,
        centroids,
        topography,
        audit,
    ) = _make_annual_parcel_features()
    assignments = _load_assignments()
    for polygon_id, values in parcel_data.items():
        values.update(topography.get(polygon_id, {}))
        if polygon_id in assignments:
            values["AREA_HA"] = assignments[polygon_id].get("AREA_HA", "")
            values["PARTICION"] = assignments[polygon_id].get(
                "CONJUNTO", values.get("PARTICION", "")
            ).upper()
            target_value = _number(assignments[polygon_id].get(TARGET))
            if target_value is not None:
                values[TARGET] = target_value

    descriptive_columns = [name for name in audit["headers"] if name]
    descriptive_columns.extend(
        [
            "TOPOGRAFIA__elevacion_media_m_snm",
            "TOPOGRAFIA__pendiente_media_grados",
        ]
    )
    model_features = [
        column for column in feature_names if column != "AREA_HA"
    ]
    dictionary_columns = [
        *descriptive_columns,
        *model_features,
        "RENDIMIENTO_T_2025_T_HA",
    ]
    dictionary = _variable_dictionary(dict.fromkeys(dictionary_columns))
    workbook_path = OUTPUT / "datos_agrocebada_organizados.xlsx"
    _write_workbook(
        descriptive_columns,
        dictionary,
        topography,
        workbook_path,
    )
    _write_reorganized_csv(
        descriptive_columns,
        topography,
        OUTPUT / "datos_reorganizados_todas_observaciones.csv",
    )
    _write_audit(audit, OUTPUT / "auditoria_columnas.csv")
    _write_iqr_outlier_review(parcel_data, feature_names, topography)
    _write_parcel_features(
        OUTPUT / "datos_modelado_parcela_2026.csv",
        model_features,
        parcel_data,
        topography,
        assignments,
    )
    target_values = _read_spatial_target_data()
    _write_descriptive_outputs(annual_features, topography, target_values)
    model_results = _train_evaluate_predict(
        feature_names, parcel_data, topography, centroids
    )
    _write_test_residual_plot(
        model_results["test_actual"], model_results["test_prediction"]
    )
    _write_chart_interpretations(
        annual_features, topography, target_values, model_results
    )
    _write_report(audit, model_results, target_values)
    _write_traceability(preserved)
    print(f"Base procesada: {EVOLUTION_PATH}")
    print(f"Filas analizadas: {audit['row_count']:,}; duplicados exactos: {audit['duplicate_rows']}")
    print(f"Libro reorganizado: {workbook_path}")
    print(f"Modelo seleccionado: {model_results['selected_model']}")
    print(f"Transferencia robusta: {model_results['robust']}")
    print(f"Informe: {OUTPUT / 'informe_ejecutivo_analisis_2026.md'}")

if __name__ == "__main__":
    main()