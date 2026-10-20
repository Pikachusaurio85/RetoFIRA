# outputs/predictor — Módulo predictor AgroCebada 2026

Decisión: **ADOPTAR PRIOR** (media municipal con encogimiento calibrado). Ver `INFORME_DECISION.md`.
Contrato: README 01 — Claude; verificación punto por punto en `CUMPLIMIENTO_README.md`.

## Ejecutar

Coloca esta carpeta en `cebada_datos_baseline_1/outputs/predictor/`. Las tablas se leen de `../../data_processed/`.

```powershell
python -m pip install -r requirements.txt
python run_all.py          # auditoría -> fase1 -> fase2 -> diagnóstico -> fase3 -> verificar  (~2 min)
```

Cada script acepta `--base RUTA\cebada_datos_baseline_1` y `--out RUTA`. No escribe fuera de esta carpeta.

## Integración con la UI

```python
from modelo_final import predecir_parcela        # solo biblioteca estándar
predecir_parcela({"ID_POLIGONO": "AGC_011"})     # -> objeto del contrato JSON 1.0 (ver ejemplo_prediccion.json)
predecir_parcela({"loc_municipio": "Chignahuapan"})
```

O desde la terminal: `python modelo_final.py --entrada ejemplo_entrada.json --salida salida.json`. Las salidas listas son `predicciones_ui.json` (197 parcelas) y `predicciones_modelo_final.csv`.

## Archivos

| Archivo | Contenido |
|---|---|
| `modelo_final.py`, `modelo_final.json` | Función predictiva y artefacto único (JSON, sin pickle) |
| `ejemplo_prediccion.json` | Salida del contrato para una parcela PREDICCION |
| `explicacion_referencia_municipal.csv` | Explicación principal del modelo final: media, peso e intervalo por municipio |
| `auditoria_datos.json`, `auditoria_datos_tablas.csv` | Auditoría de datos (Fase 1 del README) |
| `CUMPLIMIENTO_README.md` | Revisión punto por punto contra el README 01 |
| `splits.csv` | Particiones compartidas `scheme,repeat,fold,ID_POLIGONO,role` (138 etiquetadas) |
| `predicciones_oof.csv` | OOF de todos los modelos y esquemas (`schema_version,model_id,scheme,repeat,fold,ID_POLIGONO,y_true_t_ha,y_pred_t_ha`, más `municipio` y `error_t_ha`) |
| `errores_pareados_parcela.csv` | Error cuadrático medio por parcela y diferencias contra PRIOR |
| `comparacion_final.csv` | Métricas, ΔRMSE/ΔMAE con IC, prueba t corregida, etiqueta |
| `contrastes_pareados.csv`, `comparacion_por_municipio.csv`, `metricas_por_repeticion.csv` | Detalle de la comparación |
| `decision_fase1.json`, `regla_decision_preregistrada.json` | Decisión y regla fijada antes de correr |
| `validacion_intervalo_prior.csv` | Cobertura del intervalo 80 % en CV |
| `comparativo_ridge148/` | **Análisis auxiliar, no explica el modelo final**: `explicaciones_shap.csv`, locales representativas, familias del contrato (`importancia_familias_contrato.csv`), SHAP vs permutación, diagnóstico post hoc |
| `manifest_modelo.json` | Variables, fuentes (SHA-256), particiones, configuración, advertencias, hashes de salidas |
| `verificacion.json` | Resultado de las 14 pruebas de `verificar.py` |
| `ejemplo_entrada.json`, `ejemplo_salida.json` | Ejemplo de integración |
