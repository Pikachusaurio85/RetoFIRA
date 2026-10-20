# Verificación contra README 01 — Claude: Predictor final y explicabilidad del modelo

La primera entrega se hizo con el README equivocado (el de OpenCode). Este documento revisa cada punto del README correcto y lista lo que se corrigió. **La decisión y las métricas no cambiaron**: `splits.csv`, `comparacion_final.csv`, `contrastes_pareados.csv` y `comparacion_por_municipio.csv` son byte a byte iguales a los de la primera entrega.

Estado: ✅ cumple · 🔧 corregido en esta revisión · ⚠️ cumple con salvedad

## B. Contrato técnico compartido

| Punto del README | Estado | Evidencia |
|---|---|---|
| Clave `ID_POLIGONO`; cruces por ID, no por posición | ✅ | `comun.cargar_datos` hace un `merge(..., validate="one_to_one")` por `ID_POLIGONO` |
| Objetivo `RENDIMIENTO_T_HA` (t/ha); 138 ENTRENAMIENTO / 59 PREDICCION | ✅ | Comprobado con asserts y en `auditoria_datos.json` (C) |
| Preservar originales y registrar SHA-256 | ✅ | `manifest_modelo.json` (`fuentes`) e `integridad_fuentes_sha256.csv` (90 archivos sin cambios, prueba de `verificar.py`) |
| No usar rendimiento, identificadores ni partición como predictores | ✅ | El predictor final solo usa `loc_municipio`. Ridge 148 excluye ID, PARTICION, rendimiento y AREA_HA (`auditoria_datos.json` D). El ID solo sirve para buscar el municipio en el artefacto; no es una variable del modelo |
| CV aleatoria repetida de la auditoría previa, si es recuperable | ⚠️ | No estaba en el ZIP. Se regeneró con las semillas documentadas (1000–1009) y reproduce las métricas previas a 4 decimales. Se declara comparación exploratoria |
| `splits.csv` con `scheme,repeat,fold,ID_POLIGONO,role`; role `train`/`test`; sin solapamiento | ✅ | `verificar.py`: `particiones_validas` |
| Selección, regularización, imputación, escalado y ajuste supervisado solo con entrenamiento | ✅ | k del prior, alpha de RidgeCV y el pipeline completo se ajustan dentro de cada fold |
| `predicciones_oof.csv` con `schema_version,model_id,scheme,repeat,fold,ID_POLIGONO,y_true_t_ha,y_pred_t_ha` | 🔧 | Antes las columnas eran `modelo,y_obs,y_pred`. Ahora son las del contrato, más `municipio` y `error_t_ha` (prueba `oof_columnas_contrato`) |
| JSON de predicción con el esquema lógico 1.0 | ✅ | Mismas claves y mismo orden (prueba `contrato_json_1_0`) |
| `null` = sin calcular; no inventar intervalos ni atribuciones | 🔧 | Los componentes son el cálculo real (media global + ajuste municipal = predicción). El intervalo es empírico y está validado en CV (cobertura 0.87 con nominal 0.80). **Corregido:** para un municipio no visto ya no se da intervalo (`pending_validation`, límites `null`), porque no hay forma de validarlo |
| `prediction_components` con componentes reales (referencia municipal…) | ✅ | `media_global_entrenamiento` + `ajuste_municipal` |

## C. Responsabilidades

| Punto del README | Estado | Evidencia |
|---|---|---|
| **Fase 1:** auditar identidad de parcelas, etiquetas, columnas y hashes | 🔧 | Nuevo `auditoria_datos.py` → `auditoria_datos.json` y `auditoria_datos_tablas.csv`: 6/6 OK. Antes solo había asserts y hashes, sin un reporte de auditoría |
| ¿`tabla_modelado_parcelaria.csv` tiene las 148 columnas de los experimentos previos? | 🔧 | Sí: el mismo conjunto que Ridge, OLS, RF y ET 148; sin faltantes ni columnas constantes. Ridge 148 = 0.4980 se reproduce exacto |
| Diferencias entre los datos centralizados y los que consumían los scripts originales | 🔧 | `data_processed/tabla_modelado_parcelaria.csv` es byte a byte igual a `desarrollo/salidas/tabla_modelado_parcelaria.csv`. Las etiquetas son iguales a `datos/entradas/ID_area_rendimiento_70_30` y a su copia en `salidas/originales`. Notas: no están `calidad_observaciones.csv` ni `reports/`; las 148 variables se documentan en `salidas/diccionario_features_comparacion.csv` |
| No modificar tablas sin registrar la transformación | ✅ | Ninguna transformación; todo es solo lectura |
| **Fase 2:** comparar de forma pareada prior calibrado, Ridge 148 y (opcional) prior + topografía | ✅ ⚠️ | Hecho. La implementación previa del híbrido no estaba en el ZIP; se reimplementó de forma reproducible (`comun.ModeloHibridoTopo`) |
| RMSE, MAE, R², diferencias pareadas, IC y desempeño por municipio | ✅ | `comparacion_final.csv`, `contrastes_pareados.csv`, `comparacion_por_municipio.csv` y `errores_pareados_parcela.csv` |
| No declarar superioridad con diferencias pequeñas; ante empate, el más simple | ✅ | ADOPTAR PRIOR con la regla fijada antes de correr (`regla_decision_preregistrada.json`) |
| **Fase 3:** si gana el prior, la referencia municipal es la explicación principal y el SHAP de Ridge solo auxiliar y separado | 🔧 | `explicacion_referencia_municipal.csv` (nuevo) y `prediction_components`. El SHAP de Ridge va aparte en `comparativo_ridge148/`, etiquetado `RIDGE148_COMPARATIVO_NO_ADOPTADO` |
| Agrupar por familias: ubicación, clima, satélite, topografía, geometría | 🔧 | Columna `familia_contrato` en `explicaciones_shap.csv` y nuevo `importancia_familias_contrato.csv`. Ridge 148 no tiene variables de ubicación ni de geometría |
| Estrategia explícita para variables correlacionadas | ✅ | SHAP intervencional exacto (supuesto de independencia, declarado) con agregación y permutación por bloque. La versión condicional no aplica porque n < p |
| Las contribuciones reproducen la salida | ✅ | SHAP: error ≤ 1.8·10⁻¹⁵. Prior: la suma de componentes da la predicción (prueba `contrato_json_1_0`) |
| Sin significado causal | ✅ | Advertencias en el informe, el manifiesto y los JSON |
| **Fase 4:** entrenar con las 138 al cerrar; artefacto único; función que acepte una parcela y devuelva el contrato | 🔧 | `modelo_final.json` es el único artefacto. Se añadió `modelo_final.predecir_parcela(datos_parcela)`; antes solo existían `predecir_uno` y `predecir` |

## D. Entregables (`outputs/predictor/`)

| Entregable | Estado | Archivo |
|---|---|---|
| `splits.csv` | ✅ | `splits.csv` |
| `predicciones_oof.csv` | 🔧 | Columnas del contrato |
| `comparacion_final.csv` | ✅ | `comparacion_final.csv` |
| `modelo_final.py` | ✅ | Incluye `predecir_parcela` |
| `modelo_final.joblib` o equivalente | ✅ | `modelo_final.json` (JSON en lugar de pickle; el prior solo necesita una tabla) |
| `manifest_modelo.json` | ✅ | Incluye modelo, características, fuentes, hashes, particiones, versiones, advertencias, contrato y auditoría |
| `ejemplo_prediccion.json` | 🔧 | Nuevo; se mantienen además `ejemplo_entrada.json` y `ejemplo_salida.json` |
| `explicaciones_shap.csv`, si aplica | ✅ | No aplica al modelo final; está en `comparativo_ridge148/` como análisis auxiliar |
| `INFORME_DECISION.md` | ✅ | Justifica el modelo elegido, los descartados y las limitaciones |
| `requirements.txt` | ✅ | `requirements.txt` |

## E. Límites de responsabilidad

| Límite | Estado | Evidencia |
|---|---|---|
| No modificar `xai/` ni implementar prototipos | ✅ | Solo se escribió en `outputs/predictor/` |
| No sobrescribir resultados históricos | ✅ | Los 90 archivos originales de `cebada_datos_baseline_1` son idénticos (hash) |
| No cambiar nombres ni definiciones del contrato común | ✅ 🔧 | JSON y `splits.csv` intactos. `predicciones_oof.csv` se alineó al contrato |
| No construir UI; no usar las 59 como observadas; no presentar la evaluación espacial como garantía temporal | ✅ | Prueba `sin_rendimiento_para_parcelas_PREDICCION`; KM6 y LMO solo como diagnóstico |

## Nota para el equipo

`cebada_datos_baseline_1/README_CLAUDE.md` dentro del ZIP sigue teniendo el contenido de OpenCode («02 — OpenCode»). No lo modifiqué porque está fuera de `outputs/predictor/`; conviene reemplazarlo por el README 01 correcto.
