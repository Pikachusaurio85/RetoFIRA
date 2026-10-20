# 01 — Claude: Predictor final y explicabilidad del modelo

**Proyecto:** AgroCebada 2026  
**Rol:** Modelado predictivo, validación, explicaciones del predictor y entrega para integración.  
**Prioridad:** Cerrar hoy una implementación reproducible.

## A. Contexto científico compartido

AgroCebada busca estimar el rendimiento agrícola de parcelas de cebada utilizando datos satelitales, climáticos, topográficos y geográficos.

Se dispone de 197 parcelas: 138 con rendimiento observado y 59 sin etiqueta.

La fuente oficial establecida por el equipo es `DATASET_RETO_AGRO_2026.zip`. Para el trabajo integrado se han centralizado las tablas procesadas bajo `cebada_datos_baseline_1/data_processed/`.

Tablas principales:

- `features_parcela.csv`: características del baseline, municipio, identificadores y etiquetas.
- `tabla_modelado_parcelaria.csv`: conjunto histórico utilizado por Ridge 148 y otros modelos.
- `series_dekadales_2025.csv`: series satelitales.
- `clima_mensual_parcela.csv`: clima histórico.
- `parcelas_base.csv`: información geométrica y topográfica.
- `diccionario_features.csv`: significado de variables.

### Hallazgos anteriores

La variación entre municipios explica aproximadamente 69 % de la variación observada en rendimiento.

Existe una posible confusión entre municipio, características climáticas y régimen de captura de rendimiento.

El prior municipal con encogimiento k=3 estaba sobre-regularizado. La auditoría anterior obtuvo RMSE cercano a 0.512 t/ha utilizando una calibración interna.

Ridge 148 obtuvo RMSE 0.498 en una validación aleatoria de cinco folds, pero todavía no demostró superioridad robusta frente al prior recalibrado.

No existen etiquetas verificadas de múltiples campañas que permitan demostrar generalización temporal.

Las variables satelitales, SHAP o los coeficientes no deben interpretarse como evidencia causal.

## B. Contrato técnico compartido

### Datos

- Clave única: `ID_POLIGONO`.
- Unidad: una parcela.
- Objetivo: `RENDIMIENTO_T_HA`, medido en t/ha.
- `ENTRENAMIENTO`: 138 parcelas etiquetadas.
- `PREDICCION`: 59 parcelas sin rendimiento observado.
- Todos los cruces entre tablas deben realizarse por `ID_POLIGONO`, no por posición.
- Preservar los datos originales y registrar SHA-256 de las tablas utilizadas.
- Nunca usar el rendimiento, identificadores ni columnas de partición como características predictoras.

### Validación

La referencia principal será la CV aleatoria repetida documentada en la auditoría previa, si sus particiones son recuperables.

Claude es responsable de producir:

`outputs/predictor/splits.csv`

Formato obligatorio:

`scheme,repeat,fold,ID_POLIGONO,role`

Donde `role` es `train` o `test`.

Para cada combinación de `scheme`, `repeat` y `fold`, una parcela no puede aparecer simultáneamente en entrenamiento y prueba.

La selección de variables, regularización, imputación, escalado y cualquier ajuste supervisado deben permanecer dentro de entrenamiento.

### Predicciones fuera de muestra

Exportar:

`outputs/predictor/predicciones_oof.csv`

Columnas mínimas:

`schema_version,model_id,scheme,repeat,fold,ID_POLIGONO,y_true_t_ha,y_pred_t_ha`

El identificador de cada fila será la combinación de modelo, esquema, repetición, fold y parcela.

### Predicción final para integración

La salida del predictor deberá seguir este esquema lógico:

```json
{
  "schema_version": "1.0",
  "ID_POLIGONO": "AGC_001",
  "model_id": "modelo_final",
  "prediction_t_ha": null,
  "uncertainty": {
    "method": "pending_validation",
    "lower_t_ha": null,
    "upper_t_ha": null
  },
  "prediction_components": [],
  "warnings": []
}
```

Los valores `null` son campos sin calcular, no resultados del modelo.

La predicción no debe inventar intervalos de confianza o atribuciones.

Cuando proceda, `prediction_components` describirá componentes reales del cálculo, por ejemplo referencia municipal y corrección Ridge.

## C. Responsabilidades exclusivas de Claude

### Fase 1 — Auditoría de datos

Verificar identidad de parcelas, etiquetas, columnas y hashes.

Comprobar específicamente si `tabla_modelado_parcelaria.csv` coincide con las 148 características esperadas por los experimentos anteriores.

Detectar diferencias entre datos centralizados y datos que consumían los scripts originales.

No modificar las tablas para solucionar incompatibilidades sin registrar la transformación.

### Fase 2 — Decisión predictiva

Comparar de forma pareada:

1. Prior municipal calibrado dentro del entrenamiento.
2. Ridge 148 con preprocesamiento y alfa ajustados dentro de cada fold.
3. Opcional: prior + corrección topográfica, únicamente si existe una implementación reproducible.

Reutilizar resultados y particiones de auditorías anteriores cuando correspondan, evitando repetir búsquedas extensas.

Reportar RMSE, MAE, R², diferencias pareadas, intervalos de incertidumbre y desempeño por municipio.

No declarar superioridad estadística por diferencias pequeñas o intervalos inconcluyentes.

En caso de empate o incertidumbre, priorizar el modelo más simple y robusto.

### Fase 3 — SHAP y explicación predictiva

Si Ridge participa en el predictor final, evaluar SHAP lineal con una estrategia explícita para tratar variables correlacionadas.

Agrupar resultados descriptivamente por familias: ubicación, clima, satélite, topografía y geometría.

Comprobar que las contribuciones reproduzcan la salida del modelo explicado.

Si el modelo ganador es exclusivamente municipal, mostrar la referencia municipal como explicación principal. SHAP de Ridge sólo podrá figurar como análisis auxiliar, claramente separado.

No atribuir significado causal a SHAP, coeficientes o importancia por permutación.

### Fase 4 — Cierre del predictor

Entrenar el modelo seleccionado utilizando las 138 etiquetas disponibles después de cerrar la evaluación.

Generar un único artefacto oficial.

Implementar una función que acepte los datos de una parcela y devuelva el contrato JSON.

## D. Entregables exclusivos

Todos los archivos propios de esta tarea deberán escribirse bajo `outputs/predictor/`.

- `splits.csv`
- `predicciones_oof.csv`
- `comparacion_final.csv`
- `modelo_final.py`
- `modelo_final.joblib` o artefacto equivalente
- `manifest_modelo.json`
- `ejemplo_prediccion.json`
- `explicaciones_shap.csv`, si aplica
- `INFORME_DECISION.md`
- `requirements.txt`

`manifest_modelo.json` documentará modelo, características, fuentes, hashes, particiones, versiones y advertencias.

`INFORME_DECISION.md` justificará el modelo elegido, los descartados y las limitaciones.

## E. Límites de responsabilidad

Claude NO deberá:

- Modificar el código del módulo `xai/`.
- Implementar el sistema de prototipos del equipo OpenCode.
- Sobrescribir resultados históricos.
- Cambiar nombres o definiciones del contrato común sin coordinación.
- Construir una UI.
- Tratar las 59 predicciones ocultas como datos observados.
- Presentar una evaluación espacial como garantía de transferencia temporal.

### Definición de terminado

Existe un predictor único y reproducible, un manifiesto, una comparación documentada, predicciones fuera de muestra y un JSON de inferencia que puede consumir la UI.

La prioridad es terminar y congelar el predictor, no continuar indefinidamente optimizando modelos.