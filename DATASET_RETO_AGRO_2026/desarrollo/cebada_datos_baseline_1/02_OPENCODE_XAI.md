# 02 — OpenCode: Conceptos, prototipos y módulo XAI

**Proyecto:** AgroCebada 2026  
**Rol:** Representaciones interpretables, prototipos, análisis de similitud y preparación de resultados para UI.  
**Prioridad:** Entregar hoy un módulo XAI funcional, independiente del predictor oficial.

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
- Nunca utilizar rendimiento, identificadores ni particiones como atributos de similitud descriptiva.

### Validación

Claude es responsable de generar la referencia compartida:

`outputs/predictor/splits.csv`

Esquema:

`scheme,repeat,fold,ID_POLIGONO,role`

OpenCode reutilizará estas particiones cuando evalúe cuantitativamente un predictor basado en conceptos o prototipos contra rendimiento observado.

Si las particiones aún no existen, puede implementar y demostrar el módulo descriptivo, pero no declarar desempeño predictivo validado.

Los prototipos y transformaciones deben ajustarse únicamente con las parcelas de entrenamiento dentro de cada fold cuando sean parte de una evaluación supervisada.

### Contrato de integración

El predictor oficial entrega, para cada parcela:

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

Este formato es propiedad del módulo predictor.

El módulo XAI no recalcula ni sobrescribe `prediction_t_ha`.

OpenCode entrega un objeto independiente `xai_profile`, asociado por `ID_POLIGONO`, que la UI puede presentar junto con la predicción.

## C. Objetivo de investigación

Estudiar si una representación basada en conceptos observacionales y parcelas prototipo proporciona interpretaciones comprensibles y científicamente defendibles.

No es obligatorio mejorar el RMSE para que el módulo descriptivo sea útil.

Sin embargo, cualquier afirmación de que los prototipos mejoran la predicción requiere validación fuera de muestra frente al prior municipal calibrado.

## D. Conceptos observacionales iniciales

Investigar un conjunto pequeño, preferentemente entre cuatro y seis descriptores.

Familias candidatas:

1. Cobertura espectral de línea base.
2. Dinámica temporal del índice vegetal.
3. Exposición climática registrada.
4. Pendiente y elevación.
5. Variabilidad de observaciones.
6. Calidad o completitud de series satelitales.

Las definiciones deben derivarse de columnas realmente existentes.

Antes de implementar, verificar el diccionario de características y las unidades.

No inventar mediciones de fertilidad, nutrientes, estrés fisiológico o fenología confirmada.

En caso de usar puntuaciones normalizadas, conservar también los valores físicos originales cuando tengan significado.

## E. Prototipos

Construir un conjunto reducido de parcelas reales representativas.

Metodología inicial:

- Seleccionar una representación compacta de conceptos.
- Estandarizar características numéricas.
- Calcular distancias sobre las variables definidas.
- Seleccionar prototipos mediante medoides u otro procedimiento documentado.
- Recuperar los prototipos más próximos para cada parcela.
- Identificar qué características explican descriptivamente la similitud.

No utilizar rendimiento en la selección descriptiva de prototipos.

Evitar que variables redundantes dominen artificialmente la distancia.

Distinguir similitud dentro del municipio frente a similitud entre municipios; ambas responden preguntas diferentes.

No interpretar distancia pequeña como igualdad de rendimiento.

## F. Experimento opcional de prototipos predictivos

Si el módulo descriptivo funciona y queda tiempo, construir una representación basada en similitudes con prototipos.

Usar esas similitudes como entrada de un predictor Ridge pequeño.

Evaluar con las mismas particiones que utiliza Claude.

Seleccionar prototipos y transformar variables dentro del entrenamiento del fold.

Comparar contra el prior municipal calibrado y Ridge 148, sin atribuir a la representación mejoras que no sean consistentes.

Este experimento es opcional y no debe retrasar el módulo XAI funcional.

## G. Salida del módulo XAI

Exportar:

`outputs/xai/perfiles_conceptuales.csv`

Clave: `ID_POLIGONO`.

Exportar:

`outputs/xai/prototipos.csv`

Clave: `prototype_id`.

Exportar:

`outputs/xai/similitudes.csv`

Columnas mínimas:

`ID_POLIGONO,prototype_id,rank,distance,similarity`

La distancia y la similitud deberán documentar su métrica y transformación.

Exportar:

`outputs/xai/ejemplo_xai.json`

Ejemplo de estructura, con valores ilustrativos sin resultados calculados:

```json
{
  "schema_version": "1.0",
  "ID_POLIGONO": "AGC_001",
  "xai_profile": {
    "representation_version": "concepts_v1",
    "concepts": [],
    "nearest_prototypes": [],
    "data_quality": {},
    "interpretation_warnings": []
  }
}
```

Cada concepto deberá contener un identificador, etiqueta descriptiva, valor, unidad si corresponde, fuente y método de cálculo.

Cada prototipo deberá incluir un ID real de parcela, distancia o similitud, rango y descripción de los atributos compartidos.

Los campos explicativos deben describir evidencia observacional, no causalidad.

## H. Entregables exclusivos

Todos los resultados deberán escribirse en `outputs/xai/`.

- `conceptos.py`
- `prototipos.py`
- `xai_service.py`
- `perfiles_conceptuales.csv`
- `prototipos.csv`
- `similitudes.csv`
- `ejemplo_xai.json`
- `manifest_xai.json`
- `INFORME_XAI.md`
- `requirements.txt`

El manifiesto documentará fuentes, hashes, conceptos, transformaciones, escalado, métrica de distancia y versiones.

El informe explicará qué demuestra el módulo, qué no demuestra y qué pruebas adicionales serían necesarias para darle significado agronómico.

## I. Límites de responsabilidad

OpenCode NO deberá:

- Modificar el modelo predictivo oficial.
- Sobrescribir resultados de Claude.
- Modificar las tablas de origen.
- Presentar similitud como relación causal.
- Tratar el rendimiento observado como concepto de entrada.
- Producir explicaciones SHAP de un predictor distinto y atribuirlas al modelo final.
- Implementar una red neuronal compleja o CBM completo sin justificación temporal.
- Modificar el esquema de integración compartido.

### Definición de terminado

Existe una representación conceptual reproducible, un conjunto de prototipos reales, una forma de recuperar parcelas similares y un JSON de salida independiente del modelo predictor.

El módulo funciona incluso si el predictor final resulta ser únicamente el prior municipal.