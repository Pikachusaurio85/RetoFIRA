# AgroCebada 2026 — Random Forest Regressor

## Objetivo

Construir y evaluar modelos Random Forest para estimar `RENDIMIENTO_T_HA` a partir de características de parcelas. La unidad de predicción es **la parcela**, no cada observación satelital. Las 138 parcelas del conjunto ENTRENAMIENTO sí tienen rendimiento observado; las 59 del conjunto PREDICCION no tienen etiqueta y **no se usaron para evaluación**.

## Ejecución en VS Code

1. Coloca `DATASET_RETO_AGRO_2026.zip` y `random_forest_agro.py` en la misma carpeta (no hace falta descomprimir el ZIP original).
2. Instala las dependencias:

   ```bash
   python -m pip install -r requirements_random_forest_agro.txt
   ```

3. Ejecuta:

   ```bash
   python random_forest_agro.py
   ```

4. Los resultados aparecen en `resultados_random_forest_agro/`. También puedes cambiar las rutas:

   ```bash
   python random_forest_agro.py --zip ruta/al/DATASET_RETO_AGRO_2026.zip --salida ruta/de/salida
   ```

## Fuente de los datos

Dos CSV leídos directamente del ZIP original:

- `desarrollo/cebada_datos_baseline_1/data_processed/features_parcela.csv`
- `desarrollo/salidas/tabla_modelado_parcelaria.csv`

Se verifica que coincidan los identificadores y rendimientos. No se usa el identificador de parcela ni `RENDIMIENTO_T_HA` como variables predictoras. Las características históricas excluyen también `PARTICION` y `AREA_HA` (identificador de partición o variable fuera del set elegido en los experimentos previos).

## Configuración y validación

Tres variantes de variables, fijadas antes de evaluar:

- `RF_8_agronomicas`: 8 variables agronómicas.
- `RF_24_curadas`: 24 índices, topografía y geometría.
- `RF_148_historicas`: 148 variables de históricos satelitales, clima y topografía.

Configuración compartida de RandomForestRegressor: `n_estimators=350`, `min_samples_leaf=3`, `max_features=0.7`, `bootstrap=True`, `max_depth=None`, `random_state=2026`. No se realizó búsqueda de hiperparámetros. `SimpleImputer(strategy='median')` se entrena dentro de cada partición mediante Pipeline, aunque las variables analizadas no presentan valores faltantes. Random Forest no necesita normalizar columnas.

Para comparar directamente con Ridge/OLS se utilizan **las mismas** particiones:

- Aleatoria: `KFold(n_splits=5, shuffle=True, random_state=2026)`.
- Espacial: `KMeans(n_clusters=6, n_init=10, random_state=2026)` sobre coordenadas UTM de parcelas etiquetadas, seguido por `GroupKFold(n_splits=6)`.

Los errores agregados se calculan sobre las 138 predicciones **fuera de muestra**, cada parcela predicha una vez en cada esquema de validación. Los modelos `.joblib` se ajustan *posteriormente* con las 138 etiquetas y son artefactos exploratorios, no estimadores de error.

## Resultados

| Modelo | RMSE aleatorio (t/ha) | MAE espacial (t/ha) | RMSE espacial (t/ha) | R² espacial |
|---|---:|---:|---:|---:|
| RF 8 | 0.7301 | 0.9503 | 1.0751 | -0.6072 |
| RF 24 | 0.6859 | 0.6989 | 0.8862 | -0.0920 |
| RF 148 | 0.5384 | 0.7243 | 0.8862 | -0.0918 |
| Ridge 148 (resultado del experimento anterior) | 0.4980 | 0.5018 | 0.6839 | 0.3497 |
| Media municipal suavizada (resultado anterior) | 0.5328 | 0.5298 | 0.6847 | 0.3482 |
| OLS 24 (resultado anterior) | 0.7177 | 0.7413 | 0.8908 | -0.1032 |

Los modelos RF 24 y 148 prácticamente **empatan** en RMSE espacial (0.886238 vs 0.886151); la ventaja de RF 148 en esa prueba es despreciable. Ridge 148 es superior en el protocolo espacial actual.

**Sobreajuste**: en RF-148 el RMSE de entrenamiento medio por fold en validación espacial es aprox. **0.277 t/ha**, mientras que el RMSE agregado de validación espacial alcanza **0.886 t/ha**.

**Importancia de variables (no causal)**: según importancia basada en impureza del RF-148 final, `HISTORICO_PROMEDIO__CLIMA__precipitacion_mm_05` contribuye alrededor de 41.9%, pero este indicador puede sesgarse por correlación entre variables o por la distribución regional. No significa que esa variable cause el 41.9% del rendimiento.

**Limitaciones fundamentales**:

- Los rendimientos observados no tienen documentado claramente su ciclo/año agrícola, por lo que no es posible declarar validada la cosecha 2026.
- Calpulalpan tiene 30 parcelas etiquetadas con exactamente 3.0 t/ha; confirmar con la fuente si es una observación real para cada parcela o un valor asignado por grupo.
- Algunos municipios apenas tienen una parcela etiquetada.
- La división aleatoria puede ser optimista cuando hay cercanía o semejanza regional.
- Este experimento compara configuraciones prefijadas; otra parametrización podría cambiar el resultado. No seleccionar modelos exclusivamente por la validación aleatoria.
- La importancia por impureza en árboles no demuestra causalidad ni es necesariamente estable al cambiar zona o temporada.

## Entregables

- `random_forest_agro.py`: script que reconstruye el experimento leyendo tu ZIP.
- `resultados_random_forest_agro/metricas_random_forest.csv`: comparación numérica de tres RF.
- `resultados_random_forest_agro/metricas_por_fold_random_forest.csv`: resultados de cada partición, con error de entrenamiento y prueba.
- `resultados_random_forest_agro/predicciones_validacion_random_forest.csv`: predicción externa de cada parcela para auditar errores.
- `resultados_random_forest_agro/errores_por_municipio.csv`: comportamiento espacial del mejor RF.
- `resultados_random_forest_agro/importancias_RF_*.csv`: importancia de variables, basada en disminución de impureza.
- `resultados_random_forest_agro/importancias_por_fold_random_forest.csv`: variación de importancias por fold.
- `resultados_random_forest_agro/RF_*.joblib`: modelos finales completos con columnas y metadatos.
- `resultados_random_forest_agro/random_forest_validacion_espacial.png`: observado vs. estimado.
- `resultados_random_forest_agro/importancia_variables_rf148.png`: 12 importancias principales.
- `resultados_random_forest_agro/comparacion_rmse_espacial.png`: comparación general.
- `resultados_random_forest_agro/comparacion_modelos.csv`: comparación con métricas anteriores cuando están disponibles.

## Próximos pasos sugeridos

Antes de hacer predicciones operativas, verificar ciclo y fuente de etiquetas, probar validación *leave-one-municipality-out* con suficiente representatividad y comparar Random Forest optimizado mediante **validación anidada** con Ridge y modelos de referencia. No confundir mejora en entrenamiento con mejora fuera de muestra.
