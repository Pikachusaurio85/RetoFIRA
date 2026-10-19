# AgroCebada 2026 — Extra Trees Regressor

## ¿Qué se hizo?

Se entrenaron tres modelos de **Extra Trees Regressor** para estimar `RENDIMIENTO_T_HA` de cebada. La unidad de análisis es la parcela y no cada registro satelital. Se emplearon **138 parcelas etiquetadas**; otras **59 sin etiqueta** se reservaron para generar estimaciones *exploratorias*, no para evaluar los modelos.

Se hicieron **dos evaluaciones** con exactamente las mismas particiones externas que Ridge, regresión lineal y Random Forest anteriores:

- `KFold(n_splits=5, shuffle=True, random_state=2026)`: validación aleatoria (más optimista en datos espacialmente correlacionados).
- `KMeans(n_clusters=6, n_init=10, random_state=2026)` con coordenadas UTM + `GroupKFold(n_splits=6)`: validación espacial por bloques completos.

Los resultados son *out of fold*: cada una de las 138 parcelas aparece en la prueba externa exactamente una vez por escenario/modelo. Se calcularon MAE, RMSE y R² globales sobre las predicciones retenidas. La imputación con medianas se ajusta solo con el entrenamiento de cada partición.

## Resultados reales

| Modelo | Variables | RMSE aleatorio (t/ha) | MAE espacial (t/ha) | RMSE espacial (t/ha) | R² espacial |
|---|---:|---:|---:|---:|---:|
| Extra Trees 8 | 8 | 0.7223 | 0.9030 | 1.0200 | -0.4466 |
| Extra Trees 24 | 24 | 0.6752 | 0.7213 | 0.8882 | -0.0968 |
| **Extra Trees 148** | **148** | **0.5256** | **0.6628** | **0.8441** | **0.0093** |

En el esquema **espacial**, Extra Trees 148 es mejor que RF 148 (RMSE 0.8862), pero no mejora Ridge 148 (RMSE 0.6839). La referencia municipal suavizada también logró un RMSE de 0.6847. Las cifras corresponden únicamente a las 138 parcelas etiquetadas existentes, no a una cosecha verificada de 2026.

### ¿Por qué no está listo para desplegarse?

- Con Extra Trees 148, el RMSE de entrenamiento en los seis bloques está entre ~0.17 y ~0.20 t/ha, mientras que fuera de muestra la métrica espacial es 0.8441 t/ha. Esto indica sobreajuste o falta de transferibilidad a zonas distintas.
- En Singuilucan el RMSE espacial fue aproximadamente 1.454 t/ha (16 parcelas) y en Almoloya 1.150 t/ha (10 parcelas).
- En **Calpulalpan las 30 parcelas etiquetadas muestran rendimiento exactamente 3.0 t/ha**. Hace falta verificar si se midió cada parcela o se asignó un valor común al municipio.
- Algunos municipios contienen 1–5 etiquetas, por lo que las métricas regionales allí son inestables.
- No está documentada de forma inequívoca la campaña agrícola de las etiquetas; no se puede declarar validado el rendimiento de 2026.
- La importancia por impureza en árboles **no implica causalidad**. Muchas características climáticas están correlacionadas, pueden reflejar el municipio y pueden desplazar otras importancias.

## Configuración

Tres grupos de variables fijados antes del entrenamiento:

1. `ET_8_agronomicas`: índices vegetativos y topografía/área.
2. `ET_24_curadas`: índices satelitales y topografía/geométricas.
3. `ET_148_historicas`: históricos satelitales, clima y topografía.

`ExtraTreesRegressor(n_estimators=350, min_samples_leaf=3, max_features=0.7, max_depth=None, bootstrap=False, random_state=2026, n_jobs=-1)`.

Se usó `bootstrap=False`, que es el comportamiento habitual de Extra Trees; Random Forest previo usó `bootstrap=True`. Por tanto la diferencia es entre **configuraciones típicas completas** de cada algoritmo, no un estudio aislado del efecto de los umbrales de división.

No se optimizaron hiperparámetros sobre la prueba espacial. Los modelos `.joblib` se volvieron a entrenar al final con las 138 parcelas etiquetadas: **no** se usaron para calcular las métricas cruzadas.

## Cómo reproducir en VS Code

1. Coloca `DATASET_RETO_AGRO_2026.zip` en la carpeta `experiments/` (sin descomprimir el ZIP original).
2. Instala las dependencias:

   ```bash
   python -m pip install -r modelos/ExtraTrees_model/requirements_extra_trees_agro.txt
   ```

3. Ejecuta:

   ```bash
   python modelos/ExtraTrees_model/extra_trees_agro.py
   ```

4. Revisa `modelos/ExtraTrees_model/resultados_extra_trees_agro/`.

Para especificar otras rutas:

```bash
python modelos/ExtraTrees_model/extra_trees_agro.py --zip ruta/DATASET_RETO_AGRO_2026.zip --salida ruta/resultados
```

Se leen estos dos CSV directamente dentro del ZIP:

- `desarrollo/cebada_datos_baseline_1/data_processed/features_parcela.csv`
- `desarrollo/salidas/tabla_modelado_parcelaria.csv`

No se utiliza `ID_POLIGONO`, rendimiento ni el identificador de partición como entrada del modelo. El script incorpora automáticamente los CSV de métricas previos de los otros experimentos integrados. El archivo `comparacion_modelos.csv` que acompaña este paquete **sí incluye** las comparaciones de la sesión original.

## Archivos entregados

- `extra_trees_agro.py`: código completo reproducible.
- `requirements_extra_trees_agro.txt`: dependencias.
- `resultados_extra_trees_agro/metricas_extra_trees.csv`: métricas globales por modelo y escenario.
- `resultados_extra_trees_agro/metricas_por_fold_extra_trees.csv`: errores de entrenamiento/prueba por partición.
- `resultados_extra_trees_agro/predicciones_validacion_extra_trees.csv`: 138 predicciones out-of-fold por configuración y escenario.
- `resultados_extra_trees_agro/errores_por_municipio.csv`: errores y promedios de ET 148 en validación espacial.
- `resultados_extra_trees_agro/importancias_ET_*.csv`: importancia por impureza sobre el ajuste final (descriptiva; no causal).
- `resultados_extra_trees_agro/importancias_por_fold_extra_trees.csv`: importancias de cada fold.
- `resultados_extra_trees_agro/ET_*.joblib`: 3 pipelines finales con columnas y metadatos.
- `resultados_extra_trees_agro/extra_trees_validacion_espacial.png`: dispersión observados/predichos.
- `resultados_extra_trees_agro/comparacion_rmse_espacial.png`: comparación espacial con otros algoritmos.
- `resultados_extra_trees_agro/importancia_variables_et148.png`: doce características con mayor importancia.
- `resultados_extra_trees_agro/predicciones_reserva_extra_trees_NO_VALIDADAS.csv`: **estimaciones sobre las 59 parcelas sin rendimiento, NO comprobadas** (3 modelos, 177 filas).

### Importante acerca de las 59 predicciones

El archivo `predicciones_reserva_extra_trees_NO_VALIDADAS.csv` contiene **predicciones exploratorias**, que pueden ser inexactas. No mezclar esas predicciones con etiquetas medidas, ni mostrarlas como cosecha verificada de 2026. No sirven para calcular precisión o R² al no tener resultados reales.

## Próxima prueba recomendada

Hacer selección de variables y ajuste de `min_samples_leaf`, `max_features` y `max_depth` usando **validación anidada por bloques geográficos**, y comparar contra Ridge y la referencia municipal en el mismo conjunto externo. Revisar la fuente y año de los rendimientos antes de construir un dashboard operativo.
