# Primer experimento de Ridge Regression — AgroCebada

## Objetivo
Estimar `RENDIMIENTO_T_HA` (toneladas de cebada por hectárea) mediante regresión Ridge, comparando dos representaciones de las parcelas:

1. **Ridge_24_curadas**: 24 variables seleccionadas de fenología 2025, índices satelitales, topografía y geometría, tomadas de la selección que ya existe en el proyecto.
2. **Ridge_148_historicas**: 148 atributos de 2022-2025 consolidados en `tabla_modelado_parcelaria.csv`, sin `ID_POLIGONO`, `PARTICION`, `RENDIMIENTO_T_HA` ni `AREA_HA`.

Se utilizaron las **138 parcelas etiquetadas** para entrenamiento/validación. Las **59 parcelas sin rendimiento** no se emplearon para medir precisión ni para optimizar hiperparámetros.

## Ejecutar en Windows, Linux o macOS

1. Instala Python 3.10 o posterior.
2. Crea una carpeta y coloca juntos `ridge_regresion_agro.py` y tu archivo original `DATASET_RETO_AGRO_2026.zip`. No necesitas extraer el ZIP de datos.
3. Abre la terminal de esa carpeta y ejecuta:

```bash
python -m pip install numpy pandas scikit-learn matplotlib joblib
python ridge_regresion_agro.py
```

Se creará `resultados_ridge_agrocebada/` con las métricas, las predicciones fuera de muestra, los coeficientes, los valores óptimos de alfa y los modelos `.joblib`.

**Si tu ZIP tiene otro nombre o ubicación:**

```bash
python ridge_regresion_agro.py --zip ruta/al/archivo.zip
```

**Predicciones EXPLORATORIAS en las 59 parcelas sin etiqueta** (no son pronósticos validados para 2026):

```bash
python ridge_regresion_agro.py --predecir-reserva
```

## ¿Cómo funciona Ridge?

La predicción es lineal:

`rendimiento_estimado = beta0 + beta1*x1 + ... + betap*xp`

Ridge minimiza el error cuadrático con una penalización `alpha * suma(beta_j^2)`. Cuanto mayor es `alpha`, más se reduce la magnitud de los coeficientes y menos tiende a sobreajustarse en datos con predictores altamente correlacionados.

El programa utiliza `SimpleImputer -> StandardScaler -> RidgeCV` **dentro de cada partición**, de modo que la escala y los valores faltantes no aprenden de los registros de validación. RidgeCV selecciona `alpha` entre 13 valores de 0.01 a 10000 usando validación cruzada sobre los datos de entrenamiento del fold.

## Resultados de esta ejecución

| Evaluación | Modelo | MAE (t/ha) | RMSE (t/ha) | R² |
|---|---|---:|---:|---:|
| Aleatoria 5 folds | Media global | 0.757 | 0.859 | -0.027 |
| Aleatoria 5 folds | Media municipal suavizada | 0.371 | 0.533 | 0.605 |
| Aleatoria 5 folds | Ridge 24 | 0.534 | 0.687 | 0.345 |
| Aleatoria 5 folds | Ridge 148 | 0.371 | 0.498 | 0.655 |
| Espacial 6 bloques | Media global | 0.898 | 0.991 | -0.366 |
| Espacial 6 bloques | Media municipal suavizada | 0.530 | 0.685 | 0.348 |
| Espacial 6 bloques | Ridge 24 | 0.752 | 0.899 | -0.124 |
| Espacial 6 bloques | Ridge 148 | 0.502 | 0.684 | 0.350 |

Métricas agregadas sobre todas las predicciones *fuera de muestra* en cada esquema de validación. En la validación aleatoria usamos `KFold(5, shuffle=True, random_state=2026)`. En la espacial generamos seis bloques con KMeans aplicado a coordenadas y evaluamos con `GroupKFold(6)`.

**Interpretación**: Ridge 148 es el Ridge más eficaz de los dos, pero su RMSE en validación espacial (0.684) casi empata con el promedio municipal (0.685). No se ha demostrado una ventaja clara frente a ese control geográfico.

## Archivos generados

- `metricas_ridge.csv`: rendimiento de Ridge 24, Ridge 148 y dos referencias.
- `predicciones_validacion_ridge.csv`: rendimiento real, predicho y error por parcela; predicciones *out-of-fold*, no predicciones del entrenamiento.
- `alfas_elegidos_ridge.csv`: valor de `alpha` usado en cada fold.
- `coeficientes_Ridge_24_curadas.csv`, `coeficientes_Ridge_148_historicas.csv`: coeficientes tras estandarización, ordenados por magnitud; **no son efectos causales** ni importancias independientes, dada la correlación entre índices.
- `Ridge_24_curadas.joblib`, `Ridge_148_historicas.joblib`: pipelines entrenados con 138 parcelas para reutilización exploratoria (cargar con `joblib.load` solo desde fuentes confiables).
- `ridge_validacion_espacial.png`: gráfica observado vs. predicho de ambos Ridge usando predicciones fuera de muestra espacial.
- `resumen_experimento.json`: metadatos y métricas.

## Advertencias científicas

- La fuente de rendimiento **no identifica el año o ciclo agrícola de cada etiqueta**; por tanto, no se ha comprobado el rendimiento prospectivo para 2026.
- Las 24 características curadas ya aparecían en experimentos previos del ZIP. Si esa selección usó las mismas etiquetas, la estimación de desempeño puede estar optimistamente sesgada por selección previa.
- Con 138 parcelas y múltiples variables correlacionadas, los coeficientes pueden cambiar entre validaciones; no inferir causalidad a partir de su signo.
- La validación espacial con KMeans/GroupKFold es una aproximación, no una garantía de precisión en municipios o campañas enteramente nuevos.
- El modelo municipal se calcula **solo con las etiquetas del entrenamiento de cada fold** para evitar fuga de datos.
- No publicar proyecciones 2026 como respaldadas sin etiquetas correspondientes al ciclo y validación externa.
