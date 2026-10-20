# AgroCebada 2026 · Experimento 2: regresión lineal múltiple (OLS)

## Objetivo

Explicar y predecir `RENDIMIENTO_T_HA` (toneladas por hectárea) con **regresión lineal ordinaria sin regularización**, comparando sus resultados contra el experimento previo de Ridge. Se trabaja con **138 parcelas etiquetadas** y se preservan sin usar las **59 parcelas sin rendimiento** (no son ejemplos de prueba porque no hay una respuesta conocida).

## Resultados

| Algoritmo | RMSE aleatorio (t/ha) | R² aleatorio | RMSE espacial (t/ha) | R² espacial |
|---|---:|---:|---:|---:|
| OLS 8 variables | 0.7990 | 0.1124 | 1.0572 | −0.5539 |
| OLS 24 variables | 0.7177 | 0.2839 | 0.8908 | −0.1032 |
| OLS 148 variables | 2.4073 | −7.0576 | 10.5197 | −152.8661 |
| Ridge 24 variables (experimento anterior) | 0.6865 | 0.3447 | 0.8991 | −0.1239 |
| Ridge 148 variables (experimento anterior) | 0.4980 | 0.6551 | 0.6839 | 0.3497 |
| Media municipal suavizada (referencia anterior) | 0.5328 | 0.6053 | 0.6847 | 0.3482 |

**Comparación:** se usan exactamente los mismos ID, características 24 y 148, semilla `2026`, 5 folds aleatorios con `KFold(shuffle=True)` y 6 folds espaciales con `KMeans` de las coordenadas UTM y `GroupKFold` del trabajo anterior. Las métricas de Ridge proceden del archivo del experimento anterior, sin reentrenar Ridge en este script.

**Qué significa:** OLS 24 es la variante OLS menos mala en validación espacial, pero R² espacial negativo indica generalización deficiente. OLS 148 **sobreajusta gravemente**: entrenamiento R²=1.0000 y RMSE aproximadamente cero; validación espacial R²≈−152.87 con valores predichos entre −33 y 36 t/ha. Esto ocurre porque hay más variables (148) que observaciones de entrenamiento (138 en total, menos por fold), y por correlaciones entre características. No usar OLS 148 para predecir.

La referencia de municipio sigue siendo competitiva con Ridge 148; un modelo agronómico útil debe mejorar de forma consistente ese benchmark en nuevas zonas.

## Requisitos

- Python 3.10 o posterior
- Paquetes: `numpy pandas scikit-learn matplotlib joblib`

## Ejecución

1. Coloca `regresion_lineal_agro.py` y `DATASET_RETO_AGRO_2026.zip` en la misma carpeta (no hace falta descomprimir el ZIP original).
2. En esa carpeta ejecuta:

```bash
python -m pip install numpy pandas scikit-learn matplotlib joblib
python regresion_lineal_agro.py
```

Opcional:

```bash
python regresion_lineal_agro.py --zip /ruta/dataset.zip --salida /ruta/resultados
```

El programa guarda archivos en `resultados_regresion_lineal_agro/`.

## Archivos entregados

- `regresion_lineal_agro.py`: entrenamiento, evaluación, exportación y gráficos.
- `resultados_regresion_lineal_agro/metricas_regresion_lineal.csv`: métricas globales fuera de muestra.
- `resultados_regresion_lineal_agro/metricas_por_fold_lineal.csv`: métricas de cada bloque de validación.
- `resultados_regresion_lineal_agro/comparacion_ridge_vs_lineal.csv`: comparación con el experimento Ridge previo.
- `resultados_regresion_lineal_agro/predicciones_validacion_lineal.csv`: observados y predichos **fuera del fold de entrenamiento**, por parcela y escenario.
- `resultados_regresion_lineal_agro/diagnostico_sobreajuste_lineal.csv`: métricas dentro de la muestra y rangos matriciales.
- `resultados_regresion_lineal_agro/coeficientes_Lineal_*.csv`: coeficientes de modelos entrenados con toda la muestra. Los coeficientes están en unidades de 1 desviación estándar del predictor; no representan causalidad.
- `resultados_regresion_lineal_agro/Lineal_*.joblib`: pipelines finales ajustados con las 138 etiquetas (solo exploratorios; **OLS-148 no apto para uso práctico**).
- `resultados_regresion_lineal_agro/lineal_validacion_espacial.png`: predicción frente a valor observado, OLS 8 y 24.
- `resultados_regresion_lineal_agro/comparacion_rmse_lineal.png`: error de OLS 8/24/148, escala logarítmica.
- `resultados_regresion_lineal_agro/resumen_experimento_lineal.json`: metadatos y métricas.

## Puntos metodológicos

1. Una fila = una parcela. El ID nunca entra como predictor. Rendimiento nunca entra como predictor.
2. Escalado e imputación se ajustan exclusivamente con las parcelas de entrenamiento de cada fold. La separación en bloques geográficos es no supervisada (se realiza con las coordenadas) para evaluar generalización fuera de zona.
3. En OLS no se elige `alpha` porque no tiene término de regularización. Esto diferencia a OLS de Ridge.
4. OLS 8 utiliza 8 variables agronómicas fijadas previamente. OLS 24 reproduce las 24 variables del experimento Ridge. OLS 148 reproduce las 148 históricas de Ridge; se incluye para **documentar el sobreajuste**, no como alternativa recomendada.
5. La magnitud de coeficientes con variables correlacionadas (como NDVI, EVI y LAI) es inestable y no indica importancia causal.
6. **Advertencia temporal:** no se dispone de una etiqueta que feche claramente el rendimiento respecto de los periodos de las variables. Por tanto, estas validaciones NO prueban que el modelo prediga correctamente la cosecha de 2026.
7. Por prudencia no se generan predicciones para las 59 parcelas sin etiqueta.

## Conclusión

Ridge 148 (RMSE espacial ~0.684 t/ha) es preferible a la regresión lineal ordinaria entrenada con las mismas 148 variables. La regresión lineal de 24 variables (RMSE espacial ~0.891 t/ha) se mantiene como **baseline interpretativo**, pero no alcanza una calidad suficiente para extrapolar a parcelas nuevas.
