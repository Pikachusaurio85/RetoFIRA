# Informe de evidencia: conceptos agronómicos y prototipos temporales AgroCebada 2025

## Resumen ejecutivo

**La colección permite descriptores espectrales y temporales observables por parcela, pero todavía no permite declarar estados fisiológicos, estrés hídrico o fases fenológicas como verdad agronómica.** Las observaciones válidas cubren las 197 parcelas a lo largo de los meses de 2025; aun así, el porcentaje de filas aceptables cae en la ventana vegetativa calendárica, y el alisado de tres meses cambia sistemáticamente el pico. Los valores de índices con el mismo nombre entre Sentinel-2 y Planet están correlacionados, pero no son intercambiables. Los prototipos de tres grupos no son suficientemente estables. En la evaluación espacial, las características agregadas de abril-octubre superan a las temporales más elaboradas en las tres configuraciones, pero no vencen consistentemente las referencias geográficas en los bloques externos.

Conclusión principal: **usar canales y descriptores espectrales/temporales por sensor con fecha y calidad; presentar las interpretaciones agronómicas como hipótesis; no usar hoy clusters como clases XAI ni llamar a un índice “estrés” o “LAI medido”.** No se generaron pseudo-etiquetas de concepto a partir del rendimiento.

## Diseño y límites de inferencia

| Campo | Contenido |
|---|---|
| **Hipótesis** | Determinar si existe soporte observacional suficiente para conceptos y prototipos temporales estables, verificables y útiles para XAI. |
| **Experimento** | Se leyeron las fuentes originales BASIC/PRO de 2022–2025, asignación de parcelas, y los productos existentes parcelarios de clima/topografía. La evaluación detallada de curvas se limita a 2025. Se preservan fila, fecha, ID, sensor, archivo, índices y nubosidad. No se usaron las dekadas previamente interpoladas para la auditoría de cobertura ni para clustering. |
| **Resultado** | 155,470 capturas en BASIC+PRO; 75,560 capturas fechadas en 2025. Se conservaron 138 etiquetas de entrenamiento y 59 parcelas de predicción sin etiqueta. |
| **Incertidumbre** | Las etiquetas de rendimiento no especifican ciclo agrícola/año, y no hay mediciones de campo de cobertura, LAI, agua, etapas fenológicas o manejo que verifiquen los conceptos. |
| **Interpretación** | La evidencia respalda la disponibilidad de señales y descriptores estadísticos, no su causalidad ni validez fisiológica. |
| **Decisión XAI** | Mantener cada observación como señal fechada y específica del sensor; agregar un valor de confianza/soporte observacional. Las etiquetas semánticas son candidatas pendientes de validación. |

## A. Calidad y cobertura

### Cobertura por sensor

El universo espacial tiene 197 parcelas y nueve municipios. No se encontraron IDs fuera de la capa parcelaria, fechas no parseables, claves repetidas `(ID_POLIGONO, fecha, sensor)` ni filas completas duplicadas. Cada parcela aparece en las fuentes para cada mes de 2025; además, para Sentinel-2 y Planet, cada combinación parcela-mes tuvo al menos una señal con el filtro de nubosidad utilizado. Este indicador es presencia de al menos una observación en el mes, **no** significa que todas las capturas fueran utilizables ni que la densidad temporal fuera homogénea.

| Sensor | Capturas 2025 | Señal principal no nula | Señal + umbral de nubes | Nubosidad media de todas las filas |
|---|---:|---:|---:|---:|
| Sentinel-2 | 16,820 | NDVI: 77.8 % | 67.9 % | 27.0 % |
| Planet | 47,804 | NDVI: 74.8 % | 67.5 % | 30.1 % |
| Landsat | 10,936 | VI6T: 66.9 % | 58.7 % | 33.3 % |

La nube media elevada y la mediana igual a cero en Sentinel-2/Planet reflejan distribuciones asimétricas con muchas filas despejadas y capturas muy nubladas; no deben resumirse con una sola media. Landsat se audita mediante VI6T porque la fuente no proporciona NDVI Landsat en esta tabla. Sus 6,414 observaciones utilizables son VI6T no nulo con nubosidad <=5 %; no se reinterpretó VI6T como NDVI.

| Año | Sentinel-2: capturas (primera–última) | Landsat: capturas (primera–última) | Planet: capturas (primera–última) |
|---:|---|---|---|
| 2022 | 13,922 (01-ene–27-dic) | 12,806 (05-ene–31-dic) | — |
| 2023 | 13,917 (01-ene–27-dic) | 13,125 (01-ene–27-dic) | — |
| 2024 | 14,381 (01-ene–31-dic) | 11,759 (03-ene–29-dic) | — |
| 2025 | 16,820 (05-ene–31-dic) | 10,936 (05-ene–31-dic) | 47,804 (01-ene–30-dic) |

La proporción de filas con señal principal y filtro de nube admisible también cambia por municipio: S2 va de 64.1 % en Singuilucan a 79.8 % en Apan; Planet, de 57.8 % a 75.5 %; Landsat, de 45.7 % en Almoloya a 74.8 % en Emiliano Zapata. Los extremos municipales de 1–3 parcelas son inestables. `calidad_parcela_sensor_2025.csv` detalla por parcela fechas válidas, primera/última fecha y brechas temporales; `calidad_parcela_sensor_etapa_2025.csv` entrega las brechas dentro de cada ventana.

La mediana por parcela fue de 58 observaciones núcleo utilizables para S2, 168 para Planet y 30 para Landsat. Las 197 parcelas tuvieron al menos una observación núcleo por mes con S2 y Planet; con Landsat, 139/197 tuvieron una observación núcleo en los 12 meses. La mediana parcelaria de la brecha entre observaciones válidas fue 5 días para S2, 1 para Planet y 8 para Landsat; el percentil 90 de la brecha máxima entre una parcela y otra fue 28, 25 y 72 días, respectivamente. Esto respalda cobertura temporal densa para las dos primeras fuentes, aunque no uniformidad completa entre sensores.

| Ventana calendárica — hipótesis, no fase observada | Sentinel-2: filas señal+nube admisible | Planet: filas señal+nube admisible | Landsat: filas VI6T+nube admisible |
|---|---:|---:|---:|
| Pre-ciclo DOY 91–151 | 74.5 % | 76.9 % | 74.3 % |
| Vegetativo DOY 152–212 | 47.0 % | 49.0 % | 38.2 % |
| Pico DOY 213–273 | 63.2 % | 51.1 % | 44.9 % |
| Senescencia DOY 274–304 | 59.7 % | 56.0 % | 43.0 % |

Los porcentajes de la tabla son fracciones de **filas capturadas** que contienen la señal principal y cumplen el filtro de nube; no son fracciones de parcelas. Los CSV de resultados incluyen el denominador, conteo por etapa, mes, municipio y parcela, e incorporan combinaciones parcela-sensor-etapa con cero observaciones.

### Municipios, faltantes y trazabilidad

La cobertura de parcelas con al menos una observación usable por mes fue completa en los nueve municipios, pero sus tamaños varían de 1 a 100 parcelas; por ello, una aparente diferencia municipal no se debe interpretar como efecto territorial robusto en municipios pequeños. La frecuencia de observaciones filtradas cambia por mes y etapa. Los datos faltantes dependen de la variable y del sensor; la proporción de señal principal no nula es 77.8 % para NDVI S2, 74.8 % para NDVI Planet y 66.9 % para VI6T Landsat en 2025. `faltantes_por_variable_sensor.csv` conserva faltantes por variable e incluye el conjunto temporal completo.

La trazabilidad se conserva en `observaciones_2025_trazables.csv` (nombre del archivo fuente, número de fila original, fecha sin procesar, fecha parseada, parcela, sensor, porcentaje de nubes, índice y banderas de validez). El umbral de nube es una regla de selección usada para esta auditoría, no una certificación universal de calidad.

La asignación se preserva en `auditoria_etiquetas_por_parcela.csv` con su fila original y sin asignar un año/ciclo a la etiqueta. Las series mensuales usadas para exposición climática se exportan como `clima_mensual_trazable_2025.csv` (parcela, celda, año/mes, valores y fuentes CHIRPS/CHIRTS-ERA5); los rasgos topográficos parcelarios están en `topografia_por_parcela.csv` con nombre del derivado fuente.

### Clima y topografía

Las 197 parcelas tienen los 12 meses de clima de 2025; los datos CHIRPS/CHIRTS-ERA5 representan **25 celdas de 0.05° (~5 km)**, no 197 mediciones meteorológicas independientes. La precipitación y temperatura son exposición regional aproximada. La precipitación acumulada mediana abril-octubre varía entre municipios (por ejemplo, 888 mm Calpulalpan, 936 mm Chignahuapan y 1,275 mm Singuilucan); no es agua disponible en la raíz ni un indicador medido de estrés.

INEGI CEM4 proporciona elevación y pendiente para 197/197 parcelas. La mediana de elevación media parcelaria es 2,639 m y las medias parcelarias van de 2,496 a 2,905 m; la pendiente media parcelaria va de 0.29° a 8.28°. Los productos vienen de rásteres de 120 m; las parcelas pequeñas pueden cubrir pocos píxeles y un descriptor cartográfico no demuestra efecto agronómico.

## B. Curvas, picos, suavizado y fases posibles

Las curvas mensuales se construyen como medianas de capturas originales por mes que pasan el filtro de nube; la banda sombreada es el rango intercuartílico. **No** se rellenan meses ni se interpola el final/inicio para esta figura.

![Medianas e IQR por sensor](./resultados/graficas/curvas_mediana_por_sensor.png)

![Índices relevantes Sentinel-2](./resultados/graficas/indices_relevantes_s2_2025.png)

Se aprecia una señal espectral estacional: NDVI/EVI/LAI de Sentinel-2 ascienden aproximadamente de junio a agosto-septiembre y descienden después. Planet presenta máximos algo más tardíos en sus medianas. Landsat VI6T mantiene escala y comportamiento distintos, por lo que se representa separado. NDWI, MSI y NDDI también varían a lo largo del año, pero son **respuestas espectrales calculadas**, no observaciones de agua del suelo ni mediciones de estrés.

Las características por parcela conservan, según señal y sensor, medias estacionales, picos y DOY, amplitud frente a la mediana de DOY <=151, integral del exceso sobre esa base, pendientes entre observaciones con brecha <=45 días, ventanas calendáricas y cruces de media amplitud cuando hay soporte. El área bajo la curva es un descriptor en índice-días, no radiación absorbida. Los cruces SOS/EOS quedan nulos cuando la serie no permite hallar cruces contiguos válidos; no se extrapolan.

| Campo | Contenido |
|---|---|
| **Hipótesis** | Una curva suavizada permite identificar picos y resumir duración/integrales sin borrar cambios temporales reales. |
| **Experimento** | Se comparó NDVI mensual observado con una media móvil centrada de tres meses en 197 parcelas, por separado para S2 y Planet. |
| **Resultado** | En ambos sensores, el pico suavizado se desplazó una mediana de **un mes** respecto al no suavizado; el valor pico cambió una mediana de **−0.041 NDVI**. El percentil 90 del cambio absoluto del pico fue 0.086 para S2 y 0.092 para Planet. |
| **Incertidumbre** | El perfil mensual oculta variación intra-mes; el alisado de tres meses reduce los picos. No hay etapa fenológica observada que determine cuál curva es biológicamente correcta. |
| **Interpretación** | Peak, SOS/EOS y duración dependen de la elección de filtro y umbral; en este conjunto deben mostrarse junto con la curva sin suavizar, el número de capturas y la fecha de captura más cercana. |
| **Decisión XAI** | Preferir observaciones y descriptores transparentes a una curva suavizada opaca; si se alisa, conservar método y parámetros en la explicación. |

## C. Sensores y combinación

Se compararon observaciones de Sentinel-2 y Planet dentro de una misma parcela, con umbrales S2 <=5 %, Planet <=20 %, y emparejamiento por fecha más cercana dentro de cinco días. Se obtuvieron 11,284 pares distribuidos en las 197 parcelas; el desfase mediano fue cero días.

| Variable con nombre común | Mediana Planet − S2 | MAE de diferencia | Pearson | Lectura |
|---|---:|---:|---:|---|
| NDVI | +0.079 | 0.082 | 0.983 | Correlación alta, desplazamiento sistemático apreciable |
| EVI | +0.007 | 0.032 | 0.974 | Correlación alta, dispersión aún no nula |
| LAI | +0.957 | 1.057 | 0.924 | Escalas claramente distintas |

| Campo | Contenido |
|---|---|
| **Hipótesis** | Los índices con igual nombre pueden combinarse directamente entre sensores. |
| **Experimento** | Pares temporales por parcela y variable con tolerancia 3, 5 y 10 días; filtros de nubosidad por sensor; se reportan diferencia, MAE/RMSE y correlación. |
| **Resultado** | La correlación de NDVI a cinco días es 0.983, pero Planet presenta sesgo medio +0.081 respecto de S2. Para LAI, el sesgo medio es +1.057 y la diferencia mediana +0.957. |
| **Incertidumbre** | Una correlación alta no calibra reflectancia ni iguala composición espacial, resolución, procesado, máscaras o fórmulas. Las observaciones se repiten dentro de parcela. |
| **Interpretación** | La covariación y el orden relativo de parcelas son prometedores; los valores absolutos no son equivalentes ni están armonizados. VI6T no tiene pareja semántica NDVI en esta fuente. |
| **Decisión XAI** | No concatenar ni promediar índices de sensores como si compartieran escala. Mantener prefijos/canales separados y calibrar con escenas/mediciones independientes antes de fusionar. |

## D. Grupos y prototipos de curvas

El clustering descriptivo utiliza sólo Sentinel-2 NDVI, medianas observadas de abril a octubre y filtro de nube <=5 %. Las 197 parcelas (etiquetadas y no etiquetadas) tuvieron siete meses observados y fueron elegibles. No se consultó rendimiento para definir, seleccionar o describir los grupos. KMeans euclidiano con k=3 se emplea únicamente para producir una figura legible; no es un número de clases validado.

![Curvas y prototipos descriptivos](./resultados/graficas/prototipos_curvas_s2_ndvi.png)

| Campo | Contenido |
|---|---|
| **Hipótesis** | Existen grupos con forma de curva similar y prototipos estables a remuestreo, métrica y calidad. |
| **Experimento** | KMeans k=2–5; silhouette; 100 bootstrap de observaciones por parcela para k=3; comparación de euclidiana con clustering jerárquico por correlación; sensibilidad de filtro 1/5/10 % de nubes. |
| **Resultado** | Para k=3, tamaños 65/76/56 y silhouette 0.245. La silhouette mayor observada fue k=2: 0.337. ARI bootstrap medio 0.580 (DE 0.073; rango 0.376–0.738); ARI frente a métrica de correlación 0.668. Cambiar filtro de 5 % a 1 % o 10 % produce ARI 0.955 y 0.938, respectivamente. |
| **Incertidumbre** | La selección de k altera asignaciones; el remuestreo de observaciones y la distancia cambian grupos. La sensibilidad a nubes por sí sola es alta, pero no compensa la inestabilidad estructural. Los grupos pueden mezclar localidad, variedad o manejo no observados. |
| **Interpretación** | Las tres líneas son prototipos ilustrativos de las curvas disponibles, no arquetipos agronómicos demostrados ni etiquetas apropiadas para supervisar XAI. |
| **Decisión XAI** | No codificar estos clusters como conceptos ni categorías de parcela. Mostrar curvas representativas/medoides sólo como resumen exploratorio; conservar distancias, algoritmo, k y estabilidad. |

## E. Temporal frente a estadística agregada

Comparación supervisada de utilidad predictiva de variables espectrales, no validación de conceptos agronómicos. Las etiquetas solo se usan como objetivo de esta comparación y no en el clustering/construcción de conceptos. Se mantienen features por sensor. Ambos grupos se construyen con las mismas capturas utilizables de abril a octubre; temporal contiene 120 descriptores de forma/ventanas y agregado 40 estadísticas anuales estacionales (media, mediana, SD, p10 y p90 por sensor/variable). Los descriptores de temporal son determinísticos por parcela. Imputación por mediana, escalado y regresión Ridge (alfa=10) se ajustan exclusivamente dentro de cada fold.

Se comparte el diseño del Equipo A implementado en `analisis/comparison.py`: bandas espaciales contiguas y balanceadas sobre el eje principal de centroides UTM, configuraciones de 4, 6 y 8 bloques, bloque extremo reservado externamente y GroupKFold en bloques internos restantes. Todos los candidatos comparten los mismos índices. Las pruebas externas cambian de tamaño (35, 23 y 18), por lo que deben leerse como sensibilidades del protocolo, no promediarse como un test único. Se incluyen referencias de media municipal con encogimiento y k-NN espacial k=5; el rendimiento de los vecinos/priores procede exclusivamente de las parcelas de entrenamiento de cada fold.

| Bloques | Evaluación | Temporal RMSE / MAE / R² | Agregado RMSE / MAE / R² | Mejor referencia geográfica RMSE |
|---:|---|---|---|---:|
| 4 | Interna OOF | 0.701 / 0.508 / 0.062 | **0.631 / 0.457 / 0.241** | 0.582 |
| 4 | Externa (n=35) | 1.178 / 1.057 / −8.778 | **1.096 / 1.005 / −7.454** | 1.298 |
| 6 | Interna OOF | 0.758 / 0.558 / 0.076 | **0.680 / 0.520 / 0.256** | 0.679 |
| 6 | Externa (n=23) | 0.861 / 0.703 / −3.562 | **0.788 / 0.629 / −2.822** | **0.456** |
| 8 | Interna OOF | 0.758 / 0.585 / 0.100 | **0.712 / 0.547 / 0.205** | 0.696 |
| 8 | Externa (n=18) | 0.887 / 0.755 / −4.179 | **0.719 / 0.573 / −2.403** | **0.444** |

| Campo | Contenido |
|---|---|
| **Hipótesis** | Conservar la forma temporal añade utilidad frente a resumir la señal del ciclo, fuera del espacio de entrenamiento. |
| **Experimento** | Ridge fijo, características temporales vs agregadas de las mismas observaciones/filtros, particiones espaciales comunes del Equipo A y referencias geográficas. No se ajustaron features ni hiperparámetros con el bloque externo. |
| **Resultado** | El agregado estacional obtuvo menor RMSE y MAE que temporal en las seis comparaciones internas/externas (3 configuraciones × 2 particiones). El R² externo de todos los candidatos es negativo. En bloques 6 y 8, las referencias geográficas superan a ambos grupos satelitales; el bloque 4 externo muestra una gran degradación de las referencias, prueba de sensibilidad al bloque. |
| **Incertidumbre** | n=138; holdouts pequeños; una etiqueta sin año/ciclo, por lo que esta evaluación no demuestra predicción temporal 2025 ni 2026. Ridge y features fijas son una prueba sencilla, no búsqueda exhaustiva. |
| **Interpretación** | No hay evidencia de que 120 features temporales sean mejores que 40 estadísticas agregadas para esta tarea y protocolo; tampoco se demuestra que los agregados sean conceptos agronómicos. La separación geográfica domina en partes de la prueba. |
| **Decisión XAI** | Empezar con estadísticas por sensor y pocas variables temporales verificables; justificar cualquier incremento temporal con mejoras estables en los mismos folds y con etiqueta/ciclo conocido. Mantener las predicciones OOF/externas para auditoría. |

## F. Catálogo candidato de conceptos

El CSV `resultados/catalogo_conceptos.csv` contiene definiciones operativas, variables, evidencia, límites, confianza y validación sugerida. La confianza expresa la solidez de la **interpretación propuesta**, no la calidad general del archivo de datos.

| Concepto | Nivel actual | Decisión |
|---|---|---|
| Vigor espectral (NDVI/EVI) | Moderado como señal espectral; medio para el candidato | Se puede explicar como índice observado, no como vigor fisiológico probado. Comparar por sensor. |
| Cobertura vegetal | Bajo-moderado, indirecto | Candidato para validar con cobertura fotográfica/cuadrantes; no llamarlo cobertura medida. |
| Dinámica fenológica | Moderado en forma de curva; bajo en etapa biológica | Presentar pico/amplitud/integral con fecha y filtro; no asignar fase sin calendario de campo. |
| Condición hídrica espectral | Exploratorio/bajo | No usar como etiqueta de estrés o humedad. Validar con humedad de suelo, estado hídrico e irrigación. |
| Exposición climática | Alta cobertura del raster; media como exposición local | Apto como contexto de celda regional; no como microclima/agua disponible. |
| Topografía | Moderado-alto como descriptor cartográfico | Apto como covariable estática con cautela de resolución; el efecto agronómico requiere validación. |
| Grupos/prototipos temporales | Insuficiente para clases estables | No usar como concepto etiquetado en XAI. |
| Equivalencia de índices entre sensores | Evidencia alta de no equivalencia absoluta en esta colección | Separar sensores; armonizar/calibrar antes de combinar. |

## G. Integridad de etiquetas y disponibilidad temporal

| Campo | Contenido |
|---|---|
| **Hipótesis** | Las etiquetas pueden alinearse sin ambigüedad con un ciclo/fecha de cierre y servir para escenarios de predicción anticipada y al cierre. |
| **Experimento** | Reconciliación uno-a-uno de IDs entre la asignación canónica y las 197 parcelas; conteos de etiqueta y rango; contraste con el calendario de las fuentes satelitales. |
| **Resultado** | 138/138 parcelas `ENTRENAMIENTO` tienen rendimiento (2.00–5.59 t/ha; mediana 4.34); las 59 `PREDICCION` no tienen etiqueta. La asignación no incluye campo de año/ciclo. Las observaciones satelitales 2025 llegan hasta 30-dic (Planet) y 31-dic (S2/Landsat). |
| **Incertidumbre** | No se puede establecer que las etiquetas correspondan al ciclo 2025 ni identificar fecha de siembra, madurez, cosecha o corte de información. Que existan imágenes durante 2025 no prueba que sean previas a la etiqueta o al cierre agronómico. |
| **Interpretación** | La prueba de E mide asociación espacial entre features satelitales observadas en 2025 y la etiqueta disponible, pero no valida un pronóstico al cierre de 2025 ni una predicción anticipada. |
| **Decisión XAI** | **Cierre del ciclo:** exigir cierre/fecha de cosecha por parcela y ciclo, excluir observaciones posteriores al cierre definido y emparejar rendimiento del mismo ciclo. **Anticipada:** fijar una fecha de corte y usar solo capturas/clima con fecha <= corte; entrenar con etiquetas fechadas y usar partición espacial/temporal. No producir resultados de transferencia temporal hasta contar con las etiquetas correspondientes. |

## Recomendaciones de representación XAI

1. Mantener observaciones y features específicos por sensor (`s2_*`, `planet_*`, `ls_*`); no agrupar valores por el nombre de índice.
2. Para cada explicación, exponer fecha/ventana, filas disponibles, filtro de nube, porcentaje faltante, fuente y versión de extracción. Mostrar intervalos y no solo una predicción de concepto.
3. Priorizar variables interpretables: NDVI/EVI/LAI **derivados**, amplitud, fecha de máximo, integral, pendientes y agregados estacionales. Separar señal primaria, descriptor temporal e indicador derivado.
4. Llamar “pico de NDVI observado” a un pico espectral; reservar “espigamiento”, “senescencia”, “estrés hídrico”, “cobertura (%)” o “LAI medido” para variables con verificación de campo.
5. No alimentar el modelo con clusters actuales como etiquetas ni presentar prototipos como tipos de cultivo. Las curvas representativas pueden servir como ejemplos visuales con su incertidumbre.
6. Para futuras pruebas, congelar folds espaciales y cortes temporales compartidos; dentro de cada fold, ajustar imputación/escalado/selección y cualquier embedding únicamente en train. Conservar predicciones y métricas por fold, no solo promedios.
7. Versionar las fórmulas de índices, máscaras QA, procesado de reflectancia y resoluciones antes de una fusión multisensor.

## Entregables y reproducción

- Código: [analizar.py](./analizar.py).
- Reproducibilidad, fuentes, parámetros y SHA-256: [manifest.json](./resultados/manifest.json).
- Métricas, predicciones y tablas de calidad: archivos CSV en `resultados/`.
- Curvas mensuales: [curvas_mediana_por_sensor.png](./resultados/graficas/curvas_mediana_por_sensor.png), [indices_relevantes_s2_2025.png](./resultados/graficas/indices_relevantes_s2_2025.png).
- Prototipos exploratorios: [prototipos_curvas_s2_ndvi.png](./resultados/graficas/prototipos_curvas_s2_ndvi.png).

Los CSV de series `series_dekadales_2025.csv` y features `features_parcela.csv` del baseline se consultaron como contexto/documentación, pero este informe no los usa para recuperar faltantes ni para definir las curvas analizadas.
