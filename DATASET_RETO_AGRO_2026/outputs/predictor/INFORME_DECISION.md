# AgroCebada 2026 — Decisión del predictor final

**Decisión: ADOPTAR PRIOR.** El predictor final es la media municipal con encogimiento calibrado (k = 0.078, elegido por leave-one-out en las 138 parcelas). Ridge 148 y el híbrido prior + topografía no muestran una mejora defendible en comparación pareada y quedan como **experimentales**. Como las diferencias no son concluyentes, se elige el modelo más sencillo.

## 1. Protocolo (README 01 — Claude; ver `CUMPLIMIENTO_README.md`)

- **Auditoría de datos** (`auditoria_datos.json`, 6/6 OK, sin transformar tablas). Las 197 parcelas coinciden en las 5 tablas y las etiquetas y la partición coinciden con la fuente oficial. Las 148 columnas son las mismas de Ridge/OLS/RF/ET previos, y la tabla centralizada es byte a byte la que leían esos scripts.
- **Particiones** (`splits.csv`). Las de la auditoría previa no venían en el ZIP. Se regeneraron con sus semillas documentadas: KFold(5, shuffle, 1000–1009) × 10 sobre las 138 parcelas ordenadas por `ID_POLIGONO`. Reproducen sus métricas a 4 decimales (prior k = 3: 0.5245; calibrado: 0.5123), aunque no se pudo verificar byte a byte. **Es una comparación exploratoria**: las 138 etiquetas ya se usaron en análisis previos, no hay prueba independiente y no se usaron las 59 ocultas.
- **Ajuste dentro del fold.** k, imputación, escalado y alpha se ajustan solo con entrenamiento; el residuo del híbrido usa el prior *leave-self-out*.
- **Regla fijada antes de correr** (`regla_decision_preregistrada.json`). Un candidato se adopta solo si cumple **E1** (IC95 de ΔRMSE por bootstrap de 25 microclústeres espaciales < 0), **E2** (Nadeau–Bengio, p Holm < 0.05), **E3** (gana en ≥ 9/10 repeticiones) y **S1** (ΔRMSE ≤ 0 en ≥ 3 de los 4 municipios con n ≥ 10). KM6 y leave-municipality-out (LMO) son solo diagnóstico.

## 2. Resultados en CV aleatoria 5 × 10

Esquema más parecido al reto: test→train a 426 m de mediana, frente a 350 m de las 59 ocultas.

| Modelo | RMSE (sd reps) | MAE | R² | ΔRMSE vs PRIOR [IC95 clúster] | ΔMAE [IC95] | Gana | p NB (Holm) |
|---|---|---|---|---|---|---|---|
| Media global | 0.854 | 0.754 | −0.01 | | | | |
| Prior k = 3 (equipo) | 0.525 (0.004) | 0.365 | 0.62 | +0.012 [+0.001, +0.028] | | 0/10 | |
| **PRIOR calibrado** | **0.512 (0.004)** | **0.336** | **0.64** | — | — | — | — |
| Ridge 148 | 0.506 (0.010) | 0.376 | 0.64 | −0.006 [−0.056, +0.052] | +0.039 [−0.003, +0.082] | 7/10 | 0.81 (0.81) |
| Híbrido prior + TOPO | 0.500 (0.005) | 0.351 | 0.65 | −0.012 [−0.026, +0.004] | +0.015 [0.000, +0.033] | 10/10 | 0.18 (0.37) |

- **Ridge 148 (falla E1, E2, E3 y S1).** Su 0.498 se reproduce exacto con la partición del equipo (seed 2026), pero es el extremo favorable del rango entre repeticiones (−0.020 a +0.018 frente al prior calibrado). Tiene más varianza y peor MAE. Por municipio: Chignahuapan −0.031; Calpulalpan +0.030, Singuilucan +0.090, Almoloya +0.031.
- **Híbrido (falla E1, E2 y S1; cumple E3).** Gana 10/10 repeticiones, pero el IC llega a +0.004 y pierde en Calpulalpan (+0.018) y Almoloya (+0.041). Queda como hipótesis.
- **Diagnóstico espacial.** KM6: prior 0.679, Ridge 0.744, híbrido 0.674. LMO: el prior cae a la media global (1.153) y Ridge da 0.722 porque reconstruye la región; las 59 ocultas están en municipios etiquetados, así que no es el escenario del reto.
- **Diagnóstico post hoc** (no candidato). Prior + Ridge 148 sobre el residuo da 0.518 (+0.005 [−0.002, +0.017]), con alpha en el tope (10⁴) en 31 de 50 folds. Dado el municipio, las 148 variables no aportan señal detectable.

## 3. SHAP de Ridge 148 (análisis comparativo; NO explica la predicción final)

- **Explicador.** SHAP lineal intervencional exacto en unidades originales, a través de la imputación y el escalado: φⱼ = wⱼ/sⱼ·(xⱼ − media del fondo). Supone independencia (la versión condicional no aplica: n_train ≈ 110 < p = 148). Base + Σφ reproduce la predicción (error ≤ 1.8·10⁻¹⁵) y coincide con `shap.LinearExplainer` y los coeficientes del equipo.
- **Familias del contrato** (descriptivas; η² = fracción de la varianza SHAP entre municipios; ΔRMSE al permutar el bloque):

| Familia (n var.) | \|SHAP\| t/ha | η² mun. | Perm. global | Perm. intra-mun. |
|---|---|---|---|---|
| Ubicación (0) / Geometría (0) | — | — | — | — |
| Clima (36) | 0.52 | 0.89 | +0.449 | +0.043 |
| Satélite (110) | 0.20 | 0.20 | +0.088 | +0.048 |
| Topografía (2) | 0.06 | 0.16 | +0.011 | +0.009 |

- **Lectura.** El clima atribuye casi todo entre municipios (η² de las predicciones de Ridge 0.87; del prior 0.997): es un proxy geográfico, no causal. Las variables correlacionadas se tratan por bloque (agregación y permutación conjunta): BÁSICO y PRO se compensan dentro del municipio (r = −0.64; por separado +0.093 y +0.074, juntos +0.051). La permutación intra-municipio mide la dependencia de este modelo, sin término municipal, no información adicional sobre el prior.
- **Parcelas representativas** (`explicaciones_locales_representativas.csv`): acierto, error alto, y mínimo y y máximo. Ningún modelo captura los extremos dentro del municipio.

## 4. Predictor final, pruebas y limitaciones

- **Predicción** = media global (3.955) + peso municipal × (media municipal − media global), entrenada con las 138 etiquetas. Es la explicación principal: las dos `prediction_components` del JSON y `explicacion_referencia_municipal.csv`. Artefacto único `modelo_final.json`; `predecir_parcela()` devuelve el contrato (`ejemplo_prediccion.json`).
- **Incertidumbre.** Intervalo de predicción empírico (no de confianza): ± cuantil 80 % de los residuos leave-one-out del prior, por municipio si n ≥ 10 y agrupado si no. Recalculado por fold en RCV5x10 cubre 0.87 (0.85–0.89; Chignahuapan 0.81, Calpulalpan 0.99, Singuilucan 0.81), ancho medio 1.24 t/ha. Un municipio no visto recibe la media global con `pending_validation` y límites `null`.
- **Pruebas:** 14/14 OK (detalle en `verificacion.json` y `CUMPLIMIENTO_README.md`). Solo falló un primer intento de la Fase 3 por una ruta (corregido).
- **Limitaciones.** (1) No hay prueba independiente; con IC de semiancho 0.015–0.054 t/ha, una mejora de ≈ 0.01 no se distingue de cero. (2) El nivel municipal está confundido con el régimen de registro del rendimiento (múltiplos de 0.5). (3) Sin ciclo en las etiquetas: no hay validación temporal ni para 2026. (4) Municipios con n < 10 (tres con 1 parcela) tienen medias inestables. (5) El intervalo se validó con las mismas 138 parcelas. (6) Para municipios no vistos, Ridge 148 supera a la media global (LMO), pero no está adoptado.
