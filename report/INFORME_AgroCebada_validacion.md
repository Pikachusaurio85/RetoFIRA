# AgroCebada FIRA 2026 — ¿Qué información aporta cada fuente y qué arquitectura interpretable se justifica?

**Informe ejecutivo con hallazgos verificados** · Sesión de análisis para Mike (equipo AgroCebada) · 14-oct-2026
Semilla global `20261013` · Todas las cifras salen de `results/*.csv` (la fuente de cada una se indica entre corchetes) y se regeneran con los scripts de `scripts/` (ver §7).

> **Cómo leer este informe.** Primero el resumen (§0). Luego, una tabla por investigación con el formato pedido (Hipótesis / Experimento / Resultado / Incertidumbre / Interpretación / Decisión XAI) en §2–§4. Después: pregunta de la "curiosidad" (§5), arquitectura XAI e hipótesis descartadas (§6, entregable D), preguntas para agronomía (§8, entregable E) y el cierre "sabemos / sospechamos / no podemos afirmar" (§9). Las unidades son t/ha; RMSE salvo que se indique. "Δ" = diferencia de RMSE respecto a la referencia (negativo = mejor). Zona de equivalencia práctica (ROPE) = ±0.02 t/ha.

---

## 0. Resumen ejecutivo

1. **La prueba oficial es un problema de interpolación, no de transferencia regional.** Las 59 parcelas ocultas están intercaladas con las 138 etiquetadas: AUC de validación adversaria 0.44–0.50 (percentil 95 bajo H0: 0.58–0.62) y distancia mediana a la parcela etiquetada más cercana de 350 m. La CV aleatoria (425 m) es el esquema que más se le parece; los bloques espaciales (1.4–9.4 km) miden otra pregunta —transferencia a zonas sin datos— que no es la del reto. [00_validacion_adversaria_ocultas, 01_descriptor_esquemas_validacion]
2. **El 69 % de la varianza del rendimiento es entre municipios (η² = 0.686; IC95 bootstrap 0.57–0.80), pero ese efecto es casi por completo un contraste de dos regímenes de captura del dato**: municipios con rendimiento continuo (≈ 4.5 t/ha: Chignahuapan, Almoloya, Cuautepec, Tepeapulco) frente a municipios con rendimiento en múltiplos de 0.5 t/ha (≈ 3.0 t/ha: Calpulalpan —30/30 valores en {2.5, 3.0, 3.5}—, Singuilucan, Apan, E. Zapata, Nanacamilpa). El régimen solo explica R² = 0.675; añadir municipio sube solo +0.011 (p = 0.72). **No podemos distinguir "diferencia agronómica real" de "diferencia en cómo se registra el dato"** con lo que hay en el ZIP. [02_varianza_entre_dentro_municipio, 02_regimen_vs_municipio_secuencial, 00_cuantizacion_por_municipio]
3. **Dentro de cada municipio no hay autocorrelación espacial detectable a escalas ≥ 1 km** (Moran del rendimiento centrado por municipio: I = −0.008, p = 0.77), con **potencia limitada** (n = 71 en el municipio mayor: 28–65 % de potencia para estructuras moderadas). Solo hay un indicio débil a < 500 m (I = 0.33, p = 0.008, no sobrevive a corrección conservadora) y una estructura de escala de kilómetros visible cuando se retiran bloques completos. [02_moran_rendimiento, 02_correlograma_moran, 02_potencia_moran_chignahuapan]
4. **Ningún modelo supera de forma distinguible al prior municipal en CV aleatoria.** Prior municipal del equipo (encogimiento k = 3) 0.524; mejor residual 0.506 (Ridge + topografía, Δ = −0.019, IC95 por clúster espacial [−0.037, +0.001]). En los 46 contrastes "residuo sobre prior" bajo CV aleatoria hay 0 mejoras relevantes y 2 empeoramientos; ninguno pasa Holm. **Además, el propio prior está sobre‑encogido:** con k elegido dentro del fold (≈ media municipal) baja a **0.512** (Δ = −0.012 [−0.029, −0.0005] vs k = 3) y contra ese prior los residuales de CUR/CLIM/SAT/LOC/ALLNOLOC quedan en +0.000 a +0.004; solo topografía se mantiene (−0.013 [−0.026, +0.004]). [04_contrastes_pareados, 06b_prior_shrink_control]
5. **El satélite por sí solo es peor que la ubicación sola** (ET 0.641 vs 0.519; Ridge 0.667 vs 0.532; Δ = +0.12/+0.14, ambos significativos tras Holm) **y no aporta nada medible una vez dados municipio + ubicación + clima + topografía + geometría.** Satélite, clima y ubicación *identifican* municipio (exactitud 83–100 %, AUC de régimen 0.96–1.00): por eso ayudan solo en esquemas regionales (leave-municipality-out: hasta −0.27 t/ha), donde actúan como proxy de región/régimen (en LMO los modelos directos con ubicación/clima/satélite llegan a 0.66–0.70 frente a 1.15 de la media global). [04_contrastes_pareados, 02_identificabilidad_municipio_por_familia]
6. **Lo que los modelos "aprenden" es geografía.** En el ET directo con todas las variables, 58 % de la |SHAP| es clima y 23 % ubicación, con 98 % de su varianza *entre* municipios; permutar clima o ubicación *dentro* del municipio no cambia el error (ΔRMSE −0.004 y −0.0004). Clima ≈ ubicación (R² de CV de los componentes principales de clima desde las coordenadas: 0.55–0.93), así que repartir mérito entre ambas es arbitrario. [07_shap_grupos, 07_permutacion_grupos, 07_redundancia_entre_grupos]
7. **Señal intra‑municipio: solo candidatos débiles, ninguno confirmado.** Contra el prior bien calibrado solo sobrevive la **topografía** (pendiente/elevación; −0.013 [−0.026, +0.004]); a ella se suma el índice de línea base **`s2_ndvi_base`** (ρ intra-municipio = +0.30; p cruda 0.0066; FWER 0.11; FDR 0.24; mismo signo en 4/4 municipios; persistente entre años). Ninguno supera la corrección por multiplicidad. Son **hipótesis para validar**, no hallazgos. [08_evidencia_univariada_intra_municipio, 06_permutacion_intra_municipio, 06_dentro_de_municipio_cv_y_permutacion]
8. **Sesgos metodológicos medidos**: elegir variables *fuera* del fold infla el desempeño en 0.006–0.012 t/ha (del orden de las "mejoras" que de otro modo reportaríamos); la lista curada CUR24 **no** se distingue de conjuntos al azar con su misma composición (30 % de 300 sorteos la superan); las familias temporales T_VEG, T_PICO y T_PHTIM con ET *empeoran* de forma significativa (+0.025 a +0.044). [05_fuga_por_seleccion, 05b_cur24_control_composicion, 04_contrastes_pareados]
9. **No ejecutable: transferencia temporal.** Solo hay un ciclo etiquetado (2025) y los datos de rendimiento no traen año. Como proxy *sin etiquetas*: los índices de línea base (`ndvi/evi/lai_base`) son persistentes entre años (ρ 0.60–0.79) y los de pico son casi ruido interanual (ρ 0.08–0.31). [09_persistencia_temporal_sentinel2]
10. **Recomendación XAI:** arquitectura de dos capas explícitas —**prior municipal con encogimiento ajustado en el fold** (auditable, rinde 0.512) + **corrección dispersa** (≤ 5–6 variables, lineal con signos esperados, tratada como hipótesis)— con una regla de adopción basada en intervalos, no en el RMSE puntual (§6). No presentar SHAP de clima/ubicación como "explicación agronómica" y no usar satélite como predictor principal.

**Integridad (verificada con `scripts/99_verify_integrity.py`)**: el árbol original del ZIP quedó **byte a byte idéntico** (274 archivos; hash previo = posterior); **ningún ID de las 59 parcelas ocultas aparece** en particiones, OOF, SHAP, errores ni tablas; esas 59 solo se usaron *sin etiqueta* como features en la validación adversaria y en la persistencia temporal. [99_verificacion_integridad]

---

## 1. Diseño experimental (lo común a todo)

**Datos.** 197 parcelas (138 con rendimiento 2025, 59 sin), 9 municipios, 25 celdas climáticas (23 con parcelas etiquetadas), 219 columnas. Municipios con n ≥ 10 etiquetadas: Chignahuapan (71), Calpulalpan (30), Singuilucan (16), Almoloya (10); el resto tiene 1–5. Los archivos originales se copiaron a `data/` (solo lectura); nada se escribió sobre el pipeline original.

**Familias de variables** (`scripts/common.py::feature_families`): LOC (2 coordenadas UTM), S2 (124), PL (48), LS (5) → SAT (177), CLIM (26), TOPO (4), GEOM (3); familias *temporales* de SAT: T_PRE, T_VEG, T_PICO, T_SEN, T_PHTIM, T_PHLEV, T_HET; CUR (24 curadas del baseline), ALLNOLOC (todo menos ubicación), B1 (= SAT + TOPO + GEOM, el Baseline 1 del equipo), ALL, y ALL‑{SAT, CLIM, TOPO, GEOM}.

**Esquemas de validación** (particiones generadas una sola vez, guardadas en `splits/splits_long.csv`, huella SHA‑256 `56c85a25a3cd31a8…`, y reutilizadas idénticas por **todas** las configuraciones):

| Esquema | Qué mide | Reps | Folds | Dist. mediana test→train (m) | % test con vecino < 500 m |
|---|---|---|---|---|---|
| **Referencia: 59 ocultas → 138 etiquetadas** | el reto real | – | – | **350** | 66 % |
| **RCV5×10** (CV aleatoria 5 folds × 10 reps) | interpolación (≈ reto) | 10 | 50 | 425 | 60 % |
| SPA_GRID3B05 (celdas de 3 km, buffer 0.5 km) | interpolación con huecos | 10 | 60 | 1 368 | 0 % |
| SPA_GRID8B1 (celdas de 8 km, buffer 1 km) | transferencia local | 10 | 60 | 2 741 | 0 % |
| SPA_BAND6 (6 bandas por eje principal) | transferencia direccional | 1 | 6 | 1 669 | 14 % |
| SPA_KM4 / KM6 / KM8 (bloques k‑means) | transferencia regional | 1 | 4 / 6 / 8 | 9 443 / 3 348 / 3 347 | ≈ 0–2 % |
| LMO (leave‑municipality‑out) | municipio nuevo | 1 | 9 | 4 861 | 2 % |
| BLOO r = 0–8 km (LOO con buffer) | decaimiento con la distancia | 1 | 138 por radio | 400 → 8 296 | 67 % → 0 % |

[01_descriptor_esquemas_validacion] — **Distinción pedida:** *CV aleatoria* = RCV; *validación espacial* = GRID/BAND/KM/LMO/BLOO; *transferencia temporal* = **no ejecutable** (ver §2.6).

**Todo lo entrenable se ajusta dentro del fold:** prior municipal (media encogida con k = 3, *leave‑self‑out* para filas de entrenamiento), prior kNN espacial (k = 5), escalado, α de RidgeCV, selección de variables (filtro, Lasso, forward anidado) y cualquier importancia usada para seleccionar. Modelos: Ridge (StandardScaler + RidgeCV), ExtraTrees (300 árboles, `min_samples_leaf = 3`, `max_features = 0.5`), CatBoost (profundidad 4, 300 it.; limitado a 18 configuraciones por costo). Cada configuración usa las mismas semillas por (rep, fold) para que los errores sean *pareables*. Configuraciones evaluadas: 136 (esquemas principales) + 19 (BLOO). **Reconstrucción del baseline del equipo** (CV aleatoria): media global 0.854 (equipo: 0.85), prior municipal 0.524 (0.52), ET sat+topo+geom 0.638 (0.64), ET + clima 0.523–0.532 (0.54), híbrido prior + residual 0.513–0.517 (0.51): coincide con su tabla. La carpeta `reports/`/`figures/` del ZIP está vacía, por lo que no pude comparar contra OOF originales, solo contra las cifras del README.

**Inferencia pareada.** Pérdida por parcela = media sobre repeticiones del error cuadrático. ΔRMSE con bootstrap por parcela **y por 25 micro‑clústeres espaciales** (el segundo, más conservador, es el que se usa para decidir), Wilcoxon sobre errores, corrección de Holm por familia de contrastes. Etiquetas de decisión: *A MEJOR/PEOR (relevante)* = IC95 fuera de ±0.02 y Holm < 0.05; *EQUIVALENTE* = IC95 dentro de ±0.02; *INDETERMINADO* = el resto. [04_contrastes_pareados: 1 120 contrastes]

---

## 2. Hallazgos por investigación

### 2.0 Auditoría de datos (previa a todo)

| Campo | Contenido |
|---|---|
| **Hipótesis** | (a) Las etiquetas de las 59 ocultas no están filtradas en el ZIP; (b) el reparto oficial 70/30 es aleatorio dentro de municipio; (c) el rendimiento se registra con la misma resolución en todos los municipios. |
| **Experimento** | Escaneo de 19 archivos con columna de rendimiento; reparto oficial por municipio; validación adversaria (¿se distingue oculta vs etiquetada con features? 5 conjuntos, CV repetida y nulo permutado); cuantización del rendimiento por municipio (fracción de múltiplos de 0.5, binomial contra 2 % esperado). |
| **Resultado** | (a) 0 valores no nulos de rendimiento en las 59 ocultas en los 18 archivos con ID; 1 archivo (`datos_reorganizados_todas_observaciones.csv`) no tiene columna ID y no pudo verificarse por ID. (b) 28.6–31.8 % de ocultas en los 4 municipios con n ≥ 10; AUC adversaria 0.44–0.50 (nulo p95 0.58–0.62). (c) **No**: Calpulalpan 30/30 en {2.5, 3.0, 3.5}; Singuilucan 56 % múltiplos; Emiliano Zapata/Apan/Nanacamilpa 100 %; Chignahuapan 1/71 (1.4 %), 54 valores distintos; Almoloya, Cuautepec, Tepeapulco 0 %. |
| **Incertidumbre** | La validación adversaria detecta diferencias *observables en features*; no excluye diferencias en variables no medidas. Con 59 ocultas el AUC tiene poca potencia para desviaciones pequeñas. |
| **Interpretación** | El reto es interpolación aleatoria dentro de municipio. La resolución del dato de rendimiento **está confundida con el municipio**: una parte del "efecto municipio" podría ser instrumento de medición, no agronomía. |
| **Decisión XAI** | Tratar "municipio / régimen de captura" como un término explícito del modelo, no como algo que variables satelitales "expliquen". Pedir a agronomía el origen del dato (§8, P1). |

### 2.1 Investigación 1 — Variabilidad entre/dentro de municipios y autocorrelación espacial

| Campo | Contenido |
|---|---|
| **Hipótesis** | H1a: la mayor parte de la varianza es entre municipios. H1b: dentro de municipio hay estructura espacial del rendimiento. H1c: los residuos de los modelos conservan autocorrelación espacial explotable. |
| **Experimento** | ANOVA a una vía (η², ω², ICC), REML con bootstrap paramétrico, régimen de captura vs. municipio (SS secuenciales), celda climática dado municipio (permutación intra-municipio); I de Moran con permutación (global, intra-municipio con pesos y permutación dentro de grupo, por municipio; kNN‑5/8 y bandas ≤ 1/2/5 km), correlograma por anillos, semivariograma con envolvente de permutación, **simulación de potencia** (campo gaussiano exponencial sobre las coordenadas reales de Chignahuapan), Moran de residuos OOF de 12 configuraciones en 3 esquemas. |
| **Resultado** | **Entre/dentro:** η² = 0.686 [0.57, 0.80]; ω² = 0.665; ICC(REML) = 0.705 [0.26, 0.85]; sd total 0.851 vs sd intra‑municipio agrupada 0.491. Solo régimen: R² = 0.675; municipio dado régimen: +0.011 (p = 0.72). Dentro del régimen *fino* los 4 municipios son indistinguibles (η² = 0.003 [0.0005, 0.025]); dentro del *grueso* η² = 0.128 [0.10, 0.28]. Chignahuapan aporta 77 % de la suma de cuadrados intra. Celda climática dado municipio: ΔR² = 0.024 (esperado bajo H0: 0.026; p = 0.47). **Moran del rendimiento:** global (incluye entre municipios) I = 0.694 (z = 14.2); **centrado por municipio** I = −0.008 (kNN‑5, p = 0.77), −0.005 (kNN‑8), 0.011 / −0.070 / −0.010 (bandas 2 / 1 / 5 km, p bilateral ≥ 0.16); Chignahuapan solo: I = 0.0075 (p = 0.73). Únicos indicios: anillo 0–0.5 km I = 0.327 (p = 0.008; intra‑pooled 0.262, p = 0.011) y banda ≤ 5 km en Chignahuapan (p = 0.044 a dos colas). Semivariograma: dentro de la envolvente en todos los rezagos. **Potencia (n = 71, p < 0.05):** 9–25 % si solo 20 % de la varianza es espacial; 28–65 % con 40 % y rango 1–4 km; ≥ 70 % solo con ≥ 60 % de varianza espacial y rango ≥ 2 km. **Residuos del prior municipal bajo CV aleatoria:** I = 0.054 (p = 0.11 sobre 138); intra‑centrado −0.010 (p = 0.36); Chignahuapan 0.006 (p = 0.35); residuos de modelos residuales: I entre −0.02 y −0.09, ninguno positivo significativo. Bajo bloques k‑means (KM6) los residuos del prior sí están autocorrelacionados (I = 0.11–0.17, p ≈ 0.007–0.02) —ver interpretación. **LOO con buffer (radio 0 / 0.5 / 1 / 2 / 4 / 8 km, RMSE):** prior municipal 0.520 / 0.525 / 0.524 / 0.547 / 0.612 / 1.053; kNN espacial 0.513 / 0.578 / 0.531 / 0.569 / 0.648 / 0.749; ET solo ubicación 0.513 / 0.524 / 0.521 / 0.552 / 0.631 / 0.754; prior + Ridge(CUR24) 0.510 / 0.514 / 0.513 / 0.533 / 0.588 / 1.004; ET todo 0.527 / 0.549 / 0.549 / 0.575 / 0.662 / 0.950. [04b_bloo_rmse, fig07] |
| **Incertidumbre** | n = 71 en el municipio mayor, 30 y 16 en los siguientes: la ausencia de autocorrelación **no** equivale a su inexistencia (potencia moderada para estructuras suaves/cortas). Se probaron 5 pesos × 4 subconjuntos y 5 anillos: los p ≈ 0.01 aislados no sobreviven a corrección conservadora (Bonferroni sobre 5 anillos: 0.04 y 0.055). Los IC de ICC son muy anchos (ICC de REML inestable con 9 grupos y 4 efectivos). Las celdas climáticas (23) son casi colineales con municipio. |
| **Interpretación** | H1a: **sí**, pero el efecto es un contraste de dos regímenes de captura (≈ 98 % del η² municipal lo reproduce el régimen). H1b: **no hay evidencia de estructura espacial intra‑municipio a ≥ 1 km**; hay un indicio tenue < 500 m (parcelas contiguas, posiblemente el mismo productor/manejo) que no es explotable por un kNN (el prior kNN‑5 empata con el municipal: 0.5226 vs 0.5245, Δ = −0.002 [−0.048, +0.045]). H1c: bajo CV aleatoria **no** quedan residuos autocorrelacionados que un modelo espacial pudiera aprovechar; bajo bloques sí, pero es en gran parte *artefacto del diseño* (al retirar un bloque completo de un municipio, todas sus parcelas comparten el mismo error del prior) y señala una tendencia suave de escala de km dentro de Chignahuapan que RCV no puede ver. **BLOO:** entre excluir nada (vecino mediano a 0.4 km) y excluir 1 km, casi todos los modelos cambian ≤ 0.03 t/ha (el prior municipal, 0.004; excepciones: Ridge solo satélite +0.05 y el kNN, que salta +0.065 en 0.5 km y vuelve a 0.531 en 1 km, es decir, ruido de una sola partición): los vecinos a < 1 km no aportan casi nada más allá de pertenecer al municipio; el deterioro a ≥ 4 km ocurre sobre todo porque se retiran las parcelas del *propio municipio* (se pierde el prior), no porque exista un decaimiento espacial suave. |
| **Decisión XAI** | Prior **municipal** (no un kernel espacial) como base. No añadir un término de kriging/residuo espacial: no hay evidencia a la escala útil y el kNN no mejora. Mantener el hallazgo < 500 m como hipótesis (§8, P6). |

### 2.2 Investigación 2 — Ablaciones controladas por familia (mismas particiones)

| Campo | Contenido |
|---|---|
| **Hipótesis** | H2a: cada familia (LOC, S2, PL, LS, SAT, CLIM, TOPO, GEOM, temporales) añade información sobre rendimiento más allá del municipio. H2b: SAT añade información dados ubicación + clima + topografía + geometría. |
| **Experimento** | 136 configuraciones × 8 esquemas con particiones idénticas: **directo** (modelo sobre el rendimiento), **residual** (prior municipal + modelo sobre y − prior), referencias (media global, prior municipal, kNN, prior + kNN del residuo); familias aisladas, secuencia incremental MUN → LOC → CLIM → TOPO → GEOM → SAT, *drop‑one* desde ALL, y SAT dado todo lo demás. Contrastes pareados C1–C8 con bootstrap por parcela/clúster, Wilcoxon y Holm. |
| **Resultado (CV aleatoria, RMSE)** | Media global 0.854 · **prior municipal 0.524** (sd entre reps 0.004) · kNN espacial 0.523 · ET solo ubicación **0.519** · ET solo clima 0.523 · ET todo 0.532 · ET sin ubicación 0.533 · CatBoost todo 0.567 · **ET solo satélite 0.641 · ET Baseline 1 (sat+topo+geom) 0.638 · Ridge solo satélite 0.667**. **Residual sobre prior:** Ridge+TOPO 0.506 · ET+LOC 0.512 · Ridge+CUR 0.513 · Ridge+CLIM 0.514 · Ridge+SAT 0.518 · ET+SAT 0.527 · ET+ALL 0.530 · CatBoost+ALL 0.552. **Contrastes:** satélite vs ubicación (ET) Δ = +0.122 [+0.075, +0.173], Holm 0.004 → *peor*; (Ridge) +0.135 [+0.081, +0.185] → *peor*; directo[F] vs directo[LOC] (Ridge/ET/CatBoost, 240 contrastes): 139 peores / 0 mejores / 101 indeterminados. Residual vs prior (C1, 368 contrastes, 8 esquemas): bajo RCV 0 mejoras, 2 peores (ET + T_PICO +0.044; ET + T_PHTIM +0.034), 37 indeterminados; **mejoras relevantes solo bajo esquemas regionales** (LMO 40/46, KM4 31/46, KM6 18/46, KM8 14/46, GRID8 10/46) y 0 bajo GRID3 y BAND6. SAT dado todo lo demás (C6): 0 mejoras, 1 peor, 3 equivalentes de 16. Regional: ET+ALL sobre prior en LMO 0.885 vs 1.153 (Δ = −0.268 [−0.380, −0.180]); en KM6 0.651 vs 0.688 (−0.037 [−0.058, −0.018]). [04_metricas_por_esquema, 04_contrastes_pareados, 04_mejor_config_por_esquema] |
| **Incertidumbre** | Con n = 138 y 4 municipios de n ≥ 10, el IC95 por clúster de las diferencias en CV aleatoria mide ≈ ±0.025–0.04 t/ha, mayor que el ROPE: "indeterminado" es el resultado honesto para casi todo lo que está dentro de ±0.02. Esquemas KM/BAND/LMO tienen **una sola partición** (sin varianza entre repeticiones y dependientes del diseño). Hiperparámetros fijos (ver sensibilidad en 2.3). CatBoost solo en 18 configuraciones. |
| **Interpretación** | La información útil **y distinguible** es: municipio (−0.33 vs media global, IC [−0.44, −0.22]). Ubicación y clima aportan lo mismo que el prior municipal (son su proxy). El **satélite solo reconstruye parcialmente el municipio** (identifica municipio con 87 % de exactitud) y por eso, solo, rinde peor que la etiqueta municipal; dado el municipio, no se ve aporte. Las familias temporales de SAT no añaden nada, y con ET algunas añaden ruido (T_VEG, T_PICO, T_PHTIM). Las "mejoras" regionales de los modelos residuales son transferencia entre regímenes/regiones (no tienen que ver con el reto). **Advertencia sobre la referencia:** el prior con k = 3 está sobre‑encogido (con k anidado 0.512), de modo que las diferencias de −0.01 a −0.02 de los residuales *frente a k = 3* están en buena parte infladas por recalibración; el control de §2.4 lo corrige para Ridge. |
| **Decisión XAI** | Cuestiona una arquitectura "satélite → rendimiento" como pieza principal. Favorece: **prior municipal explícito + correcciones pequeñas con evidencia intra‑municipio**. Cualquier familia debe justificarse con ΔRMSE *intra‑municipio*, no con ganancias que reproducen el municipio. |

### 2.3 Investigación 3 — Estabilidad de modelos, coeficientes, selección y errores

| Campo | Contenido |
|---|---|
| **Hipótesis** | H3a: el ranking de modelos es estable entre esquemas. H3b: las variables y coeficientes seleccionados son estables. H3c: los errores por parcela y por bloque son homogéneos. H3d: elegir variables fuera del fold no sesga. |
| **Experimento** | Correlación de rangos de 136 configuraciones entre esquemas; sensibilidad a semillas (ET) y a hiperparámetros (`min_samples_leaf` × `max_features` en RCV y KM6); estabilidad de selección (Jaccard, frecuencia) con filtro Spearman top‑10, Lasso‑CV y top‑10 de importancia de ET, dentro de fold, 6 esquemas; estabilidad de coeficientes Ridge (correlación de Spearman entre vectores, consistencia de signo); **experimento de fuga** (selección dentro vs fuera del fold); **CUR24 vs 300 sorteos de igual composición (20 SAT + 2 TOPO + 2 GEOM)**; errores pareados por parcela y métricas por bloque/municipio. |
| **Resultado** | **Ranking:** correlación de Spearman entre esquemas: RCV ↔ GRID3 0.93, ↔ BAND6 0.90, ↔ GRID8 0.83, ↔ KM6 0.75, ↔ KM4 **0.15**, ↔ LMO **0.19** (KM4 ↔ LMO 0.90): hay dos regímenes de ranking —interpolación vs extrapolación regional— sin relación entre sí. **Semillas:** sd = 0.0018–0.0019 (ET); **hiperparámetros ET:** rango de RMSE 0.518–0.539 (RCV) y 0.649–0.661 (KM6, residual): menor que el IC de cualquier contraste. **Selección (objetivo = residuo del prior, RCV):** Jaccard medio filtro top‑10 0.27–0.49, Lasso 0.10–0.23, ET‑top‑10 0.20–0.39; 0–4 variables aparecen en ≥ 80 % de los folds según el método (las más frecuentes: `s2_evi_base`, `s2_lai_base`, `s2_ndvi_base`). Con objetivo *directo* la selección es más estable (Jaccard 0.49–0.79; Lasso 0.22–0.45) porque elige proxies de municipio. **Coeficientes Ridge:** correlación media entre vectores 0.71–0.85; consistencia de signo del top‑10 0.83–1.0, pero solo 37–57 % del total de variables mantiene signo en ≥ 90 % de los ajustes; solape del top‑10 (Jaccard) 0.29–0.62. **Fuga por selección (RCV, residual, k = 5 / 10 / 24):** fuera del fold 0.5000 / 0.5015 / 0.5031 vs dentro 0.5093 / 0.5104 / 0.5091 → **optimismo 0.009 / 0.009 / 0.006 t/ha**. **CUR24** 0.5148 vs 300 sorteos de igual composición (20 SAT + 2 TOPO + 2 GEOM) 0.5187 (p5–p95: 0.510–0.530): el 30 % de los sorteos la supera; sin sus 4 variables TOPO/GEOM, 0.5171 vs 0.5196 de 20 SAT al azar (32 % la supera). **Selección honesta** top‑24 por Spearman dentro de fold: 0.5091. **Por municipio (RCV, RMSE):** Chignahuapan 0.59 (sd intra 0.584), Calpulalpan 0.39 (sd 0.371), Singuilucan 0.48 (sd 0.43 → mejor residual ET+ALL 0.419), Almoloya 0.27; ningún modelo baja de ≈ 0.57 en Chignahuapan. **Por bloque (KM6):** el RMSE del prior va de 0.41 (Calpulalpan) a **1.07** (Singuilucan, municipio ausente del entrenamiento); ET‑SAT va de 0.28 (Almoloya) a **1.06** (Chignahuapan): los modelos con satélite ganan o pierden según el bloque. [05_*, 04_metricas_por_bloque, 04_metricas_por_municipio, 04_correlacion_ranking_entre_esquemas] |
| **Incertidumbre** | Un solo lote de semillas por esquema regional; Jaccard sobre solo 6–25 folds; CatBoost sin barrido de hiperparámetros; los errores por bloque en KM/LMO provienen de una partición (no hay IC por bloque). |
| **Interpretación** | Un ranking ganado en bloques **no predice** el ranking en la prueba oficial (ρ = 0.15–0.19 con KM4/LMO; 0.75–0.93 con KM6/GRID/BAND). La selección de variables sobre el residuo es **inestable** (salvo familia "base" de S2 y algo de topografía), así que cualquier lista corta debe presentarse como tentativa. El optimismo por fuga es del tamaño de las "ventajas" que se reportarían. CUR24 no tiene evidencia de superioridad sobre variables al azar. |
| **Decisión XAI** | Elegir el modelo con CV aleatoria *y* mostrar los esquemas espaciales como prueba de robustez regional, nunca al revés. Selección siempre dentro de fold. Reportar siempre IC pareado por clúster; ninguna diferencia menor a 0.02 en valor absoluto se declara mejora. |

### 2.4 Investigación 4 — ¿Aprenden variación dentro del municipio o solo diferencias geográficas? ¿Añade señal estable un residual sobre el prior?

| Campo | Contenido |
|---|---|
| **Hipótesis** | H4a: los modelos capturan variación *intra‑municipio*. H4b: un modelo residual sobre el prior municipal añade señal estable (entre repeticiones, esquemas y municipios). |
| **Experimento** | (i) Descomposición de las predicciones OOF: η² de las predicciones por municipio, correlación global e **intra‑municipio** (agrupada y por municipio), R² intra. (ii) Estabilidad del aporte residual: por repetición (10 reps RCV) y por municipio. (iii) **Permutación intra‑municipio de X con todo el pipeline** (Ridge: 15 conjuntos × 200 permutaciones; ET: 4 × 100; 2 reps CV; Holm sobre 19). (iv) **Prueba limpia dentro de un municipio** (CV de 5 folds solo con parcelas de Chignahuapan n = 71, Calpulalpan n = 30, Singuilucan n = 16; skill = 1 − MSE/MSE de la media municipal; permutación de *y* dentro del municipio, 200/100/60 permutaciones; Holm por municipio y modelo). (v) **Control del prior** (`06b`): el encogimiento k = 3 heredado del equipo es un hiperparámetro; se compara contra un prior con k elegido *dentro del fold* (error leave‑one‑out del entrenamiento; malla 0.05–10). |
| **Resultado** | **(i) Predicciones:** η² municipal de las predicciones: prior 0.999, ET‑ubicación 0.988, ET‑todo 0.941, ET‑satélite 0.686, Ridge‑satélite 0.691; r(predicción, y) global 0.79–0.81 (con ubicación/clima o prior) vs 0.64–0.67 (satélite solo). **R² intra‑municipio de las predicciones:** ET‑SAT −0.315, Ridge‑SAT −0.500, ET‑ALL −0.137, ET‑LOC −0.025, prior −0.046 (con r = −0.62 por construcción del leave‑one‑out), residual Ridge+CUR −0.010 y **residual Ridge+TOPO +0.018 (el único positivo)**; correlación intra en Chignahuapan: +0.258 (Ridge+TOPO), +0.179 (Ridge+CUR), +0.059 (ET‑LOC), −0.064 (ET‑SAT). **(ii) Estabilidad:** Ridge+TOPO mejora al prior en 10/10 reps (Δ de −0.027 a −0.013), Ridge+CUR 10/10 (−0.017 a −0.005), Ridge+CLIM 10/10, Ridge+SAT 9/10; ET 1–4/10; CatBoost **0/10** (+0.013 a +0.036). Por municipio (RCV): Ridge+SAT Chignahuapan 0.000 [−0.019, +0.018], Calpulalpan −0.012, Singuilucan −0.006, Almoloya −0.036 [−0.060, −0.007] (n = 10, único IC que excluye 0); ET+ALL en Chignahuapan **+0.027** y CatBoost+ALL **+0.061 [+0.024, +0.102]** (peor). **(iii) Permutación (Ridge, RCV; ΔRMSE observado vs media de la nula):** SAT −0.0090 vs −0.0040 (p = 0.16) · S2 −0.0111 vs −0.0043 (0.10) · CLIM −0.0145 vs −0.0075 (0.13) · CUR −0.0141 vs −0.0047 (0.12) · ALLNOLOC −0.0122 vs −0.0050 (0.13) · **TOPO −0.0206 vs −0.0023 (p = 0.005; Holm 0.095)** · GEOM 0.000 (p = 0.74); ET: todos p ≥ 0.17. **La nula de Ridge no es cero**: aun con X permutada, "mejora" −0.001 a −0.0075 sobre el prior (fig09, derecha). **(iv) Dentro de Chignahuapan:** skill negativo en casi todo (SAT −0.046, S2 −0.041, PL −0.084, CLIM −0.012, ALLNOLOC −0.031, ET‑SAT −0.138); positivos: **TOPO +0.063 [−0.033, +0.137] (p = 0.015; Holm 0.24)**, LOC +0.024 (p = 0.065), GEOM +0.007 (p = 0.040; Holm 0.60), ET‑TOPO +0.020 (p = 0.089). Calpulalpan (valores en {2.5, 3.0, 3.5}): todo ≤ 0 (mejor: Ridge T_PICO +0.007, p = 0.055). Singuilucan (n = 16): IC95 de hasta [−1.4, +0.2]; hay "positivos" (ET‑CUR +0.337, p = 0.016, Holm 0.10; Ridge T_PICO +0.319; Ridge LOC +0.107) pero con 22 conjuntos probados y n = 16 no son interpretables. **(v) Prior:** RMSE RCV con k = 0.5 → 0.5128; k = 1 → 0.5145; **k = 3 (equipo) → 0.5245**; k = 10 → 0.5649; con k elegido en cada fold (0.05–0.1 en los 50 folds ≈ media municipal sin encogimiento) → **0.5123** (Δ = −0.0122 [−0.029, −0.0005] vs k = 3; p = 0.043; ≈ 0.6 del ROPE). **Contra ese prior mejor calibrado** los residuales Ridge quedan en: CUR +0.0023, CLIM +0.0026, SAT +0.0040, LOC +0.0001, ALLNOLOC +0.0039 (todos IC ∋ 0) y **TOPO −0.0128 [−0.026, +0.004] (p = 0.13)**. [06_entre_vs_dentro_predicciones, 06_estabilidad_por_repeticion, 06_aporte_residual_por_municipio, 06_permutacion_intra_municipio, 06_dentro_de_municipio_cv_y_permutacion, 06_sensibilidad_encogimiento_prior, 06b_prior_shrink_control, fig09] |
| **Incertidumbre** | Chignahuapan es 51 % de las parcelas: las líneas de evidencia de TOPO (permutación agrupada, prueba dentro de Chignahuapan, SHAP/PFI, ρ de `topo_pend_media`) **no son independientes**, y se probaron 19 conjuntos en la permutación agrupada y 22 por municipio. Calpulalpan (3 valores) y Singuilucan (n = 16) no tienen potencia. 2 repeticiones de CV en las permutaciones; ET con 100 permutaciones. El k anidado se eligió con error leave‑one‑out en entrenamiento y cae en el borde bajo de la malla (0.05–0.1); el control `06b` solo cubre Ridge (ET/CatBoost no se rehicieron sobre el prior ajustado). |
| **Interpretación** | H4a: **los modelos aprenden geografía, no variación intra‑municipio**: las predicciones con satélite tienen η² municipal de 0.69 y R² intra negativo (−0.3 a −0.5); con ubicación/clima son casi un prior municipal (η² ≥ 0.94). H4b: lo que parecía señal residual estable (Ridge: −0.010 a −0.019 en 9–10 de 10 reps) es **en su mayor parte recalibración de un prior sobre‑encogido**: la nula con X permutada ya da −0.004 a −0.0075 y un prior con k anidado da −0.012 por sí solo; contra ese prior los residuales de CUR, CLIM, SAT, LOC y ALLNOLOC no aportan (0.000 a +0.004). **Solo TOPO sobrevive** (−0.013; mejora en 10/10 reps; p = 0.005 en la permutación agrupada y 0.015 dentro de Chignahuapan), pero su IC95 contra el prior ajustado incluye 0 y Holm lo deja en 0.095–0.24. Los árboles (ET/CatBoost) sobre el residuo empeoran (CatBoost 0/10 reps; +0.061 en Chignahuapan). |
| **Decisión XAI** | Capa 1 con prior de encogimiento **ajustado dentro del fold** (≈ media municipal; 0.512), no el k = 3 heredado. Capa 2: a lo sumo un término **lineal de topografía** (pendiente/elevación; opcionalmente `s2_ndvi_base`) tratado como hipótesis. **No** usar residuales con árboles/CatBoost ni satélite masivo como corrección. |


### 2.5 Investigación 5 — SHAP agrupado por familias y permutación de grupos

| Campo | Contenido |
|---|---|
| **Hipótesis** | H5a: la importancia agregada por familia refleja cuánta información aporta cada fuente. H5b: permutar grupos confirma esa ordenación. H5c: la correlación entre grupos no distorsiona la atribución. |
| **Experimento** | SHAP calculado **solo sobre parcelas fuera de entrenamiento** (TreeExplainer en ET; `ShapValues` nativo en CatBoost; SHAP lineal exacto en Ridge) en RCV5×10 (3 reps) y en GRID3B05 y KM6 (1 rep), para 8 configuraciones: ET/Ridge/CatBoost directos con todas las variables, ET sin ubicación, ET solo satélite, y los 3 modelos *sobre el residuo del prior* (el prior aparece como componente `MUN_prior`). Se agrega por familia, por familia temporal y por concepto espectral (vigor, residuo/labranza, agua/estrés, térmico) y se descompone la varianza SHAP de cada grupo en entre‑ e intra‑municipio. **Permutación de grupos** de tres tipos: global, **intra‑municipio** e **intra‑clúster** (25 clústeres espaciales). **Redundancia entre grupos:** R² de CV de los 1–3 primeros componentes de cada grupo desde (i) las coordenadas, (ii) todo lo demás sin coordenadas. Acuerdo de rangos de SHAP entre modelos (Spearman). |
| **Resultado** | **ET directo (RCV):** participación en la SHAP absoluta media: CLIM 58.5 %, LOC 23.3 %, S2 9.9 %, PL 4.6 %, TOPO 1.7 %, LS 1.0 %, GEOM 1.0 %; fracción de la varianza SHAP que es *entre* municipios: LOC 0.98, CLIM 0.98, PL 0.29, S2 0.17, TOPO 0.02. **Permutación global (ΔRMSE):** CLIM +0.300, LOC +0.071, S2 +0.009, resto ≈ 0 o negativo. **Permutación intra‑municipio:** CLIM −0.004, LOC −0.0004, S2 −0.0008, PL −0.0085, **TOPO +0.0019** (el único positivo en este modelo); intra‑clúster: todos ≈ 0 (−0.004 a +0.0002). Se repite con CatBoost (CLIM 49 %, S2 18 %; permutación global CLIM +0.25) y Ridge (S2 24 %, CLIM 49 %). **Modelos residuales (prior + modelo):** el prior municipal concentra 71–86 % de la SHAP absoluta; permutar un grupo casi nunca *empeora* el error: PL, S2, CLIM y GEOM suelen *mejorarlo* al permutarse (hasta −0.015 en PL → aportan ruido); **TOPO es el único grupo con ΔRMSE intra‑municipio positivo en los 6 modelos** (+0.0008 a +0.0037; el mayor positivo en 5 de 6; magnitud ≪ ROPE; en Ridge directo S2 da +0.0046). **Redundancia:** R² de CV de los PCs de CLIM desde las coordenadas = 0.55 (PC1–3) y **0.93 (PC1)**; S2 y PL son predecibles desde el resto (PC1: R² 0.97 y 0.94); S2 y PL tienen 10 y 8 componentes para el 90 % de la varianza de 124 y 48 variables. **Acuerdo de rangos SHAP:** entre modelos directos 0.52–1.00; entre un directo y un residual −0.14 a 0.60 (la atribución depende de la formulación). [07_shap_grupos, 07_permutacion_grupos, 07_redundancia_entre_grupos, 07_acuerdo_rangos_shap_entre_modelos, 07_top20_features_gemelos_correlacionados] |
| **Incertidumbre** | PFI con 3 repeticiones (RCV) o 1 (espaciales): sd entre reps de 0.025 (CLIM), 0.002 (LOC), 0.005 (S2, PL intra). SHAP describe la dependencia del *modelo*, no un mecanismo causal. Con grupos casi colineales (CLIM ≈ LOC; las 124 S2 con pocos componentes efectivos) el reparto de mérito **no es identificable**: el mismo conocimiento geográfico aparece como "clima" en un modelo y "ubicación" en otro. |
| **Interpretación** | La importancia de CLIM y LOC es importancia **entre municipios** (geografía/régimen), no explicación agronómica. S2 y PL varían *dentro* del municipio en la SHAP (sd intra 0.08 y 0.04 en el ET directo) pero **permutarlas dentro del municipio no daña el error**: el modelo las usa para ajustar sin que eso generalice. TOPO tiene SHAP pequeña pero su permutación intra‑municipio empeora el error en los 6 modelos (magnitud diminuta, ≪ ROPE): coherente con el candidato débil de la Investigación 4, no una prueba. **Advertencia:** cualquier SHAP por variable individual en S2/PL es inestable por redundancia; solo se interpretan agregados por grupo/concepto. |
| **Decisión XAI** | No reportar importancias de CLIM/LOC como mecanismos. Presentar la atribución en dos niveles: (1) **prior municipal** (explícito, no SHAP); (2) **correcciones intra‑municipio** con importancia por *permutación intra‑municipio* y SHAP del residuo, por grupo; y acompañar siempre con la tabla de redundancia entre grupos. |


### 2.6 Familias temporales y transferencia temporal

| Campo | Contenido |
|---|---|
| **Hipótesis** | H6: las familias temporales de SAT (pre‑siembra, vegetativo, pico, senescencia, fenología) aportan señal diferenciada de rendimiento; el modelo generaliza a otros ciclos. |
| **Experimento** | Ablación por familia temporal (T_PRE, T_VEG, T_PICO, T_SEN, T_PHTIM, T_PHLEV, T_HET) como residual sobre prior; **persistencia interanual sin etiquetas** de índices Sentinel‑2 2022–2025 (Spearman global e intra‑municipio, 6 pares de años). |
| **Resultado** | Ninguna familia temporal mejora al prior en RCV (Ridge: Δ de −0.008 a +0.011, todos equivalentes o indeterminados; ET: T_PICO +0.044 y T_PHTIM +0.034 *peor*, Holm 0.036 y 0.030; T_VEG +0.025 *peor* sin sobrevivir Holm, p = 0.39). Bajo KM6 T_PRE con Ridge Δ = −0.051 [−0.087, −0.024], Holm 0.008 (mejora regional, proxy de región). **Persistencia:** `s2_evi_base`, `s2_lai_base`, `s2_ndvi_base` ρ = 0.59–0.79 (intra‑municipio 0.60–0.74); `s2_evi_media_estacional` 0.09–0.37; `s2_evi_pico` 0.08–0.31. |
| **Incertidumbre** | **No se pudo ejecutar transferencia temporal**: solo existe rendimiento 2025 y los archivos de etiquetas no traen año; no hay validación leave‑one‑year‑out ni extrapolación de ciclo. La persistencia es una proxy descriptiva, no evidencia predictiva. |
| **Interpretación** | Las variables "base" se comportan como atributo casi estático de la parcela (suelo, residuo, cobertura previa); las de pico/estacionales cambian de año a año (clima y manejo anuales). Un modelo basado en variables de pico, ajustado con un solo año, no puede validarse temporalmente. |
| **Decisión XAI** | Separar en el modelo "atributos de parcela persistentes" (base, topografía, geometría) de "estado del ciclo" (pico, AUC, fenología), y exigir validación temporal antes de usar estas últimas con fines explicativos. |

---

## 3. Resultados por parcela y por bloque (entregable C)

- **OOF por repetición** (todas las configuraciones): `oof/main/<cfg>.csv.gz` (cfg, scheme, rep, fold, id, y, pred, n_train); BLOO: `oof/bloo/`; SHAP fuera de entrenamiento: `oof/shap/`.
- **OOF promedio y error por parcela**: `oof/oof_promedio_reps_todas_configs.csv.gz` (cfg, scheme, id, municipio, y, pred_oof_mean, abs_err, sq_err_mean_reps) y `results/04_errores_por_parcela_configs_clave.csv.gz`.
- **Métricas por bloque** (esquema × bloque × configuración: n, municipios, sesgo, RMSE, MAE): `results/04_metricas_por_bloque.csv`; **por municipio**: `results/04_metricas_por_municipio.csv`.
- **Contrastes pareados** por parcela y por clúster: `results/04_contrastes_pareados.csv`.

---

## 4. Figuras (entregable B)

| Figura | Contenido |
|---|---|
| `fig01_rendimiento_municipio_y_varianza.png` | Rendimiento por municipio y régimen de captura; descomposición de varianza |
| `fig02_mapa_parcelas.png` | Mapa: etiquetadas y ocultas intercaladas |
| `fig03_autocorrelacion_espacial.png` | Semivariograma, correlograma de Moran y potencia |
| `fig04_features_identifican_municipio.png` | Cuánto identifica cada familia el municipio/régimen |
| `fig05_rmse_por_esquema.png` | RMSE de 18 configuraciones clave × 6 esquemas (mismas particiones) |
| `fig06_aporte_incremental_sobre_prior.png` | ΔRMSE de cada familia sobre el prior municipal con IC95 por clúster y zona de equivalencia |
| `fig07_decaimiento_con_buffer.png` | RMSE al excluir vecinos en radios de 0–8 km |
| `fig09_senal_dentro_de_municipio.png` | Señal dentro de Chignahuapan y pruebas de permutación |
| `fig10_shap_y_permutacion_por_familia.png` | SHAP agrupado y permutación de grupos (global / intra‑municipio / intra‑clúster) |
| `fig11_pocas_variables_interpretables.png` | Evidencia para el conjunto de 5–10 variables |
| `fig12_persistencia_temporal.png` | Persistencia interanual de índices Sentinel‑2 |

---

## 5. La pregunta de la curiosidad

> *"Si tuvieras prohibido usar una red neuronal compleja y tuvieras que predecir el rendimiento con cinco a diez variables interpretables, ¿cuáles escogerías, por qué y qué evidencia experimental respalda esa elección?"*

**Respuesta corta.** Antes que cualquier variable de parcela, elegiría **el municipio como prior explícito** (media municipal con encogimiento pequeño elegido dentro del fold). Es lo único con evidencia fuerte: baja el RMSE de 0.854 a 0.524 con el k = 3 del equipo (Δ = −0.330, IC95 [−0.44, −0.22], Holm 0.002) y a **0.512** con k anidado. Encima añadiría, **como hipótesis y no como hallazgo**, un máximo de seis variables de parcela con corrección lineal pequeña, en este orden de apoyo:

| # | Variable | Por qué (lectura agronómica tentativa) | Evidencia experimental (intra‑municipio, dentro de fold) |
|---|---|---|---|
| 0 | **Prior municipal** (media con encogimiento ajustado en el fold) | Régimen de captura + sistema productivo de la zona | RMSE 0.524 (k = 3) / **0.512** (k anidado) vs 0.854 (global); η² = 0.69; en gran parte régimen de captura (§2.1) |
| 1 | `s2_ndvi_base` | Cobertura/vigor de línea base (ene‑may): suelo, residuo, cultivo previo; atributo casi estático | ρ parcial intra = **+0.30** (p = 0.0066; FWER 0.11; FDR 0.24); signo + en 4/4 municipios (+0.32, +0.08, +0.04, +0.71); elegida en **92 %** de los folds anidados; persistencia interanual ρ ≈ 0.6–0.7 |
| 2 | `topo_pend_media` | Pendiente: drenaje, erosión, heladas | ρ = +0.27 (p = 0.013; FDR 0.24); 3/4 signos; **TOPO** (4 variables) es la única familia con prueba de permutación intra‑municipio "casi" significativa: Δ = −0.0206 vs −0.0023 esperado, p = 0.005 (Holm 0.095) y skill +0.063 en Chignahuapan (p = 0.015) |
| 3 | `topo_elev_media` | Elevación (temperatura/heladas) | ρ = +0.17 (p = 0.13); seleccionada en 56 % de los folds; PFI intra‑municipio positiva en los 6 modelos (§2.5) |
| 4 | `geo_area_ha` | Tamaño de parcela (manejo/heterogeneidad) | ρ = −0.25 (p = 0.019; FDR 0.24); 3/4 signos (−0.27 Chignahuapan, −0.44 Almoloya, ≈ 0 Calpulalpan y Singuilucan) |
| 5 | `s2_msi_pico` | Estrés hídrico en el pico | ρ = +0.20 (p = 0.067); elegida en 68 % de los folds; **signo opuesto en Singuilucan (−0.41)** |
| 6 | `s2_ndti_pre` | Residuo/labranza antes de siembra | ρ = +0.19 (p = 0.095); elegida en 58 % de los folds; 2/4 signos |

**Lo que dicen los experimentos sobre esa lista** [08_*, fig11]:

- **Selección anidada** (la lista se elige solo con el entrenamiento de cada fold; Ridge sobre el residuo del prior k = 3) con 5 variables: RMSE 0.5161 vs prior 0.5245, Δ = **−0.0083 [−0.037, +0.019]**, p = 0.56 → *no distinguible* del prior; con 10: −0.0027; con k elegido por CV interna: −0.0029. Con bloques 3 km: −0.0084 [−0.040, +0.021]. **Frente al prior con k anidado (0.5123) la selección anidada de 5 variables (0.5161) ni siquiera mejora el RMSE puntual** (la "ganancia" frente a k = 3 era en gran parte recalibración, §2.4).
- **Conjunto a priori** (10 variables fijadas por criterio agronómico antes de la selección: `s2_ndvi_peak`, `s2_ndvi_auc`, `s2_lai_pico`, `s2_msi_pico`, `s2_crc_pre`, `clim_prec_temp`, `clim_tmin_minmes`, `topo_elev_media`, `topo_pend_media`, `geo_area_ha`): −0.0025 [−0.031, +0.034]; su versión de 5 variables **empeora** (+0.0057 en RCV; +0.0130 [+0.004, +0.023] en bloques 3 km). *Nota de honestidad:* esta lista se fijó antes de correr la selección pero **después** de haber visto resultados de ablaciones; no es un pre‑registro estricto.
- **Evidencia univariada intra‑municipio:** de 37 candidatas, ninguna pasa FWER (mínimo 0.11; p95 del máximo de abs(ρ) bajo H0 = 0.33) y ninguna FDR < 0.10 (mínimo q = 0.24). Con n = 138 y 4 municipios efectivos, la potencia para abs(ρ) ≈ 0.25 es baja (≈ 20 % con aproximación normal, SE ≈ 0.086, umbral FWER 0.33).
- **Bajo bloques regionales (KM6, una sola partición)** la selección anidada sí mejora: −0.039 [−0.074, −0.012], p = 0.005. Pero eso es transferencia regional (las variables de línea base distinguen región/régimen) y no el reto.
- **Qué *no* incluiría:** coordenadas y clima como "variables explicativas": identifican municipio (§2.2) y las 25 celdas climáticas son 25 valores distintos para todo el set (muestra efectiva ≈ 25, no 138).

**Mi respuesta honesta:** *prior municipal + (`s2_ndvi_base`, `topo_pend_media`, `topo_elev_media`, `geo_area_ha`, `s2_msi_pico`, `s2_ndti_pre`)*, entendiendo que la ganancia esperada sobre un prior bien calibrado es ≈ 0.01 t/ha (≈ 2 %; solo `topo_*` la sostiene: −0.013 [−0.026, +0.004]), que **no es distinguible de cero con los datos actuales**, y que si la regla de adopción de §6 no se cumple, la mejor predicción defendible es **solo el prior municipal**.

---

## 6. Recomendación de arquitectura XAI (entregable D)

**Arquitectura de dos capas explícitas**

```
ŷ(parcela) =  μ̂_municipio (media con encogimiento k ajustado dentro del fold ≈ 0.05–0.1)   ← Capa 1: prior auditable
           +  Σ_j β_j · z_j(parcela),  j ∈ lista corta (≤ 5–6), Ridge, signos esperados   ← Capa 2: corrección intra‑municipio (hipótesis)
```

**Reglas de uso**

1. **Capa 1 siempre.** Es la única fuente con evidencia fuerte. El k = 3 heredado sobre‑encoge (0.524); con k elegido en el fold el RMSE es 0.512. Con k ≈ 0 los municipios con 1–3 parcelas quedan casi sin contracción (en el reto casi no hay parcelas ocultas ahí: 5 de 59 en municipios con n < 10), por lo que conviene un nivel intermedio de régimen de captura **si agronomía confirma ese metadato** (§8, P1) en lugar de derivarlo de los propios rendimientos. Para municipios *nuevos* (LMO) el prior cae a la media global (RMSE 1.15). Los modelos directos con ubicación/clima/satélite sí recuperan parte del nivel (mejores LMO: Ridge ALL‑TOPO 0.661, Ridge ALL 0.665, ET‑LOC 0.70), pero lo hacen **identificando región/régimen**, con una sola partición y solo 6 municipios evaluables, y fallan cuando el vecino más cercano es de otro régimen (Singuilucan: 1.31 con ET‑LOC vs 1.07 con la media global).
2. **Capa 2 solo se activa si cumple las tres condiciones** (hoy ningún candidato las cumple completas): (a) ΔRMSE frente al prior con IC95 por clúster espacial < 0 en RCV y en BLOO r ≥ 0.5 km; (b) prueba de permutación intra‑municipio con p < 0.05 **después de Holm**; (c) signo coherente en ≥ 3 de los 4 municipios grandes y mecanismo agronómico plausible confirmado por el equipo.
3. **Atribución:** capa 1 = tabla de medias por municipio (no SHAP); capa 2 = coeficientes estandarizados con IC bootstrap por clúster y permutación *intra‑municipio* por grupo. No reportar importancias de clima/ubicación como mecanismos; no interpretar variables individuales de S2/PL (redundancia).
4. **Validación:** decisión con CV aleatoria repetida (≥ 10 reps) e IC pareado por clúster; esquemas espaciales/LMO/BLOO solo como prueba de robustez regional; **transferencia temporal pendiente** (§2.6, §10). Selección de variables siempre dentro del fold.
5. **Incertidumbre predictiva:** mientras no haya señal intra‑municipio, el piso de error es la desviación intra‑municipal (Chignahuapan 0.58, Singuilucan 0.43, Calpulalpan 0.37, Almoloya 0.19); reportar intervalos por municipio, no un único RMSE global.

**Hipótesis que hay que descartar (con la evidencia que lo respalda)**

| # | Hipótesis | Estado | Evidencia |
|---|---|---|---|
| D1 | "El satélite predice rendimiento mejor que la ubicación/municipio" | **Descartada** | ET‑SAT 0.641 vs ET‑LOC 0.519 (Δ +0.122 [+0.075, +0.173], Holm 0.004); Ridge +0.135; SAT dado todo lo demás: 0 mejoras en 16 contrastes |
| D2 | "ET/CatBoost extraen más que Ridge" | **Descartada** | Ningún residual (Ridge, ET, CatBoost) supera al prior de forma distinguible en RCV; CatBoost residual +0.027 [−0.019, +0.066] (no mejor); sensibilidad de hiperparámetros ET 0.518–0.539 |
| D3 | "Hay autocorrelación espacial de residuos explotable a ≥ 1 km" | **Descartada para la escala útil** (con potencia limitada) | Moran intra I = −0.008 (p = 0.77); kNN‑5 ≈ prior municipal (Δ −0.002 [−0.048, +0.045]); BLOO casi plano hasta 1 km (prior 0.520 → 0.524; +0.027 a 2 km). Queda abierto < 500 m |
| D4 | "El clima explica diferencias de rendimiento dentro del municipio" | **Descartada** | Celda climática dado municipio ΔR² = 0.024 (p = 0.47); permutar CLIM intra‑municipio ΔRMSE −0.004; CLIM ≈ ubicación (R² de PC1 0.93) |
| D5 | "La lista curada CUR24 es mejor que variables al azar" | **Descartada** | 0.5148 vs 0.5187 (sorteos de igual composición); 30 % de los sorteos la supera |
| D6 | "Las familias temporales T_VEG/T_PICO/T_PHTIM aportan (con ET)" | **Descartada** | Δ = +0.025 / +0.044 / +0.034 (peor; Holm 0.39 / 0.036 / 0.030) |
| D7 | "Ganar en bloques espaciales implica ganar en el reto" | **Descartada** | ρ de ranking RCV ↔ KM4 = 0.15, ↔ LMO = 0.19 |
| D8 | "SHAP de clima/ubicación = explicación agronómica" | **Descartada** | 98 % de su varianza es entre municipios; permutación intra ≈ 0; colinealidad CLIM–LOC |
| D9 | "Seleccionar variables con todos los datos y luego validar es inocuo" | **Descartada** | Optimismo 0.006–0.012 t/ha (§2.3) |
| D10 | "La diferencia entre municipios es puramente agronómica" | **No se puede afirmar ni descartar** | Confundida con régimen de captura (ΔR² de municipio dado régimen = 0.011) |

---

## 7. Reproducibilidad y conservación (qué se guardó para cada decisión)

| Qué | Dónde |
|---|---|
| Scripts (orden de ejecución) | `scripts/00_audit.py` → `01_make_splits.py` → `02_variance_spatial.py` → `03_run_cv_grid.py --group main` y `--group bloo` → `02b_residual_autocorr.py` → `04_ablation_analysis.py` → `04b_bloo_analysis.py` → `05_stability.py` → `05b_cur24_control.py` → `06_within_municipality.py` → `06b_prior_shrink_control.py` → `07_shap_groups.py` → `08_sparse_interpretable.py` → `09_temporal_persistence.py` → `10_figures.py` → `99_verify_integrity.py`; orquestador `run_all.sh` |
| Configuración / semillas | `config/splits_config.json` (semilla global 20261013; semillas RCV 1000–1009; malla 2000+rep+lado; k‑means n_init = 10, rs = 0; shrink = 3.0; k kNN = 5); `scripts/grid.py` (lista de 136 + 19 configuraciones); semillas de modelos por (rep, fold) |
| Particiones | `splits/splits_long.csv` (18 659 filas; SHA‑256 `56c85a25a3cd31a8…`), descriptor en `results/01_descriptor_esquemas_validacion.csv` |
| Predicciones fuera de muestra | `oof/main/*.csv.gz` (136), `oof/bloo/*.csv.gz` (19), `oof/shap/*.csv.gz`, `oof/oof_promedio_reps_todas_configs.csv.gz` |
| Errores por parcela / bloque / municipio | `results/04_errores_por_parcela_configs_clave.csv.gz`, `04_metricas_por_bloque.csv`, `04_metricas_por_municipio.csv`, `04_contrastes_pareados.csv` (1 120 contrastes con IC pareados y por clúster, Wilcoxon, Holm) |
| Integridad | `results/00_original_tree_sha256_BEFORE.txt`, `00_zip_sha256.txt`, `00_input_copies_sha256.txt`, `99_verificacion_integridad.csv` |
| Entorno | Python 3.13.16 · numpy 2.5.3 · scipy 1.18.1 · pandas 3.0.5 · scikit‑learn 1.9.1 · CatBoost 1.2.10 · shap 0.53.0 · statsmodels 0.15.0 · matplotlib 3.11.2 |
| Costos de cómputo (2 núcleos) | malla principal 4 171 s; BLOO 2 361 s; permutaciones intra‑municipio (06) ≈ 1 h |

`scripts/common.py` fija `EXP=/home/claude/experiments`; si se mueve el directorio hay que cambiar esa constante. Cada script se ejecutó individualmente en esta sesión; **`run_all.sh` no se corrió completo de una sola vez en un entorno limpio**.

---

## 8. Preguntas pendientes para el equipo agronómico (entregable E)

1. **Origen y resolución del rendimiento.** ¿Cómo se registró en Tlaxcala (Calpulalpan, Nanacamilpa) y en Hidalgo (Singuilucan, Emiliano Zapata, Apan)? ¿Estimación del productor, redondeo a 0.5 t/ha, pesaje en báscula? ¿Y en Chignahuapan, Almoloya, Cuautepec y Tepeapulco (valores continuos)? **Esto decide si el contraste 4.5 vs 3.0 t/ha es real o instrumental.**
2. **Sistema productivo por municipio:** cebada maltera vs forrajera, variedad, fecha de siembra, temporal vs riego, fertilización y paquete tecnológico. ¿Son dos sistemas distintos?
3. **¿Qué representan `s2_*_base` (línea base ene‑may)?** ¿Cobertura/rastrojo/cultivo previo, color y humedad del suelo? ¿Por qué persisten entre años (ρ 0.6–0.8)? ¿Es plausible un efecto positivo en el rendimiento dentro del municipio?
4. **Topografía:** ¿hay un mecanismo (heladas, drenaje, erosión, temperatura) por el que pendiente y elevación expliquen diferencias dentro de Chignahuapan? ¿Cuál es el rango real de elevación del municipio?
5. **Área de parcela (ρ = −0.25):** ¿la relación negativa área–rendimiento es manejo (parcelas grandes menos atendidas), heterogeneidad, o forma de estimar el rendimiento?
6. **¿Parcelas contiguas comparten productor o manejo?** (indicio de estructura < 500 m).
7. **Rendimientos de 2022–2024** de las mismas parcelas: sin ellos no se puede validar transferencia temporal. ¿Existen?
8. **¿Qué ocurrió en 2025?** Sequía, heladas, granizo, fecha de cosecha vs la ventana de senescencia de Sentinel‑2.
9. **¿Los 59 polígonos ocultos mantienen la misma mezcla de municipios y régimen de captura?** Asumimos que sí (29–32 % por municipio grande).
10. **Variables no medidas pero probablemente más informativas** que el sensor remoto: variedad, fecha de siembra, dosis de fertilización, análisis de suelo, cultivo previo, riego. ¿Se pueden conseguir para al menos Chignahuapan?
11. **Polígonos vs unidades de manejo:** ¿un polígono = un productor/lote de manejo único?

---

## 9. Resumen final: qué sabemos, qué sospechamos y qué no podemos afirmar

**Lo que sabemos (verificado, con incertidumbre cuantificada)**

- La prueba oficial es interpolación aleatoria dentro de municipio; las ocultas están intercaladas con las etiquetadas. [AUC adversaria 0.44–0.50; 350 m]
- El municipio explica ≈ 69 % de la varianza (η² = 0.686 [0.57, 0.80]); un prior municipal encogido baja el RMSE de 0.854 a 0.524 (Δ = −0.330 [−0.44, −0.22]).
- Ese efecto municipal es un contraste de dos regímenes de captura del dato: régimen R² = 0.675; municipio dado régimen +0.011 (p = 0.72).
- Bajo CV aleatoria, ningún modelo con satélite/clima/topografía/geometría supera de forma distinguible al prior municipal (todos los IC95 por clúster contienen 0 o caen dentro de ±0.02; 0 mejoras relevantes de 46 contrastes).
- El prior municipal del equipo (k = 3) está sobre‑encogido: con k elegido dentro del fold el RMSE baja a 0.512 (−0.012 [−0.029, −0.0005]) y contra ese prior las correcciones de CUR/CLIM/SAT/LOC/ALLNOLOC no aportan (0.000 a +0.004).
- El satélite solo es significativamente peor que la ubicación sola (Δ = +0.12 a +0.14, Holm 0.004) y no añade nada dado todo lo demás (0/16 mejoras).
- Satélite, clima y ubicación identifican municipio (exactitud 83–100 %); su "importancia" SHAP es importancia entre municipios (98 % de la varianza SHAP de CLIM/LOC).
- No hay autocorrelación espacial intra‑municipio detectable a ≥ 1 km (Moran I = −0.008, p = 0.77) —con potencia limitada—.
- Elegir variables fuera del fold infla el desempeño en 0.006–0.012 t/ha; CUR24 no supera a variables al azar de igual composición; algunas familias temporales con ET empeoran el modelo.
- Los rankings de modelos en bloques regionales/LMO no se relacionan con los de CV aleatoria (ρ = 0.15–0.19 para KM4/LMO).
- Integridad: árbol original intacto; cero uso de las etiquetas ocultas y de sus predicciones.

**Lo que sospechamos (consistente en varias pruebas, pero sin pasar corrección por multiplicidad)**

- Topografía (pendiente/elevación) y `s2_ndvi_base` llevan una señal intra‑municipio pequeña (≈ 0.01 t/ha de RMSE; ≈ 2 %): permutación TOPO p = 0.005 (Holm 0.095), −0.013 contra el prior ajustado (IC95 [−0.026, +0.004]), mejora en 10/10 reps, PFI intra‑municipio positiva en los 6 modelos, `s2_ndvi_base` ρ = +0.30 (FWER 0.11). Las evidencias de TOPO comparten datos (Chignahuapan = 51 %).
- Existe una tendencia suave de escala de kilómetros dentro de Chignahuapan (residuos autocorrelacionados cuando se retiran bloques completos; el prior se degrada con radios ≥ 4 km) y quizá un efecto de vecindad < 500 m (I = 0.33 en el primer anillo).
- Una parte del "efecto municipio" puede ser de medición (resolución del dato) y no agronómica.
- Las variables de línea base (`ndvi/evi/lai_base`) son atributos casi estáticos de la parcela; las de pico, atributos del ciclo.
- Un modelo con más información agronómica no medida (variedad, siembra, fertilización, suelo) superaría a cualquier combinación de las variables remotas actuales.

**Lo que no podemos afirmar**

- Que el satélite, el clima o cualquier variable "expliquen" el rendimiento intra‑municipio: no hay señal confirmada.
- Que los modelos generalicen a otro ciclo agrícola: **no se pudo ejecutar transferencia temporal**.
- Que los modelos generalicen de forma fiable a municipios nuevos (LMO, una partición: 0.66–0.70 en el mejor caso vs 1.15 de la media global, a costa de identificar región/régimen; ET‑LOC falla en Singuilucan, 1.31 vs 1.07).
- Que el efecto municipal sea agronómico y no instrumental.
- Que no exista estructura espacial débil o de corta distancia (potencia 28–65 % para estructuras moderadas con n = 71).
- Que una lista de 5–10 variables sea "la correcta": las listas son hipótesis; ninguna variable pasa FWER/FDR.
- Que un ligero empate de RMSE (|Δ| < 0.02) implique equivalencia o superioridad.

---

## 10. Limitaciones y pruebas que no pude ejecutar

1. **Transferencia temporal: no ejecutable.** Un solo ciclo con rendimiento (2025); las etiquetas no traen año; no hay validación leave‑one‑year‑out. Solo se hizo una proxy sin etiquetas (persistencia interanual de índices).
2. **El ZIP no incluye salidas del baseline** (`reports/` y `figures/` vacías): no se compararon OOF originales con los míos; solo las cifras del README (coinciden).
3. **Tamaño muestral:** n = 138; solo 4 municipios con n ≥ 10 (Chignahuapan = 51 % de las parcelas, 77 % de la suma de cuadrados intra). Los IC por clúster son anchos (±0.025–0.04) y casi todo cae en "indeterminado".
4. **Esquemas con una sola partición** (KM4/6/8, BAND6, LMO, BLOO): sin varianza entre repeticiones y dependientes del diseño; los residuos autocorrelacionados bajo bloques son en parte artefacto de retirar bloques completos.
5. **Hiperparámetros fijos** para ET/CatBoost; sensibilidad medida solo para ET (rango 0.518–0.539); CatBoost solo en 18 configuraciones. El **prior de referencia (k = 3) no se ajustó** en la malla principal: la corrección con k anidado (`06b`) solo se hizo para Ridge; los contrastes de 04 frente al prior k = 3 sobrestiman la ganancia de los residuales (≈ 0.01 t/ha).
6. **Pruebas de permutación** con 200 (Ridge) o 100 (ET) permutaciones y 2 repeticiones de CV: resolución de p de 0.005–0.01; las pruebas intra‑municipio de Calpulalpan y Singuilucan usan 60 permutaciones en ET.
7. **Multiplicidad:** se corrigió con Holm/FWER/FDR *dentro de familias de contrastes*, pero hay ≈ 1 100 contrastes pareados y varias familias de pruebas; los p ≈ 0.01 aislados deben leerse como exploratorios. El conjunto "a priori" se fijó tras ver ablaciones (no es un pre‑registro).
8. **Elecciones arbitrarias:** 25 microclústeres para el bootstrap espacial; ROPE de ±0.02 t/ha; shrink = 3 (la sensibilidad del encogimiento se evaluó en `06_sensibilidad_encogimiento_prior.csv`).
9. **SHAP/PFI** con 3 repeticiones (RCV) o 1 (espaciales); SHAP describe el modelo, no mecanismos; con grupos colineales (CLIM–LOC; 124 variables S2 con ≈ 10 componentes) la atribución no es identificable.
10. **Un archivo (`datos_reorganizados_todas_observaciones.csv`) no tiene ID de parcela**: no pudo verificarse por ID que no incluya rendimientos ocultos (la columna no se usó).
11. **`run_all.sh` no se ejecutó completo en un entorno limpio** (cada script se corrió por separado).

