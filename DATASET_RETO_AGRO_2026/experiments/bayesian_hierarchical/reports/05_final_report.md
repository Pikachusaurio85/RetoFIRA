# Informe final — ¿Mejora la información ambiental el prior municipal?

**Proyecto:** Reto AgroCebada 2026 — experimento bayesiano jerárquico independiente
**Ubicación:** `experiments/bayesian_hierarchical/` (no se modificó ningún módulo oficial)
**Fecha:** 2026-10-19

## 1. Pregunta y respuesta corta

*¿La información ambiental permite mejorar de forma robusta el prior municipal, o el límite está
fijado por la cantidad/calidad de etiquetas?*

**Respuesta:** El límite está fijado por las etiquetas. Ningún conjunto de covariables ambientales
mejora de forma robusta al baseline municipal; el modelo jerárquico **solo-municipio** iguala al
prior municipal oficial. Conclusión **Escenario C**.

### Marco de decisión
- **A — Adoptar covariables ambientales:** mejora robusta y consistente → **rechazado**.
- **B — Adopción condicional:** mejora solo en un subconjunto/municipios grandes → **no observado**.
- **C — Sin mejora robusta (límite = etiquetas):** mantener el prior municipal; documentar que lo
  ambiental no aporta → **ELEGIDO**.
- **D — Indeterminado/inviable:** **no aplica** (todos los modelos convergieron).

## 2. Evidencia principal (RCV5x10, 50 folds, NUTS 2000×4)

| Modelo | RMSE | MAE | CRPS | cov80 | cov95 | R-hat máx | div |
|---|---|---|---|---|---|---|---|
| **B0** (solo municipio) | **0.5109** | 0.3402 | 0.2696 | 0.878 | 0.940 | 1.006 | 31 |
| B3 (3 ambientales) | 0.5125 | 0.3597 | 0.2731 | 0.867 | 0.958 | 1.005 | 51 |
| B5 (5 conceptos XAI) | 0.5184 | 0.3614 | 0.2756 | 0.860 | 0.956 | 1.006 | 60 |
| PRIOR municipal oficial | 0.5123 | **0.3364** | — | — | — | — | — |

1. **B0 es el mejor en RMSE**, pero su ventaja sobre el prior oficial **no es robusta**
   (IC95 % de la diferencia de error cuadrático = [−0.0044, +0.0014], cruza 0). En MAE el prior
   oficial gana. → **empate técnico con el baseline ya desplegado**.
2. **Ningún set ambiental mejora a B0.** B3 es indistinguible; **B5 es significativamente peor**
   (IC de Δerror² = [+0.0019, +0.0138], todo > 0).
3. La degradación es **monótona** con el número de variables (B0 < B3 < B5 < B8 < B12 < B20):
   firma clásica de sobreajuste sin señal real.
4. El patrón es idéntico en las repeticiones de **evaluación limpia (5–9)** y en las de selección,
   descartando que sea un artefacto de selección.

## 3. Mecanismo (por qué no funciona)

- **El municipio domina:** ICC = 0.695 [0.464, 0.878] (70 % de la varianza es entre municipios).
  Añadir covariables apenas mueve el ICC: no absorben estructura municipal.
- **Efectos ambientales nulos:** el único coeficiente con soporte es la **elevación**
  (~+0.10–0.14 t/ha por DE, P>0 ≈ 0.98–0.99). El resto incluye 0. Su magnitud es ~5× menor que el
  spread municipal (tau ≈ 0.8 t/ha).
- **Test intra-municipio:** la correlación entre el ajuste ambiental (`B5 − B0`) y el residuo
  *dentro* de municipio es **+0.05** agrupada y con signos inconsistentes por municipio → las
  covariables **no** contienen señal predictiva intra-municipio.
- **Lo que sí aporta el enfoque jerárquico:** pooling parcial estable para municipios con n=1
  (λ≈0.72; Nanacamilpa pasa de −1.46 a −0.86 t/ha). Esto es una mejora de *robustez*, no de RMSE.

## 4. Recomendación

1. **Mantener el prior municipal oficial** (`prior_municipal_calibrado_v1`). Es tan bueno como lo
   mejor que encontró este estudio; no procede reemplazarlo por un modelo con covariables.
2. Si se quisiera robustez ante municipios pequeños, el modelo **B0 jerárquico** es un sustituto
   equivalente y más principiado, pero **no** supera al oficial de forma robusta.
3. **La palanca real es la recolección de etiquetas**, sobre todo en los 5 municipios con n<10
   (Cuautepec, Emiliano Zapata, Nanacamilpa, Apan, Tepeapulco).
4. Las covariables ambientales **no** deben promocionarse como predictoras de rendimiento a nivel
   de predio con los datos actuales.

## 5. Limitaciones

- n = 138 etiquetas, repartidas en 9 municipios; 5 con n<10.
- Los conjuntos ambientales usan features pre-ingenierizadas y 5 conceptos XAI; podría existir otra
  parametrización con señal, pero el patrón monótono y el test intra-municipio sugieren que no.
- Meta-redondeo de etiquetas (14.5 % múltiplos de 0.5) añade ruido irreducible.
- Las conclusiones son sobre predicción *out-of-fold*; no implican que las variables ambientales no
  tengan relación agronómica con el rendimiento.

## 6. Reproducibilidad

Ver `README.md`. Todo el pipeline es reanudable, con semilla fija `SEED=20260219` y checkpoints en
`results/`.
