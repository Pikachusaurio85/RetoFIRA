# Matriz de Comparabilidad de Modelos — AgroCebada 2026

> **Regla fundamental**: Solo son directamente comparables los modelos evaluados sobre **las mismas particiones** (mismos folds, mismas repeticiones, mismas parcelas test). Cualquier combinación distinta requiere una figura/tabla separada con advertencia explícita.

---

## 1. Grupos de Comparabilidad

| Grupo | Esquema | Modelos | Base común | Comparables |
|-------|---------|---------|------------|-------------|
| **A** | **RCV5x10** (50 folds, 138 parcelas × 10 reps) | MEDIA_GLOBAL, PRIOR, PRIOR_K3, RIDGE148, HIBRIDO_TOPO, **B0, B3, B5** | Mismas 50 particiones, mismas 1380 predicciones OOF | ✅ **SÍ** — Gráfica principal |
| **B** | KM6 (6 clusters) | MEDIA_GLOBAL, PRIOR, PRIOR_K3, RIDGE148, HIBRIDO_TOPO | 6 clusters, 1 rep | ❌ No con A |
| **C** | LMO (Leave-Municipio-Out) | MEDIA_GLOBAL, PRIOR, PRIOR_K3, RIDGE148, HIBRIDO_TOPO | 9 municipios, 1 rep | ❌ No con A ni B |
| **D** | RCV5_S2026 | MEDIA_GLOBAL, PRIOR, PRIOR_K3, RIDGE148, HIBRIDO_TOPO | 5 folds × 1 rep (distinto a RCV5x10) | ❌ No con A |
| **E** | Bayesiano Screening | B0, B3, B5, B8, B12, B20 | Reps 0-4 (25 folds), NUTS 500×2 | ❌ No con A (diferente n_reps, NUTS corto, etiqueta EXPLORATORIO) |
| **F** | Desarrollo Aleatoria | ExtraTrees/RF/Ridge/Lineal (8/24/148 vars) | 5-fold aleatorio, 138 parcelas | ❌ No con ningún oficial |
| **G** | Desarrollo Espacial | ExtraTrees/RF/Ridge/Lineal (8/24/148 vars) | 6 bloques espaciales, 138 parcelas | ❌ No con F ni oficiales |
| **H** | ProtoDash | — | — | **No es predictor** — Figura independiente obligatoria |

---

## 2. Matriz de Comparabilidad Par-a-Par

| | A (RCV5x10) | B (KM6) | C (LMO) | D (RCV5_S2026) | E (Bayes Screen) | F (Dev Aleat) | G (Dev Espac) |
|---|---|---|---|---|---|---|---|
| **A (RCV5x10)** | ✅ | ❌ | ❌ | ❌ | ⚠️* | ❌ | ❌ |
| **B (KM6)** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **C (LMO)** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **D (RCV5_S2026)** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **E (Bayes Screen)** | ⚠️* | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **F (Dev Aleat)** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **G (Dev Espac)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

⚠️* Grupo E (screening) usa **subconjunto de particiones** (reps 0-4) y **NUTS corto**; no es comparable directo con Grupo A aunque comparta esquema RCV5x10. Se muestra en figura separada (progresión B0–B20).

---

## 3. Recomendación de Figuras para el Anexo (4 cuartillas)

### **Figura 1 — Comparación Homogénea RCV5x10 (Grupo A) — PÁGINA 1**
- **Tipo**: Barras agrupadas RMSE / MAE con IC95 bootstrap pareado vs PRIOR
- **Modelos**: MEDIA_GLOBAL, PRIOR, PRIOR_K3, RIDGE148, HIBRIDO_TOPO, **B0, B3, B5**
- **Eje X**: Modelo (ordenado por RMSE)
- **Color**: Familia (Oficial vs Bayesiano)
- **Anotación**: ★ = significativamente distinto de PRIOR (p<0.05, Wilcoxon pareado n=138)
- **Nota al pie**: "50 folds RCV5x10, 138 parcelas únicas, 10 repeticiones. IC95 bootstrap percentile (5000) sobre diferencias de pérdida agregadas por parcela. Wilcoxon signed-rank, dos colas."

### **Figura 2 — Progresión Experimental Bayesiana B0–B20 — PÁGINA 2**
- **Panel izquierdo (Screening EXPLORATORIO)**: RMSE/MAE ± SD para B0, B3, B5, B8, B12, B20 (reps 0-4, NUTS 500×2)
- **Panel derecho (Finalistas)**: RMSE/MAE para B0, B3, B5 en tres bloques: Selección (reps 0-4), Evaluación (reps 5-9), Todas (50 folds)
- **Etiquetas claras**: "EXPLORATORIO" / "EVALUACIÓN FINAL" en cada panel
- **Línea de referencia**: PRIOR oficial (RMSE=0.5123, MAE=0.3364)
- **Nota**: "Screening: NUTS 500+500, 2 cadenas. Finalistas: NUTS 2000+1000, 4 cadenas. Mismas 50 particiones RCV5x10."

### **Figura 3 — ProtoDash: 5 Prototipos Oficiales + Ejemplo Local — PÁGINA 3**
- **Tabla superior**: 5 prototipos oficiales (AGC_076, AGC_133, AGC_162, AGC_074, AGC_170) con pesos normalizados, municipio, partición
- **Panel inferior**: Ejemplo 1 parcela (AGC_001) con sus 5 prototipos locales, distancia RBF, similitud, top-3 componentes diferenciadores
- **Leyenda**: "ProtoDash optimiza MMD (representatividad distributiva), NO importancia de variables ni contribución a predicción. Pesos = representatividad global; Similitud local = RBF en espacio compacto 5D."

### **Tablas anexas (Página 4 o material suplementario)**
- Tabla 1: `modelos_comparativa.csv` completo (todos los modelos ejecutados)
- Tabla 2: Diagnósticos inferenciales bayesianos (R-hat, ESS, divergencias, ICC, pooling)
- Tabla 3: Coeficientes ambientales posteriores (B3, B5) con IC95 y P(β>0)
- Tabla 4: Test intra-municipal (correlación delta_B5 vs residuo por municipio)

---

## 4. Modelos que DEBEN aparecer en figura separada

| Modelo/Grupo | Razón | Ubicación recomendada |
|--------------|-------|----------------------|
| Oficiales KM6 / LMO / RCV5_S2026 | Distinto esquema de validación | Tabla histórica (no gráfica principal) |
| Bayesianos Screening B8, B12, B20 | Solo exploratorio, NUTS corto | Panel izquierdo Figura 2 (etiquetado EXPLORATORIO) |
| Desarrollo (ExtraTrees, RF, Ridge, Lineal) | Validación aleatoria/espacial distinta | Tabla histórica |
| ProtoDash | No predice rendimiento | Figura 3 dedicada |

---

## 5. Resumen de Decisiones de Visualización

| Decisión | Justificación |
|------------|---------------|
| Solo Grupo A en Figura 1 | Único con particiones idénticas y OOF completo |
| Separar Screening vs Finalistas en Figura 2 | Diferente presupuesto computacional y etiqueta exploratoria |
| ProtoDash en figura propia | Naturaleza cualitativa distinta (explicativa, no predictiva) |
| No mezclar RMSE/MSE en un mismo eje | Unidades diferentes; Figura 1 usa RMSE y MAE (ambas en t/ha) |
| IC95 bootstrap pareado, no IC del RMSE | Intervalos sobre la **diferencia de pérdidas**, no sobre el RMSE puntual |

---

## 6. Archivos de Datos para Figuras (ya generados)

| Figura | Archivo CSV | Descripción |
|--------|-------------|-------------|
| Figura 1 | `pareados_corregidos.csv` + `modelos_comparativa.csv` (filas AGREGADO Grupo A) | RMSE, MAE, IC95 diff vs PRIOR, p-Wilcoxon |
| Figura 2 | `modelos_comparativa.csv` (filas bayesiano screening + finalistas) | RMSE/MAE por config y bloque |
| Figura 3 | `prototipos_resumen.csv`, `prototipos_similitudes_locales.csv` | 5 prototipos oficiales + ejemplo local |

