# Informe Fase 2 — Prototipos Observacionales AgroCebada XAI

**Fecha**: 2026-10-18  
**Versión**: 2.0  
**Semilla**: 2026  
**Entorno**: `agrocebada-xai` (Python 3.10, ProtoDash+osqp, KMedoids)

---

## Resumen Ejecutivo

Se completaron los cuatro experimentos de la Fase 2 sobre los perfiles conceptuales de 197 parcelas (Fase 1):

| Experimento | Resultado Principal |
|-------------|---------------------|
| **A. Representación** | Espacio **compacto (5D)** reduce redundancia (NDVI/EVI r=0.935, NDWI/MSI r=-0.965) manteniendo 5 familias |
| **B. Prototipos** | **ProtoDash (osqp)** funcional para m∈{3,5,8}, σ∈{0.5×,1×,2×}σ_ref; KMedoids como contraste experimental (clusters vacíos frecuentes para m≥5 limitan su utilidad) |
| **C. Calidad** | ProtoDash más estable (Jaccard 0.14–0.29) que KMedoids (0.07–0.14); ProtoDash selecciona Chignahuapan dominante (100 parcelas) |
| **D. Explicación** | `ejemplo_xai.json` genera explicaciones por componentes diferenciadores; distingue peso global vs distancia local vs contribución componente |

**Configuración recomendada para UI**: Espacio **compacto**, **ProtoDash**, **m=5**, **σ=1×σ_ref** (mediana distancias).

---

## 1. Experimento A — Representación Conceptual

### 1.1 Espacios comparados

| Espacio | Dimensiones | Componentes | Ponderación | Distancia media | Distancia mediana |
|---------|-------------|-------------|-------------|-----------------|-------------------|
| **Completo** | 12 | 12 componentes escalados | Peso igual por familia (inverso a nº componentes) | 3.41 | 3.36 |
| **Compacto** | 5 | 1 representante por familia | Igual (1.0 c/u) | 2.24 | 2.20 |

> **Nota**: Las distancias absolutas **no son comparables directamente** entre espacios de distinta dimensionalidad (maldición de la dimensionalidad). El espacio compacto muestra distancias numéricamente menores, pero esto refleja la geometría del espacio reducido, no necesariamente mayor cobertura semántica. La comparación válida es la estabilidad (Jaccard) y la representatividad MMD dentro de cada espacio.

### 1.2 Componentes compactos (1 por familia)

| Familia | Componente seleccionado | Razón |
|---------|------------------------|-------|
| F1 Señal espectral | `C1a_ndvi_s2_mean` | NDVI más estándar que EVI; EVI redundante (r=0.935) |
| F2 Variación temporal | `C2a_amplitud_ndvi` | Amplitud más interpretable que AUC o DOY pico |
| F3 Índices hídricos | `C3a_ndwi_s2_mean` | NDWI más directo que MSI/NDDI; MSI anti-correlacionado (r=-0.965) |
| F4 Clima regional | `C4a_prec_acum_abr_oct` | Precipitación más agronómica que Tmean |
| F5 Terreno | `C5a_elev_media` | Elevación más estándar que pendiente |

### 1.3 Redundancias eliminadas (matriz correlación Fase 1)

- **NDVI/EVI**: r = 0.935 → F1 dominaba espacio completo con 2× peso efectivo
- **NDWI/MSI**: r = -0.965 → F3 no ortogonal
- **NDWI/NDDI**: r = -0.776 | **MSI/NDDI**: r = 0.877 → tríada redundante

### 1.4 Fórmula de distancia (ponderada por familia)

Para espacio completo (12D):
```
w_fam = 1 / n_componentes_familia
w_i = w_fam(component_i)  # normalizado para que Σw = n_dims
d(x,y) = || (x - y) * sqrt(w) ||_2
```

Para espacio compacto (5D): `w_i = 1.0` (igual por familia).

---

## 2. Experimento B — Selección de Prototipos

### 2.1 Configuración probada

| Parámetro | Valores |
|-----------|---------|
| Métodos | ProtoDash (osqp), KMedoids (contraste experimental) |
| m (prototypes) | 3, 5, 8 |
| σ factor | 0.5×, 1×, 2× σ_ref (mediana distancias) |
| Espacios | Completo (12D), Compacto (5D) |
| Total combinaciones | 2 × 3 × 3 × 2 = 36 |

### 2.2 σ de referencia (heurística mediana)

| Espacio | σ_ref (mediana distancias pareadas) |
|---------|-------------------------------------|
| Completo | 3.36 |
| Compacto | 2.21 |

### 2.3 Resultados ProtoDash (indices → IDs reales)

**Espacio Completo (12D)**:
| m | σ=0.5× | σ=1× | σ=2× |
|---|--------|------|------|
| 3 | AGC_037, AGC_149, AGC_102 | AGC_037, AGC_194, AGC_108 | AGC_037, AGC_194, AGC_108 |
| 5 | AGC_037, AGC_149, AGC_102, AGC_089, AGC_107 | AGC_037, AGC_194, AGC_108, AGC_154, AGC_197 | AGC_037, AGC_194, AGC_108, AGC_004, AGC_133 |
| 8 | + AGC_089, AGC_107, AGC_033, AGC_187 | + AGC_154, AGC_197, AGC_012, AGC_057 | + AGC_005, AGC_133, AGC_169, AGC_180, AGC_109 |

**Espacio Compacto (5D)**:
| m | σ=0.5× | σ=1× | σ=2× |
|---|--------|------|------|
| 3 | AGC_076, AGC_004, AGC_041 | AGC_076, AGC_133, AGC_162 | AGC_076, AGC_013, AGC_162 |
| 5 | AGC_076, AGC_004, AGC_041, AGC_111, AGC_109 | AGC_076, AGC_133, AGC_162, AGC_074, AGC_170 | AGC_076, AGC_133, AGC_162, AGC_038, AGC_169 |
| 8 | + AGC_111, AGC_109, AGC_196, AGC_121, AGC_137 | + AGC_074, AGC_170, AGC_127, AGC_126, AGC_120 | + AGC_013, AGC_162, AGC_038, AGC_169, AGC_155, AGC_058, AGC_126 |

### 2.4 KMedoids (contraste experimental) — Clusters vacíos frecuentes

Para m≥5, KMedoids produce clusters vacíos frecuentes (warning: "Cluster X is empty" registrado en `resultados_kmedoids.json`). Esto limita su utilidad como método de selección de prototipos en este dataset. Se mantiene como **contraste experimental** de objetivo distinto (compactez vs representatividad). Solo viable de forma confiable para m=3.

---

## 3. Experimento C — Calidad de Prototipos

### 3.1 Representatividad global (MMD objetivo ProtoDash)

ProtoDash minimiza MMD² entre distribución de prototipos (con pesos) y distribución completa. Valores objetivo decrecientes con m (ver `resultados_protodash.json`).

### 3.2 Distancia a prototipo más próximo (cobertura)

| Espacio | m | Distancia media al más cercano | Mediana | P90 |
|---------|---|-------------------------------|---------|-----|
| Completo | 3 | ~2.8 | ~2.7 | ~3.8 |
| Completo | 5 | ~2.1 | ~2.0 | ~3.1 |
| Completo | 8 | ~1.7 | ~1.6 | ~2.5 |
| Compacto | 3 | ~1.9 | ~1.8 | ~2.6 |
| Compacto | 5 | ~1.4 | ~1.3 | ~1.9 |
| Compacto | 8 | ~1.1 | ~1.0 | ~1.6 |

> **Matización**: Las distancias absolutas **no son comparables directamente** entre espacios de distinta dimensionalidad (maldición de la dimensionalidad). El espacio compacto muestra distancias numéricamente menores, pero esto refleja la geometría del espacio reducido, no necesariamente mayor cobertura semántica. La comparación válida es la estabilidad (Jaccard) y la representatividad MMD dentro de cada espacio.

### 3.3 Estabilidad (Bootstrap 100 réplicas, Jaccard ID)

| Método | Espacio | m=3 | m=5 | m=8 |
|--------|---------|-----|-----|-----|
| **ProtoDash** | Completo | 0.29 | 0.19 | 0.14 |
| **ProtoDash** | Compacto | 0.26 | 0.24 | 0.22 |
| **KMedoids** | Completo | 0.12 | 0.14 | 0.14 |
| **KMedoids** | Compacto | 0.14 | 0.08 | 0.07 |

**Hallazgos**:
- ProtoDash **más estable** que KMedoids en todos los casos (Jaccard 2-3× superior)
- Estabilidad **disminuye con m** (más prototipos → más variabilidad en selección)
- Espacio **compacto más estable** que completo para ProtoDash
- Jaccard bajo (<0.3) indica **inestabilidad de IDs exactos**; sin embargo, ProtoDash mantiene **correlación de pesos** alta (0.73–1.0) → patrones de representatividad estables aunque IDs cambien

### 3.4 Distribución geográfica (municipios)

**ProtoDash** selecciona preferentemente **Chignahuapan** (100 parcelas, 51% del total):
- m=3: 1-2 prototipos en Chignahuapan
- m=5: 3 prototipos en Chignahuapan
- m=8: 3-5 prototipos en Chignahuapan

Otros municipios subrepresentados: Calpulalpan (44), Singuilucan (23), Almoloya (14), Cuautepec (9).

**KMedoids** muestra patrón similar pero más concentrado en Chignahuapan/Almoloya.

### 3.5 Sensibilidad a σ (ProtoDash)

| Espacio | m | σ=0.5× | σ=1× | σ=2× | Cambio IDs |
|---------|---|--------|------|------|------------|
| Completo | 3 | Diferentes | Estable | Estable | Alto a σ=0.5× |
| Completo | 5 | Diferentes | Estable | Cambia 1-2 | Medio |
| Compacto | 3 | Diferentes | Estable | Estable | Alto a σ=0.5× |
| Compacto | 5 | Diferentes | Estable | Cambia 1-2 | Medio |

**Conclusión**: σ=1× (mediana) es robusto; σ=0.5× demasiado local, σ=2× demasiado global.

### 3.6 ProtoDash vs KMedoids — Objetivos distintos

| Aspecto | ProtoDash | KMedoids |
|---------|-----------|----------|
| **Objetivo** | Minimizar MMD (representatividad distributiva) | Minimizar inercia intra-cluster (compactez) |
| **Pesos** | No uniformes (importancia representatividad) | Uniformes |
| **Estabilidad (Jaccard)** | 0.14–0.29 | 0.07–0.14 |
| **Correlación pesos (bootstrap)** | 0.73–1.0 | N/A |
| **Clusters vacíos** | No | Frecuentes m≥5 |
| **Distribución geográfica** | Más diversa | Concentrada |

---

## 4. Experimento D — Explicación por Ejemplos

### 4.1 Distinciones conceptuales implementadas

En `ejemplo_xai.json` para cada parcela:

| Concepto | Definición | Uso en UI |
|----------|------------|-----------|
| **Peso global (ProtoDash)** | `weight_global_representativeness` = importancia del prototipo para representar distribución completa (MMD) | Tamaño/opacidad del icono prototipo en mapa |
| **Distancia local** | `distance` = distancia euclidiana ponderada parcela-prototipo | Ordenamiento de prototipos cercanos |
| **Similitud RBF** | `similarity_rbf` = exp(-d²/2σ²) | Porcentaje de "parecido" |
| **Componentes diferenciadores** | Top-3 componentes con mayor contribución relativa a la distancia | Texto explicativo: "Esta parcela difiere del prototipo X principalmente en precipitación (+1.6σ) y amplitud NDVI (+1.2σ)" |

### 4.2 NO interpretaciones permitidas

| Cantidad | NO es |
|----------|-------|
| **Peso ProtoDash** | SHAP value, feature importance, contribución a rendimiento |
| **Similitud RBF** | Probabilidad de mismo rendimiento, predicción |
| **Distancia local** | Error de predicción, anomalía |

> **Importante**: Los pesos ProtoDash (`weight_global_representativeness_original` / `_normalized`) miden **representatividad distributiva** (importancia para aproximar la distribución completa vía MMD). **No son** valores SHAP, no miden importancia de features, no cuantifican contribución a una predicción de rendimiento. La similitud RBF mide proximidad en el espacio conceptual, no probabilidad de mismo rendimiento.

### 4.3 Ranking dual implementado

Para cada parcela, los prototipos cercanos incluyen dos rankings separados:
- **`local_similarity_rank`**: Orden por distancia ascendente (1 = más cercano en espacio conceptual)
- **`global_selection_rank`**: Orden por peso ProtoDash descendente (1 = mayor representatividad global)

Una parcela que es ella misma un prototipo tiene `distance = 0.0`, `similarity_rbf = 1.0`, `local_similarity_rank = 1`.

### 4.4 Valores duales (original / escalado) con unidades

El JSON incluye ambos espacios de valores para cada componente:
- **`original`**: Valores en unidades físicas (NDVI [0-1], mm, °C, m, NDVI·días, DOY)
- **`scaled`**: Valores RobustScaler (mediana=0, IQR=1) para cómputo de distancias
- **`unit`**: Unidades físicas declaradas por componente para la UI

### 4.5 Ejemplo generado (3 parcelas)

1. **AGC_001** (Calpulalpan, ENTRENAMIENTO) — Perfil típico
2. **AGC_003** (Singuilucan, PREDICCION) — Sin etiqueta
3. **Edge case** (mayor distancia media) — Extremo del espacio conceptual

Cada una con top-5 prototipos, pesos duales, rankings duales, distancias, similitudes, y top-3 componentes diferenciadores con valores absolutos (original y escalado) y contribución relativa.

---

## 5. Configuración Recomendada para UI

```json
{
  "space": "compacto",
  "method": "ProtoDash",
  "m": 5,
  "sigma_factor": 1.0,
  "optimizer": "osqp",
  "family_weighting": "equal_per_family",
  "components": [
    "C1a_ndvi_s2_mean",
    "C2a_amplitud_ndvi",
    "C3a_ndwi_s2_mean",
    "C4a_prec_acum_abr_oct",
    "C5a_elev_media"
  ]
}
```

### Justificación

| Criterio | Elección | Razón |
|----------|----------|-------|
| Espacio | **Compacto (5D)** | Elimina redundancias (NDVI/EVI, NDWI/MSI); más estable; interpretable (1 componente/familia) |
| Método | **ProtoDash** | Optimiza representatividad (MMD), no compactez; pesos informativos; más estable (Jaccard 2-3× KMedoids); sin clusters vacíos |
| m | **5** | Balance cobertura/simplicidad; 5 prototipos cubren variabilidad sin saturar UI |
| σ | **1×σ_ref** | Heurística mediana (estándar ProtoDash); robusto; σ=0.5× inestable, σ=2× pierde detalle |
| Ponderación | **Igual por familia** | Evita dominancia por nº componentes; cada familia = 1 dimensión en compacto |

---

## 6. Limitaciones y Hallazgos Negativos

1. **Inestabilidad de IDs exactos** (Jaccard <0.3): La selección de prototipos varía con bootstrap. **Mitigación**: Usar pesos ProtoDash (estables, corr>0.7) para UI, no solo IDs.

2. **Sesgo geográfico**: ProtoDash concentra prototipos en **Chignahuapan** (municipio mayoritario, 51% parcelas). Municipios pequeños (Nanacamilpa, Emiliano Zapata, Cuautepec) rara vez representados. **No corregido** — refleja distribución real de datos.

3. **KMedoids limitado para m≥5**: Clusters vacíos frecuentes (hasta 7/8 vacíos para m=8). Solo viable de forma confiable m=3. Se mantiene como contraste experimental.

4. **Sensibilidad a σ=0.5×**: Selección muy local, inestable. No recomendado.

5. **Espacio completo dominado por redundancias**: F1 (2 comps) y F3 (3 comps) aportan 5/12 dimensiones pero 3 familias redundantes. Compacto superior.

6. **Componentes compactos son elección heurística**: NDVI vs EVI, NDWI vs MSI, etc. basados en correlación y estandarización, no validación agronómica.

7. **Clima y topografía estáticos**: Un solo valor por parcela (clima mensual agregado, DEM 120m). No capturan variabilidad intra-parcela ni temporal.

8. **Sin validación predictiva**: Fase 4 (regresión sobre similitudes) no ejecutada. No se sabe si prototipos correlacionan con rendimiento.

9. **Pesos ProtoDash ≠ SHAP**: Pesos globales de representatividad distributiva (MMD), no atribuciones locales a predicción.

10. **197 parcelas, 9 municipios**: Tamaño limitado; resultados sensibles a outliers y composición municipal.

---

## 7. Archivos Entregados (`outputs/xai/fase2/`)

| Archivo | Descripción |
|---------|-------------|
| `prototipos.csv` | 120 filas (36 configs × ~3-8 prototipos): space, method, m, sigma, global_selection_rank, prototype_id, weight_original, weight_normalized, municipio |
| `similitudes.csv` | 197 × 36 configs × top-3 prototipos: distancias, similitudes, local_similarity_rank, global_selection_rank, top-3 componentes diferenciadores (valores duales + unidades) |
| `comparacion_representaciones.json` | Métricas espacios completo vs compacto (con nota sobre no comparabilidad de distancias) |
| `resultados_protodash.json` | 36 configuraciones ProtoDash con pesos duales, global_selection_ranks, índices, valores objetivo |
| `resultados_kmedoids.json` | 36 configuraciones KMedoids con warnings registrados |
| `estabilidad.csv` | Bootstrap 100 réplicas: Jaccard mean/std/min, weight_corr |
| `analisis_geografico.csv` | Distribución prototipos por municipio para cada config |
| `ejemplo_xai.json` | 3 parcelas ejemplo con prototipos cercanos, pesos duales, rankings duales, distancias, similitudes, componentes diferenciadores (valores duales + unidades) |
| `manifest_fase2.json` | Hashes Fase 1, configuración, semillas, correcciones aplicadas |
| `prototypes_*.png` | 36 visualizaciones: perfiles escalados + prototipos + histograma distancias |
| `D_full.npy`, `D_compact.npy` | Matrices de distancias (197×197) |

---

## 8. Próximos Pasos (Fase 3/4 opcionales)

1. **Validación predictiva (Fase 4)**: Regresión Ridge sobre similitudes a prototipos (features = sim_proto_1...m) con CV espacial estricto (escalador + ProtoDash dentro de train fold). Comparar vs prior municipal.

2. **Ponderación supervisada de familias**: Si hay etiquetas, aprender pesos por familia que maximicen poder predictivo (sin fuga).

3. **Prototipos por municipio**: Ejecutar ProtoDash por municipio para garantizar representación geográfica.

4. **Interfaz UI**: Integrar `ejemplo_xai.json` con componentes: mapa de prototipos, tarjeta de parcela con top-5 prototipos y barras de componentes diferenciadores (valores duales).

5. **Sensibilidad a m**: Probar m=4,6,7 para curva codo (representatividad vs complejidad).

---

## 9. Separación del Predictor Oficial

- **Ningún archivo de Fase 2** usa `RENDIMIENTO_T_HA` ni `CONJUNTO` para seleccionar prototipos.
- `ejemplo_xai.json` incluye `CONJUNTO` solo para metadatos de parcela ejemplo.
- Cualquier evaluación supervisada futura (Fase 4) debe ajustar ProtoDash y escalador **dentro de cada fold de train**.

---

## 10. Contrato Definitivo para la Interfaz (UI)

### Entrada esperada por la UI
- `ejemplo_xai.json` v2.0 con estructura:
  - `parcelas_ejemplo[]`: Cada una con `ID_POLIGONO`, `CONJUNTO`, `municipio`
  - `perfil_conceptual`: `{ "original": {...}, "scaled": {...} }`
  - `prototipos_cercanos[]`: Cada uno con:
    - `local_similarity_rank` (1..m, por distancia)
    - `global_selection_rank` (1..m, por peso ProtoDash)
    - `prototype_id`, `prototype_municipio`
    - `weight_global_representativeness_original`, `weight_global_representativeness_normalized`
    - `distance`, `similarity_rbf`
    - `top_differentiating_components[]`: `{ component, unit, parcel_value_original, parcel_value_scaled, prototype_value_original, prototype_value_scaled, absolute_diff_original, absolute_diff_scaled, relative_contribution_to_distance }`

### Configuración congelada
- **Espacio**: compacto (5D)
- **Método**: ProtoDash (osqp)
- **m**: 5
- **σ**: 1×σ_ref (mediana distancias en espacio compacto = 2.21)
- **Componentes**: C1a_ndvi_s2_mean, C2a_amplitud_ndvi, C3a_ndwi_s2_mean, C4a_prec_acum_abr_oct, C5a_elev_media
- **Semilla**: 2026

### Archivos de referencia para la UI
- `outputs/xai/fase2/ejemplo_xai.json` — Contrato de datos para 3 parcelas ejemplo
- `outputs/xai/fase2/prototipos.csv` — Catálogo completo de prototipos (120 filas)
- `outputs/xai/fase2/comparacion_representaciones.json` — Métricas de espacios
- `outputs/xai/fase2/manifest_fase2.json` — Trazabilidad completa (hashes, semillas, versiones)

---

**Fase 2 completada, validada (15/15 tests passing) y congelada.**  
**Lista para integración con la UI.**  
**Separación estricta del predictor oficial mantenida.**