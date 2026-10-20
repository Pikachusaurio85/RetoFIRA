# Catálogo de Conceptos Observacionales — AgroCebada XAI Fase 1

**Versión**: 1.0  
**Fecha**: 2026-10-18  
**Semilla**: 2026  
**Fuentes**: Sentinel-2 (BASIC 2025, nube ≤5%), CHIRPS/CHIRTS (clima 2025 abr-oct), INEGI CEM4 (topografía)  
**Parcelas**: 197 (9 municipios)  
**Restricción**: No se usan etiquetas de rendimiento ni conceptos supervisados.

---

## Estructura: 12 componentes base organizados en 5 familias conceptuales

| Familia | Componentes | Dimensiones | Peso relativo inicial |
|---------|-------------|-------------|----------------------|
| F1: Señal espectral S2 | C1a, C1b | 2 | 1.0 c/u |
| F2: Variación temporal | C2a, C2b, C2c | 3 | 1.0 c/u |
| F3: Índices hídricos | C3a, C3b, C3c | 3 | 1.0 c/u |
| F4: Clima regional | C4a, C4b | 2 | 1.0 c/u |
| F5: Terreno | C5a, C5b | 2 | 1.0 c/u |

**Total**: 12 dimensiones. Pesos iguales por componente → familias con más componentes pesan más en distancia euclidiana no ponderada. Ver `manifest_xai.json` para parámetros de RobustScaler.

---

## F1: Señal espectral Sentinel-2

### C1a: ndvi_s2_mean
- **Nombre observacional**: NDVI Sentinel-2 medio temporal abr-oct 2025
- **Variable origen**: `ndvi_promedio` en BASIC (Sentinel-2)
- **Fórmula**: `mean(NDVI_s2 | sensor=S2, year=2025, cloud≤5%, DOY∈[91,304])`
- **Unidades**: Adimensional [0, 1] (índice reflectancia)
- **Ventana**: DOY 91–304 (abril–octubre), filtro nube ≤5%
- **Normalización**: RobustScaler (mediana, IQR) — ver `manifest_xai.json`
- **Interpretación permitida**: "Parcela con NDVI medio superior a la mediana del conjunto"
- **Interpretación NO permitida**: "Mayor vigor fisiológico", "mejor estado nutricional", "mayor biomasa"
- **Problemas conocidos**:
  - Sensor: solo Sentinel-2 (no Planet/Landsat)
  - Temporalidad: media simple, no ponderada por intervalo entre observaciones
  - Correlación: alta con EVI (r≈0.97 en análisis previo)
  - Disponibilidad: todas las 197 parcelas tienen 7 meses válidos (media 31 obs/parcela)

### C1b: evi_s2_mean
- **Nombre observacional**: EVI Sentinel-2 medio temporal abr-oct 2025
- **Variable origen**: `evi_promedio` en BASIC (Sentinel-2)
- **Fórmula**: `mean(EVI_s2 | sensor=S2, year=2025, cloud≤5%, DOY∈[91,304])`
- **Unidades**: Adimensional [0, 1] (índice reflectancia corregido atmósfera)
- **Ventana**: DOY 91–304 (abril–octubre), filtro nube ≤5%
- **Normalización**: RobustScaler (mediana, IQR) — ver `manifest_xai.json`
- **Interpretación permitida**: "Parcela con EVI medio superior a la mediana del conjunto"
- **Interpretación NO permitida**: "Mayor vigor fotosintético", "menor influencia de suelo"
- **Problemas conocidos**:
  - Sensor: solo Sentinel-2
  - Correlación muy alta con NDVI (r≈0.97) → redundancia en distancia euclidiana
  - EVI menos sensible a saturación en biomasa alta, pero no validado en campo

---

## F2: Variación temporal (forma de curva NDVI S2)

### C2a: amplitud_ndvi
- **Nombre observacional**: Amplitud de NDVI S2 (pico − baseline pre-ciclo)
- **Variables origen**: `ndvi_promedio` en BASIC (Sentinel-2)
- **Fórmula**: `peak_NDVI − baseline_pre151` donde:
  - `peak_NDVI = max(NDVI_s2 | DOY∈[91,304])`
  - `baseline_pre151 = median(NDVI_s2 | DOY≤151)` (pre-ciclo, abril–mayo)
- **Unidades**: Adimensional NDVI (diferencia de índices)
- **Ventana**: Pico en DOY 91–304; baseline DOY ≤151
- **Normalización**: RobustScaler (mediana, IQR) — ver `manifest_xai.json`
- **Interpretación permitida**: "Parcela con mayor rango dinámico NDVI entre pre-ciclo y pico"
- **Interpretación NO permitida**: "Mayor desarrollo vegetativo", "mejor llenado de grano"
- **Problemas conocidos**:
  - Baseline usa mediana pre-ciclo (pocas observaciones en algunas parcelas)
  - Pico = máximo observado, no suavizado → sensible a outliers individuales
  - No hay etiqueta fenológica que valide correspondencia con espigamiento/madurez

### C2b: auc_exceso_ndvi
- **Nombre observacional**: Área bajo curva de exceso NDVI sobre baseline (trapezoidal)
- **Variables origen**: `ndvi_promedio` en BASIC (Sentinel-2)
- **Fórmula**: `∫ max(NDVI_s2(doy) − baseline_pre151, 0) d(doy)` integrado por trapecios sobre DOY ordenados 91–304
- **Unidades**: NDVI·días (índice-día)
- **Ventana**: DOY 91–304, baseline DOY ≤151
- **Normalización**: RobustScaler (mediana, IQR) — ver `manifest_xai.json`
- **Interpretación permitida**: "Parcela con mayor integral de exceso NDVI respecto a pre-ciclo"
- **Interpretación NO permitida**: "Mayor radiación absorbida", "mayor producción de biomasa", "LAI integrado"
- **Problemas conocidos**:
  - Integración trapezoidal asume linealidad entre observaciones (intervalos irregulares)
  - No pondera por intervalo temporal entre capturas
  - Baseline fijo (mediana pre-ciclo) no captura dinámica temprana

### C2c: doy_pico_ndvi
- **Nombre observacional**: Día del año (DOY) del pico NDVI S2 observado
- **Variables origen**: `ndvi_promedio`, `fecha_captura` en BASIC (Sentinel-2)
- **Fórmula**: `DOY del primer máximo de NDVI_s2 en DOY∈[91,304]`
- **Unidades**: DOY (día del año, 1–365)
- **Ventana**: Búsqueda en DOY 91–304
- **Normalización**: RobustScaler (mediana, IQR) — ver `manifest_xai.json`
- **Interpretación permitida**: "Parcela con pico NDVI más tardío/temprano en la ventana observada"
- **Interpretación NO permitida**: "Fecha de espigamiento", "fecha de madurez", "ciclo más largo/corto"
- **Problemas conocidos**:
  - Pico = máximo observado crudo (sin suavizado) → sensible a ruido
  - Ventana de búsqueda fija (91–304) puede excluir picos reales fuera de rango
  - No hay etiqueta fenológica de campo para validar correspondencia biológica

---

## F3: Índices hídricos espectrales Sentinel-2

### C3a: ndwi_s2_mean
- **Nombre observacional**: NDWI Sentinel-2 medio temporal abr-oct 2025
- **Variable origen**: `ndwi_promedio` en BASIC (Sentinel-2)
- **Fórmula**: `mean(NDWI_s2 | sensor=S2, year=2025, cloud≤5%, DOY∈[91,304])`
- **Unidades**: Adimensional (índice agua: (NIR-SWIR)/(NIR+SWIR))
- **Ventana**: DOY 91–304, filtro nube ≤5%
- **Normalización**: RobustScaler (mediana, IQR) — ver `manifest_xai.json`
- **Interpretación permitida**: "Parcela con mayor/respuesta NDWI medio en la ventana"
- **Interpretación NO permitida**: "Mayor contenido de agua en dosel", "menor estrés hídrico", "humedad de suelo"
- **Problemas conocidos**:
  - NDWI sensible a estructura de dosel y fondo de suelo, no solo agua
  - Confusión con senescencia (baja reflectancia NIR)
  - No hay mediciones de agua de suelo/foliar para validación

### C3b: msi_s2_mean
- **Nombre observacional**: MSI Sentinel-2 medio temporal abr-oct 2025
- **Variable origen**: `msi_promedio` en BASIC (Sentinel-2)
- **Fórmula**: `mean(MSI_s2 | sensor=S2, year=2025, cloud≤5%, DOY∈[91,304])`
- **Unidades**: Adimensional (índice humedad: SWIR/NIR)
- **Ventana**: DOY 91–304, filtro nube ≤5%
- **Normalización**: RobustScaler (mediana, IQR) — ver `manifest_xai.json`
- **Interpretación permitida**: "Parcela con mayor/respuesta MSI medio en la ventana"
- **Interpretación NO permitida**: "Mayor estrés hídrico", "menor contenido de agua"
- **Problemas conocidos**:
  - MSI inversamente relacionado con contenido de agua (mayor MSI = más seco)
  - Sensible a estructura de dosel y residuos de cosecha
  - Correlación negativa esperada con NDWI (ver matriz correlación en LIMITACIONES_FASE1.md)

### C3c: nddi_s2_mean
- **Nombre observacional**: NDDI Sentinel-2 medio temporal abr-oct 2025
- **Variable origen**: `nddi_promedio` en BASIC (Sentinel-2)
- **Fórmula**: `mean(NDDI_s2 | sensor=S2, year=2025, cloud≤5%, DOY∈[91,304])`
- **Unidades**: Adimensional (índice sequía normalizado: (NDVI-NDWI)/(NDVI+NDWI))
- **Ventana**: DOY 91–304, filtro nube ≤5%
- **Normalización**: RobustScaler (mediana, IQR) — ver `manifest_xai.json`
- **Interpretación permitida**: "Parcela con mayor/respuesta NDDI medio en la ventana"
- **Interpretación NO permitida**: "Índice de sequía validado", "estrés hídrico medido"
- **Problemas conocidos**:
  - Combinación no lineal de NDVI y NDWI → amplifica ruido de ambos
  - Rango teórico [-1,1], valores observados positivos (ver estadísticas)
  - No validado contra mediciones de potencial hídrico foliar o humedad de suelo

---

## F4: Clima regional (exposición, no microclima)

### C4a: prec_acum_abr_oct
- **Nombre observacional**: Precipitación acumulada CHIRPS abr-oct 2025
- **Variable origen**: `prec_mm` en `clima_mensual_parcela.csv` (CHIRPS 0.05°)
- **Fórmula**: `sum(prec_mm | anio=2025, mes∈[4,10])` por parcela (via `celda_clima`)
- **Unidades**: mm (milímetros)
- **Ventana**: Meses 4–10 (abril–octubre) 2025
- **Normalización**: RobustScaler (mediana, IQR) — ver `manifest_xai.json`
- **Interpretación permitida**: "Parcela en celda climática con mayor precipitación acumulada abr-oct"
- **Interpretación NO permitida**: "Mayor agua disponible en raíz", "menor estrés hídrico", "lluvia efectiva"
- **Problemas conocidos**:
  - Resolución 0.05° (~5 km) → 25 celdas para 197 parcelas (no independientes)
  - CHIRPS = precipitación satelital, no pluviómetro en parcela
  - No considera escorrentía, infiltración, evaporación, riego
  - Mismo valor para todas las parcelas en una celda climática

### C4b: tmean_abr_oct
- **Nombre observacional**: Temperatura media CHIRTS-ERA5 abr-oct 2025
- **Variable origen**: `tmin`, `tmax` en `clima_mensual_parcela.csv` (CHIRTS-ERA5 0.05°)
- **Fórmula**: `mean((tmin + tmax)/2 | anio=2025, mes∈[4,10])` por parcela (via `celda_clima`)
- **Unidades**: °C (grados Celsius)
- **Ventana**: Meses 4–10 (abril–octubre) 2025
- **Normalización**: RobustScaler (mediana, IQR) — ver `manifest_xai.json`
- **Interpretación permitida**: "Parcela en celda climática con mayor temperatura media abr-oct"
- **Interpretación NO permitida**: "Mayor estrés térmico", "mayor demanda evaporativa", "acumulación grados-día"
- **Problemas conocidos**:
  - Misma limitación espacial que precipitación (25 celdas)
  - CHIRTS-ERA5 = reanálisis, no estación en parcela
  - No captura heladas, picos diarios, amplitud térmica diaria

---

## F5: Terreno (descriptores cartográficos estáticos)

### C5a: elev_media
- **Nombre observacional**: Elevación media parcelaria INEGI CEM4
- **Variable origen**: `elev_media` en `parcelas_base.csv` (INEGI CEM4 120m)
- **Fórmula**: `mean(elevación_DEM | píxeles dentro del polígono)`
- **Unidades**: m (metros sobre nivel del mar)
- **Ventana**: Estático (una medición por parcela)
- **Normalización**: RobustScaler (mediana, IQR) — ver `manifest_xai.json`
- **Interpretación permitida**: "Parcela a mayor/menor altitud media"
- **Interpretación NO permitida**: "Mayor/menor temperatura por gradiente altitudinal", "zona agroclimática distinta"
- **Problemas conocidos**:
  - Resolución 120m → parcelas pequeñas cubren pocos píxeles (mediana 6 pix, mínimo 1)
  - DEM = modelo digital de elevación, no medición in situ
  - Efecto agronómico de elevación no validado (confusión con clima, suelo, manejo)

### C5b: pend_media
- **Nombre observacional**: Pendiente media parcelaria INEGI CEM4
- **Variable origen**: `pend_media` en `parcelas_base.csv` (INEGI CEM4 120m)
- **Fórmula**: `mean(pendiente_DEM | píxeles dentro del polígono)`
- **Unidades**: Grados (0–90°)
- **Ventana**: Estático (una medición por parcela)
- **Normalización**: RobustScaler (mediana, IQR) — ver `manifest_xai.json`
- **Interpretación permitida**: "Parcela con mayor/menor pendiente media"
- **Interpretación NO permitida**: "Mayor erosión", "menor retención de agua", "manejo más difícil"
- **Problemas conocidos**:
  - Misma limitación de resolución que elevación (pocos píxeles/parcela)
  - Pendiente derivada de DEM 120m, no levantamiento topográfico
  - Efecto en cultivo mediado por suelo, drenaje, manejo (no directo)

---

## Flags de calidad por parcela

| Flag | Descripción | Valores observados |
|------|-------------|-------------------|
| `flag_n_meses_s2_abr_oct` | Meses con ≥1 obs S2 válida abr-oct | Todas 7 (media 7.0) |
| `flag_n_obs_s2_abr_oct` | Total observaciones S2 válidas abr-oct | Media 31.1, rango 25–38 |
| `flag_meses_clima_2025` | Meses con datos clima 2025 | Todas 7 (media 7.0) |
| `flag_topo_completa` | Topografía (elev+pend) disponible | Todas 1 (100%) |

---

## Concordancia Sentinel-2 vs Planet (resumen)

| Variable | Parcelas pareadas | Sesgo medio (Planet−S2) | MAE | Pearson r |
|----------|------------------|------------------------|-----|-----------|
| NDVI | 197 | +0.081 | 0.082 | 0.983 |
| EVI | 197 | +0.007 | 0.032 | 0.974 |

**Conclusión**: Alta correlación pero sesgo sistemático apreciable en NDVI (+0.08). No intercambiar ni promediar sin calibración. Ver `concordancia_sensores.csv` por parcela.

---

## Limitaciones transversales

1. **Ningún componente mide estado fisiológico** — son descriptores espectrales/temporales/climáticos/cartográficos
2. **Ventanas calendáricas, no fenológicas** — DOY 91–304 y ≤151 son cortes fijos, no etapas observadas
3. **Sensores no armonizados** — S2 y Planet tienen escalas/sesgos distintos; solo S2 en espacio conceptual
4. **Clima no parcelario** — 25 celdas 0.05° para 197 parcelas
5. **Topografía 120m** — Parcelas pequeñas = pocos píxeles DEM
6. **Etiquetas sin año/ciclo** — No se puede alinear observaciones 2025 con etiqueta de ciclo correspondiente
7. **Ponderación igual por defecto** — Familias con más componentes (F2, F3 = 3) dominan distancia euclidiana
8. **Baseline pre-ciclo** — Mediana DOY≤151 puede ser ruidosa (pocas obs en algunas parcelas)

---

## Archivos asociados

- `perfiles_conceptuales.csv` — 197 × 29 columnas (ID + 12 base + 12 escalados + 4 flags + 1 ID)
- `manifest_xai.json` — Hashes fuentes, parámetros scaler, semillas, ventanas
- `concordancia_sensores.csv` — Pareo S2-Planet ±5 días por parcela
- `conceptos.py` — Script reproducible
- `test_conceptos.py` — Pruebas automatizadas
- `LIMITACIONES_FASE1.md` — Correlaciones, discrepancias vs baseline, sensibilidad bandwidth