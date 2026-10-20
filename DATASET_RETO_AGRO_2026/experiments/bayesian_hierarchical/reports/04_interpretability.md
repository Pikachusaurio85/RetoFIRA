# Fase 4 — Interpretación del modelo

Ajustes sobre las **138 parcelas de ENTRENAMIENTO** con priors finales (NUTS 2000×4, 4 cadenas,
R-hat máx. < 1.01). Artefactos en `results/phase4/`, figuras en `reports/figures/`.

## 1. Descomposición de varianza

| Modelo | ICC (media, IC95%) | tau | sigma |
|---|---|---|---|
| B0 | **0.695** [0.464, 0.878] | 0.796 | 0.496 |
| B3 | 0.672 [0.425, 0.868] | 0.749 | 0.493 |
| B5 | 0.665 [0.430, 0.880] | 0.750 | 0.494 |

Coincide con la descomposición descriptiva de Fase 1 (frac_between = 0.686). **≈70 % de la varianza
del rendimiento es entre municipios**; solo ~30 % queda como margen *dentro* de municipio, que es
donde podrían actuar las covariables ambientales. El ICC apenas cambia al añadir variables: las
covariables no absorben varianza municipal.

## 2. Efectos municipales (pooling parcial)

La contracción (shrinkage) es la esperada: λ = tau² / (tau² + sigma²/n).

| Municipio | n | Desv. media obs. | Efecto posterior u | λ |
|---|---|---|---|---|
| Chignahuapan | 71 | +0.522 | +0.738 | 0.995 |
| Calpulalpan | 30 | −0.955 | −0.724 | 0.987 |
| Singuilucan | 16 | −0.874 | −0.633 | 0.976 |
| Almoloya | 10 | +0.566 | +0.750 | 0.963 |
| Cuautepec | 5 | +0.643 | +0.790 | 0.928 |
| Emiliano Zapata | 3 | −0.455 | −0.202 | 0.885 |
| Apan | 1 | −0.955 | −0.516 | 0.720 |
| Tepeapulco | 1 | +0.495 | +0.502 | 0.720 |
| Nanacamilpa | 1 | −1.455 | −0.856 | 0.720 |

Municipios con n=1 (λ≈0.72) ven su efecto contraído ~28 % hacia la media global: **Nanacamilpa
pasa de −1.46 a −0.86**. Esto es el valor añadido del prior jerárquico frente a medias crudas
inestables. El resto queda prácticamente sin contraer por su tamaño. Figura: `figures/B0_municipios.png`.

## 3. Coeficientes ambientales (t/ha por 1 DE de la covariable)

### B3 (mejor set ambiental)
| Covariable | beta (IC95%) | P(beta>0) |
|---|---|---|
| clim_prec_temp | −0.037 [−0.166, 0.087] | 0.28 |
| topo_elev_media | +0.100 [0.002, 0.197] | 0.98 |
| s2_ndvi_amp | −0.056 [−0.150, 0.039] | 0.12 |

### B5 (conceptos XAI oficiales)
| Concepto | beta (IC95%) | P(beta>0) |
|---|---|---|
| C1a_ndvi_s2_mean | +0.067 [−0.074, 0.208] | 0.82 |
| C2a_amplitud_ndvi | −0.040 [−0.150, 0.074] | 0.24 |
| C3a_ndwi_s2_mean | −0.098 [−0.240, 0.048] | 0.10 |
| C4a_prec_acum_abr_oct | −0.022 [−0.153, 0.113] | 0.37 |
| C5a_elev_media | **+0.141 [0.026, 0.257]** | **0.99** |

**Único efecto con soporte:** la **elevación** (C5a / topo_elev_media), con un efecto positivo muy
pequeño (~0.10–0.14 t/ha por DE). Todos los demás coeficientes incluyen 0 en su IC95 %. Aun siendo
"creíble" la elevación, su magnitud es **~0.1 t/ha**, cinco veces menor que el spread municipal
(tau ≈ 0.8 t/ha), y por eso no se traduce en mejora de RMSE en validación cruzada. Figuras:
`figures/B3_betas.png`, `figures/B5_betas.png`.

> Nota: estos `beta` **no** son comparables numéricamente con los pesos de ProtoDash
> (`outputs/xai/fase2/`). ProtoDash es un método basado en *prototipos representativos*, no una
> medida de importancia de variables; forzar una correlación entre ambos sería un error de
> interpretación. La comparación aquí es puramente cualitativa sobre el signo del efecto.

## 4. Test de discriminación *dentro* de municipio

Si las covariables ambientales aportaran señal, el ajuste ambiental `delta = pred_B5 − pred_B0`
debería correlacionar con el residuo *dentro* de cada municipio. Resultado (OOF, 50 folds):

| Municipio | n pares | corr(delta, resid) |
|---|---|---|
| Almoloya | 100 | +0.358 |
| Chignahuapan | 710 | +0.133 |
| Calpulalpan | 300 | −0.068 |
| Cuautepec | 50 | −0.414 |
| Singuilucan | 160 | −0.467 |
| Apan / Emiliano Zapata / Nanacamilpa / Tepeapulco | 10–30 | n/d (sin varianza) |

**Correlación agrupada (centrada por municipio) = +0.051.** Prácticamente cero, y con **signos
inconsistentes** entre municipios. Es decir, el ajuste ambiental de B5 **no sigue** la variación
real de rendimiento dentro de los municipios: las covariables disponibles no llevan información
predictiva intra-municipio. Figura: `figures/within_municipio_delta.png`.

## 5. Síntesis

- El efecto municipal domina (ICC≈0.70) y el pooling parcial estabiliza bien los municipios
  pequeños (n=1, λ≈0.72).
- La única covariable con efecto creíble es la elevación, de magnitud despreciable.
- El test intra-municipio confirma que las covariables no capturan la variación residual
  relevante.
- El cuello de botella es **la cantidad/calidad de etiquetas por municipio**, no la falta de
  covariables ambientales.
