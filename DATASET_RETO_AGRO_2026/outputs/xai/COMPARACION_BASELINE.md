# Comparación con Baseline (features_parcela.csv)

Parcelas comunes: 197/197

## Correlaciones por componente

- **C1a_ndvi_s2_mean** vs **s2_ndvi_base** (NDVI baseline pre-ciclo vs media abr-oct (distintas definiciones)): r=0.3754, n=197, mean_new=0.3618, mean_base=0.1581
- **C1a_ndvi_s2_mean** vs **s2_ndvi_veg** (NDVI media vegetativo): r=0.6751, n=197, mean_new=0.3618, mean_base=0.3779
- **C1a_ndvi_s2_mean** vs **s2_ndvi_pico** (NDVI media pico): r=0.7254, n=197, mean_new=0.3618, mean_base=0.5699
- **C2a_amplitud_ndvi** vs **s2_ndvi_amp** (Amplitud pico-baseline): r=0.9541, n=197, mean_new=0.5134, mean_base=0.4797
- **C2b_auc_exceso_ndvi** vs **s2_ndvi_auc** (AUC exceso sobre baseline): r=0.9461, n=197, mean_new=49.0308, mean_base=51.6987
- **C2c_doy_pico_ndvi** vs **s2_ndvi_peak_doy** (DOY pico NDVI): r=0.8013, n=197, mean_new=238.4924, mean_base=240.7284
- **C2c_doy_pico_ndvi** vs **s2_ndvi_sos_doy** (DOY inicio temporada (SOS)): r=0.3654, n=197, mean_new=238.4924, mean_base=184.1604
- **C2c_doy_pico_ndvi** vs **s2_ndvi_eos_doy** (DOY fin temporada (EOS)): r=0.5874, n=197, mean_new=238.4924, mean_base=297.9391
- **C1b_evi_s2_mean** vs **s2_evi_base** (EVI baseline pre-ciclo): r=0.1625, n=197, mean_new=0.2905, mean_base=0.1162
- **C1b_evi_s2_mean** vs **s2_evi_veg** (EVI media vegetativo): r=0.6307, n=197, mean_new=0.2905, mean_base=0.3055
- **C1b_evi_s2_mean** vs **s2_evi_pico** (EVI media pico): r=0.7113, n=197, mean_new=0.2905, mean_base=0.4716
- **C3a_ndwi_s2_mean** vs **s2_ndwi_veg** (NDWI media vegetativo): r=0.5212, n=197, mean_new=0.0465, mean_base=0.0562
- **C3b_msi_s2_mean** vs **s2_msi_veg** (MSI media vegetativo): r=0.5256, n=197, mean_new=0.9825, mean_base=0.9356
- **C3c_nddi_s2_mean** vs **s2_nddi_veg** (NDDI media vegetativo): r=0.6318, n=197, mean_new=0.0643, mean_base=0.0606

## Análisis de discrepancias

- **C1a (media abr-oct) vs s2_ndvi_base (pre-ciclo)**: r=0.3754. Diferentes ventanas → esperamos correlación moderada.
- **C2a vs s2_ndvi_amp**: r=0.9541, diff media absoluta=0.0338. Mismo concepto, ligeras diferencias por definición de baseline.
- **C2b vs s2_ndvi_auc**: r=0.9461. C2b es exceso sobre baseline, baseline AUC es área total.
- **C2c vs s2_ndvi_peak_doy**: r=0.8013, diff media DOY=9.4. Mismo concepto, diferencias por detección de pico (suavizado vs crudo).
