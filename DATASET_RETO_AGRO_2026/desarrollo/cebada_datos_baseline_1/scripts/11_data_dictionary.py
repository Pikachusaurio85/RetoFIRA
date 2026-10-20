"""Genera diccionario_features.csv a partir de los nombres de columna de features_parcela.csv"""
import pandas as pd, re
from paths import DATA_PROCESSED
W=str(DATA_PROCESSED)
d = pd.read_csv(f"{W}/features_parcela.csv", index_col=0)
NOM = {'ndvi':'NDVI (vigor verde)','evi':'EVI (vigor, menos efecto suelo/atmosfera)','crc':'CRC (cobertura de residuos)','mcrc':'MCRC (residuos, variante)','ndsvi':'NDSVI (vegetacion senescente)',
 'srndi':'SRNDI (infrarrojo onda corta vs rojo)','mra':'MRA (nomenclatura del proveedor)','lai':'LAI (indice de area foliar)','fapar':'FAPAR (radiacion fotosintetica absorbida)','savi':'SAVI (ajustado por suelo)',
 'ndti':'NDTI (labranza)','sti':'STI (labranza simple)','ndwi':'NDWI (agua/humedad)','endwi':'ENDWI (agua, variante)','msi':'MSI (estres hidrico)','emsi':'EMSI (estres, variante)',
 'dswi2':'DSWI2','dswi3':'DSWI3','dswi4':'DSWI4','dswi5':'DSWI5 (estres hidrico/condicion vegetal)','nddi':'NDDI (sequia)','enddi':'ENDDI (sequia, variante)','vi6t':'VI6T (espectral-termico, Landsat)','msavi':'MSAVI (ajustado por suelo)'}
WIN = {'pre':'ventana pre-siembra/siembra (abr-may)','veg':'ventana crecimiento (jun-jul)','pico':'ventana pico (ago-sep)','sen':'ventana senescencia/cosecha (oct)'}
PH = {'base':'linea base ene-may (suelo/rastrojo), mediana de la curva','peak':'valor maximo de la curva (jun-oct)','amp':'amplitud = pico - linea base','peak_doy':'dia del anio del pico','sos_doy':'dia del anio de inicio de temporada (50% de la amplitud)',
 'eos_doy':'dia del anio de fin de temporada (50% de la amplitud)','los':'duracion de temporada (dias) = fin - inicio','auc':'area bajo la curva sobre la linea base, jun-nov','slope_up':'pendiente maxima de subida (unidades/dia)','slope_down':'pendiente maxima de bajada (unidades/dia)'}
SEN = {'s2':'Sentinel-2 (nubes<=5%)','pl':'Planet (nubes<=20%)','ls':'Landsat (nubes<=5%)'}
CL_OVR = {'clim_prec_cv':'Coeficiente de variacion mensual de la lluvia, abr-oct','clim_prec_anom_pct':'Anomalia (%) de lluvia de temporada abr-oct vs media 2022-2024 (referencia debil: 3 anios)','clim_tmax_anom':'Anomalia de Tmax de temporada abr-oct vs media 2022-2024 (C)'}
rows = []
for c in d.columns:
    if c in ('CONJUNTO','RENDIMIENTO_T_HA'):
        rows.append((c, 'objetivo/particion', 'Partición oficial (ENTRENAMIENTO/PREDICCION)' if c=='CONJUNTO' else 'Rendimiento de cebada ciclo abr-oct 2025 (t/ha); vacío en PREDICCION')); continue
    p = c.split('_')
    if p[0] in SEN:
        sensor = SEN[p[0]]; ix = p[1]; rest = '_'.join(p[2:])
        m = re.match(r'(std|min|max)_pico$', rest)
        if rest in WIN: desc = f"Promedio de {NOM[ix]} en la {WIN[rest]}"
        elif rest in PH: desc = f"Fenologia de {NOM[ix]}: {PH[rest]}"
        elif m: desc = f"Heterogeneidad intra-parcela: promedio de {m.group(1)} espacial de {NOM[ix]} en la {WIN['pico']}"
        else: desc = rest
        rows.append((c, f"satelite {sensor}", desc))
    elif c in CL_OVR:
        rows.append((c, 'clima (CHIRPS/CHIRTS, celda 0.05 grados)', CL_OVR[c]))
    elif p[0] == 'clim':
        rows.append((c, 'clima (CHIRPS/CHIRTS, celda 0.05 grados)', {
          'prec':'Precipitacion acumulada (mm)','tmin':'Temperatura minima media (C)','tmax':'Temperatura maxima media (C)','dtr':'Rango termico diurno Tmax-Tmin (C)','tmean':'Temperatura media (C)','gdd0':'Grados-dia base 0 C aprox. (medias mensuales), abr-oct','prec_cv':'Coef. de variacion mensual de lluvia abr-oct','prec_anom_pct':'Anomalia % de lluvia de temporada vs media 2022-2024','tmax_anom':'Anomalia Tmax de temporada vs 2022-2024 (C)'}.get(p[1], p[1]) + {'pre':' - abr-may','veg':' - jun-jul','pico':' - ago-sep','sen':' - oct','temp':' - temporada abr-oct','prev':' - previa nov 2024-mar 2025','minmes':' - mes mas frio de la temporada'}.get(p[2] if len(p)>2 else '', '')))
    elif p[0] == 'topo':
        rows.append((c, 'topografia (INEGI CEM 120 m)', {'elev':'Elevacion (m)','pend':'Pendiente (grados)'}[p[1]] + {'media':', media en el poligono','std':', desviacion estandar en el poligono'}[p[2]]))
    elif p[0] == 'geo':
        rows.append((c, 'geometria de parcela', {'area':'Area (ha)','compacidad':'Compacidad 4*pi*A/P^2 (1 = circulo)','perim':'Perimetro (m)'}[p[1]]))
    elif p[0] == 'loc':
        rows.append((c, 'ubicacion', {'lon':'Longitud centroide (WGS84)','lat':'Latitud centroide (WGS84)','x':'X centroide UTM 14N (m)','y':'Y centroide UTM 14N (m)','municipio':'Municipio','estado':'Estado','celda':'Celda de clima (fila_col) de CHIRPS/CHIRTS'}[p[1]]))
D = pd.DataFrame(rows, columns=['columna','grupo','descripcion']); D.to_csv(f"{W}/diccionario_features.csv", index=False)
print(D.shape); print(D.sample(12, random_state=1).to_string())
assert (D.descripcion.str.len()>3).all() and len(D)==d.shape[1]
