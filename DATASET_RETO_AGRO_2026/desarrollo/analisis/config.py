#Configuración y vocabulario compartido del análisis AgroCebada

import matplotlib
matplotlib.use("Agg")

from config import DATA_ROOT, GEOSPATIAL_DATA_ROOT, OUTPUT_FOLDER

OUTPUT = OUTPUT_FOLDER
EVOLUTION_PATH = OUTPUT / "ml_evolucion_parcelas_2022_2025.csv"
ASSIGNMENT_PATH = DATA_ROOT / "entradas" / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv"
TOPOGRAPHY_ROOT = GEOSPATIAL_DATA_ROOT / "Reto_AgroCebada_Topografia_INEGI_CEM4"

TARGET = "RENDIMIENTO_T_HA"

SEED = 2026

YEARS = (2022, 2023, 2024, 2025)

SPECTRAL_INDICES: dict[str, tuple[str, str]] = {
    "ndvi": (
        "Índice de vegetación de diferencia normalizada (NDVI)",
        "Índices de vegetación",
    ),
    "evi": ("Índice de vegetación mejorado (EVI)", "Índices de vegetación"),
    "crc": (
        "Índice de cobertura de residuos de cultivo (CRC)",
        "Residuos y labranza",
    ),
    "mcrc": (
        "Cobertura modificada de residuos de cultivo (MCRC)",
        "Residuos y labranza",
    ),
    "ndsvi": (
        "Índice de diferencia normalizada de vegetación senescente (NDSVI)",
        "Residuos y labranza",
    ),
    "srndi": (
        "Índice normalizado infrarrojo de onda corta-rojo (SRNDI)",
        "Residuos y labranza",
    ),
    "mra": ("Variable espectral MRA", "Índices espectrales sin expansión"),
    "lai": (
        "Índice de área foliar (LAI)",
        "Estructura del cultivo",
    ),
    "fapar": (
        "Fracción de radiación fotosintéticamente activa absorbida (FAPAR)",
        "Estructura del cultivo",
    ),
    "savi": (
        "Índice de vegetación ajustado al suelo (SAVI)",
        "Índices de vegetación",
    ),
    "vi6t": ("Variable VI6T derivada de Landsat (VI6T)", "Índices espectrales sin expansión"),
    "ndti": ("Índice normalizado de labranza (NDTI)", "Residuos y labranza"),
    "sti": ("Índice simple de labranza (STI)", "Residuos y labranza"),
    "ndwi": ("Índice de agua de diferencia normalizada (NDWI)", "Agua y estrés hídrico"),
    "endwi": (
        "Índice mejorado de agua de diferencia normalizada (ENDWI)",
        "Agua y estrés hídrico",
    ),
    "msi": ("Índice de estrés por humedad (MSI)", "Agua y estrés hídrico"),
    "emsi": (
        "Índice mejorado de estrés por humedad (EMSI)",
        "Agua y estrés hídrico",
    ),
    "dswi2": ("Índice de estrés enfermedad-agua 2 (DSWI2)", "Agua y estrés hídrico"),
    "dswi3": ("Índice de estrés enfermedad-agua 3 (DSWI3)", "Agua y estrés hídrico"),
    "dswi4": ("Índice de estrés enfermedad-agua 4 (DSWI4)", "Agua y estrés hídrico"),
    "dswi5": ("Índice de estrés enfermedad-agua 5 (DSWI5)", "Agua y estrés hídrico"),
    "nddi": ("Índice de sequedad de diferencia normalizada (NDDI)", "Agua y estrés hídrico"),
    "enddi": (
        "Índice mejorado de sequedad de diferencia normalizada (ENDDI)",
        "Agua y estrés hídrico",
    ),
    "msavi": (
        "Índice de vegetación ajustado al suelo modificado (MSAVI)",
        "Índices de vegetación",
    ),
}

CLASS_NOTES = {
    "Identificación y referencia": "Claves, partición, fuente y atributos auxiliares para relacionar las observaciones. ID_POLIGONO identifica una parcela y puede repetirse entre fechas.",
    "Tiempo de captura": "Fecha y componentes de calendario de cada observación. Los años/meses son temporales, no resultados del cultivo.",
    "Índices de vegetación": "Índices satelitales asociados con vigor o cobertura vegetal; las columnas de estadísticos resumen la parcela en una captura.",
    "Estructura del cultivo": "Variables biofísicas asociadas con superficie foliar y radiación absorbida.",
    "Agua y estrés hídrico": "Índices satelitales asociados con agua, humedad, sequedad o estrés; no son mediciones directas de rendimiento.",
    "Residuos y labranza": "Índices asociados con residuos vegetales, vegetación senescente o condiciones de labranza.",
    "Índices espectrales sin expansión": "Abreviaturas cuya expansión completa no está respaldada por los metadatos consultados; se conserva el identificador sin inventar su significado.",
    "Precipitación": "Precipitación mensual acumulada CHIRPS en milímetros por mes.",
    "Temperatura": "Promedio mensual de temperatura mínima o máxima CHIRTS-ERA5 en grados Celsius.",
    "Topografía": "Atributos estáticos extraídos del ráster INEGI por polígono: elevación media y pendiente media.",
    "Calidad de observación": "Porcentaje de nubosidad y variables que describen calidad/cobertura de la captura.",
    "Variable objetivo": "Rendimiento etiquetado usado como respuesta para entrenar la transferencia a 2026; no es una variable predictora.",
}