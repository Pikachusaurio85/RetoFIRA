# AgroCebada

Este proyecto transforma las fuentes geoespaciales y tabulares de AgroCebada en tablas normalizadas, variables climáticas por parcela, conjuntos para aprendizaje automático y un informe con análisis descriptivo y evaluación espacial. La ejecución normal comienza en `main.py`; los demás módulos se agrupan por responsabilidad para que cada paso pueda leerse, ejecutarse y mantenerse por separado.

## Mapa de carpetas

La carpeta de trabajo se llama `desarrollo` (minúsculas). Las carpetas de datos grandes se mantienen en la raíz del proyecto para conservar su ubicación original.

```text
DATASET_RETO_AGRO_2026/
├── desarrollo/
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   ├── lecturas/
│   │   ├── csv_reader.py
│   │   ├── dataset_readers.py
│   │   └── anuales/
│   ├── procesamiento/
│   │   ├── data_normalization.py
│   │   ├── climate_features.py
│   │   └── ml_dataset.py
│   ├── analisis/
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── organization.py
│   │   ├── visualization.py
│   │   ├── modeling.py
│   │   ├── report.py
│   │   └── pipeline.py
│   └── salidas/
├── Parcelas_Reto_AGC_CONJUNTO_70_30/
├── Reto_AgroCebada_CHIRPS_Precipitacion_2022_2025/
├── Reto_AgroCebada_Temperatura_2022_2025/
├── Reto_AgroCebada_Topografia_INEGI_CEM4/
├── Conjunto_datos_BASICO_AgroCebada2026.csv
├── Conjunto_datos_PRO_AgroCebada.csv
├── ID_area_rendimiento_70_30_Reto_AgroCebada.csv
├── Descripcion_general_variables_DataSet.docx
└── Diccionario_de_indices_satelitales.docx
```

### Qué hace cada carpeta y módulo

#### `desarrollo/`

Contiene el código Python, las dependencias, documentación y resultados derivados. No es la carpeta donde residen los datos de origen. Los módulos usan rutas construidas desde `config.py`, así que pueden localizar las fuentes en la carpeta raíz aunque el comando se ejecute desde otro directorio.

#### `desarrollo/main.py`

Es el **orquestador y punto de entrada**. Interpreta los argumentos de consola, valida la combinación de dataset/año, llama a los lanzadores anuales correspondientes y, cuando procede, encadena la exportación ML y el análisis. Sin argumentos ejecuta el flujo completo en orden: lectura inicial, BASIC 2022–2025, PRO 2025, tablas ML y análisis.

#### `desarrollo/config.py`

Centraliza las rutas compartidas de raíz del proyecto, salida de resultados y shapefile de parcelas. Mantenerlas aquí evita rutas absolutas locales y diferencias según la carpeta actual de la terminal.

#### `desarrollo/lecturas/`

Lee las tablas originales y prepara las observaciones BASIC/PRO asociadas a los polígonos de predicción.

- `csv_reader.py`: localiza el archivo de asignación de parcelas en CSV/Excel, selecciona `CONJUNTO = PREDICCION`, valida los identificadores, normaliza la tabla inicial de parcelas y escribe CSV.
- `dataset_readers.py`: contiene la implementación común de lectura por año. Verifica dataset y año, identifica columnas por su nombre normalizado, filtra capturas por `ID_POLIGONO` y fecha, registra incidencias de calidad, añade los atributos climáticos del año y exporta la tabla y su reporte.
- `anuales/`: tiene lanzadores pequeños para cada conjunto-año disponible: BASIC 2022, 2023, 2024 y 2025, además de PRO 2025. Cada uno fija la combinación y delega en `dataset_readers.py`. La lógica común **no se copia** entre años.

#### `desarrollo/procesamiento/`

Contiene las transformaciones reutilizables que preparan las variables y las tablas para análisis/modelado.

- `data_normalization.py`: utilidades para reconocer faltantes, normalizar texto, convertir números/fechas y acumular reportes de calidad.
- `climate_features.py`: encuentra las geometrías del shapefile por `ID_POLIGONO`, reproyecta cada geometría al CRS de cada ráster y calcula promedios zonales mensuales de precipitación y temperatura. Omite celdas NoData; errores de CRS, archivo, geometría o ausencia de celdas válidas se notifican explícitamente.
- `ml_dataset.py`: combina BASIC y PRO por parcela y captura, adjunta clima mensual y metadatos de fecha/fuente/sensor, mantiene la etiqueta solo donde está disponible y genera las tablas longitudinal, de entrenamiento, de predicción y calidad.

#### `desarrollo/analisis/`

Implementa el análisis reproducible que consume la tabla longitudinal `salidas/ml_evolucion_parcelas_2026.csv`. Los módulos se dividen para que la lectura/preparación, la exportación, las gráficas, el modelo y el informe no estén mezclados.

- `config.py`: define constantes y vocabulario del análisis, nombres de salida, semilla, años, notas sobre clases e índices satelitales.
- `data.py`: audita filas/columnas, consolida características históricas a nivel parcela, extrae elevación y pendiente desde rásteres topográficos, carga asignaciones y preserva hashes/copia de fuentes.
- `organization.py`: asigna clases y etiquetas descriptivas a las variables, crea el diccionario y escribe el Excel de presentación y el CSV reorganizado sin columnas separadoras.
- `visualization.py`: genera resúmenes descriptivos por año, gráficas climáticas/topográficas y de evaluación, más el CSV de interpretación en lenguaje llano.
- `modeling.py`: prepara predictores sin `ID_POLIGONO`, crea particiones espaciales por parcela, compara modelos con una línea base, mide errores y sesgo, revisa el dominio y solo publica predicciones reservadas si los criterios de transferencia se satisfacen.
- `report.py`: produce el informe ejecutivo a partir de métricas, diagnóstico y resultados existentes.
- `pipeline.py`: coordina las etapas anteriores y es el punto de entrada interno del análisis.

`analisis_modelado_2026.py` en la raíz de `desarrollo` es un lanzador compatible; para uso normal se recomienda `main.py`.

#### `desarrollo/salidas/`

Contiene productos generados, no fuentes originales editables. Incluye CSV anuales y de calidad, las tres tablas ML, Excel organizado, CSV limpio para herramientas, auditorías, métricas, predicciones de prueba, estado de estimación 2026, interpretaciones, informe y gráficas. `originales/` guarda copias verificables de los CSV fuente empleados y `trazabilidad_archivos.csv` registra huellas SHA-256.

No todos los CSV son redundantes: algunos son entradas para una etapa posterior y otros son entregables para inspección/reproducibilidad. No se deben borrar manualmente para “limpiar” el flujo a menos que se confirme antes que no son fuente, dependencia ni producto requerido.

#### Carpetas de datos en la raíz

- `Parcelas_Reto_AGC_CONJUNTO_70_30/`: geometrías y archivos auxiliares del shapefile; define las parcelas y su CRS.
- `Reto_AgroCebada_CHIRPS_Precipitacion_2022_2025/`: rásteres mensuales de precipitación CHIRPS.
- `Reto_AgroCebada_Temperatura_2022_2025/`: rásteres mensuales de Tmin/Tmax CHIRTS-ERA5.
- `Reto_AgroCebada_Topografia_INEGI_CEM4/`: elevación y pendiente estáticas.
- CSV y documentos DOCX de la raíz: tablas de observaciones/asignaciones y metadatos/diccionarios del reto. Son entradas; el flujo no debe modificarlos.

## Funcionamiento general

1. `main.py` resuelve el flujo completo o la etapa indicada por argumentos.
2. El lector inicial identifica los polígonos `PREDICCION`; los lectores anuales seleccionan capturas de BASIC/PRO por fecha y parcela.
3. Los módulos de normalización preservan texto/identificadores y registran faltantes o formatos inválidos. El enriquecimiento climático calcula promedios zonales de los rásters mensuales usando las geometrías y CRS.
4. `ml_dataset.py` conserva la granularidad de captura en la tabla de evolución y produce una tabla de entrenamiento etiquetada y otra de predicción sin etiqueta.
5. El análisis consume la tabla de evolución, resume variables por parcela/año, extrae atributos topográficos, crea el diccionario/Excel/CSV, audita calidad, produce gráficas y evalúa transferencia espacial.
6. El diagnóstico, las predicciones históricas del bloque espacial y el informe se escriben en `salidas/`. La etiqueta disponible se atribuye a 2025 según los metadatos generales, aunque algunos nombres de archivo usan 2026; la discrepancia se comunica en los resultados.

La repetición de `ID_POLIGONO` entre capturas es longitudinal y no se elimina como duplicado. Para modelar, se divide por parcelas, no por filas de captura. No hay etiquetas anuales verificables de rendimiento para 2022–2025, por lo que esos años se describen pero no se presentan como backtest de rendimiento.

## Instalación y ejecución

### Crear un entorno reproducible

Desde la carpeta del proyecto, crea un entorno virtual y luego instala las dependencias declaradas:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\desarrollo\requirements.txt
```

Si ya tienes un Python y un entorno configurados, omite la creación/activación y ejecuta la instalación desde la raíz con `python -m pip install -r .\desarrollo\requirements.txt`.

### Ejecutar

Desde `desarrollo`, la cadena completa es:

```powershell
python main.py
```

Para limitarla:

```powershell
python main.py --dataset basico --year 2022
python main.py --dataset pro --year 2025 --initial
python main.py --initial
python main.py --ml
python main.py --analysis
python main.py --help
```

- `--dataset` permite seleccionar `basico` o `pro`.
- `--year` limita la lectura al año solicitado y requiere `--dataset`.
- `--initial` añade la exportación inicial de parcelas.
- `--ml` genera las tablas longitudinal, de entrenamiento y predicción.
- `--analysis` regenera primero la base ML si es necesario y ejecuta el análisis.
- Sin argumentos se ejecutan todas las etapas disponibles.

Los lanzadores anuales también se pueden ejecutar como módulo desde `desarrollo`, por ejemplo `python -m lecturas.anuales.lectura_conjunto_de_datos_basico2022`.

## Dependencias y archivos auxiliares

`requirements.txt` enumera las librerías de lectura, análisis espacial, Excel, gráficas y modelos. `.venv/` (si se crea en la raíz del proyecto) es un entorno local reinstalable: contiene el intérprete/paquetes, no los datos ni el código. Puedes eliminarlo si ya no lo necesitas; para volver a ejecutar después tendrás que seleccionar otro Python o crearlo de nuevo e instalar `desarrollo/requirements.txt`.

`catboost_info/`, cuando se genera, contiene archivos de diagnóstico/registro del entrenamiento CatBoost (por ejemplo errores por iteración y tiempo restante). No es una fuente de datos ni lo consume el flujo posterior: se puede borrar si no necesitas inspeccionar el historial de entrenamiento. Una ejecución futura de CatBoost puede volver a generarlo.

## Normalización, calidad y clima

Los números con coma decimal se serializan con punto decimal; las fechas válidas se normalizan a `AAAA-MM-DD`. Los valores faltantes y numéricos/temporales inválidos se registran en los reportes de calidad. Una fecha inválida no puede asignarse con seguridad a un año, por lo que esa observación se omite del lector anual.

BASIC está disponible para 2022–2025 y PRO para 2025. Los CSV anuales de lector se limitan a parcelas de predicción y añaden precipitación mensual en mm/mes y Tmin/Tmax en °C. Si falta un ráster, CRS, geometría o conjunto de celdas válidas, el procesamiento falla con un error explícito.

## Tablas ML y alcance estadístico

La columna objetivo es `RENDIMIENTO_T_HA`. La descripción general atribuye las etiquetas conocidas de 138 parcelas al ciclo abril–octubre de 2025, aunque algunos nombres de archivo/código usan 2026; esta discrepancia permanece documentada. La tabla de evolución conserva múltiples filas por parcela y la clave de parcela; el modelo no utiliza `ID_POLIGONO` como predictor.

Las gráficas 2022–2025 son descriptivas: no existen etiquetas verificables de rendimiento anual para construir un backtest temporal. La transferencia a 2026 se evalúa por bloques espaciales de parcelas y se compara con una línea base y CatBoost/XGBoost/LightGBM. Solo se llenan las estimaciones de las parcelas reservadas si se satisfacen los criterios conservadores de métricas, sesgo y cobertura descritos en el diagnóstico. De lo contrario, las estimaciones permanecen vacías y no se presentan como pronóstico fiable. La importancia por permutación informa sobre contribución predictiva en una prueba espacial; no demuestra causalidad ni garantiza generalización.