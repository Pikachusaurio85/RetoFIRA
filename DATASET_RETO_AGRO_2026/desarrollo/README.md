# AgroCebada

Este proyecto transforma las fuentes geoespaciales y tabulares de AgroCebada en tablas normalizadas, variables climáticas por parcela, conjuntos para aprendizaje automático y un informe con análisis descriptivo y evaluación espacial. La ejecución normal comienza en `main.py`; los demás módulos se agrupan por responsabilidad para que cada paso pueda leerse, ejecutarse y mantenerse por separado.

## Mapa de carpetas

El flujo principal, sus datos y el paquete de comparación se agrupan bajo `desarrollo`. En la raíz también se conservan documentación metodológica (`docs/`), el experimento bayesiano independiente (`experiments/`), recursos del predictor y XAI (`outputs/`) y registros de CatBoost (`catboost_info/`). El mapa general está en el README de la raíz. Estas carpetas se mantienen en su ubicación porque los proyectos contienen referencias entre rutas.

```text
DATASET_RETO_AGRO_2026/
├── docs/                    # anexo metodológico
├── experiments/             # experimentos independientes
├── outputs/                 # recursos y artefactos predictor/XAI
├── catboost_info/           # registros de entrenamiento presentes
└── desarrollo/              # flujo principal, datos y salidas
    ├── main.py              # punto de entrada principal
    ├── config.py            # rutas del flujo
    ├── requirements.txt
    ├── datos/
    │   ├── entradas/        # CSV y documentos fuente
    │   └── geoespaciales/   # shapefile y rásteres
    ├── cebada_datos_baseline_1/
    ├── lecturas/
    ├── procesamiento/
    ├── analisis/
    └── salidas/             # productos derivados
```

### Qué hace cada carpeta y módulo

#### `desarrollo/`

Contiene código Python, dependencias, documentación, fuentes originales, productos derivados del baseline y resultados del flujo. Los módulos usan rutas construidas desde `config.py`, así que pueden localizar datos y salidas aunque el comando se ejecute desde otro directorio.

#### `desarrollo/main.py`

Es el **orquestador y punto de entrada**. Interpreta los argumentos de consola, valida la combinación de dataset/año, llama a los lanzadores anuales correspondientes y, cuando procede, encadena la exportación ML y el análisis. Sin argumentos ejecuta el flujo completo en orden: lectura inicial, BASIC 2022–2025, PRO 2025, tablas ML y análisis.

#### `desarrollo/config.py`

Centraliza las rutas compartidas de datos y salida de resultados. Mantenerlas aquí evita rutas absolutas locales y diferencias según la carpeta actual de la terminal.

#### `desarrollo/lecturas/`

Lee las tablas originales y prepara las observaciones BASIC/PRO asociadas a los polígonos de predicción.

- `csv_reader.py`: localiza el archivo de asignación de parcelas en CSV/Excel, selecciona `CONJUNTO = PREDICCION`, valida los identificadores, normaliza la tabla inicial de parcelas y escribe CSV.
- `dataset_readers.py`: contiene la implementación común de lectura por año. Verifica dataset y año, identifica columnas por su nombre normalizado, filtra capturas por `ID_POLIGONO` y fecha, informa incidencias en consola, añade los atributos climáticos del año y exporta las tablas procesadas.
- `anuales/`: tiene lanzadores pequeños para cada conjunto-año disponible: BASIC 2022, 2023, 2024 y 2025, además de PRO 2025. Cada uno fija la combinación y delega en `dataset_readers.py`. La lógica común **no se copia** entre años.

#### `desarrollo/procesamiento/`

Contiene las transformaciones reutilizables que preparan las variables y las tablas para análisis/modelado.

- `data_normalization.py`: utilidades para reconocer faltantes, normalizar texto y convertir números/fechas.
- `climate_features.py`: encuentra las geometrías del shapefile por `ID_POLIGONO`, reproyecta cada geometría al CRS de cada ráster y calcula promedios zonales mensuales de precipitación y temperatura. Omite celdas NoData; errores de CRS, archivo, geometría o ausencia de celdas válidas se notifican explícitamente.
- `ml_dataset.py`: combina BASIC y PRO por parcela y captura, adjunta clima mensual y metadatos de fecha/fuente/sensor, mantiene la etiqueta solo donde está disponible y genera las tablas longitudinal, de entrenamiento y de predicción.

#### `desarrollo/analisis/`

Implementa el análisis reproducible que consume la tabla longitudinal `salidas/ml_evolucion_parcelas_2022_2025.csv`. Los años de esos nombres identifican el período real de observaciones, no el año objetivo de una estimación.

- `config.py`: define constantes y vocabulario del análisis, nombres de salida, semilla, años, notas sobre clases e índices satelitales.
- `data.py`: audita filas/columnas, consolida características históricas a nivel parcela, extrae elevación y pendiente desde rásteres topográficos, carga asignaciones y preserva hashes/copia de fuentes.
- `organization.py`: asigna clases y etiquetas descriptivas a las variables, crea el diccionario y escribe el Excel de presentación y el CSV reorganizado sin columnas separadoras.
- `visualization.py`: genera resúmenes descriptivos por año, gráficas climáticas/topográficas y de evaluación, más el CSV de interpretación en lenguaje llano.
- `modeling.py`: prepara predictores sin `ID_POLIGONO`, crea particiones espaciales por parcela, compara modelos con una línea base, mide errores y sesgo, revisa el dominio y solo publica predicciones reservadas si los criterios de transferencia se satisfacen.
- `report.py`: produce el informe ejecutivo a partir de métricas, diagnóstico y resultados existentes.
- `pipeline.py`: coordina las etapas anteriores y es el punto de entrada interno del análisis.

`analisis_modelado_2026.py` en la raíz de `desarrollo` es un lanzador compatible; para uso normal se recomienda `main.py`.

#### `desarrollo/salidas/`

Contiene productos generados, no fuentes originales editables. Incluye CSV anuales, tablas de evolución/entrenamiento/predicción, tabla parcelaria, Excel organizado, métricas, evaluación espacial, estado de transferencia al objetivo 2026, interpretaciones, informes y gráficas. No se generan reportes separados de calidad. `originales/` guarda copias verificables de los CSV fuente empleados y `trazabilidad_archivos.csv` registra huellas SHA-256.

No todos los CSV son redundantes: algunos son entradas para una etapa posterior y otros son entregables para inspección/reproducibilidad. No se deben borrar manualmente para “limpiar” el flujo a menos que se confirme antes que no son fuente, dependencia ni producto requerido.

#### Datos fuente

`datos/entradas/` conserva los CSV de observaciones/asignación y los documentos DOCX originales. `datos/geoespaciales/` contiene el shapefile de parcelas, los rásteres mensuales de precipitación y temperatura y los rásteres estáticos de elevación y pendiente. Son entradas; el flujo no debe modificarlas.

## Funcionamiento general

1. `main.py` resuelve el flujo completo o la etapa indicada por argumentos.
2. El lector inicial identifica los polígonos `PREDICCION`; los lectores anuales seleccionan capturas de BASIC/PRO por fecha y parcela.
3. Los módulos de normalización preservan texto/identificadores y registran faltantes o formatos inválidos. El enriquecimiento climático calcula promedios zonales de los rásters mensuales usando las geometrías y CRS.
4. `ml_dataset.py` conserva la granularidad de captura en la tabla de evolución y produce una tabla de entrenamiento etiquetada y otra de predicción sin etiqueta.
5. El análisis consume la tabla de evolución, resume variables por parcela/año, extrae atributos topográficos, crea el diccionario/Excel/CSV, produce gráficas y evalúa transferencia espacial.
6. El diagnóstico, las predicciones del bloque espacial y el informe se escriben en `salidas/`. No hay datos observados de 2026. La fuente canónica de particiones y etiquetas es `ID_area_rendimiento_70_30_Reto_AgroCebada.csv`: las 138 parcelas `ENTRENAMIENTO` tienen `RENDIMIENTO_T_HA`; las 59 parcelas `PREDICCION` no. La fuente no especifica el ciclo de esos rendimientos.

La repetición de `ID_POLIGONO` entre capturas es longitudinal y no se elimina como duplicado. Para modelar, se divide por parcelas, no por filas de captura. No hay etiquetas anuales verificables de rendimiento para 2022–2025, por lo que esos años se describen pero no se presentan como backtest de rendimiento.

## Instalación y ejecución

### Crear un entorno reproducible

Desde la carpeta del proyecto, crea un entorno virtual y luego instala las dependencias declaradas:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\desarrollo\requirements.txt
```

Si ya tienes un Python y un entorno configurados, omite la creación/activación y ejecuta la instalación desde `desarrollo` con `python -m pip install -r .\requirements.txt`.

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
python main.py --comparison
python main.py --help
```

- `--dataset` permite seleccionar `basico` o `pro`.
- `--year` limita la lectura al año solicitado y requiere `--dataset`.
- `--initial` añade la exportación inicial de parcelas.
- `--ml` genera las tablas longitudinal, de entrenamiento y predicción.
- `--analysis` regenera primero la base ML si es necesario y ejecuta el análisis.
- `--comparison` regenera desarrollo y ejecuta la comparación parcelaria con `cebada_datos_baseline_1` bajo bloques espaciales externos comunes.
- Sin argumentos se ejecutan todas las etapas disponibles.

Los lanzadores anuales también se pueden ejecutar como módulo desde `desarrollo`, por ejemplo `python -m lecturas.anuales.lectura_conjunto_de_datos_basico2022`.

## Dependencias y archivos auxiliares

`requirements.txt` enumera las librerías de lectura, análisis espacial, Excel, gráficas y modelos. `.venv/` (si se crea en la raíz del proyecto) es un entorno local reinstalable: contiene el intérprete/paquetes, no los datos ni el código. Puedes eliminarlo si ya no lo necesitas; para volver a ejecutar después tendrás que seleccionar otro Python o crearlo de nuevo e instalar `desarrollo/requirements.txt`.

`catboost_info/`, cuando se genera, contiene archivos de diagnóstico/registro del entrenamiento CatBoost (por ejemplo errores por iteración y tiempo restante). No es una fuente de datos ni lo consume el flujo posterior: se puede borrar si no necesitas inspeccionar el historial de entrenamiento. Una ejecución futura de CatBoost puede volver a generarlo.

## Normalización y clima

Los números con coma decimal se serializan con punto decimal; las fechas válidas se normalizan a `AAAA-MM-DD`. Los valores faltantes y numéricos/temporales inválidos se validan durante la lectura y las incidencias se informan en consola; no se generan reportes separados de calidad. Una fecha inválida no puede asignarse con seguridad a un año, por lo que esa observación se omite del lector anual.

BASIC está disponible para 2022–2025 y PRO para 2025. El nombre heredado `Conjunto_datos_BASICO_AgroCebada2026.csv` no significa que contenga capturas de 2026; el lector admite años hasta 2025. Los CSV anuales de lector se limitan a parcelas de predicción y añaden precipitación mensual en mm/mes y Tmin/Tmax en °C. Si falta un ráster, CRS, geometría o conjunto de celdas válidas, el procesamiento falla con un error explícito.

## Momento de estimación y disponibilidad temporal

El horizonte de transferencia es **antes de la siembra 2026**; esto no significa que existan observaciones del ciclo 2026. Los predictores disponibles tienen observaciones hasta 2025. La fuente canónica de rendimiento y partición es `ID_area_rendimiento_70_30_Reto_AgroCebada.csv`; sus rendimientos etiquetados no llevan año/ciclo en esa tabla.

| Momento de estimación | Features permitidas según la información disponible | Features excluidas / uso |
|---|---|---|
| Antes de la siembra 2026 (horizonte de transferencia) | Ubicación, geometría y topografía estáticas; imágenes y curvas satelitales de 2025; clima observado hasta 2025; resúmenes históricos 2022–2025. | No hay datos observados de 2026. Cualquier estimación para ese objetivo sigue siendo exploratoria y requiere validación posterior. |
| Durante un ciclo futuro | Features estáticas e históricas; clima observado únicamente hasta una fecha de corte y pronósticos meteorológicos disponibles en esa fecha. | No se incorporan observaciones posteriores al corte. Antes de ejecutar, fijar fecha y fuente de pronóstico. |
| Al final de un ciclo futuro | Features del ciclo observadas hasta el cierre definido, para estimación retrospectiva o monitoreo tardío. | No describe un pronóstico pre-siembra; requiere etiquetas del ciclo correspondiente para validación temporal. |

Esta política usa el alcance documentado de BASIC (2022–2025), PRO (2025) y el clima disponible hasta 2025. Las fechas de captura, el rendimiento etiquetado sin ciclo explícito en la tabla fuente y el horizonte objetivo se conservan como conceptos distintos.

## Tablas ML y alcance estadístico

La columna objetivo es `RENDIMIENTO_T_HA`, leída de la tabla canónica `ID_area_rendimiento_70_30_Reto_AgroCebada.csv`. Hay 138 parcelas `ENTRENAMIENTO` con rendimiento y 59 parcelas `PREDICCION` sin etiqueta. La fuente no declara el año/ciclo de esos rendimientos. La tabla de evolución conserva múltiples filas por parcela y la clave de parcela; el modelo no utiliza `ID_POLIGONO` como predictor.

El nombre de los productos distingue el dato del horizonte: `ml_evolucion_parcelas_2022_2025.csv` identifica observaciones fechadas; `ml_entrenamiento_etiquetado.csv`, `ml_prediccion_sin_etiqueta.csv` y `tabla_modelado_parcelaria.csv` no asignan año a la etiqueta; únicamente los resultados de transferencia llevan el sufijo `transferencia_objetivo_2026`.

Las gráficas 2022–2025 son descriptivas: no existen etiquetas verificables de rendimiento anual para construir un backtest temporal y no hay datos observados de 2026. La evaluación espacial de etiquetas disponibles sirve para revisar generalización entre parcelas; no valida temporalmente el objetivo 2026. Solo se llenan las estimaciones reservadas si se satisfacen los criterios conservadores de métricas, sesgo y cobertura; de lo contrario, permanecen vacías. La importancia por permutación informa sobre contribución predictiva en una prueba espacial; no demuestra causalidad ni garantiza generalización.

Para reconstruir baseline_1 desde las fuentes, desde `desarrollo` ejecuta primero `python .\cebada_datos_baseline_1\scripts\run_pipeline.py` (consulta su README para las rutas). Después ejecuta `python main.py --comparison` para regenerar la tabla de desarrollo y la comparación unificada. Se producen la tabla maestra parcelaria, el diccionario de features, hashes de fuentes, resultados por fold/configuración/municipio, predicciones únicamente para holdouts etiquetados y el informe. La tabla de asignación es la referencia de verdad para partición y etiquetas; los candidatos comparten parcelas, etiquetas y bloques; la selección se hace en validación espacial interna y el bloque externo se reserva. Las 59 parcelas sin etiqueta nunca eligen el modelo. Los criterios predefinidos y sus resultados no validan temporalmente el objetivo 2026.