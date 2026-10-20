# DATASET_RETO_AGRO_2026

Este repositorio reúne el flujo principal de AgroCebada, los datos fuente y derivados, y varios análisis y artefactos relacionados. Este archivo sirve como índice de navegación; las instrucciones específicas de instalación y ejecución están en la documentación de cada flujo.

## Mapa del repositorio

| Ruta | Contenido |
|---|---|
| [`desarrollo/`](desarrollo/) | Flujo principal en Python. Incluye `main.py`, `config.py`, `requirements.txt`, lectores, procesamiento, análisis, datos y salidas. |
| [`desarrollo/README.md`](desarrollo/README.md) | Descripción del flujo principal, argumentos, dependencias y advertencias de ejecución. |
| [`desarrollo/datos/`](desarrollo/datos/) | CSV y documentos de entrada, capas geoespaciales y rásteres climáticos/topográficos. |
| [`desarrollo/salidas/`](desarrollo/salidas/) | Productos derivados del flujo principal y copias de fuentes empleadas. |
| [`desarrollo/cebada_datos_baseline_1/`](desarrollo/cebada_datos_baseline_1/) | Pipeline de baseline, datos procesados, scripts, resultados y su propio README. Los cuatro modelos independientes se mantienen en [`modelos_experimentales/`](desarrollo/cebada_datos_baseline_1/modelos_experimentales/). |
| [`desarrollo/analisis_xai_2025/`](desarrollo/analisis_xai_2025/) | Análisis XAI paralelo y documentación asociada. |
| [`experiments/bayesian_hierarchical/`](experiments/bayesian_hierarchical/) | Experimento bayesiano independiente, con configuración, código fuente, pruebas, resultados, informes y runners por fase. Consulte su README antes de ejecutarlo. |
| [`outputs/predictor/`](outputs/predictor/) | Código, configuración, documentación y artefactos del módulo predictor. |
| [`outputs/xai/`](outputs/xai/) | Código, documentación y artefactos relacionados con conceptos XAI. |
| [`docs/anexo_metodologico/`](docs/anexo_metodologico/) | Scripts, tablas y documentación metodológica de comparaciones y análisis bayesianos. |
| [`catboost_info/`](catboost_info/) | Registros de entrenamiento de CatBoost presentes en esta copia. |

## Puntos de entrada y dependencias

- El flujo principal comienza en [`desarrollo/main.py`](desarrollo/main.py). Sus opciones y comandos están documentados en [`desarrollo/README.md`](desarrollo/README.md).
- El baseline tiene instrucciones propias en [`desarrollo/cebada_datos_baseline_1/README.md`](desarrollo/cebada_datos_baseline_1/README.md).
- El experimento bayesiano tiene runners por fase y documentación propia en [`experiments/bayesian_hierarchical/README.md`](experiments/bayesian_hierarchical/README.md).
- Las dependencias se declaran por flujo, no en un manifiesto único: por ejemplo, [`desarrollo/requirements.txt`](desarrollo/requirements.txt), [`outputs/predictor/requirements.txt`](outputs/predictor/requirements.txt) y [`outputs/xai/environment.yml`](outputs/xai/environment.yml).

## Nota de organización

Se conservan las ubicaciones actuales. Los módulos construyen rutas relativas a sus directorios de proyecto y algunos experimentos leen carpetas hermanas desde la raíz; mover datos, código o resultados sin revisar esas referencias podría romper la ejecución. Los archivos y carpetas de resultados, caché y registros se mantienen intactos.
