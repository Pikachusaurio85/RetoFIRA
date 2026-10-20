import argparse
from collections.abc import Sequence
from analisis.pipeline import main as run_analysis
from analisis.comparison import run_comparison
from lecturas.anuales.lectura_conjunto_de_datos_basico2022 import run_basico_2022
from lecturas.anuales.lectura_conjunto_de_datos_basico2023 import run_basico_2023
from lecturas.anuales.lectura_conjunto_de_datos_basico2024 import run_basico_2024
from lecturas.anuales.lectura_conjunto_de_datos_basico2025 import run_basico_2025
from lecturas.anuales.lectura_conjunto_de_datos_pro2025 import run_pro_2025
from lecturas.csv_reader import run_initial_reader
from lecturas.dataset_readers import DATASET_YEARS
from procesamiento.ml_dataset import run_machine_learning_export

YEAR_RUNNERS = {
    ("basico", 2022): run_basico_2022,
    ("basico", 2023): run_basico_2023,
    ("basico", 2024): run_basico_2024,
    ("basico", 2025): run_basico_2025,
    ("pro", 2025): run_pro_2025,
}

def build_jobs(
    dataset: str | None,
    year: int | None,
    include_initial: bool,
) -> list[tuple[str, int | None]]:
    #Valida la selección del usuario y prepara los lectores en orden
    if year is not None and dataset is None:
        raise ValueError("--year requiere indicar --dataset.")

    jobs: list[tuple[str, int | None]] = []
    if include_initial:
        jobs.append(("inicial", None))

    selected_datasets = [dataset] if dataset else list(DATASET_YEARS)
    for dataset_name in selected_datasets:
        if year is not None:
            if year not in DATASET_YEARS[dataset_name]:
                available = ", ".join(map(str, DATASET_YEARS[dataset_name]))
                raise ValueError(
                    f"Año {year} no disponible para {dataset_name}; "
                    f"años válidos: {available}."
                )
            jobs.append((dataset_name, year))
        else:
            jobs.extend(
                (dataset_name, available_year)
                for available_year in DATASET_YEARS[dataset_name]
            )
    return jobs

def main(argv: Sequence[str] | None = None) -> None:
    #Orquesta lecturas, exportación ML y análisis según los argumentos
    parser = argparse.ArgumentParser(
        description="Procesa AgroCebada y genera salidas reproducibles."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_YEARS),
        help="Ejecutar solo el conjunto BASIC o PRO.",
    )
    parser.add_argument(
        "--year",
        type=int,
        choices=range(2022, 2026),
        help="Limitar la ejecución a un año; requiere --dataset.",
    )
    parser.add_argument(
        "--initial",
        action="store_true",
        help="Incluir también la lectura inicial de predicciones.",
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Generar las tablas de entrenamiento, predicción y evolución por parcela.",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Evaluar etiquetas disponibles y la transferencia al objetivo 2026.",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help=(
            "Regenerar desarrollo y comparar ambos pipelines con evaluación "
            "espacial externa común."
        ),
    )
    args = parser.parse_args(argv)

    has_reader_selection = args.dataset is not None or args.year is not None
    run_all = (
        not has_reader_selection
        and not args.initial
        and not args.ml
        and not args.analysis
        and not args.comparison
    )
    if (
        (args.ml or args.analysis or args.comparison)
        and not has_reader_selection
        and not args.initial
    ):
        jobs = []
    elif args.initial and not has_reader_selection:
        jobs = [("inicial", None)]
    else:
        include_initial = args.initial or run_all
        jobs = build_jobs(args.dataset, args.year, include_initial)
    for dataset_name, year in jobs:
        if dataset_name == "inicial":
            run_initial_reader()
        else:
            YEAR_RUNNERS[(dataset_name, year)]()

    run_ml = args.ml or args.comparison or run_all
    if run_ml:
        run_machine_learning_export()
    if args.analysis or args.comparison or run_all:
        if not run_ml:
            run_machine_learning_export()
        run_analysis()
    if args.comparison or run_all:
        run_comparison()

if __name__ == "__main__":
    main()