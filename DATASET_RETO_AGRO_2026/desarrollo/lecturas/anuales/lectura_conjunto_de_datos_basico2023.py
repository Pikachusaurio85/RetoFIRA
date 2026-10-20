from pathlib import Path
from lecturas.dataset_readers import run_dataset_reader

def run_basico_2023() -> Path:
    return run_dataset_reader("basico", 2023)

def main() -> None:
    run_basico_2023()

if __name__ == "__main__":
    main()