from pathlib import Path
from lecturas.dataset_readers import run_dataset_reader

def run_basico_2024() -> Path:
    return run_dataset_reader("basico", 2024)

def main() -> None:
    run_basico_2024()

if __name__ == "__main__":
    main()