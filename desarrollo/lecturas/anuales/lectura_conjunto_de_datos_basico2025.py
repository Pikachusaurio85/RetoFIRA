from pathlib import Path
from lecturas.dataset_readers import run_dataset_reader

def run_basico_2025() -> Path:
    return run_dataset_reader("basico", 2025)

def main() -> None:
    run_basico_2025()

if __name__ == "__main__":
    main()