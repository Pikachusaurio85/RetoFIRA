from pathlib import Path
from lecturas.dataset_readers import run_dataset_reader

def run_pro_2025() -> Path:
    return run_dataset_reader("pro", 2025)

def main() -> None:
    run_pro_2025()

if __name__ == "__main__":
    main()