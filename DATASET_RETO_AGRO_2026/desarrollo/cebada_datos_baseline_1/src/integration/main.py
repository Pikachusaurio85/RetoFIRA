"""Punto de entrada CLI para la integración AgroCebada 2026."""
from __future__ import annotations
import sys
from pathlib import Path

from .build import build
from .validate import validate_all


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        errors = []
        # Importar aquí para evitar importación circular
        from .validate import validate_all
        errors = validate_all()
        if errors:
            print(f"\n❌ {len(errors)} validaciones fallaron")
            sys.exit(1)
        else:
            print("\n✅ Todas las validaciones pasaron")
            sys.exit(0)
    else:
        build()


if __name__ == "__main__":
    main()