import csv
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import unicodedata

DATE_FORMATS = ("%d/%m/%Y", "%Y-%m-%d")

def is_missing(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())

def clean_text(value: object) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFC", str(value)).strip()

def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFD", clean_text(value).casefold())
    return "".join(
        character for character in text if not unicodedata.combining(character)
    )

def parse_date(value: object, field_name: str, row_number: int) -> datetime:
    text = clean_text(value)
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue
    accepted_formats = " o ".join(DATE_FORMATS)
    raise ValueError(
        f"Fecha inválida en {field_name}, fila {row_number}: {text!r}. "
        f"Formatos admitidos: {accepted_formats}."
    )

def normalize_date(value: object, field_name: str, row_number: int) -> str:
    return parse_date(value, field_name, row_number).date().isoformat()

def normalize_number(value: object) -> str | None:
    #Interpreta números con separadores decimales comunes sin alterar su valor
    if is_missing(value):
        return None

    text = clean_text(value).replace(" ", "").replace("\u00a0", "")
    if "," in text and "." in text:
        decimal_separator = "," if text.rfind(",") > text.rfind(".") else "."
        grouping_separator = "." if decimal_separator == "," else ","
        text = text.replace(grouping_separator, "")
        if decimal_separator == ",":
            text = text.replace(",", ".")
    elif "," in text:
        if text.count(",") != 1:
            raise ValueError(f"Formato numérico inválido: {value!r}.")
        text = text.replace(",", ".")

    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"Formato numérico inválido: {value!r}.") from error
    if not number.is_finite():
        raise ValueError(f"El valor numérico debe ser finito: {value!r}.")

    normalized = format(number, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"

class QualityReport:
    #Acumula incidencias por campo y tipo para exportarlas de forma resumida

    def __init__(self, dataset: str, year: int | None = None) -> None:
        self.dataset = dataset
        self.year = year
        self._counts: Counter[tuple[str, str]] = Counter()

    def add(
        self,
        field_name: str,
        issue: str,
        row_number: int,
        value: object = "",
    ) -> None:
        key = (field_name, issue)
        self._counts[key] += 1

    def records(self) -> list[dict[str, object]]:
        return [
            {
                "conjunto": self.dataset,
                "anio": self.year if self.year is not None else "",
                "campo": field_name,
                "tipo_incidencia": issue,
                "cantidad": count,
            }
            for (field_name, issue), count in sorted(self._counts.items())
        ]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        headers = (
            "conjunto",
            "anio",
            "campo",
            "tipo_incidencia",
            "cantidad",
        )
        with path.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(self.records())