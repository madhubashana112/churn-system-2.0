"""Turn uploaded files into the filename-keyed frames the pipeline consumes.

Every stage downstream — schema resolution, synthesis, enrichment — is keyed by
a logical table name, so this module is the only place that knows whether that
name came from a CSV on disk or from one sheet of a multi-sheet workbook.

A workbook contributes one entry per sheet, named ``workbook.xlsx::Sheet1``.
``table_stem`` in the synthesizer already splits on ``::``, so each sheet gets
its own feature prefix rather than all of them sharing the workbook's name.
"""

from __future__ import annotations

import io
import logging
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, NamedTuple, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

SHEET_SEPARATOR = "::"
SAMPLE_ROWS = 3

CSV_EXTENSIONS = frozenset({".csv", ".txt"})
TSV_EXTENSIONS = frozenset({".tsv"})
EXCEL_EXTENSIONS = frozenset({".xlsx", ".xlsm"})
SUPPORTED_EXTENSIONS = CSV_EXTENSIONS | TSV_EXTENSIONS | EXCEL_EXTENSIONS

# The legacy binary workbook format needs xlrd, which is not a dependency here.
LEGACY_EXCEL_EXTENSIONS = frozenset({".xls"})

# Real exports are frequently cp1252 rather than UTF-8, and a smart quote in a
# free-text complaint column is enough to abort the whole upload.
FALLBACK_ENCODING = "cp1252"


class UnsupportedFileError(ValueError):
    """Raised for an extension this platform cannot read."""


class IngestedFiles(NamedTuple):
    dataframes: Dict[str, pd.DataFrame]
    samples: Dict[str, str]


def extension_of(file_name: str) -> str:
    return PurePosixPath(file_name or "").suffix.lower()


def parse_bytes(file_name: str, contents: bytes) -> Dict[str, pd.DataFrame]:
    """Parse one upload into one or more frames, keyed by logical table name."""
    extension = extension_of(file_name)
    if extension in EXCEL_EXTENSIONS:
        return _parse_excel(file_name, contents)
    if extension in CSV_EXTENSIONS | TSV_EXTENSIONS:
        return {file_name: _parse_delimited(contents, sep="\t" if extension in TSV_EXTENSIONS else ",")}
    if extension in LEGACY_EXCEL_EXTENSIONS:
        raise UnsupportedFileError(
            f"{file_name!r} is a legacy .xls workbook, which this platform cannot "
            "read. Re-save it as .xlsx and upload again."
        )
    raise UnsupportedFileError(
        f"{file_name!r} has an unsupported extension {extension or '(none)'!r}; "
        f"upload one of {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _parse_delimited(contents: bytes, sep: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(contents), sep=sep)
    except UnicodeDecodeError:
        logger.warning("Upload was not valid UTF-8; retrying as %s", FALLBACK_ENCODING)
        return pd.read_csv(io.BytesIO(contents), sep=sep, encoding=FALLBACK_ENCODING)


def _parse_excel(file_name: str, contents: bytes) -> Dict[str, pd.DataFrame]:
    sheets = pd.read_excel(io.BytesIO(contents), sheet_name=None, engine="openpyxl")

    frames = {}
    for sheet_name, df in sheets.items():
        if df.empty:
            logger.info("Skipping empty sheet %r in %s", sheet_name, file_name)
            continue
        frames[f"{file_name}{SHEET_SEPARATOR}{sheet_name}"] = df
    if not frames:
        raise ValueError(f"{file_name!r} contained no sheets with data")
    return frames


def sample_csv(df: pd.DataFrame, rows: int = SAMPLE_ROWS) -> str:
    """Deterministic rows spread across the export, including first and last."""
    count = min(max(rows, 0), len(df))
    indices = [round(i * (len(df) - 1) / (count - 1)) for i in range(count)] if count > 1 else list(range(count))
    return df.iloc[indices].to_csv(index=False)


def ingest(uploads: Iterable[Tuple[str, bytes]]) -> IngestedFiles:
    """Parse a whole upload set, rejecting duplicate table names.

    A collision is an error rather than a last-write-wins: silently dropping one
    of two tables named `transactions.csv` would remove a signal from the
    analysis with nothing in the response to show it happened.
    """
    dataframes: Dict[str, pd.DataFrame] = {}
    duplicates: List[str] = []

    for file_name, contents in uploads:
        for table_name, df in parse_bytes(file_name, contents).items():
            if table_name in dataframes:
                duplicates.append(table_name)
                continue
            dataframes[table_name] = df

    if duplicates:
        raise ValueError(f"Duplicate table names in the upload: {', '.join(sorted(duplicates))}")
    if not dataframes:
        raise ValueError("No readable tables found in the upload")

    return IngestedFiles(dataframes=dataframes, samples={n: sample_csv(df) for n, df in dataframes.items()})
