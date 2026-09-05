"""Upload parsing: CSV, TSV and multi-sheet workbooks.

Spec section 1 promised Excel and multi-sheet ingestion; the endpoint read CSV
only, so an `.xlsx` upload crashed with a parser error. These tests pin the
contract the rest of the pipeline relies on — one entry per logical table, named
so that two sheets from one workbook never collide.
"""

from __future__ import annotations

import io
from typing import Dict, List, Tuple

import pandas as pd
import pytest

from churn_platform.infrastructure.parsers.file_ingestion import (
    SAMPLE_ROWS,
    SHEET_SEPARATOR,
    UnsupportedFileError,
    ingest,
    parse_bytes,
    sample_csv,
)
from churn_platform.infrastructure.parsers.feature_synthesizer import table_stem


def csv_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def workbook_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


FRAME = pd.DataFrame({"user_id": ["u1", "u2"], "amount": [10.0, 20.0]})


class TestDelimitedFiles:
    def test_a_csv_is_keyed_by_its_filename(self):
        frames = parse_bytes("invoices.csv", csv_bytes("user_id,amount\nu1,10.0\n"))

        assert list(frames) == ["invoices.csv"]
        assert frames["invoices.csv"]["amount"].tolist() == [10.0]

    def test_a_tsv_is_split_on_tabs(self):
        frames = parse_bytes("events.tsv", "user_id\tamount\nu1\t10.0\n".encode("utf-8"))

        assert list(frames["events.tsv"].columns) == ["user_id", "amount"]

    def test_a_plain_txt_file_is_read_as_csv(self):
        assert list(parse_bytes("export.txt", csv_bytes("user_id,amount\nu1,1\n"))) == ["export.txt"]

    def test_a_cp1252_export_is_read_rather_than_rejected(self):
        """A smart quote in a complaint must not abort the whole upload."""
        raw = b"user_id,notes\nu1,can\x92t log in\n"
        with pytest.raises(UnicodeDecodeError):
            pd.read_csv(io.BytesIO(raw))

        frames = parse_bytes("complaints.csv", raw)

        assert "log in" in frames["complaints.csv"]["notes"].iloc[0]


class TestWorkbooks:
    def test_every_sheet_becomes_its_own_table(self):
        frames = parse_bytes("export.xlsx", workbook_bytes({"users": FRAME, "invoices": FRAME}))

        assert sorted(frames) == [
            f"export.xlsx{SHEET_SEPARATOR}invoices",
            f"export.xlsx{SHEET_SEPARATOR}users",
        ]

    def test_sheet_names_get_distinct_feature_prefixes(self):
        """The whole reason for the ``::`` separator.

        Without it both sheets would reduce to the workbook's stem and their
        aggregates would overwrite each other.
        """
        frames = parse_bytes("export.xlsx", workbook_bytes({"users": FRAME, "invoices": FRAME}))
        stems = {table_stem(name) for name in frames}

        assert stems == {"users", "invoices"}

    def test_an_empty_sheet_is_skipped(self):
        frames = parse_bytes(
            "export.xlsx",
            workbook_bytes({"users": FRAME, "scratch": pd.DataFrame()}),
        )

        assert list(frames) == [f"export.xlsx{SHEET_SEPARATOR}users"]

    def test_a_workbook_with_no_data_at_all_is_an_error(self):
        with pytest.raises(ValueError, match="no sheets with data"):
            parse_bytes("empty.xlsx", workbook_bytes({"scratch": pd.DataFrame()}))


class TestRejectedUploads:
    @pytest.mark.parametrize("name", ["data.json", "data.parquet", "data.pdf", "noextension"])
    def test_an_unreadable_extension_is_rejected(self, name):
        with pytest.raises(UnsupportedFileError, match="unsupported extension"):
            parse_bytes(name, b"whatever")

    def test_a_legacy_xls_explains_how_to_fix_it(self):
        """xlrd is not a dependency, so say so instead of raising an ImportError."""
        with pytest.raises(UnsupportedFileError, match="Re-save it as .xlsx"):
            parse_bytes("old.xls", b"\xd0\xcf\x11\xe0")


class TestIngest:
    def test_a_mixed_upload_yields_one_sample_per_table(self):
        uploads: List[Tuple[str, bytes]] = [
            ("users.csv", csv_bytes("user_id,tier\nu1,pro\n")),
            ("export.xlsx", workbook_bytes({"invoices": FRAME})),
        ]
        ingested = ingest(uploads)

        assert sorted(ingested.dataframes) == sorted(ingested.samples)
        assert len(ingested.dataframes) == 2

    def test_a_duplicate_table_name_is_rejected(self):
        """Last-write-wins would silently drop a table from the analysis."""
        body = csv_bytes("user_id,amount\nu1,10.0\n")
        with pytest.raises(ValueError, match="Duplicate table names"):
            ingest([("a.csv", body), ("a.csv", body)])

    def test_two_sheets_named_the_same_in_different_workbooks_are_not_a_collision(self):
        body = workbook_bytes({"users": FRAME})
        ingested = ingest([("june.xlsx", body), ("july.xlsx", body)])

        assert len(ingested.dataframes) == 2

    def test_an_empty_upload_set_is_an_error(self):
        with pytest.raises(ValueError, match="No readable tables"):
            ingest([])

    def test_samples_carry_the_header_and_a_few_rows(self):
        rows = "".join(f"u{i},{i}\n" for i in range(50))
        ingested = ingest([("users.csv", csv_bytes(f"user_id,amount\n{rows}"))])

        lines = ingested.samples["users.csv"].strip().splitlines()
        assert lines[0] == "user_id,amount"
        assert len(lines) == SAMPLE_ROWS + 1

    def test_the_caller_can_control_how_many_sample_rows_are_taken(self):
        frame = sample_csv(FRAME, rows=1)

        assert len(frame.strip().splitlines()) == 2
