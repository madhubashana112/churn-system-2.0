"""Preserve human-readable metric evidence without guessing its business meaning."""
import math
from numbers import Real
import re
import unicodedata
import pandas as pd


def currency_number(value):
    if pd.isna(value) or str(value).strip() == "":
        return float("nan")
    if isinstance(value, Real):
        return float(value) if math.isfinite(value) else float("nan")
    text = str(value).strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = "".join(c for c in text if unicodedata.category(c) != "Sc").strip()
    text = re.sub(r"^(?:USD|EUR|GBP|LKR|INR|Rs\.?)\s*|\s*(?:USD|EUR|GBP|LKR|INR)$", "", text, flags=re.I).strip()
    if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text):
        return float("nan")
    number = float(text.replace(",", ""))
    return (-number if negative else number) if math.isfinite(number) else float("nan")


def metric_evidence(series):
    values = series.dropna().astype(str).tolist()
    samples = list(dict.fromkeys(values))[:3]
    evidence = {"sample_values": samples, "observation_count": len(values)}
    matches = [re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([a-zA-Z%µμ°/]+)?\s*", value) for value in values]
    if values and all(matches):
        units = {m.group(2) or "" for m in matches}
        numbers = [float(m.group(1)) for m in matches]
        if len(units) == 1 and all(math.isfinite(n) for n in numbers):
            lo, hi = min(numbers), max(numbers)
            evidence["numeric_summary"] = {"unit": next(iter(units)), "min": lo, "max": hi,
                "mean": sum(numbers) / len(numbers),
                "minimum_observation": values[numbers.index(lo)],
                "maximum_observation": values[numbers.index(hi)]}
    return evidence


def review_amount_mappings(schema, dataframes):
    """Check proposed/remembered amount roles against every uploaded row."""
    issues = []
    for table in schema.tables:
        frame = dataframes.get(table.file_name)
        if frame is None:
            continue
        for column in table.columns:
            if column.canonical_role != "TRANSACTION_AMOUNT" or column.source_column not in frame:
                continue
            values = frame[column.source_column]
            invalid = values.notna() & values.astype(str).str.strip().ne("") & values.map(currency_number).isna()
            if not invalid.any():
                continue
            example = str(values[invalid].iloc[0])[:200]
            message = (f"{table.file_name}: {column.source_column} cannot be Transaction Amount: "
                       f"{int(invalid.sum())} non-empty values are not numeric amounts. "
                       "Choose Attribute or Ignore for identifiers, and map the actual amount column to Transaction Amount.")
            column.confidence = 0.0
            column.reasoning = message
            column.sample_values = list(dict.fromkeys([example] + column.sample_values))[:3]
            column.confidence_status()
            issues.append(message)
    return issues
