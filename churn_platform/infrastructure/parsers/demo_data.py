"""The bundled sample exports, as bytes the ingest pipeline already accepts.

``data/<sector>/*.csv`` is committed and shipped with the deployment, and
``ingest`` takes ``(file_name, bytes)`` rather than paths — so a visitor can be
handed a complete customer base without ever opening a file picker. This is
deliberately separate from ``generate_mock_data.py``: that script owns how the
sample data is *made*, this only owns handing over what was made.
"""

from __future__ import annotations

from typing import List, Tuple

from churn_platform.config import REPO_ROOT
from churn_platform.domain.models.sector import normalize_sector

DEMO_DATA_DIR = REPO_ROOT / "data"


def demo_files(sector: str) -> List[Tuple[str, bytes]]:
    """Every sample CSV for a sector, as ``(file_name, contents)``.

    The sector is normalised first, so a tenant registered under a free-text
    label like "subscription" still reaches the SaaS exports.

    Raises ``ValueError`` for a sector the platform does not model, and
    ``FileNotFoundError`` when the exports are not on disk — the two have
    different remedies, so the endpoint answers them differently.
    """
    resolved = normalize_sector(sector)
    if resolved is None:
        raise ValueError(
            f"Unknown sector {sector!r}; expected SaaS, Telecom or FinTech"
        )

    folder = DEMO_DATA_DIR / resolved
    files = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    if not files:
        raise FileNotFoundError(
            f"No sample exports in {folder}. Run `python generate_mock_data.py` "
            "to create them."
        )
    return [(path.name, path.read_bytes()) for path in files]
