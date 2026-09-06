"""The sectors this platform understands.

This is domain vocabulary rather than a parser detail: tenant registration
validates against it, the composition root dispatches cores by it, the enrichers
dispatch feature maths by it, and the dashboard picks a template from it. Every
one of those needs the same canonical spelling, so it lives here once.
"""

from __future__ import annotations

from typing import Dict, Optional

SECTOR_SAAS = "saas"
SECTOR_TELECOM = "telecom"
SECTOR_FINTECH = "fintech"

SECTORS = (SECTOR_SAAS, SECTOR_TELECOM, SECTOR_FINTECH)

# Display forms. A tenant registered as "isp" is stored and shown as "Telecom",
# so every later lookup sees one spelling instead of whatever was typed.
SECTOR_LABELS: Dict[str, str] = {
    SECTOR_SAAS: "SaaS",
    SECTOR_TELECOM: "Telecom",
    SECTOR_FINTECH: "FinTech",
}

SECTOR_ALIASES: Dict[str, str] = {
    "saas": SECTOR_SAAS, "subscription": SECTOR_SAAS,
    "telecom": SECTOR_TELECOM, "isp": SECTOR_TELECOM, "telco": SECTOR_TELECOM,
    "fintech": SECTOR_FINTECH, "banking": SECTOR_FINTECH, "bank": SECTOR_FINTECH,
    "wallet": SECTOR_FINTECH,
}


def normalize_sector(sector: Optional[str]) -> Optional[str]:
    """Map a user-supplied sector label onto a canonical key, or None."""
    if not sector:
        return None
    return SECTOR_ALIASES.get(sector.strip().lower())


def canonical_sector_label(sector: Optional[str]) -> Optional[str]:
    """Map a user-supplied sector label onto its display form, or None."""
    resolved = normalize_sector(sector)
    return SECTOR_LABELS[resolved] if resolved else None
