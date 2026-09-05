"""Score the generated mock base with a real model and report how good it is.

The mock datasets have a known churning cohort (entity ids 1-25 of 100), so a
hosted model's output can be measured rather than eyeballed: coverage, cohort
separation and AUC against that ground truth, next to the same numbers for the
offline oracle the app falls back to.

This is the gate that decides whether a free hosted model is good enough to
leave in `.env`, and which knob to turn when it is not. Free tiers cannot serve
weights you tuned yourself, so what is tunable here is the prompt, the batch
size and the model choice — the numbers below are what tells you whether a
change helped.

    python evaluate_ai_provider.py                        # every sector
    python evaluate_ai_provider.py --sector saas --entities 40 --batch-size 5
    python evaluate_ai_provider.py --mock                 # oracle only, no key
    python evaluate_ai_provider.py --json report.json --strict
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from churn_platform.application.use_cases.execute_sector_analysis import ExecuteSectorAnalysisUseCase
from churn_platform.application.use_cases.resolve_multi_sheet_schema import ResolveMultiSheetSchemaUseCase
from churn_platform.application.use_cases.synthesize_features import SynthesizeFeaturesUseCase
from churn_platform.config import get_settings
from churn_platform.domain.models.analysis_run import AT_RISK_TIERS
from churn_platform.domain.models.churn_prediction import ChurnPrediction
from churn_platform.domain.models.retention_playbook import RetentionPlaybook
from churn_platform.infrastructure.ai.cores.fintech_core import FintechCore
from churn_platform.infrastructure.ai.cores.saas_core import SaasCore
from churn_platform.infrastructure.ai.cores.telecom_core import TelecomCore
from churn_platform.infrastructure.ai.mock_qwen_gateway import MockQwenGateway, risk_tier
from churn_platform.infrastructure.parsers.feature_synthesizer import PandasFeatureSynthesizer
from churn_platform.infrastructure.parsers.file_ingestion import ingest
from churn_platform.infrastructure.parsers.schema_resolver import AISchemaResolver
from churn_platform.infrastructure.parsers.sector_feature_enrichers import enrich_features
from churn_platform.presentation.api.dependencies import build_ai_gateway
from generate_mock_data import CHURN_COHORT_SIZE, N_ENTITIES

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CORES = {"saas": SaasCore, "telecom": TelecomCore, "fintech": FintechCore}

# A run is usable when it scored almost everybody and separated the cohorts
# clearly. Below these the dashboard would be showing noise with a straight face.
COVERAGE_GATE = 0.95
AUC_GATE = 0.75

logger = logging.getLogger("evaluate")


class RepairCounter(logging.Handler):
    """Counts the repairs the normaliser made to a model's replies.

    The app fixes a drifted tier or an out-of-range probability and carries on,
    which is right for a dashboard and wrong for an evaluation: without this the
    report would say nothing about how much of the output needed rescuing.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1

    def reset(self) -> None:
        self.count = 0


REPAIRS = RepairCounter()
logging.getLogger("churn_platform.infrastructure.ai.normalise_prediction").addHandler(REPAIRS)


def ground_truth(entity_id: str) -> bool:
    """True when the entity belongs to the generated churning cohort."""
    match = re.search(r"(\d+)$", entity_id)
    if match is None:
        raise ValueError(f"entity id {entity_id!r} carries no cohort index")
    return int(match.group(1)) <= CHURN_COHORT_SIZE


def auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Rank AUC (Mann-Whitney), ties split, so a plateau is scored honestly."""
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return float("nan")
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in positives
        for n in negatives
    )
    return wins / (len(positives) * len(negatives))


def load_uploads(sector: str) -> List[Tuple[str, bytes]]:
    folder = DATA_DIR / sector
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs in {folder}; run `python generate_mock_data.py` first")
    return [(path.name, path.read_bytes()) for path in files]


async def score_sector(
    sector: str,
    gateway,
    batch_size: int,
    limit: Optional[int],
) -> Dict[str, object]:
    """Run the same pipeline an upload runs, and keep what came back."""
    REPAIRS.reset()
    ingested = ingest(load_uploads(sector))
    schema = await ResolveMultiSheetSchemaUseCase(AISchemaResolver(gateway)).execute(ingested.samples)
    features = SynthesizeFeaturesUseCase(
        PandasFeatureSynthesizer(),
        enricher=enrich_features,
    ).execute(schema, ingested.dataframes, sector=sector)
    if limit:
        features = features[:limit]

    payload_chars = sum(len(json.dumps(f.model_dump(), default=str)) for f in features)
    calls = 1 + (1 if batch_size <= 0 else math.ceil(len(features) / batch_size))

    started = time.perf_counter()
    results = await ExecuteSectorAnalysisUseCase(batch_size=batch_size).execute(CORES[sector](gateway), features)
    elapsed = time.perf_counter() - started

    return {
        "sector": sector,
        "submitted": len(features),
        "calls": calls,
        # Rough but honest: a token is ~4 characters of this kind of payload, and
        # free tiers meter tokens, not requests.
        "prompt_tokens": payload_chars // 4,
        "seconds": elapsed,
        "repairs": REPAIRS.count,
        "results": results,
        "primary_entity_key": schema.primary_entity_key,
        "files": [table.file_name for table in schema.tables],
    }


def summarize(run: Dict[str, object]) -> Dict[str, object]:
    results: List[Tuple[ChurnPrediction, RetentionPlaybook]] = run["results"]
    submitted = int(run["submitted"])
    labels = [ground_truth(pred.entity_id) for pred, _ in results]
    scores = [pred.churn_probability for pred, _ in results]

    churning = [s for s, y in zip(scores, labels) if y]
    healthy = [s for s, y in zip(scores, labels) if not y]
    mean_churning = sum(churning) / len(churning) if churning else float("nan")
    mean_healthy = sum(healthy) / len(healthy) if healthy else float("nan")

    tiers = {pred.risk_tier for pred, _ in results}
    alarming = {pred.entity_id for pred, _ in results if pred.risk_tier in AT_RISK_TIERS}
    mismatched_tier = sum(1 for pred, _ in results if risk_tier(pred.churn_probability) != pred.risk_tier)
    incomplete_playbook = sum(
        1 for _, playbook in results
        if not playbook.action_type or not playbook.channel
    )

    caught = sum(1 for pred, _ in results if ground_truth(pred.entity_id) and pred.entity_id in alarming)
    false_alarms = sum(1 for pred, _ in results if not ground_truth(pred.entity_id) and pred.entity_id in alarming)

    return {
        "sector": run["sector"],
        "submitted": submitted,
        "scored": len(results),
        "coverage": len(results) / submitted if submitted else float("nan"),
        "calls": run["calls"],
        "prompt_tokens": run["prompt_tokens"],
        "seconds": run["seconds"],
        "mean_churning": mean_churning,
        "mean_healthy": mean_healthy,
        "separation": mean_churning - mean_healthy,
        "auc": auc(labels, scores),
        "tiers": sorted(tiers),
        "caught": caught,
        "cohort_size": sum(labels),
        "false_alarms": false_alarms,
        "healthy_size": len(labels) - sum(labels),
        "mismatched_tier": mismatched_tier,
        "repairs": run["repairs"],
        "incomplete_playbook": incomplete_playbook,
    }


def verdict(row: Dict[str, object]) -> Tuple[bool, List[str]]:
    """Pass/fail plus what to turn when it fails."""
    failures: List[str] = []
    coverage = float(row["coverage"])
    value = row["auc"]

    if coverage < COVERAGE_GATE:
        failures.append(
            f"coverage {coverage:.0%} < {COVERAGE_GATE:.0%}: batches are being dropped. "
            "Lower BATCH_SIZE, raise QWEN_MAX_RETRIES, or raise MAX_ENTITIES headroom."
        )
    if value != value:
        # One cohort only, so there is nothing to rank against. Still a failure —
        # an unmeasurable model is not a certified one — but blaming the ranking
        # would send the reader to the wrong knob.
        failures.append(
            "AUC is not measurable: every scored customer came from one cohort. "
            "Fix coverage first, or raise --entities so both cohorts are present."
        )
    elif float(value) < AUC_GATE:
        failures.append(
            f"AUC {float(value):.2f} < {AUC_GATE:.2f}: the model is not ranking the churning cohort "
            "above the healthy one. Try a stronger model on the same host before blaming the prompt."
        )
    if row["mismatched_tier"]:
        failures.append(
            f"{row['mismatched_tier']} risk tiers disagree with their own probability: "
            "the model is inventing labels instead of applying the bands."
        )
    return not failures, failures


def num(value: object) -> str:
    """A measured value, or "n/a" when the cohort needed to compute it was empty."""
    number = float(value)
    return f"{number:.3f}" if number == number else "n/a"


def print_report(label: str, oracle: Dict[str, object], live: Optional[Dict[str, object]]) -> bool:
    rows = [("offline oracle", oracle)] + ([(label, live)] if live else [])
    fields = [
        ("coverage", lambda r: f"{r['scored']}/{r['submitted']} ({r['coverage']:.0%})"),
        ("model calls", lambda r: f"{r['calls']} (~{r['prompt_tokens']:,} prompt tokens)"),
        ("elapsed", lambda r: f"{r['seconds']:.1f}s"),
        ("mean p, churning", lambda r: num(r["mean_churning"])),
        ("mean p, healthy", lambda r: num(r["mean_healthy"])),
        ("separation", lambda r: num(r["separation"])),
        ("AUC", lambda r: num(r["auc"])),
        ("churning caught in high/critical", lambda r: f"{r['caught']}/{r['cohort_size']}"),
        ("healthy wrongly alarmed", lambda r: f"{r['false_alarms']}/{r['healthy_size']}"),
        ("risk tiers seen", lambda r: ", ".join(r["tiers"]) or "-"),
        ("tier/probability mismatches", lambda r: str(r["mismatched_tier"])),
        ("replies the app had to repair", lambda r: str(r["repairs"])),
        ("incomplete playbooks", lambda r: str(r["incomplete_playbook"])),
    ]

    width = max(len(name) for name, _ in rows) + 2
    print(f"\n{oracle['sector']}  (entity key {oracle['files_key']}, files: {', '.join(oracle['files'])})")
    print(f"  {'':<34}" + "".join(f"{name:>{width + 14}}" for name, _ in rows))
    ok = True
    for field, render in fields:
        print(f"  {field:<34}" + "".join(f"{render(row):>{width + 14}}" for _, row in rows))
    for name, row in rows:
        passed, failures = verdict(row)
        mark = "PASS" if passed else "FAIL"
        print(f"  gate for {name:<24}{mark}")
        for failure in failures:
            print(f"    - {failure}")
        if name != "offline oracle" and not passed:
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sector", default="all", choices=("all", *CORES), help="which mock base to score")
    parser.add_argument("--entities", type=int, default=0, help="cap entities per sector (0 = all %d)" % N_ENTITIES)
    parser.add_argument("--batch-size", type=int, default=None, help="entities per model call (default: the host's, else BATCH_SIZE)")
    parser.add_argument("--mock", action="store_true", help="score with the offline oracle only; no key, no network")
    parser.add_argument("--json", metavar="PATH", help="write the report as JSON")
    parser.add_argument("--strict", action="store_true", help="exit non-zero when the live model fails the gate")
    parser.add_argument("--verbose", action="store_true", help="show per-batch and HTTP logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    offline = MockQwenGateway()
    live = None if args.mock else build_ai_gateway(settings)
    if isinstance(live, MockQwenGateway):
        live = None
        print(
            "No provider key configured, so only the offline oracle is measured.\n"
            "Add GROQ_API_KEY (free, no card) to .env to measure a real model."
        )
    batch_size = args.batch_size if args.batch_size is not None else settings.resolved_batch_size
    sectors = list(CORES) if args.sector == "all" else [args.sector]
    limit = args.entities or None

    label = "offline mock"
    if live is not None:
        label = f"{settings.api_provider} / {settings.resolved_model}"
        print(f"Measuring {label} at {settings.resolved_base_url}")

    report: List[Dict[str, object]] = []
    passed = True
    for sector in sectors:
        oracle_run = asyncio.run(score_sector(sector, offline, 0, limit))
        oracle = summarize(oracle_run)
        oracle["files"] = oracle_run["files"]
        oracle["files_key"] = oracle_run["primary_entity_key"]

        live_row = None
        if live is not None:
            print(f"\n{sector}: calling {label} ...")
            try:
                live_run = asyncio.run(score_sector(sector, live, batch_size, limit))
            except Exception as exc:
                # The schema call is not batched, so a host that rejects it leaves
                # nothing to report. Which model refused, and why, is the answer
                # this harness exists to give; a client traceback is not.
                print(f"{sector}: {label} could not be measured — {type(exc).__name__}: {str(exc)[:300]}")
                passed = False
            else:
                live_row = summarize(live_run)
                live_row["files"] = live_run["files"]
                live_row["files_key"] = live_run["primary_entity_key"]

        passed = print_report(label, oracle, live_row) and passed
        report.append({"oracle": oracle, "live": live_row, "label": label})

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {args.json}")

    print()
    return 1 if (args.strict and not passed) else 0


if __name__ == "__main__":
    sys.exit(main())
