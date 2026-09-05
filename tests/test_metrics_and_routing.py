"""The metrics layer and the sector routing that serves the dashboards.

Everything here runs against the real FastAPI app through ``TestClient``, so a
broken Jinja block or a mis-wired dependency is a test failure rather than a
blank page in someone's browser. No test touches the network: with no API key
configured the composition root selects the offline gateway.
"""

from __future__ import annotations

import os
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

from churn_platform.application.dtos.metrics_dto import (
    CHART_BAR,
    CHART_DOUGHNUT,
    CHART_LINE,
    CHART_SCATTER,
    TONE_CRITICAL,
    TONE_GOOD,
    TONE_NEUTRAL,
    TONE_WARNING,
)
from churn_platform.domain.models.analysis_run import AT_RISK_TIERS, AnalysisRun, EntityOutcome
from churn_platform.domain.models.churn_prediction import ChurnPrediction
from churn_platform.domain.models.retention_playbook import RetentionPlaybook
from churn_platform.main import app

from conftest import SECTOR_SCHEMAS, SECTORS, sector_schema

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

SECTOR_KPI_KEYS: Dict[str, List[str]] = {
    "SaaS": ["mrr_at_risk", "payment_failures", "velocity_collapse", "data_export", "negative_tickets"],
    "Telecom": ["regional_impact", "worst_tower", "dropped_calls", "recharge_lapse", "port_out_intent"],
    "FinTech": ["liquidity_drain", "net_outflow", "rapid_drain", "dormant_accounts", "p2p_failures"],
}

SECTOR_CHART_KEYS: Dict[str, List[str]] = {
    "SaaS": ["segment_risk", "usage_dropoff", "velocity_spread"],
    "Telecom": ["region_risk", "tower_health", "recharge_gap"],
    "FinTech": ["drain_vs_risk", "dormancy_mix", "withdrawal_share"],
}

SECTOR_TEMPLATE_MARKERS: Dict[str, str] = {
    "SaaS": "Subscription churn dashboard",
    "Telecom": "Subscriber churn dashboard",
    "FinTech": "Liquidity and dormancy dashboard",
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def analyzed_tenants(client: TestClient) -> Dict[str, str]:
    """Register one tenant per sector and push that sector's exports through it."""
    tenants: Dict[str, str] = {}
    for sector in SECTORS:
        response = client.post("/api/v1/tenants/", json={"name": f"{sector} Metrics Co", "sector": sector})
        assert response.status_code == 200, response.text
        tenant_id = response.json()["tenant_id"]

        uploads = [
            ("files", (table.file_name, open(os.path.join(DATA_DIR, sector.lower(), table.file_name), "rb"), "text/csv"))
            for table in SECTOR_SCHEMAS[sector][1]
        ]
        analysis = client.post("/api/v1/upload/analyze", data={"tenant_id": tenant_id}, files=uploads)
        assert analysis.status_code == 200, analysis.text
        assert analysis.json()["entities_analyzed"] == 100
        tenants[sector] = tenant_id
    return tenants


@pytest.fixture(scope="module")
def summaries(client: TestClient, analyzed_tenants: Dict[str, str]) -> Dict[str, dict]:
    return {
        sector: client.get(f"/api/v1/analytics/metrics?tenant_id={tenant_id}").json()
        for sector, tenant_id in analyzed_tenants.items()
    }


class TestMetricsEndpoint:
    def test_a_tenant_without_a_run_is_a_404_not_an_empty_summary(self, client: TestClient):
        """An empty summary would render a dashboard of zeroes; the UI needs the 404
        to know it should show the upload prompt instead."""
        registered = client.post("/api/v1/tenants/", json={"name": "Fresh Co", "sector": "SaaS"})
        tenant_id = registered.json()["tenant_id"]

        response = client.get(f"/api/v1/analytics/metrics?tenant_id={tenant_id}")

        assert response.status_code == 404
        assert "Upload" in response.json()["detail"]

    @pytest.mark.parametrize("sector", SECTORS)
    def test_every_sector_gets_its_own_kpis_plus_the_shared_one(self, summaries, sector):
        kpis = summaries[sector]["kpis"]

        for key in SECTOR_KPI_KEYS[sector] + ["at_risk"]:
            assert key in kpis, f"{sector} is missing KPI {key}"
        assert all(kpis[key]["key"] == key for key in kpis)
        assert all(kpis[key]["label"] and kpis[key]["value"] for key in kpis)

    @pytest.mark.parametrize("sector", SECTORS)
    def test_kpi_tones_come_from_the_shared_vocabulary(self, summaries, sector):
        tones = {card["tone"] for card in summaries[sector]["kpis"].values()}

        assert tones <= {TONE_NEUTRAL, TONE_GOOD, TONE_WARNING, TONE_CRITICAL}

    @pytest.mark.parametrize("sector", SECTORS)
    def test_sector_charts_and_the_shared_distribution_charts_are_present(self, summaries, sector):
        charts = summaries[sector]["charts"]

        for key in SECTOR_CHART_KEYS[sector] + ["tier_mix", "probability_histogram"]:
            assert key in charts, f"{sector} is missing chart {key}"
        assert all(spec["kind"] in {CHART_BAR, CHART_LINE, CHART_DOUGHNUT, CHART_SCATTER}
                   for spec in charts.values())

    @pytest.mark.parametrize("sector", SECTORS)
    def test_chart_series_are_shaped_the_way_the_frontend_reads_them(self, summaries, sector):
        for spec in summaries[sector]["charts"].values():
            assert spec["datasets"], spec["key"]
            for dataset in spec["datasets"]:
                if spec["kind"] == CHART_SCATTER:
                    assert dataset["points"], f"{spec['key']}/{dataset['label']} has no scatter points"
                elif spec["kind"] == CHART_DOUGHNUT:
                    assert len(dataset["values"]) == len(spec["labels"])
                else:
                    assert len(dataset["values"]) == len(spec["labels"])

    @pytest.mark.parametrize("sector", SECTORS)
    def test_tier_slices_account_for_every_analyzed_entity(self, summaries, sector):
        summary = summaries[sector]
        counts = {slice_["tier"]: slice_["count"] for slice_ in summary["tiers"]}

        assert sum(counts.values()) == summary["entities_analyzed"] == 100
        at_risk = sum(counts.get(tier, 0) for tier in AT_RISK_TIERS)
        assert summary["at_risk_count"] == at_risk
        assert summary["at_risk_share"] == pytest.approx(at_risk / 100, abs=1e-6)

    @pytest.mark.parametrize("sector", SECTORS)
    def test_rows_are_sorted_worst_first_and_carry_their_evidence(self, summaries, sector):
        rows = summaries[sector]["rows"]
        probabilities = [row["churn_probability"] for row in rows]

        assert len(rows) == 100
        assert probabilities == sorted(probabilities, reverse=True)
        assert all(row["playbook"]["action_type"] and row["playbook"]["channel"] for row in rows)
        assert all(row["highlights"] for row in rows)
        # The churning cohort is generated first, so the worst row is one of theirs.
        assert int(rows[0]["entity_id"].rsplit("_", 1)[1]) <= 25

    @pytest.mark.parametrize("sector", SECTORS)
    def test_the_summary_says_it_was_scored_offline(self, summaries, sector):
        assert summaries[sector]["offline_mode"] is True
        assert summaries[sector]["entities_uploaded"] == 100


class TestStatusEndpoint:
    def test_it_reports_the_offline_configuration(self, client: TestClient):
        status = client.get("/api/v1/analytics/status").json()

        assert status["offline_mode"] is True
        assert status["batch_size"] == 0
        assert status["batch_size_note"]


class TestDashboardRouting:
    def test_without_a_tenant_the_bootstrap_page_is_served(self, client: TestClient):
        response = client.get("/dashboard")

        assert response.status_code == 200
        assert "localStorage" in response.text

    @pytest.mark.parametrize("sector", SECTORS)
    def test_the_template_follows_the_tenants_sector(self, client, analyzed_tenants, sector):
        response = client.get(f"/dashboard?tenant_id={analyzed_tenants[sector]}")

        assert response.status_code == 200
        assert SECTOR_TEMPLATE_MARKERS[sector] in response.text
        assert f'data-sector="{sector.lower()}"' in response.text

    def test_a_tenant_the_server_has_forgotten_goes_to_onboarding(self, client: TestClient):
        """Rendering the bootstrap page here would bounce the browser straight back
        to this same URL, because the bootstrap reads the stale localStorage id."""
        response = client.get("/dashboard?tenant_id=no-such-tenant", follow_redirects=False)

        assert response.status_code == 303
        assert response.headers["location"] == "/"


class TestAnalysisRun:
    def outcome(self, tier: str) -> EntityOutcome:
        return EntityOutcome(
            prediction=ChurnPrediction(entity_id=f"u_{tier}", churn_probability=0.5, risk_tier=tier),
            playbook=RetentionPlaybook(action_type="DISCOUNT", action_payload="x", channel="EMAIL"),
        )

    def test_at_risk_is_critical_and_high_only(self):
        run = AnalysisRun(
            tenant_id="t", sector="SaaS",
            schema_mapping=sector_schema("SaaS"),
            outcomes=[self.outcome(tier) for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW")],
        )

        assert [o.prediction.risk_tier for o in run.at_risk] == ["CRITICAL", "HIGH"]
