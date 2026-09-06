"""The per-customer detail layer: reason fallbacks, ranking, peer percentiles.

Unit tests drive the pure helpers and the use case through the real in-memory
repository; HTTP-level tests live further down and run against ``TestClient``
like the rest of the suite.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from churn_platform.application.dtos.entity_detail_dto import EntityDetail
from churn_platform.application.use_cases.describe_entity import (
    DescribeEntityUseCase,
    _evidence,
    _median,
    _percentile,
    _risk_rank,
)
from churn_platform.application.use_cases.summarize_analysis import reason_for
from churn_platform.domain.models.analysis_run import AnalysisRun, EntityOutcome
from churn_platform.domain.models.churn_prediction import ChurnPrediction
from churn_platform.domain.models.retention_playbook import RetentionPlaybook
from churn_platform.infrastructure.repositories.memory_analysis_repo import (
    MemoryAnalysisRepository,
)
from churn_platform.main import app

from conftest import SECTOR_SCHEMAS, run_async, sector_schema

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def make_outcome(
    entity_id: str,
    probability: float,
    tier: str = "MEDIUM",
    features: dict | None = None,
    root_cause: str | None = None,
    drivers: list[str] | None = None,
    dormancy: str | None = None,
) -> EntityOutcome:
    return EntityOutcome(
        prediction=ChurnPrediction(
            entity_id=entity_id,
            churn_probability=probability,
            risk_tier=tier,
            root_cause=root_cause,
            primary_drivers=drivers,
            dormancy_type=dormancy,
        ),
        playbook=RetentionPlaybook(action_type="DISCOUNT", action_payload="x", channel="EMAIL"),
        features=features or {},
    )


def make_run(*outcomes: EntityOutcome, tenant_id: str = "t1") -> AnalysisRun:
    return AnalysisRun(
        tenant_id=tenant_id,
        sector="SaaS",
        schema_mapping=sector_schema("SaaS"),
        outcomes=list(outcomes),
    )


class TestReasonFor:
    def test_root_cause_wins_over_everything_else(self):
        outcome = make_outcome("u1", 0.9, root_cause="Payment fatigue", drivers=["ignored"])

        assert reason_for(outcome) == "Payment fatigue"

    def test_drivers_are_joined_when_there_is_no_root_cause(self):
        outcome = make_outcome("u1", 0.9, drivers=["Usage collapse", "Ticket storm"])

        assert reason_for(outcome) == "Usage collapse; Ticket storm"

    def test_dormancy_is_the_last_fallback(self):
        outcome = make_outcome("u1", 0.9, dormancy="SILENT")

        assert reason_for(outcome) == "Dormancy: SILENT"

    def test_a_prediction_with_no_explanation_has_no_reason(self):
        assert reason_for(make_outcome("u1", 0.9)) is None


class TestRiskRank:
    def test_rank_one_is_the_highest_probability(self):
        outcomes = [
            make_outcome("u_low", 0.1),
            make_outcome("u_high", 0.9),
            make_outcome("u_mid", 0.5),
        ]

        assert _risk_rank(outcomes, "u_high") == 1
        assert _risk_rank(outcomes, "u_mid") == 2
        assert _risk_rank(outcomes, "u_low") == 3

    def test_equal_probabilities_break_on_entity_id_so_the_rank_is_stable(self):
        # Deliberately inserted out of id order: the sort, not the storage
        # order, must decide who ranks first.
        outcomes = [make_outcome("u_b", 0.7), make_outcome("u_a", 0.7)]

        assert _risk_rank(outcomes, "u_a") == 1
        assert _risk_rank(outcomes, "u_b") == 2


class TestPercentile:
    def test_exclusive_rank_over_n_plus_one(self):
        peers = [10.0, 20.0, 30.0, 40.0]

        percentiles = [_percentile(peers, v) for v in peers]

        assert percentiles == pytest.approx([0.2, 0.4, 0.6, 0.8])

    def test_tied_peers_share_the_mid_rank(self):
        peers = [10.0, 10.0, 30.0, 40.0]

        # Both 10s occupy ranks 1 and 2, so they share (0 + 1.5) / 5.
        assert _percentile(peers, 10.0) == pytest.approx(0.3)

    def test_nobody_reaches_one_because_the_entity_is_its_own_peer(self):
        assert _percentile([1.0, 2.0, 3.0], 3.0) < 1.0

    def test_median_of_an_even_population_averages_the_middle(self):
        assert _median([10.0, 20.0, 30.0, 40.0]) == pytest.approx(25.0)

    def test_median_of_an_odd_population_takes_the_middle(self):
        assert _median([30.0, 10.0, 20.0]) == pytest.approx(20.0)


class TestEvidence:
    def population(self, target_features: dict) -> tuple[list[EntityOutcome], EntityOutcome]:
        target = make_outcome("u_t", 0.5, features=target_features)
        peers = [
            make_outcome("u_1", 0.2, features={"x": 10}),
            make_outcome("u_2", 0.3, features={"x": 20}),
            make_outcome("u_3", 0.4, features={"x": 30}),
        ]
        return peers + [target], target

    def test_flags_and_text_are_not_magnitudes_and_get_no_percentile(self):
        outcomes, target = self.population({"x": 40, "plan": "gold", "waived": True})

        evidence = _evidence(outcomes, target)

        assert [f.key for f in evidence] == ["x"]

    def test_a_population_too_small_to_compare_against_has_no_percentile(self):
        lonely = make_outcome("u_only", 0.5, features={"x": 10})

        evidence = _evidence([lonely], lonely)

        assert evidence[0].percentile is None
        assert evidence[0].unusualness is None
        # The median of one value is still that value — it is the percentile
        # that is meaningless without peers.
        assert evidence[0].tenant_median == pytest.approx(10.0)

    def test_the_strangest_signals_lead_the_table(self):
        outcomes, target = self.population({"x": 40, "flat": 5})
        # Give every peer the same "flat" value so the target sits exactly at
        # the middle of that distribution.
        for outcome in outcomes:
            outcome.features["flat"] = 5

        evidence = _evidence(outcomes, target)

        assert [f.key for f in evidence] == ["x", "flat"]
        assert evidence[0].unusualness == pytest.approx(0.3)
        assert evidence[1].unusualness == pytest.approx(0.0)

    def test_low_is_as_unusual_as_high(self):
        outcomes, target = self.population({"x": 40})
        bottom = make_outcome("u_b", 0.1, features={"x": 0})
        outcomes.append(bottom)

        evidence = _evidence(outcomes, bottom)

        # 0 is below all five peers: (0 + 1) / 6.
        assert evidence[0].percentile == pytest.approx(1 / 6)
        assert evidence[0].unusualness == pytest.approx(0.5 - 1 / 6)


class TestDescribeEntityUseCase:
    def use_case(self, run: AnalysisRun | None) -> DescribeEntityUseCase:
        repo = MemoryAnalysisRepository()
        if run is not None:
            run_async(repo.save(run))
        return DescribeEntityUseCase(repo)

    def test_a_tenant_without_a_run_has_no_detail(self):
        use_case = self.use_case(None)

        assert run_async(use_case.execute("t1", "u1")) is None

    def test_an_entity_missing_from_the_run_has_no_detail(self):
        run = make_run(make_outcome("u1", 0.5))

        assert run_async(self.use_case(run).execute("t1", "ghost")) is None

    def test_the_payload_positions_the_customer_in_its_population(self):
        run = make_run(
            make_outcome("u1", 0.9, tier="CRITICAL", features={"x": 40}, root_cause="Payment fatigue"),
            make_outcome("u2", 0.4, features={"x": 20}),
            make_outcome("u3", 0.2, features={"x": 10}),
        )

        detail = run_async(self.use_case(run).execute("t1", "u1"))

        assert isinstance(detail, EntityDetail)
        assert detail.entity_id == "u1"
        assert detail.risk_rank == 1
        assert detail.population_size == 3
        assert detail.reason == "Payment fatigue"
        assert detail.risk_tier == "CRITICAL"
        assert detail.playbook.action_type == "DISCOUNT"
        assert detail.created_at == run.created_at
        assert detail.features[0].key == "x"
        assert detail.features[0].percentile == pytest.approx(0.75)
        assert detail.features[0].tenant_median == pytest.approx(20.0)

    def test_the_wrong_tenant_never_leaks_another_tenants_run(self):
        run = make_run(make_outcome("u1", 0.5), tenant_id="t1")

        assert run_async(self.use_case(run).execute("t2", "u1")) is None


# -- HTTP level -----------------------------------------------------------------
#
# Driven through the real app like test_metrics_and_routing.py, so a mis-wired
# dependency or a broken response model fails here rather than in the browser.


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def analyzed_tenant(client: TestClient) -> str:
    """One SaaS tenant pushed through the real upload pipeline."""
    response = client.post("/api/v1/tenants/", json={"name": "Detail Co", "sector": "SaaS"})
    assert response.status_code == 200, response.text
    tenant_id = response.json()["tenant_id"]

    uploads = [
        ("files", (table.file_name, open(os.path.join(DATA_DIR, "saas", table.file_name), "rb"), "text/csv"))
        for table in SECTOR_SCHEMAS["SaaS"][1]
    ]
    analysis = client.post("/api/v1/upload/analyze", data={"tenant_id": tenant_id}, files=uploads)
    assert analysis.status_code == 200, analysis.text
    return tenant_id


@pytest.fixture(scope="module")
def worst_entity(client: TestClient, analyzed_tenant: str) -> str:
    rows = client.get(f"/api/v1/analytics/metrics?tenant_id={analyzed_tenant}").json()["rows"]
    return rows[0]["entity_id"]


class TestCustomerEndpoint:
    def test_a_scored_customer_comes_back_with_its_population_context(
        self, client: TestClient, analyzed_tenant: str, worst_entity: str
    ):
        response = client.get(
            f"/api/v1/analytics/customer?tenant_id={analyzed_tenant}&entity_id={worst_entity}"
        )

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["tenant_id"] == analyzed_tenant
        assert payload["entity_id"] == worst_entity
        assert payload["population_size"] == 100
        # The row the summary calls worst must be the customer the detail ranks first.
        assert payload["risk_rank"] == 1
        assert 0.0 <= payload["churn_probability"] <= 1.0
        assert payload["playbook"]["action_type"] and payload["playbook"]["channel"]
        assert payload["features"], "the SaaS synthesizer always stores numeric features"
        assert all(f["percentile"] is not None for f in payload["features"])
        assert all(f["tenant_median"] is not None for f in payload["features"])

    def test_an_entity_that_was_never_scored_is_a_404(
        self, client: TestClient, analyzed_tenant: str
    ):
        response = client.get(
            f"/api/v1/analytics/customer?tenant_id={analyzed_tenant}&entity_id=ghost"
        )

        assert response.status_code == 404
        assert "ghost" in response.json()["detail"]

    def test_a_tenant_without_a_run_is_a_404(self, client: TestClient):
        registered = client.post("/api/v1/tenants/", json={"name": "Fresh Detail Co", "sector": "SaaS"})
        tenant_id = registered.json()["tenant_id"]

        response = client.get(f"/api/v1/analytics/customer?tenant_id={tenant_id}&entity_id=usr_1")

        assert response.status_code == 404
        assert "has not uploaded exports" in response.json()["detail"]

    def test_both_query_parameters_are_required(self, client: TestClient, analyzed_tenant: str):
        assert client.get(f"/api/v1/analytics/customer?tenant_id={analyzed_tenant}").status_code == 422
        assert client.get("/api/v1/analytics/customer?entity_id=usr_1").status_code == 422


class TestCustomerPage:
    def test_the_page_renders_with_the_customer_and_a_way_back(
        self, client: TestClient, analyzed_tenant: str, worst_entity: str
    ):
        response = client.get(f"/customer?tenant_id={analyzed_tenant}&entity_id={worst_entity}")

        assert response.status_code == 200, response.text
        assert worst_entity in response.text
        assert f'data-entity-id="{worst_entity}"' in response.text
        assert f"/dashboard?tenant_id={analyzed_tenant}" in response.text
        # The SaaS colourway: the theme comes from the tenant record's sector.
        assert "--accent:#4f46e5" in response.text

    def test_a_missing_parameter_goes_to_onboarding(self, client: TestClient, analyzed_tenant: str):
        for url in ("/customer", f"/customer?tenant_id={analyzed_tenant}", "/customer?entity_id=usr_1"):
            response = client.get(url, follow_redirects=False)

            assert response.status_code == 303, url
            assert response.headers["location"] == "/"

    def test_a_tenant_the_server_has_forgotten_goes_to_onboarding(self, client: TestClient):
        response = client.get("/customer?tenant_id=no-such-tenant&entity_id=usr_1", follow_redirects=False)

        assert response.status_code == 303
        assert response.headers["location"] == "/"
