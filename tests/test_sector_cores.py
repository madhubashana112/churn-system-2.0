"""The three sector cores, driven end to end against the offline gateway.

These go through `IChurnCore.analyze`, so they cover the prompt the core builds,
the gateway's dispatch on it, the JSON shape it returns, and the core's parsing
of that shape into domain models — the whole seam that a live Qwen deployment
would also have to cross.
"""

from __future__ import annotations

import statistics
from typing import List

import pytest

from churn_platform.domain.models.churn_prediction import ChurnPrediction
from churn_platform.domain.models.customer_features import CustomerFeatures
from churn_platform.domain.models.retention_playbook import RetentionPlaybook
from churn_platform.infrastructure.ai.cores.fintech_core import FintechCore
from churn_platform.infrastructure.ai.cores.saas_core import SaasCore
from churn_platform.infrastructure.ai.cores.telecom_core import TelecomCore
from churn_platform.infrastructure.ai.mock_qwen_gateway import MockQwenGateway

from conftest import cohort_of, run_async, sector_features

CORES = {"SaaS": SaasCore, "Telecom": TelecomCore, "FinTech": FintechCore}

VALID_TIERS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

# Channels each sector's playbook is allowed to use, per the spec's prompt
# schemas. A SaaS retention offer arriving by USSD would be a wiring bug.
EXPECTED_CHANNELS = {
    "SaaS": {"EMAIL", "IN_APP", "PHONE"},
    "Telecom": {"SMS", "USSD"},
    "FinTech": {"PUSH_NOTIFICATION", "EMAIL"},
}


@pytest.fixture
def core(sector: str):
    return CORES[sector](MockQwenGateway())


class TestPredictionShape:
    def test_every_entity_comes_back_as_a_domain_model(self, sector, core):
        results = run_async(core.analyze(sector_features(sector)))

        assert len(results) == 100
        for prediction, playbook in results:
            assert isinstance(prediction, ChurnPrediction)
            assert isinstance(playbook, RetentionPlaybook)

    def test_probabilities_and_tiers_are_well_formed(self, sector, sector_predictions):
        for prediction, _ in sector_predictions[sector]:
            assert 0.0 <= prediction.churn_probability <= 1.0
            assert prediction.risk_tier in VALID_TIERS
            assert prediction.entity_id

    def test_the_playbook_channel_is_one_the_sector_actually_uses(self, sector, sector_predictions):
        channels = {playbook.channel for _, playbook in sector_predictions[sector]}

        assert channels <= EXPECTED_CHANNELS[sector]

    def test_sector_specific_fields_are_populated(self, sector, sector_predictions):
        """Each core asks for its own extra fields, so each must fill them in."""
        predictions = sector_predictions[sector]
        if sector == "SaaS":
            assert all(p.primary_drivers for p, _ in predictions)
        elif sector == "Telecom":
            assert all(p.root_cause for p, _ in predictions)
            assert any(p.regional_network_impact_flag for p, _ in predictions)
        else:
            assert all(p.dormancy_type for p, _ in predictions)

    def test_a_core_does_not_invent_another_sector_s_fields(self, sector, sector_predictions):
        """The telecom regional flag must not leak into a SaaS response."""
        owned = {"SaaS": "primary_drivers", "Telecom": "root_cause", "FinTech": "dormancy_type"}
        for other_sector, field in owned.items():
            if other_sector == sector:
                continue
            assert all(getattr(p, field) is None for p, _ in sector_predictions[sector])


class TestCohortSeparation:
    def test_the_seeded_churners_outscore_the_healthy_cohort(self, sector, sector_predictions):
        probabilities = {"CHURN": [], "HEALTHY": []}
        for prediction, _ in sector_predictions[sector]:
            probabilities[cohort_of(prediction.entity_id)].append(prediction.churn_probability)

        assert statistics.mean(probabilities["CHURN"]) > statistics.mean(probabilities["HEALTHY"])
        # Not just on average: the generator seeds a correlated pattern, so the
        # weakest churner should still sit above a typical healthy customer.
        assert min(probabilities["CHURN"]) > statistics.median(probabilities["HEALTHY"])

    def test_no_seeded_churner_is_written_off_as_low_risk(self, sector, sector_predictions):
        tiers = [p.risk_tier for p, _ in sector_predictions[sector] if cohort_of(p.entity_id) == "CHURN"]

        assert "LOW" not in tiers

    def test_no_healthy_customer_is_escalated_to_critical(self, sector, sector_predictions):
        tiers = [p.risk_tier for p, _ in sector_predictions[sector] if cohort_of(p.entity_id) == "HEALTHY"]

        assert "CRITICAL" not in tiers


class TestDegenerateBatches:
    def test_an_empty_batch_returns_nothing(self, core):
        assert run_async(core.analyze([])) == []

    def test_a_single_entity_is_scored_rather_than_rejected(self, sector, core):
        """One customer gives the batch nothing to normalise against.

        Every signal is constant across a batch of one, so none can discriminate
        and the entity lands on the midpoint instead of crashing on an empty
        quantile.
        """
        results: List = run_async(core.analyze(sector_features(sector)[:1]))

        assert len(results) == 1
        prediction, _ = results[0]
        assert prediction.churn_probability == pytest.approx(0.5)
        assert prediction.risk_tier == "MEDIUM"

    def test_entities_with_no_features_at_all_are_still_scored(self, core):
        bare = [
            CustomerFeatures(entity_id="ghost_1", features={}),
            CustomerFeatures(entity_id="ghost_2", features={}),
        ]
        results = run_async(core.analyze(bare))

        assert [p.entity_id for p, _ in results] == ["ghost_1", "ghost_2"]
        assert all(p.churn_probability == pytest.approx(0.5) for p, _ in results)
