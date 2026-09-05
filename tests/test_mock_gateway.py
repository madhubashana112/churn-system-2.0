"""The offline gateway's own contract: determinism, dispatch and score shape.

The point of these tests is that offline mode is a *function*, not a stub. Two
calls with the same prompt must return the same bytes, a prompt it does not
recognise must fail loudly rather than fall back to something plausible, and a
batch of indistinguishable customers must produce indistinguishable scores
rather than a canned spread.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List

import pandas as pd
import pytest

from churn_platform.infrastructure.ai.mock_qwen_gateway import (
    CRITICAL_TIER,
    FINTECH_MARKER,
    FINTECH_SIGNALS,
    MIDPOINT_SCORE,
    NO_SIGNAL_DRIVERS,
    RISK_THRESHOLDS,
    SAAS_MARKER,
    SAAS_SIGNALS,
    SCHEMA_RESOLVER_MARKER,
    SIGMOID_STEEPNESS,
    TELECOM_MARKER,
    WINSORISE_QUANTILE,
    MockQwenGateway,
    _normalisation_bounds,
    _sigmoid,
    extract_payload,
    risk_tier,
)
from churn_platform.infrastructure.ai.prompts.fintech_prompts import FINTECH_CORE_SYSTEM_PROMPT
from churn_platform.infrastructure.ai.prompts.saas_prompts import SAAS_CORE_SYSTEM_PROMPT
from churn_platform.infrastructure.ai.prompts.schema_resolver_prompts import (
    SCHEMA_RESOLVER_SYSTEM_PROMPT,
)
from churn_platform.infrastructure.ai.prompts.telecom_prompts import TELECOM_CORE_SYSTEM_PROMPT

from conftest import run_async


def payload(*entities: Dict[str, Any]) -> str:
    """Render entities into the prompt a sector core would build."""
    body = [{"entity_id": eid, "features": features} for eid, features in entities]
    return f"Analyze these customer features:\n{json.dumps(body, default=str)}"


def entity(eid: str, **features: Any) -> Dict[str, Any]:
    return eid, features


def analyze(system_prompt: str, user_prompt: str) -> dict:
    return run_async(MockQwenGateway().generate_json(system_prompt, user_prompt))


def predictions(response: dict) -> List[dict]:
    return response["predictions"]


class TestPromptDispatch:
    def test_the_markers_are_substrings_of_the_real_prompts(self):
        """The gateway routes on prompt wording, so pin the wording down.

        Matched rather than compared so the prompts can be reworded freely, but
        a rewrite that drops one of these phrases fails here instead of turning
        every offline request into a ValueError at runtime.
        """
        assert SCHEMA_RESOLVER_MARKER in SCHEMA_RESOLVER_SYSTEM_PROMPT
        assert SAAS_MARKER in SAAS_CORE_SYSTEM_PROMPT
        assert TELECOM_MARKER in TELECOM_CORE_SYSTEM_PROMPT
        assert FINTECH_MARKER in FINTECH_CORE_SYSTEM_PROMPT

    def test_each_sector_prompt_reaches_its_own_scorer(self):
        body = payload(entity("u1", failure_rate=0.9), entity("u2", failure_rate=0.0))
        for system_prompt, signals in (
            (SAAS_CORE_SYSTEM_PROMPT, SAAS_SIGNALS),
            (FINTECH_CORE_SYSTEM_PROMPT, FINTECH_SIGNALS),
        ):
            response = analyze(system_prompt, body)
            assert len(predictions(response)) == 2
            assert "failure_rate" in signals

    def test_the_schema_prompt_reaches_the_resolver(self):
        response = analyze(
            SCHEMA_RESOLVER_SYSTEM_PROMPT,
            'Analyze these file samples:\n{"a.csv": "user_id,tier\\nu1,pro\\n"}',
        )
        assert response["primary_entity_key"] == "user_id"

    def test_an_unrecognised_system_prompt_raises(self):
        """No silent default: an unknown caller is a bug, not a fallback case."""
        with pytest.raises(ValueError, match="no offline behaviour"):
            analyze("You are a poetry generator.", payload(entity("u1", failure_rate=0.5)))

    def test_a_sector_prompt_given_an_object_instead_of_a_list_raises(self):
        with pytest.raises(ValueError, match="Expected a list of customer features"):
            analyze(SAAS_CORE_SYSTEM_PROMPT, 'Analyze these:\n{"entity_id": "u1"}')


class TestPayloadExtraction:
    def test_prose_before_the_json_is_skipped(self):
        assert extract_payload('Some words, then [{"a": 1}]') == [{"a": 1}]

    def test_a_bracket_inside_the_prose_does_not_end_the_scan(self):
        """The first bracket that fails to parse is skipped, not fatal."""
        assert extract_payload("Ratio [x] then {\"a\": 1}") == {"a": 1}

    def test_multiline_indented_json_is_parsed(self):
        rendered = json.dumps({"a.csv": "user_id\nu1"}, indent=2)
        assert extract_payload(f"Analyze these file samples:\n{rendered}") == {"a.csv": "user_id\nu1"}

    def test_no_json_at_all_raises(self):
        with pytest.raises(ValueError, match="No JSON payload"):
            extract_payload("just prose, nothing else")


class TestDeterminism:
    def test_two_calls_return_identical_output(self):
        body = payload(
            entity("u1", failure_rate=0.8, activity_velocity=0.1, text_churn_score=4.0),
            entity("u2", failure_rate=0.0, activity_velocity=1.2, text_churn_score=0.0),
        )
        first = analyze(SAAS_CORE_SYSTEM_PROMPT, body)
        second = analyze(SAAS_CORE_SYSTEM_PROMPT, body)

        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_entity_order_in_the_batch_does_not_change_any_score(self):
        """Normalising across a batch must not make it order-sensitive."""
        a = entity("u1", failure_rate=0.8, activity_velocity=0.1)
        b = entity("u2", failure_rate=0.2, activity_velocity=1.1)
        forward = {p["entity_id"]: p for p in predictions(analyze(SAAS_CORE_SYSTEM_PROMPT, payload(a, b)))}
        backward = {p["entity_id"]: p for p in predictions(analyze(SAAS_CORE_SYSTEM_PROMPT, payload(b, a)))}

        assert forward == backward


class TestScoreShape:
    def test_identical_customers_are_indistinguishable(self):
        """Nothing is canned: with no variation there is nothing to score on.

        Every signal is constant across the batch, so none can discriminate, and
        both entities land on the midpoint rather than being given a made-up
        spread.
        """
        features = {"failure_rate": 0.4, "activity_velocity": 1.0}
        response = analyze(
            SAAS_CORE_SYSTEM_PROMPT,
            payload(entity("u1", **features), entity("u2", **features)),
        )
        probabilities = [p["churn_prediction"]["churn_probability"] for p in predictions(response)]

        assert probabilities == [0.5, 0.5]
        assert all(p["churn_prediction"]["primary_drivers"] == NO_SIGNAL_DRIVERS for p in predictions(response))

    def test_a_stronger_signal_produces_a_higher_probability(self):
        response = analyze(
            SAAS_CORE_SYSTEM_PROMPT,
            payload(entity("bad", failure_rate=0.9), entity("good", failure_rate=0.0)),
        )
        by_id = {p["entity_id"]: p["churn_prediction"]["churn_probability"] for p in predictions(response)}

        # failure_rate is the only varying signal, so the raw score is exactly
        # the normalised value and the probability is the sigmoid of it. Rounded
        # to four decimals on the way out, hence the tolerance.
        assert by_id["good"] == pytest.approx(_sigmoid(SIGMOID_STEEPNESS * (0.0 - MIDPOINT_SCORE)), abs=1e-4)
        assert by_id["bad"] == pytest.approx(_sigmoid(SIGMOID_STEEPNESS * (1.0 - MIDPOINT_SCORE)), abs=1e-4)
        assert by_id["bad"] > by_id["good"]

    def test_probabilities_stay_inside_the_unit_interval(self):
        response = analyze(
            SAAS_CORE_SYSTEM_PROMPT,
            payload(
                entity("u1", failure_rate=1e9, activity_velocity=0.0, recency_days=1e9),
                entity("u2", failure_rate=0.0, activity_velocity=1e9, recency_days=0.0),
            ),
        )
        for prediction in predictions(response):
            assert 0.0 <= prediction["churn_prediction"]["churn_probability"] <= 1.0

    def test_a_missing_feature_does_not_push_an_entity_to_an_extreme(self):
        """An entity with no rows in a table sits at the batch average."""
        response = analyze(
            SAAS_CORE_SYSTEM_PROMPT,
            payload(
                entity("high", failure_rate=1.0),
                entity("low", failure_rate=0.0),
                entity("absent"),
            ),
        )
        by_id = {p["entity_id"]: p["churn_prediction"]["churn_probability"] for p in predictions(response)}

        assert by_id["low"] < by_id["absent"] < by_id["high"]

    def test_every_prediction_carries_a_well_formed_playbook(self):
        from churn_platform.domain.models.retention_playbook import RetentionPlaybook

        response = analyze(
            SAAS_CORE_SYSTEM_PROMPT,
            payload(entity("u1", failure_rate=0.9), entity("u2", failure_rate=0.1)),
        )
        for prediction in predictions(response):
            playbook = RetentionPlaybook(**prediction["retention_playbook"])
            assert playbook.action_type and playbook.action_payload and playbook.channel

    def test_drivers_are_capped_and_ordered_by_contribution(self):
        response = analyze(
            SAAS_CORE_SYSTEM_PROMPT,
            payload(
                entity("worst", failure_rate=1.0, text_churn_score=5.0, recency_days=60.0, export_ratio=1.0),
                entity("best", failure_rate=0.0, text_churn_score=0.0, recency_days=0.0, export_ratio=0.0),
            ),
        )
        drivers = next(p for p in predictions(response) if p["entity_id"] == "worst")["churn_prediction"]["primary_drivers"]

        assert len(drivers) <= 3
        # failure_rate carries the largest weight, so it is named first.
        assert drivers[0] == SAAS_SIGNALS["failure_rate"].label


class TestRiskTiers:
    def test_tiers_are_monotonic_in_probability(self):
        probabilities = [i / 100 for i in range(101)]
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, CRITICAL_TIER: 3}
        ranks = [order[risk_tier(p)] for p in probabilities]

        assert ranks == sorted(ranks)

    def test_thresholds_are_the_documented_boundaries(self):
        assert risk_tier(0.29) == "LOW"
        assert risk_tier(0.30) == "MEDIUM"
        assert risk_tier(0.54) == "MEDIUM"
        assert risk_tier(0.55) == "HIGH"
        assert risk_tier(0.79) == "HIGH"
        assert risk_tier(0.80) == CRITICAL_TIER

    def test_every_threshold_is_strictly_increasing(self):
        values = [threshold for threshold, _ in RISK_THRESHOLDS]
        assert values == sorted(values)
        assert len(set(values)) == len(values)


class TestNormalisationBounds:
    def test_a_binary_flag_is_left_on_its_natural_scale(self):
        """Winsorising a rare flag would erase it entirely.

        A flag held by one customer in ten has a 90th percentile of zero, so the
        span collapses to zero and the signal is dropped from the score.
        """
        values = pd.Series([True] + [False] * 9).astype("float64")

        assert _normalisation_bounds(values) == (0.0, 1.0)

    def test_a_continuous_signal_is_winsorised(self):
        values = pd.Series([float(i) for i in range(101)])
        low, high = _normalisation_bounds(values)

        assert low == pytest.approx(values.quantile(WINSORISE_QUANTILE))
        assert high == pytest.approx(values.quantile(1.0 - WINSORISE_QUANTILE))
        assert high < values.max()

    def test_a_single_outlier_does_not_flatten_the_batch(self):
        """One tenfold spike must not push every real customer to zero."""
        values = pd.Series([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03, 0.97, 1.01, 40.0])
        low, high = _normalisation_bounds(values)

        assert high < 40.0
        assert math.isfinite(high - low) and high - low > 0


class TestSigmoid:
    def test_zero_maps_to_a_coin_flip(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_it_is_strictly_increasing(self):
        xs = [-10.0, -1.0, 0.0, 1.0, 10.0]
        ys = [_sigmoid(x) for x in xs]

        assert ys == sorted(ys)
        assert len(set(ys)) == len(ys)

    def test_it_stays_inside_the_unit_interval_over_the_reachable_domain(self):
        """Raw scores are bounded, so the sigmoid is only asked about a narrow range.

        Every normalised signal is clipped to [0, 1] and the raw score is their
        weighted mean divided by the total absolute weight, which puts it in
        [-1, 1]. Probing `_sigmoid` at 1e6 would be testing an input the scorer
        cannot produce.
        """
        for raw in (-1.0, -0.5, 0.0, MIDPOINT_SCORE, 0.5, 1.0):
            probability = _sigmoid(SIGMOID_STEEPNESS * (raw - MIDPOINT_SCORE))
            assert 0.0 <= probability <= 1.0
