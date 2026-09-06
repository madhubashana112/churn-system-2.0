"""Batching: chunking, ordering, and surviving a batch that fails.

Batching exists because a live LLM has a context window. It is exercised here
against a stub core so the assertions are about the use case's own behaviour —
call count, ordering, partial failure — and not about scoring.
"""

from __future__ import annotations

import statistics
from typing import List, Tuple

from churn_platform.application.use_cases.execute_sector_analysis import (
    DEFAULT_BATCH_SIZE,
    ExecuteSectorAnalysisUseCase,
    chunked,
)
from churn_platform.domain.models.churn_prediction import ChurnPrediction
from churn_platform.domain.models.customer_features import CustomerFeatures
from churn_platform.domain.models.retention_playbook import RetentionPlaybook
from churn_platform.infrastructure.ai.cores.saas_core import SaasCore
from churn_platform.infrastructure.ai.mock_qwen_gateway import MockQwenGateway

from conftest import run_async, sector_features


def features(n: int) -> List[CustomerFeatures]:
    return [CustomerFeatures(entity_id=f"u{i}", features={"failure_rate": i / max(n - 1, 1)}) for i in range(n)]


def _score(results: List[Tuple[ChurnPrediction, RetentionPlaybook]]) -> List[Tuple[str, float]]:
    return [(p.entity_id, p.churn_probability) for p, _ in results]


class RecordingCore:
    """Stands in for a sector core and records the batches it was handed."""

    def __init__(self, fail_on_batch: int | None = None) -> None:
        self.batches: List[List[str]] = []
        self.fail_on_batch = fail_on_batch

    async def analyze(self, batch: List[CustomerFeatures]):
        self.batches.append([f.entity_id for f in batch])
        if self.fail_on_batch is not None and len(self.batches) == self.fail_on_batch:
            raise RuntimeError("provider returned a 429")
        return [
            (
                ChurnPrediction(entity_id=f.entity_id, churn_probability=0.5, risk_tier="MEDIUM"),
                RetentionPlaybook(action_type="X", action_payload="y", channel="EMAIL"),
            )
            for f in batch
        ]


class TestChunking:
    def test_evenly_divisible_input(self):
        assert [[f.entity_id for f in c] for c in chunked(features(6), 3)] == [
            ["u0", "u1", "u2"], ["u3", "u4", "u5"],
        ]

    def test_a_remainder_forms_a_short_final_batch(self):
        assert [len(c) for c in chunked(features(7), 3)] == [3, 3, 1]

    def test_a_batch_larger_than_the_input_is_one_batch(self):
        assert [len(c) for c in chunked(features(3), 100)] == [3]

    def test_a_non_positive_size_means_a_single_call(self):
        """The escape hatch offline mode uses: no chunking at all."""
        assert [len(c) for c in chunked(features(5), 0)] == [5]
        assert [len(c) for c in chunked(features(5), -1)] == [5]

    def test_the_default_batch_size_is_the_documented_ten(self):
        assert DEFAULT_BATCH_SIZE == 10


class TestExecuteSectorAnalysis:
    def test_every_entity_is_returned_exactly_once_in_order(self):
        core = RecordingCore()
        results = run_async(ExecuteSectorAnalysisUseCase(batch_size=4).execute(core, features(10)))

        assert [p.entity_id for p, _ in results] == [f"u{i}" for i in range(10)]
        assert [len(b) for b in core.batches] == [4, 4, 2]

    def test_batches_are_awaited_in_sequence(self):
        """Sequential rather than gathered, so a provider rate limit is respected."""
        core = RecordingCore()
        run_async(ExecuteSectorAnalysisUseCase(batch_size=3).execute(core, features(9)))

        assert core.batches == [["u0", "u1", "u2"], ["u3", "u4", "u5"], ["u6", "u7", "u8"]]

    def test_a_batch_size_of_zero_sends_the_whole_population(self):
        core = RecordingCore()
        run_async(ExecuteSectorAnalysisUseCase(batch_size=0).execute(core, features(25)))

        assert len(core.batches) == 1
        assert len(core.batches[0]) == 25

    def test_a_failing_batch_does_not_take_the_run_down_with_it(self):
        core = RecordingCore(fail_on_batch=2)
        results = run_async(ExecuteSectorAnalysisUseCase(batch_size=3).execute(core, features(9)))

        assert len(core.batches) == 3
        assert [p.entity_id for p, _ in results] == ["u0", "u1", "u2", "u6", "u7", "u8"]

    def test_the_shortfall_is_visible_to_the_caller(self):
        """The endpoint reports it by comparing analyzed against uploaded."""
        core = RecordingCore(fail_on_batch=1)
        uploaded = features(6)
        results = run_async(ExecuteSectorAnalysisUseCase(batch_size=3).execute(core, uploaded))

        assert len(results) < len(uploaded)

    def test_an_empty_batch_list_returns_nothing_and_calls_nothing(self):
        core = RecordingCore()

        assert run_async(ExecuteSectorAnalysisUseCase(batch_size=5).execute(core, [])) == []
        assert core.batches == []

    def test_results_are_domain_models_not_dicts(self):
        results = run_async(ExecuteSectorAnalysisUseCase(batch_size=2).execute(RecordingCore(), features(3)))

        assert all(isinstance(p, ChurnPrediction) and isinstance(b, RetentionPlaybook) for p, b in results)


class TestWhyOfflineModeIsNotChunked:
    def test_chunking_the_offline_scorer_changes_its_verdict(self):
        """The reason `get_analysis_batch_size` returns 0 for the mock.

        The offline scorer normalises every signal across the batch it is
        handed. Slice a hundred customers into tens and each slice re-normalises
        against its own worst and best, so the healthiest customer in an
        all-healthy slice is scored as if they were the riskiest in the tenant.
        """
        all_features = sector_features("SaaS")
        core = SaasCore(MockQwenGateway())

        whole = dict(_score(run_async(core.analyze(all_features))))
        chunk_scores: dict = {}
        for batch in chunked(all_features, 10):
            chunk_scores.update(_score(run_async(core.analyze(list(batch)))))

        assert set(whole) == set(chunk_scores)
        deltas = [abs(whole[eid] - chunk_scores[eid]) for eid in whole]
        assert max(deltas) > 0.05
        assert statistics.mean(deltas) > 0.01
