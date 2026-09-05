"""Run a sector core over every entity, chunking the calls it makes.

Chunking exists because a live LLM has a context window: a thousand feature
dictionaries in one prompt does not fit. Calls are awaited sequentially rather
than gathered, so the provider's rate limit is respected, and a batch that fails
is logged and skipped instead of taking the whole run down with it.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Sequence, Tuple

from churn_platform.domain.interfaces.i_churn_core import IChurnCore
from churn_platform.domain.models.churn_prediction import ChurnPrediction
from churn_platform.domain.models.customer_features import CustomerFeatures
from churn_platform.domain.models.retention_playbook import RetentionPlaybook

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 10


def chunked(items: Sequence[CustomerFeatures], size: int) -> Iterator[Sequence[CustomerFeatures]]:
    """Yield consecutive slices of ``size``. A non-positive size yields one slice."""
    if size <= 0:
        yield items
        return
    for start in range(0, len(items), size):
        yield items[start:start + size]


class ExecuteSectorAnalysisUseCase:
    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.batch_size = batch_size

    async def execute(
        self,
        core: IChurnCore,
        features: List[CustomerFeatures],
    ) -> List[Tuple[ChurnPrediction, RetentionPlaybook]]:
        if not features:
            return []

        batches = list(chunked(features, self.batch_size))
        results: List[Tuple[ChurnPrediction, RetentionPlaybook]] = []

        for index, batch in enumerate(batches, start=1):
            try:
                results.extend(await core.analyze(list(batch)))
            except Exception:
                # The caller can see the shortfall by comparing the number of
                # predictions against the number of entities it uploaded.
                logger.exception(
                    "Batch %d/%d failed (%d entities); continuing with the rest",
                    index, len(batches), len(batch),
                )

        return results
