"""Response shape for an analysis run.

Typed rather than ``List[dict]`` so the OpenAPI schema documents what the
dashboard is actually consuming.
"""

from pydantic import BaseModel
from typing import List

from churn_platform.domain.models.churn_prediction import ChurnPrediction
from churn_platform.domain.models.retention_playbook import RetentionPlaybook
from churn_platform.domain.models.schema_mapping import SchemaMapping


class PredictionResult(BaseModel):
    prediction: ChurnPrediction
    playbook: RetentionPlaybook


class AnalysisResponse(BaseModel):
    schema_mapping: SchemaMapping
    predictions: List[PredictionResult]

    entities_uploaded: int
    entities_analyzed: int
    # Anything the caller should know that is not an outright failure: a batch
    # that errored, or an upload truncated by MAX_ENTITIES.
    warnings: List[str] = []
    # True when predictions were computed locally instead of by Qwen, so the UI
    # can say so rather than implying an LLM judged every customer.
    offline_mode: bool = False
