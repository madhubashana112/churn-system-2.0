"""Orchestrates feature synthesis plus optional sector-specific enrichment."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import pandas as pd

from churn_platform.domain.interfaces.i_feature_synthesizer import IFeatureSynthesizer
from churn_platform.domain.models.customer_features import CustomerFeatures
from churn_platform.domain.models.schema_mapping import SchemaMapping

# Injected rather than imported, so this use case never depends on the
# infrastructure layer that happens to provide the implementation.
FeatureEnricher = Callable[
    [str, List[CustomerFeatures], SchemaMapping, Dict[str, pd.DataFrame]],
    List[CustomerFeatures],
]


class SynthesizeFeaturesUseCase:
    def __init__(
        self,
        synthesizer: IFeatureSynthesizer,
        enricher: Optional[FeatureEnricher] = None,
    ) -> None:
        self.synthesizer = synthesizer
        self.enricher = enricher

    def execute(
        self,
        schema: SchemaMapping,
        dataframes: Dict[str, pd.DataFrame],
        sector: Optional[str] = None,
    ) -> List[CustomerFeatures]:
        schema, dataframes = self.synthesizer.prepare(schema, dataframes)
        features = self.synthesizer.synthesize(schema, dataframes)
        if sector and self.enricher is not None:
            features = self.enricher(sector, features, schema, dataframes)
        return features
