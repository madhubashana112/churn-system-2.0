from abc import ABC, abstractmethod
from typing import Dict, List
from churn_platform.domain.models.schema_mapping import SchemaMapping
from churn_platform.domain.models.customer_features import CustomerFeatures
import pandas as pd

class IFeatureSynthesizer(ABC):
    def prepare(self, schema, dataframes):
        return schema, dataframes

    @abstractmethod
    def synthesize(self, schema: SchemaMapping, dataframes: Dict[str, pd.DataFrame]) -> List[CustomerFeatures]:
        pass
