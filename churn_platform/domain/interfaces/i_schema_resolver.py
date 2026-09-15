from abc import ABC, abstractmethod
from typing import Dict, List
from churn_platform.domain.models.schema_mapping import SchemaMapping

class ISchemaResolver(ABC):
    @abstractmethod
    async def resolve(self, file_samples: Dict[str, str], remembered: dict | None = None) -> SchemaMapping:
        pass
