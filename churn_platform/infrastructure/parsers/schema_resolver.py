from typing import Dict
from churn_platform.domain.interfaces.i_ai_gateway import IAIGateway
from churn_platform.domain.interfaces.i_schema_resolver import ISchemaResolver
from churn_platform.domain.models.schema_mapping import SchemaMapping, TableClassification
from churn_platform.infrastructure.ai.prompts.schema_resolver_prompts import SCHEMA_RESOLVER_SYSTEM_PROMPT
import json

class AISchemaResolver(ISchemaResolver):
    def __init__(self, gateway: IAIGateway):
        self.gateway = gateway

    async def resolve(self, file_samples: Dict[str, str]) -> SchemaMapping:
        user_prompt = f"Analyze these file samples:\n{json.dumps(file_samples, indent=2)}"

        response = await self.gateway.generate_json(SCHEMA_RESOLVER_SYSTEM_PROMPT, user_prompt)

        return SchemaMapping(
            primary_entity_key=response["primary_entity_key"],
            tables=[TableClassification(**table_data) for table_data in response.get("tables", [])],
        )
