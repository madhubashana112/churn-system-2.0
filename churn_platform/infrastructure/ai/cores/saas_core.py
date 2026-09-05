from typing import List, Tuple
from churn_platform.domain.interfaces.i_churn_core import IChurnCore
from churn_platform.domain.interfaces.i_ai_gateway import IAIGateway
from churn_platform.domain.models.customer_features import CustomerFeatures
from churn_platform.domain.models.churn_prediction import ChurnPrediction
from churn_platform.domain.models.retention_playbook import RetentionPlaybook
from churn_platform.infrastructure.ai.prompts.saas_prompts import SAAS_CORE_SYSTEM_PROMPT
import json

class SaasCore(IChurnCore):
    def __init__(self, gateway: IAIGateway):
        self.gateway = gateway

    async def analyze(self, features: List[CustomerFeatures]) -> List[Tuple[ChurnPrediction, RetentionPlaybook]]:
        features_json = json.dumps([f.model_dump() for f in features], default=str)
        user_prompt = f"Analyze these customer features:\n{features_json}"
        
        response = await self.gateway.generate_json(SAAS_CORE_SYSTEM_PROMPT, user_prompt)
        
        results = []
        for pred_data in response.get("predictions", []):
            churn_pred = ChurnPrediction(
                entity_id=pred_data["entity_id"],
                churn_probability=pred_data["churn_prediction"]["churn_probability"],
                risk_tier=pred_data["churn_prediction"]["risk_tier"],
                primary_drivers=pred_data["churn_prediction"].get("primary_drivers")
            )
            playbook = RetentionPlaybook(**pred_data["retention_playbook"])
            results.append((churn_pred, playbook))
            
        return results
