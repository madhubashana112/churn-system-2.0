"""Complete a resolved upload using injected synthesis, scoring and persistence ports."""
from churn_platform.application.dtos.analysis_response_dto import AnalysisResponse, PredictionResult
from churn_platform.application.use_cases.synthesize_features import SynthesizeFeaturesUseCase
from churn_platform.domain.models.analysis_run import AnalysisRun, EntityOutcome


class AnalysisUnavailable(RuntimeError):
    """No submitted customer could be scored; leave the previous run intact."""


class CompleteUploadAnalysisUseCase:
    def __init__(self, synthesizer, enricher, analysis, repository, cap, offline):
        self.synthesizer, self.enricher = synthesizer, enricher
        self.analysis, self.repository = analysis, repository
        self.cap, self.offline = cap, offline

    async def execute(self, tenant, schema, dataframes, original_files, core):
        features = SynthesizeFeaturesUseCase(
            self.synthesizer,
            enricher=self.enricher,
        ).execute(schema, dataframes, sector=tenant.sector)

        warnings: list[str] = []
        entities_uploaded = len(features)
        cap = self.cap
        if cap > 0 and entities_uploaded > cap:
            warnings.append(
                f"MAX_ENTITIES={cap} is set, so only the first {cap} of "
                f"{entities_uploaded} entities were analyzed"
            )
            features = features[:cap]

        results = await self.analysis.execute(core, features)
        if features and not results:
            raise AnalysisUnavailable("No customers could be scored. The analysis service may be busy or rate-limited. "
                                      "Please retry later; your previous results have been kept.")
        if len(results) < len(features):
            warnings.append(
                f"{len(features) - len(results)} of the {len(features)} submitted entities "
                "could not be scored; see the server log for the failed batch"
            )

        # offline_mode describes this run, not the deployment: the dashboard reloads
        # a stored analysis long after the request, and both engines can now be in
        # play in the same process.
        offline = self.offline

        # The features travel with the run: the sector KPIs are recomputed from them
        # on every dashboard load, so a stored prediction without its evidence could
        # only ever be restated, not re-aggregated.
        features_by_id = {entry.entity_id: entry.features for entry in features}
        await self.repository.save(AnalysisRun(
            tenant_id=tenant.tenant_id,
            sector=tenant.sector,
            schema_mapping=schema,
            original_files=original_files,
            outcomes=[
                EntityOutcome(prediction=pred, playbook=playbook, features=features_by_id.get(pred.entity_id, {}))
                for pred, playbook in results
            ],
            entities_uploaded=entities_uploaded,
            entities_analyzed=len(results),
            offline_mode=offline,
            warnings=warnings,
        ))

        return AnalysisResponse(
            schema_mapping=schema,
            predictions=[
                PredictionResult(prediction=pred, playbook=playbook)
                for pred, playbook in results
            ],
            entities_uploaded=entities_uploaded,
            entities_analyzed=len(results),
            warnings=warnings,
            offline_mode=offline,
        )
