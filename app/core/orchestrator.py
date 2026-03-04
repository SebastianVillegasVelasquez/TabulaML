from app.core.context.run_context import RunContext
from app.core.context.stages import Stages


class Orchestrator:
    def __init__(self,
                 context: RunContext
                 ) -> None:
        self.context = context

    def run(self):
        self.data_inspection_stage()
        self.feature_selection_stage()

    def data_inspection_stage(self):
        from app.core.stages.data_inspection.data_inspection import DataInspectionStage

        DataInspectionStage(context=self.context).run()

    def feature_selection_stage(self):
        from app.core.stages.feature_selection.feature_selection_stage import FeatureSelectionStage

        FeatureSelectionStage(context=self.context).run()

        self.evaluation_stage(Stages.FEATURE_SELECTION)

    def model_selection_stage(self):
        from app.core.stages.model_selection.model_selection_stage import ModelSelectionStage

        ModelSelectionStage(context=self.context).run()

    def evaluation_stage(self, stage: Stages):
        from app.core.stages.evaluation.evaluation_stage import EvaluationStage
        EvaluationStage(
            stage=stage,
            context=self.context
        ).run()
