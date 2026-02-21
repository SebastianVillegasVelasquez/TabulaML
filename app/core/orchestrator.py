from app.core.context.run_context import RunContext


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

        print(FeatureSelectionStage(context=self.context).run())
