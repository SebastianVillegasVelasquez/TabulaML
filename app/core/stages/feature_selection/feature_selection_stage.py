from app.core.context.run_context import RunContext
from app.core.context.stages import Stages
from app.core.domain.experiments.experiment import Experiment
from app.core.ml.preprocessing_stage import PreprocessingBuilder
from app.core.stages.registry import get_stage_experiments


class FeatureSelectionStage:

    def __init__(self, context=RunContext):
        self.context = context

    def run(self):
        definitions = get_stage_experiments(Stages.FEATURE_SELECTION)
        preprocessing = PreprocessingBuilder(
            feature_configs=self.context.stage_results[Stages.DATA_HANDLER].results["feature_configs"]
        ).build()

        results = []

        for definition in definitions:
            builder = definition.builder(preprocessing)
            experiment = Experiment(
                name=str(self.context.current_stage) + "_" + definition.name,
                pipeline_builder=builder,
                scoring=self.context.config.scoring,
                cv=5,
                metadata=definition.metadata
            )

            results.append(experiment
                       .run(self.context.config.X_train, self.context.config.y_train))

        return results
