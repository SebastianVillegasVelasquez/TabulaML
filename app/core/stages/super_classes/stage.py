from abc import ABC, abstractmethod

from app.core.context import Context, StageResult
from app.core.enums import Stages
from app.utils.logger import logger
from app.core.experiments import Experiment


class Stage(ABC):
    """
    Base class for pipeline stages.

    Responsibilities:
    - Run experiments for different configurations
    - Store raw results in context
    - Evaluation/sorting/selection is handled by EvaluationStage
    """

    def __init__(self, context: Context):
        from app.core.stages.factories.registry import get_stage_experiments

        self.context = context
        self.stage = self.get_stage_type()
        self.definitions = get_stage_experiments(self.stage, context=self.context)

    @abstractmethod
    def get_stage_type(self) -> Stages:
        """Child classes must define their stage model_based"""
        pass

    def run(self):
        """Execute experiments for this stage."""

        X = self.context.stage_results[Stages.DATA_HANDLER].results["df_transformed"]

        results = []
        for definition in self.definitions:
            logger.debug(f"Metadata: {definition.metadata}")

            experiment = Experiment(
                name=f"{self.stage.value}_{definition.name}",
                pipeline=definition.pipeline_builder,
                context=self.context,
                stage=self.stage,
                cv=5,
                metadata=definition.metadata,
                evaluation_type=definition.evaluation_type,
            )

            result = experiment.run(X, self.context.config.dataset.y_train)
            results.append(result)

        self.context.stage_results[self.stage] = StageResult(
            name=self.stage, results=results, metadata={"total_experiments": len(results)}
        )
