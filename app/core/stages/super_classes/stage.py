from abc import ABC, abstractmethod

from app.core.context.run_context import RunContext, StageResult
from app.core.enums.stages import Stages
from app.core.domain.experiments.experiment import Experiment
from app.core.stages.factories.registry import get_stage_experiments
from app.utils.logger import logger


class Stage(ABC):
    """
    Base class for pipeline stages.
    
    Responsibilities:
    - Run experiments for different configurations
    - Store raw results in context
    - Evaluation/sorting/selection is handled by EvaluationStage
    """

    def __init__(self, context: RunContext):
        self.context = context
        self.stage = self.get_stage_type()
        self.definitions = get_stage_experiments(self.stage, context=self.context)

    @abstractmethod
    def get_stage_type(self) -> Stages:
        """Child classes must define their stage type"""
        pass

    def run(self):
        """Execute experiments for this stage."""
        logger.info(f"Running {self.stage} stage...")

        preprocessing = self.context.stage_results[Stages.DATA_HANDLER].results["preprocessing"]
        results = []
        for definition in self.definitions:

            pipeline_or_builder = definition.builder(preprocessing)

            experiment = Experiment(
                name=f"{self.stage.value}_{definition.name}",
                pipeline_builder=pipeline_or_builder,
                context=self.context,
                cv=5,
                metadata=definition.metadata
            )

            result = experiment.run(self.context.config.X_train, self.context.config.y_train)
            results.append(result)
            logger.info(f"Finished experiment {experiment.name}")

        self.context.stage_results[self.stage] = StageResult(
            name=self.stage,
            results=results,
            metadata={"total_experiments": len(results)}
        )

        logger.info(f"Completed {len(results)} experiments for {self.stage.value}")