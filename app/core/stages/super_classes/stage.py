from abc import ABC, abstractmethod

from app.core.context.run_context import RunContext, StageResult
from app.core.domain.experiments.experiment import Experiment
from app.core.enums.stages import Stages
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
        from app.core.stages.factories.registry import get_stage_experiments
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

            logger.debug(f"builder: {definition.builder}")
            logger.debug(f"builder type: {type(definition.builder)}")
            logger.debug(f"builder callable: {callable(definition.builder)}")
            logger.debug(f"builder has __call__: {hasattr(definition.builder, '__call__')}")
            logger.debug(f"parameters: {dir(definition.builder)}")


            try:
                if callable(definition.builder) and hasattr(definition.builder, '__call__'):
                    pipeline_or_builder = definition.builder(preprocessing).build()
            except AttributeError:
                logger.debug(f"It is not a PipelineBuilder function but a Pipeline")
                pipeline_or_builder = definition.builder


            # if callable(definition.builder) and hasattr(definition.builder, 'build'):
            #     pipeline_or_builder = definition.builder(preprocessing).build()
            # else:
            #     pipeline_or_builder = definition.builder


            experiment = Experiment(
                name=f"{self.stage.value}_{definition.name}",
                pipeline_builder=pipeline_or_builder,
                context=self.context,
                cv=5,
                metadata=definition.metadata,
                evaluation_type=definition.evaluation_type
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
