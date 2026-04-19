from abc import ABC, abstractmethod

from sklearn.pipeline import Pipeline

from app.core.context.run_context import RunContext, StageResult
from app.core.domain.experiments.experiment import Experiment
from app.core.enums.stages import Stages
from app.core.ml import pipeline_builder, PipelineBuilder
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

            # logger.debug(f"DEFINITION OBJECT:: {definition}")
            #
            # logger.debug(f"builder: {definition.pipeline_builder}")
            # logger.debug(f"builder type: {type(definition.pipeline_builder)}")
            # logger.debug(f"builder callable: {callable(definition.pipeline_builder)}")
            # logger.debug(f"builder has __call__: {hasattr(definition.pipeline_builder, '__call__')}")
            # logger.debug(f"parameters: {dir(definition.pipeline_builder)}")
            # logger.debug(f"pipeline before the handling: {definition.pipeline_builder.build()}")
            pipeline = self._handle_pipeline_builder_callable(definition.pipeline_builder)

            experiment = Experiment(
                name=f"{self.stage.value}_{definition.name}",
                pipeline=pipeline,
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

    @staticmethod
    def _handle_pipeline_builder_callable(pipeline_builder: PipelineBuilder) -> Pipeline | None:
        pipeline = None
        try:
            if hasattr(pipeline_builder, '__call__'):
                pipeline = pipeline_builder.build()
                # logger.debug(f"pipeline: {pipeline}")
            else:
                pipeline = Pipeline(steps=pipeline_builder.steps)

        except AttributeError:
            logger.debug(f"It is not a PipelineBuilder function but a Pipeline")
            logger.debug(f"builder: {type(pipeline_builder)}")
        # logger.debug(f"pipeline after the handling: {pipeline}")
        return pipeline
