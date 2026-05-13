from app.core.context.context import Context
from app.core.enums import Stages
from app.core.stages.evaluation.evaluator_factory import EvaluatorFactory
from app.utils.logger import logger
from app.core.experiments import ExperimentResult


class EvaluationStage:
    """
    Orchestrates evaluation of experiment results.

    Workflow:
    1. Get experiments from context
    2. Select the best experiment based on metrics
    3. Delegate to stage-specific evaluator

    No stage-specific logic here - all delegated via factory.
    """

    def __init__(self, stage: Stages, context: Context):
        self.stage = stage
        self.context = context
        self.config = context.config

    def run(self):
        """Execute the evaluation workflow."""
        logger.info(f"Running evaluation stage: {self.stage}...")

        # Get experiments from context
        try:
            experiments = self._get_experiments()
        except RuntimeError as e:
            logger.warning(f"No experiments to evaluate for {self.stage.value}: {e}")
            return

        # Delegate to stage-specific evaluator
        evaluator = EvaluatorFactory.create(self.stage, self.context)

        if not evaluator:
            logger.info(f"No evaluator for {self.stage.value}, skipping evaluation")
            return

        try:
            evaluator.evaluate(experiments)
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")

        logger.info(
            f"Metadata from {self.stage}: {self.context.stage_results[self.stage].metadata}"
        )

    def _get_experiments(self) -> list[ExperimentResult]:
        """Get experiments from context."""
        stage_result = self.context.stage_results.get(self.stage, {})

        if not stage_result or not stage_result.results:
            raise RuntimeError(
                f"No experiments found for stage {self.context.stage_results} "
            )

        return stage_result.results
