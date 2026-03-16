from app.core.context.problems_type import ProblemsType
from app.core.context.run_context import RunContext
from app.core.context.stages import Stages
from app.core.domain.experiments.experiment_result import ExperimentResult
from app.core.stages.evaluation.evaluator import Evaluator
from app.core.stages.evaluation.evaluator_factory import EvaluatorFactory
from app.utils.logger import logger


class EvaluationStage:
    """
    Orchestrates evaluation of experiment results.
    
    Workflow:
    1. Get experiments from context
    2. Select best experiment based on metrics
    3. Delegate to stage-specific evaluator
    
    No stage-specific logic here - all delegated via factory.
    """

    def __init__(self, stage: Stages, context: RunContext):
        self.stage = stage
        self.context = context
        self.config = context.config

    def run(self):
        """Execute evaluation workflow."""
        logger.info(f"Running evaluation stage: {self.stage}...")
        
        # Get experiments
        experiments = self._get_experiments()
        
        # Evaluate and select best
        best_experiment = self._evaluate(experiments)
        logger.info(f"Best experiment for {self.stage}: {best_experiment.name}")
        
        # Delegate to stage-specific evaluator
        evaluator = EvaluatorFactory.create(self.stage, self.context)
        evaluator.evaluate(experiments)

    def _get_experiments(self) -> list[ExperimentResult]:
        """Get experiments from context."""
        stage_result = self.context.stage_results.get(self.stage, {})

        if not stage_result or not stage_result.results:
            raise RuntimeError(f"No experiments found for stage {self.stage}")

        return stage_result.results

    def _evaluate(self, experiments: list[ExperimentResult]) -> ExperimentResult:
        """Select the best experiment based on configured metrics."""

        if self.config.problem_type == ProblemsType.REGRESSION:
            mode = "min"
        else:
            mode = "max"

        evaluator = Evaluator(context=self.context, mode=mode)

        for exp in experiments:
            evaluator.add_result(exp)

        return evaluator.get_best()

