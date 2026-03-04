from app.core.context.problems_type import ProblemsType
from app.core.context.run_context import RunContext
from app.core.context.stages import Stages
from app.core.domain.experiments.experiment_result import ExperimentResult
from app.core.stages.evaluation.evaluator import Evaluator
from app.core.stages.evaluation.model_registry import ModelRegistry
from app.utils.logger import logger


class EvaluationStage:

    def __init__(
            self,
            stage: Stages,
            context: RunContext,
            registry: ModelRegistry | None = None
    ):
        self.stage = stage
        self.context = context
        self.config = context.config
        self.registry = registry or ModelRegistry()

    def run(self) -> None:
        experiments = self._get_experiments()
        best_experiment = self._evaluate(experiments)
        self._handle_stage_specific_logic(best_experiment)

        if self.stage == Stages.FEATURE_SELECTION:
            stage_result = self.context.stage_results[self.stage]
            stage_result.metadata['selector'] = best_experiment.config.get('selector')
            stage_result.metadata['predictor'] = best_experiment.config.get('predictor')
            self.context.update_context(self.stage, stage_result)
            self.context.stage_results[self.stage].results = None
            logger.info(f"Updated context, final context results {self.context.stage_results[self.stage]}")

        artifact_path = self._persist(best_experiment)

        self._update_context(best_experiment, artifact_path)
        return None

    def _get_experiments(self) -> list[ExperimentResult]:
        stage_result = self.context.stage_results.get(self.stage)

        if not stage_result or not stage_result.results:
            raise RuntimeError(
                f"No experiments found for stage {self.stage}"
            )

        return stage_result.results

    def _evaluate(
            self,
            experiments: list[ExperimentResult]
    ) -> ExperimentResult:
        priority = self.config.priority_metrics

        if self.config.problem_type == ProblemsType.REGRESSION:
            mode = "min"
        else:
            mode = "max"

        evaluator = Evaluator(
            metric=priority,
            mode=mode
        )

        for exp in experiments:
            evaluator.add_result(exp)

        return evaluator.get_best()

    def _handle_stage_specific_logic(
            self,
            experiment: ExperimentResult
    ):
        if self.stage == Stages.FEATURE_SELECTION:
            self._handle_feature_selection(experiment)

    def _handle_feature_selection(
            self,
            experiment: ExperimentResult
    ):
        pipeline = experiment.pipeline

        if "feature_selection" not in pipeline.named_steps:
            return

        selector = pipeline.named_steps["feature_selection"]

        if not hasattr(selector, "get_support"):
            return

        mask = selector.get_support()

        original_columns = self.config.X_train.columns.tolist()

        selected_columns = [
            col for col, keep in zip(original_columns, mask) if keep
        ]

        experiment.feature_mask = mask
        experiment.selected_features = selected_columns

        stage_result = self.context.stage_results[self.stage]
        stage_result.feature_importance = {
            col: 1.0 for col in selected_columns
        }

    def _persist(
            self,
            experiment: ExperimentResult
    ) -> str:
        model_name = f"{self.stage.name.lower()}_best"

        return self.registry.register(
            result=experiment,
            name=model_name
        )

    def _update_context(
            self,
            experiment: ExperimentResult,
            artifact_path: str
    ):
        if self.stage == Stages.FEATURE_SELECTION:
            return

        stage_result = self.context.stage_results[self.stage]

        stage_result.best_pipeline_path = artifact_path
        stage_result.best_experiment = experiment

        self.context.update_context(self.stage, stage_result)
