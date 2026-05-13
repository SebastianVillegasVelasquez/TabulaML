"""
Data Handler Evaluator: Simple pass-through evaluator for data inspection stage.

The data inspection stage doesn't produce experiments to evaluate,
so this evaluator is a no-op. It simply logs that data preparation was successful.
"""

from typing import Any

from app.core.experiments import ExperimentResult
from app.core.stages.evaluation.base_evaluator import BaseEvaluator
from app.utils.logger import logger


class DataHandlerEvaluator(BaseEvaluator):
    """
    No-op evaluator for data inspection/handling stage.

    Data handling stage only prepares and validates the dataset.
    It doesn't produce experiments, so evaluation is trivial.
    """

    def _extract_stage_specific_data(
        self, sorted_results: list[ExperimentResult], best_experiment: ExperimentResult
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """No-op: data handler doesn't produce results to extract."""
        logger.debug("Data handling stage has no experiments to evaluate")
        return None

    def _update_context(
        self,
        sorted_results: list[ExperimentResult],
        best_experiment: ExperimentResult,
        stage_specific_data: dict[str, Any] | list[dict[str, Any]],
    ) -> None:
        """No-op: data handler output is already in context."""
        logger.debug("Data handling stage evaluation completed (no updates needed)")
        self._log_best_experiment(best_experiment)
