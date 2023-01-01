"""
Base Evaluator: Template for stage-specific evaluation logic.

This module defines the abstract base class for all stage evaluators.
Each stage (FeatureSelection, ModelSelection, etc.) has unique requirements
for processing results. This class provides the common template while
allowing subclasses to implement stage-specific behavior.

Design Pattern: Template Method + Strategy
"""

from abc import ABC, abstractmethod
from typing import Any

from app.core.context.context import Context
from app.core.enums import Stages
from experiments import ExperimentResult
from app.utils.logger import logger


class BaseEvaluator(ABC):
    """
    Abstract base class for stage-specific evaluators.

    Provides a template method pattern for evaluation workflow:
    1. Sort experiments by primary metric
    2. Extract best experiment
    3. Handle stage-specific logic
    4. Update context with results

    Subclasses implement hook methods for custom behavior.
    """

    def __init__(self, stage: Stages, context: Context):
        """
        Initialize evaluator.

        Args:
            stage: The stage type being evaluated
            context: The pipeline context
        """
        self.stage = stage
        self.context = context
        self.config = context.config
        self.priority_metric = self.config.priority_metric

    def evaluate(self, results: list[ExperimentResult]) -> None:
        """
        Template method: Execute the evaluation workflow.

        This method orchestrates the complete evaluation process.
        Subclasses should NOT override this but should override
        specific hook methods instead.

        Args:
            results: List of experiment results to evaluate
        """
        logger.info(f"Evaluating {self.stage.value} results...")

        sorted_results = self._sort_results(results)

        best_experiment = sorted_results[0] if sorted_results else None

        if not best_experiment:
            logger.warning(f"No results to evaluate for {self.stage.value}")
            return

        stage_specific_data = self._extract_stage_specific_data(sorted_results, best_experiment)

        self._update_context(sorted_results, best_experiment, stage_specific_data)


    # ========== Hook Methods (Override in Subclasses) ==========

    @abstractmethod
    def _extract_stage_specific_data(
        self, sorted_results: list[ExperimentResult], best_experiment: ExperimentResult
    ) -> dict[str, Any] | list[dict[str, Any]] :
        """
        Extract stage-specific data from results.

        Each stage has unique requirements:
        - FeatureSelection: Extract top-k selectors, feature masks
        - ModelSelection: Extract top-k models from different families
        - FineTuning: Extract best hyperparameters

        Args:
            sorted_results: Results sorted by primary metric
            best_experiment: The best experiment

        Returns:
            Dictionary with stage-specific data to store in context
        """
        pass

    @abstractmethod
    def _update_context(
        self,
        sorted_results: list[ExperimentResult],
        best_experiment: ExperimentResult,
        stage_specific_data: dict[str, Any] | list[dict[str, Any]],
    ) -> None:
        """
        Update pipeline context with evaluation results.

        Args:
            sorted_results: All results sorted by metric
            best_experiment: The best experiment
            stage_specific_data: Stage-specific data from extraction
        """
        pass

    # ========== Common Methods (Reusable) ==========

    def _sort_results(self, results: list[ExperimentResult]) -> list[ExperimentResult]:
        """
        Sort experiments by primary metric in descending order.

        Returns:
            Sorted list (best first)
        """

        sorted_results = sorted(
            results, key=lambda r: r.metrics.get(self.priority_metric, 0), reverse=True
        )

        logger.debug(f"Results sorted by {self.priority_metric}")

        return sorted_results

    @staticmethod
    def _get_model_family(experiment: ExperimentResult) -> str:
        """
        Extract the model family from experiment config.

        Examples: 'RandomForest', 'LogisticRegression', 'XGBoost'

        Args:
            experiment: The experiment to extract from

        Returns:
            Model family name as a string
        """
        model_name = experiment.config.get("model", "unknown")
        return str(model_name)

    def _extract_top_k_by_family(self, sorted_results: list[ExperimentResult], k: int = 3) -> dict:
        """
        Extract top-k results grouping by model family.

        For each family, keeps the best (first) occurrence.
        Returns top-k families overall.

        Args:
            sorted_results: Results already sorted by metric
            k: Number of families to extract

        Returns:
            Dict mapping family name to best result of that family
        """
        family_best = {}

        for result in sorted_results:
            family = self._get_model_family(result)

            # Keep only first (best) occurrence of each family
            if family not in family_best:
                family_best[family] = result

        # Return top-k families
        top_k = dict(list(family_best.items())[:k])
        logger.debug(f"Extracted top {len(top_k)} families: {list(top_k.keys())}")

        return top_k
