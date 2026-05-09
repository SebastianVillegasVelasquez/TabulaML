from typing import Any

from app.core.context.context import StageResult
from app.core.stages.evaluation.base_evaluator import BaseEvaluator
from app.utils.logger import logger
from app.core.experiments import ExperimentResult


class FeatureSelectionEvaluator(BaseEvaluator):
    """This class is responsible for managing the evaluation logic of the Feature Selection stage.

    The idea is to extract the top-3 chain selectors and their associated feature mask.
    Save the metadata from the selector model_based (linear or non-linear) and the number of features selected.

    Then in the model selection stage, we can use the feature mask to avoid feature extraction
    in each pipeline instead use the mask to select the features combine to just focusing
    if the selector is linear or non-linear.

    """

    def _extract_stage_specific_data(
        self, sorted_results: list[ExperimentResult], best_experiment: ExperimentResult
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        This method extracts the specific data for the Feature Selection stage.

        This includes the top-3 selectors by model_based, the feature mask, and the number of features selected.

        Args:
            sorted_results (list[ExperimentResult]): List of ExperimentResult sorted by performance
            best_experiment (ExperimentResult): The best ExperimentResult

        """

        # Extract top-3 selectors by model_based
        return self._extract_top_k_chain_selectors(sorted_results, k=3)

    def _update_context(
        self,
        sorted_results: list[ExperimentResult],
        best_experiment: ExperimentResult,
        stage_specific_data: dict[str, Any] | list[dict[str, Any]],
    ):
        """
        Update the context to ensure the metadata for the model selection stage is available.
        Based on this information, the model selection stage can use the feature mask to avoid feature extraction
        in each pipeline instead use the mask to select the feature combine to just focusing on the selector model_based.

        Also, using the information about the predictor used in the feature extractor step, the model selection stage
        can use a predictor of its model_based as non-linear, linear, both or just use the model based as a tree, svm, etc.

        """
        try:
            stage_result = StageResult(
                name=self.stage,
                results=None,
                best_experiment=best_experiment,
                metadata=stage_specific_data,
            )
        except Exception as e:
            logger.error(f"Error updating context: {e}")
            return

        logger.debug(f"Stage Result: {stage_result}")

        self.context.update_stage_context(self.stage, stage_result)

    @staticmethod
    def _extract_top_k_chain_selectors(
        sorted_results: list[ExperimentResult], k=3
    ) -> list[dict[str, Any]]:
        """
        This method extracts the top-k chain selectors from a list of ExperimentResult.
        From the first k experiments, it needs to extract the chain selector, it can be 1, 2, or more selectors.
        Also, it needs to keep track of the metadata of the final predictor.

        """

        metadata_to_model_selection_stage = []

        for result in sorted_results[:k]:
            metadata_to_model_selection_stage.append(
                {
                    "selectors": result.metadata.get("selectors", []),
                    "predictor": result.metadata.get("model", None),
                    "selector_type": result.metadata.get("selector_type", None),
                    "n_features_selected": result.metadata.get("n_features_selected", None),
                }
            )

        logger.debug(f"Top {k} selectors: {metadata_to_model_selection_stage}")

        return metadata_to_model_selection_stage
