from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from app.core.context.context import StageResult
from app.core.stages.evaluation.base_evaluator import BaseEvaluator
from app.utils.logger import logger
from experiments import ExperimentResult


class FeatureSelectionEvaluator(BaseEvaluator):
    """This class is responsible for managing the evaluation logic of the Feature Selection stage.

    The idea is to extract the top-3 chain selectors and their associated feature mask.
    Save the metadata from the selector type (linear or non-linear) and the number of features selected.

    Then in the model selection stage, we can use the feature mask to avoid feature extraction
    in each pipeline instead use the mask to select the features combine to just focusing
    if the selector is linear or non-linear.

    """

    def _extract_stage_specific_data(
            self, sorted_results: list[ExperimentResult], best_experiment: ExperimentResult
    ):
        """
        This method extracts the specific data for the Feature Selection stage.

        This includes the top-3 selectors by type, the feature mask, and the number of features selected.

        Args:
            sorted_results (list[ExperimentResult]): List of ExperimentResult sorted by performance
            best_experiment (ExperimentResult): The best ExperimentResult

        """

        # Extract top-3 selectors by type
        top_k_selectors = self._extract_top_k_chain_selectors(sorted_results, k=3)



    def _update_context(self, sorted_results, best_experiment, stage_specific_data):
        """Update context with feature selection results."""
        stage_result = StageResult(
            name=self.stage,
            results=None,
            best_experiment=best_experiment,
            metadata={
                "top_k_selectors": stage_specific_data["top_k_selectors"],
                "selector": stage_specific_data["best_selector"],
                "predictor": stage_specific_data["best_predictor"],
                "n_features_selected": stage_specific_data["n_features_selected"],
                "total_experiments": stage_specific_data["total_experiments"],
                "selector_estimator": best_experiment.pipeline.named_steps.get(
                    "feature_selection", None
                ),
            },
        )

        if stage_specific_data["selected_features"]:
            stage_result.feature_importance = {
                col: 1.0 for col in stage_specific_data["selected_features"]
            }

        self.context.update_stage_context(self.stage, stage_result)

        # Store feature data in experiment for downstream use
        best_experiment.feature_mask = stage_specific_data.get("feature_mask")
        best_experiment.selected_features = stage_specific_data.get("selected_features")

    @staticmethod
    def _extract_top_k_chain_selectors(sorted_results:list[ExperimentResult], k=3) -> dict:
        """
        This method extracts the top-k chain selectors from a list of ExperimentResult.
        From the first k experiments, it needs to extract the chain selector, it can be 1, 2, or more selectors.
        Also, it needs to keep track of the metadata of the final predictor.

        """
        from collections import defaultdict

        chain_data = {}

        # logger.debug(f"Results received: {len(sorted_results)}")
        #
        #
        # for result in sorted_results[:k]:
        #     pipeline: Pipeline = result.pipeline
        #
        #     chain_data[result.name] = {'model': model,
        #                                'selectors': selectors,
        #                                'features_selected': }
        #
        #
        #
        #     logger.debug(f"Chain selector: {selectors[:-1]}")
        #
        # logger.debug(f"Top {k} chain selectors: {chain_data}")
        return chain_data