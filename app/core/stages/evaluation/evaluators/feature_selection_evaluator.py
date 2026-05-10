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
    ) -> None:
        """Persists the feature-selection outcome into the shared context.

        Stores a class `StageResult` that contains:

        - "best_experiment": the single top-performing experiment, **with
          "model_type" and "model_based" removed from its own
          "metadata"** so those keys live only in the stage-level metadata
          and are not accidentally double-read by the next stage.
        - "metadata": the "stage_specific_data" list produced by
          :func:`_extract_top_k_chain_selectors`, which already contains
          "model_type" and "model_based" from every top-k result.

        The model-selection stage reads "model_type" and "model_based"
        from "context.stage_results[FEATURE_SELECTION].metadata[0]" (the
        best entry), not from "best_experiment.metadata".

        Args:
            sorted_results: All experiment results, best-first.  Not stored
                directly but available for future extensions.
            best_experiment: The top-ranking
                :class:`~app.core.experiments.experiment_result.ExperimentResult`.
                Its "metadata" is mutated in-place to remove
                ""model_type"" and ""model_based"" before storage.
            stage_specific_data: Structured metadata produced by
                :func:`_extract_top_k_chain_selectors` for the top-k results.
        """
        try:
            # Remove model_type / model_based from the best experiment's own
            # metadata so there is a single authoritative source of truth:
            # the stage-level metadata list built by
            # _extract_top_k_chain_selectors.
            best_experiment.metadata.pop("model_type", None)
            best_experiment.metadata.pop("model_based", None)

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
        sorted_results: list[ExperimentResult], k: int = 3
    ) -> list[dict[str, Any]]:
        """Extracts selector-chain metadata from the top-k feature-selection results.

        Reads the keys that "_build_experiment" actually writes into each
        experiment's "metadata" dict — ""selectors"", ""model"",
        ""model_type"", and ""model_based"" — and re-packages them into a
        flat list of dicts suitable for storage in :class:`StageResult`.

        ""selector_type"" and ""n_features_selected"" were previously
        collected here but are not written by "_build_experiment", so they
        have been removed to avoid silent "None" values.

        Args:
            sorted_results: Experiment results ordered from best to worst
                (e.g. descending by primary metric).  Only the first *k*
                entries are consumed.
            k: Number of top results to include.  Defaults to 3.

        Returns:
            list[dict[str, Any]]: One dict per top-k result, each containing:

            - ""selectors"" (list[str]): Ordered selector names used in
              the chain (e.g. "["selectkbest", "rfe_non_linear"]").
            - ""model"" (str | None): Name of the predictor used during
              feature selection (e.g. ""RandomForestClassifier"").
            - ""model_type"" (:class:`~app.core.enums.ModelSpecType` | None):
              Broad family of that predictor — "LINEAR" or "NON_LINEAR".
            - ""model_based"" (:class:`~app.core.enums.ModelSpecType` | None):
              Structural sub-family — "TREE", "SVM", etc.
            - ""selected_features"" (list[str] | None): Feature names
              chosen by this chain, copied from
              :attr:`~app.core.experiments.experiment_result.ExperimentResult.selected_features`.
        """
        metadata_to_model_selection_stage: list[dict[str, Any]] = []

        for result in sorted_results[:k]:
            metadata_to_model_selection_stage.append(
                {
                    "selectors": result.metadata.get("selectors", []),
                    "model": result.metadata.get("model", None),
                    "model_type": result.metadata.get("model_type", None),
                    "model_based": result.metadata.get("model_based", None),
                    "selected_features": result.selected_features,
                }
            )

        logger.debug(f"Top {k} selectors: {metadata_to_model_selection_stage}")

        return metadata_to_model_selection_stage
