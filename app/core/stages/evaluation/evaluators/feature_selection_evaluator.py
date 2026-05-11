from typing import Any

from app.core.stages.evaluation import BaseEvaluator


from app.core.context import StageResult
from app.core.experiments import ExperimentResult
from app.utils.logger import logger


class FeatureSelectionEvaluator(BaseEvaluator):
    """Evaluation logic for the feature-selection stage.

    Inherits the template-method workflow from :class:`BaseEvaluator` and
    implements two hook methods:

    - :meth:`_extract_stage_specific_data` — collects the top-k selector
      chains, excluding the best experiment (which is stored separately in
      :class:`StageResult`) to avoid redundancy.
    - :meth:`_update_context` — removes ``model_type`` and ``model_based``
      from the best experiment's own metadata before persisting, keeping a
      single authoritative source of those values in the stage-level metadata.

    The data stored in :attr:`StageResult.metadata` is a list of dicts (one
    per chain, best-first, starting from rank 2) that the model-selection
    stage reads via
    ``context.stage_results[Stages.FEATURE_SELECTION].metadata``.
    """

    def _extract_stage_specific_data(
        self,
        sorted_results: list[ExperimentResult],
        best_experiment: ExperimentResult,
    ) -> list[dict[str, Any]]:
        """Extracts top-k selector-chain metadata, excluding the best experiment.

        The best experiment is stored separately in
        :attr:`~app.core.context.StageResult.best_experiment`, so it is
        excluded here (``sorted_results[1:]``) to avoid duplicating the same
        information in :attr:`~app.core.context.StageResult.metadata`.

        Args:
            sorted_results: All experiment results sorted best-first by the
                primary metric.
            best_experiment: The top-performing
                :class:`~app.core.experiments.ExperimentResult`.  Passed by
                the base class but not consumed here; the first slot is
                intentionally skipped in ``sorted_results``.

        Returns:
            list[dict[str, Any]]: One dict per runner-up chain (ranks 2 … k+1),
            each with the keys documented in
            :meth:`_extract_top_k_chain_selectors`.
        """
        logger.debug(
            f"Extracting stage-specific data for {self.stage}. "
            f"Total sorted results: {len(sorted_results)}"
        )
        return self._extract_top_k_chain_selectors(sorted_results, k=3)

    def _update_context(
        self,
        sorted_results: list[ExperimentResult],
        best_experiment: ExperimentResult,
        stage_specific_data: list[dict[str, Any]],
    ) -> None:
        """Persists the feature-selection outcome into the shared context.

        Before storage, ``model_type`` and ``model_based`` are popped from
        ``best_experiment.metadata`` so that those keys exist in exactly one
        place: the stage-level metadata list produced by
        :meth:`_extract_top_k_chain_selectors`.  Downstream retrievers must
        read those keys from
        ``context.stage_results[Stages.FEATURE_SELECTION].metadata``, not
        from ``best_experiment.metadata``.

        Args:
            sorted_results: All experiment results, best-first.  Retained for
                signature compatibility; not consumed directly.
            best_experiment: The top-ranking
                :class:`~app.core.experiments.ExperimentResult`.  Mutated
                in-place to remove ``"model_type"`` and ``"model_based"``
                before storage.
            stage_specific_data: Runner-up chain metadata produced by
                :meth:`_extract_stage_specific_data`.
        """
        try:
            best_experiment.metadata.pop("model_type", None)
            best_experiment.metadata.pop("model_based", None)

            stage_result = StageResult(
                name=self.stage,
                results=None,
                best_experiment=best_experiment,
                metadata=stage_specific_data,
            )
        except Exception as exc:
            logger.error(f"Error updating context for {self.stage}: {exc}")
            return

        logger.debug(f"Stage Result: {stage_result}")
        self.context.update_stage_context(self.stage, stage_result)

    @staticmethod
    def _extract_top_k_chain_selectors(
        sorted_results: list[ExperimentResult],
        k: int = 3,
    ) -> list[dict[str, Any]]:
        """Extracts selector-chain metadata from the runner-up experiments.

        Skips ``sorted_results[0]`` (the best experiment) because that entry
        is already stored in :attr:`~app.core.context.StageResult.best_experiment`.
        Takes the next *k* results (ranks 2 … k+1) and packages them into
        flat dicts for the model-selection stage.

        Each dict contains the keys that
        :func:`~app.core.stages.feature_selection.builder._build_experiment`
        writes into ``ExperimentResult.metadata``:

        - ``"selectors"`` — ordered selector names (e.g.
          ``["selectkbest", "rfe_non_linear"]``).
        - ``"model"`` — predictor name string used during feature selection.
        - ``"model_type"`` — :class:`~app.core.enums.ModelSpecType` broad
          family (``LINEAR`` or ``NON_LINEAR``).
        - ``"model_based"`` — :class:`~app.core.enums.ModelSpecType`
          structural sub-family (``TREE``, ``SVM``, etc.).
        - ``"selected_features"`` — post-selection feature names taken from
          :attr:`~app.core.experiments.ExperimentResult.selected_features`
          (not from ``metadata``, where this key does not exist).

        Args:
            sorted_results: All experiment results ordered best-first.  The
                first element is skipped; at most *k* subsequent elements are
                consumed.
            k: Maximum number of runner-up chains to include.  Defaults to 3.

        Returns:
            list[dict[str, Any]]: At most *k* dicts, one per runner-up chain.
            May be shorter than *k* when fewer than *k*+1 results exist.
        """
        # Skip index 0 — that is the best experiment stored separately.
        runner_ups = sorted_results[1 : k + 1]

        metadata_list: list[dict[str, Any]] = []
        for result in runner_ups:
            metadata_list.append(
                {
                    "selectors": result.metadata.get("selectors", []),
                    "model": result.metadata.get("model", None),
                    "model_type": result.metadata.get("model_type", None),
                    "model_based": result.metadata.get("model_based", None),
                    "selected_features": result.selected_features,
                }
            )

        logger.debug(f"Top {k} runner-up chains: {metadata_list}")
        return metadata_list
