from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.enums import ModelSpecType, ProblemType
from app.core.model_bank import ModelSpec, BaseModelRetriever
from app.core.context import StageResult, Metadata, Context
from app.utils.logger import logger
from app.core.enums import Stages

_BOOST_MIN_SAMPLES: int = 1_000
"""Minimum number of training rows required to include XGBoost / LightGBM."""

_BOOST_MIN_FEATURES: int = 10
"""Minimum number of selected features required to include XGBoost / LightGBM."""


@dataclass
class ChainModelPairing:
    """A selector chain paired with the candidate models suggested for it.

    Attributes:
        chain_metadata: The chain dict produced by
            :meth:`FeatureSelectionEvaluator._extract_top_k_chain_selectors`,
            containing ``"selectors"``, ``"model"``, ``"model_type"``,
            ``"model_based"``, and ``"selected_features"``.
        suggested_models: Ordered list of :class:`ModelSpec` instances
            recommended for this chain, derived from its ``model_type``
            and ``model_based`` values.
    """

    chain_metadata: dict[str, Any]
    suggested_models: list[ModelSpec] = field(default_factory=list)


class PredictorModelRetriever(BaseModelRetriever):
    """Produces candidate-model pairings for every feature-selection chain.

    Reads the feature-selection outcome from
    ``context.stage_results[Stages.FEATURE_SELECTION]``.  That
    :class:`~app.core.context.StageResult` contains:

    - ``best_experiment`` — the single top-performing result whose
      ``model_type`` / ``model_based`` drive the primary model pool.
    - ``metadata`` — a list of runner-up chain dicts (ranks 2 … k+1),
      each also carrying ``model_type`` / ``model_based``.

    :meth:`load_defaults` returns a flat :class:`list` of
    :class:`ModelSpec` objects built from the **best** chain, preserving the
    existing contract expected by the experiment builder.

    :meth:`load_all_chain_pairings` returns a :class:`list` of
    :class:`ChainModelPairing` objects — one per chain (best + runner-ups) —
    so that callers who want per-chain model suggestions can iterate over
    them directly.

    Decision logic (applied independently per chain):

    - ``model_type == LINEAR`` → logistic/linear regression family.
    - ``model_type == NON_LINEAR`` → random forest + extra-trees; SVM added
      when ``model_based != TREE``.
    - ``model_based == TREE`` **and** dataset passes size thresholds
      (``n_samples > 1 000``, ``n_features > 10``) → XGBoost and LightGBM
      appended.

    Attributes:
        problem_type: Task type (``CLASSIFICATION`` or ``REGRESSION``).
        context: Shared runtime context exposing stage results and dataset
            metadata.
    """

    def __init__(self, problem_type: ProblemType, context: Optional[Context] = None):
        """Initialize the PredictorModelRetriever.

        Args:
            problem_type: Task type (CLASSIFICATION or REGRESSION).
            context: Optional shared runtime context.
        """
        super().__init__(problem_type=problem_type, context=context)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_defaults(self) -> list[ModelSpec]:
        """Returns candidate models for the best feature-selection chain.

        Reads ``model_type`` and ``model_based`` from the best experiment
        stored in the feature-selection :class:`~app.core.context.StageResult`,
        then builds the appropriate model pool.

        This method preserves the flat :class:`list` contract so it is a
        drop-in replacement for the previous implementation.

        Returns:
            list[ModelSpec]: Ordered model specifications for the best chain.
            Falls back to the full non-linear pool when metadata is absent.
        """
        best_meta = self._read_best_experiment_metadata()
        return self._build_pool_for_chain(best_meta)

    def load_all_chain_pairings(self) -> list[ChainModelPairing]:
        """Returns model suggestions paired with every stored selector chain.

        Combines the best experiment (from
        :attr:`~app.core.context.StageResult.best_experiment`) with the
        runner-up chains (from
        :attr:`~app.core.context.StageResult.metadata`) and builds a
        :class:`ChainModelPairing` for each one.

        The first element in the returned list always corresponds to the best
        chain; subsequent elements correspond to runner-ups in descending
        performance order.

        Returns:
            list[ChainModelPairing]: One pairing per chain.  Empty list when
            the feature-selection stage has not yet run or its context entry
            is missing.
        """
        stage_result = self._read_stage_result()
        if stage_result is None:
            return []

        pairings: list[ChainModelPairing] = []

        # --- best experiment ---
        best_meta = self._metadata_from_best_experiment(stage_result)
        pairings.append(
            ChainModelPairing(
                chain_metadata=best_meta,
                suggested_models=self._build_pool_for_chain(best_meta),
            )
        )

        # --- runner-up chains ---
        runner_ups: list[dict[str, Any]] = stage_result.metadata or []
        for chain_meta in runner_ups:
            pairings.append(
                ChainModelPairing(
                    chain_metadata=chain_meta,
                    suggested_models=self._build_pool_for_chain(chain_meta),
                )
            )

        return pairings

    # ------------------------------------------------------------------
    # Pool builder (core decision logic)
    # ------------------------------------------------------------------

    def _build_pool_for_chain(self, chain_meta: dict[str, Any]) -> list[ModelSpec]:
        """Builds the model pool appropriate for a single selector chain.

        Args:
            chain_meta: A chain metadata dict containing at minimum the keys
                ``"model_type"`` and ``"model_based"``.  Missing keys trigger
                safe fallback values.

        Returns:
            list[ModelSpec]: Ordered model specifications for this chain.
        """
        model_type: ModelSpecType = chain_meta.get(
            "model_type", ModelSpecType.NON_LINEAR
        )
        model_based: ModelSpecType = chain_meta.get("model_based", ModelSpecType.TREE)

        if model_type == ModelSpecType.LINEAR:
            return self._build_linear_pool()

        specs = self._build_non_linear_pool(model_based)

        if model_based == ModelSpecType.TREE and self._boost_eligible(chain_meta):
            specs.extend(self._build_boosting_pool())

        return specs

    # ------------------------------------------------------------------
    # Pool builders
    # ------------------------------------------------------------------

    def _build_linear_pool(self) -> list[ModelSpec]:
        """Returns all linear model specs appropriate for the problem type.

        For regression: Ridge, Lasso, ElasticNet.
        For classification: LogisticRegression with L2 and elasticnet penalties.

        Returns:
            list[ModelSpec]: Linear model specifications.
        """
        if self.problem_type == ProblemType.REGRESSION:
            return [
                self._build_ridge(),
                self._build_lasso(),
                self._build_elasticnet_regressor(),
            ]
        return [
            self._build_logistic_l2(),
            self._build_logistic_elasticnet(),
        ]

    def _build_non_linear_pool(self, model_based: ModelSpecType) -> list[ModelSpec]:
        """Returns the base non-linear model specs.

        Random forest and extra-trees are always included.  SVM is appended
        when ``model_based`` is not ``TREE``, because SVMs add meaningful
        diversity to non-tree ensembles but overlap heavily with trees on
        structured tabular data.

        Args:
            model_based: Structural sub-family from the chain metadata.
                :attr:`~app.core.enums.ModelSpecType.TREE` suppresses SVM.

        Returns:
            list[ModelSpec]: Non-linear model specifications.
        """
        specs = [
            self._build_random_forest(),
            self._build_extra_trees(),
        ]
        if model_based != ModelSpecType.TREE:
            specs.append(self._build_svm())
        return specs

    def _build_boosting_pool(self) -> list[ModelSpec]:
        """Returns XGBoost and LightGBM specs when their libraries are available.

        Each library is imported lazily and silently omitted when not installed,
        so the rest of the pipeline is never blocked by an optional dependency.

        Returns:
            list[ModelSpec]: Boosting specs for whichever libraries are
            importable.  May be empty if neither is installed.
        """
        specs: list[ModelSpec] = []
        xgb = self._build_xgboost()
        if xgb is not None:
            specs.append(xgb)
        lgbm = self._build_lightgbm()
        if lgbm is not None:
            specs.append(lgbm)
        return specs

    # ------------------------------------------------------------------
    # Linear model builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ridge() -> ModelSpec:
        """Builds a Ridge regression spec.

        Returns:
            ModelSpec: Ridge regression with default alpha.
        """
        from sklearn.linear_model import Ridge

        return ModelSpec(
            name="ridge",
            factory=lambda: Ridge(alpha=1.0),
            spec_type=ModelSpecType.LINEAR,
            model_based=ModelSpecType.LINEAR,
        )

    @staticmethod
    def _build_lasso() -> ModelSpec:
        """Builds a Lasso regression spec.

        Returns:
            ModelSpec: Lasso regression with default alpha and extended
            iteration budget to aid convergence on sparse problems.
        """
        from sklearn.linear_model import Lasso

        return ModelSpec(
            name="lasso",
            factory=lambda: Lasso(alpha=1.0, max_iter=10_000),
            spec_type=ModelSpecType.LINEAR,
            model_based=ModelSpecType.LINEAR,
        )

    @staticmethod
    def _build_elasticnet_regressor() -> ModelSpec:
        """Builds an ElasticNet regression spec.

        Returns:
            ModelSpec: ElasticNet with equal L1/L2 mixing ratio.
        """
        from sklearn.linear_model import ElasticNet

        return ModelSpec(
            name="elasticnet_regressor",
            factory=lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10_000),
            spec_type=ModelSpecType.LINEAR,
            model_based=ModelSpecType.LINEAR,
        )

    @staticmethod
    def _build_logistic_l2() -> ModelSpec:
        """Builds a logistic regression spec with L2 regularisation.

        Returns:
            ModelSpec: LogisticRegression with L2 penalty and the
            ``liblinear`` solver, suitable for small-to-medium datasets.
        """
        from sklearn.linear_model import LogisticRegression

        return ModelSpec(
            name="logistic_l2",
            factory=lambda: LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="liblinear",
                max_iter=1_000,
                random_state=42,
            ),
            spec_type=ModelSpecType.LINEAR,
            model_based=ModelSpecType.LINEAR,
        )

    @staticmethod
    def _build_logistic_elasticnet() -> ModelSpec:
        """Builds a logistic regression spec with elasticnet regularisation.

        Uses the ``saga`` solver, which is required for the ``elasticnet``
        penalty and scales well to larger datasets.

        Returns:
            ModelSpec: LogisticRegression with elasticnet penalty.
        """
        from sklearn.linear_model import LogisticRegression

        return ModelSpec(
            name="logistic_elasticnet",
            factory=lambda: LogisticRegression(
                penalty="elasticnet",
                C=1.0,
                l1_ratio=0.5,
                solver="saga",
                max_iter=2_000,
                random_state=42,
            ),
            spec_type=ModelSpecType.LINEAR,
            model_based=ModelSpecType.LINEAR,
        )

    # ------------------------------------------------------------------
    # Non-linear model builders
    # ------------------------------------------------------------------

    def _build_random_forest(self) -> ModelSpec:
        """Builds a Random Forest spec appropriate for the problem type.

        Returns:
            ModelSpec: ``RandomForestClassifier`` or
            ``RandomForestRegressor`` with conservative defaults safe for
            cross-validation.
        """
        if self.problem_type == ProblemType.REGRESSION:
            from sklearn.ensemble import RandomForestRegressor

            return ModelSpec(
                name="random_forest",
                factory=lambda: RandomForestRegressor(
                    n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
                ),
                spec_type=ModelSpecType.NON_LINEAR,
                model_based=ModelSpecType.TREE,
            )

        from sklearn.ensemble import RandomForestClassifier

        return ModelSpec(
            name="random_forest",
            factory=lambda: RandomForestClassifier(
                n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
            ),
            spec_type=ModelSpecType.NON_LINEAR,
            model_based=ModelSpecType.TREE,
        )

    def _build_extra_trees(self) -> ModelSpec:
        """Builds an Extra Trees spec appropriate for the problem type.

        Extra Trees introduces additional randomness compared to Random
        Forest, which often reduces variance at the cost of a slight bias
        increase.

        Returns:
            ModelSpec: ``ExtraTreesClassifier`` or ``ExtraTreesRegressor``.
        """
        if self.problem_type == ProblemType.REGRESSION:
            from sklearn.ensemble import ExtraTreesRegressor

            return ModelSpec(
                name="extra_trees",
                factory=lambda: ExtraTreesRegressor(
                    n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
                ),
                spec_type=ModelSpecType.NON_LINEAR,
                model_based=ModelSpecType.TREE,
            )

        from sklearn.ensemble import ExtraTreesClassifier

        return ModelSpec(
            name="extra_trees",
            factory=lambda: ExtraTreesClassifier(
                n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
            ),
            spec_type=ModelSpecType.NON_LINEAR,
            model_based=ModelSpecType.TREE,
        )

    def _build_svm(self) -> ModelSpec:
        """Builds an SVM spec appropriate for the problem type.

        SVMs are included only when the previous-stage predictor was not
        tree-based, to maximise hypothesis diversity in the pool.

        Returns:
            ModelSpec: ``SVC`` or ``SVR`` with an RBF kernel; probability
            estimates enabled for classifiers.
        """
        if self.problem_type == ProblemType.REGRESSION:
            from sklearn.svm import SVR

            return ModelSpec(
                name="svm",
                factory=lambda: SVR(kernel="rbf", C=1.0, epsilon=0.1),
                spec_type=ModelSpecType.NON_LINEAR,
                model_based=ModelSpecType.SVM,
            )

        from sklearn.svm import SVC

        return ModelSpec(
            name="svm",
            factory=lambda: SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
            spec_type=ModelSpecType.NON_LINEAR,
            model_based=ModelSpecType.SVM,
        )

    # ------------------------------------------------------------------
    # Boosting model builders (optional dependencies)
    # ------------------------------------------------------------------

    def _build_xgboost(self) -> ModelSpec | None:
        """Builds an XGBoost spec, or returns ``None`` if not installed.

        The library is imported lazily so that environments without XGBoost
        are not affected.

        Returns:
            ModelSpec | None: XGBoost spec, or ``None`` when the ``xgboost``
            package is not importable.
        """
        try:
            if self.problem_type == ProblemType.REGRESSION:
                from xgboost import XGBRegressor

                return ModelSpec(
                    name="xgboost",
                    factory=lambda: XGBRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        n_jobs=-1,
                        random_state=42,
                        verbosity=0,
                    ),
                    spec_type=ModelSpecType.NON_LINEAR,
                    model_based=ModelSpecType.TREE,
                    metadata={"library": "xgboost"},
                )

            from xgboost import XGBClassifier

            return ModelSpec(
                name="xgboost",
                factory=lambda: XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=42,
                    verbosity=0,
                ),
                spec_type=ModelSpecType.NON_LINEAR,
                model_based=ModelSpecType.TREE,
                metadata={"library": "xgboost"},
            )

        except ImportError:
            return None

    def _build_lightgbm(self) -> ModelSpec | None:
        """Builds a LightGBM spec, or returns ``None`` if not installed.

        The library is imported lazily so that environments without LightGBM
        are not affected.

        Returns:
            ModelSpec | None: LightGBM spec, or ``None`` when the
            ``lightgbm`` package is not importable.
        """
        try:
            if self.problem_type == ProblemType.REGRESSION:
                from lightgbm import LGBMRegressor

                return ModelSpec(
                    name="lightgbm",
                    factory=lambda: LGBMRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        n_jobs=-1,
                        random_state=42,
                        verbose=-1,
                    ),
                    spec_type=ModelSpecType.NON_LINEAR,
                    model_based=ModelSpecType.TREE,
                    metadata={"library": "lightgbm"},
                )

            from lightgbm import LGBMClassifier

            return ModelSpec(
                name="lightgbm",
                factory=lambda: LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1,
                ),
                spec_type=ModelSpecType.NON_LINEAR,
                model_based=ModelSpecType.TREE,
                metadata={"library": "lightgbm"},
            )

        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_stage_result(self) -> StageResult | None:
        """Retrieves the feature-selection :class:`StageResult` from context.

        Returns:
            StageResult | None: The stored result, or ``None`` when it
            cannot be found.
        """
        try:
            assert self.context is not None
            return self.context.stage_results[Stages.FEATURE_SELECTION]
        except (AssertionError, KeyError, AttributeError, TypeError):
            logger.warning(
                "Feature-selection stage result not found in context; "
                "falling back to default model pool."
            )
            return None

    def _read_best_experiment_metadata(self) -> dict[str, Any]:
        """Reads the metadata dict reconstructed from the best experiment.

        The best experiment's own ``metadata`` dict no longer carries
        ``"model_type"`` or ``"model_based"`` (they are popped during
        :meth:`FeatureSelectionEvaluator._update_context`).  Instead, this
        method reconstructs an equivalent dict from the attributes of
        ``best_experiment`` itself — specifically ``selected_features`` —
        and fills ``"model_type"`` / ``"model_based"`` from
        ``best_experiment.metadata`` for any keys that were not removed (e.g.
        ``"selectors"``, ``"model"``).

        To retrieve ``model_type`` and ``model_based`` for the best chain,
        callers should use :meth:`load_all_chain_pairings` where the first
        pairing corresponds to the best experiment, or read them from
        ``stage_result.metadata[0]`` (the first runner-up), acknowledging
        that the best experiment's type info lives only in the evaluator's
        pop-before-store logic.

        Returns:
            dict[str, Any]: Best-experiment metadata enriched with
            ``"selected_features"``, or ``{}`` on failure.
        """
        stage_result = self._read_stage_result()
        if stage_result is None or stage_result.best_experiment is None:
            return {}

        best = stage_result.best_experiment
        meta: dict[str, Any] = dict(best.metadata or {})
        meta.setdefault("selected_features", best.selected_features)
        return meta

    @staticmethod
    def _metadata_from_best_experiment(stage_result: StageResult) -> dict[str, Any]:
        """Extracts a chain-compatible metadata dict from the best experiment.

        Builds the same dict shape that runner-up chains use, sourcing
        ``"selected_features"`` from
        :attr:`~app.core.experiments.ExperimentResult.selected_features`
        and the remaining keys from
        :attr:`~app.core.experiments.ExperimentResult.metadata`.

        Args:
            stage_result: The feature-selection
                :class:`~app.core.context.StageResult`.

        Returns:
            dict[str, Any]: Chain-compatible metadata dict, or ``{}`` when
            ``best_experiment`` is ``None``.
        """
        best = stage_result.best_experiment
        if best is None:
            return {}

        meta = dict(best.metadata or {})
        meta.setdefault("selected_features", best.selected_features)
        return meta

    def _boost_eligible(self, chain_meta: dict[str, Any]) -> bool:
        """Returns ``True`` when the dataset is large enough for boosting.

        Both conditions must hold simultaneously:

        - ``n_samples > _BOOST_MIN_SAMPLES`` (default 1 000)
        - ``n_features > _BOOST_MIN_FEATURES`` (default 10)

        The feature count is taken from ``"selected_features"`` in the chain
        metadata (the post-selection count), falling back to ``n_features``
        stored in :attr:`~app.core.context.Metadata`.  The post-selection
        count is preferred because it is the actual number of columns the
        boosting model will process.

        Args:
            chain_meta: A chain metadata dict, as returned by
                :meth:`_read_best_experiment_metadata` or from
                :attr:`~app.core.context.StageResult.metadata`.

        Returns:
            bool: ``True`` when both size thresholds are satisfied.
        """
        assert self.context is not None

        dataset_metadata: Metadata = self.context.metadata
        n_samples: int = getattr(dataset_metadata, "dataset_length", 0)

        selected: list[str] | None = chain_meta.get("selected_features")
        n_features: int = (
            len(selected)
            if selected is not None
            else getattr(dataset_metadata, "n_features", 0)
        )

        eligible = n_samples > _BOOST_MIN_SAMPLES and n_features > _BOOST_MIN_FEATURES

        if not eligible:
            logger.debug(
                f"Boosting skipped — n_samples={n_samples} "
                f"(threshold={_BOOST_MIN_SAMPLES}), "
                f"n_features={n_features} (threshold={_BOOST_MIN_FEATURES})."
            )

        return eligible
