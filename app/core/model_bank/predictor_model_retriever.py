from __future__ import annotations

from typing import Any

from app.core.enums import ModelSpecType, ProblemType
from app.core.model_bank import ModelSpec, BaseModelRetriever
from app.utils.logger import logger
from app.core.context import StageResult

_BOOST_MIN_SAMPLES: int = 1_000
"""Minimum number of training rows required to include XGBoost / LightGBM."""

_BOOST_MIN_FEATURES: int = 10
"""Minimum number of selected features required to include XGBoost / LightGBM."""


class PredictorModelRetriever(BaseModelRetriever):
    """Produces :class:`ModelSpec` lists for the model-selection stage.

    Reads the feature-selection outcome stored by the previous stage in
    ``context.stage_results[FEATURE_SELECTION].metadata``, which is a
    list of dicts produced by :func:`_extract_top_k_chain_selectors`.
    The first entry in that list corresponds to the best-performing
    feature-selection chain and drives the model-pool decision.

    Decision logic in :meth:`load_defaults`:

    - ``model_type == LINEAR`` → logistic/linear regression family only.
    - ``model_type == NON_LINEAR`` → random-forest + extra-trees as the
      base pool; SVM added when ``model_based != TREE``.
    - ``model_based == TREE`` AND ``n_samples > 1 000`` AND
      ``n_features > 10`` → XGBoost and LightGBM appended.

    Attributes:
        problem_type: Task type (``CLASSIFICATION`` or ``REGRESSION``).
            Controls which estimator variants are instantiated.
        context: Shared runtime context that exposes dataset statistics
            and the results of previous stages.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_defaults(self) -> list[ModelSpec]:
        """Returns the default candidate models for the model-selection stage.

        Resolves ``model_type`` and ``model_based`` from the
        feature-selection stage metadata stored in the context, then
        combines those signals with dataset statistics to build the pool.

        Linear problems yield a lightweight regression/logistic family.
        Non-linear problems yield tree ensembles; boosting libraries are
        appended only when the dataset is large enough to justify the
        memory and compute cost.

        Returns:
            list[ModelSpec]: Ordered list of model specifications ready
            to be wrapped in
            :class:`~app.core.experiments.experiment.Experiment` instances.
            Falls back to the full non-linear pool when metadata is
            missing or ambiguous.
        """
        best_chain_meta = self._read_best_chain_metadata()
        model_type: ModelSpecType = best_chain_meta.get(
            "model_type", ModelSpecType.NON_LINEAR
        )
        model_based: ModelSpecType = best_chain_meta.get(
            "model_based", ModelSpecType.TREE
        )

        if model_type == ModelSpecType.LINEAR:
            return self._build_linear_pool()

        specs = self._build_non_linear_pool(model_based)

        if model_based == ModelSpecType.TREE and self._boost_eligible(best_chain_meta):
            specs.extend(self._build_boosting_pool())

        return specs

    # ------------------------------------------------------------------
    # Pool builders
    # ------------------------------------------------------------------

    def _build_linear_pool(self) -> list[ModelSpec]:
        """Returns all linear model specs appropriate for the problem type.

        For regression: Ridge, Lasso, ElasticNet.
        For classification: LogisticRegression (L2 and elasticnet penalties).

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

        Always includes random forest and extra-trees.  SVM is appended
        when ``model_based`` is not ``TREE``, because SVMs add meaningful
        diversity to non-tree ensembles but overlap heavily with trees on
        structured tabular data.

        Args:
            model_based: Structural sub-family from the feature-selection
                metadata.  Only :attr:`~app.core.enums.ModelSpecType.TREE`
                suppresses the SVM addition.

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

        Each library is imported lazily and silently skipped if not
        installed, so the rest of the pipeline is never blocked by an
        optional dependency.

        Returns:
            list[ModelSpec]: Boosting model specifications for whichever
            libraries are importable.  May be empty if neither is
            installed.
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
            ModelSpec: Lasso regression with default alpha.
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
            ModelSpec: LogisticRegression with L2 penalty and liblinear
            solver, suitable for small-to-medium datasets.
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
            ModelSpec: RandomForestClassifier or RandomForestRegressor
            with conservative defaults safe for cross-validation.
        """
        if self.problem_type == ProblemType.REGRESSION:
            from sklearn.ensemble import RandomForestRegressor

            return ModelSpec(
                name="random_forest",
                factory=lambda: RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    n_jobs=-1,
                    random_state=42,
                ),
                spec_type=ModelSpecType.NON_LINEAR,
                model_based=ModelSpecType.TREE,
            )

        from sklearn.ensemble import RandomForestClassifier

        return ModelSpec(
            name="random_forest",
            factory=lambda: RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                n_jobs=-1,
                random_state=42,
            ),
            spec_type=ModelSpecType.NON_LINEAR,
            model_based=ModelSpecType.TREE,
        )

    def _build_extra_trees(self) -> ModelSpec:
        """Builds an Extra Trees spec appropriate for the problem type.

        Extra Trees introduces more randomness than Random Forest, which
        often reduces variance at the cost of a slight bias increase.

        Returns:
            ModelSpec: ExtraTreesClassifier or ExtraTreesRegressor.
        """
        if self.problem_type == ProblemType.REGRESSION:
            from sklearn.ensemble import ExtraTreesRegressor

            return ModelSpec(
                name="extra_trees",
                factory=lambda: ExtraTreesRegressor(
                    n_estimators=100,
                    max_depth=10,
                    n_jobs=-1,
                    random_state=42,
                ),
                spec_type=ModelSpecType.NON_LINEAR,
                model_based=ModelSpecType.TREE,
            )

        from sklearn.ensemble import ExtraTreesClassifier

        return ModelSpec(
            name="extra_trees",
            factory=lambda: ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                n_jobs=-1,
                random_state=42,
            ),
            spec_type=ModelSpecType.NON_LINEAR,
            model_based=ModelSpecType.TREE,
        )

    def _build_svm(self) -> ModelSpec:
        """Builds an SVM spec appropriate for the problem type.

        SVMs are included only when the previous-stage predictor was not
        tree-based, to maximise hypothesis diversity in the pool.

        Returns:
            ModelSpec: SVC or SVR with an RBF kernel; probability
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
            factory=lambda: SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                random_state=42,
            ),
            spec_type=ModelSpecType.NON_LINEAR,
            model_based=ModelSpecType.SVM,
        )

    # ------------------------------------------------------------------
    # Boosting model builders (optional dependencies)
    # ------------------------------------------------------------------

    def _build_xgboost(self) -> ModelSpec | None:
        """Builds an XGBoost spec, or returns ``None`` if not installed.

        The library is imported lazily so that environments without
        XGBoost are not affected.

        Returns:
            ModelSpec | None: XGBoost spec, or ``None`` when the
            ``xgboost`` package is not importable.
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

        The library is imported lazily so that environments without
        LightGBM are not affected.

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

    def _read_best_chain_metadata(self) -> dict[str, Any]:
        """Reads the metadata dict for the best feature-selection chain.

        The feature-selection stage stores a *list* of top-k chain dicts
        in ``StageResult.metadata``; the first entry is the best one.
        This method navigates that chain safely and returns the first
        entry, or an empty dict when the chain is incomplete.

        The keys available in the returned dict (written by
        :func:`_extract_top_k_chain_selectors`) are:

        - "selectors" — list of selector names.
        - "model" — predictor name string.
        - "model_type" — :class:`~app.core.enums.ModelSpecType`.
        - "model_based" — :class:`~app.core.enums.ModelSpecType`.
        - "selected_features" — list of feature name strings.

        Returns:
            dict[str, Any]: Metadata for the best chain, or ``{}`` on
            failure.
        """
        try:
            assert self.context is not None
            from app.core.enums import Stages

            stage_result: StageResult = self.context.stage_results[
                Stages.FEATURE_SELECTION
            ]
            top_k: list[dict[str, Any]] = stage_result.metadata or []

            if not top_k:
                logger.warning(
                    "Feature-selection stage metadata is empty; "
                    "falling back to default model pool."
                )
                return {}

            return top_k[0]

        except (AttributeError, KeyError, TypeError):
            logger.warning(
                "Could not read feature-selection metadata from context; "
                "falling back to default model pool."
            )
            return {}

    def _boost_eligible(self, best_chain_meta: dict[str, Any]) -> bool:
        """Returns ``True`` when the dataset is large enough for boosting.

        Both conditions must hold simultaneously:

        - ``n_samples > _BOOST_MIN_SAMPLES`` (default 1 000)
        - ``n_features > _BOOST_MIN_FEATURES`` (default 10)

        The feature count is taken from ``selected_features`` in the
        best chain metadata (the post-selection count), falling back to
        the raw feature count stored in the context.  The post-selection
        count is preferred because it is the number of features that the
        boosting model will actually process.

        Args:
            best_chain_meta: The dict returned by
                :meth:`_read_best_chain_metadata`.

        Returns:
            bool: Whether XGBoost and LightGBM should be included.
        """
        n_samples: int = getattr(self.context, "n_samples", 0)

        selected: list[str] | None = best_chain_meta.get("selected_features")
        if selected is not None:
            n_features = len(selected)
        else:
            n_features = getattr(self.context, "n_features", 0)

        eligible = n_samples > _BOOST_MIN_SAMPLES and n_features > _BOOST_MIN_FEATURES

        if not eligible:
            logger.debug(
                f"Boosting skipped: n_samples={n_samples} "
                f"(min={_BOOST_MIN_SAMPLES}), "
                f"n_features={n_features} (min={_BOOST_MIN_FEATURES})."
            )

        return eligible
