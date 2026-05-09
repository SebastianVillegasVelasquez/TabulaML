from __future__ import annotations

from app.core.enums import ModelSpecType, ProblemType
from app.core.model_bank import ModelSpec, BaseModelRetriever

_BOOST_MIN_SAMPLES: int = 1_000
"""Minimum number of training rows required to include XGBoost / LightGBM."""

_BOOST_MIN_FEATURES: int = 10
"""Minimum number of selected features required to include XGBoost / LightGBM."""


class PredictorModelRetriever(BaseModelRetriever):
    """Produces :class:`ModelSpec` lists for the model-selection stage.

    The factory reads the best experiment result from the previous
    feature-selection stage and uses its metadata (``model_type``,
    ``model_based``) together with current dataset statistics to decide
    which models to include.

    Decision logic in :meth:`load_defaults`:

    - ``model_type == LINEAR``  → logistic/linear regression family only.
    - ``model_type == NON_LINEAR`` → random forest + extra-trees family
      as the base pool; SVM added when ``model_based != TREE``.
    - ``model_based == TREE`` AND dataset satisfies
      ``n_samples > 1000`` AND ``n_features > 10`` → XGBoost and
      LightGBM are appended to the non-linear pool.

    Attributes:
        problem_type: Task model_based (``CLASSIFICATION`` or ``REGRESSION``).
            Controls which estimator variants are instantiated.
        context: Shared runtime context used to access dataset statistics.
        stage_result: Result object from the feature-selection stage.
            Its ``best_experiment.metadata`` supplies ``model_type`` and
            ``model_based``.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_defaults(self) -> list[ModelSpec]:
        """Returns the default set of candidate models for this experiment.

        Reads ``model_type`` and ``model_based`` from the best experiment
        metadata of the previous stage, then combines those signals with
        dataset statistics to build the model list.

        Linear problems yield a lightweight regression/logistic family.
        Non-linear problems yield tree ensembles; boosting libraries are
        added only when the dataset is large enough to justify the
        memory and compute cost.

        Returns:
            list[ModelSpec]: Ordered list of model specifications ready
            to be wrapped in :class:`~app.core.experiments.experiment.Experiment`
            instances.  Never empty — falls back to the full non-linear
            pool when metadata is missing or ambiguous.
        """
        metadata = self._read_metadata()
        model_type: ModelSpecType = metadata.get("model_type", ModelSpecType.NON_LINEAR)
        model_based: ModelSpecType = metadata.get("model_based", ModelSpecType.TREE)

        if model_type == ModelSpecType.LINEAR:
            return self._build_linear_pool()

        # Non-linear path
        specs = self._build_non_linear_pool(model_based)

        if model_based == ModelSpecType.TREE and self._boost_eligible():
            specs.extend(self._build_boosting_pool())

        return specs

    # ------------------------------------------------------------------
    # Pool builders
    # ------------------------------------------------------------------

    def _build_linear_pool(self) -> list[ModelSpec]:
        """Returns all linear model specs appropriate for the problem model_based.

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

        Always includes random forest and extra-trees variants.  An SVM
        is appended when ``model_based`` is not ``TREE``, because SVMs
        add meaningful diversity to non-tree ensembles but overlap
        heavily with trees on structured tabular data.

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
            solver, suitable for small to medium datasets.
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
        """Builds a Random Forest spec appropriate for the problem model_based.

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
        """Builds an Extra Trees spec appropriate for the problem model_based.

        Extra Trees adds more randomness than Random Forest, which often
        reduces variance at the cost of a slight bias increase.

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
        """Builds an SVM spec appropriate for the problem model_based.

        SVMs are included only when the previous-stage model was not
        tree-based, to maximise hypothesis diversity.

        Returns:
            ModelSpec: SVC or SVR with an RBF kernel and probability
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

        XGBoost is only attempted when the dataset passes the size
        thresholds checked in :meth:`_boost_eligible`.  The library is
        imported lazily so that environments without XGBoost are not
        affected.

        Returns:
            ModelSpec | None: XGBoost spec, or ``None`` when the
            ``xgboost`` package is not available.
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
                    use_label_encoder=False,
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

        LightGBM is only attempted when the dataset passes the size
        thresholds checked in :meth:`_boost_eligible`.  The library is
        imported lazily so that environments without LightGBM are not
        affected.

        Returns:
            ModelSpec | None: LightGBM spec, or ``None`` when the
            ``lightgbm`` package is not available.
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

    def _read_metadata(self) -> dict:
        """Extracts the metadata dict from the best feature-selection experiment.

        Safely navigates the stage result chain.  Returns an empty dict
        rather than raising if any part of the chain is ``None``, so
        :meth:`load_defaults` can fall back gracefully.

        Returns:
            dict: The ``metadata`` dict from ``stage_result.best_experiment``,
            or an empty dict when the chain is incomplete.
        """
        try:
            assert self.context is not None

            return self.context.stage_results.best_experiment.metadata or {}
        except AttributeError:
            return {}

    def _boost_eligible(self) -> bool:
        """Returns ``True`` when the dataset is large enough for boosting.

        Both conditions must hold:

        - ``n_samples > _BOOST_MIN_SAMPLES`` (default 1 000)
        - ``n_features > _BOOST_MIN_FEATURES`` (default 10)

        The feature count is read from the selected features of the best
        experiment when available, and falls back to the full feature
        count stored in the context.

        Returns:
            bool: Whether XGBoost and LightGBM should be included.
        """
        n_samples: int = getattr(self.context, "n_samples", 0)

        # Prefer the post-selection feature count over the raw one.
        best = getattr(self.stage_result, "best_experiment", None)
        selected: list[str] | None = getattr(best, "selected_features", None)
        if selected is not None:
            n_features = len(selected)
        else:
            n_features = getattr(self.context, "n_features", 0)

        return n_samples > _BOOST_MIN_SAMPLES and n_features > _BOOST_MIN_FEATURES
