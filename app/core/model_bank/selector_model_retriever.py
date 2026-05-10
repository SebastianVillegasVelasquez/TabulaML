import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from app.core.enums import ModelSpecType, SelectorSpecType
from app.core.enums import ProblemType
from app.core.model_bank import BaseModelRetriever
from app.core.model_bank import SelectorSpec


class ShapSelector:
    """SHAP-based feature selector for tree ensemble models.

    Uses SHAP (SHapley Additive exPlanations) values to rank feature
    importance and retain the top-k most influential features.

    This selector is memory-intensive. Place it only after a faster
    upstream selector has already reduced the feature space, so that
    SHAP operates on a manageable number of columns.

    Attributes:
        model_factory: A zero-argument callable that returns a fitted-able
            tree model (e.g., "lambda: RandomForestClassifier()").
        k: Number of top features to retain.
        model: The tree model instance created by "model_factory" and
            trained during :meth:`fit`. "None" before fitting.
        selected_idx_: 1-D integer array of column indices (into the
            transformed feature matrix) that were selected. "None"
            before fitting.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> selector = ShapSelector(
        ...     model_factory=lambda: RandomForestClassifier(n_estimators=100),
        ...     k=10,
        ... )
        >>> selector.fit(X_train, y_train)
        >>> X_reduced = selector.transform(X_train)
    """

    def __init__(self, model_factory, k: int = 20) -> None:
        """Initialises the selector without training any model.

        Args:
            model_factory: A zero-argument callable that returns a
                fresh, unfitted tree-based estimator each time it is
                called.  Using a factory (rather than a pre-built model)
                lets the selector create a clean instance on every
                :meth:`fit` call, which is important inside cross-
                validation loops.
            k: Number of features to keep.  Features are ranked by mean
                absolute SHAP value across the training set; the *k*
                highest-ranked ones are retained.  Defaults to 20.
        """
        self.selected_idx_ = None
        self.model = None
        self.model_factory = model_factory
        self.k = k

    def fit(self, X, y) -> "ShapSelector":
        """Fits the internal model and computes SHAP-based feature importance.

        Trains a tree model via "model_factory", computes SHAP values
        with :class:`shap.TreeExplainer`, and stores the indices of the
        *k* most important features in :attr:`selected_idx_`.

        Multi-class SHAP outputs (shape "[n_samples, n_features,
        n_classes]") are reduced by averaging across classes before
        ranking.

        Args:
            X: Training feature matrix of shape "(n_samples, n_features)".
                Accepts any array-like accepted by the underlying model.
            y: Target labels of shape "(n_samples,)".

        Returns:
            self: The fitted selector instance, enabling method chaining.
        """
        import shap

        self.model = self.model_factory()
        self.model.fit(X, y)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)

        # shap_values is a list for multi-output models; use only the
        # first output for binary classification compatibility.
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.array(shap_values)

        # Shape (n_samples, n_features, n_classes) -> (n_samples, n_features)
        if shap_values.ndim == 3:
            shap_values = shap_values.mean(axis=2)

        importance = np.abs(shap_values).mean(axis=0)

        # argsort is ascending; take the last k indices for the top-k.
        self.selected_idx_ = np.argsort(importance)[-self.k :]

        return self

    def transform(self, X):
        """Reduces the feature matrix to the columns selected during fitting.

        Args:
            X: Feature matrix of shape "(n_samples, n_features)".
                Must have at least as many columns as the matrix used in
                :meth:`fit`.

        Returns:
            numpy.ndarray: Reduced matrix of shape
            "(n_samples, k)", containing only the selected columns.

        Raises:
            ValueError: If :meth:`fit` has not been called yet and
                :attr:`selected_idx_` is "None".
        """
        X = np.asarray(X)
        return X[:, self.selected_idx_]

    def fit_transform(self, X, y):
        """Fits the selector and returns the reduced feature matrix in one step.

        Equivalent to calling :meth:`fit` followed by :meth:`transform`,
        but slightly more convenient inside pipeline definitions.

        Args:
            X: Training feature matrix of shape "(n_samples, n_features)".
            y: Target labels of shape "(n_samples,)".

        Returns:
            numpy.ndarray: Reduced matrix of shape "(n_samples, k)".
        """
        return self.fit(X, y).transform(X)


class SelectorModelRetriever(BaseModelRetriever):
    def load_defaults(self) -> list[SelectorSpec]:
        function_score = self._load_score_func()

        return [
            self._build_selectkbest(function_score),
            self._build_elasticnet(),
            self._build_shap(),
            self._build_linear_rfe_cv(),
            self._build_non_linear_rfe_cv(),
        ]

    def _load_score_func(self):
        from sklearn.feature_selection import f_regression, f_classif

        return (
            f_regression if self.problem_type == ProblemType.REGRESSION else f_classif
        )

    """
    Statistical models for feature selection.
    """

    @staticmethod
    def _build_selectkbest(func_score) -> SelectorSpec:
        from sklearn.feature_selection import SelectKBest

        return SelectorSpec(
            name="selectkbest",
            factory=lambda: SelectKBest(score_func=func_score, k=10),
            spec_type=ModelSpecType.LINEAR,
            type=SelectorSpecType.STATISTICAL,
        )

    """
    Tree-based models for feature selection.
    """

    @staticmethod
    def _build_extratrees() -> SelectorSpec:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel

        return SelectorSpec(
            name="extratrees",
            factory=lambda: SelectFromModel(
                ExtraTreesClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                ),
                threshold="median",
            ),
            spec_type=ModelSpecType.NON_LINEAR,
            type=SelectorSpecType.TREE_BASED,
        )

    """
    L1 based models for feature selection.
    """

    @staticmethod
    def _build_lasso() -> SelectorSpec:
        from sklearn.linear_model import Lasso
        from sklearn.feature_selection import SelectFromModel

        return SelectorSpec(
            name="lasso",
            factory=lambda: SelectFromModel(
                Lasso(alpha=0.01, max_iter=3000, random_state=42), threshold="median"
            ),
            spec_type=ModelSpecType.LINEAR,
            type=SelectorSpecType.L1,
        )

    """
    L1 + L2 based models for feature selection.
    """

    @staticmethod
    def _build_elasticnet() -> SelectorSpec:
        from sklearn.linear_model import ElasticNet
        from sklearn.feature_selection import SelectFromModel

        return SelectorSpec(
            name="elasticnet",
            factory=lambda: SelectFromModel(
                ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=3000, random_state=42),
                threshold="median",
            ),
            spec_type=ModelSpecType.LINEAR,
            type=SelectorSpecType.L1_L2,
        )

    """
    SHAP selector
    """

    @staticmethod
    def _build_shap() -> SelectorSpec:
        from sklearn.ensemble import ExtraTreesClassifier

        return SelectorSpec(
            name="shap",
            factory=lambda: ShapSelector(
                model_factory=lambda: ExtraTreesClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                ),
                k=20,
            ),
            spec_type=ModelSpecType.NON_LINEAR,
            type=SelectorSpecType.SHAP,
        )

    """
    RFE selectors
    """

    @staticmethod
    def _build_linear_rfe_cv() -> SelectorSpec:
        from sklearn.feature_selection import RFECV
        from sklearn.linear_model import LogisticRegression

        return SelectorSpec(
            name="rfe_linear",
            factory=lambda: RFECV(
                estimator=LogisticRegression(
                    solver="liblinear", max_iter=1000, random_state=42
                ),
                step=0.2,
            ),
            spec_type=ModelSpecType.LINEAR,
            type=SelectorSpecType.RFE,
        )

    @staticmethod
    def _build_non_linear_rfe_cv() -> SelectorSpec:
        from sklearn.feature_selection import RFECV

        return SelectorSpec(
            name="rfe_non_linear",
            factory=lambda: RFECV(
                estimator=ExtraTreesClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            spec_type=ModelSpecType.NON_LINEAR,
            type=SelectorSpecType.RFE,
        )
