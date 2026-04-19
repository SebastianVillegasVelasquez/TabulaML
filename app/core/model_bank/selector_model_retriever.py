import numpy as np

from app.core.enums import (
    ModelSpecType,
    SelectorSpecInfo)
from app.core.enums import ProblemsType
from app.core.model_bank import BaseModelRetriever
from app.core.model_bank import SelectorSpec


class ShapSelector:

    def __init__(self, model_factory, k=20):
        self.selected_idx_ = None
        self.model = None
        self.model_factory = model_factory
        self.k = k

    def fit(self, X, y):
        import shap

        self.model = self.model_factory()
        self.model.fit(X, y)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)

        # clasificación vs regresión
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values.mean(axis=2)

        importance = np.abs(shap_values).mean(axis=0)

        self.selected_idx_ = np.argsort(importance)[-self.k:]

        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.selected_idx_]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class SelectorModelRetriever(BaseModelRetriever):

    def load_defaults(self) -> list[SelectorSpec]:
        function_score = self._load_score_func()

        return [
            self._build_selectkbest(function_score),
            self._build_extratrees(),
            self._build_lasso(),
            self._build_elasticnet(),
            self._build_shap(),
            self._build_rfe()
        ]

    def _load_score_func(self):
        from sklearn.feature_selection import f_regression, f_classif

        return f_regression if self.problem_type == ProblemsType.REGRESSION else f_classif

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
            type=SelectorSpecInfo.STATISTICAL
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
            factory=lambda: SelectFromModel(ExtraTreesClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            ),
                threshold="median"),
            spec_type=ModelSpecType.NON_LINEAR,
            type=SelectorSpecInfo.TREE_BASED
        )

    """
    L1 based models for feature selection.
    """

    @staticmethod
    def _build_lasso():
        from sklearn.linear_model import Lasso
        from sklearn.feature_selection import SelectFromModel
        return SelectorSpec(
            name="lasso",
            factory=lambda: SelectFromModel(Lasso(alpha=0.01, max_iter=3000, random_state=42),
                                            threshold="median"),
            spec_type=ModelSpecType.LINEAR,
            type=SelectorSpecInfo.L1
        )

    """
    L1 + L2 based models for feature selection.
    """

    @staticmethod
    def _build_elasticnet():
        from sklearn.linear_model import ElasticNet
        from sklearn.feature_selection import SelectFromModel
        return SelectorSpec(
            name="elasticnet",
            factory=lambda: SelectFromModel(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=3000, random_state=42),
                                            threshold="median"),
            spec_type=ModelSpecType.LINEAR,
            type=SelectorSpecInfo.L1_L2,
        )

    """
    SHAP selector
    """

    @staticmethod
    def _build_shap():
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
                k=20
            ),
            spec_type=ModelSpecType.NON_LINEAR,
            type=SelectorSpecInfo.SHAP
        )

    """
    RFE selector
    """

    @staticmethod
    def _build_rfe():
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression

        return SelectorSpec(
            name="rfe",
            factory=lambda: RFE(
                estimator=LogisticRegression(
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42
                ),
                n_features_to_select=20,
                step=0.2
            ),
            spec_type=ModelSpecType.LINEAR,
            type=SelectorSpecInfo.WRAPPER
        )
