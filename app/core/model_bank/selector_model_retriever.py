from app.core.enums import (
    ModelSpecType,
    SelectorSpecInfo)
from app.core.enums import ProblemsType
from app.core.model_bank import BaseModelRetriever
from app.core.model_bank import SelectorSpec


class SelectorModelRetriever(BaseModelRetriever):

    def load_defaults(self):
        function_score = self._load_score_func()

        return [
            self._build_selectkbest(function_score),
            self._build_extratrees(),
            self._build_lasso(),
            self._build_elasticnet(),
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
