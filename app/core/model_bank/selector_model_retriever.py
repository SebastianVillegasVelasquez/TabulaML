from app.core.model_bank import BaseModelRetriever
from app.core.model_bank import SelectorSpec

class SelectorModelRetriever(BaseModelRetriever):


    def load_defaults(self):
        return [
            self._build_selectkbest(),
            self._build_extratrees(),
        ]

    """
    Statistical models for feature selection.
    """

    @staticmethod
    def _build_selectkbest() -> SelectorSpec:
        from sklearn.feature_selection import SelectKBest
        return SelectorSpec(
            name="selectkbest",
            factory=lambda score_func: SelectKBest(score_func=score_func, k=10),
            type="statistical"
        )

    @staticmethod
    def _build_extratrees() -> SelectorSpec:
        from sklearn.ensemble import ExtraTreesClassifier
        return SelectorSpec(
            name="extratrees",
            factory=lambda: ExtraTreesClassifier(n_estimators=100),
            type='tree_based'
        )



