from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from app.core.enums import ModelSpecType
from app.core.model_bank.base_model_retriever import BaseModelRetriever
from app.core.model_bank.model_spects import ModelSpec


class BaselineModelRetriever(BaseModelRetriever):
    """
    Store baseline models for ML tasks.
    This class provides a method to retrieve a predefined list of baseline models,
    which can be used as a starting point for model selection
    and comparison in various machine learning tasks.
    These baseline models included are commonly used algorithms that serve as a
    reference point for evaluating the performance of more complex models.

    The models are stored in a list of ModelSpec objects,
    each representing a BaseModel with a name and a factory function for lazy loading.
    """

    def load_defaults(self) -> list[ModelSpec]:
        return [
            ModelSpec(name="RandomForestClassifier",
                      factory=self._build_randomforest,
                      spec_type=ModelSpecType.NON_LINEAR,
                      type=ModelSpecType.TREE),
            ModelSpec(name="LogisticRegression",
                      factory=self._build_logisticregression,
                      spec_type=ModelSpecType.LINEAR,
                      type=ModelSpecType.LINEAR),
        ]

    @staticmethod
    def _build_logisticregression():
        return LogisticRegression(
            solver="liblinear",
            max_iter=3000,
            random_state=42,
        )

    @staticmethod
    def _build_randomforest():
        return ExtraTreesClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
