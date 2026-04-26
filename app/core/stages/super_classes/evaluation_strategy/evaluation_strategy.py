from abc import ABC, abstractmethod

from sklearn.pipeline import Pipeline

from app.core.context import Context


class EvaluationStrategy(ABC):

    @abstractmethod
    def evaluate(self, pipeline: Pipeline, X, y, context: Context, cv=5, threshold=None):
        pass
