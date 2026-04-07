from abc import ABC, abstractmethod

from sklearn.pipeline import Pipeline

from app.core.context import RunContext


class EvaluationStrategy(ABC):

    @abstractmethod
    def evaluate(self,
                 pipeline: Pipeline,
                 X,
                 y,
                 context:RunContext,
                 cv=5,
                 threshold=None):
        pass