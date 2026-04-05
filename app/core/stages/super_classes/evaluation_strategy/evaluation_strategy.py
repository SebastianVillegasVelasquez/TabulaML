from abc import ABC, abstractmethod

class EvaluationStrategy(ABC):

    @abstractmethod
    def evaluate(self, pipeline, X, y, context, cv=5, threshold=None):
        pass