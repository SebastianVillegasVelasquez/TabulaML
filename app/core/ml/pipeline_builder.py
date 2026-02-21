from sklearn.base import BaseEstimator
from typing import Tuple, List

from sklearn.pipeline import Pipeline


class PipelineBuilder:
    def __init__(
            self,
            steps: List[Tuple[str, BaseEstimator]]
    ):
        self.steps = self._validate_steps(steps)

    def build(self) -> Pipeline:
        return Pipeline(self.steps)

    def add_step(self, step: Tuple[str, BaseEstimator]):
        self.steps.append(step)

    @staticmethod
    def _validate_steps(steps: List[Tuple[str, BaseEstimator]]):
        if not steps:
            raise ValueError("Pipeline must have at least one step.")
        for name, estimator in steps:
            if not isinstance(name, str):
                raise TypeError(f"Step name must be a string. Got {type(name)}")
            if not isinstance(estimator, BaseEstimator):
                raise TypeError(f"Step estimator must be a sklearn BaseEstimator. Got {type(estimator)}")
        return steps