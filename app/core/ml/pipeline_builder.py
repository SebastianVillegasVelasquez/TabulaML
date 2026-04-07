from typing import List, Tuple, Optional

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class PipelineBuilder:
    """Utility class for constructing and managing scikit-learn pipelines.

    This class provides a structured way to define, validate, and build
    machine learning pipelines. It ensures that all pipeline steps conform
    to scikit-learn's expected format and allows incremental construction
    of pipelines.

    The pipeline_builder pattern enables deferred pipeline creation, which is useful
    in experiment tracking systems where pipelines must be instantiated
    multiple times or lazily.

    Attributes:
        steps (List[Tuple[str, BaseEstimator]]): Ordered list of pipeline steps,
            where each step is defined as a tuple of (name, estimator).
    """

    def __init__(
            self,
            steps: List[Tuple[str, BaseEstimator]] = None,
    ):
        """Initializes the PipelineBuilder with validated steps.

        Args:
            steps (List[Tuple[str, BaseEstimator]]): Initial pipeline steps.

        Raises:
            ValueError: If no steps are provided.
            TypeError: If any step name is not a string or estimator is not a BaseEstimator.
        """
        self.steps = self._validate_steps(steps) if steps else []

    def build(self) -> Pipeline:
        """Builds and returns a scikit-learn Pipeline instance.

        Returns:
            Pipeline: A scikit-learn Pipeline constructed from the defined steps.
        """
        return Pipeline(self.steps)

    def preprend_step(self,step: Tuple[str, BaseEstimator]) -> None:
        """Adds a new step to the pipeline at the beginning."""
        self.steps.insert(0, step)

    def add_step(self,
                 step: Tuple[str, BaseEstimator],
                 at_index: Optional[int] = None) -> None:
        """Adds a new step to the pipeline.

        Args:
            step (Tuple[str, BaseEstimator]): A tuple containing:
                - Step name (str)
                - Estimator (BaseEstimator)
            at_index (int, optional): Index at which to insert the step. Defaults to None.

        Raises:
            TypeError: If the step is not valid.
        """
        name, estimator = step
        if not isinstance(name, str):
            raise TypeError(f"Step name must be a string. Got {type(name)}")
        if not isinstance(estimator, BaseEstimator):
            raise TypeError(f"Step estimator must be a sklearn BaseEstimator. Got {type(estimator)}")

        if at_index is not None:
            self.steps.insert(at_index, step)
            return

        self.steps.append(step)

    @staticmethod
    def _validate_steps(steps: List[Tuple[str, BaseEstimator]]) -> List[Tuple[str, BaseEstimator]]:
        """Validates pipeline steps.

        Ensures that:
        - At least one step exists
        - Each step name is a string
        - Each estimator is a scikit-learn BaseEstimator

        Args:
            steps (List[Tuple[str, BaseEstimator]]): Steps to validate.

        Returns:
            List[Tuple[str, BaseEstimator]]: Validated steps.

        Raises:
            ValueError: If steps list is empty.
            TypeError: If any step is invalid.
        """
        if not steps:
            raise ValueError("Pipeline must have at least one step.")

        for name, estimator in steps:
            if not isinstance(name, str):
                raise TypeError(f"Step name must be a string. Got {type(name)}")
            if not isinstance(estimator, BaseEstimator):
                raise TypeError(
                    f"Step estimator must be a sklearn BaseEstimator. Got {type(estimator)}"
                )

        return steps
