from abc import ABC, abstractmethod
from typing import Optional

from app.core.context import Context, StageResult
from app.core.enums import ProblemType
from app.core.model_bank.model_spec import ModelSpec


class BaseModelRetriever(ABC):
    """
    Base class for retrieving function models.

    This class provides a structured way
    to manage and retrieve different types of models,
    such as baseline models, selector models, and ensembler models.
    It defines a common interface for retrieving models
    and allows for easy extension by subclassing and implementing
    """

    def __init__(self, problem_type: ProblemType, context: Optional[Context] = None):
        self.problem_type = problem_type
        self.models = self.load_defaults()
        self.context = context

    def retrieve_models(self) -> list[ModelSpec]:
        """
        Retrieve the list of baseline models.

        This functions servers as a Public API
        to access the predefined baseline models stored in the class.

        Returns:
            list[ModelSpec]: A list of ModelSpec objects representing the baseline models.

        Example:
            RetrieveBaseLineModels().retrieve_models()

        """
        if self.models is None:
            raise ValueError("No models have been registered.")
        return self.models

    def register_model(self, model_spec: ModelSpec) -> None:
        """
        Register a new baseline model.

        This function allows users to add new baseline models to the list of available models.

        Args:
            model_spec (ModelSpec): A ModelSpec object representing the new baseline model.

        Example:
            RetrieveBaseLineModels().retrieve_models(
                ModelSpec(name="MyNewModel", factory=MyNewModel))


        """
        self.models.append(model_spec)

    @abstractmethod
    def load_defaults(self) -> list[ModelSpec]:
        pass
