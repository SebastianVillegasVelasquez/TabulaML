from abc import ABC, abstractmethod
from typing import List

from app.core.context.context import Context
from experiments import ExperimentDefinition


class BaseExperimentFactory(ABC):
    """
    Abstract base class for all stage experiment factories.
    """

    @abstractmethod
    def create_experiments(self, context: Context = None) -> List[ExperimentDefinition]:
        """
        Creates and returns all experiment definitions for this stage.

        Args:
            context: Context containing pipeline configuration and state

        Returns:
            ExperimentDefinition instances for this stage.
        """
        pass
