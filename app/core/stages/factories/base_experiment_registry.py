from abc import ABC, abstractmethod
from typing import List

from app.core.context.run_context import RunContext
from app.core.domain.experiments.experiment_definition import ExperimentDefinition


class BaseExperimentFactory(ABC):
    """
    Abstract base class for all stage experiment factories.
    """

    @abstractmethod
    def create_experiments(self, context: RunContext = None) -> List[ExperimentDefinition]:
        """
        Creates and returns all experiment definitions for this stage.

        Args:
            context: El contexto de ejecución (puede ser None para stages simples)

        Returns:
            Lista de ExperimentDefinition registrados para esta stage
        """
        pass
