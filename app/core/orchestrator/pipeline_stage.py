from abc import ABC, abstractmethod
from typing import Optional

from app.core.context.context import Context
from app.core.enums import Stages
from app.core.orchestrator.stage_validator import StageValidator


class PipelineStage(ABC):
    """
    Base interface for stages in the ML pipeline architecture.

    Stages are discrete units of work that transform pipeline context.
    Each stage must provide:
    - Stage type identification
    - Validator for preconditions
    - Execute method for business logic
    """

    def __init__(self, context: Context):
        self.context = context

    @abstractmethod
    def get_stage_type(self) -> Stages:
        """
        Return the stage type identifier.

        Returns:
            Stages enum value representing this stage
        """
        pass

    @abstractmethod
    def execute(self, context: Context) -> None:
        """
        Execute the stage business logic.

        Called by orchestrator after preconditions are validated.

        Args:
            context: Current pipeline context

        Raises:
            Any exception raised will be caught and handled by orchestrator
        """
        pass

    @abstractmethod
    def get_validator(self) -> Optional[StageValidator]:
        """
        Return the precondition validator for this stage.

        Returns:
            StageValidator instance if stage has dependencies, None otherwise
        """
        pass
