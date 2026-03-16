from abc import ABC, abstractmethod
from typing import Optional, Tuple

from app.core.context.run_context import RunContext


class StageValidator(ABC):
    """
    Base interface for stage precondition validators.
    
    Validators verify that a stage has all required dependencies available
    before execution. Each stage validator implements domain-specific logic
    to check preconditions.
    """
    
    @abstractmethod
    def validate(self, context: RunContext) -> Tuple[bool, Optional[str]]:
        """
        Validate stage preconditions.
        
        Args:
            context: Current pipeline context
        
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if preconditions are met
            - error_message: Description if validation fails, None otherwise
        """
        pass
