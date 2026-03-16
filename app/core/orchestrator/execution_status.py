from enum import Enum


class ExecutionStatus(Enum):
    """
    Represents the execution state of a pipeline stage.
    
    States:
    - PENDING: Stage is queued but not yet executing
    - RUNNING: Stage is currently executing
    - SUCCESS: Stage completed successfully
    - FAILED: Stage terminated with an error
    - SKIPPED: Stage was skipped due to failed preconditions
    """
    
    PENDING = "pending"      # Not yet started
    RUNNING = "running"      # In progress
    SUCCESS = "success"      # Completed successfully
    FAILED = "failed"        # Completed with error
    SKIPPED = "skipped"      # Preconditions aren't met
