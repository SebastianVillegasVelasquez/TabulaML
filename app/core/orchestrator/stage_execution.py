from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.core.context.stages import Stages
from app.core.orchestrator.execution_status import ExecutionStatus


@dataclass
class StageExecution:
    """
    Tracks execution metrics and status for a single pipeline stage.
    
    This record captures all relevant information about a stage execution,
    including timing, status transitions, retry attempts, and error information.
    """
    
    stage: Stages                                    # Stage identifier
    status: ExecutionStatus = ExecutionStatus.PENDING  # Current execution state
    start_time: Optional[datetime] = None           # Execution start timestamp
    end_time: Optional[datetime] = None             # Execution end timestamp
    error: Optional[Exception] = None               # Exception if execution failed
    duration_seconds: float = 0.0                   # Total execution time
    retry_count: int = 0                            # Number of retry attempts
    skip_reason: Optional[str] = None               # Reason if stage was skipped
    
    @property
    def is_complete(self) -> bool:
        """Returns True if stage execution has reached terminal state."""
        return self.status in [ExecutionStatus.SUCCESS, ExecutionStatus.FAILED, ExecutionStatus.SKIPPED]
    
    @property
    def was_successful(self) -> bool:
        """Returns True if stage completed successfully."""
        return self.status == ExecutionStatus.SUCCESS
    
    @property
    def was_skipped(self) -> bool:
        """Returns True if stage was skipped due to preconditions."""
        return self.status == ExecutionStatus.SKIPPED
