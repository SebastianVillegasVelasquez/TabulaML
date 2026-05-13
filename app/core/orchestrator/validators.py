"""
Stage-specific validators for precondition checking.

These validators ensure that stages have all required dependencies
before attempting to execute.
"""

from typing import Optional, Tuple

from app.core.context.context import Context
from app.core.enums import ProblemType
from app.core.enums import Stages
from app.core.orchestrator.stage_validator import StageValidator


class FeatureSelectionValidator(StageValidator):
    """
    Validates that FeatureSelectionStage can be executed.

    Preconditions:
    - DATA_HANDLER stage must be completed
    - DATA_HANDLER must have produced results
    """

    def validate(self, context: Context) -> Tuple[bool, Optional[str]]:
        # Check if DATA_HANDLER stage exists in context
        if Stages.DATA_HANDLER not in context.stage_results:
            return False, "DATA_HANDLER stage not completed"

        # Check if DATA_HANDLER produced any results
        data_handler_result = context.stage_results[Stages.DATA_HANDLER]
        if data_handler_result.results is None or len(data_handler_result.results) == 0:
            return False, "DATA_HANDLER produced no results"

        # All preconditions met
        return True, None


class ModelSelectionValidator(StageValidator):
    """
    Validates that ModelSelectionStage can be executed.

    Preconditions:
    - FEATURE_SELECTION stage must be completed
    """

    def validate(self, context: Context) -> Tuple[bool, Optional[str]]:
        # Check if FEATURE_SELECTION stage exists in context
        if Stages.FEATURE_SELECTION not in context.stage_results:
            return False, "FEATURE_SELECTION stage not completed"

        # All preconditions met
        return True, None


class FineTuningValidator(StageValidator):
    """
    Validates that FineTuningStage can be executed.

    Preconditions:
    - MODEL_SELECTION stage must be completed
    """

    def validate(self, context: Context) -> Tuple[bool, Optional[str]]:
        # Check if MODEL_SELECTION stage exists in context
        if Stages.MODEL_SELECTION not in context.stage_results:
            return False, "MODEL_SELECTION stage not completed"

        # All preconditions met
        return True, None


class ModelEnsembleValidator(StageValidator):
    """
    Validates that ModelEnsemble can be executed.

    Preconditions:
    - FINE_TUNING stage must be completed
    """

    def validate(self, context: Context) -> Tuple[bool, Optional[str]]:
        # Check if MODEL_SELECTION stage exists in context
        if Stages.FINE_TUNING.value not in context.stage_results:
            return False, "FINE_TUNING stages not completed"
        # All preconditions met
        return True, None


class ModelThresholdExtractionValidator(StageValidator):
    """
    Validates that ModelThresholdExtraction can be executed.

    Preconditions:
    - The problem has to be classification

    """

    def validate(self, context: Context) -> Tuple[bool, Optional[str]]:
        # Check if MODEL_SELECTION stage exists in context
        if context.config.problem_type != ProblemType.CLASSIFICATION:
            return False, "FINE_TUNING stages not completed"
        # All preconditions met
        return True, None
