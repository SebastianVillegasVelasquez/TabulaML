"""
Stage Adapters: Bridges between legacy stage implementations and the pipeline framework.

These adapters provide a compatibility layer, allowing existing stage implementations
to work within the new pipeline architecture. Adapters implement the PipelineStage
interface and delegate execution to the original stage classes.

Design pattern: Adapter Pattern
- Adapts existing stage implementations to the PipelineStage interface
- Maintains backward compatibility with existing code
- Adds validation and lifecycle management without modifying original stages
"""

from typing import Optional
from app.core.context.run_context import RunContext
from app.core.enums.stages import Stages
from app.core.orchestrator.pipeline_stage import PipelineStage
from app.core.orchestrator.stage_validator import StageValidator
from app.core.orchestrator.validators import (
    FeatureSelectionValidator,
    ModelSelectionValidator,
    FineTuningValidator, ModelEnsembleValidator
)
from app.utils.logger import logger


class DataInspectionStageAdapter(PipelineStage):
    """
    Adapter for the data inspection stage.
    
    Responsibilities:
    - Executes initial data loading and validation
    - Generates baseline statistics and data profiles
    - Prepares preprocessing transformers for downstream stages
    
    No preconditions: This is the first stage in the pipeline.
    """

    
    def get_stage_type(self) -> Stages:
        return Stages.DATA_HANDLER
    
    def get_validator(self) -> Optional[StageValidator]:
        # First stage - no preconditions required
        return None
    
    def execute(self, context: RunContext) -> None:
        from app.core.stages.data_inspection.data_inspection import DataInspectionStage
        
        logger.debug("Executing data inspection stage...")
        DataInspectionStage(context=context).run()


class FeatureSelectionStageAdapter(PipelineStage):
    """
    Adapter for the feature selection stage.
    
    Responsibilities:
    - Identifies optimal feature subsets through experimentation
    - Evaluates multiple feature selection strategies
    - Selects best performing features for downstream modeling
    
    Precondition: DATA_HANDLER stage must be completed successfully.
    """
    
    def get_stage_type(self) -> Stages:
        return Stages.FEATURE_SELECTION
    
    def get_validator(self) -> Optional[StageValidator]:
        return FeatureSelectionValidator()
    
    def execute(self, context: RunContext) -> None:
        from app.core.stages.feature_selection.feature_selection_stage import FeatureSelectionStage
        
        logger.debug("Executing feature selection stage...")
        FeatureSelectionStage(context=context).run()


class ModelSelectionStageAdapter(PipelineStage):
    """
    Adapter for the  model selection stage.
    
    Responsibilities:
    - Trains and evaluates multiple ML algorithms
    - Compares model performance using configured metrics
    - Identifies and registers best performing model
    
    Precondition: FEATURE_SELECTION stage must be completed successfully.
    """
    
    def get_stage_type(self) -> Stages:
        return Stages.MODEL_SELECTION
    
    def get_validator(self) -> Optional[StageValidator]:
        return ModelSelectionValidator()
    
    def execute(self, context: RunContext) -> None:
        from app.core.stages.model_selection.model_selection_stage import ModelSelectionStage
        
        logger.debug("Executing model selection stage...")
        ModelSelectionStage(context=context).run()


class FineTuningStageAdapter(PipelineStage):
    """
    Adapter for hyperparameter fine-tuning stage.
    
    Responsibilities:
    - Performs hyperparameter optimization on selected models
    - Evaluates configurations using cross-validation
    - Produces final tuned model for deployment
    
    Precondition: MODEL_SELECTION stage must be completed successfully.
    """
    
    def get_stage_type(self) -> Stages:
        return Stages.FINE_TUNING
    
    def get_validator(self) -> Optional[StageValidator]:
        return FineTuningValidator()
    
    def execute(self, context: RunContext) -> None:
        from app.core.stages.fine_tuning.fine_tuning_stage import FineTuningStage
        
        logger.debug("Executing fine-tuning stage...")
        FineTuningStage(context=context).run()


class ModelEnsambleStageAdapter(PipelineStage):
    """
    Adapter for model ensamble stage.

    Responsibilities:
    - Combines multiple models to improve performance
    - Evaluates ensemble strategies (e.g., voting, stacking)
    - Produces a final ensemble model for deployment

    Precondition: MODEL_SELECTION and FINE_TUNING stages must be completed successfully.
    """

    def get_stage_type(self) -> Stages:
        return Stages.MODEL_ENSEMBLE

    def get_validator(self) -> Optional[StageValidator]:
        # Ensemble can be done after model selection, no strict preconditions
        return ModelEnsembleValidator()

    def execute(self, context: RunContext) -> None:
        from app.core.stages.model_ensemble.model_ensemble_stage import ModelEnsembleStage

        logger.debug("Executing model ensamble stage...")
        ModelEnsembleStage(context=context).run()


