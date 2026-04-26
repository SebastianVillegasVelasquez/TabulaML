from app.core.context.context import Context
from app.core.enums import Stages
from app.core.stages.factories.stages_experiments_factories.feature_selection_factory import (
    FeatureSelectionExperimentFactory,
)
from app.core.stages.factories.stages_experiments_factories.fine_tuning_factory import (
    FineTuningExperimentsFactory,
)
from app.core.stages.factories.stages_experiments_factories.model_selection_factory import (
    ModelSelectionExperimentsFactory,
)
from app.core.stages.factories.stages_experiments_factories.model_ensemble_experiment_factory import (
    ModelEnsembleExperimentFactory,
)
from app.core.stages.factories.stages_experiments_factories.final_evaluation_factory import (
    FinalEvaluationFactory,
)
from app.utils.logger import logger


class StageExperimentFactory:
    _FACTORIES = {
        Stages.FEATURE_SELECTION: FeatureSelectionExperimentFactory,
        Stages.MODEL_SELECTION: ModelSelectionExperimentsFactory,
        Stages.FINE_TUNING: FineTuningExperimentsFactory,
        Stages.MODEL_ENSEMBLE: ModelEnsembleExperimentFactory,
        Stages.FINAL_EVALUATION: FinalEvaluationFactory,
    }

    @classmethod
    def create(
        cls,
        stage: Stages,
        context: Context = None,
    ):
        """Create experiment factory for the given stage."""
        if stage not in cls._FACTORIES:
            logger.debug(f"Stage '{stage}' no tiene un factory registrado.")
            raise ValueError(
                f"Stage '{stage}' no está registrada. "
                f"Stages disponibles: {list(cls._FACTORIES.keys())}"
            )

        factory_class = cls._FACTORIES[stage]
        logger.debug(f"Usando factory: {factory_class.__name__} para stage: {stage}")
        return factory_class()
