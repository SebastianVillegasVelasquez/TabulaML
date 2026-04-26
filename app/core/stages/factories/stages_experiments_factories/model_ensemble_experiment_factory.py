from app.core.context.context import Context
from app.core.stages.factories.base_experiment_registry import BaseExperimentFactory
from app.core.stages.model_ensemble.model_ensemble_experiments import get_model_ensemble_experiments


class ModelEnsembleExperimentFactory(BaseExperimentFactory):

    def create_experiments(self, context: Context = None):
        if context is None:
            raise ValueError(
                f"Context parameter must not be None for ModelEnsembleExperimentFactory\n"
                f"It has to be able to get the models from the feature selection stage"
            )

        return get_model_ensemble_experiments(context=context)
