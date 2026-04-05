from app.core.context import RunContext
from app.core.stages.factories.base_experiment_registry import BaseExperimentFactory
from app.core.stages.final_evaluation.final_evaluation_experiments import get_final_evaluation_experiments


class FinalEvaluationFactory(BaseExperimentFactory):

    def create_experiments(self, context:RunContext=None):
        if context is None:
            raise ValueError(f"RunContext parameter must not be None for ModelEnsembleExperimentFactory\n"
                     f"It has to be able to get the models from the feature selection stage")

        return get_final_evaluation_experiments(context=context)