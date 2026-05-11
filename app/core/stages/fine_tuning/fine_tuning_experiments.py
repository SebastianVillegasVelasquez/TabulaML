from app.core.context.context import Context
from app.core.experiments import ExperimentDefinition
from app.core.stages.fine_tuning.tuner_factory import FineTunerFactory
from app.core.stages.fine_tuning.tuner_strategies import TunerStrategy
from app.utils.logger import logger


def get_fine_tuning_experiments(context: Context):
    """
    Prepare and return a list of ExperimentDefinition objects for fine tuning.

    Each model from the model selection stage will be fine-tuned using a strategy defined in define_tuner_strategy.
    The strategies could be GridSearchCV or Optuna.

    param: context: Context object containing the stage results.
    """
    from app.core.enums import Stages
    from sklearn.base import clone

    stage_results = context.stage_results.get(Stages.MODEL_SELECTION, None)
    if not stage_results:
        raise ValueError("Not found stage results for model selection")

    experiments = []

    for model_name, model_results in stage_results.metadata[
        "top_k_models_by_family"
    ].items():
        tuner_strategy = define_tuner_strategy(model_name)
        logger.info(f"Fine tuning with {tuner_strategy} for {model_name}")
        tuner = FineTunerFactory.create_tuner(
            tuner_strategy=tuner_strategy, context=context
        )
        try:
            pipeline = clone(model_results.pipeline)

            tune_result = tuner.tune(model_name=model_name, pipeline=pipeline)

            tuned_pipeline = tune_result["best_pipeline"]

            experiments.append(
                ExperimentDefinition(
                    name=f"{model_name}_fine_tuning_{tuner_strategy.value}",
                    stage=Stages.FINE_TUNING,
                    pipeline_builder=lambda prep, pipe=tuned_pipeline: (
                        pipe
                    ),  # This returns the pipeline
                    metadata={
                        "model": model_name,
                        "tuner_strategy": tuner_strategy.value,
                        "tuner_params": tune_result["best_params"],
                        "tuner_score": tune_result["best_score"],
                    },
                )
            )
        except Exception as e:
            logger.error(f"Error tuning {model_name}: {str(e)}", exc_info=True)

        logger.info(f"Experiments: {experiments}")
    return experiments


def define_tuner_strategy(model_name: str):
    if model_name in ["logistic_regression", "ridge_classifier", "sgd_classifier"]:
        return TunerStrategy.GRID_SEARCH
    elif model_name in [
        "random_forest",
        "gradient_boosting",
        "extra_trees",
        "decision_tree",
    ]:
        return TunerStrategy.OPTUNA
    else:
        return TunerStrategy.OPTUNA
