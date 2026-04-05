from app.core.context import RunContext
from app.core.domain.experiments.experiment_definition import ExperimentDefinition
from app.core.enums import EvaluationType
from app.core.enums.problems_type import ProblemsType
from app.core.enums.stages import Stages


def get_final_evaluation_experiments(context: RunContext) -> list[ExperimentDefinition]:
    models_with_threshold = _get_models(context)

    results = []

    for item in models_with_threshold:
        threshold_value = item['threshold']
        use_threshold = item['use_threshold']
        model = item['model']

        def builder(preprocessing, model=model):
            return model

        results.append(
            ExperimentDefinition(
                name=f"{item['model'].named_steps['model']}_{Stages.FINAL_EVALUATION}",
                stage=Stages.FINAL_EVALUATION,
                builder=builder,
                use_threshold=use_threshold,
                evaluation_type=EvaluationType.THRESHOLD,
                threshold=threshold_value,
                metadata={
                    "threshold": threshold_value
                }
            )
        )

    return results


def _get_models(context: RunContext):
    from app.core.stages.threshold_optimizer import ThresholdOptimizer

    is_classification = context.config.problem_type == ProblemsType.CLASSIFICATION

    models = [
        context.stage_results[Stages.FINE_TUNING].best_experiment.pipeline,
        context.stage_results[Stages.MODEL_ENSEMBLE].best_experiment.pipeline,
    ]

    results = []

    for model in models:
        results.append({
            "model": model,
            "threshold": ThresholdOptimizer(context).find_best_threshold(model),
            "use_threshold": is_classification,
        })
    else:
        results.append({
            "model": model,
            "threshold": None,
            "use_threshold": False,
        })

    return results
