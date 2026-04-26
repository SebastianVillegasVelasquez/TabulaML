from typing import List, Tuple, Any

from sklearn.base import BaseEstimator

from app.core.domain.experiments import ExperimentDefinition
from app.core.enums import ProblemType
from app.core.context.context import Context
from app.core.enums import Stages
from app.core.ml.pipeline_builder import PipelineBuilder
from app.core.domain.experiments.experiment_result import ExperimentResult


def get_model_ensemble_experiments(context: Context) -> list[ExperimentDefinition]:
    """Builds ensemble experiment definitions using shared preprocessing and feature selection.

    This function constructs ensemble-based experiments (e.g., voting and stacking)
    using models obtained from the fine-tuning stage. All generated models are wrapped
    into a unified pipeline that includes:

    - A shared preprocessing step
    - A shared feature selection step
    - The ensemble model (voting or stacking)

    This design ensures consistency with previous evaluation stages, as all base
    models were originally trained under the same preprocessing and feature selection
    conditions.

    Args:
        context (Context): Execution context containing:
            - Fine-tuned model results
            - Preprocessing configuration
            - Feature selection metadata
            - Problem type configuration

    Returns:
        PipelineBuilder: A pipeline pipeline_builder instance that constructs a pipeline with the following steps:
            - Preprocessing
            - Feature Selection
            - Ensemble Model (voting or stacking)
    """
    from app.core.domain.experiments.experiment_definition import ExperimentDefinition

    fine_tuned_results = context.stage_results[Stages.FINE_TUNING].results
    feature_selection_step = context.stage_results[Stages.FEATURE_SELECTION].metadata[
        "selector_estimator"
    ]

    results = []

    models = get_models(
        problem_type=context.config.problem_type,
        results=fine_tuned_results,
    )

    for name, model in models:

        def _build_pipeline(
            preprocessing_step, feature_selection_step=feature_selection_step, model=model
        ) -> PipelineBuilder:
            """Constructs a unified pipeline for an ensemble model.

            This pipeline applies a shared preprocessing and feature selection
            workflow before passing the transformed data to the ensemble model.

            The design ensures that all ensemble models operate under the same
            data transformation conditions used during previous training stages,
            improving consistency and comparability of results.

            This may changed in a futuro for better individualization of the models,
            each having it's own preprocessing and feature selection steps.

            Args:
                preprocessing_step: A scikit-learn compatible transformer for data preprocessing.
                feature_selection_step: A scikit-learn compatible transformer for feature selection.
                model: An instantiated ensemble model (e.g., VotingClassifier, StackingRegressor)
                coming from the 'get_models' function.

            Returns:
                PipelineBuilder: A pipeline pipeline_builder instance that constructs a pipeline with the following steps:
                    - Preprocessing
                    - Feature Selection
                    - Ensemble Model
            """
            steps = [("preprocessing", preprocessing_step)]
            steps.append(("selector", feature_selection_step))
            steps.append(("model", model))

            return PipelineBuilder(steps=steps)

        results.append(
            ExperimentDefinition(
                name=f"{name}_{context.config.problem_type.value}",
                stage="model_ensemble",
                pipeline_builder=_build_pipeline,
                metadata={"model": name},
            )
        )

    return results


def get_models(problem_type: ProblemType, results) -> list[Tuple[str, Any]]:
    """Builds ensemble models based on the problem type and training results.

    This function orchestrates the creation of ensemble models by:
    - Extracting pipelines from previous results
    - Building estimators
    - Selecting appropriate ensemble strategies depending on the problem type
    - Configuring voting strategy when applicable (classification only)

    Args:
        problem_type (ProblemType): Type of machine learning problem
            (e.g., CLASSIFICATION or REGRESSION).
        results (Any): Object containing trained pipeline results.

    Returns:
        list[tuple[str, object]]: List of tuples where each tuple contains:
            - Model name (str)
            - Instantiated ensemble model
    """
    pipelines = _get_pipelines_from_results(results)
    estimators = _build_estimators(pipelines)

    match problem_type:
        case ProblemType.CLASSIFICATION:
            support_proba = _supports_proba(pipelines)
            ensemble_models = _get_ensemble_model_for_classification()
            return _build_classification_ensemble_models(
                ensemble_models,
                estimators,
                support_proba,
            )

        case ProblemType.REGRESSION:
            ensemble_models = _get_ensemble_model_for_regression()
            return _build_regression_ensemble_models(
                ensemble_models,
                estimators,
            )

        case _:
            raise ValueError(f"Unsupported problem type: {problem_type}")


def _build_estimators(pipelines: List[Any]) -> List[Tuple[str, Any]]:
    from sklearn.pipeline import Pipeline

    """Builds a list of estimators from trained pipelines.

    Each estimator is represented as a tuple containing:
    - A short identifier (sigla) derived from the model class name
    - The pipeline steps excluding the initial preprocessing step(s)

    This structure is typically used for ensemble methods in scikit-learn,
    where estimators must follow the format: (name, estimator).

    Args:
        pipelines (list[Any]): List of fitted pipeline objects. Each pipeline
            is expected to contain a step named "model".

    Returns:
        list[tuple[str, Any]]: List of tuples where each tuple contains:
            - A short name (str) for the estimator
            - The estimator or pipeline steps (Any)
    """
    return [
        (
            get_sigla(pipeline.named_steps["model"].__class__.__name__),
            Pipeline(steps=[("model", pipeline.named_steps["model"])]),
        )
        for pipeline in pipelines
    ]


def get_sigla(name: str) -> str:
    """Generates a short identifier (sigla) from a class name.

    The sigla is constructed by extracting uppercase letters from the
    class name, converting them to lowercase, and taking the first two.
    If fewer than two uppercase letters are found, the first two characters
    of the name are used instead.

    Examples:
        "RandomForestClassifier" -> "rf"
        "LogisticRegression" -> "lr"
        "KNN" -> "kn"

    Args:
        name (str): Name of the class or model.

    Returns:
        str: A short lowercase identifier of length 2.
    """
    sigla = "".join(c.lower() for c in name if c.isupper())
    return sigla[:2] if len(sigla) >= 2 else name[:2].lower()


def _supports_proba(pipeline: list | None = None, estimator: BaseEstimator | None = None) -> bool:
    """Determines whether estimators support probability predictions.

    This function checks if a given estimator or a collection of estimators
    implements the `predict_proba` method. When a list (or iterable) is provided,
    it verifies that all elements support probability prediction.

    Args:
        pipeline (list | None, optional): Iterable of estimators or pipelines.
            If provided, the function evaluates all elements recursively.
        estimator (BaseEstimator | None, optional): A single estimator to evaluate.

    Returns:
        bool: True if all evaluated estimators support `predict_proba`,
        False otherwise.
    """
    from collections.abc import Iterable

    if pipeline and isinstance(pipeline, Iterable):
        return all(_supports_proba(estimator=est) for est in pipeline)

    return hasattr(estimator, "predict_proba") and callable(estimator.predict_proba)


def _get_pipelines_from_results(results: list[ExperimentResult]) -> list[Any]:
    """Extracts pipelines from experiment results.

    This function retrieves the `pipeline` attribute from each result object.
    If a result does not contain a pipeline, `None` is returned in its place.

    Args:
        results (list[ExperimentResult]): List of experiment result objects,
            each potentially containing a trained pipeline.

    Returns:
        list[Any]: List of pipeline objects or None values if not present.
    """
    return [getattr(result, "pipeline", None) for result in results]


def _build_classification_ensemble_models(ensemble_models, estimators, support_proba):
    """Build ensemble models for classification problems.

    For classification, this function dynamically selects between
    soft and hard voting depending on whether all estimators support
    probability predictions.

    Args:
        ensemble_models (list[tuple[str, type]]): List of ensemble model
            definitions as (name, class).
        estimators (list[tuple[str, object]]): Base estimators.
        support_proba (bool): Whether all estimators support `predict_proba`.

    Returns:
        list[tuple[str, object]]: Instantiated ensemble models.
    """
    params = {
        "estimators": estimators,
        "voting": "soft" if support_proba else "hard",
    }

    return [
        (
            name,
            cls(**params) if name == "voting" else cls(estimators=estimators),
        )
        for name, cls in ensemble_models
    ]


def _build_regression_ensemble_models(ensemble_models, estimators):
    """Build ensemble models for regression problems.

    Unlike classification, regression ensembles do not rely on voting
    strategies based on probabilities. All models are instantiated
    using the provided estimators.

    Args:
        ensemble_models (list[tuple[str, type]]): List of ensemble model
            definitions as (name, class).
        estimators (list[tuple[str, object]]): Base estimators.

    Returns:
        list[tuple[str, object]]: Instantiated ensemble models.
    """
    return [(name, cls(estimators=estimators)) for name, cls in ensemble_models]


def _get_ensemble_model_for_classification() -> List[Tuple[str, Any]]:
    """
    Return a list of tuples containing the name of the ensemble model and
    the corresponding estimator class for classification problems.

    returns:
        List[Tuple[str, BaseEstimator]]: A list of tuples containing the name and the
        estimator class.
    """
    from sklearn.ensemble import VotingClassifier, StackingClassifier

    return [
        ("stacking", StackingClassifier),
        ("voting", VotingClassifier),
    ]


def _get_ensemble_model_for_regression() -> List[Tuple[str, Any]]:
    """
    Return a list of tuples containing the name of the ensemble model and the
    corresponding estimator class for regression problems.

    Returns:
        List[Tuple[str, BaseEstimator]]: A list of tuples containing the name and the
        estimator class.
    """
    from sklearn.ensemble import VotingRegressor, StackingRegressor

    return [
        ("stacking", StackingRegressor),
        ("voting", VotingRegressor),
    ]
