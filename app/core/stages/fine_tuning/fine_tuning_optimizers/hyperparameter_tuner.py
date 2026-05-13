from app.core.enums import ProblemType
from app.core.stages.fine_tuning.tuner_strategies import TunerStrategy

tuning = {
    ProblemType.CLASSIFICATION: {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "saga"],
        },
        "ridge_classifier": {"model__alpha": [0.1, 1.0, 10.0]},
        "sgd_classifier": {
            "model__alpha": [1e-4, 1e-3, 1e-2],
            "model__loss": ["hinge", "log_loss"],
            "model__penalty": ["l2", "l1"],
        },
        "random_forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [10, 15, 20, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
            "model__subsample": [0.8, 1.0],
        },
        "extra_trees": {
            "model__n_estimators": [100, 150, 300],
            "model__max_depth": [10, 15, 20, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "decision_tree": {
            "model__max_depth": [5, 10, 20, None],
            "model__min_samples_split": [2, 10, 20],
            "model__min_samples_leaf": [1, 2, 5],
        },
        "kneighbors": {
            "model__n_neighbors": [3, 5, 7, 11],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["minkowski", "euclidean", "manhattan"],
        },
        "gaussian_nb": {"model__var_smoothing": [1e-9, 1e-8, 1e-7]},
    },
    ProblemType.REGRESSION: {
        "random_forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [10, 20, None],
            "model__min_samples_split": [2, 5, 10],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 5],
        },
        "decision_tree": {
            "model__max_depth": [5, 10, None],
            "model__min_samples_split": [2, 10],
        },
        "kneighbors": {
            "model__n_neighbors": [3, 5, 7],
            "model__weights": ["uniform", "distance"],
        },
    },
}


def _convert_to_optuna_space(
    param_grid: dict,
) -> dict[str, Union[tuple[str, Any], tuple[str, Any, Any]]]:
    """
    Transform helper param_grid to Optuna space.
    """

    optuna_space: dict[str, Union[tuple[str, Any], tuple[str, Any, Any]]] = {}

    for param, values in param_grid.items():
        if any(v is None for v in values):
            optuna_space[param] = ("categorical", values)
            continue

        if all(isinstance(v, int) for v in values):
            optuna_space[param] = ("int", min(values), max(values))

        elif all(isinstance(v, float) for v in values):
            optuna_space[param] = ("float", min(values), max(values))

        else:
            optuna_space[param] = ("categorical", values)

    return optuna_space


def get_set_hyperparameter(
    problem_type: ProblemType,
    model: str,
    tuner_strategy: TunerStrategy = TunerStrategy.OPTUNA,
) -> dict:
    try:
        param_grid = tuning[problem_type][model]
    except KeyError:
        raise ValueError(
            f"No hay hiperparámetros definidos para el modelo '{model}' "
            f"en el problema '{problem_type.name}'"
        )

    if tuner_strategy == TunerStrategy.GRID_SEARCH:
        return param_grid

    elif tuner_strategy == TunerStrategy.OPTUNA:
        return _convert_to_optuna_space(param_grid)

    else:
        raise ValueError(f"Tuner strategy '{tuner_strategy}' no soportada")


get_set_hyperparameter(ProblemType.CLASSIFICATION, "extra_trees", TunerStrategy.OPTUNA)
