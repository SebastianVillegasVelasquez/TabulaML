from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline

from app.core.metrics.metrics import get_primary_metric
from app.core.enums.problems_type import ProblemsType
from app.core.context.run_context import RunContext
from app.core.stages.fine_tuning.fine_tuning_optimizers.hyperparameter_tuner import get_set_hyperparameter
from app.core.stages.fine_tuning.tuner_strategies import TunerStrategy
from app.utils.logger import logger


class BaseTuner(ABC):
    def __init__(self,
                 context: RunContext,
                 ):
        self.context = context

    @abstractmethod
    def tune(self, model_name:str, pipeline: Pipeline): ...

    @abstractmethod
    def get_tuner_strategy(self) -> TunerStrategy:
        """Return the strategy type for this tuner"""
        ...

    def _get_hyperparameters(self, model_name: str) -> dict:
        logger.info(f"Getting hyperparameters for {model_name} to {self.context.config.problem_type} problem")
        return get_set_hyperparameter(
            problem_type=self.context.config.problem_type,
            model=model_name,
            tuner_strategy=self.get_tuner_strategy()
        )


class OptunaTunerStrategy(BaseTuner):

    def get_tuner_strategy(self) -> TunerStrategy:
        return TunerStrategy.OPTUNA

    def tune(self, model_name: str, pipeline: Pipeline):
        import optuna
        from sklearn.model_selection import cross_val_score

        X = self.context.config.X_train
        y = self.context.config.y_train

        param_config = self._get_hyperparameters(model_name=model_name)

        def objective(trial):
            params = self._build_optuna_params(trial, param_config)

            pipeline.set_params(**params)

            score = cross_val_score(
                pipeline,
                X,
                y,
                cv=3,
                scoring=get_primary_metric(self.context.config.problem_type)
            ).mean()

            return score

        study = optuna.create_study(direction=self._get_direction())
        study.optimize(objective, n_trials=30)

        pipeline.set_params(**study.best_params)

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "best_pipeline": pipeline,
        }

    @staticmethod
    def _build_optuna_params(trial, param_config: dict):

        params = {}

        for param_name, config in param_config.items():
            param_type = config[0]

            if param_type == "int":
                _, low, high = config
                params[param_name] = trial.suggest_int(param_name, low, high)

            elif param_type == "float":
                _, low, high = config
                params[param_name] = trial.suggest_float(param_name, low, high)

            elif param_type == "categorical":
                _, choices = config
                params[param_name] = trial.suggest_categorical(param_name, choices)

            else:
                raise ValueError(f"Unsupported param type: {param_type}")

        return params

    def _get_direction(self):
        problem_type = self.context.config.problem_type
        return "maximize" if problem_type == ProblemsType.CLASSIFICATION else "minimize"


class GridSearchCVTunerStrategy(BaseTuner):

    def get_tuner_strategy(self) -> TunerStrategy:
        return TunerStrategy.GRID_SEARCH

    def tune(self, model_name: str, pipeline: Pipeline):
        from sklearn.model_selection import GridSearchCV

        X = self.context.config.X_train
        y = self.context.config.y_train

        param_grid = self._get_hyperparameters(model_name=model_name)

        scoring = get_primary_metric(self.context.config.problem_type)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        best_pipeline = grid_search.best_estimator_

        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "best_pipeline": best_pipeline,
        }

