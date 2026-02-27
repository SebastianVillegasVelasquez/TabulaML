from typing import Union, List

from app.core.domain.experiments.experiment_result import ExperimentResult


class Evaluator:

    def __init__(self, metric: Union[List[str], str], mode: str = "max"):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")

        self.metric = metric
        self.mode = mode
        self.results: list[ExperimentResult] = []

    def add_result(self, result: ExperimentResult):

        if isinstance(self.metric, list):
            for m in self.metric:
                if m not in result.metrics:
                    raise ValueError(
                        f"Metric '{m}' not found in experiment {result.name}"
                    )
        else:
            if self.metric not in result.metrics:
                raise ValueError(
                    f"Metric '{self.metric}' not found in experiment {result.name}"
                )

        self.results.append(result)

    def get_best(self) -> ExperimentResult:

        if not self.results:
            raise RuntimeError("No experiments evaluated.")

        if isinstance(self.metric, list):

            def key_func(r):
                return tuple(r.metrics[m] for m in self.metric)

        else:
            key_func = lambda r: r.metrics[self.metric]

        if self.mode == "max":
            return max(self.results, key=key_func)
        else:
            return min(self.results, key=key_func)