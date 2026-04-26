from typing import List

from app.core.enums import Stages
from app.core.domain.experiments.experiment_result import ExperimentResult
from app.core.context.context import Context, StageResult


class Evaluator:

    def __init__(
        self, mode: str = "max", results: List[ExperimentResult] = None, context: Context = None
    ):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")

        self.mode = mode
        self.results = results if results is not None else []
        self.context = context
        self.priority_metric = self.context.config.priority_metric
        self.stage = self.context.current_stage

    def add_result(self, result: ExperimentResult):
        """
        This function adds an experiment result to the evaluator.
        It has to be called before the returning best models functions.

        """

        if isinstance(self.priority_metric, list):
            for m in self.priority_metric:
                if m not in result.metrics:
                    raise ValueError(f"Metric '{m}' not found in experiment {result.name}")
        else:
            if self.priority_metric not in result.metrics:
                raise ValueError(
                    f"Metric '{self.priority_metric}' not found in experiment {result.name}"
                )

        self.results.append(result)

    def get_best(self) -> ExperimentResult:

        if not self.results:
            raise RuntimeError("No experiments evaluated.")

        if isinstance(self.priority_metric, list):

            def key_func(r):
                return tuple(r.metrics[m] for m in self.priority_metric)

        else:
            key_func = lambda r: r.metrics[self.priority_metric]

        if self.mode == "max":
            return max(self.results, key=key_func)
        else:
            return min(self.results, key=key_func)

    def extract_best_experiments(self, return_best=False, k=3) -> ExperimentResult:
        """
        Extract top-k best experiments based on the priority metric.
        Handles ties by including all experiments with the same metric value.

        :param k: Number of top experiments to extract
        :param return_best: If True, returns only the best experiment,
        otherwise returns a list of ExperimentResult objects
        :return: List of top-k ExperimentResult objects
        """
        # Sort the results based on the primary metric
        # (default to the first metric in scoring if no priority is set)

        priority_metric = (
            self.priority_metric
            if self.priority_metric
            else list(self.results[0].metrics.keys())[0]
        )

        best_experiment = None

        results_sorted = sorted(
            self.results, key=lambda r: r.metrics.get(f"test_{priority_metric}", 0), reverse=True
        )

        # Get and return only the best experiment if there is no need for top k best models
        # (i.e., final ensemble model)
        if return_best:
            best_experiment = results_sorted[0] if results_sorted else None
            return best_experiment

        # For feature selection, extract top-k selectors (group by selector type)
        top_k_selectors = (
            self._extract_top_k_selectors(results_sorted)
            if self.stage == Stages.FEATURE_SELECTION
            else {}
        )

        # Store results in the context
        self.context.stage_results[self.stage] = StageResult(
            name=self.stage,
            results=results_sorted,
            best_experiment=best_experiment,
            metadata={"top_k_selectors": top_k_selectors, "total_experiments": len(self.results)},
        )

        # Extract top-k unique selectors

    @staticmethod
    def _extract_top_k_selectors(sorted_results, k=3):
        """
        Extract top-k unique selectors from feature selection results.
        Groups by selector name and takes the best performing configuration.

        :param sorted_results: List of ExperimentResult sorted by performance
        :param k: Number of top selectors to extract
        :return: Dictionary mapping selector names to their best ExperimentResult
        """
        selector_best = {}

        for result in sorted_results:
            selector_name = result.config.get("selector", "unknown")

            # Keep only the first (best) occurrence of each selector
            if selector_name not in selector_best:
                selector_best[selector_name] = result

        # Return top-k selectors
        top_k = list(selector_best.items())[:k]
        return dict(top_k)
