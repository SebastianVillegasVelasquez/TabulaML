from app.core.domain.experiments.experiment_result import ExperimentResult


class Evaluator:
    """
    Tracks, compares and formats experiment results.
    """

    def __init__(self, metric: str):
        self.metric = metric
        self.results: list[ExperimentResult] = []

    def add_result(self, result: ExperimentResult) -> None:
        self.results.append(result)

    def get_best(self) -> ExperimentResult:
        if not self.results:
            raise RuntimeError("No experiments have been evaluated.")

        return max(
            self.results,
            key=lambda r: r.metrics[self.metric]
        )

    def summary(self) -> list[dict]:
        """
        Returns a summary of all experiments.
        """
        return [
            {
                "model": r.config["model"],
                self.metric: r.metrics[self.metric]
            }
            for r in self.results
        ]