from abc import ABC, abstractmethod

from app.core.context.run_context import RunContext, StageResult
from app.core.context.stages import Stages
from app.core.domain.experiments.experiment import Experiment
from app.core.stages.registry import get_stage_experiments
from app.utils.logger import logger



class Stage(ABC):

    def __init__(self, context: RunContext):
        self.context = context
        self.stage = self.get_stage_type()
        self.definitions = get_stage_experiments(self.stage, context=self.context)

    @abstractmethod
    def get_stage_type(self) -> Stages:
        """Child classes must define their stage type"""
        pass

    def run(self):
        logger.info(f"Running {self.stage} stage...")

        # Get the preprocessing ColumnTransformer from the DATA_HANDLER stage
        preprocessing = self.context.stage_results[Stages.DATA_HANDLER].results["preprocessing"]

        results = []

        for definition in self.definitions:
            # Build the pipeline using the definition's builder
            pipeline_builder = definition.builder(preprocessing)

            # Create and run the experiment
            experiment = Experiment(
                name=f"{self.stage.value}_{definition.name}",
                pipeline_builder=pipeline_builder,
                context=self.context,
                cv=5,
                metadata=definition.metadata
            )

            result = experiment.run(self.context.config.X_train, self.context.config.y_train)
            results.append(result)
            logger.info(f"Finished experiment {experiment.name}")

        # Sort results by primary metric (descending)
        # Use the first scoring metric as primary
        primary_metric = self.context.config.scoring[0] if isinstance(self.context.config.scoring, list) else self.context.config.scoring
        results_sorted = sorted(
            results,
            key=lambda r: r.metrics.get(f"test_{primary_metric}", 0),
            reverse=True
        )

        # Get the best experiment
        best_experiment = results_sorted[0] if results_sorted else None

        # For feature selection, extract top-k selectors (group by selector type)
        top_k_selectors = self._extract_top_k_selectors(results_sorted) if self.stage == Stages.FEATURE_SELECTION else {}

        # Store results for THIS stage
        self.context.stage_results[self.stage] = StageResult(
            name=self.stage,
            results=results_sorted,
            best_experiment=best_experiment,
            metadata={
                "top_k_selectors": top_k_selectors,
                "total_experiments": len(results)
            }
        )

        logger.info(f"Best experiment: {best_experiment.name} with {primary_metric}={best_experiment.metrics.get(f'test_{primary_metric}', 0):.4f}")

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