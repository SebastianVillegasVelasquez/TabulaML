from typing import List

from app.core.context.run_context import RunContext
from app.core.domain.experiments.experiment_definition import ExperimentDefinition
from app.core.stages.factories.base_experiment_registry import BaseExperimentFactory
from app.core.stages.feature_selection.feature_selection_experiments import get_feature_selection_experiments


class FeatureSelectionExperimentFactory(BaseExperimentFactory):

    def create_experiments(self, context: RunContext = None) -> List[ExperimentDefinition]:
        return get_feature_selection_experiments(context=context)