from typing import List

from app.core.context.context import Context
from experiments import ExperimentDefinition
from app.core.stages.factories.base_experiment_registry import BaseExperimentFactory
from app.core.stages.feature_selection.feature_selection_experiments import (
    get_feature_selection_experiments,
)


class FeatureSelectionExperimentFactory(BaseExperimentFactory):

    def create_experiments(self, context: Context = None) -> List[ExperimentDefinition]:
        return get_feature_selection_experiments(context=context)
