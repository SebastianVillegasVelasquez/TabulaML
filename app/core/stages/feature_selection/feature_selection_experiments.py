from app.core.context import Context
from app.core.enums import ModelRetrieveType, ProblemType
from app.core.model_bank import ModelRetrieveFactory
from app.core.stages.feature_selection.composer import ExperimentComposer


def get_feature_selection_experiments(context: Context):
    """
    This method generates feature selection experiments using the ExperimentComposer.
    First, it retrieves the preprocessing pipeline from the data handler stage.
    Then, it loads the default selectors and models for classification.

    """

    # Retrieve the preprocessing pipeline from the data handler stage
    # preprocessing = context.stage_results[Stages.DATA_HANDLER].results["preprocessing"]

    selectors = (
        ModelRetrieveFactory.create(
            model_retrieve_type=ModelRetrieveType.SELECTOR,
            problem_type=ProblemType.CLASSIFICATION,
        )
    ).load_defaults()

    models = (
        ModelRetrieveFactory.create(
            model_retrieve_type=ModelRetrieveType.BASELINE,
            problem_type=ProblemType.CLASSIFICATION,
        )
    ).load_defaults()

    return ExperimentComposer(context, selectors, models).generate()
