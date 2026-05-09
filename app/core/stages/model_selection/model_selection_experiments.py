from app.core.context import Context
from app.core.model_bank import ModelRetrieveFactory
from app.core.enums import ModelRetrieveType, ProblemType
from app.utils.logger import logger


def get_model_selection_experiments(context: Context):
    models = ModelRetrieveFactory.create(
        model_retrieve_type=ModelRetrieveType.PREDICTOR, problem_type=ProblemType.CLASSIFICATION
    ).load_defaults()

    logger.info(f"Models loaded in the model selection stage: {models}")

    return None
