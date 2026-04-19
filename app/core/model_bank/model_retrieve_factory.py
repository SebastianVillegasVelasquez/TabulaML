from app.core.enums import ModelRetrieveType, ProblemsType


class ModelRetrieveFactory:
    """
    Factory class for creating model retrievers.

    These retrievers are responsible for retrieving models from the model bank.
    Such as Baselines, Selectors, Ensembles, etc.

    """
    _MODELS_TYPE_TO_FACTORY = {}


    @classmethod
    def register(cls, model_type, builder):
        cls._MODELS_TYPE_TO_FACTORY[model_type] = builder

    @classmethod
    def register_defaults(cls):
        from app.core.model_bank import BaselineModelRetriever
        from app.core.model_bank import SelectorModelRetriever
        if not cls._MODELS_TYPE_TO_FACTORY:
            cls._MODELS_TYPE_TO_FACTORY = {
                ModelRetrieveType.BASELINE: BaselineModelRetriever,
                ModelRetrieveType.SELECTOR: SelectorModelRetriever,
            }

    @classmethod
    def create(cls, model_retrieve_type: ModelRetrieveType,
               problem_type: ProblemsType):
        cls.register_defaults()
        model_bank_class = cls._MODELS_TYPE_TO_FACTORY.get(model_retrieve_type)
        return model_bank_class(problem_type=problem_type)