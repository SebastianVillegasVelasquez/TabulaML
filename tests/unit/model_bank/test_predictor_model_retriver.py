from typing import Callable

from mypy.nodes import Context

from app.core.enums import ModelRetrieveType, ProblemType
from app.core.model_bank import BaseModelRetriever, ModelSpec


class TestPredictorModelRetriever:

    def test_instance_retrieve(self,
                               build_context: Context,
                               load_models_from_factory: Callable[
                                   [ModelRetrieveType, ProblemType, Context], BaseModelRetriever]):

        factory = load_models_from_factory(retrieve_type=ModelRetrieveType.PREDICTOR,
                                           problem_type=ProblemType,
                                           context=build_context)

        assert isinstance(factory, BaseModelRetriever)

    def test_instance_retrieve_return_list_model_spec(self,
                               build_context: Context,
                               load_models_from_factory: Callable[
                                   [ModelRetrieveType, ProblemType, Context], BaseModelRetriever]):
        factory = load_models_from_factory(retrieve_type=ModelRetrieveType.PREDICTOR,
                                           problem_type=ProblemType,
                                           context=build_context).load_defaults()

        assert isinstance(factory, list)
        assert all(isinstance(m, ModelSpec) for m in factory)
