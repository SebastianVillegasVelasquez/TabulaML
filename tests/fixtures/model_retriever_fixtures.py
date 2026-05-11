import pytest

from app.core.context import Context
from app.core.enums import ModelRetrieveType, ProblemType
from app.core.model_bank import ModelRetrieveFactory


@pytest.fixture
def load_models_from_factory():
    def _factory(retrieve_type:ModelRetrieveType, problem_type:ProblemType, context: Context = None):
        return ModelRetrieveFactory.create(model_retrieve_type=retrieve_type,
                                           problem_type=problem_type,
                                           context=context)
    return _factory