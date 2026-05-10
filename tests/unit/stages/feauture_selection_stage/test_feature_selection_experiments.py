from app.core.enums import ModelRetrieveType, ProblemType
from app.core.model_bank import ModelRetrieveFactory, SelectorSpec, ModelSpec


class TestFeatureSelectionExperiments:
    def test_retrieve_selectors(self):
        selectors = (
            ModelRetrieveFactory.create(
                model_retrieve_type=ModelRetrieveType.SELECTOR,
                problem_type=ProblemType.CLASSIFICATION,
            )
        ).load_defaults()

        assert len(selectors) > 0
        assert isinstance(selectors, list)
        assert all(isinstance(s, SelectorSpec) for s in selectors)

    def test_retrieve_models(self):
        models = (
            ModelRetrieveFactory.create(
                model_retrieve_type=ModelRetrieveType.BASELINE,
                problem_type=ProblemType.CLASSIFICATION,
            )
        ).load_defaults()

        assert len(models) > 0
        assert isinstance(models, list)
        assert all(isinstance(s, ModelSpec) for s in models)

    def test_retrieve_selectors_regression(self):
        selectors = (
            ModelRetrieveFactory.create(
                model_retrieve_type=ModelRetrieveType.SELECTOR,
                problem_type=ProblemType.REGRESSION,
            )
        ).load_defaults()

        assert len(selectors) > 0

    def test_selector_retriever_not_none(self):
        retriever = ModelRetrieveFactory.create(
            model_retrieve_type=ModelRetrieveType.SELECTOR,
            problem_type=ProblemType.CLASSIFICATION,
        )
        assert retriever is not None

    def test_baseline_retriever_not_none(self):
        retriever = ModelRetrieveFactory.create(
            model_retrieve_type=ModelRetrieveType.BASELINE,
            problem_type=ProblemType.CLASSIFICATION,
        )
        assert retriever is not None
