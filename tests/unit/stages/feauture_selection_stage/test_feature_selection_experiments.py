from app.core.context import Context
from app.core.enums import ModelRetrieveType, ProblemType
from app.core.model_bank import ModelRetrieveFactory, SelectorSpec, ModelSpec
from app.core.stages.feature_selection.composer import ExperimentComposer


class TestFeatureSelectionExperiments:
    """Test suite for FeatureSelectionExperiments."""

    def test_retrieve_selectors(self, run_context):
        selectors = (
            ModelRetrieveFactory.create(
                model_retrieve_type=ModelRetrieveType.SELECTOR,
                problem_type=ProblemType.CLASSIFICATION,
            )
        ).load_defaults()

        # Is retrieving selectors
        assert len(selectors) > 0

        # It is a list
        assert isinstance(selectors, list)

        # Each of them is a SelectorSpec
        [isinstance(s, SelectorSpec) for s in selectors]

    def test_retrieve_models(self, run_context):
        models = (
            ModelRetrieveFactory.create(
                model_retrieve_type=ModelRetrieveType.BASELINE,
                problem_type=ProblemType.CLASSIFICATION,
            )
        ).load_defaults()

        # Is retrieving models
        assert len(models) > 0

        # It is a list
        assert isinstance(models, list)

        # Each of them is a ModelSpec
        [isinstance(s, ModelSpec) for s in models]


# def get_feature_selection_experiments(context: Context | None):
#     # preprocessing = context.stage_results[Stages.DATA_HANDLER].results["preprocessing"]
#
#     selectors = (ModelRetrieveFactory
#                  .create(model_retrieve_type=ModelRetrieveType.SELECTOR,
#                          problem_type=ProblemType.CLASSIFICATION)
#                  ).load_defaults()
#
#     models = (ModelRetrieveFactory
#               .create(model_retrieve_type=ModelRetrieveType.BASELINE,
#                       problem_type=ProblemType.CLASSIFICATION)
#               ).load_defaults()
#
#     composer = ExperimentComposer(selectors, models)
#
#     experiments = []
#
#     for exp in composer.generate():
#         logger.debug(f"Generated experiment: {exp}")
#         builder = exp.pipeline_builder
#         builder.steps.insert(0, ("preprocessing", preprocessing))
#
#         experiments.append(exp)
#
#     logger.debug(f"Generated {len(experiments)} feature selection experiments.")
#     logger.debug(f"Experiments: {experiments}")
#
#     return experiments
