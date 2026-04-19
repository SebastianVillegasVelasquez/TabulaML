from app.core.context import RunContext
from app.core.enums import Stages, ModelRetrieveType, ProblemsType
from app.core.model_bank import ModelRetrieveFactory
from app.core.stages.feature_selection.composer import ExperimentComposer
from app.utils.logger import logger

"""
Feature Selection Stage

The purpose of this stage is to identify the BEST SELECTOR, not the best final model.
We use lightweight predictors to validate different feature selection approaches.

Feature Selection Approaches:
2. Statistical: SelectKBest with f_classif (fast, linear relationships)
3. Statistical: SelectKBest with mutual_info (moderate, non-linear relationships)
4. L1-based: Lasso (fast, linear sparsity)
5. L1-based: ElasticNet (moderate, linear with L2 regularization)
6. Tree-based: ExtraTrees (moderate, non-linear importance)
7. RFE: Recursive Feature Elimination (slow but thorough)

Each selector is validated with both linear and non-linear predictors to ensure
the selected features generalize across model families.

Resource Optimization:
- Reduced n_estimators for tree-based methods
- Simplified hyperparameters
- Fast solvers where possible
"""


# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================
def get_feature_selection_experiments(context: RunContext | None):
    preprocessing = context.stage_results[Stages.DATA_HANDLER].results["preprocessing"]

    selectors = (ModelRetrieveFactory
                 .create(model_retrieve_type=ModelRetrieveType.SELECTOR,
                         problem_type=ProblemsType.CLASSIFICATION)
                 ).load_defaults()

    models = (ModelRetrieveFactory
              .create(model_retrieve_type=ModelRetrieveType.BASELINE,
                      problem_type=ProblemsType.CLASSIFICATION)
              ).load_defaults()

    composer = ExperimentComposer(selectors, models)

    experiments = []

    for exp in composer.generate():
        builder = exp.pipeline_builder
        builder.steps.insert(0, ("preprocessing", preprocessing))

        experiments.append(exp)

    logger.debug(f"Generated {len(experiments)} feature selection experiments.")
    logger.debug(f"Experiments: {experiments}")

    return experiments
