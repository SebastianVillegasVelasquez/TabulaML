from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from app.core.context.context import Context
from app.core.enums import Stages
from experiments import ExperimentDefinition
from app.core.stages.data_inspection.pipeline_builder import PipelineBuilder
from app.utils.logger import logger

"""
Enhanced Model Selection Stage - Resource Optimized

This stage takes the top-k selectors from feature selection and tests them
with a comprehensive suite of models (both linear and non-linear).

The goal is to find the best MODEL × SELECTOR × HYPERPARAMETERS combination.

Model Categories:
1. Linear Models (Fast):
   - LogisticRegression
   - RidgeClassifier
   - SGDClassifier

2. Non-Linear Models (Moderate to Slow):
   - RandomForest
   - GradientBoosting
   - ExtraTrees
   - AdaBoost
   - DecisionTree
   - KNeighbors
   - SVC (if dataset is small)
   - GaussianNB

Resource Optimization:
- Reduced n_estimators for ensemble methods
- Limited max_depth to prevent overfitting
- Simplified hyperparameters (future: add grid search)
"""


def get_model_selection_experiments(context: Context):
    """
    Dynamically generate model selection experiments based on top-k selectors
    from the feature selection stage.
    """
    logger.info("Generating model selection experiments from top-k selectors...")

    fs_stage_result = context.stage_results[Stages.FEATURE_SELECTION]
    top_k_selectors = fs_stage_result.metadata.get("top_k_selectors", {})

    if not top_k_selectors:
        logger.warning("No top-k selectors found. Using all features (no selector).")
        # Fallback: use no selector
        return _generate_experiments_for_selector(selector_name="none", selector_pipeline_step=None)

    logger.info(f"Found {len(top_k_selectors)} top selectors: {list(top_k_selectors.keys())}")

    # Generate experiments for each top-k selector
    all_experiments = []
    for selector_name, selector_experiment_result in top_k_selectors.items():
        # Extract the selector step from the best pipeline
        selector_step = _extract_selector_from_pipeline(selector_experiment_result.pipeline)

        experiments = _generate_experiments_for_selector(
            selector_name=selector_name, selector_pipeline_step=selector_step
        )
        all_experiments.extend(experiments)

    logger.info(f"Generated {len(all_experiments)} model selection experiments")
    return all_experiments


def _extract_selector_from_pipeline(pipeline):
    """
    Extract the feature_selection step from a fitted pipeline.
    Returns None if no feature selection step exists.
    """
    if hasattr(pipeline, "named_steps"):
        return pipeline.named_steps.get("feature_selection", None)
    return None


def _generate_experiments_for_selector(selector_name, selector_pipeline_step):
    """
    Generate model experiments for a specific selector.
    Tests both linear and non-linear models.
    """
    experiments = []

    # Model builders
    models = {
        # Linear models
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "ridge_classifier": RidgeClassifier(random_state=42),
        "sgd_classifier": SGDClassifier(max_iter=1000, random_state=42, n_jobs=-1),
        # Non-linear models
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=150, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=10, min_samples_split=20, random_state=42
        ),
        "kneighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "gaussian_nb": GaussianNB(),
    }

    for model_name, model in models.items():
        exp_name = f"{selector_name}_{model_name}"

        def builder(preprocessing, sel=selector_pipeline_step, mod=model):
            steps = [("preprocessing", preprocessing)]

            # Add a selector if it exists
            if sel is not None:
                steps.append(("feature_selection", sel))

            steps.append(("model", mod))

            return PipelineBuilder(steps=steps)

        experiments.append(
            ExperimentDefinition(
                name=exp_name,
                stage="model_selection",
                pipeline_builder=builder,
                metadata={
                    "selector": selector_name,
                    "model": model_name,
                    "model_family": _get_model_family(model_name),
                },
            )
        )

    return experiments


def _get_model_family(model_name):
    """Categorize model into family for analysis"""
    linear_models = {"logistic_regression", "ridge_classifier", "sgd_classifier"}
    tree_models = {"random_forest", "gradient_boosting", "extra_trees", "decision_tree"}
    other_models = {"kneighbors", "gaussian_nb", "svc"}

    if model_name in linear_models:
        return "linear"
    elif model_name in tree_models:
        return "tree_based"
    else:
        return "other"
