from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import ElasticNet, LogisticRegression, Lasso

from app.core.context import RunContext
from app.core.domain.experiments.experiment_definition import ExperimentDefinition
from app.core.ml.pipeline_builder import PipelineBuilder

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
# STATISTICAL SELECTORS (Fast)
# ============================================================================

def selectkbest_f_classif_linear_builder():
    """SelectKBest with F-test - fast, captures linear relationships."""
    selector = SelectKBest(score_func=f_classif, k=10)

    return PipelineBuilder(
        steps=[
            ("feature_selection", selector),
            ("model", build_base_linear_model()),
        ]
    )


def selectkbest_f_classif_nonlinear_builder():
    """SelectKBest with F-test validated with a non-linear model."""
    selector = SelectKBest(score_func=f_classif, k=10)

    return PipelineBuilder(
        steps=[
            ("feature_selection", selector),
            ("model", build_base_non_linear_model()),
        ]
    )


def selectkbest_mutual_info_linear_builder():
    """SelectKBest with mutual info - captures non-linear relationships."""
    selector = SelectKBest(score_func=mutual_info_classif, k=10)

    return PipelineBuilder(
        steps=[
            ("feature_selection", selector),
            ("model", build_base_linear_model()),
        ]
    )


def selectkbest_mutual_info_nonlinear_builder():
    """SelectKBest with mutual info validated with a non-linear model."""
    selector = SelectKBest(score_func=mutual_info_classif, k=10)

    return PipelineBuilder(
        steps=[
            ("feature_selection", selector),
            ("model", build_base_non_linear_model()),
        ]
    )


# ============================================================================
# L1-BASED SELECTORS (Moderate Speed)
# ============================================================================

def lasso_selector_linear_builder():
    """Lasso-based selection - fast L1 regularization."""
    selector = SelectFromModel(
        Lasso(alpha=0.01, max_iter=3000, random_state=42),
        threshold="median",
    )

    return PipelineBuilder(
        steps=[
            ("feature_selection", selector),
            ("model", build_base_linear_model()),
        ]
    )


def lasso_selector_nonlinear_builder():
    """Lasso-based selection validated with a non-linear model."""
    selector = SelectFromModel(
        Lasso(alpha=0.01, max_iter=3000, random_state=42),
        threshold="median",
    )

    return PipelineBuilder(
        steps=[
            ("feature_selection", selector),
            ("model", build_base_non_linear_model()),
        ]
    )


def elasticnet_selector_linear_builder():
    """ElasticNet-based selection - L1 + L2 regularization."""
    selector = SelectFromModel(
        ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=3000, random_state=42),
        threshold="median",
    )

    return PipelineBuilder(
        steps=[
            ("feature_selection", selector),
            ("model", build_base_linear_model()),
        ]
    )


def elasticnet_selector_nonlinear_builder():
    """ElasticNet-based selection validated with a non-linear model."""
    selector = SelectFromModel(
        ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=3000, random_state=42),
        threshold="median",
    )

    return PipelineBuilder(
        steps=[
            ("feature_selection", selector),
            ("model", build_base_non_linear_model()),
        ]
    )


# ============================================================================
# TREE-BASED SELECTORS (Moderate to Slow)
# ============================================================================

def extratrees_selector_linear_builder():
    """ExtraTrees-based selection - captures non-linear feature importance."""
    selector = SelectFromModel(
        ExtraTreesClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        threshold="median",
    )

    return PipelineBuilder(
        steps=[
            ("feature_selection", selector),
            ("model", build_base_linear_model()),
        ]
    )


def extratrees_selector_nonlinear_builder():
    """ExtraTrees-based selection validated with a non-linear model."""
    selector = SelectFromModel(
        ExtraTreesClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        threshold="median",
    )

    return PipelineBuilder(
        steps=[
            ("feature_selection", selector),
            ("model", build_base_non_linear_model()),
        ]
    )


# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

def get_feature_selection_experiments(context: RunContext) -> list[ExperimentDefinition]:
    from app.core.context.stages import Stages

    preprocessing = context.stage_results[Stages.DATA_HANDLER].results["preprocessing"]

    results = []

    for definition in FEATURE_SELECTION_EXPERIMENTS:

        builder: PipelineBuilder = definition.pipeline_builder()

        builder.steps.insert(0, ("preprocessing", preprocessing))

        results.append(
            ExperimentDefinition(
                name=definition.name,
                stage=definition.stage,
                pipeline_builder=builder,
            )
        )

    return results


FEATURE_SELECTION_EXPERIMENTS = [
    # Baseline experiments (no selection)
    ExperimentDefinition(
        name="no_selector_linear",
        stage="feature_selection",
        pipeline_builder=no_selector_linear_builder,
        metadata={"selector": "none", "selector_type": "baseline", "validator": "linear"}
    ),
    ExperimentDefinition(
        name="no_selector_nonlinear",
        stage="feature_selection",
        pipeline_builder=no_selector_nonlinear_builder,
        metadata={"selector": "none", "selector_type": "baseline", "validator": "nonlinear"}
    ),

    # Statistical selectors (Fast - ~1-2s)
    ExperimentDefinition(
        name="selectkbest_fclassif_linear",
        stage="feature_selection",
        pipeline_builder=selectkbest_f_classif_linear_builder,
        metadata={"selector": "selectkbest_f", "selector_type": "statistical", "validator": "linear"}
    ),
    ExperimentDefinition(
        name="selectkbest_fclassif_nonlinear",
        stage="feature_selection",
        pipeline_builder=selectkbest_f_classif_nonlinear_builder,
        metadata={"selector": "selectkbest_f", "selector_type": "statistical", "validator": "nonlinear"}
    ),
    ExperimentDefinition(
        name="selectkbest_mutual_info_linear",
        stage="feature_selection",
        pipeline_builder=selectkbest_mutual_info_linear_builder,
        metadata={"selector": "selectkbest_mi", "selector_type": "statistical", "validator": "linear"}
    ),
    ExperimentDefinition(
        name="selectkbest_mutual_info_nonlinear",
        stage="feature_selection",
        pipeline_builder=selectkbest_mutual_info_nonlinear_builder,
        metadata={"selector": "selectkbest_mi", "selector_type": "statistical", "validator": "nonlinear"}
    ),

    # L1-based selectors (Moderate - ~5-10s)
    ExperimentDefinition(
        name="lasso_selector_linear",
        stage="feature_selection",
        pipeline_builder=lasso_selector_linear_builder,
        metadata={"selector": "lasso", "selector_type": "l1_based", "validator": "linear"}
    ),
    ExperimentDefinition(
        name="lasso_selector_nonlinear",
        stage="feature_selection",
        pipeline_builder=lasso_selector_nonlinear_builder,
        metadata={"selector": "lasso", "selector_type": "l1_based", "validator": "nonlinear"}
    ),
    ExperimentDefinition(
        name="elasticnet_selector_linear",
        stage="feature_selection",
        pipeline_builder=elasticnet_selector_linear_builder,
        metadata={"selector": "elasticnet", "selector_type": "l1_based", "validator": "linear"}
    ),
    ExperimentDefinition(
        name="elasticnet_selector_nonlinear",
        stage="feature_selection",
        pipeline_builder=elasticnet_selector_nonlinear_builder,
        metadata={"selector": "elasticnet", "selector_type": "l1_based", "validator": "nonlinear"}
    ),

    # Tree-based selectors (Moderate - ~10-20s)
    ExperimentDefinition(
        name="extratrees_selector_linear",
        stage="feature_selection",
        pipeline_builder=extratrees_selector_linear_builder,
        metadata={"selector": "extratrees", "selector_type": "tree_based", "validator": "linear"}
    ),
    ExperimentDefinition(
        name="extratrees_selector_nonlinear",
        stage="feature_selection",
        pipeline_builder=extratrees_selector_nonlinear_builder,
        metadata={"selector": "extratrees", "selector_type": "tree_based", "validator": "nonlinear"}
    ),

]
