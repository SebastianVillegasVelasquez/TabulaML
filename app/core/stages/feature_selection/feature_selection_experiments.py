from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.linear_model import ElasticNet, LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier

from app.core.domain.experiments.experiment_definition import ExperimentDefinition
from app.core.ml.pipeline_builder import PipelineBuilder

"""
Enhanced Feature Selection Stage - Resource Optimized

The purpose of this stage is to identify the BEST SELECTOR, not the best final model.
We use lightweight predictors to validate different feature selection approaches.

Feature Selection Approaches:
1. No selection (baseline - all features)
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
# VALIDATION MODELS (Lightweight for Feature Selection)
# ============================================================================

def build_base_linear_model():
    """Lightweight linear model for feature validation"""
    return LogisticRegression(
        max_iter=1000,  # Reduced from 5000
        C=1.0,
        solver="lbfgs",
        random_state=42
    )


def build_base_non_linear_model():
    """Lightweight non-linear model for feature validation"""
    return RandomForestClassifier(
        n_estimators=100,  # Reduced from 200 for speed
        max_depth=10,      # Limit depth to reduce overfitting and computation
        min_samples_split=10,  # Prevent deep trees
        random_state=42,
        n_jobs=-1
    )


# ============================================================================
# BASELINE - No Feature Selection
# ============================================================================

def no_selector_linear_builder(preprocessing):
    """Baseline: all features with linear model"""
    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("model", build_base_linear_model())
        ]
    )


def no_selector_nonlinear_builder(preprocessing):
    """Baseline: all features with non-linear model"""
    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("model", build_base_non_linear_model())
        ]
    )


# ============================================================================
# STATISTICAL SELECTORS (Fast)
# ============================================================================

def selectkbest_f_classif_linear_builder(preprocessing):
    """SelectKBest with F-test - fast, captures linear relationships"""
    selector = SelectKBest(score_func=f_classif, k=10)  # Keep top 10 features

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", build_base_linear_model())
        ]
    )


def selectkbest_f_classif_nonlinear_builder(preprocessing):
    """SelectKBest with F-test validated with non-linear model"""
    selector = SelectKBest(score_func=f_classif, k=10)

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", build_base_non_linear_model())
        ]
    )


def selectkbest_mutual_info_linear_builder(preprocessing):
    """SelectKBest with mutual info - captures non-linear relationships"""
    selector = SelectKBest(score_func=mutual_info_classif, k=10)

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", build_base_linear_model())
        ]
    )


def selectkbest_mutual_info_nonlinear_builder(preprocessing):
    """SelectKBest with mutual info validated with non-linear model"""
    selector = SelectKBest(score_func=mutual_info_classif, k=10)

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", build_base_non_linear_model())
        ]
    )


# ============================================================================
# L1-BASED SELECTORS (Moderate Speed)
# ============================================================================

def lasso_selector_linear_builder(preprocessing):
    """Lasso-based selection - fast L1 regularization"""
    selector = SelectFromModel(
        Lasso(alpha=0.01, max_iter=1000, random_state=42),
        threshold="median"
    )

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", build_base_linear_model())
        ]
    )


def lasso_selector_nonlinear_builder(preprocessing):
    """Lasso-based selection validated with non-linear model"""
    selector = SelectFromModel(
        Lasso(alpha=0.01, max_iter=1000, random_state=42),
        threshold="median"
    )

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", build_base_non_linear_model())
        ]
    )


def elasticnet_selector_linear_builder(preprocessing):
    """ElasticNet-based selection - L1 + L2 regularization"""
    selector = SelectFromModel(
        ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000, random_state=42),
        threshold="median"
    )

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", build_base_linear_model())
        ]
    )


def elasticnet_selector_nonlinear_builder(preprocessing):
    """ElasticNet-based selection validated with non-linear model"""
    selector = SelectFromModel(
        ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000, random_state=42),
        threshold="median"
    )

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", build_base_non_linear_model())
        ]
    )


# ============================================================================
# TREE-BASED SELECTORS (Moderate to Slow)
# ============================================================================

def extratrees_selector_linear_builder(preprocessing):
    """ExtraTrees-based selection - captures non-linear feature importance"""
    selector = SelectFromModel(
        ExtraTreesClassifier(
            n_estimators=50,  # Reduced from 200
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        threshold="median"
    )

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", build_base_linear_model())
        ]
    )


def extratrees_selector_nonlinear_builder(preprocessing):
    """ExtraTrees-based selection validated with non-linear model"""
    selector = SelectFromModel(
        ExtraTreesClassifier(
            n_estimators=50,  # Reduced from 200
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        threshold="median"
    )

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", build_base_non_linear_model())
        ]
    )


# ============================================================================
# RFE-BASED SELECTORS (Slowest but Thorough)
# ============================================================================

def rfe_linear_builder(preprocessing):
    """RFE with linear estimator - recursive elimination"""
    selector = RFE(
        estimator=LogisticRegression(max_iter=500, random_state=42),
        n_features_to_select=10,
        step=0.2  # Remove 20% of features at each iteration
    )

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", build_base_linear_model())
        ]
    )


def rfe_nonlinear_builder(preprocessing):
    """RFE with tree estimator validated with non-linear model"""
    selector = RFE(
        estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
        n_features_to_select=10,
        step=0.2
    )

    return PipelineBuilder(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            ("model", build_base_non_linear_model())
        ]
    )


# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

FEATURE_SELECTION_EXPERIMENTS = [
    # Baseline experiments (no selection)
    ExperimentDefinition(
        name="no_selector_linear",
        stage="feature_selection",
        builder=no_selector_linear_builder,
        metadata={"selector": "none", "selector_type": "baseline", "validator": "linear"}
    ),
    ExperimentDefinition(
        name="no_selector_nonlinear",
        stage="feature_selection",
        builder=no_selector_nonlinear_builder,
        metadata={"selector": "none", "selector_type": "baseline", "validator": "nonlinear"}
    ),

    # Statistical selectors (Fast - ~1-2s)
    ExperimentDefinition(
        name="selectkbest_fclassif_linear",
        stage="feature_selection",
        builder=selectkbest_f_classif_linear_builder,
        metadata={"selector": "selectkbest_f", "selector_type": "statistical", "validator": "linear"}
    ),
    ExperimentDefinition(
        name="selectkbest_fclassif_nonlinear",
        stage="feature_selection",
        builder=selectkbest_f_classif_nonlinear_builder,
        metadata={"selector": "selectkbest_f", "selector_type": "statistical", "validator": "nonlinear"}
    ),
    ExperimentDefinition(
        name="selectkbest_mutual_info_linear",
        stage="feature_selection",
        builder=selectkbest_mutual_info_linear_builder,
        metadata={"selector": "selectkbest_mi", "selector_type": "statistical", "validator": "linear"}
    ),
    ExperimentDefinition(
        name="selectkbest_mutual_info_nonlinear",
        stage="feature_selection",
        builder=selectkbest_mutual_info_nonlinear_builder,
        metadata={"selector": "selectkbest_mi", "selector_type": "statistical", "validator": "nonlinear"}
    ),

    # L1-based selectors (Moderate - ~5-10s)
    ExperimentDefinition(
        name="lasso_selector_linear",
        stage="feature_selection",
        builder=lasso_selector_linear_builder,
        metadata={"selector": "lasso", "selector_type": "l1_based", "validator": "linear"}
    ),
    ExperimentDefinition(
        name="lasso_selector_nonlinear",
        stage="feature_selection",
        builder=lasso_selector_nonlinear_builder,
        metadata={"selector": "lasso", "selector_type": "l1_based", "validator": "nonlinear"}
    ),
    ExperimentDefinition(
        name="elasticnet_selector_linear",
        stage="feature_selection",
        builder=elasticnet_selector_linear_builder,
        metadata={"selector": "elasticnet", "selector_type": "l1_based", "validator": "linear"}
    ),
    ExperimentDefinition(
        name="elasticnet_selector_nonlinear",
        stage="feature_selection",
        builder=elasticnet_selector_nonlinear_builder,
        metadata={"selector": "elasticnet", "selector_type": "l1_based", "validator": "nonlinear"}
    ),

    # Tree-based selectors (Moderate - ~10-20s)
    ExperimentDefinition(
        name="extratrees_selector_linear",
        stage="feature_selection",
        builder=extratrees_selector_linear_builder,
        metadata={"selector": "extratrees", "selector_type": "tree_based", "validator": "linear"}
    ),
    ExperimentDefinition(
        name="extratrees_selector_nonlinear",
        stage="feature_selection",
        builder=extratrees_selector_nonlinear_builder,
        metadata={"selector": "extratrees", "selector_type": "tree_based", "validator": "nonlinear"}
    ),

    # RFE selectors (Slow - ~30-60s) - Optional, can comment out for speed
    ExperimentDefinition(
        name="rfe_linear",
        stage="feature_selection",
        builder=rfe_linear_builder,
        metadata={"selector": "rfe_linear", "selector_type": "wrapper", "validator": "linear"}
    ),
    ExperimentDefinition(
        name="rfe_nonlinear",
        stage="feature_selection",
        builder=rfe_nonlinear_builder,
        metadata={"selector": "rfe_tree", "selector_type": "wrapper", "validator": "nonlinear"}
    ),
]
