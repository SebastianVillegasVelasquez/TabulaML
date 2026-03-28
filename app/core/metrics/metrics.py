from app.core.enums.problems_type import ProblemsType

DEFAULT_METRICS = {
    ProblemsType.CLASSIFICATION: [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ],
    ProblemsType.REGRESSION: [
        "neg_mean_squared_error",
        "neg_mean_absolute_error",
        "r2"
    ]
}

PRIMARY_METRICS = {
    ProblemsType.CLASSIFICATION: "f1",
    ProblemsType.REGRESSION: "neg_mean_squared_error"
}

def get_primary_metric(problem_type):
    return PRIMARY_METRICS[problem_type]

def get_default_metrics(problem_type):
    return DEFAULT_METRICS[problem_type]