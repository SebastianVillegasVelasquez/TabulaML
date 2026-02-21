from app.core.context.problems_type import ProblemsType

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