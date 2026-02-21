import pandas as pd
import pytest

from app.core.context.init_context import init_context
from app.core.context.metrics import DEFAULT_METRICS
from app.core.context.problems_type import ProblemsType


@pytest.mark.parametrize("problem_type, expected_metrics",
                         [
                             (ProblemsType.CLASSIFICATION, DEFAULT_METRICS[ProblemsType.CLASSIFICATION]),
                             (ProblemsType.REGRESSION, DEFAULT_METRICS[ProblemsType.REGRESSION])
                         ])
def test_context_init_context(problem_type, expected_metrics):
    X_train = pd.DataFrame({"a": [1, 2, 3]})
    y_train = pd.Series([0, 1, 0])

    X_test = pd.DataFrame({"a": [4, 5]})
    y_test = pd.Series([1, 0])

    context = init_context(
        X=(X_train, y_train),
        y=(X_test, y_test),
        problem_type=problem_type
    )

    assert context.config.X_train.equals(X_train)
    assert context.config.y_train.equals(y_train)
    assert context.config.X_test.equals(X_test)
    assert context.config.y_test.equals(y_test)

    assert context is not None
    assert context.config.problem_type == problem_type
    assert context.config.scoring == expected_metrics

    assert hasattr(context, "update_context")

    assert context.stage_results == {}
    assert context.metadata == {}
    assert context.metadata == {}
    assert context.current_stage is None
