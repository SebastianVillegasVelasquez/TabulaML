import pytest
import pandas as pd

from app.core.context import Context, DatasetBundle
from app.core.enums import ProblemType, Stages


class TestContext:
    """Unit tests for Context class."""

    def test_context_create_classification(self, dataset_bundle):
        """Test Context creation with classification problem model_based."""
        context = Context.create(
            dataset=dataset_bundle,
            problem_type=ProblemType.CLASSIFICATION,
            priority_metric="f1",
            target_column="target",
        )

        assert context.config.problem_type == ProblemType.CLASSIFICATION
        assert context.config.dataset == dataset_bundle
        assert context.metadata.target_column == "target"
        assert context.metadata.columns == ["feature1", "feature2"]
        assert context.metadata.columns_length == 2

    def test_context_create_regression(self, dataset_bundle):
        """Test Context creation with regression problem model_based."""
        context = Context.create(
            dataset=dataset_bundle,
            problem_type=ProblemType.REGRESSION,
            target_column="target",
        )

        assert context.config.problem_type == ProblemType.REGRESSION
        assert context.config.priority_metric == "test_neg_mean_squared_error"

    def test_context_dataset_access(self, dataset_bundle):
        """Test accessing datasets from context configuration."""
        context = Context.create(
            dataset=dataset_bundle,
            problem_type=ProblemType.CLASSIFICATION,
            target_column="target",
        )

        assert context.config.dataset.X_train.shape == (4, 2)
        assert context.config.dataset.y_train.shape == (4,)
        assert context.config.dataset.X_test.shape == (2, 2)
        assert context.config.dataset.y_test.shape == (2,)

    def test_context_update_stage_context(self, dataset_bundle):
        """Test updating stage results in context."""
        from app.core.context import StageResult

        context = Context.create(
            dataset=dataset_bundle,
            problem_type=ProblemType.CLASSIFICATION,
            target_column="target",
        )

        stage_result = StageResult(name=Stages.DATA_HANDLER, artifacts_path="/tmp/artifacts")
        context.update_stage_context(Stages.DATA_HANDLER, stage_result)

        assert context.current_stage == Stages.DATA_HANDLER
        assert Stages.DATA_HANDLER in context.stage_results
        assert context.stage_results[Stages.DATA_HANDLER].artifacts_path == "/tmp/artifacts"

    def test_context_priority_metric_default_classification(self, dataset_bundle):
        """Test default priority metric for classification."""
        context = Context.create(
            dataset=dataset_bundle,
            problem_type=ProblemType.CLASSIFICATION,
            target_column="target",
        )

        assert context.config.priority_metric == "test_f1"

    def test_context_priority_metric_custom(self, dataset_bundle):
        """Test custom priority metric."""
        context = Context.create(
            dataset=dataset_bundle,
            problem_type=ProblemType.CLASSIFICATION,
            priority_metric="accuracy",
            target_column="target",
        )

        assert context.config.priority_metric == "test_accuracy"
        assert context.config.priority_metric_normalized == "accuracy"

    def test_context_invalid_problem_type(self, dataset_bundle):
        """Test Context creation with invalid problem model_based raises error."""
        with pytest.raises(ValueError, match="Invalid problem model_based"):
            Context.create(
                dataset=dataset_bundle,
                problem_type="invalid_type",
                target_column="target",
            )

    def test_context_initial_stage_results_empty(self, dataset_bundle):
        """Test that context starts with empty stage results."""
        context = Context.create(
            dataset=dataset_bundle,
            problem_type=ProblemType.CLASSIFICATION,
            target_column="target",
        )

        assert len(context.stage_results) == 0


class TestDatasetBundle:
    """Unit tests for DatasetBundle model."""

    def test_dataset_bundle_creation(self, dataset_bundle):
        """Test creating a DatasetBundle from sample data."""
        assert isinstance(dataset_bundle.X_train, pd.DataFrame)
        assert isinstance(dataset_bundle.y_train, pd.Series)
        assert isinstance(dataset_bundle.X_test, pd.DataFrame)
        assert isinstance(dataset_bundle.y_test, pd.Series)

    def test_dataset_bundle_shapes(self, dataset_bundle):
        """Test DatasetBundle preserves correct shapes."""
        assert dataset_bundle.X_train.shape == (4, 2)
        assert dataset_bundle.y_train.shape == (4,)
        assert dataset_bundle.X_test.shape == (2, 2)
        assert dataset_bundle.y_test.shape == (2,)

    def test_dataset_bundle_arbitrary_types_allowed(self):
        """Test that DatasetBundle allows DataFrame and Series (arbitrary types)."""
        X_train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y_train = pd.Series([0, 1])
        X_test = pd.DataFrame({"a": [5], "b": [6]})
        y_test = pd.Series([1])

        bundle = DatasetBundle(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        assert bundle.X_train is X_train
        assert bundle.y_train is y_train
        assert bundle.X_test is X_test
        assert bundle.y_test is y_test
