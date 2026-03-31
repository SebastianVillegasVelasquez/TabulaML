import pandas as pd
import pytest

from app.core.context import init_context, RunContext, ProjectConfig
from app.core.context.init_context import _decouple_tuples, _get_priority_metric, _get_metadata
from app.core.enums import ProblemsType
from tests.conftest import run_context_params


class TestRunContext:
    """Test suite for RunContext."""


    class TestInitContext:

        @pytest.mark.unit
        def test_invalid_problem_type(self, sample_data):
            X, y = sample_data

            with pytest.raises(ValueError, match="Invalid problem type"):
                init_context("INVALID", X, y)

        @pytest.mark.unit
        def test_context_instance(self, run_context_params):
            assert isinstance(run_context_params, RunContext)

        @pytest.mark.unit
        def test_config_instance(self, run_context_params):
            assert isinstance(run_context_params.config, ProjectConfig)

        @pytest.mark.unit
        def test_problem_type_assignment(self, run_context_params):
            assert run_context_params.config.problem_type in [
                ProblemsType.CLASSIFICATION,
                ProblemsType.REGRESSION,
            ]

        @pytest.mark.unit
        def test_data_integrity(self, sample_data):
            X, y = sample_data
            context = init_context(ProblemsType.CLASSIFICATION, X, y)

            assert context.config.X_train.equals(X[0])
            assert context.config.y_train.equals(X[1])
            assert context.config.X_test.equals(y[0])
            assert context.config.y_test.equals(y[1])

        @pytest.mark.unit
        def test_default_random_state(self, run_context_params):
            assert run_context_params.config.random_state == 42

        @pytest.mark.unit
        def test_default_priority_metric_classification(self, sample_data):
            X, y = sample_data
            context = init_context(ProblemsType.CLASSIFICATION, X, y)

            assert context.config.priority_metric == "test_f1"

        @pytest.mark.unit
        def test_default_priority_metric_regression(self, sample_data):
            X, y = sample_data
            context = init_context(ProblemsType.REGRESSION, X, y)

            assert context.config.priority_metric == "test_neg_mean_squared_error"

        @pytest.mark.unit
        def test_custom_priority_metric(self, sample_data):
            X, y = sample_data
            context = init_context(
                ProblemsType.CLASSIFICATION,
                X,
                y,
                priority_metric="accuracy"
            )

            assert context.config.priority_metric == "test_accuracy"

        @pytest.mark.unit
        def test_custom_priority_metric_normalized(self, sample_data):
            X, y = sample_data
            context = init_context(
                ProblemsType.CLASSIFICATION,
                X,
                y,
                priority_metric="accuracy"
            )

            assert context.config.priority_metric_normalized == "accuracy"

        @pytest.mark.unit
        def test_metadata_generation(self, sample_data):
            X, y = sample_data
            context = init_context(ProblemsType.CLASSIFICATION, X, y)

            metadata = context.metadata

            assert "total_columns" in metadata
            assert "total_rows" in metadata
            assert "original_shape" in metadata

        @pytest.mark.unit
        def test_metadata_values(self, sample_data):
            X, y = sample_data
            context = init_context(ProblemsType.CLASSIFICATION, X, y)

            X_train = X[0]
            metadata = context.metadata

            assert metadata["total_columns"] == len(X_train.columns)
            assert metadata["total_rows"] == len(X_train)

        @pytest.mark.unit
        def test_empty_stage_results(self, run_context_params):
            assert run_context_params.stage_results == {}

        @pytest.mark.unit
        def test_current_stage_is_none(self, run_context_params):
            assert run_context_params.current_stage is None


class TestHelpers:

    @pytest.mark.unit
    def test_decouple_tuples(self, sample_data):
        X, y = sample_data

        X_train, y_train, X_test, y_test = _decouple_tuples(X, y)

        assert X_train.equals(X[0])
        assert y_train.equals(X[1])
        assert X_test.equals(y[0])
        assert y_test.equals(y[1])

    @pytest.mark.unit
    def test_get_priority_metric_custom(self):
        result = _get_priority_metric(ProblemsType.CLASSIFICATION, "accuracy")
        assert result == "test_accuracy"

    @pytest.mark.unit
    def test_get_priority_metric_default_classification(self):
        result = _get_priority_metric(ProblemsType.CLASSIFICATION)
        assert result == "test_f1"

    @pytest.mark.unit
    def test_get_priority_metric_default_regression(self):
        result = _get_priority_metric(ProblemsType.REGRESSION)
        assert result == "test_neg_mean_squared_error"

    @pytest.mark.unit
    def test_get_metadata(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        metadata = _get_metadata(df)

        assert metadata["total_columns"] == 2
        assert metadata["total_rows"] == 2

    @pytest.mark.unit
    def test_none_inputs(self):
        with pytest.raises(ValueError):
            init_context(ProblemsType.CLASSIFICATION, None, None)

    @pytest.mark.unit
    def test_invalid_tuple_structure(self):
        X = (pd.DataFrame(),)
        y = (pd.Series(),)

        with pytest.raises(ValueError):
            init_context(ProblemsType.CLASSIFICATION, X, y)
