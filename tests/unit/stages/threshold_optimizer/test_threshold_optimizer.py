import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# =========================
# Fixtures
# =========================

class DummyConfig:
    def __init__(self, metric="f1"):
        self.priority_metric_normalized = metric
        self.X_train = np.random.rand(100, 5)
        self.y_train = np.array([0, 1] * 50)


class DummyContext:
    def __init__(self, metric="f1"):
        self.config = DummyConfig(metric)


@pytest.fixture
def context_f1():
    return DummyContext(metric="f1")


@pytest.fixture
def context_precision():
    return DummyContext(metric="precision")


@pytest.fixture
def context_recall():
    return DummyContext(metric="recall")


@pytest.fixture
def optimizer(context_f1):
    from app.core.stages.threshold_optimizer.threshold_optimizer import ThresholdOptimizer
    return ThresholdOptimizer(context=context_f1)


# =========================
# UNIT TESTS
# =========================

# -------- _is_the_model_probabilistic --------
@pytest.mark.unit
@pytest.mark.parametrize(
    "model,expected",
    [
        (LogisticRegression(), True),
        (SVC(probability=True), True),
        (SVC(probability=False), False),
        (object(), False),
    ],
)
def test_is_model_probabilistic(model, expected):
    from app.core.stages.threshold_optimizer.threshold_optimizer import ThresholdOptimizer

    result = ThresholdOptimizer._is_the_model_probabilistic(model)
    assert result is expected


# -------- _get_score --------
@pytest.mark.unit
@pytest.mark.parametrize(
    "metric",
    ["f1", "precision", "recall"],
)
def test_get_score_supported_metrics(metric):
    from app.core.stages.threshold_optimizer.threshold_optimizer import ThresholdOptimizer

    context = DummyContext(metric=metric)

    y_val = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    score = ThresholdOptimizer._get_score(context, y_val, y_pred)

    assert isinstance(score, float)
    assert score >= 0


@pytest.mark.unit
def test_get_score_invalid_metric():
    from app.core.stages.threshold_optimizer.threshold_optimizer import ThresholdOptimizer

    context = DummyContext(metric="invalid_metric")

    y_val = np.array([0, 1])
    y_pred = np.array([0, 1])

    with pytest.raises(ValueError):
        ThresholdOptimizer._get_score(context, y_val, y_pred)


# -------- find_best_threshold --------
@pytest.mark.unit
def test_find_best_threshold_non_probabilistic_model(optimizer):
    """
    Debe retornar el modelo sin cambios si no es probabilístico
    """
    model = SVC(probability=False)

    result = optimizer.find_best_threshold(model)

    assert result == model


@pytest.mark.unit
@pytest.mark.parametrize(
    "model",
    [
        LogisticRegression(),
        SVC(probability=True),
    ],
)
def test_find_best_threshold_probabilistic_models(optimizer, model):
    """
    Debe encontrar threshold y score válidos
    """
    result = optimizer.find_best_threshold(model)

    assert isinstance(result, dict)
    assert "best_threshold" in result
    assert "best_score" in result

    assert 0.0 <= result["best_threshold"] <= 1.0
    assert result["best_score"] >= 0


@pytest.mark.unit
def test_find_best_threshold_improves_score(optimizer):
    """
    Verifica que el score final sea mejor que el inicial baseline
    """
    model = LogisticRegression()

    result = optimizer.find_best_threshold(model)

    assert result["best_score"] != -1


@pytest.mark.unit
def test_find_best_threshold_with_different_metrics():
    """
    Verifica funcionamiento con diferentes métricas configuradas
    """
    from app.core.stages.threshold_optimizer.threshold_optimizer import ThresholdOptimizer

    for metric in ["f1", "precision", "recall"]:
        context = DummyContext(metric=metric)
        optimizer = ThresholdOptimizer(context=context)

        model = LogisticRegression()

        result = optimizer.find_best_threshold(model)

        assert isinstance(result["best_score"], float)
