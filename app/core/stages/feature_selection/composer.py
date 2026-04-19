from dataclasses import dataclass
from enum import Enum
from typing import Callable

from app.core.enums import SelectorSpecInfo, ModelSpecType
from app.core.model_bank.model_spects import SelectorSpec, ModelSpec


class ExperimentPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BLOCKED = "blocked"


@dataclass
class CompositionRule:
    """
    A rule that evaluates a (selector, predictor) pair and returns
    a priority or blocks the combination entirely.

    Rules are evaluated in order; the first matching rule wins.
    """
    name: str
    description: str
    match: Callable[[SelectorSpec, ModelSpec], bool]
    priority: ExperimentPriority


_RULES: list[CompositionRule] = [
    CompositionRule(
        name="tree_on_tree",
        description=(
            "Blocks tree-based selectors paired with tree predictors. "
            "Both share the same inductive bias (splitting criterion), creating "
            "circular feature selection — the selected features are already biased "
            "toward what a tree finds useful, so diversity and independence are lost."
        ),
        match=lambda s, m: (
                s.type == SelectorSpecInfo.TREE_BASED
                and m.type == ModelSpecType.TREE
        ),
        priority=ExperimentPriority.BLOCKED,
    ),
    CompositionRule(
        name="statistical_linear",
        description=(
            "High priority for statistical selectors paired with linear predictors. "
            "SelectKBest measures univariate linear correlation; a linear predictor "
            "exploits exactly that signal. Selector and predictor assumptions align."
        ),
        match=lambda s, m: (
                s.type == SelectorSpecInfo.STATISTICAL
                and m.spec_type == ModelSpecType.LINEAR
        ),
        priority=ExperimentPriority.HIGH,
    ),
    CompositionRule(
        name="tree_based_linear",
        description=(
            "High priority cross-paradigm combination: tree-based selector captures "
            "global nonlinear feature importance, then a linear predictor models the "
            "reduced high-signal space. Different inductive biases eliminate circular "
            "reasoning — the strongest cross-paradigm pairing."
        ),
        match=lambda s, m: (
                s.type == SelectorSpecInfo.TREE_BASED
                and m.spec_type == ModelSpecType.LINEAR
        ),
        priority=ExperimentPriority.HIGH,
    ),
    CompositionRule(
        name="l1_linear",
        description=(
            "Medium priority: L1/L1+L2 selectors with linear predictors share the "
            "same linear paradigm. The selector performs hard sparsity (zeroing "
            "coefficients); the predictor then models the remaining features linearly. "
            "Partially redundant with predictor regularization but valid for explicit "
            "feature exclusion."
        ),
        match=lambda s, m: (
                s.type in {SelectorSpecInfo.L1, SelectorSpecInfo.L1_L2}
                and m.spec_type == ModelSpecType.LINEAR
        ),
        priority=ExperimentPriority.MEDIUM,
    ),
    CompositionRule(
        name="statistical_tree",
        description=(
            "Medium priority: statistical selectors impose no inductive bias, so any "
            "predictor can follow. However, univariate statistics miss feature "
            "interactions that tree predictors are designed to exploit — the tree "
            "may receive a suboptimal feature set."
        ),
        match=lambda s, m: (
                s.type == SelectorSpecInfo.STATISTICAL
                and m.type == ModelSpecType.TREE
        ),
        priority=ExperimentPriority.MEDIUM,
    ),
    CompositionRule(
        name="l1_tree",
        description=(
            "Low priority: L1/L1+L2 selection assumes linear relevance; a tree "
            "predictor then operates on features chosen for linear signal. Features "
            "weak individually but important in nonlinear combinations may be "
            "discarded. Exploratory value for coverage, but inductive bias mismatch."
        ),
        match=lambda s, m: (
                s.type in {SelectorSpecInfo.L1, SelectorSpecInfo.L1_L2}
                and m.type == ModelSpecType.TREE
        ),
        priority=ExperimentPriority.LOW,
    ),
]

_PRIORITY_ORDER = {
    ExperimentPriority.HIGH: 0,
    ExperimentPriority.MEDIUM: 1,
    ExperimentPriority.LOW: 2,
}


def _evaluate_pair(selector: SelectorSpec, model: ModelSpec) -> ExperimentPriority:
    for rule in _RULES:
        if rule.match(selector, model):
            return rule.priority
    return ExperimentPriority.MEDIUM


class ExperimentComposer:
    def __init__(self, selectors: list[SelectorSpec], models: list[ModelSpec]):
        self.selectors = selectors
        self.models = models

    def generate(self):
        experiments = []

        for selector in self.selectors:
            for model in self.models:
                priority = _evaluate_pair(selector, model)

                if priority == ExperimentPriority.BLOCKED:
                    continue

                experiment = self._build_experiment(selector, model, priority)
                experiments.append((priority, experiment))

        experiments.sort(key=lambda pair: _PRIORITY_ORDER[pair[0]])
        return [experiment for _, experiment in experiments]

    @staticmethod
    def _build_experiment(selector: SelectorSpec, model: ModelSpec, priority: ExperimentPriority):
        from app.core.ml import PipelineBuilder
        from app.core.domain.experiments import ExperimentDefinition

        builder = PipelineBuilder()

        builder.steps = [
            ("feature_selection", selector.factory()),
            ("model", model.factory()),
        ]

        return ExperimentDefinition(
            name=f"{selector.name}__{model.name}",
            stage="feature_selection",
            pipeline_builder=builder,
            metadata={
                "priority": priority.value,
                "selector_type": selector.type.value,
                "model_type": model.type.value,
            }
        )
