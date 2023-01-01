from dataclasses import dataclass
from enum import Enum
from typing import Callable

from app.core.context import Context
from experiments import ExperimentDefinition
from app.core.enums import SelectorSpecType, ModelSpecType
from app.core.stages.data_inspection.pipeline_builder import PipelineBuilder
from app.core.model_bank.model_spects import SelectorSpec, ModelSpec
from app.utils.logger import logger


class ExperimentPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BLOCKED = "blocked"


@dataclass
class SelectorChain:
    selectors: list[SelectorSpec]
    name: str
    type: str


@dataclass
class CompositionRule:
    """
    A rule that evaluates a (selector, predictor) pair and returns
    a priority or blocks the combination entirely.

    Rules are evaluated in order; the first matching rule wins.
    """

    name: str
    description: str
    match: Callable[[SelectorChain, ModelSpec], bool]
    priority: ExperimentPriority


class SelectorChainFactory:

    def __init__(self, selectors: list[SelectorSpec]):
        self.selectors = selectors

    def build(self):

        chains = []

        filters, embedded, rfe, shap = self._decoupled_selectors(selectors=self.selectors)

        # Collect all selectors from the list of selectors,
        # then filter each group of selectors based on the type attribute.
        for s in self.selectors:
            if s.type in {SelectorSpecType.RFE, SelectorSpecType.SHAP}:
                continue
            chains.append(SelectorChain([s], name=s.name, type="simple"))

        # This filter is used to combine a quick filter as SelectKBest
        # with a more complex embedded selector such as SelectFromModel.
        # Such models as RandomForest or Linear models
        for f in filters:
            for e in embedded:
                chains.append(
                    SelectorChain([f, e], name=f"{f.name}__{e.name}", type="filter_embedded")
                )
            for r in rfe:
                chains.append(SelectorChain([f, r], name=f"{f.name}__{r.name}", type="rfe"))

        # These filters combine a quick filter first such as SelectKBest
        # with a more complex RFE selector such as RFECV.
        # for f in filters:
        #     for r in rfe:
        #         chains.append(SelectorChain([f, r], name=f"{f.name}__{r.name}", type="rfe"))

        # These filters combine a quick filter first such as SelectKBest
        # with a more complex RFE and a final SHAP selector.
        # This is a combination for a heavy-duty task
        for f in filters:
            for s in shap:
                chains.append(SelectorChain([f, s], name=f"{f.name}__{s.name}", type="filter_shap"))
        return chains

    @staticmethod
    def _decoupled_selectors(selectors: list[SelectorSpec]) -> tuple:
        """
        This method helps to decouple the selectors into 4 groups:
        - Filters
        - Embedded
        - Rfe
        - Sha

        All the selectors are being grouped using the type attribute.

        Args:
            - selectors: List of selectors

        Returns:
            tuple: A tuple containing the decoupled selectors
        """
        filters = [s for s in selectors if s.type == SelectorSpecType.STATISTICAL]

        embedded = [
            s
            for s in selectors
            if s.type in {SelectorSpecType.TREE_BASED, SelectorSpecType.L1, SelectorSpecType.L1_L2}
        ]

        rfe = [s for s in selectors if s.type == SelectorSpecType.RFE]
        shap = [s for s in selectors if s.type == SelectorSpecType.SHAP]

        return filters, embedded, rfe, shap


_RULES: list[CompositionRule] = [
    # Tree heavy pipeline approach
    CompositionRule(
        name="tree_heavy_pipeline",
        description=(
            "Penalizes pipelines where tree-based selectors are followed by a tree model. "
            "Although valid, this creates strong inductive bias overlap and may reduce diversity."
        ),
        match=lambda chain, model: (
            any(s.type == SelectorSpecType.TREE_BASED for s in chain.selectors)
            and model.type == ModelSpecType.TREE
        ),
        priority=ExperimentPriority.BLOCKED,
    ),
    CompositionRule(
        name="linear_redundancy_block",
        description="Blocks pipelines where a linear RFE/Selector is followed by a linear model.",
        match=lambda chain, model: (
            chain.selectors[-1].spec_type == ModelSpecType.LINEAR
            and model.spec_type == ModelSpecType.LINEAR
        ),
        priority=ExperimentPriority.BLOCKED,
    ),
    CompositionRule(
        name="tree_on_tree_block",
        description="Blocks pipelines where a tree-based selector is followed by a tree model.",
        match=lambda chain, model: (
            any(s.type == SelectorSpecType.TREE_BASED for s in chain.selectors)
            and model.type == ModelSpecType.TREE
        ),
        priority=ExperimentPriority.BLOCKED,
    ),
    # Multi stage pipeline approach
    CompositionRule(
        name="multi_stage_bonus",
        description=(
            "Multi-stage feature selection pipelines (e.g. filter + embedded) are more robust, "
            "as they progressively refine the feature space."
        ),
        match=lambda chain, model: len(chain.selectors) > 1,
        priority=ExperimentPriority.HIGH,
    ),
    # Statistical based on linear approach
    CompositionRule(
        name="statistical_to_linear",
        description=(
            "Statistical selectors (e.g. SelectKBest) align perfectly with linear models, "
            "since both rely on linear relationships."
        ),
        match=lambda chain, model: (
            chain.selectors[-1].type == SelectorSpecType.STATISTICAL
            and model.spec_type == ModelSpecType.LINEAR
        ),
        priority=ExperimentPriority.HIGH,
    ),
    # Tree based on Linear approach
    CompositionRule(
        name="tree_to_linear",
        description=(
            "Tree-based selection captures nonlinear interactions, then linear models exploit "
            "a cleaner feature space. Strong cross-paradigm synergy."
        ),
        match=lambda chain, model: (
            any(s.type == SelectorSpecType.TREE_BASED for s in chain.selectors)
            and model.spec_type == ModelSpecType.LINEAR
        ),
        priority=ExperimentPriority.HIGH,
    ),
    # Regularization based on Linear approach
    CompositionRule(
        name="l1_to_linear",
        description=(
            "L1-based selection combined with linear models introduces redundancy, since both "
            "perform sparsity. Still valid but not optimal."
        ),
        match=lambda chain, model: (
            any(s.type in {SelectorSpecType.L1, SelectorSpecType.L1_L2} for s in chain.selectors)
            and model.spec_type == ModelSpecType.LINEAR
        ),
        priority=ExperimentPriority.BLOCKED,
    ),
    # Statistical based on Tree based approach
    CompositionRule(
        name="statistical_to_tree",
        description=(
            "Statistical selectors ignore feature interactions, which tree models depend on. "
            "Useful as a pre-filter but not ideal alone."
        ),
        match=lambda chain, model: (
            chain.selectors[0].type == SelectorSpecType.STATISTICAL
            and model.type == ModelSpecType.TREE
        ),
        priority=ExperimentPriority.MEDIUM,
    ),
    # Regularization based on Tree based approach
    CompositionRule(
        name="l1_to_tree",
        description=(
            "L1-based selection assumes linear importance, which may remove features useful "
            "for nonlinear tree-based models."
        ),
        match=lambda chain, model: (
            any(s.type in {SelectorSpecType.L1, SelectorSpecType.L1_L2} for s in chain.selectors)
            and model.type == ModelSpecType.TREE
        ),
        priority=ExperimentPriority.LOW,
    ),
    # Filter based + embedded approach
    CompositionRule(
        name="filter_then_embedded",
        description=(
            "Fast statistical filtering followed by embedded selection is a strong and efficient "
            "feature selection strategy."
        ),
        match=lambda chain, model: (
            len(chain.selectors) >= 2
            and chain.selectors[0].type == SelectorSpecType.STATISTICAL
            and chain.selectors[1].type
            in {
                SelectorSpecType.TREE_BASED,
                SelectorSpecType.L1,
                SelectorSpecType.L1_L2,
            }
        ),
        priority=ExperimentPriority.HIGH,
    ),
    CompositionRule(
        name="filter_then_rfe",
        description=(
            "Fast statistical filtering followed by RFE is a strong and efficient feature selection strategy."
        ),
        match=lambda chain, model: (
            len(chain.selectors) >= 2
            and chain.selectors[0].type == SelectorSpecType.STATISTICAL
            and chain.selectors[1].type == SelectorSpecType.RFE
        ),
        priority=ExperimentPriority.HIGH,
    ),
    # SHAP approach
    CompositionRule(
        name="shap_bonus",
        description=(
            "SHAP-based feature selection provides high-quality global importance estimates."
        ),
        match=lambda chain, model: any(
            s.type == SelectorSpecType.EXPLAINABLE for s in chain.selectors
        ),
        priority=ExperimentPriority.HIGH,
    ),
    CompositionRule(
        name="cross_paradigm_synergy",
        description="Combines multiple selectors and models to create a powerful feature selection pipeline.",
        match=lambda chain, model: (
            chain.selectors[-1].spec_type == ModelSpecType.NON_LINEAR
            and model.spec_type == ModelSpecType.LINEAR
        ),
        priority=ExperimentPriority.HIGH,
    ),
    CompositionRule(
        name="efficient_multistage",
        description="Combines multiple selectors and models to create a powerful feature selection pipeline.",
        match=lambda chain, model: (
            len(chain.selectors) >= 2 and chain.selectors[0].type == SelectorSpecType.STATISTICAL
        ),
        priority=ExperimentPriority.HIGH,
    ),
    CompositionRule(
        name="non_linear_redundancy_penalty",
        description="Low priority for non-linear selectors followed by non-linear models.",
        match=lambda chain, model: (
            chain.selectors[-1].spec_type == ModelSpecType.NON_LINEAR
            and model.spec_type == ModelSpecType.NON_LINEAR
        ),
        priority=ExperimentPriority.BLOCKED,
    ),
]

_PRIORITY_ORDER = {
    ExperimentPriority.HIGH: 0,
    ExperimentPriority.MEDIUM: 1,
    ExperimentPriority.LOW: 2,
    ExperimentPriority.BLOCKED: 99,
}


class ExperimentComposer:
    """
    Composes feature selection experiments by combining selector chains
    with predictive models and evaluating them through a rule-based system.

    This class is responsible for:
    - Generating selector chains via SelectorChainFactory
    - Evaluating compatibility and priority using composition rules
    - Building ExperimentDefinition objects
    - Avoiding duplicate experiment configurations
    - Sorting experiments by priority

    Attributes:
        selectors (list[SelectorSpec]): Available feature selectors.
        models (list[ModelSpec]): Available predictive models.
    """

    def __init__(self, context: Context, selectors: list[SelectorSpec], models: list[ModelSpec]):
        """
        Initialize the ExperimentComposer.

        Args:
            selectors (list[SelectorSpec]): List of selector specifications.
            Models (list[ModelSpec]): List of model specifications.
        """
        self.selectors = selectors
        self.models = models

    def generate(self):
        """
        Generate all valid experiment combinations.

        Workflow:
            1. Build selector chains using SelectorChainFactory
            2. Iterate over all (chain, model) combinations
            3. Evaluate priority using composition rules
            4. Skip blocked combinations
            5. Avoid duplicate experiments
            6. Build ExperimentDefinition objects
            7. Sort experiments by priority

        Returns:
            list[ExperimentDefinition]: Sorted list of experiments.
        """
        experiments = []
        seen = set()

        chains = SelectorChainFactory(self.selectors).build()
        for chain in chains:
            for model in self.models:

                priority = self._evaluate_chain(chain, model)

                if priority == ExperimentPriority.BLOCKED:
                    continue

                key = (chain.name, model.name)
                if key in seen:
                    continue
                seen.add(key)

                experiment = self._build_experiment(chain, model, priority)
                experiments.append((priority, experiment))

        experiments.sort(key=lambda pair: _PRIORITY_ORDER[pair[0]])
        return [exp for _, exp in experiments]

    @staticmethod
    def _evaluate_chain(chain, model):
        """
        Evaluate the priority of a selector chain with a given model.

        This method applies all composition rules and selects the
        highest priority (lowest numeric value).

        Args:
            chain (SelectorChain): Sequence of selectors.
            model (ModelSpec): Model to evaluate.

        Returns:
            ExperimentPriority: Computed priority level.
        """
        priorities = []

        for rule in _RULES:
            if rule.match(chain, model):
                # logger.debug(f"Rule {rule.name} matched chain: {chain.name} and model: {model.name}")
                priorities.append(rule.priority)

        if not priorities:
            return ExperimentPriority.MEDIUM
        return min(priorities, key=lambda p: _PRIORITY_ORDER[p])

    @staticmethod
    def _build_experiment(chain: SelectorChain, model: ModelSpec, priority: ExperimentPriority):
        """
        Build an ExperimentDefinition from a selector chain and model.

        The resulting pipeline includes:
            - One or more feature selection steps
            - A final predictive model

        Args:
            chain (SelectorChain): Ordered selectors to apply.
            model (ModelSpec): Model specification.
            priority (ExperimentPriority): Assigned priority.

        Returns:
            ExperimentDefinition: Configured experiment.
        """

        builder = PipelineBuilder()

        steps = []

        for i, selector in enumerate(chain.selectors):
            steps.append((f"selector_{i}", selector.factory()))

        steps.append(("model", model.factory()))

        builder.steps = steps

        return ExperimentDefinition(
            name=f"{chain.name}__{model.name}",
            stage="feature_selection",
            pipeline_builder=builder,
            metadata={
                "selectors": [s.name for s in chain.selectors],
                "model": model.name,
                "model_type": model.spec_type,
                "model_based": model.type,
            },
        )
