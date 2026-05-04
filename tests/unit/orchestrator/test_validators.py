from app.core.context import StageResult
from app.core.orchestrator.validators import FeatureSelectionValidator
import pytest


class TestValidators:

    @pytest.mark.skip(reason="Not implemented yet")
    def test_validate_feature_selection_expected_stage_not_completed(self, run_context):
        validator = FeatureSelectionValidator()
        expected = False, "DATA_HANDLER stage not completed"
        assert validator.validate(build_context) == expected

    @pytest.mark.skip(reason="Not implemented yet")
    def test_validate_feature_selection_expected_no_results(self, run_context):
        build_context.stage_results["DATA_HANDLER"] = StageResult(name="DATA_HANDLER", results=[])

        validator = FeatureSelectionValidator()
        expected = False, "DATA_HANDLER produced no results"
        assert validator.validate(build_context) == expected
