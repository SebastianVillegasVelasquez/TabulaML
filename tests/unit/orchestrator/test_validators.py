from app.core.context import StageResult
from app.core.enums import Stages
from app.core.orchestrator.validators import FeatureSelectionValidator


class TestValidators:
    # @pytest.mark.skip(reason="Not implemented yet")
    def test_validate_feature_selection_expected_stage_not_completed(
        self, build_context
    ):
        validator = FeatureSelectionValidator()
        expected = False, "DATA_HANDLER stage not completed"
        assert validator.validate(build_context) == expected

    # @pytest.mark.skip(reason="Not implemented yet")
    def test_validate_feature_selection_expected_no_results(self, build_context):
        build_context.stage_results["DATA_HANDLER"] = StageResult(
            name=Stages.DATA_HANDLER, results=[]
        )

        validator = FeatureSelectionValidator()
        expected = False, "DATA_HANDLER stage not completed"
        assert validator.validate(build_context) == expected
