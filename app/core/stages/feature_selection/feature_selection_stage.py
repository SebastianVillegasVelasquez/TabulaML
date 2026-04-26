from app.core.enums import Stages
from app.core.stages.super_classes.stage import Stage


class FeatureSelectionStage(Stage):

    def get_stage_type(self) -> Stages:
        return Stages.FEATURE_SELECTION
