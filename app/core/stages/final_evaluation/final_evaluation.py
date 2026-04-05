from app.core.enums import Stages
from app.core.stages.super_classes.stage import Stage


class FinalEvaluationStage(Stage):

    def get_stage_type(self) -> Stages:
        """Child classes must define their stage type"""
        return Stages.FINAL_EVALUATION