from app.core.enums import Stages
from app.core.stages.super_classes.stage import Stage


class FineTuningStage(Stage):
    def get_stage_type(self) -> Stages:
        return Stages.FINE_TUNING
