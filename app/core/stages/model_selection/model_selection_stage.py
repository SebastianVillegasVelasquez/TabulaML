from app.core.context.stages import Stages
from app.core.stages.super_classes.stage import Stage


class ModelSelectionStage(Stage):

    def get_stage_type(self) -> Stages:
        return Stages.MODEL_SELECTION