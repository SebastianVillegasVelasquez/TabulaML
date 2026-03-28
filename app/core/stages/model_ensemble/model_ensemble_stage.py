from app.core.enums.stages import Stages
from app.core.stages.super_classes.stage import Stage


class ModelEnsembleStage(Stage):

    def get_stage_type(self) -> Stages:
        """Child classes must define their stage type"""
        return Stages.MODEL_ENSEMBLE
