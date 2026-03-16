from app.core.context.stages import Stages
from app.core.stages.super_classes.stage import Stage
from app.utils.logger import logger


class FineTuningStage(Stage):

    def get_stage_type(self) -> Stages:
        return Stages.FINE_TUNING
