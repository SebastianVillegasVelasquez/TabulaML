from app.core.context.run_context import RunContext
from app.core.stages.fine_tuning.tuner_strategies import TunerStrategy
from app.utils.logger import logger


class FineTunerFactory:
    _TUNERS = {}

    @classmethod
    def _register_defaults(cls):
        from app.core.stages.fine_tuning.tuners import OptunaTunerStrategy, GridSearchCVTunerStrategy

        if not cls._TUNERS:
            try:
                cls._TUNERS = {
                    TunerStrategy.OPTUNA.value: OptunaTunerStrategy,
                    TunerStrategy.GRID_SEARCH.value: GridSearchCVTunerStrategy
                }
                logger.info("Default tuners registered successfully")
            except Exception as e:
                logger.error(f"Error registering default tuners: {str(e)}", exc_info=True)

    @classmethod
    def create_tuner(cls,
                     tuner_strategy: TunerStrategy,
                     context: RunContext):

        cls._register_defaults()
        try:
            logger.info(f"Creating tuner for {tuner_strategy}")
            return cls._TUNERS[tuner_strategy.value](context)
        except KeyError:
            raise ValueError(f"Tuner '{tuner_strategy.value}' not found")

    @classmethod
    def register_tuner(cls, tuner_name: str, tuner_class):
        if tuner_name in cls._TUNERS:
            logger.warning(f"Tuner '{tuner_name}' is already registered and will be overwritten")
        cls._TUNERS[tuner_name] = tuner_class
        logger.info(f"Tuner '{tuner_name}' registered successfully")
