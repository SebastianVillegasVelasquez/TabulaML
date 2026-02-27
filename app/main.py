from app.core.context.init_context import init_context
from app.core.orchestrator import Orchestrator
from app.services.loader import load_data
from app.utils.logger import logger


# Initialize any global context, configurations, or resources here
def main():
    # TODO: After implementing the data loading and preprocessing,
    #  replace the hardcoded dataset with the actual data loading logic.
    # (X_train, y_train), (X_test, y_test)
    X, y = load_data('C:\\WorkSpace\\TabulaML\\data_test\\train.csv', 'Survived')

    context = init_context(X=X, y=y)
    if context:
        logger.info("Context initialized successfully.")

    Orchestrator(context).run()

if __name__ == '__main__':
    main()
