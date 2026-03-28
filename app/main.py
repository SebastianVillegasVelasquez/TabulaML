from app.core.context import init_context
from app.core.orchestrator import Orchestrator
from app.services import load_data
from app.utils.logger import logger


def main():
    """
    Main entry point for the ML pipeline.
    
    Responsibilities:
    - Loads training data
    - Initializes pipeline context with configuration
    - Orchestrates execution of pipeline stages
    - Logs execution results
    """
    # Load dataset
    X, y = load_data('C:\\WorkSpace\\TabulaML\\data_test\\train.csv', 'Survived')

    # Initialize pipeline context
    context = init_context(X=X, y=y)
    if context:
        logger.info("RunContext initialized successfully.")

    # Run a pipeline with an orchestrator
    orchestrator = Orchestrator(context, max_retries=2)
    orchestrator.run()

    # Log execution report
    logger.info("\nExecution Report:")
    logger.info(orchestrator.get_execution_report_json())


if __name__ == '__main__':
    main()
