from app.core.context.context import Context
from app.core.orchestrator import Orchestrator
from app.utils.logger import logger


def main():
    """
    Main entry point for the ML pipeline.

    Responsibilities:
    - Initializes pipeline context with data loading and configuration
    - Orchestrates execution of pipeline stages
    - Logs execution results
    """
    # Initialize pipeline context with integrated data loading
    context = Context.create(
        file_path="C:\\WorkSpace\\TabulaML\\data_test\\train.csv",
        target_column="Survived",
        priority_metric="f1"
    )
    logger.info("Context initialized successfully.")

    # Run a pipeline with an orchestrator
    orchestrator = Orchestrator(context, max_retries=2)
    orchestrator.run()

    # Log execution report
    logger.info("\nExecution Report:")
    logger.info(orchestrator.get_execution_report_json())


if __name__ == "__main__":
    main()
