from app.core.context.context import Context
from app.core.enums import ProblemType
from app.core.orchestrator import Orchestrator
from app.services.loader import DataLoader
from app.utils.logger import logger


def main():
    """
    Main entry point for the ML pipeline.

    Responsibilities:
    - Loads data using DataLoader
    - Initializes pipeline context with configuration
    - Orchestrates execution of pipeline stages
    - Logs execution results
    """
    # Load data using DataLoader
    loader = DataLoader()
    dataset = loader.load_data(
        file_path="C:\\WorkSpace\\TabulaML\\data_test\\train.csv",
        target="Survived",
    )
    logger.info("Dataset loaded successfully.")

    # Initialize pipeline context with dataset
    context = Context.create(
        dataset=dataset,
        problem_type=ProblemType.CLASSIFICATION,
        priority_metric="f1",
        target_column="Survived",
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
