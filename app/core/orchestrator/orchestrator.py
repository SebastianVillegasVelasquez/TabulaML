"""
Pipeline Orchestrator: Coordinates execution of ML pipeline stages.

Responsibilities:
- Validates preconditions before executing each stage
- Executes stages with automatic retry mechanism for transient failures
- Tracks execution metrics (timing, status, errors) for each stage
- Runs evaluation phase after successful stage execution
- Provides detailed execution reports for auditing and monitoring

Architecture:
- Stages are executed sequentially with dependency validation
- Each stage reports its execution status (success, failed, skipped)
- Failed stages are retried up to max_retries before stopping
- If preconditions fail, stage is skipped without affecting the pipeline
"""

import json
import traceback
from datetime import datetime
from typing import List, Dict, Any

from app.core.context import Context
from app.core.enums import ExecutionStatus, Stages
from app.core.orchestrator.pipeline_stage import PipelineStage
from app.core.orchestrator.stage_execution import StageExecution
from app.core.orchestrator.stages_adapters import (
    DataInspectionStageAdapter,
    FeatureSelectionStageAdapter,
    ModelSelectionStageAdapter,
    FineTuningStageAdapter,
    ModelEnsambleStageAdapter,
    # ModelThresholdExtractorAdapter
)
from app.utils.logger import logger


class Orchestrator:
    """
    Orchestrates the execution of ML pipeline stages with validation and resilience.

    Usage:
        context = Context(config=project_config)
        orchestrator = Orchestrator(context, max_retries=2)
        summary = orchestrator.run()

        # Access detailed execution report
        report = orchestrator.get_execution_report()
    """

    def __init__(self, context: Context, max_retries: int = 2):
        """
        Initialize the orchestrator.

        Args:
            context: The Context containing pipeline configuration and state
            max_retries: Number of retry attempts for failed stages (default: 2)
        """
        self.context = context
        self.max_retries = max_retries
        self.executions: List[StageExecution] = []
        self._pipeline: List[PipelineStage] = []

    def _build_pipeline(self) -> List[PipelineStage]:
        """
        Build the ordered pipeline of stages.

        Modify this method to add, remove, or reorder pipeline stages.
        Each stage is executed in the order returned by this method.

        Returns:
            List of PipelineStage instances in execution order
        """
        return [
            DataInspectionStageAdapter(self.context),
            FeatureSelectionStageAdapter(self.context),
            ModelSelectionStageAdapter(self.context),
            FineTuningStageAdapter(self.context),
            ModelEnsambleStageAdapter(self.context),
            # FinalEvaluationAdapter(self.context)
        ]

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline with robust error handling and validation.

        Execution flows for each stage:
        1. Validate preconditions (if a validator exists)
        2. If validation fails: Mark as SKIPPED, continue to the next stage
        3. If validation passes: Execute with automatic retries
        4. If execution fails: Mark as FAILED, continue to the next stage
        5. If execution succeeds: Run evaluation, mark as SUCCESS
        6. Record execution metrics (timing, status, error information)

        Returns:
            Dict with execution summary:
            {
                'success': [list of successful stage names],
                'failed': [list of failed stage names],
                'skipped': [list of skipped stage names]
            }
        """
        logger.info("=" * 80)
        logger.info("PIPELINE ORCHESTRATION STARTED")
        logger.info("=" * 80)

        # Build pipeline stages
        self._pipeline = self._build_pipeline()

        # Initialize execution summary
        summary: Dict[str, Any] = {"success": [], "failed": [], "skipped": []}

        # Execute each stage in the pipeline
        for stage in self._pipeline:
            stage_type = stage.get_stage_type()

            # Create execution record
            execution = StageExecution(stage=stage_type)

            try:
                # Step 1: Validate preconditions
                logger.info(f"\n[{stage_type.value}] Validating preconditions...")
                validator = stage.get_validator()

                if validator:
                    is_valid, error_msg = validator.validate(self.context)
                    if not is_valid:
                        # Preconditions not met - skip stage
                        logger.warning(
                            f"[{stage_type.value}] Precondition failed: {error_msg}"
                        )
                        execution.status = ExecutionStatus.SKIPPED
                        execution.skip_reason = error_msg
                        execution.end_time = datetime.now()
                        summary["skipped"].append(stage_type.value)
                        self.executions.append(execution)
                        continue

                logger.info(f"[{stage_type.value}] Preconditions validated ✓")

                # Step 2: Execute stage with retry logic
                execution = self._execute_with_retry(stage, execution)

                # Step 3: Run evaluation if stage succeeded
                if execution.status == ExecutionStatus.SUCCESS:
                    try:
                        self._run_evaluation(stage_type)
                    except Exception as e:
                        logger.warning(
                            f"[{stage_type.value}] Evaluation failed (stage still considered successful): {str(e)}",
                            exc_info=False,
                        )

                # Record outcome
                summary[execution.status.value].append(stage_type.value)

            except Exception as e:
                # Unexpected unhandled error
                logger.error(
                    f"[{stage_type.value}] Unexpected error: {str(e)}", exc_info=True
                )
                execution.status = ExecutionStatus.FAILED
                execution.error = e
                execution.error_traceback = traceback.format_exc()
                execution.end_time = datetime.now()
                if execution.start_time:
                    execution.duration_seconds = (
                        execution.end_time - execution.start_time
                    ).total_seconds()
                summary["failed"].append(stage_type.value)

            finally:
                # Record execution always
                self.executions.append(execution)

        # Log execution summary
        self._log_summary(summary)

        return summary

    def _execute_with_retry(
        self, stage: PipelineStage, execution: StageExecution
    ) -> StageExecution:
        """
        Execute a stage with automatic retry on transient failures.

        Retry strategy:
        - Attempts execution up to max_retries times
        - Returns on first success (SUCCESS status)
        - Returns after all retries exhausted (FAILED status)

        Args:
            stage: The PipelineStage to execute
            execution: The StageExecution record to update with metrics

        Returns:
            Updated StageExecution with final status and metrics
        """
        stage_type = stage.get_stage_type()

        for attempt in range(1, self.max_retries + 1):
            try:
                # Mark stage as running
                execution.status = ExecutionStatus.RUNNING
                execution.start_time = datetime.now()
                execution.retry_count = attempt

                logger.info(
                    f"[{stage_type.value}] Execution attempt {attempt}/{self.max_retries}"
                )

                # Execute stage logic
                stage.execute(self.context)

                # Execution succeeded
                execution.status = ExecutionStatus.SUCCESS
                execution.end_time = datetime.now()
                execution.duration_seconds = (
                    execution.end_time - execution.start_time
                ).total_seconds()

                logger.info(
                    f"[{stage_type.value}] Completed successfully ({execution.duration_seconds:.2f}s)"
                )

                return execution

            except Exception as e:
                # Log failure and retry if applicable
                logger.warning(
                    f"[{stage_type.value}] Attempt {attempt} failed: {str(e)}"
                )

                if attempt == self.max_retries:
                    # All retries exhausted
                    execution.status = ExecutionStatus.FAILED
                    execution.error = e
                    execution.error_traceback = traceback.format_exc()
                    execution.end_time = datetime.now()
                    if execution.start_time:
                        execution.duration_seconds = (
                            execution.end_time - execution.start_time
                        ).total_seconds()

                    logger.error(
                        f"[{stage_type.value}] Failed after {self.max_retries} attempts"
                    )

                    return execution

        return execution

    def _run_evaluation(self, stage: Stages) -> None:
        """
        Execute evaluation phase for a completed stage.

        This evaluates the results of the stage execution and updates
        the context with evaluation metrics and best results.

        Args:
            stage: The stage model_based to evaluate

        Raises:
            Exception if evaluation fails
        """
        try:
            logger.info(f"[EVALUATION] Evaluating {stage.value} results...")

            from app.core.stages.evaluation.evaluation_stage import EvaluationStage

            evaluator = EvaluationStage(stage=stage, context=self.context)
            evaluator.run()

            logger.info(f"[EVALUATION] Evaluation completed for {stage.value}")

        except Exception as e:
            logger.error(
                f"[EVALUATION] Evaluation failed for {stage.value}: {str(e)}",
                exc_info=True,
            )
            raise

    def _log_summary(self, summary: Dict[str, Any]) -> None:
        """
        Log the final execution summary with detailed metrics.

        Displays execution status, timing, and error information for each stage.

        Args:
            summary: The execution summary dictionary
        """
        logger.info("\n" + "=" * 80)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 80)

        # Log each execution record
        for execution in self.executions:
            status_icon = {
                ExecutionStatus.SUCCESS: "✓",
                ExecutionStatus.FAILED: "✗",
                ExecutionStatus.SKIPPED: "⊘",
            }.get(execution.status, "?")

            status_str = f"{status_icon} {execution.stage.value:20}"
            duration_str = f"Duration: {execution.duration_seconds:7.2f}s"
            status_val = f"Status: {execution.status.value:10}"

            log_line = f"{status_str} | {status_val} | {duration_str}"

            # Add contextual information
            if execution.status == ExecutionStatus.FAILED and execution.error:
                log_line += f" | Error: {str(execution.error)}"
                logger.info(log_line)
                if execution.error_traceback:
                    logger.error(
                        f"\nTraceback for {execution.stage.value}:\n{execution.error_traceback}"
                    )
            elif execution.status == ExecutionStatus.SKIPPED and execution.skip_reason:
                log_line += f" | Reason: {execution.skip_reason}"
                logger.info(log_line)
            else:
                logger.info(log_line)

        logger.info("=" * 80)
        logger.info(f"✓ Successful: {len(summary['success'])}")
        logger.info(f"✗ Failed:     {len(summary['failed'])}")
        logger.info(f"⊘ Skipped:    {len(summary['skipped'])}")
        logger.info("=" * 80 + "\n")

    def get_execution_report(self) -> List[Dict[str, Any]]:
        """
        Get detailed execution report as structured data.

        Provides complete execution metrics for each stage, suitable for
        logging systems, APIs, dashboards, or audit trails.

        Returns:
            List of execution records with all metrics:
            [
                {
                    'stage': 'feature_selection',
                    'status': 'success',
                    'duration_seconds': 5.67,
                    'error': None,
                    'timestamp': '2026-03-13T14:30:45.123456',
                    'retry_count': 1,
                    'skip_reason': None
                },
                ...
            ]
        """
        return [
            {
                "stage": exec.stage.value,
                "status": exec.status.value,
                "duration_seconds": round(exec.duration_seconds, 2),
                "error": str(exec.error) if exec.error else None,
                "error_traceback": exec.error_traceback,
                "timestamp": exec.start_time.isoformat() if exec.start_time else None,
                "retry_count": exec.retry_count,
                "skip_reason": exec.skip_reason,
            }
            for exec in self.executions
        ]

    def get_execution_report_json(self) -> str:
        """
        Get execution report as formatted JSON string.

        Returns:
            JSON string of execution report for logging or transmission
        """
        return json.dumps(self.get_execution_report(), indent=2)
