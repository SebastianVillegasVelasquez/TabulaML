from app.core.stages.evaluation.base_evaluator import BaseEvaluator


class FineTuningEvaluator(BaseEvaluator):
    def _extract_stage_specific_data(self, sorted_results, best_experiment):

        self._log_best_experiment(best_experiment)

        return {
            "best_model": best_experiment,
            "results": sorted_results,
            "total_experiments": len(sorted_results),
        }

    def _update_context(self, sorted_results, best_experiment, stage_specific_data):
        from app.core.context.context import StageResult

        stage_result = StageResult(
            name=self.stage,
            results=sorted_results,
            best_experiment=best_experiment,
            metadata={
                "best_model": stage_specific_data["best_model"],
                "results": stage_specific_data["results"],
                "total_experiments": stage_specific_data["total_experiments"],
            },
        )

        self.context.update_context(self.stage, stage_result)
