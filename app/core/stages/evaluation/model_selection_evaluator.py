from app.core.stages.evaluation.base_evaluator import BaseEvaluator
from app.utils.logger import logger


class ModelSelectionEvaluator(BaseEvaluator):
    """Evaluator for Model Selection stage.
    
    Extracts top-3 models from DIFFERENT model families.
    This ensures diversity in model types for downstream ensemble or selection.
    """
    
    def _extract_stage_specific_data(self, sorted_results, best_experiment):
        # Extract top-3 models from different families
        top_models_by_family = self._extract_top_k_by_family(sorted_results, k=3)
        self._log_best_experiment(best_experiment)
        
        return {
            'top_k_models_by_family': top_models_by_family,
            'best_model': best_experiment.config.get('model', 'unknown'),
            'best_selector': best_experiment.config.get('selector', 'unknown'),
            'total_experiments': len(sorted_results),
            'models_by_family': self._get_all_models_by_family(sorted_results)
        }
    
    def _update_context(self, sorted_results, best_experiment, stage_specific_data):
        from app.core.context.run_context import StageResult
        
        stage_result = StageResult(
            name=self.stage,
            results=sorted_results,
            best_experiment=best_experiment,
            metadata={
                'top_k_models_by_family': stage_specific_data['top_k_models_by_family'],
                'best_model': stage_specific_data['best_model'],
                'best_selector': stage_specific_data['best_selector'],
                'models_by_family': stage_specific_data['models_by_family']
            }
        )
        
        self.context.update_context(self.stage, stage_result)
    
    def _get_all_models_by_family(self, sorted_results):
        """Get all models organized by family."""
        models_dict = {}
        for result in sorted_results:
            family = self._get_model_family(result)
            if family not in models_dict:
                models_dict[family] = []
            models_dict[family].append(result)
        
        logger.debug(f"Found model families: {list(models_dict.keys())}")
        return models_dict

