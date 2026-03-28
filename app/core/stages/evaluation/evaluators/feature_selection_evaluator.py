from app.core.stages.evaluation.base_evaluator import BaseEvaluator
from app.core.context.run_context import StageResult
from app.utils.logger import logger


class FeatureSelectionEvaluator(BaseEvaluator):
    """Evaluator for Feature Selection stage.
    
    Extracts top-3 feature selectors based on performance.
    Groups by selector type to ensure diversity.
    """

    def _extract_stage_specific_data(self, sorted_results, best_experiment):
        """Extract feature selection specific data."""
        # Extract top-3 selectors by type
        top_k_selectors = self._extract_top_k_selectors(sorted_results, k=3)
        
        # Extract feature data from best
        feature_data = self._extract_feature_data(best_experiment)
        
        self._log_best_experiment(best_experiment)

        return {
            'top_k_selectors': top_k_selectors,
            'best_selector': best_experiment.config.get('selector', 'unknown'),
            'best_predictor': best_experiment.config.get('predictor', 'unknown'),
            'feature_mask': feature_data.get('feature_mask'),
            'selected_features': feature_data.get('selected_features'),
            'n_features_selected': feature_data.get('n_features_selected', 0),
            'total_experiments': len(sorted_results)
        }

    def _update_context(self, sorted_results, best_experiment, stage_specific_data):
        """Update context with feature selection results."""
        stage_result = StageResult(
            name=self.stage,
            results=None,
            best_experiment=best_experiment,
            metadata={
                'top_k_selectors': stage_specific_data['top_k_selectors'],
                'selector': stage_specific_data['best_selector'],
                'predictor': stage_specific_data['best_predictor'],
                'n_features_selected': stage_specific_data['n_features_selected'],
                'total_experiments': stage_specific_data['total_experiments'],
                'selector_estimator': best_experiment.pipeline.named_steps.get('feature_selection', None),
            }
        )
        
        if stage_specific_data['selected_features']:
            stage_result.feature_importance = {
                col: 1.0 for col in stage_specific_data['selected_features']
            }
        
        self.context.update_context(self.stage, stage_result)
        
        # Store feature data in experiment for downstream use
        best_experiment.feature_mask = stage_specific_data.get('feature_mask')
        best_experiment.selected_features = stage_specific_data.get('selected_features')

    def _extract_top_k_selectors(self, sorted_results, k=3):
        """Extract top-k selectors grouping by selector type."""
        selector_best = {}
        
        for result in sorted_results:
            selector_name = result.config.get('selector', 'unknown')
            if selector_name not in selector_best:
                selector_best[selector_name] = result
        
        top_k = dict(list(selector_best.items())[:k])
        logger.debug(f"Top {len(top_k)} selectors: {list(top_k.keys())}")
        return top_k

    def _extract_feature_data(self, best_experiment):
        """Extract feature mask and selected features from best experiment."""
        pipeline = best_experiment.pipeline
        
        if "feature_selection" not in pipeline.named_steps:
            return {'feature_mask': None, 'selected_features': None, 'n_features_selected': 0}
        
        selector = pipeline.named_steps["feature_selection"]
        
        if not hasattr(selector, "get_support"):
            return {'feature_mask': None, 'selected_features': None, 'n_features_selected': 0}
        
        try:
            mask = selector.get_support()
            original_columns = self.config.X_train.columns.tolist()
            selected_columns = [col for col, keep in zip(original_columns, mask) if keep]
            
            logger.info(f"Selected {len(selected_columns)} features out of {len(original_columns)}")
            
            return {
                'feature_mask': mask,
                'selected_features': selected_columns,
                'n_features_selected': len(selected_columns)
            }
        except Exception as e:
            logger.error(f"Error extracting feature data: {e}")
            return {'feature_mask': None, 'selected_features': None, 'n_features_selected': 0}
