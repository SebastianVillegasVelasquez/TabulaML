from app.core.context import RunContext


class Deployment:

    def __init__(self, context:RunContext):
        self.context = context


    def deploy(self):
        """
        Deploy the best model to production.
        This strategy is serve the .joblib model file.

        This is a placeholder method and should be implemented with actual deployment logic.
        """
        pass


    def save_model_on_storage(self):
        """
        Save the .joblib model file on the storage cloud service.

        """
        pass