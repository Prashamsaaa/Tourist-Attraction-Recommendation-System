class DynamicModelManager:
    def __init__(self, model):
        self.model = model

    def save_model(self, path):
        """
        Save the model to a specified path.
        """
        self.model.save(path)

    def load_model(self, path):
        """
        Load a model from a specified path.
        """
        self.model.load(path)