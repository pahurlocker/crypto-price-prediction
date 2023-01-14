class BasePredictor(object):
    def __init__(self, config):
        self.config = config
        self.data = self._load_data()
        self.model = self._load_model()

    def _load_model(self):
        raise NotImplementedError

    def _load_data(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError