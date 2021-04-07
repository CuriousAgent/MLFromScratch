class BaseEstimator:  
    def __init__(self, **args):
        pass
    def fit(self, X, y, **kwargs):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError