import numpy as np
from .base_model import BaseEstimator

class MLP(BaseEstimator):
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y):
        return self
    
    def predict(self, X):
#       todo
        return self
   

class MLPRegression(MLP):
    def __init__(self):
        super().__init__('regression')
        
class MLPClassifier(MLP):
    def __init__(self):
        super().__init__('classification')