import numpy as np
from .base_model import BaseEstimator

class DecisionTree(BaseEstimator):
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y):
        return self
    
    def predict(self, X):
#       todo
        return self
   

class DecisionTreeRegression(DecisionTree):
    def __init__(self):
        super().__init__('regression')
        
class DecisionTreeClassifier(DecisionTree):
    def __init__(self):
        super().__init__('classification')