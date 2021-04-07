import numpy as np
from .base_estimator import BaseEstimator

class LinearModel(BaseEstimator):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.coefs = None
        self.intercept = None
        
    def fit(self, X, y):
        if X.shape[1] > X.shape[0]:
            raise ValueError('Features cannot be more than samples')
        X_ = np.concatenate([X, np.ones((len(X),1))], axis=1)
        #compute coefficients and intercept
        B = y * np.linalg.pinv(X_)
        self.coefs, self.intercept = np.mean(B[:-1], axis=1), np.mean(B[-1])    
        return self
    
    def predict(self, X):
        if self.coefs is None:
            raise AssertionError('Estimator is not fitted!')
        return np.sum(self.coefs * X,axis=1) + self.intercept
            
class LinearRegression(LinearModel):
    def __init__(self):
        super().__init__('regression')
        
class LogisticRegression(LinearModel):
    def __init__(self):
        super().__init__('classification')
        
    def fit(self, X, y):
        self.n_labels = len(set(y))
        self.models = []
        for i in range(self.n_labels):
            fitted_model = LinearModel('classification').fit(X, (y==i).astype(float))
            self.models.append(fitted_model)
        return self
    
    def predict(self, X):
        y_probs = np.zeros((len(X), self.n_labels))
        for i in range(self.n_labels):
            y_probs[:,i] = self.models[i].predict(X)
        y_probs = y_probs / np.sum(np.exp(y_probs), axis=1, keepdims=True)
        return np.argmax(y_probs, axis=1)