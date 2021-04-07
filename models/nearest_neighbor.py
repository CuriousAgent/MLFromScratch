from scipy.stats import mode
import numpy as np
from .base_estimator import BaseEstimator
from .utils import get_distance_fn

class NearestNeighbor(BaseEstimator):
    def __init__(self, mode, neighbors, distance):
        self.neighbors = neighbors
        super().__init__()
        self.mode = mode
        self.distance = distance
        self.distance_fn = get_distance_fn(self.distance)
        
    def fit(self, X, y):
        if self.neighbors > len(X):
            raise ValueError('Neighbours are more than the number of samples!')
        if self.mode == 'classification':
            self.n_labels = len(set(y))
        self.X = X
        self.y = y
        return self
    
    def predict(self, X):
        distances = self.distance_fn(X, self.X)
        if self.mode == 'classification':
            y_prob = np.zeros((len(X), self.n_labels))
        y_pred = np.zeros(len(X))
        closest_ids = np.argsort(distances, axis=1)
        for i, x_i in enumerate(X):
            votes_i = self.y[closest_ids[i, :self.neighbors]]
            if self.mode=='regression':
                y_pred[i] = np.mean(votes_i)
            if self.mode=='classification':
                y_pred[i] = mode(votes_i).mode[0]
        return y_pred
    
class NearestNeighborClassifier(NearestNeighbor):
    def __init__(self, neighbors=3, distance='euclidean'):
        super().__init__('classification', neighbors, distance)
        
class NearestNeighborRegression(NearestNeighbor):
    def __init__(self, neighbors=3, distance='euclidean'):
        super().__init__('regression', neighbors, distance)