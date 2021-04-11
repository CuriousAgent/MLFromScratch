import numpy as np
    
def get_distance_fn(distance_name):
    if distance_name == 'euclidean':
        def euclidean_distance(X1, X2):
            dmat = np.zeros((len(X1),len(X2)))
            for i in range(len(X1)):
                dmat[i] = np.sqrt(np.sum((X1[i] - X2)**2, axis=1))
            return dmat    
        return euclidean_distance
    if distance_name == 'l1':
        def absolute_distance(X1, X2):
            dmat = np.zeros((len(X1),len(X2)))
            for i in range(len(X1)):
                dmat[i] = np.sum(np.abs(X1[i] - X2), axis=1)
            return dmat
        return absolute_distance