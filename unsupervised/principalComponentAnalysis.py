from utils.functions import covariance_matrix
import numpy as np


class PCA():
    def __init__(self, num_components):
        self.num_components = num_components

    def transform(self, X):
        covariance = covariance_matrix(X)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance)
        sorted_idx = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, sorted_idx][:, :self.num_components]
        return X.dot(eigen_vectors)
