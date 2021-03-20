import numpy as np


def manhattan_distance(point, X):
    return np.sum(np.abs(point - X), axis=1, dtype=np.float)


def euclidean_distance(point, X):
    return np.sqrt(np.sum(np.square(point - X), axis=1))