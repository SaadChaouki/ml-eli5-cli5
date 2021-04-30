import numpy as np


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def __get_closest_indices(self, observation):
        return np.argsort([np.linalg.norm(observation - nn) for nn in self.x])[:self.k]

    def __get_closest_classes(self, closest_indices):
        return [self.y[i] for i in closest_indices]

    @staticmethod
    def __getPrediction(closest_classes):
        return np.bincount(closest_classes).argmax()

    def __single_prediction(self, observation):
        closest_indices = self.__get_closest_indices(observation)
        closest_classes = self.__get_closest_classes(closest_indices)
        return self.__getPrediction(closest_classes)

    def predict(self, x):
        return [self.__single_prediction(observation) for observation in x]
