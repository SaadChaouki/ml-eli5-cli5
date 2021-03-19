import numpy as np


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def __getClosestIndices(self, observation):
        return np.argsort([np.linalg.norm(observation - nn) for nn in self.x])[:self.k]

    def __getClosestClasses(self, closestIndices):
        return [self.y[i] for i in closestIndices]

    def __getPrediction(self, closestClasses):
        return np.bincount(closestClasses).argmax()

    def __singlePrediction(self, observation):
        closestIndices = self.__getClosestIndices(observation)
        closestClasses = self.__getClosestClasses(closestIndices)
        return self.__getPrediction(closestClasses)

    def predict(self, x):
        return [self.__singlePrediction(observation) for observation in x]
