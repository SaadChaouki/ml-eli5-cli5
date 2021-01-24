from utils.devutils import generateClassificationData
import numpy as np
from utils.metrics import accuracy

class KNN():
    def __init__(self, k):
        self.k = k

    def fit(self, x , y):
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

if __name__ == '__main__':
    knn = KNN(10)
    x, y = generateClassificationData(1000)
    knn.fit(x, y)
    ypred = knn.predict(x)
    print(accuracy(y, ypred))
