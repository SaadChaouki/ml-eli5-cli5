from utils.devutils import generateClassificationData
from utils.metrics import accuracy
from supervised.decisionTree import DecisionTree
import numpy as np
import random


class RandomForest:
    def __init__(self, nEstimators=100, maxDepth=2, maxFeatures=None, minSampleSplit=10):
        self.nEstimators = nEstimators
        self.maxDepth = maxDepth
        self.maxFeatures = maxFeatures
        self.treeIndexes = []
        self.minSampleSplit = minSampleSplit

    def fit(self, x, y):
        # Get the size of the data.
        nFeatures = x.shape[1]
        indexes = list([k for k in range(nFeatures)])
        self.maxFeatures = int(np.round(np.sqrt(nFeatures), 0)) if self.maxFeatures is None else self.maxFeatures
        if self.maxFeatures > nFeatures:
            raise Exception('Cannot set maximum number of features higher than number of features.')

        # Create the trees and the indexes
        for _ in range(self.nEstimators):
            self.treeIndexes.append(
                {
                    'tree': DecisionTree(maxDepth=self.maxDepth, minimumSample=self.minSampleSplit),
                    'indexes': random.sample(indexes, k=self.maxFeatures)
                }
            )

        # Fit the trees
        for estimator in self.treeIndexes:
            trainingData = x[:, estimator['indexes']]
            estimator['tree'].fit(trainingData, y)

    def __singlePrediction(self, x):
        predictions = []
        for estimator in self.treeIndexes:
            estimatorData = x[:, estimator['indexes']]
            predictions.append(estimator['tree'].predict(estimatorData)[0])
        classes, counts = np.unique(predictions, return_counts=True)
        return classes[np.argmax(counts)]

    def predict(self, x):
        return [self.__singlePrediction(np.array([sample])) for sample in x]