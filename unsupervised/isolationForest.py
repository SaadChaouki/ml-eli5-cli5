import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from utils.functions import cFactor


class IsolationNode:
    def __init__(self, featureIndex=None, featureThreshold=None, leftNode=None,
                 rightNode=None, sample=None, depth=None):
        self.featureIndex = featureIndex
        self.featureThreshold = featureThreshold
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.sample = sample
        self.depth = depth

    def describe(self):
        return f"Node in depth {self.depth} and sample size of {len(self.sample)}. " \
               f"Splitting on {self.featureIndex} by {self.featureThreshold}"


class IsolationTree:
    def __init__(self, maxDepth):
        self.root = None
        self.maxDepth = maxDepth

    def fit(self, X):
        self.root = self.__growTree(X)

    def __growTree(self, X, depth=0):
        if depth >= self.maxDepth + 1 or len(X) <= 1:
            return None

        # Get a split
        randomSplit = self.__randomSplit(X)

        # Start the node
        node = IsolationNode(featureIndex=randomSplit['index'], featureThreshold=randomSplit['value'],
                             depth=depth, sample=X)

        # Recursive
        node.leftNode = self.__growTree(randomSplit['leftData'], depth=depth + 1)
        node.rightNode = self.__growTree(randomSplit['rightData'], depth=depth + 1)
        return node

    def __print_tree(self, tree=None, indent="--"):
        tree = self.root if tree is None else tree
        print(indent + ' ' + tree.describe())
        if tree.leftNode is not None:
            self.__print_tree(tree.leftNode, indent=indent + indent)
        if tree.rightNode is not None:
            self.__print_tree(tree.rightNode, indent=indent + indent)

    def __randomSplit(self, X):
        randomIndex = np.random.randint(0, X.shape[1])
        randomValue = np.random.uniform(min(X[:, randomIndex]), max(X[:, randomIndex]))
        leftData = X[X[:, randomIndex] < randomValue]
        rightData = X[X[:, randomIndex] >= randomValue]
        return {'index': randomIndex, 'value': randomValue, 'leftData': leftData, 'rightData': rightData}

    def pathLength(self, X, pathLength=0, node=None):
        node = self.root if node is None else node
        if X[node.featureIndex] < node.featureThreshold and isinstance(node.leftNode, IsolationNode):
            return self.pathLength(X, pathLength + 1, node.leftNode)
        elif X[node.featureIndex] >= node.featureThreshold and isinstance(node.rightNode, IsolationNode):
            return self.pathLength(X, pathLength + 1, node.rightNode)
        else:
            return pathLength


class IsolationForest:
    def __init__(self, numEstimators):
        self.maxDepth = 100
        self.numEstimators = numEstimators
        self.estimators = [IsolationTree(maxDepth=self.maxDepth) for _ in range(self.numEstimators)]

    def fit(self, X):
        self.sampleSize = X.shape[0]
        for tree in self.estimators:
            tree.fit(X)

    def __singlePrediction(self, observation):
        pathDepth = np.mean([tree.pathLength(observation) for tree in self.estimators])
        return pathDepth

    def predict(self, X):
        predictions = [self.__singlePrediction(observation) for observation in X]
        return np.array([np.power(2, -length / cFactor(self.sampleSize)) for length in predictions])


if __name__ == '__main__':
    ilf = IsolationForest(numEstimators=10)
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]  # diagonal covariance
    Nobjs = 10000
    x, y = np.random.multivariate_normal(mean, cov, Nobjs).T
    # Add manual outlier
    x[0] = 5
    y[0] = 5

    X = np.array([x, y]).T
    X = pd.DataFrame(X, columns=['feat1', 'feat2'])

    k = X.values
    ilf.fit(k)
    pred = ilf.predict(k)
    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, c=pred, cmap='Blues');
    plt.show()
