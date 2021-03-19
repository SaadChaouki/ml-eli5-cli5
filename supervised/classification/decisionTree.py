import numpy as np


class Node:
    def __init__(self, gini, predictedClass, featureIndex=None, featureThreshold=None, leftNode=None,
                 rightNode=None, samples=None, depth=None):
        self.gini = gini
        self.predictedClass = predictedClass
        self.featureIndex = featureIndex
        self.featureThreshold = featureThreshold
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.samples = samples
        self.depth = depth

    def describe(self):
        return f'Gini: {self.gini}. Splitting on feature {self.featureIndex} with a threshold of {self.featureThreshold}. ' \
               f'Total Samples : {self.samples}. Depth: {self.depth}.'


class DecisionTree:
    def __init__(self, maxDepth=2, minimumSample=100):
        self.maxDept = maxDepth
        self.minimumSample = minimumSample

    def __bestSplit(self, x, y):
        # Calculate the gini of parent to evaluate next split.
        parentGini = self.__computeGini(y)
        bestSplit = None
        bestSets = None
        for feature in range(self.numberFeatures):
            featureTarget = {'x': x[:, feature], 'y': y}
            for threshold in np.unique(x[:, feature]):
                # Split the data
                left, right = self.__divideData(featureTarget, threshold)
                if left is None: continue

                # Compute gini coefficients
                weightedGini = (self.__computeGini(left) * len(left) + self.__computeGini(right) * len(right)) / y.size

                # Check if the split is worth it
                if weightedGini < parentGini and len(left) >= self.minimumSample and len(right) >= self.minimumSample:
                    parentGini = weightedGini
                    bestSplit = {'featureIndex': feature, 'threshold': threshold, 'weightedGini': weightedGini}
                    bestSets = {
                        'leftX': x[x[:, bestSplit['featureIndex']] < bestSplit['threshold']],
                        'leftY': y[x[:, bestSplit['featureIndex']] < bestSplit['threshold']],
                        'rightX': x[x[:, bestSplit['featureIndex']] >= bestSplit['threshold']],
                        'rightY': y[x[:, bestSplit['featureIndex']] >= bestSplit['threshold']]
                    }

        return bestSplit, bestSets

    def __divideData(self, featureTarget, threshold):
        condition = featureTarget['x'] < threshold
        left = featureTarget['y'][condition]
        right = featureTarget['y'][~condition]
        if len(left) == 0 or len(right) == 0: return None, None
        return left, right

    def __computeGini(self, y):
        countClasses = [np.sum(y == c) for c in self.classes]
        return 1.0 - np.sum([(countClass / y.size) ** 2 for countClass in countClasses])

    def fit(self, x, y):
        self.classes = np.unique(y)
        self.numberFeatures = x.shape[1]
        self.root = self.__growTree(x, y, depth=0)

    def __growTree(self, x, y, depth):
        classes, counts = np.unique(y, return_counts=True)
        node = Node(
            gini=self.__computeGini(y),
            predictedClass=classes[np.argmax(counts)],
            depth=depth,
            samples=len(y)
        )

        if node.gini == 0: return node

        # Find best split
        bestSplit, bestSets = self.__bestSplit(x, y)

        # Record best split
        if bestSplit is not None:
            node.featureIndex = bestSplit['featureIndex']
            node.featureThreshold = bestSplit['threshold']

        # Recursion
        if depth < self.maxDept and bestSplit is not None:
            node.leftNode = self.__growTree(bestSets['leftX'], bestSets['leftY'], depth=depth + 1)
            node.rightNode = self.__growTree(bestSets['rightX'], bestSets['rightY'], depth=depth + 1)

        return node

    def __predict_value(self, x, tree=None):
        tree = self.root if tree is None else tree
        if tree.leftNode is None:
            return tree.predictedClass
        nextTree = tree.leftNode if x[tree.featureIndex] < tree.featureThreshold else tree.rightNode
        return self.__predict_value(x, nextTree)

    def predict(self, x):
        predictions = [self.__predict_value(sample) for sample in x]
        return predictions

    def print_tree(self, tree=None, indent="--"):
        tree = self.root if tree is None else tree
        print(indent + ' ' + tree.describe())
        if tree.leftNode is not None:
            self.print_tree(tree.leftNode, indent=indent + indent)
            self.print_tree(tree.rightNode, indent=indent + indent)

