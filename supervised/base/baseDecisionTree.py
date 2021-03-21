import numpy as np


class Node(object):
    def __init__(self, gain, predicted_value, feature_index=None, feature_threshold=None, left_node=None,
                 right_node=None, samples=None, depth=None):
        self.gain = gain
        self.predicted_value = predicted_value
        self.feature_index = feature_index
        self.feature_threshold = feature_threshold
        self.left_node = left_node
        self.right_node = right_node
        self.samples = samples
        self.depth = depth

    def describe(self):
        return f'Gini: {self.gain}. Splitting on feature {self.feature_index} with a threshold of ' \
               f'{self.feature_threshold}. ' \
               f'Total Samples : {self.samples}. Depth: {self.depth}.'


class BaseDecisionTree(object):
    def __init__(self, max_depth=2, minimum_sample_leaf=10):
        self.max_depth = max_depth
        self.minimum_sample_leaf = minimum_sample_leaf
        self.root = None

    def __best_split(self, X, y):
        # Calculate the gini of parent to evaluate next split.
        parentGini = self.gain_function(y)
        best_split = None
        best_sets = None
        for feature in range(self.number_features):
            feature_target = {'x': X[:, feature], 'y': y}
            for threshold in np.unique(feature_target['x']):
                # Split the data
                left, right = self.__divide_data(feature_target, threshold)
                if left is None:
                    continue

                # Compute gini coefficients
                computed_impurity = (self.gain_function(left) * len(left) + self.gain_function(right) * len(
                    right)) / y.size

                # Check if the split is worth it
                if computed_impurity < parentGini and len(left) >= self.minimum_sample_leaf\
                        and len(right) >= self.minimum_sample_leaf:
                    parentGini = computed_impurity
                    best_split = {'featureIndex': feature, 'threshold': threshold, 'weightedGini': computed_impurity}
                    best_sets = {
                        'leftX': X[X[:, best_split['featureIndex']] < best_split['threshold']],
                        'leftY': y[X[:, best_split['featureIndex']] < best_split['threshold']],
                        'rightX': X[X[:, best_split['featureIndex']] >= best_split['threshold']],
                        'rightY': y[X[:, best_split['featureIndex']] >= best_split['threshold']]
                    }

        return best_split, best_sets

    @staticmethod
    def __divide_data(featureTarget, threshold):
        condition = featureTarget['x'] < threshold
        left = featureTarget['y'][condition]
        right = featureTarget['y'][~condition]
        if len(left) == 0 or len(right) == 0: return None, None
        return left, right

    def fit(self, X, y):
        self.number_features = X.shape[1]
        self.root = self.__grow_tree(X, y, depth=0)

    def __grow_tree(self, X, y, depth):

        node = Node(
            gain=self.gain_function(y),
            predicted_value=self.prediction_function(y),
            depth=depth,
            samples=len(y)
        )

        # Perfect node, no need to split
        if node.gain == 0:
            return node

        # Find best split
        bestSplit, bestSets = self.__best_split(X, y)

        # Record best split
        if bestSplit is not None:
            node.feature_index = bestSplit['featureIndex']
            node.feature_threshold = bestSplit['threshold']

        # Recursion
        if depth < self.max_depth and bestSplit is not None:
            node.left_node = self.__grow_tree(bestSets['leftX'], bestSets['leftY'], depth=depth + 1)
            node.right_node = self.__grow_tree(bestSets['rightX'], bestSets['rightY'], depth=depth + 1)

        return node

    def __predict_value(self, x, tree=None):
        tree = self.root if tree is None else tree
        if tree.left_node is None:
            return tree.predicted_value
        next_node = tree.left_node if x[tree.feature_index] < tree.feature_threshold else tree.right_node
        return self.__predict_value(x, next_node)

    def predict(self, x):
        predictions = [self.__predict_value(sample) for sample in x]
        return predictions

    def print_tree(self, tree=None, indent="--"):
        tree = self.root if tree is None else tree
        print(indent + ' ' + tree.describe())
        if tree.left_node is not None:
            self.print_tree(tree.left_node, indent=indent + indent)
            self.print_tree(tree.right_node, indent=indent + indent)


