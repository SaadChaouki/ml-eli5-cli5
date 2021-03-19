import numpy as np
from utils.functions import c_factor


class IsolationNode:
    def __init__(self, feature_index=None, feature_threshold=None, left_node=None,
                 right_node=None, sample=None, depth=None):
        self.feature_index = feature_index
        self.feature_threshold = feature_threshold
        self.left_node = left_node
        self.right_node = right_node
        self.sample = sample
        self.depth = depth

    def describe(self):
        return f"Node in depth {self.depth} and sample size of {len(self.sample)}. " \
               f"Splitting on {self.feature_index} by {self.feature_threshold}"


class IsolationTree:
    def __init__(self, max_depth):
        self.root = None
        self.maxDepth = max_depth

    def fit(self, X):
        self.root = self.__grow_tree(X)

    def __grow_tree(self, X, depth=0):
        if depth >= self.maxDepth + 1 or len(X) <= 1:
            return None

        # Get a split
        split_random = self.__random_split(X)

        # Start the node
        node = IsolationNode(feature_index=split_random['index'], feature_threshold=split_random['value'],
                             depth=depth, sample=X)

        # Recursive
        node.left_node = self.__grow_tree(split_random['leftData'], depth=depth + 1)
        node.right_node = self.__grow_tree(split_random['rightData'], depth=depth + 1)
        return node

    def __print_tree(self, tree=None, indent="--"):
        tree = self.root if tree is None else tree
        print(indent + ' ' + tree.describe())
        if tree.left_node is not None:
            self.__print_tree(tree.left_node, indent=indent + indent)
        if tree.right_node is not None:
            self.__print_tree(tree.right_node, indent=indent + indent)

    @staticmethod
    def __random_split(X):
        randomIndex = np.random.randint(0, X.shape[1])
        randomValue = np.random.uniform(min(X[:, randomIndex]), max(X[:, randomIndex]))
        leftData = X[X[:, randomIndex] < randomValue]
        rightData = X[X[:, randomIndex] >= randomValue]
        return {'index': randomIndex, 'value': randomValue, 'leftData': leftData, 'rightData': rightData}

    def path_length(self, X, length_path=0, node=None):
        node = self.root if node is None else node
        if X[node.feature_index] < node.feature_threshold and isinstance(node.left_node, IsolationNode):
            return self.path_length(X, length_path + 1, node.left_node)
        elif X[node.feature_index] >= node.feature_threshold and isinstance(node.right_node, IsolationNode):
            return self.path_length(X, length_path + 1, node.right_node)
        else:
            return length_path


class IsolationForest:
    def __init__(self, num_estimators=100):
        self.max_depth = 100
        self.num_estimators = num_estimators
        self.estimators = [IsolationTree(max_depth=self.max_depth) for _ in range(self.num_estimators)]

    def fit(self, X):
        self.sample_size = X.shape[0]
        for tree in self.estimators:
            tree.fit(X)

    def __single_prediction(self, observation):
        path_depth = np.mean([tree.path_length(observation) for tree in self.estimators])
        return path_depth

    def predict(self, X):
        predictions = [self.__single_prediction(observation) for observation in X]
        return np.array([np.power(2, -length / c_factor(self.sample_size)) for length in predictions])
