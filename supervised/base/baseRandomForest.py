import numpy as np
import random


class BaseRandomForest(object):
    def __init__(self, num_estimators=100, max_depth=2, max_features=None, min_sample_leaf=10):
        self.num_estimators = num_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree_indexes = []
        self.min_sample_leaf = min_sample_leaf

    def fit(self, x, y):
        # Get the size of the data.
        n_features = x.shape[1]
        indexes = list([k for k in range(n_features)])
        self.max_features = int(np.round(np.sqrt(n_features), 0)) if self.max_features is None else self.max_features
        if self.max_features > n_features:
            raise Exception('Cannot set maximum number of features higher than number of features.')

        # Create the trees and the indexes
        for _ in range(self.num_estimators):
            self.tree_indexes.append(
                {
                    'tree': self.tree_model(max_depth=self.max_depth, minimum_sample_leaf=self.min_sample_leaf),
                    'indexes': random.sample(indexes, k=self.max_features)
                }
            )

        # Fit the trees
        for estimator in self.tree_indexes:
            training_data = x[:, estimator['indexes']]
            estimator['tree'].fit(training_data, y)

    def __single_prediction(self, x):
        predictions = []
        for estimator in self.tree_indexes:
            estimator_data = x[:, estimator['indexes']]
            predictions.append(estimator['tree'].predict(estimator_data)[0])
        return self.prediction_combination(predictions)

    def predict(self, x):
        return [self.__single_prediction(np.array([sample])) for sample in x]
