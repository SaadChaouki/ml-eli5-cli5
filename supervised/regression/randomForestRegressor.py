from supervised.regression.decisionTreeRegressor import DecisionTreeRegressor
from supervised.base.baseRandomForest import BaseRandomForest
import numpy as np


class RandomForestRegressor(BaseRandomForest):
    def __init__(self, num_estimators=100, max_depth=2, max_features=None, minimum_sample_leaf=10):
        super(RandomForestRegressor, self).__init__(num_estimators=num_estimators, max_depth=max_depth,
                                                    max_features=max_features, min_sample_leaf=minimum_sample_leaf)
        self.prediction_combination = self.__mean_prediction
        self.tree_model = DecisionTreeRegressor

    @staticmethod
    def __mean_prediction(predictions):
        return np.mean(predictions)
