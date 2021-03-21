from supervised.base.baseDecisionTree import BaseDecisionTree
import numpy as np


class DecisionTreeRegressor(BaseDecisionTree):
    def __init__(self, max_depth=2, minimum_sample_leaf=10):
        super(DecisionTreeRegressor, self).__init__(max_depth=max_depth, minimum_sample_leaf=minimum_sample_leaf)
        self.gain_function = self.__variance_reduction
        self.prediction_function = self.__average_value

    @staticmethod
    def __average_value(y):
        return np.mean(y)

    @staticmethod
    def __variance_reduction(y):
        return np.var(y)
