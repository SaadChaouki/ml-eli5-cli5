import abc
import numpy as np


class StatisticalTest(object, metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def test(self, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, **kwargs):
        pass


class ZTest(StatisticalTest):

    @staticmethod
    def test(sample: np.array, population_mean: float, population_variance: float) -> float:
        print('test')
        return None

    def __call__(self, sample: np.array, population_mean: float, population_variance: float):
        return self.test(sample=sample,
                         population_mean=population_mean,
                         population_variance=population_variance)


if __name__ == '__main__':
    print('saad')
    s = ZTest.test(np.array([0, 10, 0, 0, 0]), 10, 100)


