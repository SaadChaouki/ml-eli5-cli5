from abc import ABC

import numpy as np


class Activation(object):
    def __call__(self, x):
        raise NotImplementedError

    def gradient(self, x):
        raise NotImplementedError


class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class ReLU(Activation):
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


class TanH(Activation):
    def __call__(self, x):
        return (2 / (1 + np.exp(-2 * x))) - 1

    def gradient(self, x):
        return 1 - self.__call__(x) ** 2


class LeakyReLU(Activation):
    def __init__(self):
        self.alpha = .01

    def __call__(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x > 0, 1, self.alpha)


class ELU(Activation):
    """ Exponential linear unit """

    def __init__(self, alpha=1.):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        return np.where(x > 0, 1, self.__call__(x) + self.alpha)


class SELU(Activation):
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.l = 1.0507009873554804934193349852946

    def __call__(self, x):
        return self.l * np.where(x > 0, x, self.alpha * np.exp(x) - self.alpha)

    def gradient(self, x):
        return self.l * np.where(x > 0, 1, self.alpha * np.exp(x))


class Softmax(Activation):
    def __call__(self, x):
        e_x = np.exp(x)
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)


class LogSoftmax(Activation, ABC):
    def __call__(self, x):
        e_x = np.exp(x)
        return np.log(e_x / np.sum(e_x, axis=1, keepdims=True))
