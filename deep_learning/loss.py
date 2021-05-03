from abc import ABC
from deep_learning.activations import LogSoftmax
from torch import nn
import torch
import numpy as np


class Loss(object):
    def __init__(self):
        pass

    def __call__(self, x, y):
        raise NotImplementedError

    @staticmethod
    def gradient(y_true, y_predicted):
        raise NotImplementedError


class L1Loss(Loss):
    def __call__(self, y_true, y_predicted):
        return np.mean(np.abs(y_true - y_predicted))

    @staticmethod
    def gradient(y_true, y_predicted):
        return np.sign(y_predicted - y_true) / y_true.size


class MSELoss(Loss):
    def __call__(self, y_true, y_predicted):
        return np.mean(np.square(y_true - y_predicted))

    @staticmethod
    def gradient(y_true, y_predicted):
        return (2 * (y_predicted - y_true)) / y_true.size


# class NLLLoss(Loss):
#     def __call__(self, labels, y_predicted):
#         data = []
#         for i, label in enumerate(labels):
#             data.append(y_predicted[i][label])
#
#         print(np.mean(data))


class BCELoss(Loss):
    def __call__(self, y_true, y_predicted):
        return - np.mean(((y_true * np.log(y_predicted + 1e-15)) + ((1 - y_true) * np.log(1 - y_predicted + 1e-15))))

    @staticmethod
    def gradient(y_true, y_predicted):
        return (- (y_true / (y_predicted + 1e-15)) + (1 - y_true) / (1 - y_predicted + 1e-15)) / len(y_true)


class CrossEntropyLoss(Loss):
    def __call__(self, labels, y_predicted):
        y_predicted = LogSoftmax()(y_predicted)
        return - np.sum(y_predicted * labels)/labels.shape[0]

    def gradient(y_true, y_predicted):
        pass

def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

if __name__ == '__main__':
    from deep_learning.activations import LogSoftmax

    torch.manual_seed(10)

    softmax = nn.Softmax(dim=1)

    loss = nn.CrossEntropyLoss()
    input = torch.rand(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()

    # Mine
    y_pred = input.detach().numpy()
    y_true = target.numpy()

    print(to_categorical(y_true, 5))

    print(y_true)

    # SoftMax Transformation
    # print('-' * 50)
    # print(f'Torch Softmax {softmax(input)}')

    # Loss
    print('-'*50)
    print(f'Torch Loss {output}.')
    print(f'Custom Loss {CrossEntropyLoss()(to_categorical(y_true, 5), y_pred)}.')

    # Gradient
    print('-' * 50)
    print(f'Torch Gradient {input.grad}')
    # print(f'Custom Gradient {CrossEntropyLoss().gradient(y_true, y_pred)}')
    print('-' * 50)
