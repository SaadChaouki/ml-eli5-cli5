import math


def sigmoid(x):
    return (1/(1+math.exp(-x)))


if __name__ == '__main__':
    print(sigmoid(0))
