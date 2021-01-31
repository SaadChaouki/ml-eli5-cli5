import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import random

def generateClassificationData(n = 1000):
    features, output = make_classification(n_samples = n, n_features = 2,
                                n_informative = 2, n_redundant = 0,
                                n_classes = 2, random_state= 10)
    return features, output

def plot2d(data, x = 'var_1', y = 'var_2', c = 'target'):
    plt.clf()
    plt.title('Scatter 2D')
    plt.scatter(x = data[x], y = data[y], c = data[c])
    plt.show()

def generateLRData(n = 100):
    data = pd.DataFrame([random.randint(0, 10) for _ in range(n)], columns = ['var'])
    data['target'] = [1 if row['var'] >= 6 else 0 for _, row in data.iterrows()]
    return data



if __name__ == '__main__':
    data = generateClassificationData(1000)
    plot2d(data)
    newData = generateLRData(100)
    plot2d(newData, 'var', 'target')
