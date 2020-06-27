from utils.devutils import *
from utils.lossFunctions import logLoss
from utils.activationFunctions import sigmoid
import math
import random

class LogisticRegression():

    def __init__(self, learningRate = 0.01):
        print('initialised the logistic regression')

        

if __name__ == '__main__':
    generatedData = generateClassificationData(500)
    lr = LogisticRegression()
    print(lr._sigmoid(0))
    print('here')
