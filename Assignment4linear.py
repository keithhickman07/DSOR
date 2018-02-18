# Assignment 4
# from sklearn import linear_model
from sklearn import preprocessing
import pandas as pd
import numpy as np
from numpy.linalg import inv

data = pd.read_csv("~/PycharmProjects/DSOR/clones/DSOR/Resources/Concrete_Data.csv")

print(data.head())

def zScoreNorm(targetCol, data):
    for i in data.columns:
        if i == targetCol:
            continue
        # mean = data[i].mean
        # std = data[i].std
        # data[i] = data[i].apply(lambda d: float(d - mean) / float(d - std))
        data[i] = preprocessing.scale(data[i])


def filterData(dataFrame, targetVarName):
    targetValues = dataFrame[targetVarName]
    column = range(1, len(dataFrame.columns))
    data = dataFrame[dataFrame.columns[column]]
    data.insert(0, "x0", [1] * len(dataFrame))
    return targetValues, data

def calOptimalWeight(dataFrame, target):
    innerProduct = dataFrame.T.dot(dataFrame)
    inverse = inv(innerProduct)
    product = inverse.dot(dataFrame.T)
    weight = product.dot(target)
    print(weight.shape)
    return weight

def predict(weights, X):
    predictedValue = X.dot(weights)
    return predictedValue

def calSSE(target, predicted):
    m = len(target)
    SSE = (np.asarray(target) - np.asarray(predicted)) ** 2 / (float(2 * m))
    return sum(SSE)

zScoreNorm('Strength', data)
Y, X = filterData(data, 'Strength')
optimalWeight = calOptimalWeight(X, Y)
predictedValue = predict(optimalWeight, X)
print(calSSE(Y,predictedValue))
