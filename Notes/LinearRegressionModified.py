import pandas
import os
from numpy.linalg import inv
import numpy as np
relativePath = os.getcwd()
dataFilePath =  os.path.join(relativePath,"concrete.csv")
data = pandas.read_csv(dataFilePath)

# http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
# http://archive.ics.uci.edu/ml/datasets/Wine+Quality


def Z_ScoreNormalization(data, targetColName):  # we need to normalize the data so that each attribute is brought to same scale
    for i in data.columns:
        if i == targetColName:  # do not modify target values
            continue
        mean = data[i].mean()
        std = data[i].std()
        data[i] = data[i].apply(lambda d: float(d - mean) / float(std))  # perform z-score normalization

def filterData(dataFrame, targetVarName):  # we need to remove target value from the data frame and need to put x0 values.
    targetValues = dataFrame[targetVarName]
    colum = range(1, len(dataFrame.columns))  # all the column except 0th which is target
    data = dataFrame[dataFrame.columns[colum]]  # keep all column except 0th
    data.insert(0, "x0", [1] * len(dataFrame))  # in 0th column set all 1 and name the column as X0 this is to accommodate biases weights i.e. w0
    return targetValues, data  # return target value and modified dataframe


def calOptimalWeight(dataFrame, target):  # this function calculates W'=(X^TX)^-1X^TY
    innerProduct = dataFrame.T.dot(dataFrame)  # calculating X^TX
    inverse = inv(innerProduct)  # calculate (X^TX)^-1
    product = inverse.dot(dataFrame.T)  # (X^TX)^-1X^T
    weight = product.dot(target)  # (X^TX)^-1X^TY
    print(weight.shape)
    return weight  # return W'


def predict(weights, X):  # this function calculates Y'=W^TX or y'=XW
    predictedValue = X.dot(weights)
    return predictedValue


def calSSE(target, predicted):  # calculate sum of squared error
    m = len(target)
    SSE = ((np.asarray(target) - np.asarray(predicted)) ** 2) / float(2 * m)
    return sum(SSE)


Z_ScoreNormalization(data, 'ccs')  # step 1 Normalize
Y, X = filterData(data, 'ccs')  # Step 2 Filter the data i.e. seperate data from target value
optimalWeight = calOptimalWeight(X, Y)  # Step 3 calculate W'
predictedValue = predict(optimalWeight, X)  # Step 4 predict value using W'
print(calSSE(Y, predictedValue))  # calculate S.S.E using actual target value Y and predicted target value Y
