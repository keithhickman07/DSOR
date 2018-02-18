import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("~/PycharmProjects/DSOR/clones/DSOR/Resources/carMPG.csv")

learningRate = 0.001  # learning rate
convergeThreshold = 0.001  # threshold to check condition of convergence

def meanNormalization(data, targetColName):  # we need to normalize the data so that each attribute is brought to same scale
    for i in data.columns:
        if i == targetColName:  # do not modify target values
            continue
        mean = data[i].mean()
        min = data[i].min()
        max = data[i].max()
        data[i] = data[i].apply(lambda d: float(d - mean) / float(max - min))

def filterData(dataFrame):  # we need to remove target value from the data frame and need to put x0 values.
    targetValues = dataFrame.iloc[:0]
    column = range(1, len(dataFrame.columns))
    data = dataFrame[dataFrame.columns[column]]
    data.insert(0, "x0", [1] * len(dataFrame))
    return targetValues, data  # return target value and modified dataframe

def initializeWeightVector(allZeros, numberOfFeatures):  # weights has to be initialized. either to all zeros or to random values
    if allZeros:  # if all zero flag is true then make it all zero
        return np.zeros((numberOfFeatures, 1))
    else:
        return np.array(np.random.uniform(-2, 2, size=numberOfFeatures)).reshape(numberOfFeatures, 1)

def calCost(weight, data, target):  # calculate cost after each iteration to look for convergance
    m = len(data)
    predictedValue = np.dot(data, weight)
    return (((predictedValue - target) ** 2).sum()) / (2 * m)

def calGradient(data, weights, target, column, learningRate):  # calculate gradient
    m = len(data)
    predictedValue = np.dot(data, weights)
    return learningRate * ((((predictedValue - target) * (data[[column]].as_matrix())).sum()) / float(m))

def gradientDescent():
    prevCost = 0
    iterationCount = 0
    costList = []
    meanNormalization(data, "mpg")  # second parameter is the name of the target column
    target, dataMatrix = filterData(data)
    weights = initializeWeightVector(True, len(dataMatrix.columns))
    cost = calCost(weights, dataMatrix, target)
    while abs(prevCost - cost[0]) > convergeThreshold:  # check for convergence
        iterationCount += 1
        prevCost = cost[0]
        weightSub = []
        for i in range(len(data.columns)):
            weightSub.append(calGradient(dataMatrix, weights, target, i, learningRate))
        weights = weights - weightSub
        print ("new Weights", weights.T)
        cost = calCost(weights, dataMatrix, target)
        print ("new cost", float(cost))
        costList.append(float(cost))
    print ("Algo converged in ", iterationCount, "iteration")
    print ("final weights ", weights.T)
    return costList

def plotGraph(x, y):
    # Create a figure of size 8x6 inches, 80 dots per inch
    plt.figure(figsize=(8, 6), dpi=80)

    # Create a new subplot from a grid of 1x1
    plt.subplot(1, 1, 1)  # rows , column , serial number starting from 1
    # Plot cosine with a blue continuous line of width 1 (pixels)
    plt.plot(x, y, color="blue", linewidth=2.5, linestyle="-", label="cost")
    plt.show()

costList = gradientDescent()
plotGraph(range(1, len(costList) + 1), costList)