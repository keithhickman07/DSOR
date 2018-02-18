import pandas as pd
import matplotlib as plt
import numpy as np

data = pd.read_csv("~/PycharmProjects/DSOR/clones/DSOR/Resources/carMPG.csv")

learningRate = .001
convergeThreshold = .001

def meanNormalization(data, targetColName):

    for i in data.columns:
        if i == targetColName:
            continue
        mean = data[i].mean()
        min = data[i].min()
        max = data[i].max()
        data[i] = data[i].apply(lambda d: float(d-mean)/float(max-min))

meanNormalization(data, "mpg")

def filterData(dataFrame):
    targetValues = dataFrame.iloc[:0]
    column = range(1, len(dataFrame.columns))
    data = dataFrame[column]
    data.insert(0, "x0", [1] * len(dataFrame))
    return targetValues, data

filterData(data)