from sklearn import preprocessing
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn import linear_model

# Problem 1

data = pd.read_csv("~/PycharmProjects/DSOR/clones/DSOR/Resources/iris.csv")
print(data.head())
data['class'] = data['class'].replace(['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], ['1', '0', '0'])


def zScoreNorm(targetCol, data):
    for i in data.columns:
        if i == targetCol:
            continue
        data[i] = preprocessing.scale(data[i])


def dataFilter(targetColumn, data):
    columnName = [col for col in data.columns if col not in targetColumn]
    dataFrame = data[columnName]
    labelFrame = data[[targetColumn]]
    size = len(dataFrame)
    trainingData = dataFrame.loc[range(1,int(size/2))]
    trainingLabel = labelFrame.loc[range(1,int(size/2))]
    testData = dataFrame.loc[range(int(size/2),size)]
    testLabel = labelFrame.loc[range(int(size/2),size)]
    print(trainingData.shape, trainingLabel.shape, testData.shape, testLabel.shape)
    return trainingData, np.asarray(trainingLabel).flatten(), \
           testData, np.asarray(testLabel).flatten()


# dataFilter('class', data) examining output

def calBaseLine(data):
    classValues = np.unique(data)
    highest = 0
    baseClass = ""

    for label in classValues:
        count = len(data[data==label])
        if count > highest:
            highest = count
            baseClass = label
    print("Base class: ", baseClass)
    print("Base Line: ",(float(highest)/len(data))*100)

def calAccuracy(testLabel, predictLabel):
    count = 0
    for i in range(len(testLabel)):
        if testLabel[i] in predictLabel[i]:
            count += 1
    print ("Accuracy = ", (float(count)/len(testLabel)*100))

zScoreNorm('class', data)

training, label, test, testLabel = dataFilter('class', data)

logreg = linear_model.LogisticRegression(C=.09, n_jobs=-1)

calBaseLine(testLabel)

logreg.fit(training, label)

predictedLabel = logreg.predict(test)

calAccuracy(testLabel, predictedLabel)

# Problem 2

data = pd.read_csv("~/PycharmProjects/DSOR/clones/DSOR/Resources/iris.csv")
print(data.head())
data['class'] = data['class'].replace(['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], ['1', '0', '0'])


def zScoreNorm(targetCol, data):
    for i in data.columns:
        if i == targetCol:
            continue
        data[i] = preprocessing.scale(data[i])


def dataFilter(targetColumn, data):
    columnName = [col for col in data.columns if col not in targetColumn]
    dataFrame = data[columnName]
    labelFrame = data[[targetColumn]]
    size = len(dataFrame)
    trainingData = dataFrame.loc[range(1,int(size/2))]
    trainingLabel = labelFrame.loc[range(1,int(size/2))]
    testData = dataFrame.loc[range(int(size/2),size)]
    testLabel = labelFrame.loc[range(int(size/2),size)]
    print(trainingData.shape, trainingLabel.shape, testData.shape, testLabel.shape)
    return trainingData, np.asarray(trainingLabel).flatten(), \
           testData, np.asarray(testLabel).flatten()


# dataFilter('class', data) examining output

# Problem 2 - 5 best features.

