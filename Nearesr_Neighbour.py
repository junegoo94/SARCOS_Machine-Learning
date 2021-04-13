import math
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


class KNN():
    def fitPredict(self, X_train, X_test, y_train, y_test, k):

        predictions = []
        for x in range(len(X_test)):
            neighbours = self.getNeighbours(X_train, X_test[x], y_train, k)
            result = self.predict(neighbours, k)
            predictions.append(result)
        accuracy = self.getMSE(y_test, predictions)
        # print('predictions :', predictions)
        print('MSE :', accuracy)
        return accuracy

    def euclideanDistance(self, instance1, instance2, length):

        distance = 0
        for x in range(length):
            distance = distance + pow((instance1[x] - instance2[x]), 2)

        return math.sqrt(distance)

    def getNeighbours(self, trainingSet, testInstance, training_out, k):
        dists = []
        length = len(testInstance) - 1
        for x in range(len(trainingSet)):
            # print('y_train', training_out)
            dist = self.euclideanDistance(testInstance, trainingSet[x], length)
            dists.append((trainingSet[x], training_out[x], dist))
        dists.sort(key=operator.itemgetter(2))

        neighbours = []
        for i in range(k):
            neighbours.append(dists[i][1])
        # print('neighbours = ', neighbours)
        return neighbours

    def predict(self, neighbours, k):
        sum = 0
        for x in range(len(neighbours)):
            response = neighbours[x]
            sum = sum + response
        return float(sum / float(k))

    def getMSE(self, testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            # print('test=', testSet[x])
            correct += pow((testSet[x] - predictions[x]), 2) / float(len(testSet))
        # return (pow(correct, 0.5))
        return correct


###############################################################################
''' Toy Problem '''
file = 'toy_prob_4.csv'
with open(file, 'rt', encoding='UTF-8') as data:
    reader = csv.reader(data)
    toy_data = []
    for row in reader:
        toy_data.append(row)


def toy(data):
    x_data = []
    y_data = []
    for d in data:
        x = d[0:3]
        y = d[2]
        x = np.array(x)
        y = np.asarray(y)
        x = x.reshape(3, 1)

        x_data.append(x)
        y_data.append(y)

    split = int(len(data) * 0.85)
    X_train = x_data[:split]
    X_test = x_data[split:]

    y_train = y_data[:split]
    y_test = y_data[split:]

    # print(X_test)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')
    y_train = y_train.astype('float64')
    y_test = y_test.astype('float64')

    return X_train, X_test, y_train, y_test


x_toy_train, x_toy_test, y_toy_train, y_toy_test = toy(toy_data)

print('Actual Value :', y_toy_test)

obj = KNN()
print()
print('K = 1')
print('Expectations: [2.5, 5.5, 8.5]')
k1 = obj.fitPredict(x_toy_train, x_toy_test, y_toy_train, y_toy_test, 1)
print()
print('K = 2')
print('Expectations: [3.0, 6.0, 9.0]')
k2 = obj.fitPredict(x_toy_train, x_toy_test, y_toy_train, y_toy_test, 2)

###########################################################################
'''Experiment'''

file = 'pre_processed_data.csv'
with open(file, 'rt', encoding='UTF-8') as data:
    reader = csv.reader(data)
    sarcos_data = []
    for row in reader:
        sarcos_data.append(row)


def experiment_data(data):
    x_data = []
    y_data = []
    full_data = sarcos_data  # [:1000]
    for d in full_data:
        x = d[0:21]
        y = d[21]
        x = np.array(x)
        y = np.asarray(y)
        x = x.reshape(21, 1)
        x_data.append(x)
        y_data.append(y)

    split = int(len(full_data) * 0.8)
    X_train = x_data[:split]
    X_test = x_data[split:]

    y_train = y_data[:split]
    y_test = y_data[split:]

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')
    y_train = y_train.astype('float64')
    y_test = y_test.astype('float64')

    return X_train, X_test, y_train, y_test


X_sar_train, X_sar_test, y_sar_train, y_sar_test = experiment_data(sarcos_data)

mse_lst = []
k_lst = []
for k in range(1, 10):
    print(k)
    k_lst.append(k)
    obj = KNN()
    mse = obj.fitPredict(X_sar_train, X_sar_test, y_sar_train, y_sar_test, k)
    mse_lst.append(rsme)
    print()

# plot the cost
fig, ax = plt.subplots()
ax.plot(k_lst, rsme_lst, 'r')
ax.set_xlabel('K- Value')
ax.set_ylabel('Mean Squared Error')
# plt.savefig("knn_sarcos_MSE.png")

optimal_k = k_lst[mse_lst.index(min(msee_lst))]
print('Optimal K value :', optimal_k)
