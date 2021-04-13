import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math


def cost(X, y, theta):
    Y = y.reshape(y.shape[0], 1)
    error = np.dot(X, theta.transpose()) - Y
    summation = np.sum(np.power(error, 2))
    cost_result = summation / (len(X))
    return cost_result


# gradient descent
def gradientDescent(X, y, theta, iters, alpha):
    Y = y.reshape(y.shape[0], 1)
    costs = np.zeros(iters)
    for i in range(iters):
        error = np.dot(X, theta.transpose()) - Y
        theta = np.array(theta - (alpha / len(X)) * np.sum(X * error, axis=0))
        costs[i] = cost(X, y, theta)
        if costs[i] < 0.075:
            print('Iteration: ', i)
            break

    return theta, costs, i


############################################################################
''' Toy problem Implementation '''

dataset = pd.read_csv('toy_prob_4.csv', header=None)
dataset = pd.DataFrame(dataset)
train = dataset[:17]
test = dataset[17:]

# setting the matrixes
X_train = train.iloc[:, :-1]
ones = np.ones([X_train.shape[0], 1])
X_train = np.concatenate((ones, X_train), axis=1)

y_train = train.iloc[:, -1].values  # .values converts it from pandas.core.frame.DataFrame to numpy.ndarray
theta = np.zeros([1, len(train.columns)])

# set hyper parameters
alpha = 0.01
iters = 10

# running the gd and cost function
coefficient_toy, cost_toy, iteration_toy = gradientDescent(X_train, y_train, theta, iters, alpha)
coefficient_toy.tolist
print('Coefficients :', coefficient_toy)
print('MSE:', cost_toy[iteration_toy])

# plot the cost
fig, ax = plt.subplots()
ax.plot(np.arange(iters), cost_toy, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Mean Squared Error')
# plt.savefig("toy_Mean Squared Error.png")

prediction = test[1] * coefficient_toy[0][1] + test[1] * coefficient_toy[0][2] + coefficient_toy[0][0]
predictions = [prediction[17], prediction[18], prediction[19]]
expectations = [3, 6, 9]

print('Expectations: ', expectations)
print('Predictions: ', predictions)
print()
print('MSE: ', cost_toy[iteration_toy])

############################################################################
'''Experiment'''

dataset = pd.read_csv('pre_processed_data.csv', header=None)
dataset = pd.DataFrame(dataset)

# dataset = dataset.sample(frac=1).reset_index(drop=True)# shuffle the rows
ratio = 0.8
split = int(len(dataset) * 0.8)
train = dataset[:split]
test = dataset[split:]

# setting the matrixes
X_train = train.iloc[:, :-1]
ones = np.ones([X_train.shape[0], 1])
X_train = np.concatenate((ones, X_train), axis=1)

y_train = train.iloc[:, -1].values  # .values converts it from pandas.core.frame.DataFrame to numpy.ndarray

X_test = test.iloc[:, :-1]
ones = np.ones([X_test.shape[0], 1])
X_test = np.concatenate((ones, X_test), axis=1)

y_test = test.iloc[:, -1].values  # .values converts it from pandas.core.frame.DataFrame to numpy.ndarray

theta = np.zeros([1, len(train.columns)])
# set hyper parameters
alpha = 0.01
iters = 2682

# running the gd and cost function
coefficient_sarcos, cost_sarcos, iter_sarcos = gradientDescent(X_train, y_train, theta, iters, alpha)
coefficient_sarcos.tolist
print('Coefficients :', coefficient_sarcos)
print('MSE:', cost_sarcos[-1])

# plot the cost
fig, ax = plt.subplots()
ax.plot(np.arange(iters), cost_sarcos, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Mean Squared Error')
# plt.savefig("LR_sarcos_MSE.png")

actual_data = y_test.reshape(y_test.shape[0], 1)
predictive = np.dot(X_test, coefficient_sarcos.transpose())
predictive = pd.DataFrame(predictive)
predictive.to_csv('predictive_LR.csv', index=False, header=False)

