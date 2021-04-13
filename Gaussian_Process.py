
import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt


def kernel(X1, X2, length=1.0, sigma_f=1.0):

    sqdist = np.sum(X 1* *2, 1).reshape(-1, 1) + np.sum(X 2* *2, 1) - 2 * np.dot(X1, X2.T)
    ker = sigma_ f* *2 * np.exp(-0.5 / lengt h* *2 * sqdist)
    # print(ker)
    return ker


def posterior_predictive(X_s, X_train, Y_train, length=1.0, sigma_f=1.0, sigma_y=1e-8):

    K = kernel(X_train, X_train, length, sigma_f) + sigma_ y* *2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, length, sigma_f)
    K_ss = kernel(X_s, X_s, length, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)


    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s

###############################################################################
'''Toy Problem'''

def toy_data(file_name):
    dataset = pd.read_csv(file_name, header = None)

    ratio = 0.85
    split = int(len(dataset) * ratio)
    train = dataset[:split]
    test = dataset[split:]

    # setting the matrixes
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1].values  # .values converts it from pandas.core.frame.DataFrame to numpy.ndarray
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1].values

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    return X_train, X_test, y_train, y_test

X_train_toy, X_test_toy, y_train_toy, y_test_toy = toy_data('toy_prob.csv')

noise = 0.1
mu_s, cov_s = posterior_predictive(X_test_toy, X_train_toy, y_train_toy, sigma_y=noise)


##############################################################################
'''Experiment'''

def sarcos_data():
    dataset = pd.read_csv('pre_processed_data.csv', header = None)
    dataset[:10]
    ratio = 0.8
    split = int(len(dataset) * ratio)
    train = dataset[:split]
    test = dataset[split:]

    # setting the matrixes
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1].values  # .values converts it from pandas.core.frame.DataFrame to numpy.ndarray
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1].values

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    return X_train, X_test, y_train, y_test


X_train_sar, X_test_sar, y_train_sar, y_test_sar = sarcos_data()

