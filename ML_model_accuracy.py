def mse(actual, prediction):
    errors = prediction - actual
    rss = sum(errors ** 2)
    MSE = rss / len(prediction)
    print('Model Performance')
    print('MSE:', MSE)

    return MSE


def mae(actual, prediction):
    errors = prediction - actual
    rss = sum(abs(errors))
    MAE = rss / len(prediction)
    print('Model Performance')
    print('MAE:', MAE)

    return MAE

import pandas as pd
import numpy as np
dataset = pd.read_csv('pre_processed_data.csv', header = None)
dataset = pd.DataFrame(dataset)

# dataset = dataset.sample(frac=1).reset_index(drop=True)# shuffle the rows
ratio = 0.8
split = int(len(dataset) * 0.8)
train = dataset[:split]
test = dataset[split:]

#####################################################################
######################### LR Accuracy ###############################
actual = test.iloc[:, -1]
actual_array = np.array(actual)
expectation = actual_array.reshape(actual_array.shape[0], 1)
pred = pd.read_csv('predictive_LR.csv', header = None)
pred_LR = np.array(pred)

MSE_LR = mse(expectation, pred_LR)
MAE_LR = mae(expectation, pred_LR)

#####################################################################
######################### KNN Accuracy ##############################
actual = test.iloc[:, -1]
actual_array = np.array(actual)
expectation = actual_array.reshape(actual_array.shape[0], 1)
pred = pd.read_csv('predictive_KNN.csv', header = None)
pred_KNN = np.array(pred)

MSE_KNN = mse(expectation, pred_KNN)
MAE_KNN = mae(expectation, pred_KNN)

#####################################################################
#################### Forest Regression Accuracy #####################
actual = test.iloc[:, -1]
actual_array = np.array(actual)
expectation = actual_array.reshape(actual_array.shape[0], 1)
pred = pd.read_csv('predictive_RF.csv', header = None)
pred_RF = np.array(pred)

MSE_RF = mse(expectation, pred_RF)
MAE_RF = mae(expectation, pred_RF)

#####################################################################
#################### Gaussian Process Accuracy ######################
GP_dataset = pd.DataFrame(dataset[:1000])

# dataset = dataset.sample(frac=1).reset_index(drop=True)# shuffle the rows
ratio = 0.8
split = int(len(GP_dataset) * 0.8)
train = GP_dataset[:split]
test = GP_dataset[split:]
GP_dataset.head()

actual = test.iloc[:, -1]
actual_array = np.array(actual)
expectation = actual_array.reshape(actual_array.shape[0], 1)
pred = pd.read_csv('predictive_GPR.csv', header = None)
pred_RF = np.array(pred)

MSE_RF = mse(expectation, pred_RF)
MAE_RF = mae(expectation, pred_RF)