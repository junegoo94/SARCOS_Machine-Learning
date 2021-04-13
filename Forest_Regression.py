import random
import copy
import scipy
import scipy.optimize
import numpy as np
import csv
import matplotlib.pyplot as plt


class Tree(object):
    def __init__(self, error, predict, std, start, num_points):
        self.error = error
        self.predict = predict
        self.std = std

        self.split_var = None
        self.split_val = None
        self.split_lab = None
        self.left = None
        self.right = None
        self.num_points = num_points

    def lookup(self, x):
        ### Finding the predicted value
        if self.left == None:
            return self.predict

        if x[self.split_var] <= self.split_val:
            return self.left.lookup(x)
        return self.right.lookup(x)

    def predict_all(self, data):
        ### Finding the predicted values for some list of data
        return map(lambda x: self.lookup(x), data)

    def smallest_alpha(self):
        ### Finding the smallest value of alpha
        if self.right == None:
            return float("inf"), [self]
        b_error, num_nodes = self.cost()
        alpha = (self.error - b_error) / (num_nodes - 1)
        alpha_right, tree_right = self.right.smallest_alpha()
        alpha_left, tree_left = self.left.smallest_alpha()
        if smallest_alpha == alpha:
            smallest_trees.append(self)
        if smallest_alpha == alpha_right:
            smallest_trees = smallest_trees + tree_right
        if smallest_alpha == alpha_left:
            smallest_trees = smallest_trees + tree_left
        return smallest_alpha, smallest_trees

    def prune_tree(self):
        ### Finding alpha and trees -> choose the right size of tree
        trees = [copy.deepcopy(self)]
        alphas = [0]
        new_tree = copy.deepcopy(self)
        while 1:
            alpha, nodes = new_tree.smallest_alpha()
            for node in nodes:
                node.right = None
                node.left = None
            trees.append(copy.deepcopy(new_tree))
            alphas.append(alpha)
            # when reached root node
            if node.start == True:
                break
            return alphas, trees

    def cost(self):
        ### Finding the branch error and number of nodes
        if self.right == None:
            return self.error, 1
        error, num_nodes = self.right.cost()
        left_error, left_num = self.left.cost()
        error = error + left_error
        num_nodes = num_nodes + left_num
        return error, num_nodes

    def length(self):
        ### Finding the length of the tree
        if self.right == None:
            return 1
        right_length = self.right.length()
        left_length = self.left.length()
        return max(right_length, left_length) + 1


def grow_tree(data, depth, max_depth=500, Nmin=5, labels={}, start=False, feature_bagging=False):
    ### Function to get a regression tree
    root = Tree(mean_sq(list(data.values())), np.mean(np.array(list(data.values()))),
                np.std(np.array(list(data.values()))), start, len(list(data.values())))

    # regions < Nmin data
    if len(data.values()) <= Nmin:
        return root

    # length of tree > max_depth
    if depth >= max_depth:
        return root
    num_vars = len(list(data.keys())[0])

    min_error = -1
    min_split = -1
    split_var = -1

    if feature_bagging:
        # maximum features
        max_features = random.sample(range(num_vars), int(num_vars ** (0.5)))
        # max_features = random.sample(range(num_vars), int(num_vars))
        # print('max_features:', max_features)
    else:
        max_features = range(num_vars)
    # iterating
    for i in max_features:
        var_space = [x[i] for x in data]
        if min(var_space) == max(var_space):
            continue
        # finding oprimal split point
        split, error, ierr, numf = scipy.optimize.fminbound(error_function, min(var_space), max(var_space),
                                                            args=(i, data), full_output=1)
        # minimising error
        if ((error < min_error) or (min_error == -1)):
            min_error = error
            min_split = split
            split_var = i

    if split_var == -1:
        return root

    root.split_var = split_var
    root.split_val = min_split
    if split_var in labels:
        root.split_lab = labels[split_var]
    # print('labels:', labels)
    data1 = {}
    data2 = {}
    for i in data:
        if i[split_var] <= min_split:
            data1[i] = data[i]
        else:
            data2[i] = data[i]
    root.left = grow_tree(data1, depth + 1, max_depth=max_depth, Nmin=Nmin, labels=labels,
                          feature_bagging=feature_bagging)
    root.right = grow_tree(data2, depth + 1, max_depth=max_depth, Nmin=Nmin, labels=labels,
                           feature_bagging=feature_bagging)
    return root


def mean_sq(data):
    ### sum of squared error
    data = np.array(data)
    mean_squared_error = np.sum((data - np.mean(data)) ** 2) / len(data)
    return mean_squared_error


def error_function(split_point, split_var, data):
    ### to choose split point
    data1 = []
    data2 = []
    for i in data:
        if i[split_var] <= split_point:
            data1.append(data[i])
        else:
            data2.append(data[i])
    return mean_sq(data1) + mean_sq(data2)


class Forest(object):
    def __init__(self, trees):
        self.trees = trees

    def lookup(self, x):
        ###finding the predicted value
        predict = map(lambda t: t.lookup(x), self.trees)
        return np.mean(list(predict))

    def predict_all(self, data):
        ### Finding the predicted values for some list of data
        return map(lambda x: self.lookup(x), data)


def bootstrap(pairs, n):
    ### building bootstrap
    index = np.random.choice(n, size=n, replace=True)
    # print('index:', index)
    return dict(map(lambda x: pairs[x], index))


def make_forest(data, B, max_depth=500, Nmin=5, labels={}):
    ### random forest
    ### B is for bootstrap B -> number of trees
    trees = []
    n = len(data)
    pairs = list(data.items())
    for b in range(B):
        boot = bootstrap(pairs, n)
        trees.append(
            grow_tree(boot, 0, max_depth=max_depth, Nmin=Nmin, labels=labels, start=True, feature_bagging=True))
    return Forest(trees)


##############################################################################
'''Toy Problem'''

f = open("toy_prob_4.csv", "rt", encoding='UTF-8-sig')
lines = f.readlines()
f.close()
train_dict = {}
test_dict = {}
labels = dict()
for i in range(2):
    labels[i] = str(i)

total_len = len(lines)
# print(total_len)
ratio = 0.85
train_len = int(total_len * ratio)
# print(train_len)
for i in range(0, len(lines)):
    tmp = lines[i].strip().split(",")
    dat = []
    for j in range(2):
        # print('tmp', float(tmp[j]))

        dat.append(float(tmp[j]))
    res = float(tmp[2])
    if i < train_len:
        train_dict[tuple(dat)] = res
    else:
        test_dict[str(i)] = [tuple(dat), res]

MSE_lst = []
B_lst = []
for B in range(1, 50):
    B_lst.append(B)
    forest = make_forest(train_dict, B, max_depth=1000, Nmin=1, labels=labels)

    error = []
    predictive = []
    actual = []
    for i in test_dict.keys():
        predict = forest.lookup(list(test_dict[i][0]))
        predictive.append(predict)
        act_val = test_dict[i][1]
        actual.append(act_val)
        error.append(((act_val - predict) * (act_val - predict)) ** 0.5)

    print('Actual Value: ', actual)
    print('predictive: ', predictive)
    print()
    print('MSE: ', np.mean(np.array(error)))
    print()
    print()
    MSE_lst.append(np.mean(np.array(error)))

##############################################################################
'''Experiment'''
# plot the cost
print("reading csv data")

f = open("pre_processed_data.csv", "r")
lines = f.readlines()
f.close()
train_dict = {}
test_dict = {}
labels = dict()
for i in range(21):
    labels[i] = str(i)

total_len = len(lines)
ratio = 0.8
train_len = int(total_len * ratio)

for i in range(0, len(lines)):
    tmp = lines[i].strip().split(",")
    dat = []
    for j in range((21)):
        dat.append(float(tmp[j]))
    res = float(tmp[21])
    if i < train_len:
        train_dict[tuple(dat)] = res
    else:
        test_dict[str(i)] = [tuple(dat), res]

print(tmp)
print(len(train_dict))
print(test_dict)
B = 50
print("fitting")
forest = make_forest(train_dict, B, max_depth=100, Nmin=5, labels=labels)

print("saving log")

error = []
predictive = []
for i in test_dict.keys():
    predict = forest.lookup(list(test_dict[i][0]))
    predictive.append(predict)
    act_val = test_dict[i][1]
    error.append(((act_val - predict) * (act_val - predict)) ** 0.5)

print("mse")
print(np.mean(np.array(error)))
print('predictive', predictive)