# coding:utf-8
import numpy as np
from sklearn.datasets import load_iris
import random


class KNN(object):
    def __init__(self, x, y, n_neighbors=1, dist_func='l1'):
        self.n_neighbors = n_neighbors
        if dist_func == 'l1':
            self.dist_func = self.l1_distance
        else:
            self.dist_func = self.l2_distance
        self.x_train = x
        self.y_train = y

    def pred(self, x):
        y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)
        for i, x_test in enumerate(x):
            distance = self.dist_func(self.x_train, x_test)
            nn_index = np.argsort(distance)
            nn_pred = self.y_train[nn_index][:self.n_neighbors].ravel()
            y_pred[i] = np.argmax(np.bincount(nn_pred))
        return y_pred

    @staticmethod
    def l1_distance(a, b):
        return np.sum(np.abs(a - b), axis=1)

    @staticmethod
    def l2_distance(a, b):
        return np.sqrt(np.sum((a - b)**2, axis=1))


def load_data():
    iris = load_iris()
    x = iris.data
    y = iris.target.reshape(-1, 1)
    return x, y


def train_test_split(x, y, test_ratio):
    data_nums = len(x)
    temp = range(0, data_nums)
    test_index = random.sample(temp, int(data_nums*test_ratio))
    train_index = list(set(temp).difference(set(test_index)))
    train_data = x[train_index]
    train_label = y[train_index]
    test_data = x[test_index]
    test_label = y[test_index]
    return train_data, train_label, test_data, test_label


def accuracy_score(pred, label):
    temp = (pred - label).ravel().tolist()
    correct_nums = temp.count(0)
    return correct_nums / float(pred.shape[0])


def main():
    x, y = load_data()
    train_data, train_label, test_data, test_label = train_test_split(x, y, 0.3)
    knn = KNN(train_data, train_label, 9, 'l2')
    pred = knn.pred(test_data)
    print(accuracy_score(pred, test_label))


if __name__ == '__main__':
    main()
