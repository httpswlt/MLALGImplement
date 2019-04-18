# coding:utf-8
https://github.com/yingzk/MyML/blob/master/1-Logistic%20Regession/Logistic%20%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%95%E5%8F%8APython%E5%AE%9E%E7%8E%B0.md
import numpy as np

def load_data():
    x = []
    y = []
    with open('dataset.csv') as f:
        for line in f.readlines():
            lineArr = line.strip().split(',')
            x.append([1.0, float(lineArr[0]), float(lineArr[1])])
            y.append(int(lineArr[2]))
    return x, y


def sigmoid(x):
    # activate function
    return 1 / (1 + np.exp(-x))


def forward_propagrate(x, w, b):
    # forward propagate
    return sigmoid(w * x + b)


def loss_function(pred, y):
    # loss function
    N = 10
    cost = -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)) / N
    return cost


def comput_gradient(x, pred, y):
    # back propagate
    N = 10
    w_gradient = np.dot(x, np.transpose(pred - y)) / N
    b_gradient = np.sum(pred) / N
    return w_gradient, b_gradient


def optimize(x, w, b, lr, y, iters):
    # fit the two classify
    for i in range(iters):
        pred = forward_propagrate(x, w, b)
        w_gradient, b_gradient = comput_gradient(x, pred, y)
        w = w - lr * w_gradient
        b = b - lr * b_gradient

def main():
    load_data()

if __name__ == '__main__':
    main()

























