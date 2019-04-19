# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    x = []
    y = []
    with open('dataset.csv') as f:
        for line in f.readlines():
            lineArr = line.strip().split(',')
            x.append([float(lineArr[0]), float(lineArr[1])])
            y.append(int(lineArr[2]))
    return (x, y)


def sigmoid(x):
    # activate function
    return 1 / (1 + np.exp(-x))


def forward_propagrate(x, w, b):
    # forward propagate
    pred = sigmoid(np.dot(x, w) + b)
    return pred


def loss_function(pred, y):
    # loss function
    N = pred.shape[0]
    cost = -np.sum(np.multiply(y, np.log(pred)) + np.multiply((1 - y), np.log(1 - pred))) / N
    return cost


def comput_gradient(x, pred, y):
    # back propagate
    N = x.shape[0]
    w_gradient = np.sum(np.dot(np.transpose(x), pred - y), 1) / N
    b_gradient = np.sum(pred - y) / N
    return w_gradient, b_gradient


def optimize(x, y, w, b, lr, iters):
    # fit the two classify
    for i in range(iters):
        pred = forward_propagrate(x, w, b)
        w_gradient, b_gradient = comput_gradient(x, pred, y)
        w = w - lr * w_gradient
        b = b - lr * b_gradient
        if iters % 10 == 0:
            print("iter nums : {0}, loss : {1}".format(i, loss_function(pred, y)))
    return w, b


def plt_data(datas, w=np.array([[1], [1], [1]]), b=0):
    p_x, p_y = [], []
    n_x, n_y = [], []
    for data, label in zip(datas[0], datas[1]):
        if label == 1:
            p_x.append(data[0])
            p_y.append(data[1])
        else:
            n_x.append(data[0])
            n_y.append(data[1])
    plt.scatter(p_x, p_y, s=30, c='red', marker='s')
    plt.scatter(n_x, n_y, s=30, c='blue', marker='o')
    x = np.arange(-3.0, 3.0, 0.1)
    # set sigmoid value is 1/2, so f(x) = -(w0x0 + w1x1 + b) = 0,
    # so x2=(−w0x0−w1x1-b)/w2 ,so y = (w0*x - b) / w1
    w = w.tolist()
    y = -(w[0][0] * x + b) / w[1][0]
    plt.plot(x, y)
    plt.show()


def Logist_Regression(data):
    lr = 0.1
    iters = 1000
    x = np.mat(data[0])
    y = np.mat(data[1]).transpose()
    row, col = x.shape
    init_w = np.ones((col, 1))
    init_b = 0
    w, b = optimize(x, y, init_w, init_b, lr, iters)
    plt_data(data, w, b)


def main():
    data = load_data()
    # plt_data(data)
    Logist_Regression(data)


if __name__ == '__main__':
    main()
























