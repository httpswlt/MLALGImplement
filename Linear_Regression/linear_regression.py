# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt


def model(x, w, b):
    # define  y =wx+b
    return w * x + b


def loss_function(x, y, w, b):
    return np.sum(np.square(y - model(x, w, b))) / len(x)


def optimizer(x, y, w, b, lr, iters):
    for i in range(iters):
        w, b = compute_gradient(x, y, w, b, lr)
        if i % 10 == 0:
            print ("iter {0} : loss={1}".format(i, loss_function(x, y, w, b)))
    return w, b


def compute_gradient(x, y, w, b, lr):
    N = float(len(x))
    w_gradient = np.sum(-2 * (y - model(x, w, b)) * x) / N
    b_gradient = np.sum(-2 * (y - model(x, w, b))) / N
    w -= lr * w_gradient
    b -= lr * b_gradient
    return w, b


def linear_regression():
    x = [13854, 12213, 11009, 10655, 9503]
    x = np.reshape(x, newshape=(5, 1)) / 10000.0
    y = [21332, 20162, 19138, 18621, 18016]
    y = np.reshape(y, newshape=(5, 1)) / 10000.0
    # set hyper-parameters
    lr = 0.1
    b = 0
    w = 0
    num_iter = 2000
    w, b = optimizer(x, y, w, b, lr, num_iter)
    print ('final formula parmaters:\n w={1}\n b={2}\n error={3} \n'.
           format(num_iter, w, b, loss_function(x, y, w, b)))
    plt.title("lr : {0}".format(lr))
    plt.scatter(x, y)
    plt.plot(x, model(x, w, b))
    plt.show()


if __name__ == '__main__':
    linear_regression()


