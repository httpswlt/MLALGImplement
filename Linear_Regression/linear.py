# coding:utf-8
# inference:https://github.com/Shicoder/DeepLearning_Demo/tree/master/linear_regression_use_gradient_decent
import numpy as np
import matplotlib.pyplot as plt

def model(x, w, b):
    return w * x + b

def loss_function(x, y, w, b):
    return np.sum(np.square(y - model(x, w, b))) / len(x)

def optimizer(x, y, w, b, lr, iters):
    b_gradient = 0
    w_gradient = 0
    N = float(len(x))

    for i in range(iters):
        pass






def compute_gradient(data,w, b, lr):
    w_gradient = 0
    b_gradient = 0

    N = float(len(data[0]))
    x = data[0]
    y = data[1]






def Linear_regression():
    x = [13854, 12213, 11009, 10655, 9503]  #程序员工资，顺序为北京，上海，杭州，深圳，广州
    x = np.reshape(x, newshape=(5, 1)) / 10000.0
    y = [21332, 20162, 19138, 18621, 18016]
    y = np.reshape(y, newshape=(5, 1)) / 10000.0
    # define hyperparamters
    # learning_rate is used for update gradient
    # defint the number that will iteration
    # define  y =wx+b
    learning_rate = 0.001
    b = 0.0
    w = 1
    num_iter = 1000
    # plt.scatter(x, y)
    # plt.show()
    loss_function(x,y,w,b)
    # print 'initial variables:\n initial_b = {0}\n intial_m = {1}\n error of begin = {2} \n' \
    #     .format(initial_b, initial_m, compute_error(initial_b, initial_m, data))




if __name__ == '__main__':
    Linear_regression()


