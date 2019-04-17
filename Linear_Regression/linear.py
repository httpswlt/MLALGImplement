# coding:utf-8
# inference:https://github.com/Shicoder/DeepLearning_Demo/tree/master/linear_regression_use_gradient_decent
import numpy as np
import matplotlib.pyplot as plt




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
    initial_b = 0.0
    initial_w = 0.0
    num_iter = 1000
    plt.scatter(x, y)
    plt.show()
    # print 'initial variables:\n initial_b = {0}\n intial_m = {1}\n error of begin = {2} \n' \
    #     .format(initial_b, initial_m, compute_error(initial_b, initial_m, data))


if __name__ == '__main__':
    Linear_regression()


