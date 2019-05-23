import cv2
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt


def sign(y):
    if y > 0:
        return 1
    else:
        return -1


def visual(x, y):
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)
    # plt.show()


x, y = make_blobs(100, n_features=2, centers=2, random_state=2)

# define model: y = w * x + b
# define default w and b
w = np.zeros(x.shape[-1])
b = 0
lr = 0.1

# define arithmetic
continue_flag = True
epoch = 0
while continue_flag:
    continue_flag = False
    epoch += 1
    step = 0
    print("epoch is {}".format(epoch))
    for features, label in zip(x, y):
        label = 1 if label else -1
        # define strategy(can be called loss function)
        pred = sign(np.dot(w, features.transpose()) + b)
        # y' representative label , y representative predict value.
        # define arithmetic: positive: y' * y > 0 negative y' * y < 0
        if label * pred < 0:
            w = w + lr * (label * features)
            b = b + lr * label
            continue_flag = True
            step += 1
            print("step is {}".format(step))


def line(x, w, b):
    return -(w[0] / w[1]) * x - b / w[1]


visual(x, y)
plt.plot((x.min(), x.max()), (line(x.min(), w, b), line(x.max(), w, b)))
plt.show()








