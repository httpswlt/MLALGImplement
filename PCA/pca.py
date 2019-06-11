# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    return np.array(pd.read_csv("data.txt", sep='\t', header=None)).astype(np.float32)


def mean(data):
    return np.mean(data, axis=0)


def pca(x, k):
    average = mean(x)
    row, col = np.shape(x)
    normal_x = x - average
    cov_x = np.cov(normal_x.T)
    feature_value, feature_vector = np.linalg.eig(cov_x)
    index = np.argsort(-feature_value)
    if k > col:
        print("k must lower than feature number.")
        return
    select_vec = np.matrix(feature_vector.T[index[:k]])
    final_data = normal_x * select_vec.T
    recon_data = final_data * select_vec + average
    return final_data, recon_data


def plot(data1, data2):
    data1 = np.array(data1)
    data2 = np.array(data2)
    row = np.shape(data1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(row):
        axis_x1.append(data1[i, 0])
        axis_y1.append(data1[i, 1])
        axis_x2.append(data2[i, 0])
        axis_y2.append(data2[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    # plt.savefig("outfile.png")
    plt.show()


def main():
    x = load_data()
    k = 2
    final_data, recon_data = pca(x, k)
    plot(final_data, recon_data)


if __name__ == '__main__':
    main()
