# coding:utf-8
import pandas as pd
import random
import math
import sys
import collections
sys.path.append('../../MLALGImplement')
from Decision_Tree.ID3_tree import ID3


def load_data():
    df = pd.read_csv('./data/wine.txt', header=None)
    return df


def random_datasets(datasets):
    n, m = datasets.shape
    features = random.sample(list(datasets.columns.values[:-1]), int(math.sqrt(m - 1)))
    features.append(datasets.columns.values[-1])
    rows = [random.randint(0, n-1) for _ in range(n)]
    train_data = datasets.iloc[rows][features]
    train_data.drop_duplicates(inplace=True)
    return train_data.values.tolist(), features


def main():
    datasets = load_data()
    forest = []
    tree_nums = 10
    for i in range(tree_nums):
        sample_data, sample_feature = random_datasets(datasets)
        id3 = ID3(sample_data, sample_feature)
        id3.build_tree()
        forest.append(id3)

    label_pred = []
    for id3 in forest:
        test_data = [12, 0.92, 2, 19, 86, 2.42, 2.26, 0.3, 1.43, 2.5, 1.38, 3.12, 278]
        label = id3.pred(test_data)
        label_pred.append(label)

    label = collections.Counter(label_pred)
    print(label)


if __name__ == '__main__':
    main()
