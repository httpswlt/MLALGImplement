# coding:utf-8
import os
import sys
sys.path.append('../../MLALGImplement')
print(sys.path)
import pandas as pd
import random
import math
from Decision_Tree.ID3.dTreeID3 import decision_tree_ID3

def load_data():
    df = pd.read_csv('./data/housing.txt')
    return df


def random_datasets(datasets):
    n, m = datasets.shape
    features = random.sample(list(datasets.columns.values[:-1]), int(math.sqrt(m - 1)))
    features.append(datasets.columns.values[-1])
    rows = [random.randint(0, n-1) for _ in range(n)]
    train_data = datasets.iloc[rows][features]
    return train_data.values.tolist(), features



def main():
    datasets = load_data()
    tree_nums = 10
    for i in range(tree_nums):
        sample_data, sample_feature = random_datasets(datasets)
        pass



if __name__ == '__main__':
    main()