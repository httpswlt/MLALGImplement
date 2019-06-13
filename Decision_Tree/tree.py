# coding:utf-8
import math


class Tree(object):
    def __init__(self, datasets, feature):
        """
        :param feature: feature names set,format [name_1, name_2,...]
        :param train_data:format: [[name_1.value, name_2.value,,..., label],
                                   [name_1.value, name_2.value,,..., label],
                                   ...]
        """
        self.datasets = datasets
        self.feature = feature

    @staticmethod
    def calculate_entropy(datasets):
        datasets_len = len(datasets)
        # calculate the total nums for every classification
        label_count = {}
        for data in datasets:
            label = data[-1]
            if label not in label_count.keys():
                label_count.setdefault(label, 1)
            else:
                label_count[label] += 1

        # calculate the entropy of the whole data
        datasets_entropy = 0
        for label in label_count.keys():
            pi = float(label_count[label]) / datasets_len
            datasets_entropy -= pi * math.log(pi, 2)
        return datasets_entropy

    @staticmethod
    def split_datasets(datas, feature_value, feature_idx):
        """
            remove feature_value dim.
        :param datas:
        :param feature_value:
        :param feature_idx:
        :return:
        """
        new_datasets = []
        for data in datas:
            if data[feature_idx] == feature_value:
                temp = data[:feature_idx]   # left data
                temp.extend(data[feature_idx+1:])   # right data
                new_datasets.append(temp)
        return new_datasets




    @staticmethod
    def split_datasets_double(datas, feature_value, feature_idx):
        d1 = []
        d2 = []
        for data in datas:
            if data[feature_idx] == feature_value:
                d1.append(data)
            else:
                d2.append(data)
        return [d1, d2]

    def select_best_tree_point(self, train_data=None, train_feature=None):
        pass

    def build_tree(self, train_data=None, train_feature=None):
        pass

    def pred(self, test_data, tree):
        pass
