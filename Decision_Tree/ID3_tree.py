# coding:utf-8
from .tree import Tree


class ID3(Tree):
    def __init__(self, datasets, feature):
        Tree.__init__(self, datasets, feature)
        self.tree = None

    def select_best_tree_point(self, train_data=None, train_feature=None):
        if not train_feature:
            train_feature = self.feature
        if not train_data:
            train_data = self.datasets

        entropy = self.calculate_entropy(train_data)
        best_feature_point_index = -1
        max_info_gain = 0
        for feature_name in train_feature:
            feature_idx = train_feature.index(feature_name)
            feature_values = [temp[feature_idx] for temp in train_data]
            unique_feature_value = set(feature_values)
            new_feature_entropy = 0
            for feature_value in unique_feature_value:
                new_datasets = self.split_datasets(train_data, feature_value, feature_idx)
                prop = float(len(new_datasets)) / len(train_data)
                new_feature_entropy += prop * self.calculate_entropy(new_datasets)
            info_gain = entropy - new_feature_entropy
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature_point_index = feature_idx
        return best_feature_point_index

    @staticmethod
    def count_nums(classify):
        classify_count = {}
        for cls in classify:
            if cls not in classify_count.keys():
                classify_count.setdefault(cls, 0)
            classify_count[cls] += 1
        result = sorted(classify_count.items(), key=lambda d: d[1], reverse=True)
        return result[0][0]

    def build_tree(self, train_data=None, train_feature=None):
        if not train_feature:
            train_feature = self.feature
        if not train_data:
            train_data = self.datasets

        classification = [info[-1] for info in train_data]

        # all data classifications are the same
        if classification.count(classification[0]) == len(classification):
            return classification[0]

        # when the point is the last.
        if len(train_data[0]) == 1:
            return self.count_nums(classification)

        # acquire the max entropy of feature as the root.
        best_feature_indx = self.select_best_tree_point(train_data, train_feature)
        # acquire what the feature name
        best_feature_name = train_feature[best_feature_indx]
        best_feature_value = [temp[best_feature_indx] for temp in train_data]
        unique_best_feature_value = set(best_feature_value)
        tree = {best_feature_name: {}}
        # delete the root point, then start calculate next feature
        del(train_feature[best_feature_indx])
        # start branch calculation
        for feature_value in unique_best_feature_value:
            train_feature_copy = train_feature[:]
            train_data_copy = self.split_datasets(train_data, feature_value, best_feature_indx)
            tree[best_feature_name][feature_value] = self.build_tree(train_data_copy, train_feature_copy)
        self.tree = tree
        return tree

    def pred(self, test_data=None, tree=None):
        if not tree:
            tree = self.tree
        if not isinstance(tree, dict):
            return tree
        first_feature = list(tree.keys())[0]
        compare_data = test_data[first_feature]
        assert_error = 100
        best_branch = None
        for feature_value in list(tree[first_feature].keys()):
            err = abs(feature_value - compare_data)
            if err <= assert_error:
                best_branch = feature_value
                assert_error = err

        sub_tree = tree[first_feature][best_branch]
        label = self.pred(test_data, sub_tree)
        return label


def load_data():
    datasets = [['long', 'coarse', 'male'],
                ['short', 'coarse', 'male'],
                ['short', 'coarse', 'male'],
                ['long', 'thin', 'female'],
                ['short', 'thin', 'female'],
                ['short', 'coarse', 'female'],
                ['long', 'coarse', 'female'],
                ['long', 'coarse', 'female']]
    feature = ['hair', 'sound']
    return datasets, feature


def main():
    train_data, feature_names = load_data()
    id3_tree = ID3(train_data, feature_names)
    print(id3_tree.build_tree())


if __name__ == '__main__':
    main()
