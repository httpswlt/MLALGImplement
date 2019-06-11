from ID3_tree import ID3
import math


class C45(ID3):
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
            feature_self_entropy = 0
            for feature_value in unique_feature_value:
                new_datasets = self.split_datasets(train_data, feature_value, feature_idx)
                prop = float(len(new_datasets)) / len(train_data)
                new_feature_entropy += prop * self.calculate_entropy(new_datasets)
                feature_self_entropy -= prop * math.log(prop, 2)
            info_gain = entropy - new_feature_entropy
            # add C4.5 arithmetic
            info_fain_ratio = info_gain / feature_self_entropy
            if info_fain_ratio > max_info_gain:
                max_info_gain = info_fain_ratio
                best_feature_point_index = feature_idx
        return best_feature_point_index


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
    c45_tree = C45(train_data, feature_names)
    print(c45_tree.build_tree())


if __name__ == '__main__':
    main()