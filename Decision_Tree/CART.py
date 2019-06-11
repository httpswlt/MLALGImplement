from tree import Tree


class CART(Tree):
    def __init__(self, datasets, feature):
        super(CART, self).__init__(datasets, feature)
        self.nums_feature = len(feature)


    def build_tree(self, train_data=None, train_feature=None):

        for i in range(self.nums_feature):
            new_data = self.split_datasets_double(self.datasets, self.feature[i], i)
            self.calculate_gini_dichotomy(self.datasets, new_data)


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
    c45_tree = CART(train_data, feature_names)
    print(c45_tree.build_tree())


if __name__ == '__main__':
    main()