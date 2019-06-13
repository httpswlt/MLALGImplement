from tree import Tree


class CART(Tree):
    def __init__(self, datasets, feature):
        super(CART, self).__init__(datasets, feature)
        # self.nums_feature = len(feature)
        self.d_normal = len(datasets)
        self.dichotomy_1 = datasets[0][-1]

    def build_tree(self, train_data=None, train_feature=None):

        for i, feature_name in enumerate(self.feature):
            self.calculate_ginibyfeature_name(feature_name)

            # self.calculate_gini_dichotomy(self.datasets, new_data)

    def calculate_ginibyfeature_name(self, feature_name):
        i = self.feature.index(feature_name)
        feature_values = set([data[i] for data in self.datasets])

        for feature_value in feature_values:
            new_data = self.split_datasets_double(self.datasets, feature_value, i)
            d1 = new_data[0]
            d2 = new_data[1]
            prop_1 = len(d1) / self.d_normal
            prop_2 = len(d2) / self.d_normal
            gini1 = self.calculate_gini_dichotomy(d1)
            gini2 = self.calculate_gini_dichotomy(d2)
            gini_value = prop_1 * gini1 + prop_2 * gini2
            print(gini_value)

    def calculate_gini_dichotomy(self, datasets):
        d_normal = len(datasets)
        label_count = {}
        for data in datasets:
            label = data[-1]
            count = label_count.get(label)
            if count:
                count += 1
            else:
                count = 1
            label_count[label] = count
        p = float(label_count.get(self.dichotomy_1, 0)) / d_normal
        gini = 2 * p * (1 - p)
        return gini

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
    c45_tree.build_tree()


if __name__ == '__main__':
    main()