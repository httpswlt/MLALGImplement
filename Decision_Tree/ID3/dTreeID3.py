# coding:utf-8
import math

# https://www.jb51.net/article/146584.htm
# https://blog.csdn.net/csqazwsxedc/article/details/65697652
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


def count_nums(classify):
    classify_count = {}
    for cls in classify:
        if cls not in classify_count.keys():
            classify_count.setdefault(cls, 0)
        classify_count[cls] += 1
    result = sorted(classify_count.items(), key=lambda d: d[1], reverse=True)
    return result[0][0]


def calculate_entropy(datasets):
    # calculate the entropy of dataSet
    data_num = len(datasets)
    label_count = {}
    for data in datasets:
        label = data[-1]
        if label not in label_count.keys():
            label_count.setdefault(label, 0)
        label_count[label] += 1
    entropy = 0.0
    for key in label_count.keys():
        p_i = float(label_count[key] / data_num)
        entropy -= p_i * math.log(p_i, 2)
    return entropy


def split_datasets(datasets, feature_index, feature):
    new_datasets = []
    for data in datasets:
        if data[feature_index] == feature:
            temp = data[:feature_index]
            temp.extend(data[feature_index+1:])
            new_datasets.append(temp)
    return new_datasets


def select_best_tree(datasets):
    feature_num = len(datasets[0]) - 1
    datasets_entropy = calculate_entropy(datasets)

    max_info_gain = 0.0
    selected_best_feature_index = -1
    for i in range(feature_num):
        feature_list = [temp[i] for temp in datasets]
        unique_feature = set(feature_list)
        new_feature_entropy = 0.0
        for feature in unique_feature:
            sub_data = split_datasets(datasets, i, feature)
            prob = len(sub_data) / float(len(datasets))
            new_feature_entropy += prob*calculate_entropy(sub_data)
        info_gain = datasets_entropy - new_feature_entropy
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            selected_best_feature_index = i
    return selected_best_feature_index


def decision_tree_ID3(datasets, feature):
    classify = [info[-1] for info in datasets]

    if classify.count(classify[0]) == len(classify):
        return classify[0]
    if len(datasets[0]) == 1:
        return count_nums(classify)

    best_feature_index = select_best_tree(datasets)
    best_feature = feature[best_feature_index]
    del(feature[best_feature_index])
    decision_tree = {best_feature: {}}
    best_feature_value = [temp[best_feature_index] for temp in datasets]
    unique_best_feature_value = set(best_feature_value)
    for feature_value in unique_best_feature_value:
        copy_feature = feature[:]
        decision_tree[best_feature][feature_value] = decision_tree_ID3\
            (split_datasets(datasets, best_feature_index, feature_value), copy_feature)
    return decision_tree


def main():
    datasets, feature = load_data()
    print(decision_tree_ID3(datasets, feature))


if __name__ == '__main__':
    main()
