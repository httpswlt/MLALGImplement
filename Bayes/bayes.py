# coding:utf-8
import numpy as np


def load_data():
    words_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                  ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1 bad, 0 good
    class_vec = [0, 1, 0, 1, 0, 1]
    return words_list, class_vec


def data2list(data):
    features = []
    for feature in data:
        features.extend(feature)
    return list(set(features))


def word2vect(all_words, input_words):
    vects = [0] * len(all_words)
    for word in input_words:
        word_index = all_words.index(word)
        vects[word_index] += 1
    return vects


def train(train_data, label):
    classify_nums = len(label)

    all_features_nums = len(train_data[0])
    label0_vect = np.zeros(all_features_nums)
    label1_vect = np.zeros(all_features_nums)

    label0_count = 0
    label1_count = 0
    for i in range(classify_nums):
        if label[i] == 1:
            label1_count += 1
            label1_vect += train_data[i]
        else:
            label0_count += 1
            label0_vect += train_data[i]

    # calculate the probability of every feature
    label0_prob = label0_vect / label0_count
    label1_prob = label1_vect / label1_count
    p_1 = sum(label) / float(classify_nums)
    return label0_prob, label1_prob, p_1


def pred(test_data, p0_vec, p1_vec, p_1):
    p1 = sum(test_data * p1_vec) + np.log(p_1)
    p0 = sum(test_data * p0_vec) + np.log(1 - p_1)
    if p0 > p1:
        return 0
    return 1


def main():
    doc_list, label = load_data()
    all_words_vec = data2list(doc_list)
    train_vec = list(map((lambda x: word2vect(all_words_vec, x)), doc_list))
    p0_vec, p1_vec, p_1 = train(train_vec, label)
    test_entry = ['love', 'my', 'dalmation']
    test_vect = word2vect(all_words_vec, test_entry)
    print pred(test_vect, p0_vec, p1_vec, p_1)
    test_entry = ['stupid', 'garbage']
    test_vect = word2vect(all_words_vec, test_entry)
    print pred(test_vect, p0_vec, p1_vec, p_1)


if __name__ == '__main__':
    main()
