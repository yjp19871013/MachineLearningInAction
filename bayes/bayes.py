import numpy as np


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]    # 1 is abusive, 0 not
    return posting_list, class_vec


def create_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in input_set:
            return_vec[vocab_list.index(word)] = 1

    return return_vec


def bag_of_words2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in input_set:
            return_vec[vocab_list.index(word)] += 1

    return return_vec


def trainNB0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])

    p1_vect = np.log(p1_num / p1_denom)
    p0_vect = np.log(p0_num / p0_denom)
    return p0_vect, p1_vect, p_abusive


def classifyNB(vec2classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec2classify * p1_vec) + np.log(p_class1)
    p0 = sum(vec2classify * p0_vec) + np.log(1.0 - p_class1)

    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    list_post, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_post)
    train_mat = []
    for postin_doc in list_post:
        train_mat.append(set_of_words2vec(my_vocab_list, postin_doc))

    p0_v, p1_v, p_ab = trainNB0(np.array(train_mat), np.array(list_classes))

    test_entry = ['love', 'my', 'dalmation']
    this_doc = np.array(set_of_words2vec(my_vocab_list, test_entry))
    print(str(test_entry) + ' classified as: ' + str(classifyNB(this_doc, p0_v, p1_v, p_ab)))

    test_entry = ['stupid', 'garbage']
    this_doc = np.array(set_of_words2vec(my_vocab_list, test_entry))
    print(str(test_entry) + ' classified as: ' + str(classifyNB(this_doc, p0_v, p1_v, p_ab)))


def demo1():
    list_post, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_post)
    print(set_of_words2vec(my_vocab_list, list_post[0]))
    print(set_of_words2vec(my_vocab_list, list_post[1]))


if __name__ == '__main__':
    testingNB()