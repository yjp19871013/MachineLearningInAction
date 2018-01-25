import numpy as np
import operator


def create_data_set():
    group = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1],
    ])

    labels = ['A', 'A', 'B', 'B']

    return group, labels


def classify0(in_x, data_set, labels, k=3):
    # Calculate distance
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5

    #Sort and get indicies
    sorted_dist_indicies = distances.argsort()

    #Find the first k labels and vote
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1

    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]


def file2matrix(filename, col_num=3):
    fr = open(filename)
    lines = fr.readlines()
    lines_num = len(lines)
    ret_mat = np.zeros((lines_num, col_num))
    class_label_vector = []
    index = 0
    for line in lines:
        line = line.strip()
        list_from_line = line.split('\t')
        ret_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1

    return ret_mat, class_label_vector


def auto_norm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals

    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))

    return norm_data_set, ranges, min_vals


def dating_class_test(dating_data_mat, dating_labels, ho_radio=0.10, k=3):
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_radio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
                                      dating_labels[num_test_vecs:m], k)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print("the total error rate is: %f and error count is: %f" % (error_count / float(num_test_vecs), error_count))


def first_demo():
    group, lables = create_data_set()
    print(classify0([0, 0], group, lables))


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input('percentage of time spent playing video games?'))
    ff_miles = float(input('frequent flier miles earned per year?'))
    ice_cream = float(input('liters if ice cream consumed per year?'))
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = np.array([ff_miles, percent_tats, ice_cream])

    # Remember to normalize the in_arr
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat, dating_labels)
    print("You will probably like this person: ", result_list[classifier_result - 1])


if __name__ == '__main__':
    classify_person()
