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


def classify0(in_x, data_set, labels, k):
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


if __name__ == '__main__':
    group, lables = create_data_set()
    print(classify0([0, 0], group, lables, 3))
