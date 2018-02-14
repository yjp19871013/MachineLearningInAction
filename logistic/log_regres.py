import numpy as np
import random


def load_data_set():
    data_mat = []
    label_mat = []

    fr = open('testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))

    return data_mat, label_mat


def sigmoid(in_x):
    return 1.0 / (1 + np.exp(-in_x))


def grad_ascent(data_in_mat, class_labels):
    data_matrix = np.mat(data_in_mat)
    label_mat = np.mat(class_labels).T

    m, n = np.shape(data_matrix)
    alpha = 0.001
    max_cycle = 500
    weights = np.ones((n, 1))
    for k in range(max_cycle):
        h = sigmoid(data_matrix * weights)
        error = label_mat - h
        weights = weights + alpha * data_matrix.T * error

    return weights


def plot_best_fit(weights):
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_data_set()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stoc_grad_ascent0(data_matrix, class_labels):
    m, n = np.shape(data_matrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]

    return weights


def stoc_grad_ascent1(data_matrix, class_labels, num_iter = 150):
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            del(data_index[rand_index])

    return weights


def demo1():
    data_arr, label_mat = load_data_set()
    print(grad_ascent(data_arr, label_mat))


def demo2():
    data_arr, label_mat = load_data_set()
    weights = grad_ascent(data_arr, label_mat)
    plot_best_fit(weights.getA())


def demo3():
    data_arr, label_mat = load_data_set()
    weights = stoc_grad_ascent0(np.array(data_arr), label_mat)
    plot_best_fit(weights)


def demo4():
    data_arr, label_mat = load_data_set()
    weights = stoc_grad_ascent1(np.array(data_arr), label_mat)
    plot_best_fit(weights)


if __name__ == '__main__':
    demo4()
