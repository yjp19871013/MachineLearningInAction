import random
import numpy as np


def load_data_set(filename):
    data_mat = []
    label_mat = []
    with open(filename) as fr:
        for line in fr.readlines():
            line_arr = line.strip().split('\t')
            data_mat.append([float(line_arr[0]), float(line_arr[1])])
            label_mat.append(float(line_arr[2]))

    return data_mat, label_mat


def select_jrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))

    return j


def clip_alpha(aj, h, l):
    if aj > h:
        aj = h

    if l > aj:
        aj = l

    return aj


def smo_simple(data_mat_in, class_labels, c, toler, max_iter):
    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).T
    b = 0
    m, n = np.shape(data_matrix)
    alphas = np.zeros((m, 1))
    iter = 0
    while iter < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            fxi = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
            ei = fxi - float(label_mat[i])
            if (label_mat[i] * ei < -toler and alphas[i] < c) or (label_mat[i] * ei > toler and alphas[i] > 0):
                j = select_jrand(i, m)
                fxj = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                ej = fxj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:
                    l = min(0, alphas[j] - alphas[i])
                    h = max(c, c + alphas[j] - alphas[i])
                else:
                    l = min(0, alphas[i] + alphas[j] - c)
                    h = max(c, alphas[i] + alphas[j])
                if l == h:
                    print('l == h')
                    continue

                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - \
                    data_matrix[i, :] * data_matrix[i, :].T - \
                    data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    print('eta>=0')
                    continue

                alphas[j] = alphas[j] - label_mat[j] * (ei - ej) / eta
                alphas[j] = clip_alpha(alphas[j], h, l)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print('j not moving enough')
                    continue
                alphas[i] = alphas[i] + label_mat[j] * label_mat[j] * (alpha_j_old - alphas[j])
                b1 = b - ei - label_mat[i] * (alphas[i] - alpha_i_old) * \
                     data_matrix[i, :] * data_matrix[i, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * \
                     data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - ei - label_mat[i] * (alphas[i] - alpha_j_old) * \
                     data_matrix[i, :] * data_matrix[j, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * \
                     data_matrix[j, :] * data_matrix[j, :].T
                if (alphas[i] > 0) and (c > alphas[i]):
                    b = b1
                elif (alphas[j] > 0) and (c > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                    alpha_pairs_changed += 1
                    print("iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed))

            if alpha_pairs_changed == 0:
                iter += 1
            else:
                iter = 0
            print("iteration number: %d" % iter)

    return b, alphas


def demo1():
    data_arr, label_arr = load_data_set('testSet.txt')
    print(data_arr)
    print(label_arr)


def demo2():
    data_arr, label_arr = load_data_set('testSet.txt')
    b, alphas = smo_simple(data_arr, label_arr, 0.6, 0.001, 40)
    print(b)
    print(alphas[alphas > 0])
    print(np.shape(alphas))
    for i in range(100):
        if alphas[i] > 0.0:
            print((data_arr[i], label_arr[i]))


if __name__ == '__main__':
    demo2()
