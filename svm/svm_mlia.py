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


class OptStruct:
    def __init__(self, data_mat_in, class_labels, c, toler):
        self.x = data_mat_in
        self.label_mat = class_labels
        self.c = c
        self.tol = toler
        self.m = np.shape(data_mat_in)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.ecache = np.mat(np.zeros((self.m, 2)))


def calc_ek(opt_struct, k):
    fxk = float(np.multiply(opt_struct.alphas, opt_struct.label_mat).T *
                (opt_struct.x * opt_struct.x[k, :].T)) + opt_struct.b
    ek = fxk - float(opt_struct.label_mat[k])
    return ek


def select_j(i, opt_struct, ei):
    max_k = -1
    max_delta_e = 0
    ej = 0
    valid_ecache_list = np.nonzero(opt_struct.ecache[:, 0].A)[0]
    if len(valid_ecache_list) > 1:
        for k in valid_ecache_list:
            if k == i:
                continue
            ek = calc_ek(opt_struct, k)
            delta_e = abs(ei - ek)
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                ej = ek
        return max_k, ej
    else:
        j = select_jrand(i, opt_struct.m)
        ej = calc_ek(opt_struct, j)
        return j, ej


def inner_l(i, opt_struct):
    ei = calc_ek(opt_struct, i)
    if (opt_struct.label_mat[i] * ei < -opt_struct.tol and opt_struct.alphas[i] < opt_struct.c) or \
            (opt_struct.label_mat[i] * ei > opt_struct.tol and opt_struct.alphas[i] > 0):
        j, ej = select_j(i, opt_struct, ei)
        alpha_i_old = opt_struct.alphas[i].copy()
        alpha_j_old = opt_struct.alphas[j].copy()
        if opt_struct.label_mat[i] != opt_struct.label_mat[j]:
            l = min(0, opt_struct.alphas[j] - opt_struct.alphas[i])
            h = max(opt_struct.c, opt_struct.c + opt_struct.alphas[j] - opt_struct.alphas[i])
        else:
            l = min(0, opt_struct.alphas[i] + opt_struct.alphas[j] - opt_struct.c)
            h = max(opt_struct.c, opt_struct.alphas[i] + opt_struct.alphas[j])
        if l == h:
            print('l == h')
            return 0

        eta = 2.0 * opt_struct.x[i, :] * opt_struct.x[j, :].T - \
            opt_struct.x[i, :] * opt_struct.x[i, :].T - \
            opt_struct.x[j, :] * opt_struct.x[j, :].T
        if eta >= 0:
            print('eta>=0')
            return 0

        opt_struct.alphas[j] = opt_struct.alphas[j] - opt_struct.label_mat[j] * (ei - ej) / eta
        opt_struct.alphas[j] = clip_alpha(opt_struct.alphas[j], h, l)
        update_ek(opt_struct, j)
        if abs(opt_struct.alphas[j] - alpha_j_old) < 0.00001:
            print('j not moving enough')
            return 0

        opt_struct.alphas[i] = opt_struct.alphas[i] + opt_struct.label_mat[j] * \
                            opt_struct.label_mat[j] * (alpha_j_old - opt_struct.alphas[j])
        update_ek(opt_struct, i)
        b1 = opt_struct.b - ei - opt_struct.label_mat[i] * (opt_struct.alphas[i] - alpha_i_old) * \
            opt_struct.x[i, :] * opt_struct.x[i, :].T - \
            opt_struct.label_mat[j] * (opt_struct.alphas[j] - alpha_j_old) * \
            opt_struct.x[i, :] * opt_struct.x[j, :].T
        b2 = opt_struct.b - ei - opt_struct.label_mat[i] * (opt_struct.alphas[i] - alpha_j_old) * \
            opt_struct.x[i, :] * opt_struct.x[j, :].T - \
            opt_struct.label_mat[j] * (opt_struct.alphas[j] - alpha_j_old) * \
            opt_struct.x[j, :] * opt_struct.x[j, :].T
        if (opt_struct.alphas[i] > 0) and (opt_struct.c > opt_struct.alphas[i]):
            opt_struct.b = b1
        elif (opt_struct.alphas[j] > 0) and (opt_struct.c > opt_struct.alphas[j]):
            opt_struct.b = b2
        else:
            opt_struct.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smo_p(data_mat_in, class_labels, c, toler, max_iter, k_tup=('lin', 0)):
    os_struct = OptStruct(np.mat(data_mat_in), np.mat(class_labels).T, c, toler)
    iter = 0
    entire_set = True
    alpha_pairs_changed = 0
    while (iter < max_iter) and ((alpha_pairs_changed > 0) or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(os_struct.m):
                alpha_pairs_changed += inner_l(i, os_struct)
                print('fillSet, iter: %d i: %d, pairs changed %d' % (iter, i, alpha_pairs_changed))
            iter += 1
        else:
            non_bound_is = np.nonzero((os_struct.alphas.A > 0) * (os_struct.alphas.A < c))[0]
            for i in non_bound_is:
                alpha_pairs_changed += inner_l(i, os_struct)
                print('non-bound, iter: %d i: %d, pairs changed %d' % (iter, i, alpha_pairs_changed))
            iter += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True
        print('iteration number: %d' % iter)
    return os_struct.b, os_struct.alphas


def update_ek(os, k):
    ek = calc_ek(os, k)
    os.ecache[k] = [1, ek]


def calc_ws(alphas, data_arr, class_labels):
    x = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(x)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * label_mat[i], x[i, :].T)
    return w


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


def demo3():
    data_arr, label_arr = load_data_set('testSet.txt')
    b, alphas = smo_p(data_arr, label_arr, 0.6, 0.001, 40)
    print(b)
    print(alphas[alphas > 0])
    print(np.shape(alphas))
    for i in range(100):
        if alphas[i] > 0.0:
            print((data_arr[i], label_arr[i]))


if __name__ == '__main__':
    demo3()
