import numpy as np


def load_simp_data():
    dat_mat = np.mat([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return dat_mat, class_labels


def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    ret_array = np.ones((np.shape(data_matrix)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0

    return ret_array


def build_stump(data_arr, class_labels, D):
    data_matrix = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_matrix)
    num_steps = 10.0
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf
    for i in range(n):
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:
                thresh_val = range_min + step_size * float(j)
                predicted_vals = stump_classify(data_matrix, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                weighted_error = D.T * err_arr
                #print('splite: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f' %
                #      (i, thresh_val, inequal, weighted_error))
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal

    return best_stump, min_error, best_class_est


def ada_boost_train_DS(data_arr, class_labels, num_it=40):
    weak_class_arr = []
    m = np.shape(data_arr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)
        #print('D: ', D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        #print('classEst: ', class_est.T)
        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        agg_class_est += alpha * class_est
        #print('aggClassEst: ', agg_class_est.T)
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        #print('total error: ', error_rate)
        if error_rate == 0.0:
            break

    return weak_class_arr


def ada_classify(data_to_class, classifier_arr):
    data_matrix = np.mat(data_to_class)
    m = np.shape(data_matrix)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(data_matrix, classifier_arr[i]['dim'],
                                   classifier_arr[i]['thresh'],
                                   classifier_arr[i]['ineq'])
        agg_class_est += classifier_arr[i]['alpha'] * class_est
        print(agg_class_est)

    return np.sign(agg_class_est)


def demo1():
    dat_mat, class_labels = load_simp_data()
    D = np.mat(np.ones((5, 1)) / 5)
    print(build_stump(dat_mat, class_labels, D))


def demo2():
    dat_mat, class_labels = load_simp_data()
    ada_boost_train_DS(dat_mat, class_labels, 9)


def demo3():
    dat_mat, class_labels = load_simp_data()
    classifier_arr = ada_boost_train_DS(dat_mat, class_labels, 9)
    print(ada_classify([[0, 0], [5, 5]], classifier_arr))


if __name__ == '__main__':
    demo3()
