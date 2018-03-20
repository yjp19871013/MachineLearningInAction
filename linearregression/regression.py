import numpy as np


def load_data_set(filename):
    num_feat = len(open(filename).readline().split('\t')) - 1
    data_mat = []
    label_mat = []
    with open(filename) as fr:
        for line in fr.readlines():
            line_arr = []
            cur_line = line.strip().split('\t')
            for i in range(num_feat):
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)
            label_mat.append(float(cur_line[-1]))
        return data_mat, label_mat


def stand_regres(x_arr, y_arr):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    x_tx = x_mat.T * x_mat
    if np.linalg.det(x_tx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return

    ws = x_tx.I * (x_mat.T * y_mat)
    return ws


def demo1():
    x_arr, y_arr = load_data_set('ex0.txt')
    ws = stand_regres(x_arr, y_arr)

    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])

    x_copy = x_mat.copy()
    x_copy.sort(0)
    y_hat = x_copy * ws
    ax.plot(x_copy[:, 1], y_hat)

    plt.show()

    print(np.corrcoef(y_hat.T, y_mat))


def lwlr(test_point, x_arr, y_arr, k=1.0):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    m = np.shape(x_mat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff_mat = test_point - x_mat[j, :]
        weights[j, j] = np.exp(diff_mat * diff_mat.T / -2.0 * k ** 2)
    x_tx = x_mat.T * (weights * x_mat)
    if np.linalg.det(x_tx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return

    ws = x_tx.I * (x_mat.T * (weights * y_mat))
    return test_point * ws


def lwlr_test(test_arr, x_arr, y_arr, k):
    m = np.shape(test_arr)[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return y_hat


def demo2():
    x_arr, y_arr = load_data_set('ex0.txt')
    y_hat = lwlr_test(x_arr, x_arr, y_arr, 0.003)
    x_mat = np.mat(x_arr)
    srt_ind = x_mat[:, 1].argsort(0)
    x_sort = x_mat[srt_ind][:, 0]

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_sort[:, 1], y_hat[srt_ind])
    ax.scatter(x_mat[:, 1].flatten().A[0], np.mat(y_arr).T.flatten().A[0], s=2, c='red')
    plt.show()


def ridge_regres(x_mat, y_mat, lam=0.2):
    x_tx = x_mat.T * x_mat
    denom = x_tx + np.eye(np.shape(x_mat)[1]) * lam

    # if lam == 0.0
    if np.linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws


def ridge_test(x_arr, y_arr):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_mat = regularize(x_mat)
    num_test_pts = 30
    w_mat = np.zeros((num_test_pts, np.shape(x_mat)[1]))
    for i in range(num_test_pts):
        ws = ridge_regres(x_mat, y_mat, np.exp(i-10))
        w_mat[i, :] = ws.T
    return w_mat


def regularize(x_mat):
    x_means = np.mean(x_mat, 0)
    x_var = np.var(x_mat, 0)
    return (x_mat - x_means) / x_var


def demo3():
    x_arr, y_arr = load_data_set('abalone.txt')
    ridge_weights = ridge_test(x_arr, y_arr)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_weights)
    plt.show()


def rss_error(y_arr, y_hat_arr):
    return ((y_arr - y_hat_arr) ** 2).sum()


def stage_wise(x_arr, y_arr, eps=0.01, num_it=100):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_mat = regularize(x_mat)
    m, n = np.shape(x_mat)
    return_mat = np.zeros((num_it, n))
    ws = np.zeros((n, 1))
    ws_max = ws.copy()
    for i in range(num_it):
        print(ws.T)
        lowest_error = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps * sign
                y_test = x_mat * ws_test
                rss_e = rss_error(y_mat.A, y_test.A)
                if rss_e < lowest_error:
                    lowest_error = rss_e
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T

    return return_mat


def demo4():
    x_arr, y_arr = load_data_set('abalone.txt')
    print(stage_wise(x_arr, y_arr, 0.01, 200))


if __name__ == '__main__':
    demo4()
