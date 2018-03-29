import numpy as np


def load_data_set(filename):
    data_mat = []
    with open(filename) as fr:
        for line in fr.readlines():
            cur_line = line.strip().split('\t')
            flt_line = list(map(float, cur_line))
            data_mat.append(flt_line)
    return data_mat


def bin_spilt_data_set(data_set, feature, value):
    mat0 = np.mat([])
    mat1 = np.mat([])

    nonzero0 = np.nonzero(data_set[:, feature] > value)[0]
    if len(nonzero0) > 0:
        mat0 = data_set[nonzero0, :]

    nonzero1 = np.nonzero(data_set[:, feature] <= value)[0]
    if len(nonzero1) > 0:
        mat1 = data_set[nonzero1, :]

    return mat0, mat1


def reg_leaf(data_set):
    return np.mean(data_set[:, -1])


def reg_err(data_set):
    return np.var(data_set[:, -1]) * np.shape(data_set)[0]


def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    feat, val = choose_best_spilt(data_set, leaf_type, reg_err, ops)
    if feat is None:
        return val

    ret_tree = dict()
    ret_tree['spInd'] = feat
    ret_tree['spVal'] = val
    l_set, r_set = bin_spilt_data_set(data_set, feat, val)
    ret_tree['left'] = create_tree(l_set, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(r_set, leaf_type, err_type, ops)

    return ret_tree


def choose_best_spilt(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    tol_s = ops[0]
    tol_n = ops[1]
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(data_set)

    m, n = np.shape(data_set)
    s = err_type(data_set)
    best_s = np.inf
    best_index = 0
    best_value = 0
    for feat_index in range(n - 1):
        for spilt_val in set(data_set[:, feat_index].T.tolist()[0]):
            mat0, mat1 = bin_spilt_data_set(data_set, feat_index, spilt_val)
            if (np.shape(mat0)[0] < tol_n) or (np.shape(mat1)[0] < tol_n) or \
                    (np.shape(mat0)[1] == 0) or (np.shape(mat1)[1] == 0):
                continue
            new_s = err_type(mat0) + err_type(mat1)
            if new_s < best_s:
                best_index = feat_index
                best_value = spilt_val
                best_s = new_s

    if (s - best_s) < tol_s:
        return None, leaf_type(data_set)

    mat0, mat1 = bin_spilt_data_set(data_set, best_index, best_value)
    if (np.shape(mat0)[0] < tol_n) or (np.shape(mat1)[0] < tol_n) or \
            (np.shape(mat0)[1] == 0) or (np.shape(mat1)[1] == 0):
        return None, leaf_type

    return best_index, best_value


def is_tree(obj):
    return type(obj) is dict


def get_mean(tree):
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])

    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])

    return (tree['left'] + tree['right']) / 2


def prune(tree, test_data):
    if np.shape(test_data)[0] == 0:
        return get_mean(tree)

    if is_tree(tree['right']) or is_tree(tree['left']):
        l_set, r_set = bin_spilt_data_set(test_data, tree['spInd'], tree['spVal'])

    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], l_set)

    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], r_set)

    if not is_tree(tree['left']) and not is_tree(tree['right']):
        l_set, r_set = bin_spilt_data_set(test_data, tree['spInd'], tree['spVal'])

        error_no_merge = 0
        if l_set.size != 0:
            error_no_merge += np.sum(np.power(l_set[:, -1] - tree['left'], 2))
        if r_set.size != 0:
            error_no_merge += np.sum(np.power(r_set[:, -1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right']) / 2
        error_merge = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
        if error_merge < error_no_merge:
            print('merging')
            return tree_mean
        else:
            return tree

    else:
        return tree


def linear_solve(data_set):
    m, n = np.shape(data_set)
    x = np.mat(np.ones((m, n)))
    y = np.mat(np.ones((m, 1)))
    x[:, 1:n] = data_set[:, 0:n-1]
    y = data_set[:, -1]
    x_tx = x.T * x
    if np.linalg.det(x_tx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, \n')
    ws = x_tx.T * (x.T * y)
    return ws, x, y


def model_leaf(data_set):
    ws, x, y = linear_solve(data_set)
    return ws


def model_err(data_set):
    ws, x, y = linear_solve(data_set)
    y_hat = x * ws
    return np.sum(np.power(y - y_hat, 2))


def demo1():
    my_data = load_data_set('ex00.txt')
    print(create_tree(np.mat(my_data)))

    my_data1 = load_data_set('ex0.txt')
    print(create_tree(np.mat(my_data1)))


def demo2():
    my_data = load_data_set('ex2.txt')
    my_tree = create_tree(np.mat(my_data))

    my_test_data = load_data_set('ex2test.txt')
    print(prune(my_tree, np.mat(my_test_data)))


def demo3():
    my_data = load_data_set('exp2.txt')
    print(create_tree(np.mat(my_data), model_leaf, model_err, (1, 10)))


if __name__ == '__main__':
    demo3()
