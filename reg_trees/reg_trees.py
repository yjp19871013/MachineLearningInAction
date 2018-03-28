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


def demo1():
    my_data = load_data_set('ex00.txt')
    print(create_tree(np.mat(my_data)))

    my_data1 = load_data_set('ex0.txt')
    print(create_tree(np.mat(my_data1)))


if __name__ == '__main__':
    demo1()