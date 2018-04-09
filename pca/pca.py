import numpy as np


def load_data_set(filename, delim='\t'):
    with open(filename) as fr:
        string_arr = [line.strip().split(delim) for line in fr.readlines()]
        dat_arr = [list(map(float, line)) for line in string_arr]
    return np.mat(dat_arr)


def pca(data_mat, top_n_feat=9999999):
    mean_vals = np.mean(data_mat, axis=0)
    mean_removed = data_mat - mean_vals
    cov_mat = np.cov(mean_removed, rowvar=0)
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
    eig_val_ind = np.argsort(eig_vals)
    eig_val_ind = eig_val_ind[:-(top_n_feat):-1]
    red_eig_vects = eig_vects[:, eig_val_ind]
    low_d_data_mat = mean_removed * red_eig_vects
    recon_mat = (low_d_data_mat * red_eig_vects.T) + mean_vals
    return low_d_data_mat, recon_mat


def demo1():
    data_mat = load_data_set('testSet.txt')
    low_d_mat, recon_mat = pca(data_mat, 1)
    print(np.shape(low_d_mat))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], marker='o', s=50)
    plt.show()


if __name__ == '__main__':
    demo1()