import numpy as np
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as SCSP
import scipy
from scipy.sparse.linalg import svds


def load_matrix(fname, shape):
    row_lis, col_lis, val_lis = [], [], []
    with open(fname) as f:
        for line in f:
            if line:
                vals = line.strip().split()
                try:
                    assert len(vals) == 3
                except AssertionError:
                    print('Wrong format in input file')
                    return
                else:
                    row, col, val = vals
                    row_lis.append(int(row))
                    col_lis.append(int(col))
                    val_lis.append(float(val))

                    
    r, c, v = np.array(row_lis), np.array(col_lis), np.array(val_lis)
    mat = SCSP.csc_matrix((v, (r, c)), shape=shape)
    assert scipy.sparse.isspmatrix_csc(mat), "IMPOSSIBLE: matrix m should always be csc, but it is NOT"
    return mat


def svd_randomized_sklearn(mat, k):
    assert k <= min(mat.shape), "The input k is too big"
    return randomized_svd(mat, n_components=k)


def svd_scipy(mat, k):
    assert k is None or k <= min(mat.shape), "The value of input k is wrong"
    if k is None:
        k = min(mat.shape)
    u, s, vh = svds(mat, k=k)
    sorted_indices = np.argsort(s)[::-1]
    return u[:, sorted_indices], s[sorted_indices], vh[sorted_indices, :]
