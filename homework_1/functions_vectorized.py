import numpy as np


def prod_non_zero_diag(X):
    diag = X.diagonal().copy()
    diag[diag == 0] = 1
    return diag.prod()


def are_multisets_equal(x, y):
    return np.array_equal(np.sort(x), np.sort(y))


def max_after_zero(x):
    y = np.roll(x, 1)
    y[0] = -1
    try:
        return x[y == 0].max()
    except:
        return None


def convert_image(arr, coefs):
    return np.dot(arr, coefs)


def run_length_encoding(x):
    y = np.cumsum(np.ones(len(x)))
    d = np.hstack([np.array([1]), np.diff(x)])
    val = x[d != 0]
    dist = np.diff(np.hstack([y[d != 0], np.array([len(x) + 1])])).astype('int64')
    return (val, dist)


def pairwise_distance(X, Y):
    xx = np.diagonal(np.dot(X, X.T)).reshape(len(X), 1)
    yy = np.diagonal(np.dot(Y, Y.T)).reshape(1, len(Y))
    xy = np.dot(X, Y.T) * (-2)
    xy += xx
    xy += yy
    return xy
