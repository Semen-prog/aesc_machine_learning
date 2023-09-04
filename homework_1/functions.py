import numpy as np

'''

np.int64(...) вставлен, чтобы невекторизованное решение переполнялось так же, как векторизованное

'''

def prod_non_zero_diag(X):
    n, m = len(X), len(X[0])
    cnt = np.int64(1)
    for i in range(min(n, m)):
        if X[i][i] != 0:
            cnt *= X[i][i]
    return cnt


def are_multisets_equal(x, y):
    return sorted(x) == sorted(y)


def max_after_zero(x):
    ind = -1
    for i in range(1, len(x)):
        if x[i - 1] != 0:
            continue
        if ind == -1 or x[ind] < x[i]:
            ind = i
    if ind == -1:
        return None
    return x[ind]


def convert_image(arr, coefs):
    height = len(arr)
    width = len(arr[0])
    res = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            for k in range(len(coefs)):
                res[i][j] += arr[i][j][k] * coefs[k]
    return res


def run_length_encoding(x):
    a = []
    cnt = []
    for i in range(len(x)):
        if i == 0 or x[i] != x[i - 1]:
            a.append(x[i])
            cnt.append(0)
        cnt[-1] += 1
    return (a, cnt)


def pairwise_distance(X, Y):
    arr = []
    for x in X:
        arr.append([])
        for y in Y:
            cnt = np.int64(0)
            for xi, yi in zip(x, y):
                cnt += (xi - yi) ** 2
            arr[-1].append(cnt)
    return arr
