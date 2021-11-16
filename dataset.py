import numpy as np


def get_seeds_dataset():
    seeds = np.loadtxt('seeds_dataset.txt')
    X = seeds[:, :7]
    y = seeds[:, 7].astype(np.int)
    return X, y


if __name__ == '__main__':
    X, y = get_seeds_dataset()
    print(X)
    print(y)
    print(X.dtype)
    print(y.dtype)
    print(X.shape)
    print(y.shape)
