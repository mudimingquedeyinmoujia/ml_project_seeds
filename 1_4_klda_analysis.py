import pandas as pd
import pandas as pd
from scipy.linalg import eigh
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions as pre
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import dataset

data = load_iris() # 用另一个数据集iris测试效果
X = data.data
y = data.target
# X, y = dataset.get_seeds_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

##输入样本要求行排列
##计算分子M矩阵
def rbf_kernel_lda_m(X, gamma=0.01, y=[]):
    n = X.shape[0]
    c_all = np.unique(y)  # 所有不重复类别
    c_len = len(c_all)  # 类的数量
    K_m = np.zeros((n, c_len))

    for k, c in enumerate(c_all):
        c_len = len([i for i in y if i == c])
        for i in range(n):
            K_val = 0.0
            for c_row in X[y == c]:
                K_val += np.exp(-gamma * (np.sum((X[i] - c_row) ** 2)))
            K_m[i, k] = (K_val / c_len)

    M = np.zeros((n, n))
    for p in combinations(K_m.T, r=2):
        M += (p[0] - p[1])[:, np.newaxis].dot((p[0] - p[1])[np.newaxis, :])
    return M


##计算M矩阵
def rbf_kernel_lda_m_two(X, gamma=0.01, y=[]):
    n = X.shape[0]
    c_all = np.unique(y)  # 所有不重复类别
    c_len = len(c_all)  # 类的数量
    K_m = np.zeros((n, c_len))

    for k, c in enumerate(c_all):
        for i in range(n):
            K_m[i, k] = np.array(np.sum([np.exp(-gamma * np.sum((X[i] - c_row) ** 2)) for c_row in X[y == c]])) / c_len

    M = np.zeros((n, n))
    for p in combinations(K_m.T, r=2):
        M += (p[0] - p[1])[:, np.newaxis].dot((p[0] - p[1])[np.newaxis, :])
    return M


##计算N矩阵
def rbf_kernel_lda_n(X, gamma=0.01, y=[]):
    n = X.shape[0]
    c_all = np.unique(y)  # 所有不重复类别

    ##K_c = np.zeros((X.shape[0],c_len))
    N = np.zeros((n, n))
    for k, c in enumerate(c_all):
        c_num = len([i for i in y if i == c])
        I = np.eye(c_num)
        I_c = np.ones((c_num, c_num)) / c_num
        I_n = np.eye(n)
        K_c = np.zeros((n, c_num))
        for i in range(n):
            K_c[i, :] = np.array([np.exp(-gamma * np.sum((X[i] - c_row) ** 2)) for c_row in X[y == c]])
        N += K_c.dot(I - I_c).dot(K_c.T)  ##+ I_n*0.001

    return N


##计算新样本点映射后的值；alphas 是其中一个映射向量
def project_x(X_new, X, gamma=0.01, alphas=[]):
    n = X.shape[0]
    X_proj = np.zeros((n, len(alphas)))
    for p in range(len(alphas)):
        for i in range(len(X_new)):
            k = np.exp(-gamma * np.array([np.sum((X_new[i] - row) ** 2) for row in X]))
            X_proj[i, p] = np.real(k[np.newaxis, :].dot(alphas[p]))  ##不能带虚部
    return X_proj


for g_params in list([500, 1000]):
    X = X_train_std
    y = y_train
    p = 2
    n = X.shape[0]
    ##求判别式广义特征值和特征向量
    ##方式1计算K_m矩阵
    K_m = rbf_kernel_lda_m(X, g_params, y)
    ##方式2计算K_m矩阵
    ##K_m = rbf_kernel_lda_m_two(X, g_params , y)

    ##方式1计算K_m矩阵
    # K_n = np.zeros((N,N))
    # for i in  np.unique(y):
    # K_n += rbf_kernel_lda_n(X, g_params , c=i)
    K_n = rbf_kernel_lda_n(X, g_params, y)

    ##方式1
    from numpy import linalg

    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(K_n).dot(K_m))
    eigen_pairs = [(np.abs(eigvals[i]), eigvecs[:, i]) for i in range(len(eigvals))]
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    alphas1 = eigen_pairs[0][1][:, np.newaxis]
    alphas2 = eigen_pairs[1][1][:, np.newaxis]
    p = 2
    alphas = []
    for i in range(p):
        alphas.append(eigen_pairs[i][1][:, np.newaxis])

    ##方式2
    # from scipy.linalg import eigh
    #
    # eigvals1, eigvecs1 = eigh(np.linalg.inv(K_n).dot(K_m))
    # eigen_pairs_one = [(np.abs(eigvals1[i]), eigvecs1[:, i]) for i in range(len(eigvals1))]
    # eigen_pairs_two = sorted(eigen_pairs_one, key=lambda k: k[0], reverse=True)
    # alphas_one = eigen_pairs_two[0][1][:, np.newaxis]
    # alphas_two = eigen_pairs_two[1][1][:, np.newaxis]
    # alphas1 = eigvecs1[-1][:, np.newaxis]
    # alphas2 = eigvecs1[-2][:, np.newaxis]

    ##新样本点
    X_new = np.zeros((n, p))
    X_new = project_x(X, X, g_params, alphas)
    print(X_new)
    print(X_new.shape)
    plt.scatter(X_new[y == 0, 0], X_new[y == 0, 1], c='red', marker='s', label='train one')
    plt.scatter(X_new[y == 1, 0], X_new[y == 1, 1], c='blue', marker='o', label='train two')
    plt.scatter(X_new[y == 2, 0], X_new[y == 2, 1], c='green', marker='+', label='train three')
    plt.legend(loc='upper right')
    plt.show()
    lr = LogisticRegression(C=1000, random_state=1, penalty='l2')
    lr.fit(X_new, y)

    pre(X_new, y, lr)
    plt.show()
