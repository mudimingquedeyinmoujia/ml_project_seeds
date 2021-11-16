from sklearn import datasets
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import dataset
from sklearn import metrics
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.optimize import linear_sum_assignment
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
iris = datasets.load_iris()
X, y = dataset.get_seeds_dataset()


def FCM(X, c_clusters=3, m=2, eps=10):
    membership_mat = np.random.random((len(X), c_clusters))
    membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])

    while True:
        working_membership_mat = membership_mat ** m
        Centroids = np.divide(np.dot(working_membership_mat.T, X),
                              np.sum(working_membership_mat.T, axis=1)[:, np.newaxis])

        n_c_distance_mat = np.zeros((len(X), c_clusters))
        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                n_c_distance_mat[i][j] = np.linalg.norm(x - c, 2)

        new_membership_mat = np.zeros((len(X), c_clusters))

        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                new_membership_mat[i][j] = 1. / np.sum((n_c_distance_mat[i][j] / n_c_distance_mat[i]) ** (2 / (m - 1)))
        if np.sum(abs(new_membership_mat - membership_mat)) < eps:
            break
        membership_mat = new_membership_mat
    return np.argmax(new_membership_mat, axis=1)


def evaluate(y, t):
    a, b, c, d = [0 for i in range(4)]
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if y[i] == y[j] and t[i] == t[j]:
                a += 1
            elif y[i] == y[j] and t[i] != t[j]:
                b += 1
            elif y[i] != y[j] and t[i] == t[j]:
                c += 1
            elif y[i] != y[j] and t[i] != t[j]:
                d += 1
    return a, b, c, d


def external_index(a, b, c, d, m):
    JC = a / (a + b + c)
    FMI = np.sqrt(a ** 2 / ((a + b) * (a + c)))
    RI = 2 * (a + d) / (m * (m + 1))
    return JC, FMI, RI


def evaluate_it(y, t):
    a, b, c, d = evaluate(y, t)
    return external_index(a, b, c, d, len(y))

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

# raw
test_y = FCM(X)

JC, FMI, RI = evaluate_it(y, test_y)
print('AC ',metrics.accuracy_score(y, test_y),acc(y,test_y))

print('JC: ', JC)
print('FMI: ', FMI)
print('RI: ', RI)
X_reduced = PCA(n_components=2).fit_transform(X)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=test_y, cmap=plt.cm.Set1)
plt.show()
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Set1)
plt.show()

# pca
test_y = FCM(X_reduced)

JC, FMI, RI = evaluate_it(y, test_y)
print('AC ',metrics.accuracy_score(y, test_y),acc(y,test_y))
print('JC: ', JC)
print('FMI: ', FMI)
print('RI: ', RI)
# X_reduced = PCA(n_components=2).fit_transform(X)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=test_y, cmap=plt.cm.Set1)
plt.show()
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Set1)
plt.show()


# lda
# X_reduced = PCA(n_components=2).fit_transform(X)
X_reduced = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
test_y = FCM(X_reduced)

JC, FMI, RI = evaluate_it(y, test_y)
print('AC ',metrics.accuracy_score(y, test_y),acc(y,test_y))
print('JC: ', JC)
print('FMI: ', FMI)
print('RI: ', RI)
# X_reduced = PCA(n_components=2).fit_transform(X)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=test_y, cmap=plt.cm.Set1)
plt.show()
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Set1)
plt.show()