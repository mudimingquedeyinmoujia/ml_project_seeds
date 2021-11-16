import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import sklearn
import dataset
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


# iris = datasets.load_iris()
# X = iris.data[:, :2] # we only take the first two features. We could
# y = iris.target
X, y = dataset.get_seeds_dataset()
sklearn_pca = PCA(n_components=2)
X = sklearn_pca.fit_transform(X)
X = X[:, :2]


C = 1.0  # SVM regularization parameter

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                            random_state=28)
lr = LogisticRegression(solver='sag', max_iter=1400).fit(x_train, y_train)
# svc = svm.SVC(kernel='linear', C=1, gamma='auto').fit(x_train, y_train)
precision = lr.score(x_test, y_test)
print('log: ',precision)
y_hat = lr.predict(x_test)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = abs((x_max / x_min) / 100)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_hat, cmap=plt.cm.Paired)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.xlim(xx.min(), xx.max())
plt.title('logistic regression pca')
plt.show()

lr = LogisticRegression(solver='liblinear', max_iter=1400).fit(x_train, y_train)
# svc = svm.SVC(kernel='linear', C=1, gamma='auto').fit(x_train, y_train)
precision = lr.score(x_test, y_test)
print('log: ',precision)
y_hat = lr.predict(x_test)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = abs((x_max / x_min) / 100)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_hat, cmap=plt.cm.Paired)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.xlim(xx.min(), xx.max())
plt.title('logistic regression pca')
plt.show()

