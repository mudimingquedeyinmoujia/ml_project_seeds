import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import sklearn
import dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# iris = datasets.load_iris()
# X = iris.data[:, :2] # we only take the first two features. We could
# y = iris.target
X, y = dataset.get_seeds_dataset()
X = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
X = X[:, :2]

C = 1.0  # SVM regularization parameter

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                            random_state=28)
# 1. linear kernel
svc = svm.SVC(kernel='linear', C=1, gamma='auto').fit(x_train, y_train)
precision = svc.score(x_test, y_test)
print('linear: ',precision)
y_hat = svc.predict(x_test)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = abs((x_max / x_min) / 100)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_hat, cmap=plt.cm.Paired)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()

# poly kernel
svc = svm.SVC(kernel='poly', C=1,gamma='auto').fit(X, y)
precision = svc.score(x_test, y_test)
print('poly: ',precision)
y_hat = svc.predict(x_test)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = abs((x_max / x_min) / 100)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_hat, cmap=plt.cm.Paired)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with poly kernel')
plt.show()


# rbf kernel auto
svc = svm.SVC(kernel='rbf', C=1,gamma='auto').fit(X, y)
precision = svc.score(x_test, y_test)
print('rbf auto: ',precision)
y_hat = svc.predict(x_test)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = abs((x_max / x_min) / 100)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_hat, cmap=plt.cm.Paired)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with rbf kernel gamma = auto')
plt.show()


# rbf kernel 10
svc = svm.SVC(kernel='rbf', C=1,gamma=10).fit(X, y)
precision = svc.score(x_test, y_test)
print('rbf 10: ',precision)
y_hat = svc.predict(x_test)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = abs((x_max / x_min) / 100)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_hat, cmap=plt.cm.Paired)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with rbf kernel gamma = 10')
plt.show()


# rbf kernel 100
svc = svm.SVC(kernel='rbf', C=1,gamma=100).fit(X, y)
precision = svc.score(x_test, y_test)
print('rbf 100: ',precision)
y_hat = svc.predict(x_test)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = abs((x_max / x_min) / 100)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_hat, cmap=plt.cm.Paired)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with rbf kernel gamma = 100')
plt.show()