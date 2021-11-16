import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import dataset
import torch.utils.data as Data
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import sklearn
import dataset
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


class MyAnn(nn.Module):
    def __init__(self, input_n, output_n):
        super(MyAnn, self).__init__()
        self.archi = nn.Sequential(
            nn.Linear(input_n, 256),
            nn.ReLU(),
            nn.Linear(256, output_n)
        )

    def forward(self, x):
        y = self.archi(x)
        return y


class MyDataSet():
    def __init__(self, X, y):
        super(MyDataSet, self).__init__()
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __getitem__(self, index):
        return X[index], y[index]

    def __len__(self):
        return len(X)

def softmax(x):
    x_exp = x.exp()
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition


def cal_acc(test_iter, net):
    acc_sum = 0.0
    n = 0
    with torch.no_grad():
        for features, labels in test_iter:
            # print(features.dtype)
            output_raw = net(features)
            y_hat = softmax(output_raw)
            acc = (y_hat.argmax(dim=1) == labels).float().sum().item()
            acc_sum += acc
            n += labels.shape[0]
        return acc_sum / n

def loss_curve(epochs, loss_list):
    epochs_list = np.arange(epochs) + 1

    plt.plot(epochs_list, loss_list, label="loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc=0, ncol=1)  # 参数：loc设置显示的位置，0是自适应；ncol设置显示的列数

    plt.show()

def acc_curve(epochs, acc_list):
    epochs_list = np.arange(epochs) + 1

    plt.plot(epochs_list, acc_list, label="acc")
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.legend(loc=0, ncol=1)  # 参数：loc设置显示的位置，0是自适应；ncol设置显示的列数

    plt.show()

batch_size = 10
X, y = dataset.get_seeds_dataset()
X=X.astype(np.float32)
y=(y-1).astype(np.float32)
input_n=X.shape[1]
output_n=3
lra=0.01
epochs=500
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                            random_state=28)

train_dataset = MyDataSet(x_train, y_train)
test_dataset = MyDataSet(x_test, y_test)

train_iter = Data.DataLoader(train_dataset, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_dataset, batch_size, shuffle=True)

# for features,labels in test_iter:
#     print(features.dtype)
#     print(labels.dtype)
net=MyAnn(input_n,output_n)
loss = nn.CrossEntropyLoss()

for params in net.parameters():
    init.normal_(params, mean=0, std=1)

optimizer = optim.SGD(net.parameters(), lr=lra)
acc_rate = cal_acc(test_iter, net)
loss_list=[]
acc_list=[]
print("not train acc:{}".format(acc_rate))
for epoch in range(1, epochs + 1):
    print("epoch {} start".format(epoch))
    for features, labels in train_iter:
        outputs = net(features)
        # print(outputs.dtype,labels.dtype)
        l = loss(outputs, labels.to(torch.long))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    print('iter ok')
    acc_rate_test = cal_acc(test_iter, net)
    acc_rate_train = cal_acc(train_iter, net)
    loss_list.append(l.item())
    acc_list.append(acc_rate_test)
    print("epoch {}, loss:{}, acc_rate_train:{}, acc_rate_test:{}".format(epoch, l.item(), acc_rate_train,
                                                                          acc_rate_test))
    if epoch%100==0:
        loss_curve(epoch,loss_list)
        acc_curve(epoch,acc_list)



