from sklearn.datasets import fetch_openml
import numpy as np

mnist_X, mnist_y = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

X = mnist_X.astype(np.float32) / 255
y = mnist_y.astype(np.int32)

import matplotlib.pyplot as plt

plt.imshow(X[0].reshape(28,28), cmap='gray')
# print("The labe/l of this data is {:.0f}".format(y[0]))
# plt.show()

# ========================================================================================================
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/7, random_state=0)
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

ds_train = TensorDataset(X_train,y_train)
ds_test = TensorDataset(X_test,y_test)

loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)

#Define NN
from torch import nn
model = nn.Sequential()
model.add_module('fc1', nn.Linear(28*28*1,100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100,100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100,10))

# print(model)

#Define Optimizer
from torch import optim

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr = 0.01)


def train(epoch):
    model.train()

    for data, targets in loader_train:
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

    print("epoch{}: end\n".format(epoch))


def test():
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, targets in loader_test:
            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data.view_as(predicted)).sum()

    data_num = len(loader_test.dataset)
    print('Accracy: {}/{} ({:.0f}%)\n'.format(correct, data_num, 100. * correct /data_num))


for epoch in range(3):
    train(epoch)

# test()

index = 2018

model.eval()
data = X_test[index]
output = model(data)
_, predicted = torch.max(output.data, 0)
print("Predicted:{}".format(predicted))

X_test_show = (X_test[index]).numpy()
plt.imshow(X_test_show.reshape(28, 28), cmap='gray')
print("The correct answer is {:.0f}".format(y_test[index]))
plt.show()