from sklearn.datasets import fetch_openml
import numpy as np

mnist_X, mnist_y = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

X = mnist_X.astype(np.float32) / 255
y = mnist_y.astype(np.int32)

import matplotlib.pyplot as plt

plt.imshow(X[0].reshape(28,28), cmap='gray')
print("The label of this data is {:.0f}".format(y[0]))
plt.show()

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

X_train, X_test, y_test
