if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dezero


# train_set = dezero.datasets.MNIST(train=True, transform=None)
# test_set = dezero.datasets.MNIST(train=False, transform=None)

# print(len(train_set))
# print(len(test_set))


import math
import numpy as np


from dezero import optimizers, DataLoader
import dezero.functions as F
from dezero.models import MLP
import dezero


# ハイパーパラメータ
max_epoch = 5
batch_size = 100
hidden_size = 1000

# データ読み込み
train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)

# data_size = len(train_set)
# max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    sum_acc, sum_loss = 0, 0

    for x, t in train_loader:
        # 勾配の算出、パラメータ更新
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print(f"epoch: {epoch}")
    print(f"train_loss: {(sum_loss / len(train_set)):.4f}, accuracy: {(sum_acc / len(train_set)):.4f}")

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print(f"test_loss: {(sum_loss / len(test_set)):.4f}, accuracy: {(sum_acc / len(test_set)):.4f}")
