if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np


from dezero import optimizers, DataLoader
import dezero.functions as F
from dezero.models import MLP
import dezero


# ハイパーパラメータ
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# データ読み込み
train_set = dezero.datasets.Spiral(train=True)
test_set = dezero.datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)

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

    print(f"test_loss: {(sum_loss / len(test_set)):.4f}, accuracy: {(sum_acc / len(test_set)):.4f}")
