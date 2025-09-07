if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os
import dezero
import dezero.functions as F
from dezero import optimizers, DataLoader
from dezero.models import MLP

# ハイパーパラメータ
max_epoch = 5
batch_size = 100
hidden_size = 1000

# データ読み込み
train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)

model = MLP((hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)

if os.path.exists("my_mlp.npz"):
    model.load_weights("my_mlp.npz")

for epoch in range(max_epoch):
    sum_loss = 0

    for x, t in train_loader:
        # 勾配の算出、パラメータ更新
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)

    print(f"epoch: {epoch}")
    print(f"train_loss: {(sum_loss / len(train_set)):.4f}")

model.save_weights("my_mlp.npz")
