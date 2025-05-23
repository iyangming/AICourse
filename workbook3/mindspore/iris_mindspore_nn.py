import mindspore as ms
from mindspore import nn, Tensor, context
from mindspore.dataset import NumpySlicesDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置模式与设备
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 1. 数据加载与预处理
iris = load_iris()
X, y = iris.data, iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot 编码
y_onehot = np.eye(3)[y]

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# 转为 Tensor 数据集
train_dataset = NumpySlicesDataset({"data": X_train.astype(np.float32), "label": y_train.astype(np.float32)}, shuffle=True)
test_dataset = NumpySlicesDataset({"data": X_test.astype(np.float32), "label": y_test.astype(np.float32)}, shuffle=False)

train_dataset = train_dataset.batch(16)
test_dataset = test_dataset.batch(16)

# 2. 定义网络结构
class SingleLayerNN(nn.Cell):
    def __init__(self):
        super(SingleLayerNN, self).__init__()
        self.fc = nn.Dense(4, 3)

    def construct(self, x):
        return self.fc(x)

# 3. 定义损失函数与优化器
net = SingleLayerNN()
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
optimizer = nn.Adam(net.trainable_params(), learning_rate=0.01)

# 4. 模型训练
model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={"Accuracy": nn.Accuracy()})

print("Training...")
start = time.time()
model.train(epoch=100, train_dataset=train_dataset, callbacks=[ms.LossMonitor()])
train_time = time.time() - start

# 5. 模型评估
print("Evaluating...")
start = time.time()
acc = model.eval(test_dataset)
test_time = time.time() - start

print(f"Accuracy on test set: {acc['Accuracy']:.4f}")
print(f"Training time: {train_time:.2f}s")
print(f"Test time: {test_time:.2f}s")

# 6. 可视化预测
X_test_tensor = Tensor(X_test.astype(np.float32))
y_pred_logits = model.predict(X_test_tensor)
y_pred = y_pred_logits.asnumpy()
y_true = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

# 混淆矩阵和准确率可视化
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()