import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from matplotlib.colors import ListedColormap
import pandas as pd
import time

# 加载数据
iris = load_iris()
X = iris.data[:, 2:4]  # 使用花瓣长度和宽度
y = iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# 训练Softmax回归模型
model = LogisticRegression(solver='lbfgs', C=100, max_iter=1000)
model.fit(X_train, y_train)

# 测试集预测与准确率
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("测试集准确率：", acc)

# 可视化分类边界
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']))
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.title("Softmax Classification Boundary")
plt.show()


# 超参数对比
learning_rates = [0.01, 0.1, 1]
max_iters = [50, 100, 200]
results = []

for lr in learning_rates:
    for iters in max_iters:
        model = LogisticRegression(solver='lbfgs',
                                   C=1/lr, max_iter=iters)
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        test_loss = log_loss(y_test, model.predict_proba(X_test))
        results.append([lr, iters, train_loss, test_loss, end - start])

# 表格输出
df = pd.DataFrame(results, columns=["Learning Rate", "Max Iter", "Train Loss", "Test Loss", "Train Time (s)"])
print(df)


# 不同属性组合比较准确率
attribute_sets = {
    "petal only (2 features)": iris.data[:, 2:4],
    "petal + sepal length (3 features)": iris.data[:, [0, 2, 3]],
    "all features (4 features)": iris.data
}

print("\\n不同属性组合的分类性能比较：")
for desc, X_sel in attribute_sets.items():
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=0)
    model = LogisticRegression(solver='lbfgs', C=100)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{desc:<35} -> Accuracy: {acc:.4f}")

# 不同训练集比例对模型性能的影响
ratios = [0.6, 0.7, 0.8, 0.9]
X_sel = iris.data[:, 2:4]  # 固定选择 petal 特征

print("\\n不同训练集比例的分类准确率：")
for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=1 - ratio, random_state=42)
    model = LogisticRegression(solver='lbfgs', C=100)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Train ratio: {int(ratio*100)}% -> Test Accuracy: {acc:.4f}")