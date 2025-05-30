import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# 1. 生成模拟数据（两类二维数据）
X, y = make_blobs(n_samples=300, centers=2, cluster_std=1.5, random_state=42)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建并训练高斯朴素贝叶斯分类器
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 4. 在测试集上预测
y_pred = gnb.predict(X_test)

# 5. 计算准确率
accuracy = np.mean(y_pred == y_test)
print(f"测试集准确率: {accuracy:.2f}")

# 6. 绘制决策边界和测试点

# 创建网格点坐标
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# 预测网格点类别
Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘图
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# 画训练集点
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label='训练集', edgecolor='k')

# 画测试集点（标记错误样本）
correct = (y_pred == y_test)
plt.scatter(X_test[correct, 0], X_test[correct, 1], c=y_test[correct], marker='^', label='测试集 正确', edgecolor='k')
plt.scatter(X_test[~correct, 0], X_test[~correct, 1], c='red', marker='x', label='测试集 错误', edgecolor='k')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

plt.title("高斯朴素贝叶斯分类器 - 决策边界与测试集结果")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.legend()
plt.show()