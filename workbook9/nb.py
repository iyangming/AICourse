import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 生成模拟数据，4个特征，2类
X, y = make_classification(n_samples=500, n_features=4, n_informative=3, n_redundant=0,
                           n_classes=2, random_state=42)

# 对MultinomialNB和BernoulliNB需要非负整数或二值数据
# 这里做简单缩放和离散化处理：
scaler = MinMaxScaler(feature_range=(0, 10))
X_scaled = scaler.fit_transform(X)
X_int = np.round(X_scaled).astype(int)
X_bin = (X_scaled > 5).astype(int)  # 简单二值化

# 划分训练测试集
X_train_g, X_test_g, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_m, X_test_m, _, _ = train_test_split(X_int, y, test_size=0.3, random_state=42)
X_train_b, X_test_b, _, _ = train_test_split(X_bin, y, test_size=0.3, random_state=42)

# 训练和预测
gnb = GaussianNB()
gnb.fit(X_train_g, y_train)
acc_g = gnb.score(X_test_g, y_test)

mnb = MultinomialNB()
mnb.fit(X_train_m, y_train)
acc_m = mnb.score(X_test_m, y_test)

bnb = BernoulliNB()
bnb.fit(X_train_b, y_train)
acc_b = bnb.score(X_test_b, y_test)

print(f"GaussianNB准确率: {acc_g:.3f}")
print(f"MultinomialNB准确率: {acc_m:.3f}")
print(f"BernoulliNB准确率: {acc_b:.3f}")

# 可视化：只取前2个特征降维画图（用高斯朴素贝叶斯）
plt.figure(figsize=(8,6))
plt.scatter(X_test_g[:, 0], X_test_g[:, 1], c=y_test, edgecolor='k', cmap=plt.cm.coolwarm, alpha=0.7)
plt.title("测试集样本散点图（前两个特征）")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.xlabel("特征1")
plt.ylabel("特征2")
plt.show()