import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 超参数组合
param_grid = [
    {"optimizer": "adam", "learning_rate": 0.01, "epochs": 100},
    {"optimizer": "adam", "learning_rate": 0.001, "epochs": 100},
    {"optimizer": "sgd", "learning_rate": 0.01, "epochs": 100},
    {"optimizer": "sgd", "learning_rate": 0.001, "epochs": 100},
    {"optimizer": "adam", "learning_rate": 0.001, "epochs": 200},
]

results = []

# 实验执行
for params in param_grid:
    tf.keras.backend.clear_session()
    optimizer_name = params["optimizer"]
    learning_rate = params["learning_rate"]
    epochs = params["epochs"]

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, input_shape=(4,), activation='softmax')
    ])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    start = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_data=(X_test, y_test))
    duration = time.time() - start

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    results.append({
        "Optimizer": optimizer_name,
        "Learning Rate": learning_rate,
        "Epochs": epochs,
        "Test Accuracy": round(test_acc, 4),
        "Test Loss": round(test_loss, 4),
        "Training Time (s)": round(duration, 4)
    })

# 转为 DataFrame 展示
df = pd.DataFrame(results)
print(df.sort_values(by="Test Accuracy", ascending=False))

# 可视化结果
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 准确率对比
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
for opt in df['Optimizer'].unique():
    subset = df[df['Optimizer'] == opt]
    plt.plot(subset['Learning Rate'], subset['Test Accuracy'], 'o-', label=opt)
plt.xscale('log')
plt.xlabel('学习率(log scale)')
plt.ylabel('测试准确率')
plt.title('不同优化器的准确率对比')
plt.legend()

# 2. 损失对比
plt.subplot(1, 3, 2)
for opt in df['Optimizer'].unique():
    subset = df[df['Optimizer'] == opt]
    plt.plot(subset['Learning Rate'], subset['Test Loss'], 'o-', label=opt)
plt.xscale('log')
plt.xlabel('学习率(log scale)')
plt.ylabel('测试损失')
plt.title('不同优化器的损失对比')
plt.legend()

# 3. 训练时间对比
plt.subplot(1, 3, 3)
for opt in df['Optimizer'].unique():
    subset = df[df['Optimizer'] == opt]
    plt.plot(subset['Learning Rate'], subset['Training Time (s)'], 'o-', label=opt)
plt.xscale('log')
plt.xlabel('学习率(log scale)')
plt.ylabel('训练时间(s)')
plt.title('不同优化器的训练时间对比')
plt.legend()

plt.tight_layout()
plt.show()