# iris_nn_tf.py - TensorFlow Implementation

import time
import numpy as np
import matplotlib.pyplot as plt
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

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_shape=(4,), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
start_time = time.time()
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=0)
training_time = time.time() - start_time
print(f"训练时间：{training_time:.4f} 秒")

# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"测试集准确率：{test_acc:.4f}")
print(f"测试集损失：{test_loss:.4f}")

# 可视化结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.legend()
plt.title("Accuracy")
plt.show()


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, input_shape=(4,), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


