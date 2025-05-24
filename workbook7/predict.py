# # 加载预训练模型
# from tensorflow.keras.models import load_model
# model = load_model('best_mnist_model.h5')

# # 预测函数
# def predict_images(images):
#     # 图像预处理
#     images = images.astype('float32') / 255.
#     # 预测
#     predictions = model.predict(images)
#     return np.argmax(predictions, axis=1)


import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# 加载模型
model = load_model('best_mnist_model.h5')

def preprocess_image(img_path):
    # 加载图像，灰度
    img = Image.open(img_path).convert('L')
    # 缩放到28x28
    img = img.resize((28, 28))
    # 二值化（黑白）
    img = img.convert('1', dither=Image.NONE)
    # 反色处理，转成 numpy 数组，归一化
    img_array = 1.0 - np.asarray(img).astype(np.float32) / 255.0
    # 保持28x28输入形状
    return img_array

def load_dataset(folder_path, labels_dict=None):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            path = os.path.join(folder_path, filename)
            img_array = preprocess_image(path)
            images.append(img_array)
            if labels_dict is not None:
                # 从文件名或labels_dict映射得到标签
                label = labels_dict.get(filename, None)
                if label is not None:
                    labels.append(label)
    images = np.array(images)
    images = images.reshape((-1, 28, 28))
    if labels:
        labels = to_categorical(labels, 10)
        return images, labels
    else:
        return images

def evaluate_on_dataset(folder_path, labels_dict=None):
    data = load_dataset(folder_path, labels_dict)
    if labels_dict is not None:
        x_test, y_test = data
        y_pred = model.predict(x_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(y_pred_labels == y_true_labels)
        print(f"Dataset: {folder_path} - Accuracy: {accuracy:.4f}")
        return accuracy, y_pred_labels, y_true_labels
    else:
        # 无标签时直接预测
        x_test = data
        y_pred = model.predict(x_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        print(f"Dataset: {folder_path} - Prediction done.")
        return y_pred_labels

# 假设你有对应的标签字典（文件名 -> 数字标签）
# 例如：labels_a = {'img1.png': 3, 'img2.png': 7, ...}
# 这里用示例字典，需要替换成你的实际标签
labels_a = {'0.png': 0, '1.png': 1, '2.png': 2, '3.png': 3, '4.png': 4, '5.png': 5, '6.png': 6, '7.png': 7, '8.png': 8, '9.png': 9}
labels_b = {'1.jpg': 1, '2.jpg': 2, '3.jpg': 3, '4.jpg': 4, '5.jpg': 5, '6.jpg': 6, '7.jpg': 7, '8.jpg': 8, '9.jpg': 9}
labels_c = {'1.png': 1, '2.png': 2, '3.png': 3, '4.png': 4, '5.png': 5, '6.png': 6, '7.png': 7, '8.png': 8, '9.png': 9}

# (1) 自制手写数字-a组
print("Evaluating dataset a group")
accuracy_a, pred_a, true_a = evaluate_on_dataset('./images/agroup', labels_a)

# (2) 自制手写数字-b组
print("Evaluating dataset b group")
accuracy_b, pred_b, true_b = evaluate_on_dataset('./images/bgroup', labels_b)

# (3) 自制手写数字-c组
print("Evaluating dataset c group")
accuracy_c, pred_c, true_c = evaluate_on_dataset('./images/cgroup', labels_c)

# (4) 简单分析对比
print(f"Accuracy comparison:\nA组: {accuracy_a:.4f}\nB组: {accuracy_b:.4f}\nC组: {accuracy_c:.4f}")

# 你可以根据识别率较低的组进一步分析：
# - 图像质量
# - 预处理方式是否合适
# - 数据集风格差异（比如笔迹、背景噪声）
# - 是否需要更多数据增强或模型微调


# **识别率偏低的原因分析**:
# 1. **图像质量**: 某些图像的清晰度较低，可能影响模型识别
# 2. **预处理方式**: 当前的二值化处理可能丢失了部分有用信息
# 3. **数据集风格差异**: 不同组的笔迹风格和背景噪声可能存在较大差异
# 4. **数据量不足**: 每组仅包含10张图像，可能导致模型泛化能力不足

# **改进建议**:
# 1. **优化预处理**: 尝试不同的图像增强技术，如高斯模糊、直方图均衡化等
# 2. **数据增强**: 通过旋转、平移、缩放等方式增加训练数据
# 3. **模型微调**: 在现有模型基础上，使用自制数据集进行微调
# 4. **错误样本分析**: 重点分析识别错误的样本，找出共同特征并针对性改进

        