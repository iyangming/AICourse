
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from scipy.ndimage import rotate, zoom
import random
import matplotlib

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # For MacOS
plt.rcParams['axes.unicode_minus'] = False

# 下载并加载数据
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

# 查看数据集信息
print("训练集样本数：", train_x.shape[0], "形状：", train_x.shape[1:])
print("测试集样本数：", test_x.shape[0], "形状：", test_x.shape[1:])
label_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
print("标签类别：", label_names)

# 提取索引为0和8的图片
img_0 = train_x[0]
img_8 = train_x[8]

# 局部采样函数
def sample_image(img, step):
    sampled_img = np.zeros_like(img)
    sampled_img[::step, :] = img[::step, :]
    return sampled_img

sample2 = sample_image(img_0, 2)
sample4 = sample_image(img_0, 4)

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.imshow(img_0, cmap='gray')
plt.title('原图')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sample2, cmap='gray')
plt.title('隔2行采样')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sample4, cmap='gray')
plt.title('隔4行采样')
plt.axis('off')

plt.tight_layout()
plt.show()

# 图像增强函数
def augment_image(img):
    return [
        img,
        np.transpose(img),
        np.flipud(img),
        np.fliplr(img),
        rotate(img, angle=-10, reshape=False, mode='constant', cval=0),
        rotate(img, angle=10, reshape=False, mode='constant', cval=0)
    ]

train_x_aug1 = []
for i in range(10):
    augmented = augment_image(train_x[i])
    train_x_aug1.append(augmented)
train_x_aug1 = np.array(train_x_aug1)

# 显示10×6子图
titles = ["原图", "转置", "上下翻转", "水平镜像", "逆时针10°", "顺时针10°"]
plt.figure(figsize=(12, 16))
for i in range(10):
    for j in range(6):
        plt.subplot(10, 6, i * 6 + j + 1)
        plt.imshow(train_x_aug1[i, j], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title(titles[j], fontsize=10, color='blue')
plt.suptitle("Fashion MNIST数据增强", fontsize=16, color='darkred')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# 随机旋转增强 train_x_aug2
train_x_aug2 = []
for i in range(10):
    img = train_x[i]
    for _ in range(2):
        angle = random.uniform(-30, 30)
        img_rot = rotate(img, angle=angle, reshape=False, mode='constant', cval=0)
        train_x_aug2.append(img_rot)
train_x_aug2 = np.array(train_x_aug2)

plt.figure(figsize=(8, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(train_x_aug2[i], cmap='gray')
    plt.axis('off')
plt.suptitle("前10张图像的随机旋转结果", fontsize=14)
plt.tight_layout()
plt.show()

# train_x_aug3 - 随机变换
def random_transform(img):
    transforms = [
        lambda x: np.transpose(x),
        lambda x: np.flipud(x),
        lambda x: np.fliplr(x),
        lambda x: rotate(x, angle=random.choice([-15, 15]), reshape=False, mode='constant', cval=0)
    ]
    selected = random.sample(transforms, 3)
    return [f(img) for f in selected]

train_x_aug3 = []
for i in range(100):
    aug = random_transform(train_x[i])
    train_x_aug3.append(aug)
train_x_aug3 = np.array(train_x_aug3)

# 展示随机10张图及其增强
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(4, 5, i+1)
    plt.imshow(train_x[i], cmap='gray')
    plt.title("原图")
    plt.axis('off')
    for j in range(3):
        plt.subplot(4, 5, 10 + i+1 + j*10)
        plt.imshow(train_x_aug3[i, j], cmap='gray')
        plt.axis('off')
plt.suptitle("随机选择的10张图像及其增强效果", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 缩放保持28x28函数
def resize_to_28x28(img, scale):
    h, w = img.shape
    img_scaled = zoom(img, zoom=scale)
    sh, sw = img_scaled.shape
    result = np.zeros((28, 28))
    start_h = (28 - sh) // 2
    start_w = (28 - sw) // 2
    end_h = start_h + sh
    end_w = start_w + sw
    result[start_h:end_h, start_w:end_w] = img_scaled
    return result

img = train_x[0]
smaller = resize_to_28x28(img, 0.9)
larger = resize_to_28x28(img, 1.1)

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("原图")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(smaller, cmap='gray')
plt.title("缩小10%")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(larger, cmap='gray')
plt.title("放大10%")
plt.axis('off')
plt.tight_layout()
plt.show()
