# MNIST 手写数字处理项目

## 项目简介
本项目实现了手写数字图像的预处理和格式转换功能，将手写数字图片转换为MNIST数据集兼容的格式。

## 数据处理流程
1. 图像预处理：二值化、反色处理
2. 尺寸调整：统一调整为28x28像素
3. 格式转换：将图像转换为CSV格式的数值矩阵

## 文件结构
- `handwritten_digits/`: 原始手写数字图片(0-9.png)
- `mnist_format_digits/`: 处理后的MNIST格式CSV文件
- `mnist_image_process.py`: 主处理脚本
- `mnist_image_process.ipynb`: Jupyter Notebook版本

## 运行指南
1. 安装依赖：
```bash
pip install pillow numpy matplotlib
```
2. 运行处理脚本：
```bash
python mnist_image_process.py
```
3. 处理结果将保存在`mnist_format_digits/`目录下

## 依赖库说明
### NumPy
- `np.array()`: 将图像数据转换为NumPy数组
- `np.savetxt()`: 将数组保存为CSV文件
- `np.loadtxt()`: 从CSV文件加载数组数据

### PIL (Python Imaging Library)
- `Image.open()`: 打开图像文件
- `ImageOps.invert()`: 反色处理图像
- `Image.point()`: 应用阈值处理
- `Image.resize()`: 调整图像尺寸

### Matplotlib
- `plt.subplots()`: 创建多子图布局
- `plt.imshow()`: 显示图像
- `plt.show()`: 显示所有图表