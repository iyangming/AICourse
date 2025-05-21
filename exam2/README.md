# Fashion MNIST 图像处理项目

## 运行环境要求
- Python 3.9+ (推荐使用Anaconda环境)

### Conda环境设置
1. 创建conda虚拟环境:
```bash
conda create -n fashion_mnist python=3.9
```
2. 激活环境:
```bash
conda activate fashion_mnist
```

### 安装依赖库
1. 使用conda安装:
```bash
conda install numpy matplotlib scipy
conda install -c conda-forge tensorflow
```
2. 或者使用pip安装:
```bash
pip install numpy matplotlib scipy tensorflow
```

- 必需库:
  - numpy
  - matplotlib
  - tensorflow (用于加载Fashion MNIST数据集)
  - scipy

## 使用的库及功能

### NumPy
- 主要功能：数组操作和数学运算
- 项目应用：
  - 处理图像数据矩阵
  - 实现图像采样和像素级操作
  - 提供高效的数值计算支持

### Matplotlib
- 主要功能：数据可视化
- 项目应用：
  - 显示原始和处理后的图像
  - 创建多子图对比展示
  - 自定义图表样式和布局

### TensorFlow
- 主要功能：深度学习框架
- 项目应用：
  - 加载Fashion MNIST数据集
  - 提供便捷的数据预处理接口
  - 支持批量数据加载

### SciPy
- 主要功能：科学计算工具
- 项目应用：
  - 实现图像旋转、缩放等变换
  - 提供高级图像处理函数
  - 支持插值和滤波操作

## 项目说明
本项目实现了对Fashion MNIST数据集的多种图像处理功能，包括:
1. 图像采样(隔行采样)
2. 多种图像增强技术:
   - 转置
   - 翻转(上下/左右)
   - 旋转(固定角度/随机角度)
   - 缩放
3. 结果可视化展示

## 数据来源
- 使用tensorflow.keras.datasets中的fashion_mnist数据集
- 包含10类时尚物品的28x28灰度图像

## 参考资源
1. [Fashion MNIST官方文档](https://github.com/zalandoresearch/fashion-mnist)
2. [Scipy图像处理文档](https://docs.scipy.org/doc/scipy/reference/ndimage.html)
3. [Matplotlib可视化指南](https://matplotlib.org/stable/contents.html)

## 常见问题

### 中文字符显示问题
1. **问题描述**: 在使用Matplotlib绘图时，中文字符可能显示为方框或乱码
2. **解决方案**:
   - 安装中文字体包:
     ```bash
     conda install -c conda-forge matplotlib-base fontconfig
     ```
   - 在代码中添加以下设置:
     ```python
import matplotlib.pyplot as plt
# Windows/Linux系统使用
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
# MacOS系统使用
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
```
3. **其他注意事项**:
   - 确保系统已安装中文字体:
  - Windows/Linux: SimHei
  - MacOS: Arial Unicode MS (默认已安装)
   - 对于Linux系统，可能需要额外配置字体缓存:
     ```bash
     fc-cache -fv
     ```