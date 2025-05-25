# AICourse 课程项目

## 项目简介
本课程项目包含多个实验项目，涵盖人工智能和机器学习的基础知识和实践应用。

## 目录结构
- exam1/: 第一个实验项目
- exam2/: 第二个实验项目(Fashion MNIST图像处理)
- exam3/: 第三个实验项目(MNIST手写数字识别)
- workbook1/: 逻辑回归实验项目
- workbook2/: Softmax回归实验项目
- workbook3/: 神经网络实验项目
  - pytorch/: PyTorch实现
  - tensorflow/: TensorFlow实现
  - mindspore/: MindSpore实现
- workbook4/: MindSpore基础知识学习
- workbook5/: MindSpore进阶编程

## 实验项目概述
### exam1
- 基础Python编程练习
- 数据可视化基础
- 使用Matplotlib绘制图表

### exam2
- Fashion MNIST数据集处理
- 图像采样和增强技术
- 使用TensorFlow加载数据集
- CNN模型训练与评估

### exam3
- MNIST手写数字识别
- 手写数字图像预处理
- 数字图像格式转换(CSV)
- 使用Keras构建神经网络模型
- 包含手写数字样本(handwritten_digits/)和MNIST格式样本(mnist_format_digits/)

### workbook1
- Iris数据集逻辑回归分类(Setosa vs Virginica)
- 特征选择: 花瓣长度和宽度
- 超参数调优实验(学习率0.001-10.0, 迭代次数100-200)
- 测试准确率: 100% (最佳参数组合)
- 决策边界可视化
- 学习率与损失函数关系分析
- 迭代次数与准确率关系分析

### workbook2
- Iris数据集Softmax回归多分类
- 使用LogisticRegression实现Softmax回归
- 特征选择: 花瓣长度和宽度
- 默认参数: C=100, max_iter=1000
- 测试准确率: 约97%

### workbook3
#### PyTorch实现
- Iris数据集分类任务
- 单层神经网络架构
  - 输入层: 4个特征(花萼长度、花萼宽度、花瓣长度、花瓣宽度)
  - 输出层: 3个类别(setosa, versicolor, virginica)
  - 激活函数: Softmax
- 训练过程:
  - 数据预处理: 标准化特征值
  - 损失函数: 交叉熵损失
  - 优化器比较: SGD, Adam, RMSprop
  - 超参数调优:
    - 学习率: 0.001, 0.01, 0.1
    - 训练轮次: 50, 100, 200
- 可视化对比不同超参数组合下的模型表现

#### TensorFlow实现
- Iris数据集分类任务
- 神经网络架构
- 训练过程和超参数调优

#### MindSpore实现
- Iris数据集分类任务
- 使用MindSpore构建神经网络
- 训练过程和性能评估


### workbook4
- MindSpore 基础知识学习
- 内容包括：Tensors, Datasets, Models, Transforms, Training, Saving/Loading Models
- 使用 MNIST 数据集进行实践

### workbook5
- MindSpore进阶编程
- 内容包括：数据集处理、模型构建、错误分析、混合精度训练
- 使用 MindSpore 进行静态图编程实践

## 运行要求
- Python 3.x
- 相关依赖库:
  - PyTorch (workbook3/pytorch)
  - TensorFlow (workbook3/tensorflow)
  - MindSpore (workbook3/mindspore)
  - scikit-learn
  - matplotlib
  - pandas

## 项目进展
- [x] exam1: 基础练习
- [x] exam2: 图像处理
- [x] exam3: 手写数字识别
- [x] workbook1: 逻辑回归
- [x] workbook2: Softmax回归
- [x] workbook3: 神经网络
- [x] workbook4: MindSpore基础
- [x] workbook5: MindSpore进阶编程
- [x] workbook6: 生成对抗网络(GAN)与自然语言处理(NLP)
- [x] workbook7: MNIST手写数字识别
- [x] workbook8: Cifar10图像分类
