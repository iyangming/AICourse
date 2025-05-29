# AICourse

## 项目简介
本课程项目包含多个实验项目，涵盖人工智能和机器学习的基础知识和实践应用。

## 目录结构
- demo1/: 番茄钟计时器Web应用
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
- workbook6/: 生成对抗网络(GAN)与自然语言处理(NLP)
- workbook7/: MNIST手写数字识别
- workbook8/: Cifar10图像分类
- llm/: 深度学习与Transformer模型
- d2l-mindspore/: 深度学习实践课程
  - chapter_02_preliminaries/: 预备知识
  - chapter_03_linear-networks/: 线性网络
  - chapter_04_multilayer-perceptrons/: 多层感知机
  - chapter_05_deep-learning-computation/: 深度学习计算
  - chapter_06_convolutional-neural-networks/: 卷积神经网络
  - chapter_07_convolutional-modern/: 现代卷积网络
  - chapter_08_recurrent-neural-networks/: 循环神经网络
  - chapter_09_recurrent-modern/: 现代循环网络
  - chapter_10_attention_mechanisms/: 注意力机制
  - chapter_11_optimization/: 优化算法
  - chapter_12_computational-performance/: 计算性能
  - chapter_13_computer-vision/: 计算机视觉
  - chapter_14_natural_language_processing_pretraining/: 自然语言处理预训练
  - chapter_15_natural_language_processing_applications/: 自然语言处理应用
- d2l-paddle/: 深度学习实践课程
  - chapter_attention-mechanisms/: 注意力机制
  - chapter_computer-vision/: 计算机视觉
  - chapter_convolutional-modern/: 现代卷积网络
  - chapter_installation/: 安装指南
  - chapter_introduction/: 介绍
  - chapter_linear-networks/: 线性网络
  - chapter_multilayer-perceptrons/: 多层感知机
  - chapter_notation/: 符号说明
  - chapter_optimization/: 优化算法
  - chapter_preface/: 前言
  - chapter_preliminaries/: 预备知识
  - chapter_recurrent-modern/: 现代循环网络
  - chapter_references/: 参考文献
- d2l-tensorflow/: 深度学习实践课程
  - chapter_attention-mechanisms/: 注意力机制
  - chapter_computer-vision/: 计算机视觉
  - chapter_convolutional-modern/: 现代卷积网络
  - chapter_installation/: 安装指南
  - chapter_introduction/: 介绍
  - chapter_linear-networks/: 线性网络
  - chapter_multilayer-perceptrons/: 多层感知机
  - chapter_notation/: 符号说明
  - chapter_optimization/: 优化算法
  - chapter_preface/: 前言
  - chapter_preliminaries/: 预备知识
  - chapter_recurrent-modern/: 现代循环网络
  - chapter_references/: 参考文献
- d2l-pytorch/: 深度学习实践课程
  - chapter_attention-mechanisms/: 注意力机制
  - chapter_computer-vision/: 计算机视觉
  - chapter_convolutional-modern/: 现代卷积网络
  - chapter_installation/: 安装指南
  - chapter_introduction/: 介绍
  - chapter_linear-networks/: 线性网络
  - chapter_multilayer-perceptrons/: 多层感知机
  - chapter_notation/: 符号说明
  - chapter_optimization/: 优化算法
  - chapter_preface/: 前言
  - chapter_preliminaries/: 预备知识
  - chapter_recurrent-modern/: 现代循环网络
  - chapter_references/: 参考文献

## 实验项目概述
### demo1
- 番茄钟计时器Web应用
- 功能特性:
  - 可自定义工作时间、短休息和长休息时长
  - 任务管理系统(添加、完成、删除任务)
  - 统计功能(每日专注时间、每周番茄钟数量、任务完成数)
  - 数据可视化图表(Chart.js)
  - 主题切换(明暗模式)
  - 音效通知
  - 键盘快捷键支持
  - PWA支持(Progressive Web App)
  - 本地存储数据持久化
- 技术栈: HTML5, CSS3, JavaScript, Chart.js
- 文件结构:
  - index.html: 主页面
  - css/styles.css: 样式文件
  - js/app.js: 核心逻辑
  - assets/: 图标和音效资源
  - manifest.json: PWA配置

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
- Web浏览器 (demo1)

## 项目进展
- [x] demo1: 番茄钟计时器Web应用
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
- [x] llm: 深度学习与Transformer模型
- [x] d2l-mindspore: 深度学习实践课程
- [x] d2l-paddle: 深度学习实践课程
- [x] d2l-tensorflow: 深度学习实践课程
- [x] d2l-pytorch: 深度学习实践课程
