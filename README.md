# AICourse 课程项目

## 项目简介
本课程项目包含多个实验项目，涵盖人工智能和机器学习的基础知识和实践应用。

## 目录结构
- exam1/: 第一个实验项目
- exam2/: 第二个实验项目(Fashion MNIST图像处理)
- exam3/: 第三个实验项目(MNIST手写数字识别)
- workbook1/: 逻辑回归实验项目

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
- 实验结论:
  - 花瓣长度和宽度是区分两类鸢尾花的最佳特征
  - 逻辑回归模型计算效率高(训练时间<0.002s)
  - 学习率在0.01-10.0范围内对性能影响不大

## 通用使用说明
1. 每个实验项目包含:
   - Python源代码(.py)
   - Jupyter Notebook(.ipynb)
   - 结果图片
   - README说明文件
2. 推荐使用Anaconda环境
3. 详细依赖请参考各项目README
4. 运行方式:
   - exam1: 直接运行e1.py或打开e1.ipynb
   - exam2: 运行fashion_mnist_processing.py或打开fashion_mnist_processing.ipynb
   - exam3: 运行mnist_image_process.py或打开mnist_image_process.ipynb
   - workbook1: 运行iris_logistic_regression.py或打开iris_logistic_regression.ipynb